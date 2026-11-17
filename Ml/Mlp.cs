using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text.Json;
using Lpc.Strategies.Planner;

namespace Lpc.Ml;

/// <summary>A training (or validation) matrix: Rows x InputCount raw (unstandardized) features.</summary>
public sealed record TrainingSet(float[] X, float[] Y, float[]? Weights, int Rows);

public sealed record FitResult(int EpochsRun, int BestEpoch, double InitialValLoss, double BestValLoss,
    double FinalTrainLoss, double RowsPerSec, string Backend);

/// <summary>
/// Small fully connected regression net: ReLU hidden layers, linear scalar output (default 28-64-64-1,
/// ~6k parameters). Inputs are standardized with a mean/std stored in the model and clipped to
/// +-<see cref="InputClip"/> so an out-of-distribution live feature can't blow up the output.
///
/// Inference matrix products go through <see cref="Blas.Sgemm"/> (Accelerate/AMX on Apple silicon;
/// ~1000 rows per turn, well under a millisecond). Training (<see cref="Fit"/>) runs on the Metal GPU
/// via <see cref="MetalMlpTrainer"/> when <see cref="Trainer.TrainDevice"/> allows it (default "auto"):
/// a whole epoch of minibatch steps is encoded into a few command buffers so dispatch overhead is
/// amortised; otherwise it falls back to the CPU path (<see cref="FitCpu"/>).
/// </summary>
public sealed class Mlp
{
    public const int FormatVersion = 1;
    public const float InputClip = 10f;
    public const float StdFloor = 1e-3f;

    readonly int[] sizes;          // [in, hidden..., 1]
    internal readonly float[][] W; // W[l] stored [sizes[l], sizes[l+1]] row-major
    internal readonly float[][] B; // B[l] length sizes[l+1]

    public int InputCount => sizes[0];
    public int[] Hidden => sizes[1..^1];
    public float[] Mean { get; }
    public float[] Std { get; }
    public Dictionary<string, string> Metadata { get; } = new();

    public int ParameterCount
    {
        get { int s = 0; for (int l = 0; l < W.Length; l++) s += W[l].Length + B[l].Length; return s; }
    }

    public Mlp(int inputCount, int[]? hidden = null, int seed = 1)
    {
        hidden ??= [64, 64];
        if (inputCount <= 0) throw new ArgumentOutOfRangeException(nameof(inputCount));
        if (hidden.Any(h => h <= 0)) throw new ArgumentException("Hidden sizes must be positive.", nameof(hidden));
        sizes = [inputCount, .. hidden, 1];
        int layers = sizes.Length - 1;
        W = new float[layers][];
        B = new float[layers][];
        var rng = new Random(seed);
        for (int l = 0; l < layers; l++)
        {
            int fin = sizes[l], fout = sizes[l + 1];
            float scale = MathF.Sqrt((l < layers - 1 ? 2f : 1f) / fin); // He init for ReLU layers
            W[l] = new float[fin * fout];
            B[l] = new float[fout];
            for (int i = 0; i < W[l].Length; i++) W[l][i] = (float)Gaussian(rng) * scale;
        }
        Mean = new float[inputCount];
        Std = new float[inputCount];
        Array.Fill(Std, 1f);
    }

    Mlp(int[] sizes, float[][] w, float[][] b, float[] mean, float[] std)
    {
        this.sizes = sizes; W = w; B = b; Mean = mean; Std = std;
    }

    public Mlp Clone()
    {
        var m = new Mlp((int[])sizes.Clone(), W.Select(a => (float[])a.Clone()).ToArray(),
            B.Select(a => (float[])a.Clone()).ToArray(), (float[])Mean.Clone(), (float[])Std.Clone());
        foreach (var kv in Metadata) m.Metadata[kv.Key] = kv.Value;
        return m;
    }

    static double Gaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>Sets Mean/Std from raw training rows (std floored at <see cref="StdFloor"/>).</summary>
    public void FitNormalization(ReadOnlySpan<float> x, int rows)
    {
        int d = InputCount;
        var sum = new double[d];
        var sq = new double[d];
        for (int r = 0; r < rows; r++)
        {
            var row = x.Slice(r * d, d);
            for (int j = 0; j < d; j++) { double v = row[j]; sum[j] += v; sq[j] += v * v; }
        }
        for (int j = 0; j < d; j++)
        {
            double mean = rows > 0 ? sum[j] / rows : 0;
            double var = rows > 0 ? Math.Max(0, sq[j] / rows - mean * mean) : 1;
            Mean[j] = (float)mean;
            Std[j] = Math.Max(StdFloor, (float)Math.Sqrt(var));
        }
    }

    internal void Standardize(ReadOnlySpan<float> raw, int rows, Span<float> dst)
    {
        int d = InputCount;
        for (int r = 0; r < rows; r++)
        {
            var src = raw.Slice(r * d, d);
            var o = dst.Slice(r * d, d);
            for (int j = 0; j < d; j++)
            {
                float v = (src[j] - Mean[j]) / Std[j];
                if (float.IsNaN(v)) v = 0f;
                else if (v > InputClip) v = InputClip;
                else if (v < -InputClip) v = -InputClip;
                o[j] = v;
            }
        }
    }

    // ---------------------------------------------------------------- inference

    /// <summary>Reusable activation buffers for <see cref="Predict"/>. Not thread-safe: one per thread.</summary>
    public sealed class Workspace
    {
        internal float[][] Act = [];
        int[] shape = [];
        public int Capacity { get; private set; }

        internal void Ensure(int[] sizes, int rows)
        {
            if (rows <= Capacity && shape.AsSpan().SequenceEqual(sizes)) return;
            int cap = Math.Max(256, (int)BitOperations.RoundUpToPowerOf2((uint)Math.Max(1, rows)));
            Act = new float[sizes.Length][];
            for (int l = 0; l < sizes.Length; l++) Act[l] = new float[cap * sizes[l]];
            shape = (int[])sizes.Clone();
            Capacity = cap;
        }
    }

    public Workspace CreateWorkspace(int rows = 1024)
    {
        var ws = new Workspace();
        ws.Ensure(sizes, rows);
        return ws;
    }

    /// <summary>Batched forward pass. raw is rows x InputCount unstandardized features; writes rows outputs.</summary>
    public void Predict(ReadOnlySpan<float> raw, int rows, Span<float> output, Workspace ws)
    {
        if (rows <= 0) return;
        if (raw.Length < rows * InputCount) throw new ArgumentException("Input too small.", nameof(raw));
        if (output.Length < rows) throw new ArgumentException("Output too small.", nameof(output));
        ws.Ensure(sizes, rows);
        Standardize(raw, rows, ws.Act[0]);
        ForwardCore(ws.Act, rows);
        ws.Act[^1].AsSpan(0, rows).CopyTo(output);
    }

    [ThreadStatic] static Workspace? threadWorkspace;

    /// <summary>
    /// Batched forward pass using a per-thread workspace. The model's parameters are only read during
    /// inference, so one Mlp can be shared by any number of threads calling this concurrently.
    /// </summary>
    public void Predict(ReadOnlySpan<float> rows, int rowCount, Span<float> output) =>
        Predict(rows, rowCount, output, threadWorkspace ??= new Workspace());

    /// <summary>Convenience single-shot prediction (allocates; not for the hot path).</summary>
    public float[] Predict(float[] raw, int rows)
    {
        var o = new float[rows];
        Predict(raw, rows, o, CreateWorkspace(rows));
        return o;
    }

    void ForwardCore(float[][] act, int rows)
    {
        int layers = W.Length;
        for (int l = 0; l < layers; l++)
        {
            int din = sizes[l], dout = sizes[l + 1];
            var y = act[l + 1].AsSpan(0, rows * dout);
            Blas.Sgemm(false, false, rows, dout, din, 1f, act[l].AsSpan(0, rows * din), W[l], 0f, y);
            AddBias(y, rows, dout, B[l], relu: l < layers - 1);
        }
    }

    static void AddBias(Span<float> y, int rows, int dout, float[] bias, bool relu)
    {
        int vc = Vector<float>.Count;
        bool vec = Vector.IsHardwareAccelerated && dout % vc == 0;
        var vb = vec ? MemoryMarshal.Cast<float, Vector<float>>(bias.AsSpan()) : default;
        for (int r = 0; r < rows; r++)
        {
            var row = y.Slice(r * dout, dout);
            if (vec)
            {
                var vr = MemoryMarshal.Cast<float, Vector<float>>(row);
                if (relu) for (int i = 0; i < vr.Length; i++) vr[i] = Vector.Max(vr[i] + vb[i], Vector<float>.Zero);
                else for (int i = 0; i < vr.Length; i++) vr[i] += vb[i];
            }
            else
            {
                for (int i = 0; i < dout; i++)
                {
                    float v = row[i] + bias[i];
                    row[i] = relu && v < 0f ? 0f : v;
                }
            }
        }
    }

    /// <summary>Mean squared error on raw rows.</summary>
    public double Evaluate(float[] x, float[] y, int rows)
    {
        if (rows <= 0) return double.NaN;
        const int chunk = 4096;
        var ws = CreateWorkspace(Math.Min(rows, chunk));
        var outBuf = new float[chunk];
        double sum = 0;
        for (int s = 0; s < rows; s += chunk)
        {
            int n = Math.Min(chunk, rows - s);
            Predict(x.AsSpan(s * InputCount, n * InputCount), n, outBuf, ws);
            for (int i = 0; i < n; i++) { double e = outBuf[i] - y[s + i]; sum += e * e; }
        }
        return sum / rows;
    }

    // ---------------------------------------------------------------- training

    /// <summary>
    /// Minibatch Adam on weighted MSE (or Huber when o.HuberDelta &gt; 0) with coupled L2 weight decay.
    /// Uses the model's current Mean/Std (call <see cref="FitNormalization"/> first for a fresh model).
    /// With a validation set, early-stops on validation MSE and restores the best weights (the
    /// starting weights count as epoch 0, so fine-tuning can never make validation loss worse).
    /// </summary>
    public FitResult Fit(TrainingSet train, TrainingSet? val, TrainOptions o, float learningRate, int epochs)
    {
        if (Trainer.UseGpu)
        {
            try { return MetalMlpTrainer.Fit(this, train, val, o, learningRate, epochs); } // weights untouched on failure
            catch (Exception e) when (Trainer.TrainDevice == "auto" && e is MetalException or DllNotFoundException or EntryPointNotFoundException)
            {
                (o.Log ?? Console.WriteLine)($"fit: GPU training failed ({e.Message}); falling back to CPU");
            }
        }
        return FitCpu(train, val, o, learningRate, epochs);
    }

    /// <summary>CPU (Accelerate/managed BLAS) implementation of <see cref="Fit"/>.</summary>
    public FitResult FitCpu(TrainingSet train, TrainingSet? val, TrainOptions o, float learningRate, int epochs)
    {
        int d = InputCount, n = train.Rows, layers = W.Length;
        if (n <= 0) throw new ArgumentException("Empty training set.", nameof(train));
        int bs = Math.Max(1, Math.Min(o.BatchSize, n));
        var log = o.Log;

        var xs = new float[n * d];
        Standardize(train.X, n, xs);

        var act = new float[layers + 1][];
        var grad = new float[layers + 1][];
        for (int l = 0; l <= layers; l++) { act[l] = new float[bs * sizes[l]]; grad[l] = new float[bs * sizes[l]]; }
        var gW = W.Select(a => new float[a.Length]).ToArray();
        var gB = B.Select(a => new float[a.Length]).ToArray();
        var mW = W.Select(a => new float[a.Length]).ToArray();
        var vW = W.Select(a => new float[a.Length]).ToArray();
        var mB = B.Select(a => new float[a.Length]).ToArray();
        var vB = B.Select(a => new float[a.Length]).ToArray();
        float[][] bestW = W.Select(a => (float[])a.Clone()).ToArray();
        float[][] bestB = B.Select(a => (float[])a.Clone()).ToArray();

        bool hasVal = val is { Rows: > 0 };
        double initialVal = hasVal ? Evaluate(val!.X, val.Y, val.Rows) : double.NaN;
        double bestVal = hasVal ? initialVal : double.PositiveInfinity;
        int bestEpoch = 0, bad = 0, epoch = 0;
        double lastTrain = double.NaN;
        long totalRows = 0, totalTicks = 0;

        var idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        var rng = new Random(o.Seed);
        float delta = o.HuberDelta;
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        int step = 0;

        log?.Invoke($"fit: rows={n} val={(hasVal ? val!.Rows : 0)} params={ParameterCount} shape={string.Join('-', sizes)} " +
                    $"batch={bs} lr={learningRate:G3} wd={o.WeightDecay:G3} loss={(delta > 0 ? $"huber({delta})" : "mse")} backend={Blas.Backend}" +
                    (hasVal ? $" initialValMse={initialVal:G5}" : ""));

        for (epoch = 1; epoch <= epochs; epoch++)
        {
            long t0 = Stopwatch.GetTimestamp();
            for (int i = n - 1; i > 0; i--) { int j = rng.Next(i + 1); (idx[i], idx[j]) = (idx[j], idx[i]); }
            double lossSum = 0, wSum = 0;

            for (int start = 0; start < n; start += bs)
            {
                int b = Math.Min(bs, n - start);
                for (int r = 0; r < b; r++)
                    xs.AsSpan(idx[start + r] * d, d).CopyTo(act[0].AsSpan(r * d, d));
                ForwardCore(act, b);

                float sw = 0f;
                for (int r = 0; r < b; r++) sw += train.Weights?[idx[start + r]] ?? 1f;
                if (sw <= 0f) continue;
                var outp = act[layers];
                var dOut = grad[layers];
                for (int r = 0; r < b; r++)
                {
                    int i = idx[start + r];
                    float wi = train.Weights?[i] ?? 1f;
                    float e = outp[r] - train.Y[i];
                    float loss, g;
                    if (delta > 0f)
                    {
                        float ae = MathF.Abs(e);
                        loss = ae <= delta ? 0.5f * e * e : delta * (ae - 0.5f * delta);
                        g = Math.Clamp(e, -delta, delta);
                    }
                    else { loss = e * e; g = 2f * e; }
                    lossSum += wi * loss;
                    dOut[r] = wi * g / sw;
                }
                wSum += sw;

                // backward
                for (int l = layers - 1; l >= 0; l--)
                {
                    int din = sizes[l], dout = sizes[l + 1];
                    var dY = grad[l + 1].AsSpan(0, b * dout);
                    var x = act[l].AsSpan(0, b * din);
                    Blas.Sgemm(true, false, din, dout, b, 1f, x, dY, 0f, gW[l]);   // dW = X^T dY
                    var gb = gB[l];
                    Array.Clear(gb);
                    for (int r = 0; r < b; r++)
                    {
                        var row = dY.Slice(r * dout, dout);
                        for (int j = 0; j < dout; j++) gb[j] += row[j];
                    }
                    if (l > 0)
                    {
                        var dX = grad[l].AsSpan(0, b * din);
                        Blas.Sgemm(false, true, b, din, dout, 1f, dY, W[l], 0f, dX); // dX = dY W^T
                        for (int i = 0; i < dX.Length; i++) if (x[i] <= 0f) dX[i] = 0f; // ReLU mask
                    }
                }

                step++;
                float c1 = 1f - MathF.Pow(b1, step), c2 = 1f - MathF.Pow(b2, step);
                float lrT = learningRate * MathF.Sqrt(c2) / c1;
                for (int l = 0; l < layers; l++)
                {
                    Adam(W[l], gW[l], mW[l], vW[l], lrT, o.WeightDecay, b1, b2, eps);
                    Adam(B[l], gB[l], mB[l], vB[l], lrT, 0f, b1, b2, eps);
                }
            }

            long ticks = Stopwatch.GetTimestamp() - t0;
            totalTicks += ticks;
            totalRows += n;
            lastTrain = wSum > 0 ? lossSum / wSum : double.NaN;
            double rowsPerSec = n / Math.Max(1e-9, ticks / (double)Stopwatch.Frequency);

            if (!double.IsFinite(lastTrain))
            {
                log?.Invoke($"epoch {epoch}: training loss diverged ({lastTrain}); restoring best weights.");
                break;
            }

            string msg = $"epoch {epoch}/{epochs} train={lastTrain:G5}";
            if (hasVal)
            {
                double v = Evaluate(val!.X, val.Y, val.Rows);
                msg += $" valMse={v:G5}";
                if (v < bestVal)
                {
                    bestVal = v; bestEpoch = epoch; bad = 0;
                    for (int l = 0; l < layers; l++) { W[l].CopyTo(bestW[l], 0); B[l].CopyTo(bestB[l], 0); }
                    msg += " *";
                }
                else bad++;
            }
            else
            {
                bestEpoch = epoch;
                for (int l = 0; l < layers; l++) { W[l].CopyTo(bestW[l], 0); B[l].CopyTo(bestB[l], 0); }
            }
            log?.Invoke($"{msg} rows/s={rowsPerSec:N0} ({Blas.Backend})");
            if (hasVal && bad >= o.Patience) { log?.Invoke($"early stop: no val improvement for {bad} epochs"); epoch++; break; }
        }

        for (int l = 0; l < layers; l++) { bestW[l].CopyTo(W[l], 0); bestB[l].CopyTo(B[l], 0); }
        double avgRps = totalRows / Math.Max(1e-9, totalTicks / (double)Stopwatch.Frequency);
        var result = new FitResult(epoch - 1, bestEpoch, initialVal, hasVal ? bestVal : double.NaN, lastTrain, avgRps, Blas.Backend);
        log?.Invoke($"fit done: epochs={result.EpochsRun} best={bestEpoch} bestValMse={result.BestValLoss:G5} avgRows/s={avgRps:N0} backend={Blas.Backend}");
        return result;
    }

    static void Adam(float[] w, float[] g, float[] m, float[] v, float lrT, float wd, float b1, float b2, float eps)
    {
        for (int i = 0; i < w.Length; i++)
        {
            float gi = g[i] + wd * w[i];
            float mi = m[i] = b1 * m[i] + (1f - b1) * gi;
            float vi = v[i] = b2 * v[i] + (1f - b2) * gi * gi;
            w[i] -= lrT * mi / (MathF.Sqrt(vi) + eps);
        }
    }

    // ---------------------------------------------------------------- persistence

    sealed class Dto
    {
        public int Version { get; set; }
        public int FeatureVersion { get; set; }
        public string[] FeatureNames { get; set; } = [];
        public int InputCount { get; set; }
        public int[] Hidden { get; set; } = [];
        public float[] Mean { get; set; } = [];
        public float[] Std { get; set; } = [];
        public float[][] Weights { get; set; } = [];
        public float[][] Biases { get; set; } = [];
        public Dictionary<string, string> Metadata { get; set; } = new();
    }

    static readonly JsonSerializerOptions JsonOpts = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = false };

    public void Save(string path)
    {
        var dto = new Dto
        {
            Version = FormatVersion,
            FeatureVersion = Features.Version,
            FeatureNames = InputCount == Features.Count ? Features.Names : [],
            InputCount = InputCount,
            Hidden = Hidden,
            Mean = Mean,
            Std = Std,
            Weights = W,
            Biases = B,
            Metadata = Metadata,
        };
        var dir = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        var tmp = path + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(dto, JsonOpts));
        File.Move(tmp, path, overwrite: true);
    }

    /// <summary>Loads a model; throws InvalidDataException if it doesn't match the current <see cref="Features"/> layout.</summary>
    public static Mlp Load(string path)
    {
        var dto = JsonSerializer.Deserialize<Dto>(File.ReadAllText(path), JsonOpts)
                  ?? throw new InvalidDataException($"{path}: empty model file.");
        if (dto.Version != FormatVersion) throw new InvalidDataException($"{path}: model format v{dto.Version}, expected v{FormatVersion}.");
        if (dto.FeatureVersion != Features.Version) throw new InvalidDataException($"{path}: featureVersion {dto.FeatureVersion} != current {Features.Version}.");
        if (dto.InputCount != Features.Count) throw new InvalidDataException($"{path}: inputCount {dto.InputCount} != Features.Count {Features.Count}.");
        if (!dto.FeatureNames.SequenceEqual(Features.Names)) throw new InvalidDataException($"{path}: feature names differ from the current Features.Names.");
        int[] sizes = [dto.InputCount, .. dto.Hidden, 1];
        int layers = sizes.Length - 1;
        if (dto.Hidden.Any(h => h <= 0) || dto.Weights.Length != layers || dto.Biases.Length != layers ||
            dto.Mean.Length != dto.InputCount || dto.Std.Length != dto.InputCount)
            throw new InvalidDataException($"{path}: inconsistent model shape.");
        for (int l = 0; l < layers; l++)
            if (dto.Weights[l].Length != sizes[l] * sizes[l + 1] || dto.Biases[l].Length != sizes[l + 1])
                throw new InvalidDataException($"{path}: layer {l} has the wrong size.");
        if (dto.Weights.Concat(dto.Biases).Append(dto.Mean).Append(dto.Std).Any(a => a.Any(v => !float.IsFinite(v))))
            throw new InvalidDataException($"{path}: non-finite parameter.");
        for (int j = 0; j < dto.Std.Length; j++) dto.Std[j] = Math.Max(StdFloor, dto.Std[j]);
        var m = new Mlp(sizes, dto.Weights, dto.Biases, dto.Mean, dto.Std);
        foreach (var kv in dto.Metadata) m.Metadata[kv.Key] = kv.Value;
        return m;
    }
}
