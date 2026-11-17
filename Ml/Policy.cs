using System.Numerics;
using System.Text.Json;
using Lpc.Strategies.Planner;

namespace Lpc.Ml;

/// <summary>
/// Candidate-ranking policy: a shared MLP f(row) -> logit (default 28-64-64-1, ReLU) applied to every
/// candidate row of one worker decision; pi(a|s) = softmax over that decision's rows of logit / T.
///
/// Inputs are standardized with a stored mean/std (std floored at <see cref="StdFloor"/>, clipped to
/// +-<see cref="InputClip"/>). Forward and the hand-written backward pass go through <see cref="Blas.Sgemm"/>.
/// <see cref="Logits"/> is safe for concurrent readers (per-thread scratch); training mutates the
/// weights and must not overlap with inference on the same instance (train a <see cref="Clone"/>).
/// </summary>
public sealed class PolicyNet
{
    public const int FormatVersion = 1;
    public const float InputClip = 10f;
    public const float StdFloor = 1e-3f;

    readonly int[] sizes;          // [in, hidden..., 1]
    internal readonly float[][] W; // W[l] stored [sizes[l], sizes[l+1]] row-major
    internal readonly float[][] B; // B[l] length sizes[l+1]
    readonly ThreadLocal<Workspace> inferWs = new(() => new Workspace());

    public int InputCount => sizes[0];
    public int[] Hidden => sizes[1..^1];
    internal int[] Sizes => sizes;
    internal int LayerCount => W.Length;
    public float[] Mean { get; }
    public float[] Std { get; }
    public Dictionary<string, string> Metadata { get; } = new();

    public int ParameterCount
    {
        get { int s = 0; for (int l = 0; l < W.Length; l++) s += W[l].Length + B[l].Length; return s; }
    }

    public PolicyNet(int inputCount = Features.Count, int[]? hidden = null, int seed = 1)
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
            // He init for ReLU layers; small output layer so a fresh policy starts near uniform.
            float scale = l < layers - 1 ? MathF.Sqrt(2f / fin) : 0.1f * MathF.Sqrt(1f / fin);
            W[l] = new float[fin * fout];
            B[l] = new float[fout];
            for (int i = 0; i < W[l].Length; i++) W[l][i] = (float)Gaussian(rng) * scale;
        }
        Mean = new float[inputCount];
        Std = new float[inputCount];
        Array.Fill(Std, 1f);
    }

    PolicyNet(int[] sizes, float[][] w, float[][] b, float[] mean, float[] std)
    {
        this.sizes = sizes; W = w; B = b; Mean = mean; Std = std;
    }

    public PolicyNet Clone()
    {
        var m = new PolicyNet((int[])sizes.Clone(), W.Select(a => (float[])a.Clone()).ToArray(),
            B.Select(a => (float[])a.Clone()).ToArray(), (float[])Mean.Clone(), (float[])Std.Clone());
        foreach (var kv in Metadata) m.Metadata[kv.Key] = kv.Value;
        return m;
    }

    static double Gaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    // ---------------------------------------------------------------- shared log-prob

    /// <summary>
    /// Stable log-softmax: log pi(chosen) = z_c - logsumexp(z) with z = logits / temperature.
    /// The planner (behavior log-prob) and <see cref="PolicyTrainer"/> both use this so they match exactly.
    /// Returns NaN if chosen is out of range or any logit is non-finite.
    /// </summary>
    public static float LogProb(ReadOnlySpan<float> logits, int chosen, double temperature)
    {
        if ((uint)chosen >= (uint)logits.Length) return float.NaN;
        double invT = 1.0 / Math.Max(1e-6, temperature);
        double max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            double z = logits[i] * invT;
            if (!double.IsFinite(z)) return float.NaN;
            if (z > max) max = z;
        }
        double sum = 0;
        for (int i = 0; i < logits.Length; i++) sum += Math.Exp(logits[i] * invT - max);
        return (float)(logits[chosen] * invT - max - Math.Log(sum));
    }

    // ---------------------------------------------------------------- normalization

    /// <summary>Sets Mean/Std from raw rows (std floored at <see cref="StdFloor"/>).</summary>
    public void FitNormalization(ReadOnlySpan<float> x, int rows)
    {
        int d = InputCount;
        var sum = new double[d];
        var sq = new double[d];
        for (int r = 0; r < rows; r++)
        {
            var row = x.Slice(r * d, d);
            for (int j = 0; j < d; j++) { double v = row[j]; if (!double.IsFinite(v)) v = 0; sum[j] += v; sq[j] += v * v; }
        }
        for (int j = 0; j < d; j++)
        {
            double mean = rows > 0 ? sum[j] / rows : 0;
            double var = rows > 0 ? Math.Max(0, sq[j] / rows - mean * mean) : 1;
            Mean[j] = (float)mean;
            Std[j] = Math.Max(StdFloor, (float)Math.Sqrt(var));
        }
    }

    /// <summary>Fits Mean/Std on every candidate row of the given decisions.</summary>
    public void FitNormalization(IReadOnlyList<DecisionRecord> decisions)
    {
        int d = InputCount;
        var sum = new double[d];
        var sq = new double[d];
        long rows = 0;
        foreach (var rec in decisions)
        {
            if (rec.Rows <= 0 || rec.Candidates.Length < rec.Rows * d) continue;
            for (int r = 0; r < rec.Rows; r++)
            {
                for (int j = 0; j < d; j++)
                {
                    double v = rec.Candidates[r * d + j];
                    if (!double.IsFinite(v)) v = 0;
                    sum[j] += v; sq[j] += v * v;
                }
            }
            rows += rec.Rows;
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

    // ---------------------------------------------------------------- forward / backward

    /// <summary>Activation and delta buffers. One per thread.</summary>
    internal sealed class Workspace
    {
        public float[] Raw = [];
        public float[][] Act = [];   // Act[0] standardized input, Act[l] output of layer l-1 (post-ReLU on hidden)
        public float[][] Delta = []; // Delta[l] = dLoss/d(pre-activation) of layer l-1, same shape as Act[l]
        int[] shape = [];
        public int Capacity { get; private set; }

        public void Ensure(int[] sizes, int rows)
        {
            if (rows <= Capacity && shape.AsSpan().SequenceEqual(sizes)) return;
            int cap = Math.Max(256, (int)BitOperations.RoundUpToPowerOf2((uint)Math.Max(1, rows)));
            Raw = new float[cap * sizes[0]];
            Act = new float[sizes.Length][];
            Delta = new float[sizes.Length][];
            for (int l = 0; l < sizes.Length; l++)
            {
                Act[l] = new float[cap * sizes[l]];
                Delta[l] = l == 0 ? [] : new float[cap * sizes[l]];
            }
            shape = (int[])sizes.Clone();
            Capacity = cap;
        }
    }

    internal Workspace NewWorkspace(int rows)
    {
        var ws = new Workspace();
        ws.Ensure(sizes, rows);
        return ws;
    }

    /// <summary>Forward from ws.Act[0] (already standardized); logits land in ws.Act[^1][0..rows].</summary>
    internal void Forward(Workspace ws, int rows)
    {
        int L = W.Length;
        for (int l = 0; l < L; l++)
        {
            int fin = sizes[l], fout = sizes[l + 1];
            var outp = ws.Act[l + 1].AsSpan(0, rows * fout);
            var b = B[l];
            for (int r = 0; r < rows; r++) b.AsSpan().CopyTo(outp.Slice(r * fout, fout));
            Blas.Sgemm(false, false, rows, fout, fin, 1f, ws.Act[l].AsSpan(0, rows * fin), W[l], 1f, outp);
            if (l < L - 1) Relu(outp);
        }
    }

    static void Relu(Span<float> x)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && x.Length >= Vector<float>.Count)
        {
            var vx = System.Runtime.InteropServices.MemoryMarshal.Cast<float, Vector<float>>(x);
            for (int v = 0; v < vx.Length; v++) vx[v] = Vector.Max(vx[v], Vector<float>.Zero);
            i = vx.Length * Vector<float>.Count;
        }
        for (; i < x.Length; i++) if (x[i] < 0f) x[i] = 0f;
    }

    /// <summary>Gradient buffers shaped like the parameters.</summary>
    internal sealed class Grads
    {
        public readonly float[][] W, B;
        public Grads(PolicyNet net)
        {
            W = net.W.Select(a => new float[a.Length]).ToArray();
            B = net.B.Select(a => new float[a.Length]).ToArray();
        }
    }

    /// <summary>
    /// Backward pass. Expects ws.Delta[^1][0..rows] = dLoss/dLogit and the activations of the matching Forward.
    /// Overwrites g with the gradient (sum over rows).
    /// </summary>
    internal void Backward(Workspace ws, int rows, Grads g)
    {
        int L = W.Length;
        for (int l = L - 1; l >= 0; l--)
        {
            int fin = sizes[l], fout = sizes[l + 1];
            var dz = ws.Delta[l + 1].AsSpan(0, rows * fout);
            // dW = A^T dZ   (A rows x fin stored -> transA)
            Blas.Sgemm(true, false, fin, fout, rows, 1f, ws.Act[l].AsSpan(0, rows * fin), dz, 0f, g.W[l]);
            var gb = g.B[l];
            Array.Clear(gb);
            for (int r = 0; r < rows; r++)
            {
                var row = dz.Slice(r * fout, fout);
                for (int j = 0; j < fout; j++) gb[j] += row[j];
            }
            if (l == 0) break;
            // dA = dZ W^T, then ReLU mask (Act[l] > 0 iff pre-activation > 0)
            var da = ws.Delta[l].AsSpan(0, rows * fin);
            Blas.Sgemm(false, true, rows, fin, fout, 1f, dz, W[l], 0f, da);
            var a = ws.Act[l].AsSpan(0, rows * fin);
            for (int i = 0; i < da.Length; i++) if (a[i] <= 0f) da[i] = 0f;
        }
    }

    /// <summary>Adam moments + step counter.</summary>
    internal sealed class AdamState
    {
        public readonly float[][] MW, VW, MB, VB;
        public int T;
        public AdamState(PolicyNet net)
        {
            MW = net.W.Select(a => new float[a.Length]).ToArray();
            VW = net.W.Select(a => new float[a.Length]).ToArray();
            MB = net.B.Select(a => new float[a.Length]).ToArray();
            VB = net.B.Select(a => new float[a.Length]).ToArray();
        }
    }

    /// <summary>Clips g to global norm maxNorm (in place) and applies one Adam step. Returns the pre-clip norm.</summary>
    internal double AdamStep(Grads g, AdamState s, float lr, float maxNorm, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
    {
        double sq = 0;
        foreach (var a in g.W) foreach (var v in a) sq += (double)v * v;
        foreach (var a in g.B) foreach (var v in a) sq += (double)v * v;
        double norm = Math.Sqrt(sq);
        if (!double.IsFinite(norm)) return norm; // skip a poisoned step entirely
        float scale = norm > maxNorm && maxNorm > 0 ? (float)(maxNorm / norm) : 1f;
        s.T++;
        float bc1 = 1f - MathF.Pow(b1, s.T), bc2 = 1f - MathF.Pow(b2, s.T);
        float lrT = lr * MathF.Sqrt(bc2) / bc1;
        for (int l = 0; l < W.Length; l++)
        {
            Adam(W[l], g.W[l], s.MW[l], s.VW[l], scale, lrT, b1, b2, eps);
            Adam(B[l], g.B[l], s.MB[l], s.VB[l], scale, lrT, b1, b2, eps);
        }
        return norm;
    }

    static void Adam(float[] w, float[] g, float[] m, float[] v, float gScale, float lrT, float b1, float b2, float eps)
    {
        for (int i = 0; i < w.Length; i++)
        {
            float gi = g[i] * gScale;
            m[i] = b1 * m[i] + (1 - b1) * gi;
            v[i] = b2 * v[i] + (1 - b2) * gi * gi;
            w[i] -= lrT * m[i] / (MathF.Sqrt(v[i]) + eps);
        }
    }

    /// <summary>All parameters flattened (W0, B0, W1, B1, ...). For tests / finite differences.</summary>
    public float[] GetParameters()
    {
        var p = new float[ParameterCount];
        int o = 0;
        for (int l = 0; l < W.Length; l++)
        {
            W[l].CopyTo(p, o); o += W[l].Length;
            B[l].CopyTo(p, o); o += B[l].Length;
        }
        return p;
    }

    public void SetParameters(ReadOnlySpan<float> p)
    {
        if (p.Length != ParameterCount) throw new ArgumentException("Wrong parameter count.", nameof(p));
        int o = 0;
        for (int l = 0; l < W.Length; l++)
        {
            p.Slice(o, W[l].Length).CopyTo(W[l]); o += W[l].Length;
            p.Slice(o, B[l].Length).CopyTo(B[l]); o += B[l].Length;
        }
    }

    internal static void Flatten(Grads g, float[] dst)
    {
        int o = 0;
        for (int l = 0; l < g.W.Length; l++)
        {
            g.W[l].CopyTo(dst, o); o += g.W[l].Length;
            g.B[l].CopyTo(dst, o); o += g.B[l].Length;
        }
    }

    // ---------------------------------------------------------------- inference

    /// <summary>
    /// Logits for rowCount raw (unstandardized) feature rows (rowCount x InputCount, row-major).
    /// Thread-safe for concurrent readers: scratch is per thread.
    /// </summary>
    public void Logits(ReadOnlySpan<float> rows, int rowCount, Span<float> output)
    {
        if (rowCount <= 0) return;
        if (rows.Length < rowCount * InputCount) throw new ArgumentException("Input too small.", nameof(rows));
        if (output.Length < rowCount) throw new ArgumentException("Output too small.", nameof(output));
        var ws = inferWs.Value!;
        ws.Ensure(sizes, rowCount);
        Standardize(rows, rowCount, ws.Act[0]);
        Forward(ws, rowCount);
        ws.Act[^1].AsSpan(0, rowCount).CopyTo(output);
    }

    // ---------------------------------------------------------------- persistence

    sealed class Dto
    {
        public string Kind { get; set; } = "";
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
            Kind = "policy",
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

    /// <summary>Loads a policy; throws InvalidDataException if it doesn't match the current <see cref="Features"/> layout.</summary>
    public static PolicyNet Load(string path)
    {
        Dto? dto;
        try { dto = JsonSerializer.Deserialize<Dto>(File.ReadAllText(path), JsonOpts); }
        catch (JsonException e) { throw new InvalidDataException($"{path}: malformed policy JSON ({e.Message}).", e); }
        if (dto is null) throw new InvalidDataException($"{path}: empty model file.");
        // Explicit JSON nulls bypass the property initializers.
        if (dto.Hidden is null || dto.Weights is null || dto.Biases is null || dto.Mean is null || dto.Std is null ||
            dto.FeatureNames is null || dto.Weights.Any(a => a is null) || dto.Biases.Any(a => a is null))
            throw new InvalidDataException($"{path}: missing (null) model field.");
        dto.Metadata ??= new();
        if (dto.Hidden.Any(h => h > 4096)) throw new InvalidDataException($"{path}: hidden layer too large.");
        if (dto.Kind != "policy") throw new InvalidDataException($"{path}: not a policy model (kind '{dto.Kind}').");
        if (dto.Version != FormatVersion) throw new InvalidDataException($"{path}: policy format v{dto.Version}, expected v{FormatVersion}.");
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
        var m = new PolicyNet(sizes, dto.Weights, dto.Biases, dto.Mean, dto.Std);
        foreach (var kv in dto.Metadata) m.Metadata[kv.Key] = kv.Value;
        return m;
    }
}
