using System.Globalization;
using Lpc.Strategies.Planner;

namespace Lpc.Ml;

public sealed class TrainOptions
{
    public int Epochs = 25;
    public int BatchSize = 512;
    /// <summary>Minibatch size when training on the GPU (0 = use BatchSize).</summary>
    public int GpuBatchSize = 0;
    public float LearningRate = 1e-3f;
    public float WeightDecay = 1e-5f;
    public double ValidationFraction = 0.1;
    public int Patience = 4;
    public int[] Hidden = [64, 64];
    public int Seed = 1;
    /// <summary>Fine-tuning: share of the fine-tuning training mix drawn from sim replay (0.3 = 30% sim, 70% live).</summary>
    public double ReplayFraction = 0.3;
    /// <summary>Fine-tuning: sample weight of live rows (sim replay rows have weight 1).</summary>
    public float LiveWeight = 2f;
    /// <summary>Fine-tuning learning rate (Pretrain uses LearningRate).</summary>
    public float FinetuneLearningRate = 3e-4f;
    /// <summary>Fine-tuning epoch cap (Pretrain uses Epochs).</summary>
    public int FinetuneEpochs = 10;
    /// <summary>0 = MSE; &gt; 0 = Huber loss with this delta.</summary>
    public float HuberDelta = 0f;
    public Action<string>? Log;
}

public static class Trainer
{
    static volatile string trainDevice = Environment.GetEnvironmentVariable("LPC_TRAIN_DEVICE") is { Length: > 0 } env ? Normalize(env) : "auto";

    /// <summary>
    /// Device used by <see cref="Mlp.Fit"/> for training only (inference always stays on the CPU):
    /// "auto" (Metal GPU when available, else CPU), "gpu" (throws if Metal is unavailable) or "cpu".
    /// Defaults to the LPC_TRAIN_DEVICE environment variable, else "auto".
    /// </summary>
    public static string TrainDevice
    {
        get => trainDevice;
        set => trainDevice = Normalize(value);
    }

    static string Normalize(string v) => v.Trim().ToLowerInvariant() switch
    {
        "auto" => "auto",
        "gpu" or "metal" => "gpu",
        "cpu" => "cpu",
        _ => throw new ArgumentException($"Unknown training device '{v}' (expected auto|gpu|cpu)."),
    };

    /// <summary>True when the next Mlp.Fit will run on the GPU.</summary>
    public static bool UseGpu => TrainDevice switch
    {
        "cpu" => false,
        "gpu" => MetalMlpTrainer.Available ? true : throw new InvalidOperationException("TrainDevice=gpu but Metal is unavailable: " + MetalMlpTrainer.UnavailableReason),
        _ => MetalMlpTrainer.Available,
    };

    /// <summary>Human-readable training device for logs.</summary>
    public static string DeviceDescription => UseGpuSafe() ? $"gpu (Metal: {Metal.DeviceName})" : $"cpu ({Blas.Backend})";

    static bool UseGpuSafe() { try { return UseGpu; } catch (InvalidOperationException) { return false; } }

    /// <summary>Trains a fresh model on simulator rows (normalization computed from the training split).</summary>
    public static Mlp Pretrain(IReadOnlyList<TripRecord> simData, TrainOptions o)
    {
        var rows = Clean(simData, o.Log, "sim");
        if (rows.Count < 2) throw new ArgumentException("Need at least 2 valid rows to pretrain.", nameof(simData));
        var rng = new Random(o.Seed);
        var order = Shuffled(rows.Count, rng);
        int nv = Math.Clamp((int)(rows.Count * o.ValidationFraction), 0, rows.Count - 1);
        var val = Pack(rows, order.AsSpan(0, nv), null);
        var train = Pack(rows, order.AsSpan(nv), null);

        var model = new Mlp(Features.Count, o.Hidden, o.Seed);
        model.FitNormalization(train.X, train.Rows);
        o.Log?.Invoke($"pretrain: {train.Rows} train / {val.Rows} val rows, device={DeviceDescription}");
        var r = model.Fit(train, val.Rows > 0 ? val : null, o, o.LearningRate, o.Epochs);

        Stamp(model, "pretrain", r, train.Rows, val.Rows);
        model.Metadata["simRows"] = rows.Count.ToString(CultureInfo.InvariantCulture);
        return model;
    }

    /// <summary>
    /// Fine-tunes a copy of <paramref name="pretrained"/> (keeping its normalization) on live rows mixed
    /// with sim replay (ReplayFraction of the mix) to limit forgetting. Live rows get LiveWeight.
    /// Early stopping uses held-out LIVE rows only; the pretrained weights count as epoch 0, so the result
    /// is never worse than the pretrained net on that held-out set.
    /// </summary>
    public static Mlp Finetune(Mlp pretrained, IReadOnlyList<TripRecord> liveData, IReadOnlyList<TripRecord> simReplay, TrainOptions o)
    {
        if (pretrained.InputCount != Features.Count) throw new ArgumentException("Pretrained model has the wrong input count.", nameof(pretrained));
        var live = Clean(liveData, o.Log, "live");
        var sim = Clean(simReplay, o.Log, "sim replay");
        if (live.Count == 0) throw new ArgumentException("No valid live rows to fine-tune on.", nameof(liveData));
        var rng = new Random(o.Seed);

        var liveOrder = Shuffled(live.Count, rng);
        int nv = Math.Clamp((int)(live.Count * o.ValidationFraction), 0, live.Count - 1);
        var val = Pack(live, liveOrder.AsSpan(0, nv), null);
        int nl = live.Count - nv;

        double rf = Math.Clamp(o.ReplayFraction, 0, 0.95);
        int nReplay = Math.Min(sim.Count, (int)Math.Round(nl * rf / (1 - rf)));
        var simOrder = Shuffled(sim.Count, rng); // first nReplay = sample without replacement

        var mixed = new List<TripRecord>(nl + nReplay);
        var weights = new float[nl + nReplay];
        for (int i = 0; i < nl; i++) { mixed.Add(live[liveOrder[nv + i]]); weights[i] = o.LiveWeight; }
        for (int i = 0; i < nReplay; i++) { mixed.Add(sim[simOrder[i]]); weights[nl + i] = 1f; }
        var all = Enumerable.Range(0, mixed.Count).ToArray();
        var train = Pack(mixed, all, weights);

        var model = pretrained.Clone();
        o.Log?.Invoke($"finetune: {nl} live (w={o.LiveWeight}) + {nReplay} sim replay train rows, {val.Rows} live val rows, " +
                      $"lr={o.FinetuneLearningRate:G3}, device={DeviceDescription}");
        var r = model.Fit(train, val.Rows > 0 ? val : null, o, o.FinetuneLearningRate, o.FinetuneEpochs);

        Stamp(model, "finetune", r, train.Rows, val.Rows);
        model.Metadata["liveRows"] = live.Count.ToString(CultureInfo.InvariantCulture);
        model.Metadata["replayRows"] = nReplay.ToString(CultureInfo.InvariantCulture);
        if (val.Rows > 0) model.Metadata["liveValMseBefore"] = r.InitialValLoss.ToString("G6", CultureInfo.InvariantCulture);
        return model;
    }

    static void Stamp(Mlp m, string mode, FitResult r, int trainRows, int valRows)
    {
        var inv = CultureInfo.InvariantCulture;
        m.Metadata["mode"] = mode;
        m.Metadata["trainedAtUtc"] = DateTime.UtcNow.ToString("O", inv);
        m.Metadata["trainRows"] = trainRows.ToString(inv);
        m.Metadata["valRows"] = valRows.ToString(inv);
        m.Metadata["epochs"] = r.EpochsRun.ToString(inv);
        m.Metadata["bestEpoch"] = r.BestEpoch.ToString(inv);
        m.Metadata["bestValMse"] = r.BestValLoss.ToString("G6", inv);
        m.Metadata["finalTrainLoss"] = r.FinalTrainLoss.ToString("G6", inv);
        m.Metadata["rowsPerSec"] = r.RowsPerSec.ToString("F0", inv);
        m.Metadata["backend"] = r.Backend;
    }

    static List<TripRecord> Clean(IReadOnlyList<TripRecord> data, Action<string>? log, string what)
    {
        var ok = new List<TripRecord>(data.Count);
        foreach (var r in data)
            if (r?.F is { Length: Features.Count } f && float.IsFinite(r.Y) && Array.TrueForAll(f, float.IsFinite))
                ok.Add(r);
        if (ok.Count != data.Count) log?.Invoke($"{what}: dropped {data.Count - ok.Count} invalid rows (wrong length or non-finite)");
        return ok;
    }

    static int[] Shuffled(int n, Random rng)
    {
        var a = new int[n];
        for (int i = 0; i < n; i++) a[i] = i;
        for (int i = n - 1; i > 0; i--) { int j = rng.Next(i + 1); (a[i], a[j]) = (a[j], a[i]); }
        return a;
    }

    static TrainingSet Pack(IReadOnlyList<TripRecord> rows, ReadOnlySpan<int> idx, float[]? weights)
    {
        int d = Features.Count;
        var x = new float[idx.Length * d];
        var y = new float[idx.Length];
        for (int i = 0; i < idx.Length; i++)
        {
            var r = rows[idx[i]];
            r.F.AsSpan().CopyTo(x.AsSpan(i * d, d));
            y[i] = r.Y;
        }
        return new TrainingSet(x, y, weights, idx.Length);
    }
}
