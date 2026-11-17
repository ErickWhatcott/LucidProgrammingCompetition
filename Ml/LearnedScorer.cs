using System.Diagnostics;
using Lpc.Strategies.Planner;

namespace Lpc.Ml;

/// <summary>
/// Scores planner candidates with an <see cref="Mlp"/> (predicted gold-equivalent banked per turn).
/// Thread-safe: the model is read-only during inference and scratch buffers are per thread, so one
/// scorer (or one model behind many scorers) can serve parallel self-play games.
///
/// Latency guard: before running, the expected cost (fixed overhead + rows x running-average cost per
/// row, times <see cref="SafetyFactor"/>) is compared with the remaining turn budget; if it doesn't
/// fit, Score returns false without touching any candidate and the planner keeps the heuristic.
/// </summary>
public sealed class LearnedScorer : ICandidateScorer
{
    sealed class Scratch
    {
        public float[] Input = new float[256 * Features.Count];
        public float[] Output = new float[256];
        public readonly Mlp.Workspace Ws = new();

        public void Ensure(int rows)
        {
            if (Output.Length >= rows) return;
            int cap = Math.Max(256, (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)rows));
            Input = new float[cap * Features.Count];
            Output = new float[cap];
        }
    }

    readonly ThreadLocal<Scratch> scratch = new(() => new Scratch());
    readonly object statsLock = new();
    long calls, skipped, failed, rowsTotal;
    double totalMs, maxMs;
    double msPerRow; // running average (EMA) of inference cost per row, from calls with >= 32 rows

    public Mlp Model { get; }
    public string Name => "learned";
    public bool NeedsFeatures => true;

    /// <summary>Multiplier on the estimated inference time before comparing against the remaining budget.</summary>
    public double SafetyFactor { get; set; } = 3.0;
    /// <summary>Fixed per-call allowance (gather, dispatch, timer noise) added to the estimate, in ms.</summary>
    public double FixedOverheadMs { get; set; } = 0.5;

    public LearnedScorer(Mlp model)
    {
        if (model.InputCount != Features.Count)
            throw new ArgumentException($"Model expects {model.InputCount} inputs; Features.Count is {Features.Count}.", nameof(model));
        Model = model;
        // Warm up (JIT, Accelerate init, buffers) and seed the per-row cost estimate.
        const int warmRows = 1024;
        var x = new float[warmRows * Features.Count];
        var o = new float[warmRows];
        var ws = model.CreateWorkspace(warmRows);
        double best = double.MaxValue;
        for (int i = 0; i < 20; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            model.Predict(x, warmRows, o, ws);
            best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
        }
        // Use a pessimistic seed (2x best warm time) until real measurements arrive.
        msPerRow = Math.Max(1e-5, 2 * best / warmRows);
    }

    public static LearnedScorer Load(string path) => new(Mlp.Load(path));

    public double EstimateMs(int rows) => FixedOverheadMs + rows * Volatile.Read(ref msPerRow) * SafetyFactor;

    public bool Score(Span<Candidate> candidates, float[] features, double remainingBudgetMs)
    {
        int n = candidates.Length;
        if (n == 0) return true;
        if (EstimateMs(n) > remainingBudgetMs)
        {
            Interlocked.Increment(ref skipped);
            return false;
        }

        long t0 = Stopwatch.GetTimestamp();
        var s = scratch.Value!;
        s.Ensure(n);
        int d = Features.Count;
        for (int i = 0; i < n; i++)
        {
            int off = candidates[i].FeatureRow * d;
            if ((uint)off > (uint)(features.Length - d)) { Interlocked.Increment(ref failed); return false; }
            features.AsSpan(off, d).CopyTo(s.Input.AsSpan(i * d, d));
        }
        Model.Predict(s.Input, n, s.Output, s.Ws);
        for (int i = 0; i < n; i++)
            if (!float.IsFinite(s.Output[i])) { Interlocked.Increment(ref failed); return false; }
        for (int i = 0; i < n; i++) candidates[i].Score = s.Output[i];

        double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        lock (statsLock)
        {
            calls++;
            rowsTotal += n;
            totalMs += ms;
            if (ms > maxMs) maxMs = ms;
            if (n >= 32) msPerRow = 0.9 * msPerRow + 0.1 * (ms / n);
        }
        return true;
    }

    public string Stats
    {
        get
        {
            lock (statsLock)
            {
                double mean = calls > 0 ? totalMs / calls : 0;
                double avgRows = calls > 0 ? (double)rowsTotal / calls : 0;
                return $"learned: calls={calls} meanMs={mean:F3} maxMs={maxMs:F3} avgRows={avgRows:F0} " +
                       $"skippedForBudget={skipped} failed={failed} usPerRow={msPerRow * 1000:F3} backend={Blas.Backend}";
            }
        }
    }
}
