using System.Diagnostics;
using Lpc.Strategies.Planner;

namespace Lpc.Ml;

/// <summary>
/// Scores planner candidates with a <see cref="GbdtModel"/> (predicted gold-equivalent banked per turn).
/// Thread-safe: the model is read-only and scratch buffers are per thread.
/// Latency guard: estimated cost = FixedOverheadMs + rows x running-average ms/row x SafetyFactor; if that
/// exceeds the remaining budget, Score returns false without touching any candidate (planner keeps heuristic).
/// </summary>
public sealed class GbdtScorer : ICandidateScorer
{
    [ThreadStatic] static float[]? tInput;
    [ThreadStatic] static float[]? tOutput;

    readonly object statsLock = new();
    long calls, skipped, failed, rowsTotal;
    double totalMs, maxMs;
    double msPerRow;

    public GbdtModel Model { get; }
    public string Name => "gbdt";
    public bool NeedsFeatures => true;
    public bool Stochastic => false;
    public double Temperature => 1.0;
    public double SafetyFactor { get; set; } = 3.0;
    public double FixedOverheadMs { get; set; } = 0.3;

    public GbdtScorer(GbdtModel model)
    {
        Model = model;
        if (Volatile.Read(ref model.WarmMsPerRow) <= 0)
        {
            // Warm up once per model (JIT + caches) and seed the per-row cost pessimistically (2x best).
            const int warmRows = 1024;
            var x = new float[warmRows * Features.Count];
            var rng = new Random(5);
            for (int i = 0; i < x.Length; i++) x[i] = (float)rng.NextDouble();
            var o = new float[warmRows];
            double best = double.MaxValue;
            for (int i = 0; i < 5; i++)
            {
                long t0 = Stopwatch.GetTimestamp();
                model.Predict(x, warmRows, o);
                best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            }
            Volatile.Write(ref model.WarmMsPerRow, Math.Max(1e-5, 2 * best / warmRows));
        }
        msPerRow = model.WarmMsPerRow;
    }

    public static GbdtScorer Load(string path) => new(GbdtModel.Load(path));

    public double EstimateMs(int rows) => FixedOverheadMs + rows * Volatile.Read(ref msPerRow) * SafetyFactor;

    public bool Score(Span<Candidate> candidates, float[] features, double remainingBudgetMs)
    {
        int n = candidates.Length;
        if (n == 0) return true;
        if (EstimateMs(n) > remainingBudgetMs) { Interlocked.Increment(ref skipped); return false; }

        long t0 = Stopwatch.GetTimestamp();
        int d = Features.Count;
        if (tOutput is null || tOutput.Length < n)
        {
            int cap = Math.Max(256, (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)n));
            tInput = new float[cap * d];
            tOutput = new float[cap];
        }
        var input = tInput!; var output = tOutput;
        for (int i = 0; i < n; i++)
        {
            int off = candidates[i].FeatureRow * d;
            if (off < 0 || off > features.Length - d) { Interlocked.Increment(ref failed); return false; }
            features.AsSpan(off, d).CopyTo(input.AsSpan(i * d, d));
        }
        Model.Predict(input, n, output);
        for (int i = 0; i < n; i++)
            if (!float.IsFinite(output[i])) { Interlocked.Increment(ref failed); return false; }
        for (int i = 0; i < n; i++) candidates[i].Score = output[i];

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
                return $"gbdt: calls={calls} meanMs={mean:F3} maxMs={maxMs:F3} avgRows={avgRows:F0} " +
                       $"skippedForBudget={skipped} failed={failed} usPerRow={msPerRow * 1000:F3} trees={Model.TreeCount}";
            }
        }
    }
}
