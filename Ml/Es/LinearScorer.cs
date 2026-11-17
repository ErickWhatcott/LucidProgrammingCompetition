using System.Diagnostics;
using Lpc.Strategies.Planner;

namespace Lpc.Ml.Es;

/// <summary>
/// The 'es' engine's candidate scorer: Score = dot(weights, feature row). Weights are evolved by
/// CMA-ES together with the planner knobs (see <see cref="EsGenome"/>); weights = e24 reproduces the
/// heuristic exactly. Thread-safe (weights are read-only, scratch is per thread); about tens of ns per row.
/// Like the other scorers it returns false without touching candidates if it can't fit the budget,
/// a feature row is out of range, or a score is not finite.
/// </summary>
public sealed class LinearScorer : ICandidateScorer
{
    [ThreadStatic] static float[]? scratch;

    readonly float[] weights;
    readonly object statsLock = new();
    long calls, skipped, failed, rowsTotal;
    double totalMs, maxMs;

    public LinearScorer(float[] weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (weights.Length != Features.Count)
            throw new ArgumentException($"Expected {Features.Count} weights, got {weights.Length}.", nameof(weights));
        this.weights = (float[])weights.Clone();
    }

    public ReadOnlySpan<float> Weights => weights;
    public string Name => "es";
    public bool NeedsFeatures => true;
    public bool Stochastic => false;
    public double Temperature => 1.0;

    /// <summary>Fixed allowance plus a pessimistic 1 us/row; well under any sane budget.</summary>
    public static double EstimateMs(int rows) => 0.05 + rows * 0.001;

    public bool Score(Span<Candidate> candidates, float[] features, double remainingBudgetMs)
    {
        int n = candidates.Length;
        if (n == 0) return true;
        if (features is null) { Interlocked.Increment(ref failed); return false; }
        if (EstimateMs(n) > remainingBudgetMs) { Interlocked.Increment(ref skipped); return false; }

        long t0 = Stopwatch.GetTimestamp();
        var s = scratch;
        if (s == null || s.Length < n) scratch = s = new float[Math.Max(256, (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)n))];
        const int d = Features.Count;
        ReadOnlySpan<float> wv = weights;
        for (int i = 0; i < n; i++)
        {
            long off = (long)candidates[i].FeatureRow * d; // long: no int overflow into a valid offset
            if (off < 0 || off > features.Length - d) { Interlocked.Increment(ref failed); return false; }
            var row = features.AsSpan((int)off, d);
            float v = 0;
            for (int j = 0; j < d; j++) v += wv[j] * row[j];
            if (!float.IsFinite(v)) { Interlocked.Increment(ref failed); return false; }
            s[i] = v;
        }
        for (int i = 0; i < n; i++) candidates[i].Score = s[i];

        double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        lock (statsLock)
        {
            calls++; rowsTotal += n; totalMs += ms;
            if (ms > maxMs) maxMs = ms;
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
                return $"es: calls={calls} meanMs={mean:F4} maxMs={maxMs:F4} avgRows={avgRows:F0} skippedForBudget={skipped} failed={failed}";
            }
        }
    }
}
