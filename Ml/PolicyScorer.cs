using System.Diagnostics;
using Lpc.Strategies.Planner;

namespace Lpc.Ml;

/// <summary>
/// "ppo" engine: writes <see cref="PolicyNet"/> logits into Candidate.Score. Stochastic: the planner samples each
/// worker's target from softmax(Score / Temperature) over the rows it offered and records
/// <see cref="PolicyNet.LogProb"/> of the pick as BehaviorLogProb.
///
/// Thread-safe: the net is read-only during inference and scratch is per thread. Latency guard as in
/// <see cref="LearnedScorer"/>: if the estimated cost doesn't fit the remaining budget, Score returns false
/// without touching any candidate.
/// </summary>
public sealed class PolicyScorer : ICandidateScorer
{
    sealed class Scratch
    {
        public float[] Input = new float[256 * Features.Count];
        public float[] Output = new float[256];

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
    readonly double temperature;
    long calls, skipped, failed, rowsTotal;
    double totalMs, maxMs;
    double msPerRow;

    public PolicyNet Net { get; }
    public string Name => "ppo";
    public bool NeedsFeatures => true;
    public bool Stochastic => true;
    public double Temperature => temperature;

    public double SafetyFactor { get; set; } = 3.0;
    public double FixedOverheadMs { get; set; } = 0.5;

    public PolicyScorer(PolicyNet net, double temperature = 1.0)
    {
        if (net.InputCount != Features.Count)
            throw new ArgumentException($"Policy expects {net.InputCount} inputs; Features.Count is {Features.Count}.", nameof(net));
        if (!(temperature > 0) || !double.IsFinite(temperature)) throw new ArgumentOutOfRangeException(nameof(temperature));
        Net = net;
        this.temperature = temperature;
        const int warmRows = 1024;
        var x = new float[warmRows * Features.Count];
        var o = new float[warmRows];
        double best = double.MaxValue;
        for (int i = 0; i < 20; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            net.Logits(x, warmRows, o);
            best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
        }
        msPerRow = Math.Max(1e-5, 2 * best / warmRows);
    }

    public static PolicyScorer Load(string path, double temperature = 1.0) => new(PolicyNet.Load(path), temperature);

    /// <summary>Same as <see cref="PolicyNet.LogProb"/>.</summary>
    public static float LogProb(ReadOnlySpan<float> logits, int chosen, double temperature) => PolicyNet.LogProb(logits, chosen, temperature);

    public double EstimateMs(int rows) => FixedOverheadMs + rows * Volatile.Read(ref msPerRow) * SafetyFactor;

    public bool Score(Span<Candidate> candidates, float[] features, double remainingBudgetMs)
    {
        int n = candidates.Length;
        if (n == 0) return true;
        if (!(EstimateMs(n) <= remainingBudgetMs)) // also refuses a NaN budget
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
            long off = (long)candidates[i].FeatureRow * d; // long: no int overflow on a corrupt FeatureRow
            if (off < 0 || off > features.Length - d) { Interlocked.Increment(ref failed); return false; }
            features.AsSpan((int)off, d).CopyTo(s.Input.AsSpan(i * d, d));
        }
        Net.Logits(s.Input, n, s.Output);
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
                return $"ppo: calls={calls} meanMs={mean:F3} maxMs={maxMs:F3} avgRows={avgRows:F0} T={temperature:G3} " +
                       $"skippedForBudget={skipped} failed={failed} usPerRow={msPerRow * 1000:F3} backend={Blas.Backend}";
            }
        }
    }
}
