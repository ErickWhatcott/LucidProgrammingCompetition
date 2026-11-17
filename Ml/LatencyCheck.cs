using System.Diagnostics;

namespace Lpc.Ml;

public static class LatencyCheck
{
    /// <summary>
    /// Times <paramref name="iterations"/> batched forward passes of <paramref name="rows"/> random rows
    /// (after a warm-up) using the same reusable-workspace path as <see cref="LearnedScorer"/>.
    /// </summary>
    public static (double meanMs, double p99Ms, double maxMs) Measure(Mlp model, int rows, int iterations)
    {
        rows = Math.Max(1, rows);
        iterations = Math.Max(1, iterations);
        var rng = new Random(123);
        var x = new float[rows * model.InputCount];
        for (int i = 0; i < x.Length; i++) x[i] = (float)rng.NextDouble();
        var o = new float[rows];
        var ws = model.CreateWorkspace(rows);
        for (int i = 0; i < 50; i++) model.Predict(x, rows, o, ws);

        var times = new double[iterations];
        for (int i = 0; i < iterations; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            model.Predict(x, rows, o, ws);
            times[i] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        }
        Array.Sort(times);
        int p99 = Math.Min(iterations - 1, (int)Math.Ceiling(iterations * 0.99) - 1);
        return (times.Average(), times[p99], times[^1]);
    }
}
