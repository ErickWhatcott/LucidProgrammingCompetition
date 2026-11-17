using Lpc.Strategies.Planner;

namespace Lpc.Ml.Es;

/// <summary>
/// Maps a CMA-ES point x in normalized space [0,1]^Dimension to (PlannerConfig, linear scorer weights)
/// and back. Each coordinate maps linearly onto its range; values outside [0,1] are clamped here (the
/// optimizer keeps its samples in the same box). Integer knobs are rounded.
///
/// Weight scaling: the planner's use of Score is invariant to a positive rescale, and the heuristic
/// rate is only ~0.05..0.5 gold/turn. So heuristic_rate's weight stays positive ([0.25, 1.75], 0.5 -> 1)
/// and every other weight is a small additive correction ([-0.25, 0.25], 0.5 -> 0). (The previous
/// [-2, 2] range gave a per-weight std of 0.8 at sigma 0.2, drowning the heuristic in noise and letting
/// its weight go negative, i.e. inverting the planner.)
/// </summary>
public static class EsGenome
{
    public const double CorrectionRange = 0.25, HeuristicWeightMin = 0.25, HeuristicWeightMax = 1.75;

    /// <summary>Decoded range of feature weight <paramref name="j"/>.</summary>
    public static (double Min, double Max) WeightBounds(int j) =>
        j == HeuristicFeature ? (HeuristicWeightMin, HeuristicWeightMax) : (-CorrectionRange, CorrectionRange);

    readonly record struct Knob(string Name, double Min, double Max, bool IsInt,
        Func<PlannerConfig, double> Get, Action<PlannerConfig, double> Set);

    static readonly Knob[] Knobs =
    [
        new("StopInvestFraction", 0.3, 0.75, false, c => c.StopInvestFraction, (c, v) => c.StopInvestFraction = v),
        new("MaxRecruits", 2, 14, true, c => c.MaxRecruits, (c, v) => c.MaxRecruits = (int)Math.Round(v)),
        new("MaxBases", 1, 6, true, c => c.MaxBases, (c, v) => c.MaxBases = (int)Math.Round(v)),
        new("IronValue", 0.4, 2.5, false, c => c.IronValue, (c, v) => c.IronValue = v),
        new("RecruitSafety", 0.8, 3.0, false, c => c.RecruitSafety, (c, v) => c.RecruitSafety = v),
        new("BaseSafety", 0.6, 3.0, false, c => c.BaseSafety, (c, v) => c.BaseSafety = v),
        new("PriorRate", 0.2, 0.6, false, c => c.PriorRate, (c, v) => c.PriorRate = v),
        new("CenterBaseBonus", 0, 60, false, c => c.CenterBaseBonus, (c, v) => c.CenterBaseBonus = v),
        new("ClaimBonus", 0, 0.5, false, c => c.ClaimBonus, (c, v) => c.ClaimBonus = v),
        new("TradeSafety", 0.8, 4.0, false, c => c.TradeSafety, (c, v) => c.TradeSafety = v),
        new("Hysteresis", 0, 0.4, false, c => c.Hysteresis, (c, v) => c.Hysteresis = v),
        new("ExploreRate", 0.1, 0.6, false, c => c.ExploreRate, (c, v) => c.ExploreRate = v),
        new("CenterExploreBonus", 0, 1.5, false, c => c.CenterExploreBonus, (c, v) => c.CenterExploreBonus = v),
        new("ContestPenalty", 0, 6, false, c => c.ContestPenalty, (c, v) => c.ContestPenalty = v),
        new("StaleDiscount", 0.4, 1.0, false, c => c.StaleDiscount, (c, v) => c.StaleDiscount = v),
        new("HiddenCost", 1.2, 2.2, false, c => c.HiddenCost, (c, v) => c.HiddenCost = v),
        // Game-phase knobs: late-game ends of the blended settings, plus the opening center rush.
        new("ExploreRateLate", 0.1, 0.6, false, c => c.ExploreRateLate ?? c.ExploreRate, (c, v) => c.ExploreRateLate = v),
        new("TradeSafetyLate", 0.8, 4.0, false, c => c.TradeSafetyLate ?? c.TradeSafety, (c, v) => c.TradeSafetyLate = v),
        new("BaseSafetyLate", 0.6, 3.0, false, c => c.BaseSafetyLate ?? c.BaseSafety, (c, v) => c.BaseSafetyLate = v),
        new("RecruitSafetyLate", 0.8, 3.0, false, c => c.RecruitSafetyLate ?? c.RecruitSafety, (c, v) => c.RecruitSafetyLate = v),
        new("CenterRushBonus", 0, 2.0, false, c => c.CenterRushBonus, (c, v) => c.CenterRushBonus = v),
        new("CenterRushTurns", 20, 250, false, c => c.CenterRushTurns, (c, v) => c.CenterRushTurns = v),
        // Opponent awareness.
        new("RaceDiscount", 0, 1.0, false, c => c.RaceDiscount, (c, v) => c.RaceDiscount = v),
        new("CrowdPenalty", 0, 0.4, false, c => c.CrowdPenalty, (c, v) => c.CrowdPenalty = v),
        new("BehindCenterBonus", 0, 1.5, false, c => c.BehindCenterBonus, (c, v) => c.BehindCenterBonus = v),
        new("InvestCommit", 0, 0.6, false, c => c.InvestCommit, (c, v) => c.InvestCommit = v),
    ];

    /// <summary>Number of evolved PlannerConfig knobs (they come first in x).</summary>
    public static int KnobCount => Knobs.Length;
    /// <summary>KnobCount + Features.Count.</summary>
    public static int Dimension => Knobs.Length + Features.Count;
    /// <summary>Feature index whose weight starts at +1 (heuristic_rate).</summary>
    public const int HeuristicFeature = 24;

    /// <summary>Coordinate names: knob names, then "w_" + feature name.</summary>
    public static string[] Names =>
        [.. Knobs.Select(k => k.Name), .. Features.Names.Select(f => "w_" + f)];

    static double Clamp01(double v) => double.IsFinite(v) ? Math.Clamp(v, 0, 1) : 0.5;

    public static double[] Encode(PlannerConfig config, float[] weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        if (weights.Length != Features.Count) throw new ArgumentException($"Expected {Features.Count} weights.", nameof(weights));
        var x = new double[Dimension];
        for (int i = 0; i < Knobs.Length; i++)
        {
            var k = Knobs[i];
            x[i] = Clamp01((k.Get(config) - k.Min) / (k.Max - k.Min));
        }
        for (int j = 0; j < Features.Count; j++)
        {
            var (lo, hi) = WeightBounds(j);
            x[Knobs.Length + j] = Clamp01((weights[j] - lo) / (hi - lo));
        }
        return x;
    }

    /// <summary>Clamps x into [0,1], applies the knobs to a clone of <paramref name="template"/> (other settings kept).</summary>
    public static (PlannerConfig Config, float[] Weights) Decode(double[] x, PlannerConfig template)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(template);
        if (x.Length != Dimension) throw new ArgumentException($"Expected {Dimension} coordinates.", nameof(x));
        var cfg = template.Clone();
        for (int i = 0; i < Knobs.Length; i++)
        {
            var k = Knobs[i];
            k.Set(cfg, k.Min + Clamp01(x[i]) * (k.Max - k.Min));
        }
        var w = new float[Features.Count];
        for (int j = 0; j < w.Length; j++)
        {
            var (lo, hi) = WeightBounds(j);
            w[j] = (float)(lo + Clamp01(x[Knobs.Length + j]) * (hi - lo));
        }
        return (cfg, w);
    }

    /// <summary>Weights that reproduce the hand-written scorer: 0 everywhere, +1 on heuristic_rate.</summary>
    public static float[] HeuristicWeights()
    {
        var w = new float[Features.Count];
        w[HeuristicFeature] = 1;
        return w;
    }

    /// <summary>Starting mean: the defaults' knobs plus <see cref="HeuristicWeights"/>, so generation 0 ~ the hand-written planner.</summary>
    public static double[] InitialMean(PlannerConfig defaults) => Encode(defaults, HeuristicWeights());

    /// <summary>One line per evolved knob/weight, for logs.</summary>
    public static string Describe(PlannerConfig config, float[] weights)
    {
        var parts = new List<string>();
        foreach (var k in Knobs) parts.Add($"{k.Name}={k.Get(config):0.###}");
        for (int j = 0; j < weights.Length; j++)
            if (Math.Abs(weights[j]) >= 0.005f) parts.Add($"w_{Features.Names[j]}={weights[j]:0.###}");
        return string.Join(" ", parts);
    }
}
