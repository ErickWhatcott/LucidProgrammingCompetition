using System.Text.Json;

namespace Lpc.Strategies.Planner;

/// <summary>Tunable knobs for the scored planner. `lpc tune` searches over these.</summary>
public sealed class PlannerConfig
{
    // Investment
    public double StopInvestFraction { get; set; } = 0.55;  // no new workers/bases after this share of the match
    public int MaxRecruits { get; set; } = 10;
    public int MaxBases { get; set; } = 4;
    public double IronGoldCost { get; set; } = 1.0;         // gold-equivalent cost of 1 kg iron when weighing investments
    public double IronValue { get; set; } = 1.1;            // iron's value to miners while an investment waits on iron
    public double RecruitSafety { get; set; } = 1.6;        // required benefit / cost for a new worker
    public double BaseSafety { get; set; } = 1.4;           // required benefit / cost for a new base
    public double PriorRate { get; set; } = 0.35;           // prior kg gold / worker / turn before we have data
    public double CenterBaseBonus { get; set; } = 20;       // gold-equivalent for a base near the rich center
    public double ClaimBonus { get; set; } = 0.15;          // per kg of unclaimed gold our base would claim
    public double TradeSafety { get; set; } = 1.5;          // benefit / gold spent needed to trade gold for iron
    public double InvestCommit { get; set; } = 0.2;         // while saving for a worker (or base), the other must beat it by this share

    // Opponent awareness (0 = off). Applied to mine targets after scoring.
    public double RaceDiscount { get; set; } = 0;           // share of score lost when a visible enemy reaches the deposit first and can drain it
    public double CrowdPenalty { get; set; } = 0;           // score share lost per enemy worker within 4 tiles of the deposit
    public double BehindCenterBonus { get; set; } = 0;      // extra share on center gold when trailing the leader (full at 150 kg behind)

    // Game phase. Optional "late" values: the planner blends linearly from the early value at turn 0 to the
    // late value at the last turn (null = same as early). CMA-ES evolves both ends.
    public double? ExploreRateLate { get; set; }
    public double? TradeSafetyLate { get; set; }
    public double? BaseSafetyLate { get; set; }
    public double? RecruitSafetyLate { get; set; }
    public double CenterRushBonus { get; set; } = 0;        // opening: extra value on rich-center gold, fading out...
    public double CenterRushTurns { get; set; } = 100;      // ...linearly by this turn

    public static double Phase(double early, double? late, double t) => late is { } l ? early + (l - early) * t : early;

    // Targeting
    public bool Coordinate { get; set; } = true;            // reserve deposit stock across workers
    public double Hysteresis { get; set; } = 0.15;          // bonus for keeping last turn's target
    public double ExploreRate { get; set; } = 0.30;
    public double CenterExploreBonus { get; set; } = 0.6;
    public double ContestPenalty { get; set; } = 3;         // kg assumed lost per enemy worker on a deposit
    public double StaleDiscount { get; set; } = 0.75;       // remembered-but-unseen deposits may be mined by others
    public int StaleAfter { get; set; } = 25;

    // Movement and timing
    public int EndgameMargin { get; set; } = 1;
    public double HiddenCost { get; set; } = 1.6;           // expected cost of entering an unseen tile
    public int ObstacleCost { get; set; } = 3;              // obstacleTraversalCost (entry + 2 cooldown turns)
    public double TurnBudgetMs { get; set; } = 100;         // hard per-turn decision budget

    public PlannerConfig Clone() => (PlannerConfig)MemberwiseClone();

    static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };
    public static PlannerConfig Load(string path) => JsonSerializer.Deserialize<PlannerConfig>(File.ReadAllText(path), JsonOptions)!;
    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        File.WriteAllText(path, JsonSerializer.Serialize(this, JsonOptions));
    }

    /// <summary>The built-in Intermediate practice bot as described in the rules, expressed as planner settings.</summary>
    public static PlannerConfig Intermediate() => new()
    {
        StopInvestFraction = 0.5,
        MaxRecruits = 3,
        MaxBases = 2,
        Coordinate = false,
        Hysteresis = 0.05,
        CenterBaseBonus = 0,
        ClaimBonus = 0,
        CenterExploreBonus = 0,
    };
}
