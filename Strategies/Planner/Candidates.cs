namespace Lpc.Strategies.Planner;

public enum CandidateKind : byte { Mine, Return, Explore }

/// <summary>One option for one worker: go mine a deposit, go bank the load, or explore.</summary>
public struct Candidate
{
    public CandidateKind Kind;
    public int Worker;        // index into the planner's worker list
    public int Cell;          // target tile (-1 for Return: nearest base)
    public float Travel;      // turns until the worker is ready on the target
    public float ReturnTurns; // turns from target back to the nearest base
    public int Minable;       // kg we expect to mine there
    public float ResourceValue;
    public float CarriedValue;
    public float Heuristic;   // value per turn under the hand-written model
    public float Score;       // what the active scorer says (heuristic or learned)
    public int FeatureRow;    // row in the feature buffer

    public static float Rate(float carriedValue, int minable, float resourceValue, float travel, float ret) =>
        (carriedValue + minable * resourceValue) / (travel + minable + ret + 1f);
}

/// <summary>
/// Shared feature encoding for the learned scorer and for the training records written during
/// simulation and scrimmages. Changing this layout invalidates saved models.
/// </summary>
public static class Features
{
    public const int Count = 28;
    public const int Version = 1;

    public static readonly string[] Names =
    [
        "kind_mine", "kind_return", "kind_explore", "travel", "mine_turns", "return_turns", "total_turns",
        "minable", "amount", "is_gold", "in_center", "carried_gold", "carried_iron", "free_capacity",
        "turns_left", "turn", "enemies_on_tile", "enemies_near", "friendlies_near", "claimed_by_us",
        "claimed_by_enemy", "gold_nearby_count", "gold_nearby_amount", "info_age", "heuristic_rate",
        "iron_shadow", "explore_value", "team_workers",
    ];

    public struct Context
    {
        public float CarriedGold, CarriedIron, FreeCapacity, TurnsLeft, Turn, IronShadow, TeamWorkers;
    }

    public struct Target
    {
        public float Amount, IsGold, InCenter, EnemiesOnTile, EnemiesNear, FriendliesNear,
            ClaimedByUs, ClaimedByEnemy, GoldNearbyCount, GoldNearbyAmount, InfoAge, ExploreValue;
    }

    public static void Fill(Span<float> row, in Candidate c, in Context ctx, in Target t)
    {
        row[0] = c.Kind == CandidateKind.Mine ? 1 : 0;
        row[1] = c.Kind == CandidateKind.Return ? 1 : 0;
        row[2] = c.Kind == CandidateKind.Explore ? 1 : 0;
        row[3] = Math.Min(c.Travel, 80) / 30f;
        row[4] = c.Minable / 8f;
        row[5] = Math.Min(c.ReturnTurns, 80) / 30f;
        row[6] = Math.Min(c.Travel + c.Minable + c.ReturnTurns + 1, 160) / 60f;
        row[7] = c.Minable * c.ResourceValue / 8f;
        row[8] = t.Amount / 18f;
        row[9] = t.IsGold;
        row[10] = t.InCenter;
        row[11] = ctx.CarriedGold / 8f;
        row[12] = ctx.CarriedIron / 8f;
        row[13] = ctx.FreeCapacity / 8f;
        row[14] = ctx.TurnsLeft / 480f;
        row[15] = ctx.Turn / 480f;
        row[16] = t.EnemiesOnTile / 3f;
        row[17] = t.EnemiesNear / 5f;
        row[18] = t.FriendliesNear / 5f;
        row[19] = t.ClaimedByUs;
        row[20] = t.ClaimedByEnemy;
        row[21] = t.GoldNearbyCount / 5f;
        row[22] = t.GoldNearbyAmount / 40f;
        row[23] = Math.Min(t.InfoAge, 100) / 50f;
        row[24] = c.Heuristic;
        row[25] = ctx.IronShadow;
        row[26] = t.ExploreValue;
        row[27] = ctx.TeamWorkers / 10f;
    }
}

/// <summary>Turns candidate features into scores. The heuristic scorer just returns the hand-written rate.</summary>
public interface ICandidateScorer
{
    string Name { get; }
    bool NeedsFeatures { get; }
    /// <summary>
    /// True for policies (ppo): Score holds logits and the planner samples each worker's target from
    /// softmax(logit / Temperature) over the candidates still available, recording log-probabilities.
    /// False: Score is a value and the planner takes the best (worker, target) pairs greedily.
    /// </summary>
    bool Stochastic => false;
    double Temperature => 1.0;
    /// <summary>Writes Score for each candidate. Returns false if it could not finish within the budget.</summary>
    bool Score(Span<Candidate> candidates, float[] features, double remainingBudgetMs);
}

public sealed class HeuristicScorer : ICandidateScorer
{
    public static readonly HeuristicScorer Instance = new();
    public string Name => "heuristic";
    public bool NeedsFeatures => false;

    public bool Score(Span<Candidate> candidates, float[] features, double remainingBudgetMs)
    {
        foreach (ref var c in candidates) c.Score = c.Heuristic;
        return true;
    }
}
