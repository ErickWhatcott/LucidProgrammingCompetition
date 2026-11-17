namespace Lpc.Strategies.Planner;

/// <summary>
/// One worker decision with everything every engine needs to learn from it:
/// - value scorers (mlp, gbdt) use the chosen row + Return (same as TripRecord),
/// - imitation (ppo stage 1) uses all candidate rows + Chosen, weighted toward strong seats (FinalPlace),
/// - PPO (ppo stage 2) additionally uses BehaviorLogProb from its own stochastic seats.
/// Records are flushed at match end, when FinalGold/FinalPlace are known.
/// </summary>
public sealed record DecisionRecord(
    float[] Candidates,     // Rows x Features.Count, row-major; only candidates the worker could actually pick
    int Rows,
    int Chosen,             // index into Candidates
    float BehaviorLogProb,  // log pi(chosen) under the policy that acted; NaN for deterministic scorers
    float Return,           // gold-equivalent banked per turn from this decision to the next deposit (0 if never banked)
    string Model,           // engine that acted: mlp | gbdt | ppo | es | heuristic
    string Team,            // team id within the match
    int Turn,
    float FinalGold,        // that team's banked gold at match end
    int FinalPlace,         // 1 = winner
    int TeamCount,
    string Src);            // "sim" or "live"

public interface IDecisionRecorder
{
    /// <summary>Thread-safe; called once per match per team with all of that team's decisions.</summary>
    void AddRange(IReadOnlyList<DecisionRecord> records);
}
