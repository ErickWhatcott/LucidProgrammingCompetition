using Lpc.Client;
using Lpc.Strategies;
using Lpc.Strategies.Planner;

namespace Lpc.Training;

/// <summary>
/// Learn from what every team does, not just our engines. Wraps a non-ML opponent and runs a shadow
/// planner on the same view to compute the candidates our engines would consider. The choice the
/// opponent actually made is inferred from what happened next (the deposit it mined, or going home to
/// deposit), and labelled with the gold it actually banked per turn, so its play becomes value rows and
/// imitation examples in the shared pool.
/// </summary>
public sealed class ObservedBot(IMarsStrategy inner, ITripRecorder trips, IDecisionRecorder decisions, int sampleEvery = 3) : IMarsStrategy
{
    sealed class Pending
    {
        public int Turn;
        public required float[] Flat;
        public int Rows;
        public required CandidateKind[] Kinds;
        public required int[] Cells;
        public int Chosen = -1;
    }

    readonly PlannerStrategy shadow = new("shadow", new PlannerConfig(), HeuristicScorer.Instance);
    readonly Dictionary<string, List<Pending>> pending = [];
    readonly Dictionary<string, (Coordinate Pos, long Gold, long Iron)> last = [];
    readonly List<(Pending P, float Return)> finished = [];
    int turn;

    public string Name => inner.Name;

    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity)
    {
        var actions = inner.CalculateActions(state, identity);
        var view = state.View;
        if (view == null) return actions;
        turn = view.Turn;
        int width = view.Map.Width;
        var bases = view.Bases.Where(b => b.TeamId == identity.TeamId).Select(b => b.Position).ToHashSet();
        foreach (var u in view.Units)
        {
            if (u.TeamId != identity.TeamId) continue;
            long gold = u.Inventory?.Gold ?? 0, iron = u.Inventory?.Iron ?? 0;
            if (last.TryGetValue(u.UnitId, out var prev) && pending.TryGetValue(u.UnitId, out var list) && list.Count > 0)
            {
                if (u.Position == prev.Pos && gold + iron > prev.Gold + prev.Iron)
                {
                    // Mined here: decisions still open chose this deposit (if it was among their options).
                    int cell = u.Position.Y * width + u.Position.X;
                    foreach (var p in list)
                        if (p.Chosen < 0)
                            for (int j = 0; j < p.Rows; j++)
                                if (p.Kinds[j] == CandidateKind.Mine && p.Cells[j] == cell) { p.Chosen = j; break; }
                }
                else if (gold + iron == 0 && prev.Gold + prev.Iron > 0 && bases.Contains(u.Position))
                {
                    // Deposited: anything still open was a trip home; label everything with the banked value.
                    float value = prev.Gold + prev.Iron * 0.25f;
                    foreach (var p in list)
                    {
                        if (p.Chosen < 0) p.Chosen = Array.IndexOf(p.Kinds, CandidateKind.Return);
                        if (p.Chosen >= 0) finished.Add((p, value / Math.Max(1, turn - p.Turn)));
                    }
                    list.Clear();
                }
            }
            last[u.UnitId] = (u.Position, gold, iron);
        }

        // The shadow updates its memory every turn; candidates are sampled to bound cost and data volume.
        shadow.CandidateObserver = turn % sampleEvery == 0 ? Observe : null;
        shadow.CalculateActions(state, identity);
        return actions;
    }

    void Observe(string unitId, IReadOnlyList<(CandidateKind Kind, int Cell, float[] Row)> candidates)
    {
        var flat = new float[candidates.Count * Features.Count];
        for (int j = 0; j < candidates.Count; j++) candidates[j].Row.CopyTo(flat, j * Features.Count);
        if (!pending.TryGetValue(unitId, out var list)) pending[unitId] = list = [];
        if (list.Count >= 6) list.RemoveAt(0);
        list.Add(new Pending
        {
            Turn = turn, Flat = flat, Rows = candidates.Count,
            Kinds = candidates.Select(c => c.Kind).ToArray(), Cells = candidates.Select(c => c.Cell).ToArray(),
        });
    }

    public void OnMatchEnd(GameState<MarsView>? finalState, Identity identity)
    {
        inner.OnMatchEnd(finalState, identity);
        foreach (var list in pending.Values)
            foreach (var p in list)
                if (p.Chosen >= 0) finished.Add((p, 0)); // never banked
        var scores = finalState?.View?.Scores ?? [];
        long gold = scores.GetValueOrDefault(identity.TeamId);
        int place = 1 + scores.Count(s => s.Key != identity.TeamId && s.Value > gold);
        var records = new List<DecisionRecord>(finished.Count);
        foreach (var (p, ret) in finished)
        {
            trips.Add(new TripRecord(p.Flat.AsSpan(p.Chosen * Features.Count, Features.Count).ToArray(), ret, "sim"));
            records.Add(new DecisionRecord(p.Flat, p.Rows, p.Chosen, float.NaN, ret, inner.Name, identity.TeamId, p.Turn,
                gold, place, Math.Max(1, scores.Count), "sim"));
        }
        decisions.AddRange(records);
        finished.Clear();
        pending.Clear();
    }
}
