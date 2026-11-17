using System.Diagnostics;
using Lpc.Client;

namespace Lpc.Strategies.Planner;

/// <summary>
/// The scored planner (research approach 1, upgraded): remembered map, value-per-turn targeting
/// with real path costs, assignment across workers, claim awareness, a worker/base build schedule
/// and deadline-aware end-game. The scorer is pluggable: hand-written rate (greedy mode) or the
/// learned model (ml mode).
/// </summary>
public sealed class PlannerStrategy : IMarsStrategy
{
    enum InvestmentChoice { Recruit, Base, Save }

    public string Name { get; }
    readonly PlannerConfig cfg;
    readonly ICandidateScorer scorer;
    readonly ITripRecorder? recorder;
    readonly string recordSource;
    readonly double noise;
    readonly Random rng;

    WorldMemory? mem;
    float[] enterCost = [];
    readonly PathField baseField = new();
    readonly List<PathField> fields = [];
    int[] mineBlockedUntil = [], mineFailures = [], moveBlockedUntil = [], buildBlockedUntil = [], stamp = [];
    float[] depAvail = [];
    int stampId;
    int initialWorkers = -1;
    long workerTurns;
    int planSite = -1, planWait;
    InvestmentChoice saving = InvestmentChoice.Save;
    string? planBuilder;

    sealed class WorkerState
    {
        public CandidateKind? LastKind;
        public int LastCell = -2;
        public string LastAction = "";
        public int LastActionCell = -1;
        public long LastLoad;
        public Coordinate LastPos;
        public readonly List<(float[] F, int Turn)> Pending = [];
        public readonly List<(float[] Rows, int Count, int Chosen, float LogProb, int Turn)> PendingDecisions = [];
    }
    readonly IDecisionRecorder? decisionRecorder;
    readonly List<(float[] Rows, int Count, int Chosen, float LogProb, int Turn, float Return)> finishedDecisions = [];
    Dictionary<string, long>? lastScores;
    string teamId = "";
    readonly Dictionary<string, WorkerState> workerStates = [];

    // Diagnostics
    public bool Trace { get; set; }
    /// <summary>Observe mode (shadowing another bot): receives each worker's candidates and feature rows every turn.</summary>
    public Action<string, IReadOnlyList<(CandidateKind Kind, int Cell, float[] Row)>>? CandidateObserver { get; set; }
    /// <summary>Banked gold and placement by banked gold at match end (ties on the later criteria are not resolved).</summary>
    public (long Gold, int Place, int Teams)? FinalStanding { get; private set; }
    public int Turns { get; private set; }
    public int ScorerFallbacks { get; private set; }
    public double MaxTurnMs { get; private set; }
    public string Stats => $"{Name}: turns={Turns} maxTurnMs={MaxTurnMs:F2} scorerFallbacks={ScorerFallbacks}";

    public PlannerStrategy(string name, PlannerConfig config, ICandidateScorer? scorer = null,
        ITripRecorder? recorder = null, string recordSource = "sim", double explorationNoise = 0, int seed = 0,
        IDecisionRecorder? decisionRecorder = null)
    {
        Name = name;
        cfg = config;
        this.scorer = scorer ?? HeuristicScorer.Instance;
        this.recorder = recorder;
        this.decisionRecorder = decisionRecorder;
        this.recordSource = recordSource;
        noise = explorationNoise;
        rng = new Random(seed);
    }

    struct Dep
    {
        public int Cell;
        public bool Gold, Mineable, ByUs, ByEnemy;
        public float AmountEst;
        public int EnemiesOn, EnemiesNear, GoldNearbyCount, EnemiesWithin4;
        public float GoldNearbyAmount;
        public int EnemyEta;    // turns until the nearest visible enemy with free capacity could stand on it
        public long EnemyFree;  // that enemy's free capacity
    }

    static int RecruitPrice(int recruits)
    {
        int a = 5, b = 6;
        if (recruits <= 0) return a;
        for (int i = 1; i < recruits; i++) (a, b) = (b, a + b);
        return b;
    }

    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity)
    {
        long started = Stopwatch.GetTimestamp();
        var actions = new Dictionary<string, MarsAction>();
        var view = state.View;
        if (view == null) return actions;
        string me = identity.TeamId;
        teamId = me;
        if (view.Scores != null) lastScores = view.Scores;
        mem ??= new WorldMemory(me);
        mem.Update(view);
        int w = mem.Width, h = mem.Height, n = mem.Count;
        if (enterCost.Length != n)
        {
            enterCost = new float[n];
            mineBlockedUntil = new int[n]; mineFailures = new int[n]; moveBlockedUntil = new int[n];
            buildBlockedUntil = new int[n]; stamp = new int[n]; depAvail = new float[n];
        }
        int turn = view.Turn;
        int maxTurns = state.Match.MaxTurns > 0 ? state.Match.MaxTurns : 480;
        int turnsLeft = maxTurns - turn;
        double phase = (double)turn / maxTurns;
        double rush = cfg.CenterRushBonus * Math.Max(0, 1 - turn / Math.Max(1, cfg.CenterRushTurns));
        var mine = view.Units.Where(u => u.TeamId == me).OrderBy(u => u.UnitId, StringComparer.Ordinal).ToList();
        Turns++;
        if (mine.Count == 0) return actions;
        if (initialWorkers < 0) initialWorkers = mine.Count;
        var bank = view.TeamResources?.GetValueOrDefault(me) ?? new Inventory(0, 0);

        // ---- Terrain costs (enemy bases and tiles we failed to enter are walls) ----
        for (int i = 0; i < n; i++)
        {
            float c = mem.Kind[i] switch
            {
                Tile.Unknown => (float)cfg.HiddenCost,
                Tile.Mountain or Tile.Crater => cfg.ObstacleCost,
                Tile.Void => float.PositiveInfinity,
                _ => 1f,
            };
            if (moveBlockedUntil[i] > turn) c = float.PositiveInfinity;
            var owner = mem.BaseOwnerAt[i];
            if (owner != null) c = owner == me ? 1f : float.PositiveInfinity;
            enterCost[i] = c;
        }
        var myBases = mem.MyBases.ToList();
        baseField.To(myBases.Select(b => mem.Index(b.Position)), enterCost, w, h);
        while (fields.Count < mine.Count) fields.Add(new PathField());
        // Targets more than ~45 turns away are never worth it within a 480-turn match; capping the
        // search keeps the full turn (all workers) in the low milliseconds.
        for (int k = 0; k < mine.Count; k++) fields[k].From(mem.Index(mine[k].Position), enterCost, w, h, 45);

        // ---- Learn from last turn's outcomes ----
        foreach (var unit in mine)
        {
            if (!workerStates.TryGetValue(unit.UnitId, out var ws)) continue;
            long load = Load(unit);
            int pos = mem.Index(unit.Position);
            if (ws.LastAction == "mine" && unit.Position == ws.LastPos && load == ws.LastLoad
                && load < unit.InventoryCapacity && mem.Kind[pos] is Tile.Gold or Tile.Iron && mem.Amount[pos] > 0)
            {
                if (++mineFailures[pos] >= 2) { mineBlockedUntil[pos] = turn + 25; mineFailures[pos] = 0; }
            }
            else if (ws.LastAction == "mine" && load > ws.LastLoad) mineFailures[pos] = 0;
            if (ws.LastAction == "move" && unit.Position == ws.LastPos && turn >= unit.ReadyOnTurn && ws.LastActionCell >= 0)
                moveBlockedUntil[ws.LastActionCell] = turn + 40;
            if (ws.LastAction == "build" && mem.BaseOwnerAt[pos] != me)
            {
                buildBlockedUntil[pos] = turn + 60;
                if (planSite == pos) { planSite = -1; planBuilder = null; }
            }
        }
        workerTurns += mine.Count;

        // ---- Known deposits ----
        var enemyAt = new Dictionary<int, int>();
        var friendAt = new Dictionary<int, int>();
        foreach (var u in view.Units)
        {
            var dict = u.TeamId == me ? friendAt : enemyAt;
            int i = mem.Index(u.Position);
            dict[i] = dict.GetValueOrDefault(i) + 1;
        }
        var deps = new List<Dep>();
        for (int i = 0; i < n; i++)
        {
            if (mem.Kind[i] is not (Tile.Gold or Tile.Iron) || mem.Amount[i] <= 0 || mineBlockedUntil[i] > turn) continue;
            var (mineable, byUs, byEnemy) = mem.Claim(i);
            float est = mem.Amount[i];
            if (!mem.Visible[i] && turn - mem.LastSeen[i] > cfg.StaleAfter) est *= (float)cfg.StaleDiscount;
            deps.Add(new Dep
            {
                Cell = i, Gold = mem.Kind[i] == Tile.Gold, Mineable = mineable, ByUs = byUs, ByEnemy = byEnemy,
                AmountEst = est, EnemiesOn = enemyAt.GetValueOrDefault(i), EnemiesNear = CountNear(enemyAt, i, 2),
            });
        }
        // Opponent race: the first visible enemy worker with room to mine that could reach each deposit (8-way moves).
        bool opponentAware = cfg.RaceDiscount > 0 || cfg.CrowdPenalty > 0;
        var enemyWorkers = opponentAware ? view.Units.Where(u => u.TeamId != me).ToList() : [];
        for (int a = 0; a < deps.Count; a++)
        {
            var d = deps[a];
            d.EnemyEta = int.MaxValue;
            if (opponentAware)
            {
                int x = mem.X(d.Cell), y = mem.Y(d.Cell);
                foreach (var e in enemyWorkers)
                {
                    long free = e.InventoryCapacity - (e.Inventory?.Gold ?? 0) - (e.Inventory?.Iron ?? 0);
                    if (free <= 0) continue;
                    int eta = Math.Max(Math.Abs(e.Position.X - x), Math.Abs(e.Position.Y - y)) + Math.Max(0, e.ReadyOnTurn - turn);
                    if (eta < d.EnemyEta) { d.EnemyEta = eta; d.EnemyFree = free; }
                }
                d.EnemiesWithin4 = CountNear(enemyAt, d.Cell, 4);
            }
            for (int b = 0; b < deps.Count; b++)
            {
                if (a == b || !deps[b].Gold || Manhattan(d.Cell, deps[b].Cell) > 3) continue;
                d.GoldNearbyCount++;
                d.GoldNearbyAmount += deps[b].AmountEst;
            }
            deps[a] = d;
        }
        float knownGold = deps.Where(d => d.Gold && d.Mineable).Sum(d => d.AmountEst);

        // ---- Investment: workers and bases ----
        long carriedGold = mine.Sum(u => u.Inventory?.Gold ?? 0);
        double rate = (bank.Gold + carriedGold + cfg.PriorRate * 60) / (workerTurns + 60.0);
        bool invest = turn < maxTurns * cfg.StopInvestFraction;
        int recruits = mine.Count - initialWorkers;
        int price = RecruitPrice(recruits);
        double supply = Math.Max(Math.Clamp(knownGold / (10.0 * (mine.Count + 1)), 0.3, 1.0), 1.0 - turn / 80.0);
        double recruitBenefit = rate * supply * Math.Max(0, turnsLeft - 15);
        bool recruitOk = invest && recruits < cfg.MaxRecruits && recruitBenefit >= PlannerConfig.Phase(cfg.RecruitSafety, cfg.RecruitSafetyLate, phase) * price * cfg.IronGoldCost;

        if (planSite >= 0 && (!invest || !SiteValid(planSite, turn) || mine.All(u => u.UnitId != planBuilder)))
        {
            planSite = -1; planBuilder = null;
        }
        int site = -1;
        double siteBenefit = 0;
        if (invest && planSite < 0 && myBases.Count < cfg.MaxBases)
            (site, siteBenefit) = BestSite(deps, mine, rate, turn, turnsLeft, maxTurns);
        bool baseOk = site >= 0 && siteBenefit >= PlannerConfig.Phase(cfg.BaseSafety, cfg.BaseSafetyLate, phase) * 6 * cfg.IronGoldCost;

        long reserved = planSite >= 0 ? 6 : 0;
        long avail = bank.Iron - reserved;
        double rr = recruitOk ? recruitBenefit / (price * cfg.IronGoldCost) : 0;
        double br = baseOk ? siteBenefit / (6 * cfg.IronGoldCost) : 0;
        // Commit to what we are already saving for: flip-flopping lets cheap bases eat the iron a worker waits on.
        if (saving == InvestmentChoice.Recruit) br /= 1 + cfg.InvestCommit;
        else if (saving == InvestmentChoice.Base) rr /= 1 + cfg.InvestCommit;
        var choice = br > 0 && br >= rr ? InvestmentChoice.Base : rr > 0 ? InvestmentChoice.Recruit : InvestmentChoice.Save;
        saving = choice;
        long desired = 0;
        double desiredBenefit = 0;
        string? spawner = null;
        if (choice == InvestmentChoice.Base)
        {
            desiredBenefit = siteBenefit;
            if (avail >= 6)
            {
                planSite = site; planWait = 0; saving = InvestmentChoice.Save;
                planBuilder = NearestReady(mine, site, turn);
                avail -= 6;
                if (planBuilder == null) planSite = -1;
            }
            else desired = 6;
            if (recruitOk && avail >= price) spawner = PickSpawner(mine, turn);
        }
        else if (choice == InvestmentChoice.Recruit)
        {
            desiredBenefit = recruitBenefit;
            if (avail >= price) { spawner = PickSpawner(mine, turn); if (spawner != null) saving = InvestmentChoice.Save; }
            else desired = price;
        }
        float ironShadow = desired > avail ? (float)cfg.IronValue : invest ? 0.15f : 0.25f;
        if (Trace && turn % 20 == 0)
            Console.Error.WriteLine($"[trace {Name} t{turn}] bank {bank.Gold}g/{bank.Iron}i workers {mine.Count} bases {myBases.Count} rate {rate:F2} supply {supply:F2} "
                + $"price {price} recruitBenefit {recruitBenefit:F0} ok={recruitOk} site {site} siteBenefit {siteBenefit:F0} ok={baseOk} choice {choice} desired {desired} knownGold {knownGold:F0} deps {deps.Count}");

        // ---- Exploration frontier ----
        var frontier = new List<(int Cell, float Value)>();
        if (turnsLeft > 40)
        {
            for (int i = 0; i < n; i++)
            {
                if (mem.Kind[i] is Tile.Unknown or Tile.Void || float.IsInfinity(enterCost[i])) continue;
                int x = mem.X(i), y = mem.Y(i);
                bool edge = (x > 0 && mem.Kind[i - 1] == Tile.Unknown) || (x < w - 1 && mem.Kind[i + 1] == Tile.Unknown)
                    || (y > 0 && mem.Kind[i - w] == Tile.Unknown) || (y < h - 1 && mem.Kind[i + w] == Tile.Unknown);
                if (!edge) continue;
                int unknown = 0;
                for (int dy = -3; dy <= 3; dy++)
                for (int dx = -3 + Math.Abs(dy); dx <= 3 - Math.Abs(dy); dx++)
                    if (mem.InBounds(x + dx, y + dy) && mem.Kind[mem.Index(x + dx, y + dy)] == Tile.Unknown) unknown++;
                frontier.Add((i, unknown / 25f));
            }
        }
        float exploreNeed = (float)Math.Clamp(1 - knownGold / (12.0 * mine.Count), 0.15, 1);

        // ---- Candidates ----
        var candidates = new List<Candidate>();
        var rows = new List<float[]>();
        bool wantFeatures = scorer.NeedsFeatures || recorder != null || decisionRecorder != null || CandidateObserver != null;
        var contexts = new Features.Context[mine.Count];
        foreach (var d in deps) depAvail[d.Cell] = Math.Max(0, d.AmountEst - (float)cfg.ContestPenalty * d.EnemiesOn);
        var topExplore = new List<(int Cell, float Value, float Score, float Dist)>();
        for (int k = 0; k < mine.Count; k++)
        {
            var u = mine[k];
            if (u.UnitId == spawner || u.UnitId == planBuilder) continue;
            var field = fields[k];
            long gold = u.Inventory?.Gold ?? 0, iron = u.Inventory?.Iron ?? 0;
            int free = (int)(u.InventoryCapacity - gold - iron);
            float carried = gold + iron * ironShadow;
            float wait = Math.Max(0, u.ReadyOnTurn - turn);
            int pos = mem.Index(u.Position);
            contexts[k] = new Features.Context
            {
                CarriedGold = gold, CarriedIron = iron, FreeCapacity = free, TurnsLeft = turnsLeft, Turn = turn,
                IronShadow = ironShadow, TeamWorkers = mine.Count,
            };

            foreach (var d in deps)
            {
                if (!d.Mineable) continue;
                float value = d.Gold ? 1f : ironShadow;
                if (d.Gold && rush > 0 && mem.InCenter(d.Cell)) value *= (float)(1 + rush);
                if (value < 0.05f) continue;
                float travel = field.Dist[d.Cell];
                float ret = baseField.Dist[d.Cell];
                if (float.IsInfinity(travel) || float.IsInfinity(ret)) continue;
                travel += wait;
                int minable = (int)Math.Min(Math.Min(depAvail[d.Cell], free), turnsLeft - cfg.EndgameMargin - travel - ret - 1);
                if (minable <= 0) continue;
                var c = new Candidate
                {
                    Kind = CandidateKind.Mine, Worker = k, Cell = d.Cell, Travel = travel, ReturnTurns = ret,
                    Minable = minable, ResourceValue = value, CarriedValue = carried,
                    Heuristic = Candidate.Rate(carried, minable, value, travel, ret),
                };
                AddCandidate(c, new Features.Target
                {
                    Amount = d.AmountEst, IsGold = d.Gold ? 1 : 0, InCenter = mem.InCenter(d.Cell) ? 1 : 0,
                    EnemiesOnTile = d.EnemiesOn, EnemiesNear = d.EnemiesNear,
                    FriendliesNear = CountNear(friendAt, d.Cell, 2) - (Manhattan(pos, d.Cell) <= 2 ? 1 : 0),
                    ClaimedByUs = d.ByUs ? 1 : 0, ClaimedByEnemy = d.ByEnemy ? 1 : 0,
                    GoldNearbyCount = d.GoldNearbyCount, GoldNearbyAmount = d.GoldNearbyAmount,
                    InfoAge = turn - mem.LastSeen[d.Cell],
                });
            }

            if (gold + iron > 0)
            {
                float ret = baseField.Dist[pos] + wait;
                if (!float.IsInfinity(ret))
                {
                    var c = new Candidate
                    {
                        Kind = CandidateKind.Return, Worker = k, Cell = -1, Travel = ret, ReturnTurns = 0,
                        CarriedValue = carried, Heuristic = carried / (ret + 1),
                    };
                    AddCandidate(c, new Features.Target { InCenter = mem.InCenter(pos) ? 1 : 0, FriendliesNear = CountNear(friendAt, pos, 2) - 1 });
                }
            }

            if (frontier.Count > 0)
            {
                topExplore.Clear();
                foreach (var (cell, value) in frontier)
                {
                    float d = field.Dist[cell];
                    if (float.IsInfinity(d)) continue;
                    float s = (float)(PlannerConfig.Phase(cfg.ExploreRate, cfg.ExploreRateLate, phase) * exploreNeed * (0.2 + value)
                        * (mem.InCenter(cell) ? 1 + cfg.CenterExploreBonus + rush : 1) * 10 / (10 + d));
                    if (topExplore.Count < 5) topExplore.Add((cell, value, s, d));
                    else
                    {
                        int worst = 0;
                        for (int t = 1; t < topExplore.Count; t++) if (topExplore[t].Score < topExplore[worst].Score) worst = t;
                        if (s > topExplore[worst].Score) topExplore[worst] = (cell, value, s, d);
                    }
                }
                foreach (var (cell, value, s, d) in topExplore)
                {
                    float ret = baseField.Dist[cell];
                    var c = new Candidate
                    {
                        Kind = CandidateKind.Explore, Worker = k, Cell = cell, Travel = d + wait,
                        ReturnTurns = float.IsInfinity(ret) ? 60 : ret, CarriedValue = carried, Heuristic = s,
                    };
                    AddCandidate(c, new Features.Target
                    {
                        InCenter = mem.InCenter(cell) ? 1 : 0, ExploreValue = value, EnemiesNear = CountNear(enemyAt, cell, 2),
                        InfoAge = 0,
                    });
                }
            }
        }

        void AddCandidate(Candidate c, Features.Target t)
        {
            if (wantFeatures)
            {
                var row = new float[Features.Count];
                Features.Fill(row, c, contexts[c.Worker], t);
                c.FeatureRow = rows.Count;
                rows.Add(row);
            }
            candidates.Add(c);
        }

        // ---- Score ----
        var span = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(candidates);
        bool stochastic = scorer.Stochastic;
        if (scorer.NeedsFeatures)
        {
            var flat = new float[rows.Count * Features.Count];
            for (int r = 0; r < rows.Count; r++) rows[r].CopyTo(flat, r * Features.Count);
            double remaining = cfg.TurnBudgetMs - Stopwatch.GetElapsedTime(started).TotalMilliseconds - 15;
            if (!scorer.Score(span, flat, remaining))
            {
                ScorerFallbacks++;
                stochastic = false; // heuristic values are not logits; act greedily this turn
                HeuristicScorer.Instance.Score(span, flat, remaining);
            }
        }
        else scorer.Score(span, [], cfg.TurnBudgetMs);
        if (!stochastic)
        {
            foreach (ref var c in span)
            {
                if (noise > 0) c.Score *= (float)Math.Exp(noise * Gaussian());
                var ws = workerStates.GetValueOrDefault(mine[c.Worker].UnitId);
                if (ws != null && ws.LastKind == c.Kind && (c.Kind == CandidateKind.Return || ws.LastCell == c.Cell))
                    c.Score *= (float)(1 + cfg.Hysteresis);
            }
        }

        // ---- Opponent awareness: races we would lose, crowded deposits, catching up via the rich center ----
        long leaderGap = 0;
        if (cfg.BehindCenterBonus > 0 && view.Scores is { Count: > 1 } scores)
            leaderGap = scores.Where(s => s.Key != me).Max(s => s.Value) - scores.GetValueOrDefault(me);
        if (opponentAware || leaderGap > 0)
        {
            var depAt = new Dictionary<int, int>(deps.Count);
            for (int i = 0; i < deps.Count; i++) depAt[deps[i].Cell] = i;
            float behind = (float)(cfg.BehindCenterBonus * Math.Clamp(leaderGap / 150.0, 0, 1));
            foreach (ref var c in span)
            {
                if (c.Kind != CandidateKind.Mine || !depAt.TryGetValue(c.Cell, out int di)) continue;
                var d = deps[di];
                float penalty = 0;
                if (d.EnemyEta < c.Travel && d.AmountEst > 0)
                    penalty += (float)cfg.RaceDiscount * Math.Min(1f, d.EnemyFree / d.AmountEst);
                penalty += (float)cfg.CrowdPenalty * d.EnemiesWithin4;
                float bonus = d.Gold && behind > 0 && mem.InCenter(d.Cell) ? behind : 0;
                c.Score += Math.Abs(c.Score) * (bonus - Math.Min(0.95f, penalty));
            }
        }

        if (CandidateObserver != null)
            for (int k = 0; k < mine.Count; k++)
            {
                var list = new List<(CandidateKind, int, float[])>();
                foreach (var c in span) if (c.Worker == k) list.Add((c.Kind, c.Cell, rows[c.FeatureRow]));
                if (list.Count > 0) CandidateObserver(mine[k].UnitId, list);
            }

        var chosen = new int[mine.Count];
        Array.Fill(chosen, -1);
        var logProbs = new float[mine.Count];
        Array.Fill(logProbs, float.NaN);
        var offered = new List<int>[mine.Count];
        for (int k = 0; k < mine.Count; k++) offered[k] = [];
        var reservedAt = new Dictionary<int, float>();
        var exploreTargets = new List<int>();

        // ---- Policy: each worker samples from softmax(logit / T) over what is still available ----
        if (stochastic)
        {
            double temperature = scorer.Temperature;
            var logits = new List<double>();
            for (int k = 0; k < mine.Count; k++)
            {
                logits.Clear();
                for (int i = 0; i < span.Length; i++)
                {
                    if (span[i].Worker != k) continue;
                    if (span[i].Kind == CandidateKind.Mine && depAvail[span[i].Cell] - reservedAt.GetValueOrDefault(span[i].Cell) < 1) continue;
                    offered[k].Add(i);
                    logits.Add(span[i].Score / temperature);
                }
                if (offered[k].Count == 0) continue;
                double max = logits.Max(), sum = 0;
                foreach (var l in logits) sum += Math.Exp(l - max);
                double u = rng.NextDouble() * sum, acc = 0;
                int pickIndex = logits.Count - 1;
                for (int j = 0; j < logits.Count; j++)
                {
                    acc += Math.Exp(logits[j] - max);
                    if (u <= acc) { pickIndex = j; break; }
                }
                int i0 = offered[k][pickIndex];
                chosen[k] = i0;
                logProbs[k] = (float)(logits[pickIndex] - max - Math.Log(sum));
                if (span[i0].Kind == CandidateKind.Mine)
                {
                    float left = depAvail[span[i0].Cell] - reservedAt.GetValueOrDefault(span[i0].Cell);
                    reservedAt[span[i0].Cell] = reservedAt.GetValueOrDefault(span[i0].Cell) + Math.Min(span[i0].Minable, left);
                }
            }
        }
        else
        {
            for (int i = 0; i < span.Length; i++) offered[span[i].Worker].Add(i);
        }

        // ---- Values: repeatedly take the best (worker, target) pair, reserving deposit stock ----
        for (int round = 0; !stochastic && round < mine.Count; round++)
        {
            // Learned scores may be negative: only the ranking matters, so any valid candidate can win.
            int best = -1;
            float bestScore = float.NegativeInfinity;
            for (int i = 0; i < span.Length; i++)
            {
                ref var c = ref span[i];
                if (chosen[c.Worker] >= 0) continue;
                float s = Adjusted(ref c, out _);
                if (s > bestScore) { bestScore = s; best = i; }
            }
            if (best < 0) break;
            ref var pick = ref span[best];
            chosen[pick.Worker] = best;
            Adjusted(ref pick, out int minable);
            if (pick.Kind == CandidateKind.Mine) reservedAt[pick.Cell] = reservedAt.GetValueOrDefault(pick.Cell) + minable;
            if (pick.Kind == CandidateKind.Explore) exploreTargets.Add(pick.Cell);
        }

        float Adjusted(ref Candidate c, out int minable)
        {
            minable = c.Minable;
            if (!cfg.Coordinate) return c.Score;
            if (c.Kind == CandidateKind.Mine)
            {
                float left = depAvail[c.Cell] - reservedAt.GetValueOrDefault(c.Cell);
                if (left < 1) return float.NegativeInfinity;
                if (left < c.Minable)
                {
                    minable = (int)left;
                    float r = Candidate.Rate(c.CarriedValue, minable, c.ResourceValue, c.Travel, c.ReturnTurns);
                    return c.Heuristic > 0 ? c.Score * r / c.Heuristic : r;
                }
            }
            else if (c.Kind == CandidateKind.Explore)
            {
                foreach (var t in exploreTargets) if (Manhattan(t, c.Cell) <= 6) return c.Score * 0.4f;
            }
            return c.Score;
        }

        (MarsAction, string, int) Hoard(int k, Worker u, int pos, long gold, long iron, bool canBank)
        {
            if (!canBank && iron >= 3) return (MarsAction.Trade(Resource.Iron), "trade", -1); // carried gold outranks carried iron
            if (u.InventoryCapacity - gold - iron <= 0) return (MarsAction.Pass(), "pass", -1);
            int target = -1;
            float bestCost = float.PositiveInfinity;
            foreach (var d in deps)
            {
                if (!d.Mineable) continue;
                float dist = fields[k].Dist[d.Cell];
                if (float.IsInfinity(dist) || dist + 1 > turnsLeft) continue;
                float cost = dist + (d.Gold ? 0 : 6); // prefer gold unless iron is much closer
                if (cost < bestCost) { bestCost = cost; target = d.Cell; }
            }
            if (target < 0) return (MarsAction.Pass(), "pass", -1);
            if (target == pos) return (MarsAction.Mine(), "mine", -1);
            int step = fields[k].FirstStep(target);
            return step >= 0 ? (Move(u.Position, step), "move", step) : (MarsAction.Pass(), "pass", -1);
        }

        // ---- Actions ----
        long ironBought = 0;
        for (int k = 0; k < mine.Count; k++)
        {
            var u = mine[k];
            if (!workerStates.TryGetValue(u.UnitId, out var ws)) workerStates[u.UnitId] = ws = new WorkerState();
            int pos = mem.Index(u.Position);
            long gold = u.Inventory?.Gold ?? 0, iron = u.Inventory?.Iron ?? 0;
            MarsAction action = MarsAction.Pass();
            string kind = "pass";
            int actionCell = -1;
            Candidate? pick = chosen[k] >= 0 ? span[chosen[k]] : null;
            float home = baseField.Dist[pos] + Math.Max(0, u.ReadyOnTurn - turn);
            bool canBank = home + 1 <= turnsLeft;

            if (turn < u.ReadyOnTurn) kind = "cooldown";
            else if (u.UnitId == spawner) { action = MarsAction.Spawn(); kind = "spawn"; }
            else if (u.UnitId == planBuilder && planSite >= 0)
            {
                if (pos == planSite)
                {
                    bool alone = view.Units.Count(o => o.Position == u.Position) == 1;
                    if (alone && bank.Iron >= 6) { action = MarsAction.BuildBase(); kind = "build"; }
                    else if (++planWait > 3) { buildBlockedUntil[planSite] = turn + 30; planSite = -1; planBuilder = null; }
                }
                else
                {
                    int step = fields[k].FirstStep(planSite);
                    if (step >= 0) { action = Move(u.Position, step); kind = "move"; actionCell = step; }
                    else { planSite = -1; planBuilder = null; }
                }
            }
            else if (mem.BaseOwnerAt[pos] == me && gold + iron > 0)
            {
                // Iron deposits are small and often unknown: buy the missing iron with carried gold (3:1)
                // when the investment waiting on it is worth clearly more than the gold spent.
                long ironShort = desired - avail - iron - ironBought;
                if (invest && ironShort > 0 && gold >= 3 && desiredBenefit >= PlannerConfig.Phase(cfg.TradeSafety, cfg.TradeSafetyLate, phase) * 3 * ironShort)
                {
                    action = MarsAction.Trade(Resource.Gold); kind = "trade";
                    ironBought++;
                }
                else if (!invest && iron >= 3 && turnsLeft >= 2 + cfg.EndgameMargin) { action = MarsAction.Trade(Resource.Iron); kind = "trade"; }
                else
                {
                    action = MarsAction.Deposit(); kind = "deposit";
                    FinishTrip(ws, gold + iron * ironShadow, turn);
                }
            }
            else if (gold + iron > 0 && canBank && turnsLeft <= home + 2 + cfg.EndgameMargin)
            {
                // Banked gold is the score (and banked iron the first tiebreak): go home whatever the scorer says.
                int back = baseField.StepToward(pos);
                if (back >= 0) { action = Move(u.Position, back); kind = "move"; actionCell = back; }
            }
            else if (pick == null && canBank && gold + iron > 0)
            {
                // No usable candidate (e.g. a non-finite score) while carrying: bank it rather than stand still.
                int back = baseField.StepToward(pos);
                if (back >= 0) { action = Move(u.Position, back); kind = "move"; actionCell = back; }
            }
            else if (!canBank || (pick == null && turnsLeft <= 40))
            {
                // Banking is impossible (or nothing is worth a trip): build the later tiebreaks instead,
                // carried gold first, then carried iron. Never spend banked iron this late.
                (action, kind, actionCell) = Hoard(k, u, pos, gold, iron, canBank);
            }
            else if (pick is { } c)
            {
                switch (c.Kind)
                {
                    case CandidateKind.Mine when c.Cell == pos:
                        action = MarsAction.Mine(); kind = "mine"; break;
                    case CandidateKind.Return:
                        int back = baseField.StepToward(pos);
                        if (back >= 0) { action = Move(u.Position, back); kind = "move"; actionCell = back; }
                        break;
                    default:
                        int step = fields[k].FirstStep(c.Cell);
                        if (step >= 0) { action = Move(u.Position, step); kind = "move"; actionCell = step; }
                        break;
                }
                bool changed = ws.LastKind != c.Kind || ws.LastCell != c.Cell;
                if (recorder != null && kind is "mine" or "move" && (changed || turn % 3 == 0))
                    ws.Pending.Add((rows[c.FeatureRow], turn));
                // Decision samples for imitation/PPO. Policies keep every offered row so the recorded
                // log-probability stays exact; value scorers keep the 16 best rows (chosen always included).
                if (decisionRecorder != null && rng.NextDouble() < (stochastic ? 0.6 : 0.3))
                {
                    var set = offered[k];
                    if (!stochastic && set.Count > 16)
                    {
                        var copy = span.ToArray();
                        int chosenIndex = chosen[k];
                        set = set.OrderByDescending(i => i == chosenIndex ? float.MaxValue : copy[i].Score).Take(16).ToList();
                    }
                    var flat = new float[set.Count * Features.Count];
                    int at = -1;
                    for (int j = 0; j < set.Count; j++)
                    {
                        rows[span[set[j]].FeatureRow].CopyTo(flat, j * Features.Count);
                        if (set[j] == chosen[k]) at = j;
                    }
                    if (at >= 0) ws.PendingDecisions.Add((flat, set.Count, at, logProbs[k], turn));
                }
                ws.LastKind = c.Kind;
                ws.LastCell = c.Cell;
            }
            actions[u.UnitId] = action;
            ws.LastAction = kind;
            ws.LastActionCell = actionCell;
            ws.LastLoad = gold + iron;
            ws.LastPos = u.Position;
        }

        MaxTurnMs = Math.Max(MaxTurnMs, Stopwatch.GetElapsedTime(started).TotalMilliseconds);
        return actions;
    }

    void FinishTrip(WorkerState ws, float value, int turn)
    {
        if (recorder != null)
            foreach (var (f, t) in ws.Pending) recorder.Add(new TripRecord(f, value / (turn + 1 - t), recordSource));
        ws.Pending.Clear();
        foreach (var d in ws.PendingDecisions) finishedDecisions.Add((d.Rows, d.Count, d.Chosen, d.LogProb, d.Turn, value / (turn + 1 - d.Turn)));
        ws.PendingDecisions.Clear();
    }

    public void OnMatchEnd(GameState<MarsView>? finalState, Identity identity)
    {
        // Loads that never reached the bank were worth nothing.
        foreach (var ws in workerStates.Values)
        {
            if (recorder != null) foreach (var (f, _) in ws.Pending) recorder.Add(new TripRecord(f, 0, recordSource));
            foreach (var d in ws.PendingDecisions) finishedDecisions.Add((d.Rows, d.Count, d.Chosen, d.LogProb, d.Turn, 0));
            ws.Pending.Clear();
            ws.PendingDecisions.Clear();
        }
        {
            var final = finalState?.View?.Scores ?? lastScores ?? [];
            string me = identity.TeamId.Length > 0 ? identity.TeamId : teamId;
            long myGold = final.GetValueOrDefault(me);
            if (final.Count > 0) FinalStanding = (myGold, 1 + final.Count(s => s.Key != me && s.Value > myGold), final.Count);
        }
        if (decisionRecorder != null && finishedDecisions.Count > 0)
        {
            var scores = finalState?.View?.Scores ?? lastScores ?? [];
            string team = identity.TeamId.Length > 0 ? identity.TeamId : teamId;
            long gold = scores.GetValueOrDefault(team);
            int place = 1 + scores.Count(s => s.Key != team && s.Value > gold);
            decisionRecorder.AddRange(finishedDecisions.Select(d => new DecisionRecord(d.Rows, d.Count, d.Chosen, d.LogProb, d.Return,
                Name, team, d.Turn, gold, place, Math.Max(1, scores.Count), recordSource)).ToList());
        }
        finishedDecisions.Clear();
    }

    bool SiteValid(int s, int turn) =>
        mem!.Kind[s] == Tile.Plain && mem.BaseOwnerAt[s] == null && mem.FarFromEnemyBases(s) && buildBlockedUntil[s] <= turn;

    (int Site, double Benefit) BestSite(List<Dep> deps, List<Worker> mine, double rate, int turn, int turnsLeft, int maxTurns)
    {
        var m = mem!;
        stampId++;
        int best = -1;
        double bestValue = 0;
        foreach (var g in deps)
        {
            if (!g.Gold) continue;
            int gx = m.X(g.Cell), gy = m.Y(g.Cell);
            for (int dy = -3; dy <= 3; dy++)
            for (int dx = -3; dx <= 3; dx++)
            {
                int x = gx + dx, y = gy + dy;
                if (!m.InBounds(x, y)) continue;
                int s = m.Index(x, y);
                if (stamp[s] == stampId) continue;
                stamp[s] = stampId;
                if (!SiteValid(s, turn) || baseField.Dist[s] < 3) continue;
                float builderDist = float.PositiveInfinity;
                for (int k = 0; k < mine.Count; k++) builderDist = Math.Min(builderDist, fields[k].Dist[s]);
                if (float.IsInfinity(builderDist)) continue;

                double value = -builderDist * rate;
                foreach (var d in deps)
                {
                    if (!d.Gold) continue;
                    int ddx = Math.Abs(m.X(d.Cell) - x), ddy = Math.Abs(m.Y(d.Cell) - y);
                    if (Math.Max(ddx, ddy) > 7) continue;
                    bool claims = ddx + ddy <= 2;
                    if (!d.Mineable)
                    {
                        if (claims) value += d.AmountEst * 0.6;
                        continue;
                    }
                    float current = baseField.Dist[d.Cell];
                    if (float.IsInfinity(current)) current = 40;
                    double saved = Math.Max(0, current - Math.Max(ddx, ddy) * 1.15);
                    value += d.AmountEst / 7.0 * 2 * saved * rate;
                    if (claims && !d.ByUs && !d.ByEnemy) value += d.AmountEst * cfg.ClaimBonus;
                }
                if (x >= m.CenterX0 - 2 && y >= m.CenterY0 - 2 && x < m.CenterX0 + m.CenterW + 2 && y < m.CenterY0 + m.CenterH + 2)
                    value += cfg.CenterBaseBonus * turnsLeft / maxTurns;
                if (value > bestValue) { bestValue = value; best = s; }
            }
        }
        return (best, bestValue);
    }

    string? NearestReady(List<Worker> mine, int site, int turn)
    {
        string? best = null;
        float bestDist = float.PositiveInfinity;
        for (int k = 0; k < mine.Count; k++)
        {
            float d = fields[k].Dist[site] + Math.Max(0, mine[k].ReadyOnTurn - turn);
            if (d < bestDist) { bestDist = d; best = mine[k].UnitId; }
        }
        return best;
    }

    string? PickSpawner(List<Worker> mine, int turn)
    {
        string? best = null;
        double bestScore = double.NegativeInfinity;
        foreach (var u in mine)
        {
            if (turn < u.ReadyOnTurn || u.UnitId == planBuilder) continue;
            int pos = mem!.Index(u.Position);
            double s = -Load(u) / 8.0;
            if (mem.Kind[pos] is Tile.Gold && mem.Amount[pos] >= 3) s += 3;
            if (mem.BaseOwnerAt[pos] == mem.TeamId) s += 2;
            if (s > bestScore) { bestScore = s; best = u.UnitId; }
        }
        return best;
    }

    MarsAction Move(Coordinate from, int to) => MarsAction.Move(Directions.Of(mem!.X(to) - from.X, mem.Y(to) - from.Y));

    static long Load(Worker u) => (u.Inventory?.Gold ?? 0) + (u.Inventory?.Iron ?? 0);

    int Manhattan(int a, int b) => Math.Abs(mem!.X(a) - mem.X(b)) + Math.Abs(mem.Y(a) - mem.Y(b));

    int CountNear(Dictionary<int, int> at, int cell, int radius)
    {
        int total = 0;
        foreach (var (i, count) in at) if (Manhattan(i, cell) <= radius) total += count;
        return total;
    }

    double Gaussian()
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }
}
