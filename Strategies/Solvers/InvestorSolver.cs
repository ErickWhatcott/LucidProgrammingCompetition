using Lpc.Client;
using Lpc.Strategies.Planner;

namespace Lpc.Strategies.Solvers;

/// <summary>
/// Investor: decides WHEN to grow by full-horizon NPV over growth schedules. Every 10 turns (or after an
/// investment) a scalar economy model (turn, workers, bases, iron, gold) simulates every interleaving of
/// 0..5 recruits and 0..3 bases to the last turn with Fibonacci prices, supply saturation, trip-time gains
/// and iron procurement (mine vs trade); the first step of the best schedule is the current goal.
/// Micro: per-worker targets scored by opportunity cost (value - rho * turns), stock reserved across workers.
/// </summary>
public sealed class InvestorSolver(int seed = 0) : IMarsStrategy
{
    enum Goal { None, Recruit, Base }

    public string Name => "investor";

    const double PriorRate = 0.35;   // kg gold / worker / turn before data
    const int MacroEvery = 10;
    const int RecruitLag = 6;        // turns before a new worker banks at the team rate
    const double CenterDensity = 0.6, OuterDensity = 0.12; // prior kg per unseen tile
    const double Hysteresis = 0.6;   // kg bonus for keeping last turn's target
    const double ContestKg = 3;      // kg assumed lost per enemy worker on a deposit

    readonly Random rng = new(seed);
    WorldMemory? mem;
    float[] enterCost = [], depAvail = [], reservedAt = [];
    int[] mineBlockedUntil = [], mineFailures = [], moveBlockedUntil = [], buildBlockedUntil = [];
    readonly PathField baseField = new();
    readonly List<PathField> fields = [];
    int initialWorkers = -1;

    // Economy measurement
    long lastGoldTotal = -1;
    int lastWorkers, goldTradesLast;
    double cumGold, cumWorkerTurns, rhoEma = PriorRate, tripEma = 24;

    // Macro state
    Goal goal = Goal.None;
    bool goalTrade;
    int lastMacro = -100, lastW = -1, lastB = -1;
    int site = -1;
    string? builder, ironRunner;
    double siteGain, siteExtra;
    float siteDist;
    public string LastPlan { get; private set; } = "";
    static readonly bool Trace = Environment.GetEnvironmentVariable("INVESTOR_TRACE") == "1";

    sealed class WState
    {
        public string LastAction = "";
        public int LastActionCell = -1, LastCell = -2, LastDeposit;
        public long LastLoad;
        public Coordinate LastPos;
    }
    readonly Dictionary<string, WState> states = [];

    struct Dep { public int Cell; public bool Gold; public float Est; public int EnemiesOn; }

    struct Cand { public int Worker, Cell; public byte Kind; public float Net, Travel, Ret; public int Take; public float Value, Carried; }
    const byte KMine = 0, KReturn = 1, KExplore = 2;

    static int RecruitPrice(int recruits)
    {
        int a = 5, b = 6;
        if (recruits <= 0) return a;
        for (int i = 1; i < recruits; i++) (a, b) = (b, a + b);
        return b;
    }

    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity)
    {
        var actions = new Dictionary<string, MarsAction>();
        var view = state.View;
        if (view == null) return actions;
        string me = identity.TeamId;
        mem ??= new WorldMemory(me);
        mem.Update(view);
        int w = mem.Width, h = mem.Height, n = mem.Count;
        if (enterCost.Length != n)
        {
            enterCost = new float[n]; depAvail = new float[n]; reservedAt = new float[n];
            mineBlockedUntil = new int[n]; mineFailures = new int[n]; moveBlockedUntil = new int[n]; buildBlockedUntil = new int[n];
        }
        int turn = view.Turn;
        int maxTurns = state.Match.MaxTurns > 0 ? state.Match.MaxTurns : 480;
        int left = maxTurns - turn;
        var mine = view.Units.Where(u => u.TeamId == me).OrderBy(u => u.UnitId, StringComparer.Ordinal).ToList();
        if (mine.Count == 0) return actions;
        if (initialWorkers < 0) initialWorkers = mine.Count;
        var bank = view.TeamResources?.GetValueOrDefault(me) ?? new Inventory(0, 0);

        // ---- Terrain and fields ----
        for (int i = 0; i < n; i++)
        {
            float c = mem.Kind[i] switch
            {
                Tile.Unknown => 1.6f,
                Tile.Mountain or Tile.Crater => 3f,
                Tile.Void => float.PositiveInfinity,
                _ => 1f,
            };
            if (moveBlockedUntil[i] > turn) c = float.PositiveInfinity;
            var owner = mem.BaseOwnerAt[i];
            if (owner != null) c = owner == me ? 1f : float.PositiveInfinity;
            enterCost[i] = c;
        }
        var myBases = mem.MyBases.Select(b => mem.Index(b.Position)).ToList();
        baseField.To(myBases, enterCost, w, h);
        while (fields.Count < mine.Count) fields.Add(new PathField());
        for (int k = 0; k < mine.Count; k++) fields[k].From(mem.Index(mine[k].Position), enterCost, w, h, 45);

        // ---- Learn from last turn: failed mines / moves / builds ----
        foreach (var u in mine)
        {
            if (!states.TryGetValue(u.UnitId, out var ws)) continue;
            long load = Load(u);
            int pos = mem.Index(u.Position);
            if (ws.LastAction == "mine" && u.Position == ws.LastPos && load == ws.LastLoad
                && load < u.InventoryCapacity && mem.Kind[pos] is Tile.Gold or Tile.Iron && mem.Amount[pos] > 0)
            {
                if (++mineFailures[pos] >= 2) { mineBlockedUntil[pos] = turn + 25; mineFailures[pos] = 0; }
            }
            else if (ws.LastAction == "mine" && load > ws.LastLoad) mineFailures[pos] = 0;
            if (ws.LastAction == "move" && u.Position == ws.LastPos && turn >= u.ReadyOnTurn && ws.LastActionCell >= 0)
                moveBlockedUntil[ws.LastActionCell] = turn + 40;
            if (ws.LastAction == "build" && mem.BaseOwnerAt[pos] != me)
            {
                buildBlockedUntil[pos] = turn + 60;
                if (site == pos) { site = -1; builder = null; }
            }
        }

        // ---- Measure the per-worker rate (rho) ----
        long goldTotal = bank.Gold + mine.Sum(u => u.Inventory?.Gold ?? 0);
        if (lastGoldTotal >= 0)
        {
            double inc = goldTotal - lastGoldTotal + 3 * goldTradesLast;
            cumGold += inc; cumWorkerTurns += lastWorkers;
            rhoEma += 0.025 * (inc / Math.Max(1, lastWorkers) - rhoEma);
        }
        lastGoldTotal = goldTotal; lastWorkers = mine.Count; goldTradesLast = 0;
        double rhoCum = (cumGold + PriorRate * 80) / (cumWorkerTurns + 80);
        double rho = Math.Max(0.08, turn < 100 ? rhoCum : 0.5 * (rhoCum + rhoEma));

        // ---- Known deposits ----
        var enemyAt = new Dictionary<int, int>();
        foreach (var u in view.Units)
            if (u.TeamId != me) { int i = mem.Index(u.Position); enemyAt[i] = enemyAt.GetValueOrDefault(i) + 1; }
        var deps = new List<Dep>();
        double supply = 0;
        for (int i = 0; i < n; i++)
        {
            if (mem.Kind[i] is not (Tile.Gold or Tile.Iron) || mem.Amount[i] <= 0 || mineBlockedUntil[i] > turn) continue;
            if (!mem.Claim(i).Mineable) continue;
            float est = mem.Amount[i];
            if (!mem.Visible[i] && turn - mem.LastSeen[i] > 25) est *= 0.7f;
            var d = new Dep { Cell = i, Gold = mem.Kind[i] == Tile.Gold, Est = est, EnemiesOn = enemyAt.GetValueOrDefault(i) };
            deps.Add(d);
            depAvail[i] = Math.Max(0, est - (float)ContestKg * d.EnemiesOn);
            if (d.Gold && baseField.Dist[i] < 60) supply += est;
        }
        int unknown = 0;
        for (int i = 0; i < n; i++) if (mem.Kind[i] == Tile.Unknown) unknown++;
        supply += 40.0 * unknown / n;

        long carriedIron = mine.Sum(u => u.Inventory?.Iron ?? 0);
        int recruits = mine.Count - initialWorkers;

        // ---- Macro: growth-schedule optimiser ----
        if (site >= 0 && (!SiteValid(site, turn) || (builder != null && mine.All(u => u.UnitId != builder)))) { site = -1; builder = null; if (goal == Goal.Base) lastMacro = -100; }
        if (turn - lastMacro >= MacroEvery || mine.Count != lastW || myBases.Count != lastB)
        {
            lastMacro = turn; lastW = mine.Count; lastB = myBases.Count;
            if (builder == null) (site, siteGain, siteExtra, siteDist) = BestSite(deps, mine, rho, turn);
            double ironRate = IronRate(deps);
            (goal, goalTrade) = Optimize(turn, maxTurns, mine.Count, myBases.Count, bank.Iron + carriedIron, recruits, rho,
                supply, ironRate);
            if (goal != Goal.Base && builder == null) site = -1;
        }

        // ---- Execute the goal ----
        long need = goal switch { Goal.Recruit => RecruitPrice(recruits), Goal.Base => 6, _ => 0 };
        long shortfall = Math.Max(0, need - bank.Iron - carriedIron);
        string? spawner = null;
        if (goal == Goal.Recruit && bank.Iron >= need && left > 30)
        {
            spawner = PickSpawner(mine, turn);
            if (spawner != null) lastMacro = -100; // replan next turn
        }
        if (goal == Goal.Base && site >= 0 && builder == null && bank.Iron >= 6)
            builder = NearestReady(mine, site, turn);
        if (goal != Goal.Base && builder != null) { builder = null; site = -1; }

        // Iron logistics: an iron run when cheaper than trading 3 gold per kg, else trade carried gold at a base.
        bool tradeIron = false;
        int runnerK = -1;
        if (ironRunner != null && (shortfall <= 0 || mine.All(u => u.UnitId != ironRunner))) ironRunner = null;
        if (shortfall > 0 && left > 40)
        {
            double bestCost = double.PositiveInfinity;
            string? best = null;
            for (int k = 0; k < mine.Count; k++)
            {
                var u = mine[k];
                if (u.UnitId == builder || u.UnitId == spawner) continue;
                int free = (int)(u.InventoryCapacity - Load(u));
                if (free < 2) continue;
                foreach (var d in deps)
                {
                    if (d.Gold || depAvail[d.Cell] < 1) continue;
                    float tr = fields[k].Dist[d.Cell], ret = baseField.Dist[d.Cell];
                    if (float.IsInfinity(tr) || float.IsInfinity(ret)) continue;
                    int take = (int)Math.Min(Math.Min(depAvail[d.Cell], free), shortfall);
                    double perKg = rho * (tr + take + ret + 1) / take;
                    if (u.UnitId == ironRunner) perKg *= 0.8;
                    if (perKg < bestCost) { bestCost = perKg; best = u.UnitId; }
                }
            }
            if (best != null && bestCost < 3) ironRunner = best;
            else { ironRunner = null; tradeIron = goalTrade; }
        }
        for (int k = 0; k < mine.Count; k++) if (mine[k].UnitId == ironRunner) runnerK = k;

        if (Trace && turn % 10 == 0) Console.Error.WriteLine($"[investor] {LastPlan} | {LastSchedule} W={mine.Count} B={myBases.Count} bank={bank.Gold}g/{bank.Iron}i trip={tripEma:F0}");
        LastPlan = $"t{turn} goal={goal} trade={goalTrade} need={need} short={shortfall} rho={rho:F2} S={supply:F0} site={site} gain={siteGain:F2}";

        // ---- Exploration frontier ----
        var frontier = new List<(int Cell, float Gold)>();
        if (left > 40)
        {
            for (int i = 0; i < n; i++)
            {
                if (mem.Kind[i] is Tile.Unknown or Tile.Void || float.IsInfinity(enterCost[i])) continue;
                int x = mem.X(i), y = mem.Y(i);
                bool edge = (x > 0 && mem.Kind[i - 1] == Tile.Unknown) || (x < w - 1 && mem.Kind[i + 1] == Tile.Unknown)
                    || (y > 0 && mem.Kind[i - w] == Tile.Unknown) || (y < h - 1 && mem.Kind[i + w] == Tile.Unknown);
                if (!edge) continue;
                double g = 0;
                for (int dy = -3; dy <= 3; dy++)
                for (int dx = -3 + Math.Abs(dy); dx <= 3 - Math.Abs(dy); dx++)
                {
                    if (!mem.InBounds(x + dx, y + dy)) continue;
                    int t = mem.Index(x + dx, y + dy);
                    if (mem.Kind[t] == Tile.Unknown) g += mem.InCenter(t) ? CenterDensity : OuterDensity;
                }
                frontier.Add((i, (float)Math.Min(8, g * 0.7)));
            }
        }

        // ---- Candidates: net = value - rho * turns ----
        var cands = new List<Cand>();
        for (int k = 0; k < mine.Count; k++)
        {
            var u = mine[k];
            if (u.UnitId == spawner || u.UnitId == builder) continue;
            var field = fields[k];
            long gold = u.Inventory?.Gold ?? 0, iron = u.Inventory?.Iron ?? 0;
            int free = (int)(u.InventoryCapacity - gold - iron);
            float carried = gold + iron * (shortfall > 0 ? 3f : 0.1f);
            float wait = Math.Max(0, u.ReadyOnTurn - turn);
            int pos = mem.Index(u.Position);
            int lastCell = states.TryGetValue(u.UnitId, out var st) ? st.LastCell : -2;
            bool runner = k == runnerK;
            if (free > 0)
                foreach (var d in deps)
                {
                    float value = d.Gold ? 1f : runner ? 3f : 0f;
                    if (value <= 0) continue;
                    float travel = field.Dist[d.Cell], ret = baseField.Dist[d.Cell];
                    if (float.IsInfinity(travel) || float.IsInfinity(ret)) continue;
                    travel += wait;
                    int take = (int)Math.Min(Math.Min(depAvail[d.Cell], free), left - 1 - travel - ret - 1);
                    if (!d.Gold) take = (int)Math.Min(take, shortfall);
                    if (take <= 0) continue;
                    float net = carried + take * value - (float)rho * (travel + take + ret + 1);
                    if (d.Cell == lastCell) net += (float)Hysteresis;
                    cands.Add(new Cand { Worker = k, Cell = d.Cell, Kind = KMine, Travel = travel, Ret = ret, Take = take, Value = value, Carried = carried, Net = net });
                }
            if (gold + iron > 0 && !float.IsInfinity(baseField.Dist[pos]))
            {
                float ret = baseField.Dist[pos] + wait;
                float net = carried - (float)rho * (ret + 1) + (lastCell == -1 ? (float)Hysteresis : 0);
                cands.Add(new Cand { Worker = k, Cell = -1, Kind = KReturn, Travel = ret, Carried = carried, Net = net });
            }
            if (free >= 4 && frontier.Count > 0)
            {
                // Speculative trip: go to the frontier, expect to fill from what it reveals, come back.
                var top = new List<Cand>();
                foreach (var (cell, g) in frontier)
                {
                    float d = field.Dist[cell];
                    if (float.IsInfinity(d)) continue;
                    float ret = baseField.Dist[cell];
                    if (float.IsInfinity(ret)) ret = 30;
                    float take = Math.Min(g, free);
                    float net = carried + take - (float)rho * (d + wait + take + 2 + ret + 1) + (cell == lastCell ? (float)Hysteresis : 0);
                    var c = new Cand { Worker = k, Cell = cell, Kind = KExplore, Travel = d + wait, Ret = ret, Carried = carried, Net = net };
                    if (top.Count < 4) top.Add(c);
                    else
                    {
                        int worst = 0;
                        for (int t = 1; t < top.Count; t++) if (top[t].Net < top[worst].Net) worst = t;
                        if (net > top[worst].Net) top[worst] = c;
                    }
                }
                cands.AddRange(top);
            }
        }

        // ---- Greedy assignment with stock reservation ----
        var chosen = new int[mine.Count];
        Array.Fill(chosen, -1);
        var exploreTargets = new List<int>();
        var touched = new List<int>();
        for (int round = 0; round < mine.Count; round++)
        {
            int best = -1;
            float bestNet = float.NegativeInfinity;
            for (int i = 0; i < cands.Count; i++)
            {
                var c = cands[i];
                if (chosen[c.Worker] >= 0) continue;
                float s = Adjusted(c, out _);
                if (s > bestNet) { bestNet = s; best = i; }
            }
            if (best < 0) break;
            var pick = cands[best];
            chosen[pick.Worker] = best;
            Adjusted(pick, out int take);
            if (pick.Kind == KMine) { reservedAt[pick.Cell] += take; touched.Add(pick.Cell); }
            if (pick.Kind == KExplore) exploreTargets.Add(pick.Cell);
        }
        foreach (int c in touched) reservedAt[c] = 0;

        float Adjusted(Cand c, out int take)
        {
            take = c.Take;
            if (c.Kind == KMine)
            {
                float rest = depAvail[c.Cell] - reservedAt[c.Cell];
                if (rest < 1) return float.NegativeInfinity;
                if (rest < c.Take)
                {
                    take = (int)rest;
                    return c.Net - (c.Take - take) * (c.Value - (float)rho);
                }
            }
            else if (c.Kind == KExplore)
                foreach (var t in exploreTargets) if (Manhattan(t, c.Cell) <= 6) return c.Net - 3;
            return c.Net;
        }

        // ---- Actions ----
        long tradesLeft = tradeIron ? shortfall : 0;
        for (int k = 0; k < mine.Count; k++)
        {
            var u = mine[k];
            if (!states.TryGetValue(u.UnitId, out var ws)) states[u.UnitId] = ws = new WState { LastDeposit = turn };
            int pos = mem.Index(u.Position);
            long gold = u.Inventory?.Gold ?? 0, iron = u.Inventory?.Iron ?? 0;
            MarsAction action = MarsAction.Pass();
            string kind = "pass";
            int actionCell = -1, targetCell = -2;
            float home = baseField.Dist[pos] + Math.Max(0, u.ReadyOnTurn - turn);
            bool canBank = home + 1 <= left;
            Cand? pick = chosen[k] >= 0 ? cands[chosen[k]] : null;

            if (turn < u.ReadyOnTurn) kind = "cooldown";
            else if (u.UnitId == spawner) { action = MarsAction.Spawn(); kind = "spawn"; }
            else if (u.UnitId == builder && site >= 0)
            {
                if (pos == site)
                {
                    bool alone = view.Units.Count(o => o.Position == u.Position) == 1;
                    if (alone && bank.Iron >= 6) { action = MarsAction.BuildBase(); kind = "build"; }
                    else { buildBlockedUntil[site] = turn + 30; site = -1; builder = null; lastMacro = -100; }
                }
                else
                {
                    int step = fields[k].FirstStep(site);
                    if (step >= 0) { action = Move(u.Position, step); kind = "move"; actionCell = step; }
                    else { buildBlockedUntil[site] = turn + 30; site = -1; builder = null; lastMacro = -100; }
                }
            }
            else if (mem.BaseOwnerAt[pos] == me && gold + iron > 0)
            {
                if (tradesLeft > 0 && gold >= 3 && left > 30)
                {
                    action = MarsAction.Trade(Resource.Gold); kind = "trade";
                    tradesLeft--; goldTradesLast++;
                }
                else if (shortfall <= 0 && left <= 30 && iron >= 3 && left >= 3) { action = MarsAction.Trade(Resource.Iron); kind = "trade"; }
                else
                {
                    action = MarsAction.Deposit(); kind = "deposit";
                    int dt = Math.Clamp(turn - ws.LastDeposit, 4, 80);
                    tripEma += 0.1 * (dt - tripEma);
                    ws.LastDeposit = turn;
                }
            }
            else if (gold + iron > 0 && canBank && left <= home + 3)
            {
                int back = baseField.StepToward(pos);
                if (back >= 0) { action = Move(u.Position, back); kind = "move"; actionCell = back; }
            }
            else if (!canBank || (pick == null && left <= 40))
                (action, kind, actionCell) = Hoard(k, u, pos, gold, iron, canBank, deps, left);
            else if (pick is { } c)
            {
                targetCell = c.Cell;
                switch (c.Kind)
                {
                    case KMine when c.Cell == pos:
                        action = MarsAction.Mine(); kind = "mine"; break;
                    case KReturn:
                        int back = baseField.StepToward(pos);
                        if (back >= 0) { action = Move(u.Position, back); kind = "move"; actionCell = back; }
                        break;
                    default:
                        int step = fields[k].FirstStep(c.Cell);
                        if (step >= 0) { action = Move(u.Position, step); kind = "move"; actionCell = step; }
                        break;
                }
            }
            else if (gold + iron > 0 && canBank)
            {
                int back = baseField.StepToward(pos);
                if (back >= 0) { action = Move(u.Position, back); kind = "move"; actionCell = back; }
            }
            else if (kind == "pass") { action = Wander(pos); kind = "move"; }
            actions[u.UnitId] = action;
            ws.LastAction = kind; ws.LastActionCell = actionCell; ws.LastLoad = gold + iron; ws.LastPos = u.Position;
            ws.LastCell = targetCell;
        }
        return actions;
    }

    // ---- Macro model ----

    static readonly List<byte[]> Schedules = BuildSchedules();

    /// <summary>Every interleaving of r recruits (0) and b bases (1), r = 0..5, b = 0..3. Index 0 is the empty schedule.</summary>
    static List<byte[]> BuildSchedules()
    {
        var all = new List<byte[]>();
        for (int len = 0; len <= 8; len++)
            for (int mask = 0; mask < 1 << len; mask++)
            {
                int b = System.Numerics.BitOperations.PopCount((uint)mask), r = len - b;
                if (b > 3 || r > 5) continue;
                var s = new byte[len];
                for (int i = 0; i < len; i++) s[i] = (byte)((mask >> i) & 1);
                all.Add(s);
            }
        return all;
    }

    (Goal, bool Trade) Optimize(int turn, int maxTurns, int w0, int b0, long iron0, int recruits0, double rho,
        double supply, double ironRate)
    {
        double sf0 = supply / (supply + 6.0 * w0);
        double best = double.NegativeInfinity;
        Goal bestGoal = Goal.None;
        bool bestTrade = false;
        string bestDesc = "";
        double zero = 0;
        foreach (var seq in Schedules)
        {
            if (site < 0 && Array.IndexOf(seq, (byte)1) >= 0) continue;
            for (int tr = 0; tr < (seq.Length == 0 ? 1 : 2); tr++)
            {
                double g = Simulate(seq, tr == 1, turn, maxTurns, w0, b0, iron0, recruits0, rho, supply, sf0, ironRate);
                if (seq.Length == 0) zero = g;
                // Ties go to the simpler schedule (enumerated first).
                if (g > best + 0.5)
                {
                    best = g; bestTrade = tr == 1;
                    bestGoal = seq.Length == 0 ? Goal.None : seq[0] == 0 ? Goal.Recruit : Goal.Base;
                    bestDesc = string.Concat(seq.Select(x => x == 0 ? 'R' : 'B'));
                }
            }
        }
        LastSchedule = $"{bestDesc}{(bestTrade ? "+trade" : "")} -> {best:F0} (zero {zero:F0})";
        return (bestGoal, bestTrade);
    }

    public string LastSchedule { get; private set; } = "";

    double Simulate(byte[] seq, bool trade, int turn, int maxTurns, int w0, int b0, double iron, int recruits,
        double rho, double supply, double sf0, double ironRate)
    {
        double g = 0;
        int idx = 0, workers = w0, bases = b0, pendingW = 0, pendingB = 0;
        int nextW = int.MaxValue, nextB = int.MaxValue;
        double tf = 1, s = supply;
        // Mining iron with one worker beats trading when it yields more than a third of its gold rate.
        bool mineIron = !trade || ironRate * 3 > rho;
        int lastInvest = maxTurns - 25;
        for (int t = turn; t < maxTurns; t++)
        {
            if (t >= nextW) { workers += pendingW; pendingW = 0; nextW = int.MaxValue; }
            if (t >= nextB) { bases += pendingB; pendingB = 0; nextB = int.MaxValue; int k = bases - b0; tf *= 1 + siteGain * Math.Pow(0.6, k - 1); if (k == 1) s += siteExtra; }
            double inc = rho * workers * (s / (s + 6.0 * workers)) / sf0 * tf;
            if (idx < seq.Length && t < lastInvest)
            {
                double cost = seq[idx] == 0 ? RecruitPrice(recruits) : 6;
                if (iron >= cost)
                {
                    iron -= cost;
                    if (seq[idx] == 0) { recruits++; pendingW++; nextW = Math.Min(nextW, t + RecruitLag); }
                    else { pendingB++; nextB = Math.Min(nextB, t + (int)siteDist + 1); g -= rho * siteDist; }
                    idx++;
                }
                else if (!mineIron) { iron += inc / 3; inc = 0; }
                else if (ironRate > 0) { iron += ironRate; inc -= inc / workers; }
            }
            g += inc;
        }
        return g;
    }

    /// <summary>Best kg iron per turn for one worker shuttling from our bases.</summary>
    double IronRate(List<Dep> deps)
    {
        double best = 0.02;
        foreach (var d in deps)
        {
            if (d.Gold || depAvail[d.Cell] < 1) continue;
            float ret = baseField.Dist[d.Cell];
            if (float.IsInfinity(ret)) continue;
            double y = Math.Min(depAvail[d.Cell], 8);
            best = Math.Max(best, y / (2 * ret + y + 1));
        }
        return best;
    }

    /// <summary>Best base site: trip-time gain from the measured mean trip and unmineable gold the site would co-claim.</summary>
    (int Site, double Gain, double Extra, float Dist) BestSite(List<Dep> deps, List<Worker> mine, double rho, int turn)
    {
        var m = mem!;
        int bestSite = -1;
        double bestScore = 0, bestGain = 0, bestExtra = 0;
        float bestDist = 0;
        var seen = new HashSet<int>();
        double totalA = 0;
        var gx0 = new List<(int X, int Y, float Est, float Cur)>();
        foreach (var d in deps)
        {
            if (!d.Gold) continue;
            float cur = baseField.Dist[d.Cell];
            if (!float.IsInfinity(cur)) totalA += d.Est;
            gx0.Add((m.X(d.Cell), m.Y(d.Cell), d.Est, float.IsInfinity(cur) ? 40 : cur));
        }
        if (totalA <= 0) return (-1, 0, 0, 0);
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
                if (!seen.Add(s) || !SiteValid(s, turn) || baseField.Dist[s] < 4) continue;
                float bd = float.PositiveInfinity;
                for (int k = 0; k < mine.Count; k++) bd = Math.Min(bd, fields[k].Dist[s]);
                if (float.IsInfinity(bd)) continue;
                double saved = 0;
                foreach (var (dx0, dy0, est, cur) in gx0)
                {
                    double ds = Math.Max(Math.Abs(dx0 - x), Math.Abs(dy0 - y)) * 1.15;
                    if (ds < cur) saved += est * 2 * (cur - ds);
                }
                saved /= totalA;
                // Gold within claim range we cannot mine today (enemy claim) becomes ours to share.
                double extra = 0;
                for (int ey = -2; ey <= 2; ey++)
                for (int ex = -2 + Math.Abs(ey); ex <= 2 - Math.Abs(ey); ex++)
                {
                    if (!m.InBounds(x + ex, y + ey)) continue;
                    int t = m.Index(x + ex, y + ey);
                    if (m.Kind[t] == Tile.Gold && m.Amount[t] > 0 && !m.Claim(t).Mineable) extra += m.Amount[t];
                }
                double gain = tripEma / (tripEma - Math.Min(saved, 0.6 * tripEma)) - 1;
                double score = gain * 100 + extra * 0.5 - rho * bd;
                if (score > bestScore) { bestScore = score; bestSite = s; bestGain = gain; bestExtra = extra; bestDist = bd; }
            }
        }
        return (bestSite, bestGain, bestExtra, bestDist);
    }

    bool SiteValid(int s, int turn) =>
        mem!.Kind[s] == Tile.Plain && mem.BaseOwnerAt[s] == null && mem.FarFromEnemyBases(s) && buildBlockedUntil[s] <= turn;

    string? NearestReady(List<Worker> mine, int s, int turn)
    {
        string? best = null;
        float bestDist = float.PositiveInfinity;
        for (int k = 0; k < mine.Count; k++)
        {
            float d = fields[k].Dist[s] + Math.Max(0, mine[k].ReadyOnTurn - turn) + Load(mine[k]) * 0.3f;
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
            if (turn < u.ReadyOnTurn || u.UnitId == builder) continue;
            int pos = mem!.Index(u.Position);
            double s = -Load(u) / 8.0;
            if (mem.Kind[pos] is Tile.Gold && mem.Amount[pos] >= 3) s += 3;
            if (mem.BaseOwnerAt[pos] == mem.TeamId) s += 2;
            if (s > bestScore) { bestScore = s; best = u.UnitId; }
        }
        return best;
    }

    (MarsAction, string, int) Hoard(int k, Worker u, int pos, long gold, long iron, bool canBank, List<Dep> deps, int left)
    {
        if (!canBank && iron >= 3) return (MarsAction.Trade(Resource.Iron), "trade", -1);
        if (u.InventoryCapacity - gold - iron <= 0) return (MarsAction.Pass(), "pass", -1);
        int target = -1;
        float bestCost = float.PositiveInfinity;
        foreach (var d in deps)
        {
            float dist = fields[k].Dist[d.Cell];
            if (float.IsInfinity(dist) || dist + 1 > left) continue;
            float cost = dist + (d.Gold ? 0 : 6);
            if (cost < bestCost) { bestCost = cost; target = d.Cell; }
        }
        if (target < 0) return (MarsAction.Pass(), "pass", -1);
        if (target == pos) return (MarsAction.Mine(), "mine", -1);
        int step = fields[k].FirstStep(target);
        return step >= 0 ? (Move(u.Position, step), "move", step) : (MarsAction.Pass(), "pass", -1);
    }

    MarsAction Wander(int pos)
    {
        var m = mem!;
        for (int a = 0; a < 12; a++)
        {
            var (dx, dy, dir) = Directions.All[rng.Next(8)];
            int x = m.X(pos) + dx, y = m.Y(pos) + dy;
            if (m.InBounds(x, y) && !float.IsInfinity(enterCost[m.Index(x, y)])) return MarsAction.Move(dir);
        }
        return MarsAction.Pass();
    }

    MarsAction Move(Coordinate from, int to) => MarsAction.Move(Directions.Of(mem!.X(to) - from.X, mem.Y(to) - from.Y));

    static long Load(Worker u) => (u.Inventory?.Gold ?? 0) + (u.Inventory?.Iron ?? 0);

    int Manhattan(int a, int b) => Math.Abs(mem!.X(a) - mem.X(b)) + Math.Abs(mem.Y(a) - mem.Y(b));
}
