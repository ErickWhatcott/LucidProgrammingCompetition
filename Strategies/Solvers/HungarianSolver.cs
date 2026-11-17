using Lpc.Client;
using Lpc.Strategies.Planner;

namespace Lpc.Strategies.Solvers;

/// <summary>
/// Global optimal matching. Every turn all workers are matched to columns (8 kg deposit slots, two-stop tours,
/// return home, frontier cells) by a rectangular Hungarian solve. The rate objective sum(v)/sum(t) is linearised
/// with Dinkelbach (profit = v - rho*t, rho re-set to the achieved rate and solved again). Investments come from
/// counterfactual re-solves: a phantom worker on the spawner's tile, or every return time shortened by a new base.
/// </summary>
public sealed class HungarianSolver : IMarsStrategy
{
    public string Name => "hungarian";

    const double Big = 1e6, IdleCost = 1e5;
    const int StopInvestTurn = 300, MaxBases = 6, SlotKg = 8;

    enum ColKind : byte { Slot, Tour, Return, Explore }

    struct Col
    {
        public ColKind Kind;
        public int Cell, Cell2;   // target (tour: first stop), tour second stop
        public int Amount, Amount2;
        public float Value, Value2; // gold-equivalent per kg
        public int Owner;           // Return: the only row that may take it
        public float Info;          // Explore: expected kg found
    }

    sealed class Row
    {
        public int K = -1;          // index into mine, -1 = phantom
        public PathField Field = null!;
        public float Wait;
        public int Free, Pos;
        public float Carried;
    }

    sealed class WorkerState
    {
        public ColKind? LastKind;
        public int LastCell = -2;
        public string LastAction = "";
        public int LastActionCell = -1;
        public long LastLoad;
        public Coordinate LastPos;
        public int Streak;
    }

    WorldMemory? mem;
    float[] enterCost = [];
    readonly PathField baseField = new();
    readonly List<PathField> fields = [];
    int[] mineBlockedUntil = [], mineFailures = [], moveBlockedUntil = [], buildBlockedUntil = [];
    int initialWorkers = -1;
    double rhoEma = 0.35;
    long lastGold = -1;
    int planSite = -1, planWait;
    string? planBuilder;
    enum Invest { None, Recruit, Base }
    Invest saving = Invest.None;
    readonly Dictionary<string, WorkerState> states = [];

    // Per-turn context used by Eval
    readonly List<Col> cols = [];
    int turnsLeft, site = -1;
    WorkerState?[] rowState = [];
    double[] V = [], T = [], cost = [], u = [], v = [], minv = [];
    int[] p = [], way = [];
    bool[] used = [];

    static readonly int TraceFrom = int.TryParse(Environment.GetEnvironmentVariable("HUNGARIAN_FROM"), out var f) ? f : 9999;
    static readonly bool Trace = Environment.GetEnvironmentVariable("HUNGARIAN_TRACE") == "1";

    public HungarianSolver(int seed) { _ = seed; }

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
            enterCost = new float[n];
            mineBlockedUntil = new int[n]; mineFailures = new int[n]; moveBlockedUntil = new int[n]; buildBlockedUntil = new int[n];
        }
        int turn = view.Turn;
        int maxTurns = state.Match.MaxTurns > 0 ? state.Match.MaxTurns : 480;
        turnsLeft = maxTurns - turn;
        var mine = view.Units.Where(u => u.TeamId == me).OrderBy(u => u.UnitId, StringComparer.Ordinal).ToList();
        if (mine.Count == 0) return actions;
        if (initialWorkers < 0) initialWorkers = mine.Count;
        var bank = view.TeamResources?.GetValueOrDefault(me) ?? new Inventory(0, 0);

        // ---- Terrain ----
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
        var myBases = mem.MyBases.ToList();
        baseField.To(myBases.Select(b => mem.Index(b.Position)), enterCost, w, h);
        while (fields.Count < mine.Count) fields.Add(new PathField());
        for (int k = 0; k < mine.Count; k++) fields[k].From(mem.Index(mine[k].Position), enterCost, w, h, 45);

        // ---- Learn from last turn ----
        foreach (var unit in mine)
        {
            if (!states.TryGetValue(unit.UnitId, out var ws)) continue;
            long load = Load(unit);
            int pos = mem.Index(unit.Position);
            if (ws.LastAction == "mine" && unit.Position == ws.LastPos && load == ws.LastLoad
                && load < unit.InventoryCapacity && mem.Kind[pos] is Tile.Gold or Tile.Iron && mem.Amount[pos] > 0)
            {
                ws.Streak = 0;
                if (++mineFailures[pos] >= 2) { mineBlockedUntil[pos] = turn + 25; mineFailures[pos] = 0; }
            }
            else if (ws.LastAction == "mine" && load > ws.LastLoad) { mineFailures[pos] = 0; ws.Streak++; }
            else ws.Streak = 0;
            if (ws.LastAction == "move" && unit.Position == ws.LastPos && turn >= unit.ReadyOnTurn && ws.LastActionCell >= 0)
                moveBlockedUntil[ws.LastActionCell] = turn + 40;
            if (ws.LastAction == "build" && mem.BaseOwnerAt[pos] != me)
            {
                buildBlockedUntil[pos] = turn + 60;
                if (planSite == pos) { planSite = -1; planBuilder = null; }
            }
        }

        // Team gold per worker-turn (EMA of mined gold).
        long goldNow = bank.Gold + mine.Sum(u => u.Inventory?.Gold ?? 0);
        if (lastGold >= 0) rhoEma += (Math.Max(0, goldNow - lastGold) / (double)mine.Count - rhoEma) / 50.0;
        lastGold = goldNow;
        rhoEma = Math.Clamp(rhoEma, 0.1, 1.5);

        // ---- Investment state ----
        bool invest = turn < StopInvestTurn;
        int recruits = mine.Count - initialWorkers;
        int price = RecruitPrice(recruits);
        if (planSite >= 0 && (!invest || !SiteValid(planSite, turn) || mine.All(u => u.UnitId != planBuilder)))
        {
            planSite = -1; planBuilder = null;
        }
        long avail = bank.Iron - (planSite >= 0 ? 6 : 0);
        long desired = saving switch { Invest.Recruit => price, Invest.Base => 6, _ => 0 };
        bool ironWanted = invest && desired > avail;
        float ironValue = ironWanted ? 1.0f : 0.2f;

        // ---- Deposits ----
        var enemyAt = new Dictionary<int, int>();
        foreach (var un in view.Units)
            if (un.TeamId != me) { int i = mem.Index(un.Position); enemyAt[i] = enemyAt.GetValueOrDefault(i) + 1; }
        var deps = new List<(int Cell, int Est, bool Gold)>();
        float knownGold = 0;
        for (int i = 0; i < n; i++)
        {
            if (mem.Kind[i] is not (Tile.Gold or Tile.Iron) || mem.Amount[i] <= 0 || mineBlockedUntil[i] > turn) continue;
            if (!mem.Claim(i).Mineable) continue;
            float est = mem.Amount[i];
            if (!mem.Visible[i] && turn - mem.LastSeen[i] > 25) est *= 0.75f;
            est -= 3 * enemyAt.GetValueOrDefault(i);
            if (est < 1) continue;
            bool gold = mem.Kind[i] == Tile.Gold;
            deps.Add((i, (int)est, gold));
            if (gold) knownGold += est;
        }

        // ---- Columns ----
        cols.Clear();
        foreach (var d in deps)
        {
            float val = d.Gold ? 1f : ironValue;
            for (int left = d.Est; left > 0; left -= SlotKg)
                cols.Add(new Col { Kind = ColKind.Slot, Cell = d.Cell, Amount = Math.Min(SlotKg, left), Value = val, Owner = -1 });
            if (d.Est >= SlotKg) continue;
            // Two-stop tour: A plus the nearest other deposit within Chebyshev 4.
            int best = -1, bestD = 5;
            for (int b = 0; b < deps.Count; b++)
            {
                if (deps[b].Cell == d.Cell) continue;
                int cd = Cheb(d.Cell, deps[b].Cell);
                if (cd < bestD || (cd == bestD && best >= 0 && deps[b].Gold && !deps[best].Gold)) { bestD = cd; best = b; }
            }
            if (best >= 0 && bestD <= 4)
                cols.Add(new Col
                {
                    Kind = ColKind.Tour, Cell = d.Cell, Amount = d.Est, Value = val, Cell2 = deps[best].Cell,
                    Amount2 = deps[best].Est, Value2 = deps[best].Gold ? 1f : ironValue, Owner = -1,
                });
        }
        for (int k = 0; k < mine.Count; k++)
            if (Load(mine[k]) > 0) cols.Add(new Col { Kind = ColKind.Return, Cell = -1, Owner = k });
        if (turnsLeft > 30) AddScanTargets(mine, turn);

        // ---- Rows ----
        Row MakeRow(int k)
        {
            var un = mine[k];
            long g = un.Inventory?.Gold ?? 0, ir = un.Inventory?.Iron ?? 0;
            return new Row
            {
                K = k, Field = fields[k], Wait = Math.Max(0, un.ReadyOnTurn - turn), Pos = mem.Index(un.Position),
                Free = (int)(un.InventoryCapacity - g - ir), Carried = g + ir * ironValue,
            };
        }
        var rows = new List<Row>();
        for (int k = 0; k < mine.Count; k++) if (mine[k].UnitId != planBuilder) rows.Add(MakeRow(k));
        rowState = mine.Select(un => states.GetValueOrDefault(un.UnitId)).ToArray();

        site = -1;
        var baseAssign = Solve(rows, out double baseThroughput, out double rho);

        // ---- Investment: counterfactual re-solves ----
        string? spawner = null;
        bool tradeForIron = false;
        if (invest)
        {
            double recruitRatio = 0, baseRatio = 0;
            int bestSite = -1;
            double recruitCost = 0, baseCost = 0;
            string? spawnCandidate = PickSpawner(mine, turn);
            if (spawnCandidate != null && turnsLeft > 40)
            {
                int sk = mine.FindIndex(un => un.UnitId == spawnCandidate);
                var phantom = new Row { K = -1, Field = fields[sk], Wait = 1, Pos = mem.Index(mine[sk].Position), Free = 8, Carried = 0 };
                rows.Add(phantom);
                Solve(rows, out double withRecruit, out _);
                rows.RemoveAt(rows.Count - 1);
                double ironCost = bank.Iron >= price ? 1 : 3;
                recruitCost = 1.4 * price * ironCost;
                double benefit = (withRecruit - baseThroughput) * (turnsLeft - 25);
                if (benefit >= recruitCost) recruitRatio = benefit / recruitCost;
            }
            if (planSite < 0 && myBases.Count < MaxBases)
            {
                double ironCost = avail >= 6 ? 1 : 3;
                baseCost = 1.3 * 6 * ironCost;
                foreach (int s in Sites(deps, turn))
                {
                    float builderDist = float.PositiveInfinity;
                    foreach (var r in rows) builderDist = Math.Min(builderDist, r.Field.Dist[s] + r.Wait);
                    if (float.IsInfinity(builderDist)) continue;
                    site = s;
                    Solve(rows, out double withBase, out _);
                    site = -1;
                    double benefit = (withBase - baseThroughput) * (turnsLeft - builderDist) - builderDist * rho;
                    if (benefit >= baseCost && benefit / baseCost > baseRatio) { baseRatio = benefit / baseCost; bestSite = s; }
                }
            }
            // Commit to what we are already saving for unless the other is clearly better.
            if (saving == Invest.Recruit) baseRatio /= 1.2;
            else if (saving == Invest.Base) recruitRatio /= 1.2;
            var choice = baseRatio > 0 && baseRatio >= recruitRatio ? Invest.Base : recruitRatio > 0 ? Invest.Recruit : Invest.None;
            saving = choice;
            if (choice == Invest.Recruit)
            {
                if (bank.Iron - (planSite >= 0 ? 6 : 0) >= price) { spawner = spawnCandidate; saving = Invest.None; }
                else { desired = price; tradeForIron = true; }
            }
            else if (choice == Invest.Base)
            {
                if (avail >= 6)
                {
                    planBuilder = NearestReady(mine, bestSite, turn);
                    if (planBuilder != null) { planSite = bestSite; planWait = 0; saving = Invest.None; }
                }
                else { desired = 6; tradeForIron = true; }
            }
            if (Trace && (turn % 10 == 0 || spawner != null || (planBuilder != null && planWait == 0 && planSite == bestSite)))
                Console.Error.WriteLine($"[hun t{turn}] bank {bank.Gold}g/{bank.Iron}i w{mine.Count} b{myBases.Count} rho {rho:F2} ema {rhoEma:F2} thr {baseThroughput:F2} "
                    + $"price {price} rr {recruitRatio:F2} br {baseRatio:F2} site {bestSite} choice {choice} spawn {spawner != null} deps {deps.Count} cols {cols.Count}");
            if (spawner != null || planBuilder != null)
            {
                // Re-solve without the worker that spawns or builds.
                rows.RemoveAll(r => mine[r.K].UnitId == spawner || mine[r.K].UnitId == planBuilder);
                baseAssign = Solve(rows, out _, out _);
            }
        }

        var target = new int[mine.Count];
        Array.Fill(target, -1);
        for (int r = 0; r < rows.Count; r++) if (baseAssign[r] >= 0) target[rows[r].K] = baseAssign[r];

        // ---- Actions ----
        long ironBought = 0;
        for (int k = 0; k < mine.Count; k++)
        {
            var un = mine[k];
            if (!states.TryGetValue(un.UnitId, out var ws)) states[un.UnitId] = ws = new WorkerState();
            int pos = mem.Index(un.Position);
            long gold = un.Inventory?.Gold ?? 0, iron = un.Inventory?.Iron ?? 0;
            MarsAction action = MarsAction.Pass();
            string kind = "pass";
            int actionCell = -1;
            float home = baseField.Dist[pos] + Math.Max(0, un.ReadyOnTurn - turn);
            bool canBank = home + 1 <= turnsLeft;

            if (turn < un.ReadyOnTurn) kind = "cooldown";
            else if (un.UnitId == spawner) { action = MarsAction.Spawn(); kind = "spawn"; }
            else if (un.UnitId == planBuilder && planSite >= 0)
            {
                if (pos == planSite)
                {
                    bool alone = view.Units.Count(o => o.Position == un.Position) == 1;
                    if (alone && bank.Iron >= 6) { action = MarsAction.BuildBase(); kind = "build"; }
                    else if (++planWait > 3) { buildBlockedUntil[planSite] = turn + 30; planSite = -1; planBuilder = null; }
                }
                else
                {
                    int step = fields[k].FirstStep(planSite);
                    if (step >= 0) { action = Move(un.Position, step); kind = "move"; actionCell = step; }
                    else { planSite = -1; planBuilder = null; }
                }
            }
            else if (mem.BaseOwnerAt[pos] == me && gold + iron > 0)
            {
                long ironShort = desired - avail - iron - ironBought;
                if (tradeForIron && ironShort > 0 && gold >= 3) { action = MarsAction.Trade(Resource.Gold); kind = "trade"; ironBought++; }
                else if (!invest && iron >= 3 && turnsLeft >= 3) { action = MarsAction.Trade(Resource.Iron); kind = "trade"; }
                else { action = MarsAction.Deposit(); kind = "deposit"; }
            }
            else if (gold + iron > 0 && canBank && home + 3 >= turnsLeft)
            {
                int back = baseField.StepToward(pos);
                if (back >= 0) { action = Move(un.Position, back); kind = "move"; actionCell = back; }
            }
            else if (!canBank)
            {
                if (iron >= 3) { action = MarsAction.Trade(Resource.Iron); kind = "trade"; }
                else if (mem.Kind[pos] == Tile.Gold && gold + iron < un.InventoryCapacity) { action = MarsAction.Mine(); kind = "mine"; }
            }
            else if (target[k] >= 0)
            {
                var c = cols[target[k]];
                switch (c.Kind)
                {
                    case ColKind.Return:
                    {
                        int back = baseField.StepToward(pos);
                        if (back >= 0) { action = Move(un.Position, back); kind = "move"; actionCell = back; }
                        break;
                    }
                    case ColKind.Slot or ColKind.Tour when c.Cell == pos:
                        action = MarsAction.Mine(); kind = "mine"; break;
                    default:
                    {
                        int step = fields[k].FirstStep(c.Cell);
                        if (step >= 0) { action = Move(un.Position, step); kind = "move"; actionCell = step; }
                        break;
                    }
                }
                if (Trace) colCount[c.Kind.ToString()] = colCount.GetValueOrDefault(c.Kind.ToString()) + 1;
                if (Trace && ws.LastCell != c.Cell && ws.LastAction == "move") colCount["switch" + c.Kind] = colCount.GetValueOrDefault("switch" + c.Kind) + 1;
                ws.LastKind = c.Kind == ColKind.Tour ? ColKind.Slot : c.Kind;
                ws.LastCell = c.Cell;
            }
            else if (gold + iron > 0)
            {
                int back = baseField.StepToward(pos);
                if (back >= 0) { action = Move(un.Position, back); kind = "move"; actionCell = back; }
            }
            actions[un.UnitId] = action;
            if (Trace && turn >= TraceFrom && turn < TraceFrom + 30)
                Console.Error.WriteLine($"  [w t{turn}] {un.UnitId} @({un.Position.X},{un.Position.Y}) load {gold}/{iron} home {home:F0} -> {(target[k] >= 0 ? $"{cols[target[k]].Kind} ({mem.X(Math.Max(0, cols[target[k]].Cell))},{mem.Y(Math.Max(0, cols[target[k]].Cell))}) amt {cols[target[k]].Amount}" : "none")} {kind}");
            if (Trace) kindCount[kind] = kindCount.GetValueOrDefault(kind) + 1;
            ws.LastAction = kind;
            ws.LastActionCell = actionCell;
            ws.LastLoad = gold + iron;
            ws.LastPos = un.Position;
        }
        if (Trace && turnsLeft == 1)
            Console.Error.WriteLine("[hun kinds] " + string.Join(" ", kindCount.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}={kv.Value}"))
                + " targets " + string.Join(" ", colCount.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}={kv.Value}")));
        return actions;
    }

    readonly Dictionary<string, int> kindCount = [];
    readonly Dictionary<string, int> colCount = [];

    // ---- Assignment ----

    float Ret(int cell)
    {
        float r = baseField.Dist[cell];
        if (site >= 0) r = Math.Min(r, Cheb(cell, site) * 1.2f);
        return r;
    }

    /// <summary>Gold-equivalent value and turns for row r taking column c (T = inf if infeasible).</summary>
    void Eval(Row r, in Col c, int ci, out double value, out double time)
    {
        value = 0; time = double.PositiveInfinity;
        switch (c.Kind)
        {
            case ColKind.Return:
            {
                if (c.Owner != r.K) return;
                float ret = Ret(r.Pos);
                if (float.IsInfinity(ret)) return;
                value = r.Carried; time = ret + r.Wait + 1;
                return;
            }
            case ColKind.Slot:
            {
                float travel = r.Field.Dist[c.Cell], ret = Ret(c.Cell);
                if (float.IsInfinity(travel) || float.IsInfinity(ret)) return;
                travel += r.Wait;
                int take = (int)Math.Min(Math.Min(c.Amount, r.Free), turnsLeft - travel - ret - 2);
                if (take <= 0) return;
                value = r.Carried + take * c.Value; time = travel + take + ret + 1;
                return;
            }
            case ColKind.Tour:
            {
                int takeA = Math.Min(c.Amount, r.Free);
                if (takeA >= r.Free) return;
                float travel = r.Field.Dist[c.Cell], ret = Ret(c.Cell2);
                if (float.IsInfinity(travel) || float.IsInfinity(ret)) return;
                travel += r.Wait;
                int takeB = Math.Min(c.Amount2, r.Free - takeA);
                double t = travel + takeA + Cheb(c.Cell, c.Cell2) * 1.2 + takeB + ret + 1;
                if (t + 1 > turnsLeft) return;
                value = r.Carried + takeA * c.Value + takeB * c.Value2; time = t;
                return;
            }
            case ColKind.Explore:
            {
                float travel = r.Field.Dist[c.Cell];
                if (float.IsInfinity(travel) || r.Free <= 0) return;
                float ret = Ret(c.Cell);
                if (float.IsInfinity(ret)) ret = 40;
                float take = Math.Min(r.Free, c.Info);
                value = r.Carried + take; time = travel + r.Wait + 4 + take + ret + 1;
                return;
            }
        }
    }

    /// <summary>
    /// Dinkelbach + rectangular Hungarian. Returns each row's column (-1 = idle), the team throughput
    /// sum(v/t) and the achieved rate sum(v)/sum(t).
    /// </summary>
    int[] Solve(List<Row> rows, out double throughput, out double rho)
    {
        int nr = rows.Count, m = cols.Count, mm = m + nr;
        var assign = new int[nr];
        Array.Fill(assign, -1);
        throughput = 0; rho = rhoEma;
        if (nr == 0) return assign;
        if (V.Length < nr * m) { V = new double[nr * m]; T = new double[nr * m]; }
        for (int r = 0; r < nr; r++)
            for (int c = 0; c < m; c++)
            {
                Eval(rows[r], cols[c], c, out double val, out double t);
                V[r * m + c] = val; T[r * m + c] = t;
            }
        if (cost.Length < nr * mm) cost = new double[nr * mm];
        for (int iter = 0; iter < 2; iter++)
        {
            for (int r = 0; r < nr; r++)
            {
                var row = rows[r];
                var ws = row.K >= 0 ? rowState[row.K] : null;
                for (int c = 0; c < m; c++)
                {
                    double t = T[r * m + c];
                    if (double.IsInfinity(t)) { cost[r * mm + c] = Big; continue; }
                    double profit = V[r * m + c] - rho * t;
                    if (ws != null)
                    {
                        var col = cols[c];
                        var kind = col.Kind == ColKind.Tour ? ColKind.Slot : col.Kind;
                        if (ws.LastKind == kind && (kind == ColKind.Return || ws.LastCell == col.Cell)) profit += 0.25 * rho * t;
                        if (ws.Streak > 0 && kind == ColKind.Slot && col.Cell == row.Pos) profit += 1.0;
                    }
                    cost[r * mm + c] = -profit;
                }
                for (int c = m; c < mm; c++) cost[r * mm + c] = IdleCost;
            }
            Hungarian(nr, mm, assign);
            double sv = 0, st = 0;
            throughput = 0;
            for (int r = 0; r < nr; r++)
            {
                int c = assign[r];
                if (c >= m || double.IsInfinity(T[r * m + c])) { assign[r] = -1; continue; }
                // Carried gold is sunk: rate and throughput count only what the trip adds.
                double gain = V[r * m + c] - rows[r].Carried;
                sv += gain; st += T[r * m + c];
                throughput += gain / T[r * m + c];
            }
            if (st > 0) rho = Math.Max(0.05, sv / st);
        }
        return assign;
    }

    /// <summary>Kuhn-Munkres with potentials, O(n^2 m) for n rows &lt;= m columns, minimising cost.</summary>
    void Hungarian(int nr, int m, int[] assign)
    {
        if (u.Length < nr + 1) u = new double[nr + 1];
        if (v.Length < m + 1) { v = new double[m + 1]; minv = new double[m + 1]; p = new int[m + 1]; way = new int[m + 1]; used = new bool[m + 1]; }
        Array.Clear(u, 0, nr + 1); Array.Clear(v, 0, m + 1); Array.Clear(p, 0, m + 1); Array.Clear(way, 0, m + 1);
        for (int i = 1; i <= nr; i++)
        {
            p[0] = i;
            int j0 = 0;
            Array.Fill(minv, double.PositiveInfinity, 0, m + 1);
            Array.Clear(used, 0, m + 1);
            do
            {
                used[j0] = true;
                int i0 = p[j0], j1 = 0;
                double delta = double.PositiveInfinity;
                int baseIdx = (i0 - 1) * m;
                for (int j = 1; j <= m; j++)
                {
                    if (used[j]) continue;
                    double cur = cost[baseIdx + j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
                for (int j = 0; j <= m; j++)
                {
                    if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                    else minv[j] -= delta;
                }
                j0 = j1;
            } while (p[j0] != 0);
            do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0 != 0);
        }
        for (int j = 1; j <= m; j++) if (p[j] != 0) assign[p[j] - 1] = j - 1;
    }

    // ---- Exploration ----

    /// <summary>
    /// Scan targets. A deposit can (re)appear on any empty tile, 35% of them in the 16x16 center with twice the gold,
    /// so a tile's expected new gold is density x mean amount, scaled by how long ago we last saw it.
    /// A cell's value is the sum over the tiles its vision would reveal (Manhattan 3).
    /// </summary>
    void AddScanTargets(List<Worker> mine, int turn)
    {
        var m = mem!;
        int w = m.Width, h = m.Height, n = m.Count;
        if (tileGain.Length != n) { tileGain = new float[n]; claimMask = new bool[n]; }
        Array.Clear(claimMask);
        foreach (var b in m.Bases.Values)
        {
            if (b.TeamId == m.TeamId) continue;
            for (int dy = -2; dy <= 2; dy++)
            for (int dx = -2 + Math.Abs(dy); dx <= 2 - Math.Abs(dy); dx++)
                if (m.InBounds(b.Position.X + dx, b.Position.Y + dy)) claimMask[m.Index(b.Position.X + dx, b.Position.Y + dy)] = true;
        }
        foreach (var b in m.MyBases)
            for (int dy = -2; dy <= 2; dy++)
            for (int dx = -2 + Math.Abs(dy); dx <= 2 - Math.Abs(dy); dx++)
                if (m.InBounds(b.Position.X + dx, b.Position.Y + dy)) claimMask[m.Index(b.Position.X + dx, b.Position.Y + dy)] = false;
        float centerGain = 17f / (m.CenterW * m.CenterH) * 15f, outerGain = 31f / (n - m.CenterW * m.CenterH) * 7.5f;
        for (int i = 0; i < n; i++)
        {
            float g = m.InCenter(i) ? centerGain : outerGain;
            tileGain[i] = claimMask[i] ? 0 : m.Kind[i] switch
            {
                Tile.Unknown => g,
                Tile.Plain => m.Visible[i] ? 0 : g * Math.Min(1f, (turn - m.LastSeen[i]) / 80f),
                _ => 0,
            };
        }
        var cand = new List<(int Cell, float Info, float Score)>();
        for (int i = 0; i < n; i++)
        {
            if (float.IsInfinity(enterCost[i])) continue;
            float near = float.PositiveInfinity;
            for (int k = 0; k < mine.Count; k++) near = Math.Min(near, fields[k].Dist[i]);
            if (float.IsInfinity(near) || near < 2) continue;
            int x = m.X(i), y = m.Y(i);
            float sum = 0;
            for (int dy = -3; dy <= 3; dy++)
            {
                if (y + dy < 0 || y + dy >= h) continue;
                int span = 3 - Math.Abs(dy);
                for (int dx = -span; dx <= span; dx++)
                    if (x + dx >= 0 && x + dx < w) sum += tileGain[(y + dy) * w + x + dx];
            }
            if (sum < 0.5f) continue;
            float info = Math.Min(12f, 0.5f * sum);
            cand.Add((i, info, info / (near + 8f)));
        }
        cand.Sort((a, b) => b.Score.CompareTo(a.Score) is var c && c != 0 ? c : a.Cell.CompareTo(b.Cell));
        var picked = new List<int>();
        foreach (var (cell, info, _) in cand)
        {
            if (picked.Any(o => Manhattan(o, cell) <= 5)) continue;
            picked.Add(cell);
            cols.Add(new Col { Kind = ColKind.Explore, Cell = cell, Owner = -1, Info = info });
            if (picked.Count >= 6) break;
        }
    }

    float[] tileGain = [];
    bool[] claimMask = [];

    // ---- Investment helpers ----

    bool SiteValid(int s, int turn) =>
        mem!.Kind[s] == Tile.Plain && mem.BaseOwnerAt[s] == null && mem.FarFromEnemyBases(s) && buildBlockedUntil[s] <= turn;

    /// <summary>The 6 best valid sites near gold clusters by time saved on known gold.</summary>
    List<int> Sites(List<(int Cell, int Est, bool Gold)> deps, int turn)
    {
        var m = mem!;
        var score = new Dictionary<int, double>();
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
                if (score.ContainsKey(s) || !SiteValid(s, turn) || baseField.Dist[s] < 4) continue;
                double value = 0;
                foreach (var d in deps)
                {
                    if (!d.Gold) continue;
                    int c = Cheb(s, d.Cell);
                    if (c > 6) continue;
                    float cur = baseField.Dist[d.Cell];
                    if (float.IsInfinity(cur)) cur = 40;
                    value += d.Est * Math.Max(0, cur - c * 1.2);
                }
                score[s] = value;
            }
        }
        return score.Where(kv => kv.Value > 0).OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key).Take(6).Select(kv => kv.Key).ToList();
    }

    string? NearestReady(List<Worker> mine, int s, int turn)
    {
        if (s < 0) return null;
        string? best = null;
        float bestDist = float.PositiveInfinity;
        for (int k = 0; k < mine.Count; k++)
        {
            float d = fields[k].Dist[s] + Math.Max(0, mine[k].ReadyOnTurn - turn);
            if (d < bestDist) { bestDist = d; best = mine[k].UnitId; }
        }
        return best;
    }

    string? PickSpawner(List<Worker> mine, int turn)
    {
        string? best = null;
        double bestScore = double.NegativeInfinity;
        foreach (var un in mine)
        {
            if (turn < un.ReadyOnTurn || un.UnitId == planBuilder) continue;
            int pos = mem!.Index(un.Position);
            double s = -Load(un) / 8.0;
            if (mem.Kind[pos] is Tile.Gold && mem.Amount[pos] >= 3) s += 3;
            if (mem.BaseOwnerAt[pos] == mem.TeamId) s += 2;
            if (s > bestScore) { bestScore = s; best = un.UnitId; }
        }
        return best;
    }

    MarsAction Move(Coordinate from, int to) => MarsAction.Move(Directions.Of(mem!.X(to) - from.X, mem.Y(to) - from.Y));

    static long Load(Worker un) => (un.Inventory?.Gold ?? 0) + (un.Inventory?.Iron ?? 0);

    int Manhattan(int a, int b) => Math.Abs(mem!.X(a) - mem.X(b)) + Math.Abs(mem.Y(a) - mem.Y(b));

    int Cheb(int a, int b) => Math.Max(Math.Abs(mem!.X(a) - mem.X(b)), Math.Abs(mem.Y(a) - mem.Y(b)));
}
