using Lpc.Client;
using Lpc.Sim;
using Lpc.Strategies.Planner;

namespace Lpc.Strategies.Solvers;

/// <summary>
/// Role split. One dedicated scout sweeps a 6-spaced waypoint lattice (vision diamonds tile the map), records an
/// enemy-activity heatmap and nominates quiet gold clusters as base sites. Miners never explore: they value
/// remembered deposits with a freshness model (stock decays faster where enemies are busy) and pick by expected
/// take per trip turn, reserving stock across miners.
/// </summary>
public sealed class ScoutSolver(int seed = 0) : IMarsStrategy
{
    public string Name => "scout";

    const int Lattice = 6, ScoutUntil = 300, RecruitUntil = 260, BaseUntil = 320, MaxBases = 4;
    const double CenterPrior = 9.4, HeatDecay = 0.98, SupplyPerMiner = 8, SiteGold = 15, SiteHeat = 20, HeatScale = 0.1, IronWeight = 2, MinMarginal = 0.2;
    const float SearchRadius = 32;

    readonly Random rng = new(seed);
    readonly PathField home = new(), scoutPath = new();
    readonly List<BoundedSearch> fields = [];
    readonly Dictionary<int, double> sites = [];
    readonly Dictionary<string, int> buildAt = [];
    readonly Dictionary<string, (int Pos, long Load)> lastMine = [];
    readonly HashSet<string> initial = [];
    readonly List<int> myBases = [], lattice = [];
    WorldMemory? mem;
    float[] cost = [], reserved = [], expStock = [];
    double[] heat = [], openShare = [];
    int[] openTurn = [];
    int[] occ = [], enemyOn = [], blockedUntil = [], mineFails = [];
    bool[] claimUs = [], claimEnemy = [];
    Tile[] prevKind = [];
    string? scoutId;
    int scoutWp = -1, lastFound = -1, w, h, turn, left;
    string me = "";

    static long RecruitPrice(int recruited)
    {
        long a = 5, b = 6;
        for (int i = 0; i < recruited; i++) (a, b) = (b, a + b);
        return a;
    }

    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity)
    {
        var view = state.View;
        if (view == null) return BotUtil.Empty;
        me = identity.TeamId;
        mem ??= new WorldMemory(me);
        if (prevKind.Length == mem.Count) Array.Copy(mem.Kind, prevKind, prevKind.Length);
        mem.Update(view);
        var m = mem;
        w = m.Width; h = m.Height;
        int n = m.Count;
        if (cost.Length != n)
        {
            cost = new float[n]; reserved = new float[n]; expStock = new float[n]; heat = new double[n]; openShare = new double[n]; openTurn = new int[n];
            occ = new int[n]; enemyOn = new int[n]; blockedUntil = new int[n]; mineFails = new int[n];
            claimUs = new bool[n]; claimEnemy = new bool[n]; prevKind = (Tile[])m.Kind.Clone();
            lattice.Clear();
            for (int y = Lattice / 2; y < h; y += Lattice)
                for (int x = Lattice / 2; x < w; x += Lattice) lattice.Add(y * w + x);
        }
        turn = view.Turn;
        int maxTurns = state.Match.MaxTurns > 0 ? state.Match.MaxTurns : 480;
        left = maxTurns - turn;
        long bankIron = view.TeamResources != null && view.TeamResources.TryGetValue(me, out var inv) ? inv.Iron : 0;

        // ---- Terrain, claims, units, heatmap ----
        Array.Clear(claimUs); Array.Clear(claimEnemy); Array.Clear(occ); Array.Clear(enemyOn); Array.Clear(reserved);
        for (int i = 0; i < n; i++)
        {
            cost[i] = m.Kind[i] switch
            {
                Tile.Unknown => BotUtil.UnknownCost,
                Tile.Mountain or Tile.Crater => BotUtil.ObstacleCost,
                Tile.Void => float.PositiveInfinity,
                _ => 1f,
            };
            heat[i] *= HeatDecay;
        }
        myBases.Clear();
        foreach (var b in m.Bases.Values)
        {
            int bi = m.Index(b.Position);
            bool mine = b.TeamId == me;
            if (mine) myBases.Add(bi); else cost[bi] = float.PositiveInfinity;
            var claim = mine ? claimUs : claimEnemy;
            ForDiamond(bi, 2, t => claim[t] = true);
        }
        if (myBases.Count > 0) home.To(myBases, cost, w, h);
        else { if (home.Dist.Length != n) { home.Dist = new float[n]; home.Prev = new int[n]; } Array.Fill(home.Dist, float.PositiveInfinity); Array.Fill(home.Prev, -1); }

        var mine0 = new List<Worker>();
        foreach (var u in view.Units)
        {
            int p = m.Index(u.Position);
            occ[p]++;
            if (u.TeamId == me) mine0.Add(u);
            else { enemyOn[p]++; ForDiamond(p, 3, t => heat[t] += 1); }
        }
        mine0.Sort((a, b) => string.CompareOrdinal(a.UnitId, b.UnitId));
        if (turn <= 0 || initial.Count == 0) foreach (var u in mine0) initial.Add(u.UnitId);
        if (scoutId == null) scoutId = mine0.FirstOrDefault(u => !initial.Contains(u.UnitId))?.UnitId;
        int myCount = mine0.Count;
        var actions = new Dictionary<string, MarsAction>(myCount);
        if (myCount == 0) return actions;

        // ---- Mine failures (contested deposit, lost the queue) ----
        foreach (var u in mine0)
        {
            int p = m.Index(u.Position);
            if (!lastMine.TryGetValue(u.UnitId, out var lm) || lm.Pos != p) continue;
            if (Carried(u) <= lm.Load && Carried(u) < u.InventoryCapacity)
            {
                if (++mineFails[p] >= 2) { blockedUntil[p] = turn + 20; mineFails[p] = 0; }
            }
            else mineFails[p] = 0;
        }
        lastMine.Clear();

        // ---- Freshness model: expected stock of every remembered deposit ----
        var deposits = new List<int>();
        double supply = 0;
        for (int i = 0; i < n; i++)
        {
            expStock[i] = 0;
            if (m.Kind[i] is not (Tile.Gold or Tile.Iron) || m.Amount[i] <= 0) continue;
            if (blockedUntil[i] > turn || !(claimUs[i] || !claimEnemy[i])) continue;
            int age = m.Visible[i] ? 0 : turn - m.LastSeen[i];
            double decay = 0.004 + 0.02 * Math.Min(heat[i] * HeatScale, 4);
            expStock[i] = (float)(m.Amount[i] * Math.Exp(-age * decay));
            if (expStock[i] < 0.6f) continue;
            deposits.Add(i);
            if (m.Kind[i] == Tile.Gold && home.Dist[i] < 30) supply += expStock[i];
        }

        // ---- Scout: lattice sweep, discoveries, site nominations ----
        Worker? scout = turn < ScoutUntil ? mine0.FirstOrDefault(u => u.UnitId == scoutId) : null;
        foreach (var u in mine0)
        {
            int sp = m.Index(u.Position);
            int best = -1; long bestAmt = 0;
            bool isScout = u == scout;
            ForDiamond(sp, 3, t =>
            {
                if (isScout && m.Kind[t] == Tile.Gold && prevKind[t] != Tile.Gold && Mineable(t) && m.Amount[t] > bestAmt) { bestAmt = m.Amount[t]; best = t; }
                if (m.Kind[t] == Tile.Plain) Nominate(t);
            });
            if (best >= 0) lastFound = best;
        }
        foreach (var s in sites.Keys.ToList())
        {
            double v = SiteValue(s);
            if (v < SiteGold) sites.Remove(s); else sites[s] = v;
        }

        // ---- Investment ----
        int basesNow = myBases.Count;
        long nextPrice = RecruitPrice(Math.Max(0, myCount - 1));
        int miners = myCount - (scout != null ? 1 : 0);
        bool openRecruit = turn <= 1 && myCount < 3;
        bool wantRecruit = openRecruit || (turn < RecruitUntil && left > 80 && (supply / Math.Max(1, miners) >= SupplyPerMiner || nextPrice <= 11));
        bool wantBase = sites.Count > 0 && basesNow < MaxBases && turn < BaseUntil && left > 60;
        long ironNeed = wantBase ? 6 : wantRecruit ? nextPrice : 0;
        bool wantIron = bankIron < ironNeed;
        long ironAvail = bankIron;
        foreach (var id in buildAt.Keys.ToList())
            if (!mine0.Exists(u => u.UnitId == id) || !sites.ContainsKey(buildAt[id]) || basesNow >= MaxBases || left < 30) buildAt.Remove(id);

        // ---- Scout moves ----
        if (scout != null && turn >= scout.ReadyOnTurn)
        {
            int sp = m.Index(scout.Position);
            if (m.BaseOwnerAt[sp] == me && Carried(scout) > 0) actions[scout.UnitId] = MarsAction.Deposit();
            else
            {
                if (scoutWp < 0 || Age(scoutWp) == 0 || Cheb(sp, scoutWp) <= 1) scoutWp = PickWaypoint(sp);
                scoutPath.From(sp, cost, w, h, Cheb(sp, scoutWp) * 2 + 8);
                int step = scoutPath.FirstStep(scoutWp);
                actions[scout.UnitId] = step >= 0 ? MoveTo(sp, step) : StepToward(sp, scoutWp);
            }
        }

        // ---- Miners ----
        var pool = new List<(Worker U, int Pos, BoundedSearch S)>();
        bool spawned = false;
        int f = 0;
        foreach (var u in mine0)
        {
            if (u == scout || turn < u.ReadyOnTurn) continue;
            int pos = m.Index(u.Position);
            long carried = Carried(u), cap = u.InventoryCapacity;
            bool onBase = m.BaseOwnerAt[pos] == me;
            float hd = home.Dist[pos];

            if (onBase && carried > 0)
            {
                long g = u.Inventory?.Gold ?? 0, ir = u.Inventory?.Iron ?? 0;
                bool trade = ironNeed > 0 && g >= 3 && ironAvail + ir < ironNeed && turn < BaseUntil;
                actions[u.UnitId] = trade ? MarsAction.Trade(Resource.Gold) : MarsAction.Deposit();
                continue;
            }
            if (carried > 0 && !float.IsInfinity(hd) && hd + 2 >= left) { actions[u.UnitId] = StepHome(pos) ?? MarsAction.Pass(); continue; }

            if (!spawned && wantRecruit && (openRecruit || !wantBase || ironAvail - 6 >= nextPrice) && ironAvail >= nextPrice && carried < cap)
            {
                actions[u.UnitId] = MarsAction.Spawn();
                ironAvail -= nextPrice; spawned = true;
                continue;
            }

            if (buildAt.TryGetValue(u.UnitId, out int site))
            {
                if (ironAvail < 6) buildAt.Remove(u.UnitId);
                else if (site == pos && occ[pos] == 1)
                {
                    actions[u.UnitId] = MarsAction.BuildBase();
                    ironAvail -= 6; basesNow++; sites.Remove(site);
                    buildAt.Remove(u.UnitId);
                    continue;
                }
                else if (site != pos)
                {
                    actions[u.UnitId] = StepToward(pos, site);
                    continue;
                }
                else buildAt.Remove(u.UnitId);
            }

            while (fields.Count <= f) fields.Add(new BoundedSearch());
            var s = fields[f++];
            s.Run(pos, cost, w, h, SearchRadius);
            pool.Add((u, pos, s));
        }

        // Base building: the miner passing closest to a nominated site detours to build it.
        if (wantBase && buildAt.Count == 0 && ironAvail >= 6)
        {
            int bi = -1, bs = -1; float bd = 10;
            for (int k = 0; k < pool.Count; k++)
                foreach (var s in sites.Keys)
                    if (pool[k].S.Reached(s) && pool[k].S.Dist[s] < bd && Carried(pool[k].U) < pool[k].U.InventoryCapacity)
                    { bd = pool[k].S.Dist[s]; bi = k; bs = s; }
            if (bi >= 0)
            {
                var (u, pos, _) = pool[bi];
                pool.RemoveAt(bi);
                if (bs == pos && occ[pos] == 1) { actions[u.UnitId] = MarsAction.BuildBase(); ironAvail -= 6; sites.Remove(bs); }
                else { buildAt[u.UnitId] = bs; actions[u.UnitId] = StepToward(pos, bs); }
            }
        }

        // Iron goes to the nearest miner only.
        var ironOwner = new Dictionary<int, int>();
        if (wantIron)
            foreach (int c in deposits)
            {
                if (m.Kind[c] != Tile.Iron) continue;
                int bk = -1; float bd = float.PositiveInfinity;
                for (int k = 0; k < pool.Count; k++)
                    if (pool[k].S.Reached(c) && pool[k].S.Dist[c] < bd) { bd = pool[k].S.Dist[c]; bk = k; }
                if (bk >= 0) ironOwner[c] = bk;
            }

        // Global greedy assignment: repeatedly commit the miner with the best expected rate.
        var done = new bool[pool.Count];
        for (int round = 0; round < pool.Count; round++)
        {
            int bk = -1, bt = -1; double bsc = double.NegativeInfinity;
            for (int k = 0; k < pool.Count; k++)
            {
                if (done[k]) continue;
                var (t, sc) = BestTarget(pool[k].U, pool[k].Pos, pool[k].S, deposits, ironOwner, k);
                if (sc > bsc) { bsc = sc; bk = k; bt = t; }
            }
            if (bk < 0) break;
            done[bk] = true;
            var (u, pos, s) = pool[bk];
            long room = u.InventoryCapacity - Carried(u);
            if (bt >= 0)
            {
                reserved[bt] += room;
                if (bt == pos) { actions[u.UnitId] = MarsAction.Mine(); lastMine[u.UnitId] = (pos, Carried(u)); continue; }
                int step = s.FirstStep(pos, bt);
                if (step >= 0) { actions[u.UnitId] = MoveTo(pos, step); continue; }
            }
            actions[u.UnitId] = Idle(u, pos, scout);
        }
        return actions;
    }

    /// <summary>Best deposit by expected take / (travel + take + return + 1); a loaded miner must beat going home.</summary>
    (int Target, double Score) BestTarget(Worker u, int pos, BoundedSearch s, List<int> deposits, Dictionary<int, int> ironOwner, int k)
    {
        var m = mem!;
        long carried = Carried(u), room = u.InventoryCapacity - carried;
        if (room <= 0) return (-1, -1);
        int best = -1;
        double bestScore = 0;
        float hp = home.Dist[pos];
        foreach (int c in deposits)
        {
            if (!s.Reached(c)) continue;
            bool gold = m.Kind[c] == Tile.Gold;
            if (!gold && (!ironOwner.TryGetValue(c, out int owner) || owner != k)) continue;
            float d = s.Dist[c];
            float back = home.Dist[c];
            if (float.IsInfinity(back)) back = 30;
            double avail = expStock[c] - reserved[c];
            if (m.Visible[c] && enemyOn[c] > 0 && c != pos) avail -= enemyOn[c] * Math.Min(d + 1, 4);
            if (avail < 1) continue;
            double take = Math.Min(avail, room);
            if (d + take + back + 2 >= left) continue;
            double score = take * (gold ? 1 : IronWeight) / (d + take + back + 1);
            if (c == pos) score *= 1.15; // keep the mining run
            if (score > bestScore) { bestScore = score; best = c; }
        }
        if (best >= 0 && carried > 0 && !float.IsInfinity(hp))
        {
            float d = s.Dist[best], back = float.IsInfinity(home.Dist[best]) ? 30 : home.Dist[best];
            double take = Math.Min(expStock[best] - reserved[best], room);
            if (take / (d + take + back - hp + 1) < MinMarginal) return (-1, carried / (hp + 1.0));
        }
        return (best, best >= 0 ? bestScore : 0);
    }

    MarsAction Idle(Worker u, int pos, Worker? scout)
    {
        var m = mem!;
        if (Carried(u) > 0) { var hm = StepHome(pos); if (hm != null) return hm; }
        int goal = lastFound >= 0 && Cheb(pos, lastFound) > 1 && Mineable(lastFound) && Age(lastFound) < 40 ? lastFound : HomeSweep(pos);
        if (goal < 0 && scout != null) goal = m.Index(scout.Position);
        if (goal < 0 || goal == pos)
            goal = m.Index(new Coordinate(m.CenterX0 + rng.Next(Math.Max(1, m.CenterW)), m.CenterY0 + rng.Next(Math.Max(1, m.CenterH))));
        return StepToward(pos, goal);
    }

    /// <summary>Waypoint score: prior gold density * min(age, 80) / (10 + distance).</summary>
    int PickWaypoint(int pos)
    {
        var m = mem!;
        int best = -1;
        double bestScore = double.NegativeInfinity;
        foreach (int c in lattice)
        {
            if (c == pos || float.IsInfinity(cost[c])) continue;
            double prior = (m.InCenter(c) ? CenterPrior : 1) * Open(c) / (1 + heat[c] * HeatScale);
            double score = prior * Math.Min(Age(c), 80) / (10.0 + Cheb(pos, c));
            if (score > bestScore) { bestScore = score; best = c; }
        }
        return best;
    }

    /// <summary>Share of the vision diamond around c not claimed by an enemy base.</summary>
    double Open(int c)
    {
        if (openTurn[c] == turn + 1) return openShare[c];
        int open = 0, all = 0;
        ForDiamond(c, 3, t => { all++; if (!claimEnemy[t] || claimUs[t]) open++; });
        openTurn[c] = turn + 1;
        return openShare[c] = (double)open / all;
    }

    /// <summary>Idle miners re-check the stalest lattice point near home (no long exploration).</summary>
    int HomeSweep(int pos)
    {
        int best = -1;
        double bestScore = 0;
        foreach (int c in lattice)
        {
            if (float.IsInfinity(cost[c]) || home.Dist[c] > 14) continue;
            double score = Math.Min(Age(c), 80) * Open(c) / (10.0 + Cheb(pos, c));
            if (score > bestScore) { bestScore = score; best = c; }
        }
        return best;
    }

    int Age(int c) => mem!.Visible[c] ? 0 : turn - mem.LastSeen[c];

    void Nominate(int s)
    {
        if (sites.ContainsKey(s)) return;
        double v = SiteValue(s);
        if (v < SiteGold) return;
        sites[s] = v;
        if (sites.Count > 12) sites.Remove(sites.MinBy(kv => kv.Value).Key);
    }

    /// <summary>Expected gold within Chebyshev 3, minus heat; 0 if the site is not buildable or too close to home.</summary>
    double SiteValue(int s)
    {
        var m = mem!;
        if (m.Kind[s] != Tile.Plain || m.BaseOwnerAt[s] != null || claimEnemy[s]) return 0; // claimEnemy = within 2 of an enemy base
        if (heat[s] > SiteHeat || (myBases.Count > 0 && home.Dist[s] < 7)) return 0;
        int sx = s % w, sy = s / w;
        double gold = 0;
        for (int dy = -3; dy <= 3; dy++)
            for (int dx = -3; dx <= 3; dx++)
            {
                int x = sx + dx, y = sy + dy;
                if (x < 0 || y < 0 || x >= w || y >= h) continue;
                int t = y * w + x;
                if (m.Kind[t] == Tile.Gold) gold += expStock[t];
            }
        return gold - heat[s];
    }

    static long Carried(Worker u) => (u.Inventory?.Gold ?? 0) + (u.Inventory?.Iron ?? 0);
    bool Mineable(int i) => claimUs[i] || !claimEnemy[i];
    int Cheb(int a, int b) => Math.Max(Math.Abs(a % w - b % w), Math.Abs(a / w - b / w));

    void ForDiamond(int c, int r, Action<int> act)
    {
        int cx = c % w, cy = c / w;
        for (int dy = -r; dy <= r; dy++)
            for (int dx = -r + Math.Abs(dy); dx <= r - Math.Abs(dy); dx++)
            {
                int x = cx + dx, y = cy + dy;
                if (x >= 0 && y >= 0 && x < w && y < h) act(y * w + x);
            }
    }

    MarsAction MoveTo(int from, int to) => MarsAction.Move(Directions.Of(to % w - from % w, to / w - from / w));

    MarsAction? StepHome(int pos)
    {
        if (float.IsInfinity(home.Dist[pos])) return null;
        int next = home.StepToward(pos);
        return next >= 0 && next != pos ? MoveTo(pos, next) : null;
    }

    /// <summary>Greedy step toward a far tile: minimize Chebyshev distance, penalize obstacles.</summary>
    MarsAction StepToward(int pos, int goal)
    {
        int px = pos % w, py = pos / w, gx = goal % w, gy = goal / w;
        double best = double.PositiveInfinity;
        int bestN = -1;
        foreach (var (dx, dy, _) in Directions.All)
        {
            int x = px + dx, y = py + dy;
            if (x < 0 || y < 0 || x >= w || y >= h) continue;
            int ni = y * w + x;
            if (float.IsInfinity(cost[ni])) continue;
            double s = Math.Max(Math.Abs(gx - x), Math.Abs(gy - y)) + (cost[ni] - 1) * 0.8
                       + 0.01 * (Math.Abs(gx - x) + Math.Abs(gy - y));
            if (s < best) { best = s; bestN = ni; }
        }
        return bestN < 0 ? MarsAction.Pass() : MoveTo(pos, bestN);
    }
}
