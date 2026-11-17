using Lpc.Client;
using Lpc.Strategies.Planner;

namespace Lpc.Strategies.Solvers;

/// <summary>
/// Territory solver: splits the map by path-distance races against known enemy bases (margin
/// m = dEnemy - dUs), mines only deposits in its own region, places bases by greedy facility location
/// (k-median over known gold plus the expected-respawn prior) and explores its region on a lane grid.
/// </summary>
public sealed class TerritorySolver(int seed = 0) : IMarsStrategy
{
    public string Name => "territory";
    public static bool Trace = Environment.GetEnvironmentVariable("TERRITORY_TRACE") == "1";

    // Expected gold arriving per tile per respawn: 35% of respawns at ~15 kg into 256 center tiles,
    // 65% at ~7.5 kg into the 2244 outside tiles.
    const double PriorCenter = 0.35 * 15 / 256, PriorOutside = 0.65 * 7.5 / 2244;
    const double RespawnsPerTurn = 1.2;
    const int MaxBases = 6, BaseUntil = 320, RecruitUntil = 280, MaxWorkers = 7, FieldCap = 45;

    WorldMemory? mem;
    float[] enterCost = [], enemyCost = [];
    readonly PathField home = new(), enemyField = new();
    readonly List<PathField> fields = [];
    int[] mineBlockedUntil = [], mineFailures = [], moveBlockedUntil = [], buildBlockedUntil = [];
    float[] reserved = [], prior = [];
    bool[] region = [], claimUs = [], claimEnemy = [];
    int enemyStamp = -1, enemyTurn = -100;
    int w, h;
    long workerTurns;
    int initialWorkers = -1;
    int planSite = -1, planWait;
    string? planBuilder;
    readonly Dictionary<string, (int Pos, string Kind, int Cell, long Load)> last = [];
    readonly Dictionary<string, int> waypoint = [];
    List<int>? virtualBases;
    double planGain, cachedGain;
    float[] weight = [];
    int cachedSite = -1, siteTurn = -100, siteBases;

    /// <summary>Starts sit evenly on a ring around the map center: guess the other teams' bases by rotating ours.</summary>
    List<int> RingGuess(int start, int teams)
    {
        var m = mem!;
        if (teams < 2) teams = 16;
        double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0, dx = m.X(start) - cx, dy = m.Y(start) - cy;
        double r = Math.Sqrt(dx * dx + dy * dy), a0 = Math.Atan2(dy, dx);
        var list = new List<int>();
        for (int i = 1; i < teams; i++)
        {
            double a = a0 + 2 * Math.PI * i / teams;
            int x = Math.Clamp((int)Math.Round(cx + r * Math.Cos(a)), 0, w - 1), y = Math.Clamp((int)Math.Round(cy + r * Math.Sin(a)), 0, h - 1);
            list.Add(m.Index(x, y));
        }
        return list;
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
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        var r = Decide(state, identity);
        double ms = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        if (Trace && ms > 4) Console.Error.WriteLine($"  slow t{state.View?.Turn} {ms:F2} ms");
        return r;
    }

    IReadOnlyDictionary<string, MarsAction> Decide(GameState<MarsView> state, Identity identity)
    {
        var actions = new Dictionary<string, MarsAction>();
        var view = state.View;
        if (view == null) return actions;
        string me = identity.TeamId;
        mem ??= new WorldMemory(me);
        mem.Update(view);
        var m = mem;
        w = m.Width; h = m.Height;
        int n = m.Count;
        if (enterCost.Length != n)
        {
            enterCost = new float[n]; enemyCost = new float[n];
            mineBlockedUntil = new int[n]; mineFailures = new int[n]; moveBlockedUntil = new int[n]; buildBlockedUntil = new int[n];
            reserved = new float[n]; prior = new float[n]; region = new bool[n]; claimUs = new bool[n]; claimEnemy = new bool[n];
            for (int i = 0; i < n; i++) prior[i] = (float)(m.InCenter(i) ? PriorCenter : PriorOutside);
        }
        int turn = view.Turn;
        int maxTurns = state.Match.MaxTurns > 0 ? state.Match.MaxTurns : 480;
        int left = maxTurns - turn;
        var mine = view.Units.Where(u => u.TeamId == me).OrderBy(u => u.UnitId, StringComparer.Ordinal).ToList();
        if (mine.Count == 0) return actions;
        if (initialWorkers < 0) initialWorkers = mine.Count;
        var bank = view.TeamResources?.GetValueOrDefault(me) ?? new Inventory(0, 0);

        // ---- Costs: enemy bases are walls for us, but sources for the enemy race field ----
        int enemyHash = 17;
        for (int i = 0; i < n; i++)
        {
            float c = m.Kind[i] switch
            {
                Tile.Unknown => 1.6f,
                Tile.Mountain or Tile.Crater => 3f,
                Tile.Void => float.PositiveInfinity,
                _ => 1f,
            };
            enemyCost[i] = c;
            if (moveBlockedUntil[i] > turn) c = float.PositiveInfinity;
            var owner = m.BaseOwnerAt[i];
            if (owner != null) { c = owner == me ? 1f : float.PositiveInfinity; enemyCost[i] = 1f; }
            enterCost[i] = c;
        }
        Array.Clear(claimUs); Array.Clear(claimEnemy);
        var myBases = new List<int>();
        var enemyBases = new List<int>();
        foreach (var b in m.Bases.Values)
        {
            int bi = m.Index(b.Position);
            bool ours = b.TeamId == me;
            if (ours) myBases.Add(bi); else { enemyBases.Add(bi); enemyHash = enemyHash * 31 + bi; }
            var claim = ours ? claimUs : claimEnemy;
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2 + Math.Abs(dy); dx <= 2 - Math.Abs(dy); dx++)
                    if (m.InBounds(b.Position.X + dx, b.Position.Y + dy)) claim[m.Index(b.Position.X + dx, b.Position.Y + dy)] = true;
        }
        home.To(myBases, enterCost, w, h);
        if (virtualBases == null && myBases.Count > 0) virtualBases = RingGuess(myBases[0], view.Scores?.Count ?? 0);
        if (enemyHash != enemyStamp || turn - enemyTurn >= 10)
        {
            enemyStamp = enemyHash; enemyTurn = turn;
            // Enemy bases we have not seen yet are guessed on the ring through our start; a guess goes once
            // a real base is found near it or its tile has been seen empty.
            var sources = new List<int>(enemyBases);
            if (virtualBases != null)
                foreach (int v in virtualBases)
                {
                    bool near = false;
                    foreach (int e in enemyBases) if (Math.Max(Math.Abs(m.X(e) - m.X(v)), Math.Abs(m.Y(e) - m.Y(v))) <= 6) { near = true; break; }
                    if (!near && m.LastSeen[v] < 0) sources.Add(v);
                }
            enemyField.To(sources, enemyCost, w, h);
        }
        var dUs = home.Dist;
        var dEn = enemyField.Dist;
        for (int i = 0; i < n; i++)
            region[i] = !float.IsInfinity(dUs[i]) && (claimUs[i] || (!claimEnemy[i] && dEn[i] - dUs[i] >= 1));

        while (fields.Count < mine.Count) fields.Add(new PathField());
        for (int k = 0; k < mine.Count; k++) fields[k].From(m.Index(mine[k].Position), enterCost, w, h, FieldCap);

        // ---- Learn from last turn: failed mines, blocked moves, failed builds ----
        foreach (var u in mine)
        {
            if (!last.TryGetValue(u.UnitId, out var l)) continue;
            int pos = m.Index(u.Position);
            long load = Load(u);
            if (l.Kind == "mine" && l.Pos == pos && load == l.Load && load < u.InventoryCapacity && m.Kind[pos] is Tile.Gold or Tile.Iron)
            {
                if (++mineFailures[pos] >= 2) { mineBlockedUntil[pos] = turn + 25; mineFailures[pos] = 0; }
            }
            else if (l.Kind == "mine" && load > l.Load) mineFailures[pos] = 0;
            if (l.Kind == "move" && l.Pos == pos && turn >= u.ReadyOnTurn && l.Cell >= 0) moveBlockedUntil[l.Cell] = turn + 40;
            if (l.Kind == "build" && m.BaseOwnerAt[pos] != me)
            {
                buildBlockedUntil[pos] = turn + 60;
                if (planSite == pos) { planSite = -1; planBuilder = null; }
            }
        }
        workerTurns += mine.Count;

        var enemyOn = new HashSet<int>();
        var occupied = new Dictionary<int, int>();
        foreach (var u in view.Units)
        {
            int p = m.Index(u.Position);
            occupied[p] = occupied.GetValueOrDefault(p) + 1;
            if (u.TeamId != me) enemyOn.Add(p);
        }

        // ---- Region deposits ----
        int recruits = mine.Count - initialWorkers;
        int price = RecruitPrice(recruits);
        long carriedGold = mine.Sum(u => u.Inventory?.Gold ?? 0);
        double rate = Math.Clamp((bank.Gold + carriedGold + 0.5 * 40) / (workerTurns + 40.0), 0.15, 1.2);
        var deps = new List<int>();
        double regionGold = 0, regionPrior = 0;
        for (int i = 0; i < n; i++)
        {
            if (region[i] && m.Kind[i] is Tile.Plain or Tile.Unknown) regionPrior += prior[i];
            if (m.Kind[i] is not (Tile.Gold or Tile.Iron) || m.Amount[i] <= 0 || mineBlockedUntil[i] > turn) continue;
            if (claimEnemy[i] && !claimUs[i] || float.IsInfinity(dUs[i])) continue;
            // Iron is rarely raced for; gold outside the region only while the race is close.
            if (m.Kind[i] == Tile.Gold && Win(i) <= 0) continue;
            deps.Add(i);
            if (m.Kind[i] == Tile.Gold && region[i]) regionGold += Est(i, turn);
        }
        // Expected respawn gold that lands in our region over the next 60 turns counts as known.
        double expected = regionGold + regionPrior * RespawnsPerTurn * 60;

        // ---- Investment: bases first (facility location), then workers ----
        bool invest = turn < BaseUntil && left > 60;
        if (planSite >= 0 && (!invest || !SiteValid(planSite, turn) || mine.All(u => u.UnitId != planBuilder)))
        {
            planSite = -1; planBuilder = null;
        }
        // Facility location is re-solved every 10 turns, when the base set changes, or when the cached site goes bad.
        if (!invest || planSite >= 0 || myBases.Count >= MaxBases) { cachedSite = -1; cachedGain = 0; }
        else if (turn - siteTurn >= 10 || myBases.Count != siteBases || (cachedSite >= 0 && !SiteValid(cachedSite, turn)))
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            (cachedSite, cachedGain) = BestSite(mine, turn, left, rate);
            if (Trace) Console.Error.WriteLine($"  bestsite t{turn} {System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds:F2} ms");
            siteTurn = turn; siteBases = myBases.Count;
        }
        int site = cachedSite;
        double siteGain = cachedGain;
        int workerTarget = Math.Min(MaxWorkers, Math.Max(turn < 150 ? 4 : 1, 1 + (int)Math.Ceiling(expected / 10)));
        bool recruitWanted = turn < RecruitUntil && left > 80 && mine.Count < workerTarget;
        double supply = Math.Clamp(expected / (10.0 * (mine.Count + 1)), 0.3, 1.0);
        double recruitPerIron = recruitWanted ? rate * supply * (left - 15) / price : 0;
        double basePerIron = site >= 0 && siteGain >= 1.5 * 6 ? siteGain / 6 : 0;
        bool baseWanted = basePerIron > 0 && basePerIron >= recruitPerIron;
        long avail = bank.Iron - (planSite >= 0 ? 6 : 0);
        long nextSpend = 0;
        string? spawner = null;
        if (baseWanted)
        {
            if (avail >= 6)
            {
                planSite = site; planWait = 0;
                planBuilder = NearestReady(mine, site, turn);
                if (planBuilder == null) planSite = -1; else avail -= 6;
            }
            else nextSpend = 6;
        }
        else if (recruitWanted)
        {
            if (avail >= price) { spawner = PickSpawner(mine, turn); if (spawner != null) avail -= price; }
            else nextSpend = price;
        }
        planGain = baseWanted ? siteGain : planSite >= 0 ? planGain : 0;
        bool wantIron = bank.Iron < Math.Max(nextSpend, planSite >= 0 ? 6 : 0) || (recruitWanted && bank.Iron < price);
        float ironValue = wantIron ? 0.8f : 0f;
        if (Trace && turn % 20 == 0)
            Console.Error.WriteLine($"[territory t{turn}] bank {bank.Gold}g/{bank.Iron}i workers {mine.Count}/{workerTarget} bases {myBases.Count} enemyBases {enemyBases.Count} "
                + $"regionGold {regionGold:F0} prior {regionPrior:F2} site {site} gain {siteGain:F1} plan {planSite} next {nextSpend} rate {rate:F2}");

        // ---- Greedy assignment over (worker, deposit) by value / (travel + take + return + 1) ----
        Array.Clear(reserved);
        var target = new int[mine.Count];
        var kind = new int[mine.Count]; // 0 none, 1 mine, 2 return
        Array.Fill(target, -1);
        var cands = new List<(int K, int Cell, float Carried, float Travel, float Ret, int Take, float Value)>();
        for (int k = 0; k < mine.Count; k++)
        {
            var u = mine[k];
            if (u.UnitId == spawner || u.UnitId == planBuilder) continue;
            long gold = u.Inventory?.Gold ?? 0, iron = u.Inventory?.Iron ?? 0;
            int free = (int)(u.InventoryCapacity - gold - iron);
            float carried = gold + iron * Math.Max(ironValue, 0.3f);
            float wait = Math.Max(0, u.ReadyOnTurn - turn);
            int pos = m.Index(u.Position);
            if (free > 0)
                foreach (int d in deps)
                {
                    float v = m.Kind[d] == Tile.Gold ? Win(d) : ironValue;
                    if (v <= 0) continue;
                    if (enemyOn.Contains(d) && d != pos && Est(d, turn) < 8) continue; // never chase contested gold
                    float travel = fields[k].Dist[d], ret = dUs[d];
                    if (float.IsInfinity(travel) || float.IsInfinity(ret)) continue;
                    travel += wait;
                    int take = (int)Math.Min(free, left - travel - ret - 2);
                    if (take <= 0) continue;
                    cands.Add((k, d, carried, travel, ret, take, v));
                }
            if (gold + iron > 0 && !float.IsInfinity(dUs[pos]))
                cands.Add((k, -1, carried, dUs[pos] + wait, 0, 0, 0));
        }
        for (int round = 0; round < mine.Count; round++)
        {
            int best = -1, bestTake = 0;
            float bestScore = 0;
            for (int i = 0; i < cands.Count; i++)
            {
                var c = cands[i];
                if (kind[c.K] != 0) continue;
                int take = 0;
                if (c.Cell >= 0)
                {
                    take = (int)Math.Min(c.Take, Est(c.Cell, turn) - reserved[c.Cell]);
                    if (take < 1) continue;
                }
                float s = (c.Carried + take * c.Value) / (c.Travel + take + c.Ret + 1);
                if (s > bestScore) { bestScore = s; best = i; bestTake = take; }
            }
            if (best < 0) break;
            var p = cands[best];
            if (p.Cell >= 0) { reserved[p.Cell] += bestTake; target[p.K] = p.Cell; kind[p.K] = 1; }
            else kind[p.K] = 2;
        }

        // ---- Exploration waypoints for idle workers: lane grid over unknown/stale region cells ----
        var taken = new List<int>();
        foreach (var (id, wp) in waypoint) if (mine.Exists(u => u.UnitId == id)) taken.Add(wp);

        // ---- Actions ----
        long ironBought = 0;
        for (int k = 0; k < mine.Count; k++)
        {
            var u = mine[k];
            int pos = m.Index(u.Position);
            long gold = u.Inventory?.Gold ?? 0, iron = u.Inventory?.Iron ?? 0;
            float hd = dUs[pos] + Math.Max(0, u.ReadyOnTurn - turn);
            MarsAction action = MarsAction.Pass();
            string kd = "pass";
            int cell = -1;
            if (turn < u.ReadyOnTurn) kd = "cooldown";
            else if (u.UnitId == spawner) { action = MarsAction.Spawn(); kd = "spawn"; }
            else if (u.UnitId == planBuilder && planSite >= 0)
            {
                if (pos == planSite)
                {
                    if (occupied.GetValueOrDefault(pos) == 1 && bank.Iron >= 6) { action = MarsAction.BuildBase(); kd = "build"; }
                    else if (++planWait > 3) { buildBlockedUntil[planSite] = turn + 30; planSite = -1; planBuilder = null; }
                }
                else (action, kd, cell) = Go(k, u, pos, planSite);
            }
            else if (m.BaseOwnerAt[pos] == me && gold + iron > 0)
            {
                // Gold buys iron only to finish a planned base that is at most 2 kg short.
                // A worker waiting on iron is bought too when it repays the gold twice over.
                bool forBase = planSite >= 0 || nextSpend == 6;
                long shortBy = (forBase ? 6 : nextSpend) - bank.Iron - iron - ironBought;
                double payoff = forBase ? planGain : recruitPerIron * price;
                if (shortBy > 0 && gold >= 3 && left > 80 && (forBase && shortBy <= 2 || payoff >= 2 * 3 * shortBy + 13.5 * shortBy * (forBase ? 1 : 0))) { action = MarsAction.Trade(Resource.Gold); kd = "trade"; ironBought++; }
                else { action = MarsAction.Deposit(); kd = "deposit"; }
            }
            else if (gold + iron > 0 && !float.IsInfinity(hd) && hd + 3 >= left)
            {
                int back = home.StepToward(pos);
                if (back >= 0) { action = Move(pos, back); kd = "move"; cell = back; }
            }
            else if (kind[k] == 1)
            {
                if (target[k] == pos) { action = MarsAction.Mine(); kd = "mine"; }
                else (action, kd, cell) = Go(k, u, pos, target[k]);
                waypoint.Remove(u.UnitId);
            }
            else if (kind[k] == 2)
            {
                int back = home.StepToward(pos);
                if (back >= 0) { action = Move(pos, back); kd = "move"; cell = back; }
                waypoint.Remove(u.UnitId);
            }
            else if (left > 25)
            {
                if (!waypoint.TryGetValue(u.UnitId, out int wp) || Math.Max(Math.Abs(m.X(wp) - m.X(pos)), Math.Abs(m.Y(wp) - m.Y(pos))) <= 1
                    || !Fresh(wp, turn))
                {
                    taken.Remove(wp);
                    wp = PickWaypoint(k, pos, turn, taken);
                    if (wp >= 0) { waypoint[u.UnitId] = wp; taken.Add(wp); } else waypoint.Remove(u.UnitId);
                }
                if (wp >= 0) (action, kd, cell) = Go(k, u, pos, wp);
            }
            actions[u.UnitId] = action;
            last[u.UnitId] = (pos, kd, cell, gold + iron);
        }
        return actions;
    }

    /// <summary>Chance we win the race to a deposit, from the base-distance margin m = dEnemy - dUs.</summary>
    float Win(int i)
    {
        if (region[i]) return 1f;
        float mg = enemyField.Dist[i] - home.Dist[i];
        return mg >= -7 ? Math.Max(0.3f, 0.8f + 0.1f * mg) : 0f;
    }

    float Est(int i, int turn)
    {
        var m = mem!;
        float a = m.Amount[i];
        if (!m.Visible[i] && turn - m.LastSeen[i] > 40) a *= 0.5f;
        return a;
    }

    /// <summary>A waypoint is still worth visiting while its vision diamond has unknown or stale region cells.</summary>
    bool Fresh(int wp, int turn) => CoverValue(wp, turn) > 0;

    double CoverValue(int wp, int turn)
    {
        var m = mem!;
        int x = m.X(wp), y = m.Y(wp);
        double v = 0;
        for (int dy = -3; dy <= 3; dy++)
            for (int dx = -3 + Math.Abs(dy); dx <= 3 - Math.Abs(dy); dx++)
            {
                if (!m.InBounds(x + dx, y + dy)) continue;
                int c = m.Index(x + dx, y + dy);
                if (m.Kind[c] is Tile.Mountain or Tile.Crater or Tile.Void) continue;
                bool unknown = m.Kind[c] == Tile.Unknown;
                if (!unknown && turn - m.LastSeen[c] <= 40) continue;
                if (!unknown && !region[c]) continue;
                v += prior[c] * (unknown ? 1.5 : 1);
            }
        return v;
    }

    int PickWaypoint(int k, int pos, int turn, List<int> taken)
    {
        var m = mem!;
        var f = fields[k].Dist;
        int best = -1;
        double bestScore = 0;
        for (int y = 3; y < h; y += 7)
            for (int x = 3; x < w; x += 7)
            {
                int g = m.Index(Math.Min(x, w - 1), Math.Min(y, h - 1));
                if (float.IsInfinity(f[g]) || float.IsInfinity(enterCost[g])) continue;
                // Region lanes, or unknown ground no known enemy base is closer to.
                if (!region[g] && !(m.Kind[g] == Tile.Unknown && enemyField.Dist[g] - home.Dist[g] >= 1)) continue;
                bool dup = false;
                foreach (int t in taken) if (Math.Abs(m.X(t) - m.X(g)) + Math.Abs(m.Y(t) - m.Y(g)) < 7) { dup = true; break; }
                if (dup) continue;
                double v = CoverValue(g, turn);
                if (v <= 0) continue;
                double s = v / (f[g] + 4);
                if (s > bestScore) { bestScore = s; best = g; }
            }
        return best;
    }

    (MarsAction, string, int) Go(int k, Worker u, int pos, int goal)
    {
        int step = fields[k].FirstStep(goal);
        if (step >= 0) return (Move(pos, step), "move", step);
        // Beyond the field cap: greedy Chebyshev step.
        var m = mem!;
        int px = m.X(pos), py = m.Y(pos), gx = m.X(goal), gy = m.Y(goal);
        double best = double.PositiveInfinity;
        int bestN = -1;
        foreach (var (dx, dy, _) in Directions.All)
        {
            int x = px + dx, y = py + dy;
            if (!m.InBounds(x, y)) continue;
            int ni = m.Index(x, y);
            if (float.IsInfinity(enterCost[ni])) continue;
            double s = Math.Max(Math.Abs(gx - x), Math.Abs(gy - y)) + (enterCost[ni] - 1) * 0.8 + 0.01 * (Math.Abs(gx - x) + Math.Abs(gy - y));
            if (s < best) { best = s; bestN = ni; }
        }
        return bestN < 0 ? (MarsAction.Pass(), "pass", -1) : (Move(pos, bestN), "move", bestN);
    }

    bool SiteValid(int s, int turn) =>
        mem!.Kind[s] == Tile.Plain && mem.BaseOwnerAt[s] == null && mem.FarFromEnemyBases(s) && buildBlockedUntil[s] <= turn;

    /// <summary>
    /// Greedy facility location: the site that most reduces F = sum w(c) * 2 * dist(c, nearest base) over region
    /// cells, plus the cells the new base wins from the enemy race frontier. Returns the gain in kg.
    /// </summary>
    (int Site, double Gain) BestSite(List<Worker> mine, int turn, int left, double rate)
    {
        var m = mem!;
        var dUs = home.Dist;
        var dEn = enemyField.Dist;
        double horizon = Math.Min(left - 20, 300);
        // Per-cell weight: known gold plus the expected respawn gold over the horizon.
        if (weight.Length != m.Count) weight = new float[m.Count];
        double total = 0;
        for (int c = 0; c < m.Count; c++)
        {
            var kc = m.Kind[c];
            double wt = kc is Tile.Plain ? prior[c] * RespawnsPerTurn * horizon
                : kc == Tile.Unknown ? prior[c] * RespawnsPerTurn * horizon * 0.7
                : kc == Tile.Gold ? Est(c, turn) + prior[c] * RespawnsPerTurn * horizon : 0;
            weight[c] = (float)wt;
            if (region[c]) total += wt;
        }
        // Gold beyond what our workers can carry in the horizon saves nothing: scale the weights down to capacity.
        double scale = Math.Min(1, (mine.Count + 1) * rate * horizon / Math.Max(1, total));
        float trip = (float)(2 * rate / 8);
        int best = -1;
        double bestGain = 0;
        for (int s = 0; s < m.Count; s++)
        {
            if (dUs[s] < 4 || float.IsInfinity(dUs[s]) || dEn[s] - dUs[s] < -1) continue;
            if (m.Kind[s] != Tile.Plain || m.BaseOwnerAt[s] != null || claimEnemy[s] || buildBlockedUntil[s] > turn) continue;
            float builder = float.PositiveInfinity;
            for (int k = 0; k < mine.Count; k++) builder = Math.Min(builder, fields[k].Dist[s]);
            if (builder > 40) continue;
            int sx = m.X(s), sy = m.Y(s);
            double gain = 0;
            for (int dy = -10; dy <= 10; dy++)
            {
                int y = sy + dy;
                if (y < 0 || y >= h) continue;
                int ady = Math.Abs(dy);
                for (int dx = -10; dx <= 10; dx++)
                {
                    int x = sx + dx;
                    if (x < 0 || x >= w) continue;
                    int c = y * w + x;
                    float wt = weight[c];
                    if (wt <= 0) continue;
                    int adx = Math.Abs(dx);
                    float ds = Math.Max(adx, ady) + 0.15f * Math.Min(adx, ady);
                    bool claims = adx + ady <= 2;
                    if (claimEnemy[c] && !claims) continue;
                    if (region[c])
                    {
                        float cur = Math.Min(dUs[c], 40);
                        if (cur > ds) gain += wt * (cur - ds) * trip;
                        if (claims && !claimUs[c]) gain += wt * 0.15f;
                    }
                    else if (dEn[c] - ds >= 1 || claims)
                        gain += wt * Math.Max(0, 0.6f - 0.02f * ds); // newly won from the race frontier
                }
            }
            gain = gain * scale - builder * rate;
            if (gain > bestGain) { bestGain = gain; best = s; }
        }
        return (best, bestGain);
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

    MarsAction Move(int from, int to) => MarsAction.Move(Directions.Of(mem!.X(to) - mem.X(from), mem.Y(to) - mem.Y(from)));

    static long Load(Worker u) => (u.Inventory?.Gold ?? 0) + (u.Inventory?.Iron ?? 0);
}
