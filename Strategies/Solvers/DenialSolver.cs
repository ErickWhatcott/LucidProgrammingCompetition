using Lpc.Client;
using Lpc.Sim;
using Lpc.Strategies.Planner;

namespace Lpc.Strategies.Solvers;

/// <summary>
/// Opponent-modelling miner. Tracks every enemy worker by id (position, heading, load inferred from turns spent
/// mining), projects where each is going, and races it: a small per-deposit mining simulation with streak
/// priority gives our expected take. Bases are weapons: they co-claim rival gold and deny rival build sites.
/// </summary>
public sealed class DenialSolver(int seed = 0) : IMarsStrategy
{
    public string Name => "denial";

    const int RecruitUntil = 150, MaxNewBases = 6, BaseUntil = 330, TradeFreeWorkers = 3;
    const double SiteMin = 14, TradeSiteMin = 25, StealBonus = 0.3;

    sealed class Track
    {
        public string Team = "";
        public int X, Y, Vx, Vy, Seen, Load, Cap = 8, Ready;
        public bool Mining;
    }

    sealed class Own
    {
        public string Last = "";
        public int LastPos = -1, LastCell = -1, Target = -1;
        public long LastLoad;
    }

    readonly record struct Racer(int Start, int Room, bool Heading);

    readonly Random rng = new(seed);
    readonly PathField home = new();
    readonly List<PathField> fields = [];
    readonly Dictionary<string, Track> tracks = [];
    readonly Dictionary<string, Own> own = [];
    readonly Dictionary<int, List<Racer>> raceAt = [];
    readonly Dictionary<int, int> reservedAt = [];
    readonly List<int> myBases = [], deps = [], frontier = [];
    readonly List<float> frontierValue = [];
    WorldMemory? mem;
    float[] cost = [], goldNear = [];
    int[] stamp = [], prevAmount = [], mineBlocked = [], mineFails = [], moveBlocked = [], buildBlocked = [], occ = [];
    bool[] claimUs = [], claimEnemy = [], noEnemyBuild = [];
    int w, h, turn, left, initialWorkers = -1, stampId;
    int buildSite = -1, buildWait, cachedSite = -1;
    double cachedScore;
    string? builder;
    string me = "";

    readonly Dictionary<string, int> kinds = [];
    static readonly bool Trace = Environment.GetEnvironmentVariable("DENIAL_TRACE") == "1";

    // Centre gold is worth extra early (rivals settle there); bases also count the return trips they shorten.
    const double KRush = 1, KSave = 0.5, KMaxW = 6, KExplore = 0.45;

    static int RecruitPrice(int recruits)
    {
        int a = 5, b = 6;
        for (int i = 0; i < recruits; i++) (a, b) = (b, a + b);
        return a;
    }

    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity)
    {
        var actions = new Dictionary<string, MarsAction>();
        var view = state.View;
        if (view == null) return actions;
        me = identity.TeamId;
        mem ??= new WorldMemory(me);
        mem.Update(view);
        var m = mem;
        w = m.Width; h = m.Height;
        int n = m.Count;
        if (cost.Length != n)
        {
            cost = new float[n]; stamp = new int[n]; goldNear = new float[n]; prevAmount = new int[n]; mineBlocked = new int[n];
            mineFails = new int[n]; moveBlocked = new int[n]; buildBlocked = new int[n]; occ = new int[n];
            claimUs = new bool[n]; claimEnemy = new bool[n]; noEnemyBuild = new bool[n];
        }
        turn = view.Turn;
        int maxTurns = state.Match.MaxTurns > 0 ? state.Match.MaxTurns : 480;
        left = maxTurns - turn;
        var bank = view.TeamResources?.GetValueOrDefault(me) ?? new Inventory(0, 0);
        var mine = view.Units.Where(u => u.TeamId == me).OrderBy(u => u.UnitId, StringComparer.Ordinal).ToList();
        if (mine.Count == 0) { SaveAmounts(); return actions; }
        if (initialWorkers < 0) initialWorkers = mine.Count;

        // ---- Terrain, claims, home field ----
        Array.Clear(claimUs); Array.Clear(claimEnemy); Array.Clear(noEnemyBuild); Array.Clear(occ);
        for (int i = 0; i < n; i++)
        {
            float c = m.Kind[i] switch
            {
                Tile.Unknown => BotUtil.UnknownCost,
                Tile.Mountain or Tile.Crater => BotUtil.ObstacleCost,
                Tile.Void => float.PositiveInfinity,
                _ => 1f,
            };
            if (moveBlocked[i] > turn) c = float.PositiveInfinity;
            cost[i] = c;
        }
        myBases.Clear();
        foreach (var b in m.Bases.Values)
        {
            int bi = m.Index(b.Position);
            bool ours = b.TeamId == me;
            if (ours) { myBases.Add(bi); cost[bi] = 1f; } else cost[bi] = float.PositiveInfinity;
            var claim = ours ? claimUs : claimEnemy;
            ForDiamond(bi, 2, t => { claim[t] = true; noEnemyBuild[t] = true; });
        }
        if (myBases.Count > 0) home.To(myBases, cost, w, h);
        else { home.To([], cost, w, h); }
        while (fields.Count < mine.Count) fields.Add(new PathField());
        for (int k = 0; k < mine.Count; k++) fields[k].From(m.Index(mine[k].Position), cost, w, h, 45);
        foreach (var u in view.Units) occ[m.Index(u.Position)]++;

        // ---- Learn from our last actions ----
        foreach (var u in mine)
        {
            if (!own.TryGetValue(u.UnitId, out var o)) continue;
            int pos = m.Index(u.Position);
            long load = Load(u);
            if (o.Last == "mine" && pos == o.LastPos && load == o.LastLoad && load < u.InventoryCapacity
                && m.Kind[pos] is Tile.Gold or Tile.Iron && m.Amount[pos] > 0)
            {
                if (++mineFails[pos] >= 2) { mineBlocked[pos] = turn + 20; mineFails[pos] = 0; }
            }
            else if (o.Last == "mine" && load > o.LastLoad) mineFails[pos] = 0;
            if (o.Last == "move" && pos == o.LastPos && turn >= u.ReadyOnTurn && o.LastCell >= 0) moveBlocked[o.LastCell] = turn + 30;
            if (o.Last == "build" && m.BaseOwnerAt[pos] != me) { buildBlocked[pos] = turn + 60; if (buildSite == pos) buildSite = -1; }
        }

        // ---- Opponents: track every enemy worker, project its target ----
        UpdateTracks(view);
        deps.Clear();
        for (int i = 0; i < n; i++)
            if (m.Kind[i] is Tile.Gold or Tile.Iron && m.Amount[i] > 0 && mineBlocked[i] <= turn) deps.Add(i);
        ProjectEnemies();

        // ---- Investment ----
        int recruits = mine.Count - initialWorkers;
        int price = RecruitPrice(recruits);
        bool recruitWanted = mine.Count < KMaxW && turn < RecruitUntil && left > 120;
        int newBases = myBases.Count - 1;
        int site = -1;
        double siteScore = 0;
        if (buildSite >= 0 && (!SiteValid(buildSite) || builder == null || mine.All(u => u.UnitId != builder) || left < 40))
        { buildSite = -1; builder = null; }
        if (buildSite < 0 && newBases < MaxNewBases && turn < BaseUntil)
        {
            // Site scoring is the costliest step: refresh every third turn, reuse a still-valid site between.
            if (turn % 3 == 0 || (cachedSite >= 0 && (!SiteValid(cachedSite) || occ[cachedSite] > 0))) (cachedSite, cachedScore) = BestSite(mine);
            (site, siteScore) = (cachedSite, cachedScore);
        }
        // Workers first while cheap; afterwards a strong site outranks the next (expensive) worker.
        bool recruitFirst = recruitWanted && (mine.Count < TradeFreeWorkers + 1 || site < 0 || siteScore < 2 * TradeSiteMin);
        long iron = bank.Iron;
        string? spawner = null;
        if (buildSite >= 0) iron -= 6;
        if (recruitWanted && (recruitFirst || iron >= price + 6) && iron >= price)
        {
            spawner = PickSpawner(mine);
            if (spawner != null) iron -= price;
        }
        if (buildSite < 0 && site >= 0 && iron >= 6 && (!recruitFirst || spawner != null || iron >= price + 6))
        {
            builder = NearestWorker(mine, site, spawner);
            if (builder != null) { buildSite = site; buildWait = 0; iron -= 6; }
        }
        long ironNeed = recruitWanted ? price : 0;
        if (site >= 0 && buildSite < 0) ironNeed = Math.Max(ironNeed, recruitFirst ? price + 6 : 6);
        long ironShort = Math.Max(0, ironNeed - Math.Max(0, iron));
        long tradeGap = 0;
        if (ironShort > 0 && left > 100)
        {
            bool tradeRecruit = recruitWanted && mine.Count >= TradeFreeWorkers;
            bool tradeBase = site >= 0 && siteScore >= TradeSiteMin;
            if (tradeRecruit || tradeBase) tradeGap = ironShort;
        }
        long carriedIron = mine.Sum(u => u.Inventory?.Iron ?? 0);
        bool wantIron = ironShort - carriedIron > 0 && left > 100;

        if (Trace && turn % 30 == 0)
            Console.Error.WriteLine($"[denial t{turn}] {string.Join(' ', kinds.Select(kv => kv.Key + "=" + kv.Value))} bank {bank.Gold}g/{bank.Iron}i workers {mine.Count} bases {myBases.Count} site {siteScore:F0} short {ironShort} gap {tradeGap} wantIron {wantIron} deps {deps.Count} tracks {tracks.Count}");

        // ---- Frontier for exploration ----
        float knownGold = 0;
        foreach (int d in deps) if (m.Kind[d] == Tile.Gold && Mineable(d)) knownGold += m.Amount[d];
        BuildFrontier();
        float exploreNeed = (float)Math.Clamp(1 - knownGold / (12.0 * mine.Count), 0.15, 1);

        // ---- Assignment: locked miners keep their streak, then greedy best (worker, target) ----
        reservedAt.Clear();
        var plan = new int[mine.Count];      // -1 none, -2 home, -3 explore, else target cell
        var exploreAt = new int[mine.Count];
        Array.Fill(plan, -1);
        var free = new bool[mine.Count];
        for (int k = 0; k < mine.Count; k++)
        {
            var u = mine[k];
            int pos = m.Index(u.Position);
            var o = own.GetValueOrDefault(u.UnitId);
            long room = u.InventoryCapacity - Load(u);
            free[k] = u.UnitId != spawner && u.UnitId != builder && turn >= u.ReadyOnTurn;
            if (!free[k]) continue;
            bool isDep = m.Kind[pos] is Tile.Gold or Tile.Iron && m.Amount[pos] > 0 && Mineable(pos) && mineBlocked[pos] <= turn;
            bool intended = o != null && (o.Target == pos || o.Last == "mine");
            float hd = home.Dist[pos];
            bool late = Load(u) > 0 && !float.IsInfinity(hd) && hd + 2 >= left;
            if (isDep && intended && room > 0 && !late && (m.Kind[pos] == Tile.Gold || wantIron || o!.Last == "mine"))
            {
                plan[k] = pos;
                reservedAt[pos] = reservedAt.GetValueOrDefault(pos) + (int)room;
                free[k] = false;
            }
        }
        // Reservations only change the assigned cell, so only workers whose best was that cell are re-scored.
        var cand = new (int Cell, int Take, double Score)[mine.Count];
        for (int k = 0; k < mine.Count; k++) if (free[k]) cand[k] = BestFor(mine[k], k, wantIron, exploreNeed);
        for (int round = 0; round < mine.Count; round++)
        {
            int bestK = -1;
            for (int k = 0; k < mine.Count; k++)
                if (free[k] && plan[k] == -1 && (bestK < 0 || cand[k].Score > cand[bestK].Score)) bestK = k;
            if (bestK < 0 || double.IsNegativeInfinity(cand[bestK].Score)) break;
            int bestCell = cand[bestK].Cell;
            plan[bestK] = bestCell;
            if (bestCell == -3) exploreAt[bestK] = PickFrontier(bestK, exploreAt, plan);
            if (bestCell < 0) continue;
            reservedAt[bestCell] = reservedAt.GetValueOrDefault(bestCell) + cand[bestK].Take;
            for (int k = 0; k < mine.Count; k++)
                if (free[k] && plan[k] == -1 && cand[k].Cell == bestCell) cand[k] = BestFor(mine[k], k, wantIron, exploreNeed);
        }

        // ---- Actions ----
        long traded = 0;
        for (int k = 0; k < mine.Count; k++)
        {
            var u = mine[k];
            if (!own.TryGetValue(u.UnitId, out var o)) own[u.UnitId] = o = new Own();
            int pos = m.Index(u.Position);
            long gold = u.Inventory?.Gold ?? 0, ir = u.Inventory?.Iron ?? 0, load = gold + ir;
            MarsAction act = MarsAction.Pass();
            string kind = "pass";
            int cell = -1;
            float hd = home.Dist[pos] + Math.Max(0, u.ReadyOnTurn - turn);
            bool canBank = hd + 1 <= left;

            if (turn < u.ReadyOnTurn) kind = "cooldown";
            else if (u.UnitId == spawner) { act = MarsAction.Spawn(); kind = "spawn"; }
            else if (u.UnitId == builder && buildSite >= 0)
            {
                if (pos == buildSite)
                {
                    if (occ[pos] == 1 && bank.Iron >= 6) { act = MarsAction.BuildBase(); kind = "build"; }
                    else if (++buildWait > 4) { buildBlocked[buildSite] = turn + 30; buildSite = -1; builder = null; }
                }
                else (act, kind, cell) = StepTo(k, u, buildSite);
                if (kind == "pass" && pos != buildSite) { buildSite = -1; builder = null; }
            }
            else if (m.BaseOwnerAt[pos] == me && load > 0)
            {
                if (tradeGap - traded - ir > 0 && gold >= 3) { act = MarsAction.Trade(Resource.Gold); kind = "trade"; traded++; }
                else if (turn >= BaseUntil && ir >= 3 && left > 3) { act = MarsAction.Trade(Resource.Iron); kind = "trade"; }
                else { act = MarsAction.Deposit(); kind = "deposit"; }
            }
            else if (load > 0 && canBank && hd + 2 >= left)
                (act, kind, cell) = StepHome(u, pos);
            else if (!canBank)
            {
                // Too late to bank: carried gold is the next tiebreak.
                if (m.Kind[pos] == Tile.Gold && m.Amount[pos] > 0 && Mineable(pos) && load < u.InventoryCapacity) { act = MarsAction.Mine(); kind = "mine"; }
            }
            else
            {
                int p = plan[k];
                if (p >= 0 && p == pos) { act = MarsAction.Mine(); kind = "mine"; }
                else if (p >= 0) (act, kind, cell) = StepTo(k, u, p);
                else if (p == -2 || (load >= u.InventoryCapacity)) (act, kind, cell) = StepHome(u, pos);
                else if (p == -3 && exploreAt[k] >= 0) (act, kind, cell) = StepTo(k, u, exploreAt[k]);
                else if (load > 0) (act, kind, cell) = StepHome(u, pos);
                o.Target = p;
            }
            actions[u.UnitId] = act;
            if (Trace) { string key = kind == "move" ? (plan[k] >= 0 ? (mem!.Kind[plan[k]] == Tile.Gold ? "toGold" : "toIron") : plan[k] == -3 ? "explore" : "home") : kind; kinds[key] = kinds.GetValueOrDefault(key) + 1; }
            o.Last = kind; o.LastCell = cell; o.LastPos = pos; o.LastLoad = load;
        }
        SaveAmounts();
        return actions;
    }

    // ---------------- Opponent model ----------------

    void UpdateTracks(MarsView view)
    {
        var m = mem!;
        foreach (var e in view.Units)
        {
            if (e.TeamId == me) continue;
            int x = e.Position.X, y = e.Position.Y, p = m.Index(x, y);
            if (!tracks.TryGetValue(e.UnitId, out var t))
            {
                tracks[e.UnitId] = t = new Track { Team = e.TeamId, X = x, Y = y, Seen = turn, Load = 2 };
            }
            else
            {
                bool still = x == t.X && y == t.Y;
                bool onDep = m.Kind[p] is Tile.Gold or Tile.Iron || prevAmount[p] > 0;
                if (still && onDep && t.Seen == turn - 1 && turn >= t.Ready)
                {
                    // Standing on ore: it mined (inventory is hidden, so count the turns).
                    t.Load = Math.Min(t.Cap, t.Load + 1);
                    t.Mining = m.Amount[p] > 0;
                }
                else t.Mining = false;
                if (!still)
                {
                    t.Vx = Math.Sign(x - t.X); t.Vy = Math.Sign(y - t.Y);
                    if (t.Seen < turn - 3) t.Load = Math.Max(t.Load, 2);
                }
            }
            if (m.BaseOwnerAt[p] == e.TeamId) t.Load = 0;
            t.X = x; t.Y = y; t.Seen = turn; t.Cap = (int)e.InventoryCapacity; t.Ready = e.ReadyOnTurn;
        }
    }

    void ProjectEnemies()
    {
        var m = mem!;
        foreach (var l in raceAt.Values) l.Clear();
        foreach (var t in tracks.Values)
        {
            if (t.Seen < turn - 2) continue;
            int p = m.Index(t.X, t.Y), room = t.Cap - t.Load, wait = Math.Max(0, t.Ready - turn) + (turn - t.Seen);
            if (t.Mining && m.Amount[p] > 0) { AddRacer(p, new Racer(0, room, false)); continue; }
            if (room <= 1) continue; // full: heading home
            int best = -1, bestD = 11;
            foreach (int d in deps)
            {
                if (m.Kind[d] != Tile.Gold || !MineableBy(t.Team, d)) continue;
                int dx = d % w - t.X, dy = d / w - t.Y, cd = Math.Max(Math.Abs(dx), Math.Abs(dy));
                if (cd >= bestD) continue;
                if ((t.Vx | t.Vy) != 0 && cd > 0 && t.Vx * Math.Sign(dx) + t.Vy * Math.Sign(dy) <= 0) continue;
                best = d; bestD = cd;
            }
            if (best >= 0) AddRacer(best, new Racer(bestD + wait, room, true));
        }
    }

    void AddRacer(int cell, Racer r)
    {
        if (!raceAt.TryGetValue(cell, out var l)) raceAt[cell] = l = [];
        l.Add(r);
    }

    bool MineableBy(string team, int cell)
    {
        if (!claimUs[cell] && !claimEnemy[cell]) return true;
        int x = cell % w, y = cell / w;
        foreach (var b in mem!.Bases.Values)
            if (b.TeamId == team && Math.Abs(b.Position.X - x) + Math.Abs(b.Position.Y - y) <= 2) return true;
        return false;
    }

    /// <summary>Our take arriving at <paramref name="eta"/>: a turn-by-turn race where earlier arrivals mine first.</summary>
    int Take(int cell, int stock, int eta, int room, out bool steal)
    {
        steal = false;
        if (stock <= 0 || room <= 0) return 0;
        if (!raceAt.TryGetValue(cell, out var rs) || rs.Count == 0) return Math.Min(room, stock);
        Span<int> left = stackalloc int[Math.Min(rs.Count, 16)];
        for (int i = 0; i < left.Length; i++)
        {
            left[i] = rs[i].Room;
            if (rs[i].Heading && rs[i].Start > eta) steal = true;
        }
        int got = 0;
        for (int t = 0; stock > 0 && got < room && t < eta + room + 2; t++)
        {
            for (int i = 0; i < left.Length && stock > 0; i++)
                if (rs[i].Start <= eta && rs[i].Start <= t && left[i] > 0) { left[i]--; stock--; }
            if (t >= eta && stock > 0) { got++; stock--; }
            for (int i = 0; i < left.Length && stock > 0; i++)
                if (rs[i].Start > eta && rs[i].Start <= t && left[i] > 0) { left[i]--; stock--; }
        }
        return got;
    }

    // ---------------- Targeting ----------------

    (int Cell, int Take, double Score) BestFor(Worker u, int k, bool wantIron, float exploreNeed)
    {
        var m = mem!;
        var f = fields[k];
        int pos = m.Index(u.Position);
        long gold = u.Inventory?.Gold ?? 0, iron = u.Inventory?.Iron ?? 0;
        int room = (int)(u.InventoryCapacity - gold - iron);
        float wait = Math.Max(0, u.ReadyOnTurn - turn);
        float carried = gold + iron * (wantIron ? 1f : 0.3f);
        float hd = home.Dist[pos];
        int bestCell = -1, bestTake = 0;
        double best = double.NegativeInfinity;
        if (gold + iron > 0 && !float.IsInfinity(hd)) { best = carried / (hd + wait + 1); bestCell = -2; }
        if (room > 0)
        {
            foreach (int d in deps)
            {
                if (!Mineable(d)) continue;
                bool isGold = m.Kind[d] == Tile.Gold;
                float travel = f.Dist[d];
                if (float.IsInfinity(travel)) continue;
                float ret = home.Dist[d];
                if (float.IsInfinity(ret)) ret = 30;
                float val = isGold && m.InCenter(d) ? 1f + (float)(KRush * Math.Max(0, 1 - turn / 160.0)) : 1f;
                if (!isGold)
                {
                    // Iron only when it is (nearly) on the way.
                    if (!wantIron) continue;
                    float detour = float.IsInfinity(hd) ? travel : travel + ret - hd;
                    float reach = turn < RecruitUntil ? 9 : 4;
                    if (detour > reach && travel > reach) continue;
                    val = turn < 150 ? 1.4f : 0.9f;
                }
                travel += wait;
                int stock = m.Amount[d] - reservedAt.GetValueOrDefault(d);
                if (!m.Visible[d] && turn - m.LastSeen[d] > 25) stock = (int)(stock * 0.75f);
                int cap = (int)Math.Min(room, left - 2 - travel - ret);
                if (cap <= 0 || stock <= 0) continue;
                int take = Take(d, stock, (int)travel, cap, out bool steal);
                if (take <= 0) continue;
                double s = (carried + take * val) / (travel + take + ret + 1);
                if (steal) s *= 1 + StealBonus;
                if (own.TryGetValue(u.UnitId, out var o) && o.Target == d) s *= 1.08;
                if (s > best) { best = s; bestCell = d; bestTake = take; }
            }
        }
        if (frontier.Count > 0 && left > 40 && room > 0)
        {
            double e = 0;
            for (int i = 0; i < frontier.Count; i++)
            {
                float d = f.Dist[frontier[i]];
                if (float.IsInfinity(d)) continue;
                e = Math.Max(e, KExplore * exploreNeed * (0.2 + frontierValue[i]) * CenterPull(frontier[i]) * 10 / (10 + d));
            }
            if (e > best) { best = e; bestCell = -3; }
        }
        return (bestCell, bestTake, best);
    }

    double CenterPull(int cell) => mem!.InCenter(cell) ? 1 + 0.6 + 1.5 * Math.Max(0, 1 - turn / 150.0) : 1;

    void BuildFrontier()
    {
        var m = mem!;
        frontier.Clear(); frontierValue.Clear();
        if (left <= 40) return;
        for (int i = 0; i < m.Count; i++)
        {
            if (m.Kind[i] is Tile.Unknown or Tile.Void || float.IsInfinity(cost[i])) continue;
            int x = i % w, y = i / w;
            bool edge = (x > 0 && m.Kind[i - 1] == Tile.Unknown) || (x < w - 1 && m.Kind[i + 1] == Tile.Unknown)
                || (y > 0 && m.Kind[i - w] == Tile.Unknown) || (y < h - 1 && m.Kind[i + w] == Tile.Unknown);
            if (!edge) continue;
            int unknown = 0;
            for (int dy = -3; dy <= 3; dy++)
                for (int dx = -3 + Math.Abs(dy); dx <= 3 - Math.Abs(dy); dx++)
                    if (m.InBounds(x + dx, y + dy) && m.Kind[m.Index(x + dx, y + dy)] == Tile.Unknown) unknown++;
            frontier.Add(i); frontierValue.Add(unknown / 25f);
        }
    }

    int PickFrontier(int k, int[] exploreAt, int[] plan)
    {
        var f = fields[k];
        int best = -1;
        double bestS = 0;
        for (int i = 0; i < frontier.Count; i++)
        {
            int c = frontier[i];
            float d = f.Dist[c];
            if (float.IsInfinity(d)) continue;
            double s = (0.2 + frontierValue[i]) * CenterPull(c) * 10 / (10 + d);
            for (int j = 0; j < plan.Length; j++)
                if (j != k && plan[j] == -3 && exploreAt[j] >= 0 && Cheb(exploreAt[j], c) <= 6) s *= 0.4;
            if (s > bestS) { bestS = s; best = c; }
        }
        return best;
    }

    // ---------------- Bases as weapons ----------------

    bool SiteValid(int s) =>
        mem!.Kind[s] == Tile.Plain && mem.BaseOwnerAt[s] == null && mem.FarFromEnemyBases(s) && buildBlocked[s] <= turn;

    (int Site, double Score) BestSite(List<Worker> mine)
    {
        var m = mem!;
        int n = m.Count;
        // Gold an enemy base built at i would claim (that we do not already claim).
        Array.Clear(goldNear);
        foreach (int d in deps)
        {
            if (m.Kind[d] != Tile.Gold || claimUs[d]) continue;
            float a = m.Amount[d];
            int dx0 = d % w, dy0 = d / w;
            foreach (var (ox, oy) in Diamond2)
            {
                int x = dx0 + ox, y = dy0 + oy;
                if (x >= 0 && y >= 0 && x < w && y < h) goldNear[y * w + x] += a;
            }
        }
        // Rivals seen recently, by coarse 8x8 cell.
        int rw = (w + 7) / 8, rh = (h + 7) / 8;
        var rivalGrid = new int[rw * rh];
        foreach (var t in tracks.Values) if (t.Seen >= turn - 40) rivalGrid[t.Y / 8 * rw + t.X / 8]++;
        // Rival presence: enemy workers seen recently near a site make denial worth something.
        int best = -1;
        double bestScore = 0;
        stampId++;
        foreach (int g in deps)
        {
            if (m.Kind[g] != Tile.Gold) continue;
            int gx = g % w, gy = g / w;
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2 + Math.Abs(dy); dx <= 2 - Math.Abs(dy); dx++)
                {
                    int x = gx + dx, y = gy + dy;
                    if (!m.InBounds(x, y)) continue;
                    int s = y * w + x;
                    if (stamp[s] == stampId) continue;
                    stamp[s] = stampId;
                    if (!SiteValid(s) || occ[s] > 0) continue;
                    if (!float.IsInfinity(home.Dist[s]) && home.Dist[s] < 4) continue;
                    float bd = float.PositiveInfinity;
                    for (int k = 0; k < mine.Count; k++) bd = Math.Min(bd, fields[k].Dist[s]);
                    if (float.IsInfinity(bd)) continue;
                    double claim = 0, center = 0, denial = 0;
                    foreach (var (ox, oy) in Diamond2)
                    {
                        int tx = x + ox, ty = y + oy;
                        if (tx < 0 || ty < 0 || tx >= w || ty >= h) continue;
                        int t = ty * w + tx;
                        if (m.Kind[t] == Tile.Gold && !claimUs[t])
                        {
                            claim += m.Amount[t];
                            if (m.InCenter(t)) center += m.Amount[t];
                        }
                        else if (m.Kind[t] == Tile.Iron && !claimUs[t]) claim += m.Amount[t] * 0.3;
                        if (m.Kind[t] == Tile.Plain && !noEnemyBuild[t]) denial += goldNear[t];
                    }
                    double rivals = 0;
                    for (int ry = Math.Max(0, (y - 8) / 8); ry <= Math.Min(rh - 1, (y + 8) / 8); ry++)
                        for (int rx = Math.Max(0, (x - 8) / 8); rx <= Math.Min(rw - 1, (x + 8) / 8); rx++) rivals += rivalGrid[ry * rw + rx];
                    double save = 0;
                    foreach (int d in deps)
                        {
                            if (m.Kind[d] != Tile.Gold || Cheb(d, s) > 6) continue;
                            float cur = home.Dist[d];
                            if (float.IsInfinity(cur)) cur = 30;
                            save += m.Amount[d] / 8.0 * Math.Max(0, cur - Cheb(d, s) * 1.1);
                        }
                    double score = KSave * save + claim + 0.5 * denial / 13 * Math.Min(1, 0.3 + 0.35 * rivals) + 2 * center * Math.Max(0.3, 1 - turn / 300.0);
                    score -= 0.25 * bd;
                    if (score >= SiteMin && score > bestScore) { bestScore = score; best = s; }
                }
        }
        return (best, bestScore);
    }

    // ---------------- Helpers ----------------

    static readonly (int X, int Y)[] Diamond2 =
        [(0, -2), (-1, -1), (0, -1), (1, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2)];

    bool Mineable(int i) => claimUs[i] || !claimEnemy[i];

    void ForDiamond(int c, int r, Action<int> f)
    {
        int cx = c % w, cy = c / w;
        for (int dy = -r; dy <= r; dy++)
            for (int dx = -r + Math.Abs(dy); dx <= r - Math.Abs(dy); dx++)
            {
                int x = cx + dx, y = cy + dy;
                if (x >= 0 && y >= 0 && x < w && y < h) f(y * w + x);
            }
    }

    int Cheb(int a, int b) => Math.Max(Math.Abs(a % w - b % w), Math.Abs(a / w - b / w));

    void SaveAmounts()
    {
        var m = mem!;
        if (prevAmount.Length != m.Count) prevAmount = new int[m.Count];
        Array.Copy(m.Amount, prevAmount, m.Count);
    }

    string? PickSpawner(List<Worker> mine)
    {
        string? best = null;
        double bestS = double.NegativeInfinity;
        foreach (var u in mine)
        {
            if (turn < u.ReadyOnTurn || u.UnitId == builder) continue;
            int pos = mem!.Index(u.Position);
            double s = -Load(u) / 8.0;
            if (mem.Kind[pos] == Tile.Gold && mem.Amount[pos] >= 3) s += 3;
            if (mem.BaseOwnerAt[pos] == me) s += 2;
            if (s > bestS) { bestS = s; best = u.UnitId; }
        }
        return best;
    }

    string? NearestWorker(List<Worker> mine, int site, string? skip)
    {
        string? best = null;
        float bestD = float.PositiveInfinity;
        for (int k = 0; k < mine.Count; k++)
        {
            if (mine[k].UnitId == skip) continue;
            float d = fields[k].Dist[site] + Math.Max(0, mine[k].ReadyOnTurn - turn) + Load(mine[k]) * 0.5f;
            if (d < bestD) { bestD = d; best = mine[k].UnitId; }
        }
        return best;
    }

    (MarsAction, string, int) StepTo(int k, Worker u, int target)
    {
        int step = fields[k].FirstStep(target);
        if (step < 0) return (MarsAction.Pass(), "pass", -1);
        return (Move(u.Position, step), "move", step);
    }

    (MarsAction, string, int) StepHome(Worker u, int pos)
    {
        int step = home.StepToward(pos);
        if (step < 0 || step == pos) return (MarsAction.Pass(), "pass", -1);
        return (Move(u.Position, step), "move", step);
    }

    MarsAction Move(Coordinate from, int to) => MarsAction.Move(Directions.Of(to % w - from.X, to / w - from.Y));

    static long Load(Worker u) => (u.Inventory?.Gold ?? 0) + (u.Inventory?.Iron ?? 0);
}
