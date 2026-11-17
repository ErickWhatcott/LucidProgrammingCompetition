using Lpc.Client;
using Lpc.Sim;
using Lpc.Strategies.Planner;

namespace Lpc.Strategies.Solvers;

/// <summary>
/// Moves the whole economy into the rich center: recruit at turn 0, march both workers in, build a forward base,
/// then grow a grid of center bases at the quadrant anchors and never walk home again. Iron comes from trading
/// carried center gold 3:1 at a center base (the center has almost no iron).
/// </summary>
public sealed class CenterBankSolver(int seed = 0) : IMarsStrategy
{
    const int MaxWorkers = 6, MaxCenterBases = 5, MaxBases = 8, LandGrab = 3, IronRange = 8, ExploreRange = 12, RecruitUntil = 230, BaseUntil = 300, ExploreAge = 30;
    const long RecruitCap = 17;

    readonly Random rng = new(seed);
    readonly BoundedSearch search = new();
    readonly PathField depot = new();
    readonly List<PathField> fieldPool = [];
    readonly Dictionary<int, PathField> goalFields = [];
    readonly List<int> myBases = [], centerBases = [], builtBases = [];
    readonly List<Worker> mine = [];
    WorldMemory? memory;
    float[] cost = [];
    int[] occ = [], enemyOn = [];
    float[] reserved = [];
    bool[] claimUs = [], claimEnemy = [];
    int w, h, fwdSite = -1, homeBase = -1;
    string me = "";
    string? builder;
    int buildSite = -1;

    public string Name => "centerbank";

    static long RecruitCost(int recruited)
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
        memory ??= new WorldMemory(me);
        memory.Update(view);
        var m = memory;
        w = m.Width; h = m.Height;
        int n = m.Count;
        if (cost.Length != n)
        {
            cost = new float[n]; occ = new int[n]; enemyOn = new int[n]; reserved = new float[n];
            claimUs = new bool[n]; claimEnemy = new bool[n];
        }
        goalFields.Clear();
        int turn = view.Turn;
        int maxTurns = state.Match.MaxTurns > 0 ? state.Match.MaxTurns : 480;
        int left = maxTurns - turn;
        long bankIron = view.TeamResources != null && view.TeamResources.TryGetValue(me, out var inv) ? inv.Iron : 0;

        myBases.Clear(); centerBases.Clear();
        Array.Clear(claimUs); Array.Clear(claimEnemy); Array.Clear(occ); Array.Clear(enemyOn); Array.Clear(reserved);
        for (int i = 0; i < n; i++)
            cost[i] = m.Kind[i] switch
            {
                Tile.Unknown => BotUtil.UnknownCost,
                Tile.Mountain or Tile.Crater => BotUtil.ObstacleCost,
                Tile.Void => float.PositiveInfinity,
                _ => 1f,
            };
        foreach (var b in m.Bases.Values)
        {
            int bi = m.Index(b.Position);
            bool ours = b.TeamId == me;
            if (ours) { myBases.Add(bi); if (InGrown(bi)) centerBases.Add(bi); }
            else cost[bi] = float.PositiveInfinity;
            var claim = ours ? claimUs : claimEnemy;
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2 + Math.Abs(dy); dx <= 2 - Math.Abs(dy); dx++)
                {
                    int x = b.Position.X + dx, y = b.Position.Y + dy;
                    if (m.InBounds(x, y)) claim[y * w + x] = true;
                }
        }
        myBases.Sort(); centerBases.Sort();
        // Phase A until the first center base exists; after a failed opening fall back to every base.
        bool phaseA = centerBases.Count == 0;
        // Depots: every base we built (center grid + satellites); the home base only before that.
        if (homeBase < 0 && myBases.Count > 0) homeBase = myBases[0];
        builtBases.Clear();
        foreach (int b in myBases) if (b != homeBase) builtBases.Add(b);
        var depots = builtBases.Count > 0 ? builtBases : myBases;
        if (depots.Count > 0) depot.To(depots, cost, w, h);
        else { if (depot.Dist.Length != n) { depot.Dist = new float[n]; depot.Prev = new int[n]; } Array.Fill(depot.Dist, float.PositiveInfinity); Array.Fill(depot.Prev, -1); }

        mine.Clear();
        foreach (var u in view.Units)
        {
            int p = m.Index(u.Position);
            occ[p]++;
            if (u.TeamId == me) mine.Add(u); else enemyOn[p]++;
        }
        mine.Sort((a, b) => string.CompareOrdinal(a.UnitId, b.UnitId));
        int workers = mine.Count;
        long nextRecruit = RecruitCost(Math.Max(0, workers - 1));
        foreach (var u in mine)
        {
            int p = m.Index(u.Position);
            if (m.Kind[p] is Tile.Gold or Tile.Iron) reserved[p] += u.InventoryCapacity - Carried(u);
        }

        // Next purchase: forward base in phase A, then alternate recruits and center bases.
        // The opening needs 6 banked iron for the forward base; short of it, phase A trades at home first.
        bool opening = phaseA && turn < 150 && bankIron >= 6;
        bool canRecruit = workers < MaxWorkers && left > 120 && (!phaseA || bankIron >= nextRecruit + 6)
                          && (turn < RecruitUntil && nextRecruit <= RecruitCap || turn < 300 && bankIron >= nextRecruit);
        int site = -1;
        if (phaseA)
        {
            if (opening) site = fwdSite = ForwardSite(fwdSite);
        }
        else if (turn < BaseUntil && myBases.Count < MaxBases && left > 100) site = NextBaseSite(centerBases.Count < MaxCenterBases);
        // Land grab: enemy bases cover the center by ~turn 100, so the first three center bases come before recruits.
        bool baseFirst = site >= 0 && (phaseA || !canRecruit || centerBases.Count < LandGrab && turn < 150 || bankIron >= 6 && turn >= 150 || workers > centerBases.Count + 1);
        long need = baseFirst || phaseA ? 6 : canRecruit ? nextRecruit : 0;
        if (builder != null && !mine.Exists(u => u.UnitId == builder)) builder = null;
        if (site < 0 || !baseFirst) builder = null;
        else
        {
            if (buildSite != site) builder = null;
            buildSite = site;
            if (builder == null && (bankIron >= 6 || phaseA)) builder = PickBuilder(site, phaseA);
        }
        bool ironWanted = need > 0 && bankIron < need;
        long ironPipe = bankIron;
        bool spawned = false;
        int explorers = 0;

        var phaseField = opening ? Field(site >= 0 ? site : Mid()) : null;
        var actions = new Dictionary<string, MarsAction>(workers);
        foreach (var u in mine)
        {
            if (turn < u.ReadyOnTurn) continue;
            int pos = m.Index(u.Position);
            long gold = u.Inventory?.Gold ?? 0, iron = u.Inventory?.Iron ?? 0, carried = gold + iron;
            long cap = u.InventoryCapacity, room = cap - carried;
            float dd = depot.Dist[pos];

            if (m.BaseOwnerAt[pos] == me && carried > 0)
            {
                if (need > 0 && gold >= 3 && ironPipe + iron < need && left > 100)
                {
                    actions[u.UnitId] = MarsAction.Trade(Resource.Gold);
                    ironPipe++;
                }
                else { actions[u.UnitId] = MarsAction.Deposit(); ironPipe += iron; }
                continue;
            }

            // Endgame: carried gold only counts once banked.
            if (carried > 0 && !float.IsInfinity(dd) && dd + 3 >= left)
            {
                actions[u.UnitId] = StepDepot(pos) ?? MarsAction.Pass();
                continue;
            }

            if (u.UnitId == builder)
            {
                if (pos == site)
                {
                    if (occ[pos] == 1 && bankIron >= 6)
                    {
                        actions[u.UnitId] = MarsAction.BuildBase();
                        bankIron -= 6; ironPipe -= 6; builder = null;
                        continue;
                    }
                    if (occ[pos] > 1 && bankIron >= 6) { actions[u.UnitId] = MarsAction.Pass(); continue; }
                }
                else if (bankIron >= 6 || phaseA)
                {
                    actions[u.UnitId] = StepTo(pos, site);
                    continue;
                }
            }

            if (!spawned && canRecruit && bankIron >= nextRecruit + (baseFirst ? 6 : 0))
            {
                actions[u.UnitId] = MarsAction.Spawn();
                bankIron -= nextRecruit; ironPipe -= nextRecruit; spawned = true;
                continue;
            }

            if (phaseField != null)
            {
                // Opening: walk in, mine only deposits that cost at most one extra step.
                float sd = phaseField.Dist[pos];
                if (room > 0 && MineableHere(pos, true)) { actions[u.UnitId] = MarsAction.Mine(); continue; }
                int dep = room > 0 ? OnTheWay(pos, phaseField, sd) : -1;
                if (dep >= 0) { reserved[dep] += room; actions[u.UnitId] = StepTo(pos, dep); continue; }
                int goal = site >= 0 ? site : Mid();
                actions[u.UnitId] = pos == goal || sd <= 1 && site >= 0 ? Hold(u, pos, site) : StepTo(pos, goal);
                continue;
            }

            if (room <= 0) { actions[u.UnitId] = StepDepot(pos) ?? MarsAction.Pass(); continue; }

            if (MineableHere(pos, ironWanted) && TimeOk(pos, left)) { actions[u.UnitId] = MarsAction.Mine(); continue; }

            int target = FindTarget(pos, room, carried, dd, ironWanted, left, turn, explorers < 2, out bool explore);
            if (target >= 0)
            {
                reserved[target] += room;
                if (explore) explorers++;
                int step = search.FirstStep(pos, target);
                if (step >= 0) { actions[u.UnitId] = MoveTo(pos, step); continue; }
            }
            if (carried > 0) { var hm = StepDepot(pos); if (hm != null) { actions[u.UnitId] = hm; continue; } }
            // Idle: head for the stalest mineable tile around the center (reuse this worker's search tree when it reaches).
            int idle = OldestCenter(pos, turn);
            int first = search.FirstStep(pos, idle);
            actions[u.UnitId] = first >= 0 ? MoveTo(pos, first) : StepTo(pos, idle);
        }
        return actions;
    }

    static long Carried(Worker u) => (u.Inventory?.Gold ?? 0) + (u.Inventory?.Iron ?? 0);

    bool InGrown(int i)
    {
        var m = memory!;
        int x = i % w, y = i / w;
        return x >= m.CenterX0 - 1 && y >= m.CenterY0 - 1 && x <= m.CenterX0 + m.CenterW && y <= m.CenterY0 + m.CenterH;
    }

    int Mid() => (memory!.CenterY0 + memory.CenterH / 2) * w + memory.CenterX0 + memory.CenterW / 2;

    bool Mineable(int i) => claimUs[i] || !claimEnemy[i];

    bool MineableHere(int pos, bool wantIron)
    {
        var m = memory!;
        if (m.Amount[pos] <= 0 || !Mineable(pos)) return false;
        return m.Kind[pos] == Tile.Gold || (m.Kind[pos] == Tile.Iron && wantIron);
    }

    bool TimeOk(int pos, int left) => float.IsInfinity(depot.Dist[pos]) || depot.Dist[pos] + 3 < left;

    bool SiteOk(int s)
    {
        var m = memory!;
        return m.Kind[s] == Tile.Plain && m.BaseOwnerAt[s] == null && m.FarFromEnemyBases(s) && occ[s] <= 1;
    }

    double GoldManhattan2(int s)
    {
        var m = memory!;
        int sx = s % w, sy = s / w;
        double g = 0;
        for (int dy = -2; dy <= 2; dy++)
            for (int dx = -2 + Math.Abs(dy); dx <= 2 - Math.Abs(dy); dx++)
            {
                int x = sx + dx, y = sy + dy;
                if (!m.InBounds(x, y)) continue;
                int t = y * w + x;
                if (m.Kind[t] == Tile.Gold && !claimUs[t]) g += m.Amount[t];
            }
        return g;
    }

    /// <summary>Forward base: 2*gold within Manhattan 2 + gold within Chebyshev 5 + closeness to the midpoint.</summary>
    int ForwardSite(int prev)
    {
        var m = memory!;
        int mx = m.CenterX0 + m.CenterW / 2, my = m.CenterY0 + m.CenterH / 2;
        int best = -1;
        double bestScore = double.NegativeInfinity, prevScore = double.NegativeInfinity;
        for (int y = m.CenterY0 - 1; y <= m.CenterY0 + m.CenterH; y++)
            for (int x = m.CenterX0 - 1; x <= m.CenterX0 + m.CenterW; x++)
            {
                if (!m.InBounds(x, y)) continue;
                int s = y * w + x;
                if (!SiteOk(s)) continue;
                double g5 = 0;
                for (int ey = -5; ey <= 5; ey++)
                    for (int ex = -5; ex <= 5; ex++)
                    {
                        int tx = x + ex, ty = y + ey;
                        if (!m.InBounds(tx, ty)) continue;
                        int t = ty * w + tx;
                        if (m.Kind[t] == Tile.Gold && m.InCenter(t)) g5 += m.Amount[t];
                    }
                double score = 2 * GoldManhattan2(s) + g5 + 3.0 * (8 - Math.Max(Math.Abs(x - mx), Math.Abs(y - my)));
                if (score > bestScore) { bestScore = score; best = s; }
                if (s == prev) prevScore = score;
            }
        return prevScore > double.NegativeInfinity && prevScore + 6 >= bestScore ? prev : best;
    }

    /// <summary>Uncovered quadrant anchor (or a co-claim next to rich enemy-claimed gold), snapped to a valid tile.</summary>
    int NextBaseSite(bool grid)
    {
        var m = memory!;
        int best = -1;
        double bestScore = double.NegativeInfinity;
        // Co-claim: rich gold under an enemy claim gets a base > 2 from theirs and <= 2 from the deposit.
        for (int i = 0; i < w * h; i++)
        {
            if (m.Kind[i] != Tile.Gold || m.Amount[i] < 10 || !claimEnemy[i] || claimUs[i] || !InGrown(i)) continue;
            int ix = i % w, iy = i / w;
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2 + Math.Abs(dy); dx <= 2 - Math.Abs(dy); dx++)
                {
                    int x = ix + dx, y = iy + dy;
                    if (!m.InBounds(x, y)) continue;
                    int s = y * w + x;
                    if (!SiteOk(s) || NearOwnBase(s, 2)) continue;
                    double score = GoldManhattan2(s);
                    if (score > bestScore) { bestScore = score; best = s; }
                }
        }
        if (best >= 0 && bestScore >= 12) return best;
        best = -1; bestScore = double.NegativeInfinity;
        if (!grid) return Satellite();
        foreach (int ax in (int[])[m.CenterX0 + 2, m.CenterX0 + 8, m.CenterX0 + 14])
            foreach (int ay in (int[])[m.CenterY0 + 2, m.CenterY0 + 8, m.CenterY0 + 14])
            {
                if (NearOwnBase(ay * w + ax, 2)) continue;
                for (int dy = -2; dy <= 2; dy++)
                    for (int dx = -2; dx <= 2; dx++)
                    {
                        int x = ax + dx, y = ay + dy;
                        if (!m.InBounds(x, y)) continue;
                        int s = y * w + x;
                        if (!SiteOk(s) || NearOwnBase(s, 3)) continue;
                        double score = GoldManhattan2(s) - 1.5 * Math.Max(Math.Abs(dx), Math.Abs(dy)) - 0.3 * depot.Dist[s];
                        if (score > bestScore) { bestScore = score; best = s; }
                    }
            }
        return best >= 0 ? best : Satellite();
    }

    /// <summary>Grid full or blocked: a base on the ring around the center where known unclaimed gold sits.</summary>
    int Satellite()
    {
        var m = memory!;
        int best = -1;
        double bestScore = 12;
        for (int y = Math.Max(0, m.CenterY0 - 8); y < Math.Min(h, m.CenterY0 + m.CenterH + 8); y++)
            for (int x = Math.Max(0, m.CenterX0 - 8); x < Math.Min(w, m.CenterX0 + m.CenterW + 8); x++)
            {
                int s = y * w + x;
                if (!SiteOk(s) || NearOwnBase(s, 3) || depot.Dist[s] > 14) continue;
                double score = GoldManhattan2(s) - 0.2 * depot.Dist[s];
                if (score > bestScore) { bestScore = score; best = s; }
            }
        return best;
    }

    bool NearOwnBase(int s, int cheb)
    {
        int sx = s % w, sy = s / w;
        foreach (int b in myBases)
            if (Math.Max(Math.Abs(b % w - sx), Math.Abs(b / w - sy)) <= cheb) return true;
        return false;
    }

    string? PickBuilder(int site, bool phaseA)
    {
        string? best = null;
        float bestD = float.PositiveInfinity;
        var f = Field(site);
        foreach (var u in mine)
        {
            int p = memory!.Index(u.Position);
            float d = f.Dist[p] + (phaseA ? 0 : 0.5f * Carried(u) / 8f);
            if (d < bestD) { bestD = d; best = u.UnitId; }
        }
        return best;
    }

    /// <summary>Gold whose detour from the walk to the forward site is at most one step.</summary>
    int OnTheWay(int pos, PathField f, float sd)
    {
        var m = memory!;
        search.Run(pos, cost, w, h, 6);
        int best = -1;
        float bestD = float.PositiveInfinity;
        foreach (int c in search.Settled)
        {
            bool isIron = m.Kind[c] == Tile.Iron;
            if (m.Kind[c] != Tile.Gold && !isIron || !Mineable(c) || m.Amount[c] - reserved[c] <= 0.5f) continue;
            float d = search.Dist[c];
            if (d + f.Dist[c] > sd + (isIron ? 2 : 1) || d >= bestD) continue;
            bestD = d; best = c;
        }
        return best;
    }

    MarsAction Hold(Worker u, int pos, int site)
    {
        // Non-builder waiting next to the site: mine nearby rather than stand still.
        var m = memory!;
        if (Carried(u) >= u.InventoryCapacity) return pos == site ? StepTo(pos, Mid()) : MarsAction.Pass();
        if (MineableHere(pos, false)) return MarsAction.Mine();
        search.Run(pos, cost, w, h, 4);
        foreach (int c in search.Settled)
            if (m.Kind[c] == Tile.Gold && Mineable(c) && m.Amount[c] - reserved[c] > 0.5f)
            {
                int step = search.FirstStep(pos, c);
                if (step >= 0) { reserved[c] += 8; return MoveTo(pos, step); }
            }
        return pos == site ? StepTo(pos, Mid()) : MarsAction.Pass();
    }

    int FindTarget(int pos, long room, long carried, float dd, bool wantIron, int left, int turn, bool allowExplore, out bool explore)
    {
        var m = memory!;
        explore = false;
        search.Run(pos, cost, w, h, 24);
        bool centerNear = false;
        foreach (int c in search.Settled)
            if (search.Dist[c] <= 10 && m.Kind[c] == Tile.Gold && m.InCenter(c) && Mineable(c) && Avail(c) > 0.5f) { centerNear = true; break; }
        float detourBase = carried > 0 && !float.IsInfinity(dd) ? dd : 0;
        int best = -1;
        double bestScore = 0;
        foreach (int c in search.Settled)
        {
            var k = m.Kind[c];
            float d = search.Dist[c];
            float back = depot.Dist[c];
            if (float.IsInfinity(back)) back = 30;
            double value, take;
            bool isExplore = false;
            if (k == Tile.Gold)
            {
                if (!Mineable(c) || (!m.InCenter(c) && centerNear)) continue;
                float avail = Avail(c);
                if (avail <= 0.5f) continue;
                take = Math.Min(avail, room);
                value = take;
            }
            else if (k == Tile.Iron && wantIron)
            {
                if (!Mineable(c) || back > IronRange) continue;
                float avail = m.Amount[c] - reserved[c];
                if (avail <= 0.5f) continue;
                take = Math.Min(avail, room);
                value = take * 3;
            }
            else if (allowExplore && carried == 0 && (m.InCenter(c) || back <= ExploreRange) && !m.Visible[c] && Mineable(c) && reserved[c] <= 0
                     && (m.LastSeen[c] < 0 || turn - m.LastSeen[c] > ExploreAge))
            {
                take = 0; value = 0.5; isExplore = true;
            }
            else continue;
            if (d + take + back + 3 >= left) continue;
            double score = value / Math.Max(1, d + take + back - detourBase + 2);
            if (score > bestScore) { bestScore = score; best = c; explore = isExplore; }
        }
        return best;
    }

    /// <summary>Contest rule: crowded stock is discounted by what the enemies standing on it will take.</summary>
    float Avail(int c) => Math.Max(0, memory!.Amount[c] - 8 * enemyOn[c]) - reserved[c];

    int OldestCenter(int pos, int turn)
    {
        var m = memory!;
        int px = pos % w, py = pos / w, best = Mid();
        double bestScore = double.NegativeInfinity;
        for (int y = Math.Max(0, m.CenterY0 - 4); y < Math.Min(h, m.CenterY0 + m.CenterH + 4); y += 2)
            for (int x = Math.Max(0, m.CenterX0 - 4); x < Math.Min(w, m.CenterX0 + m.CenterW + 4); x += 2)
            {
                int i = y * w + x;
                if (!Mineable(i)) continue;
                int age = m.LastSeen[i] < 0 ? 200 : turn - m.LastSeen[i];
                double s = age - 2.0 * Math.Max(Math.Abs(x - px), Math.Abs(y - py)) + rng.NextDouble();
                if (s > bestScore) { bestScore = s; best = i; }
            }
        return best;
    }

    PathField Field(int goal)
    {
        if (goalFields.TryGetValue(goal, out var f)) return f;
        if (goalFields.Count >= fieldPool.Count) fieldPool.Add(new PathField());
        f = fieldPool[goalFields.Count];
        f.To([goal], cost, w, h);
        goalFields[goal] = f;
        return f;
    }

    MarsAction MoveTo(int from, int to) => MarsAction.Move(Directions.Of(to % w - from % w, to / w - from / w));

    MarsAction StepTo(int pos, int goal)
    {
        if (pos == goal) return MarsAction.Pass();
        int next = Field(goal).StepToward(pos);
        return next >= 0 && next != pos ? MoveTo(pos, next) : MarsAction.Pass();
    }

    MarsAction? StepDepot(int pos)
    {
        if (float.IsInfinity(depot.Dist[pos])) return null;
        int next = depot.StepToward(pos);
        return next >= 0 && next != pos ? MoveTo(pos, next) : null;
    }
}
