using Lpc.Client;
using Lpc.Strategies;
using Lpc.Strategies.Planner;

namespace Lpc.Sim;

// Hand-written opponent styles for pretraining. All share one competent engine (mine, return full
// loads, deposit, respect claims, get home before the end) and differ in parameters/target biases.

enum ExploreMode { Random, Center, Ring, Home }

sealed class StyleConfig
{
    public int MaxWorkers = 4;
    public int RecruitUntil = 300;
    public int MaxBases = 2;
    public int BaseUntil = 300;
    /// <summary>A new base is considered only where the walk home is at least this long.</summary>
    public int MinBaseHomeDist = 7;
    public double MinSiteGold = 10;
    /// <summary>Iron kept back when recruiting (e.g. so a base can still be afforded).</summary>
    public long SpawnReserve;
    public double IronWeight = 0.35;
    /// <summary>Iron deposits are targets while bank iron is below this (-1 = next recruit price).</summary>
    public long IronWanted = 6;
    public double CenterBonus = 1.0;
    /// <summary>Multiplier bonus per enemy worker near a deposit (camper). 0 = mild avoidance.</summary>
    public double ContestBonus;
    /// <summary>&gt;0: only targets within this many turns of an own base.</summary>
    public int HomeRadius;
    public bool TradeForIron;
    /// <summary>Trade carried gold for iron only while the next recruit costs at most this.</summary>
    public long TradeCostCap = 45;
    public bool BlockEnemies;
    public double Epsilon;
    public float SearchRadius = 26;
    public ExploreMode Explore = ExploreMode.Random;
}

/// <summary>Dijkstra limited by cost; records settled nodes so callers can scan them.</summary>
sealed class BoundedSearch
{
    public float[] Dist = [];
    public int[] Prev = [];
    int[] seen = [];
    int stamp;
    public readonly List<int> Settled = [];
    readonly PriorityQueue<int, float> queue = new();

    public bool Reached(int i) => seen[i] == stamp;

    public void Run(int source, float[] cost, int w, int h, float maxCost)
    {
        int n = w * h;
        if (Dist.Length != n) { Dist = new float[n]; Prev = new int[n]; seen = new int[n]; stamp = 0; }
        stamp++;
        queue.Clear(); Settled.Clear();
        Dist[source] = 0; Prev[source] = -1; seen[source] = stamp;
        queue.Enqueue(source, 0);
        while (queue.TryDequeue(out int c, out float d))
        {
            if (d != Dist[c]) continue;
            Settled.Add(c);
            int cx = c % w, cy = c / w;
            for (int dy = -1; dy <= 1; dy++)
            {
                int ny = cy + dy;
                if (ny < 0 || ny >= h) continue;
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nx = cx + dx;
                    if ((dx | dy) == 0 || nx < 0 || nx >= w) continue;
                    int ni = ny * w + nx;
                    float nd = d + cost[ni];
                    if (float.IsInfinity(nd) || nd > maxCost) continue;
                    if (seen[ni] == stamp && nd >= Dist[ni]) continue;
                    seen[ni] = stamp; Dist[ni] = nd; Prev[ni] = c;
                    queue.Enqueue(ni, nd);
                }
            }
        }
    }

    public int FirstStep(int source, int target)
    {
        if (target == source || !Reached(target)) return -1;
        int c = target;
        while (Prev[c] != source) { c = Prev[c]; if (c < 0) return -1; }
        return c;
    }
}

sealed class ScriptedEngine(StyleConfig cfg, int seed)
{
    readonly Random rng = new(seed);
    readonly BoundedSearch search = new();
    readonly PathField home = new();
    readonly Dictionary<string, int> waypoint = [];
    readonly Dictionary<string, int> waypointAge = [];
    readonly Dictionary<string, int> buildAt = [];
    readonly List<int> myBases = [];
    WorldMemory? memory;
    float[] cost = [];
    int[] occ = [], enemyOn = [];
    float[] reserved = [];
    bool[] claimUs = [], claimEnemy = [];
    int w, h;
    string me = "";

    public StyleConfig Config => cfg;

    static long RecruitCost(int recruited)
    {
        long a = 5, b = 6;
        for (int i = 0; i < recruited; i++) (a, b) = (b, a + b);
        return a;
    }

    public IReadOnlyDictionary<string, MarsAction> Decide(GameState<MarsView> state, Identity identity)
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
        int turn = view.Turn;
        int maxTurns = state.Match.MaxTurns > 0 ? state.Match.MaxTurns : 480;
        int left = maxTurns - turn;
        long bankIron = view.TeamResources != null && view.TeamResources.TryGetValue(me, out var inv) ? inv.Iron : 0;

        myBases.Clear();
        Array.Clear(claimUs); Array.Clear(claimEnemy); Array.Clear(occ); Array.Clear(enemyOn); Array.Clear(reserved);
        for (int i = 0; i < n; i++)
            cost[i] = m.Kind[i] switch
            {
                Tile.Unknown => BotUtil.UnknownCost,
                Tile.Mountain or Tile.Crater => BotUtil.ObstacleCost,
                Tile.Void => float.PositiveInfinity,
                _ => 1f,
            };
        int enemyBaseCount = 0;
        foreach (var b in m.Bases.Values)
        {
            int bi = m.Index(b.Position);
            bool mine = b.TeamId == me;
            if (mine) myBases.Add(bi); else { cost[bi] = float.PositiveInfinity; enemyBaseCount++; }
            var claim = mine ? claimUs : claimEnemy;
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2 + Math.Abs(dy); dx <= 2 - Math.Abs(dy); dx++)
                {
                    int x = b.Position.X + dx, y = b.Position.Y + dy;
                    if (m.InBounds(x, y)) claim[y * w + x] = true;
                }
        }
        if (myBases.Count > 0) home.To(myBases, cost, w, h);
        else { if (home.Dist.Length != n) { home.Dist = new float[n]; home.Prev = new int[n]; } Array.Fill(home.Dist, float.PositiveInfinity); Array.Fill(home.Prev, -1); }

        var mine0 = new List<Worker>();
        foreach (var u in view.Units)
        {
            int p = m.Index(u.Position);
            occ[p]++;
            if (u.TeamId == me) mine0.Add(u); else enemyOn[p]++;
        }
        mine0.Sort((a, b) => string.CompareOrdinal(a.UnitId, b.UnitId));
        int myCount = mine0.Count;
        long nextCost = RecruitCost(Math.Max(0, myCount - 1));
        long ironWanted = cfg.IronWanted < 0 ? nextCost : cfg.IronWanted;
        bool wantIron = bankIron < ironWanted;
        long ironAvail = bankIron;
        bool spawned = false;
        int basesNow = myBases.Count;

        // Workers already mining keep their deposit reserved.
        foreach (var u in mine0)
        {
            int p = m.Index(u.Position);
            if (m.Kind[p] is Tile.Gold or Tile.Iron) reserved[p] += u.InventoryCapacity - Carried(u);
        }
        foreach (var id in buildAt.Keys.ToList())
            if (!mine0.Exists(u => u.UnitId == id)) buildAt.Remove(id);

        var actions = new Dictionary<string, MarsAction>(myCount);
        foreach (var u in mine0)
        {
            if (turn < u.ReadyOnTurn) continue;
            int pos = m.Index(u.Position);
            long gold = u.Inventory?.Gold ?? 0, iron = u.Inventory?.Iron ?? 0, carried = gold + iron;
            long cap = u.InventoryCapacity;
            bool onBase = m.BaseOwnerAt[pos] == me;
            float hd = home.Dist[pos];

            if (cfg.Epsilon > 0 && rng.NextDouble() < cfg.Epsilon && left > 40)
            {
                actions[u.UnitId] = rng.Next(3) == 0 && MineableHere(pos, true) && carried < cap
                    ? MarsAction.Mine()
                    : RandomMove(pos);
                continue;
            }

            if (onBase && carried > 0)
            {
                if (cfg.TradeForIron && gold >= 3 && turn < cfg.RecruitUntil && myCount < cfg.MaxWorkers
                    && ironAvail + iron < nextCost && nextCost <= cfg.TradeCostCap && left > 100)
                    actions[u.UnitId] = MarsAction.Trade(Resource.Gold);
                else actions[u.UnitId] = MarsAction.Deposit();
                continue;
            }

            // End of game: carried gold only counts once banked.
            if (carried > 0 && !float.IsInfinity(hd) && hd + 3 >= left)
            {
                actions[u.UnitId] = StepHome(pos) ?? MarsAction.Pass();
                continue;
            }

            // Pending base construction.
            if (buildAt.TryGetValue(u.UnitId, out int site))
            {
                if (basesNow >= cfg.MaxBases || ironAvail < 6 || !SiteOk(site, pos) || left < 30) buildAt.Remove(u.UnitId);
                else if (site == pos)
                {
                    if (occ[pos] == 1)
                    {
                        actions[u.UnitId] = MarsAction.BuildBase();
                        ironAvail -= 6; basesNow++;
                        buildAt.Remove(u.UnitId);
                        continue;
                    }
                    buildAt.Remove(u.UnitId);
                }
                else
                {
                    actions[u.UnitId] = StepToward(pos, site);
                    continue;
                }
            }

            // Recruit.
            if (!spawned && myCount < cfg.MaxWorkers && turn < cfg.RecruitUntil && left > 80
                && ironAvail - cfg.SpawnReserve >= nextCost)
            {
                actions[u.UnitId] = MarsAction.Spawn();
                ironAvail -= nextCost; spawned = true;
                continue;
            }

            // Consider a new base.
            if (buildAt.Count == 0 && basesNow < cfg.MaxBases && turn < cfg.BaseUntil && ironAvail >= 6 && left > 60)
            {
                int s = FindSite(pos, hd);
                if (s >= 0)
                {
                    if (s == pos && occ[pos] == 1)
                    {
                        actions[u.UnitId] = MarsAction.BuildBase();
                        ironAvail -= 6; basesNow++;
                    }
                    else
                    {
                        buildAt[u.UnitId] = s;
                        actions[u.UnitId] = StepToward(pos, s);
                    }
                    continue;
                }
            }

            if (carried >= cap)
            {
                actions[u.UnitId] = StepHome(pos) ?? Explore(u, pos);
                continue;
            }

            if (MineableHere(pos, wantIron) && TimeOk(0, pos, left))
            {
                actions[u.UnitId] = MarsAction.Mine();
                continue;
            }

            int target = FindTarget(pos, cap - carried, wantIron, left);
            if (target >= 0)
            {
                reserved[target] += cap - carried;
                int step = search.FirstStep(pos, target);
                if (step >= 0) { actions[u.UnitId] = MoveTo(pos, step); continue; }
            }
            if (carried > 0)
            {
                var hm = StepHome(pos);
                if (hm != null) { actions[u.UnitId] = hm; continue; }
            }
            actions[u.UnitId] = Explore(u, pos);
        }
        return actions;
    }

    static long Carried(Worker u) => (u.Inventory?.Gold ?? 0) + (u.Inventory?.Iron ?? 0);

    bool Mineable(int i) => claimUs[i] || !claimEnemy[i];

    bool MineableHere(int pos, bool wantIron)
    {
        var m = memory!;
        var k = m.Kind[pos];
        if (m.Amount[pos] <= 0 || !Mineable(pos)) return false;
        return k == Tile.Gold || (k == Tile.Iron && wantIron);
    }

    bool TimeOk(float d, int target, int left) => d + 1 + home.Dist[target] + 2 < left || float.IsInfinity(home.Dist[target]);

    int FindTarget(int pos, long room, bool wantIron, int left)
    {
        var m = memory!;
        search.Run(pos, cost, w, h, cfg.SearchRadius);
        int best = -1;
        double bestScore = 0;
        foreach (int c in search.Settled)
        {
            var k = m.Kind[c];
            if (k != Tile.Gold && !(k == Tile.Iron && wantIron)) continue;
            if (!Mineable(c)) continue;
            float avail = m.Amount[c] - reserved[c];
            if (avail <= 0.5f) continue;
            float d = search.Dist[c];
            float back = home.Dist[c];
            if (float.IsInfinity(back)) back = 30;
            if (cfg.HomeRadius > 0 && back > cfg.HomeRadius) continue;
            if (d + 2 + back + 2 >= left) continue;
            double take = Math.Min(avail, room);
            double value = take * (k == Tile.Gold ? 1.0 : cfg.IronWeight);
            if (k == Tile.Gold && m.InCenter(c)) value *= cfg.CenterBonus;
            int near = EnemiesNear(c);
            if (cfg.ContestBonus > 0) value *= 1 + cfg.ContestBonus * Math.Min(near, 3);
            else if (enemyOn[c] > 0 && m.Amount[c] < 8) value *= 0.5;
            double score = value / (d + take + back * 0.6 + 2);
            if (score > bestScore) { bestScore = score; best = c; }
        }
        return best;
    }

    int EnemiesNear(int c)
    {
        int cx = c % w, cy = c / w, cnt = 0;
        for (int dy = -2; dy <= 2; dy++)
            for (int dx = -2; dx <= 2; dx++)
            {
                int x = cx + dx, y = cy + dy;
                if (x >= 0 && y >= 0 && x < w && y < h) cnt += enemyOn[y * w + x];
            }
        return cnt;
    }

    bool SiteOk(int s, int pos)
    {
        var m = memory!;
        return m.Kind[s] == Tile.Plain && m.BaseOwnerAt[s] == null && m.FarFromEnemyBases(s)
               && (occ[s] == 0 || (s == pos && occ[s] == 1));
    }

    int FindSite(int pos, float hd)
    {
        var m = memory!;
        int px = pos % w, py = pos / w;
        int best = -1;
        double bestScore = 0;
        for (int dy = -3; dy <= 3; dy++)
            for (int dx = -3; dx <= 3; dx++)
            {
                int x = px + dx, y = py + dy;
                if (!m.InBounds(x, y)) continue;
                int s = y * w + x;
                if (!m.Visible[s] || !SiteOk(s, pos)) continue;
                double gold = 0, enemyGold = 0;
                for (int ey = -2; ey <= 2; ey++)
                    for (int ex = -2 + Math.Abs(ey); ex <= 2 - Math.Abs(ey); ex++)
                    {
                        int tx = x + ex, ty = y + ey;
                        if (!m.InBounds(tx, ty)) continue;
                        int t = ty * w + tx;
                        if (claimUs[t]) continue;
                        if (m.Kind[t] == Tile.Gold)
                        {
                            double v = m.Amount[t] * (m.InCenter(t) ? cfg.CenterBonus : 1);
                            gold += v;
                            if (claimEnemy[t]) enemyGold += v;
                        }
                        else if (m.Kind[t] == Tile.Iron) gold += m.Amount[t] * 0.3;
                    }
                float sd = home.Dist[s];
                bool far = myBases.Count == 0 || float.IsInfinity(sd) || sd >= cfg.MinBaseHomeDist;
                double score;
                if (cfg.BlockEnemies && enemyGold > 0) score = gold + enemyGold * 1.5;
                else if (far) score = gold;
                else continue;
                if (score < cfg.MinSiteGold) continue;
                score -= 0.5 * Math.Max(Math.Abs(dx), Math.Abs(dy));
                if (score > bestScore) { bestScore = score; best = s; }
            }
        return best;
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

    MarsAction RandomMove(int pos)
    {
        int px = pos % w, py = pos / w;
        for (int a = 0; a < 12; a++)
        {
            var (dx, dy, dir) = Directions.All[rng.Next(8)];
            int x = px + dx, y = py + dy;
            if (x < 0 || y < 0 || x >= w || y >= h || float.IsInfinity(cost[y * w + x])) continue;
            return MarsAction.Move(dir);
        }
        return MarsAction.Pass();
    }

    MarsAction Explore(Worker u, int pos)
    {
        var m = memory!;
        int age = waypointAge.GetValueOrDefault(u.UnitId);
        if (!waypoint.TryGetValue(u.UnitId, out int wp) || wp == pos || age > 45
            || Math.Max(Math.Abs(wp % w - pos % w), Math.Abs(wp / w - pos / w)) <= 1)
        {
            wp = PickWaypoint(pos);
            waypoint[u.UnitId] = wp;
            age = 0;
        }
        waypointAge[u.UnitId] = age + 1;
        return StepToward(pos, wp);
    }

    int PickWaypoint(int pos)
    {
        var m = memory!;
        int x, y;
        double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
        switch (cfg.Explore)
        {
            case ExploreMode.Center:
                x = m.CenterX0 + rng.Next(Math.Max(1, m.CenterW));
                y = m.CenterY0 + rng.Next(Math.Max(1, m.CenterH));
                break;
            case ExploreMode.Ring:
            {
                // Bases sit on a ring around the centre; wander along it toward neighbouring teams.
                double r = 0.38 * Math.Min(w, h);
                double a0 = Math.Atan2(pos / w - cy, pos % w - cx);
                double a = a0 + (rng.NextDouble() * 2 - 1) * 1.2;
                double rr = r * (0.75 + rng.NextDouble() * 0.4);
                x = (int)Math.Round(cx + rr * Math.Cos(a)); y = (int)Math.Round(cy + rr * Math.Sin(a));
                break;
            }
            case ExploreMode.Home when myBases.Count > 0:
            {
                int b = myBases[rng.Next(myBases.Count)];
                x = b % w + rng.Next(-8, 9); y = b / w + rng.Next(-8, 9);
                break;
            }
            default:
            {
                // Mostly local exploration, sometimes toward the middle.
                if (rng.NextDouble() < 0.3) { x = (int)cx + rng.Next(-10, 11); y = (int)cy + rng.Next(-10, 11); }
                else { x = pos % w + rng.Next(-14, 15); y = pos / w + rng.Next(-14, 15); }
                break;
            }
        }
        x = Math.Clamp(x, 0, w - 1); y = Math.Clamp(y, 0, h - 1);
        return y * w + x;
    }
}

/// <summary>Rushes the rich centre, builds bases there early, mines centre gold.</summary>
public sealed class RusherBot(int seed = 0) : IMarsStrategy
{
    readonly ScriptedEngine engine = new(new StyleConfig
    {
        MaxWorkers = 6, RecruitUntil = 300, MaxBases = 4, BaseUntil = 300, MinBaseHomeDist = 6, MinSiteGold = 12,
        CenterBonus = 2.5, Explore = ExploreMode.Center, SearchRadius = 30, IronWanted = 6,
    }, seed);
    public string Name => "rusher";
    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity) => engine.Decide(state, identity);
}

/// <summary>Spends iron on many bases near gold to claim territory; keeps few workers.</summary>
public sealed class ExpanderBot(int seed = 0) : IMarsStrategy
{
    readonly ScriptedEngine engine = new(new StyleConfig
    {
        MaxWorkers = 2, RecruitUntil = 200, MaxBases = 10, BaseUntil = 360, MinBaseHomeDist = 5, MinSiteGold = 6,
        IronWanted = 12, IronWeight = 0.7, SpawnReserve = 6,
    }, seed);
    public string Name => "expander";
    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity) => engine.Decide(state, identity);
}

/// <summary>Recruits as many workers as possible early, trading carried gold for iron; floods nearby deposits.</summary>
public sealed class SwarmBot(int seed = 0) : IMarsStrategy
{
    readonly ScriptedEngine engine = new(new StyleConfig
    {
        MaxWorkers = 14, RecruitUntil = 210, MaxBases = 1, BaseUntil = 0, TradeForIron = true,
        IronWanted = -1, IronWeight = 0.9, SearchRadius = 22,
    }, seed);
    public string Name => "swarm";
    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity) => engine.Decide(state, identity);
}

/// <summary>Parks on rich deposits, preferring ones other teams are working, and keeps its mining streaks.</summary>
public sealed class CamperBot(int seed = 0) : IMarsStrategy
{
    readonly ScriptedEngine engine = new(new StyleConfig
    {
        MaxWorkers = 4, RecruitUntil = 250, MaxBases = 2, BaseUntil = 250, MinBaseHomeDist = 8, MinSiteGold = 12,
        ContestBonus = 0.6, CenterBonus = 1.4, Explore = ExploreMode.Ring, SearchRadius = 28,
    }, seed);
    public string Name => "camper";
    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity) => engine.Decide(state, identity);
}

/// <summary>Builds bases next to other teams' claimed gold to co-claim (share/deny) it; otherwise mines.</summary>
public sealed class BlockerBot(int seed = 0) : IMarsStrategy
{
    readonly ScriptedEngine engine = new(new StyleConfig
    {
        MaxWorkers = 4, RecruitUntil = 250, MaxBases = 7, BaseUntil = 340, MinBaseHomeDist = 9, MinSiteGold = 6,
        BlockEnemies = true, Explore = ExploreMode.Ring, IronWanted = 12, IronWeight = 0.6, SpawnReserve = 6,
    }, seed);
    public string Name => "blocker";
    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity) => engine.Decide(state, identity);
}

/// <summary>Mines only close to its own bases with short trips; an early second base; ~5 workers.</summary>
public sealed class HomebodyBot(int seed = 0) : IMarsStrategy
{
    readonly ScriptedEngine engine = new(new StyleConfig
    {
        MaxWorkers = 5, RecruitUntil = 260, MaxBases = 3, BaseUntil = 150, MinBaseHomeDist = 5, MinSiteGold = 6,
        HomeRadius = 9, Explore = ExploreMode.Home, SearchRadius = 18, IronWanted = 8,
    }, seed);
    public string Name => "homebody";
    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity) => engine.Decide(state, identity);
}

/// <summary>Values iron like gold, recruits and builds steadily until turn ~360, ends with a large iron bank.</summary>
public sealed class IronlordBot(int seed = 0) : IMarsStrategy
{
    readonly ScriptedEngine engine = new(new StyleConfig
    {
        MaxWorkers = 7, RecruitUntil = 360, MaxBases = 3, BaseUntil = 300, MinBaseHomeDist = 7, MinSiteGold = 8,
        IronWanted = 100000, IronWeight = 1.3, SpawnReserve = 10,
    }, seed);
    public string Name => "ironlord";
    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity) => engine.Decide(state, identity);
}

/// <summary>Seeded random mixture of parameters plus epsilon-random actions; still mines most of the time.</summary>
public sealed class RandomBot(int seed = 0) : IMarsStrategy
{
    readonly ScriptedEngine engine = new(Make(seed), seed);
    public string Name => "random";

    static StyleConfig Make(int seed)
    {
        var r = new Random(seed * 7919 + 17);
        return new StyleConfig
        {
            MaxWorkers = 1 + r.Next(8), RecruitUntil = 150 + r.Next(200), MaxBases = 1 + r.Next(4),
            BaseUntil = 100 + r.Next(250), MinBaseHomeDist = 4 + r.Next(8), MinSiteGold = 6 + r.Next(10),
            IronWeight = 0.2 + r.NextDouble() * 0.7, IronWanted = 4 + r.Next(14), CenterBonus = 0.8 + r.NextDouble() * 1.7,
            ContestBonus = r.NextDouble() < 0.3 ? r.NextDouble() : 0, HomeRadius = r.NextDouble() < 0.2 ? 8 + r.Next(10) : 0,
            Epsilon = 0.05 + r.NextDouble() * 0.15, SearchRadius = 16 + r.Next(14),
            Explore = (ExploreMode)r.Next(4),
        };
    }

    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity) => engine.Decide(state, identity);
}

public static class ScriptedBots
{
    public static readonly string[] Names = ["rusher", "expander", "swarm", "camper", "blocker", "homebody", "ironlord", "random", "hungarian", "territory", "centerbank", "investor", "denial", "scout"];

    public static IMarsStrategy? Create(string name, int seed) => name switch
    {
        "rusher" => new RusherBot(seed),
        "expander" => new ExpanderBot(seed),
        "swarm" => new SwarmBot(seed),
        "camper" => new CamperBot(seed),
        "blocker" => new BlockerBot(seed),
        "homebody" => new HomebodyBot(seed),
        "ironlord" => new IronlordBot(seed),
        "random" => new RandomBot(seed),
        "hungarian" => new Lpc.Strategies.Solvers.HungarianSolver(seed),
        "territory" => new Lpc.Strategies.Solvers.TerritorySolver(seed),
        "centerbank" => new Lpc.Strategies.Solvers.CenterBankSolver(seed),
        "investor" => new Lpc.Strategies.Solvers.InvestorSolver(seed),
        "denial" => new Lpc.Strategies.Solvers.DenialSolver(seed),
        "scout" => new Lpc.Strategies.Solvers.ScoutSolver(seed),
        _ => null,
    };
}
