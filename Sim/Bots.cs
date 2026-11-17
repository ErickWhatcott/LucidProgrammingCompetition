using Lpc.Client;
using Lpc.Strategies;
using Lpc.Strategies.Planner;

namespace Lpc.Sim;

/// <summary>Dijkstra (8-way, pay to enter) that stops at the first target tile it settles.</summary>
sealed class NearestSearch
{
    float[] dist = [];
    int[] prev = [], seen = [];
    int stamp;
    readonly PriorityQueue<int, float> queue = new();

    /// <summary>Move toward the nearest tile with (flags[i] &amp; want) != 0, or null if none is reachable / we are on it.</summary>
    public MarsAction? StepToNearest(int source, float[] cost, byte[] flags, byte want, int w, int h)
    {
        int n = w * h;
        if (dist.Length != n) { dist = new float[n]; prev = new int[n]; seen = new int[n]; stamp = 0; }
        stamp++;
        queue.Clear();
        dist[source] = 0; prev[source] = -1; seen[source] = stamp;
        queue.Enqueue(source, 0);
        int found = -1;
        while (queue.TryDequeue(out int c, out float d))
        {
            if (d != dist[c]) continue;
            if ((flags[c] & want) != 0) { found = c; break; }
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
                    if (float.IsInfinity(nd)) continue;
                    if (seen[ni] == stamp && nd >= dist[ni]) continue;
                    seen[ni] = stamp; dist[ni] = nd; prev[ni] = c;
                    queue.Enqueue(ni, nd);
                }
            }
        }
        if (found < 0 || found == source) return null;
        int step = found;
        while (prev[step] != source) step = prev[step];
        return MarsAction.Move(Directions.Of(step % w - source % w, step / w - source / w));
    }
}

static class BotUtil
{
    public const byte Target = 1, OwnBase = 2, Unknown = 4;

    public const float UnknownCost = 1.6f, ObstacleCost = 3f;

    /// <summary>Momentum random walk: keep heading most of the time; never off-map or into enemy bases.</summary>
    public static MarsAction Wander(Random rng, Dictionary<string, Direction> heading, string unitId, int x, int y,
        int width, int height, float[] cost)
    {
        bool has = heading.TryGetValue(unitId, out var dir);
        for (int attempt = 0; attempt < 16; attempt++)
        {
            if (!has || attempt > 0 || rng.NextDouble() < 0.15) dir = Directions.All[rng.Next(8)].Direction;
            var (dx, dy) = Directions.Offset(dir);
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height || float.IsInfinity(cost[ny * width + nx])) continue;
            heading[unitId] = dir;
            return MarsAction.Move(dir);
        }
        return MarsAction.Pass();
    }

    public static readonly IReadOnlyDictionary<string, MarsAction> Empty = new Dictionary<string, MarsAction>();
}

/// <summary>Built-in "Novice": collects nearby visible resources and returns full loads. No memory.</summary>
public sealed class NoviceBot(int seed = 0) : IMarsStrategy
{
    readonly Random rng = new(seed);
    readonly NearestSearch search = new();
    readonly Dictionary<string, Direction> heading = [];
    readonly List<int> myBases = [];
    readonly List<Base> enemyBases = [];
    float[] cost = [];
    byte[] flags = [];

    public string Name => "novice";

    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity)
    {
        var view = state.View;
        if (view == null) return BotUtil.Empty;
        string me = identity.TeamId;
        int w = view.Map.Width, h = view.Map.Height, n = w * h;
        if (cost.Length != n) { cost = new float[n]; flags = new byte[n]; }
        Array.Clear(flags);
        var cells = view.Map.Cells;
        for (int i = 0; i < n; i++)
            cost[i] = cells[i].Type switch
            {
                "hidden" => BotUtil.UnknownCost,
                "mountain" or "crater" => BotUtil.ObstacleCost,
                "void" => float.PositiveInfinity,
                _ => 1f,
            };
        myBases.Clear(); enemyBases.Clear();
        bool anyTarget = false;
        foreach (var b in view.Bases)
        {
            int bi = b.Position.Y * w + b.Position.X;
            if (b.TeamId == me) { myBases.Add(bi); flags[bi] |= BotUtil.OwnBase; }
            else { cost[bi] = float.PositiveInfinity; enemyBases.Add(b); }
        }
        for (int i = 0; i < n; i++)
        {
            var c = cells[i];
            if (c.Type is not ("gold" or "iron") || (c.Amount ?? 0) <= 0) continue;
            if (Mineable(i % w, i / w, w)) { flags[i] |= BotUtil.Target; anyTarget = true; }
        }

        var actions = new Dictionary<string, MarsAction>();
        foreach (var u in view.Units)
        {
            if (u.TeamId != me || view.Turn < u.ReadyOnTurn) continue;
            int pos = u.Position.Y * w + u.Position.X;
            long carried = (u.Inventory?.Gold ?? 0) + (u.Inventory?.Iron ?? 0);
            if ((flags[pos] & BotUtil.OwnBase) != 0 && carried > 0) { actions[u.UnitId] = MarsAction.Deposit(); continue; }
            bool full = carried >= u.InventoryCapacity;
            if (!full && (flags[pos] & BotUtil.Target) != 0) { actions[u.UnitId] = MarsAction.Mine(); continue; }

            MarsAction? act = null;
            if (!full && anyTarget) act = search.StepToNearest(pos, cost, flags, BotUtil.Target, w, h);
            if (act == null && carried > 0) act = search.StepToNearest(pos, cost, flags, BotUtil.OwnBase, w, h);
            act ??= BotUtil.Wander(rng, heading, u.UnitId, u.Position.X, u.Position.Y, w, h, cost);
            actions[u.UnitId] = act;
        }
        return actions;
    }

    bool Mineable(int x, int y, int w)
    {
        bool own = false, enemy = false;
        foreach (int b in myBases)
            if (Math.Abs(b % w - x) + Math.Abs(b / w - y) <= 2) { own = true; break; }
        if (own) return true;
        foreach (var b in enemyBases)
            if (Math.Abs(b.Position.X - x) + Math.Abs(b.Position.Y - y) <= 2) { enemy = true; break; }
        return !enemy;
    }
}

/// <summary>Built-in "Beginner": remembers explored terrain and searches for gold. Never invests.</summary>
public sealed class BeginnerBot(int seed = 0) : IMarsStrategy
{
    readonly Random rng = new(seed);
    readonly NearestSearch search = new();
    readonly Dictionary<string, Direction> heading = [];
    WorldMemory? memory;
    float[] cost = [];
    byte[] flags = [];

    public string Name => "beginner";

    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity)
    {
        var view = state.View;
        if (view == null) return BotUtil.Empty;
        string me = identity.TeamId;
        memory ??= new WorldMemory(me);
        memory.Update(view);
        var m = memory;
        int w = m.Width, h = m.Height, n = m.Count;
        if (cost.Length != n) { cost = new float[n]; flags = new byte[n]; }
        bool anyGold = false;
        for (int i = 0; i < n; i++)
        {
            var k = m.Kind[i];
            cost[i] = k switch
            {
                Tile.Unknown => BotUtil.UnknownCost,
                Tile.Mountain or Tile.Crater => BotUtil.ObstacleCost,
                Tile.Void => float.PositiveInfinity,
                _ => 1f,
            };
            byte f = 0;
            string? owner = m.BaseOwnerAt[i];
            if (owner != null)
            {
                if (owner == me) f |= BotUtil.OwnBase; else cost[i] = float.PositiveInfinity;
            }
            if (k == Tile.Unknown) f |= BotUtil.Unknown;
            else if (k == Tile.Gold && m.Amount[i] > 0 && m.Claim(i).Mineable) { f |= BotUtil.Target; anyGold = true; }
            flags[i] = f;
        }

        var actions = new Dictionary<string, MarsAction>();
        foreach (var u in view.Units)
        {
            if (u.TeamId != me || view.Turn < u.ReadyOnTurn) continue;
            int pos = m.Index(u.Position);
            long g = u.Inventory?.Gold ?? 0, carried = g + (u.Inventory?.Iron ?? 0);
            if ((flags[pos] & BotUtil.OwnBase) != 0 && carried > 0) { actions[u.UnitId] = MarsAction.Deposit(); continue; }
            bool full = carried >= u.InventoryCapacity;
            if (!full && (flags[pos] & BotUtil.Target) != 0) { actions[u.UnitId] = MarsAction.Mine(); continue; }

            MarsAction? act;
            if (full) act = search.StepToNearest(pos, cost, flags, BotUtil.OwnBase, w, h);
            else
            {
                act = anyGold ? search.StepToNearest(pos, cost, flags, BotUtil.Target, w, h) : null;
                if (act == null && g > 0) act = search.StepToNearest(pos, cost, flags, BotUtil.OwnBase, w, h);
                act ??= search.StepToNearest(pos, cost, flags, BotUtil.Unknown, w, h);
            }
            act ??= BotUtil.Wander(rng, heading, u.UnitId, u.Position.X, u.Position.Y, w, h, cost);
            actions[u.UnitId] = act;
        }
        return actions;
    }
}
