using System.Text;
using Lpc.Client;

namespace Lpc.Sim;

/// <summary>
/// Local re-implementation of the LPC 2026 Mars game, used for self-play and tuning.
///
/// The real map generator is not published, so these parts are assumptions:
/// - Obstacles are grown as random 4-connected blobs (3–14 tiles) until ~MountainWeight / CraterWeight
///   of the tiles are mountains / craters.
/// - Bases sit evenly on a circle (radius 0.33 × min(W,H)) around the map center with a random rotation and
///   ±1 jitter; the 3×3 block around each base is cleared to plain so a base is never walled in. The starting
///   worker stands on a random neighbouring tile.
/// - Initial deposits: GoldPerTeam×teams gold, round(CenterPercent×gold) of them uniformly in the rich center
///   and the rest uniformly outside; IronPerTeam×teams iron uniformly outside the center. Deposits avoid
///   bases/workers and prefer Chebyshev spacing ≥ 2 from other deposits.
/// - Respawned iron prefers the outside zone (falls back to the center); gold picks its zone with CenterPercent.
/// - Turn numbering: the view at Turn t shows the state after t resolved turns (starting at 0). Step resolves the
///   actions sent with that view. Entering an obstacle while resolving view turn t sets ReadyOnTurn = t + ObstacleCost.
/// - The built-in tie-break randomness and team construction order come from a Random seeded with the match seed.
/// </summary>
public sealed class Simulator
{
    const byte Plain = 0, Mountain = 1, Crater = 2, Gold = 3, Iron = 4;
    static readonly string[] TypeNames = ["plain", "mountain", "crater", "gold", "iron"];
    static readonly Cell Hidden = new("hidden", null, null, null, null);
    static readonly Cell[] PlainCells = [new("plain", null, null, null, null), new("mountain", null, null, null, null), new("crater", null, null, null, null)];

    sealed class SimWorker
    {
        public required string Id;
        public int Team;
        public int X, Y;
        public int ReadyOnTurn;
        public long Gold, Iron;
        public int Streak;
        public int ArriveTurn;
        public int TieKey;
        public MarsAction? Action;
    }

    sealed class SimBase
    {
        public required string Id;
        public int Team;
        public int X, Y;
        public long DepGold, DepIron;
    }

    sealed class TeamState
    {
        public required string Id;
        public required string Name;
        public long BankGold, BankIron;
        public int Recruited;
        public int WorkerSeq, BaseSeq;
        public long CostA = 5, CostB = 6; // next two recruitment prices
        public readonly List<SimWorker> Workers = [];
        public readonly List<SimBase> Bases = [];
    }

    readonly record struct Respawn(bool IsGold, int OldTile, int DueTurn);

    readonly SimConfig cfg;
    readonly int W, H, N;
    readonly byte[] terrain;
    readonly int[] amount;
    readonly Cell?[] cellCache;
    readonly int[] baseAt;        // index into bases, -1 if none
    readonly int[] workerCount;
    readonly int[] visStamp;
    int stamp;
    readonly List<SimWorker> workers = [];
    readonly Dictionary<string, SimWorker> byId = new(StringComparer.Ordinal);
    readonly List<SimBase> bases = [];
    readonly TeamState[] teams;
    readonly int[] teamOrder;
    readonly List<Respawn> respawns = [];
    readonly Random rng;
    readonly int cx0, cy0, cw, ch;
    readonly List<string> teamIds;
    readonly List<Team> teamList;
    readonly HighGoldArea highGold;
    readonly List<int> tier0 = [], tier1 = [], tier2 = [];
    readonly List<SimWorker> scratch = [];

    public int Turn { get; private set; }
    public bool Done => Turn >= cfg.MaxTurns;
    public IReadOnlyList<string> TeamIds => teamIds;
    public SimConfig Config => cfg;
    public int Seed => cfg.Seed;

    // Bookkeeping for invariant checks.
    public long GoldMined { get; private set; }
    public long IronMined { get; private set; }
    public long TradeGoldNet { get; private set; } // gold created minus gold consumed by trades
    public long TradeIronNet { get; private set; }
    public long IronSpent { get; private set; }

    public Simulator(SimConfig config, IReadOnlyList<string> teamNames)
    {
        cfg = config;
        W = config.Width; H = config.Height; N = W * H;
        terrain = new byte[N];
        amount = new int[N];
        cellCache = new Cell?[N];
        baseAt = new int[N];
        Array.Fill(baseAt, -1);
        workerCount = new int[N];
        visStamp = new int[N];
        cw = Math.Min(config.CenterSize, W); ch = Math.Min(config.CenterSize, H);
        cx0 = (W - cw) / 2; cy0 = (H - ch) / 2;
        highGold = new HighGoldArea(cw, ch, config.CenterGoldMin, config.CenterGoldMax, (int)Math.Round(config.CenterPercent * 100));

        int n = teamNames.Count;
        teams = new TeamState[n];
        teamIds = new List<string>(n);
        teamList = new List<Team>(n);
        for (int i = 0; i < n; i++)
        {
            string id = $"team-{i:00}";
            teams[i] = new TeamState { Id = id, Name = teamNames[i], BankIron = config.StartIron };
            teamIds.Add(id);
            teamList.Add(new Team(id, teamNames[i]));
        }

        var mapRng = new Random(config.Seed);
        rng = new Random(unchecked(config.Seed * 7919 + 104729));
        teamOrder = Enumerable.Range(0, n).ToArray();
        rng.Shuffle(teamOrder);

        GenerateObstacles(mapRng);
        PlaceBases(mapRng);
        PlaceInitialDeposits(mapRng);
    }

    // ---------------------------------------------------------------- generation

    bool InCenter(int i)
    {
        int x = i % W, y = i / W;
        return x >= cx0 && y >= cy0 && x < cx0 + cw && y < cy0 + ch;
    }

    void GenerateObstacles(Random r)
    {
        var frontier = new List<int>();
        foreach (var (kind, weight) in new[] { (Mountain, cfg.MountainWeight), (Crater, cfg.CraterWeight) })
        {
            int target = (int)Math.Round(N * weight);
            int placed = 0, guard = 0;
            while (placed < target && guard++ < N * 4)
            {
                int start = r.Next(N);
                if (terrain[start] != Plain) continue;
                int size = Math.Min(r.Next(3, 15), target - placed);
                frontier.Clear();
                terrain[start] = kind; placed++; frontier.Add(start);
                int blob = 1;
                while (blob < size && frontier.Count > 0)
                {
                    int fi = r.Next(frontier.Count);
                    int c = frontier[fi];
                    int cx = c % W, cy = c / W;
                    int d = r.Next(4);
                    int nx = cx + (d == 0 ? 1 : d == 1 ? -1 : 0), ny = cy + (d == 2 ? 1 : d == 3 ? -1 : 0);
                    bool grew = false;
                    if (nx >= 0 && ny >= 0 && nx < W && ny < H && terrain[ny * W + nx] == Plain)
                    {
                        terrain[ny * W + nx] = kind; placed++; blob++; frontier.Add(ny * W + nx); grew = true;
                    }
                    if (!grew && !HasPlainNeighbour4(c)) frontier.RemoveAt(fi);
                }
            }
        }
    }

    bool HasPlainNeighbour4(int c)
    {
        int x = c % W, y = c / W;
        return (x > 0 && terrain[c - 1] == Plain) || (x < W - 1 && terrain[c + 1] == Plain)
            || (y > 0 && terrain[c - W] == Plain) || (y < H - 1 && terrain[c + W] == Plain);
    }

    void PlaceBases(Random r)
    {
        int n = teams.Length;
        double radius = 0.33 * Math.Min(W, H); // measured on real lpc2026 maps: bases sit 14.6-19.1 (mean 16.6) tiles from the center
        double mx = (W - 1) / 2.0, my = (H - 1) / 2.0;
        double offset = r.NextDouble() * Math.PI * 2;
        for (int i = 0; i < n; i++)
        {
            double a = offset + 2 * Math.PI * i / n;
            int bx = 0, by = 0;
            for (int attempt = 0; attempt < 50; attempt++)
            {
                bx = Math.Clamp((int)Math.Round(mx + radius * Math.Cos(a)) + r.Next(-1, 2), 1, W - 2);
                by = Math.Clamp((int)Math.Round(my + radius * Math.Sin(a)) + r.Next(-1, 2), 1, H - 2);
                if (baseAt[by * W + bx] < 0 && workerCount[by * W + bx] == 0) break;
            }
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    int x = bx + dx, y = by + dy;
                    if (x >= 0 && y >= 0 && x < W && y < H) { terrain[y * W + x] = Plain; amount[y * W + x] = 0; }
                }
            AddBase(i, bx, by);
            // Starting worker on a random neighbouring tile (all cleared to plain).
            int wx, wy, guard = 0;
            do
            {
                int d = r.Next(8);
                wx = bx + (d is 1 or 2 or 3 ? 1 : d is 5 or 6 or 7 ? -1 : 0);
                wy = by + (d is 0 or 1 or 7 ? -1 : d is 3 or 4 or 5 ? 1 : 0);
            } while ((wx < 0 || wy < 0 || wx >= W || wy >= H || baseAt[wy * W + wx] >= 0) && guard++ < 100);
            AddWorker(i, wx, wy);
        }
    }

    void PlaceInitialDeposits(Random r)
    {
        int n = teams.Length;
        int gold = cfg.GoldPerTeam * n;
        int center = (int)Math.Round(cfg.CenterPercent * gold, MidpointRounding.AwayFromZero);
        for (int i = 0; i < gold; i++) PlaceDeposit(Gold, i < center, -1, r);
        for (int i = 0; i < cfg.IronPerTeam * n; i++) PlaceDeposit(Iron, false, -1, r);
    }

    /// <summary>Puts a deposit in the preferred zone (falling back to the other one). Returns false if no room.</summary>
    bool PlaceDeposit(byte kind, bool preferCenter, int oldTile, Random r)
    {
        int tile = PickSite(preferCenter, oldTile, r);
        if (tile < 0) tile = PickSite(!preferCenter, oldTile, r);
        if (tile < 0) return false;
        bool inCenter = InCenter(tile);
        terrain[tile] = kind;
        amount[tile] = kind == Iron ? r.Next(cfg.IronMin, cfg.IronMax + 1)
            : inCenter ? r.Next(cfg.CenterGoldMin, cfg.CenterGoldMax + 1)
            : r.Next(cfg.OuterGoldMin, cfg.OuterGoldMax + 1);
        cellCache[tile] = null;
        return true;
    }

    int PickSite(bool center, int oldTile, Random r)
    {
        tier0.Clear(); tier1.Clear(); tier2.Clear();
        int ox = oldTile >= 0 ? oldTile % W : -100, oy = oldTile >= 0 ? oldTile / W : -100;
        for (int i = 0; i < N; i++)
        {
            if (terrain[i] != Plain || baseAt[i] >= 0 || workerCount[i] > 0 || InCenter(i) != center) continue;
            int x = i % W, y = i / W;
            tier2.Add(i);
            if (Math.Max(Math.Abs(x - ox), Math.Abs(y - oy)) < 2) continue;
            tier1.Add(i);
            if (!ResourceAdjacent(x, y)) tier0.Add(i);
        }
        var list = tier0.Count > 0 ? tier0 : tier1.Count > 0 ? tier1 : tier2;
        return list.Count == 0 ? -1 : list[r.Next(list.Count)];
    }

    bool ResourceAdjacent(int x, int y)
    {
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
            {
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                if (terrain[ny * W + nx] >= Gold) return true;
            }
        return false;
    }

    SimWorker AddWorker(int team, int x, int y)
    {
        var t = teams[team];
        var w = new SimWorker { Id = $"{t.Id}-w{++t.WorkerSeq}", Team = team, X = x, Y = y, ReadyOnTurn = Turn, ArriveTurn = Turn };
        workers.Add(w);
        t.Workers.Add(w);
        byId[w.Id] = w;
        workerCount[y * W + x]++;
        return w;
    }

    void AddBase(int team, int x, int y)
    {
        var t = teams[team];
        var b = new SimBase { Id = $"{t.Id}-b{++t.BaseSeq}", Team = team, X = x, Y = y };
        baseAt[y * W + x] = bases.Count;
        bases.Add(b);
        t.Bases.Add(b);
    }

    // ---------------------------------------------------------------- views

    Cell CellAt(int i)
    {
        var c = cellCache[i];
        if (c != null) return c;
        byte k = terrain[i];
        c = k < Gold ? PlainCells[k] : new Cell(TypeNames[k], amount[i], null, null, null);
        cellCache[i] = c;
        return c;
    }

    void Reveal(int x, int y)
    {
        int v = cfg.Vision;
        for (int dy = -v; dy <= v; dy++)
        {
            int yy = y + dy;
            if (yy < 0 || yy >= H) continue;
            int r = v - Math.Abs(dy);
            int x0 = Math.Max(0, x - r), x1 = Math.Min(W - 1, x + r);
            int row = yy * W;
            for (int xx = x0; xx <= x1; xx++) visStamp[row + xx] = stamp;
        }
        byte k = terrain[y * W + x];
        if (k is Mountain or Crater)
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && ny >= 0 && nx < W && ny < H) visStamp[ny * W + nx] = stamp;
                }
    }

    public GameState<MarsView> ViewFor(int teamIndex)
    {
        var team = teams[teamIndex];
        stamp++;
        foreach (var w in team.Workers) Reveal(w.X, w.Y);
        foreach (var b in team.Bases) Reveal(b.X, b.Y);

        var cells = new List<Cell>(N);
        for (int i = 0; i < N; i++) cells.Add(visStamp[i] == stamp ? CellAt(i) : Hidden);

        var units = new List<Worker>();
        foreach (var w in workers)
        {
            if (visStamp[w.Y * W + w.X] != stamp) continue;
            bool own = w.Team == teamIndex;
            units.Add(new Worker(w.Id, teams[w.Team].Id, new Coordinate(w.X, w.Y), w.ReadyOnTurn,
                own ? new Inventory(w.Gold, w.Iron) : null, cfg.Capacity));
        }
        var baseList = new List<Base>();
        foreach (var b in bases)
            if (visStamp[b.Y * W + b.X] == stamp) baseList.Add(new Base(b.Id, teams[b.Team].Id, new Coordinate(b.X, b.Y)));

        var scores = new Dictionary<string, long>(teams.Length);
        foreach (var t in teams) scores[t.Id] = t.BankGold;
        var deposits = new Dictionary<string, Inventory>(team.Bases.Count);
        foreach (var b in team.Bases) deposits[b.Id] = new Inventory(b.DepGold, b.DepIron);

        var view = new MarsView(Turn, new Grid(W, H, cells), units, baseList, scores,
            new Dictionary<string, Inventory> { [team.Id] = new Inventory(team.BankGold, team.BankIron) },
            deposits, highGold, null, null, null);
        var match = new Match($"sim-{cfg.Seed}", "practice", Done ? "completed" : "active", "client-action", "lpc2026",
            Turn, cfg.MaxTurns, null, null, teamIds, teamIds, teamIds, teamList);
        return new GameState<MarsView>(match, view, 0, null, null);
    }

    // ---------------------------------------------------------------- turn resolution

    public void Step(IReadOnlyList<IReadOnlyDictionary<string, MarsAction>?> actionsByTeam)
    {
        if (Done) return;
        int t = Turn;

        // 1. Due respawns.
        for (int i = 0; i < respawns.Count; i++)
        {
            var rs = respawns[i];
            if (rs.DueTurn > t) continue;
            bool preferCenter = rs.IsGold && rng.NextDouble() < cfg.CenterPercent;
            if (PlaceDeposit(rs.IsGold ? Gold : Iron, preferCenter, rs.OldTile, rng)) { respawns.RemoveAt(i); i--; }
        }

        // Accept actions (only for own, ready workers; others are consumed without effect).
        foreach (var w in workers) w.Action = null;
        for (int ti = 0; ti < teams.Length && ti < actionsByTeam.Count; ti++)
        {
            var acts = actionsByTeam[ti];
            if (acts == null) continue;
            foreach (var (id, a) in acts)
            {
                if (a == null || !byId.TryGetValue(id, out var w) || w.Team != ti || t < w.ReadyOnTurn) continue;
                w.Action = a;
            }
        }

        // 2. Moves.
        int count = workers.Count;
        for (int i = 0; i < count; i++)
        {
            var w = workers[i];
            if (w.Action is not { Type: "move", Direction: { } dir }) continue;
            var (dx, dy) = Offset(dir);
            int nx = w.X + dx, ny = w.Y + dy;
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            int ni = ny * W + nx;
            int b = baseAt[ni];
            if (b >= 0 && bases[b].Team != w.Team) continue;
            workerCount[w.Y * W + w.X]--;
            workerCount[ni]++;
            w.X = nx; w.Y = ny;
            w.ArriveTurn = t;
            w.Streak = 0;
            if (terrain[ni] is Mountain or Crater) w.ReadyOnTurn = t + cfg.ObstacleCost;
        }

        // 3. Construction, in rotating team order; within a team by worker ID.
        int nt = teams.Length;
        for (int k = 0; k < nt; k++)
        {
            int ti = teamOrder[(t + k) % nt];
            var team = teams[ti];
            scratch.Clear();
            foreach (var w in team.Workers)
                if (w.Action is { Type: "spawn" or "build-base" }) scratch.Add(w);
            if (scratch.Count == 0) continue;
            scratch.Sort((a, b) => string.CompareOrdinal(a.Id, b.Id));
            foreach (var w in scratch)
            {
                if (w.Action!.Type == "spawn")
                {
                    if (team.BankIron < team.CostA) continue;
                    team.BankIron -= team.CostA;
                    IronSpent += team.CostA;
                    (team.CostA, team.CostB) = (team.CostB, team.CostA + team.CostB);
                    team.Recruited++;
                    var nw = AddWorker(ti, w.X, w.Y);
                    nw.ReadyOnTurn = t + 1;
                    nw.ArriveTurn = t;
                }
                else
                {
                    int i = w.Y * W + w.X;
                    if (team.BankIron < 6 || terrain[i] != Plain || baseAt[i] >= 0 || workerCount[i] != 1) continue;
                    if (!FarFromEnemyBases(ti, w.X, w.Y)) continue;
                    team.BankIron -= 6;
                    IronSpent += 6;
                    AddBase(ti, w.X, w.Y);
                }
            }
        }

        // 4. Mining.
        scratch.Clear();
        for (int i = 0; i < count; i++)
        {
            var w = workers[i];
            if (w.Action is not { Type: "mine" }) { w.Streak = 0; continue; }
            int tile = w.Y * W + w.X;
            bool eligible = terrain[tile] >= Gold && amount[tile] > 0 && w.Gold + w.Iron < cfg.Capacity && ClaimOk(w.Team, w.X, w.Y);
            if (!eligible) { w.Streak = 0; continue; }
            w.TieKey = rng.Next();
            scratch.Add(w);
        }
        if (scratch.Count > 0)
        {
            scratch.Sort((a, b) =>
            {
                int ta = a.Y * W + a.X, tb = b.Y * W + b.X;
                if (ta != tb) return ta.CompareTo(tb);
                if (a.Streak != b.Streak) return b.Streak.CompareTo(a.Streak);
                if (a.ArriveTurn != b.ArriveTurn) return a.ArriveTurn.CompareTo(b.ArriveTurn);
                return a.TieKey.CompareTo(b.TieKey);
            });
            foreach (var w in scratch)
            {
                int tile = w.Y * W + w.X;
                w.Streak++;
                if (amount[tile] <= 0) continue;
                bool gold = terrain[tile] == Gold;
                amount[tile]--;
                cellCache[tile] = null;
                if (gold) { w.Gold++; GoldMined++; } else { w.Iron++; IronMined++; }
                if (amount[tile] == 0)
                {
                    terrain[tile] = Plain;
                    respawns.Add(new Respawn(gold, tile, t + 1));
                }
            }
        }

        // 5. Trade.
        for (int i = 0; i < count; i++)
        {
            var w = workers[i];
            if (w.Action is not { Type: "trade", From: { } from }) continue;
            if (from == Resource.Iron && w.Iron >= 3) { w.Iron -= 3; w.Gold += 1; TradeIronNet -= 3; TradeGoldNet += 1; }
            else if (from == Resource.Gold && w.Gold >= 3) { w.Gold -= 3; w.Iron += 1; TradeGoldNet -= 3; TradeIronNet += 1; }
        }

        // 6. Deposit.
        for (int i = 0; i < count; i++)
        {
            var w = workers[i];
            if (w.Action is not { Type: "deposit" }) continue;
            int b = baseAt[w.Y * W + w.X];
            if (b < 0 || bases[b].Team != w.Team) continue;
            var team = teams[w.Team];
            team.BankGold += w.Gold; team.BankIron += w.Iron;
            bases[b].DepGold += w.Gold; bases[b].DepIron += w.Iron;
            w.Gold = 0; w.Iron = 0;
        }

        Turn++;
    }

    static (int Dx, int Dy) Offset(Direction d) => d switch
    {
        Direction.North => (0, -1),
        Direction.NorthEast => (1, -1),
        Direction.East => (1, 0),
        Direction.SouthEast => (1, 1),
        Direction.South => (0, 1),
        Direction.SouthWest => (-1, 1),
        Direction.West => (-1, 0),
        Direction.NorthWest => (-1, -1),
        _ => (0, 0),
    };

    bool ClaimOk(int team, int x, int y)
    {
        bool claimed = false;
        foreach (var b in bases)
        {
            if (Math.Abs(b.X - x) + Math.Abs(b.Y - y) > 2) continue;
            if (b.Team == team) return true;
            claimed = true;
        }
        return !claimed;
    }

    bool FarFromEnemyBases(int team, int x, int y)
    {
        foreach (var b in bases)
            if (b.Team != team && Math.Abs(b.X - x) + Math.Abs(b.Y - y) <= 2) return false;
        return true;
    }

    // ---------------------------------------------------------------- results and debugging

    public IReadOnlyList<SimTeamResult> Standings()
    {
        var rows = teams.Select(t => new SimTeamResult(t.Id, t.Name, t.BankGold, t.BankIron,
            t.Workers.Sum(w => w.Gold), t.Workers.Sum(w => w.Iron), t.Bases.Count, t.Workers.Count, 0)).ToList();
        static (long, long, long, long, long, long) Key(SimTeamResult r) =>
            (r.BankedGold, r.BankedIron, r.CarriedGold, r.CarriedIron, r.Bases, r.Workers);
        var sorted = rows.OrderByDescending(Key).ToList();
        var result = new List<SimTeamResult>(sorted.Count);
        for (int i = 0; i < sorted.Count; i++)
        {
            int place = i > 0 && Key(sorted[i]) == Key(sorted[i - 1]) ? result[i - 1].Place : i + 1;
            result.Add(sorted[i] with { Place = place });
        }
        return result;
    }

    /// <summary>Full-state worker records (with inventories) for tests and debugging.</summary>
    public IEnumerable<Worker> AllWorkers() => workers.Select(w =>
        new Worker(w.Id, teams[w.Team].Id, new Coordinate(w.X, w.Y), w.ReadyOnTurn, new Inventory(w.Gold, w.Iron), cfg.Capacity));

    public IEnumerable<Base> AllBases() => bases.Select(b => new Base(b.Id, teams[b.Team].Id, new Coordinate(b.X, b.Y)));

    public Inventory Bank(int teamIndex) => new(teams[teamIndex].BankGold, teams[teamIndex].BankIron);

    public string TerrainAt(int x, int y) => TypeNames[terrain[y * W + x]];

    public long AmountAt(int x, int y) => amount[y * W + x];

    public int PendingRespawns => respawns.Count;

    public string RenderAscii()
    {
        var sb = new StringBuilder((W + 1) * H + 32);
        sb.Append("turn ").Append(Turn).Append('\n');
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                int i = y * W + x;
                char c;
                if (baseAt[i] >= 0)
                {
                    int team = bases[baseAt[i]].Team;
                    c = team < 26 ? (char)('A' + team) : (char)('a' + team % 26);
                }
                else if (workerCount[i] > 0) c = '@';
                else c = terrain[i] switch { Gold => 'G', Iron => 'i', Mountain => '^', Crater => 'o', _ => '.' };
                sb.Append(c);
            }
            sb.Append('\n');
        }
        return sb.ToString();
    }
}
