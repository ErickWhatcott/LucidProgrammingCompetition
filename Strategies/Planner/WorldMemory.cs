using Lpc.Client;

namespace Lpc.Strategies.Planner;

public enum Tile : byte { Unknown, Plain, Mountain, Crater, Gold, Iron, Void }

/// <summary>
/// Everything a team has seen so far. Vision is only Manhattan distance 3, so terrain, deposits
/// and bases outside vision are remembered from the last time they were visible.
/// </summary>
public sealed class WorldMemory
{
    public int Width { get; private set; }
    public int Height { get; private set; }
    public int Count => Width * Height;
    public Tile[] Kind = [];
    public int[] Amount = [];
    public int[] LastSeen = [];
    public bool[] Visible = [];
    public readonly Dictionary<string, Base> Bases = [];
    public string?[] BaseOwnerAt = [];
    public int Turn { get; private set; }
    public string TeamId { get; }
    public int CenterX0 { get; private set; }
    public int CenterY0 { get; private set; }
    public int CenterW { get; private set; }
    public int CenterH { get; private set; }

    public WorldMemory(string teamId) => TeamId = teamId;

    public int Index(int x, int y) => y * Width + x;
    public int Index(Coordinate c) => c.Y * Width + c.X;
    public int X(int index) => index % Width;
    public int Y(int index) => index / Width;
    public Coordinate At(int index) => new(index % Width, index / Width);
    public bool InBounds(int x, int y) => x >= 0 && y >= 0 && x < Width && y < Height;

    public bool InCenter(int index)
    {
        int x = X(index), y = Y(index);
        return x >= CenterX0 && y >= CenterY0 && x < CenterX0 + CenterW && y < CenterY0 + CenterH;
    }

    public static Tile Parse(string type) => type switch
    {
        "hidden" => Tile.Unknown,
        "mountain" => Tile.Mountain,
        "crater" => Tile.Crater,
        "gold" => Tile.Gold,
        "iron" => Tile.Iron,
        "void" => Tile.Void,
        _ => Tile.Plain, // the rules call it "plain"/"empty"; anything unrecognized is walkable ground
    };

    public void Update(MarsView view)
    {
        var map = view.Map;
        if (Width != map.Width || Height != map.Height)
        {
            Width = map.Width;
            Height = map.Height;
            Kind = new Tile[Count];
            Amount = new int[Count];
            LastSeen = Enumerable.Repeat(-1, Count).ToArray();
            Visible = new bool[Count];
            BaseOwnerAt = new string?[Count];
            int cw = view.HighGoldArea?.Width ?? Math.Min(16, Width);
            int ch = view.HighGoldArea?.Height ?? Math.Min(16, Height);
            // "When exact centering falls between cells, the rectangle is shifted toward the top/left."
            CenterW = cw; CenterH = ch; CenterX0 = (Width - cw) / 2; CenterY0 = (Height - ch) / 2;
        }
        Turn = view.Turn;
        var cells = map.Cells;
        for (int i = 0; i < cells.Count; i++)
        {
            var cell = cells[i];
            if (cell.Type == "hidden") { Visible[i] = false; continue; }
            Visible[i] = true;
            LastSeen[i] = Turn;
            Kind[i] = Parse(cell.Type);
            Amount[i] = (int)(cell.Amount ?? 0);
            if (Kind[i] is Tile.Gold or Tile.Iron && Amount[i] <= 0) Kind[i] = Tile.Plain;
        }

        foreach (var b in view.Bases) Bases[b.BaseId] = b;
        // A remembered base whose tile is visible but not listed is gone (or was misremembered).
        var listed = view.Bases.Select(b => b.BaseId).ToHashSet();
        foreach (var id in Bases.Keys.ToList())
            if (!listed.Contains(id) && Visible[Index(Bases[id].Position)]) Bases.Remove(id);

        Array.Clear(BaseOwnerAt);
        foreach (var b in Bases.Values) BaseOwnerAt[Index(b.Position)] = b.TeamId;
    }

    public IEnumerable<Base> MyBases => Bases.Values.Where(b => b.TeamId == TeamId);
    public IEnumerable<Base> EnemyBases => Bases.Values.Where(b => b.TeamId != TeamId);

    /// <summary>Claims are Manhattan distance 2. Mineable if unclaimed or if one of our bases claims it.</summary>
    public (bool Mineable, bool ClaimedByUs, bool ClaimedByEnemy) Claim(int index)
    {
        int x = X(index), y = Y(index);
        bool us = false, enemy = false;
        foreach (var b in Bases.Values)
        {
            if (Math.Abs(b.Position.X - x) + Math.Abs(b.Position.Y - y) > 2) continue;
            if (b.TeamId == TeamId) us = true; else enemy = true;
        }
        return (us || !enemy, us, enemy);
    }

    /// <summary>A build site must be more than two Manhattan steps from every (known) enemy base.</summary>
    public bool FarFromEnemyBases(int index)
    {
        int x = X(index), y = Y(index);
        foreach (var b in Bases.Values)
            if (b.TeamId != TeamId && Math.Abs(b.Position.X - x) + Math.Abs(b.Position.Y - y) <= 2) return false;
        return true;
    }
}
