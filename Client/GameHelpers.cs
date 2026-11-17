// C# port of the game helpers below the helper banner in the Java starter's Main.java.
namespace Lpc.Client;

public static class GameHelpers
{
    static bool InBounds(ICoordinateGrid grid, Coordinate p) =>
        p.X >= 0 && p.Y >= 0 && p.X < grid.Width && p.Y < grid.Height;

    public static ISet<string> IsClaimed(IEnumerable<Base> bases, Coordinate p)
    {
        var claims = new HashSet<string>();
        foreach (var b in bases)
        {
            int distance = Math.Abs(b.Position.X - p.X) + Math.Abs(b.Position.Y - p.Y);
            if (distance <= 2) claims.Add(b.TeamId);
        }
        return claims;
    }

    public static string? IsBase(IEnumerable<Base> bases, Coordinate p) =>
        bases.FirstOrDefault(b => b.Position == p)?.TeamId;

    public static string? IsOre(ICoordinateGrid grid, Coordinate p)
    {
        if (!InBounds(grid, p)) return null;

        var type = grid.Terrain(p);
        return (type == "gold" || type == "iron") && grid.Amount(p) > 0 ? type : null;
    }

    public static bool CanMoveOnto(ICoordinateGrid grid, IEnumerable<Base> bases, Coordinate p, string? teamId)
    {
        if (!InBounds(grid, p)) return false;

        var type = grid.Terrain(p);
        var owner = IsBase(bases, p);
        return type != "hidden" && type != "void" && (owner == null || owner == teamId);
    }

    /// <summary>Cost until ready at destination; start must be ready. Null means no known route.</summary>
    public static List<Coordinate>? PathTo(ICoordinateGrid grid, Coordinate start, Coordinate end,
        IReadOnlyCollection<Base> bases, string? teamId, int obstacleCost)
    {
        if (obstacleCost < 1
            || !CanMoveOnto(grid, bases, start, teamId)
            || !CanMoveOnto(grid, bases, end, teamId))
        {
            return null;
        }

        var distance = new Dictionary<Coordinate, long>();
        var previous = new Dictionary<Coordinate, Coordinate>();
        var queue = new PriorityQueue<(Coordinate Position, long Cost), long>();
        distance[start] = 0;
        queue.Enqueue((start, 0), 0);

        while (queue.Count > 0)
        {
            var step = queue.Dequeue();
            var p = step.Position;
            if (step.Cost != distance[p]) continue;
            if (p == end) break;

            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dy == 0) continue;
                    var next = new Coordinate(p.X + dx, p.Y + dy);
                    if (!CanMoveOnto(grid, bases, next, teamId)) continue;

                    var type = grid.Terrain(next);
                    long cost = distance[p] + (type == "mountain" || type == "crater" ? obstacleCost : 1);
                    if (cost >= distance.GetValueOrDefault(next, long.MaxValue)) continue;

                    distance[next] = cost;
                    previous[next] = p;
                    queue.Enqueue((next, cost), cost);
                }
            }
        }

        if (!distance.ContainsKey(end)) return null;

        var path = new List<Coordinate>();
        for (var p = end; p != start; p = previous[p]) path.Add(p);

        path.Reverse();
        return path;
    }

    public static List<Coordinate>? PathTo(ICoordinateGrid grid, Coordinate start, Coordinate end) =>
        PathTo(grid, start, end, [], null, 3);
}
