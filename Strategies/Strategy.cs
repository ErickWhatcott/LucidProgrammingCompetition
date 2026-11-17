using System.Diagnostics;
using Lpc.Client;

namespace Lpc.Strategies;

/// <summary>A per-team, per-match decision maker. Instances keep memory between turns.</summary>
public interface IMarsStrategy
{
    string Name { get; }
    IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity);
    void OnMatchEnd(GameState<MarsView>? finalState, Identity identity) { }
}

/// <summary>Wraps a strategy and records how long each turn's decision takes.</summary>
public sealed class TimedStrategy(IMarsStrategy inner) : IMarsStrategy
{
    readonly List<double> samples = [];
    public IMarsStrategy Inner => inner;
    public string Name => inner.Name;
    public IReadOnlyList<double> SamplesMs => samples;

    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity)
    {
        long start = Stopwatch.GetTimestamp();
        var actions = inner.CalculateActions(state, identity);
        samples.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        return actions;
    }

    public void OnMatchEnd(GameState<MarsView>? finalState, Identity identity) => inner.OnMatchEnd(finalState, identity);
}

public static class Directions
{
    public static readonly (int Dx, int Dy, Direction Direction)[] All =
    [
        (0, -1, Direction.North), (1, -1, Direction.NorthEast), (1, 0, Direction.East), (1, 1, Direction.SouthEast),
        (0, 1, Direction.South), (-1, 1, Direction.SouthWest), (-1, 0, Direction.West), (-1, -1, Direction.NorthWest),
    ];

    public static Direction Of(int dx, int dy)
    {
        foreach (var (x, y, d) in All) if (x == Math.Sign(dx) && y == Math.Sign(dy)) return d;
        throw new ArgumentException("No direction for a zero offset.");
    }

    public static (int Dx, int Dy) Offset(Direction direction)
    {
        foreach (var (x, y, d) in All) if (d == direction) return (x, y);
        throw new ArgumentOutOfRangeException(nameof(direction));
    }
}
