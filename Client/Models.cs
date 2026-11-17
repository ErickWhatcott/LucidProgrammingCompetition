// C# port of the record types at the top of the Java starter's BotClient.java.
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Lpc.Client;

public sealed record Identity(string TeamId, string? DisplayName);

public sealed record Match(string MatchId, string? Mode, string Status, string? TurnAdvanceMode,
    string? RulesVersion, int Turn, int MaxTurns, long? StartedAtMs, long? CompletedAtMs,
    List<string>? ExpectedTeamIds, List<string>? TeamIds, List<string>? ActiveTeamIds,
    List<Team>? Teams);

public sealed record Team(string Id, string? DisplayName);

public sealed record GameState<V>(Match Match, V? View, long ServerTimeMs, long? TurnEndsAtMs,
    Dictionary<string, long>? ActionRevisions);

public readonly record struct Coordinate(int X, int Y);

public sealed record Base(string BaseId, string TeamId, Coordinate Position);

public sealed record Inventory(long Gold, long Iron);

public sealed record Worker(string UnitId, string TeamId, Coordinate Position, int ReadyOnTurn,
    Inventory? Inventory, long InventoryCapacity);

public sealed record Formation(int X, int Y, int Size);

public sealed record Cell(string Type, long? Amount, Formation? Formation, string? UnitId, string? TeamId);

public interface ICoordinateGrid
{
    int Width { get; }
    int Height { get; }
    string Terrain(Coordinate point);
    long Amount(Coordinate point) => 0;
}

public sealed record Grid(int Width, int Height, List<Cell> Cells) : ICoordinateGrid
{
    public Cell Cell(Coordinate p) => Cells[p.Y * Width + p.X];
    public string Terrain(Coordinate p) => Cell(p).Type;
    public long Amount(Coordinate p) => Cell(p).Amount ?? 0;
}

public sealed record HighGoldArea(int Width, int Height, long Min, long Max, int SpawnPercent);

public sealed record Notice(string? EntityId, string? Error, string? Note);

public sealed record Effect(string Type, Coordinate? Coordinate, List<string>? EntityIds, string? Note,
    string? BaseId, string? Resource, long? Amount, long? Total);

public sealed record MarsView(int Turn, Grid Map, List<Worker> Units, List<Base> Bases,
    Dictionary<string, long>? Scores, Dictionary<string, Inventory>? TeamResources,
    Dictionary<string, Inventory>? BaseResourceDeposits, HighGoldArea? HighGoldArea,
    List<Notice>? Warnings, List<Notice>? Info, List<Effect>? Effects) : ICoordinateGrid
{
    public int Width => Map.Width;
    public int Height => Map.Height;
    public string Terrain(Coordinate p) => Map.Terrain(p);
    public long Amount(Coordinate p) => Map.Amount(p);
}

public sealed record Position(double X, double Y);

public enum Planet { Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto }

public sealed record SolarPlanet(Planet Name, Position Position, double OrbitRadius);

public sealed record Orbit(Planet Planet, double Angle);

public sealed record Rocket(string TeamId, Position Position, Position PreviousPosition, double Heading,
    Planet Target, Position TargetPosition, Orbit? Orbit);

public sealed record SolarView(int Turn, List<SolarPlanet> Planets, List<Rocket> Rockets, string? EntityId);

[JsonConverter(typeof(JsonStringEnumConverter<Resource>))]
public enum Resource
{
    [JsonStringEnumMemberName("gold")] Gold,
    [JsonStringEnumMemberName("iron")] Iron,
}

[JsonConverter(typeof(JsonStringEnumConverter<Direction>))]
public enum Direction
{
    [JsonStringEnumMemberName("north")] North,
    [JsonStringEnumMemberName("east")] East,
    [JsonStringEnumMemberName("south")] South,
    [JsonStringEnumMemberName("west")] West,
    [JsonStringEnumMemberName("north-east")] NorthEast,
    [JsonStringEnumMemberName("south-east")] SouthEast,
    [JsonStringEnumMemberName("south-west")] SouthWest,
    [JsonStringEnumMemberName("north-west")] NorthWest,
}

public interface IAction;

public sealed record MarsAction(string Type, Direction? Direction, Resource? From) : IAction
{
    public static MarsAction Move(Direction direction) => new("move", direction, null);
    public static MarsAction Mine() => new("mine", null, null);
    public static MarsAction Deposit() => new("deposit", null, null);
    public static MarsAction Spawn() => new("spawn", null, null);
    public static MarsAction BuildBase() => new("build-base", null, null);
    public static MarsAction Trade(Resource from) => new("trade", null, from);
    public static MarsAction Pass() => new("pass", null, null);

    public override string ToString() => Type switch
    {
        "move" => $"move {Direction}",
        "trade" => $"trade {From}",
        _ => Type,
    };
}

public sealed record SolarAction(string Type, double? X, double? Y) : IAction
{
    public static SolarAction Target(Planet planet) => new(planet.ToString(), null, null);
    public static SolarAction Move(double x, double y) => new("move", x, y);
}

public static class LpcJson
{
    /// <summary>Matches Gson's behavior in the Java starter: camelCase names, nulls omitted.</summary>
    public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web)
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };
}
