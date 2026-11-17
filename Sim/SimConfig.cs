namespace Lpc.Sim;

/// <summary>Match settings for the local simulator. Defaults follow the LPC 2026 rules.</summary>
public sealed class SimConfig
{
    public int Width { get; set; } = 50;
    public int Height { get; set; } = 50;
    public int MaxTurns { get; set; } = 480;
    public int CenterSize { get; set; } = 16;
    public double MountainWeight { get; set; } = 0.20;
    public double CraterWeight { get; set; } = 0.10;
    /// <summary>obstacleTraversalCost: includes the entry turn, so 3 means two extra turns of cooldown.</summary>
    public int ObstacleCost { get; set; } = 3;
    public int Vision { get; set; } = 3;
    public int Capacity { get; set; } = 8;
    public long StartIron { get; set; } = 15;
    public int GoldPerTeam { get; set; } = 3;
    public int IronPerTeam { get; set; } = 2;
    public double CenterPercent { get; set; } = 0.35;
    public int OuterGoldMin { get; set; } = 6;
    public int OuterGoldMax { get; set; } = 9;
    public int CenterGoldMin { get; set; } = 12;
    public int CenterGoldMax { get; set; } = 18;
    public int IronMin { get; set; } = 2;
    public int IronMax { get; set; } = 4;
    public int Seed { get; set; }

    public SimConfig Clone() => (SimConfig)MemberwiseClone();
}
