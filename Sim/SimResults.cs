namespace Lpc.Sim;

public sealed record SimTeamResult(string TeamId, string Name, long BankedGold, long BankedIron, long CarriedGold,
    long CarriedIron, int Bases, int Workers, int Place);

public sealed record SimResult(int Seed, IReadOnlyList<SimTeamResult> Teams, double ElapsedMs);
