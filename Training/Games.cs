using System.Collections.Concurrent;
using Lpc.Sim;
using Lpc.Strategies;

namespace Lpc.Training;

/// <summary>Runs many simulated matches in parallel (one game per core) and builds opponent rosters.</summary>
public static class Games
{
    public static int DefaultParallelism => Environment.ProcessorCount;

    public static List<SimResult> RunMany(IEnumerable<int> seeds, Func<int, IReadOnlyList<IMarsStrategy>> roster,
        int parallel, int maxTurns = 0, Action<SimResult>? done = null, Action<Simulator>? afterTurn = null)
    {
        var results = new ConcurrentBag<SimResult>();
        Parallel.ForEach(seeds, new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, parallel) }, seed =>
        {
            var config = new SimConfig { Seed = seed };
            if (maxTurns > 0) config.MaxTurns = maxTurns;
            var result = MatchRunner.Run(config, roster(seed), afterTurn);
            results.Add(result);
            done?.Invoke(result);
        });
        return results.OrderBy(r => r.Seed).ToList();
    }

    /// <summary>Opponent mixes. "mixed" approximates practice lobbies; "strong" adds copies of our own engines.</summary>
    public static string[] Mix(string mix) => mix switch
    {
        "mixed" => ["intermediate", "beginner", "intermediate", "novice", "beginner", "intermediate"],
        "strong" => ["mlp", "intermediate", "gbdt", "intermediate", "es", "beginner", "ppo", "intermediate"],
        // Every non-ML style: closest thing we have to a lobby of other teams' bots.
        "zoo" => ["intermediate", "rusher", "ironlord", "expander", "swarm", "blocker", "camper", "random", "homebody", "intermediate", "beginner", "rusher"],
        // Non-ML solver league: strong scripted algorithms, one per style.
        "solvers" => ["hungarian", "territory", "centerbank", "investor", "denial", "scout", "intermediate"],
        _ => mix.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries),
    };

    public static List<IMarsStrategy> Opponents(string mix, int count, int seed, StrategyOptions shared)
    {
        var modes = Mix(mix);
        var list = new List<IMarsStrategy>();
        for (int i = 0; i < count; i++)
        {
            var o = Clone(shared);
            o.Seed = seed * 101 + i;
            o.Recorder = null;
            o.Decisions = null;
            o.EsGenome = null;
            o.Noise = 0;
            o.Quiet = true;
            list.Add(StrategyFactory.Create(modes[(i + seed) % modes.Length], o));
        }
        return list;
    }

    public static StrategyOptions Clone(StrategyOptions o) => new()
    {
        Config = o.Config, ConfigPath = o.ConfigPath, ModelPath = o.ModelPath, Mlp = o.Mlp, Gbdt = o.Gbdt, Policy = o.Policy,
        Es = o.Es, EsGenome = o.EsGenome, Blend = o.Blend, Temperature = o.Temperature, Recorder = o.Recorder,
        Decisions = o.Decisions, RecordSource = o.RecordSource, Noise = o.Noise, Seed = o.Seed, Quiet = o.Quiet,
    };

    public static SimTeamResult Seat(SimResult result, int seat) => result.Teams.First(t => t.TeamId == $"team-{seat:00}");

    public static (double Mean, double StdErr) MeanSe(IReadOnlyList<double> values)
    {
        if (values.Count == 0) return (0, 0);
        double mean = values.Average();
        if (values.Count < 2) return (mean, 0);
        double variance = values.Sum(v => (v - mean) * (v - mean)) / (values.Count - 1);
        return (mean, Math.Sqrt(variance / values.Count));
    }
}
