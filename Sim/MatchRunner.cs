using System.Diagnostics;
using Lpc.Client;
using Lpc.Strategies;

namespace Lpc.Sim;

/// <summary>Plays one full simulated match, single-threaded (run several matches in parallel if needed).</summary>
public static class MatchRunner
{
    public static SimResult Run(SimConfig config, IReadOnlyList<IMarsStrategy> strategies, Action<Simulator>? afterTurn = null)
    {
        long start = Stopwatch.GetTimestamp();
        var sim = new Simulator(config, strategies.Select(s => s.Name).ToList());
        int n = strategies.Count;
        var identities = new Identity[n];
        for (int i = 0; i < n; i++) identities[i] = new Identity(sim.TeamIds[i], strategies[i].Name);
        var failed = new bool[n];
        var actions = new IReadOnlyDictionary<string, MarsAction>?[n];

        while (!sim.Done)
        {
            for (int i = 0; i < n; i++)
            {
                try
                {
                    actions[i] = strategies[i].CalculateActions(sim.ViewFor(i), identities[i]);
                }
                catch (Exception e)
                {
                    actions[i] = null;
                    if (!failed[i])
                    {
                        failed[i] = true;
                        Console.Error.WriteLine($"[sim seed {config.Seed}] {strategies[i].Name} ({identities[i].TeamId}) threw on turn {sim.Turn}: {e}");
                    }
                }
            }
            sim.Step(actions);
            afterTurn?.Invoke(sim);
        }

        for (int i = 0; i < n; i++)
        {
            try { strategies[i].OnMatchEnd(sim.ViewFor(i), identities[i]); }
            catch (Exception e) { Console.Error.WriteLine($"[sim seed {config.Seed}] {strategies[i].Name} OnMatchEnd threw: {e.Message}"); }
        }

        return new SimResult(config.Seed, sim.Standings(), Stopwatch.GetElapsedTime(start).TotalMilliseconds);
    }
}
