using System.Collections.Concurrent;
using System.Diagnostics;
using Lpc.Ml;
using Lpc.Ml.Es;
using Lpc.Sim;
using Lpc.Strategies;

namespace Lpc.Training;

public sealed class EsVsStrongOptions
{
    public string Start = "models/es-hailmary.json";
    public string Output = "models/es-opp.json";
    public double Minutes = 60;
    public int SeedsPerGeneration = 8;
    public int Lambda = 12;
    public double Sigma = 0.07;
    public int Parallel = Games.DefaultParallelism;
}

/// <summary>
/// CMA-ES for the 'es' engine against a field of strong players only (the tournament snapshot, the
/// hail-mary variant and the trained mlp/gbdt/ppo engines), so fitness rewards beating good teams
/// rather than farming scripted bots. Every candidate plays the same maps (paired), seat 0 vs 15 opponents;
/// fitness = mean of (its banked gold - mean banked gold of the other 15).
/// Stops after Minutes or when logs/es-opp.stop exists. Saves the decoded mean every generation.
/// </summary>
public static class EsVsStrong
{
    const string StopFile = "logs/es-opp.stop";

    public static void Run(EsVsStrongOptions o)
    {
        File.Delete(StopFile);
        var start = EsModel.Load(o.Start);
        var template = start.Config.Clone();
        // Knobs sitting on the box edge (new opponent knobs start at 0) get half their samples clamped back
        // to the edge; start them a little inside so CMA-ES actually explores them.
        var mean0 = EsGenome.Encode(start.Config, start.Weights);
        var names = EsGenome.Names;
        for (int i = 0; i < EsGenome.KnobCount; i++)
            if (names[i] is "RaceDiscount" or "CrowdPenalty" or "BehindCenterBonus" or "InvestCommit" && mean0[i] < 0.15) mean0[i] = 0.15;
        var es = new CmaEs(mean0, o.Sigma, o.Lambda, seed: 7);

        var field = new List<Func<int, IMarsStrategy>>();
        var final = EsModel.Load("models/final/es.json");
        var hail = EsModel.Load(o.Start);
        var mlp = Mlp.Load(Paths.Model("mlp"));
        var gbdt = GbdtModel.Load(Paths.Model("gbdt"));
        var ppo = PolicyNet.Load(Paths.Model("ppo"));
        field.Add(s => StrategyFactory.Create("es", new StrategyOptions { Es = final, Seed = s, Quiet = true }));
        field.Add(s => StrategyFactory.Create("es", new StrategyOptions { Es = hail, Seed = s, Quiet = true }));
        field.Add(s => StrategyFactory.Create("mlp", new StrategyOptions { Mlp = mlp, Seed = s, Quiet = true }));
        field.Add(s => StrategyFactory.Create("gbdt", new StrategyOptions { Gbdt = gbdt, Seed = s, Quiet = true }));
        field.Add(s => StrategyFactory.Create("ppo", new StrategyOptions { Policy = ppo, Seed = s, Quiet = true }));

        Console.WriteLine($"[es-opp] start {o.Start}; dimension {es.Dimension}; lambda {o.Lambda}; {o.SeedsPerGeneration} maps/generation; field es(final), es(hail), mlp, gbdt, ppo");
        var clock = Stopwatch.StartNew();
        for (int gen = 0; clock.Elapsed.TotalMinutes < o.Minutes && !File.Exists(StopFile); gen++)
        {
            var xs = es.Ask();
            var genomes = xs.Select(x => EsGenome.Decode(x, template)).ToArray();
            int firstSeed = 300_000 + gen * o.SeedsPerGeneration;
            var rel = new ConcurrentDictionary<(int, int), (double Rel, int Place)>();
            var jobs = from k in Enumerable.Range(0, xs.Length) from s in Enumerable.Range(firstSeed, o.SeedsPerGeneration) select (k, s);
            Parallel.ForEach(jobs, new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, o.Parallel) }, job =>
            {
                var (k, seed) = job;
                var roster = new List<IMarsStrategy>
                {
                    StrategyFactory.Create("es", new StrategyOptions { EsGenome = genomes[k], Seed = seed, Quiet = true }),
                };
                for (int i = 0; i < 15; i++) roster.Add(field[(i + seed) % field.Count](seed * 101 + i));
                var result = MatchRunner.Run(new SimConfig { Seed = seed }, roster);
                var me = result.Teams.First(t => t.TeamId == "team-00");
                double others = result.Teams.Where(t => t.TeamId != "team-00").Average(t => (double)t.BankedGold);
                rel[(k, seed)] = (me.BankedGold - others, me.Place);
            });
            var fitness = new double[xs.Length];
            var places = new double[xs.Length];
            for (int k = 0; k < xs.Length; k++)
            {
                var mine = rel.Where(r => r.Key.Item1 == k).Select(r => r.Value).ToList();
                fitness[k] = mine.Average(r => r.Rel);
                places[k] = mine.Average(r => r.Place);
            }
            es.Tell(xs, fitness);
            var (cfg, w) = EsGenome.Decode(es.Mean, template);
            var model = new EsModel { Config = cfg, Weights = w, Optimizer = es };
            model.Metadata["source"] = "es-opp";
            model.Metadata["generation"] = es.Generation.ToString();
            model.Save(o.Output);
            Console.WriteLine($"[es-opp] gen {es.Generation} ({clock.Elapsed.TotalMinutes:F1} min) sigma {es.Sigma:F3} | fitness mean {fitness.Average():+0.0;-0.0} best {fitness.Max():+0.0;-0.0} | place mean {places.Average():F2} best {places.Min():F2} | " +
                $"Race {cfg.RaceDiscount:F2} Crowd {cfg.CrowdPenalty:F2} BehindCenter {cfg.BehindCenterBonus:F2} Commit {cfg.InvestCommit:F2} BaseSafety {cfg.BaseSafety:F2} CenterRush {cfg.CenterRushBonus:F2}/{cfg.CenterRushTurns:F0}");
        }
        Console.WriteLine($"[es-opp] done; latest mean in {o.Output}");
    }
}
