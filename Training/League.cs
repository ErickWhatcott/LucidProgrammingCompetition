using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using Lpc.Ml;
using Lpc.Ml.Es;
using Lpc.Strategies;
using Lpc.Strategies.Planner;

namespace Lpc.Training;

public sealed class LeagueOptions
{
    public double Minutes = 60;
    public int MaxRounds = 1000;
    public int GamesPerRound = 72;
    public int Parallel = Games.DefaultParallelism;
    public double Noise = 0.3;          // exploration noise on one of each value engine's two seats
    public int MlpEpochs = 15;
    public int Lambda = 12;             // CMA-ES population; each candidate gets GamesPerRound*4/Lambda seats
    public int MaxRows = 20_000_000;    // circular window of value-training rows (oldest dropped first)
    public int ExpertRounds = 3;        // imitation uses top finishes from the last N rounds
    public int ImitationRounds = 2;     // ppo: imitation only for the first N rounds, then PPO updates
    public double PpoTemperature = 0.5;  // same temperature for acting and for the PPO ratio
    public int ReplayRows = 300_000;
    public bool Resume;
    public int MaxTurns;
}

/// <summary>
/// Pretraining league: every round, all four engines play together in the same simulated matches
/// (plus Intermediate/Beginner/Novice practice bots). Every engine seat records into ONE shared pool,
/// and then every engine trains from it:
///   mlp, gbdt  - value regression on all seats' (chosen features, realized gold/turn) rows
///   ppo        - imitation of the top-placing seats of any engine, then PPO on its own sampled decisions
///                with a critic built from the shared mlp value model
///   es         - CMA-ES over planner settings + linear scorer weights; fitness = the candidates' seats'
///                banked gold relative to the match average (removes map-difficulty noise)
/// </summary>
public static class League
{
    sealed record LeagueState(Dictionary<string, double> BestRel, Dictionary<string, List<double>> BlendHistory, int PpoRounds);

    sealed record Seat(string Mode, int EsCandidate, bool Noisy, double? Blend = null)
    {
        public string Label => Blend is { } b ? $"{Mode}@{b:0.##}" : Mode;
    }

    static readonly double[] BlendVariants = [0.5, 0.25];

    // Non-ML opponents: two strong solver seats plus four seats rotating mostly through the solvers (with some
    // scripted styles and practice bots for variety), so fitness rewards beating good play, not farming weak bots.
    static readonly string[] Solvers = ["denial", "investor", "territory", "hungarian", "scout", "centerbank"];
    static readonly string[] Rotating = ["denial", "investor", "rusher", "territory", "hungarian", "expander", "scout", "centerbank",
        "intermediate", "denial", "investor", "blocker", "territory", "hungarian", "swarm", "scout", "centerbank", "beginner"];
    const int EsSeatsPerGame = 4;

    static List<Seat> Seats(int game, int lambda)
    {
        var seats = new List<Seat>();
        for (int j = 0; j < EsSeatsPerGame; j++) seats.Add(new Seat("es", (game * EsSeatsPerGame + j) % lambda, false));
        // Pure model seats lost clearly to 50/50 blends in the first rounds; now compare two blends.
        foreach (var engine in new[] { "mlp", "gbdt" })
            foreach (var blend in BlendVariants)
                seats.Add(new Seat(engine, -1, false, blend));
        seats.Add(new Seat("ppo", -1, false));
        seats.Add(new Seat("ppo", -1, false));
        for (int j = 0; j < 2; j++) seats.Add(new Seat(Solvers[(game * 2 + j) % Solvers.Length], -1, false));
        for (int j = 0; j < 4; j++) seats.Add(new Seat(Rotating[(game * 4 + j) % Rotating.Length], -1, false));
        return seats;
    }

    public static void Pretrain(LeagueOptions o)
    {
        var clock = Stopwatch.StartNew();
        Directory.CreateDirectory("models");
        Directory.CreateDirectory("logs");
        Directory.CreateDirectory(Paths.SimData);
        var baseConfig = StrategyFactory.LoadConfig(new StrategyOptions());
        Console.WriteLine($"[league] matmul backend {Blas.Backend}; {o.Parallel} parallel games; {o.GamesPerRound} games/round; budget {o.Minutes:F0} min");

        Mlp? mlp = o.Resume && File.Exists(Paths.Model("mlp")) ? Mlp.Load(Paths.Model("mlp")) : null;
        GbdtModel? gbdt = o.Resume && File.Exists(Paths.Model("gbdt")) ? GbdtModel.Load(Paths.Model("gbdt")) : null;
        PolicyNet? ppo = o.Resume && File.Exists(Paths.Model("ppo")) ? PolicyNet.Load(Paths.Model("ppo")) : null;
        var esSaved = o.Resume && File.Exists(Paths.Model("es")) ? EsModel.Load(Paths.Model("es")) : null;
        var cma = esSaved?.Optimizer ?? (esSaved != null
            ? new CmaEs(EsGenome.Encode(esSaved.Config, esSaved.Weights), 0.12, o.Lambda, 1) // genome grew: continue from the evolved point
            : new CmaEs(EsGenome.InitialMean(baseConfig), 0.2, o.Lambda, 1));
        // Planner settings for mlp/gbdt/ppo follow the CMA-ES mean: all engines share one planner, so
        // the investment/targeting knobs ES discovers help every engine (and live play via planner-config.json).
        var sharedConfig = esSaved?.Config ?? baseConfig;

        using var store = new SimStore();
        var window = o.Resume ? store.LoadRecent(o.MaxRows) : [];
        if (o.Resume && window.Count == 0 && File.Exists(WindowPath))
        {
            window = LoadWindow(WindowPath); // first run with SQLite: import the pre-SQLite pool so nothing is lost
            store.AddRound(store.NextRound(), window, []);
        }
        Console.WriteLine($"[league] self-play store {SimStore.DefaultPath}: {store.GameCount():N0} games, {store.TripCount():N0} rows; training window {window.Count:N0}");
        var decisionRounds = new Queue<List<DecisionRecord>>();
        const string statePath = "models/league-state.json";
        var bestRel = Engines.ToDictionary(e => e, _ => double.NegativeInfinity);
        var blendHistory = new Dictionary<string, List<double>>();
        int ppoRounds = 0;
        if (o.Resume && File.Exists(statePath))
        {
            var saved = System.Text.Json.JsonSerializer.Deserialize<LeagueState>(File.ReadAllText(statePath))!;
            foreach (var (k, v) in saved.BestRel) bestRel[k] = v;
            blendHistory = saved.BlendHistory;
            ppoRounds = saved.PpoRounds;
            Console.WriteLine($"[league] resumed: ppo rounds {ppoRounds}, best rounds {string.Join(", ", bestRel.Select(x => $"{x.Key} {x.Value:+0;-0}"))}");
        }

        for (int round = 0; round < o.MaxRounds && clock.Elapsed.TotalMinutes < o.Minutes; round++)
        {
            var roundClock = Stopwatch.StartNew();
            var xs = cma.Ask();
            var genomes = xs.Select(x => EsGenome.Decode(x, baseConfig)).ToArray();
            var trips = new TripBuffer();
            var decisions = new DecisionBuffer();
            var seatMap = new ConcurrentDictionary<(int Seed, int Seat), Seat>();
            int firstSeed = 100_000 + round * o.GamesPerRound;
            var (playedMlp, playedGbdt, playedPpo) = (mlp, gbdt, ppo);
            var roundConfig = sharedConfig;

            var results = Games.RunMany(Enumerable.Range(firstSeed, o.GamesPerRound), seed =>
            {
                var rng = new Random(seed);
                var specs = Seats(seed - firstSeed, xs.Length).OrderBy(_ => rng.Next()).ToList();
                var list = new List<IMarsStrategy>();
                for (int i = 0; i < specs.Count; i++)
                {
                    var spec = specs[i];
                    seatMap[(seed, i)] = spec;
                    bool ours = Engines.Contains(spec.Mode);
                    var opts = new StrategyOptions
                    {
                        Config = roundConfig, Mlp = playedMlp, Gbdt = playedGbdt, Policy = playedPpo, Temperature = o.PpoTemperature,
                        Recorder = ours ? trips : null, Decisions = ours ? decisions : null,
                        Noise = spec.Noisy ? o.Noise : 0, Seed = seed * 31 + i, Quiet = true, Blend = spec.Blend,
                        EsGenome = spec.Mode == "es" ? genomes[spec.EsCandidate] : null,
                    };
                    var bot = StrategyFactory.Create(spec.Mode, opts);
                    // Learn from everyone: shadow half of the non-ML seats (alternating) and record their play too.
                    if (!ours && (i + seed) % 2 == 0) bot = new ObservedBot(bot, trips, decisions);
                    list.Add(bot);
                }
                return list;
            }, o.Parallel, o.MaxTurns);
            double playSeconds = roundClock.Elapsed.TotalSeconds;

            // ---- Relative scores: seat gold minus the match's mean gold ----
            var rel = new Dictionary<string, List<double>>();
            var abs = new Dictionary<string, List<double>>();
            var byLabel = new Dictionary<string, List<double>>();
            var fitnessSamples = Enumerable.Range(0, xs.Length).Select(_ => new List<double>()).ToArray();
            foreach (var r in results)
            {
                double mean = r.Teams.Average(t => (double)t.BankedGold);
                foreach (var t in r.Teams)
                {
                    var spec = seatMap[(r.Seed, int.Parse(t.TeamId[5..], CultureInfo.InvariantCulture))];
                    double v = t.BankedGold - mean;
                    (rel.TryGetValue(spec.Mode, out var lr) ? lr : rel[spec.Mode] = []).Add(v);
                    (abs.TryGetValue(spec.Mode, out var la) ? la : abs[spec.Mode] = []).Add(t.BankedGold);
                    if (spec.Blend != null) (byLabel.TryGetValue(spec.Label, out var lb) ? lb : byLabel[spec.Label] = []).Add(v);
                    if (spec.EsCandidate >= 0) fitnessSamples[spec.EsCandidate].Add(v);
                }
            }
            var fitness = fitnessSamples.Select(s => s.Count > 0 ? s.Average() : -1e6).ToArray();
            cma.Tell(xs, fitness);

            // Keep the version of each engine that just played if it had its best round so far.
            void KeepBest(string engine, Action<string> save)
            {
                double score = rel.TryGetValue(engine, out var l) ? l.Average() : double.NegativeInfinity;
                if (score > bestRel[engine]) { bestRel[engine] = score; save($"models/{engine}.best.json"); }
            }
            if (playedMlp != null) KeepBest("mlp", p => Atomic(p, playedMlp.Save));
            if (playedGbdt != null) KeepBest("gbdt", p => Atomic(p, playedGbdt.Save));
            if (playedPpo != null) KeepBest("ppo", p => Atomic(p, playedPpo.Save));

            // ---- Train every engine from the shared pool ----
            var trainClock = Stopwatch.StartNew();
            var freshTrips = trips.Records;
            long globalRound = store.NextRound();
            store.AddRound(globalRound, freshTrips, results);
            // Disk rule (user): keep at least 10 GB free on the Mac (+1 GB for the WAL and window.bin).
            // Below that, drop the oldest 10% of rows each round; SQLite reuses the freed pages, so growth stops.
            long free = new DriveInfo(Path.GetPathRoot(Path.GetFullPath(SimStore.DefaultPath))!).AvailableFreeSpace;
            if (free < 11L * 1024 * 1024 * 1024)
            {
                long keep = (long)(store.TripCount() * 0.9);
                store.Prune(keep);
                Console.WriteLine($"[league] disk free {free / (1024.0 * 1024 * 1024):F1} GB: pruned self-play store to {keep:N0} rows");
            }
            window.AddRange(freshTrips);
            if (window.Count > o.MaxRows) window.RemoveRange(0, window.Count - o.MaxRows);
            var freshDecisions = decisions.Records;
            decisionRounds.Enqueue(freshDecisions);
            while (decisionRounds.Count > o.ExpertRounds) decisionRounds.Dequeue();

            // Blend chosen from the variant seats over the last 3 rounds (only once a model played).
            foreach (var (label, values) in byLabel)
            {
                if (!blendHistory.TryGetValue(label, out var h)) blendHistory[label] = h = [];
                if ((label.StartsWith("mlp") && playedMlp != null) || (label.StartsWith("gbdt") && playedGbdt != null)) h.Add(values.Average());
                if (h.Count > 3) h.RemoveAt(0);
            }
            string BlendFor(string engine)
            {
                double Score(double blend) => blendHistory.TryGetValue($"{engine}@{blend:0.##}", out var h) && h.Count > 0 ? h.Average() : double.NegativeInfinity;
                return BlendVariants.MaxBy(Score).ToString(CultureInfo.InvariantCulture);
            }

            // Scrimmage data streams in every round: real games are re-weighted to ~10% of the value-training
            // signal (capped at 20 copies each), so the engines adapt to real opponents while the simulator
            // keeps supplying volume. Strong real finishes also feed ppo's imitation, its own real games its PPO step.
            var live = LiveData.Refresh();
            int copies = live.Trips.Count == 0 ? 0 : (int)Math.Clamp(0.10 * window.Count / live.Trips.Count, 1, 20);
            var valueRows = copies == 0 ? window : [.. window, .. Enumerable.Repeat(live.Trips, copies).SelectMany(x => x)];
            var liveExpert = live.Decisions.Where(d => d.FinalPlace <= Math.Max(2, d.TeamCount / 4)).ToList();
            var liveOnPolicy = live.Decisions.Where(d => d.Model == "ppo" && float.IsFinite(d.BehaviorLogProb)).ToList();

            // Bigger windows need fewer passes; keeps the (GPU) mlp fit at a few seconds per round.
            int epochs = (int)Math.Clamp(o.MlpEpochs * 2_000_000.0 / Math.Max(1, valueRows.Count), 3, o.MlpEpochs);
            mlp = Trainer.Pretrain(valueRows, new TrainOptions { Epochs = epochs, Seed = round + 1 });
            Console.WriteLine($"    mlp trained on {mlp.Metadata["backend"]}: {mlp.Metadata["rowsPerSec"]} rows/s, {mlp.Metadata["epochs"]} epochs, bestValMse={mlp.Metadata["bestValMse"]}");
            mlp.Metadata["blend"] = BlendFor("mlp");
            Atomic(Paths.Model("mlp"), mlp.Save);
            // GBDT trains on the CPU and grows ~linearly with rows (40-55 s at 2M); a random 600k sample keeps
            // rounds short with little accuracy cost for 300 trees of 31 leaves. (The mlp trains on the GPU.)
            var gbdtRows = window.Count <= 600_000 ? [.. window] : Sample(window, 600_000, round);
            if (copies > 0) gbdtRows.AddRange(Enumerable.Repeat(live.Trips, Math.Max(1, copies * gbdtRows.Count / Math.Max(1, window.Count))).SelectMany(x => x));
            gbdt = GbdtModel.Fit(gbdtRows, new GbdtOptions { Seed = round + 1 });
            gbdt.Metadata["blend"] = BlendFor("gbdt");
            Atomic(Paths.Model("gbdt"), gbdt.Save);

            string ppoNote;
            if (ppo == null || ppoRounds < o.ImitationRounds)
            {
                // Imitate whoever finished near the top, whichever engine it was.
                var expert = decisionRounds.SelectMany(d => d).Where(d => d.FinalPlace <= Math.Max(2, d.TeamCount / 5)).Concat(liveExpert).Concat(liveExpert).ToList();
                ppo = PolicyTrainer.Imitate(expert, new ImitationOptions { Seed = round + 1, WeightFn = d => 1f + Math.Max(0, d.FinalGold) / 500f }, ppo);
                ppoNote = $"imitation on {expert.Count:N0} top-finish decisions";
            }
            else
            {
                // Pure PPO drifted downward (+99 -> +41 over 4 rounds), so each round first distills the
                // current top finishers (whatever engine they are), then takes a PPO step on its own play.
                var expert = freshDecisions.Where(d => d.FinalPlace <= Math.Max(2, d.TeamCount / 5)).Concat(liveExpert).Concat(liveExpert).ToList();
                ppo = PolicyTrainer.Imitate(expert, new ImitationOptions { Epochs = 2, Seed = round + 1, WeightFn = d => 1f + Math.Max(0, d.FinalGold) / 500f }, ppo);
                var onPolicy = freshDecisions.Where(d => d.Model == "ppo" && float.IsFinite(d.BehaviorLogProb)).Concat(liveOnPolicy).ToList();
                var baselines = Baselines(ppo, mlp, onPolicy, o.PpoTemperature);
                var stats = PolicyTrainer.PpoUpdate(ppo, onPolicy, d => baselines[d],
                    new PpoOptions { Seed = round + 1, Temperature = o.PpoTemperature });
                ppoNote = $"distill {expert.Count:N0} top-finish decisions + PPO on {onPolicy.Count:N0} own: kl {stats.ApproxKl:F4} clip {stats.ClipFraction:F3} entropy {stats.Entropy:F3}";
            }
            ppoRounds++;
            Atomic(Paths.Model("ppo"), ppo.Save);

            var (esConfig, esWeights) = EsGenome.Decode(cma.Mean, baseConfig);
            sharedConfig = esConfig;
            Atomic(Paths.PlannerConfig, esConfig.Save);
            var esModel = new EsModel { Config = esConfig, Weights = esWeights, Optimizer = cma };
            esModel.Metadata["generation"] = cma.Generation.ToString(CultureInfo.InvariantCulture);
            esModel.Metadata["bestFitness"] = cma.BestFitness.ToString("F2", CultureInfo.InvariantCulture);
            Atomic(Paths.Model("es"), esModel.Save);
            File.WriteAllText(statePath, System.Text.Json.JsonSerializer.Serialize(new LeagueState(
                bestRel.Where(x => double.IsFinite(x.Value)).ToDictionary(), blendHistory, ppoRounds)));
            double trainSeconds = trainClock.Elapsed.TotalSeconds;

            // ---- Report ----
            var line = string.Join(" | ", Engines.Concat(StrategyFactory.PracticeBots).Where(rel.ContainsKey).Select(e =>
            {
                var (m, se) = Games.MeanSe(rel[e]);
                return $"{e} {abs[e].Average():F0}kg ({m:+0;-0}±{se:F0})";
            }));
            Console.WriteLine($"[league] round {round} ({clock.Elapsed.TotalMinutes:F1} min): play {playSeconds:F0}s train {trainSeconds:F0}s, "
                + $"+{freshTrips.Count:N0} rows (window {window.Count:N0}), +{freshDecisions.Count:N0} decisions");
            if (globalRound % 5 == 0)
                Console.WriteLine("         last 5 rounds (" + SimStore.DefaultPath + "): "
                    + string.Join(" | ", store.Leaderboard(5).Take(6).Select(x => $"{x.Bot} {x.Gold:F0}kg p{x.Place:F1}")));
            if (live.Trips.Count + live.Decisions.Count > 0)
                Console.WriteLine($"         scrimmage data: {live.Trips.Count:N0} value rows (x{copies}), {live.Decisions.Count:N0} decisions ({liveExpert.Count:N0} strong finishes, {liveOnPolicy.Count:N0} ppo) from {live.Files} files");
            Console.WriteLine($"         gold (vs match mean): {line}");
            Console.WriteLine($"         value-engine variants: {string.Join(" | ", byLabel.OrderBy(x => x.Key).Select(x => $"{x.Key} {x.Value.Average():+0;-0}"))}"
                + $" -> blend mlp {mlp.Metadata["blend"]}, gbdt {gbdt.Metadata["blend"]}");
            Console.WriteLine($"         es gen {cma.Generation} sigma {cma.Sigma:F3} best fitness {fitness.Max():+0.0;-0.0} | ppo: {ppoNote}");

            if (round % 5 == 4 || clock.Elapsed.TotalMinutes >= o.Minutes) SaveReplay(window, decisionRounds, o.ReplayRows, round);

            // Hot-swap support: a deploy drops this file; we stop after a complete round and the
            // wrapper loop restarts on the new build with --resume.
            if (File.Exists(StopFile))
            {
                File.Delete(StopFile);
                Console.WriteLine("[league] stop requested; exiting after a complete round");
                break;
            }
        }
        SaveReplay(window, decisionRounds, o.ReplayRows, -1);
        foreach (var e in Engines)
            if (File.Exists(Paths.Model(e))) File.Copy(Paths.Model(e), Paths.Pretrained(e), overwrite: true);
        Console.WriteLine($"[league] done in {clock.Elapsed.TotalMinutes:F1} min. Latest models in models/<engine>.json, "
            + "best-round snapshots in models/<engine>.best.json. Compare with `lpc bench`.");
    }

    static readonly string[] Engines = StrategyFactory.Engines;
    public const string StopFile = "logs/league.stop";

    /// <summary>Write to a temp file, then rename over the target (atomic on the same volume).</summary>
    static void Atomic(string path, Action<string> save)
    {
        var tmp = path + ".tmp";
        save(tmp);
        File.Move(tmp, path, overwrite: true);
    }
    const string WindowPath = "data/sim/window.bin";

    /// <summary>Training window on disk so restarts (new builds) keep the accumulated simulator data.</summary>
    static void SaveWindow(string path, List<TripRecord> window)
    {
        var tmp = path + ".tmp";
        using (var w = new BinaryWriter(File.Create(tmp)))
        {
            w.Write(Features.Count);
            w.Write(window.Count);
            foreach (var r in window)
            {
                foreach (var f in r.F) w.Write(f);
                w.Write(r.Y);
            }
        }
        File.Move(tmp, path, overwrite: true);
    }

    static List<TripRecord> LoadWindow(string path)
    {
        var list = new List<TripRecord>();
        if (!File.Exists(path)) return list;
        try
        {
            using var r = new BinaryReader(File.OpenRead(path));
            if (r.ReadInt32() != Features.Count) return list;
            int n = r.ReadInt32();
            list.Capacity = n;
            for (int i = 0; i < n; i++)
            {
                var f = new float[Features.Count];
                for (int j = 0; j < f.Length; j++) f[j] = r.ReadSingle();
                list.Add(new TripRecord(f, r.ReadSingle(), "sim"));
            }
        }
        catch (EndOfStreamException) { }
        return list;
    }

    static List<TripRecord> Sample(List<TripRecord> rows, int count, int seed)
    {
        var rng = new Random(seed);
        var picked = new List<TripRecord>(count);
        for (int i = 0; i < rows.Count; i++) // reservoir-free selection sampling: exact count, one pass, order kept
            if (rng.Next(rows.Count - i) < count - picked.Count) picked.Add(rows[i]);
        return picked;
    }

    /// <summary>Baselines are computed once, before the update changes the policy.</summary>
    static Dictionary<DecisionRecord, float> Baselines(PolicyNet policy, Mlp critic, List<DecisionRecord> records, double temperature)
    {
        var values = new float[records.Count];
        Parallel.For(0, records.Count, i => values[i] = StateValue(policy, critic, records[i], temperature));
        var map = new Dictionary<DecisionRecord, float>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < records.Count; i++) map[records[i]] = values[i];
        return map;
    }

    /// <summary>Critic for PPO: V(s) = sum_a pi(a|s) * Q(s,a), with Q from the shared mlp value model.</summary>
    static float StateValue(PolicyNet policy, Mlp critic, DecisionRecord d, double temperature)
    {
        var logits = new float[d.Rows];
        var q = new float[d.Rows];
        policy.Logits(d.Candidates, d.Rows, logits);
        critic.Predict(d.Candidates, d.Rows, q);
        double max = logits.Max(), sum = 0, v = 0;
        for (int i = 0; i < d.Rows; i++)
        {
            double p = Math.Exp((logits[i] - max) / temperature);
            sum += p;
            v += p * q[i];
        }
        return (float)(v / sum);
    }

    static void SaveReplay(List<TripRecord> window, Queue<List<DecisionRecord>> decisionRounds, int rows, int seed)
    {
        var rng = new Random(seed);
        var trips = window.Count <= rows ? window : window.OrderBy(_ => rng.Next()).Take(rows).ToList();
        var tripPath = Path.Combine(Paths.SimData, "replay.jsonl");
        var decisionPath = Path.Combine(Paths.SimData, "decisions-replay.jsonl");
        foreach (var p in new[] { tripPath, decisionPath }) if (File.Exists(p)) File.Delete(p);
        TrainingData.AppendJsonl(tripPath, trips);
        var expert = decisionRounds.SelectMany(d => d).Where(d => d.FinalPlace <= Math.Max(2, d.TeamCount / 5)).ToList();
        DecisionData.AppendJsonl(decisionPath, expert.Count <= 100_000 ? expert : expert.OrderBy(_ => rng.Next()).Take(100_000));
    }

    /// <summary>Mean banked gold of seat 0 over paired seeds against an opponent mix.</summary>
    public static double Evaluate(string mode, StrategyOptions options, IReadOnlyList<int> seeds, string opponents, int parallel)
    {
        var results = Games.RunMany(seeds, seed =>
        {
            var o = Games.Clone(options);
            o.Seed = seed;
            o.Quiet = true;
            return [StrategyFactory.Create(mode, o), .. Games.Opponents(opponents, 15, seed, new StrategyOptions { Quiet = true })];
        }, parallel);
        return results.Average(r => (double)Games.Seat(r, 0).BankedGold);
    }

    /// <summary>
    /// Fine-tuning on recorded scrimmages (real opponents). mlp/gbdt: value rows from live play mixed
    /// with simulator replay. ppo: PPO on its own live decisions plus imitation of strong live finishes.
    /// es: unchanged (CMA-ES needs many games per generation; use more league rounds instead).
    /// Starts from the latest league model and writes models/<engine>.live.json (live play prefers it; the
    /// league never overwrites it). Each engine is checked in the simulator first; the previous live model is kept.
    /// </summary>
    public static void Finetune(string liveDir, int evalSeeds, int parallel)
    {
        var liveTrips = Directory.Exists(liveDir)
            ? Directory.EnumerateFiles(liveDir, "trips*.jsonl", SearchOption.AllDirectories).SelectMany(TrainingData.ReadJsonl).Where(r => r.Src == "live").ToList()
            : [];
        var liveDecisions = DecisionData.ReadAll(liveDir).Where(d => d.Src == "live").ToList();
        if (liveTrips.Count == 0 && liveDecisions.Count == 0)
        {
            Console.Error.WriteLine($"[finetune] no live data under {liveDir}. Record scrimmages with `lpc run --mode <engine> --record`.");
            return;
        }
        var replayPath = Path.Combine(Paths.SimData, "replay.jsonl");
        var replay = File.Exists(replayPath) ? TrainingData.ReadJsonl(replayPath) : [];
        Console.WriteLine($"[finetune] live: {liveTrips.Count:N0} value rows, {liveDecisions.Count:N0} decisions; sim replay {replay.Count:N0} rows");
        var seeds = Enumerable.Range(900_000, evalSeeds).ToList();

        void Commit(string engine, Action<string> save, StrategyOptions before, StrategyOptions after)
        {
            double b = Evaluate(engine, before, seeds, "mixed", parallel), a = Evaluate(engine, after, seeds, "mixed", parallel);
            if (a < b * 0.95)
            {
                // Too little live data can drag a model below what it does in the simulator; don't ship that.
                save($"models/{engine}.live.rejected.json");
                Console.WriteLine($"[finetune] {engine}: simulator check {b:F1} -> {a:F1} kg (more than 5% worse); NOT used for live play "
                    + $"(kept as models/{engine}.live.rejected.json). Record more scrimmages and run finetune again.");
                return;
            }
            if (File.Exists(Paths.Live(engine))) File.Copy(Paths.Live(engine), $"models/{engine}.live.prev.json", overwrite: true);
            save(Paths.Live(engine));
            Console.WriteLine($"[finetune] {engine}: simulator check {b:F1} -> {a:F1} kg; saved {Paths.Live(engine)} (live play uses it; previous in models/{engine}.live.prev.json)"
                + (a < b * 0.9 ? " WARNING: >10% drop in the simulator; confirm in scrimmages before keeping." : ""));
        }

        Mlp? mlp = File.Exists(Paths.Model("mlp")) ? Mlp.Load(Paths.Model("mlp")) : null;
        if (mlp != null && liveTrips.Count > 0)
        {
            var tuned = Trainer.Finetune(mlp, liveTrips, replay, new TrainOptions { Log = s => Console.WriteLine("    " + s) });
            Commit("mlp", tuned.Save, new StrategyOptions { Mlp = mlp }, new StrategyOptions { Mlp = tuned });
            mlp = tuned;
        }

        var gbdtPath = Paths.Model("gbdt");
        if (File.Exists(gbdtPath) && liveTrips.Count > 0)
        {
            var before = GbdtModel.Load(gbdtPath);
            // Live rows count double; sim replay keeps ~30% of the mix so the trees don't forget the basics.
            var rng = new Random(1);
            int simRows = Math.Min(replay.Count, (int)(liveTrips.Count * 2 * 0.3 / 0.7) + 1);
            var mix = liveTrips.Concat(liveTrips).Concat(replay.OrderBy(_ => rng.Next()).Take(simRows)).ToList();
            var tuned = GbdtModel.Fit(mix, new GbdtOptions { Log = s => Console.WriteLine("    " + s) });
            Commit("gbdt", tuned.Save, new StrategyOptions { Gbdt = before }, new StrategyOptions { Gbdt = tuned });
        }

        var ppoPath = Paths.Model("ppo");
        if (File.Exists(ppoPath) && liveDecisions.Count > 0)
        {
            var before = PolicyNet.Load(ppoPath);
            var tuned = before.Clone();
            var strong = liveDecisions.Where(d => d.FinalPlace <= Math.Max(2, d.TeamCount / 4)).ToList();
            if (strong.Count > 0)
                tuned = PolicyTrainer.Imitate(strong, new ImitationOptions { Epochs = 3, LearningRate = 3e-4f, Log = s => Console.WriteLine("    " + s) }, tuned);
            var onPolicy = liveDecisions.Where(d => d.Model == "ppo" && float.IsFinite(d.BehaviorLogProb)).ToList();
            if (onPolicy.Count > 0 && mlp != null)
            {
                var baselines = Baselines(tuned, mlp, onPolicy, 1.0);
                var stats = PolicyTrainer.PpoUpdate(tuned, onPolicy, d => baselines[d], new PpoOptions());
                Console.WriteLine($"    ppo update on {onPolicy.Count:N0} live decisions: {stats}");
            }
            Commit("ppo", tuned.Save, new StrategyOptions { Policy = before }, new StrategyOptions { Policy = tuned });
        }
        Console.WriteLine("[finetune] es is not changed by scrimmages (CMA-ES needs many games per generation).");
    }
}
