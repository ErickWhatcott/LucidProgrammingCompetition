using System.Collections.Concurrent;
using System.IO.Compression;
using Lpc.Client;
using Lpc.Ml;
using Lpc.Ml.Es;
using Lpc.Sim;
using Lpc.Strategies;
using Lpc.Strategies.Planner;
using Lpc.Training;

namespace Lpc.Cli;

public static class Commands
{
    const string Help = """
        lpc - Lucid Programming Competition 2026 bot (C#)

        Engines (all share one planner; each scores targets differently):
          mlp   neural value scorer (28-64-64-1) trained on the shared pool
          gbdt  gradient-boosted trees value scorer (LightGBM-style, pure C#)
          ppo   policy: imitation of top finishers, then PPO on its own play
          es    CMA-ES evolved planner settings + linear scorer
        Simulator opponents: intermediate, beginner, novice.

        Commands:
          run    --mode mlp|gbdt|ppo|es [--record] [--model FILE] [--temperature 0.5] [--key-file api.key]
                 Play the current live match. The team key is read from api.key (working directory, or
                 upward from the binary); server defaults to https://lpc.sh (override: LUCID_API_BASE_URL).
                 --record saves scrimmage data to data/live/ for `train finetune`, plus raw states.
          whoami [--key-file api.key]
                 Check the API key (authenticates only; does not join a match).
          sim    [--mode mlp] [--seed 1] [--opponents mixed] [--teams a,b,c] [--turns 480] [--render] [--trace]
          bench  [--modes mlp,gbdt,ppo,es] [--seeds 44] [--opponents zoo] [--h2h] [--parallel N]
                 Paired-seed comparison: mean banked gold, placement and decision latency.
          train pretrain [--minutes 60] [--games 36] [--lambda 12] [--resume]
                 Self-play league in the simulator: all engines play the same games, all record into one
                 shared pool, every engine trains from it each round -> models/<engine>.json
          train finetune [--live data/live]
                 Fine-tune mlp/gbdt/ppo on recorded scrimmages (starts from the pretrained models).
          latency [--games 11]
                 Per-engine inference and full-turn decision time against the 100 ms budget.

        Opponent mixes: zoo (all non-ML styles), mixed (practice-like), strong (adds our engines), or a comma list.
        Non-ML styles: intermediate, beginner, novice, rusher, expander, swarm, camper, blocker, homebody, ironlord, random.
        """;

    public static async Task<int> Main(string[] argv)
    {
        if (argv.Length == 0 || argv[0] is "help" or "--help" or "-h") { Console.WriteLine(Help); return 0; }
        var args = new Args(argv.Skip(1));
        if (args.Get("backend") is { } backend) Blas.Force(backend);
        if (args.Get("train-device") is { } trainDevice) Trainer.TrainDevice = trainDevice;
        try
        {
            switch (argv[0])
            {
                case "run": return await RunLive(args);
                case "whoami":
                    var me = await new BotClient(args.Get("key-file")).Authenticate();
                    Console.WriteLine($"API key OK: team {me.DisplayName} ({me.TeamId})");
                    return 0;
                case "sim": return Sim(args);
                case "bench": return Bench(args);
                case "train": return Train(args);
                case "latency": return Latency(args);
                case "curve": return Curve(args);
                default:
                    Console.Error.WriteLine($"Unknown command '{argv[0]}'.\n");
                    Console.WriteLine(Help);
                    return 2;
            }
        }
        catch (Exception error) when (error is ArgumentException or FileNotFoundException or InvalidOperationException or InvalidDataException)
        {
            Console.Error.WriteLine(error.Message);
            return 1;
        }
    }

    static StrategyOptions Options(Args args) => new()
    {
        ConfigPath = args.Get("config"),
        ModelPath = args.Get("model"),
        Temperature = args.Double("temperature", 0.5),
    };

    /// <summary>
    /// Plays match after match until Ctrl+C, until another bot takes over the team's session, or until
    /// a new build of this program appears (checked between matches; exit code 3 so a wrapper can relaunch).
    /// Models are reloaded from disk for every match, so newly trained models are picked up automatically.
    /// </summary>
    static async Task<int> RunLive(Args args)
    {
        // Engines rotate match by match (default: all four), so scrimmage data covers every engine.
        // Live play is latency-bound: prefer short GC pauses over throughput (training uses server GC).
        System.Runtime.GCSettings.LatencyMode = System.Runtime.GCLatencyMode.SustainedLowLatency;
        var rotation = args.Get("mode", "es,ppo,gbdt,mlp").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        foreach (var m in rotation)
            if (!StrategyFactory.Modes.Contains(m)) throw new ArgumentException($"Unknown mode '{m}'. Modes: {string.Join(", ", StrategyFactory.Modes)}");
        if (rotation.Length > 1) Console.WriteLine($"Rotating engines each match: {string.Join(" -> ", rotation)}");
        string binary = typeof(Commands).Assembly.Location;
        DateTime build = File.GetLastWriteTimeUtc(binary);
        bool NewBuild() => File.GetLastWriteTimeUtc(binary) != build;
        string? lastMatch = null;
        int played = 0;
        // Continue the rotation where earlier runs left off (one CSV row per finished match).
        int offset = File.Exists("logs/scrimmages.csv") ? Math.Max(0, File.ReadLines("logs/scrimmages.csv").Count() - 1) : 0;
        Console.CancelKeyPress += (_, _) => Console.WriteLine($"\nStopping after {played} match(es).");
        for (;;)
        {
            if (NewBuild()) { Console.WriteLine("New build detected; stopping so the ./lpc launcher relaunches on it."); return 3; }
            string mode = rotation[(offset + played) % rotation.Length];
            try
            {
                Console.WriteLine($"Engine for the next match: {mode}");
                var match = await PlayOneMatch(args, mode, lastMatch, NewBuild);
                if (match == null) continue; // stopped waiting (new build): handled at the top of the loop
                lastMatch = match;
                played++;
                if (args.Has("once")) return 0;
                Console.WriteLine($"--- {played} match(es) played; looking for the next one (Ctrl+C to stop) ---");
            }
            catch (BotClient.ApiError error) when (error.SessionConflict)
            {
                Console.WriteLine("Another bot took over this team's session; stopping.");
                return 4;
            }
            catch (BotClient.ApiError error) when (error.Status is 401 or 403 && !error.InactiveMatch)
            {
                Console.Error.WriteLine(error.Message);
                return 1;
            }
            catch (Exception error) when (error is not OutOfMemoryException)
            {
                Console.Error.WriteLine($"{error.Message} Retrying in 5 s.");
                await Task.Delay(5000);
            }
        }
    }

    /// <summary>
    /// Runs a throwaway copy of the engine on a simulated board before joining, so JIT compilation and
    /// model warm-up happen now rather than on the first real-time turns.
    /// </summary>
    static void Warmup(string mode, StrategyOptions options)
    {
        var o = Games.Clone(options);
        o.Recorder = null;
        o.Decisions = null;
        o.Quiet = true;
        var sim = new Simulator(new SimConfig { Seed = 12345 }, ["warmup", "other"]);
        var bot = StrategyFactory.Create(mode, o);
        var identity = new Identity(sim.TeamIds[0], "warmup");
        for (int i = 0; i < 25; i++)
        {
            bot.CalculateActions(sim.ViewFor(0), identity);
            sim.Step([null, null]);
        }
    }

    static async Task<string?> PlayOneMatch(Args args, string mode, string? skipMatch, Func<bool> stopWaiting)
    {
        var options = Options(args);
        options.PreferLive = true;
        JsonlTripRecorder? trips = null;
        JsonlDecisionRecorder? decisions = null;
        StreamWriter? states = null;
        string stamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss");
        if (args.Has("record"))
        {
            Directory.CreateDirectory(Paths.LiveData);
            trips = new JsonlTripRecorder(Path.Combine(Paths.LiveData, $"trips-{stamp}.jsonl"));
            decisions = new JsonlDecisionRecorder(Path.Combine(Paths.LiveData, $"decisions-{stamp}.jsonl"));
            options.Recorder = trips;
            options.Decisions = decisions;
            options.RecordSource = "live";
            states = new StreamWriter(new GZipStream(File.Create(Path.Combine(Paths.LiveData, $"states-{stamp}.jsonl.gz")), CompressionLevel.Fastest));
            Console.WriteLine($"Recording scrimmage to {Paths.LiveData}/*-{stamp}.*");
        }
        Warmup(mode, options);
        var guard = new DeadlineGuard(StrategyFactory.Create(mode, options), new BeginnerBot(options.Seed));
        var strategy = new TimedStrategy(guard);
        var client = new BotClient(args.Get("key-file")) { SkipMatchId = skipMatch, StopWaiting = stopWaiting, UseTurnStream = !args.Has("no-stream") };
        if (states != null) client.StateObserver = s => states.WriteLine(s.ToJsonString());
        Identity? identity = null;
        try
        {
            await client.RunMars((state, id) => { identity = id; return strategy.CalculateActions(state, id); });
        }
        finally
        {
            if (identity != null) strategy.OnMatchEnd(null, identity);
            trips?.Dispose();
            decisions?.Dispose();
            states?.Dispose();
            PrintLatency(mode, strategy.SamplesMs);
            if (guard.Fallbacks > 0) Console.WriteLine($"[{mode}] deadline guard used the fallback move on {guard.Fallbacks} turn(s)");
            Console.WriteLine($"[{mode}] turns accepted late by the server: {client.LateTurns}");
            if (guard.Main is PlannerStrategy planner && planner.Turns > 0)
            {
                Console.WriteLine(planner.Stats);
                // Only completed matches count: replaced or cancelled practice matches would log bogus results.
                if (planner.FinalStanding is { } f && client.LastMatchId != null && client.FinalStatus == "completed")
                {
                    Console.WriteLine($"RESULT {mode}: {f.Gold} kg banked, place {f.Place} of {f.Teams}");
                    Directory.CreateDirectory("logs");
                    const string csv = "logs/scrimmages.csv";
                    if (!File.Exists(csv)) File.WriteAllText(csv, "utc,match,engine,gold,place,teams,maxTurnMs\n");
                    File.AppendAllText(csv, $"{DateTime.UtcNow:O},{client.LastMatchId},{mode},{f.Gold},{f.Place},{f.Teams},{planner.MaxTurnMs:F1}\n");
                }
            }
        }
        return client.LastMatchId;
    }

    static int Sim(Args args)
    {
        int seed = args.Int("seed", 1);
        var options = Options(args);
        options.Seed = seed;
        List<IMarsStrategy> roster;
        if (args.Get("teams") is { } teams)
            roster = teams.Split(',', StringSplitOptions.TrimEntries).Select((m, i) =>
            {
                var o = Games.Clone(options);
                o.Seed = seed * 101 + i;
                return StrategyFactory.Create(m, o);
            }).ToList();
        else
            roster = [StrategyFactory.Create(args.Get("mode", "mlp"), options),
                .. Games.Opponents(args.Get("opponents", "mixed"), args.Int("opponents-count", 15), seed, new StrategyOptions { Quiet = true })];
        if (args.Has("trace") && roster[0] is PlannerStrategy traced) traced.Trace = true;
        var timed = roster.Select(s => new TimedStrategy(s)).ToList();
        var config = new SimConfig { Seed = seed };
        if (args.Int("turns", 0) is > 0 and var turns) config.MaxTurns = turns;
        Simulator? last = null;
        var result = MatchRunner.Run(config, timed, s => last = s);
        Console.WriteLine($"Seed {seed}: {result.ElapsedMs / 1000:F2}s");
        Console.WriteLine($"{"place",5} {"team",-8} {"bot",-13} {"gold",6} {"iron",5} {"cGold",6} {"bases",5} {"workers",7} {"maxMs",7}");
        foreach (var t in result.Teams.OrderBy(t => t.Place))
        {
            int seat = int.Parse(t.TeamId[5..]);
            double maxMs = timed[seat].SamplesMs.Count > 0 ? timed[seat].SamplesMs.Max() : 0;
            Console.WriteLine($"{t.Place,5} {t.TeamId,-8} {t.Name,-13} {t.BankedGold,6} {t.BankedIron,5} {t.CarriedGold,6} {t.Bases,5} {t.Workers,7} {maxMs,7:F2}");
        }
        if (args.Has("render") && last != null) Console.WriteLine(last.RenderAscii());
        foreach (var s in roster.OfType<PlannerStrategy>()) if (s.ScorerFallbacks > 0) Console.WriteLine(s.Stats);
        return 0;
    }

    /// <summary>Loads each engine's model once so parallel games share it read-only.</summary>
    static StrategyOptions Preload(StrategyOptions o, IEnumerable<string> modes)
    {
        foreach (var m in modes.Distinct())
        {
            var path = o.ModelPath ?? Paths.Model(m);
            if (!File.Exists(path)) continue;
            switch (m)
            {
                case "mlp": o.Mlp = Mlp.Load(path); break;
                case "gbdt": o.Gbdt = GbdtModel.Load(path); break;
                case "ppo": o.Policy = PolicyNet.Load(path); break;
                case "es": o.Es = EsModel.Load(path); break;
            }
        }
        return o;
    }

    static int Bench(Args args)
    {
        var modes = args.Get("modes", string.Join(",", StrategyFactory.Engines)).Split(',', StringSplitOptions.TrimEntries);
        int seedCount = args.Int("seeds", 44), parallel = args.Int("parallel", Games.DefaultParallelism);
        string opponents = args.Get("opponents", "zoo");
        var seeds = Enumerable.Range(args.Int("seed-start", 900_000), seedCount).ToList();
        var options = Options(args);
        options.Quiet = true;
        foreach (var m in modes.Where(m => StrategyFactory.Engines.Contains(m) && !File.Exists(options.ModelPath ?? Paths.Model(m))))
            Console.Error.WriteLine($"[bench] no model for {m}; it will play with the heuristic scorer.");
        Preload(options, modes);
        var opponentOptions = Preload(new StrategyOptions { Quiet = true }, Games.Mix(opponents));

        var gold = modes.ToDictionary(m => m, _ => new ConcurrentDictionary<int, (long Gold, int Place)>());
        var latency = modes.ToDictionary(m => m, _ => new ConcurrentBag<double>());
        var clock = System.Diagnostics.Stopwatch.StartNew();
        bool h2h = args.Has("h2h");
        if (h2h)
        {
            // All modes share every match, filled up with opponents.
            Games.RunMany(seeds, seed =>
            {
                var seats = modes.Select((m, i) => (IMarsStrategy)new TimedStrategy(Create(m, options, seed * 7 + i))).ToList();
                return [.. seats, .. Games.Opponents(opponents, 16 - seats.Count, seed, opponentOptions)];
            }, parallel, done: r => Collect(r, modes, gold, h2h: true));
        }
        else
        {
            foreach (var mode in modes)
                Games.RunMany(seeds, seed =>
                    [new LatencySink(new TimedStrategy(Create(mode, options, seed)), latency[mode]), .. Games.Opponents(opponents, 15, seed, opponentOptions)],
                    parallel, done: r => Collect(r, [mode], gold, h2h: false));
        }

        Console.WriteLine($"\n{seedCount} seeds vs '{opponents}'{(h2h ? " (head-to-head)" : " (paired, seat 0)")}, {clock.Elapsed.TotalSeconds:F0}s");
        Console.WriteLine($"{"mode",-13} {"gold",8} {"±se",6} {"place",6} {"wins%",6} {"turn ms p50",11} {"p99",7} {"max",7}");
        foreach (var m in modes)
        {
            var (mean, se) = Games.MeanSe(gold[m].Values.Select(v => (double)v.Gold).ToList());
            double place = gold[m].Values.Average(v => v.Place);
            double wins = 100.0 * gold[m].Values.Count(v => v.Place == 1) / gold[m].Count;
            var lat = latency[m].OrderBy(x => x).ToList();
            Console.WriteLine($"{m,-13} {mean,8:F1} {se,6:F1} {place,6:F2} {wins,6:F1} {Pct(lat, 0.5),11:F2} {Pct(lat, 0.99),7:F2} {(lat.Count > 0 ? lat[^1] : 0),7:F2}");
        }
        for (int a = 0; a < modes.Length; a++)
        for (int b = a + 1; b < modes.Length; b++)
        {
            var A = gold[modes[a]]; var B = gold[modes[b]];
            var common = A.Keys.Intersect(B.Keys).ToList();
            int better = common.Count(s => A[s].Gold > B[s].Gold), worse = common.Count(s => A[s].Gold < B[s].Gold);
            var (d, dse) = Games.MeanSe(common.Select(s => (double)(A[s].Gold - B[s].Gold)).ToList());
            Console.WriteLine($"paired {modes[a]} - {modes[b]}: {d:+0.0;-0.0} ± {dse:F1} kg/match, {modes[a]} better on {better}/{common.Count} seeds, worse on {worse}");
        }
        return 0;
    }

    static IMarsStrategy Create(string mode, StrategyOptions shared, int seed)
    {
        var o = Games.Clone(shared);
        o.Seed = seed;
        return StrategyFactory.Create(mode, o);
    }

    static void Collect(SimResult r, IReadOnlyList<string> modes, Dictionary<string, ConcurrentDictionary<int, (long, int)>> gold, bool h2h)
    {
        // Seat 0 in paired mode; in head-to-head, seat i holds modes[i].
        for (int i = 0; i < modes.Count; i++)
        {
            var t = Games.Seat(r, h2h ? i : 0);
            gold[modes[i]][r.Seed] = (t.BankedGold, t.Place);
        }
    }

    static double Pct(List<double> sorted, double p) => sorted.Count == 0 ? 0 : sorted[Math.Min(sorted.Count - 1, (int)(p * sorted.Count))];

    static void PrintLatency(string name, IReadOnlyList<double> samples)
    {
        if (samples.Count == 0) return;
        var s = samples.OrderBy(x => x).ToList();
        Console.WriteLine($"[{name}] decision time over {s.Count} turns: p50 {Pct(s, 0.5):F2} ms, p99 {Pct(s, 0.99):F2} ms, max {s[^1]:F2} ms");
    }

    static int Train(Args args)
    {
        string sub = args.Positional.FirstOrDefault() ?? "pretrain";
        int parallel = args.Int("parallel", Games.DefaultParallelism);
        switch (sub)
        {
            case "pretrain":
                League.Pretrain(new LeagueOptions
                {
                    Minutes = args.Double("minutes", 60),
                    MaxRounds = args.Int("rounds", 1000),
                    GamesPerRound = args.Int("games", 72),
                    MaxRows = args.Int("max-rows", 20_000_000),
                    Lambda = args.Int("lambda", 12),
                    Parallel = parallel,
                    Noise = args.Double("noise", 0.3),
                    MlpEpochs = args.Int("epochs", 15),
                    Resume = args.Has("resume"),
                    MaxTurns = args.Int("turns", 0),
                });
                return 0;
            case "es-opp":
                EsVsStrong.Run(new EsVsStrongOptions
                {
                    Start = args.Get("model", "models/es-hailmary.json"),
                    Output = args.Get("out", "models/es-opp.json"),
                    Minutes = args.Double("minutes", 60),
                    SeedsPerGeneration = args.Int("games", 8),
                    Lambda = args.Int("lambda", 12),
                    Sigma = args.Double("sigma", 0.07),
                    Parallel = parallel,
                });
                return 0;
            case "finetune":
                League.Finetune(args.Get("live", Paths.LiveData), args.Int("eval-seeds", 11), parallel);
                return 0;
            default:
                Console.Error.WriteLine("Usage: lpc train pretrain|finetune [options]");
                return 2;
        }
    }

    /// <summary>
    /// Game-phase diagnostics: mean banked gold / workers / bases per bot type at checkpoints through the
    /// match, from head-to-head games (all engines + non-ML opponents in the same matches).
    /// </summary>
    static int Curve(Args args)
    {
        int[] checkpoints = [10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400, 480];
        var modes = args.Get("modes", string.Join(",", StrategyFactory.Engines)).Split(',', StringSplitOptions.TrimEntries);
        var options = Preload(new StrategyOptions { Quiet = true }, modes);
        var opponentOptions = Preload(new StrategyOptions { Quiet = true }, Games.Mix(args.Get("opponents", "zoo")));
        var samples = new ConcurrentBag<(string Name, int Turn, long Gold, int Workers, int Bases)>();
        Games.RunMany(Enumerable.Range(args.Int("seed-start", 800_000), args.Int("seeds", 22)), seed =>
        {
            var seats = modes.Select((m, i) => Create(m, options, seed * 7 + i)).ToList();
            return [.. seats, .. Games.Opponents(args.Get("opponents", "zoo"), 16 - seats.Count, seed, opponentOptions)];
        }, args.Int("parallel", Games.DefaultParallelism), afterTurn: sim =>
        {
            if (!checkpoints.Contains(sim.Turn)) return;
            foreach (var t in sim.Standings()) samples.Add((t.Name, sim.Turn, t.BankedGold, t.Workers, t.Bases));
        });
        var names = samples.Select(s => s.Name).Distinct().OrderByDescending(n => samples.Where(s => s.Name == n && s.Turn == 480).Average(s => s.Gold)).ToList();
        foreach (var (title, pick) in new (string, Func<(string Name, int Turn, long Gold, int Workers, int Bases), double>)[]
                 { ("banked gold", s => s.Gold), ("workers", s => s.Workers), ("bases", s => s.Bases) })
        {
            Console.WriteLine($"\nmean {title} by turn");
            Console.WriteLine($"{"bot",-13}" + string.Concat(checkpoints.Select(c => $"{c,7}")));
            foreach (var n in names)
                Console.WriteLine($"{n,-13}" + string.Concat(checkpoints.Select(c =>
                {
                    var xs = samples.Where(s => s.Name == n && s.Turn == c).Select(pick).ToList();
                    return xs.Count == 0 ? $"{"-",7}" : $"{xs.Average(),7:F1}";
                })));
        }
        return 0;
    }

    static int Latency(Args args)
    {
        Console.WriteLine($"Backend: {Blas.Backend}");
        var options = Preload(new StrategyOptions { Quiet = true }, StrategyFactory.Engines);
        if (options.Mlp != null)
            foreach (int rows in new[] { 100, 1000, 4000 })
            {
                var (mean, p99, max) = LatencyCheck.Measure(options.Mlp, rows, 2000);
                Console.WriteLine($"mlp inference, {rows,5} rows: mean {mean:F3} ms, p99 {p99:F3} ms, max {max:F3} ms");
            }
        bool allOk = true;
        foreach (var engine in StrategyFactory.Engines)
        {
            // Full planner turns (pathfinding + scoring + assignment) over whole simulated matches.
            var samples = new ConcurrentBag<double>();
            Games.RunMany(Enumerable.Range(1, args.Int("games", 11)), seed =>
                [new LatencySink(new TimedStrategy(Create(engine, options, seed)), samples), .. Games.Opponents("strong", 15, seed, options)],
                args.Int("parallel", Games.DefaultParallelism));
            var s = samples.OrderBy(x => x).ToList();
            bool ok = s.Count > 0 && s[^1] < 100;
            allOk &= ok;
            Console.WriteLine($"{engine,-5} full turn over {s.Count} turns: p50 {Pct(s, 0.5):F2} ms, p99 {Pct(s, 0.99):F2} ms, max {(s.Count > 0 ? s[^1] : 0):F2} ms {(ok ? "OK" : "OVER 100 ms")}");
        }
        Console.WriteLine(allOk ? "OK: every turn of every engine finished within 100 ms." : "WARNING: some turns exceeded 100 ms.");
        return allOk ? 0 : 1;
    }

    sealed class LatencySink(TimedStrategy inner, ConcurrentBag<double> sink) : IMarsStrategy
    {
        public string Name => inner.Name;
        public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity)
        {
            var actions = inner.CalculateActions(state, identity);
            sink.Add(inner.SamplesMs[^1]);
            return actions;
        }
        public void OnMatchEnd(GameState<MarsView>? finalState, Identity identity) => inner.OnMatchEnd(finalState, identity);
    }
}
