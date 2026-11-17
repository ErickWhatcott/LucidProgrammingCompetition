using Lpc.Ml;
using Lpc.Ml.Es;
using Lpc.Sim;
using Lpc.Strategies.Planner;

namespace Lpc.Strategies;

public sealed class StrategyOptions
{
    public PlannerConfig? Config;          // explicit planner settings (es carries its own)
    public string? ConfigPath;             // shared planner settings for mlp/gbdt/ppo
    public string? ModelPath;              // overrides models/<engine>.json for the mode being created
    public Mlp? Mlp;                       // pre-loaded models, shared read-only across parallel games
    public GbdtModel? Gbdt;
    public PolicyNet? Policy;
    public EsModel? Es;
    public (PlannerConfig Config, float[] Weights)? EsGenome; // a CMA-ES candidate during training
    public double? Blend;                  // mlp: learned/heuristic blend override
    public double Temperature = 0.5;       // ppo sampling temperature (matches league training)
    public ITripRecorder? Recorder;
    public IDecisionRecorder? Decisions;
    public string RecordSource = "sim";
    public double Noise;                   // exploration noise for value engines during self-play
    public int Seed;
    public bool Quiet;
    public bool PreferLive;                // live play: use models/<engine>.live.json when it exists
}

public static class Paths
{
    public static string Model(string engine) => $"models/{engine}.json";
    public static string Pretrained(string engine) => $"models/{engine}.pretrained.json";
    /// <summary>Fine-tuned on scrimmages; preferred for live play when present.</summary>
    public static string Live(string engine) => $"models/{engine}.live.json";
    public const string PlannerConfig = "models/planner-config.json";
    public const string SimData = "data/sim";
    public const string LiveData = "data/live";
}

public static class StrategyFactory
{
    /// <summary>Our learned engines (research ranks 1-3, plus the MLP value scorer).</summary>
    public static readonly string[] Engines = ["mlp", "gbdt", "ppo", "es"];
    /// <summary>Simulator opponents approximating the practice bots.</summary>
    public static readonly string[] PracticeBots = ["intermediate", "beginner", "novice", .. ScriptedBots.Names];
    public static readonly string[] Modes = [.. Engines, .. PracticeBots];

    public static PlannerConfig LoadConfig(StrategyOptions o)
    {
        if (o.Config != null) return o.Config.Clone();
        var path = o.ConfigPath ?? (File.Exists(Paths.PlannerConfig) ? Paths.PlannerConfig : null);
        return path != null ? PlannerConfig.Load(path) : new PlannerConfig();
    }

    public static IMarsStrategy Create(string mode, StrategyOptions o)
    {
        var config = LoadConfig(o);
        PlannerStrategy Planner(string name, PlannerConfig cfg, ICandidateScorer scorer) =>
            new(name, cfg, scorer, o.Recorder, o.RecordSource, o.Noise, o.Seed, o.Decisions);
        switch (mode)
        {
            case "mlp":
            {
                var model = o.Mlp ?? Load(mode, o, Mlp.Load);
                return Planner(mode, config, model == null ? HeuristicScorer.Instance : Blended(new LearnedScorer(model), o.Blend, model.Metadata));
            }
            case "gbdt":
            {
                var model = o.Gbdt ?? Load(mode, o, GbdtModel.Load);
                return Planner(mode, config, model == null ? HeuristicScorer.Instance : Blended(new GbdtScorer(model), o.Blend, model.Metadata));
            }
            case "ppo":
            {
                var net = o.Policy ?? Load(mode, o, PolicyNet.Load);
                return Planner(mode, config, net != null ? new PolicyScorer(net, o.Temperature) : HeuristicScorer.Instance);
            }
            case "es":
            {
                if (o.EsGenome is { } genome) return Planner(mode, genome.Config, new LinearScorer(genome.Weights));
                var es = o.Es ?? Load(mode, o, EsModel.Load);
                return es != null ? Planner(mode, es.Config, new LinearScorer(es.Weights)) : Planner(mode, config, HeuristicScorer.Instance);
            }
            case "intermediate":
                return new PlannerStrategy("intermediate", PlannerConfig.Intermediate(), seed: o.Seed);
            case "beginner":
                return new BeginnerBot(o.Seed);
            case "novice":
                return new NoviceBot(o.Seed);
            default:
                return ScriptedBots.Create(mode, o.Seed)
                    ?? throw new ArgumentException($"Unknown mode '{mode}'. Modes: {string.Join(", ", Modes)}");
        }
    }

    /// <summary>Value engines may blend their prediction with the hand-written rate; the league picks the blend.</summary>
    static ICandidateScorer Blended(ICandidateScorer learned, double? overrideBlend, Dictionary<string, string> metadata)
    {
        double blend = overrideBlend ?? (metadata.TryGetValue("blend", out var b)
            && double.TryParse(b, System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : 1.0);
        return blend >= 0.999 ? learned : new BlendedScorer(learned, blend);
    }

    /// <summary>Until an engine has been trained it plays with the hand-written scorer (bootstraps the data pool).</summary>
    static T? Load<T>(string engine, StrategyOptions o, Func<string, T> load) where T : class
    {
        var path = o.ModelPath ?? (o.PreferLive && File.Exists(Paths.Live(engine)) ? Paths.Live(engine) : Paths.Model(engine));
        if (File.Exists(path))
        {
            if (!o.Quiet && o.PreferLive) Console.WriteLine($"[{engine}] model {path} (trained {File.GetLastWriteTime(path):HH:mm:ss})");
            return load(path);
        }
        if (!o.Quiet) Console.Error.WriteLine($"[{engine}] no model at {path}; playing with the heuristic scorer until trained (`lpc train pretrain`).");
        return null;
    }
}

/// <summary>score = blend * learned + (1 - blend) * heuristic.</summary>
public sealed class BlendedScorer(ICandidateScorer learned, double blend) : ICandidateScorer
{
    public string Name => $"blend({blend:F2})";
    public bool NeedsFeatures => learned.NeedsFeatures;

    public bool Score(Span<Candidate> candidates, float[] features, double remainingBudgetMs)
    {
        if (!learned.Score(candidates, features, remainingBudgetMs)) return false;
        foreach (ref var c in candidates) c.Score = (float)(blend * c.Score + (1 - blend) * c.Heuristic);
        return true;
    }
}
