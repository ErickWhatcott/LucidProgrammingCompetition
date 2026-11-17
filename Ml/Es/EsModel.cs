using System.Text.Json;
using Lpc.Strategies.Planner;

namespace Lpc.Ml.Es;

/// <summary>
/// A saved 'es' engine: evolved planner settings + linear scorer weights, plus (optionally) the
/// CMA-ES optimizer state so training can resume across runs. Config/Weights are normally the
/// decoded optimizer mean (or best-so-far); the caller decides which to publish.
/// </summary>
public sealed class EsModel
{
    public const int FormatVersion = 1;

    public PlannerConfig Config { get; set; } = new();
    public float[] Weights { get; set; } = EsGenome.HeuristicWeights();
    public CmaEs? Optimizer { get; set; }
    public Dictionary<string, string> Metadata { get; set; } = new();

    public LinearScorer CreateScorer() => new(Weights);

    /// <summary>Fresh model at generation 0: defaults + heuristic weights, optimizer centered there.</summary>
    public static EsModel CreateInitial(PlannerConfig defaults, double sigma = 0.2, int lambda = 12, int seed = 1)
    {
        var mean = EsGenome.InitialMean(defaults);
        var (cfg, w) = EsGenome.Decode(mean, defaults);
        return new EsModel { Config = cfg, Weights = w, Optimizer = new CmaEs(mean, sigma, lambda, seed) };
    }

    /// <summary>Sets Config/Weights from the optimizer's current mean (decoded against <paramref name="template"/>).</summary>
    public void TakeMean(PlannerConfig template)
    {
        if (Optimizer == null) return;
        (Config, Weights) = EsGenome.Decode(Optimizer.Mean, template);
    }

    sealed class Dto
    {
        public int Version { get; set; }
        public int FeatureVersion { get; set; }
        public int FeatureCount { get; set; }
        public PlannerConfig Config { get; set; } = new();
        public float[] Weights { get; set; } = [];
        public JsonElement? Optimizer { get; set; }
        public Dictionary<string, string> Metadata { get; set; } = new();
    }

    static readonly JsonSerializerOptions Options = new() { WriteIndented = true };

    public void Save(string path)
    {
        var dto = new Dto
        {
            Version = FormatVersion,
            FeatureVersion = Features.Version,
            FeatureCount = Features.Count,
            Config = Config,
            Weights = Weights,
            Optimizer = Optimizer == null ? null : JsonDocument.Parse(Optimizer.ToJson()).RootElement.Clone(),
            Metadata = Metadata,
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        string tmp = path + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(dto, Options));
        File.Move(tmp, path, overwrite: true);
    }

    /// <summary>Loads a model; throws InvalidDataException if it doesn't match the current <see cref="Features"/> layout.</summary>
    public static EsModel Load(string path)
    {
        Dto dto;
        try
        {
            dto = JsonSerializer.Deserialize<Dto>(File.ReadAllText(path), Options)
                ?? throw new InvalidDataException($"{path}: empty es model.");
        }
        catch (JsonException e) { throw new InvalidDataException($"{path}: malformed es model: {e.Message}", e); }
        if (dto.Config is null || dto.Weights is null) throw new InvalidDataException($"{path}: es model is missing Config or Weights.");
        if (dto.Version != FormatVersion) throw new InvalidDataException($"{path}: es model format v{dto.Version}, expected v{FormatVersion}.");
        if (dto.FeatureVersion != Features.Version) throw new InvalidDataException($"{path}: featureVersion {dto.FeatureVersion} != current {Features.Version}.");
        if (dto.FeatureCount != Features.Count || dto.Weights.Length != Features.Count)
            throw new InvalidDataException($"{path}: feature count {dto.FeatureCount}/{dto.Weights.Length} != current {Features.Count}.");
        foreach (var v in dto.Weights)
            if (!float.IsFinite(v)) throw new InvalidDataException($"{path}: es model has non-finite weights.");
        if (!(dto.Config.TurnBudgetMs > 0) || dto.Config.TurnBudgetMs > 100) dto.Config.TurnBudgetMs = 100; // hard per-turn cap
        CmaEs? opt = null;
        if (dto.Optimizer is { ValueKind: JsonValueKind.Object } o)
        {
            opt = CmaEs.FromJson(o.GetRawText());
            // The genome grew (new knobs): keep the evolved Config/Weights, drop the optimizer state.
            // Training re-seeds CMA-ES from Encode(Config, Weights), so no evolved progress is lost.
            if (opt.Dimension != EsGenome.Dimension) opt = null;
        }
        return new EsModel { Config = dto.Config, Weights = dto.Weights, Optimizer = opt, Metadata = dto.Metadata ?? new() };
    }
}
