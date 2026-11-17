using System.Text.Json;
using Lpc.Strategies.Planner;

namespace Lpc.Training;

/// <summary>Thread-safe in-memory decision pool shared by every seat in a league round.</summary>
public sealed class DecisionBuffer : IDecisionRecorder
{
    readonly List<DecisionRecord> records = [];
    public void AddRange(IReadOnlyList<DecisionRecord> batch) { lock (records) records.AddRange(batch); }
    public List<DecisionRecord> Records { get { lock (records) return [.. records]; } }
}

/// <summary>Appends decisions to a JSONL file as each match ends (live scrimmage recording).</summary>
public sealed class JsonlDecisionRecorder(string path) : IDecisionRecorder, IDisposable
{
    readonly StreamWriter writer = new(path, append: true);
    public void AddRange(IReadOnlyList<DecisionRecord> batch)
    {
        lock (writer)
        {
            foreach (var r in batch) writer.WriteLine(JsonSerializer.Serialize(r, DecisionData.Json));
            writer.Flush();
        }
    }
    public void Dispose() { lock (writer) writer.Dispose(); }
}

public static class DecisionData
{
    /// <summary>BehaviorLogProb is NaN for deterministic engines, so named float literals must be allowed.</summary>
    public static readonly JsonSerializerOptions Json = new()
    {
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    public static void AppendJsonl(string path, IEnumerable<DecisionRecord> records)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var writer = new StreamWriter(path, append: true);
        foreach (var r in records) writer.WriteLine(JsonSerializer.Serialize(r, Json));
    }

    /// <summary>Reads decisions-*.jsonl files (recursively for a directory). Malformed lines are skipped.</summary>
    public static List<DecisionRecord> ReadAll(string pathOrDirectory)
    {
        var files = Directory.Exists(pathOrDirectory)
            ? Directory.EnumerateFiles(pathOrDirectory, "decisions*.jsonl", SearchOption.AllDirectories).Order()
            : File.Exists(pathOrDirectory) ? [pathOrDirectory] : Enumerable.Empty<string>();
        var list = new List<DecisionRecord>();
        foreach (var file in files)
            foreach (var line in File.ReadLines(file))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    var r = JsonSerializer.Deserialize<DecisionRecord>(line, Json);
                    if (r != null && r.Candidates.Length == r.Rows * Features.Count && r.Chosen >= 0 && r.Chosen < r.Rows) list.Add(r);
                }
                catch (JsonException) { }
            }
        return list;
    }
}
