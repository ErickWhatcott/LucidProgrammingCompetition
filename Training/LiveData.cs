using Lpc.Ml;
using Lpc.Strategies;
using Lpc.Strategies.Planner;

namespace Lpc.Training;

/// <summary>Scrimmage recordings under data/live, re-read only when files are added or grow.</summary>
public static class LiveData
{
    static string signature = "";
    static (List<TripRecord> Trips, List<DecisionRecord> Decisions, int Files) cached = ([], [], 0);

    /// <summary>
    /// Turns each completed scrimmage recording into training rows about the top opposing teams (once per
    /// recording; at most 2 per call to keep rounds short). Writes trips-opponents-*/decisions-opponents-* files.
    /// </summary>
    static void ExtractOpponents(string dir)
    {
        int done = 0;
        foreach (var states in Directory.EnumerateFiles(dir, "states-*.jsonl.gz").Order())
        {
            if (done >= 2) break;
            string stamp = Path.GetFileName(states)["states-".Length..^".jsonl.gz".Length];
            string marker = Path.Combine(dir, $"opponents-{stamp}.done");
            if (File.Exists(marker)) continue;
            if (DateTime.UtcNow - File.GetLastWriteTimeUtc(states) < TimeSpan.FromSeconds(60)) continue; // match may still be running
            var (trips, decisions) = LiveOpponents.Extract(states);
            if (trips.Count > 0) TrainingData.AppendJsonl(Path.Combine(dir, $"trips-opponents-{stamp}.jsonl"), trips);
            if (decisions.Count > 0) DecisionData.AppendJsonl(Path.Combine(dir, $"decisions-opponents-{stamp}.jsonl"), decisions);
            File.WriteAllText(marker, $"{trips.Count} trips, {decisions.Count} decisions\n");
            Console.WriteLine($"[league] scrim {stamp}: learned {trips.Count:N0} rows / {decisions.Count:N0} decisions from opposing teams");
            done++;
        }
    }

    public static (List<TripRecord> Trips, List<DecisionRecord> Decisions, int Files) Refresh(string dir = Paths.LiveData)
    {
        if (!Directory.Exists(dir)) return cached;
        ExtractOpponents(dir);
        var files = Directory.EnumerateFiles(dir, "*.jsonl", SearchOption.AllDirectories).Order().ToList();
        var sig = string.Join("|", files.Select(f => $"{f}:{new FileInfo(f).Length}"));
        if (sig == signature) return cached;
        try
        {
            var trips = files.Where(f => Path.GetFileName(f).StartsWith("trips")).SelectMany(TrainingData.ReadJsonl).Where(r => r.Src == "live").ToList();
            var decisions = DecisionData.ReadAll(dir).Where(d => d.Src == "live").ToList();
            cached = (trips, decisions, files.Count);
            signature = sig;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[league] could not read scrimmage data yet: {error.Message}");
        }
        return cached;
    }
}
