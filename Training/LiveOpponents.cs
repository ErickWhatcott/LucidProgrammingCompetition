using System.IO.Compression;
using System.Text.Json;
using System.Text.Json.Nodes;
using Lpc.Client;
using Lpc.Strategies.Planner;

namespace Lpc.Training;

/// <summary>
/// Learn from the other teams in real scrimmages. Replays a recorded match (states-*.jsonl.gz, our fog-of-war
/// view each turn) as if we were each of the top opponents: a shadow planner computes the candidates that
/// team had, and its choice and payoff are inferred from what we could see. Mining = its worker staying on a
/// deposit whose amount dropped; banking = its score jumping while its worker stood on its base (the jump is
/// the banked gold). Only what was inside our vision is usable, so the data is partial but real.
/// </summary>
public static class LiveOpponents
{
    sealed class Pending
    {
        public int Turn;
        public required float[] Flat;
        public int Rows;
        public required CandidateKind[] Kinds;
        public required int[] Cells;
        public int Chosen = -1;
    }

    sealed class Tracker(string team)
    {
        public readonly string Team = team;
        public readonly PlannerStrategy Shadow = new("shadow-live", new PlannerConfig(), HeuristicScorer.Instance);
        public readonly Dictionary<string, List<Pending>> Pending = [];
        public readonly Dictionary<string, (Coordinate Pos, long Load)> Workers = [];
        public readonly List<(Pending P, float Return)> Finished = [];
        public long LastScore;
    }

    /// <summary>Extracts training rows from one completed recording; returns (trips, decisions).</summary>
    public static (List<TripRecord> Trips, List<DecisionRecord> Decisions) Extract(string statesPath, int topTeams = 5)
    {
        var states = new List<JsonObject>();
        try
        {
            using var reader = new StreamReader(new GZipStream(File.OpenRead(statesPath), CompressionMode.Decompress));
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                try { if (JsonNode.Parse(line) is JsonObject o && o["view"] is JsonObject) states.Add(o); }
                catch (JsonException) { break; }
            }
        }
        catch (Exception error) when (error is IOException or InvalidDataException) { }
        var byTurn = new SortedDictionary<int, JsonObject>();
        foreach (var s in states) byTurn[(int)s["view"]!["turn"]!] = s;
        if (byTurn.Count < 300) return ([], []);

        var final = byTurn.Last().Value["view"]!["scores"]!.AsObject().ToDictionary(k => k.Key, v => (long)v.Value!);
        string me = byTurn.Last().Value["view"]!["teamResources"]!.AsObject().First().Key;
        var watch = final.Where(k => k.Key != me).OrderByDescending(k => k.Value).Take(topTeams).Select(k => new Tracker(k.Key)).ToList();
        JsonObject? previous = null;
        foreach (var (turn, raw) in byTurn)
        {
            foreach (var t in watch) Step(t, raw, previous, turn);
            previous = raw;
        }

        var trips = new List<TripRecord>();
        var decisions = new List<DecisionRecord>();
        foreach (var t in watch)
        {
            foreach (var list in t.Pending.Values) foreach (var p in list) if (p.Chosen >= 0) t.Finished.Add((p, 0));
            long gold = final.GetValueOrDefault(t.Team);
            int place = 1 + final.Count(s => s.Key != t.Team && s.Value > gold);
            foreach (var (p, ret) in t.Finished)
            {
                trips.Add(new TripRecord(p.Flat.AsSpan(p.Chosen * Features.Count, Features.Count).ToArray(), ret, "live"));
                decisions.Add(new DecisionRecord(p.Flat, p.Rows, p.Chosen, float.NaN, ret, "opponent", t.Team, p.Turn, gold, place, final.Count, "live"));
            }
        }
        return (trips, decisions);
    }

    static void Step(Tracker t, JsonObject raw, JsonObject? previous, int turn)
    {
        var view = raw["view"]!.AsObject();
        int width = (int)view["map"]!["width"]!;
        long score = (long?)view["scores"]?[t.Team] ?? t.LastScore;
        long banked = Math.Max(0, score - t.LastScore);
        t.LastScore = score;
        var cells = view["map"]!["cells"]!.AsArray();
        var prevCells = previous?["view"]?["map"]?["cells"]?.AsArray();
        var baseCells = view["bases"]!.AsArray().Where(b => (string?)b!["teamId"] == t.Team)
            .Select(b => (int)b!["position"]!["y"]! * width + (int)b!["position"]!["x"]!).ToHashSet();

        // Synthesize that team's view: its workers get our running estimate of their load.
        var mine = view["units"]!.AsArray().Where(u => (string?)u!["teamId"] == t.Team).Select(u => u!.AsObject()).ToList();
        var onBase = mine.Where(u => baseCells.Contains((int)u["position"]!["y"]! * width + (int)u["position"]!["x"]!)).ToList();
        foreach (var u in mine)
        {
            string id = (string)u["unitId"]!;
            var pos = new Coordinate((int)u["position"]!["x"]!, (int)u["position"]!["y"]!);
            int cell = pos.Y * width + pos.X;
            t.Workers.TryGetValue(id, out var prev);
            long load = prev.Load;
            bool known = t.Workers.ContainsKey(id);
            var list = t.Pending.GetValueOrDefault(id);
            if (known && prev.Pos == pos && prevCells != null && Amount(prevCells[cell]) > Amount(cells[cell]) && load < 8)
            {
                load++;
                if (list != null)
                    foreach (var p in list)
                        if (p.Chosen < 0)
                            for (int j = 0; j < p.Rows; j++)
                                if (p.Kinds[j] == CandidateKind.Mine && p.Cells[j] == cell) { p.Chosen = j; break; }
            }
            else if (banked > 0 && baseCells.Contains(cell) && load > 0)
            {
                float value = (float)banked / Math.Max(1, onBase.Count);
                if (list != null)
                {
                    foreach (var p in list)
                    {
                        if (p.Chosen < 0) p.Chosen = Array.IndexOf(p.Kinds, CandidateKind.Return);
                        if (p.Chosen >= 0) t.Finished.Add((p, value / Math.Max(1, turn - p.Turn)));
                    }
                    list.Clear();
                }
                load = 0;
            }
            t.Workers[id] = (pos, load);
            u["inventory"] = new JsonObject { ["gold"] = load, ["iron"] = 0 };
            u["inventoryCapacity"] = 8;
        }
        var synthetic = raw.DeepClone().AsObject();
        var sv = synthetic["view"]!.AsObject();
        sv["units"] = new JsonArray(view["units"]!.AsArray().Select(u => u!.DeepClone()).ToArray());
        foreach (var u in sv["units"]!.AsArray())
            if ((string?)u!["teamId"] == t.Team && mine.FirstOrDefault(m => (string?)m["unitId"] == (string?)u["unitId"]) is { } src)
                u["inventory"] = src["inventory"]!.DeepClone();
        sv["teamResources"] = new JsonObject { [t.Team] = new JsonObject { ["gold"] = score, ["iron"] = 0 } };
        GameState<MarsView> state;
        try { state = synthetic.Deserialize<GameState<MarsView>>(LpcJson.Options)!; }
        catch (JsonException) { return; }
        t.Shadow.CandidateObserver = turn % 2 == 0 ? (unit, candidates) =>
        {
            var flat = new float[candidates.Count * Features.Count];
            for (int j = 0; j < candidates.Count; j++) candidates[j].Row.CopyTo(flat, j * Features.Count);
            if (!t.Pending.TryGetValue(unit, out var l)) t.Pending[unit] = l = [];
            if (l.Count >= 6) l.RemoveAt(0);
            l.Add(new Pending { Turn = turn, Flat = flat, Rows = candidates.Count, Kinds = candidates.Select(c => c.Kind).ToArray(), Cells = candidates.Select(c => c.Cell).ToArray() });
        } : null;
        try { t.Shadow.CalculateActions(state, new Identity(t.Team, null)); }
        catch (Exception error) when (error is not OutOfMemoryException) { }
    }

    static long Amount(JsonNode? cell) => (long?)cell?["amount"] ?? 0;
}
