using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Lpc.Strategies.Planner;

namespace Lpc.Ml;

/// <summary>Thread-safe in-memory recorder (self-play runs games in parallel).</summary>
public sealed class TripBuffer : ITripRecorder
{
    readonly List<TripRecord> records = new();
    readonly object gate = new();

    public void Add(TripRecord record) { lock (gate) records.Add(record); }

    public int Count { get { lock (gate) return records.Count; } }

    /// <summary>Snapshot copy of everything recorded so far.</summary>
    public List<TripRecord> Records { get { lock (gate) return new List<TripRecord>(records); } }

    public void Clear() { lock (gate) records.Clear(); }
}

/// <summary>
/// Appends each record to a JSONL file immediately (flushed per line) so live scrimmage data
/// survives a crash. Thread-safe.
/// </summary>
public sealed class JsonlTripRecorder(string path) : ITripRecorder, IDisposable
{
    readonly object gate = new();
    StreamWriter? writer = TrainingData.OpenAppend(path);

    public string Path { get; } = path;
    public long Written { get; private set; }

    public void Add(TripRecord record)
    {
        var line = TrainingData.ToJsonLine(record);
        if (line is null) return;
        lock (gate)
        {
            if (writer is null) return;
            writer.Write(line);
            writer.Write('\n');
            writer.Flush();
            Written++;
        }
    }

    public void Dispose()
    {
        lock (gate)
        {
            writer?.Dispose();
            writer = null;
        }
    }
}

/// <summary>JSONL training-data IO. One row per line: {"f":[...28 floats],"y":0.31,"src":"sim"}.</summary>
public static class TrainingData
{
    sealed class Row
    {
        [JsonPropertyName("f")] public float[]? F { get; set; }
        [JsonPropertyName("y")] public float Y { get; set; }
        [JsonPropertyName("src")] public string? Src { get; set; }
    }

    internal static StreamWriter OpenAppend(string path)
    {
        var dir = System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        var fs = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read);
        return new StreamWriter(fs, new UTF8Encoding(false));
    }

    /// <summary>Serializes a record; returns null if it contains non-finite values (JSON can't hold them).</summary>
    internal static string? ToJsonLine(TripRecord r)
    {
        if (r.F is null || !float.IsFinite(r.Y)) return null;
        var sb = new StringBuilder(32 + r.F.Length * 10);
        sb.Append("{\"f\":[");
        for (int i = 0; i < r.F.Length; i++)
        {
            if (!float.IsFinite(r.F[i])) return null;
            if (i > 0) sb.Append(',');
            sb.Append(r.F[i].ToString("R", CultureInfo.InvariantCulture));
        }
        sb.Append("],\"y\":").Append(r.Y.ToString("R", CultureInfo.InvariantCulture));
        sb.Append(",\"src\":\"").Append(JsonEncodedText.Encode(r.Src ?? "").ToString()).Append("\"}");
        return sb.ToString();
    }

    public static void AppendJsonl(string path, IEnumerable<TripRecord> records)
    {
        using var w = OpenAppend(path);
        foreach (var r in records)
        {
            var line = ToJsonLine(r);
            if (line is null) continue;
            w.Write(line);
            w.Write('\n');
        }
    }

    /// <summary>Reads one JSONL file. Blank and malformed lines (e.g. a crash-truncated last line) are skipped.</summary>
    public static List<TripRecord> ReadJsonl(string path)
    {
        var list = new List<TripRecord>();
        foreach (var raw in File.ReadLines(path))
        {
            var line = raw.Trim();
            if (line.Length == 0) continue;
            try
            {
                var row = JsonSerializer.Deserialize<Row>(line);
                if (row?.F is null) continue;
                list.Add(new TripRecord(row.F, row.Y, row.Src ?? ""));
            }
            catch (JsonException) { }
        }
        return list;
    }

    /// <summary>Reads a single .jsonl file, or every *.jsonl under a directory (recursive, sorted by path).</summary>
    public static List<TripRecord> ReadAll(string pathOrDirectory)
    {
        if (File.Exists(pathOrDirectory)) return ReadJsonl(pathOrDirectory);
        if (!Directory.Exists(pathOrDirectory)) return [];
        var all = new List<TripRecord>();
        foreach (var f in Directory.EnumerateFiles(pathOrDirectory, "*.jsonl", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
            all.AddRange(ReadJsonl(f));
        return all;
    }
}
