using System.Runtime.InteropServices;
using Lpc.Sim;
using Lpc.Strategies.Planner;
using Microsoft.Data.Sqlite;

namespace Lpc.Training;

/// <summary>
/// SQLite store for self-play (data/sim/selfplay.db). Every league round appends its value-training rows
/// and every game's results, so the training pool survives restarts and new builds, and keeps growing
/// whether or not anyone is playing live. Training reads back the newest rows.
/// </summary>
public sealed class SimStore : IDisposable
{
    public const string DefaultPath = "data/sim/selfplay.db";
    readonly SqliteConnection db;

    public SimStore(string path = DefaultPath)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        db = new SqliteConnection($"Data Source={path}");
        db.Open();
        Exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
        Exec("""
            CREATE TABLE IF NOT EXISTS trips(id INTEGER PRIMARY KEY, round INTEGER NOT NULL, src TEXT NOT NULL, y REAL NOT NULL, f BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS games(round INTEGER NOT NULL, seed INTEGER NOT NULL, team TEXT NOT NULL, bot TEXT NOT NULL,
                gold INTEGER NOT NULL, place INTEGER NOT NULL, workers INTEGER NOT NULL, bases INTEGER NOT NULL, utc TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
            """);
    }

    void Exec(string sql)
    {
        using var cmd = db.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    /// <summary>Global round number across all runs (stored in meta).</summary>
    public long NextRound()
    {
        using var cmd = db.CreateCommand();
        cmd.CommandText = "INSERT INTO meta(k, v) VALUES('round', '0') ON CONFLICT(k) DO UPDATE SET v = CAST(v AS INTEGER) + 1 RETURNING v";
        return Convert.ToInt64(cmd.ExecuteScalar());
    }

    public long TripCount()
    {
        using var cmd = db.CreateCommand();
        cmd.CommandText = "SELECT COUNT(*) FROM trips";
        return Convert.ToInt64(cmd.ExecuteScalar());
    }

    public long GameCount()
    {
        using var cmd = db.CreateCommand();
        cmd.CommandText = "SELECT COUNT(DISTINCT round * 10000000 + seed) FROM games";
        return Convert.ToInt64(cmd.ExecuteScalar());
    }

    public void AddRound(long round, IReadOnlyList<TripRecord> trips, IReadOnlyList<SimResult> games)
    {
        using var tx = db.BeginTransaction();
        using (var cmd = db.CreateCommand())
        {
            cmd.Transaction = tx;
            cmd.CommandText = "INSERT INTO trips(round, src, y, f) VALUES($r, $s, $y, $f)";
            var pr = cmd.Parameters.Add("$r", SqliteType.Integer);
            var ps = cmd.Parameters.Add("$s", SqliteType.Text);
            var py = cmd.Parameters.Add("$y", SqliteType.Real);
            var pf = cmd.Parameters.Add("$f", SqliteType.Blob);
            cmd.Prepare();
            pr.Value = round;
            foreach (var t in trips)
            {
                ps.Value = t.Src;
                py.Value = t.Y;
                pf.Value = MemoryMarshal.AsBytes(t.F.AsSpan()).ToArray();
                cmd.ExecuteNonQuery();
            }
        }
        using (var cmd = db.CreateCommand())
        {
            cmd.Transaction = tx;
            cmd.CommandText = "INSERT INTO games VALUES($r, $seed, $team, $bot, $gold, $place, $w, $b, $utc)";
            string utc = DateTime.UtcNow.ToString("O");
            foreach (var g in games)
                foreach (var t in g.Teams)
                {
                    cmd.Parameters.Clear();
                    cmd.Parameters.AddWithValue("$r", round);
                    cmd.Parameters.AddWithValue("$seed", g.Seed);
                    cmd.Parameters.AddWithValue("$team", t.TeamId);
                    cmd.Parameters.AddWithValue("$bot", t.Name);
                    cmd.Parameters.AddWithValue("$gold", t.BankedGold);
                    cmd.Parameters.AddWithValue("$place", t.Place);
                    cmd.Parameters.AddWithValue("$w", t.Workers);
                    cmd.Parameters.AddWithValue("$b", t.Bases);
                    cmd.Parameters.AddWithValue("$utc", utc);
                    cmd.ExecuteNonQuery();
                }
        }
        tx.Commit();
    }

    /// <summary>The newest <paramref name="limit"/> rows, oldest first.</summary>
    public List<TripRecord> LoadRecent(int limit)
    {
        var list = new List<TripRecord>();
        using var cmd = db.CreateCommand();
        cmd.CommandText = "SELECT src, y, f FROM (SELECT id, src, y, f FROM trips ORDER BY id DESC LIMIT $n) ORDER BY id";
        cmd.Parameters.AddWithValue("$n", limit);
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            var bytes = (byte[])r[2];
            if (bytes.Length != Features.Count * sizeof(float)) continue;
            list.Add(new TripRecord(MemoryMarshal.Cast<byte, float>(bytes).ToArray(), (float)r.GetDouble(1), r.GetString(0)));
        }
        return list;
    }

    /// <summary>Keeps the newest rows only (the database would otherwise grow ~40 MB per round).</summary>
    public void Prune(long keep)
    {
        Exec($"DELETE FROM trips WHERE id <= (SELECT MAX(id) FROM trips) - {keep}");
    }

    /// <summary>Per-bot results over the last N rounds: (bot, games, mean gold, mean place).</summary>
    public List<(string Bot, long Games, double Gold, double Place)> Leaderboard(int lastRounds)
    {
        var list = new List<(string, long, double, double)>();
        using var cmd = db.CreateCommand();
        cmd.CommandText = """
            SELECT bot, COUNT(*), AVG(gold), AVG(place) FROM games
            WHERE round > (SELECT MAX(round) FROM games) - $n GROUP BY bot ORDER BY AVG(gold) DESC
            """;
        cmd.Parameters.AddWithValue("$n", lastRounds);
        using var r = cmd.ExecuteReader();
        while (r.Read()) list.Add((r.GetString(0), r.GetInt64(1), r.GetDouble(2), r.GetDouble(3)));
        return list;
    }

    public void Dispose() => db.Dispose();
}
