// C# port of the Java starter's BotClient.java. Control flow, retries, headers and messages
// follow the Java original; Gson's JsonObject maps to System.Text.Json.Nodes.JsonObject.
using System.Diagnostics;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Lpc.Client;

public sealed class BotClient
{
    const string RegisterPath = "/api/team/matches/current/register";
    static readonly HashSet<int> RetryStatuses = [408, 429, 500, 502, 503, 504];

    public static GameState<MarsView> ReadMarsState(string json) =>
        JsonSerializer.Deserialize<GameState<MarsView>>(json, LpcJson.Options)!;

    public static GameState<SolarView> ReadSolarState(string json) =>
        JsonSerializer.Deserialize<GameState<SolarView>>(json, LpcJson.Options)!;

    readonly string baseUrl;
    readonly string apiKey;
    string? botSessionToken;
    string? botMatchId;
    string? botSessionRevision;
    bool recoveringSession;
    readonly HttpClient http = new(new SocketsHttpHandler { ConnectTimeout = TimeSpan.FromSeconds(5) })
    {
        Timeout = TimeSpan.FromSeconds(5),
    };

    /// <summary>Called with every raw state document the client reads (used by --record).</summary>
    public Action<JsonObject>? StateObserver { get; set; }

    /// <summary>Looping play: a match id to skip (the one just finished) while waiting for the next match.</summary>
    public string? SkipMatchId { get; set; }

    /// <summary>The match this client registered for (null if it returned before registering).</summary>
    public string? LastMatchId { get; private set; }

    /// <summary>Polled while waiting for a match; returning true makes the run return without playing.</summary>
    public Func<bool>? StopWaiting { get; set; }

    /// <summary>Turns whose actions the server accepted for a later turn than intended.</summary>
    public int LateTurns { get; private set; }

    /// <summary>Status of the last match when the run returned ("completed", "cancelled", ...).</summary>
    public string? FinalStatus { get; private set; }

    /// <summary>Use the server-sent turn stream in real-time matches (default). False = fast polling.</summary>
    public bool UseTurnStream { get; set; } = true;

    /// <summary>
    /// Background reader of /api/events/matches/{id} (server-sent events: turn, match_starting, match_ended).
    /// Keeps the newest state; Next() returns as soon as the turn or status moves past the caller's state,
    /// so the bot always acts on the newest turn. Falls back to GET /state polling whenever the stream is down
    /// or quiet, and reconnects with afterTurn so nothing is replayed.
    /// </summary>
    sealed class StateFeed : IDisposable
    {
        readonly BotClient client;
        readonly string matchId, path;
        readonly CancellationTokenSource cts = new();
        readonly HttpClient http = new(new SocketsHttpHandler { ConnectTimeout = TimeSpan.FromSeconds(5) }) { Timeout = Timeout.InfiniteTimeSpan };
        readonly object gate = new();
        JsonObject latest;
        TaskCompletionSource changed = new(TaskCreationOptions.RunContinuationsAsynchronously);
        volatile bool healthy;

        public StateFeed(BotClient client, string matchId, string path, JsonObject initial)
        {
            this.client = client;
            this.matchId = matchId;
            this.path = path;
            latest = initial;
            _ = Task.Run(Pump);
        }

        static int Turn(JsonObject s) => (int)s["match"]!["turn"]!;
        static string? Status(JsonObject s) => (string?)s["match"]!["status"];

        void Publish(JsonObject s)
        {
            TaskCompletionSource signal;
            lock (gate)
            {
                bool newer = Terminal(s) || Turn(s) > Turn(latest) || (Turn(s) == Turn(latest) && Status(s) != Status(latest));
                if (!newer || Terminal(latest)) return;
                latest = s;
                signal = changed;
                changed = new(TaskCreationOptions.RunContinuationsAsynchronously);
            }
            signal.TrySetResult();
        }

        public async Task<JsonObject> Next(JsonObject current)
        {
            int turn = Turn(current);
            string? status = Status(current);
            for (;;)
            {
                Task wait;
                lock (gate)
                {
                    if (Terminal(latest) || Turn(latest) > turn || Status(latest) != status) return latest;
                    wait = changed.Task;
                }
                // Healthy stream: wait for a push (poll only as a safety net). Stream down: poll fast.
                if (await Task.WhenAny(wait, Task.Delay(healthy ? 1500 : 20)) == wait) continue;
                try { Publish(await client.Request("GET", path + "/state", null, true)); }
                catch (ApiError error) when (error.Status is 401 or 403 || error.SessionConflict) { throw; }
                catch (Exception error) when (error is not OutOfMemoryException) { Console.Error.WriteLine(error.Message); }
            }
        }

        async Task Pump()
        {
            int failures = 0;
            while (!cts.IsCancellationRequested)
            {
                try
                {
                    int after;
                    lock (gate) after = Turn(latest);
                    using var request = new HttpRequestMessage(HttpMethod.Get,
                        $"{client.baseUrl}/api/events/matches/{Uri.EscapeDataString(matchId)}?afterTurn={after}");
                    request.Headers.Add("x-api-key", client.apiKey);
                    if (client.botSessionToken != null) request.Headers.Add("x-bot-session", client.botSessionToken);
                    request.Headers.Add("accept", "text/event-stream");
                    using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cts.Token);
                    if (!response.IsSuccessStatusCode) throw new HttpRequestException($"turn stream HTTP {(int)response.StatusCode}");
                    using var reader = new StreamReader(await response.Content.ReadAsStreamAsync(cts.Token));
                    healthy = true;
                    failures = 0;
                    string name = "message";
                    var data = new StringBuilder();
                    for (;;)
                    {
                        using var idle = CancellationTokenSource.CreateLinkedTokenSource(cts.Token);
                        idle.CancelAfter(TimeSpan.FromSeconds(10)); // the official clients reconnect after 10 s of silence
                        var line = await reader.ReadLineAsync(idle.Token);
                        if (line == null) break;
                        if (line.Length == 0)
                        {
                            if (data.Length > 0) Handle(name, data.ToString());
                            name = "message";
                            data.Clear();
                        }
                        else if (line.StartsWith(':')) { }
                        else if (line.StartsWith("event:")) name = line[6..].Trim();
                        else if (line.StartsWith("data:")) { if (data.Length > 0) data.Append('\n'); data.Append(line[5..].Trim()); }
                    }
                }
                catch (OperationCanceledException) when (cts.IsCancellationRequested) { return; }
                catch (Exception error) when (error is not OutOfMemoryException)
                {
                    if (++failures <= 3 || failures % 20 == 0)
                        Console.Error.WriteLine($"Turn stream: {error.Message}; reconnecting (polling meanwhile).");
                }
                healthy = false;
                lock (gate) if (Terminal(latest)) return;
                try { await Task.Delay(Math.Min(2000, 250 * Math.Max(1, failures)), cts.Token); }
                catch (OperationCanceledException) { return; }
            }
        }

        void Handle(string name, string json)
        {
            JsonObject data;
            try { data = JsonNode.Parse(json)!.AsObject(); }
            catch (Exception) { return; }
            JsonObject current;
            lock (gate) current = latest;
            switch (name)
            {
                case "turn":
                {
                    int t = (int)data["turn"]!;
                    if (t <= Turn(current)) return;
                    var match = current["match"]!.DeepClone().AsObject();
                    match["turn"] = t;
                    match["status"] = t >= (int)match["maxTurns"]! ? "completed" : "active";
                    var s = new JsonObject
                    {
                        ["match"] = match,
                        ["view"] = data["view"]?.DeepClone(),
                        ["serverTimeMs"] = data["serverTimeMs"]?.DeepClone(),
                    };
                    if (data["turnEndsAtMs"] is { } ends) s["turnEndsAtMs"] = ends.DeepClone();
                    Publish(s);
                    break;
                }
                case "match_starting":
                {
                    var match = data["match"]!.DeepClone().AsObject();
                    if ((int)match["turn"]! < Turn(current)) return;
                    var s = current.DeepClone().AsObject();
                    s["match"] = match;
                    s["serverTimeMs"] = data["serverTimeMs"]?.DeepClone();
                    s["turnEndsAtMs"] = data["startsAtMs"]?.DeepClone();
                    Publish(s);
                    break;
                }
                case "match_ended":
                {
                    var s = current.DeepClone().AsObject();
                    s["match"] = data["match"]!.DeepClone();
                    s["serverTimeMs"] = data["serverTimeMs"]?.DeepClone();
                    Publish(s);
                    break;
                }
            }
        }

        public void Dispose()
        {
            cts.Cancel();
            http.Dispose();
        }
    }

    public BotClient(string? keyFile = null) : this(ConfiguredBaseUrl(), ReadApiKey(keyFile)) { }

    static string ConfiguredBaseUrl()
    {
        var value = Environment.GetEnvironmentVariable("LUCID_API_BASE_URL");
        return string.IsNullOrWhiteSpace(value) ? "https://lpc.sh" : value;
    }

    /// <summary>
    /// The team key is read from a text file (default "api.key", looked up in the working directory and
    /// then upward from the executable's folder). The API_KEY environment variable is only a fallback.
    /// </summary>
    public static string ReadApiKey(string? keyFile = null)
    {
        var candidates = new List<string>();
        if (keyFile != null) candidates.Add(keyFile);
        else
        {
            candidates.Add(Path.Combine(Directory.GetCurrentDirectory(), "api.key"));
            for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
                candidates.Add(Path.Combine(dir.FullName, "api.key"));
        }
        foreach (var path in candidates)
            if (File.Exists(path))
            {
                var key = File.ReadAllText(path).Trim();
                if (key.Length > 0) return key;
            }
        if (keyFile != null) throw new FileNotFoundException($"API key file {keyFile} is missing or empty.");
        return Environment.GetEnvironmentVariable("API_KEY") is { Length: > 0 } env
            ? env
            : throw new ArgumentException("No API key: put your team key in api.key next to the project (or pass --key-file PATH).");
    }

    public BotClient(string baseUrl, string apiKey)
    {
        if (string.IsNullOrWhiteSpace(apiKey) || apiKey == "replace-with-api-key")
            throw new ArgumentException("Put your team's API key in api.key before starting the bot.");
        this.baseUrl = baseUrl.TrimEnd('/');
        this.apiKey = apiKey;
    }

    async Task<JsonObject> Request(string method, string path, JsonObject? body, bool retry)
    {
        int attempts = retry ? 3 : 1;
        bool recoveredSession = false;
        for (int attempt = 0; ; attempt++)
        {
            using var message = new HttpRequestMessage(new HttpMethod(method), baseUrl + path);
            message.Headers.Add("x-api-key", apiKey);
            message.Content = body == null
                ? new ByteArrayContent([])
                : new StringContent(body.ToJsonString(), Encoding.UTF8);
            message.Content.Headers.ContentType = new("application/json");
            if (botSessionToken != null) message.Headers.Add("x-bot-session", botSessionToken);
            if (path == RegisterPath && botMatchId != null) message.Headers.Add("x-bot-match-id", botMatchId);
            if (path == RegisterPath)
            {
                if (recoveringSession) message.Headers.Add("x-bot-session-recover", "1");
                else if (botSessionRevision != null) message.Headers.Add("x-bot-session-revision", botSessionRevision);
            }
            try
            {
                using var response = await http.SendAsync(message);
                int status = (int)response.StatusCode;
                string text = await response.Content.ReadAsStringAsync();
                if (status >= 200 && status < 300)
                {
                    if (path == RegisterPath) botSessionToken = Header(response, "x-bot-session");
                    botSessionRevision = Header(response, "x-bot-session-revision") ?? botSessionRevision;
                    try
                    {
                        return JsonNode.Parse(text)!.AsObject();
                    }
                    catch (Exception error) when (error is JsonException or InvalidOperationException or NullReferenceException)
                    {
                        throw new InvalidOperationException($"Cannot read the competition API at {baseUrl}; check LUCID_API_BASE_URL.", error);
                    }
                }
                if (status == 409 && text.Contains("bot_session_missing") && !recoveringSession && !recoveredSession
                    && botMatchId != null && botSessionToken != null && path != RegisterPath)
                {
                    recoveredSession = true;
                    recoveringSession = true;
                    try { await Request("POST", RegisterPath, null, true); }
                    finally { recoveringSession = false; }
                    attempt--;
                    continue;
                }
                if (!RetryStatuses.Contains(status) || attempt + 1 >= attempts)
                    throw new ApiError(status, text);
            }
            catch (Exception error) when (error is HttpRequestException or TaskCanceledException)
            {
                if (attempt + 1 >= attempts) throw new IOException($"Cannot reach {baseUrl}; check the URL and your connection.", error);
            }
            await Task.Delay(250 << attempt);
        }
    }

    static string? Header(HttpResponseMessage response, string name) =>
        response.Headers.TryGetValues(name, out var values) ? values.FirstOrDefault() : null;

    public sealed class ApiError(int status, string body) : Exception(Describe(status, body))
    {
        public int Status { get; } = status;
        public string Body { get; } = body;

        static string Describe(int status, string body) =>
            status == 409 && body.Contains("pvp_disabled") ? "No match assigned. PvP practice is currently disabled. Create a practice match on the website or ask an organizer to assign one, then run your bot again."
            : status == 403 && body.Contains("Team is not in an active match.") ? "No current match is available; join a match before starting the bot."
            : status == 409 && body.Contains("bot_session_conflict") ? "This bot was replaced by another client; stopping."
            : status is 401 or 403 ? "API key was rejected; check API_KEY."
            : status == 404 ? "Cannot reach the competition API; check LUCID_API_BASE_URL."
            : $"Server request failed (HTTP {status}).";

        public bool SessionConflict => Status == 409 && Body.Contains("bot_session_conflict");

        public bool InactiveMatch
        {
            get
            {
                if (Status != 403) return false;
                try
                {
                    var error = JsonNode.Parse(Body)!["error"]!;
                    return (string?)error["code"] == "forbidden" && (string?)error["message"] == "Team is not in an active match.";
                }
                catch (Exception) { return false; }
            }
        }
    }

    static long CheckedRevision(long revision)
    {
        if (revision < 0 || revision > 9_007_199_254_740_991L) throw new InvalidOperationException("Action revision is outside the server's safe integer range.");
        return revision;
    }

    static bool Terminal(JsonObject state) =>
        (string?)state["match"]!["status"] is "completed" or "cancelled";

    static bool Has(JsonObject o, string key) => o.TryGetPropertyValue(key, out var value) && value != null;

    async Task<bool> MatchEnded(string path)
    {
        try
        {
            return Terminal(await Request("GET", path + "/state", null, true));
        }
        catch (ApiError error)
        {
            if (error.Status == 401 || error.SessionConflict || (error.Status == 403 && !error.InactiveMatch)) throw;
            Console.Error.WriteLine(error.Message);
        }
        catch (IOException error)
        {
            Console.Error.WriteLine(error.Message);
        }
        return false;
    }

    /// <summary>Read-only key check: authenticates and returns the team identity without registering for a match.</summary>
    public async Task<Identity> Authenticate() =>
        (await Request("POST", "/api/auth/api-key", null, true)).Deserialize<Identity>(LpcJson.Options)!;

    public Task Run(Func<GameState<MarsView>, Identity, IReadOnlyDictionary<string, MarsAction>> strategy,
                    Func<GameState<SolarView>, Identity, IReadOnlyDictionary<string, SolarAction>> solarStrategy) =>
        RunForRules(null, strategy, solarStrategy);

    public Task RunMars(Func<GameState<MarsView>, Identity, IReadOnlyDictionary<string, MarsAction>> strategy) =>
        RunForRules("lpc2026", strategy, null);

    public Task RunSolar(Func<GameState<SolarView>, Identity, IReadOnlyDictionary<string, SolarAction>> strategy) =>
        RunForRules("SolarOrbitor-demo", null, strategy);

    async Task RunForRules(string? rulesVersion,
        Func<GameState<MarsView>, Identity, IReadOnlyDictionary<string, MarsAction>>? strategy,
        Func<GameState<SolarView>, Identity, IReadOnlyDictionary<string, SolarAction>>? solarStrategy)
    {
        var identity = (await Request("POST", "/api/auth/api-key", null, true)).Deserialize<Identity>(LpcJson.Options)!;
        JsonObject current;
        string announced = "";
        for (;;)
        {
            if (StopWaiting?.Invoke() == true) return;
            try { current = await Request("GET", "/api/team/matches/current/state", null, true); }
            catch (ApiError error) when (error.InactiveMatch)
            {
                var enrollment = await Request("POST", RegisterPath, null, false);
                if (!Has(enrollment, "match"))
                {
                    if (!Has(enrollment, "queue")) throw new InvalidOperationException("Unexpected registration response.");
                    var queue = enrollment["queue"]!.AsObject();
                    string waiting = Has(queue, "autoQueue") && !(bool)queue["autoQueue"]!
                        ? "Waiting for a private practice match. Create one on the website; keep your bot running."
                        : "Waiting in the PvP practice queue; keep your bot running.";
                    if (announced != waiting) { Console.WriteLine(waiting); announced = waiting; }
                    await Task.Delay(1000);
                    continue;
                }
                current = await Request("GET", "/api/team/matches/current/state", null, true);
            }
            // When looping, the match we just finished can still be "current" for a while: wait for a new one.
            if ((string?)current["match"]!["matchId"] == SkipMatchId || Terminal(current))
            {
                const string waitingNext = "Match finished; waiting for the next match. Create one on the website (Ctrl+C to stop).";
                if (announced != waitingNext) { Console.WriteLine(waitingNext); announced = waitingNext; }
                await Task.Delay(2000);
                continue;
            }
            break;
        }
        botMatchId = (string)current["match"]!["matchId"]!;
        LastMatchId = botMatchId;
        botSessionToken = Guid.NewGuid().ToString();
        var state = await Request("POST", RegisterPath, null, true);
        if (rulesVersion != null && (string?)state["match"]!["rulesVersion"] != rulesVersion)
            throw new InvalidOperationException("This starter does not support the current game. Download its starter from Help.");

        var matchId = (string)state["match"]!["matchId"]!;
        var path = "/api/team/matches/" + Uri.EscapeDataString(matchId);
        Console.WriteLine("registered for match " + matchId);
        var revisions = new Dictionary<string, long>();
        if (Has(state, "actionRevisions"))
            foreach (var (key, value) in state["actionRevisions"]!.AsObject()) revisions[key] = CheckedRevision((long)value!);
        int submittedTurn = -1;
        long stateReceivedAt = Stopwatch.GetTimestamp();
        StateObserver?.Invoke(state);
        // Real-time matches (timed / multi-client-action) advance every ~200-1000 ms. Polling costs a full
        // round trip per turn (~200 ms to lpc.sh), so read the server's turn stream instead, like the
        // official Python/JS clients. Practice (client-action) keeps the original request/response loop.
        bool realTime = (string?)state["match"]!["turnAdvanceMode"] is { } advance && advance != "client-action";
        using var feed = realTime && UseTurnStream ? new StateFeed(this, matchId, path, state) : null;
        if (feed != null) Console.WriteLine($"Real-time match ({state["match"]!["turnAdvanceMode"]}): using the server's turn stream.");
        while (!Terminal(state))
        {
            var match = state["match"]!.AsObject();
            int turn = (int)match["turn"]!;
            if ((string?)match["status"] == "active" && Has(state, "view") && turn != submittedTurn)
            {
                IEnumerable<KeyValuePair<string, IAction>> actions;
                try
                {
                    actions = (string?)match["rulesVersion"] == "SolarOrbitor-demo"
                        ? Cast(solarStrategy!(state.Deserialize<GameState<SolarView>>(LpcJson.Options)!, identity))
                        : Cast(strategy!(state.Deserialize<GameState<MarsView>>(LpcJson.Options)!, identity));
                }
                catch (Exception error) when (error is not OutOfMemoryException)
                {
                    Console.Error.WriteLine($"Strategy failed on turn {turn}: {error}; submitting no actions.");
                    actions = [];
                }
                var submissions = new JsonArray();
                foreach (var (entityId, action) in actions)
                {
                    long revision = CheckedRevision(revisions.GetValueOrDefault(entityId) + 1);
                    revisions[entityId] = revision;
                    submissions.Add(new JsonObject
                    {
                        ["entityId"] = entityId,
                        ["revision"] = revision,
                        ["action"] = JsonSerializer.SerializeToNode(action, action.GetType(), LpcJson.Options),
                    });
                }
                var body = new JsonObject { ["submissionId"] = Guid.NewGuid().ToString(), ["submissions"] = submissions };
                try
                {
                    var result = await Request("PUT", path + "/action", body, true);
                    // Authoritative: the server says which turn it accepted our actions for. (The old local-clock
                    // estimate counted the whole reply round trip and flagged slow-server turns that were on time.)
                    bool late = false, reported = false;
                    if (Has(result, "accepted"))
                        foreach (var accepted in result["accepted"]!.AsArray())
                            if (accepted is JsonObject a && Has(a, "acceptedForTurn"))
                            {
                                reported = true;
                                if ((int)a["acceptedForTurn"]! > turn + 1) late = true;
                            }
                    if (!reported && Has(state, "turnEndsAtMs"))
                        late = (long)state["serverTimeMs"]! + ElapsedMs(stateReceivedAt) >= (long)state["turnEndsAtMs"]!;
                    if (late)
                    {
                        LateTurns++;
                        Console.Error.WriteLine($"Turn {turn} actions were accepted late (server applied them a turn later).");
                    }
                }
                catch (Exception error) when (error is not OutOfMemoryException)
                {
                    if (error is ApiError apiError && (apiError.Status is 401 or 403 || apiError.SessionConflict))
                    {
                        if (!apiError.InactiveMatch) throw;
                        if (await MatchEnded(path)) break;
                    }
                    Console.Error.WriteLine($"Could not submit turn {turn}: {error.Message}; continuing.");
                    try
                    {
                        var empty = new JsonObject { ["submissionId"] = Guid.NewGuid().ToString(), ["submissions"] = new JsonArray() };
                        await Request("PUT", path + "/action", empty, true);
                    }
                    catch (ApiError retryError)
                    {
                        if (retryError.Status is 401 or 403 || retryError.SessionConflict)
                        {
                            if (!retryError.InactiveMatch) throw;
                            if (await MatchEnded(path)) break;
                        }
                        Console.Error.WriteLine("Could not submit empty actions: " + retryError.Message);
                    }
                    catch (Exception retryError) when (retryError is not OutOfMemoryException)
                    {
                        Console.Error.WriteLine("Could not submit empty actions: " + retryError.Message);
                    }
                }
                submittedTurn = turn;
                if (feed != null)
                {
                    state = await feed.Next(state);
                    stateReceivedAt = Stopwatch.GetTimestamp();
                    StateObserver?.Invoke(state);
                    continue;
                }
                try
                {
                    state = await Request("GET", path + "/state", null, true);
                    stateReceivedAt = Stopwatch.GetTimestamp();
                    StateObserver?.Invoke(state);
                    if ((int)state["match"]!["turn"]! != turn) continue;
                }
                catch (Exception error) when (error is not OutOfMemoryException)
                {
                    if (error is ApiError api && (api.Status is 401 or 403 || api.SessionConflict)) throw;
                    Console.Error.WriteLine(error.Message);
                }
            }
            if (feed != null)
            {
                state = await feed.Next(state);
                stateReceivedAt = Stopwatch.GetTimestamp();
                StateObserver?.Invoke(state);
                continue;
            }
            long delay = realTime ? 20 : 100; // without the stream, poll real-time matches much faster
            if ((string?)match["turnAdvanceMode"] == "timed" && Has(state, "turnEndsAtMs"))
            {
                long remaining = (long)state["turnEndsAtMs"]! - (long)state["serverTimeMs"]! - ElapsedMs(stateReceivedAt);
                delay = remaining > 0 ? Math.Clamp(remaining - 25, 25, 1000) : 100;
            }
            await Task.Delay((int)delay);
            try
            {
                state = await Request("GET", path + "/state", null, true);
                stateReceivedAt = Stopwatch.GetTimestamp();
                StateObserver?.Invoke(state);
            }
            catch (Exception error) when (error is not OutOfMemoryException)
            {
                if (error is ApiError api && (api.Status is 401 or 403 || api.SessionConflict)) throw;
                Console.Error.WriteLine(error.Message);
            }
        }
        FinalStatus = (string?)state["match"]!["status"];
        Console.WriteLine(state["match"]!["cancellationReason"] is { } reason ? (string?)reason : $"Match {matchId} ended.");
    }

    static long ElapsedMs(long since) => (long)Stopwatch.GetElapsedTime(since).TotalMilliseconds;

    static IEnumerable<KeyValuePair<string, IAction>> Cast<T>(IReadOnlyDictionary<string, T>? actions) where T : IAction
    {
        if (actions == null) throw new ArgumentException("Strategy returned null.");
        return actions.Select(entry => new KeyValuePair<string, IAction>(entry.Key, entry.Value)).ToList();
    }
}
