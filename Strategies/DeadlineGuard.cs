using Lpc.Client;

namespace Lpc.Strategies;

/// <summary>
/// Never miss a turn because of our own computation. A cheap fallback bot sees every turn (so its memory is
/// current) and always has a move ready; the main engine runs with a time budget derived from the turn
/// deadline. If the engine overruns (GC pause, CPU contention), the fallback's move is submitted instead of
/// nothing. Matches without a deadline (practice) always wait for the engine.
/// </summary>
public sealed class DeadlineGuard(IMarsStrategy main, IMarsStrategy fallback) : IMarsStrategy
{
    Task<IReadOnlyDictionary<string, MarsAction>>? running;
    public IMarsStrategy Main => main;
    public int Fallbacks { get; private set; }
    public string Name => main.Name;

    /// <summary>Engine budget: time left to the deadline minus a network margin (~2 round trips), capped.</summary>
    public static double BudgetMs(GameState<MarsView> state)
    {
        if (Environment.GetEnvironmentVariable("LPC_GUARD_MS") is { } forced && double.TryParse(forced, out var ms)) return ms;
        if (state.TurnEndsAtMs is not { } ends) return double.PositiveInfinity;
        return Math.Clamp(ends - state.ServerTimeMs - 450, 30, 250);
    }

    public IReadOnlyDictionary<string, MarsAction> CalculateActions(GameState<MarsView> state, Identity identity)
    {
        var safe = fallback.CalculateActions(state, identity);
        if (running is { IsCompleted: false })
        {
            Fallbacks++; // the engine is still busy with an earlier turn
            return safe;
        }
        double budget = BudgetMs(state);
        running = Task.Run(() => main.CalculateActions(state, identity));
        try
        {
            if (double.IsPositiveInfinity(budget)) return running.Result;
            if (running.Wait(TimeSpan.FromMilliseconds(budget))) return running.Result;
        }
        catch (AggregateException error)
        {
            Console.Error.WriteLine($"Engine failed on turn {state.Match.Turn}: {error.InnerException?.Message}; using the fallback move.");
        }
        Fallbacks++;
        return safe;
    }

    public void OnMatchEnd(GameState<MarsView>? finalState, Identity identity)
    {
        try { running?.Wait(TimeSpan.FromSeconds(2)); } catch (AggregateException) { }
        main.OnMatchEnd(finalState, identity);
        fallback.OnMatchEnd(finalState, identity);
    }
}
