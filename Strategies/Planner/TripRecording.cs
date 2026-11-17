namespace Lpc.Strategies.Planner;

/// <summary>
/// One training example: the features of the candidate a worker chose, labelled with the
/// gold-equivalent it actually banked per turn until its next deposit (0 if it never banked).
/// Source is "sim" for simulator self-play or "live" for scrimmage/ranked recordings.
/// </summary>
public sealed record TripRecord(float[] F, float Y, string Src);

/// <summary>Sink for training examples. Implementations must be thread-safe (self-play runs games in parallel).</summary>
public interface ITripRecorder
{
    void Add(TripRecord record);
}
