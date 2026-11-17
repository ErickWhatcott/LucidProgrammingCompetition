namespace Lpc.Strategies.Planner;

/// <summary>
/// Dijkstra over the remembered map with 8-way moves. Costs are charged for the tile being
/// entered (mountain/crater = obstacle cost, which already includes the 2-turn cooldown).
/// Arrays are reused between turns to keep the per-turn decision well under the time budget.
/// </summary>
public sealed class PathField
{
    public float[] Dist = [];
    public int[] Prev = [];
    readonly PriorityQueue<int, float> queue = new();
    int width, height;

    void Reset(int w, int h)
    {
        if (Dist.Length != w * h) { Dist = new float[w * h]; Prev = new int[w * h]; }
        width = w; height = h;
        Array.Fill(Dist, float.PositiveInfinity);
        Array.Fill(Prev, -1);
        queue.Clear();
    }

    /// <summary>Forward search: Dist[i] = turns to reach i from source and be ready to act there.</summary>
    public void From(int source, float[] enterCost, int w, int h, float maxCost = float.PositiveInfinity)
    {
        Reset(w, h);
        Dist[source] = 0;
        queue.Enqueue(source, 0);
        while (queue.TryDequeue(out int c, out float d))
        {
            if (d != Dist[c]) continue;
            if (d > maxCost) break;
            int cx = c % w, cy = c / w;
            foreach (var (dx, dy, _) in Directions.All)
            {
                int nx = cx + dx, ny = cy + dy;
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                int n = ny * w + nx;
                float nd = d + enterCost[n];
                if (nd >= Dist[n]) continue;
                Dist[n] = nd;
                Prev[n] = c;
                queue.Enqueue(n, nd);
            }
        }
    }

    /// <summary>Reverse multi-source search: Dist[i] = turns to walk from i to the nearest source.</summary>
    public void To(IEnumerable<int> sources, float[] enterCost, int w, int h)
    {
        Reset(w, h);
        foreach (var s in sources) { Dist[s] = 0; queue.Enqueue(s, 0); }
        while (queue.TryDequeue(out int c, out float d))
        {
            if (d != Dist[c]) continue;
            int cx = c % w, cy = c / w;
            float step = enterCost[c]; // walking n -> c pays for entering c
            foreach (var (dx, dy, _) in Directions.All)
            {
                int nx = cx + dx, ny = cy + dy;
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                int n = ny * w + nx;
                if (float.IsInfinity(enterCost[n])) continue;
                float nd = d + step;
                if (nd >= Dist[n]) continue;
                Dist[n] = nd;
                Prev[n] = c;
                queue.Enqueue(n, nd);
            }
        }
    }

    /// <summary>First tile on the forward path from the search source to target, or -1.</summary>
    public int FirstStep(int target)
    {
        if (target < 0 || float.IsInfinity(Dist[target]) || Prev[target] < 0) return -1;
        int c = target;
        while (Prev[c] >= 0 && Prev[Prev[c]] >= 0) c = Prev[c];
        return c;
    }

    /// <summary>For a reverse field, the next tile toward the nearest source.</summary>
    public int StepToward(int from) => from >= 0 && from < Prev.Length ? Prev[from] : -1;
}
