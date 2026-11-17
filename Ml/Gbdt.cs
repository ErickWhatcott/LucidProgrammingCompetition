using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;
using Lpc.Strategies.Planner;

namespace Lpc.Ml;

/// <summary>Options for <see cref="GbdtModel.Fit(IReadOnlyList{TripRecord}, GbdtOptions)"/>.</summary>
public sealed class GbdtOptions
{
    public int Trees = 300;
    public int MaxLeaves = 31;
    public int MaxDepth = 8;
    public int MinLeaf = 50;
    public float LearningRate = 0.08f;
    /// <summary>L2 regularization on leaf values.</summary>
    public float Lambda = 1f;
    /// <summary>Per-tree row subsample (Bernoulli).</summary>
    public double RowSample = 0.8;
    /// <summary>Per-tree feature subsample.</summary>
    public double FeatureSample = 0.8;
    /// <summary>Max histogram bins per feature (2..255).</summary>
    public int MaxBins = 64;
    /// <summary>Held-out share for early stopping (0 = none). Ignored for fewer than 500 rows.</summary>
    public double ValidationFraction = 0.1;
    /// <summary>Stop after this many trees without validation improvement (0 = never stop early).</summary>
    public int EarlyStopping = 20;
    public int Seed = 1;
    /// <summary>0 = squared error; &gt; 0 = Huber loss with this delta.</summary>
    public float HuberDelta = 0f;
    /// <summary>Minimum split gain (in G^2/H units).</summary>
    public double MinGain = 0;
    /// <summary>Optional per-record sample weight (non-finite or negative weights drop the row). Null = 1.</summary>
    public Func<TripRecord, float>? SampleWeight;
    /// <summary>Training threads (0 = all cores).</summary>
    public int Threads = 0;
    public Action<string>? Log;
}

/// <summary>
/// LightGBM-style histogram gradient-boosted regression trees in pure C#.
/// Training: per-feature quantile binning (max 64 bins), leaf-wise growth (best-gain leaf first) capped by
/// MaxLeaves/MaxDepth, histogram subtraction (build the smaller child, derive the larger), multi-threaded
/// histogram building over row chunks, row/feature subsampling per tree, L2 leaf regularization, early
/// stopping on a validation split (model truncated to the best iteration).
/// Inference: trees are stored as one flat node array (feature, threshold, left, right, value); one
/// model is read-only and safe to share across threads.
/// </summary>
public sealed class GbdtModel
{
    public const int FormatVersion = 1;

    [StructLayout(LayoutKind.Sequential)]
    struct Node { public int F; public float T; public int L; public int R; } // leaf: F = -1, T = value

    // Branch-free inference layout: a leaf is a self-loop (C0 = C1 = self, T = leaf value, F = 0), so every
    // row walks exactly depth[tree] steps with no data-dependent branches: next = C[x[F] > T].
    [StructLayout(LayoutKind.Sequential)]
    struct INode { public int F; public float T; public int C0; public int C1; }

    readonly Node[] nodes;
    readonly int[] roots;
    readonly INode[] inodes;
    readonly int[] depths;

    public float BaseScore { get; }
    public int TreeCount => roots.Length;
    public int NodeCount => nodes.Length;
    public int FeatureCount => Features.Count;
    /// <summary>Total split gain per feature (training only; zeros after Load if not stored).</summary>
    public double[] FeatureImportance { get; }
    public Dictionary<string, string> Metadata { get; } = new();

    /// <summary>Cached per-row inference cost seed for scorers (ms/row), measured once per model.</summary>
    internal double WarmMsPerRow;

    GbdtModel(float baseScore, Node[] nodes, int[] roots, double[] importance)
    {
        BaseScore = baseScore; this.nodes = nodes; this.roots = roots; FeatureImportance = importance;
        inodes = new INode[nodes.Length];
        for (int i = 0; i < nodes.Length; i++)
        {
            var n = nodes[i];
            inodes[i] = n.F < 0
                ? new INode { F = 0, T = n.T, C0 = i, C1 = i }
                : new INode { F = n.F, T = n.T, C0 = n.L, C1 = n.R };
        }
        // Longest root-to-leaf path per node by DP in reverse index order (children always have a larger
        // index than their parent: guaranteed by Fit and validated by Load). Linear time even for a
        // DAG-shaped (hand-edited/corrupt) file, where a path-enumerating walk would be exponential.
        var nodeDepth = new int[nodes.Length];
        for (int i = nodes.Length - 1; i >= 0; i--)
            nodeDepth[i] = nodes[i].F < 0 ? 0 : 1 + Math.Max(nodeDepth[nodes[i].L], nodeDepth[nodes[i].R]);
        depths = new int[roots.Length];
        for (int t = 0; t < roots.Length; t++) depths[t] = nodeDepth[roots[t]];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static int Step(ref INode n0, ref float x, int k)
    {
        ref INode nd = ref Unsafe.Add(ref n0, k);
        bool gt = Unsafe.Add(ref x, nd.F) > nd.T; // NaN -> left
        return Unsafe.Add(ref nd.C0, Unsafe.As<bool, byte>(ref gt));
    }

    // ------------------------------------------------------------------ inference

    /// <summary>rows is rowCount x Features.Count raw features (row-major); writes rowCount predictions.
    /// Thread-safe (read-only), allocation-free. x &lt;= threshold goes left; NaN goes left.</summary>
    public void Predict(ReadOnlySpan<float> rows, int rowCount, Span<float> output)
    {
        if (rowCount <= 0) return;
        int d = Features.Count;
        if (rows.Length < rowCount * d) throw new ArgumentException("Input too small.", nameof(rows));
        if (output.Length < rowCount) throw new ArgumentException("Output too small.", nameof(output));
        output[..rowCount].Fill(BaseScore);
        if (roots.Length == 0) return;
        ref INode n0 = ref MemoryMarshal.GetArrayDataReference(inodes);
        ref float x0 = ref MemoryMarshal.GetReference(rows);
        ref float o0 = ref MemoryMarshal.GetReference(output);
        for (int t = 0; t < roots.Length; t++)
        {
            int root = roots[t], dep = depths[t];
            int r = 0;
            // Eight rows in lock-step: independent dependency chains keep the core busy.
            for (; r + 8 <= rowCount; r += 8)
            {
                ref float xa = ref Unsafe.Add(ref x0, r * d);
                ref float xb = ref Unsafe.Add(ref xa, d);
                ref float xc = ref Unsafe.Add(ref xb, d);
                ref float xd = ref Unsafe.Add(ref xc, d);
                ref float xe = ref Unsafe.Add(ref xd, d);
                ref float xf = ref Unsafe.Add(ref xe, d);
                ref float xg = ref Unsafe.Add(ref xf, d);
                ref float xh = ref Unsafe.Add(ref xg, d);
                int ka = root, kb = root, kc = root, kd = root, ke = root, kf = root, kg = root, kh = root;
                for (int l = 0; l < dep; l++)
                {
                    ka = Step(ref n0, ref xa, ka); kb = Step(ref n0, ref xb, kb);
                    kc = Step(ref n0, ref xc, kc); kd = Step(ref n0, ref xd, kd);
                    ke = Step(ref n0, ref xe, ke); kf = Step(ref n0, ref xf, kf);
                    kg = Step(ref n0, ref xg, kg); kh = Step(ref n0, ref xh, kh);
                }
                ref float o = ref Unsafe.Add(ref o0, r);
                o += Unsafe.Add(ref n0, ka).T;
                Unsafe.Add(ref o, 1) += Unsafe.Add(ref n0, kb).T;
                Unsafe.Add(ref o, 2) += Unsafe.Add(ref n0, kc).T;
                Unsafe.Add(ref o, 3) += Unsafe.Add(ref n0, kd).T;
                Unsafe.Add(ref o, 4) += Unsafe.Add(ref n0, ke).T;
                Unsafe.Add(ref o, 5) += Unsafe.Add(ref n0, kf).T;
                Unsafe.Add(ref o, 6) += Unsafe.Add(ref n0, kg).T;
                Unsafe.Add(ref o, 7) += Unsafe.Add(ref n0, kh).T;
            }
            for (; r < rowCount; r++)
            {
                ref float xr = ref Unsafe.Add(ref x0, r * d);
                int k = root;
                for (int l = 0; l < dep; l++) k = Step(ref n0, ref xr, k);
                Unsafe.Add(ref o0, r) += Unsafe.Add(ref n0, k).T;
            }
        }
    }

    public float PredictRow(ReadOnlySpan<float> row)
    {
        if (row.Length < Features.Count) throw new ArgumentException("Row too small.", nameof(row));
        ref INode n0 = ref MemoryMarshal.GetArrayDataReference(inodes);
        ref float x = ref MemoryMarshal.GetReference(row);
        float s = BaseScore;
        for (int t = 0; t < roots.Length; t++)
        {
            int k = roots[t];
            for (int l = 0; l < depths[t]; l++) k = Step(ref n0, ref x, k);
            s += Unsafe.Add(ref n0, k).T;
        }
        return s;
    }

    /// <summary>(Weighted) mean squared error on raw rows.</summary>
    public double Evaluate(float[] x, float[] y, int rows, float[]? w = null)
    {
        if (rows <= 0) return double.NaN;
        var p = new float[rows];
        Predict(x, rows, p);
        double s = 0, ws = 0;
        for (int i = 0; i < rows; i++) { double wi = w?[i] ?? 1.0, e = p[i] - y[i]; s += wi * e * e; ws += wi; }
        return ws > 0 ? s / ws : double.NaN;
    }

    // ------------------------------------------------------------------ training

    /// <summary>Trains on trip records (rows with the wrong length / non-finite values are dropped).
    /// A random ValidationFraction of rows is held out for early stopping.</summary>
    public static GbdtModel Fit(IReadOnlyList<TripRecord> data, GbdtOptions o)
    {
        int d = Features.Count;
        var keep = new List<int>(data.Count);
        var wAll = new float[data.Count];
        bool anyWeight = false;
        for (int i = 0; i < data.Count; i++)
        {
            var r = data[i];
            if (r?.F is null || r.F.Length != d || !float.IsFinite(r.Y)) continue;
            bool ok = true;
            foreach (var v in r.F) if (!float.IsFinite(v)) { ok = false; break; }
            if (!ok) continue;
            float w = o.SampleWeight?.Invoke(r) ?? 1f;
            if (!float.IsFinite(w) || w < 0) continue;
            if (w != 1f) anyWeight = true;
            wAll[i] = w;
            keep.Add(i);
        }
        if (keep.Count < 2) throw new ArgumentException("Need at least 2 valid rows to fit a GBDT.", nameof(data));
        if (keep.Count < data.Count) o.Log?.Invoke($"gbdt: dropped {data.Count - keep.Count} invalid rows");

        var rng = new Random(o.Seed);
        var order = keep.ToArray();
        for (int i = order.Length - 1; i > 0; i--) { int j = rng.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }
        int nv = keep.Count >= 500 && o.ValidationFraction > 0
            ? Math.Clamp((int)(keep.Count * o.ValidationFraction), 1, keep.Count - 2) : 0;

        TrainingSet Pack(ReadOnlySpan<int> ids)
        {
            var x = new float[ids.Length * d];
            var y = new float[ids.Length];
            var w = anyWeight ? new float[ids.Length] : null;
            for (int k = 0; k < ids.Length; k++)
            {
                var r = data[ids[k]];
                r.F.AsSpan().CopyTo(x.AsSpan(k * d, d));
                y[k] = r.Y;
                if (w != null) w[k] = wAll[ids[k]];
            }
            return new TrainingSet(x, y, w, ids.Length);
        }
        var val = nv > 0 ? Pack(order.AsSpan(0, nv)) : null;
        var train = Pack(order.AsSpan(nv));
        return Fit(train, val, o);
    }

    sealed class BNode { public int F = -1, Bin, L = -1, R = -1; public float Thr, Value; }

    struct Split { public int F, Bin; public double Gain, GL, HL, CL; }

    sealed class Leaf
    {
        public int Start, End, Depth, Node;
        public double G, H, C;
        public double[] Hist = null!;
        public Split Best;
    }

    /// <summary>Trains on explicit matrices (row-major, Features.Count columns). val may be null (no early stopping).</summary>
    public static unsafe GbdtModel Fit(TrainingSet train, TrainingSet? val, GbdtOptions o)
    {
        var sw = Stopwatch.StartNew();
        int d = Features.Count, n = train.Rows;
        if (n < 2) throw new ArgumentException("Need at least 2 training rows.", nameof(train));
        if (train.X.Length < n * d) throw new ArgumentException("train.X too small.", nameof(train));
        int nv = val?.Rows ?? 0;
        bool useVal = nv > 0 && o.EarlyStopping > 0;
        int B = Math.Clamp(o.MaxBins, 2, 255);
        int maxLeaves = Math.Max(2, o.MaxLeaves), maxDepth = Math.Max(1, o.MaxDepth);
        int minLeaf = Math.Max(1, o.MinLeaf);
        double lambda = Math.Max(0, o.Lambda);
        float lr = o.LearningRate;
        var po = new ParallelOptions { MaxDegreeOfParallelism = o.Threads > 0 ? o.Threads : Environment.ProcessorCount };
        int threads = po.MaxDegreeOfParallelism;
        var X = train.X; var Y = train.Y; var W = train.Weights;

        // ---- binning
        var cuts = new float[d][];
        Parallel.For(0, d, po, f => cuts[f] = MakeCuts(X, n, d, f, B));
        // column-major bins: binned[f * n + i]
        var binned = new byte[(long)n * d];
        Parallel.For(0, d, po, f =>
        {
            var c = cuts[f]; long off = (long)f * n;
            for (int i = 0; i < n; i++) binned[off + i] = BinOf(c, X[(long)i * d + f]);
        });
        byte[]? vbinned = null;
        if (nv > 0)
        {
            vbinned = new byte[(long)nv * d];
            var VX = val!.X;
            Parallel.For(0, d, po, f =>
            {
                var c = cuts[f]; long off = (long)f * nv;
                for (int i = 0; i < nv; i++) vbinned[off + i] = BinOf(c, VX[(long)i * d + f]);
            });
        }
        var usable = Enumerable.Range(0, d).Where(f => cuts[f].Length > 0).ToArray();

        // ---- init
        double sy = 0, sw0 = 0;
        for (int i = 0; i < n; i++) { double wi = W?[i] ?? 1.0; sy += wi * Y[i]; sw0 += wi; }
        float baseScore = sw0 > 0 ? (float)(sy / sw0) : 0f;
        var pred = new float[n]; Array.Fill(pred, baseScore);
        var vpred = new float[nv]; Array.Fill(vpred, baseScore);
        var grad = new float[n];
        var hess = new float[n];
        var idx = new int[n];
        var tmpIdx = new int[n];
        long tPart = 0, tHist = 0, tUpd = 0, tPrep = 0;
        int hsz = d * B * 3;
        var histPool = new Stack<double[]>();
        double[] RentHist() => histPool.Count > 0 ? histPool.Pop() : new double[hsz];
        var og = new float[n];
        var oh = new float[n];

        double ValMse()
        {
            double s = 0, ws = 0;
            for (int i = 0; i < nv; i++) { double wi = val!.Weights?[i] ?? 1.0, e = vpred[i] - val.Y[i]; s += wi * e * e; ws += wi; }
            return ws > 0 ? s / ws : double.NaN;
        }
        double TrainMse()
        {
            double s = 0, ws = 0;
            for (int i = 0; i < n; i++) { double wi = W?[i] ?? 1.0, e = pred[i] - Y[i]; s += wi * e * e; ws += wi; }
            return ws > 0 ? s / ws : double.NaN;
        }
        double baseVal = useVal ? ValMse() : double.NaN;
        double bestVal = useVal ? baseVal : double.PositiveInfinity;
        int bestIter = 0;

        var trees = new List<BNode[]>();
        var treeGain = new List<double[]>();
        ulong rs = (ulong)(o.Seed * 2654435761L + 12345) | 1;
        var featRng = new Random(o.Seed + 7);
        float delta = o.HuberDelta;
        o.Log?.Invoke($"gbdt fit: rows={n} val={nv} features={usable.Length}/{d} bins<={B} leaves={maxLeaves} depth={maxDepth} " +
                      $"minLeaf={minLeaf} lr={lr} lambda={lambda} rowSample={o.RowSample} featSample={o.FeatureSample} " +
                      $"loss={(delta > 0 ? $"huber({delta})" : "mse")} threads={threads}" + (useVal ? $" baseValMse={baseVal:G5}" : ""));

        // Builds the histogram of idx[s..e) over feats into hist (cleared first).
        void BuildHist(int s, int e, int[] feats, double[] hist)
        {
            long th = Stopwatch.GetTimestamp();
            try { BuildHistCore(s, e, feats, hist); } finally { tHist += Stopwatch.GetTimestamp() - th; }
        }
        void BuildHistCore(int s, int e, int[] feats, double[] hist)
        {
            Array.Clear(hist);
            int cnt = e - s;
            // ordered gradients: og[k] = grad[idx[k]] so each feature pass streams them sequentially
            for (int k = s; k < e; k++) { int i = idx[k]; og[k] = grad[i]; oh[k] = hess[i]; }
            if ((long)cnt * feats.Length < 400_000) // thread wake-up on macOS costs more than small leaves
            {
                foreach (int f in feats) AccumulateFeature(binned, (long)f * n, idx, og, oh, s, e, hist, f * B * 3);
                return;
            }
            Parallel.For(0, feats.Length, po, j =>
            {
                int f = feats[j];
                AccumulateFeature(binned, (long)f * n, idx, og, oh, s, e, hist, f * B * 3);
            });
        }

        void FindBest(Leaf leaf, int[] feats)
        {
            var best = new Split { F = -1, Gain = Math.Max(o.MinGain, 1e-12) };
            double parent = leaf.G * leaf.G / (leaf.H + lambda);
            var h = leaf.Hist;
            foreach (int f in feats)
            {
                int nb = cuts[f].Length + 1;
                double gl = 0, hl = 0, cl = 0;
                int off = f * B * 3;
                for (int b = 0; b < nb - 1; b++)
                {
                    gl += h[off + b * 3]; hl += h[off + b * 3 + 1]; cl += h[off + b * 3 + 2];
                    if (cl < minLeaf) continue;
                    double cr = leaf.C - cl;
                    if (cr < minLeaf) break;
                    double hr = leaf.H - hl, gr = leaf.G - gl;
                    if (hl < 1e-9 || hr < 1e-9) continue;
                    double gain = gl * gl / (hl + lambda) + gr * gr / (hr + lambda) - parent;
                    if (gain > best.Gain) best = new Split { F = f, Bin = b, Gain = gain, GL = gl, HL = hl, CL = cl };
                }
            }
            leaf.Best = best;
        }

        int maxTrees = Math.Max(0, o.Trees);
        for (int iter = 1; iter <= maxTrees; iter++)
        {
            long tg = Stopwatch.GetTimestamp();
            // gradients
            ParallelChunks(n, 16384, po, (s, e) =>
            {
                for (int i = s; i < e; i++)
                {
                    float wi = W?[i] ?? 1f, err = pred[i] - Y[i];
                    grad[i] = wi * (delta > 0 ? Math.Clamp(err, -delta, delta) : err);
                    hess[i] = wi;
                }
            });
            // row sample
            int m = 0;
            if (o.RowSample >= 1) { for (int i = 0; i < n; i++) idx[m++] = i; }
            else
            {
                ulong thr = (ulong)(Math.Clamp(o.RowSample, 0.01, 1.0) * ulong.MaxValue);
                for (int i = 0; i < n; i++)
                {
                    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
                    if (rs <= thr) idx[m++] = i;
                }
            }
            if (m < 2 * minLeaf) { for (int i = 0; i < n; i++) idx[i] = i; m = n; }
            // feature sample
            var fs = (int[])usable.Clone();
            for (int i = fs.Length - 1; i > 0; i--) { int j = featRng.Next(i + 1); (fs[i], fs[j]) = (fs[j], fs[i]); }
            int kf = Math.Clamp((int)Math.Round(fs.Length * Math.Clamp(o.FeatureSample, 0.01, 1.0)), 1, Math.Max(1, fs.Length));
            var feats = fs.Length == 0 ? [] : fs[..kf];
            Array.Sort(feats);

            tPrep += Stopwatch.GetTimestamp() - tg;
            // grow
            var tn = new List<BNode> { new() };
            var gains = new double[d];
            var root = new Leaf { Start = 0, End = m, Depth = 0, Node = 0, Hist = RentHist() };
            BuildHist(0, m, feats, root.Hist);
            if (feats.Length > 0)
            {
                int f0 = feats[0], off0 = f0 * B * 3;
                for (int b = 0; b < cuts[f0].Length + 1; b++) { root.G += root.Hist[off0 + b * 3]; root.H += root.Hist[off0 + b * 3 + 1]; root.C += root.Hist[off0 + b * 3 + 2]; }
            }
            else
            {
                for (int k = 0; k < m; k++) { root.G += grad[idx[k]]; root.H += hess[idx[k]]; root.C++; }
            }
            FindBest(root, feats);
            var leaves = new List<Leaf> { root };
            while (leaves.Count < maxLeaves)
            {
                Leaf? pick = null;
                foreach (var l in leaves)
                    if (l.Best.F >= 0 && l.Depth < maxDepth && (pick == null || l.Best.Gain > pick.Best.Gain)) pick = l;
                if (pick == null) break;
                var sp = pick.Best;
                // stable partition of idx[Start..End) by bin <= sp.Bin (keeps row order ascending -> sequential memory access)
                long tp = Stopwatch.GetTimestamp();
                int i0;
                int plen = pick.End - pick.Start;
                if (plen < 200_000 || threads <= 1)
                {
                    i0 = pick.Start; int nr = 0;
                    for (int k = pick.Start; k < pick.End; k++)
                    {
                        int row = idx[k];
                        if (binned[(long)sp.F * n + row] <= sp.Bin) idx[i0++] = row;
                        else tmpIdx[nr++] = row;
                    }
                    Array.Copy(tmpIdx, 0, idx, i0, nr);
                }
                else
                {
                    // parallel stable partition: count per chunk, prefix offsets, scatter into tmpIdx, copy back
                    int pc = Math.Min(threads * 2, plen / 50_000), ps = pick.Start, pBin = sp.Bin;
                    long colOff = (long)sp.F * n;
                    var lefts = new int[pc];
                    Parallel.For(0, pc, po, c =>
                    {
                        int a = ps + (int)((long)plen * c / pc), b = ps + (int)((long)plen * (c + 1) / pc), cnt = 0;
                        for (int k = a; k < b; k++) if (binned[colOff + idx[k]] <= pBin) cnt++;
                        lefts[c] = cnt;
                    });
                    int totalLeft = 0;
                    var lo = new int[pc]; var ro = new int[pc];
                    for (int c = 0; c < pc; c++) { lo[c] = totalLeft; totalLeft += lefts[c]; }
                    for (int c = 0, acc = totalLeft; c < pc; c++)
                    {
                        ro[c] = acc;
                        int a = (int)((long)plen * c / pc), b = (int)((long)plen * (c + 1) / pc);
                        acc += (b - a) - lefts[c];
                    }
                    Parallel.For(0, pc, po, c =>
                    {
                        int a = ps + (int)((long)plen * c / pc), b = ps + (int)((long)plen * (c + 1) / pc);
                        int l = lo[c], r = ro[c];
                        for (int k = a; k < b; k++)
                        {
                            int row = idx[k];
                            if (binned[colOff + row] <= pBin) tmpIdx[l++] = row; else tmpIdx[r++] = row;
                        }
                    });
                    Array.Copy(tmpIdx, 0, idx, ps, plen);
                    i0 = ps + totalLeft;
                }
                tPart += Stopwatch.GetTimestamp() - tp;
                var left = new Leaf { Start = pick.Start, End = i0, Depth = pick.Depth + 1, G = sp.GL, H = sp.HL, C = sp.CL };
                var right = new Leaf { Start = i0, End = pick.End, Depth = pick.Depth + 1, G = pick.G - sp.GL, H = pick.H - sp.HL, C = pick.C - sp.CL };
                var small = left.End - left.Start <= right.End - right.Start ? left : right;
                var large = ReferenceEquals(small, left) ? right : left;
                small.Hist = RentHist();
                BuildHist(small.Start, small.End, feats, small.Hist);
                large.Hist = pick.Hist; // parent - small
                foreach (int f in feats)
                {
                    int off = f * B * 3, len = B * 3;
                    for (int k = 0; k < len; k++) large.Hist[off + k] -= small.Hist[off + k];
                }
                var pn = tn[pick.Node];
                pn.F = sp.F; pn.Bin = sp.Bin; pn.Thr = cuts[sp.F][sp.Bin];
                left.Node = tn.Count; tn.Add(new BNode());
                right.Node = tn.Count; tn.Add(new BNode());
                pn.L = left.Node; pn.R = right.Node;
                gains[sp.F] += sp.Gain;
                leaves.Remove(pick);
                if (left.Depth < maxDepth) FindBest(left, feats); else left.Best.F = -1;
                if (right.Depth < maxDepth) FindBest(right, feats); else right.Best.F = -1;
                leaves.Add(left); leaves.Add(right);
            }
            foreach (var l in leaves)
            {
                tn[l.Node].Value = (float)(-l.G / (l.H + lambda) * lr);
                histPool.Push(l.Hist);
            }
            var tree = tn.ToArray();
            trees.Add(tree);
            treeGain.Add(gains);

            long tu = Stopwatch.GetTimestamp();
            // update predictions via binned traversal
            ParallelChunks(n, 16384, po, (s, e) =>
            {
                for (int i = s; i < e; i++) pred[i] += Walk(tree, binned, n, i);
            });
            if (nv > 0)
                ParallelChunks(nv, 16384, po, (s, e) =>
                {
                    for (int i = s; i < e; i++) vpred[i] += Walk(tree, vbinned!, nv, i);
                });

            double v = useVal ? ValMse() : double.NaN;
            tUpd += Stopwatch.GetTimestamp() - tu;
            if (useVal)
            {
                if (v < bestVal - 1e-12) { bestVal = v; bestIter = iter; }
            }
            else bestIter = iter;
            if (o.Log != null && (iter % 25 == 0 || iter == 1))
                o.Log($"gbdt tree {iter}: leaves={leaves.Count} train={TrainMse():G5}" + (useVal ? $" val={v:G5} best={bestVal:G5}@{bestIter}" : "") +
                      $" t={sw.Elapsed.TotalSeconds:F1}s");
            if (useVal && iter - bestIter >= o.EarlyStopping)
            {
                o.Log?.Invoke($"gbdt early stop at {iter}: best val {bestVal:G5} at tree {bestIter}");
                break;
            }
        }

        // ---- assemble flat model from the first bestIter trees
        int keepTrees = bestIter;
        var roots = new int[keepTrees];
        var flat = new List<Node>();
        var imp = new double[d];
        for (int t = 0; t < keepTrees; t++)
        {
            var tr = trees[t];
            int baseIdx = flat.Count;
            roots[t] = baseIdx;
            foreach (var bn in tr)
                flat.Add(bn.F < 0
                    ? new Node { F = -1, T = bn.Value, L = -1, R = -1 }
                    : new Node { F = bn.F, T = bn.Thr, L = baseIdx + bn.L, R = baseIdx + bn.R });
            for (int f = 0; f < d; f++) imp[f] += treeGain[t][f];
        }
        var model = new GbdtModel(baseScore, flat.ToArray(), roots, imp);
        var inv = CultureInfo.InvariantCulture;
        model.Metadata["engine"] = "gbdt";
        model.Metadata["trees"] = keepTrees.ToString(inv);
        model.Metadata["treesBuilt"] = trees.Count.ToString(inv);
        model.Metadata["trainRows"] = n.ToString(inv);
        model.Metadata["valRows"] = nv.ToString(inv);
        if (useVal)
        {
            model.Metadata["baseValMse"] = baseVal.ToString("G6", inv);
            model.Metadata["bestValMse"] = bestVal.ToString("G6", inv);
        }
        model.Metadata["fitSeconds"] = sw.Elapsed.TotalSeconds.ToString("F2", inv);
        model.Metadata["trainedUtc"] = DateTime.UtcNow.ToString("o", inv);
        double totImp = imp.Sum();
        var top = Enumerable.Range(0, d).OrderByDescending(f => imp[f]).Take(6)
            .Select(f => $"{Features.Names[f]}:{(totImp > 0 ? imp[f] / totImp : 0):P0}");
        model.Metadata["topFeatures"] = string.Join(' ', top);
        o.Log?.Invoke($"gbdt done: trees={keepTrees}/{trees.Count} nodes={flat.Count} " +
                      (useVal ? $"baseValMse={baseVal:G5} bestValMse={bestVal:G5} " : "") +
                      $"time={sw.Elapsed.TotalSeconds:F1}s (prep {Sec(tPrep):F1} hist {Sec(tHist):F1} part {Sec(tPart):F1} update {Sec(tUpd):F1}) top=[{model.Metadata["topFeatures"]}]");
        return model;
    }

    static double Sec(long ticks) => ticks / (double)Stopwatch.Frequency;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float Walk(BNode[] tree, byte[] bins, int stride, int i)
    {
        var nd = tree[0];
        while (nd.F >= 0) nd = tree[bins[(long)nd.F * stride + i] <= nd.Bin ? nd.L : nd.R];
        return nd.Value;
    }

    static unsafe void AccumulateFeature(byte[] binned, long colOff, int[] idx, float[] og, float[] oh, int s, int e, double[] hist, int hOff)
    {
        fixed (byte* bp = binned)
        fixed (int* ip = idx)
        fixed (float* gp = og, hp = oh)
        fixed (double* hb = hist)
        {
            byte* col = bp + colOff;
            double* H = hb + hOff;
            for (int k = s; k < e; k++)
            {
                double* cell = H + col[ip[k]] * 3;
                cell[0] += gp[k]; cell[1] += hp[k]; cell[2] += 1.0;
            }
        }
    }

    static void ParallelChunks(int n, int minChunk, ParallelOptions po, Action<int, int> body)
    {
        int chunks = Math.Clamp(n / Math.Max(1, minChunk), 1, po.MaxDegreeOfParallelism * 4);
        if (chunks <= 1) { body(0, n); return; }
        Parallel.For(0, chunks, po, c => body((int)((long)n * c / chunks), (int)((long)n * (c + 1) / chunks)));
    }

    /// <summary>Number of cuts strictly below v (bin index). v &lt;= cuts[b] iff bin &lt;= b.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static byte BinOf(float[] c, float v)
    {
        int lo = 0, hi = c.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (c[mid] < v) lo = mid + 1; else hi = mid;
        }
        return (byte)lo;
    }

    static float Mid(float a, float b)
    {
        float m = a + (b - a) * 0.5f;
        return m >= a && m < b ? m : a;
    }

    static float[] MakeCuts(float[] X, int n, int d, int f, int maxBins)
    {
        const int maxSample = 200_000;
        int sN = Math.Min(n, maxSample);
        var vals = new float[sN];
        for (int k = 0; k < sN; k++)
        {
            long i = sN == n ? k : (long)k * n / sN;
            vals[k] = X[i * d + f];
        }
        Array.Sort(vals);
        var distinct = new List<float>();
        for (int k = 0; k < sN; k++) if (k == 0 || vals[k] != vals[k - 1]) distinct.Add(vals[k]);
        if (distinct.Count <= 1) return [];
        var cuts = new List<float>();
        if (distinct.Count <= maxBins)
        {
            for (int k = 0; k + 1 < distinct.Count; k++) cuts.Add(Mid(distinct[k], distinct[k + 1]));
        }
        else
        {
            for (int q = 1; q < maxBins; q++)
            {
                float v = vals[(long)q * sN / maxBins];
                int p = distinct.BinarySearch(v);
                if (p < 0 || p + 1 >= distinct.Count) continue;
                float c = Mid(v, distinct[p + 1]);
                if (cuts.Count == 0 || c > cuts[^1]) cuts.Add(c);
            }
        }
        return cuts.ToArray();
    }

    // ------------------------------------------------------------------ persistence

    sealed class Dto
    {
        public int Version { get; set; }
        public int FeatureVersion { get; set; }
        public int FeatureCount { get; set; }
        public string[] FeatureNames { get; set; } = [];
        public float BaseScore { get; set; }
        public int[] Roots { get; set; } = [];
        public int[] Feature { get; set; } = [];
        public float[] Threshold { get; set; } = [];
        public int[] Left { get; set; } = [];
        public int[] Right { get; set; } = [];
        public float[] Value { get; set; } = [];
        public double[] Importance { get; set; } = [];
        public Dictionary<string, string> Metadata { get; set; } = new();
    }

    static readonly JsonSerializerOptions JsonOpts = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = false };

    public void Save(string path)
    {
        int c = nodes.Length;
        var dto = new Dto
        {
            Version = FormatVersion,
            FeatureVersion = Features.Version,
            FeatureCount = Features.Count,
            FeatureNames = Features.Names,
            BaseScore = BaseScore,
            Roots = roots,
            Feature = new int[c], Threshold = new float[c], Left = new int[c], Right = new int[c], Value = new float[c],
            Importance = FeatureImportance,
            Metadata = Metadata,
        };
        for (int i = 0; i < c; i++)
        {
            var nd = nodes[i];
            dto.Feature[i] = nd.F;
            if (nd.F < 0) { dto.Value[i] = nd.T; dto.Left[i] = -1; dto.Right[i] = -1; }
            else { dto.Threshold[i] = nd.T; dto.Left[i] = nd.L; dto.Right[i] = nd.R; }
        }
        var dir = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        var tmp = path + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(dto, JsonOpts));
        File.Move(tmp, path, overwrite: true);
    }

    /// <summary>Loads a model; throws InvalidDataException on a format/feature-layout mismatch or a corrupt tree.</summary>
    public static GbdtModel Load(string path)
    {
        Dto? dto;
        try { dto = JsonSerializer.Deserialize<Dto>(File.ReadAllText(path), JsonOpts); }
        catch (JsonException e) { throw new InvalidDataException($"{path}: invalid gbdt JSON: {e.Message}", e); }
        if (dto is null) throw new InvalidDataException($"{path}: empty model file.");
        if (dto.Roots is null || dto.Feature is null || dto.Threshold is null || dto.Left is null || dto.Right is null || dto.Value is null)
            throw new InvalidDataException($"{path}: missing tree arrays.");
        dto.FeatureNames ??= [];
        dto.Importance ??= [];
        dto.Metadata ??= new();
        if (dto.Version != FormatVersion) throw new InvalidDataException($"{path}: gbdt format v{dto.Version}, expected v{FormatVersion}.");
        if (dto.FeatureVersion != Features.Version) throw new InvalidDataException($"{path}: featureVersion {dto.FeatureVersion} != current {Features.Version}.");
        if (dto.FeatureCount != Features.Count) throw new InvalidDataException($"{path}: featureCount {dto.FeatureCount} != Features.Count {Features.Count}.");
        if (dto.FeatureNames.Length > 0 && !dto.FeatureNames.SequenceEqual(Features.Names))
            throw new InvalidDataException($"{path}: feature names differ from the current Features.Names.");
        int c = dto.Feature.Length;
        if (dto.Threshold.Length != c || dto.Left.Length != c || dto.Right.Length != c || dto.Value.Length != c)
            throw new InvalidDataException($"{path}: inconsistent node arrays.");
        if (!float.IsFinite(dto.BaseScore)) throw new InvalidDataException($"{path}: non-finite base score.");
        var nodes = new Node[c];
        for (int i = 0; i < c; i++)
        {
            int f = dto.Feature[i];
            if (f < 0)
            {
                if (!float.IsFinite(dto.Value[i])) throw new InvalidDataException($"{path}: node {i} non-finite value.");
                nodes[i] = new Node { F = -1, T = dto.Value[i], L = -1, R = -1 };
            }
            else
            {
                // children must come after the parent: guarantees termination
                if (f >= Features.Count || float.IsNaN(dto.Threshold[i]) ||
                    dto.Left[i] <= i || dto.Left[i] >= c || dto.Right[i] <= i || dto.Right[i] >= c)
                    throw new InvalidDataException($"{path}: node {i} is corrupt.");
                nodes[i] = new Node { F = f, T = dto.Threshold[i], L = dto.Left[i], R = dto.Right[i] };
            }
        }
        foreach (int r in dto.Roots)
            if (r < 0 || r >= c) throw new InvalidDataException($"{path}: bad tree root {r}.");
        var imp = dto.Importance.Length == Features.Count ? dto.Importance : new double[Features.Count];
        var m = new GbdtModel(dto.BaseScore, nodes, dto.Roots, imp);
        foreach (var kv in dto.Metadata) m.Metadata[kv.Key] = kv.Value;
        return m;
    }
}
