using System.Text.Json;

namespace Lpc.Ml.Es;

/// <summary>
/// (mu/mu_w, lambda)-CMA-ES with full covariance, rank-one + rank-mu updates and cumulative step-size
/// adaptation (Hansen's tutorial defaults). MAXIMIZES fitness. Gradient-free: fitness can be noisy
/// game results.
///
/// Usage: <c>var xs = es.Ask(); ... evaluate ...; es.Tell(xs, fitness);</c>. Tell recomputes each
/// sample's step from (x - mean) / sigma, so pass the exact arrays Ask returned. Ask clamps samples
/// into [<see cref="LowerBound"/>, <see cref="UpperBound"/>] (default [0,1], the normalized genome box)
/// so the mean, a convex combination of samples, never drifts outside the box where fitness is flat
/// (the decoder clamps too). Set the bounds to +-infinity for an unbounded problem.
///
/// The full state, including the RNG, round-trips through <see cref="ToJson"/>/<see cref="FromJson"/>,
/// so a resumed run produces exactly the same samples as an uninterrupted one.
///
/// Fitness used by the league (for the caller): fitness of a candidate = mean over its games of
/// (seat banked gold - mean banked gold of all seats in that game). Subtracting the per-game mean
/// removes map-difficulty noise; play every candidate on several seeds if budget allows.
/// </summary>
public sealed class CmaEs
{
    // Strategy parameters (derived from N and Lambda).
    public int Dimension { get; }
    public int Lambda { get; }
    public int Mu { get; }
    readonly double[] w;
    readonly double mueff, cc, cs, c1, cmu, damps, chiN;
    readonly int eigenEvery;

    // State.
    public int Generation { get; private set; }
    public double Sigma { get; private set; }
    public double[] Mean { get; }
    public double[]? BestX { get; private set; }
    public double BestFitness { get; private set; } = double.NegativeInfinity;
    /// <summary>Upper bound on sigma (the normalized search box is [0,1]^N, so steps > 1 are meaningless).</summary>
    public double MaxSigma { get; set; } = 1.0;
    /// <summary>Per-coordinate box applied to samples in <see cref="Ask"/>.</summary>
    public double LowerBound { get; set; } = 0.0;
    public double UpperBound { get; set; } = 1.0;

    readonly double[,] c, b; // covariance, eigenvectors (columns)
    readonly double[] d;     // sqrt(eigenvalues)
    readonly double[] pc, ps;
    int lastEigenGen;
    readonly ulong[] rng = new ulong[4];

    public CmaEs(double[] initialMean, double initialSigma, int lambda = 12, int seed = 1)
    {
        ArgumentNullException.ThrowIfNull(initialMean);
        if (initialMean.Length < 1) throw new ArgumentException("Empty mean.", nameof(initialMean));
        if (lambda < 4) throw new ArgumentOutOfRangeException(nameof(lambda), "lambda must be >= 4.");
        if (!(initialSigma > 0)) throw new ArgumentOutOfRangeException(nameof(initialSigma));
        int n = Dimension = initialMean.Length;
        Lambda = lambda;
        Mu = lambda / 2;
        w = new double[Mu];
        double sw = 0, sw2 = 0;
        for (int i = 0; i < Mu; i++) { w[i] = Math.Log(Mu + 0.5) - Math.Log(i + 1); sw += w[i]; }
        for (int i = 0; i < Mu; i++) { w[i] /= sw; sw2 += w[i] * w[i]; }
        mueff = 1 / sw2;
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n);
        cs = (mueff + 2) / (n + mueff + 5);
        c1 = 2 / ((n + 1.3) * (n + 1.3) + mueff);
        cmu = Math.Min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) * (n + 2) + mueff));
        damps = 1 + 2 * Math.Max(0, Math.Sqrt((mueff - 1) / (n + 1)) - 1) + cs;
        chiN = Math.Sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21.0 * n * n));
        eigenEvery = Math.Max(1, (int)(1 / ((c1 + cmu) * n * 10)));

        Mean = (double[])initialMean.Clone();
        Sigma = initialSigma;
        c = new double[n, n]; b = new double[n, n]; d = new double[n];
        for (int i = 0; i < n; i++) { c[i, i] = 1; b[i, i] = 1; d[i] = 1; }
        pc = new double[n]; ps = new double[n];
        SeedRng((ulong)seed);
    }

    // ---------------- Ask / Tell ----------------

    /// <summary>Samples Lambda candidates x = mean + sigma * B * D * z.</summary>
    public double[][] Ask()
    {
        int n = Dimension;
        var res = new double[Lambda][];
        var z = new double[n];
        for (int k = 0; k < Lambda; k++)
        {
            for (int i = 0; i < n; i++) z[i] = d[i] * NextGaussian();
            var x = new double[n];
            for (int i = 0; i < n; i++)
            {
                double s = 0;
                for (int j = 0; j < n; j++) s += b[i, j] * z[j];
                x[i] = Math.Clamp(Mean[i] + Sigma * s, LowerBound, UpperBound);
            }
            res[k] = x;
        }
        return res;
    }

    /// <summary>Updates the distribution. Higher fitness is better. Non-finite fitness counts as worst.</summary>
    public void Tell(double[][] candidates, double[] fitness)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(fitness);
        if (candidates.Length != fitness.Length) throw new ArgumentException("candidates/fitness length mismatch.");
        if (candidates.Length < Mu) throw new ArgumentException($"Need at least mu={Mu} candidates.");
        int n = Dimension;
        foreach (var x in candidates)
            if (x.Length != n) throw new ArgumentException("Candidate dimension mismatch.");

        var f = new double[fitness.Length];
        for (int i = 0; i < f.Length; i++) f[i] = double.IsFinite(fitness[i]) ? fitness[i] : double.NegativeInfinity;
        var order = Enumerable.Range(0, f.Length).ToArray();
        Array.Sort(order, (p, q) => f[q].CompareTo(f[p])); // descending (stable enough; ties rare)

        if (f[order[0]] > BestFitness || BestX == null)
        {
            BestFitness = f[order[0]];
            BestX = (double[])candidates[order[0]].Clone();
        }

        var oldMean = (double[])Mean.Clone();
        // y_i = (x_i - m_old) / sigma for the mu best.
        var y = new double[Mu][];
        for (int k = 0; k < Mu; k++)
        {
            var x = candidates[order[k]];
            var yk = new double[n];
            for (int i = 0; i < n; i++) yk[i] = (x[i] - oldMean[i]) / Sigma;
            y[k] = yk;
        }
        var yw = new double[n];
        for (int k = 0; k < Mu; k++)
            for (int i = 0; i < n; i++) yw[i] += w[k] * y[k][i];
        for (int i = 0; i < n; i++) Mean[i] = oldMean[i] + Sigma * yw[i];

        // C^{-1/2} yw = B D^{-1} B^T yw
        var tmp = new double[n];
        for (int j = 0; j < n; j++)
        {
            double s = 0;
            for (int i = 0; i < n; i++) s += b[i, j] * yw[i];
            tmp[j] = s / d[j];
        }
        double csn = Math.Sqrt(cs * (2 - cs) * mueff);
        for (int i = 0; i < n; i++)
        {
            double s = 0;
            for (int j = 0; j < n; j++) s += b[i, j] * tmp[j];
            ps[i] = (1 - cs) * ps[i] + csn * s;
        }
        double psNorm = Norm(ps);
        Generation++;
        bool hsig = psNorm / Math.Sqrt(1 - Math.Pow(1 - cs, 2 * Generation)) / chiN < 1.4 + 2.0 / (n + 1);
        double ccn = Math.Sqrt(cc * (2 - cc) * mueff);
        for (int i = 0; i < n; i++) pc[i] = (1 - cc) * pc[i] + (hsig ? ccn * yw[i] : 0);

        double dh = hsig ? 0 : cc * (2 - cc);
        double keep = 1 - c1 - cmu;
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
            {
                double rmu = 0;
                for (int k = 0; k < Mu; k++) rmu += w[k] * y[k][i] * y[k][j];
                double v = keep * c[i, j] + c1 * (pc[i] * pc[j] + dh * c[i, j]) + cmu * rmu;
                c[i, j] = v; c[j, i] = v;
            }

        Sigma *= Math.Exp(cs / damps * (psNorm / chiN - 1));
        if (Sigma > MaxSigma) Sigma = MaxSigma;
        if (!double.IsFinite(Sigma) || Sigma < 1e-300) Sigma = 1e-300;

        if (Generation - lastEigenGen >= eigenEvery) UpdateEigen();
    }

    void UpdateEigen()
    {
        lastEigenGen = Generation;
        int n = Dimension;
        var a = (double[,])c.Clone();
        var v = new double[n, n];
        var ev = new double[n];
        Jacobi.Eigen(a, v, ev);
        double maxEv = 0;
        for (int i = 0; i < n; i++) maxEv = Math.Max(maxEv, ev[i]);
        double floor = Math.Max(1e-20, maxEv * 1e-14); // keep C positive definite under rounding
        for (int i = 0; i < n; i++)
        {
            d[i] = Math.Sqrt(Math.Max(ev[i], floor));
            for (int j = 0; j < n; j++) b[j, i] = v[j, i];
        }
    }

    static double Norm(double[] v) { double s = 0; foreach (var x in v) s += x * x; return Math.Sqrt(s); }

    /// <summary>sqrt of the condition number of C (max/min axis length) from the last decomposition.</summary>
    public double AxisRatio
    {
        get { double mx = d.Max(), mn = d.Min(); return mx / mn; }
    }

    // ---------------- RNG (xoshiro256**, state is serialized) ----------------

    void SeedRng(ulong seed)
    {
        ulong s = seed;
        for (int i = 0; i < 4; i++)
        {
            s += 0x9E3779B97F4A7C15UL;
            ulong z = s;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            rng[i] = z ^ (z >> 31);
        }
    }

    ulong NextU64()
    {
        ulong result = ulong.RotateLeft(rng[1] * 5, 7) * 9;
        ulong t = rng[1] << 17;
        rng[2] ^= rng[0]; rng[3] ^= rng[1]; rng[1] ^= rng[2]; rng[0] ^= rng[3];
        rng[2] ^= t; rng[3] = ulong.RotateLeft(rng[3], 45);
        return result;
    }

    double NextDouble() => (NextU64() >> 11) * (1.0 / (1UL << 53));

    double NextGaussian()
    {
        double u1 = 1.0 - NextDouble(); // (0,1]
        double u2 = NextDouble();
        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    // ---------------- Serialization ----------------

    sealed class Dto
    {
        public int Version { get; set; } = 1;
        public int Dimension { get; set; }
        public int Lambda { get; set; }
        public int Generation { get; set; }
        public int LastEigenGen { get; set; }
        public double Sigma { get; set; }
        public double MaxSigma { get; set; }
        public double? LowerBound { get; set; }
        public double? UpperBound { get; set; }
        public bool BoundsSaved { get; set; } // false in states written before bounds existed
        public double[] Mean { get; set; } = [];
        public double[]? BestX { get; set; }
        public double? BestFitness { get; set; }
        public double[][] C { get; set; } = [];
        public double[][] B { get; set; } = [];
        public double[] D { get; set; } = [];
        public double[] Pc { get; set; } = [];
        public double[] Ps { get; set; } = [];
        public ulong[] Rng { get; set; } = [];
    }

    static double[][] ToJagged(double[,] m)
    {
        int n = m.GetLength(0);
        var r = new double[n][];
        for (int i = 0; i < n; i++) { r[i] = new double[n]; for (int j = 0; j < n; j++) r[i][j] = m[i, j]; }
        return r;
    }

    static bool AllFinite(double[] v) { foreach (var x in v) if (!double.IsFinite(x)) return false; return true; }

    static void FromJagged(double[][] src, double[,] dst, int n)
    {
        if (src.Length != n) throw new InvalidDataException("Matrix size mismatch.");
        for (int i = 0; i < n; i++)
        {
            if (src[i] is null || src[i].Length != n) throw new InvalidDataException("Matrix size mismatch.");
            for (int j = 0; j < n; j++)
            {
                if (!double.IsFinite(src[i][j])) throw new InvalidDataException("Matrix has non-finite values.");
                dst[i, j] = src[i][j];
            }
        }
    }

    public string ToJson() => JsonSerializer.Serialize(new Dto
    {
        Dimension = Dimension,
        Lambda = Lambda,
        Generation = Generation,
        LastEigenGen = lastEigenGen,
        Sigma = Sigma,
        MaxSigma = MaxSigma,
        LowerBound = double.IsFinite(LowerBound) ? LowerBound : null,
        UpperBound = double.IsFinite(UpperBound) ? UpperBound : null,
        BoundsSaved = true,
        Mean = (double[])Mean.Clone(),
        BestX = BestX == null ? null : (double[])BestX.Clone(),
        BestFitness = double.IsFinite(BestFitness) ? BestFitness : null,
        C = ToJagged(c),
        B = ToJagged(b),
        D = (double[])d.Clone(),
        Pc = (double[])pc.Clone(),
        Ps = (double[])ps.Clone(),
        Rng = (ulong[])rng.Clone(),
    });

    public static CmaEs FromJson(string json)
    {
        Dto dto;
        try { dto = JsonSerializer.Deserialize<Dto>(json) ?? throw new InvalidDataException("Empty CMA-ES state."); }
        catch (JsonException e) { throw new InvalidDataException("Malformed CMA-ES state: " + e.Message, e); }
        if (dto.Version != 1) throw new InvalidDataException($"CMA-ES state v{dto.Version}, expected v1.");
        int n = dto.Dimension;
        if (n < 1 || dto.Mean is null || dto.D is null || dto.Pc is null || dto.Ps is null || dto.Rng is null || dto.C is null || dto.B is null)
            throw new InvalidDataException("CMA-ES state is missing fields.");
        if (dto.Mean.Length != n || dto.D.Length != n || dto.Pc.Length != n || dto.Ps.Length != n || dto.Rng.Length != 4)
            throw new InvalidDataException("CMA-ES state vector size mismatch.");
        if (dto.Lambda < 4 || dto.Generation < 0 || dto.LastEigenGen < 0 || dto.LastEigenGen > dto.Generation)
            throw new InvalidDataException("CMA-ES state has invalid lambda/generation.");
        if (!(dto.Sigma > 0) || !double.IsFinite(dto.Sigma) || !AllFinite(dto.Mean) || !AllFinite(dto.Pc) || !AllFinite(dto.Ps))
            throw new InvalidDataException("CMA-ES state has non-finite values.");
        foreach (var v in dto.D) if (!(v > 0) || !double.IsFinite(v)) throw new InvalidDataException("CMA-ES state has non-positive axis lengths.");
        if (dto.Rng.All(r => r == 0)) throw new InvalidDataException("CMA-ES RNG state is all zero.");
        if (dto.BestX != null && (dto.BestX.Length != n || !AllFinite(dto.BestX))) throw new InvalidDataException("CMA-ES BestX invalid.");
        double lo = dto.BoundsSaved ? dto.LowerBound ?? double.NegativeInfinity : 0;
        double hi = dto.BoundsSaved ? dto.UpperBound ?? double.PositiveInfinity : 1;
        if (double.IsNaN(lo) || double.IsNaN(hi) || lo >= hi) throw new InvalidDataException("CMA-ES bounds invalid.");
        var es = new CmaEs(dto.Mean, dto.Sigma, dto.Lambda)
        {
            Generation = dto.Generation,
            lastEigenGen = dto.LastEigenGen,
            MaxSigma = dto.MaxSigma > 0 && double.IsFinite(dto.MaxSigma) ? dto.MaxSigma : 1.0,
            LowerBound = lo,
            UpperBound = hi,
            BestX = dto.BestX,
            BestFitness = dto.BestFitness ?? double.NegativeInfinity,
        };
        FromJagged(dto.C, es.c, n);
        FromJagged(dto.B, es.b, n);
        dto.D.CopyTo(es.d, 0);
        dto.Pc.CopyTo(es.pc, 0);
        dto.Ps.CopyTo(es.ps, 0);
        dto.Rng.CopyTo(es.rng, 0);
        return es;
    }
}

/// <summary>Cyclic Jacobi eigensolver for small symmetric matrices.</summary>
public static class Jacobi
{
    /// <summary>
    /// Diagonalizes symmetric <paramref name="a"/> (destroyed). On return column k of <paramref name="v"/>
    /// is the eigenvector for eigenvalue <paramref name="ev"/>[k] (unsorted).
    /// </summary>
    public static void Eigen(double[,] a, double[,] v, double[] ev, int maxSweeps = 100)
    {
        int n = ev.Length;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) v[i, j] = i == j ? 1 : 0;
        for (int sweep = 0; sweep < maxSweeps; sweep++)
        {
            double off = 0, diag = 0;
            for (int i = 0; i < n; i++)
            {
                diag += a[i, i] * a[i, i];
                for (int j = i + 1; j < n; j++) off += a[i, j] * a[i, j];
            }
            if (off <= 1e-30 * diag || off == 0) break;
            for (int p = 0; p < n - 1; p++)
                for (int q = p + 1; q < n; q++)
                {
                    double apq = a[p, q];
                    if (Math.Abs(apq) < 1e-300) continue;
                    double theta = (a[q, q] - a[p, p]) / (2 * apq);
                    double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                    if (theta == 0) t = 1;
                    double cs = 1 / Math.Sqrt(t * t + 1), sn = t * cs;
                    for (int k = 0; k < n; k++)
                    {
                        double akp = a[k, p], akq = a[k, q];
                        a[k, p] = cs * akp - sn * akq;
                        a[k, q] = sn * akp + cs * akq;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double apk = a[p, k], aqk = a[q, k];
                        a[p, k] = cs * apk - sn * aqk;
                        a[q, k] = sn * apk + cs * aqk;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double vkp = v[k, p], vkq = v[k, q];
                        v[k, p] = cs * vkp - sn * vkq;
                        v[k, q] = sn * vkp + cs * vkq;
                    }
                }
        }
        for (int i = 0; i < n; i++) ev[i] = a[i, i];
    }
}
