using System.Numerics;
using System.Runtime.InteropServices;

namespace Lpc.Ml;

/// <summary>
/// Row-major single-precision GEMM: C = alpha * op(A) * op(B) + beta * C, where op(A) is m x k,
/// op(B) is k x n and C is m x n. A is stored m x k (k x m when transA), B is stored k x n
/// (n x k when transB).
///
/// On macOS the call goes to Apple Accelerate's cblas_sgemm, which on M-series chips runs on the
/// AMX matrix coprocessor. Everywhere else (or if the P/Invoke fails, or when forced) a managed
/// SIMD implementation is used. The managed path is single-threaded so its latency is predictable.
/// </summary>
public static unsafe class Blas
{
    const string AcceleratePath = "/System/Library/Frameworks/Accelerate.framework/Accelerate";
    const int CblasRowMajor = 101, CblasNoTrans = 111, CblasTrans = 112;

    [DllImport(AcceleratePath, EntryPoint = "cblas_sgemm")]
    static extern void cblas_sgemm(int order, int transA, int transB, int m, int n, int k, float alpha,
        float* a, int lda, float* b, int ldb, float beta, float* c, int ldc);

    static readonly bool accelerateAvailable = ProbeAccelerate();
    static volatile bool useAccelerate = accelerateAvailable;

    /// <summary>True if Accelerate loaded and produced a correct result on a probe multiply.</summary>
    public static bool AccelerateAvailable => accelerateAvailable;

    /// <summary>"accelerate" or "managed".</summary>
    public static string Backend => useAccelerate ? "accelerate" : "managed";

    /// <summary>"auto" (Accelerate when available), "accelerate" (throws if unavailable) or "managed".</summary>
    public static void Force(string backend)
    {
        switch (backend.Trim().ToLowerInvariant())
        {
            case "auto": useAccelerate = accelerateAvailable; break;
            case "managed": useAccelerate = false; break;
            case "accelerate":
                if (!accelerateAvailable) throw new InvalidOperationException("Accelerate is not available on this machine.");
                useAccelerate = true;
                break;
            default: throw new ArgumentException($"Unknown BLAS backend '{backend}' (expected auto|accelerate|managed).", nameof(backend));
        }
    }

    static bool ProbeAccelerate()
    {
        if (!OperatingSystem.IsMacOS()) return false;
        try
        {
            // [1 2; 3 4] * [5 6; 7 8] = [19 22; 43 50]
            float* a = stackalloc float[] { 1, 2, 3, 4 };
            float* b = stackalloc float[] { 5, 6, 7, 8 };
            float* c = stackalloc float[4];
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1f, a, 2, b, 2, 0f, c, 2);
            return c[0] == 19 && c[1] == 22 && c[2] == 43 && c[3] == 50;
        }
        catch (Exception) // DllNotFoundException, EntryPointNotFoundException, BadImageFormatException...
        {
            return false;
        }
    }

    public static void Sgemm(bool transA, bool transB, int m, int n, int k, float alpha,
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, float beta, Span<float> c)
    {
        if (m < 0 || n < 0 || k < 0) throw new ArgumentOutOfRangeException(nameof(m), "Negative dimension.");
        if (a.Length < (long)m * k) throw new ArgumentException("A is too small.", nameof(a));
        if (b.Length < (long)k * n) throw new ArgumentException("B is too small.", nameof(b));
        if (c.Length < (long)m * n) throw new ArgumentException("C is too small.", nameof(c));
        if (m == 0 || n == 0) return;

        if (useAccelerate && k > 0)
        {
            fixed (float* pa = a)
            fixed (float* pb = b)
            fixed (float* pc = c)
            {
                cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans,
                    m, n, k, alpha, pa, transA ? m : k, pb, transB ? k : n, beta, pc, n);
            }
            return;
        }
        ManagedSgemm(transA, transB, m, n, k, alpha, a, b, beta, c);
    }

    /// <summary>Managed SIMD implementation (always available; also used by tests to cross-check Accelerate).</summary>
    public static void ManagedSgemm(bool transA, bool transB, int m, int n, int k, float alpha,
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, float beta, Span<float> c)
    {
        var cc = c[..(m * n)];
        if (beta == 0f) cc.Clear();
        else if (beta != 1f) Scale(cc, beta);
        if (alpha == 0f || k == 0) return;

        if (!transB)
        {
            // C[i,:] += alpha * sum_p op(A)[i,p] * B[p,:]  (row-axpy form; skips zeros, which ReLU makes common)
            for (int i = 0; i < m; i++)
            {
                var crow = c.Slice(i * n, n);
                for (int p = 0; p < k; p++)
                {
                    float s = transA ? a[p * m + i] : a[i * k + p];
                    if (s == 0f) continue;
                    Axpy(alpha * s, b.Slice(p * n, n), crow);
                }
            }
        }
        else if (!transA)
        {
            // B stored n x k: C[i,j] += alpha * dot(A[i,:], B[j,:])
            for (int i = 0; i < m; i++)
            {
                var arow = a.Slice(i * k, k);
                var crow = c.Slice(i * n, n);
                for (int j = 0; j < n; j++) crow[j] += alpha * Dot(arow, b.Slice(j * k, k));
            }
        }
        else
        {
            // Both transposed (rare): gather column i of the stored A into a contiguous row.
            float[]? rented = k > 1024 ? System.Buffers.ArrayPool<float>.Shared.Rent(k) : null;
            Span<float> col = rented is null ? stackalloc float[k] : rented.AsSpan(0, k);
            for (int i = 0; i < m; i++)
            {
                for (int p = 0; p < k; p++) col[p] = a[p * m + i];
                var crow = c.Slice(i * n, n);
                for (int j = 0; j < n; j++) crow[j] += alpha * Dot(col, b.Slice(j * k, k));
            }
            if (rented is not null) System.Buffers.ArrayPool<float>.Shared.Return(rented);
        }
    }

    static void Scale(Span<float> x, float s)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && x.Length >= Vector<float>.Count)
        {
            var vx = MemoryMarshal.Cast<float, Vector<float>>(x);
            var vs = new Vector<float>(s);
            for (int v = 0; v < vx.Length; v++) vx[v] *= vs;
            i = vx.Length * Vector<float>.Count;
        }
        for (; i < x.Length; i++) x[i] *= s;
    }

    static void Axpy(float s, ReadOnlySpan<float> x, Span<float> y)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && y.Length >= Vector<float>.Count)
        {
            var vx = MemoryMarshal.Cast<float, Vector<float>>(x);
            var vy = MemoryMarshal.Cast<float, Vector<float>>(y);
            var vs = new Vector<float>(s);
            for (int v = 0; v < vy.Length; v++) vy[v] += vs * vx[v];
            i = vy.Length * Vector<float>.Count;
        }
        for (; i < y.Length; i++) y[i] += s * x[i];
    }

    static float Dot(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        int i = 0;
        float sum = 0f;
        if (Vector.IsHardwareAccelerated && x.Length >= Vector<float>.Count)
        {
            var vx = MemoryMarshal.Cast<float, Vector<float>>(x);
            var vy = MemoryMarshal.Cast<float, Vector<float>>(y[..x.Length]);
            var acc = Vector<float>.Zero;
            for (int v = 0; v < vx.Length; v++) acc += vx[v] * vy[v];
            sum = Vector.Sum(acc);
            i = vx.Length * Vector<float>.Count;
        }
        for (; i < x.Length; i++) sum += x[i] * y[i];
        return sum;
    }
}
