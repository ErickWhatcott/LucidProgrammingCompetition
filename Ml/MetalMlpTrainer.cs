using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Lpc.Ml;

/// <summary>
/// Trains an <see cref="Mlp"/> entirely on the GPU (Metal): standardized train/val matrices, weights,
/// gradients and Adam moments live in shared MTLBuffers; every minibatch step (forward GEMMs with fused
/// bias+ReLU, loss gradient, backward GEMMs with fused ReLU mask / split-K atomic dW+db, Adam) is encoded
/// into a few large command buffers per epoch in a CPU-shuffled order, plus a GPU validation pass.
/// Same algorithm as <see cref="Mlp.Fit"/> (weighted MSE/Huber, coupled L2 decay, early stopping that
/// keeps the best-validation weights, starting weights count as epoch 0). Only the final weights are
/// copied back into the Mlp, so all CPU inference is unchanged.
/// </summary>
public static unsafe class MetalMlpTrainer
{
    const int Tile = 16;
    const int StepsPerCommandBuffer = 256;
    const int ValChunk = 8192;

    [StructLayout(LayoutKind.Sequential)]
    struct GemmArgs
    {
        public int M, N, K, Lda, Ldb, Ldc, TransA, TransB, GatherA, IdxOff, OnesRow, Mode, KChunk, Pad;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct LossArgs { public int B, Start, HasW, Pad; public float InvSw, Delta; }

    [StructLayout(LayoutKind.Sequential)]
    struct AdamArgs { public int P; public float LrT, B1, B2, Eps; }

    [StructLayout(LayoutKind.Sequential)]
    struct ErrArgs { public int Rows, Off; }

    const int ModeStore = 0, ModeBias = 1, ModeBiasRelu = 2, ModeMask = 3, ModeAtomicAdd = 4;

    const string Source = """
        #include <metal_stdlib>
        using namespace metal;

        struct GemmArgs { int M, N, K, lda, ldb, ldc, transA, transB, gatherA, idxOff, onesRow, mode, kChunk, pad; };
        struct LossArgs { int b, start, hasW, pad; float invSw, delta; };
        struct AdamArgs { int P; float lrT, b1, b2, eps; };
        struct ErrArgs { int rows, off; };

        #define T 16

        // C[M,N] (op)= op(A)[M,K] * op(B)[K,N] over K range [z*kChunk, (z+1)*kChunk).
        // A: !transA stored [M(or gathered rows), lda]; transA stored [K(or gathered rows), lda] (op(A)[m,k] = A[k,m]).
        // B: !transB stored [K, ldb]; transB stored [N, ldb] (op(B)[k,n] = B[n,k]).
        // onesRow: op(A) row M-1 is all ones (so the last row of C is the column sum = bias gradient).
        // mode: 0 store, 1 +bias(aux[n]), 2 relu(+bias), 3 keep where aux[m*ldc+n] > 0 (ReLU mask), 4 atomic add.
        kernel void gemm(device const float* A [[buffer(0)]],
                         device const float* B [[buffer(1)]],
                         device float* C [[buffer(2)]],
                         device const float* aux [[buffer(3)]],
                         device const int* idx [[buffer(4)]],
                         constant GemmArgs& p [[buffer(5)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         uint3 lt [[thread_position_in_threadgroup]])
        {
            threadgroup float As[T][T + 1];
            threadgroup float Bs[T][T + 1];
            const int tx = lt.x, ty = lt.y;
            const int m0 = tg.y * T, n0 = tg.x * T;
            const int kBeg = tg.z * p.kChunk;
            const int kEnd = min(p.K, kBeg + p.kChunk);
            const int aRows = p.onesRow ? p.M - 1 : p.M;
            float acc = 0.0f;
            for (int k0 = kBeg; k0 < kEnd; k0 += T) {
                if (!p.transA) {
                    int mm = m0 + ty, kk = k0 + tx;
                    float a = 0.0f;
                    if (mm < p.M && kk < kEnd) {
                        int row = p.gatherA ? idx[p.idxOff + mm] : mm;
                        a = A[row * p.lda + kk];
                    }
                    As[ty][tx] = a;
                } else {
                    int mm = m0 + tx, kk = k0 + ty;   // coalesced along m
                    float a = 0.0f;
                    if (mm < p.M && kk < kEnd) {
                        if (mm >= aRows) a = 1.0f;
                        else {
                            int row = p.gatherA ? idx[p.idxOff + kk] : kk;
                            a = A[row * p.lda + mm];
                        }
                    }
                    As[tx][ty] = a;
                }
                if (!p.transB) {
                    int kk = k0 + ty, nn = n0 + tx;
                    Bs[ty][tx] = (kk < kEnd && nn < p.N) ? B[kk * p.ldb + nn] : 0.0f;
                } else {
                    int kk = k0 + tx, nn = n0 + ty;   // coalesced along k
                    Bs[tx][ty] = (kk < kEnd && nn < p.N) ? B[nn * p.ldb + kk] : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                #pragma unroll
                for (int k = 0; k < T; k++) acc = fma(As[ty][k], Bs[k][tx], acc);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const int m = m0 + ty, n = n0 + tx;
            if (m >= p.M || n >= p.N) return;
            device float* c = C + m * p.ldc + n;
            switch (p.mode) {
                case 0: *c = acc; break;
                case 1: *c = acc + aux[n]; break;
                case 2: *c = max(acc + aux[n], 0.0f); break;
                case 3: *c = aux[m * p.ldc + n] > 0.0f ? acc : 0.0f; break;
                default: atomic_fetch_add_explicit((device atomic_float*)c, acc, memory_order_relaxed); break;
            }
        }

        kernel void loss_grad(device const float* out [[buffer(0)]],
                              device const float* Y [[buffer(1)]],
                              device const float* Wt [[buffer(2)]],
                              device const int* idx [[buffer(3)]],
                              device float* dOut [[buffer(4)]],
                              device float* rowLoss [[buffer(5)]],
                              constant LossArgs& p [[buffer(6)]],
                              uint r [[thread_position_in_grid]])
        {
            if ((int)r >= p.b) return;
            int i = idx[p.start + r];
            float w = p.hasW ? Wt[i] : 1.0f;
            float e = out[r] - Y[i];
            float l, g;
            if (p.delta > 0.0f) {
                float ae = fabs(e);
                l = ae <= p.delta ? 0.5f * e * e : p.delta * (ae - 0.5f * p.delta);
                g = clamp(e, -p.delta, p.delta);
            } else { l = e * e; g = 2.0f * e; }
            dOut[r] = w * g * p.invSw;
            rowLoss[p.start + r] = w * l;
        }

        kernel void adam(device float* w [[buffer(0)]],
                         device float* g [[buffer(1)]],
                         device float* m [[buffer(2)]],
                         device float* v [[buffer(3)]],
                         device const float* wd [[buffer(4)]],
                         constant AdamArgs& p [[buffer(5)]],
                         uint i [[thread_position_in_grid]])
        {
            if ((int)i >= p.P) return;
            float gi = g[i] + wd[i] * w[i];
            float mi = p.b1 * m[i] + (1.0f - p.b1) * gi;
            float vi = p.b2 * v[i] + (1.0f - p.b2) * gi * gi;
            m[i] = mi; v[i] = vi;
            w[i] -= p.lrT * mi / (sqrt(vi) + p.eps);
            g[i] = 0.0f; // ready for the next step's atomic accumulation
        }

        kernel void sq_err(device const float* out [[buffer(0)]],
                           device const float* Y [[buffer(1)]],
                           device float* err [[buffer(2)]],
                           constant ErrArgs& p [[buffer(3)]],
                           uint r [[thread_position_in_grid]])
        {
            if ((int)r >= p.rows) return;
            float e = out[r] - Y[p.off + r];
            err[p.off + r] = e * e;
        }
        """;

    static readonly object gate = new();      // serializes trainings (one GPU job at a time)
    static readonly object initGate = new();  // separate so Available never blocks behind a running training
    static volatile IntPtr[]? pipelines;
    static volatile string? initError;

    /// <summary>True when Metal is available and the training kernels compiled. Never throws.</summary>
    public static bool Available
    {
        get
        {
            if (!Metal.Available) return false;
            if (pipelines != null) return true;
            lock (initGate) return EnsurePipelines();
        }
    }

    /// <summary>Why <see cref="Available"/> is false (null when available).</summary>
    public static string? UnavailableReason => !Metal.Available ? "Metal device not available" : Available ? null : initError;

    static bool EnsurePipelines()
    {
        if (pipelines != null) return true;
        if (initError != null) return false;
        try { pipelines = Metal.CompilePipelines(Source, "gemm", "loss_grad", "adam", "sq_err"); return true; }
        catch (Exception e) { initError = e.Message; return false; }
    }

    public static FitResult Fit(Mlp model, TrainingSet train, TrainingSet? val, TrainOptions o, float learningRate, int epochs)
    {
        if (!Available) throw new MetalException("Metal training unavailable: " + (initError ?? "no device"));
        lock (gate)
        {
            if (!Metal.Available || pipelines == null) throw new MetalException("Metal training unavailable: " + (initError ?? "no device"));
            return FitCore(model, train, val, o, learningRate, epochs, pipelines!);
        }
    }

    static FitResult FitCore(Mlp model, TrainingSet train, TrainingSet? val, TrainOptions o, float learningRate, int epochs, IntPtr[] ps)
    {
        IntPtr pGemm = ps[0], pLoss = ps[1], pAdam = ps[2], pErr = ps[3];
        int d = model.InputCount, n = train.Rows;
        int[] sizes = [d, .. model.Hidden, 1];
        int layers = sizes.Length - 1;
        if (n <= 0) throw new ArgumentException("Empty training set.", nameof(train));
        int bsReq = o.GpuBatchSize > 0 ? o.GpuBatchSize : o.BatchSize;
        int bs = Math.Max(1, Math.Min(bsReq, n));
        var log = o.Log;
        string backend = $"metal ({Metal.DeviceName})";

        bool hasVal = val is { Rows: > 0 };
        int nv = hasVal ? val!.Rows : 0;
        int cap = Math.Max(bs, hasVal ? Math.Min(ValChunk, nv) : 0);

        // parameter layout: per layer W[l] (din*dout) then B[l] (dout), contiguous
        var offW = new int[layers];
        var offB = new int[layers];
        int P = 0;
        for (int l = 0; l < layers; l++) { offW[l] = P; P += sizes[l] * sizes[l + 1]; offB[l] = P; P += sizes[l + 1]; }
        var actOff = new long[layers + 1];   // act[0] is never stored (gathered from X)
        var delOff = new long[layers + 1];
        long actTotal = 0;
        for (int l = 1; l <= layers; l++) { actOff[l] = actTotal; delOff[l] = actTotal; actTotal += (long)cap * sizes[l]; }

        var buffers = new List<Metal.Buffer>();
        Metal.Buffer Alloc(long bytes) { var b = new Metal.Buffer(bytes); buffers.Add(b); return b; }

        try
        {
            var bX = Alloc((long)n * d * 4);
            var bY = Alloc((long)n * 4);
            var bWt = train.Weights != null ? Alloc((long)n * 4) : bY;
            var bIdx = Alloc((long)n * 4);
            var bRowLoss = Alloc((long)n * 4);
            var bXv = hasVal ? Alloc((long)nv * d * 4) : bX;
            var bYv = hasVal ? Alloc((long)nv * 4) : bY;
            var bErr = hasVal ? Alloc((long)nv * 4) : bY;
            var bParams = Alloc(P * 4L);
            var bGrad = Alloc(P * 4L);
            var bM = Alloc(P * 4L);
            var bV = Alloc(P * 4L);
            var bWd = Alloc(P * 4L);
            var bAct = Alloc(actTotal * 4);
            var bDel = Alloc(actTotal * 4);

            model.Standardize(train.X, n, bX.AsSpan<float>((long)n * d));
            train.Y.AsSpan(0, n).CopyTo(bY.AsSpan<float>(n));
            if (train.Weights != null) train.Weights.AsSpan(0, n).CopyTo(bWt.AsSpan<float>(n));
            if (hasVal)
            {
                model.Standardize(val!.X, nv, bXv.AsSpan<float>((long)nv * d));
                val.Y.AsSpan(0, nv).CopyTo(bYv.AsSpan<float>(nv));
            }
            var prm = bParams.AsSpan<float>(P);
            var wdv = bWd.AsSpan<float>(P);
            for (int l = 0; l < layers; l++)
            {
                model.W[l].CopyTo(prm[offW[l]..]);
                model.B[l].CopyTo(prm[offB[l]..]);
                wdv.Slice(offW[l], model.W[l].Length).Fill(o.WeightDecay);
            }
            bGrad.AsSpan<float>(P).Clear();
            bM.AsSpan<float>(P).Clear();
            bV.AsSpan<float>(P).Clear();
            var best = prm.ToArray();

            double initialVal = hasVal ? model.Evaluate(val!.X, val.Y, nv) : double.NaN;
            double bestVal = hasVal ? initialVal : double.PositiveInfinity;
            int bestEpoch = 0, bad = 0, epoch;
            double lastTrain = double.NaN;
            long totalRows = 0, totalTicks = 0;
            float delta = o.HuberDelta;
            const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
            int step = 0;

            var idx = bIdx.AsSpan<int>(n);
            for (int i = 0; i < n; i++) idx[i] = i;
            var rng = new Random(o.Seed);
            int kChunk = Math.Max(64, (bs / 32 + Tile - 1) / Tile * Tile); // split-K for dW: ~32 partitions at large batch

            log?.Invoke($"fit: rows={n} val={nv} params={P} shape={string.Join('-', sizes)} batch={bs} lr={learningRate:G3} " +
                        $"wd={o.WeightDecay:G3} loss={(delta > 0 ? $"huber({delta})" : "mse")} device=gpu backend={backend}" +
                        (hasVal ? $" initialValMse={initialVal:G5}" : ""));

            void Gemm(Metal.ComputeEncoder enc, Metal.Buffer a, long aOff, Metal.Buffer b, long bOff, Metal.Buffer c, long cOff,
                      Metal.Buffer aux, long auxOff, in GemmArgs g)
            {
                enc.SetBuffer(a, aOff * 4, 0);
                enc.SetBuffer(b, bOff * 4, 1);
                enc.SetBuffer(c, cOff * 4, 2);
                enc.SetBuffer(aux, auxOff * 4, 3);
                enc.SetBuffer(bIdx, 0, 4);
                enc.SetBytes(g, 5);
                int splits = g.Mode == ModeAtomicAdd ? (g.K + g.KChunk - 1) / g.KChunk : 1;
                enc.Dispatch(new MTLSize((g.N + Tile - 1) / Tile, (g.M + Tile - 1) / Tile, splits), new MTLSize(Tile, Tile, 1));
            }

            // Forward for `rows` rows: layer-0 input is X (gathered through idx at idxOff) or a contiguous slice of it.
            void Forward(Metal.ComputeEncoder enc, Metal.Buffer x, long xOffFloats, bool gather, int idxOff, int rows)
            {
                enc.SetPipeline(pGemm);
                for (int l = 0; l < layers; l++)
                {
                    int din = sizes[l], dout = sizes[l + 1];
                    var g = new GemmArgs
                    {
                        M = rows, N = dout, K = din, Lda = din, Ldb = dout, Ldc = dout,
                        GatherA = l == 0 && gather ? 1 : 0, IdxOff = idxOff,
                        Mode = l < layers - 1 ? ModeBiasRelu : ModeBias, KChunk = din,
                    };
                    if (l == 0) Gemm(enc, x, xOffFloats, bParams, offW[l], bAct, actOff[1], bParams, offB[l], g);
                    else Gemm(enc, bAct, actOff[l], bParams, offW[l], bAct, actOff[l + 1], bParams, offB[l], g);
                }
            }

            for (epoch = 1; epoch <= epochs; epoch++)
            {
                long t0 = Stopwatch.GetTimestamp();
                for (int i = n - 1; i > 0; i--) { int j = rng.Next(i + 1); (idx[i], idx[j]) = (idx[j], idx[i]); }
                double wSum = 0;
                double epochVal = double.NaN;
                // rows of skipped (zero-weight) batches are never written by loss_grad: don't count last epoch's values
                bRowLoss.AsSpan<float>(n).Clear();

                using (Metal.AutoreleasePool.Push())
                {
                    var committed = new List<IntPtr>();
                    var cb = IntPtr.Zero;
                    var enc = default(Metal.ComputeEncoder);
                    try
                    {
                    cb = Metal.NewCommandBuffer();
                    enc = Metal.ComputeEncoder.Begin(cb);
                    int stepsInCb = 0;

                    for (int start = 0; start < n; start += bs)
                    {
                        int b = Math.Min(bs, n - start);
                        float sw = 0f;
                        if (train.Weights is { } tw) for (int r = 0; r < b; r++) sw += tw[idx[start + r]];
                        else sw = b;
                        if (sw <= 0f) continue;
                        wSum += sw;

                        Forward(enc, bX, 0, true, start, b);

                        enc.SetPipeline(pLoss);
                        enc.SetBuffer(bAct, actOff[layers] * 4, 0);
                        enc.SetBuffer(bY, 0, 1);
                        enc.SetBuffer(bWt, 0, 2);
                        enc.SetBuffer(bIdx, 0, 3);
                        enc.SetBuffer(bDel, delOff[layers] * 4, 4);
                        enc.SetBuffer(bRowLoss, 0, 5);
                        enc.SetBytes(new LossArgs { B = b, Start = start, HasW = train.Weights != null ? 1 : 0, InvSw = 1f / sw, Delta = delta }, 6);
                        enc.Dispatch1D(b);

                        enc.SetPipeline(pGemm);
                        for (int l = layers - 1; l >= 0; l--)
                        {
                            int din = sizes[l], dout = sizes[l + 1];
                            // [dW; db] += [A^T; 1^T] dZ  (split-K over the batch, atomic accumulation; grads zeroed by adam)
                            var gw = new GemmArgs
                            {
                                M = din + 1, N = dout, K = b, Lda = din, Ldb = dout, Ldc = dout, TransA = 1,
                                GatherA = l == 0 ? 1 : 0, IdxOff = start, OnesRow = 1, Mode = ModeAtomicAdd, KChunk = kChunk,
                            };
                            if (l == 0) Gemm(enc, bX, 0, bDel, delOff[l + 1], bGrad, offW[l], bGrad, 0, gw);
                            else Gemm(enc, bAct, actOff[l], bDel, delOff[l + 1], bGrad, offW[l], bGrad, 0, gw);
                            if (l > 0)
                            {
                                // dA = (dZ W^T) * (A > 0)
                                var ga = new GemmArgs
                                {
                                    M = b, N = din, K = dout, Lda = dout, Ldb = dout, Ldc = din, TransB = 1,
                                    Mode = ModeMask, KChunk = dout,
                                };
                                Gemm(enc, bDel, delOff[l + 1], bParams, offW[l], bDel, delOff[l], bAct, actOff[l], ga);
                            }
                        }

                        step++;
                        float c1 = 1f - MathF.Pow(b1, step), c2 = 1f - MathF.Pow(b2, step);
                        enc.SetPipeline(pAdam);
                        enc.SetBuffer(bParams, 0, 0);
                        enc.SetBuffer(bGrad, 0, 1);
                        enc.SetBuffer(bM, 0, 2);
                        enc.SetBuffer(bV, 0, 3);
                        enc.SetBuffer(bWd, 0, 4);
                        enc.SetBytes(new AdamArgs { P = P, LrT = learningRate * MathF.Sqrt(c2) / c1, B1 = b1, B2 = b2, Eps = eps }, 5);
                        enc.Dispatch1D(P);

                        if (++stepsInCb >= StepsPerCommandBuffer)
                        {
                            // hand this chunk to the GPU now so encoding overlaps execution
                            enc.End();
                            enc = default;
                            Metal.Commit(cb);
                            committed.Add(cb);
                            cb = Metal.NewCommandBuffer();
                            enc = Metal.ComputeEncoder.Begin(cb);
                            stepsInCb = 0;
                        }
                    }

                    if (hasVal)
                    {
                        for (int s = 0; s < nv; s += cap)
                        {
                            int c = Math.Min(cap, nv - s);
                            Forward(enc, bXv, (long)s * d, false, 0, c);
                            enc.SetPipeline(pErr);
                            enc.SetBuffer(bAct, actOff[layers] * 4, 0);
                            enc.SetBuffer(bYv, 0, 1);
                            enc.SetBuffer(bErr, 0, 2);
                            enc.SetBytes(new ErrArgs { Rows = c, Off = s }, 3);
                            enc.Dispatch1D(c);
                        }
                    }
                    enc.End();
                    enc = default;
                    Metal.Commit(cb);
                    committed.Add(cb);
                    cb = IntPtr.Zero;
                    }
                    finally
                    {
                        // On an encoding failure: close the open encoder (releasing an un-ended encoder aborts in
                        // Metal) and let in-flight work finish before the pool pops and the buffers are released.
                        if (enc.Handle != IntPtr.Zero) enc.End();
                        // wait on every buffer (cheap once done) rather than relying on in-order completion
                        foreach (var x in committed) Metal.WaitUntilCompleted(x);
                    }
                    foreach (var x in committed) Metal.CheckCompleted(x);
                }

                double lossSum = 0;
                foreach (float v in bRowLoss.AsSpan<float>(n)) lossSum += v;
                if (hasVal)
                {
                    double e = 0;
                    foreach (float v in bErr.AsSpan<float>(nv)) e += v;
                    epochVal = e / nv;
                }

                long ticks = Stopwatch.GetTimestamp() - t0;
                totalTicks += ticks;
                totalRows += n;
                lastTrain = wSum > 0 ? lossSum / wSum : double.NaN;
                double rowsPerSec = n / Math.Max(1e-9, ticks / (double)Stopwatch.Frequency);

                if (!double.IsFinite(lastTrain))
                {
                    log?.Invoke($"epoch {epoch}: training loss diverged ({lastTrain}); restoring best weights.");
                    break;
                }

                string msg = $"epoch {epoch}/{epochs} train={lastTrain:G5}";
                if (hasVal)
                {
                    msg += $" valMse={epochVal:G5}";
                    if (epochVal < bestVal)
                    {
                        bestVal = epochVal; bestEpoch = epoch; bad = 0;
                        bParams.AsSpan<float>(P).CopyTo(best);
                        msg += " *";
                    }
                    else bad++;
                }
                else
                {
                    bestEpoch = epoch;
                    bParams.AsSpan<float>(P).CopyTo(best);
                }
                log?.Invoke($"{msg} rows/s={rowsPerSec:N0} (gpu)");
                if (hasVal && bad >= o.Patience) { log?.Invoke($"early stop: no val improvement for {bad} epochs"); epoch++; break; }
            }

            for (int l = 0; l < layers; l++)
            {
                best.AsSpan(offW[l], model.W[l].Length).CopyTo(model.W[l]);
                best.AsSpan(offB[l], model.B[l].Length).CopyTo(model.B[l]);
            }
            double avgRps = totalRows / Math.Max(1e-9, totalTicks / (double)Stopwatch.Frequency);
            var result = new FitResult(epoch - 1, bestEpoch, initialVal, hasVal ? bestVal : double.NaN, lastTrain, avgRps, "metal-gpu");
            log?.Invoke($"fit done: epochs={result.EpochsRun} best={bestEpoch} bestValMse={result.BestValLoss:G5} avgRows/s={avgRps:N0} device=gpu backend={backend}");
            return result;
        }
        finally
        {
            foreach (var b in buffers) b.Dispose();
        }
    }
}
