using System.Diagnostics;
using System.Globalization;
using Lpc.Strategies.Planner;

namespace Lpc.Ml;

public sealed class ImitationOptions
{
    public int Epochs { get; set; } = 15;
    public int BatchDecisions { get; set; } = 256;
    public float LearningRate { get; set; } = 1e-3f;
    public double ValidationFraction { get; set; } = 0.1;
    public int Patience { get; set; } = 3;
    public int Seed { get; set; } = 1;
    /// <summary>Softmax temperature used for the imitation loss (keep equal to the acting temperature).</summary>
    public double Temperature { get; set; } = 1.0;
    public float MaxGradNorm { get; set; } = 1.0f;
    /// <summary>Per-decision weight (e.g. favour strong seats by FinalPlace); null = 1. Weights &lt;= 0 drop the record.</summary>
    public Func<DecisionRecord, float>? WeightFn { get; set; }
    public Action<string>? Log { get; set; }
}

public sealed class PpoOptions
{
    public int Epochs { get; set; } = 4;
    public int BatchDecisions { get; set; } = 256;
    public float LearningRate { get; set; } = 3e-4f;
    public float Clip { get; set; } = 0.2f;
    public float EntropyCoef { get; set; } = 0.01f;
    /// <summary>Stop the update once a minibatch's approx KL exceeds 1.5 x TargetKl.</summary>
    public float TargetKl { get; set; } = 0.02f;
    /// <summary>Must equal the temperature the acting <see cref="PolicyScorer"/> used.</summary>
    public double Temperature { get; set; } = 1.0;
    public float MaxGradNorm { get; set; } = 1.0f;
    public int Seed { get; set; } = 1;
    public Action<string>? Log { get; set; }
}

public sealed record PpoStats(
    int Decisions,       // usable on-policy decisions (finite behavior log-prob, >= 2 rows)
    int Skipped,         // records rejected
    int EpochsRun,
    int Updates,         // optimizer steps
    bool KlStopped,
    double ApproxKl,     // mean over the last epoch's minibatches, measured before each step
    double ClipFraction,
    double Entropy,
    double MeanAdvantage,// raw (pre-normalization) mean of Return - baseline
    double PolicyLoss,
    double Seconds);

/// <summary>Imitation (cross-entropy on the chosen candidate) and PPO-clip updates for <see cref="PolicyNet"/>.</summary>
public static class PolicyTrainer
{
    static bool Usable(DecisionRecord r, int d) =>
        r.Rows >= 2 && (uint)r.Chosen < (uint)r.Rows && r.Candidates.Length >= r.Rows * d;

    /// <summary>Copies the decisions' rows into ws, standardizes, runs forward. starts[i]..starts[i+1] = rows of decision i.</summary>
    static int ForwardBatch(PolicyNet net, IReadOnlyList<DecisionRecord> recs, ReadOnlySpan<int> idx, PolicyNet.Workspace ws, int[] starts)
    {
        int d = net.InputCount;
        int total = 0;
        for (int i = 0; i < idx.Length; i++) total += recs[idx[i]].Rows;
        ws.Ensure(net.Sizes, total);
        int o = 0;
        for (int i = 0; i < idx.Length; i++)
        {
            var r = recs[idx[i]];
            starts[i] = o;
            r.Candidates.AsSpan(0, r.Rows * d).CopyTo(ws.Raw.AsSpan(o * d));
            o += r.Rows;
        }
        starts[idx.Length] = o;
        net.Standardize(ws.Raw, total, ws.Act[0]);
        net.Forward(ws, total);
        return total;
    }

    /// <summary>Softmax(logits/T) into p (double). Returns log-sum-exp offset info via logp of chosen.</summary>
    static void Softmax(ReadOnlySpan<float> z, double invT, Span<double> p)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < z.Length; i++) max = Math.Max(max, z[i] * invT);
        double sum = 0;
        for (int i = 0; i < z.Length; i++) { p[i] = Math.Exp(z[i] * invT - max); sum += p[i]; }
        for (int i = 0; i < z.Length; i++) p[i] /= sum;
    }

    // ------------------------------------------------------------------ imitation

    /// <summary>
    /// Imitation loss for one group. Writes scale * dNLL/dlogit into dOut. Returns NLL; top1 = argmax is the chosen row.
    /// </summary>
    static double ImitationGroup(ReadOnlySpan<float> z, int chosen, double temperature, double scale, Span<float> dOut, Span<double> p, out bool top1)
    {
        double invT = 1.0 / Math.Max(1e-6, temperature);
        double nll = -PolicyNet.LogProb(z, chosen, temperature);
        Softmax(z, invT, p[..z.Length]);
        int arg = 0;
        for (int i = 0; i < z.Length; i++)
        {
            if (z[i] > z[arg]) arg = i;
            dOut[i] = (float)(scale * (p[i] - (i == chosen ? 1.0 : 0.0)) * invT);
        }
        top1 = arg == chosen;
        return nll;
    }

    /// <summary>
    /// Trains (or fine-tunes <paramref name="init"/>, which is not modified) a policy to reproduce each decision's Chosen row.
    /// Normalization is fit on the training rows only when init is null. Early-stops on weighted validation NLL and
    /// returns the best-epoch weights; results are written to Metadata (il_valNll, il_valTop1, il_trainTop1, ...).
    /// </summary>
    public static PolicyNet Imitate(IReadOnlyList<DecisionRecord> decisions, ImitationOptions o, PolicyNet? init = null)
    {
        var log = o.Log ?? (_ => { });
        var net = init?.Clone() ?? new PolicyNet(Features.Count, seed: o.Seed);
        int d = net.InputCount;
        var usable = new List<DecisionRecord>(decisions.Count);
        var weights = new List<float>(decisions.Count);
        int dropped = 0;
        foreach (var r in decisions)
        {
            float w = o.WeightFn?.Invoke(r) ?? 1f;
            if (!Usable(r, d) || !(w > 0) || !float.IsFinite(w)) { dropped++; continue; }
            usable.Add(r); weights.Add(w);
        }
        if (usable.Count == 0) throw new ArgumentException("No usable decisions (need >= 2 rows, valid Chosen, positive weight).", nameof(decisions));

        var rng = new Random(o.Seed);
        var order = Enumerable.Range(0, usable.Count).ToArray();
        rng.Shuffle(order);
        int nVal = usable.Count >= 20 ? Math.Clamp((int)(usable.Count * o.ValidationFraction), 1, usable.Count - 1) : 0;
        var valIdx = order[..nVal];
        var trainIdx = order[nVal..];

        if (init is null)
        {
            var trainRecs = trainIdx.Select(i => usable[i]).ToList();
            net.FitNormalization(trainRecs);
        }

        int bs = Math.Max(1, o.BatchDecisions);
        var ws = net.NewWorkspace(bs * 16);
        var g = new PolicyNet.Grads(net);
        var adam = new PolicyNet.AdamState(net);
        var starts = new int[bs + 1];
        var p = new double[256];

        (double nll, double top1) Eval(int[] idx) => EvalCore(net, usable, weights, idx, o.Temperature, ws);

        var (bestVal, valTop1) = nVal > 0 ? Eval(valIdx) : Eval(trainIdx);
        double initialVal = bestVal;
        var best = net.Clone();
        int bestEpoch = 0, bad = 0, epochsRun = 0;
        long rowsSeen = 0;
        var sw = Stopwatch.StartNew();
        double trainTop1 = 0, trainNll = 0;
        log($"imitate: {usable.Count} decisions ({dropped} dropped), train={trainIdx.Length} val={nVal}, initial valNll={initialVal:F4} top1={valTop1:P1}");

        for (int epoch = 1; epoch <= o.Epochs; epoch++)
        {
            rng.Shuffle(trainIdx);
            double sumNll = 0, sumW = 0; int hits = 0;
            for (int b0 = 0; b0 < trainIdx.Length; b0 += bs)
            {
                var idx = trainIdx.AsSpan(b0, Math.Min(bs, trainIdx.Length - b0));
                int total = ForwardBatch(net, usable, idx, ws, starts);
                rowsSeen += total;
                double wSum = 0;
                foreach (int i in idx) wSum += weights[i];
                var logits = ws.Act[^1];
                var dOut = ws.Delta[^1];
                for (int k = 0; k < idx.Length; k++)
                {
                    var r = usable[idx[k]];
                    int s = starts[k], n = r.Rows;
                    if (p.Length < n) p = new double[n * 2];
                    double w = weights[idx[k]];
                    double nll = ImitationGroup(logits.AsSpan(s, n), r.Chosen, o.Temperature, w / wSum, dOut.AsSpan(s, n), p, out bool t1);
                    sumNll += w * nll; sumW += w;
                    if (t1) hits++;
                }
                net.Backward(ws, total, g);
                net.AdamStep(g, adam, o.LearningRate, o.MaxGradNorm);
            }
            epochsRun = epoch;
            trainNll = sumNll / Math.Max(1e-12, sumW);
            trainTop1 = (double)hits / Math.Max(1, trainIdx.Length);
            if (nVal == 0)
            {
                best = net.Clone(); bestEpoch = epoch; bestVal = trainNll;
                log($"imitate epoch {epoch}: trainNll={trainNll:F4} trainTop1={trainTop1:P1}");
                continue;
            }
            var (vn, vt) = Eval(valIdx);
            log($"imitate epoch {epoch}: trainNll={trainNll:F4} trainTop1={trainTop1:P1} valNll={vn:F4} valTop1={vt:P1}");
            if (vn < bestVal - 1e-5)
            {
                bestVal = vn; valTop1 = vt; bestEpoch = epoch; bad = 0;
                best = net.Clone();
            }
            else if (++bad >= o.Patience) break;
        }

        double rowsPerSec = rowsSeen / Math.Max(1e-9, sw.Elapsed.TotalSeconds);
        var ci = CultureInfo.InvariantCulture;
        best.Metadata["engine"] = "ppo";
        best.Metadata["stage"] = "imitation";
        best.Metadata["il_decisions"] = usable.Count.ToString(ci);
        best.Metadata["il_epochsRun"] = epochsRun.ToString(ci);
        best.Metadata["il_bestEpoch"] = bestEpoch.ToString(ci);
        best.Metadata["il_initialValNll"] = initialVal.ToString("F5", ci);
        best.Metadata["il_valNll"] = bestVal.ToString("F5", ci);
        best.Metadata["il_valTop1"] = valTop1.ToString("F4", ci);
        best.Metadata["il_trainTop1"] = trainTop1.ToString("F4", ci);
        best.Metadata["il_rowsPerSec"] = rowsPerSec.ToString("F0", ci);
        best.Metadata["temperature"] = o.Temperature.ToString("R", ci);
        log($"imitate done: bestEpoch={bestEpoch} valNll={bestVal:F4} valTop1={valTop1:P1} rows/s={rowsPerSec:F0} ({sw.Elapsed.TotalSeconds:F1}s)");
        return best;
    }

    static (double nll, double top1) EvalCore(PolicyNet net, List<DecisionRecord> recs, List<float> weights, int[] idx, double temperature, PolicyNet.Workspace ws)
    {
        if (idx.Length == 0) return (double.NaN, double.NaN);
        const int chunk = 256;
        var starts = new int[chunk + 1];
        double sumNll = 0, sumW = 0; int hits = 0;
        for (int b0 = 0; b0 < idx.Length; b0 += chunk)
        {
            var span = idx.AsSpan(b0, Math.Min(chunk, idx.Length - b0));
            ForwardBatch(net, recs, span, ws, starts);
            var logits = ws.Act[^1];
            for (int k = 0; k < span.Length; k++)
            {
                var r = recs[span[k]];
                var z = logits.AsSpan(starts[k], r.Rows);
                double w = weights[span[k]];
                sumNll += -w * PolicyNet.LogProb(z, r.Chosen, temperature);
                sumW += w;
                int arg = 0;
                for (int i = 1; i < z.Length; i++) if (z[i] > z[arg]) arg = i;
                if (arg == r.Chosen) hits++;
            }
        }
        return (sumNll / Math.Max(1e-12, sumW), (double)hits / idx.Length);
    }

    /// <summary>Unweighted mean NLL and top-1 agreement of a policy on decisions (records with &lt; 2 rows are ignored).</summary>
    public static (double Nll, double Top1) EvaluateImitation(PolicyNet net, IReadOnlyList<DecisionRecord> decisions, double temperature = 1.0)
    {
        var recs = decisions.Where(r => Usable(r, net.InputCount)).ToList();
        var w = Enumerable.Repeat(1f, recs.Count).ToList();
        return EvalCore(net, recs, w, Enumerable.Range(0, recs.Count).ToArray(), temperature, net.NewWorkspace(4096));
    }

    /// <summary>Mean imitation NLL over the batch and its exact gradient (flattened W0,B0,W1,B1,...). For gradient checks.</summary>
    public static double ImitationLossAndGradient(PolicyNet net, IReadOnlyList<DecisionRecord> batch, float[] gradOut, double temperature = 1.0)
    {
        var idx = Enumerable.Range(0, batch.Count).ToArray();
        var ws = net.NewWorkspace(256);
        var starts = new int[idx.Length + 1];
        int total = ForwardBatch(net, batch, idx, ws, starts);
        var p = new double[batch.Max(r => r.Rows)];
        double loss = 0;
        for (int k = 0; k < idx.Length; k++)
        {
            var r = batch[k];
            loss += ImitationGroup(ws.Act[^1].AsSpan(starts[k], r.Rows), r.Chosen, temperature, 1.0 / idx.Length,
                ws.Delta[^1].AsSpan(starts[k], r.Rows), p, out _);
        }
        var g = new PolicyNet.Grads(net);
        net.Backward(ws, total, g);
        PolicyNet.Flatten(g, gradOut);
        return loss / idx.Length;
    }

    // ------------------------------------------------------------------ PPO

    /// <summary>
    /// PPO-clip loss for one group (mean-scaled by scale): -min(r A, clip(r) A) - c H. Writes scale * dLoss/dlogit.
    /// </summary>
    static double PpoGroup(ReadOnlySpan<float> z, int chosen, float oldLogp, double adv, PpoOptions o, double scale,
        Span<float> dOut, Span<double> p, out double newLogp, out double entropy, out bool clipped)
    {
        double T = o.Temperature;
        double invT = 1.0 / Math.Max(1e-6, T);
        newLogp = PolicyNet.LogProb(z, chosen, T);
        Softmax(z, invT, p[..z.Length]);
        entropy = 0;
        for (int i = 0; i < z.Length; i++) if (p[i] > 0) entropy -= p[i] * Math.Log(p[i]);
        double ratio = Math.Exp(Math.Clamp(newLogp - oldLogp, -20, 20));
        double eps = o.Clip;
        double clippedRatio = Math.Clamp(ratio, 1 - eps, 1 + eps);
        clipped = (adv > 0 && ratio > 1 + eps) || (adv < 0 && ratio < 1 - eps);
        double loss = -Math.Min(ratio * adv, clippedRatio * adv) - o.EntropyCoef * entropy;
        double dLogp = clipped ? 0.0 : -ratio * adv; // d loss / d newLogp
        for (int i = 0; i < z.Length; i++)
        {
            double dlogpDz = (i == chosen ? 1.0 : 0.0) - p[i];
            double logPi = p[i] > 0 ? Math.Log(p[i]) : 0;
            double dEntLoss = o.EntropyCoef * p[i] * (logPi + entropy); // d(-c H)/dz_i
            dOut[i] = (float)(scale * (dLogp * dlogpDz + dEntLoss) * invT);
        }
        return loss;
    }

    /// <summary>
    /// One PPO-clip update of <paramref name="policy"/> (in place) on decisions gathered with it (or a recent copy).
    /// Advantage = Return - baseline(record), normalized over the batch. Uses only records with finite BehaviorLogProb,
    /// finite Return, valid Chosen and at least 2 rows. Input normalization is left unchanged.
    /// </summary>
    public static PpoStats PpoUpdate(PolicyNet policy, IReadOnlyList<DecisionRecord> onPolicy, Func<DecisionRecord, float> baseline, PpoOptions o)
    {
        var log = o.Log ?? (_ => { });
        var sw = Stopwatch.StartNew();
        int d = policy.InputCount;
        var recs = new List<DecisionRecord>(onPolicy.Count);
        var adv = new List<double>(onPolicy.Count);
        int skipped = 0;
        foreach (var r in onPolicy)
        {
            if (!Usable(r, d) || !float.IsFinite(r.BehaviorLogProb) || !float.IsFinite(r.Return)) { skipped++; continue; }
            double a = r.Return - baseline(r);
            if (!double.IsFinite(a)) { skipped++; continue; }
            recs.Add(r); adv.Add(a);
        }
        if (recs.Count == 0)
        {
            log($"ppo: no usable decisions ({skipped} skipped)");
            return new PpoStats(0, skipped, 0, 0, false, 0, 0, 0, 0, 0, sw.Elapsed.TotalSeconds);
        }
        double mean = adv.Average();
        double std = Math.Sqrt(adv.Sum(a => (a - mean) * (a - mean)) / adv.Count);
        var nAdv = adv.Select(a => (a - mean) / (std + 1e-8)).ToArray();

        int bs = Math.Max(1, o.BatchDecisions);
        var ws = policy.NewWorkspace(bs * 16);
        var g = new PolicyNet.Grads(policy);
        var adam = new PolicyNet.AdamState(policy);
        var starts = new int[bs + 1];
        var p = new double[256];
        var rng = new Random(o.Seed);
        var order = Enumerable.Range(0, recs.Count).ToArray();

        int epochsRun = 0, updates = 0;
        bool klStop = false;
        double kl = 0, clipFrac = 0, ent = 0, ploss = 0;
        for (int epoch = 1; epoch <= o.Epochs && !klStop; epoch++)
        {
            rng.Shuffle(order);
            double eKl = 0, eClip = 0, eEnt = 0, eLoss = 0; int eN = 0;
            for (int b0 = 0; b0 < order.Length; b0 += bs)
            {
                var idx = order.AsSpan(b0, Math.Min(bs, order.Length - b0));
                int total = ForwardBatch(policy, recs, idx, ws, starts);
                var logits = ws.Act[^1];
                var dOut = ws.Delta[^1];
                double bKl = 0;
                for (int k = 0; k < idx.Length; k++)
                {
                    var r = recs[idx[k]];
                    int s = starts[k], n = r.Rows;
                    if (p.Length < n) p = new double[n * 2];
                    double loss = PpoGroup(logits.AsSpan(s, n), r.Chosen, r.BehaviorLogProb, nAdv[idx[k]], o, 1.0 / idx.Length,
                        dOut.AsSpan(s, n), p, out double newLogp, out double h, out bool clipped);
                    double logRatio = newLogp - r.BehaviorLogProb;
                    double ratio = Math.Exp(Math.Clamp(logRatio, -20, 20));
                    bKl += (ratio - 1) - logRatio; // unbiased, non-negative KL(old||new) estimator
                    eEnt += h; eLoss += loss; if (clipped) eClip++;
                }
                eKl += bKl; eN += idx.Length;
                if (bKl / idx.Length > 1.5 * o.TargetKl)
                {
                    klStop = true;
                    break;
                }
                policy.Backward(ws, total, g);
                policy.AdamStep(g, adam, o.LearningRate, o.MaxGradNorm);
                updates++;
            }
            epochsRun = epoch;
            if (eN > 0) { kl = eKl / eN; clipFrac = eClip / eN; ent = eEnt / eN; ploss = eLoss / eN; }
        }

        var ci = CultureInfo.InvariantCulture;
        policy.Metadata["engine"] = "ppo";
        policy.Metadata["stage"] = "ppo";
        policy.Metadata["temperature"] = o.Temperature.ToString("R", ci);
        int prev = policy.Metadata.TryGetValue("ppo_iterations", out var s0) && int.TryParse(s0, NumberStyles.Integer, ci, out var v0) ? v0 : 0;
        policy.Metadata["ppo_iterations"] = (prev + 1).ToString(ci);
        var stats = new PpoStats(recs.Count, skipped, epochsRun, updates, klStop, kl, clipFrac, ent, mean, ploss, sw.Elapsed.TotalSeconds);
        log($"ppo: n={recs.Count} skipped={skipped} epochs={epochsRun} updates={updates} klStop={klStop} kl={kl:F4} clip={clipFrac:P1} " +
            $"entropy={ent:F3} meanAdv={mean:F4} loss={ploss:F4} ({stats.Seconds:F2}s)");
        return stats;
    }

    /// <summary>Mean PPO loss over the batch and its exact gradient (advantages used as given). For gradient checks.</summary>
    public static double PpoLossAndGradient(PolicyNet net, IReadOnlyList<DecisionRecord> batch, IReadOnlyList<float> advantages, PpoOptions o, float[] gradOut)
    {
        var idx = Enumerable.Range(0, batch.Count).ToArray();
        var ws = net.NewWorkspace(256);
        var starts = new int[idx.Length + 1];
        int total = ForwardBatch(net, batch, idx, ws, starts);
        var p = new double[batch.Max(r => r.Rows)];
        double loss = 0;
        for (int k = 0; k < idx.Length; k++)
        {
            var r = batch[k];
            loss += PpoGroup(ws.Act[^1].AsSpan(starts[k], r.Rows), r.Chosen, r.BehaviorLogProb, advantages[k], o, 1.0 / idx.Length,
                ws.Delta[^1].AsSpan(starts[k], r.Rows), p, out _, out _, out _);
        }
        var g = new PolicyNet.Grads(net);
        net.Backward(ws, total, g);
        PolicyNet.Flatten(g, gradOut);
        return loss / idx.Length;
    }
}
