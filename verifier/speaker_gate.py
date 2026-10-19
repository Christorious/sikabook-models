"""Real voice gate v1: who is speaking, and how sure are we? CPU, numpy only.

  python3 speaker_gate.py --data data/fisd_v0 data/kasa_v0                          # MFCC+LDA floor
  python3 speaker_gate.py --data data/fisd_v0 data/kasa_v0 --encoder ecapa          # real encoder
  python3 speaker_gate.py --data data/fisd_v0 data/kasa_v0 --encoder ecapa --adapt  # + session adaptation
  python3 run_eval.py --scenarios data/fisd_v0/scenarios_gate.json \
                      --preds data/fisd_v0/preds_gate.json --review-threshold 0.5 | tail -2

What v0 measured, and why it was not good enough:
  dev EER 13%, test EER 26%, and at the dev-chosen threshold FAR 10% / FRR 46%. That FRR is the
  product failing: nearly half her utterances rejected means nearly half her sales never reach
  the ledger, and she is the one person the gate exists to let through.

  Caveat on those four numbers: fit_lda below used to iterate an unordered set, so the LDA basis
  depended on PYTHONHASHSEED. On the smoke corpus that alone swung dev EER between 0% and 33%.
  The v0 numbers are therefore one draw from a wide distribution, not a measurement, and they have
  to be re-run on the real corpora before anything is concluded from them. The bug is fixed here.

This version changes two things. The order below is the leverage actually measured, which is not
the order they were expected in:

  1. THE GATE RANKS, IT DOES NOT ONLY FILTER (--target-far, --target-loss). Two thresholds
     instead of one: accept, review, reject. She confirms the day's ledger at closing time
     anyway, so an uncertain utterance does not have to be discarded - it can be queued. FRR
     then trades against review burden, which we choose, instead of against coverage, which we
     cannot get back. The accept threshold is set by the impostor rate we will tolerate and the
     reject threshold by the fraction of her speech we are willing to lose for good, so the
     script prints queue size per 100 of her utterances and the trade is visible as a number.
     This is the big lever: on synthetic data it recovers coverage from roughly 75% to roughly
     96% at an unchanged accept threshold, for every calibration and every seed.

  2. PER-SPEAKER CALIBRATION (--norm znorm|asnorm). At enrolment, score her voiceprint against a
     fixed cohort of other speakers that ships with the app, and express every later score in
     units of her own cohort distribution, so that one threshold can mean the same thing for
     voices that sit close to everyone and voices that sit far from all of them.
       znorm  - enrolment side only. Costs nothing per utterance: it is exactly equivalent to
                storing one raw-cosine threshold per trader at setup (written into the .npz as
                spk_thr_hi/spk_thr_lo, which is what the phone would actually use).
       asnorm - also normalises by the test utterance's own cohort statistics. One extra K x d
                matvec per utterance.
     Measured on synthetic embeddings (tests/test_gate_math.py) and on the smoke corpus, asnorm
     consistently lowers test EER and znorm does not reliably do anything; neither reliably closed
     the dev-to-test threshold gap, which was the reason they were written. So all three are
     always reported side by side and --norm only chooses which one writes preds_gate.json: the
     real corpora decide, and the honest statement today is that this is a second-order gain that
     grows as the encoder improves.

  3. SESSION ADAPTATION (--adapt), optional and third. Across a market day the most frequent
     speaker at the stall is her. Average the utterances this gate already accepted back into her
     voiceprint and rescore. It uses no labels at all, only the gate's own confident accepts -
     but it does adapt on her own test-day audio, so it is reported as session adaptation, never
     as a held-out gain. It can also drift onto a bystander who slipped the accept tier, which is
     why the adapted numbers are printed next to the unadapted ones rather than replacing them.

Still honest about what this is:
  - Voiceprints are MFCC+LDA by default. --encoder ecapa is the real speaker encoder; the floor
    is kept so the gain from the encoder stays measurable.
  - Cohort speakers come from the LDA-train group only, never from the dev or test speakers, so
    calibration never sees a voice it is later asked to judge.
  - Enrolment is ONE clean clip unless --enrol says otherwise, like a first-day phone setup, and
    no test utterance ever comes from an enrolment recording.
  - Run train_tiny.py first (this reads its preds_tiny.json).
"""
import argparse, os, json, random
import numpy as np
from train_tiny import FB
import soundfile as sf

GATE = 0.68      # at or above this speaker_similarity the ledger writes without asking
REVIEW = 0.50    # below GATE but at or above this: held for the day-end review queue


def voiceprint_feats(path):
    x, _ = sf.read(path, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    win, hop = 400, 160
    n = max(1, 1 + (len(x) - win) // hop)
    fr = np.stack([x[i * hop:i * hop + win] for i in range(n)]) * np.hamming(win)
    m = np.log(np.abs(np.fft.rfft(fr, 512)) ** 2 @ FB.T + 1e-8)       # (T, 40) log-mel
    e = m.mean(1); m = m[e > np.percentile(e, 30)]                     # keep voiced-ish frames
    k = np.arange(40)
    dct = np.cos(np.pi / 40 * (k[:, None] + 0.5) * np.arange(20)[None])  # MFCC 0..19
    c = m @ dct
    d = np.diff(c, axis=0) if len(c) > 1 else np.zeros_like(c)
    return np.r_[c.mean(0), c.std(0), d.std(0), m.mean(0), np.percentile(c, 10, 0), np.percentile(c, 90, 0)]


def ecapa_embed(paths):
    """Pretrained ECAPA-TDNN speaker model (speechbrain/spkrec-ecapa-voxceleb, Apache-2.0), 192-d.
    Trained on VoxCeleb (no Twi), used frozen: we never fine-tune it on our test speakers."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    torch.set_num_threads(max(1, torch.get_num_threads()))
    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir="data/_ecapa", run_opts={"device": "cpu"})
    out = []
    with torch.no_grad():
        for k, p in enumerate(paths):
            x, _ = sf.read(p, dtype="float32")
            if x.ndim > 1: x = x.mean(1)
            out.append(enc.encode_batch(torch.from_numpy(x)[None]).squeeze().numpy())
            if k % 200 == 0: print(f"  ecapa {k}/{len(paths)}")
    E = np.stack(out)
    return E


def fit_lda(X, spk, dim=80, reg=1e-3):
    # sorted(set(...)), not set(...): iterating a set of strings follows PYTHONHASHSEED, which
    # changes per process. The accumulation order only perturbs Sw/Sb in the last bits, but the
    # eigenvectors of near-degenerate directions then come out reordered, and the whole gate moves
    # with them - measured on the smoke corpus, dev EER swung between 0% and 33% across hash seeds
    # with nothing else changed. Anything reported from this file was unreproducible until here.
    mu = X.mean(0); Xc = X - mu
    Sw = np.zeros((X.shape[1],) * 2); Sb = np.zeros_like(Sw)
    for s in sorted(set(spk)):
        idx = [i for i, t in enumerate(spk) if t == s]
        ms = Xc[idx].mean(0); Sb += len(idx) * np.outer(ms, ms)
        Sw += (Xc[idx] - ms).T @ (Xc[idx] - ms)
    Sw = Sw / len(X) + reg * np.eye(len(Sw)) * np.trace(Sw) / len(X) / len(Sw)
    ev, V = np.linalg.eig(np.linalg.solve(Sw, Sb / len(X)))
    V = np.real(V[:, np.argsort(-np.real(ev))[:dim]])
    return mu, V


def embed(X, mu, V):
    E = (X - mu) @ V
    E -= E.mean(0)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def trials(E, man, idx, seed, n_imp=1, k=1, use=None):
    """Enrol each speaker with k clean clips from k different recordings (voiceprint = their average).
    Returns the TRIAL PAIRS, not scores: (utterance index, whose voiceprint it is scored against).
    Every utterance from a speaker's OTHER recordings (and noisy copies) is a genuine trial, and is
    also scored against n_imp other enrolled vendors as an impostor trial. `use` < k averages only
    the first `use` clips but keeps the same test utterances, so 1-clip and k-clip enrolment are
    compared on identical trials."""
    rng, use = random.Random(seed), use or k
    by = {}
    for i in idx: by.setdefault(man[i]["speaker"], []).append(i)
    enrol, pick = {}, {}
    for s, ii in sorted(by.items()):
        first = {}
        for i in ii:
            if man[i]["snr_db"] is None: first.setdefault(man[i]["src"], i)
        if len({man[i]["src"] for i in ii}) >= k + 1 and len(first) >= k:
            ch = rng.sample(sorted(first.values()), k)
            pick[s] = {man[i]["src"] for i in ch}
            v = E[ch[:use]].mean(0); enrol[s] = v / (np.linalg.norm(v) + 1e-9)
    gen, imp = [], []
    spks = list(enrol)
    for s in spks:
        for i in by[s]:
            if man[i]["src"] in pick[s]: continue                    # never test on an enrolment recording
            gen.append((i, s))
            for v in rng.sample([t for t in spks if t != s], min(n_imp, len(spks) - 1)):
                imp.append((i, v))
    return gen, imp, enrol


def cos(E, pairs, enrol):
    return np.array([float(E[i] @ enrol[s]) for i, s in pairs])


def build_cohort(E, man, idx, size, seed, clips=3):
    """The calibration cohort, in the two forms the two sides of the normalisation need.

    models - one averaged voiceprint per cohort speaker. This is what a TEST UTTERANCE is scored
             against, because "this clip against someone else's voiceprint" is the comparison the
             test side of asnorm has to describe.
    utts   - individual clean clips, a few per cohort speaker. This is what an ENROLMENT
             VOICEPRINT is scored against, because "her voiceprint against one clip" is what an
             actual trial is.

    Keeping these apart matters more than it looks: a mean-against-mean similarity has had the
    within-speaker noise averaged out of both sides, so its mean and spread do not describe the
    single-clip scores the gate must threshold, and normalising by it biases every trader by a
    different amount - the very thing this is supposed to remove.

    Either way this is all that ships with the app: a few hundred floats per cohort speaker, no
    audio and no identities. Drawn from the LDA-train speakers only, so calibration never sees a
    voice the gate is later measured on.
    """
    by = {}
    for i in idx:
        if man[i]["snr_db"] is None: by.setdefault(man[i]["speaker"], []).append(i)
    spk = sorted(by); random.Random(seed).shuffle(spk)
    unit = lambda v: v / (np.linalg.norm(v) + 1e-9)
    models, utts = [], []
    for s in spk[:size]:
        ii = sorted(by[s])
        models.append(unit(E[ii].mean(0)))
        utts += [E[i] for i in ii[:clips]]
    if not models: return np.zeros((0, E.shape[1])), np.zeros((0, E.shape[1]))
    return np.stack(models), np.stack(utts)


def cohort_stats(V, C, topk):
    """Mean and spread of each row's similarity to its topk NEAREST cohort speakers. Nearest
    rather than all of them: the voices that sit close are the ones that actually threaten this
    voiceprint, so they set the scale that matters at the operating point."""
    S = V @ C.T
    k = max(2, min(topk, S.shape[1]))
    T = np.sort(S, axis=1)[:, -k:]
    return T.mean(1), T.std(1) + 1e-6


def normalise(pairs, raw, enrol, E, C, mode, topk):
    """Turn raw cosines into per-speaker calibrated scores. raw: exactly what v0 thresholded.
    znorm: in units of the enrolled trader's own distribution against cohort clips. asnorm: the
    symmetric version, also in units of the test utterance's distribution against cohort models."""
    models, utts = C
    if mode == "raw" or len(utts) == 0: return np.asarray(raw, dtype=float)
    names = sorted(enrol)
    mu, sd = cohort_stats(np.stack([enrol[s] for s in names]), utts, topk)
    emu = {s: mu[j] for j, s in enumerate(names)}
    esd = {s: sd[j] for j, s in enumerate(names)}
    z = np.array([(c - emu[s]) / esd[s] for (i, s), c in zip(pairs, raw)])
    if mode == "znorm" or len(models) == 0: return z
    ui = sorted({i for i, _ in pairs})
    tmu, tsd = cohort_stats(E[ui], models, topk)
    tm = {i: (tmu[j], tsd[j]) for j, i in enumerate(ui)}
    t = np.array([(c - tm[i][0]) / tm[i][1] for (i, _), c in zip(pairs, raw)])
    return 0.5 * (z + t)


def speaker_thresholds(enrol, C, topk, z_lo, z_hi):
    """The same two thresholds expressed back in raw cosine, one pair per enrolled trader. This is
    the deployable form of znorm: the phone stores two numbers at setup and compares a plain
    cosine, with no cohort arithmetic at all while she is trading."""
    models, utts = C
    if len(utts) == 0: return {}, {}
    names = sorted(enrol)
    mu, sd = cohort_stats(np.stack([enrol[s] for s in names]), utts, topk)
    return ({s: float(mu[j] + z_lo * sd[j]) for j, s in enumerate(names)},
            {s: float(mu[j] + z_hi * sd[j]) for j, s in enumerate(names)})


def eer(g, i):
    g, i = np.asarray(g), np.asarray(i)
    best = (1.0, 0.0, 1.0)
    for t in np.unique(np.r_[g, i]):
        far, frr = (i >= t).mean(), (g < t).mean()
        if abs(far - frr) < best[0]: best = (abs(far - frr), t, (far + frr) / 2)
    return best[2], best[1]


def thresholds(g, i, target_far, target_loss):
    """accept: the score above which we tolerate only target_far of impostors writing unasked.
    reject: the score below which we are willing to lose target_loss of her speech for good.
    Everything between them is the review queue. Both are chosen on DEV speakers and then applied
    unchanged to unseen test speakers - that transfer is the thing being measured."""
    hi = float(np.quantile(i, 1 - target_far)) if len(i) else 0.0
    lo = float(np.quantile(g, target_loss)) if len(g) else 0.0
    # lo above hi means the two targets are incompatible on these speakers: the score below which
    # we would drop target_loss of her speech already sits above the accept threshold, so there is
    # no room for a review band and the gate is binary again. Collapse it rather than inverting it,
    # and let main() say so - an empty review tier is a result, not a formatting accident.
    return min(lo, hi), hi


def tiers(g, i, lo, hi, ratio):
    """Where the trials land, and what the review tier costs. `ratio` is how many bystander
    utterances the stall produces per utterance of hers - the trial set's own impostor ratio is an
    artefact of how trials are built, not a market measurement, so the burden number is reported
    against an explicit assumption instead of pretending to know it."""
    g, i = np.asarray(g), np.asarray(i)
    acc_g, rej_g = float((g >= hi).mean()), float((g < lo).mean())
    acc_i, rej_i = float((i >= hi).mean()), float((i < lo).mean())
    rev_g, rev_i = max(0.0, 1 - acc_g - rej_g), max(0.0, 1 - acc_i - rej_i)   # lo == hi: no band
    return {"accept_far": acc_i, "lost": rej_g, "auto": acc_g,
            "queue_per_100": 100 * rev_g + 100 * ratio * rev_i,
            "queue_hers": 100 * rev_g, "queue_bystander": 100 * ratio * rev_i,
            "coverage": acc_g + rev_g}


def adapt_enrolments(E, pairs, scores, enrol, hi, weight, cap=20):
    """Unsupervised session adaptation. For each enrolled trader, average the utterances this gate
    already accepted into her voiceprint. No labels are used - only the gate's own accepts, which
    is all the phone would have. Trust grows with the number of accepted clips and stops at `cap`,
    so one confident mistake cannot take the voiceprint over."""
    acc = {}
    for (i, s), z in zip(pairs, scores):
        if z >= hi: acc.setdefault(s, []).append(i)
    out = {}
    for s, v in enrol.items():
        ii = acc.get(s, [])
        if not ii:
            out[s] = v; continue
        m = E[ii].mean(0); m = m / (np.linalg.norm(m) + 1e-9)
        w = weight * min(1.0, len(ii) / cap)
        u = (1 - w) * v + w * m
        out[s] = u / (np.linalg.norm(u) + 1e-9)
    return out, {s: len(v) for s, v in acc.items()}


def calibrator(lo, hi, lohi_range, eps=1e-3):
    """Monotone map from gate score to the 0..1 speaker_similarity the ledger env reads, anchored
    so that the reject threshold lands on REVIEW and the accept threshold on GATE. run_eval.py then
    reproduces exactly these tiers with --threshold 0.68 --review-threshold 0.5. Order within each
    tier is preserved, which is what makes the review queue a ranking rather than a bag.

    Each band stops eps short of the one above it because the result is rounded to 3 decimals for
    the JSON: without the gap, a score just under the accept threshold rounds up to exactly 0.68
    and silently becomes an unasked write. eps must therefore stay at or above that rounding step.
    """
    floor, ceil = lohi_range
    def f(c):
        if c >= hi: return GATE + (1 - GATE) * min(1.0, (c - hi) / max(ceil - hi, 1e-6))
        if c >= lo: return REVIEW + (GATE - eps - REVIEW) * (c - lo) / max(hi - lo, 1e-6)
        return (REVIEW - eps) * max(0.0, (c - floor) / max(lo - floor, 1e-6))
    return lambda c: round(float(np.clip(f(c), 0, 1)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/fisd_v0"])
    ap.add_argument("--heldout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--encoder", choices=["lda", "ecapa"], default="lda")
    ap.add_argument("--enrol", type=int, default=1, help="clean enrolment clips per vendor (each a different recording)")
    ap.add_argument("--norm", choices=["raw", "znorm", "asnorm"], default="asnorm",
                    help="which calibration writes preds_gate.json; all three are always reported")
    ap.add_argument("--cohort", type=int, default=64, help="cohort speakers shipped for calibration")
    ap.add_argument("--cohort-clips", type=int, default=3, help="clips kept per cohort speaker")
    ap.add_argument("--topk", type=int, default=20, help="nearest cohort speakers used for the statistics")
    ap.add_argument("--target-far", type=float, default=0.01, help="impostors we tolerate writing unasked")
    ap.add_argument("--target-loss", type=float, default=0.02, help="her speech we are willing to lose for good")
    ap.add_argument("--bystander-ratio", type=float, default=3.0,
                    help="bystander utterances per utterance of hers at the stall, for the queue number")
    ap.add_argument("--adapt", action="store_true", help="unsupervised session adaptation on accepted audio")
    ap.add_argument("--adapt-weight", type=float, default=0.5)
    a = ap.parse_args()
    man, test_spk, dev_spk, scen = [], set(), set(), {}
    for d in a.data:   # same test speakers as train_tiny.py; dev carved from the rest
        ms = []
        for l in open(f"{d}/manifest.jsonl", encoding="utf-8"):
            m = json.loads(l); m["wav"] = f"{d}/" + m["wav"]
            m["src"] = m["id"] if m["snr_db"] is None else ms[-1]["src"]   # noisy copy follows its clean clip
            ms.append(m)
        spk = sorted({m["speaker"] for m in ms}); random.Random(a.seed).shuffle(spk)
        k = max(1, int(len(spk) * a.heldout))
        test_spk |= set(spk[:k]); dev_spk |= set(spk[k:k + max(1, int(len(spk) * 0.15))])
        man += ms; scen.update({s["id"]: s for s in json.load(open(f"{d}/scenarios.json", encoding="utf-8"))})
    dst = a.data[0]
    print(f"voiceprints ({a.encoder}) for {len(man)} clips ...")
    tr = [i for i, m in enumerate(man) if m["speaker"] not in test_spk | dev_spk]
    if a.encoder == "ecapa":
        cache = f"data/_ecapa/emb_{len(man)}.npy"     # embeddings take minutes on CPU; reuse them
        if os.path.exists(cache): E = np.load(cache)
        else: E = ecapa_embed([m["wav"] for m in man]); os.makedirs("data/_ecapa", exist_ok=True); np.save(cache, E)
        E -= E[tr].mean(0)                     # centre on training speakers only
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        mu = V = np.zeros(1)
    else:
        X = np.stack([voiceprint_feats(m["wav"]) for m in man])
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        mu, V = fit_lda(X[tr], [man[i]["speaker"] for i in tr])
        E = embed(X, mu, V)

    C = build_cohort(E, man, tr, a.cohort, a.seed, a.cohort_clips)
    print(f"cohort: {len(C[0])} speakers / {len(C[1])} clips "
          f"(of {len({man[i]['speaker'] for i in tr})} LDA-train speakers), top-{a.topk} per score")

    dev = [i for i, m in enumerate(man) if m["speaker"] in dev_spk]
    te = [i for i, m in enumerate(man) if m["speaker"] in test_spk]
    dg, di, denrol = trials(E, man, dev, a.seed, n_imp=3, k=a.enrol)
    tg, ti, tenrol = trials(E, man, te, a.seed + 1, n_imp=1, k=a.enrol)
    dgc, dic = cos(E, dg, denrol), cos(E, di, denrol)
    tgc, tic = cos(E, tg, tenrol), cos(E, ti, tenrol)
    print(f"dev:  {len(dg)} genuine / {len(di)} impostor trials over {len(denrol)} enrolled speakers")
    print(f"test: {len(tg)} genuine / {len(ti)} impostor trials over {len(tenrol)} enrolled speakers")
    if len(di) < 1 / a.target_far:
        print(f"  ! only {len(di)} dev impostor trials for a {a.target_far:.0%} target FAR: "
              f"the accept threshold is at the edge of what this many trials can resolve")

    if a.enrol > 1:   # same vendors, same test clips, only 1 enrolment clip: the fair comparison
        g1, i1, e1 = trials(E, man, te, a.seed + 1, n_imp=1, k=a.enrol, use=1)
        n1 = (normalise(g1, cos(E, g1, e1), e1, E, C, a.norm, a.topk),
              normalise(i1, cos(E, i1, e1), e1, E, C, a.norm, a.topk))
        print(f"test, 1 enrol clip ({a.norm}): EER {eer(*n1)[0]:.1%}   (same vendors and trials as below)")

    print(f"\n{'calib':<7} {'dev EER':>7} {'test EER':>8} | one dev threshold, unseen speakers: "
          f"{'FAR':>6} {'FRR':>6} | three tiers: {'acc.FAR':>8} {'lost':>6} {'auto':>6} {'queue/100':>10}")
    chosen = {}
    for mode in ("raw", "znorm", "asnorm"):
        gd = normalise(dg, dgc, denrol, E, C, mode, a.topk)
        idv = normalise(di, dic, denrol, E, C, mode, a.topk)
        gt = normalise(tg, tgc, tenrol, E, C, mode, a.topk)
        it = normalise(ti, tic, tenrol, E, C, mode, a.topk)
        dev_eer, thr = eer(gd, idv)                       # v0's single-threshold operating point
        test_eer, _ = eer(gt, it)
        lo, hi = thresholds(gd, idv, a.target_far, a.target_loss)
        t = tiers(gt, it, lo, hi, a.bystander_ratio)
        print(f"{mode:<7} {dev_eer:>6.1%} {test_eer:>8.1%} | "
              f"{(it >= thr).mean():>6.1%} {(gt < thr).mean():>6.1%} | "
              f"{t['accept_far']:>8.1%} {t['lost']:>6.1%} {t['auto']:>6.1%} {t['queue_per_100']:>10.0f}")
        if mode == a.norm:
            chosen = dict(mode=mode, gd=gd, idv=idv, gt=gt, it=it, lo=lo, hi=hi, thr=thr, tiers=t)
    print("  FAR/FRR are v0's operating point: one threshold picked on dev, applied to unseen speakers.")
    print("  acc.FAR = impostors that auto-write; lost = her speech dropped for good (NOT queued);")
    print(f"  auto = her speech written with no review; queue/100 = utterances to confirm per 100 of"
          f" hers, at {a.bystander_ratio:g} bystander utterances per one of hers.")

    lo, hi = chosen["lo"], chosen["hi"]
    gt, it = chosen["gt"], chosen["it"]
    t = chosen["tiers"]
    if lo >= hi:
        print(f"\n! no review band at these targets: the {a.target_loss:.0%} loss threshold sits at or "
              f"above the {a.target_far:.0%} FAR threshold, so the gate is binary again. Either accept "
              f"more impostors (--target-far) or be willing to drop more of her speech (--target-loss).")
    print(f"\noperating point ({chosen['mode']}): reject below {lo:.3f}, review to {hi:.3f}, accept above. "
          f"coverage {t['coverage']:.1%} of her speech reaches the ledger (auto {t['auto']:.1%} "
          f"+ confirmed {t['coverage'] - t['auto']:.1%}); queue {t['queue_per_100']:.0f} per 100 of hers "
          f"({t['queue_hers']:.0f} hers + {t['queue_bystander']:.0f} bystander)")

    adapted = None
    if a.adapt:
        pairs = list(tg) + list(ti)
        scores = np.r_[gt, it]
        enrol2, n_acc = adapt_enrolments(E, pairs, scores, tenrol, hi, a.adapt_weight)
        gt2 = normalise(tg, cos(E, tg, enrol2), enrol2, E, C, chosen["mode"], a.topk)
        it2 = normalise(ti, cos(E, ti, enrol2), enrol2, E, C, chosen["mode"], a.topk)
        t2 = tiers(gt2, it2, lo, hi, a.bystander_ratio)
        clips = sorted(n_acc.values())
        print(f"session adaptation: {len([v for v in n_acc.values() if v])}/{len(tenrol)} vendors adapted, "
              f"median {clips[len(clips) // 2] if clips else 0} accepted clips each")
        print(f"  test EER {eer(gt, it)[0]:.1%} -> {eer(gt2, it2)[0]:.1%}   "
              f"acc.FAR {t['accept_far']:.1%} -> {t2['accept_far']:.1%}   "
              f"lost {t['lost']:.1%} -> {t2['lost']:.1%}   "
              f"queue/100 {t['queue_per_100']:.0f} -> {t2['queue_per_100']:.0f}")
        print("  adapted on her own test-day audio with no labels: a session gain, not a held-out one.")
        adapted = (enrol2, gt2, it2)

    if adapted: tenrol, gt, it = adapted
    sim = calibrator(lo, hi, (float(np.r_[gt, it].min()), float(np.r_[gt, it].max())))
    tiny = json.load(open(f"{dst}/preds_tiny.json"))
    preds, out = {}, []
    for kind, rows, sc in (("vendor", tg, gt), ("bystander", ti, it)):
        for (i, who), z in zip(rows, sc):
            cid = man[i]["id"]
            if cid not in tiny: continue
            sid = f"{cid}__{kind}_{who}"
            s = dict(scen[cid], id=sid, enrolled_vendor=who, is_enrolled_vendor=kind == "vendor")
            if kind == "bystander": s["expected_state"] = {"transactions": [], "credit_book": []}
            out.append(s); preds[sid] = dict(tiny[cid], speaker_similarity=sim(z))
    json.dump(preds, open(f"{dst}/preds_gate.json", "w"), indent=1)
    json.dump(out, open(f"{dst}/scenarios_gate.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    s_lo, s_hi = speaker_thresholds(tenrol, C, a.topk, lo, hi)
    names = sorted(tenrol)
    np.savez(f"{dst}/gate_{a.encoder}.npz",
             mu=mu, V=V, thr=chosen["thr"],                    # v0 fields, unchanged for old readers
             cohort=C[0], cohort_clips=C[1], norm=chosen["mode"], topk=a.topk, z_lo=lo, z_hi=hi,
             spk=np.array(names),                              # per-trader raw-cosine thresholds:
             spk_thr_lo=np.array([s_lo.get(s, np.nan) for s in names]),   # what the phone stores at
             spk_thr_hi=np.array([s_hi.get(s, np.nan) for s in names]))   # setup, no cohort at runtime
    print(f"\nwrote {len(out)} gate scenarios ({len(tenrol)} enrolled vendors). Grade them with:")
    print(f"  python3 run_eval.py --scenarios {dst}/scenarios_gate.json --preds {dst}/preds_gate.json "
          f"--threshold {GATE} --review-threshold {REVIEW} | tail -2")


if __name__ == "__main__":
    main()
