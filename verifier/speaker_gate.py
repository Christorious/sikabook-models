"""Real voice gate v0: who is speaking? CPU, numpy only.

  python3 speaker_gate.py --data data/fisd_v0 data/kasa_v0                   # v0: MFCC + LDA, numpy only
  python3 speaker_gate.py --data data/fisd_v0 data/kasa_v0 --encoder ecapa   # pretrained ECAPA-TDNN (CPU)
  python3 run_eval.py --scenarios data/fisd_v0/scenarios_gate.json --preds data/fisd_v0/preds_gate.json | tail -1

What it does (honest):
  - Voice print = log-mel + MFCC statistics -> LDA (learns what separates speakers) -> unit vector.
  - Three speaker groups, never mixed: LDA-train speakers, dev speakers (pick the threshold),
    test speakers (the same unseen speakers train_tiny.py is graded on).
  - Enrol each test vendor on ONE clean clip, like a first-day phone setup.
  - Every test clip is played twice: once as the vendor (must pass), once as a bystander
    against someone else's enrolment (must be blocked). Actions come from preds_tiny.json.
  - Run train_tiny.py first.
  - Not a deep model. ECAPA/WavLM embeddings are the next step; this is the honest floor.
"""
import argparse, os, json, random
import numpy as np
from train_tiny import FB
import soundfile as sf

GATE = 0.68   # the ledger's fixed threshold; we calibrate scores so the dev EER point lands here


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
    mu = X.mean(0); Xc = X - mu
    Sw = np.zeros((X.shape[1],) * 2); Sb = np.zeros_like(Sw)
    for s in set(spk):
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
    Score every utterance from the speaker's OTHER recordings (and noisy copies) as genuine, and against
    n_imp other enrolled vendors as impostor. `use` < k averages only the first `use` clips but keeps the
    same test utterances, so 1-clip and k-clip enrolment are compared on identical trials."""
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
            gen.append((i, s, float(E[i] @ enrol[s])))
            for v in rng.sample([t for t in spks if t != s], min(n_imp, len(spks) - 1)):
                imp.append((i, v, float(E[i] @ enrol[v])))
    return gen, imp, enrol


def eer(g, i):
    g, i = np.array(g), np.array(i)
    best = (1.0, 0.0)
    for t in np.unique(np.r_[g, i]):
        far, frr = (i >= t).mean(), (g < t).mean()
        if abs(far - frr) < best[0]: best = (abs(far - frr), t, (far + frr) / 2)
    return best[2], best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/fisd_v0"])
    ap.add_argument("--heldout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--encoder", choices=["lda", "ecapa"], default="lda")
    ap.add_argument("--enrol", type=int, default=1, help="clean enrolment clips per vendor (each a different recording)")
    a = ap.parse_args(); rng = random.Random(a.seed)
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

    dev = [i for i, m in enumerate(man) if m["speaker"] in dev_spk]
    g, im, _ = trials(E, man, dev, a.seed, n_imp=3, k=a.enrol)
    dev_eer, thr = eer([s for *_, s in g], [s for *_, s in im])
    print(f"dev:  {len(g)} genuine / {len(im)} impostor trials   EER {dev_eer:.1%}   threshold cos {thr:.3f}")

    te = [i for i, m in enumerate(man) if m["speaker"] in test_spk]
    if a.enrol > 1:   # same vendors, same test clips, only 1 enrolment clip: the fair comparison
        g1, i1, _ = trials(E, man, te, a.seed + 1, n_imp=1, k=a.enrol, use=1)
        g1, i1 = np.array([s for *_, s in g1]), np.array([s for *_, s in i1])
        print(f"test, 1 enrol clip:  EER {eer(g1, i1)[0]:.1%}   (same vendors and trials as below)")
    g, im, enrol = trials(E, man, te, a.seed + 1, n_imp=1, k=a.enrol)
    gs, is_ = np.array([s for *_, s in g]), np.array([s for *_, s in im])
    print(f"test, {a.enrol} enrol clip{'s' * (a.enrol > 1)}: {len(g)} genuine / {len(im)} impostor trials   EER {eer(gs, is_)[0]:.1%}   "
          f"at dev threshold: FAR {(is_ >= thr).mean():.1%}  FRR {(gs < thr).mean():.1%}")

    sim = lambda c: round(float(np.clip(GATE + (c - thr), 0, 1)), 3)   # dev EER point -> 0.68
    tiny = json.load(open(f"{dst}/preds_tiny.json"))
    preds, out = {}, []
    for kind, rows in (("vendor", g), ("bystander", im)):
        for i, who, c in rows:
            cid = man[i]["id"]
            if cid not in tiny: continue
            sid = f"{cid}__{kind}_{who}"
            s = dict(scen[cid], id=sid, enrolled_vendor=who, is_enrolled_vendor=kind == "vendor")
            if kind == "bystander": s["expected_state"] = {"transactions": [], "credit_book": []}
            out.append(s); preds[sid] = dict(tiny[cid], speaker_similarity=sim(c))
    json.dump(preds, open(f"{dst}/preds_gate.json", "w"), indent=1)
    json.dump(out, open(f"{dst}/scenarios_gate.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez(f"{dst}/gate_{a.encoder}.npz", mu=mu, V=V, thr=thr)
    print(f"wrote {len(out)} gate scenarios ({len(enrol)} enrolled vendors)")


if __name__ == "__main__":
    main()
