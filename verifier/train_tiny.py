"""First tiny model: real Twi audio -> ledger action. CPU, numpy only, ~1 min.

  python3 train_tiny.py --data data/fisd_v0
  python3 train_tiny.py --data data/fisd_v0 data/kasa_v0      # + KasaSpeech money-talk negatives
  python3 run_eval.py --scenarios data/fisd_v0/scenarios_heldout.json --preds data/fisd_v0/preds_tiny.json | tail -1

What it is (honest):
  - log-mel features pooled over 3 time chunks -> softmax regression (a "linear probe").
  - Speaker-disjoint split: the model is graded on ~20% of speakers it NEVER heard.
  - The app knows which screen is open, so the screen masks the choices (like the real phone).
  - It does NOT verify the speaker yet: speaker_similarity is fixed at 0.85. That's the next model.
  - Low confidence (< 0.55) -> the ledger asks to repeat instead of guessing money.
"""
import argparse, json, random
from collections import Counter
import numpy as np, soundfile as sf

SR = 16000
CLASSES = ["NO_OP", "EXP_5", "EXP_10", "CONFIRM", "CANCEL", "CLARIFY"]
ON_SCREEN = {"HOME_LEDGER": {"NO_OP", "EXP_5", "EXP_10", "CLARIFY"},
             "CONFIRM_MODAL": {"CONFIRM", "CANCEL", "CLARIFY"}}


def cls_of(lab):
    a = lab["action"]
    if a == "ASK_CLARIFICATION": return "CLARIFY"
    if a == "LOG_EXPENSE": return "EXP_10" if lab["amount_ghs"] >= 10 else "EXP_5"
    return {"NO_OP": "NO_OP", "CONFIRM_ACTION": "CONFIRM", "CANCEL_ACTION": "CANCEL"}[a]


def mel_bank(n_fft=512, n_mels=40):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(60), hz2mel(7600), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / SR).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        fb[i, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[i, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


FB = mel_bank()


def features(path):
    x, _ = sf.read(path, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    win, hop = 400, 160
    n = max(1, 1 + (len(x) - win) // hop)
    frames = np.stack([x[i * hop:i * hop + win] for i in range(n)]) * np.hamming(win)
    spec = np.abs(np.fft.rfft(frames, 512)) ** 2
    m = np.log(spec @ FB.T + 1e-8)                      # (T, 40)
    energy = m.mean(1)
    m = m - m.mean(0)                                   # per-clip normalisation
    d = np.diff(m, axis=0, prepend=m[:1])               # how the sound moves
    parts = [np.r_[c.mean(0), c.std(0), dc.std(0)] for c, dc in
             zip(np.array_split(m, 3), np.array_split(d, 3))]
    flat = np.percentile(energy, 90) - np.percentile(energy, 10)  # low = buried in noise
    return np.r_[np.concatenate(parts), m.std(0), len(x) / SR, flat, energy.std()]


def train(X, y, k, l2=1e-2):
    """Convex softmax regression solved to the optimum with L-BFGS: same answer on every machine."""
    from scipy.optimize import minimize
    n, f = X.shape; Y = np.eye(k)[y]
    w = len(y) / (k * np.bincount(y, minlength=k).clip(1)); sw = w[y][:, None]  # balance rare classes

    def loss(theta):
        W = theta[:f * k].reshape(f, k); b = theta[f * k:]
        z = X @ W + b; z -= z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        g = (np.exp(lp) - Y) * sw / n
        val = -(sw * Y * lp).sum() / n + 0.5 * l2 * (W ** 2).sum()
        return val, np.r_[(X.T @ g + l2 * W).ravel(), g.sum(0)]

    r = minimize(loss, np.zeros(f * k + k), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return r.x[:f * k].reshape(f, k), r.x[f * k:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/fisd_v0"], help="first dir also receives the outputs")
    ap.add_argument("--heldout", type=float, default=0.2, help="share of speakers kept unseen")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--money-conf", type=float, default=0.75, help="below this, ask to repeat instead of writing money")
    a = ap.parse_args()
    man, scen, test_spk = [], {}, set()
    for d in a.data:   # split speakers per dataset, so the FISD held-out set never changes
        ms = [dict(json.loads(l), wav=f"{d}/" + json.loads(l)["wav"]) for l in open(f"{d}/manifest.jsonl", encoding="utf-8")]
        spk = sorted({m["speaker"] for m in ms}); random.Random(a.seed).shuffle(spk)
        test_spk |= set(spk[:max(1, int(len(spk) * a.heldout))])
        man += ms; scen.update({s["id"]: s for s in json.load(open(f"{d}/scenarios.json", encoding="utf-8"))})
    dst = a.data[0]
    print(f"featurising {len(man)} clips ...")
    X = np.stack([features(m["wav"]) for m in man])
    y = np.array([CLASSES.index(cls_of(m["label"])) for m in man])
    te = np.array([m["speaker"] in test_spk for m in man]); tr = ~te
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd
    W, b = train(X[tr], y[tr], len(CLASSES))
    np.savez(f"{dst}/tiny_model.npz", W=W, b=b, mu=mu, sd=sd, classes=CLASSES)

    z = X @ W + b; P = np.exp(z - z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    preds, scen_out, right, conf_m = {}, [], 0, Counter()
    for i in np.where(te)[0]:
        m = man[i]; screen = m["label"]["screen"]
        p = np.array([P[i, j] if c in ON_SCREEN[screen] else 0 for j, c in enumerate(CLASSES)])
        p /= p.sum(); j = int(p.argmax()); c = CLASSES[j]
        right += c == CLASSES[y[i]]; conf_m[(CLASSES[y[i]], c)] += 1
        out = {"speaker_similarity": 0.85, "item": "NONE", "quantity": 0, "amount_ghs": 0.0,
               "customer_name": None, "confidence": round(float(p[j]), 3), "screen": screen,
               "action": {"NO_OP": "NO_OP", "CONFIRM": "CONFIRM_ACTION", "CANCEL": "CANCEL_ACTION",
                          "CLARIFY": "ASK_CLARIFICATION"}.get(c, "LOG_EXPENSE")}
        if c.startswith("EXP") and p[j] < a.money_conf:
            out["action"] = "ASK_CLARIFICATION"
        elif c.startswith("EXP"):
            out.update(item="airtime", quantity=1, amount_ghs=10.0 if c == "EXP_10" else 5.0)
        preds[m["id"]] = out; scen_out.append(scen[m["id"]])
    json.dump(preds, open(f"{dst}/preds_tiny.json", "w"), indent=1)
    json.dump(scen_out, open(f"{dst}/scenarios_heldout.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n = int(te.sum())
    print(f"held-out: {n} clips from {len(test_spk)} unseen speakers   accuracy {right / n:.0%}")
    for src in sorted({m["id"].split("_")[0] for m in man}):
        ids = [man[i]["id"] for i in np.where(te)[0] if man[i]["id"].startswith(src + "_")]
        bad = sum(1 for k in ids if preds[k]["action"] == "LOG_EXPENSE" and scen[k]["expected_state"]["transactions"] == [])
        print(f"  {src}: {len(ids)} held-out clips, {bad} wrongly wrote money")
    print("mistakes (truth -> guess):", {f"{t}->{g}": k for (t, g), k in conf_m.items() if t != g} or "none")


if __name__ == "__main__":
    main()
