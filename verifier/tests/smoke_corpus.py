"""Build a tiny synthetic corpus so the gate pipeline can be RUN without the licensed audio.

  python3 tests/smoke_corpus.py --out data/_smoke
  python3 train_tiny.py --data data/_smoke          # writes preds_tiny.json
  python3 speaker_gate.py --data data/_smoke
  python3 run_eval.py --scenarios data/_smoke/scenarios_gate.json \
                      --preds data/_smoke/preds_gate.json --threshold 0.68 --review-threshold 0.5

This is NOT data and its numbers mean nothing: the "speakers" are harmonic stacks with different
fundamentals and formant weights, so they are far easier to tell apart than two women at adjacent
stalls. It exists so that a change to speaker_gate.py can be shown to run end to end - manifest to
voiceprints to thresholds to preds_gate.json to a graded ledger - on a machine that has neither
FISD nor KasaSpeech. Real numbers only ever come from the real corpora.
"""
import argparse, json, os
import numpy as np
import soundfile as sf

SR = 16_000
ACTIONS = ["LOG_CASH_SALE", "LOG_EXPENSE", "LOG_CREDIT_SALE"]
ITEMS = ["tilapia", "kpanla", "mackerel", "herrings"]


def say(rng, f0, weights, dur=1.5, jitter=0.0):
    """A harmonic stack with speaker-specific partial weights: enough spectral identity for the
    MFCC front end to have something to latch onto, and nothing more."""
    t = np.arange(int(SR * dur)) / SR
    f = f0 * (1 + jitter * rng.normal(size=len(t)).cumsum() / len(t))
    x = sum(w * np.sin(2 * np.pi * (k + 1) * f * t) for k, w in enumerate(weights))
    env = np.clip(np.sin(np.pi * t / dur) * 1.2, 0, 1)            # fade in and out
    x = x * env
    return (x / (np.abs(x).max() + 1e-9) * 0.7).astype(np.float32)


def add_noise(rng, x, snr_db):
    n = rng.normal(size=len(x)).astype(np.float32)
    g = np.sqrt((x ** 2).mean() / ((n ** 2).mean() + 1e-12) / 10 ** (snr_db / 10))
    y = x + g * n
    return (y / (np.abs(y).max() + 1e-9) * 0.7).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/_smoke")
    ap.add_argument("--speakers", type=int, default=40)
    ap.add_argument("--recordings", type=int, default=4)
    ap.add_argument("--snr", type=float, default=9.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(f"{a.out}/wav", exist_ok=True)
    man, scen, tiny = [], [], {}
    for si in range(a.speakers):
        spk = f"SM{si:03d}"
        f0 = rng.uniform(140, 260)                                # her voice
        w = rng.uniform(0.2, 1.0, size=8)
        for r in range(a.recordings):
            clean = say(rng, f0 * rng.uniform(0.97, 1.03), w * rng.uniform(0.85, 1.15, size=8),
                        jitter=rng.uniform(0, 0.02))
            for snr in (None, a.snr):
                cid = f"{spk}_r{r}" + ("_c" if snr is None else "_n")
                x = clean if snr is None else add_noise(rng, clean, snr)
                sf.write(f"{a.out}/wav/{cid}.wav", x, SR)
                man.append({"id": cid, "speaker": spk, "snr_db": snr, "wav": f"wav/{cid}.wav",
                            "lang": "synthetic", "text": "smoke", "label": "EXP_10"})
                amount = float(rng.choice([5, 10, 20, 50]))
                item = str(rng.choice(ITEMS))
                act = str(rng.choice(ACTIONS))
                tx = {"tx_type": "CASH_SALE" if act == "LOG_CASH_SALE" else "EXPENSE",
                      "item": item, "quantity": 1, "amount_ghs": amount}
                if act == "LOG_CREDIT_SALE":
                    exp = {"transactions": [], "credit_book": [
                        {"customer": "Ama", "item": item, "amount_owed": amount, "status": "UNPAID"}]}
                else:
                    exp = {"transactions": [tx], "credit_book": []}
                scen.append({"id": cid, "utterance": "smoke", "snr_db": snr,
                             "is_enrolled_vendor": True, "expected_state": exp})
                tiny[cid] = {"speaker_similarity": 1.0, "action": act, "item": item,
                             "quantity": 1, "amount_ghs": amount,
                             "customer_name": "Ama" if act == "LOG_CREDIT_SALE" else None,
                             "confidence": 0.9, "screen": "HOME_LEDGER"}
    with open(f"{a.out}/manifest.jsonl", "w", encoding="utf-8") as f:
        for m in man: f.write(json.dumps(m, ensure_ascii=False) + "\n")
    json.dump(scen, open(f"{a.out}/scenarios.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # The gate reads preds_tiny.json for the ACTION; writing a perfect one here isolates the gate,
    # so whatever run_eval reports afterwards is the speaker decision and nothing else.
    json.dump(tiny, open(f"{a.out}/preds_tiny.json", "w"), indent=1)
    print(f"{a.out}: {len(man)} clips, {a.speakers} synthetic speakers, "
          f"{a.recordings} recordings each (clean + {a.snr:g} dB copy)")


if __name__ == "__main__":
    main()
