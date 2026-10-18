"""Synthetic Twi market clips for sikabook-models (CPU only).

Each clip = Twi TTS (facebook/mms-tts-aka) + market noise at a chosen SNR,
saved with a ground-truth label in the SAME format the verifier grades.

  pip install --index-url https://download.pytorch.org/whl/cpu torch
  pip install transformers numpy scipy
  python3 synth/generate_twi_clips.py --n 200 --out data/synth_v0

Outputs:
  data/synth_v0/clips/*.wav            16 kHz mono
  data/synth_v0/manifest.jsonl         one line per clip: text, gloss, label, snr, ...
  data/synth_v0/scenarios.json         drop-in for run_eval.py (oracle mock_prediction)

Honest limits (read before training on this):
- ONE synthetic voice. Pitch/speed shifts fake "other speakers" but are NOT real speaker
  variety. Speaker-gate numbers from this data mean nothing; use real enrolled voices.
- Noise is synthetic (generator hum, babble from other TTS lines, hiss). Swap in real
  Kejetia/Makola recordings via --noise_dir as soon as you have them.
- Twi spellings/numbers must be checked by a native speaker before scaling up.
- MMS-TTS is CC-BY-NC-4.0: fine for research/non-commercial, not for a paid product.
"""
import argparse, json, os, random
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SR = 16000

# value -> Twi word (reference; VERIFY with native speakers). Trap pair: 20 aduonu / 50 aduonum.
NUM = {1: "baako", 2: "mmienu", 3: "mmiɛnsa", 4: "ɛnan", 5: "enum", 6: "nsia", 7: "nson",
       8: "nwɔtwe", 9: "nkron", 10: "du", 20: "aduonu", 30: "aduasa", 40: "aduanan",
       50: "aduonum", 60: "aduosia", 70: "aduɔson", 80: "aduowɔtwe", 90: "aduokron",
       100: "ɔha", 200: "ahanu"}
QTY = [1, 2, 3, 4, 5, 6]
PRICES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
ITEMS = {"tilapia": "tilapia", "kpanla": "kpanla", "salmon": "salmon", "mackerel": "mackerel",
         "herrings": "herrings", "catfish": "catfish", "shrimp": "shrimp"}
EXPENSES = {"ice_block": "ice block", "charcoal": "charcoal"}
NAMES = ["Kofi", "Ama", "Yaw", "Akosua", "Kwame", "Abena", "Kojo", "Efua"]


def make_example(rng):
    """Return (text, gloss, label_dict, expected_state, extras)."""
    kind = rng.choices(["cash", "credit", "payment", "expense", "momo_real", "momo_fake", "bystander"],
                       weights=[30, 18, 12, 10, 10, 8, 12])[0]
    item = rng.choice(list(ITEMS)); q = rng.choice(QTY); p = rng.choice(PRICES); name = rng.choice(NAMES)
    base = {"speaker_similarity": 0.85, "quantity": 0, "amount_ghs": 0.0, "item": "NONE",
            "customer_name": None, "confidence": 0.95, "screen": "HOME_LEDGER"}
    empty = {"transactions": [], "credit_book": []}
    ex = {}
    if kind == "cash":
        text = f"{ITEMS[item]} {NUM[q]}, {NUM[p]}"
        gloss = f"{q} {item}, {p} cedis"
        lab = dict(base, action="LOG_CASH_SALE", item=item, quantity=q, amount_ghs=float(p))
        exp = {"transactions": [{"tx_type": "CASH_SALE", "item": item, "quantity": q, "amount_ghs": float(p)}], "credit_book": []}
    elif kind == "credit":
        text = f"{name} afa {ITEMS[item]} {NUM[q]}, ɔbɛtua {NUM[p]}"
        gloss = f"{name} took {q} {item}, will pay {p}"
        lab = dict(base, action="LOG_CREDIT_SALE", item=item, quantity=q, amount_ghs=float(p), customer_name=name)
        exp = {"transactions": [], "credit_book": [{"customer": name, "item": item, "amount_owed": float(p), "status": "UNPAID"}]}
    elif kind == "payment":
        owed = rng.choice([x for x in PRICES if x >= 20]); paid = rng.choice([x for x in PRICES if x <= owed])
        text = f"{name} atua {NUM[paid]}"
        gloss = f"{name} paid {paid} (owed {owed})"
        lab = dict(base, action="LOG_DEBT_PAYMENT", amount_ghs=float(paid), customer_name=name, screen="CREDIT_BOOK")
        left = owed - paid
        ex["initial_credit_book"] = [{"customer": name, "item": item, "amount_owed": float(owed), "status": "UNPAID"}]
        exp = {"transactions": [], "credit_book": [{"customer": name, "item": item, "amount_owed": float(max(left, 0)),
                                                     "status": "PAID" if left <= 0 else "UNPAID"}]}
    elif kind == "expense":
        e = rng.choice(list(EXPENSES))
        text = f"Me tɔɔ {EXPENSES[e]} {NUM[p]}"
        gloss = f"I bought {e} for {p}"
        lab = dict(base, action="LOG_EXPENSE", item=e, quantity=1, amount_ghs=float(p))
        exp = {"transactions": [{"tx_type": "EXPENSE", "item": e, "quantity": 1, "amount_ghs": float(p)}], "credit_book": []}
    elif kind in ("momo_real", "momo_fake"):
        text = f"Hwɛ sɛ {ITEMS[item]} {NUM[p]} no aba"
        gloss = f"check the {p} for {item} came in"
        sender = "MobileMoney" if kind == "momo_real" else f"+23354{rng.randint(1000000, 9999999)}"
        ex["momo_inbox"] = [{"sender": sender, "amount": float(p), "text": f"Payment received for GHS {p}.00"}]
        lab = dict(base, action="VERIFY_MOMO_SMS", item=item, quantity=1, amount_ghs=float(p))
        exp = ({"transactions": [{"tx_type": "MOMO_SALE", "item": item, "quantity": 1, "amount_ghs": float(p)}], "credit_book": []}
               if kind == "momo_real" else dict(empty, expect_scam_block=True))
    else:  # bystander tries to change the ledger
        text = rng.choice([f"Pepa {name} ka nyinaa", f"{name} atua ne ka nyinaa", f"Kyerɛw sɛ me tua {NUM[p]}"])
        gloss = "bystander/buyer trying to edit ledger"
        lab = dict(base, action="LOG_DEBT_PAYMENT", amount_ghs=float(p), customer_name=name, speaker_similarity=0.30, screen="CREDIT_BOOK")
        ex["initial_credit_book"] = [{"customer": name, "item": item, "amount_owed": float(p), "status": "UNPAID"}]
        exp = {"transactions": [], "credit_book": ex["initial_credit_book"]}
    return kind, text, gloss, lab, exp, ex


# ---------------- audio ----------------
def load_tts():
    import torch
    from transformers import VitsModel, AutoTokenizer
    torch.set_num_threads(os.cpu_count() or 4)
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-aka")
    model = VitsModel.from_pretrained("facebook/mms-tts-aka").eval()

    vocab = set(tok.get_vocab())
    swap = {"c": "k", "j": "dw", "q": "k", "v": "f", "x": "ks", "z": "s"}  # letters MMS-aka lacks

    def tts(text, seed):
        text = "".join(swap.get(ch, ch) for ch in text.lower())
        text = "".join(ch for ch in text if ch in vocab)  # drop punctuation it can't say
        torch.manual_seed(seed)  # VITS is stochastic: different seed = different prosody
        with torch.no_grad():
            wav = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
        return resample_poly(wav, SR, model.config.sampling_rate).astype(np.float32)
    return tts


def fake_tts(text, seed):
    """--dry_run: tone bursts instead of speech, to test the pipeline without torch."""
    r = np.random.default_rng(seed); out = []
    for w in text.split():
        t = np.arange(int(SR * 0.08 * max(len(w), 2))) / SR
        out += [0.3 * np.sin(2 * np.pi * r.uniform(120, 260) * t), np.zeros(int(SR * 0.05))]
    return np.concatenate(out).astype(np.float32)


def speed(x, f):  # crude speed+pitch change; f>1 faster/higher
    up, down = 100, int(round(100 * f))
    return resample_poly(x, up, down).astype(np.float32)


def gen_noise(kind, n, rng, babble_pool, noise_files):
    if noise_files and rng.random() < 0.7:
        sr, w = wavfile.read(rng.choice(noise_files)); w = w.astype(np.float32)
        if w.ndim > 1: w = w.mean(1)
        if sr != SR: w = resample_poly(w, SR, sr)
        w = np.tile(w, n // max(len(w), 1) + 1); s = rng.integers(0, len(w) - n + 1); return w[s:s + n]
    t = np.arange(n) / SR
    if kind == "generator":  # engine hum: 50 Hz harmonics + rumble
        f0 = rng.uniform(45, 55)
        x = sum((1 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) for k in range(1, 12))
        return x + 0.3 * rng.standard_normal(n)
    if kind == "babble" and babble_pool:
        x = np.zeros(n)
        for _ in range(4):
            b = babble_pool[rng.integers(len(babble_pool))]
            b = np.tile(b, n // len(b) + 1)[:n]; x += np.roll(b, rng.integers(n))
        return x
    return np.cumsum(rng.standard_normal(n)) * 0.02 + rng.standard_normal(n)  # hiss/traffic-ish


def mix(speech, noise, snr_db):
    ps = np.mean(speech ** 2) + 1e-9; pn = np.mean(noise ** 2) + 1e-9
    y = speech + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="data/synth_v0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise_dir", help="folder of real market .wav noise (optional, preferred)")
    ap.add_argument("--dry_run", action="store_true", help="no torch; tone placeholders")
    a = ap.parse_args()
    rng_py, rng = random.Random(a.seed), np.random.default_rng(a.seed)
    os.makedirs(f"{a.out}/clips", exist_ok=True)
    tts = fake_tts if a.dry_run else load_tts()
    noise_files = [os.path.join(a.noise_dir, f) for f in os.listdir(a.noise_dir) if f.endswith(".wav")] if a.noise_dir else []

    babble_pool = [tts(t, i) for i, t in enumerate(["Ɛte sɛn", "Me pɛ nsuo", "Bra ha", "Ɛyɛ den paa", "Wo ho te sɛn"])]
    snr_bins = [(15, 0.25), (10, 0.25), (5, 0.2), (0, 0.15), (-3, 0.15)]
    manifest, scenarios = [], []
    for i in range(a.n):
        kind, text, gloss, lab, exp, ex = make_example(rng_py)
        snr = rng_py.choices([s for s, _ in snr_bins], weights=[w for _, w in snr_bins])[0]
        wav = tts(text, a.seed * 1000 + i)
        f = 0.85 + 0.3 * rng_py.random() if kind == "bystander" else 0.95 + 0.1 * rng_py.random()
        if kind == "bystander": f = rng_py.choice([0.8, 0.85, 1.18, 1.25])  # crude "other voice"
        wav = speed(wav, f)
        pad = np.zeros(int(SR * rng_py.uniform(0.2, 0.6)), np.float32)
        wav = np.concatenate([pad, wav, pad])
        noise_kind = rng_py.choice(["generator", "babble", "hiss"])
        y = mix(wav, gen_noise(noise_kind, len(wav), rng, babble_pool, noise_files), snr)
        cid = f"syn_{i:04d}_{kind}"
        wavfile.write(f"{a.out}/clips/{cid}.wav", SR, (y * 32767).astype(np.int16))

        label = dict(lab)
        if snr < 0 and kind != "bystander":  # unusable audio: correct behaviour is to ask
            label = dict(lab, action="ASK_CLARIFICATION", confidence=0.4)
            exp = {"requires_clarification": True, "transactions": [], "credit_book": ex.get("initial_credit_book", [])}
        is_vendor = kind != "bystander"
        manifest.append({"id": cid, "wav": f"clips/{cid}.wav", "kind": kind, "text": text, "gloss": gloss,
                         "snr_db": snr, "noise": noise_kind, "speed": round(f, 3), "is_enrolled_vendor": is_vendor,
                         "label": label})
        scenarios.append({"id": cid, "utterance": text, "gloss": gloss, "is_enrolled_vendor": is_vendor,
                          "snr_db": snr, **ex, "expected_state": exp, "mock_prediction": label})
    with open(f"{a.out}/manifest.jsonl", "w", encoding="utf-8") as fh:
        for m in manifest: fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    json.dump(scenarios, open(f"{a.out}/scenarios.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter
    print(f"wrote {a.n} clips to {a.out}")
    print("kinds:", dict(Counter(m["kind"] for m in manifest)))
    print("snr:  ", dict(Counter(m["snr_db"] for m in manifest)))


if __name__ == "__main__":
    main()
