"""Real Asante Twi clips from the Financial Inclusion Speech Dataset (Ashesi/Nokwary, CC-BY-4.0).
HF mirror: ghanaopenai/twi_multispeaker_audio_transcribed (parquet). CPU only.

  sudo apt install -y python3-pip
  pip3 install --break-system-packages pyarrow soundfile numpy scipy
  python3 synth/fisd_loader.py --out data/fisd_v0                 # test split, ~330 MB, 239 speakers
  python3 synth/fisd_loader.py --out data/fisd_v0 --noise         # also make noisy copies
  python3 run_eval.py --scenarios data/fisd_v0/scenarios.json      # oracle sanity check

What you get:
  clips/*.wav (16 kHz), manifest.jsonl (text, speaker, label), scenarios.json (verifier format),
  speakers.json (clips per speaker -> enrollment / bystander splits for FAR/FRR).

Mapping (FISD is banking/MoMo, not fish-market sales):
  "credit 5/10 cedis" -> LOG_EXPENSE airtime (a real trader expense)
  Aane / Yoo / Mepaakyɛw aane -> CONFIRM_ACTION (confirm screen)
  Daabi -> CANCEL_ACTION (confirm screen)
  balance / phone-number lines -> NO_OP (must NOT touch the ledger)
Everything else is skipped. Market-sale phrases still need real recordings.
"""
import argparse, io, json, os, re, urllib.request, random
from collections import Counter, defaultdict
import numpy as np, soundfile as sf, pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/ghanaopenai/twi_multispeaker_audio_transcribed/resolve/main/data/"
SR = 16000


def label_for(text):
    t = text.lower().replace("’", "'")
    m = re.search(r"credit(?: transfer)?\s+(\d+)\s*cedi", t)
    if m:
        amt = float(m.group(1))
        return "expense", {"action": "LOG_EXPENSE", "item": "airtime", "quantity": 1, "amount_ghs": amt,
                           "screen": "HOME_LEDGER"}, {"transactions": [{"tx_type": "EXPENSE", "item": "airtime",
                                                                          "quantity": 1, "amount_ghs": amt}], "credit_book": []}
    empty = {"transactions": [], "credit_book": []}
    if re.match(r"^\s*daabi\b", t):
        return "cancel", {"action": "CANCEL_ACTION", "screen": "CONFIRM_MODAL"}, empty
    if re.match(r"^\s*(aane|yoo)\b", t) or re.search(r"pa\w*'?kyɛw\s+aane|pawokyɛw aane", t):
        return "confirm", {"action": "CONFIRM_ACTION", "screen": "CONFIRM_MODAL"}, empty
    if "sika no aka" in t or "number" in t or re.search(r"\b0\d{2}\b", t):
        return "noop_money_talk", {"action": "NO_OP", "screen": "HOME_LEDGER"}, empty
    return None, None, None


def fetch(name, cache):
    os.makedirs(cache, exist_ok=True); p = os.path.join(cache, name)
    if not os.path.exists(p):
        print("downloading", name); urllib.request.urlretrieve(BASE + name, p)
    return p


def noisy(x, snr, rng):
    n = len(x); t = np.arange(n) / SR; f0 = rng.uniform(45, 55)
    hum = sum((1 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 10)) + 0.3 * rng.standard_normal(n)
    y = x + hum * np.sqrt((np.mean(x ** 2) + 1e-9) / ((np.mean(hum ** 2) + 1e-9) * 10 ** (snr / 10)))
    return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fisd_v0")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--shards", type=int, default=1, help="train only: how many of 6 shards")
    ap.add_argument("--cache", default="data/_fisd_cache")
    ap.add_argument("--noise", action="store_true", help="add a generator-hum copy at 10/5/0/-3 dB")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); r = random.Random(a.seed)
    files = ["test-00000-of-00001.parquet"] if a.split == "test" else \
            [f"train-{i:05d}-of-00006.parquet" for i in range(a.shards)]
    os.makedirs(f"{a.out}/clips", exist_ok=True)
    man, scen, per_spk = [], [], defaultdict(list)
    for fn in files:
        tab = pq.read_table(fetch(fn, a.cache))
        texts, auds = tab.column("transcription").to_pylist(), tab.column("audio_path")
        for i, text in enumerate(texts):
            kind, lab, exp = label_for(text or "")
            if not kind: continue
            au = auds[i].as_py(); spk = "-".join(au["path"].split("-")[:2])
            x, sr = sf.read(io.BytesIO(au["bytes"]), dtype="float32")
            if x.ndim > 1: x = x.mean(1)
            if sr != SR: continue  # FISD mirror is 16 kHz; skip anything odd
            versions = [(None, x)] + ([(s, noisy(x, s, rng)) for s in [r.choice([10, 5, 0, -3])]] if a.noise else [])
            for snr, y in versions:
                cid = f"fisd_{len(man):05d}_{kind}" + (f"_snr{snr}" if snr is not None else "")
                sf.write(f"{a.out}/clips/{cid}.wav", y, SR)
                label = {"speaker_similarity": 0.85, "item": "NONE", "quantity": 0, "amount_ghs": 0.0,
                         "customer_name": None, "confidence": 0.95, **lab}
                e = exp
                if snr is not None and snr < 0:
                    label = dict(label, action="ASK_CLARIFICATION", confidence=0.4)
                    e = {"requires_clarification": True, "transactions": [], "credit_book": []}
                man.append({"id": cid, "wav": f"clips/{cid}.wav", "text": text, "speaker": spk, "kind": kind,
                            "snr_db": snr, "label": label, "source": "FISD Asante Twi, CC-BY-4.0"})
                scen.append({"id": cid, "utterance": text, "speaker": spk, "is_enrolled_vendor": True,
                             "snr_db": 30 if snr is None else snr, "expected_state": e, "mock_prediction": label})
                per_spk[spk].append(cid)
    with open(f"{a.out}/manifest.jsonl", "w", encoding="utf-8") as f:
        for m in man: f.write(json.dumps(m, ensure_ascii=False) + "\n")
    json.dump(scen, open(f"{a.out}/scenarios.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(per_spk, open(f"{a.out}/speakers.json", "w"), indent=1)
    print(f"{len(man)} clips, {len(per_spk)} speakers ->", a.out)
    print("kinds:", dict(Counter(m["kind"] for m in man)))


if __name__ == "__main__":
    main()
