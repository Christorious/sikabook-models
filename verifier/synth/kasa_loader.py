"""Hard negatives from KasaSpeech (English-Twi code-switched, Responsible AI Lab KNUST, Apache-2.0).
HF mirror: ghananlpcommunity/Ghana_English-Twi_Code-switching_Speech (parquet, 48 kHz -> 16 kHz).

  python3 synth/kasa_loader.py --out data/kasa_v0 --noise
  python3 train_tiny.py --data data/fisd_v0 data/kasa_v0

Every clip becomes NO_OP on the home screen: people chatting about cedis, the market, fish, MoMo,
borrowing money... none of it is an order, so the ledger must stay untouched.
  kind money_talk : mentions cedis/sika/price/buy/sell/market/fish/momo/credit or a number
  kind chat       : ordinary code-switched talk (sampled, to keep the model honest)
Only clips <= 8 s are kept, so clip length matches FISD and can't be used as a shortcut.
"""
import argparse, io, json, os, re, random, sys, urllib.request
from collections import Counter, defaultdict
import numpy as np, soundfile as sf, pyarrow.parquet as pq
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(__file__))
from fisd_loader import noisy, SR

BASE = "https://huggingface.co/datasets/ghananlpcommunity/Ghana_English-Twi_Code-switching_Speech/resolve/main/data/"
MONEY = re.compile(r"cedi|ghs|gh₵|\bsika|pesewa|price|cost|expensive|cheap|\bbuy|bought|\bsell|sold|"
                   r"market|dwam|customer|\btɔ\b|tɔn|fish|tilapia|kpanla|momo|mobile money|credit|"
                   r"debt|\bowe|lend|borrow|\bpay|\b\d+\b")


def fetch(name, cache):
    os.makedirs(cache, exist_ok=True); p = os.path.join(cache, name)
    if not os.path.exists(p):
        print("downloading", name); urllib.request.urlretrieve(BASE + name, p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/kasa_v0")
    ap.add_argument("--split", default="test", choices=["test", "validation", "train"])
    ap.add_argument("--shards", type=int, default=1, help="train only: how many of 7 shards (350-800 MB each)")
    ap.add_argument("--chat", type=int, default=150, help="how many plain-chat clips to add")
    ap.add_argument("--max-sec", type=float, default=8.0)
    ap.add_argument("--cache", default="data/_kasa_cache")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); r = random.Random(a.seed)
    files = {"test": ["test-00000-of-00001.parquet"], "validation": ["validation-00000-of-00001.parquet"],
             "train": [f"train-{i:05d}-of-00007.parquet" for i in range(a.shards)]}[a.split]
    os.makedirs(f"{a.out}/clips", exist_ok=True)
    man, scen, per_spk, seen = [], [], defaultdict(list), set()
    empty = {"transactions": [], "credit_book": []}
    for fn in files:
        tab = pq.read_table(fetch(fn, a.cache), columns=["speaker_id", "transcript", "duration", "audio"])
        texts, spks, durs = (tab.column(c).to_pylist() for c in ("transcript", "speaker_id", "duration"))
        auds = tab.column("audio")
        money = [i for i, t in enumerate(texts) if t and durs[i] and durs[i] <= a.max_sec and MONEY.search(t.lower())]
        rest = [i for i, t in enumerate(texts) if t and durs[i] and durs[i] <= a.max_sec and i not in set(money)]
        picks = [(i, "money_talk") for i in money] + [(i, "chat") for i in r.sample(rest, min(a.chat, len(rest)))]
        for i, kind in picks:
            if texts[i] in seen: continue      # same prompt read twice -> keep one
            seen.add(texts[i])
            x, sr = sf.read(io.BytesIO(auds[i].as_py()["bytes"]), dtype="float32")
            if x.ndim > 1: x = x.mean(1)
            if sr == 48000: x = resample_poly(x, 1, 3).astype(np.float32)
            elif sr != SR: continue
            spk = f"kasa-{spks[i]}"
            versions = [(None, x)] + ([(s, noisy(x, s, rng)) for s in [r.choice([10, 5, 0, -3])]] if a.noise else [])
            for snr, y in versions:
                cid = f"kasa_{len(man):05d}_{kind}" + (f"_snr{snr}" if snr is not None else "")
                sf.write(f"{a.out}/clips/{cid}.wav", y, SR)
                label = {"speaker_similarity": 0.85, "action": "NO_OP", "item": "NONE", "quantity": 0,
                         "amount_ghs": 0.0, "customer_name": None, "confidence": 0.95, "screen": "HOME_LEDGER"}
                e = empty
                if snr is not None and snr < 0:
                    label = dict(label, action="ASK_CLARIFICATION", confidence=0.4)
                    e = {"requires_clarification": True, **empty}
                man.append({"id": cid, "wav": f"clips/{cid}.wav", "text": texts[i], "speaker": spk, "kind": kind,
                            "snr_db": snr, "label": label, "source": "KasaSpeech, Apache-2.0"})
                scen.append({"id": cid, "utterance": texts[i], "speaker": spk, "is_enrolled_vendor": True,
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
