"""Big-ears baseline: Meta MMS (facebook/mms-1b-all, Akan adapter) transcribes Twi -> text rules -> ledger action.

  pip3 install --break-system-packages transformers
  python3 asr_mms.py --data data/fisd_v0 data/kasa_v0          # same 224 held-out clips as train_tiny.py
  python3 run_eval.py --scenarios data/fisd_v0/scenarios_heldout.json --preds data/fisd_v0/preds_mms.json | tail -1

Honest notes:
  - MMS-1B is ~1B params (CC BY-NC 4.0: fine for a nonprofit, not for selling). Too big for a cheap phone;
    this test asks "does real Twi ASR beat the tiny model?" before we spend effort shrinking it.
  - No training on our data at all: zero-shot. Fine-tuning the adapter on Twi is the GPU step.
  - Transcripts are cached in data/fisd_v0/mms_transcripts.json, so re-runs only redo the rules.
"""
import argparse, json, os, random, re, time
import numpy as np, soundfile as sf

TWI_NUM = {"baako": 1, "biako": 1, "mmienu": 2, "abien": 2, "mmiɛnsa": 3, "mmiensa": 3, "abiɛsa": 3,
           "ɛnan": 4, "anan": 4, "nnan": 4, "enum": 5, "anum": 5, "nnum": 5, "nsia": 6, "nson": 7,
           "nwɔtwe": 8, "nkron": 9, "du": 10, "dú": 10, "aduonu": 20, "aduonum": 50, "ɔha": 100}
DIGITS = {"5": 5, "10": 10, "five": 5, "faif": 5, "ten": 10, "tɛn": 10}   # MMS often writes English numbers
YES = {"aane", "aanne", "ane", "ai", "yoo", "yooo", "yo"}                   # MMS spellings of aane / yoo


def split(data, seed=7, heldout=0.2):
    """Exactly train_tiny.py's speaker split, so the clips are the same 224."""
    man, scen, test_spk = [], {}, set()
    for d in data:
        ms = [dict(json.loads(l), wav=f"{d}/" + json.loads(l)["wav"]) for l in open(f"{d}/manifest.jsonl", encoding="utf-8")]
        spk = sorted({m["speaker"] for m in ms}); random.Random(seed).shuffle(spk)
        test_spk |= set(spk[:max(1, int(len(spk) * heldout))])
        man += ms; scen.update({s["id"]: s for s in json.load(open(f"{d}/scenarios.json", encoding="utf-8"))})
    return [m for m in man if m["speaker"] in test_spk], scen


def transcribe(clips, cache):
    done = json.load(open(cache, encoding="utf-8")) if os.path.exists(cache) else {}
    todo = [m for m in clips if m["id"] not in done]
    if todo:
        import torch
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="aka")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="aka",
                                               ignore_mismatched_sizes=True).eval()
        t0 = time.time()
        for k, m in enumerate(todo):
            x, sr = sf.read(m["wav"], dtype="float32")
            if x.ndim > 1: x = x.mean(1)
            inp = proc(x, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad(): lg = model(**inp).logits[0]
            p = lg.softmax(-1); ids = p.argmax(-1)
            speech = ids != proc.tokenizer.pad_token_id                     # frames that emit a letter
            conf = float(p.max(-1).values[speech].mean()) if speech.any() else 0.0
            done[m["id"]] = {"text": proc.decode(ids), "conf": round(conf, 3)}
            if k % 20 == 0:
                print(f"  {k}/{len(todo)}  {time.time() - t0:.0f}s  '{done[m['id']]['text'][:50]}'", flush=True)
                json.dump(done, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
        json.dump(done, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    return done


def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def act(text, screen, conf, min_conf):
    """Text rules. Deliberately simple: the point is to see what good ears alone buy us."""
    t = re.sub(r"[^\w\sɛɔ']", " ", text.lower())
    w = t.split()
    base = {"speaker_similarity": 0.85, "item": "NONE", "quantity": 0, "amount_ghs": 0.0,
            "customer_name": None, "confidence": conf, "screen": screen}
    if screen == "CONFIRM_MODAL":
        if w[:1] == ["daabi"] or "daabi" in w[:2]: return dict(base, action="CANCEL_ACTION")
        j = " ".join(w[:4])
        if any(x in YES for x in w[:3]) or "aa ne" in j: return dict(base, action="CONFIRM_ACTION")
        return dict(base, action="ASK_CLARIFICATION")
    money = any(k in t for k in ("credit", "kredit", "kɛredit", "airtime", "sidi", "cedi"))
    amt = next((DIGITS.get(x) or TWI_NUM.get(x) for x in w if x in TWI_NUM or x in DIGITS), None)
    if money and amt in (5, 10):
        if conf < min_conf: return dict(base, action="ASK_CLARIFICATION")
        return dict(base, action="LOG_EXPENSE", item="airtime", quantity=1, amount_ghs=float(amt))
    if money and amt is None and conf < min_conf:
        return dict(base, action="ASK_CLARIFICATION")
    return dict(base, action="NO_OP")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/fisd_v0"])
    ap.add_argument("--min-conf", type=float, default=0.6)
    a = ap.parse_args()
    clips, scen = split(a.data)
    dst = a.data[0]
    print(f"MMS on {len(clips)} held-out clips (first run downloads ~4 GB) ...")
    tr = transcribe(clips, f"{dst}/mms_transcripts.json")
    E = N = 0
    for m in clips:
        if m.get("snr_db") is None and m["id"].startswith("fisd"):
            e, n = wer(m["text"], tr[m["id"]]["text"]); E += e; N += n
    print(f"FISD clean clips: word error rate {E / max(N, 1):.0%}")
    for m in clips[:6]:
        print(f"  ref: {m['text'][:60]}\n  mms: {tr[m['id']]['text'][:60]}")
    preds = {m["id"]: act(tr[m["id"]]["text"], scen[m["id"]]["mock_prediction"].get("screen", "HOME_LEDGER"),
                          tr[m["id"]]["conf"], a.min_conf) for m in clips}
    json.dump(preds, open(f"{dst}/preds_mms.json", "w"), indent=1)
    print(f"wrote {dst}/preds_mms.json")


if __name__ == "__main__":
    main()
