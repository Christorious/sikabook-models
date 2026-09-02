#!/usr/bin/env python3
"""finetune_nlu.py — MiniLM intent+NER for sale-event extraction.

Stage 2 of the NLU pipeline. Two heads on one encoder:
  A. sequence classification  -> is_sale probability
  B. token classification     -> BIO slots: B/QTY, I/QTY, B/UNIT,
                                 B/PRODUCT, I/PRODUCT, B/AMOUNT, I/AMOUNT

Why not an LLM here: this is a fixed schema, and a quantized MiniLM is
~25 MB int8 ONNX, single-digit milliseconds on a Cortex-A55 — it runs on
every utterance on every device, where even Qwen3-0.6B is reserved for
6GB+ devices and only runs on detected candidates.

Training data: data/nlu/synthetic/*.jsonl (bootstrap) + real labeled sets
(data/nlu/eval_sets/, see labeling_guide.md). Runs on a free Kaggle T4.

Export: ONNX int8 -> nlu/bert/export/ (consumed by export/package_android.sh,
served with onnxruntime-android).

Usage (Kaggle):
  python nlu/bert/finetune_nlu.py --train data/nlu/synthetic/train.jsonl \
      --dev data/nlu/synthetic/dev.jsonl --epochs 4
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval"))

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 22M params
LABELS = ["O", "B-QTY", "I-QTY", "B-UNIT", "I-UNIT",
          "B-PRODUCT", "I-PRODUCT", "B-AMOUNT", "I-AMOUNT"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# unit words that count as UNIT tokens (keep in sync with sika_rules UNITS)
UNIT_WORDS = {"piece", "pieces", "pcs", "bowl", "bowls", "kokoo", "bucket",
              "buckets", "rubber", "tin", "tins", "can", "cans", "size",
              "sizes", "pack", "packs", "bag", "bags", "sachet", "sachets",
              "dozen", "crate", "gallon", "bunch", "tuber", "plate", "cup",
              "packet", "bottle", "carton", "roll", "olonka"}
CURRENCY_WORDS = {"cedi", "cedis", "cedes", "pesewa", "pesewas", "ghana",
                  "ghc", "ghs"}


def align_labels(text: str, gold: dict, tokenizer) -> dict:
    """Build token-classification labels from the gold fields.

    Strategy: anchor each slot on its surface evidence in the text.
    For synthetic data the generator guarantees a consistent surface:
      - quantity: the number word(s) adjacent to product/unit
      - unit: the unit word
      - product: a vocabulary variant string
      - amount: the number span before cedis/pesewas
    """
    words = text.split()
    labels = ["O"] * len(words)

    def find(sub_tokens: list[str]) -> int:
        for i in range(len(words) - len(sub_tokens) + 1):
            if [w.lower().strip(".,") for w in words[i:i + len(sub_tokens)]] == \
                    [t.lower() for t in sub_tokens]:
                return i
        return -1

    if gold.get("product"):
        surface = gold.get("product_surface") or gold["product"]
        idx = find(surface.split())
        if idx >= 0:
            for k in range(len(surface.split())):
                labels[idx + k] = ("B-PRODUCT" if k == 0 else "I-PRODUCT")
    if gold.get("quantity") is not None:
        # number word(s) directly before unit or product surface
        target = None
        if gold.get("unit"):
            pass  # unit below; quantity is the word(s) before it
        if gold.get("unit"):
            # find unit position, walk left over number words
            unit_words = gold["unit"].split()
            uidx = find(unit_words)
            if uidx >= 0:
                labels[uidx:uidx + len(unit_words)] = (
                    ["B-UNIT"] + ["I-UNIT"] * (len(unit_words) - 1))
                j = uidx - 1
                while j >= 0 and (words[j].lower().isdigit() or
                                  words[j].lower() in CURRENCY_WORDS or
                                  _is_number_word(words[j])):
                    if words[j].lower() in CURRENCY_WORDS:
                        break
                    labels[j] = "B-QTY" if labels[j + 1] != "B-QTY" else "I-QTY"
                    j -= 1
        elif gold.get("product"):
            surface = (gold.get("product_surface") or gold["product"]).split()
            pidx = find(surface)
            if pidx > 0 and (_is_number_word(words[pidx - 1]) or
                             words[pidx - 1].isdigit()):
                labels[pidx - 1] = "B-QTY"
    if gold.get("amount") is not None or gold.get("unit_price") is not None:
        # amount = number span immediately before a currency word
        for i, w in enumerate(words):
            if w.lower().strip(".,") in ("cedis", "cedi", "cedes", "pesewas",
                                         "pesewa"):
                j = i - 1
                k = 0
                while j >= 0 and (_is_number_word(words[j]) or
                                  words[j].lower() in ("and",) or
                                  words[j].replace(".", "").isdigit()):
                    labels[j] = "B-AMOUNT" if k == 0 else "I-AMOUNT"
                    k += 1
                    j -= 1
                if k:
                    break
    return {"labels": labels}


_NUM_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight",
              "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
              "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
              "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
              "eighty", "ninety", "hundred", "baako", "mmienu", "mmiensa",
              "du", "dun", "aduonu", "aduasa", "aduanan", "aduonum",
              "aduosia", "aduonson", "aduowotwe", "aduonkron", "ɔha", "apem"}


def _is_number_word(w: str) -> bool:
    return w.lower() in _NUM_WORDS


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=str(REPO_ROOT / "data/nlu/synthetic/train.jsonl"))
    ap.add_argument("--dev", default=str(REPO_ROOT / "data/nlu/synthetic/dev.jsonl"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default=str(REPO_ROOT / "nlu/bert/export"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification, AutoModelForTokenClassification,
            AutoTokenizer, Trainer, TrainingArguments,
        )
    except ImportError:
        print("Requires torch+transformers — run on Kaggle/Colab:\n"
              "  pip install transformers datasets", file=sys.stderr)
        return 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)
    print(f"train={len(train_rows)} dev={len(dev_rows)}")

    # --- dataset objects (joint: both heads trained per row) ---
    class Rows(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            enc = tokenizer(r["text"], truncation=True, max_length=64,
                            padding="max_length")
            slot = align_labels(r["text"], r, tokenizer)
            label_ids = [LABEL2ID[l] for l in slot["labels"]]
            label_ids = label_ids[:len(enc["input_ids"])]
            label_ids += [-100] * (len(enc["input_ids"]) - len(label_ids))
            enc["seq_labels"] = 1 if r["is_sale"] else 0
            enc["token_labels"] = label_ids
            return enc

    def collate(batch):
        import collections
        out = collections.defaultdict(list)
        for b in batch:
            for k, v in b.items():
                out[k].append(v)
        return {k: torch.tensor(v) for k, v in out.items()}

    token_model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label={i: l for l, i in LABEL2ID.items()},
        label2id=LABEL2ID, ignore_mismatched_sizes=True)
    seq_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True)

    targs = TrainingArguments(
        output_dir=str(Path(args.out) / "ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
    )

    token_trainer = Trainer(model=token_model, args=targs,
                            train_dataset=Rows(train_rows),
                            eval_dataset=Rows(dev_rows), data_collator=collate)
    token_trainer.train()
    seq_trainer = Trainer(model=seq_model, args=targs,
                          train_dataset=Rows(train_rows),
                          eval_dataset=Rows(dev_rows), data_collator=collate)
    seq_trainer.train()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    token_model.save_pretrained(out / "token_clf")
    seq_model.save_pretrained(out / "seq_clf")
    tokenizer.save_pretrained(out / "tokenizer")

    # ONNX export + int8 dynamic quantization
    try:
        from optimum.exporters.onnx import main_export
        main_export(out / "token_clf", out / "onnx", task="token-classification")
        main_export(out / "seq_clf", out / "onnx", task="text-classification")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        for name in ("model.onnx",):
            src = out / "onnx" / name
            if src.exists():
                quantize_dynamic(str(src), str(src.with_name(
                    name.replace(".onnx", ".int8.onnx"))),
                    weight_type=QuantType.QInt8)
    except ImportError:
        print("optimum/onnxruntime not installed — ONNX export skipped "
              "(run: pip install optimum onnxruntime)")

    (out / "labels.json").write_text(json.dumps(LABELS))
    print(f"Exported to {out}")
    print("Next: eval with sikabook_eval against dev/test gold, then package "
          "via export/package_android.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
