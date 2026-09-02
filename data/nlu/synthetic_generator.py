#!/usr/bin/env python3
"""synthetic_generator.py — labeled NLU training data without a field trip.

Generates market utterances with gold SaleEvent annotations across the
language mix the pipeline will face:

  - Ghanaian English  (i sold two tilapia at ten cedis each)
  - Ghanaian Pidgin   (i go take three tins of titus, chale)
  - Asante/Akuapem Twi (mepɛ apateshi mmienu — aduasa cedis)

Templates x products x quantities x prices x fillers, with gold fields
per the SaleEvent schema. Output: train/dev/test JSONL compatible with
`eval/sikabook_eval` and `nlu/bert/finetune_nlu.py`.

The generated set is used to TRAIN the BERT stage and to sanity-check the
rules stage. It is NOT a substitute for real market recordings — expect a
reality gap; the labeling guide (data/nlu/labeling_guide.md) defines how
real audio gets annotated later.

Usage:
  python3 data/nlu/synthetic_generator.py --n 20000 --seed 42 \
      --out data/nlu/synthetic
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "nlu" / "rules"))

from sika_rules.vocab import ProductVocabulary  # noqa: E402

EN_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
          7: "seven", 8: "eight", 9: "nine", 10: "ten", 12: "twelve",
          15: "fifteen", 20: "twenty"}
TWI_NUM = {1: "baako", 2: "mmienu", 3: "mmiɛnsa", 4: "ɛnan", 5: "ɛnum",
           6: "nsia", 7: "nson", 8: "nwɔtwe", 9: "nkron", 10: "du"}
TWI_TENS = {20: "aduonu", 30: "aduasa", 40: "aduanan", 50: "aduonum",
            60: "aduosia", 70: "aduonson", 80: "aduowotwe", 90: "aduonkron"}
UNITS_EN = {"piece": ["", "pieces"], "bowl": ["bowl", "bowls", "kokoo"],
            "bucket": ["bucket", "rubber"], "tin": ["tin", "tins"],
            "pack": ["pack", "packs"], "sachet": ["sachet", "sachets"],
            "bag": ["bag", "bags"], "crate": ["crate"], "dozen": ["dozen"]}

FILLERS_EN = ["please", "my friend", "madam", "boss", "eh", "o"]
FILLERS_PIDGIN = ["chale", "abeg", "oo", "small small", "no vex", "saa"]
NEGOTIATION_EN = ["how much is this", "what is the price of {surface}",
                  "that is too much", "can you reduce the price",
                  "is this the last price"]
NEGOTIATION_PIDGIN = ["how much be dis", "the price be too much abeg",
                      "reduce am for me", "you dey reduce am small"]


def money_en(value: float) -> str:
    if value == int(value):
        cedis = EN_NUM.get(int(value), str(int(value)))
        return f"{cedis} cedis"
    whole = int(value)
    pes = round((value - whole) * 100)
    return f"{EN_NUM.get(whole, str(whole))} cedis {EN_NUM.get(pes, str(pes))} pesewas"


def money_twi(value: float) -> str:
    tens = TWI_TENS.get(int(value))
    if tens:
        return f"{tens} cedis"
    return f"{TWI_NUM.get(int(value), str(int(value)))} cedis"


def pick_product(rng, vocab):
    p = rng.choice(vocab.products)
    surface = rng.choice(p.variants) if p.variants else p.canonical
    return p, surface


def price_for(rng, product):
    lo = product.min_price or 1.0
    hi = max(product.max_price or lo * 3, lo * 2)
    options = sorted({round(lo), round((lo + hi) / 2), round(hi),
                      round(lo) + 1, round((lo + hi) / 4) or 1})
    value = rng.choice([v for v in options if v > 0])
    if rng.random() < 0.15:
        return round(value - 0.5, 2)  # pesewas prices (0.50-style)
    return float(value)


def gen_sale_en(rng, vocab):
    p, surface = pick_product(rng, vocab)
    qty = rng.choice([1, 1, 2, 2, 3, 4, 5, 10, 12])
    unit = rng.choice(UNITS_EN.get((p.units or ["piece"])[0], [""]))
    unit_word = rng.choice(unit or [""]) if isinstance(unit, list) else unit
    price = price_for(rng, p)
    each = rng.random() < 0.35
    filler = rng.choice(FILLERS_EN + [""])
    if each and qty > 1:
        total = round(price * qty, 2)
        text = (f"i sold {EN_NUM[qty]} "
                f"{unit_word + ' of ' if unit_word else ''}{surface} "
                f"at {money_en(price)} each")
    else:
        total = price
        text = (f"i sold {EN_NUM[qty]} "
                f"{unit_word + ' of ' if unit_word else ''}{surface} "
                f"for {money_en(total)}")
    if filler:
        text = f"{text} {filler}".replace("  ", " ")
    gold = dict(is_sale=True, amount=total, product=p.canonical,
                quantity=qty, unit=(p.units or ["piece"])[0] if unit_word else None,
                unit_price=price if each and qty > 1 else None,
                language="en-GH")
    return text, gold


def gen_sale_pidgin(rng, vocab):
    p, surface = pick_product(rng, vocab)
    qty = rng.choice([1, 2, 3, 4, 5, 10])
    price = price_for(rng, p)
    total = price * qty
    openers = ["i go take", "give me", "make i take", "buyer take", "i go buy"]
    unit_word = rng.choice(UNITS_EN.get((p.units or ["piece"])[0], [""]) or [""])
    text = f"{rng.choice(openers)} {EN_NUM[qty]} "
    if unit_word:
        text += f"{unit_word} of {surface} na {money_en(total)} {rng.choice(FILLERS_PIDGIN)}"
    else:
        text += f"{surface} na {money_en(total)} {rng.choice(FILLERS_PIDGIN)}"
    gold = dict(is_sale=True, amount=round(total, 2), product=p.canonical,
                quantity=qty, unit=(p.units or [None])[0] if unit_word else None,
                unit_price=None, language="pidgin")
    return " ".join(text.split()), gold


def gen_sale_twi(rng, vocab):
    p, surface = pick_product(rng, vocab)
    qty = rng.choice([1, 2, 3, 4, 5])
    price = rng.choice([20, 30, 40, 50, 60])
    text = f"mepɛ {surface} {TWI_NUM[qty]} — {money_twi(price)}"
    text = text.replace("—", "")
    gold = dict(is_sale=True, amount=float(price), product=p.canonical,
                quantity=qty, unit=None, unit_price=None, language="tw")
    return text, gold


def gen_not_sale(rng, vocab):
    p, surface = pick_product(rng, vocab)
    lang = rng.random()
    if lang < 0.4:
        text = rng.choice(NEGOTIATION_EN).format(surface=surface)
        return text, dict(is_sale=False, amount=None, product=None,
                          quantity=None, unit=None, unit_price=None,
                          language="en-GH")
    if lang < 0.75:
        text = rng.choice(NEGOTIATION_PIDGIN)
        return text, dict(is_sale=False, amount=None, product=None,
                          quantity=None, unit=None, unit_price=None,
                          language="pidgin")
    # unambiguous not-sale: a price question, not a price statement
    # (a bare "tilapia aduasa cedis" is ambiguous sale/quote — that
    # ambiguity belongs to human labeling, not synthetic data)
    text = f"{surface} eye sɛn"
    return text, dict(is_sale=False, amount=None,
                      product=None, quantity=None, unit=None, unit_price=None,
                      language="tw")


GENERATORS = [
    (gen_sale_en, 0.4),
    (gen_sale_pidgin, 0.25),
    (gen_sale_twi, 0.15),
    (gen_not_sale, 0.2),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "nlu" / "synthetic"))
    ap.add_argument("--vocab", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    vocab = (ProductVocabulary.load(args.vocab) if args.vocab
             else ProductVocabulary.load())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(args.n):
        r = rng.random()
        acc = 0.0
        for gen, weight in GENERATORS:
            acc += weight
            if r <= acc:
                text, gold = gen(rng, vocab)
                break
        rows.append({
            "utt_id": f"syn{i:06d}",
            "text": text,
            **gold,
        })

    rng.shuffle(rows)
    n_dev = max(200, int(len(rows) * 0.02))
    n_test = max(200, int(len(rows) * 0.02))
    splits = {
        "train": rows[n_dev + n_test:],
        "dev": rows[:n_dev],
        "test": rows[n_dev:n_dev + n_test],
    }
    for name, data in splits.items():
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
