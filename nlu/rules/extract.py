#!/usr/bin/env python3
"""CLI: extract SaleEvents from text lines or stdin.

Usage:
  python3 nlu/rules/extract.py "i sold two tilapia at ten cedis each"
  cat utterances.txt | python3 nlu/rules/extract.py --jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sika_rules import extract
from sika_rules.vocab import ProductVocabulary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="utterance text (or omit to read stdin)")
    parser.add_argument("--vocab", default=None, help="path to product_vocabulary.json")
    parser.add_argument("--jsonl", action="store_true", help="one JSON object per input line")
    args = parser.parse_args()

    vocab = ProductVocabulary.load(args.vocab) if args.vocab else ProductVocabulary.load()
    lines = args.text if args.text else [l.strip() for l in sys.stdin if l.strip()]
    for line in lines:
        event = extract(line, vocab)
        if args.jsonl:
            print(json.dumps(event, ensure_ascii=False))
        else:
            keep = {k: event[k] for k in
                    ("is_sale", "amount", "product", "quantity", "unit",
                     "unit_price", "language", "confidence", "needs_review",
                     "validation_issues")}
            print(f"{line!r}\n  -> {json.dumps(keep, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
