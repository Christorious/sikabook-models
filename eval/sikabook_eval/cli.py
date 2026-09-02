"""CLI entry: python -m sikabook_eval gold.jsonl pred.jsonl"""

from __future__ import annotations

import argparse
import json
import sys

from .events import load_jsonl
from .metrics import evaluate, format_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sikabook_eval",
        description="Evaluate SikaBook sale-event predictions against gold.",
    )
    parser.add_argument("gold", help="gold JSONL (true utterances/fields)")
    parser.add_argument("pred", help="prediction JSONL (pipeline output)")
    parser.add_argument(
        "--tolerance", type=float, default=0.05,
        help="amount tolerance in GHS (default 0.05)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    args = parser.parse_args(argv)

    gold = load_jsonl(args.gold)
    pred = load_jsonl(args.pred)
    report = evaluate(gold, pred, amount_tolerance=args.tolerance)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
