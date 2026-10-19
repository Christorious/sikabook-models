"""Run the benchmark. CPU only, stdlib only.

  python run_eval.py                                  # grade the built-in mock predictions
  python run_eval.py --preds preds.json               # grade your model: {"S01_twi_cash_sale": {...}, ...}
  python run_eval.py --threshold 0.6                  # binary gate: FAR against FRR
  python run_eval.py --review-threshold 0.5           # three tiers: accept / review / reject

With --review-threshold the gate stops being a filter and becomes a ranker: utterances between
the two thresholds are held for the day-end review instead of being thrown away, and the last
line reports what that tier costs in queue size. Without it the gate is binary and every number
matches what this harness measured before the tier existed.
"""
import argparse, json
from sikabook.schema import ModelActionOutput
from sikabook.ledger_env import GateConfig
from sikabook.verifier import evaluate_episode

ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", default="scenarios/market_eval_set.json")
ap.add_argument("--preds")
ap.add_argument("--threshold", type=float, default=0.68, help="accept: write without asking")
ap.add_argument("--review-threshold", type=float, default=None,
                help="hold for the day-end review down to here; below it, reject for good")
a = ap.parse_args()

scen = json.load(open(a.scenarios, encoding="utf-8"))
preds = json.load(open(a.preds, encoding="utf-8")) if a.preds else {s["id"]: s["mock_prediction"] for s in scen}
cfg = GateConfig(speaker_threshold=a.threshold, review_threshold=a.review_threshold)

rows, total = [], 0.0
for s in scen:
    r, why = evaluate_episode(s, ModelActionOutput.from_dict(preds[s["id"]]), cfg)
    total += r; rows.append((s, r, why))
    print(f"{s['id']:<34} {r:+.1f}  {why}")

n = len(rows)
imp = [(s, r, w) for s, r, w in rows if not s["is_enrolled_vendor"]]
gen = [(s, r, w) for s, r, w in rows if s["is_enrolled_vendor"]]
harm = sum(1 for _, r, _ in rows if r < 0)
far = sum(1 for _, r, _ in imp if r < 0) / len(imp) if imp else 0.0
# A false reject is one of her utterances the gate dropped: it reaches neither the ledger nor the
# review queue, so the sale is gone. Held utterances are deliberately not counted here.
frr = sum(1 for _, _, w in gen if "false reject" in w) / len(gen) if gen else 0.0
print(f"\nscore {total:.1f}/{n}   success {sum(1 for _, r, _ in rows if r > 0)/n:.0%}   "
      f"harmful writes {harm/n:.0%}   FAR {far:.0%}   FRR {frr:.0%}   (threshold {a.threshold})")

if a.review_threshold is not None:
    held = sum(1 for _, _, w in gen if w.startswith("HELD"))
    noise = sum(1 for _, _, w in imp if w.startswith("NOISE"))
    print(f"review queue {held + noise}/{n}   {held} of hers held (recoverable)   "
          f"{noise} bystander (pure burden)   (review threshold {a.review_threshold})")
