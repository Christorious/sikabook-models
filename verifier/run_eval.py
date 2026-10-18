"""Run the benchmark. CPU only, stdlib only.

  python run_eval.py                      # grade the built-in mock predictions
  python run_eval.py --preds preds.json   # grade your model: {"S01_twi_cash_sale": {...ModelActionOutput...}, ...}
  python run_eval.py --threshold 0.6      # see how the voice-gate threshold trades FAR vs FRR
"""
import argparse, json
from sikabook.schema import ModelActionOutput
from sikabook.ledger_env import GateConfig
from sikabook.verifier import evaluate_episode

ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", default="scenarios/market_eval_set.json")
ap.add_argument("--preds")
ap.add_argument("--threshold", type=float, default=0.68)
a = ap.parse_args()

scen = json.load(open(a.scenarios, encoding="utf-8"))
preds = json.load(open(a.preds, encoding="utf-8")) if a.preds else {s["id"]: s["mock_prediction"] for s in scen}
cfg = GateConfig(speaker_threshold=a.threshold)

rows, total = [], 0.0
for s in scen:
    r, why = evaluate_episode(s, ModelActionOutput.from_dict(preds[s["id"]]), cfg)
    total += r; rows.append((s, r))
    print(f"{s['id']:<34} {r:+.1f}  {why}")

n = len(rows)
imp = [r for s, r in rows if not s["is_enrolled_vendor"]]
gen = [(s, r) for s, r in rows if s["is_enrolled_vendor"]]
harm = sum(1 for _, r in rows if r < 0)
far = sum(1 for r in imp if r < 0) / len(imp) if imp else 0.0
frr = sum(1 for s, r in gen if r == 0.0 and "false reject" in str(evaluate_episode(
          s, ModelActionOutput.from_dict(preds[s["id"]]), cfg)[1])) / len(gen) if gen else 0.0
print(f"\nscore {total:.1f}/{n}   success {sum(1 for _, r in rows if r > 0)/n:.0%}   "
      f"harmful writes {harm/n:.0%}   FAR {far:.0%}   FRR {frr:.0%}   (threshold {a.threshold})")
