"""SikaBook evaluation harness.

Measures what matters for the product:
  - ASR quality      : WER / CER per language
  - Sale detection   : precision / recall / F1 on is_sale
  - Slot extraction  : per-field accuracy and F1 (product, quantity, unit)
  - Amount accuracy  : exact and within-tolerance GHS error (the number
                       a trader actually cares about)

Inputs are JSONL files of per-utterance records:

  {"utt_id": "u1", "text": "i sold two tilapia at ten cedis each",
   "is_sale": true, "amount": 20.0, "product": "Tilapia",
   "quantity": 2, "unit": "piece"}

Gold records must carry the true fields; prediction records carry the
pipeline's output. Extra fields are ignored.

Usage:
  python -m sikabook_eval gold.jsonl predictions.jsonl [--tolerance 0.05]

No third-party dependencies: runs anywhere (laptop, Kaggle, device logs).
"""

__version__ = "0.1.0"

from .metrics import evaluate, EvaluationReport  # noqa: F401
from .events import load_jsonl, SaleRecord  # noqa: F401
