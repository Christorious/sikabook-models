"""Metrics: WER/CER, sale detection P/R/F1, slot extraction, amount accuracy.

All functions are pure and dependency-free so they can run in CI, on
Kaggle, or pasted into a device-log analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .events import SaleRecord


# ---------------------------------------------------------------- edit distance

def _edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance over token lists (standard WER definition)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,            # deletion
                cur[j - 1] + 1,         # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> Optional[float]:
    """Word error rate for one pair. None when reference is empty."""
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return None
    return _edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> Optional[float]:
    """Character error rate for one pair. None when reference is empty."""
    ref = list(reference.replace(" ", ""))
    hyp = list(hypothesis.replace(" ", ""))
    if not ref:
        return None
    return _edit_distance(ref, hyp) / len(ref)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------- report

@dataclass
class FieldMetrics:
    """Slot-level metrics for one field (product, quantity, unit)."""
    correct: int = 0
    predicted: int = 0      # predictions where field is not None
    gold: int = 0           # gold rows where field is not None
    # For product we also count near-misses (case/space-insensitive match)
    near_correct: int = 0

    @property
    def precision(self) -> float:
        return self.correct / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.gold if self.gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "correct": self.correct,
            "predicted": self.predicted,
            "gold": self.gold,
            "near_correct": self.near_correct,
        }


@dataclass
class EvaluationReport:
    # ASR
    wer_mean: Optional[float] = None
    cer_mean: Optional[float] = None
    wer_by_language: dict = field(default_factory=dict)
    n_asr_pairs: int = 0
    # Sale detection
    sale_precision: float = 0.0
    sale_recall: float = 0.0
    sale_f1: float = 0.0
    n_sales_gold: int = 0
    # Amount
    amount_evaluated: int = 0
    amount_exact: int = 0
    amount_within_tolerance: int = 0
    amount_mae_ghs: float = 0.0
    # Slots
    product: FieldMetrics = field(default_factory=FieldMetrics)
    quantity: FieldMetrics = field(default_factory=FieldMetrics)
    unit: FieldMetrics = field(default_factory=FieldMetrics)

    def as_dict(self) -> dict:
        return {
            "asr": {
                "pairs": self.n_asr_pairs,
                "wer": round(self.wer_mean, 4) if self.wer_mean is not None else None,
                "cer": round(self.cer_mean, 4) if self.cer_mean is not None else None,
                "wer_by_language": self.wer_by_language,
            },
            "sale_detection": {
                "precision": round(self.sale_precision, 4),
                "recall": round(self.sale_recall, 4),
                "f1": round(self.sale_f1, 4),
                "gold_sales": self.n_sales_gold,
            },
            "amount": {
                "evaluated": self.amount_evaluated,
                "exact": self.amount_exact,
                "within_tolerance": self.amount_within_tolerance,
                "mae_ghs": round(self.amount_mae_ghs, 4),
            },
            "slots": {
                "product": self.product.as_dict(),
                "quantity": self.quantity.as_dict(),
                "unit": self.unit.as_dict(),
            },
        }


# ---------------------------------------------------------------- main entry

def evaluate(
    gold: list[SaleRecord],
    pred: list[SaleRecord],
    amount_tolerance: float = 0.05,
    near_match_case_insensitive: bool = True,
) -> EvaluationReport:
    """Evaluate predictions against gold, joining on utt_id.

    Records present only in one file are counted against the other
    (missing prediction = false negative for sale detection).
    """
    report = EvaluationReport()
    gold_by_id = {r.utt_id: r for r in gold}
    pred_by_id = {r.utt_id: r for r in pred}
    if len(pred_by_id) != len(pred):
        raise ValueError("duplicate utt_id in predictions")

    # ---- ASR (only where both sides have text)
    wer_by_lang: dict[str, list[float]] = {}
    wers: list[float] = []
    cers: list[float] = []
    for utt_id, g in gold_by_id.items():
        p = pred_by_id.get(utt_id)
        if g.text and p is not None and p.text:
            w = wer(g.text, p.text)
            c = cer(g.text, p.text)
            if w is not None:
                wers.append(w)
                lang = g.language or "unknown"
                wer_by_lang.setdefault(lang, []).append(w)
            if c is not None:
                cers.append(c)
    report.n_asr_pairs = len(wers)
    report.wer_mean = _mean(wers) if wers else None
    report.cer_mean = _mean(cers) if cers else None
    report.wer_by_language = {
        lang: round(_mean(vals), 4) for lang, vals in sorted(wer_by_lang.items())
    }

    # ---- Sale detection
    tp = fp = fn = 0
    gold_sales = 0
    for utt_id, g in gold_by_id.items():
        p = pred_by_id.get(utt_id)
        g_sale = bool(g.is_sale)
        p_sale = bool(p.is_sale) if p is not None else False
        if g_sale:
            gold_sales += 1
        if g_sale and p_sale:
            tp += 1
        elif not g_sale and p_sale:
            fp += 1
        elif g_sale and not p_sale:
            fn += 1
    for utt_id, p in pred_by_id.items():
        if utt_id not in gold_by_id and p.is_sale:
            fp += 1  # predicted sale with no gold row at all
    report.n_sales_gold = gold_sales
    report.sale_precision, report.sale_recall, report.sale_f1 = _prf(tp, fp, fn)

    # ---- Amount, slots (only on rows joined by utt_id)
    amount_errors: list[float] = []
    amount_exact = amount_tol = 0
    fm_product, fm_qty, fm_unit = FieldMetrics(), FieldMetrics(), FieldMetrics()

    def _slot_update(fm: FieldMetrics, g_val, p_val, near: bool = False) -> None:
        if g_val is not None:
            fm.gold += 1
        if p_val is not None:
            fm.predicted += 1
        if g_val is not None and p_val is not None:
            if str(g_val) == str(p_val):
                fm.correct += 1
            elif near and near_match_case_insensitive and \
                    str(g_val).strip().lower() == str(p_val).strip().lower():
                fm.correct += 1
                fm.near_correct += 1

    for utt_id, g in gold_by_id.items():
        p = pred_by_id.get(utt_id)
        if p is None:
            continue
        # amount
        if g.is_sale and g.amount is not None:
            if p.amount is not None:
                report.amount_evaluated += 1
                err = abs(p.amount - g.amount)
                amount_errors.append(err)
                if err == 0:
                    amount_exact += 1
                if err <= amount_tolerance:
                    amount_tol += 1
        _slot_update(fm_product, g.product, p.product, near=True)
        _slot_update(fm_qty, g.quantity, p.quantity)
        _slot_update(fm_unit, g.unit, p.unit, near=True)

    report.amount_exact = amount_exact
    report.amount_within_tolerance = amount_tol
    report.amount_mae_ghs = _mean(amount_errors) if amount_errors else 0.0
    report.product, report.quantity, report.unit = fm_product, fm_qty, fm_unit
    return report


def format_report(report: EvaluationReport) -> str:
    d = report.as_dict()
    lines = [
        "=" * 62,
        "SikaBook evaluation report",
        "=" * 62,
        f"ASR pairs: {d['asr']['pairs']}  "
        f"WER: {d['asr']['wer']}  CER: {d['asr']['cer']}",
    ]
    for lang, w in d["asr"]["wer_by_language"].items():
        lines.append(f"  WER[{lang}] = {w}")
    lines += [
        f"Sale detection  P={d['sale_detection']['precision']} "
        f"R={d['sale_detection']['recall']} F1={d['sale_detection']['f1']} "
        f"(gold sales: {d['sale_detection']['gold_sales']})",
        f"Amount          n={d['amount']['evaluated']}  "
        f"exact={d['amount']['exact']}  "
        f"within tol={d['amount']['within_tolerance']}  "
        f"MAE={d['amount']['mae_ghs']} GHS",
        f"Product  slot   P={d['slots']['product']['precision']} "
        f"R={d['slots']['product']['recall']} F1={d['slots']['product']['f1']}",
        f"Quantity slot   P={d['slots']['quantity']['precision']} "
        f"R={d['slots']['quantity']['recall']} F1={d['slots']['quantity']['f1']}",
        f"Unit     slot   P={d['slots']['unit']['precision']} "
        f"R={d['slots']['unit']['recall']} F1={d['slots']['unit']['f1']}",
        "=" * 62,
    ]
    return "\n".join(lines)
