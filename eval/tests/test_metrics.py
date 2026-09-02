import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sikabook_eval.events import SaleRecord, parse_record
from sikabook_eval.metrics import evaluate, cer, format_report, wer


def rec(utt_id, **kw):
    return SaleRecord(utt_id=utt_id, **kw)


class TestWer(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(wer("i sold two fish", "i sold two fish"), 0.0)

    def test_one_substitution(self):
        self.assertAlmostEqual(wer("i sold two fish", "i sold to fish"), 1 / 4)

    def test_insertion_and_deletion(self):
        # ref 3 words, hyp 4 words, 1 insertion + 1 sub = 2 errors
        self.assertAlmostEqual(wer("i sold fish", "i then sold fresh fish"), 2 / 3)

    def test_empty_reference_is_none(self):
        self.assertIsNone(wer("", "anything"))

    def test_cer(self):
        self.assertAlmostEqual(cer("cedis", "cediz"), 1 / 5)
        self.assertIsNone(cer("", "x"))


class TestParseRecord(unittest.TestCase):
    def test_requires_utt_id(self):
        with self.assertRaises(ValueError):
            parse_record({"text": "hi"}, "t", 1)

    def test_coercion_and_extra_fields(self):
        rec = parse_record(
            {"utt_id": 5, "amount": "12.5", "quantity": "3", "is_sale": True,
             "note": "extra ignored"}, "t", 1)
        self.assertEqual(rec.utt_id, "5")
        self.assertEqual(rec.amount, 12.5)
        self.assertEqual(rec.quantity, 3)
        self.assertTrue(rec.is_sale)
        self.assertEqual(rec.extra, {"note": "extra ignored"})

    def test_bad_amount_is_none(self):
        rec = parse_record({"utt_id": "a", "amount": "not-a-number"}, "t", 1)
        self.assertIsNone(rec.amount)


class TestEvaluate(unittest.TestCase):
    def test_sale_detection_counts(self):
        gold = [
            rec("u1", is_sale=True, amount=20.0),
            rec("u2", is_sale=False),
            rec("u3", is_sale=True, amount=5.0),
        ]
        pred = [
            rec("u1", is_sale=True, amount=20.0),
            rec("u2", is_sale=True),          # false positive
            # u3 missing -> false negative
        ]
        rep = evaluate(gold, pred)
        self.assertEqual(rep.sale_precision, 0.5)  # 1 tp, 1 fp
        self.assertEqual(rep.sale_recall, 0.5)     # 1 tp, 1 fn

    def test_amount_accuracy_and_tolerance(self):
        gold = [rec("u1", is_sale=True, amount=10.00),
                rec("u2", is_sale=True, amount=2.00)]
        pred = [rec("u1", is_sale=True, amount=10.00),
                rec("u2", is_sale=True, amount=2.03)]
        rep = evaluate(gold, pred)
        self.assertEqual(rep.amount_evaluated, 2)
        self.assertEqual(rep.amount_exact, 1)
        self.assertEqual(rep.amount_within_tolerance, 2)
        self.assertAlmostEqual(rep.amount_mae_ghs, 0.015)

    def test_amount_null_prediction_not_evaluated(self):
        gold = [rec("u1", is_sale=True, amount=10.0)]
        pred = [rec("u1", is_sale=True)]  # no amount predicted
        rep = evaluate(gold, pred)
        self.assertEqual(rep.amount_evaluated, 0)
        # but the sale itself was detected
        self.assertEqual(rep.sale_recall, 1.0)

    def test_slot_metrics_with_near_match(self):
        gold = [rec("u1", product="Tilapia", quantity=2, unit="bowl")]
        pred = [rec("u1", product="tilapia", quantity=2, unit="Bowl")]
        rep = evaluate(gold, pred)
        self.assertEqual(rep.product.correct, 1)
        self.assertEqual(rep.product.near_correct, 1)
        self.assertEqual(rep.quantity.f1, 1.0)
        self.assertEqual(rep.unit.correct, 1)

    def test_wer_by_language(self):
        gold = [
            rec("u1", text="aduasa cedis", language="tw"),
            rec("u2", text="two cedis", language="en-GH"),
        ]
        pred = [rec("u1", text="aduasa cedis"), rec("u2", text="too cedis")]
        rep = evaluate(gold, pred)
        self.assertEqual(rep.wer_by_language["tw"], 0.0)
        self.assertAlmostEqual(rep.wer_by_language["en-GH"], 0.5)
        # unknown language bucket for unlabeled predictions' gold rows
        self.assertIn("tw", rep.wer_by_language)

    def test_duplicate_prediction_ids_rejected(self):
        gold = [rec("u1", is_sale=True)]
        pred = [rec("u1", is_sale=True), rec("u1", is_sale=False)]
        with self.assertRaises(ValueError):
            evaluate(gold, pred)

    def test_format_report_runs(self):
        gold = [rec("u1", text="a b", is_sale=True, amount=1.0, product="X")]
        pred = [rec("u1", text="a c", is_sale=True, amount=1.0, product="X")]
        text = format_report(evaluate(gold, pred))
        self.assertIn("SikaBook evaluation report", text)


if __name__ == "__main__":
    unittest.main()
