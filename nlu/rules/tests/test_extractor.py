import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "nlu" / "rules"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from sika_rules import extract  # noqa: E402
from sika_rules.numbers import parse_number_tokens  # noqa: E402
from sika_rules.vocab import ProductVocabulary  # noqa: E402

VOCAB = ProductVocabulary.load(REPO_ROOT / "data" / "nlu" / "product_vocabulary.json")


def run(text):
    return extract(text, VOCAB)


class TestNumbers(unittest.TestCase):
    def test_english(self):
        self.assertEqual(parse_number_tokens(["twenty", "one"]), 21)
        self.assertEqual(parse_number_tokens(["fifty"]), 50)
        self.assertEqual(parse_number_tokens(["one", "hundred", "and", "five"]), 105)

    def test_twi(self):
        self.assertEqual(parse_number_tokens(["aduasa"]), 30)
        self.assertEqual(parse_number_tokens(["aduonu", "mmiensa"]), 23)
        self.assertEqual(parse_number_tokens(["mmienu"]), 2)
        self.assertEqual(parse_number_tokens(["ɔha"]), 100)
        self.assertEqual(parse_number_tokens(["apem"]), 1000)

    def test_not_a_number(self):
        self.assertIsNone(parse_number_tokens(["kenkey"]))
        self.assertIsNone(parse_number_tokens([]))


class TestAmounts(unittest.TestCase):
    def test_explicit_cedis(self):
        self.assertEqual(run("i sold fish for twenty cedis")["amount"], 20.0)

    def test_pesewas_conversion(self):
        self.assertEqual(run("fifty pesewas")["amount"], 0.5)

    def test_cedis_pesewas_combination(self):
        self.assertEqual(run("three cedis fifty pesewas")["amount"], 3.5)

    def test_twi_tens(self):
        self.assertEqual(run("aduasa cedis")["amount"], 30.0)

    def test_gh_cedi_symbol(self):
        self.assertEqual(run("i sold mackerel for GH₵15")["amount"], 15.0)

    def test_last_amount_wins(self):
        # negotiation: quoted 20, agreed 15
        event = run("i said twenty cedis but we finished at fifteen cedis")
        self.assertEqual(event["amount"], 15.0)

    def test_unit_price_times_quantity(self):
        event = run("i sold two tilapia at ten cedis each")
        self.assertEqual(event["unit_price"], 10.0)
        self.assertEqual(event["amount"], 20.0)

    def test_quantity_does_not_leak_into_amount(self):
        event = run("i go take two tins of titus")
        self.assertIsNone(event["amount"])
        self.assertEqual(event["quantity"], 2)

    def test_no_amount(self):
        self.assertIsNone(run("she bought tilapia")["amount"])


class TestQuantity(unittest.TestCase):
    def test_unit_quantity(self):
        event = run("she bought five bowls of gari")
        self.assertEqual(event["quantity"], 5)
        self.assertEqual(event["unit"], "bowl")

    def test_ghana_units_canonicalized(self):
        event = run("i sold three kokoo of peppers")
        self.assertEqual(event["unit"], "bowl")
        event = run("one rubber of oil")
        self.assertEqual(event["unit"], "bucket")

    def test_twi_quantity_after_product(self):
        event = run("me pɛ apateshi mmienu")
        self.assertEqual(event["quantity"], 2)
        self.assertEqual(event["product"], "Tilapia")

    def test_price_number_is_not_quantity(self):
        event = run("aduasa cedis")
        self.assertIsNone(event["quantity"])


class TestProducts(unittest.TestCase):
    def test_canonicalization_twi(self):
        self.assertEqual(run("apateshi is finished")["product"], "Tilapia")

    def test_canonicalization_pidgin_name(self):
        self.assertEqual(run("two tins of titus")["product"], "Mackerel")

    def test_fuzzy_asr_error(self):
        event = run("i sold two tilapa this morning")
        self.assertEqual(event["product"], "Tilapia")
        self.assertTrue(event["needs_review"])  # fuzzy -> review

    def test_fuzzy_does_not_match_function_words(self):
        self.assertIsNone(run("abeg reduce am")["product"])

    def test_surface_recorded(self):
        event = run("buyer took kpanla")
        self.assertEqual(event["product"], "Mackerel")
        self.assertEqual(event["product_surface"], "kpanla")


class TestSaleDecision(unittest.TestCase):
    def test_seller_cue_with_amount(self):
        self.assertTrue(run("i sold tilapia for twenty cedis")["is_sale"])

    def test_buyer_order_pidgin(self):
        self.assertTrue(run("i go take two tins of titus")["is_sale"])

    def test_question_is_not_sale(self):
        self.assertFalse(run("how much be dis")["is_sale"])
        self.assertFalse(run("how much is this tilapia")["is_sale"])

    def test_negotiation_is_not_sale(self):
        event = run("twenty cedis is too much, abeg reduce am")
        self.assertFalse(event["is_sale"])

    def test_price_statement_only(self):
        self.assertFalse(run("aduasa cedis")["is_sale"])


class TestSchemaAndReview(unittest.TestCase):
    GOLDEN = {
        "i sold two tilapia at ten cedis each": {
            "is_sale": True, "amount": 20.0, "product": "Tilapia",
            "quantity": 2, "unit": None, "language": "en-GH",
        },
        "she bought five bowls of gari for forty cedis": {
            "is_sale": True, "amount": 40.0, "product": "Gari",
            "quantity": 5, "unit": "bowl",
        },
        "three cedis fifty pesewas": {"is_sale": False, "amount": 3.5},
        "how much be dis": {"is_sale": False},
        "aduasa cedis": {"is_sale": False, "amount": 30.0},
        "i go take two tins of titus": {
            "is_sale": True, "product": "Mackerel",
            "quantity": 2, "unit": "tin",
        },
    }

    def test_golden_set(self):
        for text, expected in self.GOLDEN.items():
            event = run(text)
            for key, value in expected.items():
                self.assertEqual(
                    event[key], value,
                    f"{key} mismatch for {text!r}: "
                    f"got {event[key]!r}, expected {value!r}")

    def test_schema_required_fields_present(self):
        schema = json.loads(
            (REPO_ROOT / "export" / "schema" / "sale_event.schema.json")
            .read_text(encoding="utf-8"))
        event = run("i sold two tilapia at ten cedis each")
        for field in schema["required"]:
            self.assertIn(field, event)

    def test_no_invented_values(self):
        event = run("how much be dis")
        self.assertIsNone(event["amount"])
        self.assertIsNone(event["product"])
        self.assertIsNone(event["quantity"])

    def test_confidence_bounds(self):
        for text in ("i sold two tilapia at ten cedis each", "how much be dis",
                     "aduasa cedis", "me pɛ apateshi mmienu"):
            event = run(text)
            self.assertGreaterEqual(event["confidence"], 0.0)
            self.assertLessEqual(event["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
