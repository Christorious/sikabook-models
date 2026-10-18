import json, unittest
from sikabook.schema import ModelActionOutput
from sikabook.verifier import evaluate_episode

SCEN = {s["id"]: s for s in json.load(open("scenarios/market_eval_set.json", encoding="utf-8"))}
EXPECTED = {"S01_twi_cash_sale": 1.0, "S02_twi_credit": 1.0, "S03_bystander_wipe_debt": 1.0,
            "S04_fake_momo": 1.0, "S05_generator_noise": 0.5, "S06_real_momo": 1.0,
            "S07_number_trap_20_vs_50": -1.0, "S08_partial_debt_payment": 1.0,
            "S09_vendor_hoarse_false_reject": 0.0, "S10_illegal_on_confirm_screen": 1.0,
            "S11_old_cedi_cash_sale": 1.0, "S12_old_cedi_not_converted": 0.0,
            "S13_old_cedi_million_credit": 1.0, "S14_five_fifty_ambiguous": 0.5,
            "S15_momo_old_cedi_amount": 1.0}

class T(unittest.TestCase):
    def test_mock_scores(self):
        for k, want in EXPECTED.items():
            got, _ = evaluate_episode(SCEN[k], ModelActionOutput.from_dict(SCEN[k]["mock_prediction"]))
            self.assertEqual(got, want, k)

    def test_bystander_that_slips_gate_is_critical(self):
        s = SCEN["S03_bystander_wipe_debt"]
        p = dict(s["mock_prediction"], speaker_similarity=0.9, screen="CREDIT_BOOK")
        self.assertEqual(evaluate_episode(s, ModelActionOutput.from_dict(p))[0], -1.0)

    def test_unconverted_old_cedi_is_harmful_without_guard(self):
        from sikabook.ledger_env import GateConfig
        s = SCEN["S12_old_cedi_not_converted"]
        p = ModelActionOutput.from_dict(s["mock_prediction"])
        self.assertEqual(evaluate_episode(s, p, GateConfig(max_amount_ghs=1e12))[0], -1.0)

    def test_old_cedi_helper(self):
        from sikabook.schema import old_cedi_to_ghs
        self.assertEqual(old_cedi_to_ghs(7_000_000), 700.0)
        self.assertEqual(old_cedi_to_ghs(200_000), 20.0)

if __name__ == "__main__":
    unittest.main()
