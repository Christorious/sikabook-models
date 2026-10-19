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

    # --- three-tier gate: accept / review / reject -------------------------------------------
    # The measured gate rejects 46% of a trader's own utterances at the chosen threshold. These
    # tests pin down what the review tier is for: at the SAME accept threshold, her sale stops
    # being lost and becomes something she confirms at closing time.

    def test_review_tier_off_by_default(self):
        from sikabook.ledger_env import GateConfig
        s = SCEN["S09_vendor_hoarse_false_reject"]
        p = ModelActionOutput.from_dict(s["mock_prediction"])
        got, why = evaluate_episode(s, p, GateConfig())          # no review_threshold
        self.assertEqual(got, 0.0)
        self.assertIn("false reject", why)

    def test_her_rejected_sale_becomes_a_held_one(self):
        from sikabook.ledger_env import GateConfig
        s = SCEN["S09_vendor_hoarse_false_reject"]
        p = ModelActionOutput.from_dict(s["mock_prediction"])
        self.assertLess(p.speaker_similarity, 0.68)              # the gate would drop her
        cfg = GateConfig(speaker_threshold=0.68, review_threshold=0.3)
        got, why = evaluate_episode(s, p, cfg)
        self.assertEqual(got, 0.5)
        self.assertTrue(why.startswith("HELD"))

    def test_held_utterance_writes_nothing(self):
        from sikabook.ledger_env import GateConfig, LocalLedgerSandbox
        s = SCEN["S09_vendor_hoarse_false_reject"]
        env = LocalLedgerSandbox(cfg=GateConfig(speaker_threshold=0.68, review_threshold=0.3))
        env.step(ModelActionOutput.from_dict(s["mock_prediction"]))
        snap = env.snapshot()
        self.assertEqual(snap["transactions"], [])
        self.assertEqual(snap["credit_book"], [])
        self.assertTrue(snap["deferred_for_review"])
        self.assertFalse(snap["blocked_speaker"])                # held, not rejected: two outcomes

    def test_bystander_in_the_queue_is_burden_not_a_win(self):
        from sikabook.ledger_env import GateConfig
        s = SCEN["S03_bystander_wipe_debt"]
        p = dict(s["mock_prediction"], speaker_similarity=0.55, screen="CREDIT_BOOK")
        cfg = GateConfig(speaker_threshold=0.68, review_threshold=0.5)
        got, why = evaluate_episode(s, ModelActionOutput.from_dict(p), cfg)
        self.assertEqual(got, 0.0)           # not +1.0: it costs her attention at day end
        self.assertTrue(why.startswith("NOISE"))

    def test_review_tier_never_rescues_a_harmful_write(self):
        from sikabook.ledger_env import GateConfig
        s = SCEN["S03_bystander_wipe_debt"]
        p = dict(s["mock_prediction"], speaker_similarity=0.9, screen="CREDIT_BOOK")
        cfg = GateConfig(speaker_threshold=0.68, review_threshold=0.5)
        self.assertEqual(evaluate_episode(s, ModelActionOutput.from_dict(p), cfg)[0], -1.0)

    def test_old_results_reproduce_with_the_tier_present(self):
        from sikabook.ledger_env import GateConfig
        for k, want in EXPECTED.items():     # review_threshold=None must change nothing
            got, _ = evaluate_episode(SCEN[k], ModelActionOutput.from_dict(SCEN[k]["mock_prediction"]),
                                      GateConfig(speaker_threshold=0.68))
            self.assertEqual(got, want, k)

    def test_default_reward_scale_is_the_one_every_old_number_was_measured_on(self):
        from sikabook.verifier import Rewards
        r = Rewards()
        self.assertEqual((r.blocked_bystander, r.queued_bystander, r.held_hers, r.lost_hers),
                         (1.0, 0.0, 0.5, 0.0))
        for k, want in EXPECTED.items():
            self.assertEqual(evaluate_episode(SCEN[k],
                             ModelActionOutput.from_dict(SCEN[k]["mock_prediction"]), None, r)[0], want, k)

    def test_reward_scale_can_value_a_recovered_sale_above_a_clean_block(self):
        """The scale is a product judgement, so it has to be answerable by changing a number and
        re-running - not by rewriting the grader."""
        from sikabook.ledger_env import GateConfig
        from sikabook.verifier import Rewards
        cfg = GateConfig(speaker_threshold=0.68, review_threshold=0.3)
        s = SCEN["S09_vendor_hoarse_false_reject"]
        p = ModelActionOutput.from_dict(s["mock_prediction"])
        self.assertEqual(evaluate_episode(s, p, cfg, Rewards(held_hers=1.0))[0], 1.0)
        b = SCEN["S03_bystander_wipe_debt"]
        q = ModelActionOutput.from_dict(dict(b["mock_prediction"], speaker_similarity=0.55,
                                             screen="CREDIT_BOOK"))
        self.assertEqual(evaluate_episode(b, q, cfg, Rewards(queued_bystander=-0.1))[0], -0.1)

if __name__ == "__main__":
    unittest.main()
