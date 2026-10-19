"""The gate's arithmetic, on synthetic embeddings. No audio, no corpus, seconds to run.

The real gate needs FISD/KasaSpeech audio, which is not in the repo, so these tests do not tell
you the gate's EER. What they do is pin down what speaker_gate.py claims, on data built to contain
exactly the problem those claims are about:

  Some voices sit close to every other voice and some sit far from all of them, and some voices
  vary much more from clip to clip than others. That alone makes one fixed cosine threshold mean
  different things for different traders, which is the shape of the measured failure (dev EER 13%,
  test EER 26%, dev threshold -> FAR 10% / FRR 46%).

What these tests found, which is worth stating because it is not what we set out to prove: on this
synthetic the three-tier ranker is the big lever (coverage ~96% against ~75% for the binary gate at
the same accept threshold, every seed, every calibration), while per-speaker calibration is a
second-order gain that grows as the encoder gets better - and znorm on its own is unreliable. So
the ranker is asserted here, asnorm's EER gain is asserted here, and the dev-to-test threshold
transfer gap is NOT asserted for either, because it did not hold across seeds. The real corpus
decides that one.
"""
import random, unittest

try:
    import numpy as np
    import speaker_gate as G
    HAVE, WHY = True, ""
except Exception as e:                       # numpy/soundfile absent: the rest of the suite is stdlib
    HAVE, WHY = False, repr(e)

# 5% rather than the script's 1% default: with ~2000 dev impostor trials a 1% quantile rests on
# about 20 of them, which is not enough to assert anything about.
TARGET_FAR, TARGET_LOSS = 0.05, 0.02


def synth(n_spk=200, n_src=8, d=32, sub=10, jit=(0.5, 1.6), noise=0.8, seed=0):
    """base_s = alpha_s * u + sqrt(1 - alpha_s^2) * (a direction of her own, drawn in a `sub`-dim
    subspace so voices crowd together). cos(base_s, base_t) grows with alpha_s * alpha_t, so a
    high-alpha speaker scores high against everyone: the hub effect one global threshold cannot
    cope with. Each speaker also gets her own jitter, so the spread of her scores differs too -
    that is what the sigma in the normalisation is for.

    `sub` and `jit` put raw test EER near 20%, in the neighbourhood of the 26% the real gate
    measures. They were chosen by targeting that difficulty, not by hunting for a difficulty that
    flatters the calibration.
    """
    rng = np.random.default_rng(seed)
    unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    B = unit(rng.normal(size=(sub, d)))
    u = unit(rng.normal(size=d))
    man, rows, spk_names = [], [], [f"S{i:03d}" for i in range(n_spk)]
    for s in spk_names:
        alpha = rng.uniform(0.05, 0.8)                     # hubness
        jitter = rng.uniform(*jit)                         # within-speaker spread
        base = unit(alpha * u + np.sqrt(1 - alpha ** 2) * unit(rng.normal(size=sub) @ B))
        for r in range(n_src):
            src = f"{s}_r{r}"
            for snr in (None, 9):                          # one clean clip, one noisy copy
                cid = f"{src}_{'c' if snr is None else 'n'}"
                man.append({"id": cid, "speaker": s, "snr_db": snr, "src": src, "wav": cid})
                extra = 0.0 if snr is None else noise      # noise pushes the copy further out
                rows.append(unit(base + (jitter + extra) * unit(rng.normal(size=d))))
    return np.stack(rows), man, spk_names


class Bench:
    """One synthetic corpus, split and trialled exactly as speaker_gate.py does it."""

    def __init__(self, seed=0):
        self.E, self.man, names = synth(seed=seed)
        spk = list(names); random.Random(0).shuffle(spk)
        k = max(1, int(len(spk) * 0.2))
        test_spk, dev_spk = set(spk[:k]), set(spk[k:k + max(1, int(len(spk) * 0.15))])
        idx = lambda want: [i for i, m in enumerate(self.man) if m["speaker"] in want]
        tr = [i for i, m in enumerate(self.man) if m["speaker"] not in test_spk | dev_spk]
        self.C = G.build_cohort(self.E, self.man, tr, 64, 0, 3)
        self.dg, self.di, self.denrol = G.trials(self.E, self.man, idx(dev_spk), 7, n_imp=5, k=1)
        self.tg, self.ti, self.tenrol = G.trials(self.E, self.man, idx(test_spk), 8, n_imp=5, k=1)
        self.raw = (G.cos(self.E, self.dg, self.denrol), G.cos(self.E, self.di, self.denrol),
                    G.cos(self.E, self.tg, self.tenrol), G.cos(self.E, self.ti, self.tenrol))

    def scores(self, mode):
        n = lambda pairs, raw, en: G.normalise(pairs, raw, en, self.E, self.C, mode, 20)
        return (n(self.dg, self.raw[0], self.denrol), n(self.di, self.raw[1], self.denrol),
                n(self.tg, self.raw[2], self.tenrol), n(self.ti, self.raw[3], self.tenrol))

    def operating_point(self, mode):
        gd, idv, gt, it = self.scores(mode)
        lo, hi = G.thresholds(gd, idv, TARGET_FAR, TARGET_LOSS)
        return gd, idv, gt, it, lo, hi


@unittest.skipUnless(HAVE, f"needs numpy + soundfile: {WHY}")
class GateMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = Bench()

    # --- the setup is honest: groups disjoint, cohort unseen, no self-impostors ----------------

    def test_cohort_has_both_forms_and_is_not_degenerate(self):
        models, utts = self.b.C
        self.assertGreater(len(models), 10)
        self.assertGreater(len(utts), len(models))       # several clips per cohort speaker
        self.assertEqual(models.shape[1], utts.shape[1])

    def test_genuine_trials_are_hers_and_impostor_trials_are_not(self):
        self.assertTrue(self.b.dg and self.b.tg and self.b.di and self.b.ti)
        for pairs in (self.b.dg, self.b.tg):
            for i, s in pairs: self.assertEqual(self.b.man[i]["speaker"], s)
        for pairs in (self.b.di, self.b.ti):
            for i, s in pairs: self.assertNotEqual(self.b.man[i]["speaker"], s)

    def test_dev_and_test_speakers_never_overlap(self):
        self.assertFalse(set(self.b.denrol) & set(self.b.tenrol))

    def test_raw_difficulty_is_in_the_range_the_real_gate_lives_in(self):
        """If this drifts far from the measured 26%, the other numbers stop meaning anything."""
        _, _, gt, it = self.b.scores("raw")
        self.assertTrue(0.10 < G.eer(gt, it)[0] < 0.35, G.eer(gt, it)[0])

    # --- claim 1: the gate ranks, and that is where the coverage comes from --------------------

    def test_the_review_tier_buys_coverage_the_binary_gate_loses(self):
        """The point of the middle tier: at the SAME accept threshold, her speech a binary gate
        would have dropped is queued instead. Coverage rises, the accept tier does not move, and
        the price is review burden."""
        for mode in ("raw", "znorm", "asnorm"):
            _, _, gt, it, lo, hi = self.b.operating_point(mode)
            binary = G.tiers(gt, it, hi, hi, 3.0)         # lo == hi: no review band, v0's gate
            three = G.tiers(gt, it, lo, hi, 3.0)
            with self.subTest(mode=mode):
                self.assertAlmostEqual(binary["queue_per_100"], 0.0, places=9)
                self.assertEqual(three["accept_far"], binary["accept_far"])  # accept tier untouched
                self.assertGreater(three["coverage"], binary["coverage"] + 0.10)
                self.assertLess(three["lost"], binary["lost"])
                self.assertGreater(three["queue_per_100"], 0.0)              # paid for in burden

    def test_coverage_gain_does_not_depend_on_calibration(self):
        """The ranker is the lever; calibration is a separate, smaller question."""
        for mode in ("raw", "znorm", "asnorm"):
            _, _, gt, it, lo, hi = self.b.operating_point(mode)
            with self.subTest(mode=mode):
                self.assertGreater(G.tiers(gt, it, lo, hi, 3.0)["coverage"], 0.90)

    def test_thresholds_hit_their_targets_on_the_speakers_they_were_chosen_on(self):
        for mode in ("raw", "znorm", "asnorm"):
            gd, idv, _, _, lo, hi = self.b.operating_point(mode)
            with self.subTest(mode=mode):
                self.assertLessEqual(lo, hi)
                self.assertAlmostEqual(float((idv >= hi).mean()), TARGET_FAR, delta=0.01)
                self.assertAlmostEqual(float((gd < lo).mean()), TARGET_LOSS, delta=0.01)

    def test_every_utterance_lands_in_exactly_one_tier(self):
        _, _, gt, it, lo, hi = self.b.operating_point("asnorm")
        t = G.tiers(gt, it, lo, hi, 3.0)
        self.assertAlmostEqual(t["auto"] + t["queue_hers"] / 100 + t["lost"], 1.0, places=6)
        self.assertAlmostEqual(t["coverage"], t["auto"] + t["queue_hers"] / 100, places=6)

    def test_burden_counts_bystanders_at_the_stated_ratio(self):
        _, _, gt, it, lo, hi = self.b.operating_point("asnorm")
        a, b = G.tiers(gt, it, lo, hi, 1.0), G.tiers(gt, it, lo, hi, 3.0)
        self.assertAlmostEqual(b["queue_bystander"], 3 * a["queue_bystander"], places=6)
        self.assertEqual(a["queue_hers"], b["queue_hers"])            # her own queue is unaffected

    # --- claim 2: per-speaker calibration. Asserted only where it held across seeds ------------

    def test_asnorm_beats_raw_on_test_eer_across_seeds(self):
        gains = []
        for seed in range(4):
            b = Bench(seed=seed) if seed else self.b
            _, _, gt, it = b.scores("raw");    raw = G.eer(gt, it)[0]
            _, _, gt, it = b.scores("asnorm"); asn = G.eer(gt, it)[0]
            gains.append(raw - asn)
        self.assertTrue(all(g >= -0.005 for g in gains), gains)   # never meaningfully worse
        self.assertGreater(sum(gains) / len(gains), 0.0, gains)   # better on average

    def test_stored_per_trader_thresholds_decide_exactly_what_znorm_decides(self):
        """The .npz ships two raw-cosine numbers per trader so the phone needs no cohort while she
        is trading. That shortcut is only legitimate if it is the same decision."""
        _, _, gt, it, lo, hi = self.b.operating_point("znorm")
        s_lo, s_hi = G.speaker_thresholds(self.b.tenrol, self.b.C, 20, lo, hi)
        self.assertEqual(set(s_hi), set(self.b.tenrol))
        for pairs, z in ((self.b.tg, gt), (self.b.ti, it)):
            for (i, s), zi in zip(pairs, z):
                c = float(self.b.E[i] @ self.b.tenrol[s])
                self.assertEqual(bool(zi >= hi), c >= s_hi[s] - 1e-9, (s, zi, c))
                self.assertEqual(bool(zi >= lo), c >= s_lo[s] - 1e-9, (s, zi, c))

    def test_calibration_is_per_speaker_and_not_one_shared_offset(self):
        """If every trader got the same correction this would all be pointless relabelling."""
        _, _, _, _, lo, hi = self.b.operating_point("znorm")
        _, s_hi = G.speaker_thresholds(self.b.tenrol, self.b.C, 20, lo, hi)
        v = np.array(sorted(s_hi.values()))
        self.assertGreater(v[-1] - v[0], 0.02, v)

    # --- the score the ledger actually reads ---------------------------------------------------

    def test_calibrator_anchors_the_tiers_the_env_expects(self):
        f = G.calibrator(-1.0, 2.0, (-4.0, 6.0))
        self.assertAlmostEqual(f(2.0), G.GATE, places=3)
        self.assertAlmostEqual(f(-1.0), G.REVIEW, places=3)
        self.assertGreaterEqual(f(2.5), G.GATE)
        self.assertTrue(G.REVIEW <= f(0.5) < G.GATE)
        self.assertLess(f(-3.0), G.REVIEW)
        self.assertEqual(f(-99.0), 0.0)
        self.assertLessEqual(f(99.0), 1.0)

    def test_calibrator_is_monotone_so_the_queue_is_a_ranking(self):
        f = G.calibrator(-1.0, 2.0, (-4.0, 6.0))
        ys = [f(x) for x in np.linspace(-6, 8, 400)]
        self.assertTrue(all(b >= a for a, b in zip(ys, ys[1:])))

    def test_calibrated_scores_reproduce_the_tiers_through_the_ledger_env(self):
        """End to end: the number written into preds_gate.json must make the sandbox accept, hold
        and reject exactly the utterances this script said it would."""
        from sikabook.ledger_env import GateConfig, LocalLedgerSandbox
        from sikabook.schema import ActionType, ModelActionOutput
        _, _, gt, it, lo, hi = self.b.operating_point("asnorm")
        f = G.calibrator(lo, hi, (float(np.r_[gt, it].min()), float(np.r_[gt, it].max())))
        cfg = GateConfig(speaker_threshold=G.GATE, review_threshold=G.REVIEW)
        for z in np.r_[gt, it]:
            env = LocalLedgerSandbox(cfg=cfg)
            env.step(ModelActionOutput(speaker_similarity=f(z), action=ActionType.LOG_CASH_SALE,
                                       item="tilapia", quantity=1, amount_ghs=20.0))
            s = env.snapshot()
            self.assertEqual(bool(s["transactions"]), bool(z >= hi), z)
            self.assertEqual(s["deferred_for_review"], bool(lo <= z < hi), z)

    # --- session adaptation --------------------------------------------------------------------

    def test_adaptation_is_a_no_op_when_the_gate_accepts_nothing(self):
        _, _, gt, it = self.b.scores("asnorm")
        pairs, z = list(self.b.tg) + list(self.b.ti), np.r_[gt, it]
        out, n = G.adapt_enrolments(self.b.E, pairs, z, self.b.tenrol, 1e9, 0.5)
        self.assertEqual(n, {})
        for s, v in self.b.tenrol.items():
            np.testing.assert_allclose(out[s], v, atol=1e-12)

    def test_adaptation_uses_only_what_the_gate_accepted_no_labels(self):
        _, _, gt, it, lo, hi = self.b.operating_point("asnorm")
        pairs, z = list(self.b.tg) + list(self.b.ti), np.r_[gt, it]
        out, n = G.adapt_enrolments(self.b.E, pairs, z, self.b.tenrol, hi, 0.5)
        accepted = {}
        for (i, s), zi in zip(pairs, z):
            if zi >= hi: accepted.setdefault(s, []).append(i)
        self.assertEqual(n, {s: len(v) for s, v in accepted.items()})
        self.assertTrue(accepted)                      # the test would be vacuous otherwise
        for s in self.b.tenrol:
            self.assertAlmostEqual(float(np.linalg.norm(out[s])), 1.0, places=6)
            if s not in accepted:
                np.testing.assert_allclose(out[s], self.b.tenrol[s], atol=1e-12)

    def test_adaptation_weight_is_bounded_so_one_mistake_cannot_take_over(self):
        """Even if every accepted clip were a bystander's, the voiceprint must stay mostly hers."""
        s = sorted(self.b.tenrol)[0]
        other = self.b.E[[i for i, m in enumerate(self.b.man) if m["speaker"] != s][:40]]
        pairs = [(i, s) for i in range(len(other))]
        out, _ = G.adapt_enrolments(other, pairs, np.ones(len(pairs)) * 10, {s: self.b.tenrol[s]},
                                    0.0, 0.5)
        self.assertGreater(float(out[s] @ self.b.tenrol[s]), 0.5)


if __name__ == "__main__":
    unittest.main()
