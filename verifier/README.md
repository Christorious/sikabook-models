# sikabook verifier (v0)

Grades what a Voice Ledger model *does to the ledger*, not what it transcribes.
Pure Python 3.9+, no installs, runs on CPU (InstaBox is fine).

## Run on InstaBox
    instabox create --name sika --template universal
    instabox ssh sika
    # upload/unzip sikabook-verifier.zip, then:
    cd sikabook-verifier
    python3 run_eval.py
    python3 -m unittest -v
    python3 run_eval.py --threshold 0.60   # watch FAR/FRR move

## What's here
- sikabook/schema.py      screen-masked action space + Twi number reference (check spellings with native speakers)
- sikabook/ledger_env.py  in-memory SQLite ledger: voice gate -> screen mask -> clarify-if-unsure -> write
- sikabook/verifier.py    +1 / +0.5 / 0 / -1 grader
- scenarios/market_eval_set.json  15 scenarios, incl. 5 old-cedi cases (S11-S15), incl. bystander, fake MoMo, 20-vs-50 trap, false reject
- run_eval.py             prints score, success, harmful-write rate, FAR, FRR

## Plug in your model later
Write preds.json mapping scenario id -> ModelActionOutput dict, then `python3 run_eval.py --preds preds.json`.

## Honest limits
- Thresholds (0.68 voice, 0.55 confidence) are placeholders. Tune on real enrolled audio.
- MoMo check trusts the SMS sender name; real spoofing needs more (sender ID + format + balance).
- 10 hand-written scenarios is a smoke test, not a benchmark. Grow it from real trader interviews.

## Real Twi clips (synth/fisd_loader.py)
Pulls real Asante Twi speech from the Ashesi/Nokwary Financial Inclusion Speech Dataset (CC-BY-4.0).
    sudo apt install -y python3-pip
    pip3 install --break-system-packages pyarrow soundfile numpy scipy
    python3 synth/fisd_loader.py --out data/fisd_v0 --noise
    python3 run_eval.py --scenarios data/fisd_v0/scenarios.json
Test split: ~470 usable clips (airtime amounts, aane/daabi, balance/number talk) from ~210 real
speakers, x2 with --noise. speakers.json lets you enroll one speaker and use the rest as bystanders.
Banking phrases only: market-sale lines (fish, quantities, credit book) still need your own recordings.
Don't commit data/ (it's in .gitignore).

## First tiny model (train_tiny.py)
    python3 train_tiny.py --data data/fisd_v0
    python3 run_eval.py --scenarios data/fisd_v0/scenarios_heldout.json --preds data/fisd_v0/preds_tiny.json | tail -1
Log-mel features -> softmax regression, numpy only, seconds on CPU. Graded on ~20% of speakers it never heard.
First run: 69% accuracy, score 135/182 (oracle 171), harmful writes 5%. Speaker check not modelled yet (fixed 0.85).
Note: the verifier can't yet tell CONFIRM from CANCEL (no pending action in these scenarios), so read the accuracy line too.

## KasaSpeech hard negatives (synth/kasa_loader.py)
    python3 synth/kasa_loader.py --out data/kasa_v0 --noise
    python3 train_tiny.py --data data/fisd_v0 data/kasa_v0
    python3 run_eval.py --scenarios data/fisd_v0/scenarios_heldout.json --preds data/fisd_v0/preds_tiny.json | tail -1
English-Twi code-switched chat about cedis/market/fish (Apache-2.0) labelled NO_OP: money talk must not write.
Result (L-BFGS, reproducible): accuracy 81%, score 186/224 (oracle 212), harmful writes 2%. Unsure money (< 0.75) -> ask to repeat.

## Voice gate v0 (speaker_gate.py)
    python3 train_tiny.py --data data/fisd_v0 data/kasa_v0
    python3 speaker_gate.py --data data/fisd_v0 data/kasa_v0
    python3 run_eval.py --scenarios data/fisd_v0/scenarios_gate.json --preds data/fisd_v0/preds_gate.json | tail -1
MFCC/log-mel stats -> LDA -> cosine vs ONE enrolment clip. Separate LDA-train / dev (threshold) / test speakers.
First run: dev EER 13%, test EER 26%; at the dev threshold FAR 10%, FRR 46%. Read the script's FAR/FRR line:
the ledger FAR in run_eval only counts bystanders whose clip would have written money, so it looks better than the gate is.

**Those four numbers have to be re-run.** `fit_lda` iterated an unordered set, so the LDA basis followed
`PYTHONHASHSEED` and changed from process to process. On the smoke corpus that alone swung dev EER between
0% and 33% with nothing else altered. Fixed in v1 (`sorted(set(...))`), verified identical across four hash
seeds. Until the re-run, treat the v0 gate line as one draw, not a measurement. `train_tiny.py` was checked
and is not affected, so the 81% / 186 / 2% line above stands.

## Voice gate v1 (speaker_gate.py)
    python3 speaker_gate.py --data data/fisd_v0 data/kasa_v0 --encoder ecapa
    python3 run_eval.py --scenarios data/fisd_v0/scenarios_gate.json \
                        --preds data/fisd_v0/preds_gate.json --threshold 0.68 --review-threshold 0.5 | tail -2
Two changes, written because FRR 46% means half her sales never reach the ledger.

**The gate ranks instead of only filtering.** Three tiers, not two: accept (write unasked), review (hold for
the day-end confirmation she already does), reject. Because she confirms the day anyway, an uncertain
utterance does not have to be destroyed. `--target-far` sets the accept threshold, `--target-loss` the reject
threshold, and the script prints queue size per 100 of her utterances, so FRR now trades against review
burden (ours to choose) instead of coverage (unrecoverable). `GateConfig.review_threshold` defaults to None,
so every number measured before this existed still reproduces.

**Per-speaker calibration** (`--norm raw|znorm|asnorm`). Score her voiceprint at enrolment against a cohort of
other speakers that ships with the app, and express later scores in units of her own distribution. `znorm` is
enrolment-side only and so is exactly equivalent to storing one raw-cosine threshold per trader at setup
(written to the `.npz` as `spk_thr_lo/hi`, which is what a phone would use - tested for exact agreement);
`asnorm` also normalises by the test utterance's cohort statistics.

**Session adaptation** (`--adapt`, optional). The most frequent speaker at the stall across a market day is
her, so her voiceprint is averaged with the utterances the gate already accepted. No labels, only the gate's
own accepts - and it adapts on her own test-day audio, so it is reported as a session gain, never held-out.

Measured so far, honestly: on real audio, nothing yet - the corpora are not in this repo, and the v0 line
above has to be re-run after the determinism fix. On synthetic embeddings (`tests/test_gate_math.py`, 28
tests, no audio needed) and on the smoke corpus, the ranker is the big lever - coverage rises from roughly
75% to roughly 96% at an unchanged accept threshold, for every calibration and every seed - while `asnorm`
consistently lowers test EER (smoke corpus: 33.3% -> 22.9%) and `znorm` does not reliably do anything.
Neither closed the dev-to-test threshold-transfer gap across seeds, which is the reason they were written,
so that claim is not being made.

**The grader's reward scale is probably wrong, and this is how we found out.** On the smoke corpus the review
tier cuts the rate at which her own speech is lost from 42% to 6% - the one failure that stops the ledger
being worth having - and the total score *falls*, 68.0 to 56.5 / 96. The arithmetic is exact: 20 bystander
utterances moved from +1.0 (cleanly blocked) to 0.0 (noise in her queue), while only 17 of hers moved from
0.0 (lost) to +0.5 (held). So the scale as written says one cleanly blocked bystander is worth two recovered
sales, and it would rather lose four sales in ten than put twenty lines in front of her at closing time. Had
this not been caught it would have silently steered every later model choice the wrong way.
The four numbers are now `verifier.Rewards`, defaults unchanged and tested to reproduce every earlier result,
so the question can be settled by changing a number and re-running. It is a product judgement, so it is
flagged, not retuned.

    python3 tests/smoke_corpus.py --out data/_smoke     # synthetic audio: pipeline check, not data
    python3 -m unittest discover -s tests -t . -q       # 28 tests, stdlib + numpy, ~2 s

Next: pretrained speaker embeddings on the real corpora (`--encoder ecapa`), 3+ enrolment phrases, and the
re-run of every number in the v0 line above.

## Real Ghanaian noise (synth/gesma_noise.py)
    pip3 install --break-system-packages openpyxl
    python3 synth/gesma_noise.py fetch
    python3 synth/gesma_noise.py remix --data data/fisd_v0 data/kasa_v0
GESMA (RAIL KNUST, CC-BY-4.0): market sounds, crowd murmurs, vendor calls, generator hum, street.
126 clips (~66 MB) read straight out of the 7 GB Zenodo zip. Test speakers only hear test-pool noise.
Re-running a loader puts the fake hum back; run remix again after it.
Result vs fake hum: accuracy 81% -> 74% (real noise is harder), harmful writes 2% -> 1%, score 186 -> 188/224.
Voice gate test EER 26% -> 24%.

## Old cedis (S11-S15)
Traders often quote pre-2007 cedis: "seven million" = GH₵700, "two hundred thousand" = GH₵20.
`schema.old_cedi_to_ghs()` converts; the model must output amount_ghs in NEW cedis.
Safety net: any amount over GateConfig.max_amount_ghs (20,000) asks instead of writing, so a
forgotten conversion becomes a 0 (miss), not a -1 (wrong money). "five fifty" is ambiguous: ask.
