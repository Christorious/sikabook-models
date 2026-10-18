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
Next: pretrained speaker embeddings (ECAPA-TDNN) and 3+ enrolment phrases.

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
