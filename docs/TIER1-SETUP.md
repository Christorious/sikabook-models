# Tier 1 Setup — Language Model Adaptation (Your Laptop, No GPU)

This guide walks you through rebuilding the Vosk language model so it's
biased toward Ghanaian market-trading vocabulary. This is the highest-ROI
step: it takes minutes, needs no GPU, and gives the biggest accuracy boost.

## What Tier 1 does

The base Vosk English model (`vosk-model-small-en-us-0.15`) was trained on
American English. It doesn't know words like "polythene", "sachet", "cedis",
"pesewas", or Ghanaian names like "Ama" and "Kofi".

Tier 1 replaces the language model (`Gr.fst`) inside the existing model with
one built from our trading corpus. The acoustic model stays the same — we're
just telling it which words are likely to appear.

**You keep the same 40 MB model size, but it now knows trader vocabulary.**

## Prerequisites

You need a Linux environment (your HP laptop with WSL2 works perfectly).
We'll compile Kaldi — the underlying toolkit that Vosk is built on.

### Step 1: Install build dependencies

```bash
sudo apt update
sudo apt install -y build-essential git wget python3 zlib1g-dev \
    automake autoconf libtool subversion
```

### Step 2: Clone and compile Kaldi (~30 min one-time)

```bash
cd ~/sikabook-models

git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi/tools
make -j4          # -j4 uses 4 CPU cores, adjust to your CPU
```

This compiles OpenFST and other Kaldi dependencies. On your HP G-series
with 16 GB RAM, `make -j4` should complete in 20–30 minutes.

### Step 3: Install OpenGrM (for n-gram language model building)

```bash
cd ~/sikabook-models/kaldi/tools
extras/install_opengrm.sh
```

This installs the `ngramcount` and `ngrammake` tools we need.

### Step 4: Clean the corpus

```bash
cd ~/sikabook-models
python3 scripts/clean_corpus.py
```

This normalizes the trading corpus — lowercases everything, removes
punctuation, deduplicates. You should see output like:

```
Corpus cleaned successfully.
  Input:   .../trading_corpus.txt (300 lines)
  Output:  .../trading_corpus.clean.txt (280 unique lines)
  Vocabulary: 85 unique words
```

### Step 5: Build the adapted language model

```bash
cd ~/sikabook-models
chmod +x scripts/build_lm.sh
./scripts/build_lm.sh
```

This script:
1. Downloads the base Vosk small English model (40 MB)
2. Copies it to `models/sikabook-en-gh-v1/`
3. Builds a new n-gram language model from your corpus
4. Replaces the `Gr.fst` file inside the model

When it finishes, your adapted model is at:
```
sikabook-models/models/sikabook-en-gh-v1/
```

### Step 6: Test the adapted model

You can test it with the Vosk Python CLI:

```bash
pip install vosk soundfile

python3 - <<'EOF'
import json
from vosk import Model, KaldiRecognizer
import wave

model = Model("sikabook-models/models/sikabook-en-gh-v1")
rec = KaldiRecognizer(model, 16000)

# If you have a test WAV file (16kHz, mono):
# wf = wave.open("test.wav", "rb")
# while True:
#     data = wf.readframes(4000)
#     if len(data) == 0: break
#     rec.AcceptWaveform(data)
# print(json.loads(rec.FinalResult())["text"])
print("Model loaded successfully. Ready for testing.")
EOF
```

## Using the adapted model in SikaBook

The SikaBook web app loads the model from a URL. To use your adapted model:

### Option A: Host it on GitHub (free)

```bash
cd sikabook-models/models
tar czf sikabook-en-gh-v1.tar.gz sikabook-en-gh-v1/
```

Upload the `.tar.gz` to a GitHub release and update `MODEL_URL` in
`sikabook/app.js` to point to the release download URL.

### Option B: Host on HuggingFace (free, better CDN)

```bash
# Create a HuggingFace model repo, then:
huggingface-cli upload your-username/sikabook-en-gh-v1 \
    sikabook-en-gh-v1.tar.gz
```

Update `MODEL_URL` in `app.js`:
```javascript
const MODEL_URL = "https://huggingface.co/your-username/sikabook-en-gh-v1/resolve/main/sikabook-en-gh-v1.tar.gz";
```

## Improving the corpus

The current corpus has ~300 sentences. To improve recognition further:

1. **Add more sentences** to `data/corpus/trading_corpus.txt`
2. **Add real transcriptions** — record yourself saying sales and add the text
3. **Add new goods** — if traders sell items not in the list, add sentences with them
4. **Re-run** `clean_corpus.py` then `build_lm.sh`

The more sentences that reflect real trader speech, the better the model gets.
Even 500–1000 sentences would make a significant difference.

## Troubleshooting

### "Kaldi not found"

The `build_lm.sh` script searches these locations:
- `~/sikabook-models/kaldi`
- `~/kaldi`
- `/opt/kaldi`

If you cloned Kaldi elsewhere, set `KALDI_ROOT` manually:
```bash
export KALDI_ROOT=/your/kaldi/path
./scripts/build_lm.sh
```

### "fstsymbols: command not found"

OpenFST isn't in your PATH. Make sure you ran `make` in `kaldi/tools/`
and that `extras/install_opengrm.sh` completed successfully.

### Model loads but recognition is worse

Check that the corpus sentences are realistic. If sentences are too
repetitive or don't reflect actual speech, the language model can over-fit.
Add variety — different sentence structures, different goods, different prices.

## What's next

Once Tier 1 is working and you've tested it in SikaBook:

- **Tier 2**: Fine-tune the acoustic model for Ghanaian English accents
  (needs a GPU — see [TIER2-3-KAGGLE.md](TIER2-3-KAGGLE.md))
- **Tier 3**: Train models for Twi, Ewe, Ga, Dagbani from scratch
  (see [TIER3-GHANAIAN-LANGUAGES.md](TIER3-GHANAIAN-LANGUAGES.md))
