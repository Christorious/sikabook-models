# Tier 2 & 3 — Free GPU Training on Kaggle

You don't have a GPU. You don't need to buy one. This guide covers training
the acoustic model (Tier 2) and full language models (Tier 3) using free
cloud GPUs.

## Why Kaggle is the best free option

| Platform | GPU | Free hours | Session limit | Credit card |
|----------|-----|------------|---------------|-------------|
| **Kaggle** | T4 or P100 (16 GB VRAM) | **30 hrs/week** | 9 hrs | No |
| Google Colab | T4 (16 GB VRAM) | ~15–30 hrs/week | 12 hrs | No |
| Lightning AI | T4/L4 | ~22 hrs/month | 4 hr restart | No |

Kaggle wins because:
- **Fixed 30 hours/week** (Colab's quota fluctuates)
- **Persistent storage** (20 GB — your data and checkpoints survive)
- **No credit card**
- **No internet during training** — but we work around this (see below)

## Setting up Kaggle for Vosk/Kaldi training

### Step 1: Create a Kaggle account

Go to [kaggle.com](https://kaggle.com) and sign up (free, no credit card).

### Step 2: Get your API key

1. Click your profile picture → **Settings**
2. Scroll to **API** section → **Create New Token**
3. This downloads `kaggle.json`

```bash
# On your laptop
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle
```

### Step 3: Upload datasets as Kaggle Datasets

Kaggle notebooks have **no internet during GPU training**. So you must
upload all data and tools as Kaggle Datasets beforehand.

#### Upload Kaldi (compiled)

```bash
# On your laptop — compile Kaldi first (CPU only, ~30 min)
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi/tools && make
cd ../src && ./configure --use-cuda=no && make -j4

# Package it
cd ~/sikabook-models
tar czf kaldi-compiled.tar.gz kaldi/

# Upload as a Kaggle dataset
kaggle datasets create -p /path/to/kaldi-dataset-folder
```

#### Upload speech datasets

For Tier 3, download these and upload each as a Kaggle Dataset:

```bash
# Twi speech data (21,138 audio-text pairs, CC-BY-4.0)
# From: https://huggingface.co/datasets/ghananlpcommunity/twi-speech-text-multispeaker-16k

# WAXAL dataset (Akan, Ewe — 1,250 hrs, CC-BY-4.0)
# From: https://huggingface.co/datasets/google/WaxalNLP

# UG Speech Data (Akan, Ewe, Dagbani, Dagaare, Ikposo — research use)
# From: https://github.com/HCI-LAB-UGSPEECHDATA
```

For each dataset:
1. Download to your laptop
2. Create a Kaggle Dataset and upload
3. Note the dataset slug (e.g., `your-username/twi-speech-data`)

### Step 4: Create a training notebook

In Kaggle, create a new Notebook:
1. **Settings → Accelerator → GPU (T4 x2 or P100)**
2. **Settings → Internet → Off** (required for GPU sessions)
3. Add your datasets: **Add Input → your datasets**

Here's the notebook structure:

```python
# ============================================================
# SikaBook Model Training — Tier 2: Ghanaian English Fine-tune
# Run on Kaggle with GPU enabled, Internet OFF
# ============================================================

import os
import sys

# --- Paths (Kaggle mounts datasets under /kaggle/input/) ---
KALDI_ROOT = "/kaggle/input/kaldi-compiled/kaldi"
DATA_DIR = "/kaggle/input/ghanaian-english-speech"
OUTPUT_DIR = "/kaggle/working/model-output"

os.environ["KALDI_ROOT"] = KALDI_ROOT
os.environ["PATH"] = f"{KALDI_ROOT}/tools/openfst/bin:{os.environ['PATH']}"
os.environ["LD_LIBRARY_PATH"] = f"{KALDI_ROOT}/tools/openfst/lib/fst"

# --- Step 1: Prepare data in Kaldi format ---
# Kaldi expects: text, wav.scp, utt2spk, spk2utt
os.makedirs(f"{OUTPUT_DIR}/data/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/data/test", exist_ok=True)

# Convert your audio + transcripts to Kaldi format
# (See scripts/prepare_kaldi_data.py for the conversion utility)

# --- Step 2: Fine-tune the acoustic model ---
# This runs the Kaldi chain model training recipe
# The script adapts the existing TDNN model to Ghanaian English

# Navigate to the Vosk training recipe
os.chdir(f"{KALDI_ROOT}/egs/mini_librispeech/s5")

# Run training (this takes 2–6 hours on a T4)
os.system("bash run_finetune.sh")

# --- Step 3: Package the model ---
os.chdir(OUTPUT_DIR)
os.system("tar czf sikabook-en-gh-v1.tar.gz sikabook-en-gh-v1/")
```

## Tier 2: Ghanaian English acoustic fine-tune

**Goal:** Make the model understand Ghanaian-accented English.

**Data needed:** 1–5 hours of transcribed Ghanaian English speech.

### Collecting the data

You can crowdsource this through SikaBook itself:

1. Add an opt-in "Help improve voice recognition" feature
2. Trader speaks a sale → the app saves the audio + the confirmed text
3. Upload batches to your server (or Kaggle Dataset) when the phone is charging + on WiFi

Even 100 recordings (about 1 hour of speech) from 10–20 different traders
would produce a noticeable improvement.

### Training time

| Data size | T4 GPU time | Fits in one Kaggle session? |
|-----------|-------------|----------------------------|
| 1 hour | ~2 hours | Yes, easily |
| 5 hours | ~4–6 hours | Yes |
| 10 hours | ~8–10 hours | Borderline (9 hr limit) |

### Checkpoint strategy

Kaldi saves checkpoints during training. If your session dies:

```python
# In your next Kaggle session, resume from checkpoint:
os.system("bash run_finetune.sh --stage 10")
# Stage 10 = resume from existing checkpoint, skip data prep
```

## Tier 3: Full Ghanaian language models

**Goal:** Train Twi, Ewe, Ga, Dagbani models from scratch.

**Data needed per language:**

| Language | Dataset | Hours | License |
|----------|---------|-------|---------|
| Twi (Akan) | UG Speech Data + Twi Speech-Text + WAXAL | ~120 hrs | Mixed |
| Ewe | UG Speech Data + WAXAL | ~100 hrs | Research + CC-BY-4.0 |
| Ga | AdwumaTech mghana-st | ~5 hrs | MIT |
| Dagbani | UG Speech Data | ~100 hrs | Research use |

### Training pipeline per language

1. **Build a phonetic lexicon** — map each word to phonemes
   - For Twi/Ewe: use the existing GhanaNLP text corpora to build word lists
   - Use `g2p` (grapheme-to-phoneme) tools if no manual lexicon exists

2. **Prepare Kaldi-format data** — `text`, `wav.scp`, `utt2spk`, `spk2utt`

3. **Train TDNN chain model** using the Vosk recipe:
   - Small model target (~40 MB, like `vosk-model-small-en-us-0.15`)
   - ivector dim 40 (not 100) for mobile
   - No pitch features (saves size + latency)

4. **Package as Vosk model** — arrange files per the Vosk layout:
   ```
   sikabook-tw-v1/
   ├── am/final.mdl
   ├── am/global_cmvn.stats
   ├── conf/
   ├── Gr.fst
   ├── HCLr.fst
   ├── words.txt
   └── README
   ```

### Estimated GPU time per language

| Language | Est. GPU hours | Kaggle sessions |
|----------|----------------|-----------------|
| Twi (120 hrs data) | 12–15 hrs | 2 sessions |
| Ewe (100 hrs data) | 10–12 hrs | 2 sessions |
| Dagbani (100 hrs data) | 10–12 hrs | 2 sessions |
| Ga (5 hrs data) | 2–3 hrs | 1 session |

With 30 Kaggle hours/week + ~20 Colab hours/week = **~50 free GPU hours/week**.
All four languages in 2–3 weeks.

## Google Colab as backup

When you run out of Kaggle hours:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Runtime → Change runtime type → T4 GPU
3. Mount Google Drive for persistent storage:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Same workflow as Kaggle, but store checkpoints in Google Drive

Colab sessions last up to 12 hours but can disconnect after ~90 min idle.
**Keep the tab open and interact periodically.**

## Checklist before each training run

- [ ] GPU enabled (T4 or P100)
- [ ] Internet OFF (Kaggle requirement for GPU)
- [ ] All datasets added as inputs
- [ ] Kaldi compiled and uploaded as a dataset
- [ ] Output directory writable (`/kaggle/working/` or `/content/drive/`)
- [ ] Checkpoint resume flag ready in case of disconnect

## Cost: $0

Total spend for all three tiers: **zero dollars.** You need:
- Your HP laptop (already have)
- A free Kaggle account
- A free Google account (for Colab + Drive)
- Time and patience
