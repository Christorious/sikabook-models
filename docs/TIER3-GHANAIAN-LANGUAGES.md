# Tier 3 — Ghanaian Language Models (Twi, Ewe, Ga, Dagbani)

This is the ambitious goal: fully offline speech recognition models for
Ghanaian languages, running in the browser via Vosk WASM, with no internet
required.

## The problem this solves

Right now, if a trader speaks Twi, Ga, Ewe, or Dagbani to SikaBook, the only
option is the Khaya AI cloud API — which requires internet. There is **no**
offline browser-based speech model for any Ghanaian language today.

This project would be the first.

## Languages and priority

| Language | Speakers in Ghana | Priority | Data available | Data ready? |
|----------|-------------------|----------|----------------|-------------|
| **Twi (Akan)** | ~9M | 1 (highest) | ~120 hrs | Yes |
| **Ewe** | ~3M | 2 | ~100 hrs | Yes |
| **Dagbani** | ~1.6M | 3 | ~100 hrs | Yes |
| **Ga** | ~1M | 4 | ~5 hrs | Partial |

Twi first because it has the most data and the most speakers.

## Datasets

### Twi (Akan)

| Dataset | Hours | License | Source |
|---------|-------|---------|--------|
| UG Speech Data — Akan | 100 hrs transcribed | Research use | [HCI-LAB-UGSPEECHDATA](https://github.com/HCI-LAB-UGSPEECHDATA) |
| Twi Speech-Text Multi-speaker | ~20 hrs (21,138 pairs) | CC-BY-4.0 | [HuggingFace](https://huggingface.co/datasets/ghananlpcommunity/twi-speech-text-multispeaker-16k) |
| WAXAL — Akan | ~50 hrs | CC-BY-4.0 | [HuggingFace](https://huggingface.co/datasets/google/WaxalNLP) |
| Twi Bible (single speaker) | ~3 hrs | Open | GhanaNLP |

**Total: ~170 hrs available** — more than enough for a good small model.

### Ewe

| Dataset | Hours | License | Source |
|---------|-------|---------|--------|
| UG Speech Data — Ewe | 100 hrs transcribed | Research use | [HCI-LAB-UGSPEECHDATA](https://github.com/HCI-LAB-UGSPEECHDATA) |
| WAXAL — Ewe | ~50 hrs | CC-BY-4.0 | [HuggingFace](https://huggingface.co/datasets/google/WaxalNLP) |
| Ewe ASR Model dataset | — | Open | [HCI-LAB-UGSPEECHDATA](https://github.com/HCI-LAB-UGSPEECHDATA) |

**Total: ~150 hrs available.**

### Dagbani

| Dataset | Hours | License | Source |
|---------|-------|---------|--------|
| UG Speech Data — Dagbani | 100 hrs transcribed | Research use | [HCI-LAB-UGSPEECHDATA](https://github.com/HCI-LAB-UGSPEECHDATA) |

**Total: ~100 hrs available.**

### Ga

| Dataset | Hours | License | Source |
|---------|-------|---------|--------|
| AdwumaTech mghana-st | ~5 hrs (7,800 samples) | MIT | [HuggingFace](https://huggingface.co/datasets/adwumatech-ai/mghana-st) |

**Total: ~5 hrs** — enough for a basic model, needs more data for production.

## Training pipeline (per language)

### Phase A: Lexicon construction

Each language needs a phonetic lexicon mapping words to phonemes.

**For Twi:**
1. Extract all unique words from the transcribed data
2. Build pronunciation dictionary using grapheme-to-phoneme (g2p)
3. Manual review by a Twi speaker for accuracy
4. Tools: [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus) for g2p

**For Ewe/Dagbani/Ga:** Same process with native speaker review.

### Phase B: Data preparation (Kaldi format)

```
data/lang/          # Language definition
├── words.txt       # Word → ID mapping
├── lexicon.txt     # Word → phonemes
└── L.fst           # Lexicon FST

data/train/         # Training set
├── text            # utterance_id transcription
├── wav.scp          # utterance_id audio_path
├── utt2spk          # utterance_id speaker_id
└── spk2utt          # speaker_id utterance_id_list
```

### Phase C: Model training (on Kaggle)

Run the [Vosk training recipe](https://github.com/alphacep/vosk-api/tree/master/training):

```bash
# Key settings for small/mobile models:
# - TDNN nnet3 with i-vectors
# - ivector dim: 40 (not 100, saves memory for mobile)
# - No pitch features
# - Target size: ~40 MB

cd $KALDI_ROOT/egs/your_language/s5
./run.sh --stage 0
```

### Phase D: Packaging

After training, arrange files in Vosk format:

```
sikabook-tw-v1/
├── am/
│   ├── final.mdl          # Acoustic model
│   ├── global_cmvn.stats   # CMVN stats
│   ├── ivector/
│   └── ...
├── conf/
│   ├── mfcc.conf
│   └── ...
├── Gr.fst                 # Language model
├── HCLr.fst               # Compiled graph
├── words.txt              # Vocabulary
└── README.md
```

Tar and compress:
```bash
tar czf sikabook-tw-v1.tar.gz sikabook-tw-v1/
```

### Phase E: Integration with SikaBook

In `sikabook/app.js`, add language selection:

```javascript
const MODELS = {
  "en-gh": "https://huggingface.co/your-username/sikabook-en-gh-v1/resolve/main/sikabook-en-gh-v1.tar.gz",
  "tw":    "https://huggingface.co/your-username/sikabook-tw-v1/resolve/main/sikabook-tw-v1.tar.gz",
  "ee":    "https://huggingface.co/your-username/sikabook-ee-v1/resolve/main/sikabook-ee-v1.tar.gz",
  "gaa":   "https://huggingface.co/your-username/sikabook-gaa-v1/resolve/main/sikabook-gaa-v1.tar.gz",
  "dag":   "https://huggingface.co/your-username/sikabook-dag-v1/resolve/main/sikabook-dag-v1.tar.gz"
};

// Trader picks language on first launch
let userLang = localStorage.getItem("sikabook-lang") || "en-gh";
const MODEL_URL = MODELS[userLang];
```

The parser (`parseSale`) already works on text regardless of source language
— it looks for numbers and keywords. But for Ghanaian languages, the parser
would need to handle language-specific number words and trading terms too.

## License considerations

| Dataset | License | Can use commercially? |
|---------|---------|---------------------|
| WAXAL | CC-BY-4.0 | Yes |
| Twi Speech-Text | CC-BY-4.0 | Yes |
| AdwumaTech mghana-st | MIT | Yes |
| UG Speech Data | Research use | **No** — need permission |
| GhanaNLP Parallel Corpora | CC-BY-NC-SA 4.0 | **No** — non-commercial only |

**Strategy:** Train the open-source models on CC-BY-4.0 and MIT data
(WAXAL, Twi Speech-Text, AdwumaTech). This gives ~70 hrs for Twi and
~50 hrs for Ewe — enough for a functional model. Contact University of
Ghana for a commercial license to the UG Speech Data for production quality.

## Timeline

| Phase | Duration | Dependency |
|-------|----------|------------|
| Tier 1 (language model) | Done in days | Nothing |
| Tier 2 (GH English acoustic) | 1–2 weeks | Collect 1–5 hrs audio |
| Twi model | 2–3 weeks | Kaggle sessions + lexicon |
| Ewe model | 2–3 weeks | Kaggle sessions + lexicon |
| Ga model | 1 week (small data) | Kaggle + lexicon + more data |
| Dagbani model | 2–3 weeks | Kaggle + lexicon |

Total: ~2–3 months of part-time work for all four languages, using only
free GPU resources.

## How to contribute

This is an open-source project. Contributors can help by:

1. **Recording audio** — speak Twi/Ewe/Ga/Dagbani sentences, transcribe them
2. **Building lexicons** — native speakers review phoneme mappings
3. **Donating GPU hours** — if you have a GPU, run training
4. **Testing models** — try models with different accents and dialects
5. **Improving the corpus** — add more trading sentences in any language

See the [main README](../README.md) for project structure and setup.
