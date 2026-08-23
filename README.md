# SikaBook Models — Open-Source Vosk Speech Models for Ghana

Offline speech recognition models tuned for Ghanaian market traders.
Built on [Vosk](https://alphacephei.com/vosk/) / [Kaldi](https://github.com/kaldi-asr/kaldi), released under Apache 2.0.

## Why this exists

[SikaBook](../sikabook) is a voice-powered bookkeeping app for Ghanaian traders.
The default Vosk English model is trained on American speech and doesn't know
market vocabulary (polythene bags, sachet water, cedis, pesewas) or Ghanaian
names (Ama, Kofi, Adwoa, Kwame). This project fixes that.

## The three tiers

| Tier | What | Where it runs | GPU needed | Effort |
|------|------|---------------|------------|--------|
| **1** | Language model adaptation (vocabulary + grammar) | Your laptop | No | Days |
| **2** | Acoustic model fine-tune (Ghanaian English accent) | Kaggle (free GPU) | Yes | Weeks |
| **3** | New models for Twi, Ewe, Ga, Dagbani | Kaggle + Colab (free GPU) | Yes | Months |

## Quick start — Tier 1 (runs on your laptop today)

```bash
# 1. Install Kaldi (one-time, ~30 min compile)
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi/tools && make
extras/install_opengrm.sh

# 2. Clean the corpus
python3 scripts/clean_corpus.py

# 3. Build the adapted language model
./scripts/build_lm.sh
```

See [docs/TIER1-SETUP.md](docs/TIER1-SETUP.md) for detailed instructions.

## Project structure

```
sikabook-models/
├── data/
│   ├── corpus/
│   │   └── trading_corpus.txt       # ~300 trader sentences (SikaBook vocabulary)
│   └── lexicon/
│       └── sikabook_lexicon.txt     # Phonetic dictionary (starter)
├── scripts/
│   ├── clean_corpus.py              # Normalizes corpus text
│   └── build_lm.sh                  # Builds Gr.fst language model (Tier 1)
├── models/                          # Output models land here
├── docs/
│   ├── TIER1-SETUP.md               # Local setup guide
│   ├── TIER2-3-KAGGLE.md            # Free GPU training on Kaggle
│   └── TIER3-GHANAIAN-LANGUAGES.md  # Twi/Ewe/Ga/Dagbani model plans
└── LICENSE                           # Apache 2.0
```

## Datasets we build on

| Dataset | Languages | Size | License |
|---------|-----------|------|---------|
| [UG Speech Data](https://github.com/HCI-LAB-UGSPEECHDATA) | Akan, Ewe, Dagbani, Dagaare, Ikposo | 5,000 hrs (100h transcribed/lang) | Research use |
| [WAXAL](https://huggingface.co/datasets/google/WaxalNLP) | Akan, Ewe + 22 African languages | 1,250 hrs ASR | CC-BY-4.0 |
| [Twi Speech-Text](https://huggingface.co/datasets/ghananlpcommunity/twi-speech-text-multispeaker-16k) | Twi (Akan) | 21,138 pairs | CC-BY-4.0 |
| [AdwumaTech mghana-st](https://huggingface.co/datasets/adwumatech-ai/mghana-st) | Twi, Ewe, Ga | 7,800+ samples | MIT |
| [GhanaNLP Parallel Corpora](https://huggingface.co/Ghana-NLP) | Twi, Fante, Ewe, Ga, Kusaal | 41,513 sentence pairs | CC-BY-NC-SA 4.0 |

## License

Apache 2.0 — same as Vosk and Kaldi. See [LICENSE](LICENSE).

Dataset attributions are documented per-dataset; some datasets (UG Speech
Data) restrict commercial use. Models trained on CC-BY-4.0 data (WAXAL,
Twi Speech-Text) can be used commercially.
