# SikaBook Models — On-Device Speech + Sale-Extraction Models for Ghana

Offline models for a real-time voice agent that listens to customer–trader
conversations in Ghanaian markets, detects sales, and extracts item,
quantity, price (GHS) and time — **entirely on the phone**.

Companion repo: [ghana-voice-ledger](https://github.com/Christorious/ghana-voice-ledger)
(the Android app). The interface between the two is the
[SaleEvent contract](export/SALE_EVENT_CONTRACT.md).

## Device target (researched, Sept 2026)

| Tier | Device | RAM | Stack |
|---|---|---|---|
| Floor | Tecno Pop / itel A-class | 3 GB | VAD + streaming ASR + rules |
| **Typical** | **Tecno Spark 30C/40 (Helio G81)** | **4 GB** | VAD + zipformer ASR + rules + MiniLM NLU |
| Upgrade | 6 GB+ / used flagships | 6–8 GB | + Qwen3-0.6B normalization (llama.cpp, GBNF) |

CPU-only inference (int8) — NPUs on Unisoc/Helio are unusable by third-party
apps. "8GB" on Tecno spec sheets is 4 GB + virtual RAM; do not design for it.

## The three model stages

```
mic foreground service → Silero VAD (sherpa-onnx)
  → streaming zipformer ASR  (target)  / adapted Vosk (baseline, built)
  → speaker embeddings (WeSpeaker) → seller-vs-customer
  → SaleEvent NLU: rules (always) → MiniLM intent+NER (always)
     → Qwen3-0.6B normalization (6 GB+ only, post-utterance)
  → SaleEvent JSON → app confirm UI → Room `transactions`
```

| Stage | Model | Size | Status |
|---|---|---|---|
| VAD | Silero v5 (via sherpa-onnx) | 1.7 MB | off-the-shelf |
| ASR baseline | vosk-model-small-en-us + market LM | ~41 MB | **built** (`models/sikabook-en-gh-v1`) |
| ASR target | streaming zipformer, fine-tuned on Ashesi CC-BY data | 40–80 MB int8 | recipe ready (`asr/zipformer/`) |
| NLU rules | this repo, Python reference (port of app's GhanaEntityExtractor + Twi/Ga numerals) | — | **built + tested** |
| NLU BERT | MiniLM intent+NER, ONNX int8 | ~25 MB | training script ready |
| NLU LLM | Qwen3-0.6B Q4 + sale_event.gbnf | ~0.4 GB | optional tier, gated |

## Repo layout

```
data/
  corpus/            trading corpus: EN + Pidgin + Twi + Ga (sentence files)
  lexicon/           ARPABET lexicons (incl. Ghanaian words the base model lacks)
  datasets/ashesi/   download + prep for the 148h CC-BY-4.0 Ashesi dataset
  nlu/               product vocabulary, synthetic generator, labeling guide,
                     asr_postprocess_map.json (cedes->cedis)
asr/
  vosk/              baseline track: docs + limitation notes
  zipformer/         target track: fine-tune recipe + Kaggle notebook
nlu/
  rules/             sika_rules package: deterministic SaleEvent extractor + tests
  bert/              MiniLM intent+NER fine-tune + ONNX export script
  llm/               Qwen3 GBNF grammar + gating notes
eval/                sikabook_eval: WER, sale P/R/F1, slot F1, amount accuracy
export/              SaleEvent JSON schema + contract, package_android.sh
docs/                license matrix, tiered roadmap
```

## Quick start

```bash
# 1. NLU rules (no dependencies, 45 tests)
python3 -m unittest discover -s nlu/rules/tests
python3 -m unittest discover -s eval/tests
python3 nlu/rules/extract.py "i sold two tilapia at ten cedis each"

# 2. Vosk baseline model (needs OpenFST/OpenGrM; ~10 min)
./scripts/build_lm.sh        # clean -> filter -> rebuild Gr.fst -> package

# 3. Synthetic NLU data + rules baseline metrics
python3 data/nlu/synthetic_generator.py --n 3000

# 4. Ashesi dataset (license check first: docs/LICENSE-MATRIX.md)
./data/datasets/ashesi/download_ashesi.sh
python3 data/datasets/ashesi/prepare_ashesi.py --max-hours 2   # smoke

# 5. Package a model bundle for the app
./export/package_android.sh
```

## Status (2026-09-02)

| Milestone | State |
|---|---|
| M0 schema/contract/eval/rules | **done** — 45 tests green |
| M1 Vosk baseline | **done with caveat** — grammar rebuilt on recognizable vocab only; Ghana-specific words need the zipformer (Vosk lexicon is compiled-in; see `asr/vosk/README.md`) |
| M2 zipformer + Ashesi | scripts/recipe/notebook ready; **license file inside archives must be verified**, then Kaggle run |
| M3 NLU v1 | synthetic data + rules baseline done (sale F1 1.0, amount exact 98% on synthetic; real recordings will be lower); BERT script ready |
| M4 LLM tier | grammar + gating doc ready; Go/No-Go after on-device latency test |
| M5 integration + field | blocked on app's compile-fix PR (#43); sherpa-onnx AAR plan documented |

## Key findings baked into this design

- The base Vosk model's compiled lexicon cannot be extended without a full
  rebuild — appended words (cedis, pesewas, ...) are unrecognizable. Money
  words survive via the `cedes` homophone + postprocess map; everything
  else is the zipformer track's job.
- Ashesi/Nokwary Financial Inclusion Speech Dataset (CC-BY-4.0, ~148 h,
  Asante/Akuapem Twi, Fante, Ga) is the only confirmed commercially usable
  finance-domain Ghanaian speech corpus. KasaSpeech and UGSpeechData need
  written permission. MMS is license-blocked.
- "PhoneLLM" is either a research artifact (PhoneLM, mllm engine only) or
  a 30B datacenter telephony model (pipecat) — neither fits; Qwen3 does.

## License

Apache 2.0 — same as Vosk and Kaldi. Model releases list their training
data attribution per `docs/LICENSE-MATRIX.md`.
