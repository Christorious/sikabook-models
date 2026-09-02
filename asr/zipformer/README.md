# ASR Track B — Fine-tuned streaming zipformer (the target model)

Goal: one streaming transducer model covering **Ghanaian English, Pidgin,
Asante/Akuapem Twi and Fante + Ga**, int8-quantized, 40–80 MB, RTF ≪ 1 on
Cortex-A55, served by sherpa-onnx on Android.

## Why this track exists (vs the Vosk baseline)

- We own the lexicon as a plain text artifact — `cedis`, `pesewas`,
  `apateshi`, Twi numerals become first-class words (the Vosk baseline
  cannot recognize any of them; see `asr/vosk/README.md`).
- Modern transducer > Kaldi TDNN on accented/noisy audio.
- sherpa-onnx gives us VAD, speaker ID, streaming ASR from one maintained
  Android AAR (`com.k2-fsa:sherpa-onnx`).

## Recipe overview

| Step | What | Where |
|---|---|---|
| 1 | Prepare Ashesi data (CC-BY-4.0 core): `data/datasets/ashesi/` | local/Kaggle |
| 2 | Add English market data (trading corpus text → TTS-free: use existing LibriSpeech/GigaSpeech subsets already in the base recipe + our corpus for LM/rescoring) | Kaggle |
| 3 | Fine-tune `pruned_transducer_stateless7_streaming` (icefall) starting from the released English streaming zipformer | Kaggle T4/P100 |
| 4 | Export ONNX, int8-quantize | Kaggle |
| 5 | Eval gates (below) | local + Kaggle |
| 6 | Package via sherpa-onnx | `export/` |

## Base checkpoint

sherpa-onnx releases English streaming zipformers trained in icefall
(e.g. `sherpa-onnx-streaming-zipformer-en-2023-06-26`, from
`egs/librispeech/ASR/pruned_transducer_stateless7_streaming`). Fine-tuning
from its icefall checkpoint keeps tokenizer compatibility: copy its
`tokens.txt` and **extend** it before training:

- Add single characters: `ɛ ɔ ŋ ɔ` variants and any Ga/Fante letters
  missing from the 500-unit BPE set, plus digits if absent.
- Do NOT retrain the tokenizer — append units, pad/warm the output
  embeddings, then train. (icefall: adjust `--num-classes` or use the
  `finetune` flags documented in `egs/*/ASR/...` README.)
- Alternative if embedding surgery is inconvenient: map ɛ→e, ɔ→o in
  transcripts (ASCII mode of `prepare_ashesi.py --normalize-ascii`) —
  costs some WER, saves the surgery. Start here for the first run.

## Data mix (first run)

| Source | Hours | License |
|---|---|---|
| Ashesi Asante+Akuapem+Fante+Ga | ~148 h | CC-BY-4.0 (verify in-archive) |
| FLEURS aka_gh | ~10 h | CC-BY-4.0 |
| WAXAL (Ghana/Togo langs) | subset | CC-BY-4.0 |
| Twi Speech-Text multispeaker | ~text pairs → LM/rescore | CC-BY-4.0 |

Held-out eval: Ashesi dev speakers (5%, speaker-disjoint, made by
`prepare_ashesi.py`) **plus** a small self-recorded market test set —
read-speech WER always flatters the model; the market test set is the real
gate. Target: ≤35% WER on in-domain market audio for v1 (this is hard;
far-field, noise, two speakers), ≤15% on Ashesi dev.

## Fine-tuning specifics (icefall `pruned_transducer_stateless7_streaming`)

1. Convert manifests to icefall format (`cuts` via lhotse): the Kaggle
   notebook does this from `manifest.jsonl`.
2. Freeze nothing on the first pass; LR 1e-4 cosine, 8–12 epochs on a
   single T4 is usually enough for the accent to land.
3. Concatenate languages per batch (shuffle all four); keep a 10%
   English replay mix (any English subset already prepared in the recipe)
   so the base model's English doesn't drift.
4. SpecAugment as in recipe; add MUSAN noise at moderate probability —
   market noise robustness is the product requirement.
5. Decode with the base model's BPE + modified tokens; rescore with an
   n-gram LM built from `data/corpus/*.txt` (our market corpus).

## Export + Android

```bash
# in icefall recipe dir
python pruned_transducer_stateless7_streaming/export-onnx.py \
  --exp-dir exp/finetuned --epoch N --avg M --streaming True
# int8 quantize (sherpa-onnx tooling)
python -m sherpa_onnx.quantize ... # or onnxruntime quantize_static
```

Ship: `encoder-epochN-avgM-int8.onnx`, `decoder-...onnx`,
`joiner-...onnx`, `tokens.txt` (see `export/package_android.sh`).

Android (sherpa-onnx AAR):

```kotlin
val config = OnlineRecognizerConfig(
  modelConfig = OnlineModelConfig(
    transducer = OnlineTransducerModelConfig(
      encoder = ".../encoder-int8.onnx",
      decoder = ".../decoder-int8.onnx",
      joiner = ".../joiner-int8.onnx"),
    tokens = ".../tokens.txt"),
)
```

## Eval gates to replace the Vosk baseline

Run through `eval/` (WER per language + end-to-end sale metrics):

1. Ashesi dev WER per language ≤ 15%.
2. Market test set WER ≤ 35% AND sale-event F1 from the full pipeline
   (ASR → rules NLU) beats the Vosk baseline by ≥ 10 points absolute.
3. RTF ≤ 0.3 int8, 2 threads, on the reference device (Spark-class A55).
4. RAM < 300 MB steady-state during streaming.

Only after all four: flip the app's default engine flag.
