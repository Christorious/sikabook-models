# ASR Track A — Vosk quick-start baseline (English + Pidgin)

Status: **built** (`models/sikabook-en-gh-v1/`, ~30 MB tarball) — Tier-1
language-model adaptation of `vosk-model-small-en-us-0.15`.

This is the fastest path to a working offline recognizer for Ghanaian
English + Pidgin. It is explicitly a baseline: the target model is the
fine-tuned streaming zipformer (see `asr/zipformer/`).

## What the pipeline does

```
scripts/clean_corpus.py            merge + normalize all corpus files
scripts/build_lm.sh                1. copy base model
                                   2. regenerate base words.txt from Gr.fst
                                   3. filter corpus to recognizable vocabulary
                                      (scripts/filter_corpus_to_vocab.py)
                                   4. rebuild Gr.fst n-gram grammar
```

## The lexicon limitation (read this before shipping)

**Verified 2026-09-02** by projecting the compiled `graph/HCLr.fst` onto its
output symbols: words appended to `words.txt` by the grammar build (cedis,
pesewas, kenkey, waakye, Twi numerals, ...) have **no pronunciations** in
the compiled lexicon graph and can never be hypothesized by the decoder.
The shipped model contains no decision-tree or phone files, so the lexicon
cannot be recompiled — a full model rebuild is required to add words.

Consequences, handled in the pipeline:

1. **Grammar mass is not wasted on unreachable words.**
   `filter_corpus_to_vocab.py` drops sentences containing OOV words and
   reports them to `data/corpus/oov_words.txt`.
2. **Money words survive via homophone substitution.** `cedis` /siːdɪs/ is
   rewritten to `cedes` (in-vocab, /siːdz/ — ASR merges final voicing), so
   every price sentence still biases the LM. The recognizer will emit
   `cedes`; the pipeline maps it back with
   `data/nlu/asr_postprocess_map.json` (`cedes -> cedis`).
   **The app MUST apply this map to transcripts before NLU.**
3. Everything else Ghanaian (Twi numerals, Pidgin function words, product
   names like apateshi/kpanla) is listed in `oov_words.txt` and is the
   concrete motivation for the zipformer track.

## Base model vocabulary facts

- Base compiled vocabulary: **152,216 words** (regenerated into
  `models/vosk-model-small-en-us-0.15/graph/words.txt` by the build).
- Recognizable today: tilapia, mackerel, sardines, polythene, sachet,
  numbers, general English. Unreachable: cedis, pesewas, kenkey, waakye,
  Ghanaian names, Twi words (see oov_words.txt).

## Android packaging

Vosk models are directories; Android loads them from assets or downloaded
storage. `com.alphacephei:vosk-android:0.3.70` (Maven Central):

```kotlin
val model = Model("$filesDir/sikabook-en-gh-v1")
val rec = KaldiRecognizer(model, 16000.0f)
// streaming: rec.AcceptWaveform(buffer) + PartialResult while speech is ongoing
```

- Untar `sikabook-en-gh-v1.tar.gz` into app-private storage on first run
  (assets are compressed inside the APK; Vosk needs a real directory).
- ~300 MB RAM while decoding — acceptable on 3–4 GB devices but the
  zipformer target needs less.
- `export/package_android.sh` produces the versioned bundle.

## When to stop using this model

Switch to the zipformer model when (a) Twi/Fante/Ga WER matters, or (b)
"cedis"-level market vocabulary must be recognized natively. Both arrive
with the Ashesi fine-tune.
