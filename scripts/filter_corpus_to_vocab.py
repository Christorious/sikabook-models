#!/usr/bin/env python3
"""filter_corpus_to_vocab.py — keep only sentences the Vosk model can speak.

WHY THIS EXISTS (verified 2026-09-02): the adapted Vosk model's decoder can
only emit words that have pronunciations compiled into graph/HCLr.fst.
build_lm.sh appends new corpus words to words.txt and the grammar FST, but
they remain UNRECOGNIZABLE — the shipped model has no tree/phone files to
recompile the lexicon. Grammar probability mass spent on unreachable words
dilutes the biasing effect on words that CAN be recognized.

Strategy:
  1. HOMOPHONE SUBSTITUTION — for the highest-value OOV words that have a
     phonetically close IN-VOCAB stand-in (cedis -> cedes), rewrite the
     corpus to the stand-in. The recognizer then outputs "cedes" where the
     trader said "cedis"; the app maps it back textually using
     data/nlu/asr_postprocess_map.json.
  2. SENTENCE FILTER — sentences still containing OOV words are dropped
     (their LM mass would be wasted). The full OOV list is written to
     data/corpus/oov_words.txt; those words need the zipformer track
     (asr/zipformer) or a full Vosk graph rebuild.

Usage:
  python3 scripts/filter_corpus_to_vocab.py
  python3 scripts/filter_corpus_to_vocab.py \
      --words models/vosk-model-small-en-us-0.15/graph/words.txt
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_WORDS = os.path.join(
    PROJECT_ROOT, "models", "vosk-model-small-en-us-0.15", "graph", "words.txt")
DEFAULT_INPUT = os.path.join(
    PROJECT_ROOT, "data", "corpus", "trading_corpus.clean.txt")
DEFAULT_OUTPUT = os.path.join(
    PROJECT_ROOT, "data", "corpus", "trading_corpus.vocabfiltered.txt")
POSTPROCESS_MAP = os.path.join(
    PROJECT_ROOT, "data", "nlu", "asr_postprocess_map.json")

# OOV word -> phonetically close word that IS in the base compiled lexicon.
# Only safe near-homophones belong here. "cedis" /"si:dIs/ vs "cedes"
# /si:dz/ differ only in final voicing, which ASR merges in practice.
# Every entry here is mirrored into asr_postprocess_map.json at runtime.
HOMOPHONES = {
    "cedis": "cedes",
    "cedi": "cedes",
}

# map persisted for the app/pipeline to undo the substitution
if os.path.exists(POSTPROCESS_MAP):
    with open(POSTPROCESS_MAP, encoding="utf-8") as f:
        _pp = json.load(f)
else:
    _pp = {}
for _oov, _stand in sorted(HOMOPHONES.items()):
    _pp[_stand] = _oov  # ascending sort: the longest/most frequent OOV wins
os.makedirs(os.path.dirname(POSTPROCESS_MAP), exist_ok=True)
with open(POSTPROCESS_MAP, "w", encoding="utf-8") as f:
    json.dump(_pp, f, indent=2, ensure_ascii=False)


def load_vocab(path: str) -> set[str]:
    vocab = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                vocab.add(parts[0])
    return vocab


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", default=DEFAULT_WORDS,
                        help="base-model words.txt (compiled lexicon vocabulary)")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not os.path.exists(args.words):
        print(f"ERROR: base words.txt not found at {args.words}.\n"
              f"  build_lm.sh generates it from the base Gr.fst symbol table;\n"
              f"  or run: fstsymbols --save_osymbols={args.words} "
              f"models/vosk-model-small-en-us-0.15/graph/Gr.fst",
              file=sys.stderr)
        return 1
    vocab = load_vocab(args.words)
    print(f"Recognizable vocabulary: {len(vocab)} words")

    # substitution applies only when the stand-in is truly in-vocab
    subs = {oov: s for oov, s in HOMOPHONES.items() if s in vocab}
    if subs:
        print(f"Homophone substitutions (mirrored to {POSTPROCESS_MAP}):")
        for oov, s in sorted(subs.items()):
            print(f"  {oov} -> {s}")

    kept, dropped, oov_words = [], [], set()
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words = [subs.get(w, w) for w in line.split()]
            missing = [w for w in words if w not in vocab]
            if missing:
                dropped.append(line)
                oov_words.update(missing)
            else:
                kept.append(" ".join(words))

    with open(args.output, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")

    print(f"Kept {len(kept)} sentences -> {args.output}")
    print(f"Dropped {len(dropped)} sentences containing {len(oov_words)} "
          f"unrecognizable words")
    if oov_words:
        sample = ", ".join(sorted(oov_words)[:15])
        print(f"  dropped vocabulary (no pronunciations in HCLr.fst): {sample}...")
        report = os.path.join(PROJECT_ROOT, "data", "corpus", "oov_words.txt")
        with open(report, "w", encoding="utf-8") as f:
            for w in sorted(oov_words):
                f.write(w + "\n")
        print(f"  full list: {report}")
        print("  These words need the zipformer track or a full graph rebuild.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
