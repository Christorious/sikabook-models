#!/usr/bin/env python3
"""
clean_corpus.py — Prepare the trading corpus for language model building.

Reads data/corpus/trading_corpus.txt, normalizes text, and outputs
the clean format that build_lm.sh expects (one sentence per line,
lowercase, no punctuation, one space between words).

Usage:
    python3 scripts/clean_corpus.py
"""

import glob
import os
import re
import sys

# Paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CORPUS_DIR = os.path.join(PROJECT_ROOT, "data", "corpus")
# All trading corpus files (main + language additions); generated files excluded
INPUT_GLOB = os.path.join(CORPUS_DIR, "trading_corpus*.txt")
EXCLUDE = ("clean", "filtered")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "corpus", "trading_corpus.clean.txt")


def clean_line(line):
    """Normalize a single line of text."""
    line = line.strip().lower()

    # Remove punctuation except apostrophes in contractions
    line = re.sub(r"[^\w\s']", " ", line)

    # Collapse multiple spaces
    line = re.sub(r"\s+", " ", line)

    # Strip leading/trailing whitespace
    line = line.strip()

    return line


def main():
    input_paths = sorted(
        p for p in glob.glob(INPUT_GLOB)
        if not any(tag in os.path.basename(p) for tag in EXCLUDE)
    )
    if not input_paths:
        print("ERROR: No corpus files found at", INPUT_GLOB)
        sys.exit(1)

    clean_lines = []
    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.lstrip().startswith("#"):
                    continue  # comment headers in language corpus files
                cleaned = clean_line(line)
                if cleaned:
                    clean_lines.append(cleaned)

    # Deduplicate while preserving order
    seen = set()
    unique_lines = []
    for line in clean_lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for line in unique_lines:
            f.write(line + "\n")

    print("Corpus cleaned successfully.")
    print("  Inputs:  {} files".format(len(input_paths)))
    for p in input_paths:
        print("    -", os.path.relpath(p, PROJECT_ROOT))
    print("  Output:  {} ({} unique lines)".format(OUTPUT_PATH, len(unique_lines)))

    # Print vocabulary stats
    vocab = set()
    for line in unique_lines:
        vocab.update(line.split())
    print("  Vocabulary: {} unique words".format(len(vocab)))


if __name__ == "__main__":
    main()
