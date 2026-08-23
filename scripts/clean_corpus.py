#!/usr/bin/env python3
"""
clean_corpus.py — Prepare the trading corpus for language model building.

Reads data/corpus/trading_corpus.txt, normalizes text, and outputs
the clean format that build_lm.sh expects (one sentence per line,
lowercase, no punctuation, one space between words).

Usage:
    python3 scripts/clean_corpus.py
"""

import os
import re
import sys

# Paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "corpus", "trading_corpus.txt")
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
    if not os.path.exists(INPUT_PATH):
        print("ERROR: Corpus not found at", INPUT_PATH)
        sys.exit(1)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clean_lines = []
    for line in lines:
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
    print("  Input:   {} ({} lines)".format(INPUT_PATH, len(lines)))
    print("  Output:  {} ({} unique lines)".format(OUTPUT_PATH, len(unique_lines)))

    # Print vocabulary stats
    vocab = set()
    for line in unique_lines:
        vocab.update(line.split())
    print("  Vocabulary: {} unique words".format(len(vocab)))


if __name__ == "__main__":
    main()
