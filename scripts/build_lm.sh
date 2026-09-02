#!/usr/bin/env bash
# ===================================================================
# build_lm.sh — Tier 1: Build a domain-specific language model (Gr.fst)
#               for the SikaBook Vosk speech model.
#
# This runs on your laptop — NO GPU needed. It rebuilds the language
# model (grammar) inside the existing small English Vosk model so it
# is biased toward market-trading vocabulary.
#
# Prerequisites:
#   - Kaldi compiled with OpenFST and OpenGrM
#   - The base Vosk model (vosk-model-small-en-us-0.15) downloaded
#
# See docs/TIER1-SETUP.md for full instructions.
#
# Usage:
#   ./scripts/build_lm.sh
# ===================================================================

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CORPUS_RAW="$PROJECT_ROOT/data/corpus/trading_corpus.txt"
CORPUS_CLEAN="$PROJECT_ROOT/data/corpus/trading_corpus.clean.txt"
OUTPUT_DIR="$PROJECT_ROOT/models/sikabook-en-gh-v1"
BASE_MODEL_NAME="vosk-model-small-en-us-0.15"
BASE_MODEL_TAR="vosk-model-small-en-us-0.15.tar.gz"
BASE_MODEL_URL="https://alphacephei.com/vosk/models/${BASE_MODEL_TAR}"

# --- Step 0: Clean the corpus ---
echo "=========================================="
echo " Step 0: Cleaning corpus"
echo "=========================================="
python3 "$SCRIPT_DIR/clean_corpus.py"

# --- Step 1: Download base model if not present ---
echo ""
echo "=========================================="
echo " Step 1: Download base Vosk model"
echo "=========================================="
if [ ! -d "$PROJECT_ROOT/models/$BASE_MODEL_NAME" ]; then
    echo "Base model not found. Downloading..."
    cd "$PROJECT_ROOT/models"
    if [ ! -f "$BASE_MODEL_TAR" ]; then
        wget -q --show-progress "$BASE_MODEL_URL"
    fi
    tar xzf "$BASE_MODEL_TAR"
    rm -f "$BASE_MODEL_TAR"
    echo "Downloaded and extracted to models/$BASE_MODEL_NAME"
else
    echo "Base model already exists at models/$BASE_MODEL_NAME"
fi

# Copy base model to our output directory
echo "Copying base model to $OUTPUT_DIR..."
rm -rf "$OUTPUT_DIR"
cp -r "$PROJECT_ROOT/models/$BASE_MODEL_NAME" "$OUTPUT_DIR"

# --- Step 2: Locate OpenFST/OpenGrM tools ---
echo ""
echo "=========================================="
echo " Step 2: Locate OpenFST/OpenGrM tools"
echo "=========================================="

# Try local install first (built without Kaldi), then Kaldi locations
OPENFST_BIN=""
for candidate in \
    "$HOME/local/bin" \
    "$PROJECT_ROOT/kaldi/tools/openfst/bin" \
    "$HOME/kaldi/tools/openfst/bin" \
    "/opt/kaldi/tools/openfst/bin" \
    "/usr/local/kaldi/tools/openfst/bin"; do
    if [ -x "$candidate/fstsymbols" ]; then
        OPENFST_BIN="$candidate"
        break
    fi
done

if [ -z "$OPENFST_BIN" ]; then
    echo "ERROR: OpenFST/OpenGrM tools not found."
    echo ""
    echo "Please install OpenFST + OpenGrM first. Quick install (no sudo):"
    echo "  mkdir -p ~/local/src && cd ~/local/src"
    echo "  wget https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.2.tar.gz"
    echo "  tar xzf openfst-1.8.2.tar.gz && cd openfst-1.8.2"
    echo "  ./configure --enable-grm --prefix=\$HOME/local && make -j4 && make install"
    echo "  cd .. && wget https://www.opengrm.org/twiki/pub/GRM/NGramDownload/ngram-1.3.15.tar.gz"
    echo "  tar xzf ngram-1.3.15.tar.gz && cd ngram-1.3.15"
    echo "  CXXFLAGS=\"-I\$HOME/local/include\" LDFLAGS=\"-L\$HOME/local/lib\" \\"
    echo "    ./configure --prefix=\$HOME/local && make -j4 && make install"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

OPENFST_PREFIX="$(dirname "$OPENFST_BIN")"
export PATH="$OPENFST_BIN:$PATH"
export LD_LIBRARY_PATH="$OPENFST_PREFIX/lib:$OPENFST_PREFIX/lib/fst:${LD_LIBRARY_PATH:-}"

# Preload the ngram FST extension so OpenFST tools can read/write ngram FST type
NGRAM_FST_SO="$OPENFST_PREFIX/lib/fst/ngram-fst.so"
if [ -f "$NGRAM_FST_SO" ]; then
    export LD_PRELOAD="$NGRAM_FST_SO:${LD_PRELOAD:-}"
fi

echo "Using OpenFST/OpenGrM at: $OPENFST_BIN"

# --- Step 2b: materialize base-model vocabulary ---
# The shipped model has no words.txt; fstsymbols regenerates it from the
# base grammar's symbol table. Needed by filter_corpus_to_vocab.py.
BASE_WORDS="$PROJECT_ROOT/models/$BASE_MODEL_NAME/graph/words.txt"
BASE_GR="$PROJECT_ROOT/models/$BASE_MODEL_NAME/graph/Gr.fst"
if [ ! -f "$BASE_WORDS" ] && [ -f "$BASE_GR" ]; then
    echo "Generating base-model words.txt from Gr.fst symbol table..."
    fstsymbols --save_osymbols="$BASE_WORDS" "$BASE_GR" > /dev/null
fi

# --- Step 2c: filter to recognizable vocabulary ---
# Words without pronunciations in HCLr.fst can never be recognized by the
# decoder; keeping their sentences in the grammar only dilutes LM mass.
echo ""
echo "=========================================="
echo " Step 2c: Filtering corpus to recognizable vocabulary"
echo "=========================================="
python3 "$SCRIPT_DIR/filter_corpus_to_vocab.py" || true

# --- Step 3: Build the language model ---
echo ""
echo "=========================================="
echo " Step 3: Build language model (Gr.fst)"
echo "=========================================="

MODEL_DIR="$OUTPUT_DIR"
WORDS_FILE="$MODEL_DIR/graph/words.txt"
OLD_GR="$MODEL_DIR/graph/Gr.fst"
# Prefer the vocabulary-filtered corpus when present (see
# filter_corpus_to_vocab.py): grammar mass goes only to recognizable words.
if [ -f "$PROJECT_ROOT/data/corpus/trading_corpus.vocabfiltered.txt" ]; then
    TEXT_FILE="$PROJECT_ROOT/data/corpus/trading_corpus.vocabfiltered.txt"
else
    TEXT_FILE="$CORPUS_CLEAN"
fi

# Some Vosk models put Gr.fst in the root, others in graph/
if [ ! -f "$OLD_GR" ]; then
    OLD_GR="$MODEL_DIR/Gr.fst"
    WORDS_FILE="$MODEL_DIR/words.txt"
fi

if [ ! -f "$OLD_GR" ]; then
    echo "ERROR: Gr.fst not found in $MODEL_DIR or $MODEL_DIR/graph/"
    exit 1
fi

echo "Building n-gram language model from corpus..."
echo "  Corpus: $TEXT_FILE"

# Extract symbol table from existing model
fstsymbols --save_osymbols="$WORDS_FILE" "$OLD_GR" > /dev/null

# Directory where Gr.fst lives (graph/ for newer Vosk models)
GR_DIR="$(dirname "$OLD_GR")"

# Add new vocabulary words from corpus to the symbol table
# The base model doesn't know words like "cedis", "pesewas", "sachet"
python3 - "$WORDS_FILE" "$TEXT_FILE" <<'PYEOF'
import sys, re

words_file = sys.argv[1]
corpus_file = sys.argv[2]

# Read existing symbol table (format: word integer_id)
existing = {}
max_id = 0
with open(words_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            word, wid = parts[0], int(parts[1])
            existing[word] = wid
            if wid > max_id:
                max_id = wid

# Extract unique words from corpus
corpus_words = set()
with open(corpus_file, 'r', encoding='utf-8') as f:
    for line in f:
        corpus_words.update(line.strip().split())

# Find new words not in symbol table
new_words = sorted(w for w in corpus_words if w not in existing)
if new_words:
    with open(words_file, 'a', encoding='utf-8') as f:
        for w in new_words:
            max_id += 1
            f.write(f"{w} {max_id}\n")
    print(f"  Added {len(new_words)} new words to vocabulary: {', '.join(new_words[:10])}{'...' if len(new_words) > 10 else ''}")
else:
    print("  All corpus words already in vocabulary")
PYEOF

# Compile corpus into FST archives, count n-grams, and build new grammar
farcompilestrings \
    --fst_type=compact \
    --symbols="$WORDS_FILE" \
    --keep_symbols \
    "$TEXT_FILE" | \
ngramcount | \
ngrammake > "$GR_DIR/Gr.new.fst"

# Replace the old grammar
mv "$GR_DIR/Gr.new.fst" "$OLD_GR"

echo ""
echo "=========================================="
echo " DONE!"
echo "=========================================="
echo "New language model written to:"
echo "  $MODEL_DIR/Gr.fst"
echo ""
echo "The adapted model is at:"
echo "  $OUTPUT_DIR"
echo ""
echo "To use it in SikaBook, point MODEL_URL to this model directory"
echo "(or host it and update app.js)."
