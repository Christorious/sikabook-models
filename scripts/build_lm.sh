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

# --- Step 2: Locate Kaldi tools ---
echo ""
echo "=========================================="
echo " Step 2: Locate Kaldi/OpenFST tools"
echo "=========================================="

# Try common Kaldi locations
KALDI_ROOT=""
for candidate in \
    "$PROJECT_ROOT/kaldi" \
    "$HOME/kaldi" \
    "/opt/kaldi" \
    "/usr/local/kaldi"; do
    if [ -d "$candidate/tools/openfst/bin" ]; then
        KALDI_ROOT="$candidate"
        break
    fi
done

if [ -z "$KALDI_ROOT" ]; then
    echo "ERROR: Kaldi not found."
    echo ""
    echo "Please install Kaldi first. Quick install:"
    echo "  git clone https://github.com/kaldi-asr/kaldi.git $PROJECT_ROOT/kaldi"
    echo "  cd $PROJECT_ROOT/kaldi/tools && make"
    echo "  extras/install_opengrm.sh"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

export PATH="$KALDI_ROOT/tools/openfst/bin:$PATH"
export LD_LIBRARY_PATH="$KALDI_ROOT/tools/openfst/lib/fst:${LD_LIBRARY_PATH:-}"

echo "Using Kaldi at: $KALDI_ROOT"

# --- Step 3: Build the language model ---
echo ""
echo "=========================================="
echo " Step 3: Build language model (Gr.fst)"
echo "=========================================="

MODEL_DIR="$OUTPUT_DIR"
WORDS_FILE="$MODEL_DIR/words.txt"
OLD_GR="$MODEL_DIR/Gr.fst"
TEXT_FILE="$CORPUS_CLEAN"

if [ ! -f "$OLD_GR" ]; then
    echo "ERROR: Gr.fst not found in $MODEL_DIR"
    exit 1
fi

echo "Building n-gram language model from corpus..."
echo "  Corpus: $TEXT_FILE"

# Extract symbol table from existing model
fstsymbols --save_osymbols="$WORDS_FILE" "$OLD_GR" > /dev/null

# Compile corpus into FST archives, count n-grams, and build new grammar
farcompilestrings \
    --fst_type=compact \
    --symbols="$WORDS_FILE" \
    --keep_symbols \
    "$TEXT_FILE" | \
ngramcount | \
ngrammake | \
fstconvert --fst_type=ngram > "$MODEL_DIR/Gr.new.fst"

# Replace the old grammar
mv "$MODEL_DIR/Gr.new.fst" "$MODEL_DIR/Gr.fst"

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
