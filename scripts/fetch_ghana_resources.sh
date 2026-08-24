#!/usr/bin/env bash
# Fetch Ghana-focused repositories into external/ for inspection.
# WARNING: This script clones repositories for local inspection only. DO NOT copy files into this repo
# until you have checked the LICENSE in each cloned repo.
set -euo pipefail
mkdir -p external
cd external

repos=(
  "https://github.com/GhanaNLP/ghanaian-nlp-datasets-models.git"
  "https://github.com/GhanaNLP/ghana-corpus-builder.git"
  "https://github.com/GhanaNLP/ABENA.git"
  "https://github.com/GhanaNLP/GhanaNouns.git"
  "https://github.com/nyarderr/okyeame-tts.git"
  "https://github.com/iamEtornam/GhanaNLP-Dart.git"
  "https://github.com/JosephAppiah-c/Ghanaian-cultural-lexicon.git"
  "https://github.com/Ahmed01ttfret/Ghana-Pigin.git"
)

for r in "${repos[@]}"; do
  name=$(basename -s .git "$r")
  if [ -d "$name" ]; then
    echo "Already cloned: $name"
    continue
  fi
  echo "Cloning $r"
  git clone --depth 1 "$r"
done

echo "Done. Inspect LICENSE files in external/* before reusing any data/code." 
