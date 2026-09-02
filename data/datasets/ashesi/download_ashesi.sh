#!/usr/bin/env bash
# ===================================================================
# download_ashesi.sh — fetch the Ashesi/Nokwary Financial Inclusion
# Speech Dataset (Lacuna Fund), CC-BY-4.0, ~148 h total.
#
#   Asante Twi  ~30 h  https://adr.ashesi.edu.gh/datasets/10
#   Ga          ~40 h  https://adr.ashesi.edu.gh/datasets/11
#   Akuapem Twi ~38 h  https://adr.ashesi.edu.gh/datasets/12
#   Fante       ~39 h  https://adr.ashesi.edu.gh/datasets/13
#
# Each page exposes two archives per language (10% / 90% split) plus a
# data.csv with path/transcription/translation per clip (.ogg audio).
#
# BEFORE TRAINING A COMMERCIAL MODEL:
#   1. Confirm the license file shipped with each archive is CC-BY-4.0
#      (README at github.com/Ashesi-Org/Financial-Inclusion-Speech-Dataset
#      says "freely available for use based on the provided open source
#      license" — verify the actual LICENSE file inside the archives).
#   2. Keep this script's attribution output in any released model card.
#
# Usage: ./data/datasets/ashesi/download_ashesi.sh [--langs asante,ga,...]
# ===================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$SCRIPT_DIR/raw"
mkdir -p "$DATA_ROOT"

declare -A PAGES=(
  [asante]="https://adr.ashesi.edu.gh/datasets/10"
  [ga]="https://adr.ashesi.edu.gh/datasets/11"
  [akuapem]="https://adr.ashesi.edu.gh/datasets/12"
  [fante]="https://adr.ashesi.edu.gh/datasets/13"
)

LANGS="${2:-}"
if [[ "${1:-}" == "--langs" ]]; then
  LANGS="$2"
fi
IFS=',' read -ra WANT <<< "${LANGS:-asante,ga,akuapem,fante}"

for lang in "${WANT[@]}"; do
  page="${PAGES[$lang]:-}"
  if [[ -z "$page" ]]; then
    echo "Unknown language: $lang (choose from: ${!PAGES[*]})"
    exit 1
  fi
  echo "=== $lang — resolving archives from $page ==="
  # The archive URLs are linked from the dataset page; discover them rather
  # than hardcoding (filenames have version suffixes).
  mapfile -t urls < <(curl -fsSL "$page" | grep -oE 'href="[^"]*\.(zip|tar\.gz|tgz)"' \
      | sed 's/href="//; s/"$//' | sort -u)
  if [[ ${#urls[@]} -eq 0 ]]; then
    echo "  No archives found in page HTML. Open $page in a browser and"
    echo "  download manually into $DATA_ROOT/$lang/ then re-run prepare."
    mkdir -p "$DATA_ROOT/$lang"
    continue
  fi
  mkdir -p "$DATA_ROOT/$lang"
  for u in "${urls[@]}"; do
    case "$u" in
      http*) url="$u" ;;
      *) url="$page/$u" ;;
    esac
    fn="$DATA_ROOT/$lang/$(basename "$url")"
    if [[ -f "$fn" ]]; then
      echo "  already downloaded: $(basename "$url")"
      continue
    fi
    echo "  downloading $(basename "$url")"
    curl -fL --retry 3 -o "$fn" "$url"
    case "$fn" in
      *.tar.gz|*.tgz) tar xzf "$fn" -C "$DATA_ROOT/$lang" ;;
      *.zip) unzip -qo "$fn" -d "$DATA_ROOT/$lang" ;;
    esac
  done
  echo "  -> $DATA_ROOT/$lang"
done

cat <<'NOTE'

Downloaded. NEXT STEPS
  1. Check the LICENSE file inside each archive; record it in
     docs/LICENSE-MATRIX.md.
  2. Run: python3 data/datasets/ashesi/prepare_ashesi.py
NOTE
