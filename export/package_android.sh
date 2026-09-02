#!/usr/bin/env bash
# ===================================================================
# package_android.sh — build a versioned SikaBook model bundle for
# ghana-voice-ledger.
#
# Bundle layout (under bundles/sikabook-<version>-android/):
#   MANIFEST.json          version, checksums, component list
#   asr/                   zipformer onnx trio + tokens.txt (when present)
#   asr/vosk/              vosk baseline model tarball (when present)
#   nlu/                   bert int8 onnx + labels (when present)
#   llm/                   Qwen3 GGUF + sale_event.gbnf (when present)
#   vocabulary/product_vocabulary.json
#   postprocess/asr_postprocess_map.json
#   schema/sale_event.schema.json
#
# The app downloads the bundle, verifies MANIFEST checksums, then loads
# components by path. Missing components degrade gracefully:
#   no llm -> rules+BERT only (all devices)
#   no bert -> rules only (must never happen in a release)
#   no zipformer -> vosk baseline (English/Pidgin only)
#
# Usage: ./export/package_android.sh [--version 0.2.0] [--out bundles]
# ===================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
VERSION="0.2.0"
OUT="$ROOT/bundles"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

BUNDLE="$OUT/sikabook-$VERSION-android"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE"/{asr,nlu,postprocess,schema,vocabulary}

copy_if_exists() {
  local src="$1" dst="$2" desc="$3"
  if [[ -e "$src" ]]; then
    cp -r "$src" "$dst"
    echo "  + $desc"
  else
    echo "  - $desc (absent, will degrade)"
  fi
}

echo "Packaging sikabook-$VERSION-android"
copy_if_exists "$ROOT/data/nlu/product_vocabulary.json" \
  "$BUNDLE/vocabulary/product_vocabulary.json" "product vocabulary"
copy_if_exists "$ROOT/data/nlu/asr_postprocess_map.json" \
  "$BUNDLE/postprocess/asr_postprocess_map.json" "ASR postprocess map"
copy_if_exists "$ROOT/export/schema/sale_event.schema.json" \
  "$BUNDLE/schema/sale_event.schema.json" "SaleEvent schema"

# ASR: zipformer (target) — encoder/decoder/joiner + tokens
for f in encoder-*-int8.onnx decoder-*-int8.onnx joiner-*-int8.onnx tokens.txt; do
  if ls "$ROOT"/asr/zipformer/export/"$f" >/dev/null 2>&1; then
    cp "$ROOT"/asr/zipformer/export/"$f" "$BUNDLE/asr/"
    echo "  + asr/$f"
  fi
done
# ASR: vosk baseline
if [[ -f "$ROOT/models/sikabook-en-gh-v1.tar.gz" ]]; then
  cp "$ROOT/models/sikabook-en-gh-v1.tar.gz" "$BUNDLE/asr/vosk-sikabook-en-gh.tar.gz"
  echo "  + asr/vosk-sikabook-en-gh.tar.gz (baseline)"
fi

# NLU: bert int8
if ls "$ROOT"/nlu/bert/export/onnx/*.int8.onnx >/dev/null 2>&1; then
  cp "$ROOT"/nlu/bert/export/onnx/*.int8.onnx "$BUNDLE/nlu/"
  cp "$ROOT"/nlu/bert/export/labels.json "$BUNDLE/nlu/" 2>/dev/null || true
  echo "  + nlu/*.int8.onnx"
fi

# LLM: optional tier
if [[ -d "$ROOT/nlu/llm/qwen3" ]]; then
  cp -r "$ROOT/nlu/llm/qwen3" "$BUNDLE/llm-qwen3"
  cp "$ROOT/nlu/llm/sale_event.gbnf" "$BUNDLE/llm-qwen3/" 2>/dev/null || true
  echo "  + llm-qwen3/ (6GB+ tier)"
fi

# manifest with checksums
python3 - "$BUNDLE" "$VERSION" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

bundle, version = Path(sys.argv[1]), sys.argv[2]
components = {}
for p in sorted(bundle.rglob("*")):
    if p.is_file():
        rel = str(p.relative_to(bundle))
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        components[rel] = {"sha256": h, "bytes": p.stat().st_size}
manifest = {
    "name": "sikabook-android",
    "version": version,
    "schema_version": 1,
    "device_tiers": {
        "3GB+": ["asr", "nlu", "vocabulary", "postprocess", "schema"],
        "6GB+": ["llm-qwen3 (optional)"],
    },
    "components": components,
}
(bundle / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"  + MANIFEST.json ({len(components)} files, "
      f"{sum(c['bytes'] for c in components.values()) / 1e6:.1f} MB)")
PYEOF

tar czf "$BUNDLE.tar.gz" -C "$OUT" "sikabook-$VERSION-android"
echo "Bundle: $BUNDLE.tar.gz ($(du -h "$BUNDLE.tar.gz" | cut -f1))"
