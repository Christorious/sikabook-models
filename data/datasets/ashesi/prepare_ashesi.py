#!/usr/bin/env python3
"""prepare_ashesi.py — turn the raw Ashesi archives into training manifests.

Input  (data/datasets/ashesi/raw/<lang>/):
  *.ogg clips + data.csv with columns (path, transcription, translation).
  The README says: "Ignore the first two directories in the file path" and
  the filename's second segment is a stable speaker ID.

Output (data/datasets/ashesi/prepared/<lang>/):
  clips_16k/*.wav          16 kHz mono PCM (the training sample rate)
  manifest.jsonl           one record per clip:
      {"audio": "...", "transcript": "...", "translation": "...",
       "speaker": "...", "duration_s": ..., "split": "train|dev"}
  stats.json               hours, clip counts, speaker counts

Splits: speaker-disjoint 95/5 (dev speakers held out entirely — the
dataset's own 10% archive is NOT a clean test set; its speakers overlap).

Twi/Ga/Fante orthography notes: transcripts may use ɛ/ɔ and tone marks.
For ASR training we keep them as-is; the tokenizer/lexicon step decides
normalization. Run --normalize-ascii to strip diacritics to eh/oh instead
if your token set is pure ASCII.

Dependencies: ffmpeg on PATH, pandas (Kaggle images have both).
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

DEFAULT_RAW = Path(__file__).resolve().parent / "raw"
DEFAULT_OUT = Path(__file__).resolve().parent / "prepared"
LANG_DIRS = {
    "asante": "tw-Asante",
    "ga": "gaa",
    "akuapem": "tw-Akuapem",
    "fante": "fat",
}

# minimal device-friendly BCP-47-ish codes used across the models repo
BCP47 = {"asante": "tw", "akuapem": "tw", "fante": "fat", "ga": "gaa"}


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def to_wav16(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src),
         "-ac", "1", "-ar", "16000", str(dst)],
        check=True)


def normalize_transcript(text: str, ascii_only: bool) -> str:
    text = " ".join(text.strip().split())
    if ascii_only:
        text = text.replace("ɛ", "eh").replace("Ɛ", "Eh")
        text = text.replace("ɔ", "oh").replace("Ɔ", "Oh")
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
    return text


def find_transcript_csv(lang_dir: Path) -> Path | None:
    candidates = list(lang_dir.rglob("data.csv"))
    return candidates[0] if candidates else None


def prepare_language(lang: str, raw_root: Path, out_root: Path,
                     ascii_only: bool, max_hours: float | None) -> dict:
    lang_dir = raw_root / lang
    csv_path = find_transcript_csv(lang_dir)
    if csv_path is None:
        print(f"  {lang}: no data.csv under {lang_dir} — download first")
        return {}

    out_dir = out_root / lang
    clips_dir = out_dir / "clips_16k"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # collect (abs_audio, transcript, translation, speaker)
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in reader.fieldnames or []}
        path_col = cols.get("path") or cols.get("audio") or cols.get("filename")
        text_col = cols.get("transcription") or cols.get("transcript") or cols.get("text")
        trans_col = cols.get("translation") or cols.get("english")
        for r in reader:
            rel = r[path_col].strip()
            # "Ignore the first two directories in the file path"
            parts = rel.split("/")
            if len(parts) > 2:
                rel = "/".join(parts[2:])
            audio = lang_dir / rel
            if not audio.exists():
                # try by basename anywhere under lang_dir
                hits = list(lang_dir.rglob(Path(rel).name))
                if not hits:
                    continue
                audio = hits[0]
            rows.append((audio, r[text_col], (r.get(trans_col, "") if trans_col else ""),
                         Path(rel).stem.split("-")[1] if "-" in Path(rel).stem else "spk"))
    if not rows:
        print(f"  {lang}: no usable rows in {csv_path}")
        return {}

    # speaker-disjoint split
    speakers = sorted({s for *_, s in rows})
    dev_speakers = set(speakers[: max(1, int(len(speakers) * 0.05))])

    manifest_path = out_dir / "manifest.jsonl"
    stats = {"language": BCP47[lang], "clips": 0, "hours": 0.0,
             "speakers": len(speakers), "dev_speakers": len(dev_speakers)}
    with open(manifest_path, "w", encoding="utf-8") as out:
        for audio, text, translation, speaker in rows:
            text = normalize_transcript(text, ascii_only)
            if not text:
                continue
            clip = clips_dir / (audio.stem + ".wav")
            if not clip.exists():
                try:
                    to_wav16(audio, clip)
                except subprocess.CalledProcessError:
                    continue
            dur = probe_duration(clip)
            hours = dur / 3600.0
            if max_hours is not None and stats["hours"] + hours > max_hours:
                break
            stats["clips"] += 1
            stats["hours"] = round(stats["hours"] + hours, 3)
            out.write(json.dumps({
                "audio": str(clip),
                "transcript": text,
                "translation": translation,
                "speaker": speaker,
                "duration_s": round(dur, 3),
                "language": BCP47[lang],
                "split": "dev" if speaker in dev_speakers else "train",
            }, ensure_ascii=False) + "\n")

    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(f"  {lang}: {stats['clips']} clips, {stats['hours']} h, "
          f"{stats['speakers']} speakers -> {out_dir}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--langs", default="asante,ga,akuapem,fante")
    ap.add_argument("--normalize-ascii", action="store_true",
                    help="strip ɛ/ɔ and diacritics to ASCII")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="cap hours per language (for smoke runs)")
    args = ap.parse_args()

    raw_root, out_root = Path(args.raw), Path(args.out)
    all_stats = {}
    for lang in args.langs.split(","):
        all_stats[lang] = prepare_language(
            lang, raw_root, out_root, args.normalize_ascii, args.max_hours)

    summary = out_root / "summary.json"
    summary.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"\nSummary -> {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
