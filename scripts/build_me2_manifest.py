#!/usr/bin/env python
"""Build the training manifest from the class's master dataset (Experiment 37 on).

The class agreed (2026-10-01) on one shared dataset and one fixed test
set: huggingface.co/datasets/airimonda/ai231-me2-voice-commands, with
splits train / test / holdout (plus a separate number-only "numerals"
set, not used here). This script turns its parquet files into this
repo's manifest format (vcm.dataset.manifest) and slot-label format
(vcm.slots.load_slot_labels), so the existing training and evaluation
code runs on it unchanged.

What it does:
- Writes every clip's WAV bytes to data/me2/<split>/<file> (16 kHz mono).
- Keeps the dataset's test and holdout splits exactly as published.
- Carves a validation split out of train, by speaker, so checkpoint
  selection never looks at test. Per source, whole speakers are moved to
  val (shuffled with --seed) until about --val-fraction of that source's
  clips. Sources with fewer than --min-speakers speakers stay entirely in
  train: this keeps the few group (Filipino) speakers and the background
  noise clips for training.
- Labels: the dataset's `command` (19 intents or OUT_OF_SCOPE).
- Slot labels: the dataset's `slot_value` (one of the schema's 3 values per
  slotted intent), mapped to vcm.slots' canonical strings. Clips whose
  value is not in the schema have a blank slot_value and get no slot label.
- Metadata (accent group, variation, transcript, ...) goes to a side CSV
  for scripts/evaluate_checkpoint.py --metadata breakdowns.
- With --numerals, also the numerals set (66,390 number-only clips, labeled
  OUT_OF_SCOPE by the dataset) as split "numerals", source prefixed
  "numerals_" so it is never taken for the train split's noise clips.
  Training uses it only through train.py's --babble-manifest and
  --extra-oos-clips.

Usage:
    python -c "from huggingface_hub import snapshot_download; snapshot_download(
        'airimonda/ai231-me2-voice-commands', repo_type='dataset', local_dir='data/me2/hf',
        allow_patterns=['data/train-*', 'data/test-*', 'data/holdout-*', 'README.md', 'variations.csv'])"
    python scripts/build_me2_manifest.py
    # optional, 2 GB more: add 'data/numerals-*' to allow_patterns, then
    python scripts/build_me2_manifest.py --numerals
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from vcm.dataset.manifest import ManifestRow, write_manifest
from vcm.dataset.sources.dataset_schema import INTENT_LABELS, OUT_OF_SCOPE
from vcm.slots import SLOT_VOCAB, parse_slot

SPLITS = ("train", "test", "holdout")
METADATA_FIELDS = (
    "audio_path", "accent_group", "variation", "slot_value", "bucket", "transcript",
    "transcript_source", "variation_match", "whisper_check",
)  # fmt: skip


def read_split(hf_dir: Path, split: str) -> list[dict]:
    rows = []
    for path in sorted(hf_dir.glob(f"data/{split}-*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    if not rows:
        raise SystemExit(f"no parquet files for split {split!r} under {hf_dir}/data")
    return rows


def choose_val_speakers(rows: list[dict], fraction: float, min_speakers: int, seed: int) -> set[tuple[str, str]]:
    """(source, speaker_id) pairs moved from train to val."""
    clips: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        clips[r["source"]][r["speaker_id"] or ""] += 1
    rng = random.Random(seed)
    val = set()
    for source in sorted(clips):
        speakers = clips[source]
        if len(speakers) < min_speakers:
            continue
        target = fraction * sum(speakers.values())
        order = sorted(speakers)
        rng.shuffle(order)
        taken = 0
        for spk in order:
            if taken >= target:
                break
            val.add((source, spk))
            taken += speakers[spk]
    return val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-dir", type=Path, default=Path("data/me2/hf"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/me2"))
    parser.add_argument("--val-fraction", type=float, default=0.12)
    parser.add_argument("--min-speakers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--numerals", action="store_true", help="Also write the numerals set (split 'numerals').")
    args = parser.parse_args()

    manifest_rows: list[ManifestRow] = []
    slot_rows: list[dict] = []
    meta_rows: list[dict] = []
    for split in SPLITS + (("numerals",) if args.numerals else ()):
        rows = read_split(args.hf_dir, split)
        val_speakers = choose_val_speakers(rows, args.val_fraction, args.min_speakers, args.seed) if split == "train" else set()
        for r in rows:
            label = r["command"]
            if label not in (*INTENT_LABELS, OUT_OF_SCOPE):
                raise SystemExit(f"unexpected command {label!r} in {split}: {r['file']}")
            audio_path = args.out_dir / split / r["file"]
            if not audio_path.exists():
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(r["audio"]["bytes"])
            speaker = r["speaker_id"] or ""
            row_split = "val" if (r["source"], speaker) in val_speakers else split
            manifest_rows.append(
                ManifestRow(
                    audio_path=str(audio_path),
                    label=label,
                    source=f"numerals_{r['source']}" if split == "numerals" else r["source"],
                    is_synthetic=bool(r["is_synthetic"]),
                    speaker_id=speaker,
                    split=row_split,
                )
            )
            if label in SLOT_VOCAB and r["slot_value"]:
                value = parse_slot(label, r["slot_value"])
                if value is None:
                    raise SystemExit(f"slot value {r['slot_value']!r} of {label} is not in vcm.slots.SLOT_VOCAB")
                slot_rows.append(
                    {"audio_path": str(audio_path), "label": label, "value": value,
                     "text_source": "synthetic" if r["is_synthetic"] else "real"}
                )  # fmt: skip
            meta_rows.append({"audio_path": str(audio_path), **{k: r.get(k) or "" for k in METADATA_FIELDS[1:]}})

    write_manifest(manifest_rows, args.out_dir / "manifest.csv")
    with (args.out_dir / "slot_labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("audio_path", "label", "value", "text_source"))
        writer.writeheader()
        writer.writerows(slot_rows)
    with (args.out_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"wrote {args.out_dir / 'manifest.csv'}, slot_labels.csv ({len(slot_rows)} rows), metadata.csv")
    by_split = Counter(r.split for r in manifest_rows)
    for split in ("train", "val", "test", "holdout", "numerals"):
        rows = [r for r in manifest_rows if r.split == split]
        real = sum(not r.is_synthetic for r in rows)
        speakers = len({(r.source, r.speaker_id) for r in rows})
        print(f"  {split:<8} {by_split[split]:>6} clips  {real:>5} real  {speakers:>4} speakers")
    val_labels = Counter(r.label for r in manifest_rows if r.split == "val")
    missing = [label for label in (*INTENT_LABELS, OUT_OF_SCOPE) if not val_labels[label]]
    if missing:
        print(f"  warning: no val clips for {missing}")


if __name__ == "__main__":
    main()
