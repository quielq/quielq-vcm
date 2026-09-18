#!/usr/bin/env python
"""Turn a local FSC extraction into a manifest.csv in this project's
common schema. No network access — FSC isn't freely auto-downloadable
(see sources/fsc.py's module docstring for where this copy came from
and how it was verified), so this script just processes whatever's
already at --fsc-dir.

Usage:
    python scripts/process_fsc.py [--fsc-dir data/external/fsc/fluent_speech_commands_dataset]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vcm.dataset.sources.fsc import coverage_report, load_records, to_manifest_rows  # noqa: E402

MANIFEST_FIELDS = ["audio_path", "label", "source", "is_synthetic", "speaker_id", "split"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fsc-dir",
        type=Path,
        default=REPO_ROOT / "data/external/fsc/fluent_speech_commands_dataset",
        help="Path to the extracted fluent_speech_commands_dataset folder",
    )
    args = parser.parse_args()

    data_dir = args.fsc_dir / "data"
    if not data_dir.exists():
        raise SystemExit(f"{data_dir} not found — extract the FSC archive there first (see DATASET.md)")

    records = []
    records += load_records(data_dir / "train_data.csv", split="train")
    records += load_records(data_dir / "valid_data.csv", split="val")
    records += load_records(data_dir / "test_data.csv", split="test")
    print(f"Loaded {len(records)} records (train+val+test)")

    report = coverage_report(records)
    print(f"\n{'Label':18s} {'matched utterances':>18s}")
    for label, n in sorted(report.items(), key=lambda x: -x[1]):
        print(f"{label:18s} {n:18d}")

    rows = to_manifest_rows(records, audio_root=args.fsc_dir)
    manifest_path = args.fsc_dir.parent / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "audio_path": row.audio_path,
                    "label": row.label,
                    "source": row.source,
                    "is_synthetic": row.is_synthetic,
                    "speaker_id": row.speaker_id,
                    "split": row.split,
                }
            )

    print(f"\nTOTAL: {len(rows)} manifest rows -> {manifest_path}")


if __name__ == "__main__":
    main()
