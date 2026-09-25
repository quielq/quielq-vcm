#!/usr/bin/env python
"""Stream the Snips SLU dataset, classify each transcript with
sources/snips_lights.py, and save only the ~42% that confidently match
one of our 4 lighting labels (LIGHT_ON, LIGHT_OFF, BRIGHTNESS, COLOR) —
the rest (mostly music requests, see that module's docstring) are
skipped, not downloaded.

Usage:
    pip install datasets soundfile
    python scripts/fetch_snips_lights.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from datasets import Audio, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vcm.dataset.sources.snips_lights import classify, speaker_split  # noqa: E402

OUT_DIR = REPO_ROOT / "data/external/snips_lights"
MANIFEST_FIELDS = ["audio_path", "label", "source", "is_synthetic", "speaker_id", "split"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("MWilinski/snips_slu_v1.0", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    manifest_rows = []
    matched = 0
    for row in ds:
        label = classify(row["text"])
        if label is None:
            continue
        out_path = OUT_DIR / f"{row['ID']}.wav"
        out_path.write_bytes(row["audio"]["bytes"])
        manifest_rows.append(
            {
                "audio_path": str(out_path.relative_to(REPO_ROOT)),
                "label": label,
                "source": "snips_lights",
                "is_synthetic": False,
                "speaker_id": f"snips_{row['worker']}",
                # By speaker, not row position: position-based splitting put
                # the same voices in train and test (see speaker_split).
                "split": speaker_split(f"snips_{row['worker']}"),
            }
        )
        matched += 1
        if matched % 500 == 0:
            print(f"...{matched} matched/saved so far", flush=True)

    manifest_path = OUT_DIR / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nTOTAL: {matched} lighting-command clips -> {manifest_path}")


if __name__ == "__main__":
    main()
