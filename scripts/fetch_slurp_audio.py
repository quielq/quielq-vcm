#!/usr/bin/env python
"""Pull real SLURP audio for the 13 taxonomy-covered labels (see
sources/slurp.py's LABEL_MAPPING), streaming from the HuggingFace parquet
mirror `yhfang/slurp_dataset_audio_subset` instead of downloading SLURP's
full 3.9GB Zenodo archive.

Requires `pip install -e ".[dev]"` plus `datasets` and `soundfile`
(not default project dependencies — this is offline dataset tooling,
not something the deployed VCM needs):
    pip install datasets soundfile

Usage:
    python scripts/fetch_slurp_audio.py

Writes:
- data/external/slurp_audio/<split>/<slurp_id>_<n>.flac — the audio
- data/external/slurp_audio/manifest.csv — already in the
  vcm.dataset.manifest.ManifestRow schema, ready for scripts/build_manifest.py

Run scripts/slurp_coverage.py first if data/external/slurp/*.jsonl
doesn't exist yet — this script reads the local metadata to know which
slurp_ids to look for, it doesn't re-download the text annotations.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

from datasets import Audio, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vcm.dataset.sources.slurp import LABEL_MAPPING, is_valid_sentence  # noqa: E402

SLURP_META_DIR = REPO_ROOT / "data/external/slurp"
OUT_DIR = REPO_ROOT / "data/external/slurp_audio"

# SLURP's own split naming -> the HF mirror's split naming -> this
# project's manifest split naming (train/val/test, see manifest.py)
SPLIT_FILE_MAP = {"train": "train.jsonl", "validation": "devel.jsonl", "test": "test.jsonl"}
SPLIT_NAME_MAP = {"train": "train", "validation": "val", "test": "test"}

MANIFEST_FIELDS = ["audio_path", "label", "source", "is_synthetic", "speaker_id", "split"]


def main() -> None:
    if not SLURP_META_DIR.exists():
        raise SystemExit(
            f"{SLURP_META_DIR} not found — run scripts/slurp_coverage.py first "
            "to download SLURP's text annotations."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    intent_to_label = {intent: label for label, intents in LABEL_MAPPING.items() for intent in intents}

    manifest_rows = []
    t_start = time.time()

    for hf_split, fname in SPLIT_FILE_MAP.items():
        with open(SLURP_META_DIR / fname) as f:
            local_records = {json.loads(l)["slurp_id"]: json.loads(l) for l in f}

        targets = {
            sid: intent_to_label[rec["intent"]]
            for sid, rec in local_records.items()
            if rec["intent"] in intent_to_label
            and is_valid_sentence(intent_to_label[rec["intent"]], rec["sentence"])
        }
        print(f"[{hf_split}] {len(targets)} target sentences to find", flush=True)

        out_split_dir = OUT_DIR / SPLIT_NAME_MAP[hf_split]
        out_split_dir.mkdir(parents=True, exist_ok=True)

        ds = load_dataset("yhfang/slurp_dataset_audio_subset", split=hf_split, streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))

        seen_count: dict[int, int] = {}
        total_saved = 0
        for row in ds:
            sid = row["slurp_id"]
            label = targets.get(sid)
            if label is None:
                continue
            n = seen_count.get(sid, 0) + 1
            seen_count[sid] = n
            out_path = out_split_dir / f"{sid}_{n}.flac"
            out_path.write_bytes(row["audio"]["bytes"])
            manifest_rows.append(
                {
                    "audio_path": str(out_path.relative_to(REPO_ROOT)),
                    "label": label,
                    "source": "slurp",
                    "is_synthetic": False,
                    "speaker_id": f"slurp_{sid}",
                    "split": SPLIT_NAME_MAP[hf_split],
                }
            )
            total_saved += 1
            if total_saved % 500 == 0:
                print(f"  ...{total_saved} saved so far ({time.time() - t_start:.0f}s elapsed)", flush=True)

        print(
            f"[{hf_split}] done: {total_saved} recordings saved, "
            f"{len(seen_count)}/{len(targets)} sentences covered",
            flush=True,
        )

    manifest_path = OUT_DIR / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nTOTAL: {len(manifest_rows)} recordings -> {manifest_path} ({time.time() - t_start:.0f}s elapsed)")


if __name__ == "__main__":
    main()
