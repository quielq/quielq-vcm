#!/usr/bin/env python
"""Check that a copy of the class master dataset is the revision this repo
was trained and tested on, clip for clip.

Every result from Experiment 43 on uses revision da92a79 (2026-10-02) of
huggingface.co/datasets/airimonda/ai231-me2-voice-commands. Its fingerprint
is committed in data/dataset_schema/me2_fingerprint_da92a79.json: for each
split, the number of clips and one SHA-1 over every clip's audio bytes and
label columns, in order. This script computes the same fingerprint for a
copy and compares:

- the class's shared copy on the DGX (/data/ai231, a Hugging Face
  `datasets` cache written by load_dataset(..., cache_dir="/data/ai231")),
- or a download of the parquet files (data/me2/hf).

Usage:
    python scripts/verify_shared_dataset.py --shared-cache /data/ai231
    python scripts/verify_shared_dataset.py --hf-dir data/me2/hf
    python scripts/verify_shared_dataset.py --hf-dir data/me2/hf --write   # (re)write the fingerprint
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = REPO_ROOT / "data/dataset_schema/me2_fingerprint_da92a79.json"
SPLITS = ("train", "test", "holdout", "numerals")
# Columns that define a clip; everything the training and evaluation code reads.
LABEL_COLUMNS = ("file", "transcript", "command", "variation", "slot_value", "out_of_scope", "bucket",
                 "speaker_id", "source", "is_synthetic", "accent_group", "duration_s")  # fmt: skip

_spec = importlib.util.spec_from_file_location("build_me2_manifest", REPO_ROOT / "scripts/build_me2_manifest.py")
_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_builder)


def split_tables(split: str, hf_dir: Path | None, shared: Path | None):
    if shared is not None:
        for path in sorted(shared.glob(f"ai231-me2-voice-commands-{split}*.arrow")):
            yield _builder.read_arrow(path)
    else:
        for path in sorted(hf_dir.glob(f"data/{split}-*.parquet")):
            yield pq.read_table(path)


def fingerprint(split: str, hf_dir: Path | None, shared: Path | None) -> dict:
    digest, n = hashlib.sha1(), 0
    for table in split_tables(split, hf_dir, shared):
        for batch in table.to_batches(max_chunksize=500):
            for row in batch.to_pylist():
                digest.update(hashlib.sha1(row["audio"]["bytes"]).digest())
                digest.update(json.dumps([row.get(c) for c in LABEL_COLUMNS], default=str).encode())
                n += 1
    return {"clips": n, "sha1": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--shared-cache", type=Path, help="e.g. /data/ai231")
    source.add_argument("--hf-dir", type=Path, help="e.g. data/me2/hf (snapshot_download output)")
    parser.add_argument("--write", action="store_true", help=f"Write the fingerprint to {FINGERPRINT.name} instead of checking.")
    args = parser.parse_args()

    shared = _builder.shared_cache_dir(args.shared_cache) if args.shared_cache else None
    result = {split: fingerprint(split, args.hf_dir, shared) for split in SPLITS}
    if args.write:
        FINGERPRINT.write_text(json.dumps({"revision": "da92a79ffde3031d5bb2a25138d9dd7d9f7ed006", **result}, indent=2) + "\n")
        print(f"wrote {FINGERPRINT}")
        return
    expected = json.loads(FINGERPRINT.read_text())
    ok = True
    for split in SPLITS:
        match = result[split] == expected[split]
        ok &= match
        print(f"{split:<9} {result[split]['clips']:>6} clips  {'matches' if match else 'DIFFERS from'} revision da92a79")
    print("identical to the revision every result is on" if ok else "NOT the same data: results would change; see docs/DATASET.md")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
