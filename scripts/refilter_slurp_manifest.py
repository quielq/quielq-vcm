#!/usr/bin/env python
"""Re-apply LABEL_MAPPING and is_valid_sentence() to an already-downloaded
SLURP audio manifest, without re-fetching any audio.

Why this exists: sources/slurp.py's LABEL_MAPPING and is_valid_sentence()
were updated after training runs showed WEATHER, TIME, MESSAGE, and
LIST_REMINDERS were consistently weak classes (see DATASET.md's
"Known per-label quality signal" section and EXPERIMENTS.md). If you
already have data/external/slurp_audio/ populated from an earlier
scripts/fetch_slurp_audio.py run, re-running that script would re-download
~900MB for no reason — this script instead re-derives the correct label
(or "drop this row") for each already-downloaded file by looking its
slurp_id back up in the local annotation jsonl files, and rewrites
manifest.csv in place. The audio files themselves are never touched or
re-downloaded; some just become unreferenced by any manifest row.

Usage:
    python scripts/refilter_slurp_manifest.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vcm.dataset.sources.slurp import LABEL_MAPPING, is_valid_sentence  # noqa: E402

SLURP_META_DIR = REPO_ROOT / "data/external/slurp"
MANIFEST_PATH = REPO_ROOT / "data/external/slurp_audio/manifest.csv"

# manifest split name -> annotation jsonl file (see fetch_slurp_audio.py's
# own SPLIT_FILE_MAP/SPLIT_NAME_MAP for the same mapping in the other direction)
SPLIT_TO_JSONL = {"train": "train.jsonl", "val": "devel.jsonl", "test": "test.jsonl"}


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found — nothing to re-filter.")

    intent_to_label = {intent: label for label, intents in LABEL_MAPPING.items() for intent in intents}

    # slurp_id -> (intent, sentence), loaded once per split
    sentence_by_id: dict[str, dict[str, dict]] = {}
    for split, fname in SPLIT_TO_JSONL.items():
        path = SLURP_META_DIR / fname
        with path.open() as f:
            sentence_by_id[split] = {json.loads(line)["slurp_id"]: json.loads(line) for line in f}

    with MANIFEST_PATH.open() as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else []

    kept, dropped_missing_intent, dropped_bad_sentence = [], 0, 0
    for row in rows:
        # audio_path looks like data/external/slurp_audio/<split>/<sid>_<n>.flac
        parts = Path(row["audio_path"]).parts
        split = parts[-2]
        sid = int(Path(row["audio_path"]).stem.split("_")[0])

        rec = sentence_by_id.get(split, {}).get(sid)
        if rec is None:
            # shouldn't happen, but don't silently keep a row we can't verify
            dropped_missing_intent += 1
            continue

        new_label = intent_to_label.get(rec["intent"])
        if new_label is None:
            # this row's raw intent is no longer mapped to anything
            # (e.g. lists_query, dropped for LIST_REMINDERS)
            dropped_missing_intent += 1
            continue
        if not is_valid_sentence(new_label, rec["sentence"]):
            dropped_bad_sentence += 1
            continue

        row["label"] = new_label  # in case the mapping itself changed, not just filtering
        kept.append(row)

    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"Before: {len(rows)} rows")
    print(f"Dropped (label no longer mapped, e.g. LIST_REMINDERS): {dropped_missing_intent}")
    print(f"Dropped (failed is_valid_sentence quality filter): {dropped_bad_sentence}")
    print(f"After: {len(kept)} rows -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
