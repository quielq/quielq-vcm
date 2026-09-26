#!/usr/bin/env python
"""Re-split the already-downloaded Snips lighting subset by speaker, in place
(EXPERIMENTS.md Experiment 32).

fetch_snips_lights.py originally split by row position, so every Snips
speaker's recordings landed in train, val *and* test. This rewrites only
the `split` column of data/external/snips_lights/manifest.csv using
sources/snips_lights.speaker_split (the same rule the fixed fetch script
now uses), keeps a .bak copy, and verifies no speaker spans two splits.
Re-run scripts/build_manifest.py afterwards to propagate it.

Usage:
    python scripts/resplit_snips_by_speaker.py [--manifest data/external/snips_lights/manifest.csv]
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from vcm.dataset.sources.snips_lights import speaker_key, speaker_split

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/external/snips_lights/manifest.csv")
    args = parser.parse_args()

    with args.manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)

    before = defaultdict(set)
    for r in rows:
        before[speaker_key(r["speaker_id"])].add(r["split"])
    leaking = sum(len(s) > 1 for s in before.values())

    changed = 0
    for r in rows:
        new = speaker_split(r["speaker_id"])
        changed += new != r["split"]
        r["split"] = new

    after = defaultdict(set)
    for r in rows:
        after[speaker_key(r["speaker_id"])].add(r["split"])
    assert all(len(s) == 1 for s in after.values()), "a speaker still spans splits"

    shutil.copy(args.manifest, args.manifest.with_suffix(".csv.bak"))
    with args.manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    speakers = Counter(next(iter(s)) for s in after.values())
    clips = Counter(r["split"] for r in rows)
    print(f"speakers: {len(before)} total, {leaking} were in more than one split before, 0 after")
    print(f"rows changed: {changed}/{len(rows)}")
    for split in ("train", "val", "test"):
        print(f"  {split:<6} {speakers[split]:>3} speakers  {clips[split]:>5} clips")
    print(f"wrote {args.manifest} (backup: {args.manifest.with_suffix('.csv.bak')})")


if __name__ == "__main__":
    main()
