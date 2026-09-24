#!/usr/bin/env python
"""Append the QA-passed targeted synthetic clips (scripts/generate_targeted_synthetic.py)
to a copy of the training manifest (EXPERIMENTS.md Experiment 31).

Writes a new file rather than rebuilding data/dataset_manifest.csv via
build_manifest.py, so the Option B QA filter already applied to the current
manifest is untouched and Experiments 28-30 stay exactly reproducible on the
original file.

Usage:
    python scripts/add_targeted_synth_to_manifest.py \
        [--base data/dataset_manifest.csv] [--out data/dataset_manifest_targeted.csv]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from vcm.dataset.manifest import read_manifest, write_manifest
from vcm.dataset.sources.targeted_synth import load_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=REPO_ROOT / "data/dataset_manifest.csv")
    parser.add_argument("--targeted", type=Path, default=REPO_ROOT / "data/external/targeted_synth/manifest.csv")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/dataset_manifest_targeted.csv")
    args = parser.parse_args()

    base = read_manifest(args.base)
    added = load_manifest(args.targeted)
    existing = {r.audio_path for r in base}
    added = [r for r in added if r.audio_path not in existing]
    write_manifest(base + added, args.out)

    print(f"base: {len(base)} rows, added: {len(added)} QA-passed targeted rows -> {args.out} ({len(base) + len(added)})")
    counts = Counter((r.split, r.label) for r in added)
    for (split, label), n in sorted(counts.items()):
        print(f"  {split:<6}{label:<12}{n:>5}")


if __name__ == "__main__":
    main()
