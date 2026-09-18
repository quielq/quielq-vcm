#!/usr/bin/env python
"""Download SLURP's annotation files (if not already cached) and print a
coverage report against the class's 19-label fixed-vs-slotted taxonomy
(see src/vcm/dataset/sources/slurp.py for how the label mapping was
built).

Usage:
    python scripts/slurp_coverage.py [--data-dir data/external/slurp]

Only downloads the ~13MB of annotation text (train/devel/test.jsonl),
not any audio.
"""

# Acknowledgment: the taxonomy this checks coverage against was shared
# by a classmate as part of the class's collective dataset effort — see
# DATASET.md's Acknowledgments section.

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from vcm.dataset.sources.slurp import DEFAULT_SPLIT_FILES, coverage_report, load_records

BASE_URL = "https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp"


def ensure_downloaded(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in DEFAULT_SPLIT_FILES:
        path = data_dir / filename
        if path.exists():
            continue
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(f"{BASE_URL}/{filename}", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/external/slurp"),
        help="Where to cache/read the SLURP jsonl files (default: data/external/slurp)",
    )
    args = parser.parse_args()

    ensure_downloaded(args.data_dir)
    records = load_records(args.data_dir)
    report = coverage_report(records)

    print(f"\nLoaded {len(records)} sentences / {sum(r.num_recordings for r in records)} recordings\n")
    print(f"{'Label':18s} {'sentences':>10s} {'recordings':>11s} {'matched intents':>16s}")
    for label, r in report.items():
        print(f"{label:18s} {r['sentences']:10d} {r['recordings']:11d} {r['matched_intents']:16d}")

    zero_coverage = [label for label, r in report.items() if r["sentences"] == 0]
    if zero_coverage:
        print(f"\nZero SLURP coverage: {', '.join(zero_coverage)}")


if __name__ == "__main__":
    main()
