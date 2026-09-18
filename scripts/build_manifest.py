#!/usr/bin/env python
"""Combine every dataset source into one training-ready manifest.

Currently combines:
- SLURP (real audio, 13 labels) — data/external/slurp_audio/manifest.csv,
  produced by streaming the HuggingFace parquet mirror filtered against
  sources/slurp.py's LABEL_MAPPING (see that module for how the mapping
  was verified). Run this first if that file doesn't exist yet.
- The class-shared "Option B" synthetic dataset (all 19 labels) —
  data/external/option_b/{manifest.csv,audio/}. See DATASET.md for the
  source repo and how to pull it locally.
- Google Speech Commands v2 background noise -> unknown_background —
  data/external/gsc/manifest.csv, produced by scripts/fetch_gsc_background.py.
- Snips SLU, lighting-domain subset only (4 labels) —
  data/external/snips_lights/manifest.csv, produced by
  scripts/fetch_snips_lights.py.
- FSC (8 labels, including TEMPERATURE/STOP/PAUSE which had zero real
  coverage anywhere else) — data/external/fsc/manifest.csv, produced by
  scripts/process_fsc.py.

Usage:
    python scripts/build_manifest.py [--out data/dataset_manifest.csv]

Writes one combined CSV in the vcm.dataset.manifest.ManifestRow schema.
"""

# Acknowledgment: the Option B dataset combined here was generated and
# shared by a classmate as part of the class's collective dataset effort
# — see DATASET.md's Acknowledgments section.

from __future__ import annotations

import argparse
from pathlib import Path

from vcm.dataset.manifest import read_manifest, write_manifest
from vcm.dataset.sources.option_b import load_manifest as load_option_b_manifest
from vcm.dataset.sources.option_b import to_manifest_rows as option_b_to_manifest_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
SLURP_MANIFEST = REPO_ROOT / "data/external/slurp_audio/manifest.csv"
OPTION_B_MANIFEST = REPO_ROOT / "data/external/option_b/manifest.csv"
OPTION_B_AUDIO_ROOT = REPO_ROOT / "data/external/option_b/audio"
GSC_BACKGROUND_MANIFEST = REPO_ROOT / "data/external/gsc/manifest.csv"
SNIPS_LIGHTS_MANIFEST = REPO_ROOT / "data/external/snips_lights/manifest.csv"
FSC_MANIFEST = REPO_ROOT / "data/external/fsc/manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/dataset_manifest.csv")
    args = parser.parse_args()

    rows = []

    if SLURP_MANIFEST.exists():
        slurp_rows = read_manifest(SLURP_MANIFEST)
        rows.extend(slurp_rows)
        print(f"SLURP:      {len(slurp_rows)} rows (real, 13 labels)")
    else:
        print(f"SLURP:      skipped, {SLURP_MANIFEST} not found (run scripts/fetch_slurp_audio.py first)")

    if OPTION_B_MANIFEST.exists():
        option_b_records = load_option_b_manifest(OPTION_B_MANIFEST)
        option_b_rows = option_b_to_manifest_rows(option_b_records, audio_root=OPTION_B_AUDIO_ROOT)
        rows.extend(option_b_rows)
        print(f"Option B:   {len(option_b_rows)} rows (synthetic, 19 labels)")
    else:
        print(f"Option B:   skipped, {OPTION_B_MANIFEST} not found")

    if GSC_BACKGROUND_MANIFEST.exists():
        gsc_rows = read_manifest(GSC_BACKGROUND_MANIFEST)
        rows.extend(gsc_rows)
        print(f"GSC bg:     {len(gsc_rows)} rows (real, unknown_background)")
    else:
        print(f"GSC bg:     skipped, {GSC_BACKGROUND_MANIFEST} not found (run scripts/fetch_gsc_background.py first)")

    if SNIPS_LIGHTS_MANIFEST.exists():
        snips_rows = read_manifest(SNIPS_LIGHTS_MANIFEST)
        rows.extend(snips_rows)
        print(f"Snips:      {len(snips_rows)} rows (real, 4 lighting labels)")
    else:
        print(f"Snips:      skipped, {SNIPS_LIGHTS_MANIFEST} not found (run scripts/fetch_snips_lights.py first)")

    if FSC_MANIFEST.exists():
        fsc_rows = read_manifest(FSC_MANIFEST)
        rows.extend(fsc_rows)
        print(f"FSC:        {len(fsc_rows)} rows (real, 8 labels)")
    else:
        print(f"FSC:        skipped, {FSC_MANIFEST} not found (run scripts/process_fsc.py first)")

    write_manifest(rows, args.out)

    by_label: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = by_label.setdefault(row.label, {"real": 0, "synthetic": 0})
        entry["synthetic" if row.is_synthetic else "real"] += 1

    print(f"\nWrote {len(rows)} total rows to {args.out}\n")
    print(f"{'Label':18s} {'real':>8s} {'synthetic':>10s} {'total':>8s}")
    for label in sorted(by_label):
        counts = by_label[label]
        total = counts["real"] + counts["synthetic"]
        print(f"{label:18s} {counts['real']:8d} {counts['synthetic']:10d} {total:8d}")


if __name__ == "__main__":
    main()
