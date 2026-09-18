#!/usr/bin/env python
"""Download Google Speech Commands v2's background-noise folder and chop
it into fixed-length unknown_background training clips.

This fills a real gap: Section 3 of the architecture review requires an
explicit background/unknown class, and Section 9 already names this as
the intended source, but nothing had actually pulled it into the
pipeline before this script.

Only downloads the official ~2.4GB archive transiently, extracts just
the ~13MB `_background_noise_/` folder (6 long noise recordings, not
the 35 keyword classes), and deletes the full archive immediately after
— the 35 keyword classes aren't needed for this and aren't kept.

Usage:
    python scripts/fetch_gsc_background.py [--clips-per-file 100]
"""

from __future__ import annotations

import argparse
import csv
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vcm.dataset.sources.gsc_background import chop_background_noise  # noqa: E402

ARCHIVE_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
OUT_DIR = REPO_ROOT / "data/external/gsc"
MANIFEST_FIELDS = ["audio_path", "label", "source", "is_synthetic", "speaker_id", "split"]


def ensure_background_noise(out_dir: Path) -> Path:
    noise_dir = out_dir / "_background_noise_"
    if noise_dir.exists():
        return noise_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / "speech_commands_v0.02.tar.gz"
    print(f"Downloading {ARCHIVE_URL} (~2.4GB, transient, deleted after extraction)...")
    urllib.request.urlretrieve(ARCHIVE_URL, archive_path)

    print("Extracting _background_noise_/ only...")
    with tarfile.open(archive_path) as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("./_background_noise_/")]
        tar.extractall(out_dir, members=members)

    archive_path.unlink()
    return noise_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-per-file", type=int, default=100)
    args = parser.parse_args()

    noise_dir = ensure_background_noise(OUT_DIR)
    rows = chop_background_noise(noise_dir, OUT_DIR / "chopped", clips_per_file=args.clips_per_file)

    manifest_path = OUT_DIR / "manifest.csv"
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

    print(f"\nTOTAL: {len(rows)} unknown_background clips -> {manifest_path}")


if __name__ == "__main__":
    main()
