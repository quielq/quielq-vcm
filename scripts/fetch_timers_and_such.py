#!/usr/bin/env python
"""Pull real TIMER/ALARM audio from Timers and Such (Lugosch et al.,
NeurIPS 2021 D&B — https://zenodo.org/records/4623772), without
downloading the full 12.2GB archive.

Verified directly (not assumed): the Zenodo file server supports HTTP
range requests (returns 206 PARTIAL_CONTENT) even though it doesn't
advertise Accept-Ranges, so `remotezip` can list the archive's central
directory and extract only the specific member files needed — ~1,071
real SetTimer/SetAlarm recordings (~134MB), not the full archive
(which is mostly a synthetic portion we don't want — the whole point
is real audio, not another TTS voice).

Requires `pip install remotezip` (not a default project dependency —
offline dataset tooling, not something the deployed VCM needs).

Usage:
    python scripts/fetch_timers_and_such.py

Writes:
- data/external/timers_and_such/{train-real,dev-real,test-real}.csv —
  the small (~650KB total) metadata files, kept for provenance/re-mapping
- data/external/timers_and_such/audio/<split>/<file>.wav — only the
  real SetTimer/SetAlarm recordings
- data/external/timers_and_such/manifest.csv — in this project's
  common ManifestRow schema, ready for scripts/build_manifest.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from remotezip import RemoteZip

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vcm.dataset.sources.timers_and_such import LABEL_MAPPING, load_records  # noqa: E402

ARCHIVE_URL = "https://zenodo.org/api/records/4623772/files/timers-and-such-v1.0.zip/content"
OUT_DIR = REPO_ROOT / "data/external/timers_and_such"
META_SPLITS = ("train-real", "dev-real", "test-real")

MANIFEST_FIELDS = ["audio_path", "label", "source", "is_synthetic", "speaker_id", "split"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir = OUT_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    with RemoteZip(ARCHIVE_URL) as zf:
        print("Fetching metadata CSVs (small, ~650KB total)...", flush=True)
        for split in META_SPLITS:
            csv_name = f"{split}.csv"
            if not (OUT_DIR / csv_name).exists():
                zf.extract(csv_name, OUT_DIR)
            print(f"  {csv_name} ready", flush=True)

        records = load_records(OUT_DIR, splits=META_SPLITS)
        print(f"{len(records)} real SetTimer/SetAlarm recordings to fetch", flush=True)

        manifest_rows = []
        for i, rec in enumerate(records, start=1):
            out_path = audio_dir / rec.split / Path(rec.path).name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                with zf.open(rec.path) as src, out_path.open("wb") as dst:
                    dst.write(src.read())
            manifest_rows.append(
                {
                    "audio_path": str(out_path.relative_to(REPO_ROOT)),
                    "label": LABEL_MAPPING[rec.intent],
                    "source": "timers_and_such",
                    "is_synthetic": False,
                    "speaker_id": f"timers_{rec.speaker_id}",
                    "split": rec.split,
                }
            )
            if i % 100 == 0:
                print(f"  ...{i}/{len(records)} fetched ({time.time() - t_start:.0f}s elapsed)", flush=True)

    manifest_path = OUT_DIR / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nTOTAL: {len(manifest_rows)} recordings -> {manifest_path} ({time.time() - t_start:.0f}s elapsed)")


if __name__ == "__main__":
    main()
