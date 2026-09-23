#!/usr/bin/env python
"""ASR-cascade prototype, milestone 1 (see MODEL.md Section 9): transcribe
every clip in data/dataset_manifest.csv with faster-whisper (base), pairing
(Whisper's own transcript, existing label) as training data for a
text-based intent classifier.

Deliberately uses Whisper's own output, not each source's original
ground-truth transcript (which our combined manifest doesn't carry through
anyway) — the text classifier should train on the same *kind* of noisy
ASR output it'll see at real inference time, not clean text it'll never
actually get.

unknown_background is skipped (it's noise, not speech — nothing to
transcribe, and it isn't a text-classification target anyway; the cascade
would still need a separate audio-level silence/noise gate upstream of
Whisper, same as today's RMS gate in demo_infer.py).

Usage:
    python scripts/transcribe_corpus_for_cascade.py [--model-size base]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vcm.dataset.qa.synthetic_check import FasterWhisperTranscriber  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "data/dataset_manifest.csv"
OUT_PATH = REPO_ROOT / "data/cascade_transcripts.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", default="base")
    args = parser.parse_args()

    with MANIFEST_PATH.open() as f:
        rows = [r for r in csv.DictReader(f) if r["label"] != "unknown_background"]

    print(f"Loading faster-whisper ({args.model_size})...", flush=True)
    transcriber = FasterWhisperTranscriber(model_size=args.model_size)

    t0 = time.time()
    out_rows = []
    for i, row in enumerate(rows, 1):
        transcript = transcriber.transcribe(Path(row["audio_path"]))
        out_rows.append(
            {
                "audio_path": row["audio_path"],
                "label": row["label"],
                "split": row["split"],
                "source": row["source"],
                "is_synthetic": row["is_synthetic"],
                "transcript": transcript,
            }
        )
        if i % 1000 == 0:
            elapsed = time.time() - t0
            print(
                f"  ...{i}/{len(rows)} transcribed ({elapsed:.0f}s elapsed, "
                f"{elapsed/i*1000:.0f}s/1000 clips, ETA {(len(rows)-i)*elapsed/i/60:.0f} min)",
                flush=True,
            )

    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "label", "split", "source", "is_synthetic", "transcript"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nDone: {len(out_rows)} transcripts -> {OUT_PATH} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
