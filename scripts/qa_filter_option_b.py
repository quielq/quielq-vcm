#!/usr/bin/env python
"""Run the transcribe-and-compare QA gate (vcm.dataset.qa.synthetic_check)
against the already-downloaded Option B synthetic manifest, and rewrite
manifest.csv in place to drop clips the TTS generator produced that don't
actually say what they were supposed to.

Why this exists: a synthetic-to-real-gap research pass (see EXPERIMENTS.md
Experiment 24) found direct published evidence that ASR-based filtering of
synthetic TTS training clips — discarding ones a real ASR can't correctly
transcribe — measurably closes the synthetic-to-real generalization gap
(89% -> 92.5% in a directly comparable study). The QA gate to do exactly
this already existed in this repo (built for Anthony Navarez's transcriber
tool, generalized during Option B's original quality checks) but had never
actually been run here. This script runs it.

Requires `pip install faster-whisper` (not a default dependency — pulls in
a speech-to-text model, offline dataset-curation tool only, never runs at
inference time).

Usage:
    python scripts/qa_filter_option_b.py [--threshold 0.8] [--device cuda]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vcm.dataset.qa.synthetic_check import FasterWhisperTranscriber, screen_clip  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "data/external/option_b/manifest.csv"
AUDIO_ROOT = REPO_ROOT / "data/external/option_b/audio"
REPORT_PATH = REPO_ROOT / "data/external/option_b/qa_report.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--model-size", default="small")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found — nothing to QA-filter.")

    with MANIFEST_PATH.open() as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

    print(f"Loading faster-whisper ({args.model_size})...", flush=True)
    transcriber = FasterWhisperTranscriber(model_size=args.model_size)

    t0 = time.time()
    kept, dropped = [], []
    qa_rows = []
    for i, row in enumerate(rows, 1):
        audio_path = AUDIO_ROOT / row["path"]
        result = screen_clip(audio_path, row["transcript"], transcriber, threshold=args.threshold)
        qa_rows.append(
            {
                "path": row["path"],
                "label": row["label"],
                "expected_text": result.expected_text,
                "transcribed_text": result.transcribed_text,
                "confidence": f"{result.confidence:.3f}",
                "passed": result.passed,
            }
        )
        if result.passed:
            kept.append(row)
        else:
            dropped.append(row)
        if i % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  ...{i}/{len(rows)} screened ({elapsed:.0f}s elapsed, {elapsed/i*1000:.0f}s/1000 clips)", flush=True)

    with REPORT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "expected_text", "transcribed_text", "confidence", "passed"])
        writer.writeheader()
        writer.writerows(qa_rows)

    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"\nBefore: {len(rows)} rows")
    print(f"Dropped (transcription confidence < {args.threshold}): {len(dropped)}")
    print(f"After: {len(kept)} rows -> {MANIFEST_PATH}")
    print(f"Full QA report (all clips, pass+fail) -> {REPORT_PATH}")

    from collections import Counter

    dropped_by_label = Counter(row["intent"] if "intent" in row else row["label"] for row in dropped)
    print("\nDropped rows by label:")
    for label, n in dropped_by_label.most_common():
        print(f"  {label:20s} {n}")


if __name__ == "__main__":
    main()
