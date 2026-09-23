#!/usr/bin/env python
"""Generate teacher soft-labels for knowledge distillation from the
ASR-cascade into the tiny direct-audio DS-CNN (see EXPERIMENTS.md
Experiment 27 and the conversation around it).

Why this exists: the assignment explicitly rules out ASR models for the
*deployed* VCM ("ASR models are not desirable for on-device computing
because of footprint... VCM must be tiny... standalone"). That's a
constraint on what runs at inference time, not on how the training data
is built. This script uses the cascade (already proven to fix the
confusable-pair problem — see EXPERIMENTS.md Experiment 26) purely
offline, as a teacher: it never touches the deployed model or runs on
the RPi. The tiny DS-CNN remains the only thing that's actually
deployed.

For each row in data/dataset_manifest.csv (except unknown_background,
which has no transcript and no cascade prediction to distill from),
looks up its already-computed Whisper transcript (data/cascade_transcripts.csv,
from scripts/transcribe_corpus_for_cascade.py) and runs it through the
trained cascade classifier (checkpoints/cascade_classifier.joblib) to
get a full probability distribution over the 19 intent labels — the
"soft label" the student model will be taught to partially match,
alongside its normal hard-label loss.

Usage:
    python scripts/generate_distillation_labels.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

MANIFEST_PATH = REPO_ROOT / "data/dataset_manifest.csv"
TRANSCRIPTS_PATH = REPO_ROOT / "data/cascade_transcripts.csv"
CLASSIFIER_PATH = REPO_ROOT / "checkpoints/cascade_classifier.joblib"
OUT_PATH = REPO_ROOT / "data/distillation_labels.csv"


def main() -> None:
    with MANIFEST_PATH.open() as f:
        manifest_rows = [r for r in csv.DictReader(f) if r["label"] != "unknown_background"]

    with TRANSCRIPTS_PATH.open() as f:
        transcript_by_path = {r["audio_path"]: r["transcript"] for r in csv.DictReader(f)}

    missing = [r["audio_path"] for r in manifest_rows if r["audio_path"] not in transcript_by_path]
    if missing:
        raise SystemExit(
            f"{len(missing)} manifest rows have no transcript (e.g. {missing[0]!r}) — "
            "run scripts/transcribe_corpus_for_cascade.py first."
        )

    pipeline = joblib.load(CLASSIFIER_PATH)
    labels = list(pipeline.classes_)
    print(f"Loaded cascade classifier ({len(labels)} labels)")

    texts = [transcript_by_path[r["audio_path"]] for r in manifest_rows]
    probs = pipeline.predict_proba(texts)  # (n_rows, n_labels), rows sum to 1

    with OUT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", *labels])
        for row, prob_row in zip(manifest_rows, probs):
            writer.writerow([row["audio_path"], *[f"{p:.6f}" for p in prob_row]])

    print(f"Wrote {len(manifest_rows)} teacher soft-labels -> {OUT_PATH}")


if __name__ == "__main__":
    main()
