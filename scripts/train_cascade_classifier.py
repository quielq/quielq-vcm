#!/usr/bin/env python
"""Train the ASR-cascade's text-intent classifier (see MODEL.md Section 9).

Pipeline: audio -> Whisper (scripts/transcribe_corpus_for_cascade.py,
already run -> data/cascade_transcripts.csv) -> this script's
TF-IDF + logistic regression classifier -> intent label.

Deliberately the cheapest plausible text classifier, not a neural model —
this project's 20-intent, largely fixed-phrasing vocabulary is a much
narrower problem than open-domain SLU, and a first-pass test (no tuning
at all) already scored 90.6% test accuracy on real audio, comfortably
ahead of every direct-audio-classification experiment (best: 75.45%,
EXPERIMENTS.md Experiment 15). Escalate to a neural text classifier only
if this genuinely plateaus below what's needed.

Usage:
    python scripts/train_cascade_classifier.py \
        [--transcripts data/cascade_transcripts.csv] [--out checkpoints/cascade_classifier.joblib]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


def _report_split(name: str, pipeline: Pipeline, rows: list[dict]) -> None:
    if not rows:
        return
    texts = [r["transcript"] for r in rows]
    labels = [r["label"] for r in rows]
    preds = pipeline.predict(texts)
    acc = accuracy_score(labels, preds)
    real_rows = [r for r in rows if r["is_synthetic"] == "False"]
    real_acc_str = ""
    if real_rows:
        real_preds = pipeline.predict([r["transcript"] for r in real_rows])
        real_acc = accuracy_score([r["label"] for r in real_rows], real_preds)
        real_acc_str = f", real-only: {real_acc:.4f} (n={len(real_rows)})"
    print(f"{name}: {acc:.4f} (n={len(rows)}){real_acc_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", type=Path, default=REPO_ROOT / "data/cascade_transcripts.csv")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints/cascade_classifier.joblib")
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    with args.transcripts.open() as f:
        rows = list(csv.DictReader(f))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, args.ngram_max), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, C=args.C)),
        ]
    )
    pipeline.fit([r["transcript"] for r in train_rows], [r["label"] for r in train_rows])

    _report_split("VAL", pipeline, val_rows)
    _report_split("TEST", pipeline, test_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.out)
    print(f"\nSaved pipeline (TfidfVectorizer + LogisticRegression) -> {args.out}")


if __name__ == "__main__":
    main()
