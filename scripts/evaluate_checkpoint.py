#!/usr/bin/env python
"""Evaluate one or more trained checkpoints on a manifest split, with the
breakdowns EXPERIMENTS.md needs to compare runs honestly.

Why this exists: through Experiment 27, direct-audio models were only
ever compared on best *val* accuracy over all rows (real + synthetic +
background), while the ASR-cascade (Experiment 26) was reported on
*test* accuracy over real speech only (90.62%). Those two numbers
aren't comparable. This script reports the direct-audio model on the
same basis as the cascade, plus the per-class and confusable-group
breakdowns previously computed by ad hoc scratch scripts.

"Real speech" matches scripts/train_cascade_classifier.py's real-only
filter: is_synthetic == False, excluding unknown_background (which has
no transcript, so the cascade was never scored on it).

Usage:
    python scripts/evaluate_checkpoint.py checkpoints/a.pt checkpoints/b.pt [--split test]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vcm.dataset.manifest import read_manifest
from vcm.train.dataset import LABELS, ManifestDataset
from vcm.train.losses import CONFUSABLE_GROUPS
from vcm.train.train import MODELS


def predict(ckpt_path: Path, rows: list, device: torch.device, num_workers: int) -> tuple[list[int], list[float]]:
    """Predicted class and its softmax probability (the model's confidence) per row."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if tuple(ckpt["labels"]) != LABELS:
        raise SystemExit(f"{ckpt_path}: label list differs from vcm.train.dataset.LABELS")
    # Pre-Experiment-29 checkpoints have no feature_config -> the defaults they were trained with.
    dataset = ManifestDataset(rows, feature_config=ckpt.get("feature_config"))
    model = MODELS[ckpt["model_name"]](**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers)
    preds: list[int] = []
    confidences: list[float] = []
    with torch.no_grad():
        for features, _ in loader:
            top = torch.softmax(model(features.to(device)), dim=1).max(dim=1)
            preds.extend(top.indices.tolist())
            confidences.extend(top.values.tolist())
    return preds, confidences


def print_confidence_table(pairs: list[tuple[int, int]], confidences: list[float]) -> None:
    """What a reject threshold ("didn't catch that, please repeat") would do:
    for each threshold, how many utterances get rejected, how accurate the
    accepted ones are, and what share of the model's errors get caught."""
    n = len(pairs)
    n_wrong = sum(t != p for t, p in pairs)
    print(f"  {'threshold':>9} {'rejected':>9} {'acc of accepted':>16} {'errors caught':>14} {'correct lost':>13}")
    for threshold in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        accepted = [(t, p) for (t, p), c in zip(pairs, confidences) if c >= threshold]
        rejected = [(t, p) for (t, p), c in zip(pairs, confidences) if c < threshold]
        acc = sum(t == p for t, p in accepted) / max(len(accepted), 1)
        caught = sum(t != p for t, p in rejected)
        lost = sum(t == p for t, p in rejected)
        print(
            f"  {threshold:>9.2f} {len(rejected) / n:>9.1%} {acc:>16.2%} "
            f"{caught / max(n_wrong, 1):>14.1%} {lost / n:>13.1%}"
        )


def accuracy(pairs: list[tuple[int, int]]) -> str:
    if not pairs:
        return "n/a (n=0)"
    correct = sum(t == p for t, p in pairs)
    return f"{correct / len(pairs):.2%} (n={len(pairs)})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_manifest.csv"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = [r for r in read_manifest(args.manifest) if r.split == args.split]
    truth = [LABELS.index(r.label) for r in rows]
    is_real_speech = [not r.is_synthetic and r.label != "unknown_background" for r in rows]
    device = torch.device(args.device)

    for ckpt_path in args.checkpoints:
        preds, confidences = predict(ckpt_path, rows, device, args.num_workers)
        pairs = list(zip(truth, preds))
        real_pairs = [pair for pair, real in zip(pairs, is_real_speech) if real]
        real_confidences = [c for c, real in zip(confidences, is_real_speech) if real]
        synth_pairs = [pair for pair, row in zip(pairs, rows) if row.is_synthetic]

        print(f"\n=== {ckpt_path} ({args.split} split)")
        print(f"all:          {accuracy(pairs)}")
        print(f"real speech:  {accuracy(real_pairs)}   <- comparable to the cascade's 90.62%")
        print(f"synthetic:    {accuracy(synth_pairs)}")
        print("by source:")
        for source in sorted({r.source for r in rows}):
            source_pairs = [pair for pair, row in zip(pairs, rows) if row.source == source]
            print(f"  {source:<16}{accuracy(source_pairs)}")

        print("\nper-class (all / real speech):")
        for i, label in enumerate(LABELS):
            cls_all = [p for p in pairs if p[0] == i]
            cls_real = [p for p in real_pairs if p[0] == i]
            print(f"  {label:<20} {accuracy(cls_all):<22} {accuracy(cls_real)}")

        print("\nconfusable groups (correct / within-group confusion / other-wrong):")
        for group in CONFUSABLE_GROUPS:
            idx = {LABELS.index(label) for label in group}
            grp = [p for p in pairs if p[0] in idx]
            n = len(grp)
            correct = sum(t == p for t, p in grp)
            within = sum(t != p and p in idx for t, p in grp)
            print(
                f"  {'/'.join(group):<36} {correct / n:.1%} / {within / n:.1%} / "
                f"{(n - correct - within) / n:.1%} (n={n})"
            )

        print("\ntop confusions:")
        confusions = Counter((t, p) for t, p in pairs if t != p)
        for (t, p), count in confusions.most_common(10):
            print(f"  {LABELS[t]} -> {LABELS[p]}: {count}")

        print("\nreject threshold on top-class confidence (real speech):")
        print_confidence_table(real_pairs, real_confidences)


if __name__ == "__main__":
    main()
