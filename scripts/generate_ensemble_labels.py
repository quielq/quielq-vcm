#!/usr/bin/env python
"""Soft labels from an ensemble of our own checkpoints, for self-distillation
(train.py --distill-weight/--distill-labels, EXPERIMENTS.md Experiment 41).

The teacher is the average softmax of several CRNNs trained on the same
master-dataset train split (different seeds and recipes), so no outside
data or model is involved. Each clip's teacher probabilities are computed
on clean audio (no augmentation), for the train split only.

Output format matches scripts/generate_distillation_labels.py:
audio_path, then one probability column per label.

Usage:
    python scripts/generate_ensemble_labels.py checkpoints/exp37*_s*.pt \\
        --manifest data/me2/manifest.csv --out data/me2/ensemble_labels.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vcm.dataset.manifest import read_manifest
from vcm.train.dataset import LABELS, ManifestDataset
from vcm.train.train import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--manifest", type=Path, default=Path("data/me2/manifest.csv"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=Path("data/me2/ensemble_labels.csv"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = [r for r in read_manifest(args.manifest) if r.split == args.split]
    device = torch.device(args.device)
    total = torch.zeros(len(rows), len(LABELS))
    feature_config = None
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if tuple(ckpt["labels"]) != LABELS:
            raise SystemExit(f"{path}: label list differs from vcm.train.dataset.LABELS")
        if feature_config is None:
            feature_config = ckpt["feature_config"]
        elif ckpt["feature_config"] != feature_config:
            raise SystemExit(f"{path}: feature_config differs from the first checkpoint's")
        model = MODELS[ckpt["model_name"]](**ckpt["model_kwargs"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        loader = DataLoader(ManifestDataset(rows, feature_config=feature_config), batch_size=256, num_workers=args.num_workers)
        probs = []
        with torch.no_grad():
            for features, _ in loader:
                probs.append(torch.softmax(model(features.to(device)), dim=1).cpu())
        total += torch.cat(probs)
        print(f"{path}: done", flush=True)
    total /= len(args.checkpoints)

    truth = torch.tensor([LABELS.index(r.label) for r in rows])
    print(f"ensemble of {len(args.checkpoints)} on {args.split}: {(total.argmax(1) == truth).float().mean():.2%} top-1")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", *LABELS])
        for row, p in zip(rows, total.tolist()):
            writer.writerow([row.audio_path, *(f"{x:.6f}" for x in p)])
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
