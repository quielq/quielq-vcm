#!/usr/bin/env python
"""Train a model on data/dataset_manifest.csv. See EXPERIMENTS.md for
real results from runs of this script, and MODEL.md/architectures.py
for what each --model option is and why.

Usage:
    python -m vcm.train.train --model dscnn [--epochs 30] [--batch-size 128] [--lr 1e-3]
    python -m vcm.train.train --model bcresnet --augment

Saves the best checkpoint (by val accuracy) to --out, including the
label list and the model name it was trained with
(vcm.train.dataset.LABELS) so the checkpoint is self-describing.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from vcm.train.architectures import BCResNet, DSCNN
from vcm.train.dataset import LABELS, ManifestDataset, class_weights

MODELS = {"dscnn": DSCNN, "bcresnet": BCResNet}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_manifest.csv"))
    parser.add_argument("--model", choices=sorted(MODELS), default="dscnn")
    parser.add_argument("--augment", action="store_true", help="Apply SpecAugment to training data (never val/test)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, for comparable experiments")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}, model: {args.model}, augment: {args.augment}, seed: {args.seed}", flush=True)

    train_ds = ManifestDataset.from_csv(args.manifest, split="train", augment=args.augment)
    val_ds = ManifestDataset.from_csv(args.manifest, split="val", augment=False)
    print(f"train: {len(train_ds)} examples, val: {len(val_ds)} examples", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    weights = class_weights(train_ds.rows).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = MODELS[args.model](num_classes=len(LABELS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)

        train_loss = running_loss / n_seen
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ({elapsed:.1f}s)",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "augment": args.augment,
                    "seed": args.seed,
                    "labels": LABELS,
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                args.out,
            )
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f}) to {args.out}", flush=True)

    print(f"\nDone. Best val_acc={best_val_acc:.4f}, checkpoint at {args.out}", flush=True)


if __name__ == "__main__":
    main()
