#!/usr/bin/env python
"""Train a model on data/dataset_manifest.csv. See EXPERIMENTS.md for
real results from runs of this script, and MODEL.md/architectures.py
for what each --model option is and why.

Usage:
    python -m vcm.train.train --model dscnn [--epochs 30] [--batch-size 128] [--lr 1e-3]
    python -m vcm.train.train --model bcresnet --augment
    python -m vcm.train.train --model bcresnet --warmup-epochs 3  # linear warmup + cosine decay,
                                                                   # per the BC-ResNet paper (arXiv:2106.04140)

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
from vcm.train.dataset import LABELS, ManifestDataset, cap_per_class, class_counts, class_weights, load_distillation_labels
from vcm.train.losses import ConfusablePairLoss, DistillationLoss, build_confusable_mask, effective_number_weights

MODELS = {"dscnn": DSCNN, "bcresnet": BCResNet}
# Generic --width/--depth CLI flags map to each model's own constructor
# kwarg names (DSCNN: num_filters/num_blocks, BCResNet: channels/num_blocks).
WIDTH_KWARG = {"dscnn": "num_filters", "bcresnet": "channels"}


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
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Linear LR warmup for this many epochs, then cosine decay to 0 over the rest. "
        "0 (default) keeps a flat --lr the whole run.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Randomly subsample this fraction of the training split (for data-scaling "
        "experiments, e.g. EXPERIMENTS.md's learning-curve test). 1.0 (default) uses all "
        "training rows. Val/test are never subsampled.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap each class's training examples at this many, randomly subsampling classes "
        "that exceed it (val/test untouched). Addresses class imbalance by shrinking large "
        "classes rather than reweighting the loss (see EXPERIMENTS.md Experiment 20) — a "
        "dataset-level fix, doesn't interact with the loss function or LR schedule the way "
        "ConfusablePairLoss's focal-loss option did. Default: no cap, use every row.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override the model's channel width (num_filters for dscnn, channels for "
        "bcresnet). Default: the model class's own default.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Override the model's block count (num_blocks, both models). Default: the "
        "model class's own default.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout before the final classifier (dscnn only — BCResNet already has its own "
        "internal dropout via BCResBlock). Default: the model's own default (0.0 for DSCNN, "
        "i.e. no regularization at all through Experiment 23). Added to test whether DSCNN's "
        "total lack of regularization is a real gap — see EXPERIMENTS.md Experiment 24.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam weight_decay (L2 regularization). 0.0 (default) matches every experiment "
        "through 23 — plain Adam had no weight decay at all.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for the cross-entropy term (plain CrossEntropyLoss's native "
        "label_smoothing, or the same passed through to ConfusablePairLoss's internal CE). "
        "0.0 (default) matches every experiment through 23. Motivated by observed overconfident "
        "wrong predictions on live audio (e.g. 0.97 confidence for a wrong class) -- see "
        "EXPERIMENTS.md Experiment 24.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Continue training from this checkpoint's weights (architecture/model_kwargs must "
        "match --model/--width/--depth) instead of a fresh random init. For cheaply testing "
        "whether newly-added data helps, without a full from-scratch retrain.",
    )
    parser.add_argument(
        "--confusable-alpha",
        type=float,
        default=0.0,
        help="Extra penalty weight on probability mass placed on classes known to be "
        "confusable with the true label (VOLUME_UP/VOLUME_DOWN/TEMPERATURE, LIGHT_ON/"
        "LIGHT_OFF — see vcm.train.losses). 0.0 (default) is plain weighted "
        "cross-entropy, i.e. the loss used through Experiment 13.",
    )
    parser.add_argument(
        "--class-balance-beta",
        type=float,
        default=0.0,
        help="Use effective-number-of-samples class weighting (Cui et al. 2019) with this "
        "beta instead of the plain inverse-frequency weights used through Experiment 17. "
        "0.0 (default) keeps plain inverse-frequency. Try 0.999 or 0.9999 — see "
        "vcm.train.losses.effective_number_weights.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="Focal-loss modulation exponent (Lin et al. 2017) — down-weights examples the "
        "model already gets right, up-weights ones it doesn't. 0.0 (default) disables this, "
        "i.e. plain weighted cross-entropy. Try 2.0 (the paper's default).",
    )
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=0.0,
        help="Knowledge-distillation weight: adds a KL-divergence term against the "
        "ASR-cascade's soft labels (scripts/generate_distillation_labels.py) on top of the "
        "normal hard-label loss. The cascade is a training-time-only teacher, never part of "
        "the deployed model — this is how ASR is used without violating the assignment's "
        "no-ASR-on-device constraint. 0.0 (default) disables this. See EXPERIMENTS.md "
        "Experiment 27.",
    )
    parser.add_argument(
        "--distill-labels",
        type=Path,
        default=Path("data/distillation_labels.csv"),
        help="Teacher soft-labels produced by scripts/generate_distillation_labels.py. Only "
        "read when --distill-weight > 0.",
    )
    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=2.0,
        help="Softmax temperature for the distillation KL term (Hinton et al. 2015's default "
        "is a mild 2-4). Only used when --distill-weight > 0.",
    )
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
    if args.max_per_class is not None:
        before = len(train_ds.rows)
        train_ds.rows = cap_per_class(train_ds.rows, args.max_per_class)
        print(f"--max-per-class {args.max_per_class}: {before} -> {len(train_ds.rows)} rows", flush=True)
    if args.train_fraction < 1.0:
        n_keep = max(1, int(len(train_ds.rows) * args.train_fraction))
        train_ds.rows = random.sample(train_ds.rows, n_keep)
    print(
        f"train: {len(train_ds)} examples (train_fraction={args.train_fraction}), "
        f"val: {len(val_ds)} examples",
        flush=True,
    )

    use_distillation = args.distill_weight > 0.0
    if use_distillation:
        train_ds.distillation_labels = load_distillation_labels(args.distill_labels)
        print(
            f"Using knowledge distillation (weight={args.distill_weight}, "
            f"temperature={args.distill_temperature}) from {args.distill_labels}",
            flush=True,
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    if args.class_balance_beta > 0.0:
        weights = effective_number_weights(class_counts(train_ds.rows), beta=args.class_balance_beta).to(device)
        print(f"Using effective-number class weights (beta={args.class_balance_beta})", flush=True)
    else:
        weights = class_weights(train_ds.rows).to(device)

    if args.confusable_alpha > 0.0 or args.focal_gamma > 0.0 or args.label_smoothing > 0.0:
        confusable_mask = build_confusable_mask(LABELS).to(device)
        criterion = ConfusablePairLoss(
            weights,
            confusable_mask,
            alpha=args.confusable_alpha,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
        print(
            f"Using ConfusablePairLoss (alpha={args.confusable_alpha}, gamma={args.focal_gamma}, "
            f"label_smoothing={args.label_smoothing})",
            flush=True,
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)

    if use_distillation:
        distill_criterion = DistillationLoss(
            criterion, distill_weight=args.distill_weight, temperature=args.distill_temperature
        )

    model_kwargs: dict[str, int | float] = {"num_classes": len(LABELS)}
    if args.width is not None:
        model_kwargs[WIDTH_KWARG[args.model]] = args.width
    if args.depth is not None:
        model_kwargs["num_blocks"] = args.depth
    if args.dropout is not None:
        if args.model != "dscnn":
            raise SystemExit("--dropout is only wired up for --model dscnn (BCResNet has its own internal dropout)")
        model_kwargs["dropout"] = args.dropout
    model = MODELS[args.model](**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)
    if args.resume_from is not None:
        resume_ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        print(
            f"Resumed weights from {args.resume_from} "
            f"(was epoch {resume_ckpt['epoch']}, val_acc {resume_ckpt['val_acc']:.4f})",
            flush=True,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0, total_iters=args.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - args.warmup_epochs, 1)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs]
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            if use_distillation:
                features, labels, teacher_probs = batch
                features, labels, teacher_probs = features.to(device), labels.to(device), teacher_probs.to(device)
            else:
                features, labels = batch
                features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = distill_criterion(logits, labels, teacher_probs) if use_distillation else criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)

        train_loss = running_loss / n_seen
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={current_lr:.2e}  ({elapsed:.1f}s)",
            flush=True,
        )
        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "augment": args.augment,
                    "seed": args.seed,
                    "warmup_epochs": args.warmup_epochs,
                    "train_fraction": args.train_fraction,
                    "max_per_class": args.max_per_class,
                    "resumed_from": str(args.resume_from) if args.resume_from else None,
                    "confusable_alpha": args.confusable_alpha,
                    "class_balance_beta": args.class_balance_beta,
                    "focal_gamma": args.focal_gamma,
                    "dropout": args.dropout,
                    "weight_decay": args.weight_decay,
                    "label_smoothing": args.label_smoothing,
                    "distill_weight": args.distill_weight,
                    "distill_temperature": args.distill_temperature if use_distillation else None,
                    "model_kwargs": model_kwargs,
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
