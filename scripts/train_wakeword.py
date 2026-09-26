#!/usr/bin/env python
"""Train the "Hey Kiwi" wake-word detector (EXPERIMENTS.md Experiment 33).

Inputs: the wakeword batch from scripts/generate_targeted_synthetic.py
(positives + near-miss negatives) and an intent manifest (ordinary command
speech and background noise as negatives). See vcm.wakeword.data for the
example kinds. Each epoch draws --samples-per-epoch examples with fixed
kind proportions, since the negatives vastly outnumber the positives.

The wakeword batch has only train/test voices; 10% of the *train* speakers
are held out here as validation for checkpoint selection, so the 20
test-split voices stay untouched for scripts/evaluate_wakeword.py.

Usage:
    python scripts/train_wakeword.py --intent-manifest data/dataset_manifest_exp32.csv --out checkpoints/kiwi_wakeword_s0.pt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from vcm.dataset.manifest import read_manifest
from vcm.train.architectures import CRNN
from vcm.train.dataset import load_noise_bank
from vcm.wakeword import LABELS, WAKE_PHRASE, WINDOW_S
from vcm.wakeword.data import WakeWordDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
KIND_WEIGHTS = {"wake": 0.30, "partial": 0.10, "hard": 0.15, "speech": 0.35, "noise": 0.10}


def _is_val_speaker(speaker: str) -> bool:
    return int(hashlib.md5(speaker.encode()).hexdigest(), 16) % 10 == 0


# Real recordings (scripts/record_wakeword.py) are few next to ~3,000 synthetic
# positives, so each real training take is listed this many times.
REAL_REPEAT = 10


def load_items(
    wake_manifest: Path, intent_manifest: Path, speech_val_cap: int = 3000, seed: int = 0, extra: list[Path] | None = None
):
    train, val = [], []
    for manifest in extra or []:
        with manifest.open(newline="") as f:
            for r in csv.DictReader(f):
                if r["qa_pass"] != "True" or r["split"] != "train":
                    continue  # real test takes stay held out for evaluate_wakeword.py
                if r["label"] == "WAKE":
                    train += [(r["audio_path"], "wake"), (r["audio_path"], "partial")] * REAL_REPEAT
                else:
                    train += [(r["audio_path"], "hard")] * 3
    with wake_manifest.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["qa_pass"] != "True" or r["split"] != "train":
                continue
            dest = val if _is_val_speaker(r["speaker_id"]) else train
            if r["label"] == "WAKE":
                dest += [(r["audio_path"], "wake"), (r["audio_path"], "partial")]
            else:
                dest.append((r["audio_path"], "hard"))
    rows = read_manifest(intent_manifest)
    speech_val = []
    for r in rows:
        kind = "noise" if r.label == "unknown_background" else "speech"
        if r.split == "train":
            train.append((r.audio_path, kind))
        elif r.split == "val":
            (speech_val if kind == "speech" else val).append((r.audio_path, kind))
    random.Random(seed).shuffle(speech_val)
    val += speech_val[:speech_val_cap]
    if not any(kind == "wake" for _, kind in val):
        # Without held-out positives the selection score can't move, and
        # only epoch 1 would ever be saved.
        raise SystemExit("no validation wake clips: need more train-split speakers in the wakeword batch")
    return train, val, [r for r in rows if r.split == "train"]


def evaluate(model, loader, device, items) -> tuple[float, dict[str, float]]:
    model.eval()
    preds = []
    with torch.no_grad():
        for features, _ in loader:
            preds += model(features.to(device)).argmax(1).tolist()
    per_kind = {}
    for kind in ("wake", "partial", "hard", "speech", "noise"):
        idx = [i for i, (_, k) in enumerate(items) if k == kind]
        if idx:
            want = 1 if kind == "wake" else 0
            per_kind[kind] = sum(preds[i] == want for i in idx) / len(idx)
    negatives = [per_kind[k] for k in ("partial", "hard", "speech", "noise") if k in per_kind]
    balanced = (per_kind.get("wake", 0) + sum(negatives) / len(negatives)) / 2
    return balanced, per_kind


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wake-manifest", type=Path, default=REPO_ROOT / "data/external/wakeword_synth/manifest.csv")
    parser.add_argument("--intent-manifest", type=Path, default=REPO_ROOT / "data/dataset_manifest.csv")
    parser.add_argument(
        "--extra-wake-manifest",
        type=Path,
        action="append",
        default=[],
        help="real recordings from scripts/record_wakeword.py (data/wakeword_real/manifest.csv); train split only",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-epoch", type=int, default=24000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--rnn-hidden", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints/kiwi_wakeword.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_items, val_items, intent_train_rows = load_items(
        args.wake_manifest, args.intent_manifest, seed=args.seed, extra=args.extra_wake_manifest
    )
    noise_bank = load_noise_bank(intent_train_rows)
    counts = {k: sum(kind == k for _, kind in train_items) for k in KIND_WEIGHTS}
    print(f"train items by kind: {counts}; val items: {len(val_items)}; noise bank: {len(noise_bank)}", flush=True)

    train_ds = WakeWordDataset(train_items, noise_bank, augment=True, seed=args.seed)
    weights = [KIND_WEIGHTS[kind] / counts[kind] for _, kind in train_items]
    sampler = WeightedRandomSampler(weights, num_samples=args.samples_per_epoch, replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(
        WakeWordDataset(val_items, noise_bank=[], augment=False), batch_size=args.batch_size, num_workers=args.num_workers
    )

    model_kwargs = {"num_classes": len(LABELS), "channels": args.channels, "rnn_hidden": args.rnn_hidden}
    model = CRNN(**model_kwargs).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, 1e-2, 1.0, total_iters=args.warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1)),
        ],
        milestones=[args.warmup_epochs],
    )
    criterion = nn.CrossEntropyLoss()
    best = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            n += labels.size(0)
        scheduler.step()
        balanced, per_kind = evaluate(model, val_loader, device, val_items)
        kinds = "  ".join(f"{k}={v:.3f}" for k, v in per_kind.items())
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={loss_sum / n:.4f}  val_balanced={balanced:.4f}  {kinds}  ({time.time() - t0:.0f}s)", flush=True)
        if balanced > best:
            best = balanced
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "crnn",
                    "model_kwargs": model_kwargs,
                    "labels": LABELS,
                    "feature_config": {"window_s": WINDOW_S, "trim": False},
                    "wake_phrase": WAKE_PHRASE,
                    "epoch": epoch,
                    "val_acc": balanced,
                    "val_per_kind": per_kind,
                    "seed": args.seed,
                },
                args.out,
            )
            print(f"  -> saved new best checkpoint (val_balanced={balanced:.4f}) to {args.out}", flush=True)
    print(f"\nDone. Best val_balanced={best:.4f}, checkpoint at {args.out}", flush=True)


if __name__ == "__main__":
    main()
