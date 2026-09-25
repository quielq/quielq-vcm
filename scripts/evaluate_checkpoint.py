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
from vcm.slots import load_slot_labels
from vcm.train.dataset import LABELS, ManifestDataset
from vcm.train.losses import CONFUSABLE_GROUPS
from vcm.train.train import MODELS


def predict(ckpt_path: Path, rows: list, device: torch.device, num_workers: int) -> dict:
    """Per row: predicted class, its softmax probability (the model's
    confidence) and, if the model has slot heads, the predicted value index
    for every slot head. Works for a .pt checkpoint or an exported .onnx
    file (scripts/export_onnx.py), so a quantized model is scored on
    exactly the same rows."""
    if ckpt_path.suffix == ".onnx":
        from vcm.deploy.runtime import OnnxIntentModel

        onnx_model = OnnxIntentModel(ckpt_path)
        labels, feature_config, slot_vocab = onnx_model.labels, onnx_model.feature_config, onnx_model.slot_vocab
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        labels, feature_config = ckpt["labels"], ckpt.get("feature_config")
        slot_vocab = ckpt.get("slot_vocab") or {}
        model = MODELS[ckpt["model_name"]](**ckpt["model_kwargs"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
    if tuple(labels) != LABELS:
        raise SystemExit(f"{ckpt_path}: label list differs from vcm.train.dataset.LABELS")
    # Pre-Experiment-29 checkpoints have no feature_config -> the defaults they were trained with.
    loader = DataLoader(ManifestDataset(rows, feature_config=feature_config), batch_size=256, num_workers=num_workers)
    out = {"preds": [], "confidences": [], "slots": {k: [] for k in slot_vocab}, "slot_vocab": slot_vocab}
    with torch.no_grad():
        for features, _ in loader:
            if ckpt_path.suffix == ".onnx":
                probs = {k: torch.from_numpy(v) for k, v in onnx_model.probabilities(features.numpy()).items()}
                intent_probs = probs["intent"]
                slot_probs = {k: probs[f"slot_{k}"] for k in slot_vocab}
            elif slot_vocab:
                logits, slot_logits = model.forward_with_slots(features.to(device))
                intent_probs = torch.softmax(logits, dim=1)
                slot_probs = {k: v.cpu() for k, v in slot_logits.items()}
            else:
                intent_probs, slot_probs = torch.softmax(model(features.to(device)), dim=1), {}
            top = intent_probs.max(dim=1)
            out["preds"].extend(top.indices.tolist())
            out["confidences"].extend(top.values.tolist())
            for k, v in slot_probs.items():
                out["slots"][k].extend(v.argmax(dim=1).tolist())
    return out


def print_slot_accuracy(rows: list, pairs: list[tuple[int, int]], result: dict, slot_labels: dict) -> None:
    """Slot value accuracy per slotted intent, over clips with a slot label:
    `slot` reads the head of the clip's true intent (the slot head on its
    own); `joint` also requires the intent to be right (what the device
    would actually do). Split by where the label came from: ground-truth
    text (synthetic clips) vs Whisper's transcript (real audio, noisy)."""
    print("\nslot values (slot-head accuracy / intent+slot joint accuracy):")
    for intent, vocab in result["slot_vocab"].items():
        for source_kind in ("ground_truth", "whisper"):
            slot_hits = joint_hits = n = 0
            for i, row in enumerate(rows):
                label = slot_labels.get(row.audio_path)
                if not label or label[0] != intent or label[2] != source_kind:
                    continue
                n += 1
                slot_ok = vocab[result["slots"][intent][i]] == label[1]
                slot_hits += slot_ok
                joint_hits += slot_ok and pairs[i][0] == pairs[i][1]
            if n:
                print(f"  {intent:<11} {source_kind:<13} {slot_hits / n:>7.2%} / {joint_hits / n:>7.2%}  (n={n})")


def print_confidence_table(pairs: list[tuple[int, int]], confidences: list[float]) -> None:
    """What a reject threshold ("didn't catch that, please repeat") would do:
    for each threshold, how many utterances get rejected, how accurate the
    accepted ones are, and what share of the model's errors get caught."""
    n = len(pairs)
    if n == 0:
        print("  (no rows)")
        return
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


def accuracy_value(pairs: list[tuple[int, int]]) -> float | None:
    return sum(t == p for t, p in pairs) / len(pairs) if pairs else None


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
    parser.add_argument(
        "--slot-labels",
        type=Path,
        default=None,
        help="scripts/build_slot_labels.py output; adds slot-value accuracy for models with slot heads.",
    )
    parser.add_argument(
        "--exclude-source",
        nargs="*",
        default=[],
        help="Drop rows from these sources before scoring (e.g. snips_lights, to compare models "
        "trained before and after its speaker re-split without leakage).",
    )
    args = parser.parse_args()

    rows = [
        r for r in read_manifest(args.manifest) if r.split == args.split and r.source not in args.exclude_source
    ]
    slot_labels = load_slot_labels(args.slot_labels) if args.slot_labels else {}
    truth = [LABELS.index(r.label) for r in rows]
    is_real_speech = [not r.is_synthetic and r.label != "unknown_background" for r in rows]
    device = torch.device(args.device)

    for ckpt_path in args.checkpoints:
        result = predict(ckpt_path, rows, device, args.num_workers)
        preds, confidences = result["preds"], result["confidences"]
        pairs = list(zip(truth, preds))
        real_pairs = [pair for pair, real in zip(pairs, is_real_speech) if real]
        real_confidences = [c for c, real in zip(confidences, is_real_speech) if real]
        synth_pairs = [pair for pair, row in zip(pairs, rows) if row.is_synthetic]

        print(f"\n=== {ckpt_path} ({args.split} split)")
        print(f"all:          {accuracy(pairs)}")
        print(f"real speech:  {accuracy(real_pairs)}   <- comparable to the cascade's 90.62%")
        per_class_real = [accuracy_value([p for p in real_pairs if p[0] == i]) for i in range(len(LABELS))]
        per_class_real = [a for a in per_class_real if a is not None]
        if per_class_real:
            print(f"real speech, macro (mean of per-class): {sum(per_class_real) / len(per_class_real):.2%} over {len(per_class_real)} classes")
        if args.exclude_source:
            print(f"(excluded sources: {', '.join(args.exclude_source)})")
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

        if result["slot_vocab"] and slot_labels:
            print_slot_accuracy(rows, pairs, result, slot_labels)

        print("\nreject threshold on top-class confidence (real speech):")
        print_confidence_table(real_pairs, real_confidences)


if __name__ == "__main__":
    main()
