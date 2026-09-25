#!/usr/bin/env python
"""Evaluate the "Hey Kiwi" detector the way it runs on the device: streaming,
a 1.5 s window every 0.1 s, with the same trigger rule as the live loop
(vcm.wakeword.detector.triggers).

Two numbers per threshold:
- False reject rate: share of held-out "hey kiwi" clips (test-split voices,
  never trained on) that never trigger. Each clip is embedded in 1 s of
  quiet before and after; measured clean and with background noise at
  10 dB SNR.
- False wake-ups per hour: triggers while streaming one long recording made
  of every test-split command clip (real and synthetic speech that never
  says the phrase), the test-split near-miss clips, and test-split
  background noise, back to back.

Usage:
    python scripts/evaluate_wakeword.py checkpoints/kiwi_wakeword_s0.pt --intent-manifest data/dataset_manifest_exp32.csv
    python scripts/evaluate_wakeword.py models/kiwi_wakeword.int8.onnx ...   # same, for the exported model
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from vcm.audio.capture import SAMPLE_RATE
from vcm.dataset.manifest import read_manifest
from vcm.train.wave_augment import add_noise
from vcm.wakeword import HOP_S
from vcm.wakeword.data import _read
from vcm.wakeword.detector import onnx_scorer, triggers, window_scores

REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)


def torch_scorer(path: Path, device: torch.device):
    from vcm.train.train import MODELS

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MODELS[ckpt["model_name"]](**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    wake = list(ckpt["labels"]).index("wake")

    def score(features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(model(torch.from_numpy(features).to(device)), dim=1)[:, wake].cpu().numpy()

    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--wake-manifest", type=Path, default=REPO_ROOT / "data/external/wakeword_synth/manifest.csv")
    parser.add_argument("--intent-manifest", type=Path, default=REPO_ROOT / "data/dataset_manifest.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--extra-wake-manifest",
        type=Path,
        action="append",
        default=[],
        help="real recordings (scripts/record_wakeword.py); their test-split 'hey kiwi' takes add a "
        "'false reject (real voice)' column",
    )
    parser.add_argument(
        "--max-false-wakes-per-hour",
        type=float,
        default=1.0,
        help="false wake-up budget for the suggested threshold",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    score = onnx_scorer(args.model) if args.model.suffix == ".onnx" else torch_scorer(args.model, torch.device(args.device))
    rng = random.Random(0)
    with args.wake_manifest.open(newline="") as f:
        wake_rows = [r for r in csv.DictReader(f) if r["qa_pass"] == "True" and r["split"] == args.split]
    positives = [r["audio_path"] for r in wake_rows if r["label"] == "WAKE"]
    near_misses = [r["audio_path"] for r in wake_rows if r["label"] == "NOT_WAKE"]
    real_positives = []
    for manifest in args.extra_wake_manifest:
        with manifest.open(newline="") as f:
            real_positives += [r["audio_path"] for r in csv.DictReader(f) if r["label"] == "WAKE" and r["split"] == "test"]
    intent_rows = [r for r in read_manifest(args.intent_manifest) if r.split == args.split]
    background = [_read(r.audio_path) for r in intent_rows if r.label == "unknown_background"]

    # False rejects: best (max) streaming score per clip, clean and in noise.
    quiet = np.zeros(SAMPLE_RATE, dtype="float32")
    per_clip = {"clean": [], "noisy": []}
    if real_positives:  # as recorded: the room's own noise, no added noise
        per_clip["real"] = [window_scores(np.concatenate([quiet, _read(p), quiet]), score) for p in real_positives]
    for path in positives:
        clip = np.concatenate([quiet, _read(path), quiet])
        per_clip["clean"].append(window_scores(clip, score))
        noisy = add_noise(clip + 1e-4, rng.choice(background), 10.0) if background else clip
        per_clip["noisy"].append(window_scores(noisy.astype("float32"), score))

    # False wake-ups: one long stream of everything that isn't the phrase.
    negatives = [(r.audio_path, f"{r.source}:{r.label}") for r in intent_rows]
    negatives += [(p, "near-miss") for p in near_misses]
    rng.shuffle(negatives)
    gap = np.zeros(int(0.3 * SAMPLE_RATE), dtype="float32")
    pieces, owner = [], []
    for path, tag in negatives:
        audio = _read(path)
        pieces += [audio, gap]
        owner += [(path, tag)] * (len(audio) + len(gap))
    stream = np.concatenate(pieces)
    hours = len(stream) / SAMPLE_RATE / 3600
    stream_scores = window_scores(stream, score)
    print(f"model: {args.model}")
    real_note = f", {len(real_positives)} real recorded takes" if real_positives else ""
    print(f"positives: {len(positives)} held-out 'hey kiwi' clips (test voices){real_note}; negative stream: {len(negatives)} clips, {hours:.2f} h\n")

    real_head = f"{'false reject (real voice)':>27}" if real_positives else ""
    print(f"{'threshold':>9}{'false reject (clean)':>22}{'false reject (10dB noise)':>27}{real_head}{'false wake-ups':>16}{'per hour':>10}")
    results = []
    for t in THRESHOLDS:
        frr = {k: sum(not triggers(s, t) for s in v) / max(len(v), 1) for k, v in per_clip.items()}
        fired = triggers(stream_scores, t)
        results.append((t, frr, fired))
        real_col = f"{frr['real']:>27.1%}" if real_positives else ""
        print(f"{t:>9.2f}{frr['clean']:>22.1%}{frr['noisy']:>27.1%}{real_col}{len(fired):>16}{len(fired) / hours:>10.2f}")

    budget = args.max_false_wakes_per_hour
    ok = [r for r in results if len(r[2]) / hours <= budget]
    key = "real" if real_positives else "noisy"
    chosen = min(ok, key=lambda r: (r[1][key], r[1]["noisy"], r[1]["clean"])) if ok else results[-1]
    print(f"\nsuggested threshold: {chosen[0]} (lowest {key} false-reject rate with <= {budget} false wake-ups/hour)")
    hop = int(HOP_S * SAMPLE_RATE)
    culprits = Counter(owner[min(i * hop + int(0.75 * SAMPLE_RATE), len(owner) - 1)] for i in chosen[2])
    print(f"false wake-ups at that threshold, by clip: {culprits.most_common(15)}")


if __name__ == "__main__":
    main()
