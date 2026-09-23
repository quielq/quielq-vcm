#!/usr/bin/env python
"""Standalone mic-to-intent demo for a trained checkpoint.

Runs entirely on your local machine (Mac laptop or RPi) — this is a
quick sanity check of the trained model's real predictions before any
main.py/dispatch.py integration work, which still needs a taxonomy
reconciliation step (see EXPERIMENTS.md and VCM_Architecture_Review.md)
since the training label space (vcm.dataset.sources.dataset_schema,
20 classes) differs from vcm.taxonomy.LABELS (10 categories) that
main.py/dispatch.py use.

Usage:
    python scripts/demo_infer.py --checkpoint checkpoints/dscnn_bigcap_confusable2_best.pt

Hold the spacebar to record (same push-to-talk mock main.py uses on a
Mac; on the RPi this uses the real HAL pushbutton), release to
classify, repeat. Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from vcm.audio.capture import SAMPLE_RATE, record_while_held
from vcm.audio.features import extract_log_mel
from vcm.hal.button import get_button
from vcm.train.train import MODELS

# The model always outputs a full softmax over all 20 classes, even for
# silence — there's no built-in "nothing was said" option, and
# unknown_background was only trained on 6 specific noisy sources (GSC's
# white noise, a running tap, an exercise bike, a dishwasher, a cat,
# pink noise), not general quiet-room silence, so it has no strong
# learned basis for recognizing plain silence either. This is a simple
# RMS-energy gate as a stopgap: skip classification entirely below this
# threshold rather than trust the model's (currently uncalibrated)
# handling of near-silent audio. Not a trained VAD — just loud enough
# to filter out "held the button, said nothing."
SILENCE_RMS_THRESHOLD = 0.01


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/dscnn_bigcap_confusable2_best.pt"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=SILENCE_RMS_THRESHOLD,
        help="RMS amplitude below which captured audio is treated as silence and skipped "
        "without running the model (0 disables this gate entirely)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save every recording as a WAV (debug_recordings/) and print the full "
        "per-class probability distribution instead of just the top-k. For diagnosing "
        "cases where predictions look wrong/stuck — lets you listen back to exactly what "
        "the model saw and see whether it's confidently wrong or a close call.",
    )
    args = parser.parse_args()

    if args.debug:
        debug_dir = Path("debug_recordings")
        debug_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    labels = list(ckpt["labels"])
    model_kwargs = ckpt.get("model_kwargs", {"num_classes": len(labels)})
    model = MODELS[ckpt["model_name"]](**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # Pre-Experiment-29 checkpoints have no feature_config -> extract_log_mel's defaults.
    feature_config = ckpt.get("feature_config", {})

    print(
        f"Loaded {ckpt['model_name']} from {args.checkpoint} "
        f"(epoch {ckpt['epoch']}, val_acc {ckpt['val_acc']:.4f}, {len(labels)} labels)"
    )

    button = get_button()
    print("Hold spacebar (Mac) or the pushbutton (RPi) and speak a command. Ctrl+C to quit.")

    while True:
        try:
            audio = record_while_held(button)
            if len(audio) == 0:
                print("(no audio captured, try again)")
                continue
            rms = float(np.sqrt(np.mean(np.square(audio))))
            if args.debug:
                wav_path = debug_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_rms{rms:.4f}.wav"
                sf.write(wav_path, audio, SAMPLE_RATE)
                print(f"(saved {wav_path}, {len(audio)/SAMPLE_RATE:.2f}s, rms={rms:.4f})")
            if rms < args.silence_threshold:
                print(f"(silence, rms={rms:.4f} < {args.silence_threshold} — skipped)")
                continue
            features = torch.from_numpy(extract_log_mel(audio, **feature_config)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = F.softmax(model(features), dim=1).squeeze(0)
            k = len(labels) if args.debug else min(args.top_k, len(labels))
            top = torch.topk(probs, k=k)
            print("  ".join(f"{labels[i]}={p:.2f}" for p, i in zip(top.values.tolist(), top.indices.tolist())))
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
