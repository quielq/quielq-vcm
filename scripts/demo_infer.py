#!/usr/bin/env python
"""Standalone mic-to-intent demo for a trained model (exported .onnx or a .pt checkpoint).

Runs entirely on your local machine (Mac laptop or RPi) — this is a
quick sanity check of the trained model's real predictions before any
main.py/dispatch.py integration work, which still needs a taxonomy
reconciliation step (see EXPERIMENTS.md and VCM_Architecture_Review.md)
since the training label space (vcm.dataset.sources.dataset_schema,
20 classes) differs from vcm.taxonomy.LABELS (10 categories) that
main.py/dispatch.py use.

Usage:
    python scripts/demo_infer.py                                   # models/vcm_intent.onnx (Experiment 34)
    python scripts/demo_infer.py --checkpoint checkpoints/exp31_crnn_targeted_s2.pt

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
from vcm.audio.features import extract_log_mel, speech_level
from vcm.deploy.runtime import OnnxIntentModel
from vcm.hal.button import get_button
from vcm.train.train import MODELS

# The model always outputs a full softmax over all 20 classes, even for
# silence — there's no built-in "nothing was said" option, and
# unknown_background was only trained on 6 specific noisy sources (GSC's
# white noise, a running tap, an exercise bike, a dishwasher, a cat,
# pink noise), not general quiet-room silence, so it has no strong
# learned basis for recognizing plain silence either. This is a simple
# energy gate as a stopgap: skip classification entirely below this
# threshold rather than trust the model's (currently uncalibrated)
# handling of near-silent audio. Not a trained VAD — just loud enough
# to filter out "held the button, said nothing."
#
# Measured on the loudest 300ms (vcm.audio.features.speech_level), not
# the whole clip: a whole-clip RMS gate at 0.01 skipped many real
# commands at 0.0065-0.0099, because silence before/after the words
# dilutes it. 0.008 sits between recordings with no speech (loudest
# window <= 0.0048) and the quietest real command (0.0165) in 133 real
# push-to-talk recordings — see speech_level's docstring.
SILENCE_THRESHOLD = 0.008

# Below this top-class probability, ask the user to repeat instead of
# acting. Chosen on the real-speech *val* split (not test;
# `scripts/evaluate_checkpoint.py --split val` prints the full table).
# For the Experiment 34 model at 0.6 it rejects 11.1% of utterances,
# catches 47% of its errors, and is 91.8% accurate on the ones it accepts
# (vs 86.2% with no threshold), at the cost of re-asking for 4.6% of
# utterances it would have gotten right. 0.7 trades more re-asks (15.5%)
# for 93.5%.
REJECT_THRESHOLD = 0.6


def load_model(path: Path, device: torch.device):
    """-> (labels, feature_config, predict), predict(features) -> (intent probs, {intent: (slot, conf)})."""
    if path.suffix == ".onnx":
        model = OnnxIntentModel(path)
        print(f"Loaded {path} ({len(model.labels)} labels, slots for {sorted(model.slot_vocab) or 'none'})")

        def predict(features: np.ndarray):
            probs = model.probabilities(features)
            slots = {}
            for intent, values in model.slot_vocab.items():
                p = probs[f"slot_{intent}"][0]
                slots[intent] = (values[int(p.argmax())], float(p.max()))
            return probs["intent"][0], slots

        return model.labels, model.feature_config, predict

    ckpt = torch.load(path, map_location=device, weights_only=False)
    labels = list(ckpt["labels"])
    model_kwargs = ckpt.get("model_kwargs", {"num_classes": len(labels)})
    torch_model = MODELS[ckpt["model_name"]](**model_kwargs).to(device)
    torch_model.load_state_dict(ckpt["model_state_dict"])
    torch_model.eval()
    print(f"Loaded {ckpt['model_name']} from {path} (epoch {ckpt['epoch']}, val_acc {ckpt['val_acc']:.4f}, {len(labels)} labels)")

    def predict(features: np.ndarray):
        with torch.no_grad():
            logits = torch_model(torch.from_numpy(features).unsqueeze(0).to(device))
        return F.softmax(logits, dim=1).squeeze(0).cpu().numpy(), {}

    # Pre-Experiment-29 checkpoints have no feature_config -> extract_log_mel's defaults.
    return labels, ckpt.get("feature_config", {}), predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/vcm_intent.onnx"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=SILENCE_THRESHOLD,
        help="Skip without running the model if the loudest 300ms of the recording has an RMS "
        "below this (vcm.audio.features.speech_level); 0 disables the gate",
    )
    parser.add_argument(
        "--reject-threshold",
        type=float,
        default=REJECT_THRESHOLD,
        help="If the top class's probability is below this, report \"didn't catch that, please "
        "repeat\" instead of a prediction (0 disables).",
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

    labels, feature_config, predict = load_model(args.checkpoint, torch.device(args.device))

    button = get_button()
    print("Hold spacebar (Mac) or the pushbutton (RPi) and speak a command. Ctrl+C to quit.")

    while True:
        try:
            audio = record_while_held(button)
            if len(audio) == 0:
                print("(no audio captured, try again)")
                continue
            level = speech_level(audio)
            if args.debug:
                wav_path = debug_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_level{level:.4f}.wav"
                sf.write(wav_path, audio, SAMPLE_RATE)
                print(f"(saved {wav_path}, {len(audio)/SAMPLE_RATE:.2f}s, speech level={level:.4f})")
            if level < args.silence_threshold:
                print(f"(silence, speech level={level:.4f} < {args.silence_threshold} — skipped)")
                continue
            probs, slots = predict(extract_log_mel(audio, **feature_config))
            order = np.argsort(probs)[::-1][: len(labels) if args.debug else min(args.top_k, len(labels))]
            scores = "  ".join(f"{labels[i]}={probs[i]:.2f}" for i in order)
            top = labels[order[0]]
            if top in slots:
                scores += f"  [{top} slot: {slots[top][0]} ({slots[top][1]:.2f})]"
            if probs[order[0]] < args.reject_threshold:
                print(f"(didn't catch that, please repeat)  {scores}")
            else:
                print(scores)
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
