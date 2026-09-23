#!/usr/bin/env python
"""Standalone mic-to-intent demo for the ASR-cascade pipeline (see MODEL.md
Section 9): audio -> Whisper transcript -> text classifier -> intent.

The direct-audio alternative (scripts/demo_infer.py, a single trained
CNN) is still the lighter-weight path and stays available — this script
tests the higher-accuracy cascade (90.6% test accuracy on real audio vs.
75.45% for the best direct-audio checkpoint, EXPERIMENTS.md) live, the
same way demo_infer.py was used to validate the direct model against
real speech. Same taxonomy-reconciliation caveat as demo_infer.py: this
is a standalone sanity check, not wired into dispatch.py/main.py yet.

Requires `pip install faster-whisper scikit-learn joblib` (not default
dependencies) and a trained classifier at --classifier (produced by
scripts/train_cascade_classifier.py).

Usage:
    python scripts/demo_infer_cascade.py [--classifier checkpoints/cascade_classifier.joblib] [--whisper-size base]

Hold the spacebar to record (same push-to-talk mock main.py uses on a
Mac; on the RPi this uses the real HAL pushbutton), release to
transcribe+classify, repeat. Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

from vcm.audio.capture import SAMPLE_RATE, record_while_held
from vcm.dataset.qa.synthetic_check import FasterWhisperTranscriber
from vcm.hal.button import get_button

# Same rationale as demo_infer.py's SILENCE_RMS_THRESHOLD: skip
# transcription entirely for near-silent holds rather than feeding
# Whisper dead air (which can hallucinate text from silence).
SILENCE_RMS_THRESHOLD = 0.01


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", type=Path, default=Path("checkpoints/cascade_classifier.joblib"))
    parser.add_argument("--whisper-size", default="base")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=SILENCE_RMS_THRESHOLD,
        help="RMS amplitude below which captured audio is treated as silence and skipped "
        "without transcribing (0 disables this gate entirely)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save every recording as a WAV (debug_recordings/) and print the full "
        "per-class probability distribution instead of just the top-k — same purpose "
        "as demo_infer.py --debug.",
    )
    args = parser.parse_args()

    if args.debug:
        debug_dir = Path("debug_recordings")
        debug_dir.mkdir(exist_ok=True)

    print(f"Loading Whisper ({args.whisper_size})...")
    transcriber = FasterWhisperTranscriber(model_size=args.whisper_size)
    pipeline = joblib.load(args.classifier)
    labels = list(pipeline.classes_)
    print(f"Loaded cascade classifier from {args.classifier} ({len(labels)} labels)")

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
                wav_path = debug_dir / f"cascade_{time.strftime('%Y%m%d_%H%M%S')}_rms{rms:.4f}.wav"
                sf.write(wav_path, audio, SAMPLE_RATE)
                print(f"(saved {wav_path}, {len(audio)/SAMPLE_RATE:.2f}s, rms={rms:.4f})")
            if rms < args.silence_threshold:
                print(f"(silence, rms={rms:.4f} < {args.silence_threshold} — skipped)")
                continue

            transcript = transcriber.transcribe(wav_path if args.debug else _write_temp(audio))
            print(f'  heard: "{transcript}"')

            probs = pipeline.predict_proba([transcript])[0]
            k = len(labels) if args.debug else min(args.top_k, len(labels))
            top_idx = np.argsort(probs)[::-1][:k]
            print("  ".join(f"{labels[i]}={probs[i]:.2f}" for i in top_idx))
        except KeyboardInterrupt:
            print("\nExiting.")
            break


def _write_temp(audio: np.ndarray) -> Path:
    """Whisper's transcribe() takes a file path, not raw samples — write a
    scratch WAV for the non-debug case (--debug already wrote one)."""
    import tempfile

    tmp = Path(tempfile.gettempdir()) / "vcm_cascade_scratch.wav"
    sf.write(tmp, audio, SAMPLE_RATE)
    return tmp


if __name__ == "__main__":
    main()
