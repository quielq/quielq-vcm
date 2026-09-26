#!/usr/bin/env python
"""Hands-free VCM loop: "Hey Kiwi" -> command -> intent + slot value.

Runs the same way on the Raspberry Pi (USB mic, over SSH) and on a laptop
(built-in mic), using only the exported ONNX models (no torch):

    python scripts/vcm_listen.py
    python scripts/vcm_listen.py --trigger button     # push-to-talk instead (spacebar / GPIO 17)

After the wake word fires, it records until you stop talking (0.7 s below
the speech level, or 5 s max), then classifies. Every result prints with
its timing, so latency can be read straight off an SSH session.
"""

from __future__ import annotations

import argparse
import queue
import time
from pathlib import Path

import numpy as np

from vcm.audio.capture import SAMPLE_RATE
from vcm.audio.features import speech_level
from vcm.deploy.runtime import OnnxIntentModel
from vcm.wakeword.detector import WakeWordDetector, onnx_scorer

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK_S = 0.1
END_SILENCE_S = 0.7
MAX_COMMAND_S = 5.0
SILENCE_THRESHOLD = 0.008  # same gate as scripts/demo_infer.py (loudest 300ms)
REJECT_THRESHOLD = 0.6  # same as scripts/demo_infer.py


def record_command(chunks: "queue.Queue[np.ndarray]", lead_in: np.ndarray) -> np.ndarray:
    """Collect mic chunks until END_SILENCE_S of quiet after some speech, or MAX_COMMAND_S."""
    audio = [lead_in]
    heard_speech, quiet_s, total_s = False, 0.0, 0.0
    while total_s < MAX_COMMAND_S:
        chunk = chunks.get()
        audio.append(chunk)
        total_s += len(chunk) / SAMPLE_RATE
        loud = speech_level(chunk) >= SILENCE_THRESHOLD
        heard_speech |= loud
        quiet_s = 0.0 if loud else quiet_s + len(chunk) / SAMPLE_RATE
        if heard_speech and quiet_s >= END_SILENCE_S:
            break
    return np.concatenate(audio)


def report(model: OnnxIntentModel, audio: np.ndarray, t_end: float) -> None:
    level = speech_level(audio)
    if level < SILENCE_THRESHOLD:
        print(f"  (silence, speech level={level:.4f} — skipped)")
        return
    t0 = time.perf_counter()
    pred = model.predict_audio(audio)
    ms = (time.perf_counter() - t0) * 1000
    slot = f"  {pred.slot_value} ({pred.slot_confidence:.2f})" if pred.slot_value else ""
    verdict = "didn't catch that, please repeat" if pred.confidence < REJECT_THRESHOLD else "->"
    print(f"  {verdict} {pred.intent} ({pred.confidence:.2f}){slot}   [model {ms:.0f} ms, {time.perf_counter() - t_end:.2f} s after end of command]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--intent-model", type=Path, default=REPO_ROOT / "models/vcm_intent.int8.onnx")
    parser.add_argument("--wake-model", type=Path, default=REPO_ROOT / "models/kiwi_wakeword.int8.onnx")
    parser.add_argument("--wake-threshold", type=float, default=0.9, help="from scripts/evaluate_wakeword.py's suggestion")
    parser.add_argument("--trigger", choices=["wakeword", "button"], default="wakeword")
    parser.add_argument("--device", default=None, help="sounddevice input device (name or index); default: system default")
    parser.add_argument("--show-scores", action="store_true", help="print the wake-word score continuously (tuning)")
    args = parser.parse_args()

    import sounddevice as sd

    model = OnnxIntentModel(args.intent_model)
    print(f"intent model: {args.intent_model.name} ({len(model.labels)} intents, slots: {', '.join(model.slot_vocab) or 'none'})")

    if args.trigger == "button":
        from vcm.audio.capture import record_while_held
        from vcm.hal.button import get_button

        button = get_button()
        print("Hold the button (spacebar on a Mac) and speak. Ctrl+C to quit.")
        while True:
            audio = record_while_held(button)
            report(model, audio, time.perf_counter())

    detector = WakeWordDetector(onnx_scorer(args.wake_model), threshold=args.wake_threshold)
    chunks: queue.Queue[np.ndarray] = queue.Queue()
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(CHUNK_S * SAMPLE_RATE),
        device=args.device,
        callback=lambda indata, frames, t, status: chunks.put(indata[:, 0].copy()),
    )
    print(f'Say "Hey Kiwi", then your command. (wake threshold {args.wake_threshold}; Ctrl+C to quit)')
    with stream:
        try:
            while True:
                chunk = chunks.get()
                if detector.feed(chunk):
                    print(f"\n[wake word, score {detector.last_score:.2f}] listening...")
                    t_wake = time.perf_counter()
                    audio = record_command(chunks, lead_in=np.zeros(0, dtype="float32"))
                    report(model, audio, time.perf_counter())
                    print(f"  ({time.perf_counter() - t_wake:.1f} s from wake word to result)")
                    detector.reset()
                    while not chunks.empty():  # drop audio queued while classifying
                        chunks.get_nowait()
                elif args.show_scores:
                    print(f"\rwake score {detector.last_score:.2f}", end="", flush=True)
        except KeyboardInterrupt:
            print("\nExiting.")


if __name__ == "__main__":
    main()
