#!/usr/bin/env python
"""Measure memory and per-command latency of one voice pipeline, in a fresh
process, stage by stage (FOOTPRINT_COMPARISON.md). Run each pipeline in its
own process so memory numbers don't mix:

    # our pipeline: int8 ONNX intent model + wake-word model
    python scripts/measure_footprint.py ours --intent-model models/vcm_intent.int8.onnx --wake-model models/kiwi_wakeword.int8.onnx --clips <dir of 16 kHz wavs>

    # the ASR cascade (Experiment 26): faster-whisper + TF-IDF classifier
    python scripts/measure_footprint.py asr --compute-type int8 --threads 4 --clips <dir of 16 kHz wavs>

"peak" is the reliable memory number: macOS compresses idle memory, so
per-stage values can even go down.
"""

from __future__ import annotations

import argparse
import glob
import os
import resource
import subprocess
import sys
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def rss_mb() -> float:
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) / 1024


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)


def read_wav(path: str):
    import numpy as np

    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype("float32") / 32768


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", choices=["ours", "asr"])
    parser.add_argument("--clips", required=True, help="directory of 16 kHz mono 16-bit wav commands")
    parser.add_argument("--intent-model", type=Path, default=REPO_ROOT / "models/vcm_intent.int8.onnx")
    parser.add_argument("--wake-model", type=Path, default=REPO_ROOT / "models/kiwi_wakeword.int8.onnx")
    parser.add_argument("--whisper-size", default="base")
    parser.add_argument("--compute-type", default="default", help="faster-whisper compute_type (default, int8, ...)")
    parser.add_argument("--threads", type=int, default=4, help="faster-whisper cpu_threads")
    parser.add_argument("--classifier", type=Path, default=REPO_ROOT / "checkpoints/cascade_classifier.joblib")
    args = parser.parse_args()

    stages = [("python start", rss_mb())]
    import numpy as np

    stages.append(("+ numpy", rss_mb()))
    files = sorted(glob.glob(os.path.join(args.clips, "*.wav")))
    if args.pipeline == "ours":
        import onnxruntime  # noqa: F401

        stages.append(("+ onnxruntime", rss_mb()))
        from vcm.deploy.runtime import OnnxIntentModel
        from vcm.wakeword.detector import WakeWordDetector, onnx_scorer

        intent = OnnxIntentModel(args.intent_model)
        stages.append(("+ intent model loaded", rss_mb()))
        detector = WakeWordDetector(onnx_scorer(args.wake_model), threshold=0.9)
        stages.append(("+ wake-word model loaded", rss_mb()))
        rng = np.random.default_rng(0)
        for _ in range(300):  # 30 s of streaming through the wake word
            detector.feed((0.05 * rng.standard_normal(1600)).astype("float32"))

        def classify(audio):
            return intent.predict_audio(audio).intent

    else:
        from faster_whisper import WhisperModel

        stages.append(("+ faster-whisper / CTranslate2", rss_mb()))
        kwargs = {} if args.compute_type == "default" else {"compute_type": args.compute_type}
        whisper = WhisperModel(args.whisper_size, device="cpu", cpu_threads=args.threads, **kwargs)
        stages.append((f"+ whisper-{args.whisper_size} loaded", rss_mb()))
        import joblib

        classifier = joblib.load(args.classifier)
        stages.append(("+ text classifier (scikit-learn)", rss_mb()))

        def classify(audio):
            segments, _ = whisper.transcribe(audio, language="en")
            return classifier.predict([" ".join(s.text for s in segments).strip()])[0]

    times, preds = [], []
    for path in files:
        audio = read_wav(path)
        t0 = time.perf_counter()
        preds.append(classify(audio))
        times.append((time.perf_counter() - t0) * 1000)
    stages.append((f"steady state (after {len(files)} commands)", rss_mb()))

    prev = 0.0
    for name, value in stages:
        print(f"{name:<42}{value:8.1f} MB  ({value - prev:+.1f})")
        prev = value
    print(f"{'peak':<42}{peak_mb():8.1f} MB")
    steady = times[1:] or times
    print(f"latency per command: median {np.median(steady):.1f} ms, max {max(steady):.1f} ms (first call {times[0]:.1f} ms)")
    print("predictions:", " ".join(f"{Path(f).stem}={p}" for f, p in zip(files, preds)))


if __name__ == "__main__":
    main()
