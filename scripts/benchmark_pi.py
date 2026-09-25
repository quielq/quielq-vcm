#!/usr/bin/env python
"""Measure what the models cost on this machine (run it on the Raspberry Pi):
per-command latency of the intent model, per-hop cost of the always-on
wake-word detector (and the share of one CPU core it keeps busy), model
file sizes and process memory. No microphone needed.

    python scripts/benchmark_pi.py
"""

from __future__ import annotations

import argparse
import os
import resource
import time
from pathlib import Path

import numpy as np

from vcm.audio.capture import SAMPLE_RATE
from vcm.audio.features import extract_log_mel
from vcm.deploy.runtime import OnnxIntentModel
from vcm.wakeword import HOP_S, WINDOW_S
from vcm.wakeword.detector import onnx_scorer

REPO_ROOT = Path(__file__).resolve().parents[1]


def timed(fn, n: int) -> float:
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--intent-model", type=Path, default=REPO_ROOT / "models/vcm_intent.int8.onnx")
    parser.add_argument("--wake-model", type=Path, default=REPO_ROOT / "models/kiwi_wakeword.int8.onnx")
    parser.add_argument("-n", type=int, default=50)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    command = (0.05 * rng.standard_normal(int(2.5 * SAMPLE_RATE))).astype("float32")
    window = (0.05 * rng.standard_normal(int(WINDOW_S * SAMPLE_RATE))).astype("float32")

    intent = OnnxIntentModel(args.intent_model)
    features = extract_log_mel(command, SAMPLE_RATE, **intent.feature_config)
    feat_ms = timed(lambda: extract_log_mel(command, SAMPLE_RATE, **intent.feature_config), args.n)
    model_ms = timed(lambda: intent.probabilities(features), args.n)

    score = onnx_scorer(args.wake_model)
    wfeat = extract_log_mel(window, SAMPLE_RATE, window_s=WINDOW_S, trim=False)
    hop_ms = timed(lambda: score(extract_log_mel(window, SAMPLE_RATE, window_s=WINDOW_S, trim=False)[None]), args.n)
    wmodel_ms = timed(lambda: score(wfeat[None]), args.n)

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if os.uname().sysname == "Darwin" else 1024)
    print(f"machine: {os.uname().nodename} ({os.uname().machine}), {os.cpu_count()} cores")
    print(f"intent model  {args.intent_model.name}: {os.path.getsize(args.intent_model) / 1024:.0f} KB")
    print(f"  per command: features {feat_ms:.1f} ms + model {model_ms:.1f} ms = {feat_ms + model_ms:.1f} ms")
    print(f"wake model    {args.wake_model.name}: {os.path.getsize(args.wake_model) / 1024:.0f} KB")
    print(f"  per {HOP_S * 1000:.0f} ms hop: {hop_ms:.1f} ms (model {wmodel_ms:.1f} ms) -> {hop_ms / (HOP_S * 1000):.0%} of one core, always on")
    print(f"peak process memory: {rss_mb:.0f} MB")


if __name__ == "__main__":
    main()
