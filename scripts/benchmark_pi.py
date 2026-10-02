#!/usr/bin/env python
"""Measure what the models cost on this machine (run it on the Raspberry Pi):
per-command latency of the intent model, per-hop cost of the always-on
wake-word detector (and the share of one CPU core it keeps busy), model
file sizes and process memory. No microphone needed.

Reports mean, p50 and p95 latency and the real-time factor (RTF: processing
time divided by the length of the audio it covers). ONNX Runtime runs with
one thread (vcm.deploy.runtime). With --clips, the per-command numbers come
from real recordings (e.g. the master dataset's holdout WAVs) instead of a
synthetic 2.5 s command.

    python scripts/benchmark_pi.py
    python scripts/benchmark_pi.py --clips data/me2/holdout/audio --json bench_pi.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import soundfile as sf

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


def per_call_ms(fn, n: int) -> np.ndarray:
    fn()  # warm-up
    out = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        out[i] = (time.perf_counter() - t0) * 1000
    return out


def load_clips(folder: Path, limit: int) -> list[np.ndarray]:
    clips = []
    for path in sorted(folder.glob("*.wav"))[:limit]:
        audio, sr = sf.read(path, dtype="float32")
        if sr != SAMPLE_RATE:
            raise SystemExit(f"{path}: {sr} Hz, expected {SAMPLE_RATE}")
        clips.append(audio if audio.ndim == 1 else audio.mean(axis=1))
    if not clips:
        raise SystemExit(f"no .wav files in {folder}")
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--intent-model", type=Path, default=REPO_ROOT / "models/vcm_intent.onnx")
    parser.add_argument("--wake-model", type=Path, default=REPO_ROOT / "models/kiwi_wakeword.onnx")
    parser.add_argument("-n", type=int, default=50)
    parser.add_argument("--clips", type=Path, default=None, help="Folder of 16 kHz WAV commands to time end to end.")
    parser.add_argument("--json", type=Path, default=None, help="Also write the results here.")
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

    # End to end per command (features + model), on real clips if given.
    clips = load_clips(args.clips, 500) if args.clips else [command] * args.n
    e2e = np.array([per_call_ms(lambda c=c: intent.predict_audio(c), 1)[0] for c in clips])
    audio_s = np.array([len(c) / SAMPLE_RATE for c in clips])
    rtf = e2e / 1000 / audio_s
    wake = per_call_ms(lambda: score(extract_log_mel(window, SAMPLE_RATE, window_s=WINDOW_S, trim=False)[None]), args.n * 4)

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if os.uname().sysname == "Darwin" else 1024)
    print(f"machine: {os.uname().nodename} ({os.uname().machine}), {os.cpu_count()} cores")
    print(f"intent model  {args.intent_model.name}: {os.path.getsize(args.intent_model) / 1024:.0f} KB")
    print(f"  per command: features {feat_ms:.1f} ms + model {model_ms:.1f} ms = {feat_ms + model_ms:.1f} ms")
    print(f"wake model    {args.wake_model.name}: {os.path.getsize(args.wake_model) / 1024:.0f} KB")
    print(f"  per {HOP_S * 1000:.0f} ms hop: {hop_ms:.1f} ms (model {wmodel_ms:.1f} ms) -> {hop_ms / (HOP_S * 1000):.0%} of one core, always on")
    print(
        f"end to end per command ({'%d real clips' % len(clips) if args.clips else 'synthetic 2.5 s, n=%d' % len(clips)}): "
        f"mean {e2e.mean():.1f} ms, p50 {np.percentile(e2e, 50):.1f} ms, p95 {np.percentile(e2e, 95):.1f} ms; "
        f"RTF mean {rtf.mean():.4f}, p95 {np.percentile(rtf, 95):.4f}"
    )
    print(f"wake word per hop: p95 {np.percentile(wake, 95):.1f} ms, RTF {wake.mean() / (HOP_S * 1000):.3f} (one {HOP_S * 1000:.0f} ms hop)")
    print("onnxruntime: CPUExecutionProvider, 1 intra-op thread, 1 inter-op thread")
    print(f"peak process memory: {rss_mb:.0f} MB")
    if args.json:
        args.json.write_text(json.dumps({
            "machine": f"{os.uname().nodename} ({os.uname().machine})", "cores": os.cpu_count(),
            "intent_model": args.intent_model.name, "intent_model_kb": os.path.getsize(args.intent_model) / 1024,
            "clips": str(args.clips) if args.clips else "synthetic", "n": len(clips),
            "e2e_ms_mean": e2e.mean(), "e2e_ms_p50": np.percentile(e2e, 50), "e2e_ms_p95": np.percentile(e2e, 95),
            "rtf_mean": rtf.mean(), "rtf_p95": np.percentile(rtf, 95),
            "features_ms": feat_ms, "model_ms": model_ms,
            "wake_hop_ms_p95": np.percentile(wake, 95), "wake_core_share": hop_ms / (HOP_S * 1000),
            "onnxruntime_threads": 1, "peak_rss_mb": rss_mb,
        }, indent=2) + "\n")  # fmt: skip


if __name__ == "__main__":
    main()
