#!/usr/bin/env python
"""Offline dress rehearsal of the class's live benchmark
(github.com/airimonda/vcm-benchmark) for our device pipeline.

The live benchmark plays "wake word + pause + command" from a laptop to the
Pi, reads the line our listener prints, and scores it. This script builds the
same trial audio with the benchmark's own code (holdout loader, trim,
loudness levelling, wake-word patching with the same pause), runs it through
our real listener pipeline without a microphone (streaming wake-word
detector, scripts/vcm_listen.py's record_command and first_speech_span, the
intent + slot model) and prints our result line, then parses and scores those
lines with the benchmark's own parser and report code. What it can't
reproduce: the room, the laptop speaker and the Pi microphone, so expect the
live numbers to be lower, mostly in wake detection.

Also runs every in-scope holdout command *without* the wake word, as the
benchmark's false-wake check does (it uses 10; this uses all of them).

Usage:
    git clone https://github.com/airimonda/vcm-benchmark /path/to/vcm-benchmark
    python scripts/class_benchmark_offline.py --bench-dir /path/to/vcm-benchmark \\
        --holdout data/me2/hf/data/holdout-00000-of-00001.parquet --out results/class_benchmark_offline
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import vcm_listen  # noqa: E402
from vcm.audio.capture import SAMPLE_RATE  # noqa: E402
from vcm.deploy.runtime import OnnxIntentModel  # noqa: E402
from vcm.wakeword.data import _read  # noqa: E402
from vcm.wakeword.detector import WakeWordDetector, onnx_scorer  # noqa: E402

CHUNK = int(vcm_listen.CHUNK_S * SAMPLE_RATE)


def device(stream: np.ndarray, model, scorer, threshold: float, chime_chunks: int) -> list[str]:
    """Our listener's loop over a recording instead of a microphone: the
    result lines it would print (one per wake-word trigger)."""
    detector = WakeWordDetector(scorer, threshold=threshold)
    chunks = [stream[i : i + CHUNK] for i in range(0, len(stream) - CHUNK + 1, CHUNK)]
    lines, levels, i = [], [], 0
    while i < len(chunks):
        chunk = chunks[i]
        i += 1
        levels = [*levels[-29:], vcm_listen.speech_level(chunk)]
        if not detector.feed(chunk):
            continue
        floor = vcm_listen.background_floor(levels)
        # The listener plays its chime here (blocking) and then looks at what the
        # mic heard meanwhile: kept only if speech is still going on after it.
        lead_in = np.zeros(0, dtype="float32")
        if chime_chunks:
            during, i = chunks[i : i + chime_chunks], i + chime_chunks
            if during and i < len(chunks):
                lead_in = vcm_listen.chime_lead_in(during, chunks[i], floor)
        q: queue.Queue = queue.Queue()
        for c in chunks[i:]:
            q.put(c)
        audio = vcm_listen.record_command(q, lead_in=lead_in, floor=floor)
        i += (len(audio) - len(lead_in)) // CHUNK
        command = vcm_listen.first_speech_span(audio)
        if vcm_listen.speech_level(command) < vcm_listen.SILENCE_THRESHOLD:
            lines.append(vcm_listen.result_line(None, 0.0, len(command) / SAMPLE_RATE * 1000))
        else:
            t0 = time.perf_counter()
            pred = model.predict_audio(command)
            lines.append(vcm_listen.result_line(pred, (time.perf_counter() - t0) * 1000, len(command) / SAMPLE_RATE * 1000))
        detector.reset()
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench-dir", type=Path, required=True, help="a clone of airimonda/vcm-benchmark")
    parser.add_argument("--holdout", default="hf", help="holdout parquet (default: the benchmark's own download)")
    parser.add_argument("--intent-model", type=Path, default=REPO_ROOT / "models/vcm_intent.onnx")
    parser.add_argument("--wake-model", type=Path, default=REPO_ROOT / "models/kiwi_wakeword.onnx")
    parser.add_argument("--wake-manifest", type=Path, default=REPO_ROOT / "data/wakeword_real/manifest.csv",
                        help="the author's held-out (test split) 'hey kiwi' takes stand in for the 3 recorded ones")
    parser.add_argument("--wake-threshold", type=float, default=0.6)
    parser.add_argument("--gap", type=float, default=0.8, help="pause between wake word and command (benchmark default)")
    parser.add_argument("--noise-dbfs", type=float, default=-55.0, help="steady room noise added to every stream")
    parser.add_argument("--no-chime", action="store_true", help="as vcm_listen.py --no-chime: no chime after the wake word")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/class_benchmark_offline")
    args = parser.parse_args()

    sys.path.insert(0, str(args.bench_dir))
    from vcmbench import audio as A  # the benchmark's own code
    from vcmbench import report as R
    from vcmbench import schema as S
    from vcmbench.dataset import load_holdout
    from vcmbench.pi import LineParser

    clips = load_holdout(args.holdout, args.bench_dir / ".cache")
    with args.wake_manifest.open(newline="") as f:
        takes = [_read(r["audio_path"]) for r in csv.DictReader(f) if r["label"] == "WAKE" and r["split"] == "test"][:3]
    takes = [A.normalize(A.trim(t)) for t in takes]
    model, scorer = OnnxIntentModel(args.intent_model), onnx_scorer(args.wake_model)
    chime_s = 0.0 if args.no_chime else len(vcm_listen.make_chime(SAMPLE_RATE)) / SAMPLE_RATE
    chime_chunks = int(np.ceil(chime_s / vcm_listen.CHUNK_S))
    rng = np.random.default_rng(args.seed)
    noise_rms = 10 ** (args.noise_dbfs / 20)

    def with_room(x: np.ndarray) -> np.ndarray:
        pad = lambda s: np.zeros(int(s * SAMPLE_RATE), dtype="float32")  # noqa: E731
        x = np.concatenate([pad(1.0), x.astype("float32"), pad(vcm_listen.MAX_COMMAND_S + 3.0)])
        return x + (noise_rms * rng.standard_normal(len(x))).astype("float32")

    parse, aliases, order = LineParser(), S.build_alias_table(), S.id_orders()["manifest"]
    trials, raw_rows = [], []
    for kind in ("wake", "no_wake"):
        for k, c in enumerate(clips):
            if kind == "no_wake" and c.intent == S.OOS:
                continue
            cmd = A.normalize(A.trim(c.audio))
            take = takes[k % len(takes)] if kind == "wake" else np.zeros(0, dtype="float32")
            x, _ = A.patch(take, cmd, args.gap if kind == "wake" else 0.0)
            lines = device(with_room(x), model, scorer, args.wake_threshold, chime_chunks)
            events = [e for e in (parse.parse(line, 0.0) for line in lines) if e is not None and e.kind == "command"]
            t = {"order": len(trials), "kind": kind, "clip_idx": c.idx, "transcript": c.transcript,
                 "true_intent": c.intent, "true_variation": c.variation, "true_slot": c.slot_value,
                 "speaker_id": c.speaker_id, "is_synthetic": c.is_synthetic, "accent_group": c.accent_group,
                 "n_command_events": len(events), "wake_logged": bool(lines)}
            if events:
                e = events[0]
                intent, slot, _, variation = S.resolve_prediction(e.intent, e.slot, e.variation, aliases, order)
                t.update(pred_intent=intent, pred_slot=slot, pred_variation=variation, pred_variation_id=None,
                         pred_raw=e.raw, infer_ms=e.infer_ms, audio_ms=e.audio_ms, latency_s=None)
            else:
                t.update(pred_intent=S.NONE, pred_slot="", pred_variation="", pred_variation_id=None,
                         pred_raw="", infer_ms=None, audio_ms=None, latency_s=None)
            trials.append(t)
            raw_rows.append({"kind": kind, "transcript": c.transcript, "true": f"{c.intent} {c.slot_value}".strip(),
                             "lines": " | ".join(lines)})

    # The benchmark scores 10 no-wake trials (one per intent, chosen by seed); report all of them here.
    random.Random(args.seed).shuffle(trials)
    m = R.score(trials, [], {}, None, 0.0, 1.0, {"mode": "offline dress rehearsal", "holdout_clips": len(clips)})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(R.render_markdown(m), encoding="utf-8")
    (args.out / "metrics.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    with (args.out / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
        w.writeheader()
        w.writerows(raw_rows)
    o = m["breakdowns"]["overall"]
    print(f"holdout clips: {len(clips)}; wake trials {sum(t['kind'] == 'wake' for t in trials)}, "
          f"no-wake trials {sum(t['kind'] == 'no_wake' for t in trials)}")
    print(f"intent accuracy {o['accuracy']:.2%}  command accuracy {o['command_accuracy']:.2%}  "
          f"false accept {o['false_accept_rate']:.2%}  false reject {o['false_reject_rate']:.2%}  "
          f"misfire {o['misfire_rate']:.2%}  slot exact {o['slot_exact_rate']:.2%}  "
          f"false wake {m['false_wake']['false_wake_rate']:.2%} ({m['false_wake']['false_wakes']}/{m['false_wake']['n']})")
    print(f"pipeline: {m['pipeline']}")
    print(f"wrote {args.out}/report.md, metrics.json, trials.csv")


if __name__ == "__main__":
    main()
