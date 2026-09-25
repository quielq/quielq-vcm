#!/usr/bin/env python
"""Record your own "hey kiwi" samples (and a few near-misses) for the wake-word
detector (EXPERIMENTS.md Experiment 34).

    python scripts/record_wakeword.py --speaker-id quielq            # 30 x "hey kiwi" + 12 near-misses, ~5 min
    python scripts/record_wakeword.py --speaker-id <friend> --reps 20  # more voices help most

Hold the spacebar (Mac) or button (Pi), say the prompt, release. Each prompt
asks for a different way of saying it (normal, quiet, fast, from across the
room...), because a wake word has to work however it's said. A take with no
speech in it is asked again.

The last third of each speaker's "hey kiwi" takes go to the test split: they
measure the real false-reject rate on a real voice
(`scripts/evaluate_wakeword.py --extra-wake-manifest ...`). The rest can be
added to training (`scripts/train_wakeword.py --extra-wake-manifest ...`).
Recordings accumulate in data/wakeword_real/ (manifest.csv, same columns as the
synthetic wake-word batch); run it again to add more.
"""

from __future__ import annotations

import argparse
import csv
import time
import wave
from pathlib import Path

import numpy as np

from vcm.audio.capture import SAMPLE_RATE, record_while_held
from vcm.audio.features import speech_level
from vcm.hal.button import get_button

REPO_ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["id", "audio_path", "label", "text", "slot_value", "speaker_id", "split", "whisper_text", "wer", "qa_pass"]
STYLES = [
    "normally",
    "a bit quieter",
    "quickly",
    "slowly",
    "from about 2 meters away",
    "as a question (Hey Kiwi?)",
    "while turned away from the mic",
    "louder, like from another room",
]
NEAR_MISSES = ["hey", "kiwi", "hey Siri", "hey Kevin", "hey kitty", "hey key", "hey, we need", "every week",
               "hey quickly", "okay", "the kiwi is ripe", "hey, can you hear me"]  # fmt: skip
MIN_LEVEL = 0.008  # same "anything said?" gate as the live demo


def save(audio: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument("--reps", type=int, default=30, help='number of "hey kiwi" takes')
    parser.add_argument("--negatives", type=int, default=len(NEAR_MISSES), help="number of near-miss takes")
    parser.add_argument("--test-fraction", type=float, default=1 / 3)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/wakeword_real")
    args = parser.parse_args()

    manifest = args.out / "manifest.csv"
    existing = []
    if manifest.exists():
        with manifest.open(newline="") as f:
            existing = list(csv.DictReader(f))
    n_test = round(args.reps * args.test_fraction)
    prompts = [("WAKE", "hey kiwi", STYLES[i % len(STYLES)], "test" if i >= args.reps - n_test else "train") for i in range(args.reps)]
    prompts += [("NOT_WAKE", NEAR_MISSES[i % len(NEAR_MISSES)], "normally", "train") for i in range(args.negatives)]

    button = get_button()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"{len(prompts)} takes. Hold the spacebar (or button), speak, release. Ctrl+C stops (takes so far are kept).\n")
    rows = []
    try:
        for i, (label, text, style, split) in enumerate(prompts, 1):
            while True:
                print(f'[{i}/{len(prompts)}] Say "{text}" {style}')
                audio = record_while_held(button)
                level = speech_level(audio) if len(audio) else 0.0
                if level >= MIN_LEVEL and 0.3 <= len(audio) / SAMPLE_RATE <= 6:
                    break
                print(f"   didn't catch that (speech level {level:.4f}, {len(audio) / SAMPLE_RATE:.1f}s): again")
            path = args.out / args.speaker_id / f"{label}_{stamp}_{i:03d}.wav"
            save(audio, path)
            rows.append({
                "id": path.stem, "audio_path": str(path.relative_to(REPO_ROOT)), "label": label, "text": text,
                "slot_value": "", "speaker_id": f"real_{args.speaker_id}", "split": split,
                "whisper_text": "", "wer": "", "qa_pass": "True",
            })  # fmt: skip
    except KeyboardInterrupt:
        print("\nStopped early.")
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing + rows)
    wake = [r for r in rows if r["label"] == "WAKE"]
    print(f"\nSaved {len(rows)} takes ({len(wake)} 'hey kiwi': {sum(r['split'] == 'test' for r in wake)} test) -> {manifest}")


if __name__ == "__main__":
    main()
