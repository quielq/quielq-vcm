#!/usr/bin/env python
"""Record real human audio for the specific phrases that currently have
no (or effectively no) real-speaker coverage in the training data.

Why this exists: live testing (see EXPERIMENTS.md and the project chat
log) found CALL, NEXT, TIMER, and LIST_REMINDERS are 100% Chatterbox
TTS with zero real-human recordings — the model has never heard an
actual person say these phrases, only TTS-cloned voices. Two more
specific phrasings ("Kill the lights" for LIGHT_OFF, "Message" for
MESSAGE) are technically covered but only by the same TTS voices, in
labels otherwise dominated by different real phrasing (FSC/SLURP style
"turn off the lights", full email sentences). This is the textbook
synthetic-to-real generalization gap: a classmate's own prior
experiment found that adding real recordings to a synthetic-only label
(CALL) measurably helped (see VCM_Architecture_Review.md Section 9) —
this script generalizes that same fix to the rest of the same gap.

Runs entirely on your local machine (Mac or RPi), using the same
push-to-talk capture as scripts/demo_infer.py and vcm.main. The more
people who run this with a few minutes each, the more speaker
diversity the fix gets — see the "TTS mode collapse" research finding
in the project chat log: speaker diversity in training data is what
actually closes the synthetic-to-real gap, not just more of one voice.

Usage:
    python scripts/record_real_examples.py --speaker-id quielq --reps 10

Writes real .wav files to data/real_recordings/<split>/<LABEL>/ and a
manifest.csv in this project's common schema, picked up automatically
by scripts/build_manifest.py (skipped if this directory doesn't exist,
same as every other optional source).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import soundfile as sf  # noqa: E402

from vcm.audio.capture import SAMPLE_RATE, record_while_held  # noqa: E402
from vcm.hal.button import get_button  # noqa: E402

# (label, phrase) pairs with no, or effectively no, real-human coverage —
# see this file's own docstring and EXPERIMENTS.md for how each was found.
TARGET_PHRASES: list[tuple[str, str]] = [
    ("CALL", "Call"),
    ("CALL", "Make a call"),
    ("CALL", "Make a phone call"),
    ("NEXT", "Skip song"),
    ("NEXT", "Next song"),
    ("NEXT", "Play next song"),
    ("TIMER", "Timer 10 seconds"),
    ("TIMER", "Timer 30 seconds"),
    ("TIMER", "Timer 1 minute"),
    ("TIMER", "Countdown for 10 seconds"),
    ("TIMER", "Countdown for 30 seconds"),
    ("TIMER", "Countdown for 1 minute"),
    ("TIMER", "Start a timer for 10 seconds"),
    ("TIMER", "Start a timer for 30 seconds"),
    ("TIMER", "Start a timer for 1 minute"),
    ("LIST_REMINDERS", "Reminders"),
    ("LIST_REMINDERS", "Show my reminders"),
    ("LIST_REMINDERS", "List my reminders"),
    ("LIGHT_OFF", "Kill the lights"),
    ("MESSAGE", "Message"),
]

MANIFEST_FIELDS = ["audio_path", "label", "source", "is_synthetic", "speaker_id", "split"]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker-id", required=True, help="Your name/handle, for speaker-disjoint splits later")
    parser.add_argument("--reps", type=int, default=10, help="How many times to repeat each phrase")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/real_recordings")
    args = parser.parse_args()

    button = get_button()
    manifest_path = args.out_dir / "manifest.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    if manifest_path.exists():
        with manifest_path.open() as f:
            existing_rows = list(csv.DictReader(f))

    print(f"Recording as speaker_id={args.speaker_id!r}, {args.reps} reps per phrase, split={args.split!r}")
    print(f"{len(TARGET_PHRASES)} phrases, {len(TARGET_PHRASES) * args.reps} recordings total.")
    print("Hold spacebar (Mac) or the pushbutton (RPi), say the phrase naturally, release. Ctrl+C to stop early.\n")

    new_rows = []
    try:
        for label, phrase in TARGET_PHRASES:
            label_dir = args.out_dir / args.split / label
            label_dir.mkdir(parents=True, exist_ok=True)
            for rep in range(1, args.reps + 1):
                print(f'[{label}] Say: "{phrase}"  (rep {rep}/{args.reps})')
                audio = record_while_held(button)
                if len(audio) == 0:
                    print("  (no audio captured, retrying this rep)")
                    continue
                out_path = label_dir / f"{_slug(phrase)}_{args.speaker_id}_{rep}.wav"
                sf.write(out_path, audio, SAMPLE_RATE)
                new_rows.append(
                    {
                        "audio_path": str(out_path.relative_to(REPO_ROOT)),
                        "label": label,
                        "source": "real_recordings",
                        "is_synthetic": False,
                        "speaker_id": args.speaker_id,
                        "split": args.split,
                    }
                )
    except KeyboardInterrupt:
        print("\nStopped early — keeping what was recorded so far.")

    all_rows = existing_rows + new_rows
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nRecorded {len(new_rows)} new clips this session ({len(all_rows)} total) -> {manifest_path}")


if __name__ == "__main__":
    main()
