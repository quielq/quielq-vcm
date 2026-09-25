#!/usr/bin/env python
"""Slot-value training/evaluation labels for every manifest row of a slotted
intent (TIMER, ALARM, BRIGHTNESS, COLOR; vcm.slots). EXPERIMENTS.md Experiment 32.

For each row, the text is the best available transcript: the intended text
for synthetic clips (Option B's `transcript` column, the targeted batches'
`text`), otherwise Whisper's transcript of the real audio
(data/cascade_transcripts.csv, Experiment 26). vcm.slots.parse_slot turns it
into a canonical value; rows whose value isn't in the vocabulary get no
label (masked in training, reported as out-of-vocabulary in evaluation).

Whisper-derived labels are noisy (it can mishear "fifteen"/"fifty"), so the
`text_source` column records which kind each label is, and evaluation
reports them separately.

Usage:
    python scripts/build_slot_labels.py [--manifest data/dataset_manifest_exp32.csv] [--out data/slot_labels.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

from vcm.dataset.manifest import read_manifest
from vcm.dataset.sources.targeted_synth import BATCHES
from vcm.slots import SLOT_VOCAB, parse_slot

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ground_truth_texts() -> dict[str, str]:
    texts = {}
    option_b = REPO_ROOT / "data/external/option_b/manifest.csv"
    if option_b.exists():
        with option_b.open(newline="") as f:
            for r in csv.DictReader(f):
                texts[os.path.basename(r["path"])] = r["transcript"]
    for batch in BATCHES.values():
        manifest = REPO_ROOT / batch["out_dir"] / "manifest.csv"
        if manifest.exists():
            with manifest.open(newline="") as f:
                for r in csv.DictReader(f):
                    texts[os.path.basename(r["audio_path"])] = r["text"]
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/dataset_manifest.csv")
    parser.add_argument("--transcripts", type=Path, default=REPO_ROOT / "data/cascade_transcripts.csv")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/slot_labels.csv")
    args = parser.parse_args()

    ground_truth = _ground_truth_texts()
    whisper = {}
    if args.transcripts.exists():
        with args.transcripts.open(newline="") as f:
            whisper = {r["audio_path"]: r["transcript"] for r in csv.DictReader(f)}

    out_rows, stats = [], Counter()
    for row in read_manifest(args.manifest):
        if row.label not in SLOT_VOCAB:
            continue
        base = os.path.basename(row.audio_path)
        if row.is_synthetic and base in ground_truth:
            text, text_source = ground_truth[base], "ground_truth"
        elif row.audio_path in whisper:
            text, text_source = whisper[row.audio_path], "whisper"
        else:
            stats[(row.label, "no text")] += 1
            continue
        value = parse_slot(row.label, text)
        stats[(row.label, "labeled" if value else "out of vocabulary")] += 1
        if value:
            out_rows.append(
                {"audio_path": row.audio_path, "label": row.label, "value": value, "text_source": text_source, "text": text}
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "label", "value", "text_source", "text"])
        writer.writeheader()
        writer.writerows(out_rows)

    for label in SLOT_VOCAB:
        print(f"{label:<11}" + "  ".join(f"{k}: {stats[(label, k)]}" for k in ("labeled", "out of vocabulary", "no text")))
    per_value = Counter((r["label"], r["value"]) for r in out_rows)
    thin = [(label, v, per_value[(label, v)]) for label in SLOT_VOCAB for v in SLOT_VOCAB[label] if per_value[(label, v)] < 20]
    print(f"values with < 20 labeled clips: {thin}")
    print(f"wrote {len(out_rows)} labels -> {args.out}")


if __name__ == "__main__":
    main()
