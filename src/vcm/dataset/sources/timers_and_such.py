"""Timers and Such (Lugosch et al., NeurIPS 2021 Datasets & Benchmarks) —
real human speech for the labels this project's own live testing found
weakest for having zero real coverage: TIMER (was 100% synthetic
Chatterbox TTS) and, to a lesser extent, ALARM.

Real audio only. Timers and Such also ships a much larger synthetic
portion, which isn't used here — the whole point of pulling this
source was to close a *real-audio* gap, not add another TTS voice's
worth of the exact thing already causing the problem.

Source: https://zenodo.org/records/4623772 (DOI 10.5281/zenodo.4623772),
Loren Lugosch, license "other-open". See the paper
(https://arxiv.org/abs/2104.01604) for the full dataset description.

The mapping below was built empirically, from the real per-split CSVs
this project downloaded and inspected (not from the paper's abstract):
SetTimer -> 545/92/80 real recordings (train/dev/test), ~71/11/10
unique real speakers; SetAlarm -> 273/41/40 real recordings, ~70/11/10
unique speakers. SimpleMath and UnitConversion have no equivalent in
this project's taxonomy and are dropped, not guessed.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

# raw Timers-and-Such intent -> canonical label. Confirmed by inspecting
# the real semantics column of every row in train/dev/test-real.csv —
# SimpleMath and UnitConversion have no taxonomy equivalent.
LABEL_MAPPING: dict[str, str] = {
    "SetTimer": "TIMER",
    "SetAlarm": "ALARM",
}

# Timers-and-Such's own split naming -> this project's manifest split
# naming (train/val/test, see manifest.py). Preserves their original
# speaker-disjoint split boundaries rather than re-shuffling.
SPLIT_NAME_MAP = {"train-real": "train", "dev-real": "val", "test-real": "test"}


@dataclass(frozen=True)
class TimersRecord:
    path: str  # relative path inside the archive, e.g. "train-real/<uuid>_prompt-181_0.wav"
    intent: str  # raw Timers-and-Such intent, e.g. "SetTimer"
    speaker_id: str
    transcription: str
    split: str  # already mapped to train/val/test


def load_records(meta_dir: Path, splits: tuple[str, ...] = ("train-real", "dev-real", "test-real")) -> list[TimersRecord]:
    """Load records from the real-speech metadata CSVs (train-real.csv etc.),
    filtered to only the intents this project's taxonomy actually covers.
    """
    records = []
    for split in splits:
        csv_path = Path(meta_dir) / f"{split}.csv"
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                semantics = ast.literal_eval(row["semantics"])
                intent = semantics["intent"]
                if intent not in LABEL_MAPPING:
                    continue
                records.append(
                    TimersRecord(
                        path=row["path"],
                        intent=intent,
                        speaker_id=row["speakerId"],
                        transcription=row["transcription"],
                        split=SPLIT_NAME_MAP[split],
                    )
                )
    return records


def coverage_report(records: list[TimersRecord]) -> dict[str, int]:
    """How many matched real recordings exist per canonical label."""
    report = {label: 0 for label in set(LABEL_MAPPING.values())}
    for r in records:
        report[LABEL_MAPPING[r.intent]] += 1
    return report
