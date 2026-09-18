"""Loader for classmate Mark Andrian Macalalad's "Option B" synthetic dataset.

Source: https://github.com/markandrian30/AI231/tree/main/MEX2/OptionB —
17,658 QA-filtered synthetic recordings (voice-cloned via Chatterbox TTS
from 100 real reference speakers: 84 foreign/LibriSpeech, 16 real
Filipino-English/SilencioPH), speaker-disjoint train/val/test split
(80/10/10 speakers), clean + noisy acoustic conditions, screened through
Anthony Navarez's `simple-audio-transcriber` (942/18,600 originals
flagged and removed).

Unlike slurp.py and fsc.py, no label mapping is needed here: this
dataset was generated directly from the same 19-label fixed-vs-slotted
taxonomy this project already uses (see sources/dataset_schema.py), so
its `intent` column already matches our canonical labels 1:1.

This closes every one of the 6 labels SLURP had zero coverage for
(PAUSE, STOP, NEXT, CALL, TIMER, TEMPERATURE) — see slurp.py's module
docstring for that gap.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from vcm.dataset.manifest import ManifestRow


@dataclass(frozen=True)
class OptionBRecord:
    path: str
    label: str  # folder-level label, includes slot value e.g. "ALARM_6_00AM"
    intent: str  # canonical taxonomy label, e.g. "ALARM"
    speaker: str
    split: str
    variant_id: str  # "clean" or "noisy" acoustic condition
    transcript: str
    slot_value: str


def load_manifest(csv_path: Path) -> list[OptionBRecord]:
    with Path(csv_path).open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            OptionBRecord(
                path=row["path"],
                label=row["label"],
                intent=row["intent"],
                speaker=row["speaker"],
                split=row["split"],
                variant_id=row["variant_id"],
                transcript=row["transcript"],
                slot_value=row["slot_value"],
            )
            for row in reader
        ]


def to_manifest_rows(records: list[OptionBRecord], audio_root: Path | None = None) -> list[ManifestRow]:
    """Convert to the project's common manifest schema (see manifest.py).

    `audio_root` is where OptionB's folder structure lives locally (the
    repo's MEX2/OptionB directory, if cloned) — audio_path is left as the
    relative path from the CSV if audio_root is omitted.
    """
    rows = []
    for r in records:
        audio_path = str(Path(audio_root) / r.path) if audio_root else r.path
        rows.append(
            ManifestRow(
                audio_path=audio_path,
                label=r.intent,
                source="mark_option_b",
                is_synthetic=True,
                speaker_id=r.speaker,
                split=r.split,
            )
        )
    return rows


def label_counts(records: list[OptionBRecord]) -> dict[str, int]:
    """How many recordings exist per canonical taxonomy label."""
    counts: dict[str, int] = {}
    for r in records:
        counts[r.intent] = counts.get(r.intent, 0) + 1
    return counts
