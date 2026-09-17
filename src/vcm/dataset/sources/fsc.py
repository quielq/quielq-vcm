"""Fluent Speech Commands (FSC) metadata loader and taxonomy coverage check.

Unlike slurp.py, this is **not wired to a live download**: FSC's official
distribution requires a Kaggle account/API key (or a Fluent.ai license
request) this environment doesn't have, and I'd rather not hardcode FSC's
label list from memory and risk getting it subtly wrong (the action/
object/location combinations aren't something I could independently
verify here the way SLURP's intents were verified against the real
downloaded jsonl). Point this module at a local copy of FSC's
`train_data.csv` (columns: path, speakerId, transcription, action,
object, location) after downloading it yourself, and it computes the
same kind of coverage report slurp.py does.

Once real coverage numbers exist, replace LABEL_MAPPING below with a
mapping verified against actual `action`/`object`/`location` values seen
in the file, the same way slurp.py's mapping was built empirically rather
than assumed.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Placeholder — verify against a real train_data.csv before trusting this.
# Section 9 already reports FSC's qualitative gaps (no weather/timer/
# alarm/reminders/calls); this mapping only covers what plausibly exists
# in its documented smart-home action/object vocabulary.
LABEL_MAPPING: dict[str, list[tuple[str, str]]] = {
    # (action, object) pairs -> canonical label
    "LIGHT_ON": [("activate", "lights")],
    "LIGHT_OFF": [("deactivate", "lights")],
    "VOLUME_UP": [("increase", "volume")],
    "VOLUME_DOWN": [("decrease", "volume")],
}


@dataclass(frozen=True)
class FscRecord:
    action: str
    object: str
    location: str
    transcription: str


def load_records(csv_path: Path) -> list[FscRecord]:
    with Path(csv_path).open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            FscRecord(
                action=row["action"],
                object=row["object"],
                location=row["location"],
                transcription=row["transcription"],
            )
            for row in reader
        ]


def action_object_counts(records: list[FscRecord]) -> Counter:
    return Counter((r.action, r.object) for r in records)


def coverage_report(
    records: list[FscRecord], mapping: dict[str, list[tuple[str, str]]] = LABEL_MAPPING
) -> dict[str, int]:
    """For each canonical label, how many FSC utterances map to it."""
    counts = action_object_counts(records)
    return {label: sum(counts.get(pair, 0) for pair in pairs) for label, pairs in mapping.items()}
