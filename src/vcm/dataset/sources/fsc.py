"""Fluent Speech Commands (FSC) loader and taxonomy coverage check.

The official distribution needs a Kaggle account or a Fluent.ai license
request; this uses a verified-complete third-party re-upload on Zenodo
instead (record 11106540, "Fluent speech commands dataset", CC BY 4.0,
uploaded by Afsara Benazir, University of Virginia) — checksum-verified
against the Zenodo API before trusting it (md5 625d5dfecef850443955a034d5f892b2,
1,545,730,387 bytes), and its `train/valid/test_data.csv` row counts
(23,132 + 3,118 + 3,793 = 30,043) match this project's own
already-cited FSC figure exactly. The archive also includes FSC's
official public license (PDF), consistent with a genuine full copy, not
a stripped-down mirror.

The label mapping below was built empirically against all 31 real
`(action, object, location)` combinations found across all three real
CSVs (not guessed from the paper) — see LABEL_MAPPING. One combination,
`("deactivate", "music", "none")`, mixes two different taxonomy labels
under one FSC category (confirmed by inspecting every transcription in
it: 315 say "pause"-style, 510 say "stop"/"turn off"-style, split is
100% clean, no ambiguous residue) — `_classify_deactivate_music()`
handles that one case by transcription text instead of the action/
object/location triple.

Real coverage found for 3 labels this taxonomy previously had **zero**
real coverage for: TEMPERATURE, STOP, PAUSE. `change language` and
`bring` (newspaper/shoes/socks/juice) commands have no taxonomy
equivalent and are dropped, not guessed.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vcm.dataset.manifest import ManifestRow

# (action, object, location) -> canonical label, verified against the
# real train/valid/test_data.csv (see module docstring).
LABEL_MAPPING: dict[tuple[str, str, str], str] = {
    ("increase", "volume", "none"): "VOLUME_UP",
    ("decrease", "volume", "none"): "VOLUME_DOWN",
    ("increase", "heat", "washroom"): "TEMPERATURE",
    ("decrease", "heat", "washroom"): "TEMPERATURE",
    ("increase", "heat", "none"): "TEMPERATURE",
    ("decrease", "heat", "none"): "TEMPERATURE",
    ("increase", "heat", "bedroom"): "TEMPERATURE",
    ("increase", "heat", "kitchen"): "TEMPERATURE",
    ("decrease", "heat", "bedroom"): "TEMPERATURE",
    ("decrease", "heat", "kitchen"): "TEMPERATURE",
    ("activate", "lights", "washroom"): "LIGHT_ON",
    ("activate", "lights", "kitchen"): "LIGHT_ON",
    ("activate", "lights", "bedroom"): "LIGHT_ON",
    ("activate", "lights", "none"): "LIGHT_ON",
    ("deactivate", "lights", "bedroom"): "LIGHT_OFF",
    ("deactivate", "lights", "kitchen"): "LIGHT_OFF",
    ("deactivate", "lights", "none"): "LIGHT_OFF",
    ("deactivate", "lights", "washroom"): "LIGHT_OFF",
    ("deactivate", "lamp", "none"): "LIGHT_OFF",
    ("activate", "lamp", "none"): "LIGHT_ON",
    ("activate", "music", "none"): "PLAY_MUSIC",
    # ("deactivate", "music", "none") is intentionally absent here —
    # handled by _classify_deactivate_music() instead, see docstring.
}

# Dropped (no taxonomy equivalent), listed for documentation, not used:
# "change language" (all objects), "bring" (newspaper/shoes/socks/juice).


@dataclass(frozen=True)
class FscRecord:
    path: str
    speaker_id: str
    transcription: str
    action: str
    object: str
    location: str
    split: str


def load_records(csv_path: Path, split: str) -> list[FscRecord]:
    with Path(csv_path).open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            FscRecord(
                path=row["path"],
                speaker_id=row["speakerId"],
                transcription=row["transcription"],
                action=row["action"],
                object=row["object"],
                location=row["location"],
                split=split,
            )
            for row in reader
        ]


def _classify_deactivate_music(transcription: str) -> str | None:
    t = transcription.lower()
    if "pause" in t:
        return "PAUSE"
    if "stop" in t or "off" in t:
        return "STOP"
    return None  # shouldn't happen, see docstring — verified 100% clean


def classify(record: FscRecord) -> str | None:
    """Return the canonical taxonomy label for one record, or None if it
    has no equivalent in this taxonomy (change language / bring object)."""
    if (record.action, record.object) == ("deactivate", "music"):
        return _classify_deactivate_music(record.transcription)
    return LABEL_MAPPING.get((record.action, record.object, record.location))


def action_object_location_counts(records: list[FscRecord]) -> Counter:
    return Counter((r.action, r.object, r.location) for r in records)


def coverage_report(records: list[FscRecord]) -> dict[str, int]:
    """For each canonical label, how many FSC utterances map to it."""
    counts: dict[str, int] = {}
    for r in records:
        label = classify(r)
        if label:
            counts[label] = counts.get(label, 0) + 1
    return counts


def to_manifest_rows(records: list[FscRecord], audio_root: Path | None = None) -> list[ManifestRow]:
    rows = []
    for r in records:
        label = classify(r)
        if label is None:
            continue
        audio_path = str(Path(audio_root) / r.path) if audio_root else r.path
        rows.append(
            ManifestRow(
                audio_path=audio_path,
                label=label,
                source="fsc",
                is_synthetic=False,
                speaker_id=f"fsc_{r.speaker_id}",
                split=r.split,
            )
        )
    return rows
