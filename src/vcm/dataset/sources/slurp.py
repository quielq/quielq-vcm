"""SLURP metadata loader and taxonomy coverage check.

Turns Section 9's qualitative "SLURP is the closest match in breadth"
claim into exact counts against the class's fixed-vs-slotted taxonomy
(see dataset_schema.py). Metadata only, no audio download: the
annotation jsonl files (~13-14MB total) are enough to answer "does this
source cover this label, and how much."

The label mapping below was built empirically, by downloading the real
train/devel/test.jsonl from github.com/pswietojanski/slurp and inspecting
every distinct `intent` value that actually occurs (93 of them) and every
`scenario` (18 of them) — not from memory or the paper's abstract. Several
of the taxonomy's 19 labels have **no SLURP match at all**: SLURP has no
"calls" scenario (CALL, MESSAGE has only a weak email-sendemail proxy), no
timer domain distinct from alarms (TIMER), and no thermostat/temperature
domain within its "iot" scenario (TEMPERATURE). These are genuine gaps,
not omissions in the mapping below.
"""

# Acknowledgment: the taxonomy this checks coverage against was shared by
# a classmate as part of the class's collective dataset effort — see
# DATASET.md's Acknowledgments section.

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# canonical_label -> SLURP intents that map to it (empirically verified
# against the real, downloaded jsonl files; empty list = confirmed no match)
LABEL_MAPPING: dict[str, list[str]] = {
    "PLAY_MUSIC": ["play_music"],
    "WEATHER": ["weather_query"],
    "TIME": ["datetime_query"],  # mixed with date queries, not time-only
    "LIGHT_ON": ["iot_hue_lighton"],
    "LIGHT_OFF": ["iot_hue_lightoff", "hue_lightoff"],
    "PAUSE": [],
    "STOP": [],
    "NEXT": [],
    "VOLUME_UP": ["audio_volume_up"],
    "VOLUME_DOWN": ["audio_volume_down"],
    "CALL": [],
    "MESSAGE": ["email_sendemail"],  # proxy only, not a phone/SMS message
    "LIST_REMINDERS": ["lists_query"],
    "TIMER": [],
    "ALARM": ["alarm_set"],
    "TEMPERATURE": [],
    "BRIGHTNESS": ["iot_hue_lightdim", "hue_lightdim", "iot_hue_lightup", "hue_lightup"],
    "COLOR": ["iot_hue_lightchange"],
    "CREATE_REMINDER": ["lists_createoradd"],
}

DEFAULT_SPLIT_FILES = ("train.jsonl", "devel.jsonl", "test.jsonl")


@dataclass(frozen=True)
class SlurpRecord:
    intent: str
    scenario: str
    sentence: str
    num_recordings: int


def load_records(data_dir: Path, split_files: tuple[str, ...] = DEFAULT_SPLIT_FILES) -> list[SlurpRecord]:
    records = []
    for filename in split_files:
        path = Path(data_dir) / filename
        with path.open() as f:
            for line in f:
                raw = json.loads(line)
                records.append(
                    SlurpRecord(
                        intent=raw["intent"],
                        scenario=raw["scenario"],
                        sentence=raw["sentence"],
                        num_recordings=len(raw.get("recordings", [])),
                    )
                )
    return records


def intent_counts(records: list[SlurpRecord]) -> Counter:
    """Sentence counts per raw SLURP intent (not audio-recording counts)."""
    return Counter(r.intent for r in records)


def coverage_report(
    records: list[SlurpRecord], mapping: dict[str, list[str]] = LABEL_MAPPING
) -> dict[str, dict[str, int]]:
    """For each canonical label, how many SLURP sentences/recordings map to it."""
    counts = intent_counts(records)
    report = {}
    for label, intents in mapping.items():
        sentences = sum(counts.get(i, 0) for i in intents)
        audio = sum(r.num_recordings for r in records if r.intent in intents)
        report[label] = {"sentences": sentences, "recordings": audio, "matched_intents": len(intents)}
    return report
