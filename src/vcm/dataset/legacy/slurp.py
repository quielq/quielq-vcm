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

`LIST_REMINDERS` was also dropped down to no SLURP match, deliberately,
after training runs (EXPERIMENTS.md Experiments 7/11) consistently
showed it as one of the two weakest classes. Inspecting all 197 unique
`lists_query` sentences found only 1 even contains the word "reminder"
(and that one is still about viewing a list of reminders, not a timed
alert) — the rest are generic shopping lists, to-do lists, and even
music playlists ("what songs are on my favorite list"). This is a real
concept mismatch, not just noisy phrasing, so the mapping was removed
rather than filtered down. **This is a data gap worth revisiting**:
LIST_REMINDERS now relies solely on Option B's 558 synthetic rows: if
the class finds or builds a real source that actually covers timed
reminders (not generic lists), re-adding real coverage for this label
is a priority, not a "nice to have."
"""

# Acknowledgment: the taxonomy this checks coverage against was shared by
# Mark Macalacad as part of the class's collective dataset effort — see
# DATASET.md's Acknowledgments section.

from __future__ import annotations

import json
import re
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
    "LIST_REMINDERS": [],  # dropped: lists_query means generic lists, not reminders — see module docstring
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


# Built empirically by inspecting all 490 real `datetime_query` sentences
# (EXPERIMENTS.md's weak-class diagnosis): ~35% turned out to be pure date
# questions ("what date is today", "is today march sixth") with no time
# content at all, mapped to TIME anyway. This regex pair excludes a
# sentence only when it has a date signal and *no* time signal, so mixed
# "date and time" queries are kept (they do mention time).
_DATE_SIGNAL = re.compile(
    r"\b(date|calendar|day of the week|what day|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|january|february|march|april|may|june|july|august|"
    r"september|october|november|december|birthday|christmas|easter|halloween|"
    r"thanksgiving|valentine|weekend)\b"
)
_TIME_SIGNAL = re.compile(
    r"\b(time|hour|clock|noon|midnight|a\.?m\.?|p\.?m\.?|eastern|pacific|central|"
    r"mountain|g\.?m\.?t\.?|zone)\b"
)

# Small, surgical exclusion lists for WEATHER and MESSAGE — unlike TIME,
# inspecting all 834 weather_query and 523 email_sendemail sentences found
# the overwhelming majority genuinely on-topic (even indirect ones like
# "do i need a coat" are legitimately weather-dependent), so these are
# just the handful of clearly nonsensical or off-topic entries, not a
# phrasing-style filter.
_WEATHER_JUNK = {
    "answer email from", "by get marks", "by our scores", "by proper formula",
    "food will be given at the exhibition", "is the city cheaper or costlier to live",
    "i always pay attention", "i mostly pay attention", "i occasionally pay attention",
    "tell mom ill be late this evening", "its not possible", "its very dangerous one", "happy",
}
_MESSAGE_JUNK = {"how it's come to us", "how its come to us", "how we can get credit"}


def is_valid_sentence(label: str, sentence: str) -> bool:
    """True if this SLURP sentence should be kept for `label`.

    Only meaningful for TIME, WEATHER, and MESSAGE — every other label
    passes through unfiltered (either the mapping already has no known
    quality issue, or the label was already dropped from LABEL_MAPPING
    entirely, like LIST_REMINDERS).
    """
    s = sentence.lower()
    if label == "TIME":
        return not (_DATE_SIGNAL.search(s) and not _TIME_SIGNAL.search(s))
    if label == "WEATHER":
        return s not in _WEATHER_JUNK
    if label == "MESSAGE":
        return s not in _MESSAGE_JUNK
    return True
