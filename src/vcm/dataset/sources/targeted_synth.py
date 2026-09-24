"""Targeted synthetic phrasings for gaps found in live testing
(EXPERIMENTS.md, "Live-voice test of Experiment 29b" / Experiment 31).

The class-shared dataset schema (sources/dataset_schema.py) defines only
3 phrasings per fixed intent and 3 slot values per slotted intent — e.g.
TIMER durations are only 10 seconds / 30 seconds / 1 minute, COLOR only
Red / Blue / Green — and the Option B synthetic set was generated from
exactly those. Live testing of Experiment 29b failed precisely outside
them: "pause audio", "timer for 5 minutes", "color purple". This module
lists the extra phrasings; scripts/generate_targeted_synthetic.py voices
them with Chatterbox (the same TTS Option B used, so the new clips don't
introduce a new engine signature confined to a few classes), cloning
real speakers from the corresponding split.

Two design points:
- **Verb minimal pairs** across PLAY_MUSIC / PAUSE / STOP ("play / pause
  / stop the song"), all voiced by the same speakers: the object word
  ("song", "music") then carries no label information, forcing the
  model onto the verb — the same idea that resolved the VOLUME_UP/DOWN
  polarity confusion.
- **Number words, not digits**, so TTS pronunciation is unambiguous;
  the QA check normalizes both sides (see normalize_for_qa).
"""

from __future__ import annotations

import csv
import itertools
import re
from pathlib import Path

from vcm.dataset.manifest import ManifestRow
from vcm.dataset.sources.dataset_schema import _PLACEHOLDER_RE, FIXED_INTENTS, SLOTTED_INTENTS

_MEDIA_OBJECTS = ("the song", "the music", "the audio", "the track", "this song", "the playback")

PHRASES: dict[str, tuple[str, ...]] = {
    "PAUSE": (
        "pause",
        "pause it",
        "pause audio",
        "pause song",
        "pause music",
        "pause playback",
        *(f"pause {o}" for o in _MEDIA_OBJECTS),
        "please pause the song",
        "can you pause the music",
        "hold the music",
    ),
    "STOP": (
        "stop",
        "stop it",
        "stop audio",
        "stop song",
        "stop music",
        "stop playback",
        *(f"stop {o}" for o in _MEDIA_OBJECTS),
        "stop playing",
        "stop playing the song",
        "turn off the music",
        "turn the music off",
        "please stop the song",
        "can you stop the music",
    ),
    "PLAY_MUSIC": (
        "play music",
        "play a song",
        "play some music",
        "play audio",
        *(f"play {o}" for o in _MEDIA_OBJECTS),
        "start the music",
        "start playing music",
        "put on some music",
        "please play the song",
        "can you play some music",
    ),
}

_TIMER_DURATIONS = (
    "ten seconds", "fifteen seconds", "twenty seconds", "thirty seconds", "forty five seconds", "ninety seconds",
    "one minute", "a minute", "a minute and a half", "two minutes", "three minutes", "four minutes",
    "five minutes", "six minutes", "seven minutes", "eight minutes", "ten minutes", "twelve minutes",
    "fifteen minutes", "twenty minutes", "twenty five minutes", "thirty minutes", "forty five minutes",
    "an hour", "one hour", "two hours", "an hour and a half",
)  # fmt: skip
_TIMER_TEMPLATES = (
    "timer for {d}",
    "timer {d}",
    "{d} timer",
    "set a timer for {d}",
    "start a timer for {d}",
    "set timer for {d}",
    "set a {d} timer",
    "start a {d} timer",
    "countdown for {d}",
    "can you set a timer for {d}",
)
_COLORS = (
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "white",
    "warm white", "cool white", "cyan", "magenta", "violet", "teal",
)  # fmt: skip
_COLOR_TEMPLATES = (
    "color {c}",
    "{c} lights",
    "make the lights {c}",
    "turn the lights {c}",
    "set the lights to {c}",
    "change the lights to {c}",
    "set the color to {c}",
    "change the color to {c}",
    "change the light color to {c}",
    "switch the lights to {c}",
)
# BRIGHTNESS counterparts to the COLOR "set/change the lights to ..." carrier,
# so that carrier stays uninformative between the two (COLOR<->BRIGHTNESS is
# a top confusion pair in 29b).
_PERCENTS = ("ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "one hundred")
_BRIGHTNESS_TEMPLATES = (
    "brightness {p} percent",
    "set the brightness to {p} percent",
    "set the lights to {p} percent",
    "change the lights to {p} percent",
    "dim the lights to {p} percent",
)

PHRASES["TIMER"] = tuple(t.format(d=d) for t, d in itertools.product(_TIMER_TEMPLATES, _TIMER_DURATIONS))
PHRASES["COLOR"] = tuple(t.format(c=c) for t, c in itertools.product(_COLOR_TEMPLATES, _COLORS))
PHRASES["BRIGHTNESS"] = tuple(t.format(p=p) for t, p in itertools.product(_BRIGHTNESS_TEMPLATES, _PERCENTS))

# The class schema's own phrasings must always be covered: they're what
# benchmarking is likely to test. They're read from dataset_schema.py (not
# retyped) so they follow the schema if it changes, and they come first in
# each label's list, so the generator (which voices a label's phrases in
# order, cycling) covers every schema phrasing before any extra phrasing
# gets a second clip. No extra weight beyond that.


def schema_phrases(label: str) -> tuple[str, ...]:
    if label in FIXED_INTENTS:
        return tuple(p.lower() for p in FIXED_INTENTS[label])
    slot = SLOTTED_INTENTS[label]
    return tuple(
        _PLACEHOLDER_RE.sub(value, template).lower()
        for template, value in itertools.product(slot["templates"], slot["values"])
    )


def _with_schema(label: str, extra: tuple[str, ...]) -> tuple[str, ...]:
    schema = schema_phrases(label)
    schema_keys = {tuple(normalize_for_qa(p)) for p in schema}
    extra = tuple(p for p in extra if tuple(normalize_for_qa(p)) not in schema_keys)
    return schema + extra

# Clips voiced per label, per split. Train is sized to roughly double the
# weakest of these (PAUSE 640 / STOP 812 train clips before this); test
# is a held-out check on unseen speakers (not used for any selection).
CLIPS_PER_LABEL = {
    "train": {"PAUSE": 900, "STOP": 900, "PLAY_MUSIC": 600, "TIMER": 1000, "COLOR": 1000, "BRIGHTNESS": 400},
    "test": {"PAUSE": 120, "STOP": 120, "PLAY_MUSIC": 100, "TIMER": 150, "COLOR": 150, "BRIGHTNESS": 60},
}

_NUMBER_WORDS = {
    "a": "1", "an": "1", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "fifteen": "15",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}  # fmt: skip


def normalize_for_qa(text: str) -> list[str]:
    """Lowercase words with number words mapped to digits, so Whisper's
    "Set a timer for 5 minutes." matches the intended "set a timer for five
    minutes" (and "forty five" / "45" both become ["40", "5"] / ["45"] ->
    compared after joining compound tens, see below)."""
    words = re.findall(r"[a-z0-9']+", text.lower().replace("%", " percent").replace("-", " "))
    words = [_NUMBER_WORDS.get(w, w) for w in words if w not in ("and",)]
    merged: list[str] = []
    for w in words:  # "40" "5" -> "45", "1" "100" -> "100"
        if merged and merged[-1].isdigit() and w.isdigit():
            prev = int(merged[-1])
            if prev % 10 == 0 and prev >= 20 and int(w) < 10:
                merged[-1] = str(prev + int(w))
                continue
            if w == "100" and prev == 1:
                merged[-1] = "100"
                continue
        merged.append(w)
    return merged


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    d = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hypothesis, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hypothesis)] / max(len(reference), 1)


def load_manifest(csv_path: Path) -> list[ManifestRow]:
    """Rows of scripts/generate_targeted_synthetic.py's manifest that passed QA."""
    rows = []
    with Path(csv_path).open(newline="") as f:
        for r in csv.DictReader(f):
            if r["qa_pass"] != "True":
                continue
            rows.append(
                ManifestRow(
                    audio_path=r["audio_path"],
                    label=r["label"],
                    source="targeted_synth",
                    is_synthetic=True,
                    speaker_id=r["speaker_id"],
                    split=r["split"],
                )
            )
    return rows


PHRASES = {label: _with_schema(label, extra) for label, extra in PHRASES.items()}
