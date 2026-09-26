"""Slot values for the slotted intents (EXPERIMENTS.md Experiment 32).

The intent model says TIMER but not *how long*, COLOR but not *which
color*. Each slotted intent here gets a closed vocabulary of values, and
the model gets a small classification head per slot (see
vcm.train.architectures.CRNN). A closed set keeps the heads tiny and
matches the class dataset schema, which fixes 3 values per slot; each
vocabulary is the schema's values plus other common ones.

Values the vocabulary doesn't cover (a 25-minute timer, 7:15 AM) aren't
forced into a wrong class: they get no slot label for training and are
reported as out-of-vocabulary in evaluation.

TEMPERATURE and CREATE_REMINDER are schema-slotted too but out of scope
for now: FSC's temperature commands carry no values ("increase the
heat"), and a reminder's task is free text, not a closed set.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from vcm.dataset.sources.dataset_schema import SLOTTED_INTENTS

# Canonical value strings. Order is the head's class order, so append
# only — reordering would break trained checkpoints.
_DURATIONS_S = (10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 360, 420, 480, 600, 720, 900, 1200, 1500, 1800, 2700, 3600, 5400, 7200)  # fmt: skip
_ALARM_TIMES = tuple(f"{h}:00 {m}" for m in ("AM", "PM") for h in (12, *range(1, 12))) + (
    "5:30 AM", "6:30 AM", "7:30 AM", "8:30 AM",
)  # fmt: skip
_PERCENTS = (10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100)
COLORS = (
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "white",
    "warm white", "cool white", "cyan", "magenta", "violet", "teal",
)  # fmt: skip


def duration_label(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60}m"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    if seconds > 60:
        return f"{seconds // 60}m{seconds % 60}s"
    return f"{seconds}s"


SLOT_VOCAB: dict[str, tuple[str, ...]] = {
    "TIMER": tuple(duration_label(s) for s in _DURATIONS_S),
    "ALARM": _ALARM_TIMES,
    "BRIGHTNESS": tuple(f"{p}%" for p in _PERCENTS),
    "COLOR": COLORS,
}
SLOT_NAMES = {"TIMER": "duration", "ALARM": "time", "BRIGHTNESS": "percent", "COLOR": "color"}

_UNITS = {"one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
          "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}  # fmt: skip
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def _tokens(text: str) -> list[str]:
    text = text.lower().replace("%", " percent ").replace("-", " ").replace("a.m.", " am ").replace("p.m.", " pm ")
    text = re.sub(r"(\d+):(\d\d)", r"\1 \2", text)  # "6:30" -> "6 30"
    text = re.sub(r"(\d)(am|pm)\b", r"\1 \2", text)  # "6am" -> "6 am"
    return re.findall(r"[a-z]+|\d+", text)


def _numbers(tokens: list[str]) -> list[tuple[int, int, int]]:
    """(value, start, end) for each number in the token list, digits or words
    ("forty five" -> 45, "one hundred" -> 100, "half" -> handled by callers)."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t.isdigit():
            out.append((int(t), i, i + 1))
            i += 1
            continue
        value, j = None, i
        if t in _TENS:
            value, j = _TENS[t], i + 1
            if j < len(tokens) and tokens[j] in _UNITS and _UNITS[tokens[j]] < 10 and tokens[j] not in ("a", "an"):
                value, j = value + _UNITS[tokens[j]], j + 1
        elif t in _UNITS:
            value, j = _UNITS[t], i + 1
        if value is not None and j < len(tokens) and tokens[j] == "hundred":
            value, j = value * 100, j + 1
        if value is not None:
            out.append((value, i, j))
            i = j
        else:
            i += 1
    return out


def parse_duration(text: str) -> int | None:
    """Total seconds for "five minutes", "a minute and a half", "an hour and
    a half", "90 seconds", "1 hour 30 minutes", "half an hour"."""
    tokens = _tokens(text)
    if "half" in tokens and ("an" in tokens or "a" in tokens) and "hour" in tokens and "and" not in tokens:
        return 1800  # "half an hour"
    total, found = 0, False
    for value, _, end in _numbers(tokens):
        if end < len(tokens):
            unit = tokens[end]
            scale = {"second": 1, "seconds": 1, "sec": 1, "secs": 1, "minute": 60, "minutes": 60, "min": 60,
                     "mins": 60, "hour": 3600, "hours": 3600}.get(unit)  # fmt: skip
            if scale:
                total += value * scale
                found = True
                rest = tokens[end + 1 : end + 4]
                if rest[:3] in (["and", "a", "half"], ["and", "an", "half"]):
                    total += scale // 2
    return total if found and total > 0 else None


def parse_alarm_time(text: str) -> str | None:
    """"6 AM", "6:30 a.m.", "seven thirty pm", "8 in the morning", "9 at night"."""
    tokens = _tokens(text)
    for value, _, end in _numbers(tokens):
        if not 1 <= value <= 12:
            continue
        minute, k = 0, end
        nums_after = [n for n in _numbers(tokens[end:]) if n[1] == 0]
        if nums_after and nums_after[0][0] in (0, 15, 30, 45):
            minute, k = nums_after[0][0], end + nums_after[0][2]
        elif k < len(tokens) and tokens[k] == "o" and k + 1 < len(tokens) and tokens[k + 1] == "clock":
            k += 2
        rest = tokens[k : k + 4]
        if rest[:1] in (["am"], ["a"]) or rest[:3] == ["in", "the", "morning"]:
            meridiem = "AM"
        elif rest[:1] in (["pm"], ["p"]) or rest[:3] in (["in", "the", "evening"], ["in", "the", "afternoon"]) or rest[:2] == ["at", "night"]:
            meridiem = "PM"
        else:
            continue
        return f"{value}:{minute:02d} {meridiem}"
    return None


def parse_percent(text: str) -> int | None:
    tokens = _tokens(text)
    for value, _, end in _numbers(tokens):
        if end < len(tokens) and tokens[end] in ("percent", "per") and 0 < value <= 100:
            return value
    if "brightness" in tokens:  # "brightness 60", "set the brightness to 60"
        nums = _numbers(tokens)
        if nums and 0 < nums[-1][0] <= 100:
            return nums[-1][0]
    return None


def parse_color(text: str) -> str | None:
    t = " " + " ".join(_tokens(text)) + " "
    for color in sorted(COLORS, key=len, reverse=True):  # "warm white" before "white"
        if f" {color} " in t:
            return color
    return None


def parse_slot(label: str, text: str) -> str | None:
    """Canonical slot value for `text` under intent `label`, or None if the
    intent has no slot here or the value isn't in its vocabulary."""
    if label == "TIMER":
        seconds = parse_duration(text)
        value = duration_label(seconds) if seconds else None
    elif label == "ALARM":
        value = parse_alarm_time(text)
    elif label == "BRIGHTNESS":
        percent = parse_percent(text)
        value = f"{percent}%" if percent else None
    elif label == "COLOR":
        value = parse_color(text)
    else:
        return None
    return value if value in SLOT_VOCAB[label] else None


def schema_values_covered() -> dict[str, bool]:
    """Every schema slot value for the intents here parses into the vocabulary."""
    out = {}
    for label in SLOT_VOCAB:
        values = SLOTTED_INTENTS[label]["values"]
        template = SLOTTED_INTENTS[label]["templates"][0]
        out[label] = all(parse_slot(label, re.sub(r"\{[^}]+\}", v, template)) is not None for v in values)
    return out


# Head order in the model: one slot head per intent in SLOT_VOCAB, in this order.
SLOT_INTENTS: tuple[str, ...] = tuple(SLOT_VOCAB)


def load_slot_labels(csv_path: Path) -> dict[str, tuple[str, str, str]]:
    """scripts/build_slot_labels.py output: audio_path -> (label, value, text_source)."""
    with Path(csv_path).open(newline="") as f:
        return {r["audio_path"]: (r["label"], r["value"], r["text_source"]) for r in csv.DictReader(f)}


def slot_target(label: str, value: str | None) -> list[int]:
    """Per-head class index for one clip: its intent's head gets the value's
    index, every other head -1 (ignored by the loss)."""
    return [SLOT_VOCAB[i].index(value) if i == label and value in SLOT_VOCAB[i] else -1 for i in SLOT_INTENTS]
