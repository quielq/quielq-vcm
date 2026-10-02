"""Loader for the class's shared fixed-vs-slotted command taxonomy.

Source: the class's shared "Dataset Schema" Google Sheet (see
DATASET.md for current status and how the class's contributions are
being incorporated).

This is intentionally decoupled from audio: it generates the phrase/value
spec (what should be said for each label) so it's ready the moment real
recordings are organized against it, without this project needing to own
or duplicate the recordings themselves.

The sheet has three option tables (A/B/C) at increasing phrasing/value
richness. The class agreed (2026-10-01) to use Option B as the final
schema for labeling and for the demo benchmark: 3 phrasing variations
per intent and 3 slot values per slotted intent, 93 phrases in all.
This module matches that final version exactly (changes/Final Dataset
Schema.csv, and variations.csv in the class's master dataset,
huggingface.co/datasets/airimonda/ai231-me2-voice-commands). `export_csv()`
below writes this same spec out as a plain CSV, a convenient format for
anyone who wants to load it with pandas/Excel/etc. without going through
Sheets at all.
"""

# Acknowledgment: this taxonomy was shared by Mark Macalacad as part of
# the class's collective dataset effort — see DATASET.md's Acknowledgments
# section.

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

FIXED_INTENTS: dict[str, tuple[str, ...]] = {
    "PLAY_MUSIC": ("Play music", "Start music", "Play some music"),
    "WEATHER": ("Weather", "What's the weather?", "Tell me the weather"),
    "TIME": ("Time", "What time is it?", "Tell me the time"),
    "LIGHT_ON": ("Lights on", "Power on the lights", "Turn on the lights"),
    "LIGHT_OFF": ("Lights out", "Kill the lights", "Shut off the lights"),
    "PAUSE": ("Pause", "Pause audio", "Pause song"),
    "STOP": ("Stop", "Stop playing", "End playback"),
    "NEXT": ("Next song", "Skip song", "Play next song"),
    "VOLUME_UP": ("Volume up", "Increase the volume", "Turn the volume up"),
    "VOLUME_DOWN": ("Volume down", "Lower the volume", "Turn the volume down"),
    "CALL": ("Call", "Make a call", "Make a phone call"),
    "MESSAGE": ("Message", "Send a message", "Send my message"),
    "LIST_REMINDERS": ("Reminders", "Show my reminders", "List my reminders"),
}

SLOTTED_INTENTS: dict[str, dict[str, tuple[str, ...]]] = {
    "TIMER": {
        "templates": (
            "Timer {duration}",
            "Countdown for {duration}",
            "Start a timer for {duration}",
        ),
        "values": ("10 seconds", "30 seconds", "1 minute"),
    },
    "ALARM": {
        "templates": ("Alarm {time}", "Wake me up at {time}", "Set an alarm for {time}"),
        "values": ("6:00 AM", "8:00 AM", "9:00 PM"),
    },
    "TEMPERATURE": {
        "templates": (
            "Temperature {degrees}",
            "Change the temperature to {degrees}",
            "Set the temperature to {degrees}",
        ),
        "values": ("18 degrees", "22 degrees", "26 degrees"),
    },
    "BRIGHTNESS": {
        "templates": (
            "Brightness {percent}",
            "Adjust brightness to {percent}",
            "Brightness level {percent}",
        ),
        "values": ("20 percent", "60 percent", "100 percent"),
    },
    "COLOR": {
        "templates": ("Change color to {color}", "Switch color to {color}", "Set color to {color}"),
        "values": ("Red", "Blue", "Green"),
    },
    "CREATE_REMINDER": {
        "templates": (
            "Reminder {task}",
            "Remind me to {task}",
            "Create a reminder to {task}",
        ),
        "values": ("Drink water", "Study", "Exercise"),
    },
}

INTENT_LABELS: tuple[str, ...] = tuple(FIXED_INTENTS) + tuple(SLOTTED_INTENTS)

# The master dataset's non-command class: noise, Filipino speech, near-miss
# requests and general speech. Models before Experiment 37 called their
# (noise-only) non-command class "unknown_background".
OUT_OF_SCOPE = "OUT_OF_SCOPE"
NON_COMMAND_LABELS: tuple[str, ...] = (OUT_OF_SCOPE, "unknown_background")

_PLACEHOLDER_RE = re.compile(r"\{[^}]+\}")


@dataclass(frozen=True)
class Phrase:
    label: str
    text: str
    is_slotted: bool
    slot_value: str | None = None


def generate_phrases() -> list[Phrase]:
    """Expand the taxonomy into every (label, phrase) pair it specifies.

    Fixed intents yield one phrase per variation. Slotted intents yield
    one phrase per (template, value) pair.
    """
    phrases: list[Phrase] = []

    for label, variations in FIXED_INTENTS.items():
        for text in variations:
            phrases.append(Phrase(label=label, text=text, is_slotted=False))

    for label, spec in SLOTTED_INTENTS.items():
        for template in spec["templates"]:
            for value in spec["values"]:
                text = _PLACEHOLDER_RE.sub(value, template, count=1)
                phrases.append(
                    Phrase(label=label, text=text, is_slotted=True, slot_value=value)
                )

    return phrases


CSV_FIELDS = ("label", "type", "phrase", "slot_value")


def export_csv(path: Path) -> None:
    """Write the full phrase spec to a plain CSV — every value here is a
    literal string by construction, so nothing downstream needs to worry
    about a spreadsheet tool auto-formatting a cell as a date/time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        for phrase in generate_phrases():
            writer.writerow(
                [
                    phrase.label,
                    "slotted" if phrase.is_slotted else "fixed",
                    phrase.text,
                    phrase.slot_value or "",
                ]
            )
