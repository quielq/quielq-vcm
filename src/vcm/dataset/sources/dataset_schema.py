"""Loader for classmate Mark Andrian Macalalad's fixed-vs-slotted taxonomy.

Source: the "Dataset Schema" Google Sheet he owns, shared into the class
Drive folder "AI 231 MEX2 Dataset" (which also holds a "1 fixed
phrase/command_50 speakers" recordings subfolder, suggesting the class
has already started collecting real audio against this exact schema).
See VCM_Architecture_Review.md's "Draft Label Taxonomy Review" section
for the full writeup.

This is intentionally decoupled from audio: it generates the phrase/value
spec (what should be said for each label) so it's ready the moment real
recordings are organized against it, without this project needing to own
or duplicate the recordings themselves.

The sheet has three option tables (A/B/C) at increasing phrasing/value
richness; this module captures Option B, the richest (3 phrasing
variations and 3 example slot values per intent), as of 2026-09-17.
Re-sync from the live sheet if Mark Andrian updates it. `export_csv()`
below writes this same spec out as a plain CSV, a convenient format for
anyone who wants to load it with pandas/Excel/etc. without going through
Sheets at all.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

FIXED_INTENTS: dict[str, tuple[str, ...]] = {
    "PLAY_MUSIC": ("Play music", "Play a song", "Start the music"),
    "WEATHER": ("Weather", "What's the weather?", "Tell me the weather"),
    "TIME": ("Time", "What time is it?", "Tell me the time"),
    "LIGHT_ON": ("Lights on", "Power on the lights", "Turn on the lights"),
    "LIGHT_OFF": ("Lights off", "Kill the lights", "Turn off the lights"),
    "PAUSE": ("Pause", "Pause the music", "Pause this song"),
    "STOP": ("Stop song", "Stop music", "Stop playing music"),
    "NEXT": ("Skip song", "Next song", "Play next song"),
    "VOLUME_UP": ("Volume up", "Increase the volume", "Turn the volume up"),
    "VOLUME_DOWN": ("Volume down", "Decrease the volume", "Turn the volume down"),
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
            "Set the brightness to {percent}",
            "Change the brightness to {percent}",
        ),
        "values": ("20 percent", "60 percent", "100 percent"),
    },
    "COLOR": {
        "templates": ("Color {color}", "Change the lights to {color}", "Set the lights to {color}"),
        "values": ("Red", "Blue", "Green"),
    },
    "CREATE_REMINDER": {
        "templates": (
            "Reminder {task}",
            "Remind me to {task}",
            "Create a reminder to {task}",
        ),
        "values": ("Drink water", "Study", "Call home"),
    },
}

INTENT_LABELS: tuple[str, ...] = tuple(FIXED_INTENTS) + tuple(SLOTTED_INTENTS)

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
