"""Snips SLU (smart-lights subset) text classifier -> our 4 lighting labels.

Source: `MWilinski/snips_slu_v1.0` on HuggingFace — 5,886 real recordings
of smart-home commands. Unlike SLURP, this dataset ships with a raw
`text` transcript per clip but **no categorical intent label**, so this
module classifies by keyword/phrase pattern instead of a lookup table.

Verified empirically (not assumed) before building this: streamed and
inspected all 5,886 transcripts. The corpus is NOT purely lighting as
Section 9 described it — roughly 2,680 mention lighting vocabulary and
2,566 are music requests ("I'd like to listen to <artist>"), with the
two overlapping close to not at all. A naive color-word match
(`"white"`) initially misclassified a music request ("I'd like to listen
to White Sea") as a COLOR command — `classify()` below explicitly
excludes music-request phrasing and requires lighting-context words
(light/lamp/bulb/room/house) alongside any color/brightness/on/off
keyword, specifically to avoid repeating that mistake. Spot-checked
~10 examples per label after this fix, no further false positives
found, but this is a hand-built heuristic classifier, not a ground-truth
label the way SLURP's `intent` field is — treat matches as good-quality
but not infallible.

Only 4 of our 19 labels have any real presence in this corpus:
LIGHT_ON, LIGHT_OFF, BRIGHTNESS, COLOR. Nothing here helps the 6
zero-real-coverage labels (checked: no weather/alarm/timer/call/
reminder/volume vocabulary anywhere in the corpus). Unmatched examples
(including the whole music-request chunk) are dropped, not guessed.
"""

from __future__ import annotations

import re

MUSIC_REQUEST_RE = re.compile(r"\blisten to\b|\bplay\b.*\b(song|music|playlist)\b|\bput on\b.*\bmusic\b", re.I)
LIGHT_CONTEXT_RE = re.compile(r"\blight|\blamp|\bbulb|\broom\b|\bhouse\b|\bappartment\b|\bapartment\b", re.I)

COLOR_WORDS = {
    "red", "blue", "green", "pink", "purple", "orange", "yellow", "white",
    "warm", "cool", "violet", "cyan", "magenta", "teal", "gold", "silver",
    "hue", "colour", "color",
}
BRIGHTNESS_RE = re.compile(
    r"\bbright(en|ness|er|ed)?\b|\bdim(mer|med|ming)?\b|\bintensity\b|\blevel\b|\bpercent\b"
    r"|\bshin(e|ing)\b|\bglow(ing)?\b|\braise(d)?\b|\blower(ed)?\b|\bturn\s+(it\s+)?(up|down)\b"
    r"|\bincrease(d)?\b|\bdecrease(d)?\b|\bboost(ed)?\b|\baugment(ed)?\b|\bnotch\b"
    r"|\bset\b.*\b(to|at)\s+\w*\s*\d|\bto\s+(ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|\d+)\b"
)
OFF_RE = re.compile(
    r"\bturn(ed)?\s+off\b|\bswitch(ed)?\s+off\b|\bdeactivat(e|ed)\b|\bkill(ed)?\b"
    r"|\bcut\b|\beliminat(e|ed)\b|\bextinguish(ed)?\b|\bdark(en(ed)?)?\b|\boff\b"
)
ON_RE = re.compile(
    r"\bturn(ed)?\s+on\b|\bswitch(ed)?\s+on\b|\bactivat(e|ed)\b|\billuminat(e|ed)\b"
    r"|\bpower\s+on\b|\bput\s+on\b|\bon\b"
)


def classify(text: str) -> str | None:
    """Return one of COLOR/BRIGHTNESS/LIGHT_OFF/LIGHT_ON, or None if the
    text doesn't confidently match a lighting command."""
    t = text.lower()
    if MUSIC_REQUEST_RE.search(t):
        return None
    words = set(re.findall(r"[a-z']+", t))
    has_light_context = bool(LIGHT_CONTEXT_RE.search(t))
    if (words & COLOR_WORDS) and has_light_context:
        return "COLOR"
    if BRIGHTNESS_RE.search(t) and has_light_context:
        return "BRIGHTNESS"
    if OFF_RE.search(t) and has_light_context:
        return "LIGHT_OFF"
    if ON_RE.search(t) and has_light_context:
        return "LIGHT_ON"
    return None
