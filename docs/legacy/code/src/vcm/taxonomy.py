"""Single source of truth for intent labels.

Uses the ten broad categories from Section 1/6 plus the explicit
unknown/background class from Section 3, not the more granular per-action
draft circulating in Section 9 — that draft is unconfirmed class-wide.
Swapping to it later is a change to this one tuple (and the dispatch
table it drives), not a structural change.
"""

from __future__ import annotations

PLAY_MUSIC = "play_music"
ASK_QUESTION_SEARCH = "ask_question_search"
LIGHTS_ON_OFF = "lights_on_off"
LIGHTS_DIM_COLOR = "lights_dim_color"
SET_TIMER = "set_timer"
SET_ALARM = "set_alarm"
ADJUST_THERMOSTAT = "adjust_thermostat"
MEDIA_CONTROL = "media_control"
REMINDERS_LISTS = "reminders_lists"
CALLS_MESSAGING = "calls_messaging"
UNKNOWN_BACKGROUND = "unknown_background"

LABELS: tuple[str, ...] = (
    PLAY_MUSIC,
    ASK_QUESTION_SEARCH,
    LIGHTS_ON_OFF,
    LIGHTS_DIM_COLOR,
    SET_TIMER,
    SET_ALARM,
    ADJUST_THERMOSTAT,
    MEDIA_CONTROL,
    REMINDERS_LISTS,
    CALLS_MESSAGING,
    UNKNOWN_BACKGROUND,
)
