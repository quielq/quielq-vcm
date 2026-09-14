"""Intent label -> action mapping.

Each handler takes no arguments and returns a short status string for TTS
feedback. Slot-filling (which song, timer duration, target temperature)
is an open question per Section 7, not yet in scope — handlers use fixed
defaults where a real command would carry a slot value; wiring real slots
through is a follow-up once that's resolved, not a structural change here.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable

from vcm.actions import alarms, calls, clock, lights, media_control, music
from vcm.actions import reminders, thermostat, timers, weather
from vcm.taxonomy import (
    ADJUST_THERMOSTAT,
    ASK_QUESTION_SEARCH,
    CALLS_MESSAGING,
    LIGHTS_DIM_COLOR,
    LIGHTS_ON_OFF,
    MEDIA_CONTROL,
    PLAY_MUSIC,
    REMINDERS_LISTS,
    SET_ALARM,
    SET_TIMER,
    UNKNOWN_BACKGROUND,
)

Handler = Callable[[], str]


def _handle_play_music() -> str:
    track = music.play()
    return f"Playing {track.name}"


def _handle_ask_question() -> str:
    try:
        return weather.get_weather()
    except Exception:
        return f"The time is {clock.get_time()}"


def _handle_lights_on_off() -> str:
    lights.turn_on()
    return "Lights on"


def _handle_lights_dim_color() -> str:
    lights.set_brightness(50)
    return "Brightness set to 50 percent"


def _handle_set_timer() -> str:
    timer = timers.set_timer(60)
    return f"Timer set for {int(timer.duration_s)} seconds"


def _handle_set_alarm() -> str:
    alarms.set_alarm(datetime.now() + timedelta(minutes=1))
    return "Alarm set for one minute from now"


def _handle_adjust_thermostat() -> str:
    return thermostat.adjust_to_comfort()


def _handle_media_control() -> str:
    media_control.pause()
    return "Playback paused"


def _handle_reminders_lists() -> str:
    items = reminders.list_reminders()
    return f"You have {len(items)} reminders" if items else "No reminders"


def _handle_calls_messaging() -> str:
    calls.call()
    return "Calling Mom"


def _handle_unknown_background() -> str:
    return ""


DISPATCH: dict[str, Handler] = {
    PLAY_MUSIC: _handle_play_music,
    ASK_QUESTION_SEARCH: _handle_ask_question,
    LIGHTS_ON_OFF: _handle_lights_on_off,
    LIGHTS_DIM_COLOR: _handle_lights_dim_color,
    SET_TIMER: _handle_set_timer,
    SET_ALARM: _handle_set_alarm,
    ADJUST_THERMOSTAT: _handle_adjust_thermostat,
    MEDIA_CONTROL: _handle_media_control,
    REMINDERS_LISTS: _handle_reminders_lists,
    CALLS_MESSAGING: _handle_calls_messaging,
    UNKNOWN_BACKGROUND: _handle_unknown_background,
}


def dispatch(label: str) -> str:
    if label not in DISPATCH:
        raise KeyError(f"No handler registered for intent label {label!r}")
    return DISPATCH[label]()
