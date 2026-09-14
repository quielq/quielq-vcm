"""Category 5 (set a timer): software-only, local scheduling."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

from vcm.tts.speak import speak


@dataclass
class Timer:
    id: str
    duration_s: float
    _handle: threading.Timer = field(repr=False, compare=False)
    label: str = "Timer"

    def cancel(self) -> None:
        self._handle.cancel()


_active_timers: dict[str, Timer] = {}


def set_timer(duration_s: float, label: str = "Timer") -> Timer:
    timer_id = str(uuid.uuid4())

    def _on_expire() -> None:
        _active_timers.pop(timer_id, None)
        speak(f"{label} is done")

    handle = threading.Timer(duration_s, _on_expire)
    handle.daemon = True
    handle.start()
    timer = Timer(id=timer_id, duration_s=duration_s, label=label, _handle=handle)
    _active_timers[timer_id] = timer
    return timer


def cancel_timer(timer_id: str) -> None:
    timer = _active_timers.pop(timer_id, None)
    if timer:
        timer.cancel()


def list_active_timers() -> list[Timer]:
    return list(_active_timers.values())
