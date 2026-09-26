"""Background scheduler: fires timers and alarms stored in HomeState.

Timers and alarms live in the state (not in threading.Timer objects), so
they survive a server restart and show on the dashboard as countdowns.
A due item is removed and reported through `on_alert` (which speaks it)
and the state's `alerts` list (which the dashboard shows).
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta

from vcm.home.state import HomeState, new_id


def duration_seconds(label: str) -> int:
    """vcm.slots timer label ("1h30m", "5m", "45s") -> seconds."""
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", label)
    if not match or not any(match.groups()):
        raise ValueError(f"not a duration label: {label!r}")
    h, m, s = (int(x) if x else 0 for x in match.groups())
    return h * 3600 + m * 60 + s


def spoken_duration(seconds: int) -> str:
    parts = []
    for unit, size in (("hour", 3600), ("minute", 60), ("second", 1)):
        n, seconds = divmod(seconds, size)
        if n:
            parts.append(f"{n} {unit}{'s' if n > 1 else ''}")
    return " and ".join(parts) or "0 seconds"


def next_alarm_time(label: str, now: datetime | None = None) -> datetime:
    """vcm.slots alarm label ("6:30 AM") -> the next time that clock time occurs."""
    now = now or datetime.now()
    clock, meridiem = label.split()
    hour, minute = (int(x) for x in clock.split(":"))
    hour = hour % 12 + (12 if meridiem.upper() == "PM" else 0)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return target if target > now else target + timedelta(days=1)


class Scheduler:
    def __init__(self, state: HomeState, on_alert: Callable[[str], None], clock: Callable[[], float] = time.time):
        self.state, self.on_alert, self.clock = state, on_alert, clock
        self._stop = threading.Event()

    def tick(self) -> list[str]:
        """Fire everything due now. Returns the alert texts (for tests)."""
        now = self.clock()
        fired: list[str] = []

        def change(data: dict) -> None:
            for timer in [t for t in data["timers"] if t["ends_at"] <= now]:
                data["timers"].remove(timer)
                fired.append(f"Your timer for {spoken_duration(timer['duration_s'])} is done.")
            for alarm in [a for a in data["alarms"] if a["next_at"] <= now]:
                data["alarms"].remove(alarm)  # one-shot, like "wake me up at 6"
                fired.append(f"It's {alarm['time']}. This is your alarm.")
            for text in fired:
                data["alerts"].append({"id": new_id(), "text": text, "at": now})

        has_due = any(t["ends_at"] <= now for t in self.state.snapshot()["timers"]) or any(
            a["next_at"] <= now for a in self.state.snapshot()["alarms"]
        )
        if has_due:
            self.state.update(change)
            for text in fired:
                self.on_alert(text)
        return fired

    def run(self, interval_s: float = 0.5) -> None:
        while not self._stop.wait(interval_s):
            self.tick()

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run, daemon=True, name="vcm-scheduler")
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()
