"""Category 6 (set an alarm): software-only, local scheduling.

Same scheduling primitive as timers.py, just computed from a
wall-clock target time instead of a duration.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from vcm.actions.timers import Timer, set_timer


def set_alarm(target: datetime, label: str = "Alarm") -> Timer:
    now = datetime.now()
    if target <= now:
        target += timedelta(days=1)
    delay_s = (target - now).total_seconds()
    return set_timer(delay_s, label=label)
