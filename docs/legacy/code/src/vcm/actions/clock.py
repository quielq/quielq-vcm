"""Category 2 (ask a question / search): time lookup via the system clock, no network needed."""

from __future__ import annotations

from datetime import datetime


def get_time() -> str:
    return datetime.now().strftime("%I:%M %p")
