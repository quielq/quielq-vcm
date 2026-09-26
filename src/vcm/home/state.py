"""Shared, persistent home state: the virtual devices the dashboard shows,
plus reminders, timers, alarms, music, and a log of commands and calls.

One HomeState instance is shared by the dispatcher, the scheduler and the
HTTP server. Every change goes through `update()`, which holds the lock,
saves to disk and notifies subscribers (the dashboard's live stream).
"""

from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

LOG_LIMIT = 50


def default_state() -> dict:
    return {
        "lights": {"on": False, "brightness": 100, "color": "warm white"},
        "thermostat": {"target_c": 24, "current_c": None},
        "reminders": [],  # {id, text, created}
        "timers": [],  # {id, label, duration_s, ends_at}
        "alarms": [],  # {id, time, next_at}
        "music": {"source": None, "playing": False, "track": None, "volume": 50},
        "weather": None,  # {text, updated}
        "calls": [],  # {id, kind: call|message, contact, text, status, at}
        "log": [],  # {at, intent, slot, confidence, reply, source}
        "alerts": [],  # {id, text, at}, recent timer/alarm/reminder alerts
    }


class HomeState:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self._subscribers: list[Callable[[dict], None]] = []
        self._data = default_state()
        if self.path and self.path.exists():
            saved = json.loads(self.path.read_text())
            for key, value in saved.items():
                if key in self._data:
                    self._data[key] = value
        self.version = 0

    def snapshot(self) -> dict:
        with self._lock:
            return {**copy.deepcopy(self._data), "version": self.version, "now": time.time()}

    def update(self, change: Callable[[dict], object]):
        """Apply `change(data)` under the lock, persist, notify. Returns its result."""
        with self._lock:
            result = change(self._data)
            for key in ("log", "calls", "alerts"):
                self._data[key] = self._data[key][-LOG_LIMIT:]
            self.version += 1
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self._data, indent=2))
                tmp.replace(self.path)
            snapshot = {**copy.deepcopy(self._data), "version": self.version, "now": time.time()}
        for callback in list(self._subscribers):
            callback(snapshot)
        return result

    def subscribe(self, callback: Callable[[dict], None]) -> Callable[[], None]:
        self._subscribers.append(callback)
        return lambda: self._subscribers.remove(callback) if callback in self._subscribers else None


def new_id() -> str:
    return uuid.uuid4().hex[:8]
