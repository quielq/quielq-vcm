"""Category 9 (reminders and lists): software-only, persistent local storage."""

from __future__ import annotations

import json
from pathlib import Path

from vcm.config import load_settings


def _storage_path() -> Path:
    settings = load_settings()
    return Path(settings.reminders.get("storage_path", "data/reminders.json"))


def _read_all() -> list[str]:
    path = _storage_path()
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _write_all(items: list[str]) -> None:
    path = _storage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2))


def add_reminder(text: str) -> None:
    items = _read_all()
    items.append(text)
    _write_all(items)


def list_reminders() -> list[str]:
    return _read_all()


def clear_reminders() -> None:
    _write_all([])
