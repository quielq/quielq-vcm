"""System output volume: the speaker attached to the device, USB or
Bluetooth alike (both are the default audio output once connected).

Linux (the Pi): PipeWire/PulseAudio via `pactl` when available (that's
also how Bluetooth speakers appear), else ALSA via `amixer`. macOS:
`osascript`, for testing on the laptop.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5).stdout


def get_volume() -> int | None:
    try:
        if platform.system() == "Darwin":
            return int(_run(["osascript", "-e", "output volume of (get volume settings)"]).strip())
        if shutil.which("pactl"):
            match = re.search(r"(\d+)%", _run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"]))
            return int(match.group(1)) if match else None
        if shutil.which("amixer"):
            match = re.search(r"\[(\d+)%\]", _run(["amixer", "-M", "get", "Master"]))
            return int(match.group(1)) if match else None
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
    return None


def set_volume(percent: int) -> int | None:
    """Set output volume (0-100); returns the new volume, or None if no mixer is available."""
    percent = max(0, min(100, int(percent)))
    try:
        if platform.system() == "Darwin":
            _run(["osascript", "-e", f"set volume output volume {percent}"])
        elif shutil.which("pactl"):
            _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"])
        elif shutil.which("amixer"):
            _run(["amixer", "-q", "-M", "set", "Master", f"{percent}%"])
        else:
            return None
    except (subprocess.SubprocessError, OSError):
        return None
    return percent


def change_volume(step: int) -> int | None:
    current = get_volume()
    return set_volume((current if current is not None else 50) + step)
