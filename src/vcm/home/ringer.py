"""Alarm and timer sounds: a beeping pattern that repeats until stopped.

When a timer or alarm fires, the Ringer beeps (like a kitchen timer or a
bedside alarm clock), says what fired, and keeps beeping until "Hey Kiwi,
stop", the dashboard's dismiss button, or RING_LIMIT_S. While the listener
records a command after "Hey Kiwi" the beeping pauses (mute), so the
command can be heard.
"""

from __future__ import annotations

import tempfile
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

RATE = 22050
RING_LIMIT_S = 60.0


def _beeps(pattern: list[tuple[float, float, float]]) -> np.ndarray:
    """(freq Hz, beep s, gap s) triples -> float samples."""
    parts = []
    for freq, beep_s, gap_s in pattern:
        t = np.arange(int(beep_s * RATE)) / RATE
        ramp = np.minimum(1.0, np.minimum(t, beep_s - t) / 0.005)  # 5 ms edges: no clicks
        parts += [0.5 * ramp * np.sin(2 * np.pi * freq * t), np.zeros(int(gap_s * RATE))]
    return np.concatenate(parts)


# One ring cycle each (~1.2 s); the Ringer repeats them.
SOUNDS = {
    "alarm": _beeps([(988, 0.1, 0.06)] * 4 + [(988, 0.1, 0.5)]),  # bedside alarm clock
    "timer": _beeps([(1568, 0.14, 0.1), (1568, 0.14, 0.1), (1568, 0.14, 0.6)]),  # kitchen timer
}


def write_wav(samples: np.ndarray, path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


class Ringer:
    def __init__(self, play: Callable[[str], None], say: Callable[[str], None], limit_s: float = RING_LIMIT_S):
        self.play, self.say, self.limit_s = play, say, limit_s
        self._dir = Path(tempfile.mkdtemp(prefix="vcm-ring-"))
        for kind, samples in SOUNDS.items():
            write_wav(samples, self._dir / f"{kind}.wav")
        self._stop, self._muted = threading.Event(), threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def ringing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def ring(self, text: str, kind: str = "alarm") -> None:
        """Start ringing in the background (a new alert replaces the current one)."""
        self.stop()
        self._stop.clear()
        self._muted.clear()
        self._thread = threading.Thread(target=self._run, args=(text, kind), daemon=True, name="vcm-ringer")
        self._thread.start()

    def stop(self) -> bool:
        """Stop ringing. Returns whether anything was ringing."""
        was = self.ringing
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        return was

    def mute(self, on: bool) -> None:
        (self._muted.set if on else self._muted.clear)()

    def _run(self, text: str, kind: str) -> None:
        sound = str(self._dir / f"{kind}.wav")
        deadline, cycles = time.time() + self.limit_s, 0
        while not self._stop.is_set() and time.time() < deadline:
            if self._muted.is_set():
                self._stop.wait(0.2)
                continue
            try:
                self.play(sound)
            except Exception as exc:  # no audio output: still say it
                print(f"(alarm sound failed: {exc})")
                self._stop.wait(1.0)
            cycles += 1
            if cycles == 2:  # after a couple of rings, say what it is
                self.say(text)
