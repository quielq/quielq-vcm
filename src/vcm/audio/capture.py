"""Microphone capture.

No platform branching here on purpose: Section 8 notes the same Python
audio libraries work on both the Mac's built-in mic and the eventual USB
mic on the RPi — it's just whichever device the OS reports as default
input. `sounddevice` (PortAudio) handles both transparently.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000  # matches the target command-length feature window


def record(duration_s: float = 1.5, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record `duration_s` seconds of mono audio, returned as float32 in [-1, 1]."""
    frames = sd.rec(
        int(duration_s * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return frames.reshape(-1)


def record_while_held(button, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record from button press to release, using a HAL PushButton (see vcm.hal.button)."""
    import queue

    chunks: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
        chunks.put(indata.copy())

    with sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32", callback=_callback
    ):
        button.wait_for_press()
        button.wait_for_release()

    collected = []
    while not chunks.empty():
        collected.append(chunks.get_nowait())
    if not collected:
        return np.zeros(0, dtype="float32")
    return np.concatenate(collected).reshape(-1)
