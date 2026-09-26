"""Streaming resampler for microphones that can't record at 16 kHz.

Some USB mics only offer 44.1/48 kHz on their raw ALSA device (the Pi's
"USB PnP Sound Device" rejects 16 kHz with PaErrorCode -9997). The models
were trained on 16 kHz, so the live loop records at the mic's rate and
converts each chunk here: a windowed-sinc low-pass below the new Nyquist
(so 8-24 kHz content doesn't alias into the mel bands), then sampling at
the new rate. 48 kHz -> 16 kHz lands exactly on every 3rd sample; other
ratios interpolate linearly between the already-filtered samples. State
carries across chunks, so a chunked stream matches resampling it whole.
numpy only, like the rest of the Pi runtime.
"""

from __future__ import annotations

import numpy as np


def lowpass_filter(cutoff: float, taps: int = 97) -> np.ndarray:
    """Windowed-sinc FIR low-pass; `cutoff` in cycles per sample (0-0.5)."""
    n = np.arange(taps) - (taps - 1) / 2
    h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
    return (h / h.sum()).astype("float32")


class StreamResampler:
    def __init__(self, src_rate: int, dst_rate: int, taps: int = 97):
        self.step = src_rate / dst_rate
        self.h = lowpass_filter(0.45 * dst_rate / src_rate, taps)
        self.history = np.zeros(taps - 1, dtype="float32")
        self.last = np.float32(0.0)  # last filtered sample of the previous chunk
        self.pos = 1.0  # next output position; index 0 is `last`

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        x = np.concatenate([self.history, chunk.astype("float32").reshape(-1)])
        self.history = x[len(x) - len(self.history) :]
        filtered = np.concatenate([[self.last], np.convolve(x, self.h, mode="valid")]).astype("float32")
        end = len(filtered) - 1
        t = np.arange(self.pos, end + 1e-9, self.step)
        out = np.interp(t, np.arange(len(filtered)), filtered).astype("float32")
        self.pos = (t[-1] + self.step if len(t) else self.pos) - end
        self.last = filtered[-1]
        return out


def input_rate(sd, device, rate: int) -> int:
    """`rate` if the input device accepts it, else the device's default rate
    (48 kHz preferred: an exact 3:1 step down to 16 kHz)."""
    for candidate in (rate, 48000):
        try:
            sd.check_input_settings(device=device, samplerate=candidate, channels=1, dtype="float32")
            return candidate
        except Exception:
            pass
    return int(sd.query_devices(device, "input")["default_samplerate"])
