"""Numpy-only versions of the three librosa functions feature extraction
used: mel spectrogram, power_to_db, and effects.trim's bounds.

Why: librosa imports numba, llvmlite and scipy, about 300 MB of memory
for what is an FFT and a matrix multiply. That rules out small boards
like the Raspberry Pi Zero 2 W (512 MB). These reproduce librosa 0.11's
defaults exactly as the project calls them (Hann window, centered frames
with zero padding, Slaney mel scale and normalization; tests/test_dsp.py
checks them against librosa), so models trained on librosa features see
the same inputs on the device.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

_AMIN = 1e-10


def _frames(y: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    """(n_frames, frame_length) view of y after centered zero padding."""
    pad = frame_length // 2
    y = np.pad(y, (pad, pad))
    n = 1 + (len(y) - frame_length) // hop
    return np.lib.stride_tricks.as_strided(y, shape=(n, frame_length), strides=(y.strides[0] * hop, y.strides[0]))


def _hz_to_mel(f):
    f = np.asanyarray(f, dtype=np.float64)
    mels = f / (200.0 / 3)
    log_region = f >= 1000.0
    return np.where(log_region, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), mels)


def _mel_to_hz(m):
    m = np.asanyarray(m, dtype=np.float64)
    freqs = (200.0 / 3) * m
    return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), freqs)


@lru_cache(maxsize=8)
def mel_filters(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels): Slaney scale, Slaney norm."""
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2.0), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


@lru_cache(maxsize=8)
def _hann(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)  # periodic, like scipy/librosa


def melspectrogram(y: np.ndarray, sr: int, n_fft: int, hop_length: int, n_mels: int) -> np.ndarray:
    """librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)."""
    frames = _frames(y.astype(np.float32), n_fft, hop_length) * _hann(n_fft)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2  # (n_frames, 1 + n_fft // 2)
    return (mel_filters(sr, n_fft, n_mels) @ power.T.astype(np.float32)).astype(np.float32)


def power_to_db_ref_max(S: np.ndarray, top_db: float = 80.0) -> np.ndarray:
    """librosa.power_to_db(S, ref=np.max) (amin 1e-10, top_db 80)."""
    log_spec = 10.0 * np.log10(np.maximum(_AMIN, S))
    log_spec -= 10.0 * np.log10(max(_AMIN, float(S.max()) if S.size else _AMIN))
    return np.maximum(log_spec, log_spec.max() - top_db) if S.size else log_spec


def trim_bounds(y: np.ndarray, top_db: float, frame_length: int = 2048, hop_length: int = 512) -> tuple[int, int]:
    """Sample bounds librosa.effects.trim(y, top_db=top_db) returns."""
    if len(y) == 0:
        return 0, 0
    mse = np.mean(_frames(y.astype(np.float32), frame_length, hop_length).astype(np.float64) ** 2, axis=1)
    db = 10.0 * np.log10(np.maximum(_AMIN, mse)) - 10.0 * np.log10(max(_AMIN, mse.max()))
    nonzero = np.flatnonzero(db > -top_db)
    if nonzero.size == 0:
        return 0, 0
    return int(nonzero[0] * hop_length), int(min(len(y), (nonzero[-1] + 1) * hop_length))
