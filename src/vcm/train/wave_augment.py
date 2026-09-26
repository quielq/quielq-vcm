"""Waveform-level training augmentation (EXPERIMENTS.md Experiment 29).

SpecAugment was ruled out (Experiments 2 and 8) because randomly
masking time bands erases the one distinguishing word in short commands
("up" vs "down"). Everything here perturbs *how* a command sounds —
background, speaking rate, position in the window, room — without
removing any of *what* was said, and targets the synthetic-to-real gap
directly (Option B scores ~94% on its own test clips versus ~71% for
real speech, see Experiment 28's evaluation).

Deliberately no gain augmentation: extract_log_mel references each
clip to its own peak (power_to_db(ref=np.max)), so a global gain change
is almost entirely normalized away before the model sees it.

Only ever applied to training data, never validation/test, and always
before feature extraction (operates on raw audio).
"""

from __future__ import annotations

import random

import numpy as np
from scipy.signal import fftconvolve

NOISE_PROB = 0.5
NOISE_SNR_DB = (5.0, 25.0)
SPEED_PROB = 0.5
SPEED_RANGE = (0.9, 1.1)
REVERB_PROB = 0.3
REVERB_RT60_S = (0.2, 0.8)
SHIFT_MAX_S = 0.3


def add_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float, rng: random.Random | None = None) -> np.ndarray:
    """Mix a random segment of `noise` (tiled if shorter) into `audio` at `snr_db`.
    Pass `rng` for a reproducible segment (evaluation); training uses the global one."""
    if len(noise) < len(audio):
        noise = np.tile(noise, len(audio) // len(noise) + 1)
    start = (rng or random).randint(0, len(noise) - len(audio))
    noise = noise[start : start + len(audio)]
    signal_power = float(np.mean(audio**2)) + 1e-10
    noise_power = float(np.mean(noise**2)) + 1e-10
    scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    return audio + scale * noise


def change_speed(audio: np.ndarray, factor: float) -> np.ndarray:
    """Resample by linear interpolation so the clip plays `factor`x faster
    (pitch shifts with it, like Kaldi-style speed perturbation)."""
    new_len = max(1, int(round(len(audio) / factor)))
    old_idx = np.linspace(0, len(audio) - 1, num=new_len)
    return np.interp(old_idx, np.arange(len(audio)), audio).astype(audio.dtype)


def add_reverb(audio: np.ndarray, sample_rate: int, rt60_s: float) -> np.ndarray:
    """Convolve with a synthetic room impulse response: exponentially
    decaying white noise reaching -60dB at rt60_s — a standard cheap
    stand-in when no measured RIR corpus is available."""
    n = int(rt60_s * sample_rate)
    t = np.arange(n) / sample_rate
    rir = np.random.randn(n) * np.exp(-6.9078 * t / rt60_s)  # ln(1000) = 6.9078 -> -60dB
    rir[0] = 1.0  # keep a direct path
    rir /= np.sqrt(np.sum(rir**2))
    wet = fftconvolve(audio, rir)[: len(audio)]
    return (wet * (np.max(np.abs(audio)) / (np.max(np.abs(wet)) + 1e-10))).astype(audio.dtype)


def shift_start(audio: np.ndarray, sample_rate: int, max_shift_s: float) -> np.ndarray:
    """Prepend 0..max_shift_s of silence, so a trimmed clip doesn't always
    start at exactly frame 0 (live push-to-talk audio never does)."""
    pad = random.randint(0, int(max_shift_s * sample_rate))
    return np.concatenate([np.zeros(pad, dtype=audio.dtype), audio])


def augment_waveform(audio: np.ndarray, sample_rate: int, noise_bank: list[np.ndarray]) -> np.ndarray:
    audio = audio.astype("float32")
    if random.random() < SPEED_PROB:
        audio = change_speed(audio, random.uniform(*SPEED_RANGE))
    if random.random() < REVERB_PROB:
        audio = add_reverb(audio, sample_rate, random.uniform(*REVERB_RT60_S))
    if noise_bank and random.random() < NOISE_PROB:
        audio = add_noise(audio, random.choice(noise_bank), random.uniform(*NOISE_SNR_DB))
    audio = shift_start(audio, sample_rate, SHIFT_MAX_S)
    return audio.astype("float32")
