"""Training examples for the wake-word detector.

Every example is one WINDOW_S window, labeled wake (1) or not (0):

- `wake`: a synthetic "hey kiwi" clip (scripts/generate_targeted_synthetic.py
  --batch wakeword), trimmed and placed at a random position in the window.
- `partial`: the same clips cut off partway ("hey ki..."), labeled 0, so
  the streaming detector fires only once the whole phrase is in the window,
  not on "hey" or on the first half.
- `hard`: the batch's near-miss phrases ("kiwi", "hey kiki", "queen",
  "every week", "hey Siri", ...), from the wake-word confusability analysis.
- `speech`: a random window of an ordinary command clip from the intent
  manifest (real and synthetic speech that never contains the phrase).
- `noise`: a window of background noise (GSC) or near-silence.

Waveform augmentation (noise, speed, reverb) applies to every kind, so the
detector can't use "sounds clean" as a cue for any class.
"""

from __future__ import annotations

import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from vcm.audio.capture import SAMPLE_RATE
from vcm.audio.features import extract_log_mel, trim_silence
from vcm.train.wave_augment import add_noise, add_reverb, change_speed
from vcm.wakeword import WINDOW_S

KINDS = ("wake", "partial", "hard", "speech", "noise")
KIND_LABEL = {"wake": 1, "partial": 0, "hard": 0, "speech": 0, "noise": 0}


def _read(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def _place(audio: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    """Put `audio` at a random offset in an n-sample window (random crop if longer)."""
    if len(audio) >= n:
        start = rng.randint(0, len(audio) - n)
        return audio[start : start + n]
    out = np.zeros(n, dtype="float32")
    start = rng.randint(0, n - len(audio))
    out[start : start + len(audio)] = audio
    return out


class WakeWordDataset(Dataset):
    def __init__(
        self,
        items: list[tuple[str, str]],
        noise_bank: list[np.ndarray],
        augment: bool = True,
        seed: int | None = None,
    ):
        """items: (audio_path, kind). noise_bank: background clips for mixing
        (train split only for training)."""
        self.items = items
        self.noise_bank = noise_bank
        self.augment = augment
        self.rng = random.Random(seed)
        self.n = int(WINDOW_S * SAMPLE_RATE)

    def __len__(self) -> int:
        return len(self.items)

    def window(self, path: str, kind: str, rng: random.Random) -> np.ndarray:
        audio = _read(path)
        if kind in ("wake", "partial", "hard"):
            audio = trim_silence(audio, SAMPLE_RATE)
        if kind == "partial":
            audio = audio[: int(len(audio) * rng.uniform(0.35, 0.7))]
        if self.augment:
            if rng.random() < 0.5:
                audio = change_speed(audio, rng.uniform(0.9, 1.1))
            if rng.random() < 0.3:
                audio = add_reverb(audio, SAMPLE_RATE, rng.uniform(0.2, 0.8))
        if kind == "noise" and rng.random() < 0.3:
            audio = np.zeros(self.n, dtype="float32")  # near-silence: quiet room
        window = _place(audio.astype("float32"), self.n, rng)
        if self.augment and self.noise_bank and rng.random() < 0.6:
            window = add_noise(window + 1e-4, rng.choice(self.noise_bank), rng.uniform(0, 25))
        elif kind == "noise" and not window.any():
            window = (1e-4 * np.random.default_rng(rng.randint(0, 2**31)).standard_normal(self.n)).astype("float32")
        return window.astype("float32")

    def __getitem__(self, idx: int):
        path, kind = self.items[idx]
        rng = random.Random(self.rng.random()) if self.augment else random.Random(idx)
        window = self.window(path, kind, rng)
        features = extract_log_mel(window, SAMPLE_RATE, window_s=WINDOW_S, trim=False)
        return torch.from_numpy(features), KIND_LABEL[kind]
