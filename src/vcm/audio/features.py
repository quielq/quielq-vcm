"""Log-mel spectrogram feature extraction (Section 3/5: DS-CNN/TC-ResNet input).

Raw audio is pad/truncated to a fixed window before extraction so the
output is always the same shape, a fixed-size CNN input is what the
Section 5 architecture (DS-CNN/TC-ResNet) expects.
"""

from __future__ import annotations

import librosa
import numpy as np

from vcm.audio.capture import SAMPLE_RATE

WINDOW_S = 3.0
# 3.0s (not the original 1.5s), sized against the real training corpus:
# a 500-clip sample across all 5 combined sources had median duration
# 2.14s and p90 3.33s — at 1.5s, 80% of real clips were getting
# truncated. 3.0s covers ~83% without truncation at a reasonable
# compute cost; see MODEL.md / the training pipeline plan for the full
# percentile breakdown behind this number.
N_MELS = 40
N_FFT = 400  # 25ms at 16kHz
HOP_LENGTH = 160  # 10ms at 16kHz

# Global feature normalization, added during the class-imbalance-bias
# research pass (EXPERIMENTS.md Experiment 24). Before this, log-mel-dB
# values went straight into the model unstandardized — per-clip
# ref=np.max already makes each clip's own peak 0dB, but the resulting
# distribution is still far from the zero-mean/unit-variance range
# neural nets are typically initialized to expect. Measured directly
# (not guessed) from a 2,000-clip random sample of the real training
# split (data/dataset_manifest.csv, seed 0): mean=-59.64, std=21.30,
# range [-80, 0] (librosa's power_to_db floor/ceiling). Fixed constants,
# not recomputed per-run, so every experiment after this point shares
# the same normalization and stays comparable to each other — but this
# IS a real change to the feature representation itself, so a
# checkpoint trained before this change should not be --resume-from'd
# after it (the model would see systematically shifted inputs it was
# never trained on); train from scratch instead.
_LOG_MEL_MEAN = -59.64
_LOG_MEL_STD = 21.30


# Silence trimming (opt-in, EXPERIMENTS.md Experiment 29). Measured on a
# 3,000-clip manifest sample: keep-the-first-3.0s cuts off speech in
# 12.2% of clips overall and 29.6% of SLURP clips — SLURP being exactly
# where real-speech accuracy is weakest. Trimming leading/trailing
# silence first, plus a 5.0s window, leaves speech truncated in only
# ~1% of clips (3.3% of SLURP). A small margin is kept on each side so
# soft word onsets/offsets below the trim threshold aren't clipped.
TRIM_TOP_DB = 30
TRIM_MARGIN_S = 0.1


def trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    _, (start, end) = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
    margin = int(TRIM_MARGIN_S * sample_rate)
    return audio[max(0, start - margin) : min(len(audio), end + margin)]


def _fix_length(audio: np.ndarray, sample_rate: int, window_s: float = WINDOW_S) -> np.ndarray:
    target_len = int(window_s * sample_rate)
    if len(audio) >= target_len:
        return audio[:target_len]
    return np.pad(audio, (0, target_len - len(audio)))


def extract_log_mel(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_s: float = WINDOW_S,
    trim: bool = False,
) -> np.ndarray:
    """Return a (N_MELS, n_frames) float32 log-mel spectrogram for one fixed-length clip,
    normalized to roughly zero-mean/unit-variance (see _LOG_MEL_MEAN/_LOG_MEL_STD above).

    window_s/trim default to the original behavior (keep the first 3.0s, no
    trimming); a checkpoint records the values it was trained with under
    "feature_config", and inference must pass the same ones."""
    audio = audio.astype("float32")
    if trim:
        audio = trim_silence(audio, sample_rate)
    fixed = _fix_length(audio, sample_rate, window_s)
    mel = librosa.feature.melspectrogram(
        y=fixed,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    normalized = (log_mel - _LOG_MEL_MEAN) / _LOG_MEL_STD
    return normalized.astype("float32")
