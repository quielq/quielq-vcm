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


def _fix_length(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    target_len = int(WINDOW_S * sample_rate)
    if len(audio) >= target_len:
        return audio[:target_len]
    return np.pad(audio, (0, target_len - len(audio)))


def extract_log_mel(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return a (N_MELS, n_frames) float32 log-mel spectrogram for one fixed-length clip."""
    fixed = _fix_length(audio.astype("float32"), sample_rate)
    mel = librosa.feature.melspectrogram(
        y=fixed,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype("float32")
