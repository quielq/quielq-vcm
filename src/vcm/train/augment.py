"""SpecAugment (Park et al., "SpecAugment: A Simple Data Augmentation
Method for Automatic Speech Recognition", arXiv:1904.08779): randomly
mask blocks of time steps and frequency bins directly on the log-mel
spectrogram. Cheap (no extra audio processing) and directly targets
overfitting to the training distribution's specific voices/phrasing —
see MODEL.md Section 4 and EXPERIMENTS.md Experiment 1's overfitting
signal (train loss kept falling while val loss plateaued).

Only ever applied to training data, never validation/test.
"""

from __future__ import annotations

import random

import torch


def spec_augment(
    features: torch.Tensor,
    freq_mask_param: int = 8,
    time_mask_param: int = 40,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """features: (n_mels, n_frames) log-mel spectrogram. Returns a copy
    with random frequency/time bands masked to the feature's own mean
    (not zero, so it doesn't look like an artificial hard edge)."""
    features = features.clone()
    n_mels, n_frames = features.shape
    mask_value = features.mean()

    for _ in range(num_freq_masks):
        width = random.randint(0, min(freq_mask_param, n_mels))
        if width == 0:
            continue
        start = random.randint(0, n_mels - width)
        features[start : start + width, :] = mask_value

    for _ in range(num_time_masks):
        width = random.randint(0, min(time_mask_param, n_frames))
        if width == 0:
            continue
        start = random.randint(0, n_frames - width)
        features[:, start : start + width] = mask_value

    return features
