"""DS-CNN — the first real model architecture (MODEL.md Section 2).

Zhang, Suda, Lai, Chandra, "Hello Edge: Keyword Spotting on
Microcontrollers" (https://arxiv.org/abs/1711.07128): a standard conv
layer to reduce the time axis, followed by depthwise-separable conv
blocks (depthwise 3x3 + pointwise 1x1, each BatchNorm+ReLU), global
average pooling, then a linear classifier.

Chosen as the *first* architecture specifically because it's the
simplest of the four MODEL.md surveys — the goal for this pass is a
verified, working training pipeline, not the most accurate model on the
first attempt. BC-ResNet (MODEL.md's actual recommendation) is the
planned next step once this pipeline is proven end-to-end.

Input: (batch, n_mels, n_frames) log-mel spectrograms, the direct output
of vcm.audio.features.extract_log_mel — this module adds the channel
dimension itself.
"""

from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class DSCNN(nn.Module):
    def __init__(self, num_classes: int, num_filters: int = 64, num_blocks: int = 4):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1)),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )
        self.ds_blocks = nn.ModuleList(
            [DepthwiseSeparableBlock(num_filters, num_filters) for _ in range(num_blocks)]
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_mels, n_frames) -> logits (batch, num_classes)."""
        x = x.unsqueeze(1)  # (batch, 1, n_mels, n_frames)
        x = self.first_conv(x)
        for block in self.ds_blocks:
            x = block(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)
