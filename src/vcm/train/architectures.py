"""Model architectures surveyed in MODEL.md Section 2.

Input to every model here: (batch, n_mels, n_frames) log-mel
spectrograms, the direct output of vcm.audio.features.extract_log_mel —
each model adds its own channel dimension.
"""

from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableBlock(nn.Module):
    """Building block for DSCNN (below) — see that class's docstring."""

    def __init__(self, in_channels: int, out_channels: int, stride: int | tuple[int, int] = 1):
        super().__init__()
        # stride=1 (default) is the original block; CRNN below uses stride 2
        # on some blocks to widen the receptive field. Stride has no
        # parameters, so existing DSCNN checkpoints load unchanged.
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class DSCNN(nn.Module):
    """Zhang, Suda, Lai, Chandra, "Hello Edge: Keyword Spotting on
    Microcontrollers" (https://arxiv.org/abs/1711.07128): a standard
    conv layer to reduce the time axis, followed by depthwise-separable
    conv blocks (depthwise 3x3 + pointwise 1x1, each BatchNorm+ReLU),
    global average pooling, then a linear classifier.

    Used as the first architecture (EXPERIMENTS.md Experiment 1)
    specifically because it's the simplest of the models MODEL.md
    surveys — the goal for that pass was a verified, working training
    pipeline, not the most accurate model on the first attempt.
    Experiment 1's confusion analysis found its main weakness: mixing
    up labels that share an identical carrier phrase and differ only
    in one polarity word (e.g. VOLUME_UP vs VOLUME_DOWN, LIGHT_ON vs
    LIGHT_OFF) — global-average-pooling likely discards exactly the
    fine temporal detail needed to catch that. BCResNet below is a
    direct response to that finding.
    """

    def __init__(self, num_classes: int, num_filters: int = 64, num_blocks: int = 4, dropout: float = 0.0):
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
        # dropout=0.0 (default) is a no-op (nn.Dropout(0) is the identity),
        # so this is a strict superset of the original architecture, not a
        # behavior change for any existing checkpoint/experiment. Added
        # during the class-imbalance-bias research pass (EXPERIMENTS.md
        # Experiment 24) — DSCNN had no regularization at all before this,
        # unlike BCResNet's BCResBlock (dropout=0.1 throughout).
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_mels, n_frames) -> logits (batch, num_classes)."""
        x = x.unsqueeze(1)  # (batch, 1, n_mels, n_frames)
        x = self.first_conv(x)
        for block in self.ds_blocks:
            x = block(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


class CRNN(nn.Module):
    """Depthwise-separable conv front end + bidirectional GRU + attention
    pooling. A direct response to a receptive-field finding about DSCNN
    above (EXPERIMENTS.md Experiment 28): DSCNN's stride-2 first conv
    followed by stride-1 3x3 blocks gives each output unit only ~24
    input frames (~240ms) of context before global average pooling —
    it classifies a 3s command as an unordered bag of quarter-second
    snippets, shorter than many single words ("temperature"). That
    matches the log: Experiment 7->11's +6.4pp for ~2K extra params
    came from one more block (a wider receptive field), and the
    carrier-phrase confusions (VOLUME vs TEMPERATURE) resisted every
    loss-level fix.

    Three changes, all cheap in parameters:
    1. Every second DS block strides by 2 in time and frequency, so the
       conv stack's receptive field grows geometrically instead of
       linearly (and the later blocks run on a much smaller map).
    2. Frequency is then collapsed by a learned 1x1 projection into a
       per-frame feature sequence.
    3. A bidirectional GRU models word order across the whole clip, and
       attention pooling (a learned per-frame weight, softmaxed over
       time) replaces global average pooling, so the one frame that
       says "up" vs "down" can dominate the summary instead of being
       averaged away.

    `sequence_features()` exposes the per-frame GRU outputs so an
    auxiliary training-only head (e.g. CTC on transcripts) can attach
    to them later without changing the deployed model.
    """

    def __init__(
        self,
        num_classes: int,
        channels: int = 64,
        num_blocks: int = 4,
        rnn_hidden: int = 64,
        n_mels: int = 40,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.ds_blocks = nn.ModuleList(
            [DepthwiseSeparableBlock(channels, channels, stride=2 if i % 2 == 1 else 1) for i in range(num_blocks)]
        )
        freq_bins = n_mels // 2  # after first_conv's stride
        for i in range(num_blocks):
            if i % 2 == 1:
                freq_bins = (freq_bins - 1) // 2 + 1  # 3x3, padding 1, stride 2
        self.freq_proj = nn.Sequential(
            nn.Conv1d(channels * freq_bins, rnn_hidden, kernel_size=1),
            nn.BatchNorm1d(rnn_hidden),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.GRU(rnn_hidden, rnn_hidden, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(2 * rnn_hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * rnn_hidden, num_classes)

    def sequence_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_mels, n_frames) -> (batch, time, 2 * rnn_hidden)."""
        x = x.unsqueeze(1)
        x = self.first_conv(x)
        for block in self.ds_blocks:
            x = block(x)
        b, c, f, t = x.shape
        x = self.freq_proj(x.reshape(b, c * f, t))  # (B, H, T)
        x, _ = self.rnn(x.transpose(1, 2))  # (B, T, 2H)
        return x

    def forward_with_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits plus the per-frame features they were pooled from, in one
        pass — for training-only auxiliary heads (see sequence_features)."""
        seq = self.sequence_features(x)
        weights = torch.softmax(self.attention(seq), dim=1)  # (B, T, 1)
        pooled = (weights * seq).sum(dim=1)
        return self.classifier(self.dropout(pooled)), seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_mels, n_frames) -> logits (batch, num_classes)."""
        return self.forward_with_sequence(x)[0]


class BCResBlock(nn.Module):
    """Broadcasted residual block — the core building block of BCResNet
    (below). Two paths, combined by addition:

    1. A frequency-depthwise 2D conv path that keeps the full frequency
       resolution (unlike DSCNN, which pools frequency away immediately).
    2. A temporal path: average-pool over frequency first (cheap), then
       a depthwise conv over time only, then a pointwise conv to mix
       channels — then *broadcast* back across the frequency dimension
       (via ordinary tensor broadcasting) and added to path 1's output.

    This keeps a full-resolution frequency-aware signal in the residual
    stream while still getting a wide temporal receptive field cheaply,
    which is the specific property DSCNN's global-average-pooling
    lacked (see DSCNN's docstring / EXPERIMENTS.md Experiment 1).

    Simplified relative to Kim, Chang, Lee, Sung, "Broadcasted Residual
    Learning for Efficient Keyword Spotting" (arXiv:2106.04140): uses
    standard BatchNorm2d rather than the paper's subspectral
    normalization, and BCResNet below stacks these at constant width
    rather than the paper's multi-stage widening schedule. What's kept
    is the mechanism the experiment is actually testing — broadcasted
    temporal+frequency residual fusion versus DSCNN's plain pool-then-
    classify — not a claim of exactly reproducing the paper's numbers.
    """

    def __init__(self, channels: int, freq_kernel: int = 3, time_kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.freq_dw = nn.Conv2d(
            channels, channels, kernel_size=(freq_kernel, 1), padding=(freq_kernel // 2, 0), groups=channels
        )
        self.freq_bn = nn.BatchNorm2d(channels)

        self.time_dw = nn.Conv2d(
            channels, channels, kernel_size=(1, time_kernel), padding=(0, time_kernel // 2), groups=channels
        )
        self.time_bn = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        freq_out = self.freq_bn(self.freq_dw(x))  # (B, C, F, T), full frequency resolution

        pooled = freq_out.mean(dim=2, keepdim=True)  # (B, C, 1, T) — cheap temporal-only view
        time_out = self.time_bn(self.time_dw(pooled))
        time_out = self.dropout(self.relu(self.pointwise(time_out)))

        out = self.relu(freq_out + time_out)  # broadcasts (B,C,1,T) across F automatically
        return out + identity


class BCResNet(nn.Module):
    """See BCResBlock above for the core mechanism and how this differs
    from the paper. A stem conv projects to `channels`, then a stack of
    BCResBlocks, global average pooling, then a linear classifier —
    MODEL.md's actual recommended architecture, tried second
    (EXPERIMENTS.md Experiment 2+) specifically to address the
    polarity-word confusion DSCNN showed in Experiment 1.
    """

    def __init__(self, num_classes: int, channels: int = 32, num_blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=5, stride=(2, 1), padding=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([BCResBlock(channels) for _ in range(num_blocks)])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_mels, n_frames) -> logits (batch, num_classes)."""
        x = x.unsqueeze(1)  # (batch, 1, n_mels, n_frames)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)
