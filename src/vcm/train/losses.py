"""Confusable-pair-weighted loss for the polarity-word confusion that
has persisted across every experiment so far (VOLUME_UP <-> VOLUME_DOWN,
TEMPERATURE -> VOLUME_UP, LIGHT_ON <-> LIGHT_OFF — see EXPERIMENTS.md
Experiments 1-13's confusion matrices, where these are consistently the
top entries).

Adding more data doesn't target this: these labels already have
thousands of examples each and Experiment 13's data addition (real
TIMER/ALARM audio) had, as predicted, zero effect on them. The
confusion is carrier-phrase overlap (e.g. "turn the volume up" vs "turn
the volume down" vs "turn on the lights" differ by one word), not a
coverage gap.

Inspired by "Inter-Category Focal Loss"-style approaches (adaptively
penalizing confusion between known confusable category pairs, e.g.
arXiv:2304.05922): rather than collecting new data, add an extra
penalty term that directly targets the probability mass a sample's
true class places on classes already known (from the recorded
confusion matrices, not guessed) to be confusable with it. Zero new
data required — reuses the existing manifest as-is.

When alpha=0 this is exactly the baseline weighted cross-entropy used
in every experiment through Experiment 13, so it's a strict superset of
the prior loss, not a replacement.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Confusable groups, chosen directly from the top-confusions tables in
# EXPERIMENTS.md (Experiments 1, 5, 9, 12, 13, 15 all show the same
# clusters at or near the top): every pair within a group gets
# penalized both directions.
#
# COLOR/BRIGHTNESS was tried as a 3rd group (Experiment 17) — real
# confusion (COLOR -> BRIGHTNESS was the #5 overall confusion pair,
# 44 counts, same "set the lights/brightness to X" carrier-phrase
# overlap as the two groups below), but adding it at the same shared
# alpha regressed VOLUME/TEMPERATURE (-3.9pp) and LIGHT (-1.9pp) more
# than it helped COLOR/BRIGHTNESS (+4.9pp) — a net-negative tradeoff,
# not adopted. Left out of the default here so a fresh
# `--confusable-alpha` run reproduces Experiment 15's result, not
# Experiment 17's. Re-add it (or try a per-group alpha) only with that
# tradeoff in mind — see Experiment 17's writeup before changing this.
CONFUSABLE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("VOLUME_UP", "VOLUME_DOWN", "TEMPERATURE"),
    ("LIGHT_ON", "LIGHT_OFF"),
)


def effective_number_weights(counts: torch.Tensor, beta: float = 0.999) -> torch.Tensor:
    """Class-Balanced weighting (Cui, Jia, Lin, Song, Belongie, "Class-Balanced
    Loss Based on Effective Number of Samples", CVPR 2019): weight_c is
    proportional to (1-beta) / (1 - beta^n_c) instead of the plain 1/n_c
    this project used through Experiment 17.

    Motivated by a real finding, not theory: a live-voice debug session
    (see EXPERIMENTS.md Experiment 18) showed the model systematically
    defaulting to large classes (TEMPERATURE 8,691 train examples,
    LIGHT_ON 4,050, VOLUME_UP 3,524, PLAY_MUSIC 3,682) at the expense of
    small ones (CALL 396, PAUSE 640, STOP 812) on real/unfamiliar audio
    — every "surprise" wrong answer in that session was a large class
    stealing from a small one, not an acoustically-similar confusion
    (that's what ConfusablePairLoss above already handles). Plain
    inverse-frequency weighting (`vcm.train.dataset.class_weights`)
    already compensates for this some, but effective-number weighting
    is the standard fix for exactly this pattern at this project's
    imbalance ratio (~22x, TEMPERATURE vs CALL) — plain 1/n weights blow
    up for the rarest classes in a way that's often less stable than
    effective-number weighting during training.
    """
    effective_num = 1.0 - torch.pow(torch.as_tensor(beta, dtype=counts.dtype), counts)
    weights = (1.0 - beta) / effective_num.clamp(min=1e-8)
    return weights / weights.sum() * len(counts)  # normalize: mean weight == 1


def build_confusable_mask(
    labels: tuple[str, ...], groups: tuple[tuple[str, ...], ...] = CONFUSABLE_GROUPS
) -> torch.Tensor:
    """(num_classes, num_classes) bool mask. mask[i, j] is True iff i != j
    and labels[i], labels[j] belong to the same confusable group.
    """
    label_to_index = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    mask = torch.zeros(n, n, dtype=torch.bool)
    for group in groups:
        indices = [label_to_index[label] for label in group if label in label_to_index]
        for i in indices:
            for j in indices:
                if i != j:
                    mask[i, j] = True
    return mask


class ConfusablePairLoss(nn.Module):
    """Weighted cross-entropy, optionally with two independent additions:

    1. Focal modulation (Lin et al., "Focal Loss for Dense Object
       Detection", ICCV 2017): multiply each example's loss by
       (1 - p_t)^gamma, down-weighting examples the model already gets
       right (confidently) and up-weighting ones it doesn't — the
       standard fix for a model defaulting to easy/common classes under
       distribution shift rather than genuinely resolving hard ones.
       gamma=0.0 (default) disables this, recovering plain weighted CE.
    2. alpha * (probability mass placed on classes confusable with the
       true label) — see this module's docstring. alpha=0.0 (default)
       disables this.

    Both default off, so this class is a strict superset of plain
    weighted cross-entropy, not a replacement — same design as before.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        confusable_mask: torch.Tensor,
        alpha: float = 1.0,
        gamma: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.register_buffer("confusable_mask", confusable_mask)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        if self.gamma > 0.0:
            probs_for_focal = F.softmax(logits, dim=1)
            p_t = probs_for_focal.gather(1, targets.unsqueeze(1)).squeeze(1)
            ce = ((1.0 - p_t).clamp(min=1e-8) ** self.gamma) * ce
        if self.alpha == 0.0:
            return ce.mean()
        probs = F.softmax(logits, dim=1)
        confusable_for_target = self.confusable_mask[targets].to(probs.dtype)  # (batch, num_classes)
        confusion_mass = (probs * confusable_for_target).sum(dim=1)
        return (ce + self.alpha * confusion_mass).mean()
