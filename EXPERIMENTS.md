# Training Experiments Log

Real results from actual training runs on the DGX (`ai-n002`, A100-40GB),
against `data/dataset_manifest.csv` (64,665 rows, 20 classes: 19 intents
+ `unknown_background`). See [MODEL.md](MODEL.md) for the technology
survey behind these choices and [DATASET.md](DATASET.md) for the data
itself. Every number below is copied from a real log file, not
estimated — log paths are given so any entry can be re-checked.

**Methodological note**: Experiments 1 and 2 below were run before
`--seed` existed in `train.py` — neither run fixed a random seed, so
some of the differences between them are run-to-run noise (weight
init, batch shuffling), not purely the effect being tested. Experiment
3 onward all pin `--seed 0` for cleaner comparisons.

## Summary table

| # | Architecture | Augmentation | Epochs | Best val acc | Log |
|---|---|---|---:|---:|---|
| 1 | DS-CNN (24,276 params) | none | 30 | **65.29%** (epoch 28) | `logs/train_dscnn_run1.log` |
| 2 | DS-CNN (24,276 params) | SpecAugment | 30 | 63.21% (epoch 28) | `logs/train_dscnn_augment_run2.log` |

## Experiment 1 — DS-CNN baseline, no augmentation

**Setup**: `python -m vcm.train.train --epochs 30 --batch-size 128`,
DS-CNN (Zhang et al., "Hello Edge", arXiv:1711.07128), class-weighted
cross-entropy, Adam lr=1e-3, no data augmentation. Chosen as the first
architecture specifically for simplicity — validating the pipeline
mattered more than the best architecture on the first attempt.

**Result**: best val accuracy **65.29%** at epoch 28/30 (~13x random
baseline for 20 classes). Train loss fell steadily to 0.57; val loss
plateaued around 1.0-1.1 from epoch ~18 onward while train loss kept
falling — an early overfitting signal, though not yet severe by epoch
30.

**Per-class accuracy** (computed from the saved checkpoint against the
val split):

| Label | Acc | n | | Label | Acc | n |
|---|---:|---:|---|---|---:|---:|
| unknown_background | 100.0% | 68 | | LIGHT_OFF | 61.3% | 511 |
| CALL | 92.6% | 54 | | COLOR | 60.6% | 322 |
| PAUSE | 88.8% | 80 | | BRIGHTNESS | 60.0% | 395 |
| STOP | 88.0% | 108 | | LIST_REMINDERS | 53.8% | 249 |
| TIMER | 86.5% | 178 | | VOLUME_DOWN | 53.8% | 442 |
| ALARM | 82.9% | 333 | | PLAY_MUSIC | 51.8% | 658 |
| LIGHT_ON | 77.1% | 562 | | MESSAGE | 51.1% | 282 |
| TEMPERATURE | 73.8% | 1166 | | WEATHER | 42.3% | 534 |
| VOLUME_UP | 70.9% | 477 | | | | |
| NEXT | 68.5% | 54 | | | | |
| CREATE_REMINDER | 63.2% | 280 | | | | |
| TIME | 71.4% | 360 | | | | |

**Top confusions** (count of val examples with true label → predicted label):

| True → Predicted | Count |
|---|---:|
| TEMPERATURE → VOLUME_UP | 167 |
| VOLUME_DOWN → VOLUME_UP | 134 |
| WEATHER → TIME | 106 |
| VOLUME_UP → VOLUME_DOWN | 70 |
| LIGHT_OFF → LIGHT_ON | 65 |
| TEMPERATURE → VOLUME_DOWN | 59 |
| PLAY_MUSIC → TIME | 58 |
| WEATHER → LIST_REMINDERS | 57 |
| WEATHER → PLAY_MUSIC | 56 |
| PLAY_MUSIC → VOLUME_UP | 55 |

**Finding, not just noise**: the dominant failure mode is confusing
labels that share an identical carrier phrase and differ only in one
polarity/direction word — `"turn {up/down} the volume"` vs `"turn
{up/down} the temperature"` (FSC's TEMPERATURE commands literally use
volume-style "turn up/down" phrasing), `VOLUME_UP` vs `VOLUME_DOWN`
themselves, `LIGHT_ON` vs `LIGHT_OFF`. DS-CNN's global-average-pooling
at the end likely discards exactly the fine-grained temporal
information needed to reliably catch a single distinguishing word.
This is the concrete motivation for trying BC-ResNet next (below) —
its dual 1D-temporal/2D time-frequency path is designed to retain more
of that structure than a plain 2D-conv-then-pool architecture.

## Experiment 2 — DS-CNN + SpecAugment

**Setup**: `python -m vcm.train.train --model dscnn --augment --epochs
30 --batch-size 128` — same DS-CNN architecture, same class-weighted
cross-entropy, same Adam lr=1e-3, only change is SpecAugment (Park et
al., arXiv:1904.08779: 2 frequency masks up to 8 bins, 2 time masks up
to 40 frames) applied to training data only. **Caveat**: this run
predates `--seed` in `train.py`, so unlike later experiments it is not
seed-pinned — see the methodological note above.

**Result**: best val accuracy **63.21%** at epoch 28/30 — **2.08
points below Experiment 1's 65.29% baseline**, a real negative result
within this 30-epoch budget. Train loss ended noticeably higher than
Experiment 1's (1.01 vs 0.57), as expected — augmentation makes the
training task itself harder — but val loss did not correspondingly
improve past baseline, so this run shows augmentation making learning
harder without yet buying back generalization in the epochs available.

**Per-class accuracy**:

| Label | Acc | n | | Label | Acc | n |
|---|---:|---:|---|---|---:|---:|
| unknown_background | 100.0% | 68 | | VOLUME_DOWN | 66.3% | 442 |
| NEXT | 100.0% | 54 | | TIME | 59.7% | 360 |
| TIMER | 94.9% | 178 | | CREATE_REMINDER | 59.6% | 280 |
| CALL | 92.6% | 54 | | MESSAGE | 55.3% | 282 |
| PAUSE | 86.3% | 80 | | BRIGHTNESS | 54.9% | 395 |
| TEMPERATURE | 85.2% | 1166 | | PLAY_MUSIC | 52.1% | 658 |
| ALARM | 80.8% | 333 | | LIGHT_OFF | 49.5% | 511 |
| STOP | 74.1% | 108 | | LIST_REMINDERS | 43.4% | 249 |
| LIGHT_ON | 70.8% | 562 | | VOLUME_UP | 38.8% | 477 |
| COLOR | 69.3% | 322 | | WEATHER | 34.8% | 534 |

**Top confusions**:

| True → Predicted | Count |
|---|---:|
| VOLUME_UP → VOLUME_DOWN | 145 |
| PLAY_MUSIC → MESSAGE | 101 |
| WEATHER → TIME | 88 |
| WEATHER → PLAY_MUSIC | 83 |
| WEATHER → MESSAGE | 82 |
| LIGHT_OFF → LIGHT_ON | 80 |
| PLAY_MUSIC → TIME | 74 |
| TEMPERATURE → VOLUME_DOWN | 74 |
| TIME → PLAY_MUSIC | 54 |
| LIST_REMINDERS → PLAY_MUSIC | 49 |

**Analysis — a genuinely mixed result, not a clean win or loss**:
some labels improved substantially over Experiment 1 — TEMPERATURE
(73.8%→85.2%), TIMER (86.5%→94.9%), NEXT (68.5%→100%) — consistent
with SpecAugment's usual effect of forcing the model to rely on more
of the signal instead of one fragile cue. But VOLUME_UP collapsed
(70.9%→38.8%, now confused with VOLUME_DOWN even more than before),
and WEATHER (42.3%→34.8%) and LIGHT_OFF (61.3%→49.5%) also got worse.
The net effect was a small overall regression. Two plausible,
non-exclusive explanations: (1) time/frequency masking can erase the
one distinguishing word in short command phrases — exactly the
polarity words (VOLUME_UP vs VOLUME_DOWN) flagged as the baseline's
weak point in Experiment 1 — actively hurting the cases it was hoped
to help; (2) with the same 30-epoch budget, the harder augmented
training task may simply need more epochs to pay off, and this run
was cut off before that happened (train loss was still falling at
epoch 30 with no sign of plateauing, unlike Experiment 1).

**Confound to flag honestly**: neither this run nor Experiment 1 fixed
a random seed, so part of the -2.08pp gap could be ordinary run-to-run
variance (weight init, batch order) rather than a pure SpecAugment
effect. This is exactly why `--seed` was added to `train.py`
immediately after this run — Experiment 3 onward isolates the variable
under test more cleanly. Given the mixed per-class picture, the next
useful test isn't "is SpecAugment good or bad" in isolation, but
whether it changes conclusions once combined with BC-ResNet, whose
architecture is specifically meant to retain the fine temporal detail
that masking may currently be erasing.
