# Training Experiments Log

Real results from actual training runs on the DGX (`ai-n002`, A100-40GB),
against `data/dataset_manifest.csv` (64,665 rows, 20 classes: 19 intents
+ `unknown_background`). See [MODEL.md](MODEL.md) for the technology
survey behind these choices and [DATASET.md](DATASET.md) for the data
itself. Every number below is copied from a real log file, not
estimated — log paths are given so any entry can be re-checked.

## Summary table

| # | Architecture | Augmentation | Epochs | Best val acc | Log |
|---|---|---|---:|---:|---|
| 1 | DS-CNN (24,276 params) | none | 30 | **65.29%** (epoch 28) | `logs/train_dscnn_run1.log` |

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
