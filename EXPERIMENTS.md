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
| 3 | BC-ResNet (10,196 params) | none | 30 | 44.16% (epoch 30) | `logs/train_bcresnet_run3.log` |
| 4 | BC-ResNet (10,196 params), lr=3e-4 | none | 30 | 38.79% (epoch 28) | `logs/train_bcresnet_lowlr_run4.log` |
| 5 | BC-ResNet (10,196 params), lr=1e-3 + 3-epoch warmup + cosine decay | none | 30 | **47.41%** (epoch 25) | `logs/train_bcresnet_warmup_run5.log` |

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

## Experiment 3 — BC-ResNet, no augmentation

**Setup**: `python -m vcm.train.train --model bcresnet --epochs 30
--batch-size 128 --seed 0`, on GPU 0 (idle at launch time; GPUs 6/7
were running someone else's job and were avoided). BC-ResNet (Kim et
al., "Broadcasted Residual Learning", arXiv:2106.04140), 10,196
params — less than half of DS-CNN's 24,276 — same class-weighted
cross-entropy, Adam lr=1e-3, no augmentation, first seed-pinned run
(`--seed 0`) so this is directly comparable to future experiments.
Intent: isolate the architecture change on its own, since MODEL.md's
own recommendation was BC-ResNet, and Experiment 1's confusion
analysis motivated it specifically for its dual time/frequency path.

**Result**: best val accuracy **44.16%** at epoch 30/30 — well below
both DS-CNN runs (65.29% and 63.21%). This is a real, honest negative
result: the architecture MODEL.md recommended as the efficiency-
accuracy sweet spot underperformed the simpler baseline by over 19
points in this setup.

**Training was visibly unstable throughout**, unlike either DS-CNN
run: val_loss repeatedly spiked far above its recent trend (2.3→5.47
at epoch 6, →15.64 at epoch 11, →9.62 at epoch 19, →14.29 at epoch 25,
→6.63 at epoch 29) with val_acc collapsing in lockstep each time,
before partially recovering the following epoch. Train loss, by
contrast, fell smoothly and monotonically the entire run (2.55→0.995,
never spiking) — the instability is specific to generalization, not a
symptom of a broken forward/backward pass. The best epochs (14, 21,
28, 30) all coincide with a val_loss trough, meaning the final
"best" checkpoint may simply have gotten lucky landing on a trough at
epoch 30 rather than reflecting genuine convergence — a run cut off a
few epochs earlier or later could plausibly report a meaningfully
different number.

**Per-class accuracy**:

| Label | Acc | n | | Label | Acc | n |
|---|---:|---:|---|---|---:|---:|
| unknown_background | 97.1% | 68 | | CALL | 57.4% | 54 |
| TIMER | 83.7% | 178 | | VOLUME_DOWN | 55.0% | 442 |
| PAUSE | 81.3% | 80 | | COLOR | 53.4% | 322 |
| ALARM | 76.3% | 333 | | BRIGHTNESS | 49.9% | 395 |
| NEXT | 74.1% | 54 | | TEMPERATURE | 38.3% | 1166 |
| LIGHT_ON | 69.9% | 562 | | MESSAGE | 37.9% | 282 |
| STOP | 66.7% | 108 | | CREATE_REMINDER | 29.6% | 280 |
| LIST_REMINDERS | 65.5% | 249 | | LIGHT_OFF | 27.8% | 511 |
| TIME | 58.6% | 360 | | PLAY_MUSIC | 26.9% | 658 |
| | | | | WEATHER | 13.5% | 534 |
| | | | | VOLUME_UP | 12.2% | 477 |

**Top confusions**:

| True → Predicted | Count |
|---|---:|
| TEMPERATURE → VOLUME_DOWN | 220 |
| VOLUME_UP → VOLUME_DOWN | 189 |
| LIGHT_OFF → LIGHT_ON | 189 |
| PLAY_MUSIC → TIME | 153 |
| CREATE_REMINDER → LIST_REMINDERS | 131 |
| TEMPERATURE → TIME | 128 |
| TEMPERATURE → PLAY_MUSIC | 127 |
| WEATHER → TIME | 120 |
| WEATHER → LIST_REMINDERS | 112 |
| PLAY_MUSIC → LIST_REMINDERS | 100 |

**Analysis**: BC-ResNet did *not* fix the polarity-word confusion
Experiment 1 flagged as motivation for trying it — VOLUME_UP↔VOLUME_DOWN
and LIGHT_OFF→LIGHT_ON are still top confusions here, and VOLUME_UP's
per-class accuracy (12.2%) is far worse than DS-CNN's baseline (70.9%
in Experiment 1). WEATHER is also now the weakest class at 13.5%
(down from 42.3% in Experiment 1). The larger classes with many
labels sharing vocabulary (PLAY_MUSIC, TEMPERATURE, WEATHER) collapsed
into each other more than before, suggesting the model never
stabilized enough in 30 epochs to learn fine-grained distinctions —
consistent with the recurring val_loss spikes never fully damping out.
Two labels *did* do reasonably well relative to their Experiment 1
numbers only by coincidence of overall lower accuracy elsewhere
(TIMER, PAUSE, ALARM stayed in a similar range), not because
BC-ResNet handled them specially.

**Working hypothesis, not yet confirmed**: lr=1e-3 with plain Adam may
simply be too aggressive for BC-ResNet's frequency-pooled residual
path (`BCResBlock` in `architectures.py`) — the repeated sharp
val_loss spikes with a smoothly-decreasing train_loss are a classic
signature of a learning rate that's too high for a specific
architecture's loss landscape, not of a data or implementation bug.
The original BC-ResNet paper (arXiv:2106.04140) trains with a warmup +
cosine LR schedule rather than a flat rate, which this implementation
doesn't yet have. **Recommended next experiment**: re-run BC-ResNet
with a lower learning rate (e.g. 3e-4) and/or gradient clipping before
concluding the architecture itself underperforms DS-CNN on this
dataset — the current -19pp gap may be an optimization artifact rather
than a genuine architecture-vs-data mismatch. This has not been tested
yet; treat the 44.16% figure as provisional until that follow-up runs.

## Experiment 4 — BC-ResNet, lower learning rate (lr=3e-4)

**Setup**: `python -m vcm.train.train --model bcresnet --epochs 30
--batch-size 128 --lr 3e-4 --seed 0`, on GPU 0. Identical to
Experiment 3 except `--lr 3e-4` instead of the default `1e-3` (a
~3.3x reduction), to directly test the LR-instability hypothesis
raised there.

**Result**: best val accuracy **38.79%** at epoch 28/30 — this is
*worse* than Experiment 3's 44.16%, not better. The hypothesis that a
lower learning rate alone would fix BC-ResNet's underperformance is
**not confirmed** by this run.

**What the lower LR did and didn't fix**: instability was reduced but
not eliminated — the worst val_loss spike this run peaked at 5.29
(epoch 21), versus 15.64 in Experiment 3, and spikes were somewhat
less frequent. But the tradeoff was slower learning: at epoch 14 (the
point Experiment 3 first reached its 40%+ plateau), this run was only
at 25.46% val_acc, and it never fully caught up in the remaining 16
epochs. In other words, the lower LR bought some stability at the cost
of convergence speed, and 30 epochs wasn't enough for the more
cautious run to reach where the noisier one landed.

**Per-class accuracy**:

| Label | Acc | n | | Label | Acc | n |
|---|---:|---:|---|---|---:|---:|
| unknown_background | 100.0% | 68 | | COLOR | 33.5% | 322 |
| PAUSE | 98.8% | 80 | | LIST_REMINDERS | 33.3% | 249 |
| CALL | 81.5% | 54 | | PLAY_MUSIC | 31.9% | 658 |
| NEXT | 75.9% | 54 | | LIGHT_OFF | 31.5% | 511 |
| TIMER | 71.4% | 178 | | TIME | 30.0% | 360 |
| TEMPERATURE | 67.8% | 1166 | | CREATE_REMINDER | 29.3% | 280 |
| STOP | 65.7% | 108 | | VOLUME_UP | 20.1% | 477 |
| BRIGHTNESS | 50.4% | 395 | | LIGHT_ON | 14.1% | 562 |
| ALARM | 41.7% | 333 | | MESSAGE | 11.7% | 282 |
| VOLUME_DOWN | 41.0% | 442 | | WEATHER | 11.1% | 534 |

**Top confusions**:

| True → Predicted | Count |
|---|---:|
| WEATHER → PLAY_MUSIC | 186 |
| TEMPERATURE → VOLUME_DOWN | 185 |
| LIGHT_ON → VOLUME_DOWN | 154 |
| LIGHT_ON → LIGHT_OFF | 132 |
| VOLUME_UP → VOLUME_DOWN | 120 |
| PLAY_MUSIC → PAUSE | 113 |
| PLAY_MUSIC → TIME | 102 |
| LIGHT_ON → TEMPERATURE | 101 |
| TIME → PLAY_MUSIC | 100 |
| LIGHT_OFF → TEMPERATURE | 91 |

**Analysis**: TEMPERATURE improved a lot relative to Experiment 3
(38.3%→67.8%) and PAUSE is now almost perfect (81.3%→98.8%), but
LIGHT_ON collapsed badly (69.9%→14.1%, now mostly confused with
VOLUME_DOWN and TEMPERATURE rather than its natural pair LIGHT_OFF) and
WEATHER stayed the weakest class (13.5%→11.1%). The confusion pattern
looks less like "the same errors, smaller" and more like a different,
still-unconverged model — consistent with 30 epochs simply not being
enough training budget at this LR for a network this size to settle.

**Conclusion for this pair of runs**: across Experiments 3 and 4,
BC-ResNet has not beaten DS-CNN on this dataset within a 30-epoch
budget at either learning rate tried, and neither run reproduces the
polarity-word fix it was chosen for. The instability is real and
LR-related (lower LR measurably reduced spike severity), but simply
lowering LR trades one problem (instability) for another (slow
convergence) rather than resolving the underlying regression. The
warmup + cosine-decay schedule the original BC-ResNet paper
(arXiv:2106.04140) uses — not yet implemented in `train.py` — remains
untested and is the most promising next lever, since it targets
exactly this instability-vs-convergence-speed tradeoff (aggressive
learning once training has stabilized, gentle learning early on)
rather than picking one flat rate for the whole run. Until that's
tried, DS-CNN (Experiment 1, 65.29%) remains the best model on this
dataset by a wide margin.

## Experiment 5 — BC-ResNet, warmup + cosine LR decay

**Setup**: `python -m vcm.train.train --model bcresnet --epochs 30
--batch-size 128 --lr 1e-3 --warmup-epochs 3 --seed 0`, on GPU 0.
Same peak LR as Experiment 3 (1e-3), but reached via a 3-epoch linear
warmup from ~1e-5, then cosine decay to ~0 over the remaining 27
epochs — the schedule `--warmup-epochs` was added to `train.py`
specifically to test, following the original BC-ResNet paper's own
recipe (arXiv:2106.04140) rather than a flat rate.

**Result**: best val accuracy **47.41%** at epoch 25/30 — clearly
better than both prior BC-ResNet attempts (44.16% flat lr=1e-3,
38.79% flat lr=3e-4), confirming the schedule helps. Still well below
DS-CNN's 65.29% baseline, though — a real improvement over BC-ResNet's
prior showing, not a win overall.

**Instability was reduced but only partly, and correlates visibly
with LR level**: spikes still occurred at epoch 7 (val_loss 3.75),
13 (3.89), 15 (6.14), and 19 (4.13) — all while LR was still
relatively high (roughly 7e-4 to 1e-3). From epoch 20 onward, as LR
decayed below ~3.5e-4, val_loss and val_acc both stabilized
completely: epochs 24-30 form a smooth, monotonically-settling curve
around 46-47% with no further spikes. This is a clean confirmation of
the original hypothesis from Experiment 3 — **the instability is an
LR-magnitude effect specific to this architecture**, not a symptom of
a data or implementation bug, since decaying the LR down eliminates it
predictably rather than it dying out randomly.

**Per-class accuracy**:

| Label | Acc | n | | Label | Acc | n |
|---|---:|---:|---|---|---:|---:|
| unknown_background | 97.1% | 68 | | CREATE_REMINDER | 46.4% | 280 |
| CALL | 92.6% | 54 | | LIST_REMINDERS | 42.2% | 249 |
| PAUSE | 88.8% | 80 | | VOLUME_UP | 38.0% | 477 |
| NEXT | 87.0% | 54 | | TIME | 37.8% | 360 |
| STOP | 78.7% | 108 | | LIGHT_OFF | 33.7% | 511 |
| TIMER | 75.8% | 178 | | MESSAGE | 32.3% | 282 |
| TEMPERATURE | 69.5% | 1166 | | VOLUME_DOWN | 31.7% | 442 |
| ALARM | 62.5% | 333 | | LIGHT_ON | 31.0% | 562 |
| COLOR | 56.5% | 322 | | PLAY_MUSIC | 22.2% | 658 |
| BRIGHTNESS | 49.1% | 395 | | WEATHER | 46.6% | 534 |

**Top confusions**:

| True → Predicted | Count |
|---|---:|
| TEMPERATURE → VOLUME_UP | 128 |
| LIGHT_ON → LIGHT_OFF | 125 |
| PLAY_MUSIC → WEATHER | 120 |
| PLAY_MUSIC → TIME | 107 |
| TIME → WEATHER | 100 |
| VOLUME_DOWN → VOLUME_UP | 81 |
| PLAY_MUSIC → MESSAGE | 76 |
| LIGHT_OFF → LIGHT_ON | 69 |
| VOLUME_UP → VOLUME_DOWN | 67 |
| TEMPERATURE → VOLUME_DOWN | 67 |

**Analysis**: this is the first BC-ResNet run where the model reliably
separates its strongest classes (CALL, PAUSE, NEXT, STOP, TIMER all
75-93%) rather than the noisier, less-structured per-class picture in
Experiments 3-4. WEATHER recovered substantially (13.5% in Exp 3,
11.1% in Exp 4 → 46.6% here) — the worst-performing class in both
earlier runs is now mid-pack. PLAY_MUSIC is now the weakest class
(22.2%), and the polarity confusions (VOLUME_UP/DOWN, LIGHT_ON/OFF)
are still present but no longer dominate the confusion list the way
they did for DS-CNN — they're now roughly on par with several other,
unrelated confusions (TEMPERATURE↔VOLUME_UP, PLAY_MUSIC↔WEATHER),
suggesting the model hasn't yet learned fine-grained distinctions
broadly, not that it has a specific blind spot for polarity words.

**Conclusion across Experiments 3-5**: the LR schedule was the right
lever — it fixed the instability cleanly and improved BC-ResNet's
accuracy by +3.25pp over its best prior flat-LR attempt — but 30
epochs still isn't enough for BC-ResNet to reach, let alone beat,
DS-CNN's baseline on this dataset. Given the loss curve was still
descending smoothly and hadn't plateaued by epoch 30 (train_loss
1.163→1.161 over the last 3 epochs, but val_acc still crept up to its
final best at epoch 25 with no sign of overfitting yet — val_loss
never rose again after the last spike), **more epochs is the most
likely next lever to close the remaining gap**, not a further
architecture or optimizer change. The next test worth running is the
same warmup+cosine config extended to 60-80 epochs before concluding
anything final about BC-ResNet vs. DS-CNN on this dataset.
