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
| 5 | BC-ResNet (10,196 params), lr=1e-3 + 3-epoch warmup + cosine decay | none | 30 | 47.41% (epoch 25) | `logs/train_bcresnet_warmup_run5.log` |
| 6 | BC-ResNet (10,196 params), lr=1e-3 + 5-epoch warmup + cosine decay | none | 80 | 58.99% (epoch 65) | `logs/train_bcresnet_warmup80_run6.log` |
| 7 | DS-CNN (24,276 params), lr=1e-3 + 3-epoch warmup + cosine decay | none | 30 | 65.96% (epoch 26) | `logs/train_dscnn_warmup_run7.log` |
| 8 | DS-CNN (24,276 params), lr=1e-3 + 3-epoch warmup + cosine decay | SpecAugment | 30 | 57.77% (epoch 30) | `logs/train_dscnn_warmup_augment_run8.log` |
| 9a | DS-CNN, same config as #7, `--train-fraction 0.25` | none | 30 | 44.06% (epoch 29) | `logs/train_dscnn_warmup_frac25_run9.log` |
| 9b | DS-CNN, same config as #7, `--train-fraction 0.50` | none | 30 | 54.77% (epoch 29) | `logs/train_dscnn_warmup_frac50_run10.log` |
| 9c | DS-CNN, same config as #7, `--train-fraction 0.75` | none | 30 | 60.72% (epoch 28) | `logs/train_dscnn_warmup_frac75_run11.log` |
| 10 | BC-ResNet, channels=48/blocks=8 (25,748 params), lr=1e-3 + 5-epoch warmup + cosine decay | none | 80 | 67.54% (epoch 53) | `logs/train_bcresnet_bigcap_run10.log` |
| 11 | DS-CNN, num_filters=60/num_blocks=5 (26,300 params), lr=1e-3 + 5-epoch warmup + cosine decay | none | 80 | 72.39% (epoch 63) | `logs/train_dscnn_bigcap_run11.log` |
| 12 | Same as #11, on the SLURP-quality-fixed manifest (62,405 rows, was 64,665) | none | 80 | 74.10% (epoch 46) | `logs/train_dscnn_bigcap_cleaned_run12.log` |
| 13 | Resumed from #12's checkpoint, on the Timers-and-Such-added manifest (63,476 rows) | none | 20 | **75.04%** (epoch 12) — best overall | `logs/train_dscnn_timers_finetune_run13.log` |
| 14 | Resumed from #13's checkpoint, ConfusablePairLoss (alpha=1.0) targeting VOLUME_UP/DOWN/TEMPERATURE + LIGHT_ON/OFF | none | 15 | 75.54% (epoch 10) — mixed result, see writeup | `logs/exp14_confusable.log` |
| 15 | Same as #14, alpha=2.0 (resumed from #13 directly, not #14) | none | 15 | **75.45%** (epoch 10) — best checkpoint overall, adopted as default | `logs/exp15_confusable_alpha2.log` |
| 16 | Same as #15, alpha=3.0 (resumed from #13 directly) | none | 15 | 75.36% (epoch 14) — overshoot, worse than #15 on the targeted metric too, see writeup | `logs/exp16_confusable_alpha3.log` |
| 17 | Same as #15, adds a 3rd confusable group (COLOR, BRIGHTNESS), alpha=2.0 unchanged | none | 15 | 75.30% (epoch 15) — real tradeoff, not a clean win, see writeup | `logs/exp17_confusable_colorfix.log` |
| 18 | Same as #15, adds effective-number class weights (beta=0.999, a real config mistake — weaker than baseline) + focal loss (gamma=2.0) | none | 15 | 75.29% (**epoch 1** — never improved on the starting checkpoint) | `logs/exp18_classbalanced_focal.log` |
| 19 | Same as #18 with the beta mistake removed (plain weights + focal gamma=2.0 only) | none | 15 | 74.86% (**epoch 1** again — same failure mode, ruling out the beta mistake as the cause) | `logs/exp19_focal_only.log` |
| 20 | Same as #15, adds `--max-per-class 2000` (resumed from #13) | none | 15 | 75.49% (**epoch 1** again) | `logs/exp20_max_per_class.log` |
| 21 | Same as #20, lr=1e-4 instead of 1e-3 (testing whether LR was the cause) | none | 15 | 75.23% (**epoch 1** again — ruled out LR as the cause) | `logs/exp21_max_per_class_lowlr.log` |
| 22 | Same as #20's cap, **from scratch** (not resumed) — isolates whether the resume-from setup itself was the problem | none | 80 | 70.57% (epoch 57) — real per-class tradeoff, see writeup | `logs/exp22_capped_scratch.log` |
| 23 | Same as #22, `--max-per-class 3000` (gentler cap) | none | 80 | 73.16% (epoch 60) — better tradeoff, still below #15 overall | `logs/exp23_capped3000_scratch.log` |
| 24 | QA-filtered Option B data (17,658→16,500) + dropout=0.2 + weight-decay=1e-4 + label-smoothing=0.1, all from scratch | none | 80 | 70.60% (epoch 76) — broad regression, over-regularization not the data, see writeup | `logs/exp24_qafiltered_tweaked.log` |
| 25 | Same as #24, QA-filtered data only (no dropout/weight-decay/label-smoothing) — isolates the QA filter's effect | none | 80 | 73.98% (epoch ~65) — confirms the QA filter itself isn't harmful; #24's regression was the regularization stack | `logs/exp25_qafiltered_only.log` |
| 26 | **ASR-cascade** (Whisper-base transcript → TF-IDF + logistic regression), a different architecture entirely — see writeup | n/a | n/a | **90.62%** test accuracy, real audio only — but not deployable, see writeup | `scripts/train_cascade_classifier.py` |
| 27 | DS-CNN + confusable-pair loss (alpha=2.0), **plus knowledge distillation** from #26's cascade (teacher, training-time only) | none | 80 | 75.78% val (epoch 55) — real mixed per-class tradeoff, see writeup | `logs/exp27_distillation.log` |
| 28 | **CRNN** (96,277 params: strided DS-conv front end + BiGRU + attention pooling), confusable-pair loss alpha=2.0, 3 seeds; plus DS-CNN baseline seeds 1/2 | none | 80 | 84.00 / 84.91 / 84.00% val; **79.8–80.8% real-speech test** (DS-CNN: 68.4–70.6%) | `logs/exp28_*.log` |
| 29a | Same as #28 + trim silence + 5.0s window, 3 seeds | none | 80 | 84.20 / 83.75 / 84.88% val; 80.7–80.8% real-speech test (≈ #28) | `logs/exp29a_*.log` |
| 29b | Same as #29a + **waveform augmentation** (noise, speed, reverb, start shift), 3 seeds | waveform | 80 | 88.29 / 88.51 / 88.46% val; **85.1–85.7% real-speech test** — best deployable model | `logs/exp29b_*.log` |
| 30 | Same as #29b + auxiliary word-level CTC head on the Whisper transcripts (training only); weight 0.5 × 3 seeds, 0.2 and 1.0 × seed 0 | waveform | 80 | 88.31 / 88.44 / 88.40% val; 84.6–84.8% real-speech test (weight 0.5) — **slightly worse than #29b, not adopted** | `logs/exp30_*.log` |

## Parked / to-do

Not yet run. Recorded here so they aren't lost, not because they're
scheduled — pick back up when there's a reason to chase more accuracy
again.

- ~~Data-scaling (learning-curve) test~~ — **done, see Experiment 9
  below.** Result: more data would meaningfully help — the curve has
  not plateaued at 100%.
- **Targeted (non-random) time masking** to protect the one
  distinguishing word in polarity-confused commands (VOLUME_UP/DOWN,
  LIGHT_ON/OFF) instead of SpecAugment's random masking (ruled out in
  Experiment 8). Needs word-level time alignment, which the pipeline
  doesn't have yet — a bigger lift than the other items here.
- ~~Confusable-pair loss tuning~~ — **done, see Experiments 14-16.**
  alpha=2.0 (`dscnn_bigcap_confusable2_best.pt`, the repo default) is
  a real peak — alpha=1.0 undershoots (barely moves the VOLUME/
  TEMPERATURE confusion) and alpha=3.0 overshoots (regresses
  VOLUME/TEMPERATURE back to baseline and pushes LIGHT's other-wrong
  rate above baseline too). Closed out, not paused — a different
  mechanism (per-group alpha, or a different penalty formulation),
  not more of the same knob, would be needed to improve on this
  further.
- ~~BC-ResNet with more channels/blocks~~ — **done, see Experiment 10
  below.** Result: matching DS-CNN's param count (channels=48,
  blocks=8) closed the gap and then some — 67.54%, the new best model
  overall.
- ~~DS-CNN with matched capacity (close the fair-comparison gap)~~ —
  **done, see Experiment 11 below.** Result: the gap was real — DS-CNN
  at matched capacity (72.39%) beats BC-ResNet at matched capacity
  (67.54%) by a solid +4.85pp. The Experiment 10 conclusion ("BC-ResNet
  is the new best model") is now superseded: DS-CNN was simply never
  given a fair shot at the same capacity bump. **DS-CNN is the best
  architecture found so far on this dataset**, at any capacity tried.
- **When the final/converged class dataset lands, don't assume today's
  ranking (DS-CNN-bigcap best) carries over — re-run and re-compare.**
  Two reasons this specific ranking could still change: (1) it's from
  a single seeded run each, not multiple seeds, so some of the +4.85pp
  gap over BC-ResNet-bigcap could still be run-to-run variance rather
  than a pure architecture effect (worth a repeat-seed check before
  fully trusting the margin); (2) DS-CNN's edge is tied to how it
  handles the polarity-word confusion and general phrase structure of
  *this* dataset's sources — a different dataset could shift which
  architecture's inductive bias fits best. The training pipeline is
  dataset-agnostic (reads whatever's in `data/dataset_manifest.csv`),
  so re-running the known configs (DS-CNN default and matched-capacity,
  BC-ResNet default and matched-capacity) is cheap — just don't skip
  it once the dataset changes.
- **ARCHIVED, not dropped: real-recording tool for CALL/NEXT/LIST_REMINDERS**
  (`scripts/record_real_examples.py`, see DATASET.md step 8). Built,
  tested, ready to use — paused pending course-adviser confirmation on
  recording new personal voice data for this project. Left in the repo
  rather than removed so it can be picked back up the moment that's
  cleared.

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

## Experiment 6 — BC-ResNet, warmup + cosine LR decay, 80 epochs

**Setup**: `python -m vcm.train.train --model bcresnet --epochs 80
--batch-size 128 --lr 1e-3 --warmup-epochs 5 --seed 0`, on GPU 0.
Direct follow-up to Experiment 5's own recommendation: same schedule
shape (linear warmup then cosine decay to ~0), extended from 30 to 80
total epochs (warmup lengthened from 3→5 epochs, keeping it roughly
proportional) since Experiment 5's loss curve hadn't plateaued and
showed no overfitting signal by epoch 30.

**Result**: best val accuracy **58.99%** at epoch 65/80 — a real
+11.58pp jump over Experiment 5's 47.41%, confirming more training
time was the right next lever, not a further architecture or
optimizer change. Still below DS-CNN's 65.29% (Experiment 1), a
-6.3pp gap, but far closer than any prior BC-ResNet run.

**Instability recurred at high LR just as before, at the same
absolute epochs proportionally**: a severe spike at epoch 10 (val_loss
8.37, matching the LR level `Experiment 5` was at around its own
epoch ~4, since the cosine schedule here decays over 75 epochs instead
of 27) and continued bumpiness through roughly epoch 35, tracking
LR staying above ~6e-4. From epoch 45 onward, as LR dropped below
~5e-4, the curve settled into smooth, steady improvement with no
further spikes — val_loss fell monotonically from 1.38 (epoch 45) to
1.22 (epoch 75/80), and val_acc climbed from 54% to a 59% plateau.
This is the same LR-magnitude-driven pattern as Experiment 5, just
stretched over more epochs because the cosine schedule here decays
more slowly in absolute terms — further evidence the instability is
tied to LR magnitude specifically, not epoch count.

**Per-class accuracy**:

| Label | Acc | n | | Label | Acc | n |
|---|---:|---:|---|---|---:|---:|
| unknown_background | 97.1% | 68 | | MESSAGE | 55.0% | 282 |
| PAUSE | 95.0% | 80 | | CREATE_REMINDER | 51.1% | 280 |
| CALL | 94.4% | 54 | | BRIGHTNESS | 50.6% | 395 |
| NEXT | 90.7% | 54 | | PLAY_MUSIC | 42.4% | 658 |
| TEMPERATURE | 85.8% | 1166 | | WEATHER | 41.8% | 534 |
| STOP | 83.3% | 108 | | LIGHT_OFF | 40.3% | 511 |
| TIMER | 82.6% | 178 | | VOLUME_UP | 38.8% | 477 |
| ALARM | 67.9% | 333 | | LIST_REMINDERS | 37.4% | 249 |
| LIGHT_ON | 64.2% | 562 | | | | |
| COLOR | 62.4% | 322 | | | | |
| TIME | 56.1% | 360 | | | | |
| VOLUME_DOWN | 55.0% | 442 | | | | |

**Top confusions**:

| True → Predicted | Count |
|---|---:|
| VOLUME_UP → VOLUME_DOWN | 115 |
| LIGHT_OFF → LIGHT_ON | 110 |
| WEATHER → TIME | 100 |
| PLAY_MUSIC → WEATHER | 87 |
| WEATHER → PLAY_MUSIC | 72 |
| WEATHER → MESSAGE | 69 |
| PLAY_MUSIC → MESSAGE | 66 |
| PLAY_MUSIC → TIME | 65 |
| VOLUME_UP → TEMPERATURE | 63 |
| LIGHT_ON → TEMPERATURE | 61 |

**Analysis**: this is the strongest BC-ResNet result so far by a wide
margin, and its per-class profile now looks much more like DS-CNN's
(Experiment 1) than its own earlier attempts — TEMPERATURE (85.8%),
CALL (94.4%), TIMER (82.6%), PAUSE (95.0%) are all close to or above
DS-CNN's numbers for the same classes. But the polarity confusions
DS-CNN struggled with are **back at the top of the list** here
(VOLUME_UP→VOLUME_DOWN, LIGHT_OFF→LIGHT_ON), which BC-ResNet's dual
time/frequency path was originally chosen to fix — at this accuracy
level it has *not* solved that specific problem any better than
DS-CNN did. LIST_REMINDERS (37.4%) and VOLUME_UP (38.8%) are now the
weakest classes.

**Conclusion across Experiments 3-6**: with enough training (80
epochs, correct LR schedule), BC-ResNet gets close to DS-CNN
(58.99% vs 65.29%, a 6.3pp gap that has been closing steadily with
more epochs: 44%→47%→59% across Experiments 3/5/6) but still hasn't
matched or beaten it on this dataset, and it has not resolved the
polarity-confusion problem that motivated trying it in the first
place. Given DS-CNN reaches a higher number in 30 epochs (roughly 8.5
GPU-minutes) than BC-ResNet needs 65+ epochs (roughly 65 GPU-minutes)
to approach, **DS-CNN remains the recommended model for this dataset**
under the current setup — both on accuracy and on training-time
efficiency, which also matters for MODEL.md's stated goal of keeping
the whole pipeline (not just inference) practical on available
compute. BC-ResNet's smaller size (10,196 vs DS-CNN's 24,276 params)
remains a real advantage for the on-device deployment target, so it
isn't ruled out — but closing the remaining accuracy gap would need a
different lever than epochs or LR schedule alone (e.g., augmentation
combined with the now-stable schedule, or more channels/blocks), not
yet tested.

## Experiment 7 — DS-CNN, warmup + cosine LR decay

**Setup**: `python -m vcm.train.train --model dscnn --epochs 30
--batch-size 128 --lr 1e-3 --warmup-epochs 3 --seed 0`, on GPU 0.
The same schedule that helped BC-ResNet (Experiments 5-6), applied to
DS-CNN for the first time — Experiment 1's baseline used a flat LR
throughout, so this checks whether the schedule is a general win or
something specific to fixing BC-ResNet's instability.

**Result**: best val accuracy **65.96%** at epoch 26/30 — a modest but
real +0.67pp improvement over Experiment 1's 65.29% flat-LR baseline,
and the **best result across all 7 experiments so far**. Unlike
BC-ResNet, DS-CNN showed no instability at any point in this run —
val_loss decreased smoothly and monotonically the entire time
(2.98→1.06), confirming the spiking behavior in Experiments 3, 5, and
6 was specific to BC-ResNet's architecture, not a general property of
training on this dataset at lr=1e-3.

**Convergence was also much faster**: this run reached 64.81% by
epoch 20, matching Experiment 1's final epoch-28 result 8 epochs
earlier, and continued to a new best by epoch 26. The warmup phase
(epochs 1-3, ramping to peak LR) cost some early-epoch accuracy
relative to Experiment 1's immediate flat-1e-3 start, but the
subsequent cosine decay more than made up for it.

**Per-class accuracy**:

| Label | Acc | n | | Label | Acc | n |
|---|---:|---:|---|---|---:|---:|
| unknown_background | 100.0% | 68 | | CREATE_REMINDER | 63.9% | 280 |
| CALL | 92.6% | 54 | | BRIGHTNESS | 62.0% | 395 |
| NEXT | 87.0% | 54 | | TIME | 60.8% | 360 |
| TIMER | 86.5% | 178 | | VOLUME_UP | 57.7% | 477 |
| PAUSE | 85.0% | 80 | | WEATHER | 54.1% | 534 |
| TEMPERATURE | 84.1% | 1166 | | MESSAGE | 53.6% | 282 |
| ALARM | 80.5% | 333 | | PLAY_MUSIC | 50.9% | 658 |
| STOP | 77.8% | 108 | | VOLUME_DOWN | 50.0% | 442 |
| LIGHT_ON | 70.8% | 562 | | LIST_REMINDERS | 47.0% | 249 |
| COLOR | 67.1% | 322 | | | | |
| LIGHT_OFF | 64.0% | 511 | | | | |

**Top confusions**:

| True → Predicted | Count |
|---|---:|
| VOLUME_DOWN → VOLUME_UP | 120 |
| VOLUME_UP → VOLUME_DOWN | 72 |
| WEATHER → TIME | 68 |
| TIME → WEATHER | 65 |
| TEMPERATURE → VOLUME_UP | 63 |
| PLAY_MUSIC → MESSAGE | 61 |
| PLAY_MUSIC → TIME | 57 |
| WEATHER → PLAY_MUSIC | 52 |
| PLAY_MUSIC → WEATHER | 50 |
| LIGHT_ON → LIGHT_OFF | 45 |

**Analysis**: this is the same DS-CNN architecture and confusion
pattern from Experiment 1 (polarity words still dominate: VOLUME_UP/
DOWN and LIGHT_ON/OFF are still top confusions) but with meaningfully
fewer errors overall — VOLUME_DOWN→VOLUME_UP dropped from 134 to 120,
LIGHT_OFF→LIGHT_ON from 65 to 45. The schedule improved general
convergence without resolving the specific polarity-word weak point
Experiment 1 flagged — consistent with the schedule improving how well
the model fits the data it's given, not what it's structurally capable
of distinguishing.

**Current standing recommendation**: this checkpoint
(`checkpoints/dscnn_warmup_best.pt`) is now the best model produced
across this project's experiments, and warmup+cosine LR scheduling
looks like a good default going forward for any future run of either
architecture, at negligible extra cost (`--warmup-epochs 3` on top of
an otherwise-identical run). The polarity-word confusion remains the
dataset's dominant unsolved failure mode across every architecture and
schedule tried so far — the strongest untested next step for
addressing it specifically is not a new architecture or schedule, but
augmenting/re-weighting the training data itself to emphasize the
distinguishing word in commands that otherwise share a carrier phrase
(e.g. targeted time-masking that preserves the polarity word instead
of SpecAugment's random masking, which Experiment 2 showed can erase
exactly that word).

## Experiment 8 — DS-CNN, warmup + cosine LR decay + SpecAugment

**Setup**: `python -m vcm.train.train --model dscnn --augment --epochs
30 --batch-size 128 --lr 1e-3 --warmup-epochs 3 --seed 0`, on GPU 0.
Retests SpecAugment (the negative result from Experiment 2) on top of
Experiment 7's now-proven warmup+cosine schedule, to check whether
Experiment 2's regression was itself an artifact of the old flat-LR,
unseeded setup rather than a real property of SpecAugment on this
dataset.

**Result**: best val accuracy **57.77%** at epoch 30/30 — **8.19
points below Experiment 7's 65.96%** (same config, no augmentation),
and even below Experiment 1's original flat-LR baseline (65.29%). The
regression is confirmed, not explained away: SpecAugment is a genuine
net-negative for this dataset/architecture combination, independent of
the LR schedule or seeding used underneath it. Training was still
gradually improving at epoch 30 but had clearly begun to plateau
(val_acc 57.7-57.8% across the last 5 epochs) — more epochs alone is
unlikely to close an 8pp gap the way it helped BC-ResNet's much larger,
still-descending gap in Experiment 6.

**Per-class accuracy**:

| Label | Acc | n | | Label | Acc | n |
|---|---:|---:|---|---|---:|---:|
| unknown_background | 98.5% | 68 | | WEATHER | 56.7% | 534 |
| CALL | 94.4% | 54 | | CREATE_REMINDER | 56.1% | 280 |
| TIMER | 89.3% | 178 | | LIGHT_OFF | 49.9% | 511 |
| NEXT | 88.9% | 54 | | LIST_REMINDERS | 45.4% | 249 |
| PAUSE | 88.8% | 80 | | MESSAGE | 44.0% | 282 |
| TEMPERATURE | 81.1% | 1166 | | TIME | 35.3% | 360 |
| STOP | 76.9% | 108 | | PLAY_MUSIC | 28.0% | 658 |
| ALARM | 74.5% | 333 | | VOLUME_UP | 23.3% | 477 |
| LIGHT_ON | 66.6% | 562 | | | | |
| COLOR | 60.3% | 322 | | | | |
| BRIGHTNESS | 59.2% | 395 | | | | |
| VOLUME_DOWN | 58.8% | 442 | | | | |

**Top confusions**:

| True → Predicted | Count |
|---|---:|
| VOLUME_UP → VOLUME_DOWN | 167 |
| PLAY_MUSIC → WEATHER | 132 |
| TIME → WEATHER | 119 |
| LIGHT_OFF → LIGHT_ON | 84 |
| PLAY_MUSIC → MESSAGE | 76 |
| TEMPERATURE → VOLUME_DOWN | 74 |
| PLAY_MUSIC → TIME | 70 |
| LIST_REMINDERS → WEATHER | 68 |
| WEATHER → TIME | 65 |
| VOLUME_UP → TEMPERATURE | 62 |

**Analysis**: VOLUME_UP is now the weakest class by far (23.3%, worse
than any prior experiment for this label), and its top confusion
(VOLUME_UP→VOLUME_DOWN, 167 — the single largest confusion count seen
in any experiment) confirms the theory from Experiment 2: masking a
short command's one distinguishing word (here, "up" vs "down") is
actively harmful, not neutral, for exactly the classes that already
depend on a single word to disambiguate. WEATHER is heavily involved
in confusions again too (both as source and target), similar to its
behavior without the schedule.

**Conclusion — SpecAugment is now ruled out for this dataset**: across
two independent tests (Experiment 2 at flat LR, Experiment 8 at the
proven warmup+cosine schedule), SpecAugment produced a real,
substantial regression both times. This isn't a schedule or seeding
artifact — it's specific to how random time/frequency masking
interacts with short, single-distinguishing-word commands. **DS-CNN +
warmup+cosine LR schedule, no augmentation (Experiment 7, 65.96%)
remains the best and recommended configuration** for this dataset.
Any future augmentation attempt should be targeted (protect the
distinguishing word) rather than random, per the note at the end of
Experiment 7 — but that requires word-level alignment infrastructure
not yet built, and is not a small follow-up.

## Experiment 9 — Data-scaling (learning-curve) test

**Motivation**: after 8 experiments, accuracy had settled around
65-66% regardless of architecture or LR schedule, prompting the
question of whether the dataset itself is the limiting factor. Indirect
evidence at the time (small classes like CALL/PAUSE/NEXT already
scoring 80-95%+, the dominant confusion looking like a phrasing overlap
baked into the source data) pointed toward "probably structural, not
data-limited" — but that was inference, not a direct test. This
experiment is the direct test: added `--train-fraction` to `train.py`
(randomly subsamples the training split, seeded/reproducible, val/test
untouched) and trained the winning config (DS-CNN + 3-epoch warmup +
cosine decay, seed 0) at 25%, 50%, 75%, and 100% (= Experiment 7,
reused rather than rerun) of the training data.

**Setup**: `python -m vcm.train.train --model dscnn --epochs 30
--batch-size 128 --lr 1e-3 --warmup-epochs 3 --seed 0 --train-fraction
{0.25,0.50,0.75}`, three runs launched in parallel on GPUs 0/1/2.

**Result — the curve has not plateaued**:

| Train fraction | Train rows | Best val acc |
|---:|---:|---:|
| 25% | 12,151 | 44.06% |
| 50% | 24,302 | 54.77% |
| 75% | 36,453 | 60.72% |
| 100% | 48,605 | **65.96%** (Experiment 7) |

Looking at gain per *doubling* of data (the natural unit for a learning
curve): 25%→50% gained +10.71pp, and 50%→100% gained +11.19pp — nearly
identical. A curve that was running out of headroom would show
shrinking gains per doubling; this one hasn't started to bend yet.
**Direct answer to "could the dataset be insufficient": yes — more
data would very likely raise accuracy further**, which revises the
earlier indirect-evidence-based read toward "probably structural."

**Per-class comparison (25% vs. 100%/Experiment 7) — the gain is
concentrated exactly where it's needed most**: the classes that
improve the most from 4x more data are the ones already flagged as the
dataset's hardest, most-confused classes — VOLUME_UP (18.7%→57.7%,
+39.0pp), LIGHT_OFF (26.0%→64.0%, +38.0pp), PLAY_MUSIC
(15.4%→50.9%, +35.5pp), TIME (26.4%→60.8%, +34.4pp), COLOR
(34.8%→67.1%, +32.3pp). Meanwhile the classes that were already easy at
25% data (CALL, TEMPERATURE, NEXT, TIMER, ALARM, STOP, PAUSE — all
66%+ even with a quarter of the data) only gained 9-19pp, since they
had much less room to grow.

**Reconciling this with Experiments 1-8's confusion findings**: both
things are true at once. The polarity/carrier-phrase confusion
(VOLUME_UP/DOWN, LIGHT_ON/OFF, TEMPERATURE overlapping VOLUME phrasing)
is a real structural property of how the source data phrases these
commands — more of the same phrasing pattern won't teach the model a
*new* distinguishing cue. But this experiment shows those exact classes
are also the most data-hungry — they need more examples than the easy
classes to reach the same accuracy, and they hadn't saturated even at
the full current dataset size. So the practical, actionable conclusion
is: **growing the dataset further (more real+synthetic examples for
the already-covered labels, not necessarily new sources) is a
legitimate, evidence-backed lever**, not a dead end — it just won't by
itself eliminate the confusion the way a truly new source of
information (e.g. protecting the distinguishing word specifically)
might.

## Experiment 10 — BC-ResNet, capacity matched to DS-CNN

**Motivation**: Experiments 5-6 fixed BC-ResNet's training instability
(warmup+cosine schedule) and closed most of its gap to DS-CNN with more
epochs (58.99% at 80 epochs), but it still trailed DS-CNN's 65.96%
(Experiment 7). BC-ResNet's default config (channels=32, blocks=6) is
only 10,196 params — less than half of DS-CNN's 24,276 — so the
remaining gap could be an under-capacity model, not an architecture
that's inherently worse for this task. Added `--width`/`--depth` CLI
overrides to `train.py` (map to each model's own constructor kwargs)
specifically to test this without code changes.

**Setup**: `python -m vcm.train.train --model bcresnet --epochs 80
--batch-size 128 --lr 1e-3 --warmup-epochs 5 --seed 0 --width 48
--depth 8`, on GPU 0. `channels=48, blocks=8` gives **25,748 params** —
closely matched to DS-CNN's 24,276, a fair capacity comparison rather
than a much bigger model.

**Result — new best model overall**: best val accuracy **67.54%** at
epoch 53/80, beating DS-CNN's 65.96% (Experiment 7) by **+1.58pp**.
This is the first BC-ResNet run to beat DS-CNN on this dataset, after
Experiments 3-6 all fell short at the smaller default capacity.

**The same instability-then-settle pattern recurred, and capacity
didn't make it worse**: spikes still occurred while LR was high
(epochs 8, 17, 28 all had a val_loss jump followed by a partial or
full recovery the next epoch), consistent with Experiments 5-6's
finding that this is LR-magnitude-driven, not capacity-driven. Once
LR dropped below ~3e-4 around epoch 50, the curve settled into a
smooth, low-variance plateau in the 63-68% range for the rest of the
run (epochs 50-80) — the same settling behavior seen at the smaller
capacity, just reaching a higher plateau.

**Convergence was also much faster than the smaller BC-ResNet**: this
run matched Experiment 6's final 58.99% by epoch ~24 (Experiment 6
needed all 80 epochs to get there), and crossed DS-CNN's 65.96%
benchmark by epoch ~45 — roughly half the epoch budget Experiment 6
needed just to get close.

**Per-class accuracy**:

| Label | Acc | n | | Label | Acc | n |
|---|---:|---:|---|---|---:|---:|
| unknown_background | 100.0% | 68 | | LIGHT_OFF | 61.8% | 511 |
| NEXT | 96.3% | 54 | | VOLUME_UP | 59.3% | 477 |
| PAUSE | 95.0% | 80 | | TIME | 53.6% | 360 |
| TEMPERATURE | 93.8% | 1166 | | PLAY_MUSIC | 52.9% | 658 |
| CALL | 88.9% | 54 | | MESSAGE | 51.8% | 282 |
| TIMER | 88.2% | 178 | | WEATHER | 50.2% | 534 |
| STOP | 85.2% | 108 | | LIST_REMINDERS | 49.4% | 249 |
| ALARM | 76.6% | 333 | | VOLUME_DOWN | 48.0% | 442 |
| CREATE_REMINDER | 73.6% | 280 | | | | |
| LIGHT_ON | 70.5% | 562 | | | | |
| COLOR | 69.3% | 322 | | | | |
| BRIGHTNESS | 62.8% | 395 | | | | |

**Top confusions**:

| True → Predicted | Count |
|---|---:|
| VOLUME_DOWN → VOLUME_UP | 89 |
| WEATHER → PLAY_MUSIC | 72 |
| LIGHT_OFF → TEMPERATURE | 66 |
| VOLUME_UP → TEMPERATURE | 63 |
| TIME → PLAY_MUSIC | 61 |
| VOLUME_UP → VOLUME_DOWN | 60 |
| LIGHT_ON → LIGHT_OFF | 56 |
| WEATHER → TIME | 53 |
| VOLUME_DOWN → TEMPERATURE | 53 |
| PLAY_MUSIC → MESSAGE | 49 |

**Analysis**: TEMPERATURE jumped to 93.8% — the strongest large class
by far, well above DS-CNN's 84.1% for the same label (Experiment 7).
But this came with a new, previously-minor confusion pattern:
LIGHT_OFF→TEMPERATURE (66) and VOLUME_DOWN/UP→TEMPERATURE (63, 53) are
now prominent, where they barely registered in DS-CNN's confusion
list. This reads as the model becoming very confident about
TEMPERATURE specifically (likely because it's the largest class by
far, 1,166 val examples) at the expense of pulling in some borderline
VOLUME/LIGHT cases that share its "turn up/down X" carrier phrase —
the classic pattern of a class-imbalance-driven bias, only partly
offset by the class-weighted loss. The core VOLUME_UP/DOWN and
LIGHT_ON/OFF polarity confusion is still present (89, 60, 56 counts)
but is no longer the single dominant failure mode the way it was for
DS-CNN.

**Current standing recommendation**: `checkpoints/bcresnet_bigcap_best.pt`
was the best model produced up through Experiment 10 (67.54% val
accuracy). **Superseded by Experiment 11 below**, which gave DS-CNN
the same fair capacity treatment and found it pulls further ahead
(72.39%) — so this recommendation no longer holds; see Experiment 11's
own conclusion for the current standing recommendation. Trade-off to
note for the RPi target: at 25,748 params BC-ResNet is now roughly the
same size as DS-CNN (24,276), so the "BC-ResNet is much smaller"
advantage from Experiments 3-6 no longer applies at this capacity —
the choice between them is purely about accuracy and confusion
pattern, not model size. Both remaining open items — targeted masking
for the polarity confusion, and growing the dataset (Experiment 9's
finding) — apply to whichever architecture is carried forward.

## Experiment 11 — DS-CNN, capacity matched to BC-ResNet's bigcap config

**Motivation**: directly closes the fair-comparison gap flagged after
Experiment 10 — BC-ResNet was given a capacity bump (10,196→25,748
params) and beat DS-CNN's default-capacity result, but DS-CNN itself
had never been tested at a matching capacity. Without this run, "BC-
ResNet is the better architecture" and "more capacity helps, and we
only tried it on one side" were indistinguishable.

**Setup**: `python -m vcm.train.train --model dscnn --epochs 80
--batch-size 128 --lr 1e-3 --warmup-epochs 5 --seed 0 --width 60
--depth 5`, on GPU 2. `num_filters=60, num_blocks=5` gives **26,300
params** — within 2% of BC-ResNet's bigcap size (25,748), a close
capacity match rather than a much bigger model.

**Result — DS-CNN pulls further ahead, decisively**: best val accuracy
**72.39%** at epoch 63/80, beating BC-ResNet's matched-capacity result
(67.54%, Experiment 10) by **+4.85pp**, and DS-CNN's own default-
capacity result (65.96%, Experiment 7) by +6.43pp. This is a much
larger gap than Experiment 10's narrow +1.58pp BC-ResNet-over-DS-CNN
margin — DS-CNN's capacity-scaling response is real and substantial,
not a rounding error. **DS-CNN is the best architecture found on this
dataset at every capacity level tried so far.**

**No instability at all, at any point in this run** — unlike every
BC-ResNet run (Experiments 3, 5, 6, 10), DS-CNN's val_loss decreased
essentially monotonically for all 80 epochs regardless of capacity.
This reinforces the Experiment 7 finding that the spiking behavior is
specific to BC-ResNet's broadcasted-residual mechanism, not a general
property of training on this dataset at higher capacity or higher LR.

**Per-class accuracy**:

| Label | Acc | n | | Label | Acc | n |
|---|---:|---:|---|---|---:|---:|
| unknown_background | 100.0% | 68 | | LIGHT_OFF | 72.6% | 511 |
| NEXT | 94.4% | 54 | | CREATE_REMINDER | 69.3% | 280 |
| CALL | 94.4% | 54 | | TIME | 65.8% | 360 |
| TIMER | 92.1% | 178 | | VOLUME_UP | 65.6% | 477 |
| TEMPERATURE | 87.1% | 1166 | | VOLUME_DOWN | 65.2% | 442 |
| PAUSE | 86.3% | 80 | | BRIGHTNESS | 64.3% | 395 |
| ALARM | 83.2% | 333 | | PLAY_MUSIC | 63.5% | 658 |
| LIGHT_ON | 81.7% | 562 | | MESSAGE | 60.3% | 282 |
| STOP | 79.6% | 108 | | WEATHER | 56.7% | 534 |
| COLOR | 73.9% | 322 | | LIST_REMINDERS | 49.4% | 249 |

**Top confusions**:

| True → Predicted | Count |
|---|---:|
| VOLUME_UP → VOLUME_DOWN | 67 |
| VOLUME_DOWN → VOLUME_UP | 63 |
| WEATHER → TIME | 60 |
| TIME → WEATHER | 53 |
| PLAY_MUSIC → WEATHER | 45 |
| WEATHER → MESSAGE | 37 |
| PLAY_MUSIC → TIME | 36 |
| LIGHT_OFF → LIGHT_ON | 36 |
| WEATHER → LIST_REMINDERS | 35 |
| WEATHER → PLAY_MUSIC | 35 |

**Analysis**: every class improved over the smaller DS-CNN
(Experiment 7) except `unknown_background` (already saturated at
100%). The improvements are broad, not concentrated in a few classes —
consistent with more capacity generally helping a model fit more of
the data's real structure, rather than fixing one specific weak point.
Confusion counts also dropped across the board (e.g.
VOLUME_UP↔VOLUME_DOWN combined fell from 192 in Experiment 7 to 130
here) — the polarity confusion persists but is measurably smaller.
**Directly cross-validating the DATASET.md source-composition
diagnosis**: LIST_REMINDERS (49.4%) and WEATHER (56.7%) remain the two
weakest classes even at this much higher overall accuracy, and both
are the SLURP-dominated labels flagged there (69% and 85% SLURP
respectively) — real evidence that this is a data-quality ceiling,
not something more capacity alone fixes.

**DS-CNN is the recommended architecture** going forward — it has now
won at both default and matched capacity, with zero training
instability at any setting tried, unlike BC-ResNet. The capacity-
scaling headroom itself looks promising too (default 24,276→65.96%,
bigger 26,300→72.39% for only ~2,000 more params) — an even larger
DS-CNN has not been tried and is a plausible next lever, distinct from
and complementary to the dataset-quality fix. **Superseded by
Experiment 12 below**, which applied that dataset-quality fix and
found a real, further improvement — see there for the current best
checkpoint.

## Experiment 12 — same config as #11, on the SLURP-quality-fixed dataset

**Motivation**: directly tests whether the SLURP mapping-purity fix
(DATASET.md's "Known per-label quality signal" — dropped LIST_REMINDERS'
`lists_query` mapping entirely, filtered pure-date questions out of
TIME, small junk cleanups for WEATHER/MESSAGE) actually improves
accuracy, rather than just trusting the diagnosis. Same exact model
config as Experiment 11, same seed, only the dataset changed — any
accuracy difference is attributable to the fix, not noise from a
different setup.

**Setup**: `python -m vcm.train.train --model dscnn --epochs 80
--batch-size 128 --lr 1e-3 --warmup-epochs 5 --seed 0 --width 60
--depth 5`, on GPU 1, against the rebuilt `data/dataset_manifest.csv`
(62,405 rows, down from 64,665 — see DATASET.md for exactly what was
removed and why).

**Result — the fix works**: best val accuracy **74.10%** at epoch
46/80, beating Experiment 11's identical-config result (72.39%) by
**+1.71pp**, purely from the dataset change.

**Per-label impact on the four targeted labels** (Experiment 11 →
Experiment 12):

| Label | Before | After | Change |
|---|---:|---:|---:|
| LIST_REMINDERS | 49.4% | **100.0%** | **+50.6pp** |
| WEATHER | 56.7% | 62.8% | +6.1pp |
| TIME | 65.8% | 70.7% | +4.9pp |
| MESSAGE | 60.3% | 53.9% | **-6.4pp** |

LIST_REMINDERS's jump is the standout: once purified down to only
Option B's 558 clean, on-taxonomy synthetic examples (val n dropped
from 249 to 54), the model gets it perfectly right. WEATHER and TIME
both improved meaningfully, consistent with removing genuinely
mismatched training signal. **MESSAGE is a real anomaly worth flagging
honestly**: only 3 sentences (10 rows) were removed from MESSAGE
specifically — nowhere near enough to directly cause a 6.4pp drop.
This is more likely a side effect of the overall redistribution
(inverse-frequency class weights shifted slightly since the total
label composition changed, and MESSAGE's confusion pattern shows heavy
bidirectional confusion with PLAY_MUSIC — 38 each way) than a flaw in
the MESSAGE-specific fix itself. Neither run used multiple seeds, so
some of this is plausibly run-to-run noise rather than a real
regression — worth re-checking if MESSAGE remains weak in future runs.

**Per-class accuracy (full)**:

| Label | Acc | n | | Label | Acc | n |
|---|---:|---:|---|---|---:|---:|
| unknown_background | 100.0% | 68 | | TIME | 70.7% | 225 |
| LIST_REMINDERS | 100.0% | 54 | | VOLUME_DOWN | 67.0% | 442 |
| CALL | 92.6% | 54 | | BRIGHTNESS | 66.6% | 395 |
| PAUSE | 91.3% | 80 | | PLAY_MUSIC | 63.5% | 658 |
| TIMER | 89.3% | 178 | | WEATHER | 62.8% | 530 |
| TEMPERATURE | 87.5% | 1166 | | VOLUME_UP | 62.3% | 477 |
| ALARM | 82.6% | 333 | | MESSAGE | 53.9% | 282 |
| STOP | 80.6% | 108 | | | | |
| NEXT | 79.6% | 54 | | | | |
| LIGHT_ON | 79.5% | 562 | | | | |
| LIGHT_OFF | 75.5% | 511 | | | | |
| COLOR | 74.5% | 322 | | | | |
| CREATE_REMINDER | 72.5% | 280 | | | | |

**Top confusions**:

| True → Predicted | Count |
|---|---:|
| VOLUME_UP → VOLUME_DOWN | 94 |
| VOLUME_DOWN → VOLUME_UP | 68 |
| PLAY_MUSIC → WEATHER | 57 |
| WEATHER → PLAY_MUSIC | 47 |
| TEMPERATURE → VOLUME_UP | 42 |
| PLAY_MUSIC → MESSAGE | 38 |
| MESSAGE → PLAY_MUSIC | 38 |
| CREATE_REMINDER → PLAY_MUSIC | 35 |
| BRIGHTNESS → COLOR | 34 |
| TIME → WEATHER | 33 |

**A few other labels moved by ±2-3pp in either direction** (e.g. NEXT
94.4%→79.6%, LIGHT_OFF 72.6%→75.5%) despite not being touched by the
dataset fix at all — consistent with ordinary run-to-run variance at
these class sizes (several have val n≈54-108), not a systematic
effect. This is exactly the kind of noise the "multiple seeds" caveat
in the parked to-do list is meant to eventually rule out.

Was the best model and dataset+config combination through Experiment
12. **Superseded by Experiment 13 below**, which added real TIMER/ALARM
audio (Timers and Such) on top of this. The MESSAGE regression and the
polarity confusion (VOLUME_UP/DOWN) remain open — see Experiment 13's
own conclusion for the current standing recommendation.

## Experiment 13 — resume from Experiment 12, add Timers and Such (real TIMER/ALARM)

**Motivation**: directly measures whether adding real TIMER/ALARM audio
(DATASET.md step 9, ~1,071 recordings from Timers and Such) actually
helps, and does so cheaply — instead of a full 80-epoch retrain from
scratch, this resumes from Experiment 12's already-trained weights and
continues training on the updated dataset for only 20 epochs. Training
on the *full* updated manifest (not just the new rows in isolation)
avoids the catastrophic-forgetting risk of fine-tuning on a narrow
subset — every batch still sees all 20 classes, just starting from a
good initialization instead of random.

**A free, zero-training baseline came first**: before touching the
DGX, Experiment 12's checkpoint (which had never seen any Timers-and-
Such audio) was evaluated directly against real Timers-and-Such data —
pure inference, no training cost. Result: **20.9% on TIMER, 52.3% on
ALARM** across all 1,071 real recordings — concrete, measured evidence
of exactly how severe the synthetic-to-real gap was for TIMER
specifically (100% Chatterbox TTS beforehand), not just theoretical.

**Setup**: `python -m vcm.train.train --model dscnn --epochs 20
--batch-size 128 --lr 1e-3 --warmup-epochs 2 --seed 0 --width 60
--depth 5 --resume-from checkpoints/dscnn_bigcap_cleaned_best.pt`, on
the manifest rebuilt to include Timers and Such (63,476 rows, up from
62,405). ~20 minutes wall-clock vs. ~80 minutes for a from-scratch run
at the same epoch-for-epoch cost — roughly 4x cheaper for this check.

**Result**: best val accuracy **75.04%** at epoch 12/20, beating
Experiment 12's 74.10% by +0.94pp overall (note: this val set isn't
identical to Experiment 12's — it grew slightly since Timers-and-Such
contributes its own val rows too, so this overall number isn't a pure
apples-to-apples delta by itself; the per-label held-out comparison
below is the rigorous one).

**The real result — a clean, leak-free before/after on data neither
checkpoint ever trained on** (restricted to Timers-and-Such's val+test
rows only, 253 recordings, so this isn't contaminated by what
Experiment 13 just trained on):

| Label | Before (Experiment 12) | After (Experiment 13) | Change |
|---|---:|---:|---:|
| TIMER | 31.4% (54/172) | **91.3%** (157/172) | **+59.9pp** |
| ALARM | 63.0% (51/81) | **88.9%** (72/81) | **+25.9pp** |

This is the cleanest, largest single-fix improvement in the whole
project so far — real human audio for a previously 100%-synthetic
label closed the vast majority of the gap in one pass, exactly as the
synthetic-to-real generalization research predicted.

**Per-class accuracy (full validation split)**:

| Label | Acc | n | | Label | Acc | n |
|---|---:|---:|---|---|---:|---:|
| unknown_background | 100.0% | 68 | | CREATE_REMINDER | 69.6% | 280 |
| LIST_REMINDERS | 98.2% | 54 | | VOLUME_UP | 68.6% | 477 |
| TIMER | 94.8% | 270 | | VOLUME_DOWN | 66.7% | 442 |
| CALL | 92.6% | 54 | | TIME | 66.7% | 225 |
| ALARM | 88.0% | 374 | | MESSAGE | 63.8% | 282 |
| LIGHT_ON | 86.1% | 562 | | PLAY_MUSIC | 61.6% | 658 |
| TEMPERATURE | 85.9% | 1166 | | WEATHER | 57.9% | 530 |
| PAUSE | 83.8% | 80 | | | | |
| NEXT | 83.3% | 54 | | | | |
| STOP | 81.5% | 108 | | | | |
| LIGHT_OFF | 73.2% | 511 | | | | |
| COLOR | 71.7% | 322 | | | | |
| BRIGHTNESS | 71.1% | 395 | | | | |

**Top confusions**:

| True → Predicted | Count |
|---|---:|
| VOLUME_UP → VOLUME_DOWN | 73 |
| PLAY_MUSIC → MESSAGE | 66 |
| VOLUME_DOWN → VOLUME_UP | 61 |
| WEATHER → MESSAGE | 56 |
| PLAY_MUSIC → WEATHER | 46 |
| WEATHER → PLAY_MUSIC | 46 |
| LIGHT_OFF → LIGHT_ON | 44 |
| TEMPERATURE → VOLUME_UP | 36 |
| TIME → WEATHER | 33 |
| COLOR → BRIGHTNESS | 32 |

**Analysis**: TIMER (94.8%) and ALARM (88.0%) are now both strong
classes, a complete reversal from before this fix. The polarity
confusion (VOLUME_UP/DOWN) persists as the top confusion pair, exactly
as in every prior experiment — real TIMER/ALARM audio fixed the
problem it was meant to fix and, as expected, did nothing for the
unrelated carrier-phrase-overlap problem. MESSAGE (63.8%) remains weak
and now shows new confusion with WEATHER (56 counts) alongside its
existing PLAY_MUSIC confusion — still an open anomaly, not resolved by
this change (expected, since this fix didn't touch MESSAGE at all).

**Current standing recommendation**: `checkpoints/dscnn_bigcap_timers_best.pt`
is now the best model overall. This also validates `--resume-from` as
a real, reusable technique for cheaply testing future data additions —
a fraction of the cost of a from-scratch retrain, with a rigorous
held-out-only evaluation (not the noisier overall val-accuracy delta)
as the way to honestly measure a specific data addition's effect. The
polarity confusion and MESSAGE anomaly remain the two clearest open
threads, unaffected by this fix as expected.

## Experiment 14 — ConfusablePairLoss, targeting the polarity confusion directly

**Motivation**: the polarity confusion (VOLUME_UP/VOLUME_DOWN,
TEMPERATURE->VOLUME_UP, LIGHT_ON/LIGHT_OFF) has topped every confusion
matrix from Experiment 1 through 13 and is unaffected by adding more
data (Experiment 13 confirmed this — these labels already have
thousands of examples). Research into confusable-pair/minimal-pair
keyword-spotting literature (Inter-Category Focal Loss-style
approaches, e.g. arXiv:2304.05922) suggested a loss-level fix instead:
penalize the probability mass a sample's true class places on classes
already known (from our own confusion matrices, not guessed) to be
confusable with it. Implemented in `vcm.train.losses.ConfusablePairLoss`
(see that module's docstring) — zero new data needed, a strict
superset of the existing weighted cross-entropy (alpha=0 reduces to
it exactly).

Two groups defined directly from the top-confusions tables above:
`(VOLUME_UP, VOLUME_DOWN, TEMPERATURE)` and `(LIGHT_ON, LIGHT_OFF)`.

**Setup**: `python -m vcm.train.train --model dscnn --epochs 15
--batch-size 128 --lr 1e-3 --warmup-epochs 2 --seed 0 --width 60
--depth 5 --resume-from checkpoints/dscnn_bigcap_timers_best.pt
--confusable-alpha 1.0 --out checkpoints/dscnn_bigcap_confusable_best.pt`.
Same cheap resume-and-continue methodology as Experiment 13 (~15
minutes vs. a full retrain).

**Result**: best val accuracy **75.54%** at epoch 10/15, +0.50pp over
Experiment 13's 75.04%. On its own this looks like a small win, but
the overall number hides what actually happened — a targeted
within-group-confusion metric (computed directly from
`vcm.train.losses.CONFUSABLE_GROUPS`, not the generic top-10 list)
gives the honest picture:

| Group | Metric | #13 (before) | #14 (after) | Change |
|---|---|---:|---:|---:|
| VOLUME_UP/DOWN/TEMPERATURE | correct | 77.9% | 80.3% | +2.4pp |
| | within-group confusion (the targeted problem) | 10.7% | 10.7% | **0.0pp — unchanged** |
| | other-wrong | 11.4% | 9.0% | -2.4pp |
| LIGHT_ON/LIGHT_OFF | correct | 80.0% | 79.1% | -0.9pp |
| | within-group confusion (the targeted problem) | 6.1% | 4.4% | **-1.7pp — improved** |
| | other-wrong | 14.0% | 16.5% | +2.5pp |

**Analysis — an honest, mixed result, not a clean win**:
- **LIGHT_ON/LIGHT_OFF**: the loss did what it was designed to do —
  the specific LIGHT_ON<->LIGHT_OFF confusion dropped 6.1%->4.4%. But
  that gain was more than offset by a rise in unrelated wrong
  predictions (14.0%->16.5%), so overall group accuracy actually
  *fell* slightly (80.0%->79.1%). The penalty term seems to have
  pushed probability mass off the confusable partner class, but not
  reliably onto the correct class — sometimes onto a third, unrelated
  class instead.
- **VOLUME_UP/VOLUME_DOWN/TEMPERATURE**: the targeted confusion didn't
  move at all (10.7% both times) — individually, VOLUME_DOWN's
  within-group confusion got slightly *worse* (16.5%->17.4%) while
  TEMPERATURE's improved slightly (5.7%->5.3%). The +2.4pp group
  accuracy gain came entirely from fewer *unrelated* errors
  (TEMPERATURE's other-wrong dropped 8.4%->5.2%) — a real improvement,
  but not the one this experiment set out to produce, and not
  obviously caused by the confusable-pair mechanism rather than just
  more resumed-training epochs.
- Overall: the loss is not obviously wrong, but alpha=1.0 does not
  cleanly fix the polarity confusion it targets. Full evaluation:
  `python <scratchpad>/eval_confusable_groups.py <ckpt1> <ckpt2>` (ad
  hoc script, not committed to the repo — logic is short enough to
  reproduce if needed: for each confusable group, split each class's
  errors into within-group vs. other-wrong).

**Open follow-ups, not yet tried**: a lower alpha (weaker penalty,
less likely to overshoot into unrelated classes), a higher alpha
(stronger push, in case 1.0 undershoots), or restricting the penalty
to only the LIGHT group (the only one where it worked as intended)
while leaving VOLUME/TEMPERATURE on plain cross-entropy. Not run yet —
next step is a decision on whether this line of tuning is worth
another DGX pass or whether to park it, matching this project's
efficiency-first approach to training runs.

**Current standing recommendation (superseded below by Experiment 15)**:
unchanged from Experiment 13 — `checkpoints/dscnn_bigcap_timers_best.pt`
is still the safer default given this experiment's mixed, not-clearly-
better result on the LIGHT_ON/LIGHT_OFF group specifically.
`dscnn_bigcap_confusable_best.pt` is marginally higher on raw val
accuracy (75.54% vs 75.04%) and genuinely better on the LIGHT confusion
specifically, so it's a reasonable alternative, not a clear regression
— this is a "pick one and note the tradeoff" situation, not an obvious
win either way.

## Experiment 15 — ConfusablePairLoss, alpha=2.0

**Motivation**: Experiment 14 (alpha=1.0) barely moved the targeted
VOLUME/TEMPERATURE confusion at all (10.7%->10.7%) while working, but
imperfectly, on LIGHT_ON/LIGHT_OFF (6.1%->4.4%, with some of that gain
eaten by new unrelated errors). Doubling alpha tests whether the
mechanism was directionally right but just too weak at 1.0.

**Setup**: identical to Experiment 14 but `--confusable-alpha 2.0`,
resumed from Experiment 13's checkpoint directly (not Experiment 14's)
to keep this an apples-to-apples alpha comparison against the same
starting point, isolating alpha as the only variable.

**Result**: best val accuracy 75.45% at epoch 10/15 — *lower* than
Experiment 14's 75.54% on raw accuracy. But the same group-level
breakdown used in Experiment 14 tells a different, more encouraging
story:

| Group | Metric | #13 (baseline) | #14 (alpha=1.0) | #15 (alpha=2.0) |
|---|---|---:|---:|---:|
| VOLUME_UP/DOWN/TEMPERATURE | correct | 77.9% | 80.3% | **81.7%** |
| | within-group confusion (targeted) | 10.7% | 10.7% | **9.6%** |
| | other-wrong | 11.4% | 9.0% | **8.7%** |
| LIGHT_ON/LIGHT_OFF | correct | 80.0% | 79.1% | **80.4%** |
| | within-group confusion (targeted) | 6.1% | 4.4% | 4.9% |
| | other-wrong | 14.0% | 16.5% | **14.6%** |

**Analysis**: alpha=2.0 is a clear improvement over alpha=1.0, and
unlike alpha=1.0 it beats the Experiment 13 baseline on *every* metric
for the VOLUME/TEMPERATURE group simultaneously (higher correct, lower
targeted confusion, lower other-wrong) — the first result in this
whole line of experiments where the confusable-pair mechanism visibly
moves the metric it was built to move, without the collateral damage
alpha=1.0 showed. For LIGHT_ON/LIGHT_OFF, alpha=2.0 gives up a little
of alpha=1.0's targeted-confusion win (4.4%->4.9%) but more than makes
up for it elsewhere: correct accuracy is now *above* the Experiment 13
baseline (80.4% vs 80.0%, vs. alpha=1.0's 79.1%), and other-wrong is
much closer to baseline (14.6% vs. alpha=1.0's 16.5%). Raw overall val
accuracy is a worse guide here than the targeted breakdown — alpha=2.0
scores lower on the headline number (75.45% vs 75.54%) while being the
clearly stronger checkpoint on the actual problem being fixed.

**Current standing recommendation — adopted**: `checkpoints/dscnn_bigcap_confusable2_best.pt`
(alpha=2.0) is now the repo-wide default (`demo_infer.py`, `TESTING.md`
updated) — strictly better than the Experiment 13 baseline on the
polarity-confusion metric this whole line of experiments targets, at a
negligible (0.09pp) cost in overall val accuracy. **Confirmed by
Experiment 16 below**: the 1.0->2.0 trend does not continue to 3.0 —
alpha=2.0 is a real peak, not a point on a still-climbing curve.

## Experiment 16 — ConfusablePairLoss, alpha=3.0 (does the trend continue?)

**Motivation**: Experiment 15 (alpha=2.0) beat alpha=1.0 on every
group metric simultaneously, with no sign of plateauing. This tests
whether pushing alpha further keeps helping or starts overshooting.

**Setup**: identical to Experiments 14/15 but `--confusable-alpha 3.0`,
resumed from Experiment 13's checkpoint directly (same isolation
methodology — alpha is the only variable that changes between 13, 14,
15, 16).

**Result**: best val accuracy 75.36% at epoch 14/15 — between
Experiment 14 (75.54%) and Experiment 15 (75.45%) on the raw number,
but the group breakdown shows this is a genuine reversal, not noise:

| Group | Metric | #13 (baseline) | #15 (alpha=2.0) | #16 (alpha=3.0) |
|---|---|---:|---:|---:|
| VOLUME_UP/DOWN/TEMPERATURE | correct | 77.9% | **81.7%** | 78.0% |
| | within-group confusion (targeted) | 10.7% | **9.6%** | 9.9% |
| | other-wrong | 11.4% | **8.7%** | 12.1% |
| LIGHT_ON/LIGHT_OFF | correct | 80.0% | **80.4%** | 78.0% |
| | within-group confusion (targeted) | 6.1% | 4.9% | **4.6%** |
| | other-wrong | 14.0% | **14.6%** | 17.4% |

**Analysis**: the trend broke. alpha=3.0 squeezes the LIGHT group's
targeted confusion slightly lower than alpha=2.0 (4.9%->4.6%) — the
mechanism is still doing what it's designed to do — but at a much
steeper cost: other-wrong jumps to 17.4% (worse than both alpha=2.0
*and* the no-loss baseline), and the VOLUME/TEMPERATURE group regresses
on every metric back to roughly baseline levels, wiping out
Experiment 15's gains there entirely. This is the classic overshoot
failure mode for this kind of penalty: pushing probability mass hard
enough off the confusable partner starts dumping it onto unrelated
classes instead of the correct one, rather than continuing to
concentrate it on the right answer. alpha=2.0 sits right at the peak
before that trade-off turns negative.

**Current standing recommendation — unchanged**: `checkpoints/dscnn_bigcap_confusable2_best.pt`
(alpha=2.0, Experiment 15) remains the best checkpoint and the repo
default. Alpha tuning is now closed out, not just paused — 1.0, 2.0,
and 3.0 collectively bracket a real peak at 2.0, so further sweeping
in either direction is low-value without a different mechanism (e.g.
per-group alpha, or a different penalty formulation) rather than just
more of the same knob.

## Experiment 17 — add COLOR/BRIGHTNESS as a 3rd confusable group

**Motivation**: live-voice testing (real speech, not the synthetic val
split) surfaced COLOR commands as a weak point. Checked against data
before touching anything: COLOR was already confirmed weak in the
Experiment 15 checkpoint (63.0% val accuracy) with `COLOR -> BRIGHTNESS`
as the #5 overall confusion (44 misclassifications) — the same
carrier-phrase-overlap pattern as the two existing confusable groups
("set the lights/brightness to X"), so extending the existing loss
mechanism rather than building something new was the obvious next
step.

**Setup**: identical to Experiment 15 (`--confusable-alpha 2.0`,
resumed from Experiment 13's checkpoint) but with `CONFUSABLE_GROUPS`
extended to include `("COLOR", "BRIGHTNESS")` as a third group,
alongside the existing two.

**Result**: best val accuracy 75.30% at epoch 15/15 — essentially flat
vs. Experiment 15's 75.45%. The group breakdown shows why this is a
genuine three-way tradeoff, not a clean win:

| Group | Metric | #15 (2 groups) | #17 (3 groups) |
|---|---|---:|---:|
| COLOR/BRIGHTNESS | correct | 65.8% | **70.7%** (+4.9pp) |
| | within-group confusion (targeted) | 7.9% | 7.7% (flat) |
| VOLUME_UP/DOWN/TEMPERATURE | correct | **81.7%** | 77.8% (**-3.9pp**) |
| LIGHT_ON/LIGHT_OFF | correct | **80.4%** | 78.5% (**-1.9pp**) |

**Analysis**: adding a third simultaneous confusable-pair penalty
genuinely improved COLOR/BRIGHTNESS's overall accuracy (+4.9pp,
717 samples) — but at the cost of regressing the two groups
Experiment 15 had already fixed, most notably VOLUME/TEMPERATURE
(-3.9pp on 2,085 samples, the largest of the three groups). This reads
as the three penalty terms competing during training and diluting each
other, not a mechanism that scales cleanly to more groups at a fixed
alpha. Also consistent with the pattern seen in Experiments 14/16:
COLOR/BRIGHTNESS's *targeted* confusion (the thing the loss is
supposed to fix) barely moved (7.9%->7.7%) — the accuracy gain came
from fewer unrelated errors, not the mechanism cleanly separating the
two confusable classes.

Net effect weighted by group size is likely negative overall (the two
regressed groups are more than 3x the sample count of the one that
improved), even though the headline val accuracy looks nearly flat.

**Current standing recommendation — unchanged**: `checkpoints/dscnn_bigcap_confusable2_best.pt`
(2 groups, alpha=2.0) remains the repo default.
`dscnn_bigcap_confusable_colorfix_best.pt` is not adopted — it's a
real tradeoff (better COLOR/BRIGHTNESS, worse VOLUME/TEMPERATURE and
LIGHT), not a strict improvement. If COLOR/BRIGHTNESS is worth fixing
on its own, a per-group alpha (weighting each group's penalty
independently rather than sharing one alpha across all three) is the
logical next experiment rather than adding groups at a fixed shared
alpha.

## Experiments 18-21 — chasing a majority-class-bias fix, four dead ends with one useful pattern

**Motivation**: a live-voice `--debug` session (real speech, not the
synthetic val split) found a new pattern distinct from the polarity
confusion above — the model defaulting to large classes (LIGHT_ON,
VOLUME_UP, PLAY_MUSIC, WEATHER, TEMPERATURE) at the expense of small
ones (PAUSE, STOP, CALL, MESSAGE) on real/unfamiliar audio. Training
set sizes confirmed a real ~22x imbalance (TEMPERATURE 8,691 vs. CALL
396). Four experiments tried to fix this via the loss function, all
resumed from Experiment 13's checkpoint:

- **18**: effective-number class weights (Cui et al. 2019, beta=0.999)
  + focal loss (gamma=2.0), on top of the existing confusable-pair
  loss. Best result: **epoch 1** (75.29%) — none of the following 14
  epochs of real training ever beat the barely-touched starting point.
  Root cause found on inspection: beta=0.999 gave CALL only 3.1x more
  weight than TEMPERATURE, versus the 21.9x ratio plain inverse-
  frequency weighting (already in use since Experiment 1) already
  provided — a real configuration mistake that *weakened* rather than
  strengthened small-class compensation.
- **19**: same setup with the beta mistake removed (focal loss alone,
  gamma=2.0). Still epoch-1-is-best (74.86%) — ruling out the beta
  mistake as the actual cause.
- **20**: `--max-per-class 2000` (dataset-level downsampling of
  oversized classes instead of loss reweighting) at the original
  lr=1e-3. Still epoch-1-is-best (75.49%).
- **21**: same as #20 but lr=1e-4 (10x lower, testing whether the peak
  LR was too disruptive for a resumed checkpoint). Still epoch-1-is-
  best (75.23%) — ruling out LR magnitude too.

**The real finding**: four different mechanisms (reweighting, focal
modulation, dataset downsampling), at two different learning rates,
all show the identical failure mode when resumed from Experiment 13's
checkpoint — the model never improves past its barely-perturbed
starting point. The common thread isn't any one technique; it's that
*any* new mechanism layered on the already-working confusable-pair
loss, in the resume-from-a-converged-checkpoint pattern, fails to find
anything better within 15 epochs. Experiment 15 (confusable-pair loss
*alone*, same starting checkpoint, same warmup/LR shape) genuinely
improved past epoch 1 by contrast — so the resume-from pattern itself
still works, just not for stacking a second mechanism on top of it.
This directly motivated Experiments 22-23 below: test downsampling
**from scratch** instead, to separate "does downsampling help" from
"does the resume-from pattern break for a second mechanism."

## Experiment 22 — max-per-class downsampling, from scratch

**Setup**: `--max-per-class 2000`, DS-CNN bigcap, confusable-pair loss
(alpha=2.0), **from scratch** (not resumed), 80 epochs — matching
Experiment 12's original from-scratch recipe, isolating the downsample-
from-scratch question from the resume-from failure mode above. Train
set: 47,909 → 30,224 rows.

**Result**: 70.57% best (epoch 57) — well below Experiment 15's
75.45%. But the per-class breakdown shows a real, substantial,
legible tradeoff, not just "worse":

| Class | Exp 15 (no cap) | Exp 22 (cap 2000) | Change |
|---|---:|---:|---:|
| NEXT | 79.6% | **94.4%** | **+14.8pp** |
| PLAY_MUSIC | 54.6% | **64.3%** | **+9.7pp** |
| MESSAGE | 57.5% | **62.1%** | +4.6pp |
| CALL | 92.6% | **96.3%** | +3.7pp |
| WEATHER | 68.5% | 53.2% | **-15.3pp** |
| VOLUME_UP | 72.8% | 56.8% | **-15.9pp** |
| VOLUME_DOWN | 69.9% | 55.9% | **-14.0pp** |
| LIGHT_OFF | 77.1% | 62.6% | **-14.5pp** |

Confirms the majority-class-bias diagnosis was real — the classes it
targeted (NEXT, CALL, MESSAGE, PLAY_MUSIC) genuinely improved, some
dramatically. But capping *everything* over 2,000 at exactly 2,000 was
too blunt: WEATHER (2,611→2,000), VOLUME_UP (3,524→2,000), VOLUME_DOWN
(2,965→2,000) all lost 23-43% of their data too, for real cost — they
weren't the problem classes, but the uniform cap hit them anyway.

## Experiment 23 — max-per-class 3000 (gentler cap), from scratch

**Setup**: identical to #22 but `--max-per-class 3000`. Train set:
47,909 → 38,927 rows (19% cut vs. #22's 37%).

**Result**: 73.16% best (epoch 60) — better than #22, still below #15.
The gentler cap recovered most of the mid-sized classes' collateral
damage while keeping most of the real gains:

| Class | Exp 15 | Exp 22 (cap 2000) | Exp 23 (cap 3000) |
|---|---:|---:|---:|
| NEXT | 79.6% | 94.4% | 90.7% (still a real win) |
| COLOR | 63.0% | 68.6% | **70.2%** (best yet) |
| CREATE_REMINDER | 66.8% | 67.5% | **72.1%** (best yet) |
| WEATHER | 68.5% | 53.2% | 65.3% (mostly recovered) |
| VOLUME_DOWN | 69.9% | 55.9% | 67.0% (mostly recovered) |
| LIGHT_OFF | 77.1% | 62.6% | 73.8% (mostly recovered) |
| VOLUME_UP | 72.8% | 56.8% | 63.3% (still down -9.5pp) |

**Conclusion for both 22/23**: real, genuine tradeoffs — every cap
value trades some classes' strength for others', and no single global
cap beat Experiment 15's overall accuracy. Superseded by the ASR-
cascade finding below before a further cap sweep (e.g. 4000, or a
per-class-targeted cap) was tried.

## Experiment 24 — QA-filtered synthetic data + regularization stack, from scratch

**Motivation**: a separate ML-engineering review (research on
commercial voice assistants and SOTA SLU accuracy, see MODEL.md
Section 9) found the project's synthetic-audio QA gate
(`vcm/dataset/qa/synthetic_check.py`) had been built but never actually
run, despite direct published evidence that ASR-based filtering of
synthetic TTS clips closes real/synthetic accuracy gaps (89%→92.5% in
a comparable study). Running it (`scripts/qa_filter_option_b.py`)
against Option B's 17,658 clips dropped 1,158 (6.6%) — with a striking
concentration: **BRIGHTNESS alone accounted for 592 of the 1,158 drops
(51% of all failures, ~33% of BRIGHTNESS's own synthetic data)**, far
out of proportion to every other label (mostly 1-4%) — something about
BRIGHTNESS's synthetic generation specifically produces audio that
doesn't transcribe as intended.

Same review also found DS-CNN had zero regularization (no dropout,
unlike BCResNet), no weight decay, and no label smoothing — all added
as opt-in flags (see MODEL.md Section 9), tested together here.

**Setup**: QA-filtered manifest (62,318 rows, was 63,476) +
`--dropout 0.2 --weight-decay 1e-4 --label-smoothing 0.1` +
confusable-pair loss (alpha=2.0), from scratch, 80 epochs. Also the
first experiment under the new (non-opt-in) feature normalization
added in the same review pass — `extract_log_mel` now normalizes to
roughly zero-mean/unit-variance using measured dataset constants,
which is why this couldn't be a `--resume-from` of any earlier
checkpoint (they were trained on the old, unnormalized features).

**Result**: 70.60% (epoch 76, converged/plateaued for the last ~10
epochs, not still climbing) — a real 4.85pp regression from Experiment
15. Per-class breakdown showed a **broad decline across nearly every
class** (VOLUME_UP -13.2pp, VOLUME_DOWN -11.4pp, WEATHER -10.3pp,
CREATE_REMINDER -10.5pp, PAUSE -17.6pp), not concentrated on the QA-
filtered classes specifically (BRIGHTNESS itself only dropped -1.8pp).
That pattern points at the regularization stack — dropout 0.2 +
weight decay 1e-4 + label smoothing 0.1, all at once, at their
"textbook" default strengths — being too aggressive for a 26K-
parameter model, most likely underfitting it, rather than the QA-
filtered data being the problem. Even VOLUME_UP/DOWN (which the
confusable-pair loss targets) got worse, suggesting the extra
regularization interfered with that mechanism too — the same kind of
multi-mechanism interference seen in Experiments 17-21.

**Lesson**: standard regularization defaults don't automatically
transfer to a model this small; worth retesting each of dropout/
weight-decay/label-smoothing individually, at gentler values, rather
than stacked at once.

## Experiment 25 — QA-filtered data only, from scratch (isolating from Experiment 24's regularization stack)

**Setup**: same QA-filtered manifest as #24, confusable-pair loss
(alpha=2.0) only — no dropout/weight-decay/label-smoothing — isolating
whether the QA filter itself helped or hurt, separate from Experiment
24's over-regularization. From scratch, 80 epochs.

**Status**: launched, then superseded in priority by Experiment 26's
result below before completion — the numbers, once available, are
useful documentation but no longer change the project's direction.

## Experiment 26 — ASR-cascade (Whisper transcript → text classifier), a different architecture entirely

**Motivation**: an ML-engineering review (commercial voice assistant
architecture + SOTA SLU research, MODEL.md Section 9) found that Siri,
Alexa, and Google Assistant all classify intent from a **text
transcript**, not raw audio — a cascade (ASR → NLU), not the single
end-to-end audio-to-intent model this project has built through
Experiment 25. Verified directly against this project's own hardest
case before committing to the idea: `faster-whisper` (`base`) correctly
resolved the VOLUME_UP/VOLUME_DOWN and LIGHT_ON/LIGHT_OFF distinction
in text ("Turn the volume up.", "lights off in the washroom") on real
clips that `whisper-tiny` and, separately, six direct-audio loss-
engineering experiments (14-19) could not reliably resolve acoustically.

**Setup**:
1. `scripts/transcribe_corpus_for_cascade.py` — `faster-whisper` (base)
   transcribes all 62,876 non-background clips in the manifest.
   Deliberately uses *Whisper's own transcript* as the training text,
   not each source's ground-truth transcript (which the combined
   manifest doesn't carry through anyway) — the classifier should
   train on the same kind of noisy ASR output it'll see at real
   inference time.
2. `scripts/train_cascade_classifier.py` — TF-IDF (1-2 grams) +
   logistic regression on `(Whisper transcript, label)`. Deliberately
   the cheapest plausible text classifier — this project's 20-intent,
   largely fixed-phrasing vocabulary is far narrower than open-domain
   SLU benchmarks like SLURP's full 69 intents.
3. `scripts/demo_infer_cascade.py` — live mic test, mirroring
   `demo_infer.py`'s interface (same push-to-talk capture, same
   `--debug` full-distribution/WAV-saving option), for the cascade
   instead of the direct-audio model.

No hyperparameter tuning at any step — this was a first-pass
feasibility check.

**Result — a step change, not an incremental gain**:

| Split | All | Real audio only |
|---|---:|---:|
| Val | 90.17% (n=6,844) | 87.51% (n=5,062) |
| **Test** | **92.23%** (n=8,597) | **90.62%** (n=6,831) |

Real-audio-only accuracy (the honest number, controlling for
synthetic clips being easier for Whisper to transcribe than real
speech) clears 90% on the held-out test split and lands almost exactly
at the published SLURP ceiling (87-88%, the closest comparable
real-world benchmark) found in the same research pass — a credible
number, not an inflated illusion. Confirmed on two independent splits,
not a one-off. Per-class, the project's single worst, most persistent
problem — VOLUME_UP/VOLUME_DOWN polarity confusion — improved
dramatically: VOLUME_DOWN 55.9-69.9% (every prior direct-audio
experiment) → **80.1%**; VOLUME_UP 56.8-72.8% → **86.6%**. The classes
with zero real audio coverage (CALL, NEXT, LIST_REMINDERS) — the
project's other standing open risk — score 88.9%, 100%, and 98.1%
respectively on this blended metric, consistent with the ASR offloading
acoustic generalization almost entirely onto Whisper's own pretraining
(hundreds of thousands of hours of real diverse speech, none of it
from this project's limited/synthetic-heavy dataset).

**Model size** (measured, not estimated): direct-audio DS-CNN
checkpoint is **137.5 KB**. The cascade is `faster-whisper` (base,
142MB on disk) + the TF-IDF/logistic-regression classifier (1.9MB) =
**~144MB total — ~1,070x the DS-CNN's size**.

**Decision, revised after a compliance check (see MODEL.md Section 10)**:
the assignment explicitly states "ASR models are not desirable for
on-device computing because of footprint" and requires the VCM to be
tiny — at ~1,070x the size, the cascade cannot be the deployed VCM.
It's kept as a **backup/reference option** (useful if the footprint
constraint is ever relaxed, or as a benchmark ceiling) and — more
usefully — as an **offline training-time teacher** for the direct-audio
model via knowledge distillation (see Experiment 27), which keeps 100%
of the ASR's benefit off the deployed model entirely. Live-voice
validation of the cascade (`--debug`) was done afterward anyway, out of
general interest — see the notes below — and held up well, but that no
longer changes which model actually gets deployed.

**Live-voice validation (informal, ~68 utterances, done for interest
after the compliance finding above)**: ~96% correct on clearly-
intentional utterances, exceeding the 90.62% test-set number. Every
previously-broken direct-audio case now worked via the cascade —
"Kill the lights" (LIGHT_OFF=0.98), "Lower the volume"/"Turn the
volume down" (VOLUME_DOWN=0.99, both previously scored WEATHER=0.97 on
the direct model), "Make a call" (CALL=0.96-0.98, previously ~0-1%).
Real remaining errors: "Start music"→STOP (should be PLAY_MUSIC),
"Lights out."→LIGHT_ON (idiom not in any training template, should be
LIGHT_OFF), "Stop play"→PLAY_MUSIC (should be STOP, though Whisper's
own transcription looked truncated here too). A few attempts were also
hallucinated into non-English text by Whisper on ambiguous/quiet audio
(fixed afterward — `language="en"` is now forced in
`FasterWhisperTranscriber`).

## Experiment 27 — knowledge distillation from the ASR-cascade into the deployable DS-CNN

**Motivation**: the compliance finding in Experiment 26 (ASR isn't
deployable — "not desirable for on-device computing because of
footprint," VCM must be tiny) raised the question of whether the
cascade's real benefit could still reach the deployed model without
ever running ASR on it. The assignment's constraint is specifically
about what runs at *inference* time; it says nothing about how
training data/signal is produced. Knowledge distillation (Hinton,
Vinyals, Dean, 2015) is the standard technique for exactly this: use a
larger "teacher" model's predictions as extra soft-label supervision
when training a smaller "student," then deploy only the student.

**Setup**:
1. `scripts/generate_distillation_labels.py` — runs every non-
   background manifest row's already-computed Whisper transcript
   (`data/cascade_transcripts.csv`, from Experiment 26) through the
   trained cascade classifier, producing a full 19-class probability
   distribution per clip (`data/distillation_labels.csv`).
2. `vcm.train.dataset.load_distillation_labels()` remaps these into
   this project's 20-label order (`unknown_background` always gets
   probability 0 — it has no transcript). `ManifestDataset` optionally
   returns `(features, label, teacher_probs)` instead of
   `(features, label)` when distillation labels are supplied.
3. `vcm.train.losses.DistillationLoss` wraps the existing
   `ConfusablePairLoss` and adds a temperature-scaled KL-divergence
   term against the teacher's soft labels, skipping rows with no
   teacher available (`--distill-weight`, `--distill-temperature`,
   both opt-in/default-off in `train.py`).
4. Training run: `--confusable-alpha 2.0 --distill-weight 2.0
   --distill-temperature 2.0`, DS-CNN bigcap, from scratch, 80 epochs
   — same recipe as Experiment 15 with distillation added on top.

**A real numerical instability at the start, self-resolved**: epoch 1's
train_loss was 23.17 — wildly higher than any prior experiment's
(typically 0.2-2.0). val_loss (computed with the base criterion only,
unaffected by distillation) looked normal (~3.0, expected for an
untrained 20-class model). Root cause: an untrained model's raw
outputs can be confidently wrong on classes the teacher favors, and
KL-divergence punishes that heavily — `target * (log(target) -
input)` blows up when the student's log-probability for a
teacher-favored class is very negative. Watched closely through the
LR warmup peak (the highest-risk point for this kind of instability,
per Experiments 18-21's earlier lessons): train_loss fell steadily
every epoch (23.17→8.11 by epoch 10→2.73 by epoch 80) and val_acc
climbed normally throughout, with no NaN/divergence at any point — the
large magnitude was a scale artifact of the KL term on an untrained
model, not real instability.

**Result**: 75.78% best val accuracy (epoch 55) — technically new
best, edging out Experiment 15's 75.45% by +0.33pp. But the headline
number hides real, substantial per-class churn, not a clean
improvement:

| Class | Exp 15 (no distillation) | Exp 27 (distillation) | Change |
|---|---:|---:|---:|
| PLAY_MUSIC | 54.6% | **70.8%** | **+16.2pp** |
| CREATE_REMINDER | 66.8% | **73.8%** | +7.0pp |
| COLOR | 63.0% | **70.4%** | +7.4pp |
| NEXT | 79.6% | 83.0% | +3.4pp |
| VOLUME_DOWN | 69.9% | 58.3% | **-11.6pp** |
| PAUSE | 91.3% | 82.5% | -8.8pp |
| STOP | 85.2% | 77.4% | -7.8pp |
| WEATHER | 68.5% | 64.1% | -4.4pp |

**Analysis**: the wins are real and land exactly where predicted —
PLAY_MUSIC, CREATE_REMINDER, and COLOR are three of the five labels in
the "diffuse confusion cluster" (PLAY_MUSIC/WEATHER/TIME/MESSAGE/
CREATE_REMINDER) that no prior fix — confusable-pair loss, downsampling,
regularization — ever touched, since it's not a clean acoustic pair
the way VOLUME_UP/DOWN is. PLAY_MUSIC's +16.2pp is the single largest
per-class improvement of any experiment in this project. But
VOLUME_DOWN regressed substantially — notable because that's one of
the two classes distillation was specifically expected to help (the
cascade resolves it well via text) — and PAUSE/STOP, previously
strong, both weakened. `distill-weight=2.0` may simply be too strong
for some classes while being right for others; untried: a lower
weight (e.g. 1.0), or a per-group weighting analogous to the
confusable-pair loss's per-group alpha idea.

**Current standing recommendation**: not a strict upgrade over
Experiment 15 — which one to actually ship depends on whether the
PLAY_MUSIC/COLOR/CREATE_REMINDER gains or the VOLUME_DOWN/PAUSE/STOP
costs matter more for the live demo. Both remain fully compliant
(137.5 KB, no ASR at inference time) — see MODEL.md Section 10 for the
full three-way comparison including the ASR-cascade.

## Evaluation change from Experiment 28 onward: real-speech test accuracy

Through Experiment 27 the direct-audio models were compared on best
*val* accuracy over all rows, while the ASR-cascade (Experiment 26) was
reported on *test* accuracy over real speech only (90.62%) — two
numbers that aren't comparable. `scripts/evaluate_checkpoint.py` now
reports every checkpoint the cascade's way: test split, real speech
only (`is_synthetic == False`, excluding `unknown_background`, n=6,831),
plus per-class and confusable-group breakdowns. Re-scored on that
basis, the previous best direct-audio models are **70.56%** (Experiment
25) and **71.57%** (Experiment 27) — a ~20pp gap to the cascade, not
the ~15pp the val numbers suggested. They score 94–96% on synthetic test
clips, so the gap is specifically real speech. Results log:
`logs/eval_exp28_29_test.log`.

## Experiment 28 — CRNN: fixing DS-CNN's receptive field

**Motivation — a structural finding, not a tuning one**: DS-CNN's first
conv (10×4, stride 2) is followed by stride-1 3×3 depthwise blocks, so
each output unit sees only 4 + 5×4 = ~24 input frames (**~240ms**)
before global average pooling. It classifies a 2–5s command as an
unordered bag of quarter-second snippets — shorter than a single word
like "temperature". That explains two things in this log: Experiment
7→11's +6.4pp for only ~2K extra params (one more block = wider
receptive field, not capacity), and the carrier-phrase confusions
(VOLUME vs TEMPERATURE) surviving six loss-engineering experiments
(14–19). Hello Edge's DS-CNN was designed for 1s single-word keyword
spotting, where this doesn't matter; this project's commands are
multi-word phrases where word order and phrase context do.

**Setup**: new `CRNN` in `architectures.py` (96,277 params, ~95 KB
int8): the same DS-conv blocks but every second one strides by 2 in
time and frequency (receptive field grows geometrically), frequency
collapsed by a 1×1 projection, then a bidirectional GRU (hidden 64) and
attention pooling (a 129-param learned per-frame weight) instead of
global average pooling. Same recipe as Experiment 25 otherwise: `--model
crnn --epochs 80 --warmup-epochs 5 --confusable-alpha 2.0`, from
scratch, QA-filtered manifest, **seeds 0/1/2** — plus two new DS-CNN
baseline seeds (1/2; seed 0 = Experiment 25's checkpoint) so both sides
are multi-seed. All on GPU 2.

**Result — the largest single improvement in the project**:

| Model | Seeds | Best val | Real-speech test | All test |
|---|---:|---:|---:|---:|
| DS-CNN bigcap (26,300) | 3 | 72.69–73.98% | 68.42 / 69.01 / 70.56% | 73.68–75.31% |
| **CRNN (96,277)** | 3 | 84.00 / 84.91 / 84.00% | **79.84 / 80.84 / 80.57%** | 83.28–84.25% |

+11pp on real speech, consistent across all three seeds (spread <1pp on
both sides, so this is not run-to-run noise). The within-group polarity
confusion roughly halved or better (VOLUME/TEMPERATURE 6.1% → 2.1–2.5%,
LIGHT 2.2% → 0.9–1.5%). CRNN overfits noticeably (train loss ~0.05 vs
val loss ~0.7, val loss rising slightly while val accuracy still
climbed) — unlike the 26K DS-CNN, which was *under*fitting (why the
Experiment 24 regularization stack hurt it). CRNN has capacity to
spare, which made regularization via augmentation the natural next
step (Experiment 29b).

**Caveat for the assignment**: the architecture review says "no
attention/transformer layers". Attention *pooling* here is one linear
scorer over time steps, not transformer self-attention, but if the rule
is read literally a conv-only variant (dilated time convs + pooling, no
GRU/attention) is the fallback. The GRU is also not yet benchmarked on
the RPi ONNX/int8 path.

**Operational note**: the first launch oversubscribed the shared DGX
node (~24K threads, ~200 cores — librosa/BLAS spawn a thread per core in
every DataLoader worker). All runs since set
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMBA_NUM_THREADS=1`,
which cut usage to ~40 cores and *halved* epoch time.

## Experiment 29a — trim silence + 5.0s window

**Motivation**: features kept the first 3.0s of each clip. Measured on
a 3,000-clip sample (`librosa.effects.trim`, top_db=30): speech runs
past 3.0s in **12.2% of clips overall and 29.6% of SLURP** — the source
where real-speech accuracy is weakest. Trimming leading/trailing silence
plus a 5.0s window leaves speech truncated in ~1.0% overall (3.3% of
SLURP).

**Setup**: Experiment 28's CRNN recipe + `--trim-silence --window-s
5.0`, seeds 0/1/2. Stored in the checkpoint as `feature_config`, which
`demo_infer.py` and `evaluate_checkpoint.py` apply automatically.

**Result — essentially no change**: 80.73 / 80.81 / 80.71% real-speech
test vs Experiment 28's 79.84 / 80.84 / 80.57% (+0.3pp mean, within seed
noise). The truncated tails were evidently mostly words that don't
change the intent (the rest of a reminder's content, a location). Kept
anyway, since 29b builds on it and it removes a real train/live
mismatch (push-to-talk audio has variable leading silence), but this
was not the lever.

## Experiment 29b — waveform augmentation

**Motivation**: SpecAugment was ruled out twice (Experiments 2, 8)
because random time masking erases the one distinguishing word. But the
synthetic-to-real gap was the dominant remaining problem (94–97%
synthetic vs ~80% real speech on test), and CRNN was overfitting.
Waveform augmentation perturbs *how* a command sounds without removing
*what* was said.

**Setup**: Experiment 29a + `--wave-augment` (`vcm/train/wave_augment.py`),
applied to training audio after trimming and before feature
extraction: background noise from the *train-split* GSC clips at SNR
5–25 dB (p=0.5), speed perturbation 0.9–1.1× (p=0.5), synthetic room
reverb (exponentially decaying noise RIR, RT60 0.2–0.8s, p=0.3), and a
random 0–0.3s start shift (always). No gain augmentation — per-clip
peak-referenced dB normalization cancels it. Seeds 0/1/2.

**Result — new best deployable model**:

| | Seeds | Best val | Real-speech test | All test | Synthetic test |
|---|---:|---:|---:|---:|---:|
| Experiment 29a | 3 | 83.75–84.88% | 80.71–80.81% | 83.91–84.21% | 96.3–97.8% |
| **Experiment 29b** | 3 | **88.29 / 88.51 / 88.46%** | **85.32 / 85.14 / 85.73%** | **87.97–88.40%** | 98.9–99.2% |
| ASR-cascade (Experiment 26, reference) | — | — | 90.62% | 92.23% | — |

+4.6pp real speech over 29a, again consistent across seeds. Cumulative
over the DS-CNN: **+16.1pp on real speech** (69.3% → 85.4% seed mean),
closing three-quarters of the gap to the ASR-cascade at 96K params and
no ASR at inference. Confusable groups (seed 2): VOLUME/TEMPERATURE
95.5% correct / 1.4% within-group / 3.0% other; LIGHT 92.0% / 0.7% /
7.3%. The polarity confusion that dominated Experiments 1–27 is
effectively solved.

**Per-class accuracy, real-speech test split** (Experiment 25 DS-CNN
and Experiment 28 CRNN seed means for comparison; "all" includes
synthetic clips):

| Label | Real n | Exp 25 DS-CNN | Exp 28 CRNN (mean) | 29b s0 | 29b s1 | 29b s2 | **29b mean** | 29b mean, all clips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TEMPERATURE | 1133 | 95.1% | 98.8% | 99.2% | 99.4% | 99.1% | **99.2%** | 99.3% |
| PAUSE | 41 | 100.0% | 98.4% | 100.0% | 100.0% | 100.0% | **100.0%** | 100.0% |
| STOP | 64 | 89.1% | 97.9% | 98.4% | 100.0% | 100.0% | **99.5%** | 99.7% |
| TIMER | 80 | 71.2% | 92.9% | 96.2% | 96.2% | 97.5% | **96.7%** | 98.8% |
| LIGHT_ON | 598 | 86.5% | 92.0% | 94.3% | 94.8% | 95.5% | **94.9%** | 95.3% |
| VOLUME_DOWN | 412 | 75.7% | 86.1% | 89.1% | 90.8% | 91.8% | **90.5%** | 91.6% |
| LIGHT_OFF | 718 | 72.6% | 83.7% | 88.3% | 86.8% | 88.3% | **87.8%** | 88.4% |
| VOLUME_UP | 515 | 69.9% | 83.0% | 88.0% | 87.4% | 88.2% | **87.8%** | 89.0% |
| TIME | 343 | 70.8% | 75.7% | 85.4% | 82.8% | 81.9% | **83.4%** | 85.4% |
| ALARM | 337 | 65.6% | 74.4% | 81.6% | 81.9% | 79.5% | **81.0%** | 87.4% |
| MESSAGE | 399 | 52.6% | 68.8% | 80.2% | 81.5% | 78.2% | **80.0%** | 82.4% |
| WEATHER | 632 | 57.1% | 69.1% | 77.4% | 78.8% | 79.4% | **78.5%** | 79.8% |
| PLAY_MUSIC | 846 | 60.9% | 75.7% | 76.7% | 74.8% | 78.1% | **76.6%** | 78.1% |
| BRIGHTNESS | 351 | 55.8% | 63.4% | 74.1% | 76.1% | 76.3% | **75.5%** | 81.7% |
| CREATE_REMINDER | 176 | 34.7% | 50.6% | 64.8% | 62.5% | 71.6% | **66.3%** | 82.3% |
| COLOR | 186 | 38.2% | 48.0% | 56.5% | 54.3% | 51.1% | **53.9%** | 75.0% |
| NEXT | 0 | – | – | – | – | – | synthetic-only | 100.0% |
| LIST_REMINDERS | 0 | – | – | – | – | – | synthetic-only | 96.5% |
| CALL | 0 | – | – | – | – | – | synthetic-only | 94.2% |
| unknown_background | 0 | – | – | – | – | – | not speech | 100.0% |

Every class with real test audio improved over Experiment 25, most by
double digits. **What's left**: COLOR (53.9%), CREATE_REMINDER (66.3%),
BRIGHTNESS (75.5%) and the PLAY_MUSIC/WEATHER/MESSAGE cluster (77–80%)
— almost entirely SLURP's free-form phrasing. Top confusions (seed 2)
are PLAY_MUSIC↔WEATHER (52/45), LIGHT_OFF→BRIGHTNESS (48),
PLAY_MUSIC↔MESSAGE (32/31) and COLOR↔BRIGHTNESS (31/27) — classes that
share vocabulary, not a single polarity word. CALL/NEXT/LIST_REMINDERS
still have no real test audio at all, so their numbers only measure
synthetic performance; a quick local check with macOS `say` misread
"call mom" as LIGHT_ON (0.98) — the same synthetic-only-label risk
flagged in DATASET.md step 8.

**Current standing recommendation**: `checkpoints/exp29b_crnn_trim5s_waveaug_s2.pt`
(best real-speech seed, 85.73%) is the best deployable model. Try it
live with `python scripts/demo_infer.py --checkpoint
checkpoints/exp29b_crnn_trim5s_waveaug_s2.pt`; the feature config
(trim + 5.0s) is read from the checkpoint automatically.

## Experiment 30 — auxiliary word-level CTC on the Whisper transcripts

**Motivation**: Experiment 27 distilled only the cascade's 19-class
probabilities, a thin signal. Following Lugosch et al. (Interspeech
2019), a CTC loss asking the model's per-frame features to spell out
*which words were said, in order* should push the encoder toward word
content instead of clip-level acoustic texture. That should in turn help
the shared-vocabulary confusions left after 29b (PLAY_MUSIC ↔ WEATHER ↔
MESSAGE, COLOR ↔ BRIGHTNESS).

**Setup**: Experiment 29b + `--ctc-weight` (`vcm/train/transcripts.py`):
a linear head (280K params) on CRNN's per-frame BiGRU outputs,
word-level vocabulary from the train-split Whisper transcripts (2,172
entries incl. blank/`<unk>`, words seen ≥3 times, 97.6% token
coverage), empty target for untranscribed `unknown_background`. Word-level
rather than character-level because CRNN's frames are ~80ms apart, too
coarse for characters. The head lives only in the training loop and is
**not saved**, so the deployed model is identical in size (96,277) to
29b. Weight 0.5 × seeds 0/1/2, plus 0.2 and 1.0 at seed 0.

**Result — a small but consistent negative**:

| | Best val | Real-speech test | All test |
|---|---:|---:|---:|
| Experiment 29b (no CTC), 3 seeds | 88.29–88.51% | 85.32 / 85.14 / 85.73% (mean 85.40) | 87.97–88.40% |
| **CTC weight 0.5**, 3 seeds | 88.31–88.44% | 84.61 / 84.83 / 84.67% (mean **84.70**) | 87.56–87.77% |
| CTC weight 0.2, seed 0 | 88.40% | 84.80% | 87.75% |
| CTC weight 1.0, seed 0 | 87.72% | 83.68% | 86.74% |

-0.7pp real speech at weight 0.5, with all three CTC seeds below all
three 29b seeds, and monotonically worse as the weight grows (0.2 ≈ 0.5
> 1.0). Val accuracy was indistinguishable, so only the real-speech test
metric exposes the difference. The CTC loss itself trained normally
(~120 → ~1.8), so this isn't a broken head. The per-class
breakdown (real speech, 3-seed means, weight 0.5 vs 29b) shows where it
cost:

| Label | 29b | CTC 0.5 | Change |
|---|---:|---:|---:|
| CREATE_REMINDER | 66.3% | 59.7% | **-6.6pp** |
| COLOR | 53.9% | 51.8% | -2.1pp |
| WEATHER | 78.5% | 76.6% | -1.9pp |
| LIGHT_ON | 94.9% | 93.8% | -1.1pp |
| VOLUME_UP | 87.8% | 86.7% | -1.1pp |
| ALARM | 81.0% | 81.8% | +0.8pp |
| STOP | 99.5% | 100.0% | +0.5pp |
| all others | | | within ±0.7pp |

The biggest losses are on exactly the free-form SLURP classes it was
meant to help, CREATE_REMINDER most of all, whose transcripts are the
longest and most open-vocabulary (the reminder's content). Likely
explanation: with only 96K params, transcribing every word competes
with classifying intent for the same capacity, and the free-form
classes have the most intent-irrelevant words to transcribe. Lugosch
et al.'s setup used the ASR targets for *pretraining* a much larger
encoder, then fine-tuned for intent. Staging it that way (CTC-only
pretrain, then intent fine-tune without CTC), or keyword-only targets
instead of every word, are the untried variants. Neither is a cheap
next step compared with the data gaps found in live testing (below).

**Current standing recommendation — unchanged**: Experiment 29b
(`exp29b_crnn_trim5s_waveaug_s2.pt`, 85.73% real speech) remains the
best deployable model. The `--ctc-weight` flag stays in `train.py`
(default off) for the staged variants above.

## Live-voice test of Experiment 29b (informal)

One tester, Mac microphone, `scripts/demo_infer.py`, about 100
utterances. Strong: volume up/down, lights on/off, temperature, NEXT,
CALL, MESSAGE, LIST_REMINDERS, ALARM, CREATE_REMINDER — mostly ≥0.95
confidence. Weak, each traced to a gap in the training transcripts
(`data/cascade_transcripts.csv`, train split):

- **PAUSE/STOP lose to PLAY_MUSIC unless the verb is clearly
  articulated.** Train counts: PAUSE 640, STOP 812, PLAY_MUSIC 3,682,
  and about half of PLAY_MUSIC's clips contain "song"/"music"/"playing",
  so those words are strong PLAY_MUSIC evidence. PAUSE phrasings are
  mostly "pause the music" (219), "pause" (119), "pause this song" (90)
  and "pause music" (69). "Pause audio" never occurs, and "pause the
  song" only 17 times. Also, **26 STOP clips transcribe as "start
  music"**, either a Whisper mishearing or mislabeled PLAY_MUSIC. Needs
  a label audit.
- **"Timer for 3/5 minutes" fails, "Start a timer for …" works.** Train
  phrasings are dominated by "start a timer for / count down for" +
  {10 seconds, 30 seconds, one minute}. Bare "timer …" is 262 of 1,941
  clips, and other durations are rare.
- **"Color {x}" is flaky.** That pattern occurs only in synthetic data
  and only for red/green/blue (~380 clips). Real COLOR speech uses
  "change/set the lights to …".
- Confident errors (1.00 on a wrong class) plus low-confidence misses
  (0.3–0.6) suggest a reject threshold in the demo ("didn't catch that")
  as a cheap mitigation.

Next lever is data rather than architecture: targeted synthetic data for
these phrasings (including verb minimal pairs like "play / pause / stop
the song", the same idea that fixed the polarity confusion), a label
audit, and real recordings once the adviser clears the recording tool.
