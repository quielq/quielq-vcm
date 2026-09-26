# Experiment 34 report: frozen-encoder slot heads, wake word v2, exported models

Every number below is copied verbatim from the logs in `logs/` on the DGX. Tables I computed from those logs are labeled as computed.

## 1. GPUs, base commit, fixes, real recordings

- **GPU 2**: all intent work (3 frozen-slot seeds, joint w0.3 run, all intent evaluations). **GPU 7**: all wake-word work (QA, 2 training seeds, evaluations). Both had no other users' processes when checked before each launch. GPU 1 was also idle but unused. Export and benchmark ran on CPU.

- **Branch** `exp/34-frozen-slots-wake`, from `origin/feat/frozen-slots-wake-recording` at **`82f4486`** (Ship fp32 models, wake threshold 0.95; write up Experiments 32-33). `pytest -q` at setup: 206 passed.

- **Nothing stashed.** No tracked files were modified. The only untracked items were Exp 32's data files and `data/wakeword_real/`, which Step 0 requires and which the target branch gitignores, so I checked there were no conflicts and left them in place instead of stashing them away.

- **No code fixes were needed**: every step ran as written.

- **Real recordings used**: `data/wakeword_real/manifest.csv` exists, every `audio_path` exists, and it was passed as `--extra-wake-manifest` to training and to every wake-word evaluation (`logs/exp34_wakeword_real_check.log`):

```
42 rows; columns: ['id', 'audio_path', 'label', 'text', 'slot_value', 'speaker_id', 'split', 'whisper_text', 'wer', 'qa_pass']
  NOT_WAKE  train  12
  WAKE      test   10
  WAKE      train  20
missing audio_path: 0 []
```

## 2. Slot-head training

Frozen seed 0:

```
Frozen from checkpoints/exp31_crnn_targeted_s2.pt (val_acc 0.8875): training only the slot heads (10,578 params)
epoch   1/30  train_loss=3.0297  val_slot_acc=0.1497  val_loss=0.3721  val_acc=0.8931  lr=1.00e-05  (155.3s)
epoch   2/30  train_loss=2.5598  val_slot_acc=0.5039  val_loss=0.3721  val_acc=0.8931  lr=5.05e-04  (104.5s)
epoch   3/30  train_loss=2.2150  val_slot_acc=0.6816  val_loss=0.3721  val_acc=0.8931  lr=1.00e-03  (88.4s)
...
epoch  28/30  train_loss=1.0427  val_slot_acc=0.8581  val_loss=0.3721  val_acc=0.8931  lr=2.81e-05  (80.0s)
epoch  29/30  train_loss=1.0517  val_slot_acc=0.8570  val_loss=0.3721  val_acc=0.8931  lr=1.25e-05  (79.8s)
epoch  30/30  train_loss=1.0412  val_slot_acc=0.8570  val_loss=0.3721  val_acc=0.8931  lr=3.14e-06  (70.0s)
Done. Best val_slot_acc=0.8626, checkpoint at checkpoints/exp34_frozen_slots_s0.pt
```

Frozen seed 1:

```
Frozen from checkpoints/exp31_crnn_targeted_s2.pt (val_acc 0.8875): training only the slot heads (10,578 params)
epoch   1/30  train_loss=3.0250  val_slot_acc=0.1240  val_loss=0.3721  val_acc=0.8931  lr=1.00e-05  (155.3s)
epoch   2/30  train_loss=2.5743  val_slot_acc=0.3978  val_loss=0.3721  val_acc=0.8931  lr=5.05e-04  (120.9s)
epoch   3/30  train_loss=2.2843  val_slot_acc=0.7084  val_loss=0.3721  val_acc=0.8931  lr=1.00e-03  (88.6s)
...
epoch  28/30  train_loss=1.0473  val_slot_acc=0.8804  val_loss=0.3721  val_acc=0.8931  lr=2.81e-05  (71.4s)
epoch  29/30  train_loss=1.0523  val_slot_acc=0.8804  val_loss=0.3721  val_acc=0.8931  lr=1.25e-05  (82.6s)
epoch  30/30  train_loss=1.0518  val_slot_acc=0.8804  val_loss=0.3721  val_acc=0.8931  lr=3.14e-06  (91.4s)
Done. Best val_slot_acc=0.8816, checkpoint at checkpoints/exp34_frozen_slots_s1.pt
```

Frozen seed 2:

```
Frozen from checkpoints/exp31_crnn_targeted_s2.pt (val_acc 0.8875): training only the slot heads (10,578 params)
epoch   1/30  train_loss=3.0248  val_slot_acc=0.1296  val_loss=0.3721  val_acc=0.8931  lr=1.00e-05  (164.4s)
epoch   2/30  train_loss=2.5621  val_slot_acc=0.4458  val_loss=0.3721  val_acc=0.8931  lr=5.05e-04  (126.8s)
epoch   3/30  train_loss=2.2372  val_slot_acc=0.6994  val_loss=0.3721  val_acc=0.8931  lr=1.00e-03  (91.2s)
...
epoch  28/30  train_loss=1.0352  val_slot_acc=0.8682  val_loss=0.3721  val_acc=0.8931  lr=2.81e-05  (80.0s)
epoch  29/30  train_loss=1.0251  val_slot_acc=0.8693  val_loss=0.3721  val_acc=0.8931  lr=1.25e-05  (82.6s)
epoch  30/30  train_loss=1.0331  val_slot_acc=0.8693  val_loss=0.3721  val_acc=0.8931  lr=3.14e-06  (87.5s)
Done. Best val_slot_acc=0.8726, checkpoint at checkpoints/exp34_frozen_slots_s2.pt
```

val_acc across all 30 epochs, per frozen seed (distinct values, computed): seed 0: ['val_acc=0.8931']; seed 1: ['val_acc=0.8931']; seed 2: ['val_acc=0.8931']. Constant, so the freeze held.

Joint training, slot weight 0.3, seed 0:

```
Model params: 106,855
epoch  78/80  train_loss=0.3248  val_slot_acc=0.9464  val_loss=0.4128  val_acc=0.8660  lr=3.94e-06  (96.2s)
epoch  79/80  train_loss=0.3258  val_slot_acc=0.9475  val_loss=0.4133  val_acc=0.8666  lr=1.75e-06  (86.9s)
epoch  80/80  train_loss=0.3236  val_slot_acc=0.9475  val_loss=0.4127  val_acc=0.8662  lr=4.39e-07  (84.3s)
Done. Best val_acc=0.8672, checkpoint at checkpoints/exp34_joint_w03_s0.pt
```

## 3. Intent + slot evaluation (`logs/eval_exp34_test_nosnips.log`, test split, `--exclude-source snips_lights`)

```
=== checkpoints/exp31_crnn_targeted_s2.pt (test split)
real speech:  85.48% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 83.84% over 16 classes
(excluded sources: snips_lights)
by source:
  fsc             99.19% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        98.81% (n=1681)
  slurp           72.67% (n=3388)
  targeted_synth  98.39% (n=558)
  targeted_synth_slots296.72% (n=366)
  timers_and_such 96.67% (n=120)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    95.4% / 1.5% / 3.1% (n=2347)
  LIGHT_ON/LIGHT_OFF                   91.7% / 0.8% / 7.5% (n=1344)
```

```
=== checkpoints/exp34_frozen_slots_s0.pt (test split)
real speech:  85.48% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 83.84% over 16 classes
(excluded sources: snips_lights)
by source:
  fsc             99.19% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        98.81% (n=1681)
  slurp           72.67% (n=3388)
  targeted_synth  98.39% (n=558)
  targeted_synth_slots296.72% (n=366)
  timers_and_such 96.67% (n=120)

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   68.86% /  68.35%  (n=395)
  TIMER       whisper        70.00% /  70.00%  (n=20)
  ALARM       ground_truth   88.79% /  86.92%  (n=321)
  ALARM       whisper        57.56% /  50.00%  (n=172)
  BRIGHTNESS  ground_truth   66.27% /  66.27%  (n=255)
  COLOR       ground_truth   84.27% /  83.98%  (n=337)
  COLOR       whisper        52.83% /  43.40%  (n=53)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    95.4% / 1.5% / 3.1% (n=2347)
  LIGHT_ON/LIGHT_OFF                   91.7% / 0.8% / 7.5% (n=1344)
```

```
=== checkpoints/exp34_frozen_slots_s1.pt (test split)
real speech:  85.48% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 83.84% over 16 classes
(excluded sources: snips_lights)
by source:
  fsc             99.19% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        98.81% (n=1681)
  slurp           72.67% (n=3388)
  targeted_synth  98.39% (n=558)
  targeted_synth_slots296.72% (n=366)
  timers_and_such 96.67% (n=120)

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   67.59% /  67.09%  (n=395)
  TIMER       whisper        65.00% /  65.00%  (n=20)
  ALARM       ground_truth   89.41% /  87.23%  (n=321)
  ALARM       whisper        61.05% /  54.07%  (n=172)
  BRIGHTNESS  ground_truth   67.84% /  67.45%  (n=255)
  COLOR       ground_truth   84.87% /  84.57%  (n=337)
  COLOR       whisper        60.38% /  49.06%  (n=53)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    95.4% / 1.5% / 3.1% (n=2347)
  LIGHT_ON/LIGHT_OFF                   91.7% / 0.8% / 7.5% (n=1344)
```

```
=== checkpoints/exp34_frozen_slots_s2.pt (test split)
real speech:  85.48% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 83.84% over 16 classes
(excluded sources: snips_lights)
by source:
  fsc             99.19% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        98.81% (n=1681)
  slurp           72.67% (n=3388)
  targeted_synth  98.39% (n=558)
  targeted_synth_slots296.72% (n=366)
  timers_and_such 96.67% (n=120)

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   68.35% /  68.10%  (n=395)
  TIMER       whisper        60.00% /  60.00%  (n=20)
  ALARM       ground_truth   89.72% /  87.54%  (n=321)
  ALARM       whisper        58.72% /  52.33%  (n=172)
  BRIGHTNESS  ground_truth   68.24% /  68.24%  (n=255)
  COLOR       ground_truth   84.27% /  83.98%  (n=337)
  COLOR       whisper        62.26% /  50.94%  (n=53)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    95.4% / 1.5% / 3.1% (n=2347)
  LIGHT_ON/LIGHT_OFF                   91.7% / 0.8% / 7.5% (n=1344)
```

```
=== checkpoints/exp34_joint_w03_s0.pt (test split)
real speech:  84.80% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 82.94% over 16 classes
(excluded sources: snips_lights)
by source:
  fsc             98.89% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        99.29% (n=1681)
  slurp           71.61% (n=3388)
  targeted_synth  98.39% (n=558)
  targeted_synth_slots299.18% (n=366)
  timers_and_such 96.67% (n=120)

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   74.18% /  73.92%  (n=395)
  TIMER       whisper        75.00% /  70.00%  (n=20)
  ALARM       ground_truth   99.38% /  99.07%  (n=321)
  ALARM       whisper        83.72% /  75.58%  (n=172)
  BRIGHTNESS  ground_truth   73.73% /  73.33%  (n=255)
  COLOR       ground_truth   86.35% /  84.87%  (n=337)
  COLOR       whisper        69.81% /  49.06%  (n=53)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    95.1% / 1.4% / 3.5% (n=2347)
  LIGHT_ON/LIGHT_OFF                   92.0% / 0.8% / 7.1% (n=1344)
```

```
=== checkpoints/exp32_crnn_slots_s1.pt (test split)
real speech:  82.07% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 80.28% over 16 classes
(excluded sources: snips_lights)
by source:
  fsc             98.70% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        98.57% (n=1681)
  slurp           66.53% (n=3388)
  targeted_synth  98.03% (n=558)
  targeted_synth_slots297.81% (n=366)
  timers_and_such 95.83% (n=120)

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   74.18% /  73.92%  (n=395)
  TIMER       whisper        70.00% /  70.00%  (n=20)
  ALARM       ground_truth   98.44% /  97.82%  (n=321)
  ALARM       whisper        77.91% /  70.35%  (n=172)
  BRIGHTNESS  ground_truth   74.12% /  74.12%  (n=255)
  COLOR       ground_truth   86.94% /  84.57%  (n=337)
  COLOR       whisper        66.04% /  49.06%  (n=53)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    93.8% / 2.0% / 4.3% (n=2347)
  LIGHT_ON/LIGHT_OFF                   91.5% / 0.7% / 7.8% (n=1344)
```

All three `exp34_frozen` seeds' "real speech" lines are identical to `exp31_crnn_targeted_s2`'s (85.48%, n=6577).

Per-class, `checkpoints/exp34_frozen_slots_s2.pt`:

```
per-class (all / real speech):
  PLAY_MUSIC           79.92% (n=981)         77.19% (n=846)
  WEATHER              81.23% (n=682)         79.75% (n=632)
  TIME                 87.06% (n=394)         85.13% (n=343)
  LIGHT_ON             96.28% (n=618)         95.91% (n=562)
  LIGHT_OFF            87.74% (n=726)         87.43% (n=668)
  PAUSE                100.00% (n=172)        100.00% (n=41)
  STOP                 100.00% (n=207)        100.00% (n=64)
  NEXT                 100.00% (n=49)         n/a (n=0)
  VOLUME_UP            89.16% (n=572)         87.96% (n=515)
  VOLUME_DOWN          91.38% (n=464)         90.29% (n=412)
  CALL                 95.65% (n=46)          n/a (n=0)
  MESSAGE              79.34% (n=455)         76.44% (n=399)
  LIST_REMINDERS       92.98% (n=57)          n/a (n=0)
  TIMER                98.95% (n=475)         98.75% (n=80)
  ALARM                87.84% (n=658)         78.93% (n=337)
  TEMPERATURE          99.54% (n=1311)        99.47% (n=1133)
  BRIGHTNESS           84.22% (n=488)         67.81% (n=233)
  COLOR                83.93% (n=473)         49.26% (n=136)
  CREATE_REMINDER      82.49% (n=354)         67.05% (n=176)
  unknown_background   100.00% (n=58)         n/a (n=0)
```

Per-class, `checkpoints/exp34_joint_w03_s0.pt`:

```
per-class (all / real speech):
  PLAY_MUSIC           77.88% (n=981)         74.70% (n=846)
  WEATHER              78.74% (n=682)         77.37% (n=632)
  TIME                 84.77% (n=394)         82.51% (n=343)
  LIGHT_ON             96.12% (n=618)         95.73% (n=562)
  LIGHT_OFF            88.57% (n=726)         88.02% (n=668)
  PAUSE                100.00% (n=172)        100.00% (n=41)
  STOP                 98.55% (n=207)         96.88% (n=64)
  NEXT                 100.00% (n=49)         n/a (n=0)
  VOLUME_UP            90.03% (n=572)         88.93% (n=515)
  VOLUME_DOWN          90.09% (n=464)         88.83% (n=412)
  CALL                 95.65% (n=46)          n/a (n=0)
  MESSAGE              82.42% (n=455)         79.95% (n=399)
  LIST_REMINDERS       96.49% (n=57)          n/a (n=0)
  TIMER                99.16% (n=475)         97.50% (n=80)
  ALARM                88.60% (n=658)         78.04% (n=337)
  TEMPERATURE          99.01% (n=1311)        98.85% (n=1133)
  BRIGHTNESS           86.27% (n=488)         71.67% (n=233)
  COLOR                82.03% (n=473)         42.65% (n=136)
  CREATE_REMINDER      82.77% (n=354)         65.34% (n=176)
  unknown_background   100.00% (n=58)         n/a (n=0)
```

Summary (computed from the log above; means over the 4 slotted intents' `ground_truth` rows):

| checkpoint | real speech | mean slot-head acc (ground truth) | mean joint acc (ground truth) |
|---|---|---|---|
| `checkpoints/exp31_crnn_targeted_s2.pt` | 85.48% | n/a (no slot heads) | n/a (no slot heads) |
| `checkpoints/exp34_frozen_slots_s0.pt` | 85.48% | 77.05% (4 intents) | 76.38% |
| `checkpoints/exp34_frozen_slots_s1.pt` | 85.48% | 77.43% (4 intents) | 76.59% |
| `checkpoints/exp34_frozen_slots_s2.pt` | 85.48% | 77.64% (4 intents) | 76.97% |
| `checkpoints/exp34_joint_w03_s0.pt` | 84.80% | 83.41% (4 intents) | 82.80% |
| `checkpoints/exp32_crnn_slots_s1.pt` | 82.07% | 83.42% (4 intents) | 82.61% |


Best frozen seed by mean slot-head accuracy: **seed 2** (77.64%).

## 4. Val reject thresholds, `checkpoints/exp34_frozen_slots_s2.pt` (`logs/eval_exp34_val_best.log`)

```
real speech:  86.17% (n=5366)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 84.77% over 16 classes
reject threshold on top-class confidence (real speech):
  threshold  rejected  acc of accepted  errors caught  correct lost
       0.00      0.0%           86.17%           0.0%          0.0%
       0.40      2.7%           87.55%          12.4%          1.0%
       0.50      6.3%           89.44%          28.4%          2.4%
       0.60     11.1%           91.80%          47.3%          4.6%
       0.70     15.5%           93.45%          60.0%          7.2%
       0.80     20.6%           95.19%          72.4%         10.6%
       0.90     27.7%           96.93%          84.0%         16.0%
       0.95     34.2%           98.13%          91.1%         21.6%
```

## 5. Wake word v2

QA, new rule (`logs/exp34_qa_wakeword.log`):

```
test  NOT_WAKE      300/300   passed (100.0%)
test  WAKE          391/400   passed (97.8%)
train NOT_WAKE     2000/2000  passed (100.0%)
train WAKE         2942/3000  passed (98.1%)
wrote /mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/manifest.csv
```

QA, Exp 33's rule (`logs/exp33_qa_wakeword.log`, the Exp 33 run; its manifest is saved as `data/external/wakeword_synth/manifest_exp33.csv`):

```
test  NOT_WAKE      300/300   passed (100.0%)
test  WAKE          195/400   passed (48.8%)
train NOT_WAKE     2000/2000  passed (100.0%)
train WAKE         1347/3000  passed (44.9%)
wrote /mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/manifest.csv
```

Training, seed 0:

```
train items by kind: {'wake': 2877, 'partial': 2877, 'hard': 1840, 'speech': 51923, 'noise': 474}; val items: 3794; noise bank: 474
Model params: 25,475
epoch  28/30  train_loss=0.0914  val_balanced=0.9643  wake=0.962  partial=0.970  hard=0.898  speech=0.998  noise=1.000  (26s)
epoch  29/30  train_loss=0.0917  val_balanced=0.9643  wake=0.962  partial=0.970  hard=0.898  speech=0.998  noise=1.000  (24s)
epoch  30/30  train_loss=0.0898  val_balanced=0.9643  wake=0.962  partial=0.970  hard=0.898  speech=0.998  noise=1.000  (28s)
Done. Best val_balanced=0.9658, checkpoint at checkpoints/kiwi_wakeword_v2_s0.pt
```

Training, seed 1:

```
train items by kind: {'wake': 2877, 'partial': 2877, 'hard': 1840, 'speech': 51923, 'noise': 474}; val items: 3794; noise bank: 474
Model params: 25,475
epoch  28/30  train_loss=0.1008  val_balanced=0.9557  wake=0.940  partial=0.970  hard=0.918  speech=0.999  noise=1.000  (27s)
epoch  29/30  train_loss=0.0972  val_balanced=0.9557  wake=0.940  partial=0.970  hard=0.918  speech=0.999  noise=1.000  (26s)
epoch  30/30  train_loss=0.1015  val_balanced=0.9557  wake=0.940  partial=0.970  hard=0.918  speech=0.999  noise=1.000  (25s)
Done. Best val_balanced=0.9701, checkpoint at checkpoints/kiwi_wakeword_v2_s1.pt
```

Evaluation, v2 seed 0 (`logs/eval_exp34_wakeword_v2_s0.log`):

```
model: checkpoints/kiwi_wakeword_v2_s0.pt
positives: 391 held-out 'hey kiwi' clips (test voices), 10 real recorded takes; negative stream: 10106 clips, 7.48 h

threshold  false reject (clean)  false reject (10dB noise)  false reject (real voice)  false wake-ups  per hour
     0.50                  2.8%                       4.9%                       0.0%             123     16.43
     0.60                  3.1%                       5.6%                       0.0%              95     12.69
     0.70                  3.6%                       8.2%                       0.0%              69      9.22
     0.80                  5.9%                      10.2%                       0.0%              40      5.34
     0.85                  6.9%                      11.5%                       0.0%              32      4.28
     0.90                  8.7%                      15.6%                       0.0%              22      2.94
     0.95                 14.1%                      21.5%                       0.0%              10      1.34
     0.98                 27.9%                      43.2%                      30.0%               3      0.40
     0.99                 46.0%                      70.1%                      50.0%               1      0.13

suggested threshold: 0.98 (lowest real false-reject rate with <= 1.0 false wake-ups/hour)
false wake-ups at that threshold, by clip: [(('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0200.wav', 'near-miss'), 1), (('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0129.wav', 'near-miss'), 1), (('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0065.wav', 'near-miss'), 1)]
```

Evaluation, v2 seed 1 (`logs/eval_exp34_wakeword_v2_s1.log`):

```
model: checkpoints/kiwi_wakeword_v2_s1.pt
positives: 391 held-out 'hey kiwi' clips (test voices), 10 real recorded takes; negative stream: 10106 clips, 7.48 h

threshold  false reject (clean)  false reject (10dB noise)  false reject (real voice)  false wake-ups  per hour
     0.50                  2.3%                       6.4%                       0.0%             171     22.85
     0.60                  2.6%                       7.9%                       0.0%             134     17.90
     0.70                  3.6%                       8.7%                       0.0%              91     12.16
     0.80                  5.6%                      10.7%                       0.0%              58      7.75
     0.85                  6.9%                      12.8%                       0.0%              49      6.55
     0.90                  8.4%                      16.6%                       0.0%              32      4.28
     0.95                 14.3%                      23.8%                      10.0%              17      2.27
     0.98                 27.6%                      40.9%                      30.0%               7      0.94
     0.99                 50.4%                      69.6%                      60.0%               1      0.13

suggested threshold: 0.98 (lowest real false-reject rate with <= 1.0 false wake-ups/hour)
false wake-ups at that threshold, by clip: [(('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0141.wav', 'near-miss'), 1), (('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0001.wav', 'near-miss'), 1), (('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0143.wav', 'near-miss'), 1), (('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0271.wav', 'near-miss'), 1), (('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0129.wav', 'near-miss'), 1), (('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0015.wav', 'near-miss'), 1), (('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0017.wav', 'near-miss'), 1)]
```

Evaluation, Exp 33 seed 1 (old model, same test set) (`logs/eval_exp34_wakeword_exp33_s1.log`):

```
model: checkpoints/kiwi_wakeword_s1.pt
positives: 391 held-out 'hey kiwi' clips (test voices), 10 real recorded takes; negative stream: 10106 clips, 7.48 h

threshold  false reject (clean)  false reject (10dB noise)  false reject (real voice)  false wake-ups  per hour
     0.50                  9.7%                      10.7%                       0.0%             113     15.10
     0.60                 11.8%                      13.3%                       0.0%              68      9.09
     0.70                 13.0%                      15.1%                       0.0%              42      5.61
     0.80                 15.1%                      18.2%                       0.0%              18      2.41
     0.85                 16.4%                      20.7%                       0.0%              15      2.00
     0.90                 20.5%                      25.8%                      10.0%               8      1.07
     0.95                 27.4%                      34.5%                      20.0%               5      0.67
     0.98                 42.5%                      51.4%                      30.0%               1      0.13
     0.99                 59.1%                      67.8%                      90.0%               1      0.13

suggested threshold: 0.95 (lowest real false-reject rate with <= 1.0 false wake-ups/hour)
false wake-ups at that threshold, by clip: [(('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0098.wav', 'near-miss'), 1), (('data/external/slurp_audio/test/2906_1.flac', 'slurp:COLOR'), 1), (('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/fsc/fluent_speech_commands_dataset/wavs/speakers/Q4vMvpXkXBsqryvZ/d29f5d40-4515-11e9-9035-7943ec3b842a.wav', 'fsc:LIGHT_ON'), 1), (('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0173.wav', 'near-miss'), 1), (('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0269.wav', 'near-miss'), 1)]
```

**Chosen: v2 seed 0** (`checkpoints/kiwi_wakeword_v2_s0.pt`). The brief's rule couldn't be applied literally. It says to compare false rejects at 0.95 among thresholds with ≤ 1.0 false wake-ups per hour, but at 0.95 neither v2 seed is under that limit (seed 0: 1.34/h, seed 1: 2.27/h).
- Seed 0 is better on the rule's own criteria at 0.95: 0.0% vs. 10.0% real-voice false rejects, and fewer false wake-ups (10 vs. 17).
- At 0.98, the first threshold where both seeds are under 1.0/h, they tie on real voice (30.0% each). Seed 0 has fewer false wake-ups there (0.40 vs. 0.94/h), but seed 1 is slightly better in noise (40.9% vs. 43.2%).
- The real-voice set has only 10 takes, so each take is 10 points.
- Under the 1.0/h budget, both v2 seeds' evaluations suggest 0.98. The base commit's message ("Ship fp32 models, wake threshold 0.95") says 0.95 ships; I didn't check the runtime config. At 0.95, seed 0 gives 1.34 false wake-ups per hour on this negative stream.

## 6. fp32 ONNX vs. checkpoint, model files, benchmark

Intent, `checkpoints/exp34_frozen_slots_s2.pt` checkpoint (`logs/eval_exp34_test_nosnips.log`):

```
real speech:  85.48% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 83.84% over 16 classes

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   68.35% /  68.10%  (n=395)
  TIMER       whisper        60.00% /  60.00%  (n=20)
  ALARM       ground_truth   89.72% /  87.54%  (n=321)
  ALARM       whisper        58.72% /  52.33%  (n=172)
  BRIGHTNESS  ground_truth   68.24% /  68.24%  (n=255)
  COLOR       ground_truth   84.27% /  83.98%  (n=337)
  COLOR       whisper        62.26% /  50.94%  (n=53)
```

`models/vcm_intent.onnx` (`logs/eval_exp34_fp32onnx_nosnips.log`):

```
real speech:  85.46% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 83.81% over 16 classes

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   68.35% /  68.10%  (n=395)
  TIMER       whisper        60.00% /  60.00%  (n=20)
  ALARM       ground_truth   89.72% /  87.54%  (n=321)
  ALARM       whisper        58.72% /  52.33%  (n=172)
  BRIGHTNESS  ground_truth   68.24% /  68.24%  (n=255)
  COLOR       ground_truth   84.27% /  83.98%  (n=337)
  COLOR       whisper        62.26% /  50.94%  (n=53)
```

Wake word, `models/kiwi_wakeword.onnx` (`logs/eval_exp34_wakeword_fp32onnx.log`); compare with v2 seed 0 in section 5:

```
model: models/kiwi_wakeword.onnx
positives: 391 held-out 'hey kiwi' clips (test voices), 10 real recorded takes; negative stream: 10106 clips, 7.48 h

threshold  false reject (clean)  false reject (10dB noise)  false reject (real voice)  false wake-ups  per hour
     0.50                  2.8%                       5.6%                       0.0%             124     16.57
     0.60                  3.1%                       7.9%                       0.0%              95     12.69
     0.70                  3.6%                       9.2%                       0.0%              69      9.22
     0.80                  5.9%                      11.5%                       0.0%              40      5.34
     0.85                  6.9%                      12.0%                       0.0%              32      4.28
     0.90                  8.7%                      14.8%                       0.0%              22      2.94
     0.95                 14.1%                      23.8%                       0.0%              10      1.34
     0.98                 27.9%                      46.5%                      30.0%               3      0.40
     0.99                 46.0%                      72.6%                      50.0%               1      0.13

suggested threshold: 0.98 (lowest real false-reject rate with <= 1.0 false wake-ups/hour)
false wake-ups at that threshold, by clip: [(('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0200.wav', 'near-miss'), 1), (('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0129.wav', 'near-miss'), 1), (('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0065.wav', 'near-miss'), 1)]
```

Export output (`logs/exp34_export_intent.log`, `logs/exp34_export_wakeword.log`, metadata lines trimmed):

```
models/vcm_intent.onnx: 426 KB (fp32)
models/vcm_intent.int8.onnx: 292 KB (int8 weights; GRU stays fp32, no int8 GRU kernel in ONNX Runtime)
models/kiwi_wakeword.onnx: 107 KB (fp32)
models/kiwi_wakeword.int8.onnx: 93 KB (int8 weights; GRU stays fp32, no int8 GRU kernel in ONNX Runtime)
```

`ls -la models/`:

```
total 927
drwxrwxr-x  2 quiel.andrew.quiwa quiel.andrew.quiwa   4096 Sep 26 14:52 .
drwxrwxr-x 14 quiel.andrew.quiwa quiel.andrew.quiwa   4096 Sep 26 13:59 ..
-rw-rw-r--  1 quiel.andrew.quiwa quiel.andrew.quiwa  95400 Sep 26 14:42 kiwi_wakeword.int8.onnx
-rw-rw-r--  1 quiel.andrew.quiwa quiel.andrew.quiwa 109334 Sep 26 14:41 kiwi_wakeword.onnx
-rw-rw-r--  1 quiel.andrew.quiwa quiel.andrew.quiwa 298992 Sep 26 14:52 vcm_intent.int8.onnx
-rw-rw-r--  1 quiel.andrew.quiwa quiel.andrew.quiwa 435875 Sep 26 14:52 vcm_intent.onnx
```

`logs/exp34_benchmark_dgx.log` (DGX CPU reference only; the Pi will be slower):

```
machine: ai-n002.hpc.coe.upd.edu.ph (x86_64), 256 cores
intent model  vcm_intent.onnx: 426 KB
  per command: features 3.2 ms + model 1.5 ms = 4.8 ms
wake model    kiwi_wakeword.onnx: 107 KB
  per 100 ms hop: 0.7 ms (model 0.2 ms) -> 1% of one core, always on
peak process memory: 61 MB
```

## 7. Unexpected

1. **Freezing keeps intent exact, but the frozen slot heads are clearly weaker.**
   - Mean slot-head accuracy on ground-truth values: 77.05 / 77.43 / 77.64% frozen, vs. 83.41% for joint w0.3 and 83.42% for Exp 32 w1.0.
   - The gap is largest for ALARM (best frozen seed 89.72%, vs. 99.38% for joint w0.3 and 98.44% for Exp 32) and TIMER (68.35%, vs. 74.18% for both).
   - Frozen val_slot_acc peaked at 0.8626 / 0.8816 / 0.8726, vs. ~0.93–0.95 for the joint runs.
2. **Joint training at slot weight 0.3 is a middle option.**
   - Real speech 84.80%: 0.68 pp below Exp 31 seed 2 (85.48%) and 2.73 pp above Exp 32 w1.0 (82.07%). Slot accuracy matches Exp 32's.
   - Most of the intent loss is on SLURP (71.61% vs. 72.67%).
   - This is one seed. Exp 31's three seeds spanned 85.08–85.48%, so −0.68 pp isn't clearly outside seed noise.
   - I shipped the frozen model as briefed. Joint w0.3 would trade ~0.7 pp of intent for ~6 pp of slot accuracy.
3. **The wake-word seed-selection rule couldn't be applied literally** (section 5): neither v2 seed is at or under 1.0 false wake-ups per hour at 0.95.
4. **Wake word v2 vs. Exp 33, same test set.**
   - At 0.95, v2 seed 0 has fewer clean false rejects (14.1% vs. 27.4%) but more false wake-ups (1.34 vs. 0.67/h).
   - The test set is now 391 positives, up from 195. I compared ids between the two QA manifests: 200 of the 391 were rejected by Exp 33's Whisper-based QA, and 4 of Exp 33's 195 fail the new rule. That likely explains why the Exp 33 model does much worse here than in its own evaluation (27.4% vs. 6.7% clean false rejects at 0.95), since it never trained on clips like the 200. I didn't check its per-clip scores.
   - On the real voice, all three models reject 0/10 takes at ≤ 0.85.
5. **The wake word's 10 dB noise column isn't reproducible between runs.**
   - `evaluate_wakeword.py` seeds its own `rng` (`random.Random(0)`), but `vcm.train.wave_augment.add_noise` picks the noise segment with the unseeded global `random`.
   - The fp32 ONNX rerun of v2 seed 0 matches the checkpoint exactly on the clean and real-voice columns. False wake-ups match at every threshold except 0.50 (124 vs. 123). That is probably a small PyTorch vs. ONNX Runtime numeric difference; I didn't check which clip. The noise column moves by a few points (e.g. 21.5% → 23.8% at 0.95).
   - I didn't change the code because no step failed.
6. **fp32 ONNX vs. checkpoint, intent.** Slot lines are identical. Real speech is 85.46% vs. 85.48%, about 1 of 6,577 clips. I didn't investigate which clip.
7. **The benchmark now runs the fp32 files**, since `benchmark_pi.py` defaults changed on this branch. It measured 1.5 ms per intent model call vs. 14.8 ms for the int8 file in Experiment 32's DGX benchmark. On this x86 CPU, fp32 is faster than int8. The Pi's ARM CPU may differ.
8. **The joint run slowed down** from ~77 s to 106–155 s per epoch late in training. It was the only process on GPU 2 and node load average was ~29 of 256 cores when I checked. I didn't find the cause.
9. The int8 files in `models/` were regenerated by `export_onnx.py` alongside fp32 and are committed for reference. They were not re-evaluated in this experiment.
10. **Frozen val_acc is 0.8931, not the 0.8875 in the "Frozen from" line.** 0.8875 is the val_acc recorded in Exp 31's checkpoint, measured on the old val split. This experiment's manifest has the Snips speaker re-split, which moved 304 more Snips clips into val (220 → 524), so the same frozen model scores differently on the new val set. On test, intent matches Exp 31 exactly (85.48%).
