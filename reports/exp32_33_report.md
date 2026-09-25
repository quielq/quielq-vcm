# Experiments 32–33 report: slot heads, "Hey Kiwi" wake word, exported models

Every number below is copied verbatim from the logs in `logs/` on the DGX.

## 1. GPUs, base commit, fixes

- **GPU 2**: Exp 32 (slots2 synthesis, slots2 QA, training, all intent evaluations). **GPU 7**: Exp 33 (wakeword synthesis, wakeword QA, training, wake-word evaluations). At launch GPUs 1, 2 and 7 were idle; another user took GPU 1 minutes before the full synthesis launch, so the wakeword batch went to GPU 7 instead. ONNX export and the benchmark ran on CPU.

- **Branch** `exp/32-33-slots-wakeword`, created from `origin/feat/hey-kiwi-slots-deploy` at **`78e87c4`** (Replace librosa with numpy log-mel so the Pi runtime fits 512 MB). Working tree was clean, nothing stashed. `pytest -q` at setup: 206 passed.

- **Fixes** (each with `pytest -q`: 206 passed):
  - `1313a14` *Create a batch's output folder before writing its shard CSV*: `synth --batch slots2/wakeword` crashed with `FileNotFoundError` on `data/external/targeted_synth_slots2/synth_shard0.csv`, because only round1's folder existed (made by the refs stage).
  - `a701268` *Wake-word QA: accept a negative clip unless Whisper hears the wake word*: with your approval, after the first wakeword QA put every label under 50% (section 3). NOT_WAKE clips no longer need WER ≤ 0.34, only that Whisper does not hear "hey kiwi". WAKE clips stay strict.

## 2. Snips speaker re-split

`logs/exp32_snips_resplit.log`:

```
speakers: 49 total, 40 were in more than one split before, 0 after
rows changed: 1238/2472
  train   38 speakers   1382 clips
  val      3 speakers    524 clips
  test     8 speakers    566 clips
wrote /mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/snips_lights/manifest.csv (backup: /mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/snips_lights/manifest.csv.bak)
```

Manifest check (`data/dataset_manifest_v2.csv` vs. `data/dataset_manifest_pre_exp32.csv`, `logs/exp32_manifest_diff_check.log`):

```
columns identical: True
row counts: 62318 62318
duplicate audio_paths: 0 0
same audio_path set: True
differing (column, source) -> rows: {('split', 'snips_lights'): 1238}
labels/sources identical: True
ONLY snips_lights split changed: True
snips_lights before: train=1998, val=220, test=254
snips_lights after: train=1382, val=524, test=566
```

Nothing else in the manifest changed: same 62,318 audio_paths, same labels and sources, and the only differing field is `split` on 1,238 `snips_lights` rows.

## 3. Generation and QA

Smoke tests (`--limit 3`, GPU 2, after fix `1313a14`): `shard 0: done (3 jobs, 18s)` for slots2 and `shard 0: done (3 jobs, 16s)` for wakeword. Durations:

```
ALARM_train_0000          'can you set an alarm for three AM'     slot='3:00 AM'     2.86s
ALARM_train_0001          'set an alarm for six PM'               slot='6:00 PM'     3.02s
ALARM_train_0002          'set my alarm for nine PM'              slot='9:00 PM'     1.52s
WAKE_train_0000           'hey kiwi'                              slot=''            0.82s
WAKE_train_0001           'hey kiwi!'                             slot=''            0.98s
WAKE_train_0002           'hey, kiwi'                             slot=''            0.98s
```

Full runs, 4 shards each, launched together at the same time (slots2 on GPU 2, wakeword on GPU 7):

```
slots2 wall: 2790s
shard 0: done (808 jobs, 2697s)
shard 1: done (808 jobs, 2707s)
shard 2: done (807 jobs, 2682s)
shard 3: done (807 jobs, 2687s)

wakeword wall: 3349s
shard 0: done (1425 jobs, 3255s)
shard 1: done (1425 jobs, 3266s)
shard 2: done (1425 jobs, 3233s)
shard 3: done (1425 jobs, 3252s)
```

Peak GPU memory during the full run: GPU 2 17,350 MiB, GPU 7 16,900 MiB (~4.3 GB per process).

**QA, slots2** (`logs/exp32_qa_slots2.log`). Every (split, label) passed at ≥ 70%, so there are no failing-row samples:

```
test  ALARM         151/200   passed (75.5%)
test  BRIGHTNESS     78/80    passed (97.5%)
test  COLOR          44/50    passed (88.0%)
test  TIMER          93/100   passed (93.0%)
train ALARM        1006/1400  passed (71.9%)
train BRIGHTNESS    495/500   passed (99.0%)
train COLOR         259/300   passed (86.3%)
train TIMER         523/600   passed (87.2%)
wrote /mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/targeted_synth_slots2/manifest.csv
```

**QA, wakeword, first run** (before `a701268`, kept as `logs/exp33_qa_wakeword_run1.log`). Every label was under 50%, so I stopped Exp 33 and asked you:

```
test  NOT_WAKE      145/300   passed (48.3%)
test  WAKE          196/400   passed (49.0%)
train NOT_WAKE      900/2000  passed (45.0%)
train WAKE         1345/3000  passed (44.8%)
wrote /mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/manifest.csv
```

Failing-row samples and diagnosis from the first run (`logs/exp33_qa_wakeword_failing_samples.txt`):

```
== WAKE: 1859 failing rows; 15 random:
id                    text                              whisper_text                                wer
WAKE_train_2423       'hey, kiwi'                       'Heavy Kiwi'                                0.500
WAKE_train_2277       'hey kiwi'                        'koi kiwi'                                  0.500
WAKE_train_1247       'hey, kiwi'                       'Hey dude, Kiwi.'                           0.500
WAKE_test_0099        'hey kiwi'                        'Thank you.'                                1.000
WAKE_train_2833       'hey kiwi!'                       'A keyway.'                                 1.000
WAKE_train_0620       'hey, kiwi'                       'Hey, give it.'                             1.000
WAKE_train_0361       'hey kiwi!'                       'Hey, the Kiwi.'                            0.500
WAKE_train_0886       'hey kiwi!'                       'Hey, Kyrie.'                               0.500
WAKE_train_0518       'hey, kiwi'                       'Thank you, we'                             1.500
WAKE_train_2613       'hey kiwi'                        'Thank you.'                                1.000
WAKE_train_1623       'hey kiwi'                        'Click UV.'                                 1.000
WAKE_train_2215       'hey kiwi!'                       'Hey, Keebee!'                              0.500
WAKE_train_0989       'hey, kiwi'                       'Hey, QE.'                                  0.500
WAKE_train_0346       'hey kiwi!'                       'Cakey-ree.'                                1.000
WAKE_train_1861       'hey kiwi!'                       'Thank you very much.'                      2.000

== NOT_WAKE: 1255 failing rows; 15 random:
id                    text                              whisper_text                                wer
NOT_WAKE_train_1238   'hey Kimmy'                       'Hey, kimi.'                                0.500
NOT_WAKE_train_1818   'hey pee-wee'                     'Hey, Peewee.'                              0.667
NOT_WAKE_train_0708   'hey sweetie'                     'K-sweety.'                                 1.000
NOT_WAKE_train_1621   'heavy week'                      'Have you week?'                            1.000
NOT_WAKE_train_0831   'hey Kevin'                       'Take care, Vin.'                           1.500
NOT_WAKE_train_0491   'hey kid'                         'Take it!'                                  1.000
NOT_WAKE_train_1562   'hey pee-wee'                     'Happy Wee! Wee!'                           0.667
NOT_WAKE_test_0245    'week we'                         'Weekly'                                    1.000
NOT_WAKE_train_0375   'hey key'                         'Hakey.  Key.'                              0.500
NOT_WAKE_train_0814   'queen'                           'We love you, Lynn!'                        4.000
NOT_WAKE_train_1855   'hey Kevin'                       'Kevin'                                     0.500
NOT_WAKE_train_1141   'heavy week'                      "can't be weak"                             1.500
NOT_WAKE_train_0735   'hey Kevin'                       'Is it a coming?'                           2.000
NOT_WAKE_train_0129   'hey Siri'                        'Hey, city!'                                0.500
NOT_WAKE_train_1913   'pee-wee'                         'And tell you where you are.'               2.500

WAKE failures, most common whisper transcripts: [('thank you', 201), ('thank you, we', 127), ('thank you very much', 87), ('kiwi', 64), ('thank you, wee', 54), ('thank you, e', 42), ('okay, kiwi', 40), ('ki kiwi', 37), ('hey, keebee', 24), ('kqe', 22), ('hey, qe', 15), ('a kiwi', 15)]
NOT_WAKE fails: heard 'hey kiwi' = 0 | wer>0.34 = 1255
NOT_WAKE texts (distinct, sample): ['a kiwi', 'every week', 'every weekday', 'he keeps', 'heavy week', 'hey', 'hey Alexa', 'hey Kevin', 'hey Kimmy', 'hey Mikey', 'hey Ricky', 'hey Siri', 'hey Tiwi', 'hey Wiwi', 'hey keep it', 'hey key', 'hey kid', 'hey kiki', 'hey kitty', 'hey pee-wee', 'hey queen', 'hey quickly', 'hey sweetie', 'hey, we need', 'hey, weekly']
WAKE pass                    n=1541  median dur=1.02s  median rms=0.1185
WAKE fail 'thank you...'     n=599   median dur=0.88s  median rms=0.1079
WAKE fail other              n=1260  median dur=0.98s  median rms=0.1150
WAKE pass rate by text: {'hey kiwi': '466/1134', 'hey kiwi!': '449/1133', 'hey, kiwi': '626/1133'}
```

**QA, wakeword, final** (after `a701268`, `logs/exp33_qa_wakeword.log`). WAKE is still under 50%; as agreed, it stays strict:

```
test  NOT_WAKE      300/300   passed (100.0%)
test  WAKE          195/400   passed (48.8%)
train NOT_WAKE     2000/2000  passed (100.0%)
train WAKE         1347/3000  passed (44.9%)
wrote /mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/manifest.csv
```

## 4. Manifest and slot labels

`logs/exp32_manifest.log`:

```
base: 62318 rows, added: 7006 QA-passed targeted rows -> data/dataset_manifest_exp32.csv (69324)
  targeted_synth        test  BRIGHTNESS     59
  targeted_synth        test  COLOR         129
  targeted_synth        test  PAUSE          81
  targeted_synth        test  PLAY_MUSIC     77
  targeted_synth        test  STOP           85
  targeted_synth        test  TIMER         127
  targeted_synth        train BRIGHTNESS    386
  targeted_synth        train COLOR         867
  targeted_synth        train PAUSE         637
  targeted_synth        train PLAY_MUSIC    454
  targeted_synth        train STOP          661
  targeted_synth        train TIMER         794
  targeted_synth_slots2 test  ALARM         151
  targeted_synth_slots2 test  BRIGHTNESS     78
  targeted_synth_slots2 test  COLOR          44
  targeted_synth_slots2 test  TIMER          93
  targeted_synth_slots2 train ALARM        1006
  targeted_synth_slots2 train BRIGHTNESS    495
  targeted_synth_slots2 train COLOR         259
  targeted_synth_slots2 train TIMER         523
```

`logs/exp32_slot_labels.log`:

```
TIMER      labeled: 3413  out of vocabulary: 535  no text: 0
ALARM      labeled: 3721  out of vocabulary: 982  no text: 0
BRIGHTNESS labeled: 2257  out of vocabulary: 2130  no text: 0
COLOR      labeled: 3748  out of vocabulary: 494  no text: 0
values with < 20 labeled clips: []
wrote 13139 labels -> data/slot_labels.csv
```

## 5. Training

Exp 32 seed 0 (GPU 2):

```
Model params: 106,855
epoch  78/80  train_loss=0.6226  val_slot_acc=0.9330  val_loss=0.4804  val_acc=0.8406  lr=3.94e-06  (42.5s)
epoch  79/80  train_loss=0.6057  val_slot_acc=0.9341  val_loss=0.4810  val_acc=0.8406  lr=1.75e-06  (42.1s)
epoch  80/80  train_loss=0.6084  val_slot_acc=0.9318  val_loss=0.4806  val_acc=0.8405  lr=4.39e-07  (40.6s)
Done. Best val_acc=0.8436, checkpoint at checkpoints/exp32_crnn_slots_s0.pt
```

Exp 32 seed 1 (GPU 2):

```
Model params: 106,855
epoch  78/80  train_loss=0.6038  val_slot_acc=0.9296  val_loss=0.4641  val_acc=0.8431  lr=3.94e-06  (44.2s)
epoch  79/80  train_loss=0.6138  val_slot_acc=0.9307  val_loss=0.4636  val_acc=0.8427  lr=1.75e-06  (41.2s)
epoch  80/80  train_loss=0.6219  val_slot_acc=0.9285  val_loss=0.4633  val_acc=0.8433  lr=4.39e-07  (40.3s)
Done. Best val_acc=0.8443, checkpoint at checkpoints/exp32_crnn_slots_s1.pt
```

Exp 32 seed 2 (GPU 2):

```
Model params: 106,855
epoch  78/80  train_loss=0.6202  val_slot_acc=0.9184  val_loss=0.4794  val_acc=0.8379  lr=3.94e-06  (45.0s)
epoch  79/80  train_loss=0.6166  val_slot_acc=0.9184  val_loss=0.4800  val_acc=0.8375  lr=1.75e-06  (40.4s)
epoch  80/80  train_loss=0.6215  val_slot_acc=0.9196  val_loss=0.4800  val_acc=0.8384  lr=4.39e-07  (39.1s)
Done. Best val_acc=0.8388, checkpoint at checkpoints/exp32_crnn_slots_s2.pt
```

Exp 33 seed 0 (GPU 7):

```
train items by kind: {'wake': 1249, 'partial': 1249, 'hard': 1804, 'speech': 51923, 'noise': 474}; val items: 3460; noise bank: 474
Model params: 25,475
epoch  28/30  train_loss=0.0740  val_balanced=0.9859  wake=0.990  partial=0.990  hard=0.939  speech=0.999  noise=1.000  (12s)
epoch  29/30  train_loss=0.0751  val_balanced=0.9865  wake=0.990  partial=0.990  hard=0.944  speech=0.999  noise=1.000  (12s)
epoch  30/30  train_loss=0.0711  val_balanced=0.9865  wake=0.990  partial=0.990  hard=0.944  speech=0.999  noise=1.000  (12s)
Done. Best val_balanced=0.9865, checkpoint at checkpoints/kiwi_wakeword_s0.pt
```

Exp 33 seed 1 (GPU 7):

```
train items by kind: {'wake': 1249, 'partial': 1249, 'hard': 1804, 'speech': 51923, 'noise': 474}; val items: 3460; noise bank: 474
Model params: 25,475
epoch  28/30  train_loss=0.0819  val_balanced=0.9812  wake=0.980  partial=0.990  hard=0.944  speech=0.998  noise=1.000  (11s)
epoch  29/30  train_loss=0.0771  val_balanced=0.9812  wake=0.980  partial=0.990  hard=0.944  speech=0.998  noise=1.000  (12s)
epoch  30/30  train_loss=0.0744  val_balanced=0.9813  wake=0.980  partial=0.990  hard=0.944  speech=0.998  noise=1.000  (13s)
Done. Best val_balanced=0.9821, checkpoint at checkpoints/kiwi_wakeword_s1.pt
```

## 6. Intent + slot evaluation (test split)

### `logs/eval_exp32_test.log` (all sources)

```
=== checkpoints/exp31_crnn_targeted_s0.pt (test split)
all:          89.20% (n=9806)
real speech:  85.72% (n=7143)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 85.15% over 16 classes
synthetic:    98.50% (n=2605)
by source:
  fsc             99.22% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        98.33% (n=1681)
  slurp           72.08% (n=3388)
  snips_lights    92.23% (n=566)
  targeted_synth  99.28% (n=558)
  targeted_synth_slots298.09% (n=366)
  timers_and_such 95.00% (n=120)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    95.5% / 1.1% / 3.5% (n=2347)
  LIGHT_ON/LIGHT_OFF                   92.0% / 0.7% / 7.3% (n=1582)
```

```
=== checkpoints/exp31_crnn_targeted_s1.pt (test split)
all:          89.26% (n=9806)
real speech:  85.69% (n=7143)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 85.37% over 16 classes
synthetic:    98.81% (n=2605)
by source:
  fsc             99.22% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        99.17% (n=1681)
  slurp           71.93% (n=3388)
  snips_lights    92.76% (n=566)
  targeted_synth  98.21% (n=558)
  targeted_synth_slots298.09% (n=366)
  timers_and_such 95.00% (n=120)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    94.8% / 1.5% / 3.6% (n=2347)
  LIGHT_ON/LIGHT_OFF                   93.0% / 0.8% / 6.2% (n=1582)
```

```
=== checkpoints/exp31_crnn_targeted_s2.pt (test split)
all:          89.29% (n=9806)
real speech:  85.87% (n=7143)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 85.61% over 16 classes
synthetic:    98.43% (n=2605)
by source:
  fsc             99.19% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        98.81% (n=1681)
  slurp           72.67% (n=3388)
  snips_lights    90.46% (n=566)
  targeted_synth  98.39% (n=558)
  targeted_synth_slots296.72% (n=366)
  timers_and_such 96.67% (n=120)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    95.4% / 1.5% / 3.1% (n=2347)
  LIGHT_ON/LIGHT_OFF                   91.1% / 0.8% / 8.2% (n=1582)
```

```
=== checkpoints/exp32_crnn_slots_s0.pt (test split)
all:          85.42% (n=9806)
real speech:  80.57% (n=7143)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 80.11% over 16 classes
synthetic:    98.39% (n=2605)
by source:
  fsc             98.27% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        98.63% (n=1681)
  slurp           66.15% (n=3388)
  snips_lights    68.20% (n=566)
  targeted_synth  97.85% (n=558)
  targeted_synth_slots298.09% (n=366)
  timers_and_such 93.33% (n=120)

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   73.16% /  72.41%  (n=395)
  TIMER       whisper        70.00% /  65.00%  (n=20)
  ALARM       ground_truth   98.75% /  98.44%  (n=321)
  ALARM       whisper        79.07% /  69.77%  (n=172)
  BRIGHTNESS  ground_truth   73.73% /  73.73%  (n=255)
  BRIGHTNESS  whisper       100.00% / 100.00%  (n=2)
  COLOR       ground_truth   86.05% /  84.27%  (n=337)
  COLOR       whisper        78.43% /  62.09%  (n=153)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    93.5% / 1.9% / 4.6% (n=2347)
  LIGHT_ON/LIGHT_OFF                   87.9% / 1.4% / 10.7% (n=1582)
```

```
=== checkpoints/exp32_crnn_slots_s1.pt (test split)
all:          85.78% (n=9806)
real speech:  81.09% (n=7143)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 80.84% over 16 classes
synthetic:    98.35% (n=2605)
by source:
  fsc             98.70% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        98.57% (n=1681)
  slurp           66.53% (n=3388)
  snips_lights    69.61% (n=566)
  targeted_synth  98.03% (n=558)
  targeted_synth_slots297.81% (n=366)
  timers_and_such 95.83% (n=120)

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   74.18% /  73.92%  (n=395)
  TIMER       whisper        70.00% /  70.00%  (n=20)
  ALARM       ground_truth   98.44% /  97.82%  (n=321)
  ALARM       whisper        77.91% /  70.35%  (n=172)
  BRIGHTNESS  ground_truth   74.12% /  74.12%  (n=255)
  BRIGHTNESS  whisper       100.00% / 100.00%  (n=2)
  COLOR       ground_truth   86.94% /  84.57%  (n=337)
  COLOR       whisper        79.08% /  62.75%  (n=153)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    93.8% / 2.0% / 4.3% (n=2347)
  LIGHT_ON/LIGHT_OFF                   88.3% / 1.8% / 9.9% (n=1582)
```

```
=== checkpoints/exp32_crnn_slots_s2.pt (test split)
all:          85.25% (n=9806)
real speech:  80.37% (n=7143)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 80.42% over 16 classes
synthetic:    98.31% (n=2605)
by source:
  fsc             98.37% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        98.81% (n=1681)
  slurp           65.55% (n=3388)
  snips_lights    68.20% (n=566)
  targeted_synth  97.13% (n=558)
  targeted_synth_slots297.81% (n=366)
  timers_and_such 95.83% (n=120)

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   76.20% /  75.19%  (n=395)
  TIMER       whisper        90.00% /  90.00%  (n=20)
  ALARM       ground_truth   98.13% /  98.13%  (n=321)
  ALARM       whisper        84.30% /  73.26%  (n=172)
  BRIGHTNESS  ground_truth   71.76% /  70.59%  (n=255)
  BRIGHTNESS  whisper         0.00% /   0.00%  (n=2)
  COLOR       ground_truth   86.94% /  85.76%  (n=337)
  COLOR       whisper        76.47% /  60.78%  (n=153)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    93.7% / 1.5% / 4.9% (n=2347)
  LIGHT_ON/LIGHT_OFF                   88.5% / 1.6% / 9.9% (n=1582)
```

### `logs/eval_exp32_test_nosnips.log` (`--exclude-source snips_lights`, the fair comparison)

```
=== checkpoints/exp31_crnn_targeted_s0.pt (test split)
all:          89.02% (n=9240)
real speech:  85.16% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 83.23% over 16 classes
(excluded sources: snips_lights)
synthetic:    98.50% (n=2605)
by source:
  fsc             99.22% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        98.33% (n=1681)
  slurp           72.08% (n=3388)
  targeted_synth  99.28% (n=558)
  targeted_synth_slots298.09% (n=366)
  timers_and_such 95.00% (n=120)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    95.5% / 1.1% / 3.5% (n=2347)
  LIGHT_ON/LIGHT_OFF                   92.4% / 0.7% / 6.8% (n=1344)
```

```
=== checkpoints/exp31_crnn_targeted_s1.pt (test split)
all:          89.05% (n=9240)
real speech:  85.08% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 83.43% over 16 classes
(excluded sources: snips_lights)
synthetic:    98.81% (n=2605)
by source:
  fsc             99.22% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        99.17% (n=1681)
  slurp           71.93% (n=3388)
  targeted_synth  98.21% (n=558)
  targeted_synth_slots298.09% (n=366)
  timers_and_such 95.00% (n=120)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    94.8% / 1.5% / 3.6% (n=2347)
  LIGHT_ON/LIGHT_OFF                   93.0% / 0.7% / 6.2% (n=1344)
```

```
=== checkpoints/exp31_crnn_targeted_s2.pt (test split)
all:          89.22% (n=9240)
real speech:  85.48% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 83.84% over 16 classes
(excluded sources: snips_lights)
synthetic:    98.43% (n=2605)
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
=== checkpoints/exp32_crnn_slots_s0.pt (test split)
all:          86.47% (n=9240)
real speech:  81.63% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 79.50% over 16 classes
(excluded sources: snips_lights)
synthetic:    98.39% (n=2605)
by source:
  fsc             98.27% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        98.63% (n=1681)
  slurp           66.15% (n=3388)
  targeted_synth  97.85% (n=558)
  targeted_synth_slots298.09% (n=366)
  timers_and_such 93.33% (n=120)

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   73.16% /  72.41%  (n=395)
  TIMER       whisper        70.00% /  65.00%  (n=20)
  ALARM       ground_truth   98.75% /  98.44%  (n=321)
  ALARM       whisper        79.07% /  69.77%  (n=172)
  BRIGHTNESS  ground_truth   73.73% /  73.73%  (n=255)
  COLOR       ground_truth   86.05% /  84.27%  (n=337)
  COLOR       whisper        64.15% /  47.17%  (n=53)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    93.5% / 1.9% / 4.6% (n=2347)
  LIGHT_ON/LIGHT_OFF                   91.4% / 1.0% / 7.7% (n=1344)
```

```
=== checkpoints/exp32_crnn_slots_s1.pt (test split)
all:          86.77% (n=9240)
real speech:  82.07% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 80.28% over 16 classes
(excluded sources: snips_lights)
synthetic:    98.35% (n=2605)
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

```
=== checkpoints/exp32_crnn_slots_s2.pt (test split)
all:          86.30% (n=9240)
real speech:  81.42% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 79.78% over 16 classes
(excluded sources: snips_lights)
synthetic:    98.31% (n=2605)
by source:
  fsc             98.37% (n=3069)
  gsc_background  100.00% (n=58)
  option_b        98.81% (n=1681)
  slurp           65.55% (n=3388)
  targeted_synth  97.13% (n=558)
  targeted_synth_slots297.81% (n=366)
  timers_and_such 95.83% (n=120)

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   76.20% /  75.19%  (n=395)
  TIMER       whisper        90.00% /  90.00%  (n=20)
  ALARM       ground_truth   98.13% /  98.13%  (n=321)
  ALARM       whisper        84.30% /  73.26%  (n=172)
  BRIGHTNESS  ground_truth   71.76% /  70.59%  (n=255)
  COLOR       ground_truth   86.94% /  85.76%  (n=337)
  COLOR       whisper        64.15% /  43.40%  (n=53)

confusable groups (correct / within-group confusion / other-wrong):
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE    93.7% / 1.5% / 4.9% (n=2347)
  LIGHT_ON/LIGHT_OFF                   91.8% / 0.8% / 7.4% (n=1344)
```

### Per-class tables: best Exp 31 seed (`checkpoints/exp31_crnn_targeted_s2.pt`) and best Exp 32 seed (`checkpoints/exp32_crnn_slots_s1.pt`), chosen by real-speech accuracy in the no-Snips log

checkpoints/exp31_crnn_targeted_s2.pt, no Snips:

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

checkpoints/exp32_crnn_slots_s1.pt, no Snips:

```
per-class (all / real speech):
  PLAY_MUSIC           73.09% (n=981)         69.03% (n=846)
  WEATHER              71.55% (n=682)         69.94% (n=632)
  TIME                 85.03% (n=394)         83.09% (n=343)
  LIGHT_ON             96.12% (n=618)         95.73% (n=562)
  LIGHT_OFF            87.60% (n=726)         87.13% (n=668)
  PAUSE                100.00% (n=172)        100.00% (n=41)
  STOP                 100.00% (n=207)        100.00% (n=64)
  NEXT                 100.00% (n=49)         n/a (n=0)
  VOLUME_UP            85.84% (n=572)         84.27% (n=515)
  VOLUME_DOWN          90.09% (n=464)         88.83% (n=412)
  CALL                 91.30% (n=46)          n/a (n=0)
  MESSAGE              82.20% (n=455)         79.70% (n=399)
  LIST_REMINDERS       96.49% (n=57)          n/a (n=0)
  TIMER                97.89% (n=475)         95.00% (n=80)
  ALARM                86.47% (n=658)         74.18% (n=337)
  TEMPERATURE          98.55% (n=1311)        98.59% (n=1133)
  BRIGHTNESS           82.99% (n=488)         64.38% (n=233)
  COLOR                79.92% (n=473)         38.97% (n=136)
  CREATE_REMINDER      77.12% (n=354)         55.68% (n=176)
  unknown_background   100.00% (n=58)         n/a (n=0)
```

checkpoints/exp31_crnn_targeted_s2.pt, all sources:

```
per-class (all / real speech):
  PLAY_MUSIC           79.92% (n=981)         77.19% (n=846)
  WEATHER              81.23% (n=682)         79.75% (n=632)
  TIME                 87.06% (n=394)         85.13% (n=343)
  LIGHT_ON             94.33% (n=758)         93.87% (n=702)
  LIGHT_OFF            88.11% (n=824)         87.86% (n=766)
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
  BRIGHTNESS           87.57% (n=708)         81.02% (n=453)
  COLOR                84.51% (n=581)         65.98% (n=244)
  CREATE_REMINDER      82.49% (n=354)         67.05% (n=176)
  unknown_background   100.00% (n=58)         n/a (n=0)
```

checkpoints/exp32_crnn_slots_s1.pt, all sources:

```
per-class (all / real speech):
  PLAY_MUSIC           73.09% (n=981)         69.03% (n=846)
  WEATHER              71.55% (n=682)         69.94% (n=632)
  TIME                 85.03% (n=394)         83.09% (n=343)
  LIGHT_ON             89.71% (n=758)         88.89% (n=702)
  LIGHT_OFF            87.01% (n=824)         86.55% (n=766)
  PAUSE                100.00% (n=172)        100.00% (n=41)
  STOP                 100.00% (n=207)        100.00% (n=64)
  NEXT                 100.00% (n=49)         n/a (n=0)
  VOLUME_UP            85.84% (n=572)         84.27% (n=515)
  VOLUME_DOWN          90.09% (n=464)         88.83% (n=412)
  CALL                 91.30% (n=46)          n/a (n=0)
  MESSAGE              82.20% (n=455)         79.70% (n=399)
  LIST_REMINDERS       96.49% (n=57)          n/a (n=0)
  TIMER                97.89% (n=475)         95.00% (n=80)
  ALARM                86.47% (n=658)         74.18% (n=337)
  TEMPERATURE          98.55% (n=1311)        98.59% (n=1133)
  BRIGHTNESS           78.25% (n=708)         66.00% (n=453)
  COLOR                78.49% (n=581)         53.69% (n=244)
  CREATE_REMINDER      77.12% (n=354)         55.68% (n=176)
  unknown_background   100.00% (n=58)         n/a (n=0)
```

## 7. Val reject thresholds, `checkpoints/exp32_crnn_slots_s1.pt` (`logs/eval_exp32_val_best.log`)

```
real speech:  79.76% (n=5366)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 78.95% over 16 classes
reject threshold on top-class confidence (real speech):
  threshold  rejected  acc of accepted  errors caught  correct lost
       0.00      0.0%           79.76%           0.0%          0.0%
       0.40      5.6%           82.68%          19.2%          1.8%
       0.50     12.8%           86.26%          40.8%          4.5%
       0.60     19.5%           89.26%          57.3%          7.9%
       0.70     25.7%           92.23%          71.5%         11.2%
       0.80     32.9%           94.95%          83.2%         16.0%
       0.90     41.6%           97.22%          92.0%         23.0%
       0.95     48.5%           98.19%          95.4%         29.2%
```

## 8. Wake word

`logs/eval_exp33_wakeword_s0.log`:

```
model: checkpoints/kiwi_wakeword_s0.pt
positives: 195 held-out 'hey kiwi' clips (test voices); negative stream: 10106 clips, 7.48 h

threshold  false reject (clean)  false reject (10dB noise)  false wake-ups  per hour
     0.50                  0.5%                       2.6%             206     27.52
     0.60                  1.0%                       2.6%             123     16.43
     0.70                  1.5%                       4.1%              61      8.15
     0.80                  3.1%                       6.2%              29      3.87
     0.85                  3.6%                       6.2%              15      2.00
     0.90                  4.6%                       8.2%               9      1.20
     0.95                  8.2%                      16.4%               4      0.53
     0.98                 25.1%                      34.9%               1      0.13
     0.99                 48.7%                      60.5%               0      0.00

suggested threshold: 0.98 (lowest noisy false-reject rate with <= 0.5 false wake-ups/hour)
false wake-ups at that threshold, by clip: [(('/mnt/jfs_hpc/home/quiel.andrew.quiwa/quielq-vcm/data/external/wakeword_synth/audio/NOT_WAKE/NOT_WAKE_test_0098.wav', 'near-miss'), 1)]
```

`logs/eval_exp33_wakeword_s1.log`:

```
model: checkpoints/kiwi_wakeword_s1.pt
positives: 195 held-out 'hey kiwi' clips (test voices); negative stream: 10106 clips, 7.48 h

threshold  false reject (clean)  false reject (10dB noise)  false wake-ups  per hour
     0.50                  1.5%                       2.6%             106     14.16
     0.60                  1.5%                       3.1%              75     10.02
     0.70                  1.5%                       3.6%              42      5.61
     0.80                  1.5%                       5.6%              22      2.94
     0.85                  1.5%                       7.2%              15      2.00
     0.90                  4.1%                       9.7%              10      1.34
     0.95                  6.7%                      16.4%               5      0.67
     0.98                 19.0%                      29.2%               1      0.13
     0.99                 39.0%                      51.8%               0      0.00

suggested threshold: 0.98 (lowest noisy false-reject rate with <= 0.5 false wake-ups/hour)
false wake-ups at that threshold, by clip: [(('data/external/slurp_audio/test/1933_1.flac', 'slurp:WEATHER'), 1)]
```

**Chosen: seed 1** (`checkpoints/kiwi_wakeword_s1.pt`). Both suggest threshold 0.98, where seed 1's noisy false-reject rate is 29.2% vs. seed 0's 34.9%.

## 9. int8 vs. checkpoint

Intent (`checkpoints/exp32_crnn_slots_s1.pt`), test split, no Snips:

PyTorch checkpoint (`logs/eval_exp32_test_nosnips.log`):

```
real speech:  82.07% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 80.28% over 16 classes

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   74.18% /  73.92%  (n=395)
  TIMER       whisper        70.00% /  70.00%  (n=20)
  ALARM       ground_truth   98.44% /  97.82%  (n=321)
  ALARM       whisper        77.91% /  70.35%  (n=172)
  BRIGHTNESS  ground_truth   74.12% /  74.12%  (n=255)
  COLOR       ground_truth   86.94% /  84.57%  (n=337)
  COLOR       whisper        66.04% /  49.06%  (n=53)
```

int8 ONNX (`logs/eval_exp32_int8_nosnips.log`):

```
real speech:  75.32% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 73.57% over 16 classes

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   70.89% /  69.11%  (n=395)
  TIMER       whisper        50.00% /  45.00%  (n=20)
  ALARM       ground_truth   97.82% /  96.57%  (n=321)
  ALARM       whisper        76.16% /  61.63%  (n=172)
  BRIGHTNESS  ground_truth   74.51% /  73.33%  (n=255)
  COLOR       ground_truth   84.87% /  81.31%  (n=337)
  COLOR       whisper        62.26% /  22.64%  (n=53)
```

fp32 ONNX (`logs/eval_exp32_fp32onnx_nosnips.log`, a diagnostic I added that wasn't in the brief, to separate export loss from quantization loss):

```
real speech:  82.09% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 80.29% over 16 classes

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   74.18% /  73.92%  (n=395)
  TIMER       whisper        70.00% /  70.00%  (n=20)
  ALARM       ground_truth   98.44% /  97.82%  (n=321)
  ALARM       whisper        77.91% /  70.35%  (n=172)
  BRIGHTNESS  ground_truth   74.12% /  74.12%  (n=255)
  COLOR       ground_truth   86.94% /  84.57%  (n=337)
  COLOR       whisper        66.04% /  49.06%  (n=53)
```

Wake word int8 (`logs/eval_exp33_int8.log`):

```
model: models/kiwi_wakeword.int8.onnx
positives: 195 held-out 'hey kiwi' clips (test voices); negative stream: 10106 clips, 7.48 h

threshold  false reject (clean)  false reject (10dB noise)  false wake-ups  per hour
     0.50                  1.5%                       3.1%             106     14.16
     0.60                  1.5%                       3.1%              68      9.09
     0.70                  1.5%                       4.1%              37      4.94
     0.80                  1.5%                       5.1%              19      2.54
     0.85                  2.6%                       7.7%              12      1.60
     0.90                  5.1%                      11.3%               9      1.20
     0.95                 10.3%                      16.4%               4      0.53
     0.98                 30.3%                      41.5%               1      0.13
     0.99                 56.9%                      71.3%               0      0.00

suggested threshold: 0.98 (lowest noisy false-reject rate with <= 0.5 false wake-ups/hour)
false wake-ups at that threshold, by clip: [(('data/external/slurp_audio/test/1933_1.flac', 'slurp:WEATHER'), 1)]
```

## 10. Model files and DGX benchmark

`ls -la models/`:

```
total 927
drwxrwxr-x  2 quiel.andrew.quiwa quiel.andrew.quiwa   4096 Sep 25 19:23 .
drwxrwxr-x 13 quiel.andrew.quiwa quiel.andrew.quiwa   4096 Sep 25 19:23 ..
-rw-rw-r--  1 quiel.andrew.quiwa quiel.andrew.quiwa  95397 Sep 25 19:23 kiwi_wakeword.int8.onnx
-rw-rw-r--  1 quiel.andrew.quiwa quiel.andrew.quiwa 109331 Sep 25 19:23 kiwi_wakeword.onnx
-rw-rw-r--  1 quiel.andrew.quiwa quiel.andrew.quiwa 298990 Sep 25 19:23 vcm_intent.int8.onnx
-rw-rw-r--  1 quiel.andrew.quiwa quiel.andrew.quiwa 435873 Sep 25 19:23 vcm_intent.onnx
```

Export output (`logs/exp32_export_intent.log`, `logs/exp33_export_wakeword.log`, metadata lines trimmed):

```
models/vcm_intent.onnx: 426 KB (fp32)
models/vcm_intent.int8.onnx: 292 KB (int8 weights; GRU stays fp32, no int8 GRU kernel in ONNX Runtime)
models/kiwi_wakeword.onnx: 107 KB (fp32)
models/kiwi_wakeword.int8.onnx: 93 KB (int8 weights; GRU stays fp32, no int8 GRU kernel in ONNX Runtime)
```

`logs/exp32_benchmark_dgx.log` (DGX CPU reference only; the Pi will be slower):

```
machine: ai-n002.hpc.coe.upd.edu.ph (x86_64), 256 cores
intent model  vcm_intent.int8.onnx: 292 KB
  per command: features 3.2 ms + model 14.8 ms = 18.0 ms
wake model    kiwi_wakeword.int8.onnx: 93 KB
  per 100 ms hop: 2.7 ms (model 2.2 ms) -> 3% of one core, always on
peak process memory: 72 MB
```

## 11. Unexpected

1. **Exp 32 is worse than Exp 31 at intent classification.**
   - Real-speech test accuracy, no Snips: Exp 31 is 85.16 / 85.08 / 85.48%. Exp 32 is 81.63 / 82.07 / 81.42%. That's about −3.5 pp, against a ~0.6 pp spread between seeds, so it's real.
   - Almost all of the loss is on SLURP (72.67% → 66.53% for the best seeds). By class, the biggest drops on real speech are PLAY_MUSIC (77.19 → 69.03), WEATHER (79.75 → 69.94), COLOR (49.26 → 38.97) and CREATE_REMINDER (67.05 → 55.68).
   - Best val accuracy also fell, from 88.2–88.8% to 83.9–84.4%. That comparison isn't exact, because the Snips re-split moved speakers into val.
   - Exp 32 changed three things at once: the slot heads, the Snips re-split, and the slots2 clips. The Snips re-split can't explain it, since the drop holds with Snips excluded, and SLURP didn't change. My best guess is that the slot loss competes with the intent loss inside a 107K-param model: train loss ends at ~0.61 vs. ~0.17 for Exp 31. I haven't tested that.
   - Isolating it would take one more run: the Exp 32 manifest with `--slot-labels` but `--slot-weight 0` or a lower weight such as 0.3.
   - The slot heads themselves work: ALARM 98.44% and COLOR 86.94% on ground-truth values. TIMER (74.18%) and BRIGHTNESS (74.12%) are weaker.
2. **int8 quantization costs the intent model 6.8 pp. The export itself is fine.**
   - Real speech, no Snips: PyTorch checkpoint 82.07%, fp32 ONNX 82.09%, int8 ONNX 75.32%. Slot accuracy also falls under int8, most for COLOR from Whisper transcripts (joint 49.06% → 22.64%).
   - The int8 file saves only 134 KB (292 KB vs. 426 KB). I'd deploy `vcm_intent.onnx` (fp32) on the Pi unless the benchmark there says otherwise.
   - For the wake word, int8 is close to fp32 at thresholds ≤ 0.85. At the suggested 0.98, false rejects in noise rise from 29.2% to 41.5%.
3. **Wakeword QA.**
   - The first run failed every label (<50%) and I stopped Exp 33 to ask you. The fix for the negatives is in `a701268`.
   - WAKE positives still pass only 44.9% (train) / 48.8% (test). Whisper-base mostly hears the Chatterbox "hey kiwi" clips as "Thank you…" (599 clips), "Kiwi" or "Okay, Kiwi". Those clips are normal length and loudness, so either the TTS says the wrong thing or Whisper mishears a short, unusual phrase. Only listening can tell; I sent you 4 samples.
   - The trainer reports 1,249 `wake` training items, fewer than the 1,347 train positives that passed. I didn't check where the other 98 went (probably the trainer's own val hold-out).
   - Whisper on GPU isn't deterministic: the first and second QA runs differ by 1–2 WAKE clips.
4. **The wake word's suggested threshold (0.98) sits on a steep edge.** At 0.98 the chosen seed rejects 19.0% of clean and 29.2% of noisy "hey kiwi" to reach 0.13 false wake-ups per hour. At 0.95 it rejects 6.7% / 16.4% with 0.67 per hour. The negative stream is mostly other datasets' speech (7.48 h), so real-room false-wake rates need testing on the Pi.
5. **GPU sharing.** GPU 1 was idle at the first check but another user took it before the full launch, so Exp 33 used GPU 7. Neither GPU had another user's process at any of my checks (before each launch). I didn't monitor them continuously.
6. **New data files are not gitignored.** These show as untracked and are not committed: `data/dataset_manifest_v2.csv`, `data/dataset_manifest_pre_exp32.csv`, `data/dataset_manifest_exp32.csv`, `data/slot_labels.csv`.
7. **Setup notes.** `gh` lives in `~/.local/bin` (not on the default PATH). The ONNX export printed a GRU batch-size warning and an onnxruntime "consider pre-processing before quantization" warning. There were no tracebacks in any log.
