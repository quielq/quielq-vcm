# Experiment 36 report: corrected reminder values (exercise)

**Decision: shipped seed 1** (`models/vcm_intent.onnx` = `exp36_joint_w03_s1.pt`), not the best-intent seed 0.
- **Seed 0 (84.86%) failed the ALARM gate:** 96.88% vs. Experiment 34's 99.38%, −2.50 pp against a 2 pp limit, so over by 0.5 pp.
- **Seed 1 passes every gate** and is one clip behind on intent: 84.84% vs. 84.86% on 6,577 clips.
- The first unattended run shipped nothing, because the brief's rule gates only the best-intent seed. The author then chose seed 1.
- The ONNX export matches the checkpoint (see "Shipped model" below).

Numbers are copied from the logs in `logs/` on the DGX.

## Setup

- **Branch** `exp/35-temp-reminder-slots`, pulled to `c59c0c0` (Exp 36 setup: third reminder task is exercise, not call home). `pytest -q`: 232 passed.
- **GPU 2.** It had no other users' processes: I sampled for a minute before starting and checked again before the training launch. All synthesis, QA, training and evaluation ran there.
- Every job ran single-threaded (`OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS=1`), launched with `nohup` from a script in `logs/`, with one log per run. Each stage was smoke-tested first.
- **Recipe:** identical to Experiment 35 (joint, slot weight 0.3, 80 epochs, 3 seeds). The model has 107,887 params, and slot heads train on 14,492 labeled clips (Exp 35: 14,028).

## Step 1: re-voicing the "call home" clips

- I backed up the four shard CSVs and the QA manifest as `*.exp35`, then removed every row with slot_value "call home". That was **230 rows** (45 / 71 / 44 / 70 per shard: 200 train + 30 test), and I deleted their 230 wav files. The brief expected ~460, but the Exp 35 batch only contained 230 "call home" clips.
- **Plan check** (`logs/exp36_plan_check.log`): the plan's 230 "exercise" jobs are exactly the 230 removed ids. No kept clip's planned text or speaker changed.
- **Smoke test:** shard 0 with the limit raised to reach its first "exercise" job. It generated only `CREATE_REMINDER_train_0004` ("set a reminder to exercise", 2.48 s) and skipped the existing clips.
- **Full run:** 4 shards, 150–205 s each. Result (`logs/exp36_synth_check.log`):
  - 1,380 rows again, and the newly generated ids are exactly the 230 removed ones, all "exercise".
  - Every kept row is identical to Exp 35's.
  - No "call home" text is left.

## Step 2: QA (`logs/exp36_qa.log`)

```
test  CREATE_REMINDER   83/90    passed (92.2%)
test  TEMPERATURE    89/90    passed (98.9%)
train CREATE_REMINDER  554/600   passed (92.3%)
train TEMPERATURE   591/600   passed (98.5%)
```

Per value (`logs/exp36_qa_by_value.txt`):

```
CREATE_REMINDER  drink water   test      28/30   (93.3%)
CREATE_REMINDER  drink water   train    189/200  (94.5%)
CREATE_REMINDER  exercise      test      28/30   (93.3%)
CREATE_REMINDER  exercise      train    187/200  (93.5%)
CREATE_REMINDER  study         test      27/30   (90.0%)
CREATE_REMINDER  study         train    178/200  (89.0%)
TEMPERATURE      18 degrees    test      29/30   (96.7%)
TEMPERATURE      18 degrees    train    192/200  (96.0%)
TEMPERATURE      22 degrees    test      30/30   (100.0%)
TEMPERATURE      22 degrees    train    200/200  (100.0%)
TEMPERATURE      26 degrees    test      30/30   (100.0%)
TEMPERATURE      26 degrees    train    199/200  (99.5%)
```

"Exercise" passes 93.5% (train) and 93.3% (test). Its failures are mostly the two-word "reminder exercise" failing on WER ("Remind the exercise.", "a reminder exercise.").

## Steps 3–4: manifest and slot labels

- **`data/dataset_manifest_exp36.csv`:** 70,641 rows (Exp 32's 69,324 plus 1,317 QA-passed slots3 clips).
  - No duplicate audio paths.
  - No "call home" text in any slots3 row.
  - Every slots3 audio file exists.
  - slots3 values: 18/22/26 degrees 221/230/229; drink water 217, study 205, exercise 215.
- **`data/slot_labels_exp36.csv`:** 17,920 labels.
  - CREATE_REMINDER: 2,371 labeled, 956 out of vocabulary. In Exp 35 it was 1,788 labeled and 1,541 out of vocabulary.
  - **`option_b` "exercise" clips labeled "exercise": 580** (test 60, train 461, val 59).
  - Five SLURP recordings are also labeled "exercise" from their Whisper transcripts. None of them is in the test split.
  - Full intent × source × value counts: `logs/exp36_slot_labels_check.log`.
- **`option_b` is synthetic.** The brief calls the `option_b` clips "real", but `DATASET.md` describes Option B as the class-shared **synthetic** dataset, and the manifest marks it `is_synthetic=True`. So the "exercise" gate below is measured on synthetic class clips.

## Training

```
seed 0: Done. Best val_acc=0.8680, checkpoint at checkpoints/exp36_joint_w03_s0.pt
seed 1: Done. Best val_acc=0.8728, checkpoint at checkpoints/exp36_joint_w03_s1.pt
seed 2: Done. Best val_acc=0.8680, checkpoint at checkpoints/exp36_joint_w03_s2.pt
```

## Evaluation (test split, `--manifest data/dataset_manifest_exp36.csv --slot-labels data/slot_labels_exp36.csv`)

**Intent, real speech:**

| Checkpoint | No Snips | All sources |
|---|---:|---:|
| exp36 seed 0 | **84.86%** | 83.96% |
| exp36 seed 1 | 84.84% | 83.82% |
| exp36 seed 2 | 84.46% | 83.80% |
| exp35 seed 2 (shipped before Exp 36) | 85.24% | 84.15% |
| exp34 joint w0.3 | 84.80% | 84.04% |

The Exp 35 and Exp 34 numbers are identical to their Exp 35 evaluation. The real-speech rows didn't change; only the slots3 synthetic rows did.

**Slot-head accuracy, ground-truth rows, no Snips** (computed from `logs/eval_exp36_test_nosnips.log`; `logs/exp36_summary.txt`):

| Checkpoint | TIMER | ALARM | BRIGHTNESS | COLOR | Old-4 mean | TEMPERATURE | CREATE_REMINDER |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp36 seed 0 | 76.20 | **96.88** | 73.33 | 86.94 | 83.34 | 100.00 | 99.62 |
| exp36 seed 1 | 73.92 | 98.13 | 74.90 | 86.65 | 83.40 | 100.00 | 100.00 |
| exp36 seed 2 | 76.71 | 98.13 | 75.69 | 86.65 | 84.30 | 99.25 | 100.00 |
| exp35 seed 2 | 74.43 | 97.82 | 76.08 | 86.65 | 83.75 | 100.00 | 66.28 |
| exp34 | 74.18 | 99.38 | 73.73 | 86.35 | 83.41 | — | — |

Exp 35 scores 66.28% on CREATE_REMINDER because its vocabulary has no "exercise", so it gets 0% on those clips.

**Per-value reminder and temperature slot accuracy, test, no Snips** (slot head / joint with intent; `logs/eval_exp36_per_value_slots.log`, from a scratchpad script that reuses `evaluate_checkpoint.predict` on the same rows):

| Value | Source | n | exp36 s0 | exp36 s1 | exp36 s2 | exp35 s2 |
|---|---|---:|---|---|---|---|
| drink water | option_b | 58 | 100 / 100 | 100 / 98.28 | 100 / 100 | 100 / 100 |
| drink water | slots3 | 28 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| **exercise** | **option_b** | 60 | **100 / 98.33** | **100 / 100** | **100 / 100** | 0 / 0 |
| exercise | slots3 | 28 | 100 / 100 | 100 / 100 | 100 / 100 | 0 / 0 |
| study | option_b | 60 | 100 / 100 | 100 / 98.33 | 100 / 96.67 | 100 / 96.67 |
| study | slots3 | 27 | 96.30 / 96.30 | 100 / 100 | 100 / 100 | 100 / 100 |
| 18 degrees | all | 87 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| 22 degrees | all | 90 | 100 / 100 | 100 / 100 | 97.78 / 97.78 | 100 / 100 |
| 26 degrees | all | 90 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |

**Per-class intent accuracy, all rows / real speech** (`logs/exp36_perclass_lines.txt`):

No Snips:

| Checkpoint | CREATE_REMINDER | CALL | COLOR | BRIGHTNESS | TEMPERATURE |
|---|---|---|---|---|---|
| exp36 s0 | 85.13 / 63.64 | 93.48 / n/a | 81.82 / 41.91 | 86.07 / 70.82 | 99.00 / 98.76 |
| exp36 s1 | 83.07 / 59.09 | 95.65 / n/a | 83.30 / 49.26 | 85.25 / 69.10 | 99.50 / 99.38 |
| exp36 s2 | 85.81 / 65.91 | 95.65 / n/a | 80.76 / 36.76 | 86.27 / 71.24 | 99.14 / 98.94 |
| exp35 s2 | 83.30 / 59.66 | 95.65 / n/a | 83.51 / 47.06 | 85.45 / 69.53 | 99.57 / 99.47 |
| exp34 | 85.35 / 65.34 | 95.65 / n/a | 82.03 / 42.65 | 86.27 / 71.67 | 98.86 / 98.85 |
| n | 437 / 176 | 46 / 0 | 473 / 136 | 488 / 233 | 1400 / 1133 |

All sources (adds Snips lighting to COLOR and BRIGHTNESS):

| Checkpoint | COLOR | BRIGHTNESS |
|---|---|---|
| exp36 s0 | 79.52 / 54.10 | 81.78 / 71.52 |
| exp36 s1 | 82.62 / 62.70 | 80.08 / 68.87 |
| exp36 s2 | 79.52 / 53.28 | 83.90 / 74.83 |
| exp35 s2 | 82.10 / 59.84 | 80.37 / 69.32 |
| exp34 | 80.72 / 56.97 | 83.62 / 74.61 |
| n | 581 / 244 | 708 / 453 |

- **CALL has no real-speech test clips**, only 46 synthetic `option_b` clips. This test set can't measure the "bare call" regression you heard on your own voice. On the 46 clips, only seed 0 differs (93.48% vs. 95.65%, one clip).
- **COLOR on real speech varies a lot between seeds:** 36.8–49.3% without Snips, 53.3–62.7% with. Seed 1 is the best COLOR seed in both views, above Exp 34 and Exp 35.
- **CREATE_REMINDER on real speech** (n=176, SLURP only) spans 59.1–65.9% across seeds, as it did in Exp 35. This looks like seed noise.

## Decision

- **Seed:** 0, the best real-speech intent accuracy without Snips (84.86%), as the brief specifies.
- **Gates for seed 0:**
  - Intent 84.86% ≥ 83.8%: **pass** (Exp 34 84.80%, Exp 35 85.24%).
  - Old four slot heads vs. Exp 34, per head as in Experiment 35: TIMER +2.02, **ALARM −2.50**, BRIGHTNESS −0.40, COLOR +0.59. **Fail** on ALARM: 96.88% vs. 99.38%, n=321, about 8 clips. The old-4 mean (83.34% vs. 83.41%) would pass.
  - TEMPERATURE 100.00% and CREATE_REMINDER 99.62%: **pass**. "Exercise" on option_b test clips 100.00% (joint 98.33%): **pass**.
- **First run: not shipped.** The brief's rule gates only the best-intent seed, and the two seeds differ by one clip, so I didn't substitute my own selection rule while unattended.
- **Gates for seed 1** (all pass):
  - Intent 84.84% ≥ 83.8%.
  - Old heads vs. Exp 34: TIMER −0.26, ALARM −1.25, BRIGHTNESS +1.17, COLOR +0.30.
  - TEMPERATURE 100%, CREATE_REMINDER 100%, "exercise" on option_b test clips 100% / 100%.
- **Shipped: seed 1**, at the author's direction, for the reason given at the top of this report. It also has the best real-speech COLOR of any model here.

## Shipped model: seed 1

**Export:** `scripts/export_onnx.py checkpoints/exp36_joint_w03_s1.pt --out models/vcm_intent` wrote `models/vcm_intent.onnx`, 432 KB fp32. I then restored `models/vcm_intent.int8.onnx` from git, since only fp32 ships; the int8 file is still Experiment 34's.

ONNX metadata (`logs/exp36_onnx_metadata.txt`):

```
source_checkpoint: exp36_joint_w03_s1.pt | val_acc: 0.8728
slot heads: ['TIMER', 'ALARM', 'BRIGHTNESS', 'COLOR', 'TEMPERATURE', 'CREATE_REMINDER']
CREATE_REMINDER: ['drink water', 'study', 'exercise']
TEMPERATURE: ['18 degrees', '22 degrees', '26 degrees']
```

**Re-scored on the same rows** (`logs/eval_exp36_fp32onnx_nosnips.log`: test, `--exclude-source snips_lights`). Both the ONNX file and the seed 1 checkpoint (`logs/eval_exp36_test_nosnips.log`) print these lines, character for character:

```
real speech:  84.84% (n=6577)   <- comparable to the cascade's 90.62%
real speech, macro (mean of per-class): 83.27% over 16 classes

slot values (slot-head accuracy / intent+slot joint accuracy):
  TIMER       ground_truth   73.92% /  73.16%  (n=395)
  TIMER       whisper        80.00% /  80.00%  (n=20)
  ALARM       ground_truth   98.13% /  97.82%  (n=321)
  ALARM       whisper        77.33% /  69.77%  (n=172)
  BRIGHTNESS  ground_truth   74.90% /  74.90%  (n=255)
  COLOR       ground_truth   86.65% /  85.46%  (n=337)
  COLOR       whisper        67.92% /  58.49%  (n=53)
  TEMPERATURE ground_truth  100.00% / 100.00%  (n=267)
  CREATE_REMINDER ground_truth  100.00% /  99.23%  (n=261)
```

Seed 1 per-class intent accuracy, all rows / real speech (identical for the ONNX file and the checkpoint):

```
  TEMPERATURE          99.50% (n=1400)        99.38% (n=1133)
  BRIGHTNESS           85.25% (n=488)         69.10% (n=233)
  COLOR                83.30% (n=473)         49.26% (n=136)
  CREATE_REMINDER      83.07% (n=437)         59.09% (n=176)
```

**The only differences** in the full evaluation blocks are on synthetic rows and one confidence-table row:
- One synthetic `option_b` WEATHER clip that the ONNX file gets right: all 89.08% → 89.09%, synthetic 98.88% → 98.92%, `WEATHER -> MESSAGE` 38 → 37.
- The 0.80 reject-threshold row moves by 0.1 pp. That's floating-point noise on a clip near a decision boundary.

## Caveats

- The new slot heads are measured only on synthetic speech (`option_b` and slots3). No real-speech test clip has a temperature or reminder value.
- Seed 1's CREATE_REMINDER intent on real speech (59.09%, n=176, SLURP) is the lowest of the three seeds, and below Experiment 34's 65.34%. Seeds spanned 59.1–65.9%, so live testing of reminder commands is worth doing.
- Not committed: the DGX data files (`data/dataset_manifest_exp36.csv`, `data/slot_labels_exp36.csv`, `data/external/targeted_synth_slots3/`, with Exp 35's CSVs backed up as `*.exp35`).
- No PR and no Pi deploy, as instructed.
