# Experiment 35 report: temperature and reminder slot heads

**Decision: shipped.** `models/vcm_intent.onnx` is now `exp35_joint_w03_s2.pt`, which has 6 slot heads. It passes all three gates against the Experiment 34 joint w0.3 baseline:

- Intent accuracy on real speech: 85.24% vs. 84.80%.
- Old four slot heads: the largest per-head drop is ALARM, −1.56 pp.
- TEMPERATURE and CREATE_REMINDER slot accuracy: 100% each.

Numbers are copied from the logs in `logs/` on the DGX.

## Setup

- **Branch** `exp/35-temp-reminder-slots` at `28293ad` (Exp 35 setup: temperature and reminder slot heads). `git pull`: already up to date. `pytest -q`: 232 passed.
- **GPU 7, shared.** No GPU was free: every GPU had another user's processes, and 2 minutes of sampling found none that cleared. GPU 7 was the least loaded: two small jobs from another user, about 2.7 GB used and ~11% average utilization. Synthesis, QA, training, evaluation and the ONNX check all ran there; the export ran on CPU.
- Every job ran single-threaded (`OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS=1`), launched with `nohup` from a script in `logs/`, with one log per run. Each stage was smoke-tested first: synthesis `--limit 2`, QA on those 2 clips, training `--epochs 1 --train-fraction 0.02`.
- **Recipe:** Experiment 34's joint recipe (slot weight 0.3, 80 epochs, CRNN, 5 s window, silence trim, waveform augmentation), 3 seeds packed on GPU 7. The model has 107,887 params (Exp 34: 106,855), and slot heads train on 14,028 labeled clips.

## Synthesis and QA (slots3)

The smoke test gave 2 clips, 2.84 s and 1.82 s; both passed QA (Whisper heard "18 degrees" and "22 degrees").

The full run was 4 shards, 1,380 clips, 1,498 s wall time. Per shard:

```
shard 0: done (345 jobs, 1430s)
shard 1: done (345 jobs, 1442s)
shard 2: done (345 jobs, 1421s)
shard 3: done (345 jobs, 1442s)
```

QA (`logs/exp35_qa.log`):

```
test  CREATE_REMINDER   83/90    passed (92.2%)
test  TEMPERATURE    89/90    passed (98.9%)
train CREATE_REMINDER  556/600   passed (92.7%)
train TEMPERATURE   591/600   passed (98.5%)
```

Per intent and value (`logs/exp35_qa_by_value.txt`):

```
intent           value         split        passed
CREATE_REMINDER  call home     test      28/30   (93.3%)
CREATE_REMINDER  call home     train    189/200  (94.5%)
CREATE_REMINDER  drink water   test      28/30   (93.3%)
CREATE_REMINDER  drink water   train    189/200  (94.5%)
CREATE_REMINDER  study         test      27/30   (90.0%)
CREATE_REMINDER  study         train    178/200  (89.0%)
TEMPERATURE      18 degrees    test      29/30   (96.7%)
TEMPERATURE      18 degrees    train    192/200  (96.0%)
TEMPERATURE      22 degrees    test      30/30   (100.0%)
TEMPERATURE      22 degrees    train    200/200  (100.0%)
TEMPERATURE      26 degrees    test      30/30   (100.0%)
TEMPERATURE      26 degrees    train    199/200  (99.5%)
```

Failures are a mix of two kinds:
- **TTS mispronunciation:** "trick water", "co-home".
- **WER only:** the value was heard but the transcript's WER exceeded 0.34, e.g. "Set the thermal stage to 18 degrees."

## Manifest and slot labels

`data/dataset_manifest_exp35.csv` has 70,643 rows: the Exp 32 manifest's 69,324 plus 1,319 QA-passed slots3 clips.
- No audio_path is duplicated.
- Every Exp 32 row is unchanged.
- Row counts for `targeted_synth` (round1, 4,357) and `targeted_synth_slots2` (2,649) are identical to before.

`data/slot_labels_exp35.csv` has 17,337 labels (`logs/exp35_slot_labels.log`):

```
TIMER      labeled: 3413  out of vocabulary: 535  no text: 0
ALARM      labeled: 3721  out of vocabulary: 982  no text: 0
BRIGHTNESS labeled: 2257  out of vocabulary: 2130  no text: 0
COLOR      labeled: 3748  out of vocabulary: 494  no text: 0
TEMPERATURE labeled: 2410  out of vocabulary: 9374  no text: 0
CREATE_REMINDER labeled: 1788  out of vocabulary: 1541  no text: 0
values with < 20 labeled clips: []
```

Every value of all 6 intents has labels. For TEMPERATURE and CREATE_REMINDER, the labels come from:

- **`option_b`** (the class-schema dataset; synthetic): TEMPERATURE 18/22/26 degrees, 1,730 clips. CREATE_REMINDER "drink water" and "study", 1,149 clips.
- **`targeted_synth_slots3`**: all three values of each intent.
- **No real-speech labels:** no real recording (SLURP or FSC) has a Whisper transcript containing any of these values. FSC's 9,374 temperature commands carry no number ("increase the heat"), and none of SLURP's 961 reminder recordings mention "drink water", "study" or "call home".

**Vocabulary mismatch.** `option_b`'s third reminder task is **"exercise"**, not "call home":
- `option_b` has 580 "exercise" clips ("Create a reminder to exercise", "Remind me to exercise", "Reminder exercise"). They get no slot label (out of vocabulary).
- `dataset_schema.py` and `vcm.slots` both say "Call home". So "call home" is represented only by our own slots3 clips (189 train, 28 test).
- I didn't change the vocabulary; the brief only allows committing the report, EXPERIMENTS.md and the model.

## Training

```
seed 0: Done. Best val_acc=0.8722, checkpoint at checkpoints/exp35_joint_w03_s0.pt
seed 1: Done. Best val_acc=0.8725, checkpoint at checkpoints/exp35_joint_w03_s1.pt
seed 2: Done. Best val_acc=0.8679, checkpoint at checkpoints/exp35_joint_w03_s2.pt
```

The last epochs had val_slot_acc 0.9528 / 0.9460 / 0.9494 (Exp 34 joint w0.3: 0.9475).

## Evaluation (test split, all 4 checkpoints on identical rows)

I ran the brief's command (`logs/eval_exp35_test.log`, all sources). I also ran the same command with `--exclude-source snips_lights` (`logs/eval_exp35_test_nosnips.log`), because the baseline's 84.80% / 83.41%, and so the 83.8% gate, were measured without Snips. The baseline reproduces there exactly.

**Intent, real speech:**

| Checkpoint | No Snips (gate) | All sources | Macro, no Snips |
|---|---:|---:|---:|
| exp35 seed 0 | 85.15% | 84.15% | 82.69% |
| exp35 seed 1 | 85.19% | 84.17% | 83.57% |
| **exp35 seed 2** | **85.24%** | 84.15% | 83.08% |
| exp34 joint w0.3 (baseline) | 84.80% | 84.04% | 82.94% |

**Slot-head accuracy, ground-truth rows** (identical in both logs):

| Checkpoint | TIMER | ALARM | BRIGHTNESS | COLOR | Old-4 mean | TEMPERATURE | CREATE_REMINDER |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp35 seed 0 | 73.92% | 97.51% | 74.12% | 87.83% | 83.34% | 100.00% | 100.00% |
| exp35 seed 1 | 75.44% | 97.82% | 73.73% | 86.05% | 83.26% | 100.00% | 100.00% |
| **exp35 seed 2** | 74.43% | 97.82% | 76.08% | 86.65% | **83.75%** | 100.00% | 100.00% |
| baseline | 74.18% | 99.38% | 73.73% | 86.35% | 83.41% | — | — |
| n | 395 | 321 | 255 | 337 | | 267 | 201 |

Joint (intent+slot) accuracy for seed 2: TEMPERATURE 100.00%, CREATE_REMINDER 99.00%.

## Decision

- **Seed:** 2. It has the best real-speech intent accuracy without Snips, the same measurement as the baseline and the gate, and the best old-4 slot mean. Seed 1 leads the all-sources log by 0.02 pp (about one clip). Seeds 1 and 2 both pass every gate, so the decision doesn't depend on the choice.
- **Gates for seed 2:**
  - Intent 85.24% ≥ 83.8% (+0.44 vs. baseline; all sources 84.15% vs. 84.04%).
  - Old slot heads vs. baseline: TIMER +0.25, ALARM −1.56, BRIGHTNESS +2.35, COLOR +0.30. All within 2 pp.
  - TEMPERATURE 100.00% and CREATE_REMINDER 100.00%, both ≥ 90%.
- **Export:** `models/vcm_intent.onnx` (432 KB fp32). The metadata lists 6 slot heads with outputs `slot_TIMER`, `slot_ALARM`, `slot_BRIGHTNESS`, `slot_COLOR`, `slot_TEMPERATURE`, `slot_CREATE_REMINDER` (`logs/exp35_onnx_metadata.txt`).
- **ONNX check:** re-scored on the same rows (`logs/eval_exp35_fp32onnx_nosnips.log`), real speech is 85.24% and the slot lines are identical to the checkpoint's.
- **int8 file:** `export_onnx.py` also rewrote `models/vcm_intent.int8.onnx`. I restored the committed version because the brief ships only fp32, so the int8 file in the repo is still Experiment 34's.

## Caveats (what I checked, and what I didn't)

- **The new slot accuracies are on synthetic speech only.** The 267 TEMPERATURE and 201 CREATE_REMINDER test clips are all `option_b` or slots3; no real-speech test clip has a value. 100% says the heads learned these clips, not how they do on a real voice. Live testing should check "set the thermostat to 22 degrees" and "remind me to call home".
- **CREATE_REMINDER intent accuracy on real speech is noisy and mostly lower:** 55.68 / 66.48 / 59.66% vs. the baseline's 65.34% (n=176, all SLURP).
- **BRIGHTNESS intent accuracy on real speech is lower in all three seeds:** 66.89 / 69.09 / 69.32% vs. 74.61% in the all-sources log.
- **Snips lighting is lower:** 72.61 / 72.26 / 71.55% vs. 75.27%. The baseline is one seed, so part of this may be seed noise.
- **COLOR on real speech is higher:** 57.79 / 58.20 / 59.84% vs. 56.97%.
- **The "exercise" vs. "call home" mismatch** described above.
- **No PR, no Pi deploy**, as instructed. The DGX data files (`data/dataset_manifest_exp35.csv`, `data/slot_labels_exp35.csv`, `data/external/targeted_synth_slots3/`) aren't committed.
