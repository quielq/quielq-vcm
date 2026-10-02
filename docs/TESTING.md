# Testing and results

Three kinds of testing, from most to least formal:

| | What it checks | Where |
|---|---|---|
| [1. Model evaluation](#1-model-evaluation-final-results) | Accuracy of the shipped models on held-out test data | DGX (needs the datasets) |
| [2. On-device and live tests](#2-on-device-and-live-tests) | Speed and memory on the Pi; real commands spoken to the device | Raspberry Pi |
| [3. Automated test suite](#3-automated-test-suite) | Code correctness (features, data loaders, losses, runtime, home server) | Anywhere, no hardware |

## 1. Model evaluation: final results

### How the test set is defined

- **The class's fixed test split** of the master dataset
  ([DATASET.md](DATASET.md#the-class-master-dataset-experiment-37-on)):
  4,418 clips, 47 per Option B variation (93 variations) plus 47
  out-of-scope clips, from 121 speakers and synthetic voices that are in
  no other split. This is the headline number, the same set every group in
  the class reports on.
- **Real speech** is reported next to it: the 1,003 test clips spoken by
  real people (group recordings, SLURP, FSC, SNIPS, Common Voice, …).
  Synthetic voices score ~99.5% and make up 76% of the test set, so the
  real-speech number is the honest one.
- **Holdout**: the 196 clips the class kept for the live test on the
  Raspberry Pi, scored offline the same way.
- **Choosing on validation only.** Every setting and checkpoint was chosen
  on our val split (12% of train, by speaker). Test and holdout were
  reported but never used to choose.
- Every test clip counts, including ones the device would reject for low
  confidence.

### Intent accuracy (shipped model, `models/vcm_intent.onnx`)

The ONNX file and its checkpoint (`models/vcm_intent.pt`, Experiment 41d
seed 0) give identical results.

| Measure | Result |
|---|---:|
| **Class test set, all clips** | **92.98%** (n = 4,418) |
| Real speech | 73.48% (n = 1,003) |
| Synthetic voices | 99.50% (n = 3,368) |
| Macro average over 20 classes | 88.06% |
| Exact Option B wording (the demo benchmark phrases) | 99.06% (n = 3,601) |
| Same command and value in other words | 67.66% (n = 770) |
| Filipino group recordings | 87.30% (n = 189) |
| **Holdout (Pi live-test set)** | **94.90%** (real speech 94.19%, n = 196) |
| Seeds of the same recipe (0 / 1 / 2), test all | **92.98** / 92.80 / 92.98% |
| Seeds, test real speech | **73.48** / 73.08 / 73.08% |
| Small alternative, `vcm_intent_small.onnx` (722 KB) | 92.21% all, 70.99% real, 95.41% holdout |

**By accent** (test): Filipino group recordings 87.3% (n=189), Filipino
open-source 71.4% (n=14), native English 72.8% (n=254), other non-native
67.3% (n=560), other Asian English 35.0% (n=20), synthetic 99.5%.

**By source** (test): class recordings 88.0% (n=200), FSC 78.4%, Common Voice 71.4%, SLURP 67.8%, SNIPS 56.1%.

**Per class** (test, all / real speech):

| Class | All | Real | | Class | All | Real |
|---|---:|---:|---|---|---:|---:|
| PLAY_MUSIC | 73.8% | 48.6% | | CALL | 89.4% | 66.7% |
| WEATHER | 83.0% | 68.1% | | MESSAGE | 94.3% | 55.6% |
| TIME | 91.5% | 79.0% | | LIST_REMINDERS | 96.5% | 87.2% |
| LIGHT_ON | 77.3% | 46.6% | | TIMER | 99.5% | 89.5% |
| LIGHT_OFF | 82.3% | 66.7% | | ALARM | 97.9% | 90.2% |
| PAUSE | 91.5% | 83.0% | | TEMPERATURE | 100% | 100% |
| STOP | 92.2% | 88.9% | | BRIGHTNESS | 98.6% | 88.0% |
| NEXT | 95.7% | 85.7% | | COLOR | 94.3% | 79.7% |
| VOLUME_UP | 80.1% | 61.1% | | CREATE_REMINDER | 100% | 100% |
| VOLUME_DOWN | 80.9% | 63.9% | | OUT_OF_SCOPE | 42.6% | — |

The weakest real-speech classes are dominated by SLURP and SNIPS
free-form phrasings ("olly turn on the lights", "put some music on").

**Out of scope.** 42.6% of out-of-scope test clips are labeled
OUT_OF_SCOPE; 57.4% are taken for some command, but only **21.3%** with
confidence ≥ 0.6, the level at which the device acts. On the device the
wake word filters most non-command speech before the model hears it.

**Reject threshold** (real speech): at 0.6 the device says "didn't catch
that, please repeat" for 21.8% of commands and is right on 84.2% of the
ones it accepts (53% of its errors are caught).

**Progress on the master dataset** (mean of 3 seeds; EXPERIMENTS.md
Experiments 37–41):

| Model | Params | val real | test all | test real | holdout |
|---|---:|---:|---:|---:|---:|
| DS-CNN baseline, same size (38) | 100K | 51.5% | 86.9% | 52.2% | 82.7% |
| Experiment 36 recipe, retrained (37a) | 99K | 64.0% | 90.1% | 63.6% | 92.4% |
| + 2-layer GRU, 4 heads, freq-only SpecAugment, numerals as OOS (40a) | 182K | 69.7% | 92.7% | 73.2% | 93.2% |
| + self-distillation (40b) | 182K | 70.4% | 92.4% | 71.7% | 94.6% |
| + wider (41d, **shipped**) | 372K | **72.3%** | **92.9%** | **73.2%** | **93.9%** |

**The old model on this test set.** The Experiment 36 model (old 70k-clip
dataset) scores 92.37% / 78.27% real here, but 2,439 of the 4,418 test
clips were in its training data (found by hashing the decoded audio). On
the 1,979 clips it never saw: 86.71% all, 65.82% real; our first retrain
(37a) scored 89.24% / 64.98% on the same clips
(`results/eval/exp37_eval_test_unseen_by_exp36.txt`).

### Slot values (shipped model)

The value head's accuracy on test clips whose value is one of the
schema's (labels from the dataset). "Joint" also requires the intent to be
right, which is what the device needs.

| Slot | Values | Real: slot / joint | Synthetic: slot / joint | n (real / synthetic) |
|---|---:|---:|---:|---:|
| TIMER | 10 s, 30 s, 1 min | 89.5% / 79.0% | 100% / 100% | 19 / 404 |
| ALARM | 6:00 AM, 8:00 AM, 9:00 PM | 80.4% / 74.5% | 99.2% / 98.4% | 51 / 372 |
| TEMPERATURE | 18, 22, 26 degrees | 92.9% / 92.9% | 99.0% / 99.0% | 28 / 395 |
| BRIGHTNESS | 20, 60, 100 percent | 100% / 88.0% | 98.5% / 98.2% | 25 / 398 |
| COLOR | red, blue, green | 87.3% / 72.0% | 98.4% / 98.4% | 118 / 305 |
| CREATE_REMINDER | drink water, study, exercise | 77.8% / 77.8% | 99.5% / 99.5% | 18 / 405 |

Over all 2,538 slotted test clips the slot head is right 97.9% of the time.

### Out of scope (how it is handled and tested)

The class dataset labels speech that asks for none of the 19 commands
`OUT_OF_SCOPE`: noise, Filipino speech, near-miss requests and general
speech (187 train, 47 test and 10 holdout clips). We keep it exactly as
the dataset defines it:

- **In training** it is the model's 20th class, "not a command". Bare
  numbers from the numerals set (also labeled OUT_OF_SCOPE by the dataset)
  add 1,500 examples. A model without this class would have to map every
  sound to one of the 19 commands.
- **On the device** nothing happens for it: an OUT_OF_SCOPE answer, or any
  answer below 0.6 confidence ("didn't catch that"), is never acted on.
  Before that, the wake word must hear "Hey Kiwi", which filters most
  non-command speech.
- **In testing** an out-of-scope clip is correct only if the model says
  OUT_OF_SCOPE. The headline 92.98% includes the 47 such test clips.
  Separately:

| Shipped model, test split | Result | n |
|---|---:|---:|
| Command accuracy, out-of-scope clips excluded | 93.53% (real speech 73.48%) | 4,371 |
| Out-of-scope clips labeled OUT_OF_SCOPE | 42.6% | 47 |
| **Out-of-scope clips the device ignores** (OUT_OF_SCOPE or confidence < 0.6) | **78.7%** | 47 |
| Out-of-scope clips the device would act on (false accept) | 21.3% | 47 |
| Commands the device ignores (false reject, same rule) | 5.5% | 4,371 |
| Holdout: commands 97.31% (n=186); out-of-scope ignored 60.0% (n=10) | | |

Out of scope is the weakest class: it has the fewest training clips, and
"near-miss requests" (commands outside the schema) sound like commands.

### Wake word (`models/kiwi_wakeword.onnx`)

Evaluated the way the device runs it, streaming 1.5 s windows every 0.1 s:
missed wake words on 391 held-out clips in 20 unseen voices (clean and with
10 dB noise), the author's 10 held-out real recordings, and false wake-ups
while streaming the master dataset's whole test split plus 300 near-miss
phrases (3.05 hours). Experiment 42, seed 1 (best of 3 on validation):

| Threshold | Missed, clean | Missed, 10 dB noise | Author's real takes missed | False wake-ups per hour |
|---:|---:|---:|---:|---:|
| **0.6 (default)** | 4.1% | 4.6% | 0/10 | 11.2 |
| 0.7 | 4.1% | 6.6% | 0/10 | 8.5 |
| 0.85 | 7.9% | 11.3% | 0/10 | 4.3 |
| 0.95 | 17.9% | 24.0% | 1/10 | 1.6 |

The false-wake stream deliberately includes near-miss phrases ("hey kitty",
"every week"), so a real room sees fewer. On the same evaluation the
previous detector (Experiment 34, trained with the old dataset's speech as
negatives) misses 3.1% / 6.6% and fires 19.7 times per hour at 0.6. The
master test split has no pure-noise clips, so the 10 dB check mixes in the
train split's 11 noise clips.

### Known limits of the test set

- **Mostly synthetic.** 76% of test clips are the group's synthetic
  voices, which score ~99.5%. Report real speech next to the overall
  number.
- **Small real-speech counts for some classes.** MESSAGE, TIMER and
  CREATE_REMINDER have 18–19 real test clips each; OUT_OF_SCOPE has 47
  clips in all.
- **The master dataset has no "hey kiwi" recordings.** Wake-word misses
  are measured on our own held-out synthetic voices and recordings; its
  false wake-ups on the master test split.
- **Only 47 out-of-scope test clips** (10 in holdout), so out-of-scope
  rates move by ~2 points per clip.

### Reproduce

On a GPU machine, from a fresh clone: `bash scripts/reproduce.sh`. Or, with
`data/me2/` built ([DATASET.md](DATASET.md)):
```bash
python scripts/evaluate_checkpoint.py models/vcm_intent.onnx \
  --manifest data/me2/manifest.csv --slot-labels data/me2/slot_labels.csv \
  --metadata data/me2/metadata.csv --split test        # or --split holdout
python scripts/summarize_experiments.py --logs results/eval --glob '4*'
python scripts/evaluate_wakeword.py models/kiwi_wakeword.onnx --intent-manifest data/me2/manifest.csv \
  --extra-wake-manifest data/wakeword_real/manifest.csv
```
`evaluate_checkpoint.py` accepts `.pt` checkpoints or `.onnx` files and
prints overall, real-speech, per-source, per-accent, per-class,
confusable-group, slot and reject-threshold tables.

## 2. On-device and live tests

**Speed and memory** (`scripts/benchmark_pi.py`, no microphone needed;
`--clips` times real recordings, `--json` saves the numbers). It reports
mean, p50 and p95 latency per command and the real-time factor, with ONNX
Runtime on 1 thread. On the Raspberry Pi 5 (8 GB): 13.9 ms p50 / 15.8 ms
p95 per command, RTF 0.0063 at p95, wake word 1.9% of one core, 100 MB
peak; 604 MB used system-wide with both services running
([results/bench_pi5.md](../results/bench_pi5.md)). The previous model took
9.9 ms. Details in [FOOTPRINT.md](FOOTPRINT.md).

**Preflight check** before a demo: `scripts/kiwi_doctor.py --beep` checks
power, microphone, speaker, services, the listener, the home server, the
models and the internet, and ends with READY ([RUNBOOK.md](RUNBOOK.md)).

**Live commands.** The author said 137 commands to the device, saved with
`--save-commands` and transcribed offline to check what was actually said.
[RUNBOOK.md](RUNBOOK.md#test-checklist) lists three phrasings per command
with how often each was acted on correctly, and the phrasings to avoid.
Most work every time; CALL works reliably only as "make a call". The
dashboard's "Simulate a command" box runs every action without speaking, to
tell recognition problems from action problems.

## 3. Automated test suite

244 tests, all pure logic: no microphone, speaker, network, GPU or dataset
needed. They use synthetic arrays, small fixture files and mocked hardware.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"      # train adds PyTorch, needed by the training tests
python -m pytest                   # expect: 244 passed
```

Use `python -m pytest`, not bare `pytest`: on macOS a Homebrew `pytest` can
come first on `PATH` and run against the wrong interpreter.

| Area | Test files | Covers |
|---|---|---|
| Audio | `test_dsp`, `test_features`, `test_resample` | numpy log-mel matches librosa; feature shapes; silence doesn't produce NaN; 48 → 16 kHz resampling |
| Dataset | `test_dataset_manifest`, `test_dataset_schema_loader`, `test_slurp_coverage`, `test_fsc_coverage`, `test_option_b`, `test_gsc_background`, `test_snips_lights`, `test_timers_and_such`, `test_targeted_synth`, `test_synthetic_check` | Each source's label mapping and splits, the common manifest format, the synthetic-audio QA gate |
| Training | `test_train_architectures`, `test_train_dataset`, `test_train_augment`, `test_train_wave_augment`, `test_losses`, `test_train_transcripts`, `test_slots` | Model output shapes and slot heads, class weights, augmentation, the confusable-pair loss, slot-value parsing |
| Deployment | `test_deploy`, `test_wakeword`, `test_listen` | ONNX export matches PyTorch and carries its metadata; the streaming wake-word detector; the listener's endpointing, music ducking and microphone-loss handling |
| Home server | `test_home` | Every intent has an action; music, ducking, lights, thermostat, timers, alarms, reminders, phone bridge; failing integrations don't crash; an HTTP round trip |
| Configuration | `test_config`, `test_hal_selection` | Settings loading; Mac vs. Raspberry Pi hardware selection |

New tests follow the same rule: test logic with fakes; anything that needs
real hardware goes in the live checks in section 2.
