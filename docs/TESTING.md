# Testing and results

Three kinds of testing, from most to least formal:

| | What it checks | Where |
|---|---|---|
| [1. Model evaluation](#1-model-evaluation-final-results) | Accuracy of the shipped models on held-out test data | DGX (needs the datasets) |
| [2. On-device and live tests](#2-on-device-and-live-tests) | Speed and memory on the Pi; real commands spoken to the device | Raspberry Pi |
| [3. Automated test suite](#3-automated-test-suite) | Code correctness (features, data loaders, losses, runtime, home server) | Anywhere, no hardware |

## 1. Model evaluation: final results

### How the test set is defined

- **The class's fixed test split** of the master dataset, revision
  `da92a79` (2026-10-02; see
  [DATASET.md](DATASET.md#revisions)): 4,443 clips, 47 per Option B
  variation ("Message" 43) plus 76 out-of-scope clips, from 144 speakers and
  synthetic voices that are in no other split. This is the headline
  number, the same set every group in the class reports on.
- **Real speech** is reported next to it: the 777 test clips of real
  people saying a command (class recordings, SLURP, FSC, SNIPS, Common
  Voice, …). Synthetic voices score ~99.5% and make up 81% of the test set,
  so the real-speech number is the honest one.
- **Holdout**: the 202 clips the class kept for the live test on the
  Raspberry Pi, scored offline the same way.
- **Choosing on validation only.** Every setting and checkpoint was chosen
  on our val split (12% of train, by speaker). Test and holdout were
  reported but never used to choose.
- Every test clip counts, including ones the device would reject for low
  confidence.
- Experiments 37–42 were scored on the dataset's first revision, whose
  test set had 226 more free-form real clips; those numbers are about 2
  points lower overall and are not comparable with the ones here.

### Intent accuracy (shipped model, `models/vcm_intent.onnx`)

The ONNX file and its checkpoint (`models/vcm_intent.pt`, Experiment 43b
seed 1) give identical results.

| Measure | Result |
|---|---:|
| **Class test set, all clips** | **95.50%** (n = 4,443) |
| Real speech | 78.64% (n = 777) |
| Synthetic voices | 99.45% (n = 3,619) |
| Macro average over 20 classes | 92.94% |
| Exact Option B wording (the demo benchmark phrases) | 99.22% (n = 3,823) |
| Same command and value in other words | 72.98% (n = 544) |
| Filipino speakers (class recordings) | 86.77% (n = 189) |
| **Holdout (Pi live-test set)** | **96.04%** (real speech 96.51%, n = 202) |
| Seeds of the same recipe (0 / 1 / 2), test all | 95.00 / **95.50** / 95.32% |
| Seeds, test real speech | 76.71 / **78.64** / 77.35% |
| Small alternative, `vcm_intent_small.onnx` (722 KB, Exp 43c seed 0) | 94.53% all, 76.06% real, 93.07% holdout |

**By accent** (test): Filipino class recordings 86.8% (n=189), Filipino
open-source 100% (n=14), native English 83.9% (n=224), other non-native
69.0% (n=384), synthetic 99.5%.

**By source** (test): class recordings 87.5% (n=200), FSC 87.5% (n=144),
Common Voice 100% (n=14), SLURP 69.7% (n=403), SNIPS 75.4% (n=61),
synthetic out-of-scope sentences 69.0% (n=29).

**Per class** (test, all / real speech):

| Class | All | Real | | Class | All | Real |
|---|---:|---:|---|---|---:|---:|
| PLAY_MUSIC | 90.8% | 38.1% | | CALL | 90.8% | 37.5% |
| WEATHER | 90.1% | 46.2% | | MESSAGE | 94.9% | 50.0% |
| TIME | 95.7% | 88.0% | | LIST_REMINDERS | 97.9% | 89.3% |
| LIGHT_ON | 93.6% | 62.5% | | TIMER | 98.8% | 79.0% |
| LIGHT_OFF | 86.5% | 60.4% | | ALARM | 98.8% | 94.1% |
| PAUSE | 95.7% | 90.0% | | TEMPERATURE | 100% | 100% |
| STOP | 97.2% | 94.4% | | BRIGHTNESS | 99.1% | 92.0% |
| NEXT | 96.5% | 87.5% | | COLOR | 96.0% | 85.6% |
| VOLUME_UP | 80.1% | 59.4% | | CREATE_REMINDER | 100% | 100% |
| VOLUME_DOWN | 86.5% | 71.0% | | OUT_OF_SCOPE | 69.7% | — |

The weakest real-speech classes have 12–26 real test clips each, mostly
SLURP and SNIPS free-form phrasings ("put some music on"). The most
common error is now a command taken for OUT_OF_SCOPE (VOLUME_UP 18 times,
COLOR 11), which the device answers with nothing rather than a wrong
action.

**Reject threshold** (real speech): at 0.6 the device says "didn't catch
that, please repeat" for 15.7% of commands and is right on 86.6% of the
ones it accepts (47% of its errors are caught).

**Progress, all on the current test set** (mean of 3 seeds; EXPERIMENTS.md
Experiments 38 and 43):

| Model | Params | val real | test all | test real | holdout |
|---|---:|---:|---:|---:|---:|
| BC-ResNet baseline, same size (38, first revision)\* | 89K | — | 81.29% | 41.57% | — |
| DS-CNN baseline, same size (38, first revision)\* | 100K | — | 89.50% | 57.06% | — |
| CRNN, Experiment 36 recipe (43t, 80 epochs) | 99K | 71.72% | 93.06% | 70.27% | 92.90% |
| + combo and distillation (43c) | 182K | 75.76% | 94.64% | 75.93% | 94.22% |
| + wider (43a) | 372K | 74.95% | 95.03% | 77.52% | 94.88% |
| + supplemental synthetic clips (43b, **shipped**) | 372K | **77.07%** | **95.27%** | **77.57%** | **95.38%** |
| Previous shipped model (41d, first revision)\* | 372K | — | 94.80% | 79.15% | 93.07% |

\*Trained on the first revision; none of the current test or holdout clips
were in its train split (checked by audio hash), so these are fair
scores. The previous shipped model is about level on intents but gets
31.6% of out-of-scope clips right, against 69.7% for 43b.

### Slot values (shipped model)

The value head's accuracy on test clips whose value is one of the
schema's (labels from the dataset). "Joint" also requires the intent to be
right, which is what the device needs.

| Slot | Values | Real: slot / joint | Synthetic: slot / joint | n (real / synthetic) |
|---|---:|---:|---:|---:|
| TIMER | 10 s, 30 s, 1 min | 100% / 79.0% | 100% / 99.8% | 19 / 404 |
| ALARM | 6:00 AM, 8:00 AM, 9:00 PM | 90.2% / 88.2% | 100% / 99.5% | 51 / 372 |
| TEMPERATURE | 18, 22, 26 degrees | 78.6% / 78.6% | 100% / 100% | 28 / 395 |
| BRIGHTNESS | 20, 60, 100 percent | 96.0% / 88.0% | 98.7% / 98.7% | 25 / 398 |
| COLOR | red, blue, green | 82.2% / 69.5% | 99.0% / 99.0% | 118 / 305 |
| CREATE_REMINDER | drink water, study, exercise | 88.9% / 88.9% | 99.0% / 99.0% | 18 / 405 |

Over all 2,538 slotted test clips the slot head is right 98.2% of the time.

### Out of scope (how it is handled and tested)

The class dataset labels speech that asks for none of the 19 commands
`OUT_OF_SCOPE`: noise, Filipino speech, near-miss requests and general
speech (270 train, 76 test and 16 holdout clips in the current revision,
including synthetic near-miss sentences). We keep it exactly as the
dataset defines it:

- **In training** it is the model's 20th class, "not a command". Bare
  numbers from the numerals set (also labeled OUT_OF_SCOPE by the dataset)
  add 1,500 examples. A model without this class would have to map every
  sound to one of the 19 commands.
- **On the device** nothing happens for it: an OUT_OF_SCOPE answer, or any
  answer below 0.6 confidence ("didn't catch that"), is never acted on.
  Before that, the wake word must hear "Hey Kiwi", which filters most
  non-command speech.
- **In testing** an out-of-scope clip is correct only if the model says
  OUT_OF_SCOPE. The headline 95.50% includes the 76 such test clips.
  Separately:

| Shipped model | Test | n | Holdout | n |
|---|---:|---:|---:|---:|
| Command accuracy, out-of-scope clips excluded | 95.95% (real 78.64%) | 4,367 | 98.39% | 186 |
| Out-of-scope clips labeled OUT_OF_SCOPE | 69.7% | 76 | 68.8% | 16 |
| **Out-of-scope clips the device ignores** (OUT_OF_SCOPE or confidence < 0.6) | **82.9%** | 76 | **93.8%** | 16 |
| Out-of-scope clips the device would act on (false accept) | 17.1% | 76 | 6.2% | 16 |
| Commands the device ignores (false reject, same rule) | 3.7% | 4,367 | 0.5% | 186 |

The small model ignores 72.4% of out-of-scope test clips (27.6% acted on).
Out of scope is still the weakest class: near-miss requests (commands
outside the schema) sound like commands.

### Wake word (`models/kiwi_wakeword.onnx`)

Evaluated the way the device runs it, streaming 1.5 s windows every 0.1 s:
missed wake words on 391 held-out clips in 20 unseen voices (clean and with
10 dB noise), the author's 10 held-out real recordings, and false wake-ups
while streaming the master dataset's whole test split plus 300 near-miss
phrases (2.94 hours). Experiment 43, seed 1 (best of 3 on validation):

| Threshold | Missed, clean | Missed, 10 dB noise | Author's real takes missed | False wake-ups per hour |
|---:|---:|---:|---:|---:|
| **0.6 (default)** | 3.8% | 7.2% | 0/10 | 12.9 |
| 0.7 | 4.6% | 8.4% | 0/10 | 9.5 |
| 0.85 | 8.2% | 12.5% | 0/10 | 7.2 |
| 0.95 | 13.8% | 23.3% | 0/10 | 2.0 |

The false-wake stream deliberately includes near-miss phrases ("hey kitty",
"every week"), so a real room sees fewer. On the same stream at 0.6 the
Experiment 42 detector (first revision) misses 4.1% / 4.9% with 12.3 false
wake-ups per hour: the two are about level. The master test split has no
pure-noise clips, so the 10 dB check mixes in the train split's 11 noise
clips.

### Known limits of the test set

- **Mostly synthetic.** 81% of test clips are the group's synthetic
  voices, which score ~99.5%. Report real speech next to the overall
  number.
- **Small real-speech counts for some classes.** MESSAGE, CALL, TIMER and
  CREATE_REMINDER have 12–19 real test clips each.
- **The master dataset has no "hey kiwi" recordings.** Wake-word misses
  are measured on our own held-out synthetic voices and recordings; its
  false wake-ups on the master test split.
- **Only 76 out-of-scope test clips** (16 in holdout), so out-of-scope
  rates move by ~1.3 points per clip.

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
([results/bench_pi5.md](../results/bench_pi5.md); measured with the
Experiment 41d weights, the same architecture as the shipped model). The previous model took
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
