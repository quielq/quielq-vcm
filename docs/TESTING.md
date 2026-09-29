# Testing and results

Three kinds of testing, from most to least formal:

| | What it checks | Where |
|---|---|---|
| [1. Model evaluation](#1-model-evaluation-final-results) | Accuracy of the shipped models on held-out test data | DGX (needs the datasets) |
| [2. On-device and live tests](#2-on-device-and-live-tests) | Speed and memory on the Pi; real commands spoken to the device | Raspberry Pi |
| [3. Automated test suite](#3-automated-test-suite) | Code correctness (features, data loaders, losses, runtime, home server) | Anywhere, no hardware |

## 1. Model evaluation: final results

### How the test set is defined

- **Test split only.** Each real dataset keeps its own published
  train/val/test split; synthetic clips are split by cloned voice.
- **Real speech only** for the headline number: synthetic clips score 94–99%
  and would hide real-speech weaknesses. `unknown_background` (noise) is
  excluded because it isn't speech.
- **Snips lighting excluded.** Its original split put the same speakers in
  train and test. It was re-split by speaker in Experiment 32, and earlier
  models had seen some of the new test speakers, so leaving it out keeps
  the comparison with older experiments fair. Scores with it are given too.
- **Model selection on validation, reporting on test.** Checkpoints are
  chosen by validation accuracy; seeds and configurations are compared on
  test.

That leaves **6,577 real-speech test clips** from SLURP, Fluent Speech
Commands and Timers and Such, across 16 intents.

### Intent accuracy (shipped model, `models/vcm_intent.onnx`)

| Measure | Result |
|---|---:|
| **Real-speech test accuracy** | **84.84%** (n = 6,577) |
| Real speech, macro average (mean of per-class accuracy, 16 classes) | 83.27% |
| Real speech, including Snips lighting | 83.82% |
| Seeds of the same recipe (0 / 1 / 2) | 84.86 / **84.84** / 84.46% |
| ASR cascade on the same clips, for reference (not shippable) | 90.62% |

The exported ONNX file and its PyTorch checkpoint give identical real-speech
results, so the numbers above are for the file that runs on the Pi.

**Progress over the project** (real-speech test accuracy, best seed):

| Model | Real speech | What changed |
|---|---:|---|
| DS-CNN, Experiments 25 / 27 | 70.6% / 71.6% | Baseline |
| CRNN, Experiment 28 | 80.8% | Architecture: sees the whole command in order |
| CRNN + waveform augmentation, Experiment 29b | 85.7% | Noise, speed, reverb |
| + targeted synthetic phrasings, Experiment 31 | 85.3% | Fixed live-test phrasings (98–99% on held-out targeted clips) |
| + 6 slot-value heads, Experiment 36 (**shipped**) | **84.8%**\* | Timer, alarm, brightness, color, temperature, reminder values |

\*Experiments 25–31 were scored before the Snips speaker re-split, with
Snips included; Experiment 36 without it. On Experiment 36's basis,
Experiment 31 scores 85.5%, so adding the slot heads cost about 0.6 points.

**Per class.** Measured for the shipped model on real speech: TEMPERATURE
99.4%, BRIGHTNESS 69.1%, CREATE_REMINDER 59.1%, COLOR 49.3%
([reports/exp36_report.md](reports/exp36_report.md)). The last full
per-class table (EXPERIMENTS.md Experiment 29b, same architecture) has
PAUSE and STOP at ~100%, TEMPERATURE 99%, TIMER 97% and LIGHT_ON 95%, with
the same weakest classes. Those classes are dominated by SLURP's free-form
phrasing ("olly brighten the lights", "do i need a coat") and share
vocabulary with each other ("set the lights to…").

**Reject threshold.** Below 0.6 confidence the device says "didn't catch
that, please repeat" instead of acting. On validation (measured on the
Experiment 34 model) this rejects ~11% of commands and raises accuracy on the accepted ones from 86% to 92%.

### Slot values (shipped model)

Accuracy of the predicted value on test clips whose value is known from the
script (synthetic clips and scripted real recordings), Snips excluded:

| Slot | Values | Slot accuracy | Intent and slot both right | n |
|---|---:|---:|---:|---:|
| ALARM (time) | 28 | 98.1% | 97.8% | 321 |
| COLOR | 14 | 86.7% | 85.5% | 337 |
| BRIGHTNESS (percent) | 12 | 74.9% | 74.9% | 255 |
| TIMER (duration) | 24 | 73.9% | 73.2% | 395 |
| TEMPERATURE | 3 | 100% | 100% | 267 |
| CREATE_REMINDER (task) | 3 | 100% | 99.2% | 261 |

TEMPERATURE and CREATE_REMINDER values are measured on synthetic speech
only: no real recording in the datasets says those values.

### Wake word (`models/kiwi_wakeword.onnx`)

Evaluated the way the device runs it, streaming 1.5 s windows every 0.1 s:
missed wake words on 391 held-out clips in 20 unseen voices (clean and with
10 dB noise), the author's 10 held-out real recordings, and false wake-ups
over 7.48 hours of test-split command speech.

| Threshold | Missed, clean | Missed, 10 dB noise | Author's real takes missed | False wake-ups per hour |
|---:|---:|---:|---:|---:|
| **0.6 (default)** | 3.1% | 5.6% | 0/10 | 12.7 |
| 0.7 | 3.6% | 8.2% | 0/10 | 9.2 |
| 0.85 | 6.9% | 11.5% | 0/10 | 4.3 |
| 0.95 | 14.1% | 21.5% | 0/10 | 1.3 |

The false-wake stream deliberately includes near-miss phrases ("hey kitty",
"every week"), so a real room sees fewer.

### Known limits of the test set

- **CALL, NEXT and LIST_REMINDERS have no real-speech test clips.** No
  public dataset covers them, so their scores (94–100%) are on synthetic
  speech only. Live tests found CALL the least reliable command.
- No real-speech dataset has Filipino-accented English. It appears only in
  16 of Option B's 100 voice-cloned reference speakers and in live tests.

### Reproduce

On the DGX, with the data in `data/` ([DATASET.md](DATASET.md)):
```bash
python scripts/evaluate_checkpoint.py models/vcm_intent.onnx \
  --manifest data/dataset_manifest_exp36.csv --slot-labels data/slot_labels_exp36.csv \
  --exclude-source snips_lights
python scripts/evaluate_wakeword.py checkpoints/kiwi_wakeword_v2_s0.pt \
  --extra-wake-manifest data/wakeword_real/manifest.csv
```
`evaluate_checkpoint.py` accepts `.pt` checkpoints or `.onnx` files and
prints real-speech, per-class, confusable-group, slot and reject-threshold
tables.

## 2. On-device and live tests

**Speed and memory on the Raspberry Pi 5** (`scripts/benchmark_pi.py`, no
microphone needed): 9.9 ms per command, the wake word uses 2% of one core,
94 MB peak. Details in [FOOTPRINT.md](FOOTPRINT.md).

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

237 tests, all pure logic: no microphone, speaker, network, GPU or dataset
needed. They use synthetic arrays, small fixture files and mocked hardware.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"      # train adds PyTorch, needed by the training tests
python -m pytest                   # expect: 237 passed
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
