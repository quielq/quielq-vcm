# VCM: a tiny on-device Voice Command Model ("Hey Kiwi")

Machine exercise for AI222 (Supervised Learning) and AI231 (ML Operations),
UP Diliman.

Say **"Hey Kiwi"** and a command ("set a timer for five minutes", "turn the
lights blue"). A Raspberry Pi 5 recognizes it **entirely on the device**,
with no speech-to-text, no cloud model and no LLM, then acts on it and
answers out loud. A web dashboard shows the lamp, thermostat, timers,
alarms, reminders, music and call log.

| | |
|---|---|
| **Models on the device** | 539 KB total: intent + slot model (432 KB, 108K parameters) and wake word (107 KB, 25K parameters), fp32 ONNX |
| **Commands** | 19 intents covering all 10 required categories, plus values for timers, alarms, brightness, color, temperature and reminders |
| **Accuracy** | **84.8%** on 6,577 real-speech test clips; slot values 74–100% |
| **Speed on the Pi 5** | **9.9 ms** per command; the always-on wake word uses 2% of one CPU core |
| **Memory on the Pi 5** | 103 MB for the voice pipeline; ~300 MB for everything we run |

## Contents

- [How it works](#how-it-works)
- [The final model, and how we chose it](#the-final-model-and-how-we-chose-it)
- [Dataset](#dataset)
- [Training](#training)
- [Test results](#test-results)
- [Codebase structure](#codebase-structure)
- [Quick start](#quick-start)
- [Documentation](#documentation)
- [Future enhancements](#future-enhancements)
- [Acknowledgments](#acknowledgments)

## How it works

```
microphone → "Hey Kiwi" wake word ─→ record the command ─→ log-mel features ─→ intent + slot model
             (every 100 ms)            (until 0.6 s quiet)    (numpy)             (ONNX Runtime)
                                                                                      │
             spoken reply + dashboard ←── home server acts on it ←── confidence ≥ 0.6 ┘
```

Two services run on the Pi: `scripts/vcm_listen.py` (listening and
recognition) and `vcm.home.server` (actions, spoken replies and the
dashboard). Only the actions use the internet (weather, Spotify, and calls
relayed through a Mac to an iPhone); recognition never does. Full design:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## The final model, and how we chose it

**Shipped:** a **CRNN** (convolutional + recurrent network) that reads a
40-band log-mel spectrogram of the command and outputs one of 20 classes
(19 intents + background noise), plus six **slot heads** that pick the value
("5 minutes", "blue", "7:00 AM") from a fixed list. It is checkpoint
`exp36_joint_w03_s1.pt` (Experiment 36, seed 1), exported as
`models/vcm_intent.onnx`. Details: [docs/MODEL.md](docs/MODEL.md).

**How we got there** (36 experiments, all logged in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)):

| Step | Real-speech accuracy | Lesson |
|---|---:|---|
| DS-CNN and BC-ResNet, the standard keyword-spotting models (Exp 1–27) | ~70% | DS-CNN sees only ~240 ms at a time, shorter than one word, so it confused "volume **up**" / "volume **down**". Loss tweaks, rebalancing and regularization barely helped. |
| **CRNN**: strided convolutions + GRU + attention pooling (Exp 28) | 80.8% | Reading the whole command in order was the biggest single gain (+11 points). |
| + waveform augmentation: noise, speed, reverb (Exp 29b) | 85.7% | Closed most of the gap between synthetic and real voices. |
| + targeted synthetic phrasings (Exp 31) | 85.3% | Fixed phrasings that failed in live tests ("pause the song", "timer for 5 minutes"). |
| + slot-value heads, trained jointly at weight 0.3 (Exp 32–36) | **84.8%** | Adds values for 6 intents for ~0.6 points of intent accuracy. |

**Why not the more accurate ASR cascade?** Commercial assistants transcribe
speech first, then classify the text. We built that too (Whisper + a text
classifier, Experiment 26), and it scored **90.6%**. We measured what it
costs to run ([docs/FOOTPRINT.md](docs/FOOTPRINT.md)):

| | **Ours (shipped)** | ASR cascade | Cost of the cascade |
|---|---:|---:|---:|
| Model files | **539 KB** | ~149 MB | ~280× |
| Peak memory | **88–103 MB** | 478–696 MB | 5–7× |
| Latency per command (laptop) | **~3 ms** | 440–950 ms | 150–300× |
| Can run as the always-on wake word | **Yes** (2% of a Pi core) | No: needs a separate wake word anyway | |
| Real-speech accuracy | 84.8% | **90.6%** | +5.8 points |

About 6 points of accuracy for 280× the size and 150× the latency, plus a
second model to listen for the wake word, breaks the assignment's "tiny" and
"no ASR on the device" requirements. So the CRNN ships, and the cascade stays
as the accuracy reference.

Other choices that went into the final version: **fp32 instead of int8**
(int8 lost 6.8 points to save 134 KB), **numpy features instead of librosa**
on the device (peak memory ~330 MB → ~90 MB), and **"Hey Kiwi"** as the wake
word (fewest sound-alikes among 11 candidates). Everything we tried and
dropped is in [docs/archive/AUDIT.md](docs/archive/AUDIT.md).

## Dataset

**70,641 labelled clips, 64% real speech**, in 20 classes. Built from
public datasets, a synthetic dataset shared by the class, and our own
QA-screened synthetic clips. Full reproduction steps:
[docs/DATASET.md](docs/DATASET.md).

| Source | Clips | Type | Covers |
|---|---:|---|---|
| Fluent Speech Commands | 24,223 | Real | Lights, volume, temperature, music, pause, stop |
| SLURP | 17,452 | Real | Weather, time, music, messages, alarms, lights, reminders |
| Snips SLU (lighting) | 2,472 | Real | Lights, brightness, color |
| Timers and Such | 1,071 | Real | Timers, alarms |
| Google Speech Commands background noise | 600 | Real noise | `unknown_background` |
| Option B (class-shared, voice-cloned, QA-filtered) | 16,500 | Synthetic | All 19 intents |
| Our targeted and slot-value clips (Chatterbox TTS, Whisper-checked) | 8,323 | Synthetic | Weak phrasings and every slot value |

Every source's labels were mapped from its real transcripts, not its
documentation, and audited (for example, SLURP "lists" turned out to be
shopping lists, not reminders, and were dropped). CALL, NEXT and
LIST_REMINDERS have no public real recordings, so they are synthetic only.

## Training

PyTorch on UP's shared DGX (A100 GPUs). Final recipe: 80 epochs, Adam with
warm-up and cosine decay, class-weighted cross-entropy plus a penalty for
confusing one-word pairs (up/down, on/off), slot loss at weight 0.3, and
waveform augmentation. 3 seeds per configuration; models are selected on
validation and compared on the **real-speech test** split.
[docs/TRAINING.md](docs/TRAINING.md) covers running on the shared node and
the exact commands to reproduce the shipped models.

## Test results

Final numbers for the shipped models; full tables and method in
[docs/TESTING.md](docs/TESTING.md).

**Intent** (test split, real speech, n = 6,577): **84.84%** accuracy, 83.27%
macro average. TEMPERATURE reaches 99%; the weakest classes are COLOR 49%,
CREATE_REMINDER 59% and BRIGHTNESS 69%, where SLURP's free-form phrasing
dominates.

**Slot values** (slot head correct):

| TIMER | ALARM | BRIGHTNESS | COLOR | TEMPERATURE\* | CREATE_REMINDER\* |
|---:|---:|---:|---:|---:|---:|
| 73.9% | 98.1% | 74.9% | 86.7% | 100% | 100% |

\*Measured on synthetic speech only; no real recording has these values.

**Wake word** at the default threshold 0.6: misses 3.1% of held-out "hey
kiwi" clips (5.6% with background noise), none of the author's real
recordings, and fires 12.7 times per hour on a stream built from
deliberately confusing phrases.

**Automated tests:** 237 unit tests, no hardware needed (`python -m pytest`).

## Codebase structure

```
.
├── README.md                  ← you are here
├── pyproject.toml             package + dependencies (extras: dev, train, deploy, rpi)
├── requirements-pi.txt        the only packages the Raspberry Pi needs
├── configs/
│   └── settings.example.toml  template for API keys and devices (copy to settings.toml)
├── models/                    the trained models (ONNX)
│   ├── vcm_intent.onnx          ★ shipped intent + slot model
│   ├── kiwi_wakeword.onnx       ★ shipped wake word
│   ├── vcm_intent_frozen.onnx   earlier variant (Exp 34), for comparison
│   └── kiwi_wakeword.int8.onnx  int8 variant, not used
├── src/vcm/                   the Python package
│   ├── audio/                 microphone capture, log-mel features, numpy DSP, resampling
│   ├── wakeword/              "Hey Kiwi" streaming detector
│   ├── deploy/                ONNX export and ONNX Runtime inference
│   ├── slots.py               slot-value vocabularies and parsers
│   ├── home/                  home server: actions, scheduler, dashboard, Spotify, phone bridge client
│   ├── actions/               weather API, optional Xiaomi bulb, local music (used by home/)
│   ├── tts/                   spoken replies (Piper, espeak-ng, macOS say)
│   ├── hal/                   hardware abstraction: GPIO button, Sense HAT temperature
│   ├── config.py              settings loading, platform detection
│   ├── dataset/               dataset loaders per source, manifest format, synthetic-audio QA
│   └── train/                 architectures, training loop, losses, augmentation
├── scripts/                   command-line tools (see below)
├── tests/                     237 unit tests (pytest), no hardware needed
├── deploy/                    systemd services and audio-device rules for the Pi
├── data/dataset_schema/       the class's 19-intent taxonomy (other data is downloaded, not in git)
└── docs/                      all documentation (see below)
    ├── reports/               detailed reports for Experiments 32–36
    └── archive/               audit of dropped options, original plans, legacy code
```

`scripts/`, by where they run:

| Stage | Scripts |
|---|---|
| **Device** (Pi or laptop) | `vcm_listen.py` (the voice loop), `kiwi_doctor.py` (preflight check), `benchmark_pi.py`, `measure_footprint.py` |
| **Laptop** | `deploy_pi.sh` (deploy to the Pi), `mac_phone_bridge.py` (calls/messages via iPhone), `spotify_auth.py`, `record_wakeword.py` |
| **Build the dataset** (DGX) | `slurp_coverage.py`, `fetch_slurp_audio.py`, `refilter_slurp_manifest.py`, `process_fsc.py`, `fetch_snips_lights.py`, `resplit_snips_by_speaker.py`, `fetch_gsc_background.py`, `fetch_timers_and_such.py`, `qa_filter_option_b.py`, `generate_targeted_synthetic.py`, `add_targeted_synth_to_manifest.py`, `build_manifest.py`, `build_slot_labels.py` |
| **Train and evaluate** (DGX) | `python -m vcm.train.train`, `train_wakeword.py`, `evaluate_checkpoint.py`, `evaluate_wakeword.py`, `export_onnx.py` |
| **ASR cascade and analysis** (DGX) | `transcribe_corpus_for_cascade.py` (Whisper transcripts, also used for slot labels), `train_cascade_classifier.py`, `generate_distillation_labels.py` (Exp 27), `wakeword_confusability.py` |

## Quick start

**Run the tests** (any machine, no hardware):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train,deploy]"
python -m pytest
```

**Try it on a laptop** with its built-in microphone:
```bash
cp configs/settings.example.toml configs/settings.toml   # add a weather key etc. if you like
python -m vcm.home.server                                # terminal 1: actions + dashboard at http://127.0.0.1:8000
python scripts/vcm_listen.py --server http://127.0.0.1:8000 --show-scores   # terminal 2
```
Then say "Hey Kiwi, what time is it".

**Deploy to a Raspberry Pi** (64-bit Raspberry Pi OS, reachable over SSH):
```bash
scripts/deploy_pi.sh raspberrypi.local --services
```
Step-by-step: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Day-to-day use and
demo checklist: [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Documentation

| Document | What's in it |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | The assignment, system design, key decisions, hardware |
| [MODEL.md](docs/MODEL.md) | The shipped models: features, architecture, training recipe, wake word, export |
| [DATASET.md](docs/DATASET.md) | Every data source, label mapping and quality fix, with reproduction steps |
| [TRAINING.md](docs/TRAINING.md) | Training on the shared DGX; commands to reproduce the shipped models |
| [EXPERIMENTS.md](docs/EXPERIMENTS.md) | All 36 experiments with results and log paths |
| [TESTING.md](docs/TESTING.md) | Final test results, evaluation method, the automated test suite |
| [FOOTPRINT.md](docs/FOOTPRINT.md) | SD card and RAM use on the Pi; ours vs. the ASR cascade |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Setting up a Raspberry Pi from a blank SD card |
| [RUNBOOK.md](docs/RUNBOOK.md) | Starting, testing and demoing the system; troubleshooting |
| [reports/](docs/reports/) | Detailed reports for Experiments 32–36 |
| [archive/AUDIT.md](docs/archive/AUDIT.md) | Everything considered and not shipped, and why |

## Future enhancements

**Accuracy**
- Raise the weakest real-speech classes (COLOR 49%, CREATE_REMINDER 59%,
  BRIGHTNESS 69%, down from 75% in Experiment 29b) with a second round of
  targeted phrasings modelled on SLURP's free-form wording.
- Real recordings from classmates for CALL, NEXT and LIST_REMINDERS, which
  have no real training or test audio (tool built, waiting on the adviser's
  approval).
- A label audit: review clips where the ASR cascade confidently disagrees
  with the label.
- A larger CRNN (~200–300K parameters, still under 1.5 MB) over 3 seeds.
- Confirm with the adviser that attention pooling is allowed; otherwise test
  plain average pooling.

**Features**
- Contacts and message text for CALL / MESSAGE (a contact slot); today both
  go to one default contact.
- Free-text reminders with a due time, and repeating alarms.
- "Play <song>" (needs a song slot); STOP currently maps to pause on Spotify.
- Send COLOR to the real Xiaomi bulb (on/off and brightness already work).
- Calls without the Mac: Bluetooth hands-free on the Pi, or a phone
  notification / iOS Shortcut.
- The dashboard as an installable phone web app.

**Engineering**
- Add a login or token to the home server and dashboard (today anyone on the
  same network can use them).
- Split `pyproject.toml` so a plain install is the minimal device runtime and
  training/desktop libraries (librosa, pynput, python-miio) are extras.
- Measure on a Pi Zero 2 W (512 MB), the smallest board that should fit.
- Run the test suite in CI on every pull request.

## Acknowledgments

The dataset is a class effort. Mark Macalacad shared the working taxonomy
and generated the synthetic "Option B" dataset; Anthony Navarez built the
transcribe-and-compare QA tool our synthetic-audio screening generalizes.
Other classmates contributed dataset leads and corrections. Details in
[docs/DATASET.md](docs/DATASET.md#acknowledgments). Public datasets: SLURP,
Fluent Speech Commands, Snips SLU, Timers and Such, Google Speech Commands.
