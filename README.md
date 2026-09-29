# VCM: a tiny on-device Voice Command Model ("Hey Kiwi")

Machine exercise for AI222 (Supervised Learning) and AI231 (ML Operations),
UP Diliman.

Say **"Hey Kiwi"** and then a command, like "set a timer for five minutes"
or "turn the lights blue". A Raspberry Pi 5 recognizes the command on the
device. It does not use speech-to-text, a cloud model or an LLM. The Pi then
acts on the command and answers out loud. A web dashboard shows the lamp,
thermostat, timers, alarms, reminders, music and call log.

| | |
|---|---|
| **Models on the device** | 539 KB in total. Intent + slot model: 432 KB, 108K parameters. Wake word: 107 KB, 25K parameters. Both fp32 ONNX. |
| **Compute** | 54M multiply-adds per command. The wake word uses 5.3M per 1.5 s window, 10 times a second. |
| **Commands** | 19 intents that cover all 10 required categories. Six of them also return a value (timer length, alarm time, brightness, color, temperature, reminder). |
| **Accuracy** | **84.8%** on 6,577 real-speech test clips. Slot values are 74% to 100% correct. |
| **Speed on the Pi 5** | **9.9 ms** per command. The always-on wake word uses 2% of one CPU core. |
| **Memory on the Pi 5** | 103 MB for the voice pipeline. About 300 MB for everything we run. |

## Contents

- [How it works](#how-it-works)
- [The final model, and how we chose it](#the-final-model-and-how-we-chose-it)
- [Dataset](#dataset)
- [Training](#training)
- [Test results](#test-results)
- [Course concepts applied](#course-concepts-applied)
- [Codebase structure](#codebase-structure)
- [Quick start](#quick-start)
- [Documentation](#documentation)
- [Future enhancements](#future-enhancements)
- [Acknowledgments](#acknowledgments)

## How it works

```
microphone → "Hey Kiwi" wake word → record the command → log-mel features → intent + slot model
             (every 100 ms)          (until 0.6 s quiet)   (numpy)            (ONNX Runtime)
                                                                                    │
             spoken reply + dashboard ← home server acts on it ← confidence ≥ 0.6 ──┘
```

Two services run on the Pi. `scripts/vcm_listen.py` listens and recognizes
the command. `vcm.home.server` runs the action, speaks the reply and serves
the dashboard. Recognition runs offline. The internet is used only by some
actions: weather, Spotify, and calls that go through a Mac to an iPhone.
The full design is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## The final model, and how we chose it

**The shipped model is a CRNN** (convolutional recurrent neural network). It
reads a 40-band log-mel spectrogram of the command. It outputs one of 20
classes: 19 intents and background noise. Six **slot heads** also pick the
value, like "5 minutes", "blue" or "7:00 AM", from a fixed list. The model
is checkpoint `exp36_joint_w03_s1.pt` (Experiment 36, seed 1), exported as
`models/vcm_intent.onnx`. Details are in [docs/MODEL.md](docs/MODEL.md).

**How we got there.** We ran 36 experiments. All of them are logged in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

| Step | Real-speech accuracy | What we learned |
|---|---:|---|
| DS-CNN and BC-ResNet, the usual keyword-spotting models (Exp 1 to 27) | ~70% | DS-CNN sees only about 240 ms at a time. That is shorter than one word, so it mixed up "volume up" and "volume down". Loss changes, rebalancing and regularization barely helped. |
| **CRNN**: strided convolutions, a GRU and attention pooling (Exp 28) | 80.8% | Reading the whole command in order gave the biggest single gain, +11 points. |
| + waveform augmentation: noise, speed and reverb (Exp 29b) | 85.7% | This closed most of the gap between synthetic and real voices. |
| + targeted synthetic phrasings (Exp 31) | 85.3% | This fixed phrasings that failed in live tests, like "pause the song". |
| + slot-value heads, trained together with the intent at weight 0.3 (Exp 32 to 36) | **84.8%** | The model now returns values for 6 intents. It cost about 0.6 points of intent accuracy. |

**Why we didn't ship the more accurate ASR cascade.** Commercial assistants
transcribe speech first and then classify the text. We built that too, with
Whisper and a text classifier (Experiment 26). It scored **90.6%**. Then we
measured what it costs to run ([docs/FOOTPRINT.md](docs/FOOTPRINT.md)):

| | **Ours (shipped)** | ASR cascade | Cost of the cascade |
|---|---:|---:|---:|
| Model files | **539 KB** | ~149 MB | ~280x |
| Peak memory | **88 to 103 MB** | 478 to 696 MB | 5 to 7x |
| Latency per command (laptop) | **~3 ms** | 440 to 950 ms | 150 to 300x |
| Can run as the always-on wake word | **Yes** (2% of a Pi core) | No. It would need a separate wake word. | |
| Real-speech accuracy | 84.8% | **90.6%** | +5.8 points |

The cascade gains about 6 points of accuracy. In exchange it is 280 times
bigger and 150 times slower, and it still needs a second model for the wake
word. The assignment asks for a tiny model with no ASR on the device. So we
ship the CRNN and keep the cascade as the accuracy reference.

Other decisions in the final version:
- **fp32 instead of int8.** int8 lost 6.8 points of accuracy to save 134 KB.
- **numpy features instead of librosa on the device.** Peak memory dropped
  from about 330 MB to about 90 MB.
- **"Hey Kiwi" as the wake word.** It had the fewest sound-alikes among 11
  candidates.

Everything we tried and dropped is in
[docs/archive/AUDIT.md](docs/archive/AUDIT.md).

## Dataset

The training data has **70,641 labelled clips in 20 classes**. 64% of them
are real speech. The clips come from public datasets, a synthetic dataset
shared by the class, and our own synthetic clips. We screened every
synthetic clip with Whisper. The steps to rebuild it are in
[docs/DATASET.md](docs/DATASET.md).

| Source | Clips | Type | Covers |
|---|---:|---|---|
| Fluent Speech Commands | 24,223 | Real | Lights, volume, temperature, music, pause, stop |
| SLURP | 17,452 | Real | Weather, time, music, messages, alarms, lights, reminders |
| Snips SLU (lighting) | 2,472 | Real | Lights, brightness, color |
| Timers and Such | 1,071 | Real | Timers, alarms |
| Google Speech Commands background noise | 600 | Real noise | `unknown_background` |
| Option B (class-shared, voice-cloned, QA-filtered) | 16,500 | Synthetic | All 19 intents |
| Our targeted and slot-value clips (Chatterbox TTS) | 8,323 | Synthetic | Weak phrasings and every slot value |

We mapped each source's labels from its real transcripts and audited them.
For example, SLURP's "lists" intent turned out to be shopping lists, so we
dropped it from LIST_REMINDERS. CALL, NEXT and LIST_REMINDERS have no public
real recordings. Their data is synthetic only.

## Training

We trained with PyTorch on UP's shared DGX (A100 GPUs). The final recipe:

- 80 epochs, batch size 128.
- Adam optimizer, learning rate 1e-3, 5 warm-up epochs, then cosine decay.
- Class-weighted cross-entropy for class imbalance.
- An extra penalty for confusing one-word pairs (up/down, on/off).
- Slot loss at weight 0.3.
- Waveform augmentation (background noise, speed change, reverb, time shift).

We trained 3 seeds per configuration. We picked checkpoints on the
validation split and compared them on the **real-speech test** split.
[docs/TRAINING.md](docs/TRAINING.md) explains how we ran on the shared node.
It also has the exact commands to reproduce the shipped models.

## Test results

These are the final numbers for the shipped models. The full tables and
method are in [docs/TESTING.md](docs/TESTING.md).

**Intent.** On the real-speech test split (6,577 clips), accuracy is
**84.84%** and the macro average is 83.27%. TEMPERATURE reaches 99%. The
weakest classes are COLOR (49%), CREATE_REMINDER (59%) and BRIGHTNESS
(69%). Most of their real test clips come from SLURP, where people phrase
commands freely.

**Slot values.** How often the slot head picks the right value:

| TIMER | ALARM | BRIGHTNESS | COLOR | TEMPERATURE\* | CREATE_REMINDER\* |
|---:|---:|---:|---:|---:|---:|
| 73.9% | 98.1% | 74.9% | 86.7% | 100% | 100% |

\*Measured on synthetic speech only. No real recording in the datasets says
these values.

**Wake word.** At the default threshold of 0.6, it misses 3.1% of held-out
"hey kiwi" clips and 5.6% with background noise. It caught all 10 of the
author's held-out recordings. It fires 12.7 times per hour on a test stream
full of deliberately confusing phrases.

**Automated tests.** 237 unit tests run without any hardware
(`python -m pytest`).

## Course concepts applied

Where the main deep learning concepts from the course show up in this
project:

| Concept | How we used it | Where |
|---|---|---|
| Supervised learning, train/val/test splits | Validation only picks the checkpoint. The test split is used for the final comparison. We split synthetic voices by speaker and re-split Snips by speaker to avoid leakage. | [TESTING.md](docs/TESTING.md), [DATASET.md](docs/DATASET.md) |
| Datasets and dataloaders | Six sources merged into one manifest format. PyTorch `DataLoader` workers compute features on the fly. Fixed seeds for repeatable runs. | `src/vcm/dataset/`, `src/vcm/train/dataset.py` |
| Input normalization | Log-mel features are scaled to roughly zero mean and unit variance, using statistics from 2,000 training clips. | `src/vcm/audio/features.py` |
| CNN | Depthwise-separable convolutions share weights across time and frequency. Strided blocks act like pooling and widen the receptive field. | `CRNN` in `src/vcm/train/architectures.py` |
| RNN | A bidirectional GRU reads the command frame by frame, so word order matters. | same |
| Attention | Attention pooling weights the frames that matter most, like the word "up". It is a single linear scorer over time, much smaller than a Transformer's self-attention. | same, [MODEL.md](docs/MODEL.md) |
| Optimization | Adam, learning-rate warm-up and cosine decay, 3 seeds per setting. | `src/vcm/train/train.py` |
| Regularization | Data augmentation (waveform noise, speed, reverb, shift) gave the largest gain. The CRNN uses dropout of 0.1. Weight decay and label smoothing were tested and not kept. SpecAugment masking hurt because it can erase the one word that matters. | [EXPERIMENTS.md](docs/EXPERIMENTS.md) Exp 2, 8, 24, 29b |
| Class imbalance | Inverse-frequency class weights. Focal loss and per-class caps were tested and dropped. | Exp 18 to 23 |
| Evaluation | Overall and per-class accuracy, confusion pairs, macro average, and a reject threshold for low confidence. | `scripts/evaluate_checkpoint.py` |
| Efficiency | We compared models by accuracy against parameters, FLOPs, latency, memory and file size. | [FOOTPRINT.md](docs/FOOTPRINT.md) |
| Model packaging | PyTorch → ONNX export, checked against the checkpoint, then ONNX Runtime on the Pi's CPU. | `src/vcm/deploy/`, `scripts/export_onnx.py` |
| Knowledge distillation | The ASR cascade taught the small DS-CNN (Exp 27). It helped some classes and hurt others, so we dropped it. | Exp 27 |
| Quantization | int8 was tested. It lost 6.8 points of accuracy, so we ship fp32. | Exp 32 |

## Codebase structure

```
.
├── README.md                  ← you are here
├── pyproject.toml             package and dependencies (extras: dev, train, deploy, rpi)
├── requirements-pi.txt        the only packages the Raspberry Pi needs
├── configs/
│   └── settings.example.toml  template for API keys and devices (copy to settings.toml)
├── models/                    the trained models (ONNX)
│   ├── vcm_intent.onnx          ★ shipped intent + slot model
│   ├── kiwi_wakeword.onnx       ★ shipped wake word
│   ├── vcm_intent_frozen.onnx   earlier variant (Exp 34), kept for comparison
│   └── kiwi_wakeword.int8.onnx  int8 variant, not used
├── src/vcm/                   the Python package
│   ├── audio/                 microphone capture, log-mel features, numpy DSP, resampling
│   ├── wakeword/              "Hey Kiwi" streaming detector
│   ├── deploy/                ONNX export and ONNX Runtime inference
│   ├── slots.py               slot-value lists and parsers
│   ├── home/                  home server: actions, scheduler, dashboard, Spotify, phone bridge client
│   ├── actions/               weather API, optional Xiaomi bulb, local music (used by home/)
│   ├── tts/                   spoken replies (Piper, espeak-ng, macOS say)
│   ├── hal/                   hardware layer: GPIO button, Sense HAT temperature
│   ├── config.py              settings loading, platform detection
│   ├── dataset/               loaders for each data source, manifest format, synthetic-audio QA
│   └── train/                 architectures, training loop, losses, augmentation
├── scripts/                   command-line tools (see the table below)
├── tests/                     237 unit tests (pytest), no hardware needed
├── deploy/                    systemd services and audio-device rules for the Pi
├── data/dataset_schema/       the class's 19-intent taxonomy (other data is downloaded, not in git)
└── docs/                      all documentation (see below)
    ├── reports/               detailed reports for Experiments 32 to 36
    └── archive/               audit of dropped options, original plans, legacy code
```

The scripts, grouped by where they run:

| Stage | Scripts |
|---|---|
| **Device** (Pi or laptop) | `vcm_listen.py` (the voice loop), `kiwi_doctor.py` (preflight check), `benchmark_pi.py`, `measure_footprint.py` |
| **Laptop** | `deploy_pi.sh` (deploy to the Pi), `mac_phone_bridge.py` (calls and messages through the iPhone), `spotify_auth.py`, `record_wakeword.py` |
| **Build the dataset** (DGX) | `slurp_coverage.py`, `fetch_slurp_audio.py`, `refilter_slurp_manifest.py`, `process_fsc.py`, `fetch_snips_lights.py`, `resplit_snips_by_speaker.py`, `fetch_gsc_background.py`, `fetch_timers_and_such.py`, `qa_filter_option_b.py`, `generate_targeted_synthetic.py`, `add_targeted_synth_to_manifest.py`, `build_manifest.py`, `build_slot_labels.py` |
| **Train and evaluate** (DGX) | `python -m vcm.train.train`, `train_wakeword.py`, `evaluate_checkpoint.py`, `evaluate_wakeword.py`, `export_onnx.py` |
| **ASR cascade and analysis** (DGX) | `transcribe_corpus_for_cascade.py` (Whisper transcripts, also used for slot labels), `train_cascade_classifier.py`, `generate_distillation_labels.py` (Exp 27), `wakeword_confusability.py` |

## Quick start

**Run the tests** on any machine. No hardware is needed.
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train,deploy]"
python -m pytest
```

**Try it on a laptop** with its built-in microphone.
```bash
cp configs/settings.example.toml configs/settings.toml   # add a weather key if you like
python -m vcm.home.server                                # terminal 1: actions + dashboard at http://127.0.0.1:8000
python scripts/vcm_listen.py --server http://127.0.0.1:8000 --show-scores   # terminal 2
```
Then say "Hey Kiwi, what time is it".

**Deploy to a Raspberry Pi** running 64-bit Raspberry Pi OS, reachable over SSH.
```bash
scripts/deploy_pi.sh raspberrypi.local --services
```
The step-by-step setup is in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Daily
use and the demo checklist are in [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Documentation

| Document | What's in it |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | The assignment, system design, key decisions, hardware |
| [MODEL.md](docs/MODEL.md) | The shipped models: features, architecture, training recipe, wake word, export |
| [DATASET.md](docs/DATASET.md) | Every data source, label mapping and quality fix, with steps to rebuild |
| [TRAINING.md](docs/TRAINING.md) | Training on the shared DGX, and commands to reproduce the shipped models |
| [EXPERIMENTS.md](docs/EXPERIMENTS.md) | All 36 experiments with results and log paths |
| [TESTING.md](docs/TESTING.md) | Final test results, evaluation method, the automated test suite |
| [FOOTPRINT.md](docs/FOOTPRINT.md) | SD card and RAM use on the Pi, and our pipeline vs. the ASR cascade |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Setting up a Raspberry Pi from a blank SD card |
| [RUNBOOK.md](docs/RUNBOOK.md) | Starting, testing and demoing the system, and troubleshooting |
| [reports/](docs/reports/) | Detailed reports for Experiments 32 to 36 |
| [archive/AUDIT.md](docs/archive/AUDIT.md) | Everything we considered and did not ship, and why |

## Future enhancements

**Accuracy**
- Improve the weakest real-speech classes with a second round of targeted
  phrasings based on SLURP's wording. COLOR is at 49% and CREATE_REMINDER at
  59%. BRIGHTNESS is at 69%, down from 75% in Experiment 29b.
- Record classmates saying CALL, NEXT and LIST_REMINDERS. These have no real
  training or test audio. The recording tool is built and waits on the
  adviser's approval.
- Audit labels where the ASR cascade confidently disagrees with the label.
- Try a larger CRNN with 200K to 300K parameters (still under 1.5 MB), over
  3 seeds.
- Ask the adviser if attention pooling is allowed. If not, test plain
  average pooling.

**Features**
- Add contacts and message text for CALL and MESSAGE. Today both go to one
  default contact.
- Add free-text reminders with a due time, and repeating alarms.
- Add "play <song>", which needs a song slot. On Spotify, STOP currently
  pauses.
- Send COLOR to the real Xiaomi bulb. On/off and brightness already work.
- Make calls without the Mac, through Bluetooth hands-free on the Pi or a
  phone notification.
- Make the dashboard an installable phone web app.

**Engineering**
- Add a login or token to the home server and dashboard. Today anyone on
  the same network can use them.
- Split `pyproject.toml` so a plain install is the small device runtime.
  Training and desktop libraries (librosa, pynput, python-miio) would become
  extras.
- Measure on a Pi Zero 2 W (512 MB), the smallest board that should fit.
- Run the test suite in CI on every pull request.

## Acknowledgments

The dataset is a class effort. Mark Macalacad shared the working taxonomy
and generated the synthetic "Option B" dataset. Anthony Navarez built the
transcribe-and-compare QA tool that our synthetic-audio screening is based
on. Other classmates shared dataset leads and corrections. The details are
in [docs/DATASET.md](docs/DATASET.md#acknowledgments). Public datasets used:
SLURP, Fluent Speech Commands, Snips SLU, Timers and Such, and Google Speech
Commands.
