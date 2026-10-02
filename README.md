# VCM: a tiny on-device Voice Command Model ("Hey Kiwi")

Machine exercise 2 (Voice Controlled Smart Device) for AI231 (ML Operations)
and AI222 (Supervised Learning), UP Diliman. Always-on keyword and intent
recognition at the edge, with no cloud round-trip.

Say **"Hey Kiwi"** and then a command, like "set a timer for 30 seconds" or
"set color to blue". A Raspberry Pi recognizes the command on the device.
It does not use speech-to-text, a cloud model or an LLM. The Pi then acts on
the command and answers out loud. A web dashboard shows the lamp,
thermostat, timers, alarms, reminders, music and call log.

**Data:** trained and tested only on the class's shared master dataset and
final Dataset Schema (19 commands, 6 with slot values), as the class agreed
on 2026-10-01. The project's earlier version, trained on its own 70k-clip
dataset (Experiments 1–36), is on the branch
[`archive/exp36-pre-me2-schema`](https://github.com/quielq/quielq-vcm/tree/archive/exp36-pre-me2-schema)
(tag `v1-exp36`).

## Submission summary

Every value the ME2 deck asks for. Details and sources are in the linked
docs.

### Model

```
Mic · 16 kHz → log-mel 40 × 501 (5.0 s, silence-trimmed) → CRNN encoder → intent head (20) + 6 slot heads (3 values each) → actuator (home server) on the Raspberry Pi
```

| Item | Value |
|---|---|
| Input | 16 kHz mono → 40-band log-mel, 10 ms hop, **40 × 501** frames (5.0 s window after silence trim) |
| Encoder | Conv 10×4 stride 2 + 4 depthwise-separable conv blocks (80 channels) → 2-layer bidirectional GRU, 96 per direction (192-dim frame features) → 4-head attention pooling. Not causal: the command is recognized once it ends (the wake word handles streaming) |
| Intent head | Linear → 20 classes (19 intents + `OUT_OF_SCOPE`) |
| Slot heads | 6 heads, each with its own attention pooling → 3 schema values (TIMER, ALARM, TEMPERATURE, BRIGHTNESS, COLOR, CREATE_REMINDER) |
| Actuator | `vcm.home` server: lamp, thermostat, timers, alarms, reminders, Spotify/local music, volume, weather, calls/messages via phone bridge; spoken reply |
| Parameters / weights | **0.372 M** parameters · **1.46 MB** fp32 ONNX (`models/vcm_intent.onnx`); the wake word adds 0.025 M · 0.11 MB (`models/kiwi_wakeword.onnx`) |
| Wake word ("Hey Kiwi") | Small CRNN: 32 channels, 1-layer GRU of 32, 25,475 params, 107 KB fp32, 5.3M multiply-adds per 1.5 s window, scored every 100 ms. Retrained with the master dataset as its "not the wake word" examples (Experiment 42); same architecture and size as before |

### Dataset

| Item | Value |
|---|---|
| Source | Class master dataset: [huggingface.co/datasets/airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) (group recordings, Xela's recordings, SLURP, FSC, SNIPS, Timers and Such, Common Voice, Speech Commands, group synthetic set) |
| Hours / utterances | train 6.34 h / 10,682 · test 2.57 h / 4,418 · holdout 0.17 h / 196 · numerals 21.8 h / 66,390 |
| Speakers (incl. synthetic voices) | train 315 · test 121 · holdout 5 · numerals 2,547. No speaker in two splits |
| Labels | **19 intents + OUT_OF_SCOPE · 6 slots** (3 values each, 18 in all), from the final Dataset Schema (Option B, 93 phrases) |

### Training on the A100 cluster

| Item | Value |
|---|---|
| Cluster | UP DGX `ai-n002`, **1 × A100-40GB** (GPU 6), shared node. ~34 min per seed with 6 runs sharing the GPU |
| Objective | Class-weighted cross-entropy on the intent + confusable-pair penalty (alpha 2.0) + slot cross-entropy (weight 0.3) + distillation (KL, T=3) toward an ensemble of 9 of our own CRNNs. No CTC |
| Optimizer | Adam, lr 1e-3, 5 warm-up epochs then cosine decay, batch 128 |
| Steps / loss | 80 epochs × 85 steps = **6,800 steps**; final train loss 0.64 (incl. distillation term), best val loss 0.50 at epoch 62 (seed 0) |
| Seeds | 3 (0, 1, 2); shipped seed 0 |

### Validation on the Raspberry Pi

| Item | Value |
|---|---|
| Keyword / intent acc | Wake word: 95.9% of held-out "hey kiwi" clips caught (95.4% with 10 dB noise, 10/10 of the author's held-out takes) · Intent: **92.98%** on the class test set (73.48% real speech), **94.90%** on the Pi holdout set |
| False-accept rate | Wake word: 11.2 false wake-ups per hour while streaming the master test split plus near-miss phrases, threshold 0.6 · Commands: 21.3% of out-of-scope test clips acted on (confidence ≥ 0.6), see [Out of scope](#out-of-scope) |
| Latency p95 / RTF | **15.8 ms / 0.0063** end to end per command on the Raspberry Pi 5 (p50 13.9 ms: features 3.8 + model 10.0); wake word 1.9 ms per 100 ms hop, 1.9% of one core. `vcm_intent_small.onnx`: 12.1 ms / 0.0048 ([results/bench_pi5.md](results/bench_pi5.md)) |
| Runtime | onnxruntime (CPU) · **1 thread** · Raspberry Pi 5 (8 GB): 100 MB peak RSS for the listener, 604 MB used system-wide with both services |

### To be submitted

| Item | Value |
|---|---|
| GitHub repository | [github.com/quielq/quielq-vcm](https://github.com/quielq/quielq-vcm), public, MIT ([LICENSE](LICENSE)) |
| Dataset location | Hugging Face `airimonda/ai231-me2-voice-commands`; each source keeps its own license (CC BY 4.0, CC0, FSC non-commercial academic, …, see the dataset card). DOI: not minted yet (the dataset owner can create one from the Hugging Face dataset settings or Zenodo) |
| A100 cluster | `ai-n002`, 1 × A100-40GB, ~34 min per seed, seeds 0/1/2 |
| Model weights | `models/vcm_intent.onnx`, `models/vcm_intent.pt` (this repo) · release to be created (GitHub release with the two files) · licence: MIT (code and weights); the training data's own terms apply to its use |

### Reviewer checklist

| # | Item | Status |
|---|---|---|
| 1 | Repo public, one-command reproduction | `bash scripts/reproduce.sh` (data → train → evaluate → ONNX → benchmark) |
| 2 | Dataset licensed and citable (DOI) | Licensed per source on the dataset card; DOI pending, to be minted by the dataset owner |
| 3 | Training logs + final checkpoint committed | [`results/`](results/) (logs, evaluations, launchers for Experiments 37–41) and `models/vcm_intent.pt` |
| 4 | Pi latency reproduced by the posted script | `python scripts/benchmark_pi.py --json bench_pi.json` on the Pi 5: 15.8 ms p95 ([results/bench_pi5.md](results/bench_pi5.md)) |
| 5 | Held-out test set, unseen speakers | Class-fixed test split (4,418 clips, 121 speakers) and holdout (196); no speaker or synthetic voice in two splits |
| 6 | Baseline of comparable size compared | DS-CNN 99.6K and BC-ResNet 89K params, same data and recipe ([Experiment 38](docs/EXPERIMENTS.md#experiment-38--comparable-size-baselines)) |

## What the percentages mean

All accuracy numbers measure **intent recognition**: did the model pick the
right command? The model never produces text, so there is no word error
rate.

- **Test set** = the class's fixed test split of the master dataset:
  4,418 clips, 47 per Option B variation plus 47 out-of-scope clips. 76% of
  it is the group's synthetic voices and 24% real people. No test speaker
  is in training.
- **Real speech** = the 1,003 test clips spoken by real people (group
  recordings, SLURP, FSC, SNIPS, …). It is the harder, more honest number:
  synthetic voices score ~99%.
- **Holdout** = 196 clips the class set aside for the live test on the
  Raspberry Pi (2 per variation).
- **Slot accuracy** is separate: for commands with a value, is the value
  right?
- We chose every setting on our **validation** split (carved from train,
  by speaker) and only report test and holdout.

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

**The shipped model is a CRNN** (convolutional recurrent neural network),
checkpoint `exp41d_combo_wide_distill_s0.pt` (Experiment 41d, seed 0), exported as
`models/vcm_intent.onnx`. Details are in [docs/MODEL.md](docs/MODEL.md).

**How we got there on the master dataset** (3 seeds each, mean; every
experiment is in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)):

| Step | Params | val real | test all | test real | holdout |
|---|---:|---:|---:|---:|---:|
| Experiment 36 recipe retrained on the master dataset (37a) | 99K | 64.0% | 90.1% | 63.6% | 92.4% |
| DS-CNN baseline, same size (38) | 100K | 51.5% | 86.9% | 52.2% | 82.7% |
| + 2-layer GRU, 4-head attention pooling, frequency-only SpecAugment, numerals as out-of-scope (40a) | 182K | 69.7% | 92.7% | 73.2% | 93.2% |
| + distillation from an ensemble of our own CRNNs (40b) | 182K | 70.4% | 92.4% | 71.7% | 94.6% |
| + wider: 80 channels, GRU 96 (**41d, shipped**) | 372K | **72.3%** | **92.9%** | **73.2%** | **93.9%** |

The shipped file is seed 0 of 41d, the best seed on val: **92.98%** on the
class test set, **73.48%** on its real speech, **94.90%** on the holdout
set. `models/vcm_intent_small.onnx` (40b seed 1, 722 KB) is the smaller
alternative: 92.21% / 70.99% / 95.41%.

**Size.** The shipped model is 1.46 MB fp32, above the 1 MB that the
earlier models kept to. It is still 0.37 M parameters and its latency is
within a millisecond of the small one (features dominate). Pick
`--intent-model models/vcm_intent_small.onnx` to stay under 1 MB.

**Compared with the old model.** The previous model (Experiment 36, old
70k-clip dataset, `models/vcm_intent_exp36.onnx`) trained on 2,439 of the
4,418 test clips, so its 92.4% on this test set is inflated. On the 1,979
test clips it never saw it scores 86.7% (65.8% real speech); our first
retrain on the master dataset already scored 89.2% there.

What helped and what didn't (Experiment 39, one change at a time):

- **Helped:** a 2-layer GRU; 4-head attention pooling; SpecAugment with
  frequency masks only; bare numbers from the numerals set as
  out-of-scope examples; and all of them together (+9.7 points on real
  speech).
- **Attention pooling is needed:** plain mean pooling loses 5.6 points of
  real speech.
- **Hurt:** numerals as background talk, an EMA of the weights. Label
  smoothing, a wider speed range and more epochs did nothing.
- **Comparable-size baselines:** DS-CNN (99.6K params) reaches 86.9% /
  52.2% real speech and BC-ResNet (89K) 78.8% / 37.9%, against 90.1% /
  63.6% for the CRNN of the same size (Experiment 38).

**Before the master dataset** we ran 36 experiments on our own data: the
CRNN beat DS-CNN and BC-ResNet by 11 points, and waveform augmentation
added 5. An ASR cascade (Whisper + text classifier) was more accurate but
~280× bigger and 150× slower ([docs/FOOTPRINT.md](docs/FOOTPRINT.md)), and
the assignment rules out ASR on the device.

## Dataset

The class master dataset, as published, with its own train / test /
holdout splits. We only add a validation split (12% of train, by speaker)
for choosing epochs and settings. Full description, statistics and build
steps: [docs/DATASET.md](docs/DATASET.md).

| Split | Clips | Real | Synthetic | Filipino voices | Speakers | Used for |
|---|---:|---:|---:|---:|---:|---|
| train | 9,273 | 2,494 | 6,779 | 737 | 271 | Training |
| val (ours) | 1,409 | 382 | 1,027 | 5 | 44 | Choosing epochs and settings |
| **test** | **4,418** | 1,050 | 3,368 | 203 | 121 | **Reported results** |
| holdout | 196 | 96 | 100 | 76 | 5 | Raspberry Pi live test; also scored offline |
| numerals | 66,390 | all | — | — | 2,547 | Bare numbers as out-of-scope examples (1,500 sampled) |

Slot values are exactly the schema's: 10 s / 30 s / 1 min, 6:00 AM /
8:00 AM / 9:00 PM, 18 / 22 / 26 degrees, 20 / 60 / 100 percent, red /
blue / green, drink water / study / exercise.

## Training

PyTorch on one A100 of UP's shared DGX. The final recipe:

- CRNN as above, 372K parameters, dropout 0.1.
- 80 epochs, batch size 128 (85 steps per epoch, 6,800 steps); Adam, learning
  rate 1e-3, 5 warm-up epochs, then cosine decay. Best epoch on val: 62.
- Loss: class-weighted cross-entropy, plus an extra penalty for confusing
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE and LIGHT_ON/LIGHT_OFF, plus slot
  cross-entropy at weight 0.3, plus distillation (KL at temperature 3,
  weight 1) toward the averaged predictions of 9 CRNNs from Experiment 37,
  trained on the same split.
- Augmentation: background noise, speed 0.9–1.1×, reverb and start shift
  on the waveform; 2 frequency masks on the log-mel (no time masks).
- 1,500 bare-number clips from the numerals set added as OUT_OF_SCOPE.
- 3 seeds, all on one A100; checkpoints and settings chosen on val.

[docs/TRAINING.md](docs/TRAINING.md) explains how we ran on the shared
node and gives the exact commands. `bash scripts/reproduce.sh` runs all of
it from a fresh clone.

## Test results

Final numbers for the shipped model. Tables for every seed and split are
in [docs/TESTING.md](docs/TESTING.md).

**Intent** (`models/vcm_intent.onnx`, identical to its checkpoint):

| | Accuracy | n |
|---|---:|---:|
| **Class test set, all clips** | **92.98%** | 4,418 |
| Real speech | 73.48% | 1,003 |
| Synthetic voices | 99.50% | 3,368 |
| Exact Option B wording (the demo phrases) | 99.06% | 3,601 |
| Same command in other words | 67.66% | 770 |
| Filipino group recordings | 87.30% | 189 |
| Macro average over 20 classes | 88.06% | |
| **Holdout (Pi live-test set)** | **94.90%** (real 94.19%) | 196 |

Strongest classes on real speech: TEMPERATURE and CREATE_REMINDER (100%),
ALARM 90%, TIMER 89%. Weakest: LIGHT_ON 47%, PLAY_MUSIC 49%, MESSAGE 56%,
most of them SLURP's free-form phrasings. OUT_OF_SCOPE is right 43% of
the time; only 21% of out-of-scope test clips would make the device act
(confidence ≥ 0.6), and in use the wake word filters most of them first.

**Slot values** (test, slot head right / intent and slot both right):
97.9% over all 2,538 slotted clips. Real speech: TIMER 89 / 79%, ALARM
80 / 75%, BRIGHTNESS 100 / 88%, COLOR 87 / 72%, TEMPERATURE 93 / 93%,
CREATE_REMINDER 78 / 78% (n = 18–118 each).

**Reject threshold.** At 0.6 confidence the device rejects 22% of real
commands ("please repeat") and is right on 84% of those it accepts.

**Latency on the Raspberry Pi 5** (`scripts/benchmark_pi.py`, 1 thread):
13.9 ms p50 / 15.8 ms p95 per command end to end, RTF 0.0063 at p95; the
wake word uses 1.9% of one core. The previous, smaller model took 9.9 ms;
`vcm_intent_small.onnx` takes 12.1 ms p95. Details:
[results/bench_pi5.md](results/bench_pi5.md).

### Out of scope

The dataset's `OUT_OF_SCOPE` clips (noise, Filipino speech, near-miss
requests, general speech) are kept as the dataset defines them: a 20th
class in training, never acted on by the device, and scored as correct
only when the model says OUT_OF_SCOPE. Reported separately on test:

| | Result | n |
|---|---:|---:|
| Commands only (out-of-scope clips excluded) | 93.53% (real speech 73.48%) | 4,371 |
| Out-of-scope clips the device ignores (OUT_OF_SCOPE or confidence < 0.6) | 78.7% | 47 |
| Out-of-scope clips it would act on (false accept) | 21.3% | 47 |
| Commands it ignores (false reject, same rule) | 5.5% | 4,371 |

The wake word filters most non-command speech before the intent model
hears it. Details: [TESTING.md](docs/TESTING.md#out-of-scope-how-it-is-handled-and-tested).

### Wake word

Retrained in Experiment 42 with the master dataset as its negatives (its
train split's commands, out-of-scope speech and noise, plus 3,000 numerals
clips). The "hey kiwi" positives are not in any class dataset: they are
~3,000 synthetic clips and the author's own recordings, as before. On the
master test split streamed as one 3.05 h recording with near-miss phrases:

| Threshold | Missed, clean | Missed, 10 dB noise | Author's real takes missed | False wake-ups per hour |
|---:|---:|---:|---:|---:|
| **0.6 (default)** | 4.1% | 4.6% | 0/10 | 11.2 |
| 0.7 | 4.1% | 6.6% | 0/10 | 8.5 |
| 0.85 | 7.9% | 11.3% | 0/10 | 4.3 |
| 0.95 | 17.9% | 24.0% | 1/10 | 1.6 |

Same architecture and size as the previous detector (25,475 parameters,
107 KB); only the training data changed. The previous wake word
(Experiment 34, `models/kiwi_wakeword_exp34.onnx`)
fires 19.7 times per hour on the same stream at 0.6 and misses 3.1% / 6.6%.
The device uses 0.6, and 0.4 while music plays.

**Automated tests.** 244 unit tests run without any hardware
(`python -m pytest`).

## Course concepts applied

| Concept | How we used it | Where |
|---|---|---|
| Supervised learning, train/val/test splits | The class fixes train/test/holdout by speaker; we carve val from train by speaker and choose every setting on val only. | [DATASET.md](docs/DATASET.md), [TESTING.md](docs/TESTING.md) |
| Datasets and dataloaders | The Hugging Face dataset is converted to one manifest format; PyTorch `DataLoader` workers compute features on the fly. Fixed seeds. | `scripts/build_me2_manifest.py`, `src/vcm/train/dataset.py` |
| Input normalization | Log-mel features scaled to roughly zero mean and unit variance. | `src/vcm/audio/features.py` |
| CNN | Depthwise-separable convolutions; strided blocks act like pooling and widen the receptive field. | `CRNN` in `src/vcm/train/architectures.py` |
| RNN | A 2-layer bidirectional GRU reads the command frame by frame, so word order matters. | same |
| Attention | Multi-head attention pooling weights the frames that matter; removing it costs 5.6 points of real speech. | same, [MODEL.md](docs/MODEL.md) |
| Optimization | Adam, warm-up and cosine decay, 3 seeds per setting. | `src/vcm/train/train.py` |
| Regularization and augmentation | Waveform noise, speed, reverb and shift; frequency-only SpecAugment (time masks can erase the one word that matters); dropout 0.1. EMA and label smoothing tested, not kept. | [EXPERIMENTS.md](docs/EXPERIMENTS.md) Exp 39 |
| Class imbalance | Inverse-frequency class weights; extra out-of-scope examples from the numerals set. | Exp 39d |
| Knowledge distillation | An ensemble of our own CRNNs (same data) teaches a single CRNN. | Exp 40–41 |
| Evaluation | Overall, real-speech, per-class, per-accent and per-source accuracy; confusion pairs; slot accuracy; reject threshold. | `scripts/evaluate_checkpoint.py` |
| Efficiency | Accuracy against parameters, file size, latency and memory; comparable-size baselines. | [FOOTPRINT.md](docs/FOOTPRINT.md), Exp 38 |
| Model packaging | PyTorch → ONNX with labels and slot values in its metadata, then ONNX Runtime on the Pi's CPU. | `src/vcm/deploy/`, `scripts/export_onnx.py` |

## Codebase structure

```
.
├── README.md                  ← you are here
├── LICENSE                    MIT
├── pyproject.toml             package and dependencies (extras: dev, train, deploy, rpi)
├── requirements-pi.txt        the only packages the Raspberry Pi needs
├── configs/
│   └── settings.example.toml  template for API keys and devices (copy to settings.toml)
├── models/                    the trained models
│   ├── vcm_intent.onnx          ★ shipped intent + slot model (ONNX, runs on the Pi)
│   ├── vcm_intent.pt            ★ its PyTorch checkpoint
│   ├── kiwi_wakeword.onnx       ★ shipped wake word (Experiment 42), and kiwi_wakeword.pt
│   ├── vcm_intent_small.onnx    smaller intent model (Experiment 40b, 722 KB)
│   ├── vcm_intent_exp36.onnx, kiwi_wakeword_exp34.onnx   the previous models (old dataset), for comparison
│   └── vcm_intent_frozen.onnx   older variant, not used
├── results/                   training logs, evaluation outputs and launchers, Experiments 37–41
├── src/vcm/                   the Python package
│   ├── audio/                 microphone capture, log-mel features, numpy DSP, resampling
│   ├── wakeword/              "Hey Kiwi" streaming detector
│   ├── deploy/                ONNX export and ONNX Runtime inference
│   ├── slots.py               slot-value vocabularies and parsers
│   ├── home/                  home server: actions, scheduler, dashboard, Spotify, phone bridge client
│   ├── actions/               weather API, optional Xiaomi bulb, local music (used by home/)
│   ├── tts/                   spoken replies (Piper, espeak-ng, macOS say)
│   ├── hal/                   hardware layer: GPIO button, Sense HAT temperature
│   ├── config.py              settings loading, platform detection
│   ├── dataset/               manifest format, schema, loaders for the old sources
│   └── train/                 architectures, training loop, losses, augmentation
├── scripts/                   command-line tools (see the table below)
├── tests/                     unit tests (pytest), no hardware needed
├── deploy/                    systemd services and audio-device rules for the Pi
├── data/dataset_schema/       the final class schema (CSV); other data is downloaded, not in git
└── docs/                      all documentation (see below)
```

| Stage | Scripts |
|---|---|
| **Device** (Pi or laptop) | `vcm_listen.py` (the voice loop), `kiwi_doctor.py` (preflight check), `benchmark_pi.py` (latency p95 / RTF), `measure_footprint.py` |
| **Laptop** | `deploy_pi.sh` (deploy to the Pi), `mac_phone_bridge.py`, `spotify_auth.py`, `record_wakeword.py` |
| **Data** (DGX) | `build_me2_manifest.py` (master dataset → manifest, slot labels, metadata) |
| **Train and evaluate** (DGX) | `reproduce.sh` (all steps), `python -m vcm.train.train`, `generate_ensemble_labels.py` (distillation teacher), `evaluate_checkpoint.py`, `summarize_experiments.py`, `export_onnx.py`, `train_wakeword.py`, `evaluate_wakeword.py` |
| **Old dataset** (Experiments 1–36) | `build_manifest.py`, `fetch_*.py`, `process_fsc.py`, `qa_filter_option_b.py`, `generate_targeted_synthetic.py`, `build_slot_labels.py`, cascade scripts; see the archive branch |

## Quick start

**Run the tests** on any machine. No hardware is needed.
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train,deploy]"
python -m pytest
```

**Reproduce the model** on a GPU machine (downloads ~3 GB):
```bash
bash scripts/reproduce.sh
```

**Try it on a laptop** with its built-in microphone.
```bash
cp configs/settings.example.toml configs/settings.toml   # add a weather key if you like
python -m vcm.home.server                                # terminal 1: actions + dashboard at http://127.0.0.1:8000
python scripts/vcm_listen.py --server http://127.0.0.1:8000 --show-scores   # terminal 2
```
Then say "Hey Kiwi, what time is it".

## Deploy and run on the Raspberry Pi

This is the short version. The full setup from a blank SD card is in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Daily use and the demo checklist
are in [docs/RUNBOOK.md](docs/RUNBOOK.md).

### What this assumes is already set up

- **The Pi.** A Raspberry Pi 4 or 5 running **64-bit** Raspberry Pi OS,
  with SSH turned on. We use the hostname `raspberrypi`, so it is reachable
  as `raspberrypi.local`.
- **The laptop.** A Mac or Linux laptop with this repo cloned. The commands
  below run from the repo folder.
- **SSH without a password.** `ssh raspberrypi.local` logs in with a key.
  If it asks for a password, run `ssh-copy-id raspberrypi.local` once.
- **The same network.** The laptop and the Pi are on the same Wi-Fi. Venue
  Wi-Fi often blocks this, so a phone hotspot works as a backup.
- **Audio.** A USB microphone and a USB speaker are plugged into the Pi's
  black USB 2 ports.
- **Settings.** `configs/settings.toml` exists on the laptop. Copy it from
  `configs/settings.example.toml`. The weather key, Spotify and phone
  sections are optional. Without them, those commands say they are not set
  up and everything else still works.

### Commands

**1. Deploy from the laptop.** This installs the system packages, copies the
code, the models and your settings, builds the Python environment, runs a
benchmark, and installs two services that start at boot.
```bash
scripts/deploy_pi.sh raspberrypi.local --services
```

**2. Check that everything is ready.** The last line should say READY.
```bash
ssh raspberrypi.local 'cd ~/quielq-vcm && .venv/bin/python scripts/kiwi_doctor.py --beep'
```
If the microphone shows as silent right after a deploy, restart the audio
stack and the services, then run the check again:
```bash
ssh raspberrypi.local 'systemctl --user restart pipewire pipewire-pulse wireplumber && sleep 3 && systemctl --user restart vcm-home vcm'
```

**3. Measure latency on the Pi** (for the deck: p95 and RTF, 1 thread). Copy
the holdout clips over first, or leave out `--clips` to use a synthetic
command.
```bash
ssh raspberrypi.local 'cd ~/quielq-vcm && .venv/bin/python scripts/benchmark_pi.py --json bench_pi.json'
```

**4. Open the dashboard** at http://raspberrypi.local:8000. On Android, use
the Pi's IP address instead (`ssh raspberrypi.local hostname -I`).

**5. Talk to it.** Say "Hey Kiwi", then a command from the schema, like
"what time is it" or "start a timer for 30 seconds". The dashboard's
"Simulate a command" box runs the same actions without speaking.

**6. Calls and messages (optional).** These go through your Mac and iPhone.
Keep this running on the Mac:
```bash
.venv/bin/python scripts/mac_phone_bridge.py --token <bridge_token from settings.toml>
```

### Everyday commands

| To | Run |
|---|---|
| Watch what Kiwi hears | `ssh raspberrypi.local journalctl --user -u vcm -f` |
| Restart both services | `ssh raspberrypi.local systemctl --user restart vcm-home vcm` |
| Stop both services | `ssh raspberrypi.local systemctl --user stop vcm vcm-home` |
| Update the Pi after a change | `git pull`, then `scripts/deploy_pi.sh raspberrypi.local --services` |
| Push changed settings | `scripts/deploy_pi.sh raspberrypi.local --settings` |

## Documentation

| Document | What's in it |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | The assignment, system design, key decisions, hardware |
| [MODEL.md](docs/MODEL.md) | The shipped models: features, architecture, training recipe, wake word, export |
| [DATASET.md](docs/DATASET.md) | The class master dataset and how we use it; the old dataset as history |
| [TRAINING.md](docs/TRAINING.md) | Training on the shared DGX, and commands to reproduce the shipped models |
| [EXPERIMENTS.md](docs/EXPERIMENTS.md) | All experiments: 1–36 on the old dataset, 37 on on the master dataset |
| [TESTING.md](docs/TESTING.md) | Final test results, evaluation method, the automated test suite |
| [FOOTPRINT.md](docs/FOOTPRINT.md) | SD card and RAM use on the Pi, and our pipeline vs. the ASR cascade |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Setting up a Raspberry Pi from a blank SD card |
| [RUNBOOK.md](docs/RUNBOOK.md) | Starting, testing and demoing the system, and troubleshooting |
| [reports/](docs/reports/) | Detailed reports for Experiments 32 to 36 (old dataset) |
| [archive/AUDIT.md](docs/archive/AUDIT.md) | Everything we considered and did not ship, and why |

## Future enhancements

- **More real speech.** Real clips are 27% of train; synthetic voices
  score ~99% and real speakers 73.48%. Commands phrased in the
  speaker's own words are the main source of errors.
- **Out-of-scope rejection.** OUT_OF_SCOPE has the fewest train clips
  (187); the device also rejects low-confidence commands.
- Measure on a Pi Zero 2 W (512 MB), the smallest board that should fit.
- Add a login or token to the home server and dashboard.
- Run the test suite in CI on every pull request.

## Acknowledgments

The dataset is a class effort. Ailene (`airimonda`) collated the master
dataset and its documentation. Mark Macalacad shared the Dataset Schema
(Option B) and generated the group's synthetic set. Xela Ubalde shared her
recordings. Anthony Navarez built the transcribe-and-compare QA tool our
earlier synthetic-audio screening was based on. Every classmate who
recorded commands made the real-speech test possible. Public datasets in
the master dataset: SLURP, Fluent Speech Commands, SNIPS SLU, Timers and
Such, Common Voice, Google Speech Commands, MLEnd and the Multi-Sensor
Voice Command dataset.

## License

Code and model weights: [MIT](LICENSE). The dataset is not in this
repository; each of its sources keeps its own license (see the dataset
card).
