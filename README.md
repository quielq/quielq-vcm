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
on 2026-10-01. The shipped model is **Experiment 43b** (seed 1); every
number in this README is for it unless stated otherwise.

## Submission summary

Every value the ME2 deck asks for, with a short account of what each
component does and why it was chosen. The evidence behind each choice is
in the linked docs.

### Model

The task is framed as **closed-set spoken language understanding (SLU)**,
not speech recognition. The model maps an utterance directly to one of 20
intent labels and, for six intents, a slot value, without producing a
transcript. Giving up the open vocabulary of an ASR → text-classifier
cascade makes the model about 200 times smaller than Whisper `base` alone,
small enough to run on the Raspberry Pi's CPU. Recognition has two stages.
A small streaming wake-word detector runs continuously and gates a larger,
non-causal command classifier, so the larger model runs once per utterance
and only after a likely wake event.

```
Mic · 16 kHz → log-mel 40 × 501 (5.0 s, silence-trimmed) → CRNN encoder → intent head (20) + 6 slot heads (3 values each) → actuator (home server) on the Raspberry Pi
```

| Item | Value |
|---|---|
| Input representation | 16 kHz mono audio, trimmed of leading and trailing silence (30 dB below peak) and padded or cut to 5.0 s, then a 40-band **log-mel spectrogram** (25 ms window, 10 ms hop): **40 × 501**. The mel scale follows the ear's coarser resolution at high frequencies, and the log compresses loudness differences, giving a compact time–frequency image. Features are standardized with a fixed mean and variance from the training set. Trimming silence makes training clips look like live recordings, which start right after the wake word |
| Encoder: convolutional front end | A 10×4 convolution with stride 2, then 4 **depthwise-separable** blocks (80 channels; blocks 2 and 4 stride by 2). Each block splits a 3×3 convolution into a per-channel spatial filter and a 1×1 channel mixer (as in MobileNet), using about 8 times fewer parameters. The strides grow the receptive field geometrically and shorten the time axis to 63 steps of ~80 ms. A 1×1 projection then folds the remaining frequency axis into a 96-dim vector per step |
| Encoder: recurrent layers | **2-layer bidirectional GRU**, 96 units per direction (192-dim per step). The gated recurrence models word order, so commands that differ by one word ("volume up" / "volume down") separate. Running in both directions gives every step context from the whole utterance. This is the main difference from purely convolutional keyword spotters (DS-CNN, BC-ResNet), which average over time and lose order. The encoder is therefore **non-causal**: it classifies once the command ends, and the wake word handles streaming |
| Temporal pooling | **4-head attention pooling.** Each head learns a relevance score for every time step, normalizes the scores with a softmax over time, and returns the weighted mean of the steps; the 4 results are concatenated (768-dim). Unlike global average pooling, this lets the few frames carrying the deciding word dominate the summary. Replacing it with mean pooling costs 5.6 points of real-speech validation accuracy |
| Intent head | Dropout 0.1, then a linear layer → **20 classes** (19 intents + `OUT_OF_SCOPE`). The explicit reject class, together with a 0.6 softmax-confidence threshold at run time, lets the closed-set classifier decline speech that is not a command |
| Slot heads | **Multi-task learning:** 6 heads (TIMER, ALARM, TEMPERATURE, BRIGHTNESS, COLOR, CREATE_REMINDER) share the encoder. Each has its own single-head attention pooling and a 3-way classifier over the schema's values. Separate pooling lets each head attend to where the value is said ("thirty seconds", "blue") instead of the words that identify the intent. Slots are classified over a fixed vocabulary rather than tagged as spans, and a value is used only when its intent is predicted |
| Actuator | `vcm.home` server: lamp, thermostat, timers, alarms, reminders, Spotify/local music, volume, weather, calls/messages via phone bridge; spoken reply |
| Parameters / weights | **0.372 M** parameters · **1.46 MB** fp32 ONNX (`models/vcm_intent.onnx`) · 91.7 M multiply-adds per command; three quarters of the parameters are in the GRU. Kept at fp32 because dynamic int8 quantization cost 6.8 points of real-speech accuracy when tested on an earlier CRNN, and was no faster. The wake word adds 0.025 M · 0.11 MB (`models/kiwi_wakeword.onnx`) |
| Wake word ("Hey Kiwi") | The same CRNN design at a smaller width: 32 channels, a 1-layer bidirectional GRU of 32, **25,475 parameters**, 107 KB fp32, 5.3 M multiply-adds per 1.5 s window. It scores a sliding window every 100 ms as a binary classifier. Near-miss phrases ("hey kitty") are hard negatives; the master dataset's commands, out-of-scope speech, noise and numerals are the background negatives (Experiment 43, seed 1) |

### Dataset

The class master dataset combines real speech, recorded by the class or
taken from public SLU and keyword-spotting corpora, with synthetic
(voice-cloned) readings of the schema's 93 phrasings. Its splits are
**speaker-disjoint**, so test accuracy measures how well the model
generalizes to voices it has never heard, not how well it remembers
speakers. Synthetic voices are 81% of the test set and score ~99.5%, so
we also report **real-speech accuracy** on its own as the more realistic
estimate. We pick every setting on a validation split carved from train
by speaker; test and holdout are only used for reporting.

| Item | Value |
|---|---|
| Source | Class master dataset, revision `da92a79` (2026-10-02): [huggingface.co/datasets/airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands). Real speech: class recordings, SLURP, Fluent Speech Commands, SNIPS, Timers and Such, Common Voice; synthetic: the group synthetic set; noise and numerals: Speech Commands, MLEnd |
| Hours / utterances | train 6.10 h / 10,733 · test 2.45 h / 4,443 · holdout 0.18 h / 202 · numerals 21.8 h / 66,390 (not command data: 1,500 clips sampled as out-of-scope examples for the intent model) · supplemental synthetic 2.73 h / 5,856 (3,461 of train voices used for training) |
| Speakers (incl. synthetic voices) | train 377 · test 144 · holdout 7 · numerals 2,547. No speaker in two splits |
| Labels | **19 intents + OUT_OF_SCOPE · 6 slots** (3 values each, 18 in all), from the final Dataset Schema (Option B, 93 phrases). The classes are imbalanced (OUT_OF_SCOPE has 251 train clips), which the class-weighted loss corrects for |

### Training on the A100 cluster

The final model is trained with a **composite objective**. Each term
addresses a specific weakness found during development:

```
L = CE_w(intent) + α · p(confusable classes) + 0.3 · Σ CE(slot) + T² · KL(teacher_T ‖ student_T)
```

- **Class-weighted cross-entropy** (inverse class frequency) keeps rare
  classes such as OUT_OF_SCOPE from being ignored.
- The **confusable-pair penalty** (α = 2.0) adds the probability the model
  puts on classes known to be confused with the true one
  (VOLUME_UP / VOLUME_DOWN / TEMPERATURE, LIGHT_ON / LIGHT_OFF). This
  pushes the decision boundary between them further apart.
- **Slot cross-entropy** (weight 0.3) trains the slot heads on clips that
  carry a schema value. It shares the encoder with the intent task and
  acts as an auxiliary signal for it.
- **Knowledge distillation** (Hinton et al., 2015): the student matches
  the temperature-softened (T = 3) averaged predictions of an ensemble of
  9 of our own CRNNs trained on the same split. The soft targets carry how
  similar the classes are to each other, and pass most of the ensemble's
  advantage (95.0% vs 93.2% on val) to a single model at no extra cost on
  the device. There is no CTC or ASR term; the teacher is not an ASR model.

| Item | Value |
|---|---|
| Cluster | UP DGX `ai-n002`, **1 × A100-40GB** (GPU 6), shared node. Training the final model takes **~54 min** (80 epochs at ~40 s each on average, with 8 other runs and another user's job sharing the GPU) |
| Objective | Class-weighted cross-entropy + confusable-pair penalty (α 2.0) + slot cross-entropy (weight 0.3) + distillation (KL, T = 3, weight 1), as above |
| Optimizer | Adam, lr 1e-3, batch 128. A 5-epoch linear warm-up keeps the first updates small while Adam's moment estimates and BatchNorm statistics settle, then cosine decay anneals the learning rate towards zero |
| Regularization | Waveform augmentation (background noise at 5–25 dB SNR, speed 0.9–1.1×, room reverb, 0–0.3 s shift), **SpecAugment with frequency masks only** (time masks can erase the single word that decides the class), dropout 0.1 |
| Steps / loss | 80 epochs × 112 steps = **8,960 steps** (14,246 training clips: the train split, 3,461 supplemental synthetic clips of train voices and 1,500 numerals as out-of-scope); final train loss 0.52 (incl. distillation term), val loss 0.29 at the chosen epoch 65. The checkpoint is chosen by validation accuracy |
| Seeds | The final configuration was trained with 3 seeds (0, 1, 2) to measure run-to-run variance (up to 2 points on real speech); seed 1, the best on val, is shipped |

### Validation on the Raspberry Pi

**Intent accuracy** is top-1 classification accuracy; there is no
transcript, so no word error rate. **False accepts** are measured at both
stages: wake-word triggers per hour on a long stream that contains no
"hey kiwi", and the share of out-of-scope clips the command model would
act on. **Latency** is reported at the 95th percentile, which bounds the
delay a user actually notices better than the mean. The **real-time
factor** (RTF) is processing time divided by audio duration: 0.0061 on the
Raspberry Pi 5 means a command is processed about 160 times faster than
real time.

| Item | Value |
|---|---|
| Keyword / intent acc | Wake word: 96.2% of held-out "hey kiwi" clips caught (92.8% with 10 dB noise, 10/10 of the author's held-out takes) · Intent: **95.50%** on the class test set (78.64% real speech), **96.04%** on the Pi holdout set |
| False-accept rate | Wake word: 12.9 false wake-ups per hour while streaming the master test split plus near-miss phrases, threshold 0.6 · Commands: 17.1% of out-of-scope test clips acted on (confidence ≥ 0.6), see [Out of scope](#out-of-scope) |
| Latency p95 / RTF | **15.2 ms / 0.0061** end to end per command on the Raspberry Pi 5 (p50 13.9 ms: features 3.8 + model 10.1); wake word 2.0 ms per 100 ms hop at p95, 1.9% of one core. Measured with the shipped Experiment 43b weights. `vcm_intent_small.onnx`: ~12 ms / 0.0048 ([results/bench_pi5.md](results/bench_pi5.md)) |
| Runtime | onnxruntime (CPU) · **1 thread** · Raspberry Pi 5 (8 GB): 99 MB peak RSS for the listener, 719 MB used system-wide with both services |

### To be submitted

| Item | Value |
|---|---|
| GitHub repository | [github.com/quielq/quielq-vcm](https://github.com/quielq/quielq-vcm), public, MIT ([LICENSE](LICENSE)) |
| Dataset location | Hugging Face `airimonda/ai231-me2-voice-commands`, revision `da92a79` (also on the DGX at `/data/ai231`, verified identical clip for clip by `scripts/verify_shared_dataset.py`); each source keeps its own license (CC BY 4.0, CC0, FSC non-commercial academic, …, see the dataset card). DOI: [10.57967/hf/10723](https://doi.org/10.57967/hf/10723) |
| A100 cluster | `ai-n002`, 1 × A100-40GB; ~54 min to train the final model |
| Model weights | `models/vcm_intent.onnx`, `models/vcm_intent.pt` (this repo) · release to be created (GitHub release with the two files) · licence: MIT (code and weights); the training data's own terms apply to its use |

### Reviewer checklist

| # | Item | Status |
|---|---|---|
| 1 | Repo public, one-command reproduction | `bash scripts/reproduce.sh` (data → train → evaluate → ONNX → benchmark); on the DGX it reads the class's shared copy `/data/ai231` after checking it against the committed dataset fingerprint |
| 2 | Dataset licensed and citable (DOI) | Licensed per source on the dataset card; DOI [10.57967/hf/10723](https://doi.org/10.57967/hf/10723) |
| 3 | Training logs + final checkpoint committed | [`results/`](results/) (training logs, evaluations and launchers; the final model's are `exp43b_supplemental_s*`) and `models/vcm_intent.pt` |
| 4 | Pi latency reproduced by the posted script | `python scripts/benchmark_pi.py --json bench_pi.json` on the Pi 5: 15.2 ms p95 with the shipped weights ([results/bench_pi5.md](results/bench_pi5.md)) |
| 5 | Held-out test set, unseen speakers | Class-fixed test split (4,443 clips, 144 speakers) and holdout (202); no speaker or synthetic voice in two splits |
| 6 | Baseline of comparable size compared | DS-CNN 99.6K and BC-ResNet 89K params, same data and recipe as a 99K CRNN, on the test set: 89.5% / 81.3% against 92.4% for the CRNN (57.1% / 41.6% against 69.7% on real speech) ([MODEL.md](docs/MODEL.md#4-baselines)) |

## What the percentages mean

All accuracy numbers measure **intent recognition**: did the model pick the
right command? The model never produces text, so there is no word error
rate.

- **Test set** = the class's fixed test split of the master dataset
  (revision `da92a79`): 4,443 clips, 47 per Option B variation plus 76
  out-of-scope clips. 81% of it is the group's synthetic voices. No test
  speaker is in training.
- **Real speech** = the 777 test clips of real people saying a command
  (class recordings, SLURP, FSC, SNIPS, …). It is the harder, more honest
  number: synthetic voices score ~99%.
- **Holdout** = 202 clips the class set aside for the live test on the
  Raspberry Pi.
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

## The final model

**The shipped model is a CRNN** (convolutional recurrent neural network):
Experiment 43b, seed 1, checkpoint `exp43b_supplemental_s1.pt`, exported
as `models/vcm_intent.onnx` (also `models/vcm_intent.pt`). The wake word
is `models/kiwi_wakeword.onnx` (Experiment 43, seed 1). Details are in
[docs/MODEL.md](docs/MODEL.md).

| Model file | Params | Size | test all | test real | holdout |
|---|---:|---:|---:|---:|---:|
| **`models/vcm_intent.onnx`** (43b seed 1, the default) | 372K | 1.46 MB | **95.50%** | **78.64%** | **96.04%** |
| `models/vcm_intent_small.onnx` (43c seed 0, narrower, optional) | 182K | 722 KB | 94.53% | 76.06% | 93.07% |

**How it was chosen.** Every setting was chosen on our validation split,
never on test: the recipe that scored best on val (mean of 3 seeds), then
the best of its 3 seeds on val. The 3 seeds of the final recipe score
95.00 / 95.50 / 95.32% on test. Same-size DS-CNN and BC-ResNet baselines
trained the same way score 89.5% and 81.3%
([MODEL.md](docs/MODEL.md#4-baselines)).

**Size.** The shipped model is 1.46 MB fp32 (1.57 MB with the wake word),
above a 1 MB budget. It is still 0.37 M parameters and runs in 15 ms on the
Pi. To stay under 1 MB, pass `--intent-model models/vcm_intent_small.onnx`
to the listener; it costs 1 point on test and 2.6 on real speech.

The path from the first DS-CNN to this model, every experiment, and the
earlier models are in the project journal,
[docs/legacy/EXPERIMENTS.md](docs/legacy/EXPERIMENTS.md).

## Dataset

The class master dataset, as published, with its own train / test /
holdout splits. We only add a validation split (1,448 clips, ~13% of train, by speaker)
for choosing epochs and settings. Full description, statistics and build
steps: [docs/DATASET.md](docs/DATASET.md).

| Split | Clips | Real | Synthetic | Filipino voices | Out of scope | Speakers | Used for |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 9,285 | 2,019 | 7,266 | 708 | 251 | 331 | Training |
| val (ours) | 1,448 | 340 | 1,108 | 5 | 19 | 46 | Choosing epochs and settings |
| **test** | **4,443** | 824 | 3,619 | 203 | 76 | 144 | **Reported results** |
| holdout | 202 | 96 | 106 | 87 | 16 | 7 | Raspberry Pi live test; also scored offline |
| numerals | 66,390 | all | — | — | all | 2,547 | Bare numbers as out-of-scope examples (1,500 sampled) |
| supplemental | 3,461 | — | 3,461 | — | — | 60 | Extra synthetic clips of train voices (from `supplemental_synth`), used in training |

**Dataset revision.** Everything uses the 2026-10-02 revision
(`da92a79`), pinned in every download command. On the DGX the class's
shared copy `/data/ai231` is used instead, after checking it is identical
([docs/DATASET.md](docs/DATASET.md#the-shared-copy-on-the-dgx-dataai231)).

Slot values are exactly the schema's: 10 s / 30 s / 1 min, 6:00 AM /
8:00 AM / 9:00 PM, 18 / 22 / 26 degrees, 20 / 60 / 100 percent, red /
blue / green, drink water / study / exercise.

## Training

PyTorch on one A100 of UP's shared DGX. The final recipe:

- CRNN as above, 372K parameters, dropout 0.1.
- Training data: the train split (9,285 clips), the dataset's
  supplemental synthetic clips of train voices (3,461) and 1,500 bare
  numbers from the numerals set as OUT_OF_SCOPE: 14,246 clips.
- 80 epochs, batch size 128 (112 steps per epoch, 8,960 steps); Adam,
  learning rate 1e-3, 5 warm-up epochs, then cosine decay. Best epoch on
  val: 65.
- Loss: class-weighted cross-entropy, plus an extra penalty for confusing
  VOLUME_UP/VOLUME_DOWN/TEMPERATURE and LIGHT_ON/LIGHT_OFF, plus slot
  cross-entropy at weight 0.3, plus distillation (KL at temperature 3,
  weight 1) toward the averaged predictions of 9 smaller CRNNs (99K
  parameters) trained on the same split (Experiment 43t).
- Augmentation: background noise, speed 0.9–1.1×, reverb and start shift
  on the waveform; 2 frequency masks on the log-mel (no time masks).
- 3 seeds, all on one A100; checkpoints and settings chosen on val.

[docs/TRAINING.md](docs/TRAINING.md) explains how we ran on the shared
node and gives the exact commands. `bash scripts/reproduce.sh` runs all of
it from a fresh clone.

## Test results

Final numbers for the shipped models, on the current test set (revision
`da92a79`). Tables for every seed and split are in
[docs/TESTING.md](docs/TESTING.md).

**Intent** (`models/vcm_intent.onnx`, identical to its checkpoint):

| | Accuracy | n |
|---|---:|---:|
| **Class test set, all clips** | **95.50%** | 4,443 |
| Real speech | 78.64% | 777 |
| Synthetic voices | 99.45% | 3,619 |
| Exact Option B wording (the demo phrases) | 99.22% | 3,823 |
| Same command in other words | 72.98% | 544 |
| Filipino speakers (class recordings) | 86.77% | 189 |
| Macro average over 20 classes | 92.94% | |
| **Holdout (Pi live-test set)** | **96.04%** (real 96.51%) | 202 |

Strongest classes on real speech: TEMPERATURE and CREATE_REMINDER (100%),
STOP 94%, ALARM 94%, BRIGHTNESS 92%. Weakest: CALL 38%, PLAY_MUSIC 38%,
WEATHER 46%, MESSAGE 50% (12–26 real clips each, mostly SLURP's free-form
phrasings).

**Slot values** (test, slot head right / intent and slot both right):
98.2% over all 2,538 slotted clips. Real speech: TIMER 100 / 79%, ALARM
90 / 88%, BRIGHTNESS 96 / 88%, COLOR 82 / 69%, TEMPERATURE 79 / 79%,
CREATE_REMINDER 89 / 89% (n = 18–118 each).

**Reject threshold.** At 0.6 confidence the device rejects 16% of real
commands ("please repeat") and is right on 87% of those it accepts.

**Latency on the Raspberry Pi 5** (`scripts/benchmark_pi.py`, 1 thread):
13.9 ms p50 / 15.2 ms p95 per command end to end, RTF 0.0061 at p95; the
wake word uses 1.9% of one core. Measured with the shipped Experiment 43b
weights. `vcm_intent_small.onnx` takes about 12 ms p95. Details:
[results/bench_pi5.md](results/bench_pi5.md).

### Out of scope

The dataset's `OUT_OF_SCOPE` clips (noise, Filipino speech, near-miss
requests, general speech) are kept as the dataset defines them: a 20th
class in training, never acted on by the device, and scored as correct
only when the model says OUT_OF_SCOPE. Reported separately on test:

| | Result | n |
|---|---:|---:|
| Commands only (out-of-scope clips excluded) | 95.95% (real speech 78.64%) | 4,367 |
| Out-of-scope clips labeled OUT_OF_SCOPE | 69.7% | 76 |
| Out-of-scope clips the device ignores (OUT_OF_SCOPE or confidence < 0.6) | 82.9% | 76 |
| Out-of-scope clips it would act on (false accept) | 17.1% | 76 |
| Commands it ignores (false reject, same rule) | 3.7% | 4,367 |

On the holdout set the device ignores 93.8% of out-of-scope clips (n=16)
and 0.5% of commands. The wake word filters most non-command speech
before the intent model hears it. Details:
[TESTING.md](docs/TESTING.md#out-of-scope-how-it-is-handled-and-tested).

### Wake word

Trained with the master dataset as its negatives (its train split's
commands, out-of-scope speech and noise, plus 3,000 numerals clips;
Experiment 43, seed 1, the best on val). The "hey kiwi" positives are not
in any class dataset: they are ~3,000 synthetic clips and the author's own
recordings. On the master test split streamed as one 2.94 h recording with
near-miss phrases:

| Threshold | Missed, clean | Missed, 10 dB noise | Author's real takes missed | False wake-ups per hour |
|---:|---:|---:|---:|---:|
| **0.6 (default)** | 3.8% | 7.2% | 0/10 | 12.9 |
| 0.7 | 4.6% | 8.4% | 0/10 | 9.5 |
| 0.85 | 8.2% | 12.5% | 0/10 | 7.2 |
| 0.95 | 13.8% | 23.3% | 0/10 | 2.0 |

25,475 parameters, 107 KB. The device uses 0.6, and 0.4 while music
plays. Details: [docs/TESTING.md](docs/TESTING.md#wake-word-modelskiwi_wakewordonnx).

**Automated tests.** 245 unit tests run without any hardware
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
| Regularization and augmentation | Waveform noise, speed, reverb and shift; frequency-only SpecAugment (time masks can erase the one word that matters); dropout 0.1. EMA and label smoothing tested, not kept. | [MODEL.md](docs/MODEL.md#5-training-recipe-final-model) |
| Class imbalance | Inverse-frequency class weights; extra out-of-scope examples from the numerals set. | `src/vcm/train/losses.py` |
| Knowledge distillation | An ensemble of 9 smaller CRNNs (same data) teaches the final CRNN. | `scripts/generate_ensemble_labels.py`, [MODEL.md](docs/MODEL.md#5-training-recipe-final-model) |
| Evaluation | Overall, real-speech, per-class, per-accent and per-source accuracy; confusion pairs; slot accuracy; reject threshold. | `scripts/evaluate_checkpoint.py` |
| Efficiency | Accuracy against parameters, file size, latency and memory; comparable-size baselines. | [FOOTPRINT.md](docs/FOOTPRINT.md), [MODEL.md](docs/MODEL.md#4-baselines) |
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
│   ├── kiwi_wakeword.onnx       ★ shipped wake word (Experiment 43), and kiwi_wakeword.pt
│   ├── vcm_intent_small.onnx    smaller intent model (Experiment 43c, 722 KB), optional
│   └── legacy/                  earlier models (history only)
├── results/                   the final experiment's logs, evaluations and launcher; Pi and class benchmarks
│   └── legacy/                  Experiments 37–42 (history only)
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
│   ├── dataset/               manifest format, the class schema, the wake word's synthetic phrase lists
│   │   └── legacy/              loaders for the project's earlier dataset (history only)
│   └── train/                 architectures, training loop, losses, augmentation
├── scripts/                   command-line tools (see the table below)
│   └── legacy/                  earlier-dataset, ASR-cascade and wake-word-selection scripts (history only)
├── tests/                     unit tests (pytest), no hardware needed
│   └── legacy/                  tests of the legacy loaders (still run)
├── deploy/                    systemd services and audio-device rules for the Pi
├── data/dataset_schema/       the final class schema (CSV) and dataset fingerprint; other data is downloaded, not in git
└── docs/                      documentation of the final system (see below)
    └── legacy/                  the project journal (EXPERIMENTS.md) and archived records
```

| Stage | Scripts |
|---|---|
| **Device** (Pi or laptop) | `vcm_listen.py` (the voice loop), `kiwi_doctor.py` (preflight check), `benchmark_pi.py` (latency p95 / RTF), `measure_footprint.py` |
| **Laptop** | `deploy_pi.sh` (deploy to the Pi), `mac_phone_bridge.py`, `spotify_auth.py`, `record_wakeword.py` |
| **Data** (DGX) | `build_me2_manifest.py` (master dataset → manifest, slot labels, metadata; `--shared-cache /data/ai231` for the class's copy), `verify_shared_dataset.py` (checks a copy against the committed fingerprint) |
| **Train and evaluate** (DGX) | `reproduce.sh` (all steps), `python -m vcm.train.train`, `generate_ensemble_labels.py` (distillation teacher), `evaluate_checkpoint.py`, `summarize_experiments.py`, `export_onnx.py`, `train_wakeword.py`, `evaluate_wakeword.py` |
| **Wake-word data** (laptop / DGX) | `record_wakeword.py` (your own "hey kiwi" takes), `generate_targeted_synthetic.py --batch wakeword` (the synthetic "hey kiwi" positives and near-misses) |

Every `legacy/` folder is history only: nothing the final model, its training or the device uses imports from one. The project's history is in [docs/legacy/EXPERIMENTS.md](docs/legacy/EXPERIMENTS.md).

## Quick start

**Run the tests** on any machine. No hardware is needed.
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train,deploy]"
python -m pytest
```

**Reproduce the model** on a GPU machine (downloads ~3.6 GB, or reads the shared copy on the DGX):
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

## Run it on the Raspberry Pi

This section is everything needed to run the demo. The full setup from a
blank SD card is in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md), and
troubleshooting and the demo-day checklist are in
[docs/RUNBOOK.md](docs/RUNBOOK.md). Commands marked **(laptop)** run in
this repo's folder on your laptop; commands marked **(Pi)** run in an SSH
session on the Pi (`ssh raspberrypi.local`).

### What runs where

| Where | Program | What it does | Needed? |
|---|---|---|---|
| Pi | **Home server**: `python -m vcm.home.server` (port 8000), service `vcm-home` | Acts on each command, speaks the reply, serves the dashboard, rings timers and alarms | Yes. Start it **first** |
| Pi | **Listener**: `scripts/vcm_listen.py --server http://127.0.0.1:8000`, service `vcm` | Microphone → "Hey Kiwi" → records the command → intent + slot model → sends it to the home server | Yes. Start it **second** |
| Pi | raspotify (its own system service) | Makes the Pi a Spotify speaker for PLAY_MUSIC / PAUSE / STOP / NEXT | Only for Spotify |
| Mac | **Phone bridge**: `scripts/mac_phone_bridge.py` (port 8765) | Places calls and sends messages through your iPhone | Only for CALL / MESSAGE |
| Any browser | Dashboard at http://raspberrypi.local:8000 | Shows the lamp, thermostat, timers, alarms, reminders, music and call log | Optional |

The Pi needs a USB microphone and a USB speaker in its black USB 2 ports;
a Sense HAT (room temperature) and a GPIO 17 pushbutton are optional.

### Before you start

- A Raspberry Pi 4 or 5 with **64-bit** Raspberry Pi OS and SSH on, named
  `raspberrypi` (so it is `raspberrypi.local`), on the same Wi-Fi as the
  laptop. Venue Wi-Fi often blocks device-to-device traffic; a phone
  hotspot works as a backup.
- `ssh raspberrypi.local` logs in without a password. If it asks for one,
  run `ssh-copy-id raspberrypi.local` once **(laptop)**.
- `configs/settings.toml` exists on the laptop, copied from
  `configs/settings.example.toml`. The `[weather]`, `[spotify]` and
  `[phone]` sections are optional: without them those commands say they
  are not set up, and everything else works.

### Step 1. Deploy (once, and again after every change)

**(laptop)** Installs the system packages, copies the code, the models and
your settings, builds the Python environment, installs the audio-device
rules, runs a benchmark, and installs the two services so they start at
boot:
```bash
scripts/deploy_pi.sh raspberrypi.local --services
```
Add `--settings` to overwrite the Pi's `configs/settings.toml` with the
laptop's (it is copied only the first time otherwise).

### Step 2. Check the peripherals

**(Pi)** The preflight check tests power, the microphone (device and live
signal), the speaker (with a test beep), both services, the home server,
the models and the internet. The last line should say **READY**:
```bash
cd ~/quielq-vcm && .venv/bin/python scripts/kiwi_doctor.py --beep --fix
```

To check one peripheral at a time **(Pi)**:

| Peripheral | Command | Expect |
|---|---|---|
| Microphone (the input in use) | `pactl get-default-source` | The USB mic, not the soundbar |
| Microphone (hardware) | `arecord -l` | A USB card (the card number can change) |
| Speaker (the output in use) | `pactl get-default-sink` | The USB speaker / soundbar |
| Speaker volume | `pactl set-sink-volume @DEFAULT_SINK@ 100%` | Replies are audible |
| Sense HAT | `~/quielq-vcm/.venv/bin/python -c "from sense_hat import SenseHat; print(SenseHat().get_temperature())"` | A temperature in °C (reads a few degrees warm on the Pi) |
| Power | `vcgencmd get_throttled` | `throttled=0x0` (no undervoltage) |

If the microphone is silent right after a deploy, restart the audio stack
and the services, then run the preflight check again **(Pi)**:
```bash
systemctl --user restart pipewire pipewire-pulse wireplumber && sleep 3 && systemctl --user restart vcm-home vcm
```

### Step 3. Start Kiwi

**Option A: as services (the demo setup).** After step 1 with
`--services`, both programs already run, start at every boot and restart
3 s after any crash. Nothing else to start on the Pi. To restart them
**(Pi)**:
```bash
systemctl --user restart vcm-home vcm
```

**Option B: by hand, in two terminals (for debugging).** Stop the services
first, or two copies fight over port 8000 and the microphone **(Pi)**:
```bash
systemctl --user stop vcm vcm-home
```
Then open a tmux session so the programs survive a dropped SSH connection
**(Pi)**:
```bash
tmux new -s kiwi
```
Terminal 1, the home server **(Pi)**:
```bash
cd ~/quielq-vcm && .venv/bin/python -m vcm.home.server
```
Press `Ctrl-b c` for a second tmux window. Terminal 2, the listener
**(Pi)**:
```bash
cd ~/quielq-vcm && .venv/bin/python scripts/vcm_listen.py --server http://127.0.0.1:8000 --show-scores
```
`Ctrl-b n` switches windows, `Ctrl-b d` detaches (both keep running),
`tmux attach -t kiwi` comes back, `Ctrl-c` stops a program. When done,
`systemctl --user start vcm-home vcm` brings the services back.

Useful listener options: `--show-scores` prints the live wake-word score;
`--wake-threshold 0.7` if it wakes by itself too often (default 0.6, and
0.4 while music plays); `--intent-model models/vcm_intent_small.onnx` for
the smaller model; `--trigger button` for push-to-talk on GPIO 17;
`--save-commands DIR` saves each recorded command as a WAV.

### Step 4. Optional: calls, messages and Spotify

**Calls and messages** go through your Mac and iPhone. Set `[phone]` in
`configs/settings.toml` (setup in
[DEPLOYMENT.md § 8b](docs/DEPLOYMENT.md#8b-actions-and-the-web-dashboard)),
then keep this running on the Mac **(laptop)**:
```bash
.venv/bin/python scripts/mac_phone_bridge.py --token <bridge_token from settings.toml>
```
Add `--dry-run` to print calls and messages instead of sending them.

**Spotify** needs Premium, the `[spotify]` section, and raspotify
installed on the Pi once (DEPLOYMENT.md § 8b). Check that it runs **(Pi)**:
```bash
systemctl status raspotify
```

### Step 5. Use it

1. Open the dashboard at http://raspberrypi.local:8000. On Android, use
   the Pi's IP address instead (`ssh raspberrypi.local hostname -I`).
2. Say **"Hey Kiwi"**, then a command in the schema's wording, like "what
   time is it" or "start a timer for 30 seconds". The phrasings and slot
   values that work best are in
   [RUNBOOK.md](docs/RUNBOOK.md#test-checklist).
3. The dashboard's **Simulate a command** box runs the same actions
   without speaking.

### Everyday commands

| To | Run |
|---|---|
| Watch what Kiwi hears and does **(laptop)** | `ssh raspberrypi.local journalctl --user -u vcm -f` |
| Watch the home server's log **(laptop)** | `ssh raspberrypi.local journalctl --user -u vcm-home -f` |
| Restart both services **(laptop)** | `ssh raspberrypi.local systemctl --user restart vcm-home vcm` |
| Stop both services **(laptop)** | `ssh raspberrypi.local systemctl --user stop vcm vcm-home` |
| Stop them starting at boot **(laptop)** | `ssh raspberrypi.local systemctl --user disable vcm vcm-home` |
| Run the preflight check **(laptop)** | `ssh raspberrypi.local 'cd ~/quielq-vcm && .venv/bin/python scripts/kiwi_doctor.py --beep'` |
| Measure latency, p95 and RTF **(laptop)** | `ssh raspberrypi.local 'cd ~/quielq-vcm && .venv/bin/python scripts/benchmark_pi.py --json bench_pi.json'` |
| Update the Pi after a change **(laptop)** | `git pull`, then `scripts/deploy_pi.sh raspberrypi.local --services` |
| Push changed settings **(laptop)** | `scripts/deploy_pi.sh raspberrypi.local --settings` |

## Documentation

| Document | What's in it |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | The assignment, system design, key decisions, hardware |
| [MODEL.md](docs/MODEL.md) | The final models: features, architecture, training recipe, baselines, wake word, export |
| [DATASET.md](docs/DATASET.md) | The class master dataset and how we use it |
| [TRAINING.md](docs/TRAINING.md) | Training on the shared DGX, and commands to reproduce the final models |
| [TESTING.md](docs/TESTING.md) | Final test results, evaluation method, the automated test suite |
| [FOOTPRINT.md](docs/FOOTPRINT.md) | SD card and RAM use on the Pi, and our pipeline vs. an ASR cascade |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Setting up a Raspberry Pi from a blank SD card |
| [RUNBOOK.md](docs/RUNBOOK.md) | Starting, testing and demoing the system, and troubleshooting |
| [CODE_GUIDE.md](docs/CODE_GUIDE.md) | **Study guide to the code**: how the mel spectrogram, CRNN, intent and slot heads, wake word, ONNX runtime, Raspberry Pi hardware and home server are implemented, with links to the exact lines |
| [legacy/EXPERIMENTS.md](docs/legacy/EXPERIMENTS.md) | **Project history** (our journal): every experiment, the earlier models, and how they compare with the final one |

## Future enhancements

- **More real speech.** Real clips are 22% of train; synthetic voices
  score ~99% and real speakers 78.64%. Commands phrased in the
  speaker's own words are the main source of errors.
- **Out-of-scope rejection.** OUT_OF_SCOPE has the fewest train clips
  (251); 17% of out-of-scope test clips would still be acted on.
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
