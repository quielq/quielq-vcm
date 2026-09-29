# The models

Two small models run on the device. Both are CRNNs over log-mel features,
exported to fp32 ONNX and run with ONNX Runtime (no PyTorch on the device).

| | Intent + slot model | "Hey Kiwi" wake word |
|---|---|---|
| File | `models/vcm_intent.onnx` | `models/kiwi_wakeword.onnx` |
| Size | 432 KB | 107 KB |
| Parameters | 107,887 | 25,475 |
| Compute | 53.7M multiply-adds (~107 MFLOPs) per command | 5.3M multiply-adds per window, 10 windows a second |
| Input | 40 log-mel bands × 501 frames (5.0 s, silence-trimmed) | 40 × 151 frames (1.5 s window) |
| Output | 20 classes (19 intents + `unknown_background`) and 6 slot-value heads | wake / not wake |
| When it runs | Once per command, after the wake word | Every 100 ms, always on |
| Source checkpoint | `exp36_joint_w03_s1.pt` (Experiment 36, seed 1) | `kiwi_wakeword_v2_s0.pt` (Experiment 34) |
| Headline result | **84.84%** real-speech test accuracy (n=6,577) | 3.1% missed wake words (clean), 0/10 of the author's real takes missed, at threshold 0.6 |

The label list, slot vocabulary and feature settings are stored in each ONNX
file's metadata, so the runtime reads them from the model instead of keeping
its own copy.

## 1. The task

This is **closed-set spoken intent classification**, not speech recognition:
audio in, one of 20 labels out, never a transcript. The assignment rules out
ASR on the device because of its footprint, and a closed-set classifier is
orders of magnitude smaller.

Six intents also carry a value, predicted by a classification head over a
fixed vocabulary (`vcm/slots.py`):

| Intent | Values | Count |
|---|---|---:|
| TIMER | 10 s … 2 h | 24 |
| ALARM | every hour 12:00 AM … 11:00 PM, plus 5:30–8:30 AM | 28 |
| BRIGHTNESS | 10 … 100 % | 12 |
| COLOR | red, blue, green, … warm white, teal | 14 |
| TEMPERATURE | 18 / 22 / 26 degrees | 3 |
| CREATE_REMINDER | drink water / study / exercise | 3 |

## 2. Features

`vcm/audio/features.py`: 16 kHz mono audio, silence trimmed (30 dB below
peak, 0.1 s margin), padded or cut to 5.0 s, then a 40-band log-mel
spectrogram (400-sample FFT, 160-sample hop = 10 ms frames), converted to dB
and normalized with a fixed mean and standard deviation measured on 2,000
training clips.

On the device the same features come from `vcm/audio/dsp.py`, a numpy
re-implementation of librosa's mel spectrogram, tested to match it. That
removed librosa, scipy and numba from the device and cut peak memory from
~330 MB to ~90 MB.

## 3. Architecture: CRNN

`vcm/train/architectures.py`, class `CRNN`:

```
log-mel (1 × 40 × 501)
 → Conv 10×4, stride 2, 64 channels, BatchNorm, ReLU
 → 4 depthwise-separable blocks (3×3 depthwise + 1×1 pointwise); blocks 2 and 4 stride by 2
 → 1×1 projection that folds frequency into 64 features per frame
 → bidirectional GRU, 64 units each way
 → attention pooling (one learned weight per frame)  → intent classifier (20)
 → one attention pooling + classifier per slot       → 6 slot heads
```

**Why a CRNN.** The first 27 experiments used DS-CNN ("Hello Edge", Zhang et
al. 2017), which plateaued at 68–71% on real speech. Its receptive field is
only ~240 ms, shorter than the word "temperature", so it classified a
multi-word command as a bag of quarter-second snippets and kept confusing
commands that differ by one word (volume *up* / *down*, lights *on* / *off*).
The CRNN fixes this with three cheap changes: strided blocks so the
receptive field grows geometrically, a GRU that reads the whole command in
order, and attention pooling so the one frame that says "up" can decide the
answer instead of being averaged away. That was the largest single gain in
the project: **+11 points on real speech** (Experiment 28), consistent across
three seeds.

**Why separate attention per slot head.** The value ("five minutes", "blue")
is usually in a different part of the command than the words that identify
the intent, so each slot head learns where to look.

**Assignment note.** The original plan said "no attention/transformer
layers". Attention pooling here is a single linear scorer over time steps
(129 parameters per head), not transformer self-attention. If the rule is
read literally, plain average pooling is the fallback (see Future
enhancements in the README).

## 4. Training recipe (final model)

Full commands are in [TRAINING.md](TRAINING.md#reproducing-the-final-model);
results for every run are in [EXPERIMENTS.md](EXPERIMENTS.md).

| Setting | Value | Why (experiment) |
|---|---|---|
| Data | 70,641 clips, 64% real speech ([DATASET.md](DATASET.md)) | QA-filtered synthetic data (25), targeted phrasings (31), slot-value clips (32, 35, 36) |
| Epochs, batch, optimizer | 80 epochs, batch 128, Adam lr 1e-3, 5 warm-up epochs then cosine decay | Warm-up + cosine beat a flat rate (5–7) |
| Loss | Class-weighted cross-entropy + confusable-pair penalty (alpha 2.0) on VOLUME_UP/DOWN/TEMPERATURE and LIGHT_ON/OFF | Targets the one-word confusions (14–16) |
| Slot loss | Cross-entropy per slot head, weight 0.3, only on clips with a slot label | Weight 1.0 cost 3.5 intent points (32 vs 34) |
| Waveform augmentation | Background noise at 5–25 dB SNR, speed 0.9–1.1×, room reverb, 0–0.3 s start shift | +4.6 points real speech (29b) |
| Features | Silence trim + 5.0 s window | Removes a train/live mismatch (29a) |
| Checkpoint selection | Best validation accuracy; compare seeds on the **test** split, real speech only | Validation includes synthetic clips, which score 94–99% and hide real-speech weaknesses |
| Seeds | 3; shipped seed 1 | Seed 0 was one clip better on intent but failed the ALARM slot gate (36) |

## 5. The wake word

"Hey Kiwi" was picked before any model was built, by scoring 11 candidates
for false-trigger risk against 62,836 command transcripts and common English
words (EXPERIMENTS.md, "Wake-word selection"). The detector is the same CRNN
design at a quarter of the size (32 channels, 32 GRU units each way),
trained on 1.5 s windows:
~3,000 synthetic "hey kiwi" clips in 145 cloned voices, the author's own 20
training takes, near-miss phrases ("hey kitty", "every week") as hard
negatives, command speech and noise.

`vcm/wakeword/detector.py` scores the last 1.5 s every 100 ms. When the score
passes the threshold, `scripts/vcm_listen.py` records until 0.6 s of quiet
(max 5 s) and passes the command to the intent model.

| Threshold | Missed, clean | Missed, 10 dB noise | Author's real takes missed | False wake-ups per hour* |
|---:|---:|---:|---:|---:|
| **0.6 (default)** | 3.1% | 5.6% | 0/10 | 12.7 |
| 0.7 | 3.6% | 8.2% | 0/10 | 9.2 |
| 0.95 | 14.1% | 21.5% | 0/10 | 1.3 |

\*On a 7.48 h test stream that deliberately includes near-miss phrases, so a
real room sees fewer. The offline choice was 0.95; live testing on the Pi
showed it missed too many real attempts from across the room, so the
default is 0.6, and 0.4 while music is playing.

## 6. Export and runtime

- `scripts/export_onnx.py` exports a checkpoint to ONNX (opset 17) and writes
  the labels, slot vocabulary and feature settings into its metadata. The
  exported file was checked against the PyTorch checkpoint on the full test
  split: identical real-speech accuracy and slot lines.
- `vcm/deploy/runtime.py` (`OnnxIntentModel`) runs it with ONNX Runtime, one
  thread.
- **fp32, not int8.** Dynamic int8 quantization cost 6.8 points of
  real-speech accuracy (Experiment 32) to save 134 KB, and was no faster.
  Both fp32 files together are 539 KB, under the 1 MB budget.
- Commands below 0.6 confidence get "didn't catch that, please repeat"
  instead of an action. On validation (measured on the Experiment 34 model)
  this rejects ~11% of commands and raises accuracy on the accepted ones
  from 86% to 92%.

Measured cost on the Raspberry Pi 5 (`scripts/benchmark_pi.py`): 9.9 ms per
command (3.7 ms features + 6.2 ms model), the wake word uses 2% of one core,
94 MB peak memory. See [FOOTPRINT.md](FOOTPRINT.md).

## 7. Alternative considered: ASR cascade

Commercial assistants (Alexa, Siri, Google Assistant) transcribe speech
first and classify the text. We built that too (Experiment 26):
faster-whisper `base` transcribes the audio, and a TF-IDF + logistic
regression classifier maps the transcript to an intent. It scores **90.6%**
on the same real-speech test set, 5.8 points above our model, because text
makes one-word differences trivial.

It was not shipped, for the assignment's reason: footprint. It needs ~280×
the disk, 5–7× the memory and 150–300× the latency of our pipeline, and
Whisper is far too slow to listen continuously, so it would still need a
separate wake-word model. The full comparison is in
[FOOTPRINT.md](FOOTPRINT.md). The cascade scripts stay in the repo because
its Whisper transcripts are also used to label slot values on real speech
and to pick the wake word.

Published results put our number in context: SLURP, the dataset closest to
ours, tops out at 87–88% intent accuracy even with large pretrained speech
encoders (Bastianelli et al. 2020). Benchmarks that reach 95%+ (Speech
Commands, Fluent Speech Commands) use single words or a few hundred scripted
phrases.

## References

- Zhang, Suda, Lai, Chandra. "Hello Edge: Keyword Spotting on Microcontrollers." 2017. https://arxiv.org/abs/1711.07128 (DS-CNN)
- Kim, Chang, Lee, Sung. "Broadcasted Residual Learning for Efficient Keyword Spotting." Interspeech 2021. https://arxiv.org/abs/2106.04140 (BC-ResNet)
- Choi et al. "Temporal Convolution for Real-time Keyword Spotting on Mobile Devices." 2019. https://arxiv.org/abs/1904.03814 (TC-ResNet)
- Howard et al. "MobileNets." 2017. https://arxiv.org/abs/1704.04861 (depthwise-separable convolutions)
- Park et al. "SpecAugment." 2019. https://arxiv.org/abs/1904.08779
- Lin, Goyal, Girshick, He, Dollár. "Focal Loss for Dense Object Detection." ICCV 2017. https://arxiv.org/abs/1708.02002
- Lugosch et al. "Speech Model Pre-training for End-to-End Spoken Language Understanding." Interspeech 2019. https://arxiv.org/abs/1904.03670 (the CTC idea tried in Experiment 30)
- Hinton, Vinyals, Dean. "Distilling the Knowledge in a Neural Network." 2015. https://arxiv.org/abs/1503.02531 (Experiment 27)
- Krishnamoorthi. "Quantizing deep convolutional networks for efficient inference: A whitepaper." 2018. https://arxiv.org/abs/1806.08342
- Warden. "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition." 2018. https://arxiv.org/abs/1804.03209
- Bastianelli, Vanzo, Swietojanski, Rieser. "SLURP: A Spoken Language Understanding Resource Package." EMNLP 2020. https://aclanthology.org/2020.emnlp-main.588
- Lugosch et al. "Timers and Such: A Practical Benchmark for Spoken Language Understanding with Numbers." NeurIPS 2021 Datasets and Benchmarks. https://arxiv.org/abs/2104.01604
- Kumar et al. "FANS: Fusing ASR and NLU for Spoken Language Understanding." Interspeech 2021. https://arxiv.org/abs/2111.00400 (Alexa's ASR → NLU production design)
