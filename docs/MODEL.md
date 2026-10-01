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

`vcm/train/architectures.py`, class `CRNN`. A CRNN is a convolutional
recurrent neural network. The convolutions find short sound patterns. The
recurrent layer (a GRU) reads those patterns in order, like reading words in
a sentence.

### 3.1 The diagram

**Figure 1. The shipped CRNN: from a spectrogram to an intent and a slot value.**
The model turns 5 seconds of audio into a picture of sound energy over time.
Convolutions find short sound patterns in that picture. A GRU then reads the
patterns in order, from start to end and from end to start. Attention pooling
picks the moments that matter, and the classifiers name the intent and its
value.

Shapes are written as channels × mel bands × time frames. One time frame is
10 ms at the input. Parameter counts are in brackets.

```mermaid
---
title: Figure 1. The shipped CRNN (107,887 parameters, 432 KB)
---
flowchart TB
    IN["<b>Input: log-mel spectrogram</b><br/>1 × 40 × 501<br/>5 seconds of audio, 10 ms per frame"]

    subgraph CNN["① Convolutional front end: finds short sound patterns (23K)"]
        direction TB
        C1["Conv 10×4, stride 2, 64 filters, BatchNorm, ReLU<br/>→ 64 × 20 × 250 (2.8K)"]
        D1["Depthwise-separable block 1<br/>3×3 depthwise + 1×1 pointwise<br/>→ 64 × 20 × 250 (5K)"]
        D2["Block 2, stride 2<br/>→ 64 × 10 × 125 (5K)"]
        D3["Block 3<br/>→ 64 × 10 × 125 (5K)"]
        D4["Block 4, stride 2<br/>→ 64 × 5 × 63 (5K)"]
        C1 --> D1 --> D2 --> D3 --> D4
    end

    P["<b>② Frequency projection</b> (21K)<br/>1×1 conv folds 64 channels × 5 bands into 64 features<br/>→ a sequence of 63 steps × 64, one step ≈ 80 ms"]

    subgraph RNN["③ Bidirectional GRU: reads the command in order (50K)"]
        direction LR
        F["Forward GRU, 64 units<br/>start → end"]
        B["Backward GRU, 64 units<br/>end → start"]
    end

    S["63 steps × 128 features<br/>each step now knows what came before and after it"]

    A["<b>④ Attention pooling</b> (129)<br/>scores every step, softmax over time,<br/>weighted average → 128 features"]
    I["<b>Intent classifier</b> (2.6K)<br/>dropout 0.1, linear<br/>→ 20 classes"]

    subgraph SLOTS["⑤ Six slot heads (11.6K)"]
        direction LR
        SA["Own attention pooling<br/>finds where the value is said"]
        SC["Linear → value<br/>TIMER 24 · ALARM 28 · BRIGHTNESS 12<br/>COLOR 14 · TEMPERATURE 3 · REMINDER 3"]
        SA --> SC
    end

    IN --> CNN --> P --> RNN --> S
    S --> A --> I
    S --> SLOTS
```

Total: **107,887 parameters**, 432 KB as fp32 ONNX, 53.7M multiply-adds per
command. Almost half of the parameters are in the GRU.

### 3.2 Walking through it

1. **Convolutional front end.** The first convolution looks at 10 mel bands
   × 4 frames at a time and halves both axes. Then four depthwise-separable
   blocks follow. A depthwise-separable block splits a normal convolution in
   two: a 3×3 filter per channel, then a 1×1 filter that mixes channels.
   This uses about 8 times fewer parameters than a normal 3×3 convolution.
   Blocks 2 and 4 halve the map again, so the later blocks see a wider
   stretch of audio.
2. **Frequency projection.** The map is now 64 channels × 5 bands × 63
   steps. A 1×1 convolution turns each time step into one vector of 64
   numbers. The audio is now a sequence, one step per ~80 ms.
3. **Bidirectional GRU.** One GRU reads the 63 steps from start to end, and
   another reads them from end to start. Their outputs are joined, so every
   step carries context from the whole command. This is what lets the model
   tell "turn the volume up" from "turn the volume down".
4. **Attention pooling.** A single linear layer gives every step a score.
   Softmax turns the scores into weights that add up to 1. The weighted
   average becomes the summary of the command. The step that says "up" can
   get a high weight and decide the answer.
5. **Slot heads.** Each slotted intent has its own attention pooling and
   classifier. The value ("five minutes", "blue") is usually in a different
   part of the command than the words that identify the intent, so each slot
   head learns where to look. A slot value is only used when the intent
   matches, for example the TIMER head only when the intent is TIMER.

**The wake word** uses the same design at a smaller size: 32 channels, 32
GRU units each way, and a 1.5 s input (40 × 151 frames, which becomes 19
steps). It has 25,475 parameters and 2 outputs: "hey kiwi" or not.

### 3.3 Why a CRNN

The first 27 experiments used DS-CNN ("Hello Edge", Zhang et al. 2017). It
plateaued at 68% to 71% on real speech. Each of its outputs sees only about
24 frames, or 240 ms. That is shorter than the word "temperature". It then
averages everything, so word order is lost. It classified a command like a
bag of quarter-second snippets.

**Figure 2. How DS-CNN and the CRNN hear the same command.** DS-CNN cuts the
command into short pieces and averages them, so it cannot tell which word
came first. The CRNN keeps every step in order, so the last word ("up")
can decide the answer.

```
command:   "turn    the    volume    up"
DS-CNN:     [··]   [··]    [··][··]   [··]    240 ms pieces, averaged, order lost
CRNN:       ───────────────────────────▶     every step in order, attention picks "up"
```

That is why DS-CNN kept confusing commands that differ by one word, like
volume up and down, or lights on and off. The CRNN fixes this with three
small changes: strided blocks that widen what each step sees, a GRU that
reads the whole command in order, and attention pooling so the important
word decides the answer. It was the largest single gain in the project:
**+11 points on real speech** (Experiment 28), the same across three seeds.

**Assignment note.** The original plan said "no attention/transformer
layers". Attention pooling here is one linear scorer over time steps (129
parameters per head). A Transformer's self-attention compares every step
with every other step and is much larger. If the rule is meant literally,
plain average pooling is the fallback (see Future enhancements in the
README).

## 4. Compared with the other models we tested

### 4.1 DS-CNN (Experiments 1 to 27)

The standard keyword-spotting model from "Hello Edge". Shown at the
"matched capacity" size from Experiment 11, with the 3 s input used then.

**Figure 3. DS-CNN: convolutions, then one big average.** It uses the same
kind of blocks as the CRNN's front end. Its blocks never stride, so each
output only hears about 240 ms. Global average pooling then mixes all
positions into one summary, and word order is lost.

```mermaid
---
title: Figure 3. DS-CNN (26,300 parameters)
---
flowchart LR
    IN["Log-mel<br/>1 × 40 × 301<br/>(3 s)"] --> C1["Conv 10×4, stride 2<br/>60 filters<br/>→ 60 × 20 × 150"]
    C1 --> D["5 depthwise-separable blocks<br/>no striding<br/>→ 60 × 20 × 150<br/>each output sees ~240 ms"]
    D --> G["Global average pooling<br/>averages all 3,000 positions<br/>→ 60 numbers"]
    G --> L["Linear<br/>→ 20 classes"]
```

26,300 parameters. Best results: **75.5% validation** (Experiment 15) and
**70.6% real-speech test** (Experiment 25). Same
building blocks as the CRNN's front end. The difference is what happens
after: DS-CNN averages immediately, while the CRNN strides, keeps the time
order and reads it with a GRU.

### 4.2 BC-ResNet (Experiments 3 to 6, and 10)

"Broadcasted residual learning" (Kim et al. 2021). The original plan
recommended it because it beat DS-CNN on Google Speech Commands.

**Figure 4. One BC-ResNet block: a frequency path and a time path, added
together.** The frequency path keeps all 20 mel bands. The time path first
averages the bands away, which makes it cheap, and then looks along time.
Its result is copied ("broadcast") back to every band and added in. The
model stacks 8 of these blocks and still ends with a global average, like
DS-CNN.

```mermaid
---
title: Figure 4. One BC-ResNet block (the model stacks 8)
---
flowchart TB
    X["Block input<br/>48 × 20 × 301"] --> FP["Frequency path<br/>depthwise conv 3×1 over mel bands<br/>+ BatchNorm"]
    FP --> AVG["Average over the 20 mel bands<br/>→ 48 × 1 × 301"]
    AVG --> TP["Temporal path<br/>depthwise conv 1×3 over time<br/>+ BatchNorm, 1×1 conv, ReLU, dropout"]
    FP --> ADD(("+"))
    TP -->|"broadcast back<br/>over all 20 bands"| ADD
    X -->|"residual"| ADD2(("+"))
    ADD --> R["ReLU"] --> ADD2
    ADD2 --> OUT["Block output<br/>48 × 20 × 301"]
```

The full model is a 5×5 stem convolution (48 channels), 8 of these blocks,
global average pooling and a linear classifier. 25,748 parameters. **67.5%
validation**, below DS-CNN at the same size (72.4%). It also ends with a
global average, so it has the same word-order problem as DS-CNN. We stopped
testing it before real-speech test accuracy became our main measure.

### 4.3 ASR cascade (Experiment 26)

**Figure 5. The ASR cascade: speech to text, then text to intent.** Whisper
writes down what was said. A small text classifier then counts the words
and phrases and picks the intent. It was the most accurate option, but
Whisper alone is about 145 MB.

```mermaid
---
title: Figure 5. ASR cascade (about 74M parameters, 149 MB)
---
flowchart LR
    AU["Audio"] --> W["Whisper base<br/>speech-to-text<br/>74M parameters, 145 MB"]
    W --> T["Text<br/>'turn the volume up'"]
    T --> TF["TF-IDF<br/>word and phrase counts"]
    TF --> LOG["Logistic regression<br/>2 MB"]
    LOG --> O["20 classes"]
```

**90.6% real-speech test**, the best accuracy we measured. Text makes one-word
differences easy. It is about 690 times larger than the CRNN by parameter
count and takes 440 to 950 ms per command on a laptop, so it was not shipped
(see section 8 and [FOOTPRINT.md](FOOTPRINT.md)).

### 4.4 Side by side

| | DS-CNN | BC-ResNet | **CRNN (shipped)** | ASR cascade |
|---|---|---|---|---|
| Parameters | 26,300 | 25,748 | **107,887** | ~74M + classifier |
| Weights (fp32) | ~105 KB | ~103 KB | **432 KB** | ~149 MB |
| Input | 3 s log-mel | 3 s log-mel | **5 s log-mel, silence trimmed** | raw audio |
| What one output sees | ~240 ms | a short local window | **the whole command** | the whole command |
| Keeps word order? | No | No | **Yes (GRU)** | Yes (text) |
| How it summarizes | Global average | Global average | **Attention pooling** | Text classifier |
| Slot values | No | No | **6 heads** | Not built |
| Best result | 75.5% val, 70.6% real-speech test | 67.5% val | **84.8% real-speech test** | 90.6% real-speech test |
| Runs on the Pi in real time? | Yes | Yes | **Yes, 9.9 ms** | Too slow to listen always |

The CRNN is about 4 times bigger than DS-CNN. That is still tiny: it fits
the 1 MB budget twice over and runs in under 10 ms on the Pi. The extra
parameters went where DS-CNN was weak: reading the command in order.

## 5. Training recipe (final model)

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

## 6. The wake word

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

## 7. Export and runtime

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

## 8. Alternative considered: ASR cascade

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
