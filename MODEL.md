# Model Training & Deployment — Technology Choices

This documents the key technologies for the training/export/deployment
pipeline (Section 5 of [VCM_Architecture_Review.md](VCM_Architecture_Review.md)),
why each was chosen, the research they're grounded in, and options that
weren't in the original architecture writeup but are worth adopting.
Nothing here is implemented yet — this is the design reference before
writing training code (see the repo's `README.md` "Not in this pass").

## 1. Scope, restated precisely

This is **closed-set spoken intent classification, not automatic speech
recognition (ASR)**. Audio in, one of 19 intent labels out (or
`unknown_background`) — never a transcript. This matters for every
choice below: it keeps the model orders of magnitude smaller than any
general-purpose speech-to-text model (Whisper-class models are
encoder-decoder transformers producing open-vocabulary text; this task
needs a small closed-set classifier, closer to a keyword-spotting (KWS)
problem than a speech-recognition one). **The one place a real
speech-to-text model legitimately appears in this project is offline
dataset QA** (`vcm/dataset/qa/synthetic_check.py`, wrapping
`faster-whisper`) — checking whether synthetic audio says what it's
supposed to say, before training. That never runs at inference time and
isn't part of the deployed model.

## 2. Model architecture

### What the architecture doc already specifies

DS-CNN or TC-ResNet, depthwise-separable convolutions over log-mel
features, no attention/transformer layers.

### The research behind those two, and what's newer

- **DS-CNN** — Zhang, Suda, Lai, Chandra (ARM Research), *["Hello Edge:
  Keyword Spotting on Microcontrollers"](https://arxiv.org/abs/1711.07128)*
  (2017). Benchmarks DNN/CNN/RNN/CRNN/DS-CNN for KWS explicitly under
  **microcontroller** constraints (Cortex-M class, tens of KB of RAM).
  DS-CNN (depthwise-separable conv, following
  [MobileNets](https://arxiv.org/abs/1704.04861), Howard et al. 2017)
  wins on their accuracy-vs-footprint tradeoff.
- **TC-ResNet** — Choi et al. (Qualcomm AI Research),
  *["Temporal Convolution for Real-time Keyword Spotting on Mobile
  Devices"](https://arxiv.org/abs/1904.03814)* (2019). 1D temporal
  convolutions directly over stacked MFCC frames (time as the conv
  axis, not a 2D image-style spectrogram patch), residual blocks,
  designed for real-time mobile inference. Reports better
  accuracy/latency than DS-CNN on Google Speech Commands.
- **Worth adding to the comparison — BC-ResNet**, Kim, Chang, Lee, Sung
  (Qualcomm AI Research), *["Broadcasted Residual Learning for
  Efficient Keyword Spotting"](https://arxiv.org/abs/2106.04140)*
  (Interspeech 2021). Combines 1D temporal and 2D time-frequency
  convolution paths via a broadcasted residual connection; reports
  beating both DS-CNN and TC-ResNet on Google Speech Commands at
  comparable or smaller parameter counts. **Directly relevant to this
  project**: it's the architecture `SynTTS-Commands-Official` (the
  precedented synthetic dataset referenced in Section 9/DATASET.md)
  used for its own sim-to-real benchmark — their reported numbers
  (zero-shot 88.0% / 50-real-shot 93.0% accuracy on the real GSC test
  set) are a useful reference point for what a similar architecture
  achieves going from synthetic training data to real evaluation data,
  which is exactly this project's situation.
- **Also worth knowing — MatchboxNet**, Majumdar & Ginsburg (NVIDIA),
  *["MatchboxNet: 1D Time-Channel Separable Convolutional Neural
  Network Architecture for Speech Commands Recognition"](https://arxiv.org/abs/2004.08531)*
  (Interspeech 2020). 1D time-channel-separable convolutions (an even
  more parameter-efficient depthwise-separable variant), competitive
  accuracy at very small sizes (as few as ~93K params in their smallest
  variant). Part of NVIDIA NeMo, so has a maintained reference
  implementation if that ecosystem is preferred over hand-rolling.

**Recommendation**: prototype **BC-ResNet** as the primary architecture
(best published accuracy-per-parameter of the four, and the one
precedent we already have real numbers for from a dataset we're using),
keep **TC-ResNet-8** as a simpler fallback if BC-ResNet's dual-path
structure proves annoying to quantize cleanly. DS-CNN remains a fine
baseline for a first correctness check, it's the simplest to implement.

### One calibration point worth being explicit about

DS-CNN's original paper targets Cortex-**M** microcontrollers (no OS,
tens of KB RAM, no floating point unit on some parts). The Raspberry Pi
5's Cortex-**A76** is a completely different class of hardware, four
cores at 2.4GHz, gigabytes of RAM, a full Linux OS, NEON SIMD. Real-time
budget here is not the tight constraint it was for the MCU work these
papers targeted — there's headroom to prefer the more accurate
architecture (BC-ResNet) over the most minimal one, and headroom is
better spent on **accuracy/generalization** than on shaving already-tiny
latency further, given the real problem this project has already hit
empirically (large sim-to-real accuracy drop when synthetic-trained
models meet real voices) is a data/generalization problem, not a
compute-budget problem.

## 3. Feature extraction

Log-mel spectrogram, fixed ~1-2s window, already implemented
(`vcm/audio/features.py`). This is the standard input representation for
every architecture above and for GSC/SLURP-style KWS benchmarks
generally — no change recommended here.

## 4. Data augmentation — not yet in the architecture doc, worth adding

Given the dataset now mixes **real** (SLURP, accent-diverse but mostly
non-Filipino) and **synthetic** (Option B, voice-cloned including real
Filipino-English references) audio, and given the sim-to-real gap has
already shown up in an actual test run (a synthetic-trained model tested
against real voices), augmentation is worth budgeting for rather than
treating as optional polish:

- **SpecAugment** — Park et al. (Google Brain),
  *["SpecAugment: A Simple Data Augmentation Method for Automatic
  Speech Recognition"](https://arxiv.org/abs/1904.08779)* (2019).
  Randomly masks blocks of time steps and frequency bins directly on
  the log-mel spectrogram during training — cheap (no extra audio
  processing, works on the same features already being computed),
  and a standard, well-validated way to reduce overfitting to a
  narrow training distribution (exactly the synthetic-voice-specific
  overfitting already observed).
- **Noise/background augmentation**: mixing in Google Speech Commands
  v2's background-noise clips (already identified as a dataset source
  in Section 9, see the gap noted in Section 7 below) at training time,
  standard practice in the KWS literature and directly usable here.

## 5. Quantization: PTQ vs. QAT

The architecture doc currently specifies **post-training quantization
(PTQ)**. Worth reconsidering given the target is an aggressive <3%
closed-set error rate (Section 2, constraint 4):

- **PTQ** (current plan) — Krishnamoorthi (Google),
  *["Quantizing deep convolutional networks for efficient inference: A
  whitepaper"](https://arxiv.org/abs/1806.08342)* (2018). Quantize
  after training, no retraining needed, fast to iterate. Typically
  costs some accuracy, usually small for larger models, can be a
  more noticeable hit for models this small (little redundancy to
  absorb the precision loss).
- **QAT (quantization-aware training)** — Jacob et al. (Google),
  *["Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference"](https://arxiv.org/abs/1712.05877)*
  (CVPR 2018). Simulates INT8 rounding during training itself, so the
  model learns weights that are robust to quantization. Costs extra
  training complexity (fake-quant ops, a training loop change) but
  recovers accuracy PTQ tends to lose, more so for small models.

**Recommendation**: start with PTQ to get an end-to-end pipeline working
fast (matches the existing plan and the doc's Tier-1 sequencing
priorities), but budget for switching to QAT if the quantized model's
error rate doesn't clear the <3% target — this is a likely-enough
outcome given how tight that target is that it shouldn't be treated as
a surprise if it happens.

## 6. Export format and Raspberry Pi 5 runtime kernels

### Hardware being targeted

Raspberry Pi 5 = Broadcom **BCM2712**, quad-core Arm **Cortex-A76** @
2.4GHz, no on-board NPU/accelerator in the Tier-1 build (the Hailo AI
HAT+ is Tier-2/future-use only, see Section 11 — everything here assumes
**CPU-only inference**). Cortex-A76 implements Armv8.2-A, which includes
the **dot-product extension (SDOT/UDOT)** — this is the specific
instruction-level feature that makes INT8 GEMM/convolution fast on this
chip; any runtime that doesn't use it is leaving real performance on the
table.

### ONNX vs. TFLite

The architecture doc lists both as options ("exported and quantized to
TFLite or ONNX"). Concretely:

- **ONNX + ONNX Runtime**: train in PyTorch (already the stack, Section
  5), export directly via `torch.onnx.export`, no intermediate
  framework conversion. ONNX Runtime's CPU execution provider includes
  NEON-optimized kernels and can optionally be built with the Arm
  Compute Library (ACL) execution provider for further Arm-specific
  optimization. Version-pin the exact ONNX opset and ONNX Runtime
  version used for quantization, both at export time and on the Pi, to
  avoid subtle kernel-selection differences between versions.
- **TFLite**: requires converting from PyTorch (extra step, either via
  ONNX→TF→TFLite or a direct PyTorch→TFLite path), but ships with the
  **XNNPACK** delegate by default, which is heavily optimized for ARM
  NEON including INT8 dot-product kernels, and is the most
  battle-tested option specifically for small on-device audio models
  (it's what most published mobile KWS work, including the papers
  above, actually benchmarks against).

**Recommendation**: **ONNX Runtime**, since training is already in
PyTorch and direct export avoids a lossy multi-hop conversion — but
benchmark actual RPi5 latency against a TFLite/XNNPACK export of the
same trained model before committing, since XNNPACK's ARM kernel
maturity is a real, measurable advantage worth checking rather than
assuming away. This is cheap to check (same model, two export paths)
and removes a guess from a decision that affects real-time correctness.

### A library worth knowing about, not currently planned

**ncnn** (Tencent) is a C++ inference framework built specifically for
ARM/mobile edge deployment, with a strong reputation for raw ARM NEON
kernel performance. Not recommended as the primary path here (it would
mean maintaining a third export/runtime path alongside PyTorch's own
ecosystem, and ONNX Runtime/TFLite are both far more common for this
kind of benchmarking already), but worth keeping in mind if ONNX
Runtime and TFLite both come in slower than expected on the actual Pi.

## 7. Recommended stack (summary)

| Stage | Choice | Why |
|---|---|---|
| Architecture | BC-ResNet (TC-ResNet fallback) | Best published accuracy/parameter tradeoff of the options surveyed; real sim-to-real precedent via SynTTS-Commands-Official |
| Features | Log-mel spectrogram | Already implemented, standard for this model family |
| Augmentation | SpecAugment + GSC background noise mixing | Directly targets the sim-to-real gap already observed empirically |
| Training framework | PyTorch | Already the plan (Section 5), DGX-native |
| Quantization | PTQ first, QAT if <3% target isn't met | Fast iteration first, fall back to the more expensive-but-accurate option only if needed |
| Export | ONNX Runtime primary, benchmark TFLite/XNNPACK before committing | Direct PyTorch export; XNNPACK's ARM maturity is a real enough consideration to verify, not assume |
| Deployment target | Raspberry Pi 5 (BCM2712, Cortex-A76), CPU-only, no NPU | Tier-1 hardware; Hailo AI HAT+ acceleration is Tier-2/future-use only |

## 8. Dataset gap this raised — now resolved

Writing this doc surfaced a real gap: at the time, there was no
`unknown_background` class data in `data/dataset_manifest.csv`, despite
Section 3 requiring an explicit background/unknown class and Section 9
already naming Google Speech Commands v2's background-noise clips as
the intended source. **This is now fixed** — `scripts/fetch_gsc_background.py`
pulls and chops it into 600 `unknown_background` clips (see DATASET.md
step 5). FSC access remains the one still-open dataset gap of the ones
checked so far — see DATASET.md step 8.

## 9. ML-engineering review for a 90%+ target, and the ASR-cascade option

After 23 training experiments plateaued around 70-75% val accuracy
(EXPERIMENTS.md), a full review was done — codebase/pipeline/dataset
audit plus research on commercial voice assistants (Siri, Alexa,
Google Assistant) and published SLU benchmarks — specifically to
answer whether 90%+ is achievable and what it would take.

### Honest calibration: 90%+ is not realistic with current constraints

**SLURP** (real crowdsourced spoken commands + synthetic augmentation
— the closest published analog to this project's data mix) tops out
at **87-88% intent accuracy**, and only with massive scale (72K real +
69K synthetic utterances) *and* large self-supervised pretrained
encoders (wav2vec2/HuBERT). The benchmarks that do hit 90-99%
(Google Speech Commands, Fluent Speech Commands) aren't comparable —
single isolated words or ~250 scripted phrases with overlapping
train/test speakers, essentially memorization-friendly. This project's
data (mixed real+synthetic, several classes with zero real coverage,
small from-scratch model, no SSL pretraining) is structurally closer
to SLURP's harder ceiling. **80-85% is a more defensible target**
unless one of those constraints changes.

### The architectural finding that matters most: commercial assistants don't classify intent from audio

Siri, Alexa, and Google Assistant all use a **cascade**: audio →
speech-to-text (ASR) → intent classification **on the text
transcript** — not a single end-to-end audio-to-intent model. Sourced:
Amazon's own Alexa NLU papers describe ASR-transcript → domain/intent/
slot classification as the shipped production architecture, explicitly
contrasting it with end-to-end audio-to-intent as a still-unproven
research direction (FANS, Interspeech 2021). Apple's "Hey Siri" is a
tiny wake-word DNN separate from the transcript-based language
understanding. Google Assistant's on-device RNN-T ASR feeds a separate
text-understanding model.

**Why this matters here specifically**: this project's single most
persistent failure — VOLUME_UP vs. VOLUME_DOWN, LIGHT_ON vs.
LIGHT_OFF, near-identical carrier-phrase acoustic overlap (6 loss-
engineering experiments, 14-19, mixed/limited results) — nearly
disappears once classifying *text* instead of *spectrograms*. Verified
directly, not just argued: `faster-whisper` (`base`, 74M params) was
run against this project's own real LIGHT_ON/LIGHT_OFF/VOLUME_UP/
VOLUME_DOWN audio and correctly transcribed the distinguishing word
("lights off in the washroom", "Turn the volume up.") in nearly every
sample — the exact confusion that direct audio classification has
never fully resolved is largely a non-issue in text.

### Scoped prototype: ASR-cascade (audio → Whisper → text classifier → intent)

- **ASR**: `faster-whisper`, `base` size — verified `tiny` is
  meaningfully worse (garbled transcriptions, e.g. "I'd soften the
  wash arm" for a LIGHT_OFF clip) where `base` gets the same clips
  right. Both are confirmed runnable on Raspberry Pi.
- **Training data**: no new data needed. Run Whisper once over all
  63,476 existing audio clips; pair `(Whisper's own transcript,
  existing label)` as the text classifier's training data — this
  trains on the same *kind* of noisy ASR output the model will see at
  real inference time, not clean ground-truth text it'll never
  actually get.
- **Text classifier**: start with TF-IDF + logistic regression
  (cheapest plausible baseline) before reaching for a neural text
  classifier — this project's 20-intent, largely fixed-phrasing
  vocabulary is a much narrower problem than SLURP's full 69-intent
  open-domain benchmark.
- **Slot extraction** (TIMER duration, ALARM time, COLOR/BRIGHTNESS/
  TEMPERATURE values, CREATE_REMINDER task) becomes regex/rule-based
  text parsing instead of acoustic slot-filling — meaningfully easier,
  and not something the current direct-audio pipeline attempts at all.
- **Honest risks**: two models instead of one (real RAM/latency cost —
  whisper-base is **~1,070x this project's entire current DSCNN by disk
  size** (~144MB vs. 137.5KB), ~2,810x by parameter count (74M vs.
  26,300) — an earlier "~6x" estimate here was wrong, corrected after
  actually measuring both; see Section 10's size table); ASR errors
  on numbers/proper nouns could still cause slot-extraction mistakes
  even when intent classification succeeds; a genuinely bigger build
  than anything done so far, comparable in scope to the confusable
  grammar-decoder idea raised earlier and set aside — but this one has
  real production precedent and real evidence on this project's own
  data, not just a plausible argument.

### Other concrete gaps found in the same review (smaller, lower-risk)

- DS-CNN had **no regularization at all** (no dropout, unlike
  BCResNet's BCResBlock) — added as an opt-in `--dropout` flag.
- Optimizer had **no weight decay** (plain Adam) — added as
  `--weight-decay`, opt-in.
- **No label smoothing** anywhere, despite repeatedly observing
  overconfident wrong predictions on live audio (e.g. 0.97 confidence
  for the wrong class) — added to both loss paths, opt-in.
- **No feature-level normalization** beyond per-clip max-referenced dB
  — `extract_log_mel` now normalizes to roughly zero-mean/unit-variance
  using constants measured from a real 2,000-clip training sample
  (mean=-59.64, std=21.30). Unlike the above, this is *not* opt-in —
  it's a real change to the feature representation, so a checkpoint
  trained before it should be retrained from scratch, not resumed.
- The synthetic-audio QA gate (`vcm/dataset/qa/synthetic_check.py`,
  Section 11 of DATASET.md) existed but had **never actually been
  run** against Option B's 17,658 clips — direct published evidence
  found in this review shows ASR-based filtering of synthetic TTS
  clips measurably closes real/synthetic accuracy gaps (89%→92.5% in a
  directly comparable study). Now being run via
  `scripts/qa_filter_option_b.py`.

## 10. Consolidated recommendation — three real options, by accuracy and footprint

After 27 experiments (EXPERIMENTS.md), here's where things actually
stand. Sizes below are measured (`ls -la` / `du`), not estimated.

| # | Option | Accuracy | Model size | Deployable per assignment rules? |
|---|---|---|---:|---|
| 1 | **DS-CNN + confusable-pair loss** (Experiment 15) | 75.45% val | **137.5 KB** | Yes — clean baseline |
| 2 | **DS-CNN + distillation from the ASR-cascade** (Experiment 27) | 75.78% val (mixed per-class, see below) | **137.5 KB** | Yes — current best fully-compliant option |
| 3 | **ASR-cascade** (Experiment 26) | **90.62% test, real audio** | ~144 MB (whisper-base 142MB + classifier 1.9MB) | **No — backup/reference only** |

**Option 3, the ASR-cascade, is the accuracy ceiling but not a
candidate for the actual submitted VCM.** The assignment states "ASR
models are not desirable for on-device computing because of footprint"
and requires the VCM to be tiny — whisper-base alone is **~1,070x this
project's DS-CNN by disk size, ~2,810x by parameter count**. It's kept
in the repo and documented as a **backup option for systems that can
accommodate the footprint** (e.g. a benchmark reference, or a future
deployment target with more compute than an RPi4/5), not as something
that competes with Options 1/2 for the actual deliverable.

**Measured runtime footprint** (see [FOOTPRINT_COMPARISON.md](FOOTPRINT_COMPARISON.md)):
the deployed pipeline (CRNN intent model + "Hey Kiwi" wake word, int8 ONNX)
is 363 KB of models, peaks at ~90 MB of memory and takes ~3 ms per command.
The ASR-cascade needs ~149 MB of models, peaks at 478–696 MB and takes
440–950 ms per command, and it would still need a separate wake-word model.

**Option 1 vs. Option 2** — the real tradeoff, not a clean upgrade:
distillation (Option 2) uses the ASR-cascade purely as an offline
training-time teacher (see Experiment 27) — the deployed model is
identical in size and architecture to Option 1, just trained with
extra supervision. It fixed the project's long-standing diffuse
confusion cluster substantially (PLAY_MUSIC +16.2pp, COLOR +7.4pp,
CREATE_REMINDER +7.0pp — the first real progress on any of these
across 27 experiments) but caused a real regression on VOLUME_DOWN
(-11.6pp) and new weakness on PAUSE/STOP that Option 1 didn't have.
The +0.33pp headline gain hides substantial per-class churn — this is
not a strict improvement, and which one to actually ship depends on
which failure mode matters more for the live demo. Untried: a lower
`--distill-weight` (currently 2.0) to check whether the VOLUME_DOWN
cost can be reduced without losing the PLAY_MUSIC/COLOR/CREATE_REMINDER
gains.

## References

- Zhang, Suda, Lai, Chandra. "Hello Edge: Keyword Spotting on
  Microcontrollers." 2017. https://arxiv.org/abs/1711.07128
- Choi et al. "Temporal Convolution for Real-time Keyword Spotting on
  Mobile Devices." 2019. https://arxiv.org/abs/1904.03814
- Kim, Chang, Lee, Sung. "Broadcasted Residual Learning for Efficient
  Keyword Spotting." Interspeech 2021. https://arxiv.org/abs/2106.04140
- Majumdar, Ginsburg. "MatchboxNet: 1D Time-Channel Separable
  Convolutional Neural Network Architecture for Speech Commands
  Recognition." Interspeech 2020. https://arxiv.org/abs/2004.08531
- Howard et al. "MobileNets: Efficient Convolutional Neural Networks for
  Mobile Vision Applications." 2017. https://arxiv.org/abs/1704.04861
- Park et al. "SpecAugment: A Simple Data Augmentation Method for
  Automatic Speech Recognition." 2019. https://arxiv.org/abs/1904.08779
- Krishnamoorthi. "Quantizing deep convolutional networks for efficient
  inference: A whitepaper." 2018. https://arxiv.org/abs/1806.08342
- Jacob et al. "Quantization and Training of Neural Networks for
  Efficient Integer-Arithmetic-Only Inference." CVPR 2018.
  https://arxiv.org/abs/1712.05877
- Warden. "Speech Commands: A Dataset for Limited-Vocabulary Speech
  Recognition." 2018. https://arxiv.org/abs/1804.03209 (the paper
  behind Google Speech Commands v2, already a candidate source in
  Section 9)
- Bastianelli, Vanzo, Swietojanski, Rieser. "SLURP: A Spoken Language
  Understanding Resource Package." EMNLP 2020.
  https://aclanthology.org/2020.emnlp-main.588.pdf (real-world intent
  accuracy ceiling this project's data most resembles; see Section 9)
- Kumar et al. "FANS: Fusing ASR and NLU for Spoken Language
  Understanding." Amazon Alexa, Interspeech 2021.
  https://arxiv.org/pdf/2111.00400 (source for the ASR-then-NLU
  cascade being Alexa's production architecture; see Section 9)
- Lin, Goyal, Girshick, He, Dollár. "Focal Loss for Dense Object
  Detection." ICCV 2017. https://arxiv.org/abs/1708.02002 (the focal
  loss variant tried in EXPERIMENTS.md Experiments 18-19)
