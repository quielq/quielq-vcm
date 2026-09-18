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

## 8. Open question this raises for Section 7 / DATASET.md

Building this pipeline surfaced a dataset gap worth flagging explicitly:
**there is currently no `unknown_background` class data in
`data/dataset_manifest.csv`.** Section 3 requires an explicit
background/unknown class and Section 9 already identifies Google Speech
Commands v2's background-noise clips as the intended source, but that
source has never actually been pulled into the dataset pipeline. This
blocks both training a usable background-rejection class and computing
the FAR/FRR benchmark metric Section 3 proposes — worth prioritizing
alongside FSC access.

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
