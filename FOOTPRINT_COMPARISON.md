# Footprint: our pipeline vs. an ASR cascade

What it costs to run each voice pipeline on the device, measured rather
than estimated. It puts numbers on the assignment's point that "ASR models
are not desirable for on-device computing because of footprint", and on
what the extra accuracy of the ASR cascade (Experiment 26) would cost.

| | Pipeline | What runs per command |
|---|---|---|
| **Ours** | "Hey Kiwi" wake word + CRNN intent model (EXPERIMENTS.md Experiments 31–33), int8 ONNX | log-mel features → 96K-param CRNN → intent |
| **ASR cascade** | Experiment 26: faster-whisper `base` + TF-IDF/logistic-regression text classifier | audio → Whisper transcript → text classifier → intent |

## Summary

| | **Ours** | **ASR cascade** | Ratio |
|---|---:|---:|---:|
| Model files | **363 KB** (intent 270 + wake word 93) | **~149 MB** (Whisper 145 MB + tokenizer 2 MB + classifier 2 MB) | ~410× |
| Peak memory | **86–96 MB** | **478–696 MB** | 5–7× |
| Memory for the models once loaded | ~8.5 MB | ~290–370 MB | ~40× |
| Latency per command | **3.2–3.4 ms** (1 thread) | **440–950 ms** (see below) | ~150–300× |
| Python packages to install | ~149 MB: numpy, onnxruntime, sounddevice | ~330 MB: adds faster-whisper, CTranslate2, PyAV, tokenizers, scikit-learn, scipy | ~2× |
| Real-speech test accuracy | 85.1% (Experiment 31) | **90.6%** (Experiment 26) | ASR +5.5 points |
| Works as the always-on wake word? | Yes, ~1% of one core | **No.** Scoring a 1.5 s window every 0.1 s would need ~4.6 s of compute per second of audio | |

**In short:** the ASR cascade buys about 5.5 accuracy points for roughly
**400× the disk, 5–7× the memory and 150–300× the latency**, and it would
*still* need a separate wake-word model in front of it, because Whisper is
far too slow to listen continuously.

## Detail

### Our pipeline (int8 ONNX, 1 thread)

| Stage | Memory in use | Added |
|---|---:|---:|
| Python starts | 12.0 MB | +12.0 |
| + numpy | 23.6 MB | +11.6 |
| + onnxruntime | 42.0 MB | +18.4 |
| + sounddevice (microphone library) | 47.6 MB | +5.6 |
| + our runtime code | 48.4 MB | +0.8 |
| + intent model loaded | 56.0 MB | +7.6 |
| + wake-word model loaded | 57.0 MB | +0.9 |
| Running: 30 s of wake-word listening + commands | 86–96 MB | +29–39 |

The models themselves account for about 8.5 MB. Most of the running
overhead is ONNX Runtime keeping working memory around for reuse.

Removing librosa mattered here. Features used to be computed with librosa,
which pulls in numba, LLVM and scipy: importing and using it once added
**+186 MB**, and the whole pipeline peaked at ~330 MB. A numpy
re-implementation (`vcm/audio/dsp.py`, tested to match librosa exactly)
brought that to ~90 MB.

### ASR cascade (faster-whisper base)

| Setup | Loading Whisper adds | Peak memory | Latency per command |
|---|---:|---:|---:|
| The Experiment 26 demo's defaults, 4 threads | +373 MB | 696 MB | 588 ms |
| int8 weights, 4 threads (the most favorable setup) | +322 MB | 501–558 MB | 437–460 ms |
| int8 weights, 1 thread | +287 MB | 478 MB | 946 ms |

Using fewer threads saves a little memory but doubles the latency. There's
no setting where the cascade gets near our numbers.

### What this means per board

**These are projections, not Pi measurements.** They assume a Pi 5 core is
about 2–3× slower than the M-series Mac core measured here, and a Pi Zero 2 W
core about 8–15× slower. `scripts/benchmark_pi.py` and
`scripts/measure_footprint.py` measure the real numbers on a board.

| Board | Ours | ASR cascade |
|---|---|---|
| Pi 5, 8 GB | ~5–10 ms per command | Works: ~1–2.5 s per command |
| Pi 5 or Pi 4, 2 GB | Easy | Fits, but takes a quarter to a third of RAM, and each command takes seconds |
| Pi Zero 2 W, 512 MB | Fits (~96 MB), ~50 ms per command | **Doesn't fit.** Its 478 MB+ peak leaves no room for the OS on 512 MB, and each command would take ~5–10 s even if it ran |

## How this was measured

- **Machine:** Apple M-series MacBook Air, macOS, Python 3.11,
  onnxruntime 1.30, faster-whisper 1.2.1 (CTranslate2 4.8.2).
- **Commands:** the same 12 spoken commands for both pipelines, 16 kHz
  clips generated with macOS `say`. **Both pipelines got all 12 right**,
  so accuracy numbers come from the test-set evaluations in
  EXPERIMENTS.md, not from these clips.
- **Memory:** resident memory of a fresh process at each stage (`ps`),
  plus the OS-reported peak. **Peak is the reliable number:** macOS
  compresses idle memory, so a stage can even show memory going down, and
  repeated runs vary by roughly ±10%.
- **Latency:** wall time per command from audio to intent, median over the
  12 clips, excluding the first (warm-up) call.
- **Our models:** the intent model is Experiment 31 (the Experiment 32
  slot heads add ~10.6K parameters, about +40 KB). The wake-word model has
  its final architecture and size, but its weights come from a toy
  smoke-test run, which doesn't affect memory or speed.
- **Package sizes:** installed size in site-packages. Ours comes from a
  clean environment built from `requirements-pi.txt`; the ASR stack's is
  the sum of faster-whisper and its dependencies plus scikit-learn and
  scipy for the classifier.

To reproduce (use a separate process for each):
```bash
python scripts/measure_footprint.py ours --intent-model models/vcm_intent.int8.onnx --wake-model models/kiwi_wakeword.int8.onnx --clips <dir of 16 kHz wavs>
python scripts/measure_footprint.py asr --compute-type int8 --threads 4 --clips <dir of 16 kHz wavs>
```
The ASR run needs `pip install faster-whisper scikit-learn joblib`, plus
`checkpoints/cascade_classifier.joblib` from Experiment 26.
