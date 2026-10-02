# Footprint: storage and memory

> **Model update (Experiment 41d, 2026-10-02).** The measurements below were
> taken with the Experiment 36 models (intent 432 KB + wake word 107 KB =
> 539 KB). The current intent model, trained on the class master dataset,
> is 1.46 MB (1.57 MB with the wake word; `vcm_intent_small.onnx` keeps it
> at 0.83 MB). Measured on the Pi 5 on 2026-10-02: 13.9 ms p50 / 15.8 ms p95
> per command (3.8 ms features + 10.0 ms model; the Experiment 36 model took
> 9.9 ms), wake word 1.9% of one core, listener 95 MB RSS, home server 47 MB,
> 604 MB used system-wide with both services
> ([results/bench_pi5.md](../results/bench_pi5.md)). The SD-card and package
> numbers below are unchanged apart from the model files.

How much SD card space and RAM this project needs, measured rather than
estimated. Part 1 is the device, Part 2 compares our pipeline with the ASR
cascade we chose not to ship, and Part 3 covers the development and training
machines.

**Short answer:** on the Raspberry Pi the project takes about **0.3 GB of the
SD card** (0.24 GB without the optional Piper voice) and **~150 MB of RAM
with the espeak-ng voice, ~300 MB with the Piper voice** while running. The models themselves are **539 KB**. Everything
else in the repo (datasets, checkpoints, PyTorch) stays on the training
machine and never goes to the Pi.

## Part 1: on the Raspberry Pi

Measured on the demo device on 2026-09-29: Raspberry Pi 5 (8 GB), 32 GB
microSD, Raspberry Pi OS 12 (bookworm) 64-bit with the desktop, both services
running (`deploy_pi.sh --services`).

### SD card

`scripts/deploy_pi.sh` copies only the code, the two models and the
settings. It skips `docs/`, `data/`, `checkpoints/`, `logs/` and `.git`.

| What | Size | Needed? |
|---|---:|---|
| Models (`vcm_intent.onnx` + `kiwi_wakeword.onnx`) | **0.53 MB** | Yes |
| Code (`src/`, `scripts/`, `tests/`, `deploy/`, `configs/`) | ~1 MB | Yes |
| Python environment (`.venv`) | 233 MB | Yes (see below) |
| Piper voice (`models/tts/en_US-lessac-medium.onnx`) | 62 MB | Only for the natural voice; espeak-ng needs none |
| **Project total (`~/quielq-vcm`)** | **303 MB** | |
| System packages we install (espeak-ng, raspotify, portaudio, sense-hat, tmux) | ~34 MB | raspotify (20 MB) only for Spotify |
| Saved commands (`~/kiwi_commands`, `--save-commands`) | 22 MB for 319 commands (~70 KB each) | Optional, for debugging; grows with use |
| pip download cache (`~/.cache/pip`) | 84 MB | No, safe to delete |

What's in the 233 MB Python environment:

| Package | Size | Used for |
|---|---:|---|
| onnxruntime | 60 MB | Running both models |
| numpy (+ its bundled OpenBLAS) | 72 MB | Features, audio |
| piper-tts | 47 MB | Natural voice (optional) |
| pillow | 24 MB | Sense HAT library (optional) |
| pip, setuptools | 22 MB | Installing |
| requests, sounddevice and small dependencies | ~10 MB | Web APIs, microphone |

**The operating system dominates.** The card has 16 GB used, almost all of
it the full desktop OS and unrelated software (Wolfram 5.2 GB, VS Code
server 1.2 GB, Scratch 0.4 GB). With Raspberry Pi OS **Lite** (~2–3 GB
installed) instead, an **8 GB card** holds everything with room to spare.

### RAM

| Process | Peak memory (measured) |
|---|---:|
| Voice loop (`vcm_listen.py`: microphone, wake word, intent model) | **103 MB** on the Pi (94 MB in `benchmark_pi.py`) |
| of which the two models | ~8.5 MB |
| Home server + dashboard (`vcm.home.server`) with the **Piper** voice loaded | **185 MB** |
| Home server with **espeak-ng** instead | ~30 MB |
| raspotify (Spotify speaker) | 13 MB |
| **Our total** | **~300 MB with Piper, ~150 MB with espeak-ng** |
| Raspberry Pi OS Lite, idle | ~60–100 MB |

The Pi 5 has 8 GB, so this is under 4% of it. The smallest board that fits
is **512 MB** (Pi Zero 2 W or Pi 3 A+) using espeak-ng: ~150 MB for us plus
the OS leaves about half the memory free. Piper's voice model alone is the
difference between the two home-server numbers.

### Speed on the Pi 5

`scripts/benchmark_pi.py` on the device, 1 thread
([results/bench_pi5.md](../results/bench_pi5.md)):

| | Shipped (Experiment 41d, 1.46 MB) | Previous (Experiment 36, 432 KB) |
|---|---:|---:|
| Per command, p50 / p95 | **13.9 / 15.8 ms** (features 3.8 ms + model 10.0 ms), RTF 0.0063 | 9.9 ms (features 3.7 ms + model 6.2 ms) |
| Wake word, per 100 ms hop | 1.9 ms, so **1.9% of one core**, always on | 1.9 ms, 2% of one core |
| Peak memory (listener) | 100 MB | 94 MB |

## Part 2: our pipeline vs. an ASR cascade

This puts numbers on the assignment's point that "ASR models are not
desirable for on-device computing because of footprint", and on what the
extra accuracy of the ASR cascade (Experiment 26) would cost.

| | Pipeline | What runs per command |
|---|---|---|
| **Ours** | "Hey Kiwi" wake word + CRNN intent/slot model, fp32 ONNX | log-mel features → 108K-param CRNN → intent + slot value |
| **ASR cascade** | faster-whisper `base` + TF-IDF/logistic-regression text classifier | audio → Whisper transcript → text classifier → intent |

### Summary

| | **Ours** | **ASR cascade** | Ratio |
|---|---:|---:|---:|
| Model files | **539 KB** (intent 432 + wake word 107, fp32) | **~149 MB** (Whisper 145 MB + tokenizer 2 MB + classifier 2 MB) | ~280× |
| Peak memory | **88–103 MB** | **478–696 MB** | 5–7× |
| Minimum board RAM (with OS, ~60–100 MB) | **512 MB** | **1 GB+** | |
| Memory for the models once loaded | ~8.5 MB | ~290–370 MB | ~40× |
| Latency per command, laptop CPU | **3.1–3.4 ms** | **440–950 ms** | ~150–300× |
| Latency per command, Pi 5 | **15.8 ms p95** (measured, Experiment 41d; Experiment 36: 9.9 ms) | ~1–2.5 s (projected) | |
| Python packages to install | ~149 MB: numpy, onnxruntime, sounddevice | ~330 MB: adds faster-whisper, CTranslate2, PyAV, tokenizers, scikit-learn, scipy | ~2× |
| Real-speech test accuracy | 84.8% (Experiment 36) | **90.6%** (Experiment 26) | ASR +5.8 points |
| Works as the always-on wake word? | Yes, 2% of one Pi 5 core | **No.** Scoring a 1.5 s window every 0.1 s would need ~4.6 s of compute per second of audio | |

**In short:** the ASR cascade buys about 6 accuracy points for roughly
**280× the disk, 5–7× the memory and 150–300× the latency**, and it would
*still* need a separate wake-word model in front of it, because Whisper is
far too slow to listen continuously. This is why the direct audio-to-intent
CRNN ships.

### Our pipeline, stage by stage (laptop, 1 thread)

Measured first with int8 files. The deployed fp32 files measure the same:
88–89 MB peak, 3.1 ms per command. int8 isn't used: it cost the intent model
6.8 points of real-speech accuracy (Experiment 32) to save 134 KB, and on
this CPU it isn't faster (dynamic quantization converts activations at run
time).

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

The models account for about 8.5 MB. The running overhead is Python and numpy
working memory (feature arrays and buffers): turning off ONNX Runtime's
memory arena made no measurable difference (84.7–86.0 MB vs 85.8–94.8 MB,
within run-to-run noise).

Removing librosa mattered here. Features used to be computed with librosa,
which pulls in numba, LLVM and scipy: importing and using it once added
**+186 MB**, and the whole pipeline peaked at ~330 MB. A numpy
re-implementation (`vcm/audio/dsp.py`, tested to match librosa exactly)
brought that to ~90 MB.

### ASR cascade (faster-whisper base, laptop)

| Setup | Loading Whisper adds | Peak memory | Latency per command |
|---|---:|---:|---:|
| The Experiment 26 demo's defaults, 4 threads | +373 MB | 696 MB | 588 ms |
| int8 weights, 4 threads (the most favorable setup) | +322 MB | 501–558 MB | 437–460 ms |
| int8 weights, 1 thread | +287 MB | 478 MB | 946 ms |

Using fewer threads saves a little memory but doubles the latency. There's
no setting where the cascade gets near our numbers.

### Per board

The Pi 5 row for our pipeline is measured; the rest are projections from
the laptop numbers (a Pi 5 core is ~2–3× slower than the M-series core
measured here, a Pi Zero 2 W core ~8–15× slower).

| Board | Ours | ASR cascade |
|---|---|---|
| Pi 5, 8 GB | **15.8 ms p95 per command, 100 MB** (measured, Experiment 41d) | Works: ~1–2.5 s per command |
| Pi 5 or Pi 4, 2 GB | Easy | Fits, but takes a quarter to a third of RAM, and each command takes seconds |
| Pi Zero 2 W, 512 MB | Fits (~100 MB), ~80 ms per command | **Doesn't fit.** Its 478 MB+ peak leaves no room for the OS, and each command would take ~5–10 s even if it ran |

### How Part 2 was measured

- **Machine:** Apple M-series MacBook Air, macOS, Python 3.11,
  onnxruntime 1.30, faster-whisper 1.2.1 (CTranslate2 4.8.2).
- **Commands:** the same 12 spoken commands for both pipelines, 16 kHz
  clips generated with macOS `say`. Both pipelines got all 12 right, so
  accuracy numbers come from the test-set evaluations in EXPERIMENTS.md, not
  from these clips.
- **Memory:** resident memory of a fresh process at each stage (`ps`), plus
  the OS-reported peak. Peak is the reliable number: macOS compresses idle
  memory, so a stage can even show memory going down, and repeated runs vary
  by roughly ±10%.
- **Latency:** wall time per command from audio to intent, median over the
  12 clips, excluding the first (warm-up) call.
- **Our models:** the stage-by-stage table used Experiment 31's intent model
  (int8) and a same-size toy wake-word model; the fp32 re-measurement used
  the Experiment 32 intent + slot model and Experiment 33 wake word. The
  shipped Experiment 36 model has the same architecture plus two small slot
  heads (432 KB vs 426 KB), so these numbers carry over. The Pi 5 numbers
  above use the shipped models.
- **Package sizes:** installed size in site-packages. Ours comes from a
  clean environment built from `requirements-pi.txt`; the ASR stack's is
  the sum of faster-whisper and its dependencies plus scikit-learn and
  scipy for the classifier.

To reproduce (a separate process for each):
```bash
python scripts/measure_footprint.py ours --intent-model models/vcm_intent.onnx --wake-model models/kiwi_wakeword.onnx --clips <dir of 16 kHz wavs>
python scripts/measure_footprint.py asr --compute-type int8 --threads 4 --clips <dir of 16 kHz wavs>
```
The ASR run needs `pip install faster-whisper scikit-learn joblib`, plus
`checkpoints/cascade_classifier.joblib` from Experiment 26.

## Part 3: development and training machines

None of this goes to the Pi.

| What | Where | Size | In git? |
|---|---|---:|---|
| Everything tracked in git (code, docs, 4 ONNX models) | repo | ~2 MB (`.git` history 8 MB) | Yes |
| Training datasets (`data/external/`, 6 sources + synthetic batches) | DGX | ~4.5 GB | No: re-fetched by the `scripts/fetch_*` scripts |
| Manifests, slot labels, Whisper transcripts (`data/*.csv`) | DGX | ~10–30 MB | No: regenerated |
| Training environment (PyTorch + CUDA) | DGX `.venv` | ~6.1 GB | No |
| Checkpoints and logs | DGX `checkpoints/`, `logs/` | ~30 MB | No |
| Development environment (PyTorch CPU, librosa, tests) | laptop `.venv` | 1.4 GB (torch alone 587 MB) | No |

The 29 scripts and 26 test files are small (150 KB and 80 KB of source).
Most scripts run once, on the training machine, to fetch or build data; only
`vcm_listen.py`, `kiwi_doctor.py`, `benchmark_pi.py` and
`measure_footprint.py` are meant to run on the Pi. See the README's
"Codebase structure" for which is which.

**GPU memory while training:** ~2.8 GB per run on an A100-40GB, so 3–5 runs
share one GPU ([TRAINING.md](TRAINING.md)).
