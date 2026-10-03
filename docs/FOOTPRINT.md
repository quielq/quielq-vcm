# Footprint: storage and memory

How much SD card space and RAM this project needs, measured rather than
estimated. Part 1 is the device, Part 2 compares our pipeline with the ASR
cascade we chose not to ship, and Part 3 covers the development and training
machines.

**Short answer:** on the Raspberry Pi the project takes about **0.3 GB of the
SD card** (0.24 GB without the optional Piper voice) and **~150 MB of RAM
with the espeak-ng voice, ~300 MB with the Piper voice** while running. The models themselves are **1.57 MB** (intent 1.46 MB +
wake word 107 KB). Everything
else in the repo (datasets, checkpoints, PyTorch) stays on the training
machine and never goes to the Pi.

## Part 1: on the Raspberry Pi

Measured on the demo device: Raspberry Pi 5 (8 GB), 32 GB microSD,
Raspberry Pi OS 12 (bookworm) 64-bit with the desktop, both services
running (`deploy_pi.sh --services`). The listener and benchmark numbers are
from 2026-10-02 with the shipped models; the disk and home-server numbers
from 2026-09-29, which the model files don't affect beyond their own size.

### SD card

`scripts/deploy_pi.sh` copies only the code, the two models and the
settings. It skips `docs/`, `data/`, `checkpoints/`, `logs/` and `.git`.

| What | Size | Needed? |
|---|---:|---|
| Models (`vcm_intent.onnx` + `kiwi_wakeword.onnx`) | **1.57 MB** | Yes |
| Code (`src/`, `scripts/`, `tests/`, `deploy/`, `configs/`) | ~1 MB | Yes |
| Python environment (`.venv`) | 233 MB | Yes (see below) |
| Piper voice (`models/tts/en_US-lessac-medium.onnx`) | 62 MB | Only for the natural voice; espeak-ng needs none |
| **Project total (`~/quielq-vcm`)** | **~304 MB** | |
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
| Voice loop (`vcm_listen.py`: microphone, wake word, intent model) | **~100 MB** on the Pi (99.5 MB RSS running; 99 MB peak in `benchmark_pi.py`) |
| of which the two models | ~10 MB (estimate: 8.5 MB measured with a 1 MB smaller intent model) |
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

| | Shipped (Experiment 43b, 1.46 MB) |
|---|---:|
| Per command, p50 / p95 | **13.9 / 15.2 ms** (features 3.8 ms + model 10.1 ms), RTF 0.0061 |
| Wake word, per 100 ms hop | 1.9 ms, so **1.9% of one core**, always on |
| Peak memory (listener) | 99 MB |
| System memory used, both services running | 719 MB of 8 GB |

## Part 2: our pipeline vs. an ASR cascade

This puts numbers on the assignment's point that "ASR models are not
desirable for on-device computing because of footprint". The cascade was
built on the project's earlier dataset and not rebuilt for the master
dataset, so this compares footprint, not accuracy (the accuracy comparison
is in [EXPERIMENTS.md](EXPERIMENTS.md#asr-cascade-experiment-26)).

| | Pipeline | What runs per command |
|---|---|---|
| **Ours** | "Hey Kiwi" wake word + CRNN intent/slot model, fp32 ONNX | log-mel features → 372K-param CRNN → intent + slot value |
| **ASR cascade** | faster-whisper `base` + TF-IDF/logistic-regression text classifier | audio → Whisper transcript → text classifier → intent |

### Summary

| | **Ours** | **ASR cascade** | Ratio |
|---|---:|---:|---:|
| Model files | **1.57 MB** (intent 1.46 MB + wake word 107 KB, fp32) | **~149 MB** (Whisper 145 MB + tokenizer 2 MB + classifier 2 MB) | ~95× |
| Peak memory | **~100 MB** (on the Pi) | **478–696 MB** (laptop) | 5–7× |
| Minimum board RAM (with OS, ~60–100 MB) | **512 MB** | **1 GB+** | |
| Latency per command | **15.2 ms p95 on the Pi 5** | **440–950 ms on a laptop**; ~1–2.5 s on a Pi 5 (projected) | ~65–165× on the Pi 5 |
| Python packages to install | ~149 MB: numpy, onnxruntime, sounddevice | ~330 MB: adds faster-whisper, CTranslate2, PyAV, tokenizers, scikit-learn, scipy | ~2× |
| Works as the always-on wake word? | Yes, 1.9% of one Pi 5 core | **No.** Scoring a 1.5 s window every 0.1 s would need ~4.6 s of compute per second of audio | |

**In short:** the cascade needs roughly **95× the disk, 5–7× the memory and
two orders of magnitude more compute per command**, and it would *still*
need a separate wake-word model in front of it, because Whisper is far too
slow to listen continuously. This is why the direct audio-to-intent CRNN
ships.

**Why our memory is small.** The models account for about 10 MB once
loaded (estimated); the rest is Python, numpy, ONNX Runtime and working buffers.
Features used to be computed with librosa, which pulls in numba, LLVM and
scipy: importing and using it once added **+186 MB**, and the whole
pipeline peaked at ~330 MB. A numpy re-implementation (`vcm/audio/dsp.py`,
tested to match librosa exactly) brought that to ~100 MB. A stage-by-stage
breakdown, measured with an earlier model, is in
[EXPERIMENTS.md](EXPERIMENTS.md#footprint-with-the-experiment-36-models-2026-09-29).

### ASR cascade (faster-whisper base, laptop)

| Setup | Loading Whisper adds | Peak memory | Latency per command |
|---|---:|---:|---:|
| The cascade demo's defaults, 4 threads | +373 MB | 696 MB | 588 ms |
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
| Pi 5, 8 GB | **15.2 ms p95 per command, 99 MB** (measured) | Works: ~1–2.5 s per command |
| Pi 5 or Pi 4, 2 GB | Easy | Fits, but takes a quarter to a third of RAM, and each command takes seconds |
| Pi Zero 2 W, 512 MB | Fits (~100 MB), roughly 50–80 ms per command | **Doesn't fit.** Its 478 MB+ peak leaves no room for the OS, and each command would take ~5–10 s even if it ran |

### How Part 2 was measured

- **Machine:** Apple M-series MacBook Air, macOS, Python 3.11,
  onnxruntime 1.30, faster-whisper 1.2.1 (CTranslate2 4.8.2).
- **Commands:** 12 spoken commands, 16 kHz clips generated with macOS
  `say`. The cascade got all 12 right.
- **Memory:** resident memory of a fresh process at each stage (`ps`), plus
  the OS-reported peak. Peak is the reliable number: macOS compresses idle
  memory, so repeated runs vary by roughly ±10%.
- **Latency:** wall time per command from audio to intent, median over the
  12 clips, excluding the first (warm-up) call.
- **Package sizes:** installed size in site-packages. Ours comes from a
  clean environment built from `requirements-pi.txt`; the ASR stack's is
  the sum of faster-whisper and its dependencies plus scikit-learn and
  scipy for the classifier.
- **Our numbers** are the shipped models on the Raspberry Pi 5
  ([results/bench_pi5.md](../results/bench_pi5.md)).

To reproduce (a separate process for each):
```bash
python scripts/measure_footprint.py ours --intent-model models/vcm_intent.onnx --wake-model models/kiwi_wakeword.onnx --clips <dir of 16 kHz wavs>
python scripts/measure_footprint.py asr --compute-type int8 --threads 4 --clips <dir of 16 kHz wavs>
```
The ASR run needs `pip install faster-whisper scikit-learn joblib`, plus
the cascade's classifier, `checkpoints/cascade_classifier.joblib`
(trained on the earlier dataset; see EXPERIMENTS.md).

## Part 3: development and training machines

None of this goes to the Pi.

| What | Where | Size | In git? |
|---|---|---:|---|
| Everything tracked in git (code, docs, results, ONNX models) | repo | a few MB | Yes |
| Class master dataset (`/data/ai231` shared copy, or the pinned download in `data/me2/hf`) | DGX | ~3.6 GB | No: downloaded or read from the shared copy ([DATASET.md](DATASET.md)) |
| Manifest, slot labels, metadata, distillation labels (`data/me2/*.csv`) | DGX | ~10–30 MB | No: regenerated by `scripts/build_me2_manifest.py` |
| Training environment (PyTorch + CUDA) | DGX `.venv` | ~6.1 GB | No |
| Checkpoints and logs | DGX `checkpoints/`, `logs/` | ~30 MB | No |
| Development environment (PyTorch CPU, librosa, tests) | laptop `.venv` | 1.4 GB (torch alone 587 MB) | No |

The scripts and tests are small. Most scripts run once, on the training
machine, to build data, train or evaluate; only
`vcm_listen.py`, `kiwi_doctor.py`, `benchmark_pi.py` and
`measure_footprint.py` are meant to run on the Pi. See the README's
"Codebase structure" for which is which.

**GPU memory while training:** ~2.5 GB per run on an A100-40GB, so 3–5 runs
share one GPU ([TRAINING.md](TRAINING.md)).
