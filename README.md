# VCM — Voice Command Model (AI222 / AI231 Machine Exercise)

Tiny, on-device spoken-intent classifier. See [VCM_Architecture_Review.md](VCM_Architecture_Review.md)
for the full design rationale — this README only covers running the code.

## What's here vs. what's not (yet)

This is the **Tier 1 software skeleton** (Section 8's "laptop simulation" plan): a
working hardware-abstraction boundary, real audio capture/feature extraction, real
TTS, real Xiaomi bulb/plug control, real music/weather integrations, and a stub
inference backend (no trained model exists yet — that's separate dataset/DGX
training work, Section 5/9).

**Environment note**: this skeleton was built and unit-tested in a Linux dev
sandbox, not the Mac M2 the architecture doc targets. The tests below run
anywhere with no hardware. Everything that needs a real microphone, real
speaker output, or the real Xiaomi devices on your home LAN needs to be run
and verified **by you, on your Mac** — see "Verify on your Mac" below.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp configs/settings.example.toml configs/settings.toml
```

Fill in `configs/settings.toml` (gitignored) with your real values:
- Xiaomi bulb/plug host + token, obtained via `miiocli cloud` (Section 4) — and
  confirm `bulb_device_class`/`plug_device_class` actually match your hardware
  (Section 4 flags the exact bulb family as unconfirmed; check python-miio's
  supported-device list).
- An OpenWeatherMap API key (free tier) for `[weather].api_key`.
- `[music].media_dir` pointed at a local folder of mp3/m4a/flac/wav/ogg files.

## Testing

```bash
python -m pytest
```

Use `python -m pytest`, not bare `pytest` — on macOS a system/Homebrew Python
can put its own `pytest` earlier on `PATH` than `.venv/bin/pytest`, so the
bare command silently runs against the wrong interpreter and fails with
`ModuleNotFoundError` for dependencies that are actually installed. `python
-m pytest` always resolves through the active venv.

61 tests, all pure logic, run anywhere (no microphone, network, or Xiaomi
device required). For manual verification of the parts that actually need
real hardware/network — mic capture, TTS audio, the Xiaomi bulb/plug,
weather, music — see **[TESTING.md](TESTING.md)**, which walks through each
one with exact commands, expected output, and troubleshooting.

## Dataset pipeline

Dataset development (Section 9) has started: a taxonomy loader, a
verified SLURP coverage check against real data, an FSC loader, and a
generator-agnostic synthetic-audio QA gate, all in `src/vcm/dataset/`.
See **[DATASET.md](DATASET.md)** for exact commands to reproduce every
step from a fresh clone, including expected output.

## Deployment

**[STARTUP.md](STARTUP.md)** is the day-to-day runbook (what to start on the Pi and the Mac). **[DEPLOYMENT.md](DEPLOYMENT.md)** covers running the voice pipeline
("Hey Kiwi" wake word → command → intent + slot value) on a Raspberry Pi 5
over SSH, with no monitor: flashing, microphone checks, benchmarking,
field-testing the wake word, and starting on boot. The same
`scripts/vcm_listen.py` runs on a laptop for testing. Commands act through
`vcm.home.server`, which also serves a live web dashboard (virtual lamp and
thermostat, reminders, timers, alarms, music, calls): see DEPLOYMENT.md
step 8b. Open work is
tracked in **[TODO.md](TODO.md)**.

## Training

Training runs on a shared DGX node. **[TRAINING.md](TRAINING.md)** is
the runbook: picking a GPU, packing several runs per GPU, pinning
threads so DataLoader workers don't oversubscribe the node, `nohup`
launch scripts, multi-seed evaluation, and where checkpoints and logs
live. Results for every run are in **[EXPERIMENTS.md](EXPERIMENTS.md)**.

## What's deliberately mocked (per Section 8)

- **Push-to-talk button**: spacebar-hold stands in for the physical pushbutton.
- **Temperature reading**: a fixed 27.0C stands in for the Sense HAT sensor.

Both live behind `vcm/hal/`, with an RPi implementation already written
(`gpiozero`/`sense_hat`) ready to activate once the hardware and `VCM_PLATFORM=rpi`
(or auto-detection on the actual Pi) select it.

## Two model pipelines, both kept

An ML-engineering review (MODEL.md Section 9) researching commercial
voice assistants and published SLU benchmarks found that Siri, Alexa,
and Google Assistant all classify intent from a **text transcript**
(ASR → NLU), not raw audio. **Important compliance note**: this
assignment explicitly states "ASR models are not desirable for
on-device computing because of footprint" and requires the VCM to be
tiny — so the ASR-cascade below is kept as a **backup/reference
option only**, not a candidate for the actual deployed VCM. See
MODEL.md Section 10 for the full comparison. Three things exist now:

- **Direct audio → intent** (the deployable pipeline): a CRNN
  (107K params, **426 KB** fp32 ONNX) classifies a log-mel spectrogram
  directly into one of 20 intents, plus a slot value for TIMER, ALARM,
  BRIGHTNESS and COLOR. **85.48% real-speech test accuracy** (Experiment
  34; slot heads trained on a frozen Experiment 31 encoder). A 25K-param
  "Hey Kiwi" wake word (107 KB) listens in front of it.
  `scripts/vcm_listen.py` (wake word → command) or `scripts/demo_infer.py`
  (push-to-talk). Earlier DS-CNN models (137.5 KB, ~75% val, Experiments
  15 and 27) are kept for reference.
- **ASR-cascade** (backup/reference only, not for deployment):
  `faster-whisper` transcribes audio to text, then a TF-IDF + logistic
  regression classifier maps the transcript to an intent. **~144 MB —
  ~340x the direct intent model's size by disk, ~690x by parameter
  count** — but **90.62% test accuracy on real audio** vs. the direct
  pipeline's 85.5%, the accuracy ceiling for systems that can
  accommodate the footprint. See EXPERIMENTS.md Experiment 26.
  `scripts/demo_infer_cascade.py`.

Export and deployment are done: ONNX export (`scripts/export_onnx.py`),
a numpy + onnxruntime runtime (`vcm/deploy/`), a Pi benchmark
(`scripts/benchmark_pi.py`) and a headless Pi guide (DEPLOYMENT.md).
fp32 ships; int8 cost 6.8 points of accuracy to save 134 KB.
**See [EXPERIMENTS.md](EXPERIMENTS.md)** for every real training-run
result (34 experiments so far: architectures, capacity, augmentation,
data volume, dataset quality, LR schedules, loss functions, and the
cascade pivot) and **[MODEL.md](MODEL.md)** for the technology choices
and research behind both pipelines.
Dataset development itself (Section 9) has started — see the "Dataset
pipeline" section above and [DATASET.md](DATASET.md).
For Raspberry Pi hardware setup and testing (unverified, prospective
instructions), see [TESTING.md](TESTING.md) Part 3.
