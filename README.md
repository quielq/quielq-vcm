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

## What's deliberately mocked (per Section 8)

- **Push-to-talk button**: spacebar-hold stands in for the physical pushbutton.
- **Temperature reading**: a fixed 27.0C stands in for the Sense HAT sensor.

Both live behind `vcm/hal/`, with an RPi implementation already written
(`gpiozero`/`sense_hat`) ready to activate once the hardware and `VCM_PLATFORM=rpi`
(or auto-detection on the actual Pi) select it.

## Not in this pass

Model training/quantization and the benchmark harness (Sections 5, 7) are
still separate later workstreams — `vcm/inference/model.py` has a
`TFLiteIntentModel` interface ready for when a trained model exists, but
nothing trains one yet. **See [MODEL.md](MODEL.md)** for the technology
choices and research this training pipeline will be built on once that
work starts. Dataset development itself (Section 9) has started —
see the "Dataset pipeline" section above and [DATASET.md](DATASET.md).
For Raspberry Pi hardware setup and testing (unverified, prospective
instructions), see [TESTING.md](TESTING.md) Part 3.
