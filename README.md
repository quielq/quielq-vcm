# VCM — Voice Command Model (AI222 Machine Exercise)

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

## Run the test suite (works anywhere, no hardware needed)

```bash
pytest
```

18 tests, all pure logic — feature-extraction shape/dtype on synthetic audio,
dispatch-table completeness, HAL backend selection, config loading. No
microphone, network, or Xiaomi device required.

## Verify on your Mac

These need real hardware/network and can't be confirmed from this sandbox:

1. **End-to-end push-to-talk loop**:
   ```bash
   python -m vcm.main
   ```
   Hold spacebar, speak, release. You should see `Heard intent: unknown_background`
   printed (the stub model always returns that) — this confirms mic capture and
   feature extraction ran for real. macOS will prompt for microphone permission,
   and `pynput`'s keyboard listener needs Accessibility/Input Monitoring
   permission granted to your terminal app.

2. **TTS**: `python -c "from vcm.tts.speak import speak; speak('hello from the VCM')"`
   should be audible immediately via macOS's built-in `say`.

3. **Xiaomi bulb/plug**, once `configs/settings.toml` has a real host/token:
   ```bash
   python -c "from vcm.actions import lights; lights.turn_on()"
   python -c "from vcm.actions import thermostat; print(thermostat.adjust_to_comfort())"
   ```

4. **Weather**, once `[weather].api_key` is set:
   ```bash
   python -c "from vcm.actions import weather; print(weather.get_weather())"
   ```

5. **Music**, once `[music].media_dir` has audio files (requires `mpv` —
   `brew install mpv`):
   ```bash
   python -c "from vcm.actions import music; music.play()"
   python -c "from vcm.actions import media_control; media_control.pause()"
   ```

## What's deliberately mocked (per Section 8)

- **Push-to-talk button**: spacebar-hold stands in for the physical pushbutton.
- **Temperature reading**: a fixed 27.0C stands in for the Sense HAT sensor.

Both live behind `vcm/hal/`, with an RPi implementation already written
(`gpiozero`/`sense_hat`) ready to activate once the hardware and `VCM_PLATFORM=rpi`
(or auto-detection on the actual Pi) select it.

## Not in this pass

Dataset acquisition, model training/quantization, and the benchmark harness
(Sections 5, 7, 9) are separate later workstreams — `vcm/inference/model.py`
has a `TFLiteIntentModel` interface ready for when a trained model exists, but
nothing trains one yet.
