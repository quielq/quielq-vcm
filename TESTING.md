# Testing Manual

Two layers of testing exist for this codebase, and they cover different things:

| Layer | Runs where | Confirms |
|---|---|---|
| **Automated test suite** (`pytest`) | Anywhere — this sandbox, your Mac, CI | Logic correctness: shapes, dtypes, dispatch completeness, config/platform selection |
| **Manual verification** (this doc, Part 2) | Your Mac only | Real behavior: the mic actually captures, the speaker actually plays, the bulb actually turns on |

Neither layer tests the trained model itself — there isn't one yet (Section 9/5
of the architecture review are separate later work). The inference stub always
returns `unknown_background`, so every command today is a no-op through
dispatch by default; the *pipeline* is what's being verified, not
classification accuracy.

---

## Part 1 — Automated test suite

### Setup (one-time)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run everything

```bash
python -m pytest
```

Expected: `61 passed` in a few seconds. No microphone, speaker, network, or
Xiaomi device is touched — every test uses synthetic input (`np.random`
arrays, monkeypatched env vars, fake injected modules for `gpiozero`/`sense_hat`).

Always use `python -m pytest`, not the bare `pytest` command. On macOS a
system or Homebrew Python can install its own `pytest` earlier on `PATH`
than `.venv/bin/pytest` — running the bare command then silently executes
against the *wrong* interpreter and fails with `ModuleNotFoundError` for
packages that are actually installed in `.venv`. Confirm with `which pytest`;
if it doesn't point inside `.venv/bin/`, that's the shadowing. `python -m
pytest` sidesteps this entirely by always resolving through whichever
`python` is active.

### Run a subset

```bash
python -m pytest tests/test_features.py          # one file
python -m pytest tests/test_dispatch.py -v       # verbose, one file
python -m pytest -k "platform"                   # any test with "platform" in its name
```

### What each file covers

| File | Covers | Why it doesn't need hardware |
|---|---|---|
| [tests/test_config.py](tests/test_config.py) | `VCM_PLATFORM` env override (valid/invalid), auto-detection falls back sanely on this Linux sandbox, settings-file fallback to the example file | Pure string/env logic |
| [tests/test_features.py](tests/test_features.py) | Log-mel output shape is always `(N_MELS, n_frames)` regardless of input clip length, dtype is `float32`, silence doesn't produce NaN/Inf | Feeds `np.random`/`np.zeros` arrays directly into `extract_log_mel()`, bypassing the mic entirely |
| [tests/test_dispatch.py](tests/test_dispatch.py) | Every taxonomy label has a registered handler, `unknown_background` is a safe no-op, an unregistered label raises `KeyError` | Only exercises the `unknown_background` handler for real (the rest would need live hardware/network to actually run — see Part 2) |
| [tests/test_hal_selection.py](tests/test_hal_selection.py) | `get_button()`/`read_temperature()` pick the mac vs. rpi backend correctly | Patches the backend *classes* themselves (via `unittest.mock`) or injects a fake `sense_hat` module into `sys.modules`, so no real GPIO/keyboard listener is constructed |
| [tests/test_inference_stub.py](tests/test_inference_stub.py) | Default model is the stub and returns `unknown_background`; `StubIntentModel` rejects an unknown label; `RandomIntentModel` always returns a label from the taxonomy | Pure in-memory model objects |

### Adding a new test

Follow the existing pattern: if a module touches hardware or the network,
test its *logic* (input validation, branching, shape) with fake/mocked
dependencies, not the real device — real-device checks belong in Part 2 below,
not in `pytest`.

---

## Part 2 — Manual verification on your Mac

Do this after `pip install -e ".[dev]"` on your actual Mac (not this sandbox —
it has no audio hardware or LAN access to your Xiaomi devices). Copy the
config template first:

```bash
cp configs/settings.example.toml configs/settings.toml
```

Then fill in `configs/settings.toml` as you go through the sections below —
each one only needs the fields it names.

### 2.1 TTS (no config needed)

```bash
python -c "from vcm.tts.speak import speak; speak('hello from the VCM')"
```
**Expect**: audible speech within ~1s via macOS's built-in `say`.
**If it fails**: `say: command not found` means you're not actually on macOS —
this backend is Mac-only by design (Section 8); use `[tts].backend =
"espeak_ng"` instead (needs `brew install espeak-ng` or equivalent).

### 2.2 Microphone capture + feature extraction (no config needed)

```bash
python -c "
from vcm.audio.capture import record
from vcm.audio.features import extract_log_mel
audio = record(duration_s=2.0)
features = extract_log_mel(audio)
print('captured', audio.shape, 'features', features.shape)
"
```
**Expect**: prints two shapes, e.g. `captured (32000,) features (40, 151)`,
after a 2-second recording window (speak into the mic during it).
**If it fails**:
- `PortAudioError` / no default input device → check System Settings →
  Privacy & Security → Microphone has granted access to your terminal app.
- All-silence input still produces valid (if boring) numbers — this step
  doesn't fail on silence, only on the device being inaccessible.

### 2.3 Full push-to-talk loop (no config needed)

```bash
python -m vcm.main
```
Hold **spacebar**, speak a command, release it.
**Expect**: `VCM ready...` banner, then `Heard intent: unknown_background`
after each hold/release cycle (the stub model always returns this — there's
no trained model yet, so this line just confirms the full pipeline ran:
button → capture → features → inference → dispatch).
**If it fails**:
- No response to spacebar → grant your terminal app Accessibility and Input
  Monitoring permission (System Settings → Privacy & Security), required by
  `pynput`'s keyboard listener.
- Hangs after release → check the mic permission from 2.2.
- Ctrl+C to exit.

### 2.3.1 Exercising a specific intent path

Since the model is a stub, the fastest way to check a specific category's
*action code* (not the classifier) is to call `vcm.dispatch.dispatch()`
directly instead of going through `main.py`:

```bash
python -c "from vcm.dispatch import dispatch; print(dispatch('play_music'))"
```
Valid labels: see `vcm.taxonomy.LABELS`. This is how you'll exercise 2.4–2.6
below without needing the classifier to cooperate.

### 2.4 Xiaomi bulb / plug

Requires `configs/settings.toml` → `[xiaomi]` filled in:
1. Run `miiocli cloud` (one-time, needs your Mi account credentials) to get
   the local `host` + `token` for each device — see Section 4 of the
   architecture review for the full explanation.
2. Set `bulb_device_class` / `plug_device_class` to match your actual
   hardware. If unsure which family your bulb is, try `miio.Yeelight` first;
   if that raises on `.on()`, check python-miio's supported-device list for
   the Xiaomi/Philips co-branded classes instead.

```bash
python -c "from vcm.actions import lights; lights.turn_on()"
python -c "from vcm.actions import lights; lights.set_brightness(50)"
python -c "from vcm.actions import lights; lights.turn_off()"
python -c "from vcm.actions import thermostat; print(thermostat.adjust_to_comfort())"
```
**Expect**: the bulb visibly changes state; `adjust_to_comfort()` prints a
sentence and turns the plug's fan on/off depending on the mock temperature
reading (fixed at 27.0°C on Mac — see `vcm/hal/temperature.py`).
**If it fails**: `MiioNotConfigured` → host/token missing; a raw `miio`
exception on `.on()`/`.set_brightness()` → wrong `*_device_class`, revisit
step 2 above.

### 2.5 Weather

Requires `configs/settings.toml` → `[weather].api_key` (free key from
openweathermap.org) and `default_location`.

```bash
python -c "from vcm.actions import weather; print(weather.get_weather())"
```
**Expect**: a one-line string like `Quezon City,PH: broken clouds, 29C`.
**If it fails**: `WeatherNotConfigured` → api_key missing; an HTTP error →
check the location string format (`City,CountryCode`) or that the key is
active (new OpenWeatherMap keys can take up to ~2 hours to activate).

### 2.6 Music / media control

Requires `mpv` installed (`brew install mpv`) and `configs/settings.toml` →
`[music].media_dir` pointed at a folder containing at least one
`.mp3`/`.m4a`/`.flac`/`.wav`/`.ogg` file.

```bash
python -c "from vcm.actions import music; print(music.play())"
# while it's playing, in another terminal / next command:
python -c "from vcm.actions import media_control; media_control.pause()"
python -c "from vcm.actions import media_control; media_control.resume()"
python -c "from vcm.actions import media_control; media_control.volume_up()"
python -c "from vcm.actions import media_control; media_control.stop()"
```
**Expect**: audible playback starts, then responds to each control command.
**If it fails**: `MpvUnavailable: mpv not found` → install it; `FileNotFoundError`
from `music.play()` → `media_dir` is empty or wrong path.

### 2.7 Timers, alarms, reminders, calls (no config needed)

```bash
python -c "from vcm.actions import timers; timers.set_timer(3, label='Tea')"
# wait 3s — you should hear "Tea is done" via TTS
python -c "from vcm.actions import reminders; reminders.add_reminder('buy milk'); print(reminders.list_reminders())"
python -c "from vcm.actions import calls; calls.call()"
```
**Expect**: timer speaks after the delay; reminder round-trips through
`data/reminders.json` (gitignored); `calls.call()` speaks "Calling Mom".

---

### 2.8 Trained model live demo (real speech, real predictions)

Everything above this point exercises the hardware-abstraction/dispatch
skeleton with `StubIntentModel` (a stand-in that always returns the
same fixed label) — it never actually classifies your voice. This
section runs the **real trained model** from `EXPERIMENTS.md` against
live speech instead, using `scripts/demo_infer.py` (built specifically
for this — see that script's own docstring). This is a standalone
sanity check, separate from `vcm.main`: it doesn't go through
`dispatch.py`/TTS, and its label space is the 20-class training
taxonomy (`vcm.dataset.sources.dataset_schema`), not yet
`vcm.taxonomy.LABELS` — the two haven't been reconciled yet (see
`VCM_Architecture_Review.md`).

**Setup** (one-time, on top of the base `pip install -e ".[dev]"`):
```bash
pip install -e ".[train]"   # pulls in torch, needed to load a checkpoint
```

**Get a checkpoint**: checkpoints are gitignored (regeneratable
training artifacts, not committed) — you'll need one sent to you, or
train your own per `EXPERIMENTS.md`. Place it at
`checkpoints/dscnn_bigcap_confusable2_best.pt` (the current standing
recommendation, 75.45% val accuracy and the best available on the
polarity-confusion problem specifically — see `EXPERIMENTS.md`
Experiment 15 for what "best" means and how it might change), or pass
a different path explicitly.

```bash
python scripts/demo_infer.py
# or, for a specific checkpoint:
python scripts/demo_infer.py --checkpoint checkpoints/<name>.pt
```

**Expect**: it prints something like
`Loaded dscnn from checkpoints/dscnn_bigcap_confusable2_best.pt (epoch 10, val_acc 0.7545, 20 labels)`,
then `Hold spacebar (Mac) or the pushbutton (RPi) and speak a command.
Ctrl+C to quit.` — hold spacebar, say a command, release, and it prints
the top-3 predicted labels with probabilities, e.g.:
```
LIGHT_ON=0.81  LIGHT_OFF=0.12  COLOR=0.04
```

**Tips for getting a meaningful read on it, not just noise**:
- Try phrasing close to the project's own scripted taxonomy first
  (`"play music"`, `"what's the weather"`, `"turn on the lights"`) —
  that's what most of the training data actually sounds like.
  Naturalistic paraphrasing is a much harder test, especially for
  labels documented as SLURP-influenced in `DATASET.md`.
- Per-class accuracy varies a lot (see `EXPERIMENTS.md`'s per-class
  tables) — expect some commands to work reliably and others (WEATHER,
  MESSAGE, PLAY_MUSIC, VOLUME_UP/DOWN) to be noticeably weaker. That's
  documented, expected behavior, not a bug in the demo.
- If macOS blocks the mic or the spacebar listener, that's the same
  Accessibility/Input Monitoring permission issue as the rest of this
  doc — `tccutil reset Accessibility` / `tccutil reset ListenEvent` and
  re-grant to your terminal app.

---

## 2.9 Full Tier-1 smoke-test checklist

Run once after any change that touches multiple modules, or before a demo:

- [ ] `python -m pytest` — 89/89 pass
- [ ] `python -m vcm.main` — hold spacebar, speak, see `Heard intent: unknown_background`
- [ ] TTS audible (2.1)
- [ ] Bulb on/off/dim responds (2.4)
- [ ] Thermostat command runs without raising (2.4)
- [ ] Weather returns a real forecast string (2.5)
- [ ] Music plays and pause/resume/volume/stop all work (2.6)
- [ ] Timer fires and speaks after its delay (2.7)
- [ ] Reminder add/list round-trips (2.7)
- [ ] Mocked call speaks "Calling Mom" (2.7)

---

## Part 3 — Raspberry Pi 5 hardware setup and testing

**Honesty note before anything else**: everything in this Part is written
from the architecture doc's own hardware/circuit sections (Section 4, 10)
and this project's existing HAL design (`vcm/hal/`), not verified against
a physical Raspberry Pi from this session — there's no RPi hardware
reachable from here. Treat this as a runbook to follow and correct once
the actual Cytron kit is in hand, the same way Part 2 is real, tested
instructions and Part 1 is real, passing output. If a step doesn't match
reality on the actual board, that's this doc being wrong, not you.

### 3.1 Flash the OS (out of the box)

What you need: the RPi5 board, official PSU, a microSD card (the Cytron
kit bundles one, pre-loaded — reflashing it is fine and recommended so
you control the OS version), the kit's bundled USB microSD reader, and
[Raspberry Pi Imager](https://www.raspberrypi.com/software/) on another
computer.

1. Open Raspberry Pi Imager, choose **Raspberry Pi 5** as the device and
   **Raspberry Pi OS (64-bit)** as the OS (Section 5 requires 64-bit).
2. Click the gear icon (or `Ctrl+Shift+X`) for **Advanced Options**
   *before* writing, and set: hostname, enable SSH (password or your
   public key), Wi-Fi SSID/password, locale/timezone. This is the
   headless setup path the architecture doc already flagged (Section 4:
   "can skip a monitor entirely via Raspberry Pi Imager's headless
   setup") — no monitor/keyboard needed for the rest of this doc.
3. Write the image, insert the card, power on. First boot takes longer
   than normal (filesystem resize) — give it 1-2 extra minutes.
4. Find it on your network and SSH in:
   ```bash
   ssh <your-username>@<hostname>.local
   ```
   (mDNS `.local` resolution works out of the box from a Mac; if it
   doesn't resolve, check your router's connected-devices list for the
   IP instead.)

**Alternative (monitor-attached) path**: if you'd rather not do headless
setup, use the kit's bundled micro-HDMI→HDMI cable, complete the
on-screen first-boot wizard, then `sudo raspi-config` → Interface
Options → SSH → enable, and proceed with the rest of this doc over SSH
from here.

### 3.2 Base system packages

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-venv python3-pip git portaudio19-dev libatlas-base-dev mpv espeak-ng
```
- `portaudio19-dev` — needed for `sounddevice` (mic capture) to build/import correctly.
- `libatlas-base-dev` — common RPi BLAS backend numpy/scipy expect.
- `mpv`, `espeak-ng` — music playback and TTS backends (same roles as on Mac, see 2.6/2.1).

**A real RPi5-specific gotcha worth knowing before it wastes your time**:
the Pi 5 moved to a new GPIO controller chip, and the classic `RPi.GPIO`
library **does not work on it**. `gpiozero` (already a project
dependency via the `rpi` extra) handles this automatically on a
reasonably recent version by using the `lgpio` pin factory instead — but
if `hal/button.py`'s RPi path throws a pin-factory error, this is the
first thing to check (`pip install -U gpiozero lgpio`).

### 3.3 Clone and set up the project

```bash
git clone https://github.com/quielq/quielq-vcm.git
cd quielq-vcm
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rpi]"
cp configs/settings.example.toml configs/settings.toml
```
The `rpi` extra (`gpiozero`, `sense-hat`) is what activates the real
GPIO/Sense HAT code paths in `vcm/hal/` instead of the Mac mocks.

### 3.4 Configure `settings.toml`

Same fields as Part 2 (Xiaomi host/token, weather API key, media dir),
with one platform-specific change: set `[tts].backend = "espeak_ng"`
(the `mac_say` backend is Mac-only and will fail here). Platform
detection is automatic (`config.py` reads `/proc/device-tree/model`),
confirm it:
```bash
python -c "from vcm.config import get_platform; print(get_platform())"
```
**Expect**: `rpi`. If it prints `mac` instead, force it for testing with
`VCM_PLATFORM=rpi python -c "..."` and check why auto-detection didn't
pick it up (the device-tree model string should contain "raspberry pi").

### 3.5 Wire the pushbutton

Per Section 10's circuit diagram: two jumper wires only, no breadboard
or external resistor needed (the Pi's software pull-up handles it) —
GPIO 17 to one leg of the button, a GND pin to the other leg. The
Cytron kit bundles both the pushbutton and the jumper wires.

### 3.6 Connect audio hardware

Plug in the Mini USB Microphone and the powered speaker (USB or 3.5mm).
Confirm Linux sees them before testing the Python side:
```bash
arecord -l   # should list the USB mic as a capture device
aplay -l     # should list the speaker as a playback device
```
If the wrong device ends up default, `raspi-config` → System Options →
Audio lets you pick the output; for the mic, `sounddevice` picks the
system default input, same as on Mac.

### 3.7 Run the automated suite on the Pi itself

```bash
python -m pytest
```
**Expect**: `61 passed`, same as Mac/sandbox (pure logic, no hardware
touched) — worth running here anyway once, to catch any RPi-OS-specific
Python/numpy build issue early rather than during manual testing.

### 3.8 Manual verification — same checklist as Part 2, on real hardware

Re-run 2.1 through 2.7 directly on the Pi. The commands are identical;
what's different is what's actually behind them now that
`get_platform()` returns `rpi`:
- **2.1 TTS** — via `espeak_ng`, not `mac_say`.
- **2.2 mic capture** — the real USB mic, not the Mac's built-in.
- **2.3 full loop** — hold the **real pushbutton** (GPIO 17), not spacebar.
- **2.4 bulb/plug** — identical calls; confirm the Pi and the Xiaomi
  devices are actually on the same Wi-Fi network (a real, common gotcha
  — double-check this before assuming the code is broken).
- **2.5–2.7** — identical to Mac.

**Tier 2 only**: if the Sense HAT is seated, confirm `hal/temperature.py`
returns a real sensor reading instead of the Mac path's fixed 27.0°C:
```bash
python -c "from vcm.hal.temperature import read_temperature; print(read_temperature())"
```

### 3.9 Latency — the one thing that can *only* be validated here

Section 8 is explicit that Mac (M2) timing isn't representative of
RPi-class CPU performance. Once a trained, quantized model exists (see
[MODEL.md](MODEL.md)), measure real inference latency on the Pi:
```bash
python -c "
import time
import numpy as np
from vcm.inference.model import load_default_model
from vcm.audio.features import extract_log_mel

model = load_default_model()
features = extract_log_mel(np.random.randn(16000).astype('float32'))
start = time.perf_counter()
for _ in range(20):
    model.predict(features)
elapsed = (time.perf_counter() - start) / 20
print(f'{elapsed * 1000:.1f} ms per inference')
"
```
**Right now this only measures the stub model** (near-zero, meaningless
latency) — it's a template to re-run once `TFLiteIntentModel` or an
ONNX-backed model exists, not a real number yet. Don't report this
script's current output as a latency benchmark.

### 3.10 Full Tier-1 hardware smoke-test checklist

- [ ] Fresh RPi OS 64-bit flashed, booted, SSH access confirmed
- [ ] `get_platform()` returns `rpi`
- [ ] `python -m pytest` passes on the Pi itself (61/61)
- [ ] Pushbutton (not spacebar) triggers capture
- [ ] TTS audible through the real speaker
- [ ] Mic captures real audio through the USB mic
- [ ] Bulb/plug control works over the Pi's own Wi-Fi
- [ ] Weather/music/timers/reminders/calls all pass, same as Mac
- [ ] (Tier 2 only) Sense HAT temperature reads a real value
- [ ] Real inference latency measured, once a trained model exists (not yet possible)

---

## Known gaps (not bugs — future work)

- Every real command currently classifies as `unknown_background` and no-ops
  through `main.py`, because there's no trained model yet. Use `dispatch()`
  directly (2.3.1) to test action code today.
- No latency measurement possible yet anywhere — Part 3.9 has the script,
  but it's only meaningful once a trained, quantized model exists (see
  [MODEL.md](MODEL.md)).
- No benchmark harness (macro-F1, FAR/FRR, confusion matrix) yet — depends on
  a trained model and the class-wide benchmark definition (Section 7).
- Part 3 (RPi hardware setup) is unverified against physical hardware from
  this session — correct it once you've actually run it on the Cytron kit.
