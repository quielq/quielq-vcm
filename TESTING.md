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

Expected: `18 passed` in a few seconds. No microphone, speaker, network, or
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

## 2.8 Full Tier-1 smoke-test checklist

Run once after any change that touches multiple modules, or before a demo:

- [ ] `python -m pytest` — 18/18 pass
- [ ] `python -m vcm.main` — hold spacebar, speak, see `Heard intent: unknown_background`
- [ ] TTS audible (2.1)
- [ ] Bulb on/off/dim responds (2.4)
- [ ] Thermostat command runs without raising (2.4)
- [ ] Weather returns a real forecast string (2.5)
- [ ] Music plays and pause/resume/volume/stop all work (2.6)
- [ ] Timer fires and speaks after its delay (2.7)
- [ ] Reminder add/list round-trips (2.7)
- [ ] Mocked call speaks "Calling Mom" (2.7)

## Known gaps (not bugs — future work)

- Every real command currently classifies as `unknown_background` and no-ops
  through `main.py`, because there's no trained model yet. Use `dispatch()`
  directly (2.3.1) to test action code today.
- No latency measurement here — Section 8 is explicit that this Mac (or this
  sandbox) is not representative of RPi-class CPU timing; that validation
  happens only once on the actual Pi.
- No benchmark harness (macro-F1, FAR/FRR, confusion matrix) yet — depends on
  a trained model and the class-wide benchmark definition (Section 7).
