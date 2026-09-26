# Deploying the VCM to a Raspberry Pi 5 (headless, SSH only)

This gets the voice pipeline (**"Hey Kiwi" wake word → command → intent +
slot value**) running on a Raspberry Pi 5 with **no monitor or keyboard**:
everything happens over SSH from your laptop. For the rest of the device
(pushbutton wiring, Sense HAT, lights, TTS, music), see
[TESTING.md Part 3](TESTING.md#part-3--raspberry-pi-5-hardware-setup-and-testing).

What runs on the Pi is two small ONNX files (fp32), with no torch:

| Model | File | Size | Job |
|---|---|---:|---|
| Wake word | `models/kiwi_wakeword.onnx` | 107 KB | Always on: scores a 1.5 s window every 0.1 s |
| Intent + slots | `models/vcm_intent.onnx` | 426 KB | Runs once per command: 20 intents + TIMER/ALARM/BRIGHTNESS/COLOR values |

**Why fp32, not the `.int8.onnx` files:** int8 quantization cost the intent
model 6.8 points of real-speech accuracy (82.1% → 75.3%, EXPERIMENTS.md
Experiment 32) and saved only 134 KB. fp32 is 533 KB for both models,
within the 1 MB budget, uses the same memory, and is no slower.

Together they're well under the 1 MB budget. The runtime needs only
**numpy, onnxruntime and sounddevice**: features are computed with a numpy
re-implementation of librosa (`vcm/audio/dsp.py`, tested to match it), so
there's no librosa/scipy/numba and no torch on the device. The whole
pipeline peaks at **~90 MB of memory** (measured on an M-series Mac, where a
command takes ~3.5 ms and the wake word uses ~1% of one core).
`scripts/benchmark_pi.py` measures the real numbers on the board (step 6).

### Which boards

| Board | RAM | Works? | Notes |
|---|---:|---|---|
| Raspberry Pi 5 / 4 (any RAM, incl. 2 GB) | 2–8 GB | Yes | The target. Lots of headroom. |
| **Raspberry Pi Zero 2 W** | 512 MB | Yes (expected) | Same steps with 64-bit Pi OS Lite. Its 1 GHz Cortex-A53 is roughly 8–15× slower than a Mac core, which is still ~10% of one core for the wake word and ~50 ms per command, by estimate. Measure with step 6. |
| Raspberry Pi Zero / Zero W (original) | 512 MB | No | ARMv6, 32-bit only: no onnxruntime builds exist for it. It would need a pure-numpy model runtime (feasible for a model this small, but not built). |

### Minimum RAM

| What | Memory |
|---|---:|
| Voice pipeline, peak (wake word listening + commands) | **~96 MB** (measured) |
| of which the two models themselves | ~8.5 MB |
| Home server + dashboard (`vcm.home.server`, step 8b) | **~30 MB** (measured) |
| Raspberry Pi OS Lite (64-bit), idle, including SSH | ~60–100 MB (typical; confirm with `free -m`, step 6) |
| raspotify (Spotify speaker), if used | ~20–40 MB (typical, not measured) |
| **Total needed** | **~200–270 MB**, so plan for **~300 MB** with headroom |
| **Smallest workable board** | **512 MB** (Pi Zero 2 W, Pi 3 A+): about 40% of it stays free |

So any current Raspberry Pi with 512 MB or more has enough memory; with
this runtime, the limit is the CPU architecture (64-bit ARM, for
onnxruntime), not RAM. The process numbers come from an M-series Mac (see
[FOOTPRINT_COMPARISON.md](FOOTPRINT_COMPARISON.md)); Linux on the Pi will
differ somewhat, so step 6 checks them on the board. Things that did *not*
shrink it further: turning off ONNX Runtime's memory arena
(`enable_cpu_mem_arena=False`) saved nothing measurable, because the
running overhead is Python and numpy working memory, not ONNX Runtime.
Nothing here needs swap.

### Quick path: one script from the laptop

Once you can SSH in (steps 1–2), `scripts/deploy_pi.sh` does steps 3, 4
and 6 from the laptop. It installs the system packages, copies the code
and models (~2 MB; no datasets, checkpoints or recordings), builds the
Pi's Python environment and runs the benchmark:
```bash
scripts/deploy_pi.sh <username>@kiwi.local              # add --services to also do step 9
```
It copies `configs/settings.toml` the first time, switched to the
espeak-ng voice; later runs keep the Pi's copy unless you pass
`--settings`. Re-run it to update the Pi. Then continue with step 5
(microphone) and step 7.

## 1. Flash the SD card with SSH already set up (on your laptop)

1. Install [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
2. Choose your board (**Raspberry Pi 5**, or **Raspberry Pi Zero 2 W**),
   then **Raspberry Pi OS Lite (64-bit)**. Lite has no desktop, which you
   don't need without a monitor, and it leaves more memory free. It must be
   the 64-bit OS: onnxruntime has no 32-bit ARM builds.
3. Before writing, open the settings (**Edit Settings** when asked "apply
   OS customisation", or `Ctrl+Shift+X`) and set:
   - **Hostname**: `kiwi` (you'll connect to `kiwi.local`)
   - **Username / password**: your choice
   - **Wi-Fi**: network name, password, and country `PH`
   - **Services tab → Enable SSH → Allow public-key authentication only**,
     and paste your laptop's public key:
     ```bash
     cat ~/.ssh/id_ed25519.pub
     ```
     (Password authentication also works, but a key means no password
     prompts and no password to leak.)
4. Write the card, put it in the Pi, and power it on. The first boot takes
   1–2 minutes longer than usual.

## 2. Connect over SSH

```bash
ssh <username>@kiwi.local
```

If `kiwi.local` doesn't resolve, the Pi and laptop may be on different
networks, or the Wi-Fi settings didn't take. Check your router's
connected-devices list for the Pi's IP and use `ssh <username>@<ip>`.
Some campus and hotel networks block device-to-device traffic entirely;
a phone hotspot that both devices join is the quickest workaround.

**Use `tmux` for anything long-running**, so it survives a dropped SSH
connection: `tmux new -s kiwi` starts a session, `Ctrl-b d` detaches, and
`tmux attach -t kiwi` reconnects later.

## 3. System packages

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git python3-venv libportaudio2 alsa-utils tmux espeak-ng
```
`espeak-ng` is the device's voice: set `backend = "espeak_ng"` under
`[tts]` in the Pi's `configs/settings.toml` (`mac_say` only exists on macOS).

## 4. Get the code and models

If the repo is **public**, or the Pi has GitHub access:
```bash
git clone https://github.com/quielq/quielq-vcm.git ~/quielq-vcm
```

If it's **private**, the simplest route is to copy it from your laptop.
Run this *on the laptop*:
```bash
rsync -a --exclude .venv --exclude data --exclude checkpoints --exclude debug_recordings ~/quielq-vcm/ <username>@kiwi.local:~/quielq-vcm/
```

The two model files live in `models/`. If they aren't there yet, copy them
from wherever they were exported (for example the DGX), *from the laptop*:
```bash
scp models/vcm_intent.onnx models/kiwi_wakeword.onnx <username>@kiwi.local:~/quielq-vcm/models/
```

Then create the Python environment *on the Pi*. Only the three runtime
libraries are installed (`requirements-pi.txt`); the rest of the project's
dependencies (torch, librosa, python-miio, pynput) are for training and the
home-automation features, not the voice pipeline:
```bash
cd ~/quielq-vcm
python3 -m venv .venv
.venv/bin/pip install -r requirements-pi.txt
.venv/bin/pip install --no-deps -e .
```

## 5. Microphone

Plug in the USB microphone, then check that it's seen:
```bash
arecord -l                                  # note the card number, e.g. "card 1"
.venv/bin/python -c "import sounddevice as sd; print(sd.query_devices())"
```

Record 3 seconds and copy it back to your laptop to listen, since the Pi
has no screen or speaker. Use the card number from `arecord -l`:
```bash
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -d 3 ~/mictest.wav     # on the Pi
scp <username>@kiwi.local:~/mictest.wav . && afplay mictest.wav        # on the Mac
```

If it's too quiet, raise the capture level with `alsamixer -c 1` (`F4`
switches to capture controls, arrow keys adjust). If you had to lower the
silence gate on the laptop to be heard, fix the input level here instead.

## 6. Benchmark (no microphone needed)

```bash
.venv/bin/python scripts/benchmark_pi.py
```

This prints per-command latency, the wake word's share of one CPU core,
model sizes and memory use. These are the deployment numbers to report.

To confirm the minimum-RAM numbers above on the board, check the OS's own
use before starting the pipeline, and the pipeline's use while it runs:
```bash
free -m                      # "used" with nothing of ours running = the OS baseline
.venv/bin/python scripts/measure_footprint.py ours --clips <dir of 16 kHz wavs>   # our process, stage by stage
```

## 7. Run it

```bash
tmux new -s kiwi
cd ~/quielq-vcm
.venv/bin/python scripts/vcm_listen.py --device <mic index from step 5>
```

Say **"Hey Kiwi"**, pause briefly, then say a command. Each result prints
the intent, the slot value, the confidence, and timing:
```
[wake word, score 0.97] listening...
  -> TIMER (0.99)  5m (0.97)   [model 9 ms, 0.12 s after end of command]
  (2.4 s from wake word to result)
```
- `--show-scores` prints the live wake-word score, which is useful for
  seeing how close near-misses get.
- `--wake-threshold` sets the trigger level (default 0.95: in Experiment 34
  it missed 14% of clean and 22–24% of noisy synthetic "hey kiwi", none of
  the author's 10 held-out real takes, with 1.34 false wake-ups per hour of
  test speech, mostly deliberate near-misses). 0.98 cuts false wake-ups to
  0.4/h but missed 3 of the 10 real takes. Raise it if it wakes too often,
  lower it if it misses you.
- `--trigger button` skips the wake word and uses the GPIO 17 pushbutton
  instead (push-to-talk).

**Testing on a laptop** works the same way. The same script runs with the
Mac's built-in microphone:
```bash
.venv/bin/python scripts/vcm_listen.py
```
The Mac needs `onnxruntime` in its environment (`pip install -e ".[deploy]"`).

**Checked on the Mac before deploying** (Experiment 34 models, a clean venv
with only `requirements-pi.txt`, no torch or librosa):
- `benchmark_pi.py`: 3.2 ms per command, the wake word uses 1% of one
  core, and the process peaks at 82 MB.
- The author's recorded takes, streamed through the detector: all 30
  "hey kiwi" takes trigger at 0.95 (23 of 30 at 0.98), and none of the 12
  near-misses do.
- `vcm.home.server`: every simulated command works, and the dashboard
  serves.

## 8. Field-test the wake word

**Record your own "hey kiwi" first** (on the laptop, ~5 minutes):
```bash
python scripts/record_wakeword.py --speaker-id <your-name>
```
It prompts 30 "hey kiwi" takes said different ways (quiet, fast, across the
room...) plus 12 near-misses, and holds out a third of the "hey kiwi" takes
for testing. `scripts/evaluate_wakeword.py --extra-wake-manifest
data/wakeword_real/manifest.csv` then reports the false-reject rate on your
real voice, and `scripts/train_wakeword.py --extra-wake-manifest ...` adds
the rest to training. Other people's recordings help even more.

Two numbers matter, and both can be measured over SSH:
- **False rejects:** say "Hey Kiwi" 20 times, at about 1 m and 3 m, and
  count the misses.
- **False wake-ups:** leave `vcm_listen.py` running for an hour with a
  podcast or TV on nearby, then count the `[wake word` lines. Target: at
  most 1 per hour.

Compare these with the offline estimates in EXPERIMENTS.md. If false
wake-ups come mostly from one kind of audio, save a sample (step 5's
`arecord`) and add it to the wake-word training negatives.

## 8b. Actions and the web dashboard

Recognizing a command is half the job; `vcm.home.server` acts on it. It
runs the action (music, lamp, timers, reminders, calls...), speaks the
reply, and serves a **live dashboard** showing the virtual lamp and
thermostat, reminders, timers, alarms, music, calls and a command log.

```bash
# terminal 1 (tmux window): the home server
.venv/bin/python -m vcm.home.server            # --no-speak to stay silent
# terminal 2: the voice loop, sending each command to it
.venv/bin/python scripts/vcm_listen.py --server http://127.0.0.1:8000
```
Open **http://kiwi.local:8000** on your laptop or phone (same Wi-Fi). The
dashboard also has a "Simulate a command" box, so every action can be
tested without speaking. On a Mac, the same two commands work with
`http://127.0.0.1:8000`.

The server has **no login**: anyone on the same network can open it.
Keep it on your home network (TODO.md).

What each command does, and what it needs in `configs/settings.toml`:

| Commands | Action | Setup |
|---|---|---|
| LIGHT_ON/OFF, BRIGHTNESS, COLOR, TEMPERATURE | Virtual lamp / thermostat on the dashboard | none; `[xiaomi]` also drives a real bulb |
| TIMER, ALARM, CREATE/LIST_REMINDERS | Scheduled / stored on the device, spoken alerts | none |
| TIME | System clock | none |
| WEATHER | OpenWeatherMap | `[weather] api_key` (free) |
| PLAY_MUSIC, PAUSE, STOP, NEXT | Spotify Connect; local files if Spotify isn't available | `[spotify]` (Premium) and/or `[music] media_dir` + `mpv` |
| VOLUME_UP/DOWN | The speaker's volume (USB or Bluetooth) | none |
| CALL, MESSAGE | Your iPhone, through your Mac | `[phone]` + the Mac bridge; otherwise simulated |

**Spotify (needs Premium).**
1. Create an app at <https://developer.spotify.com/dashboard> with the
   redirect URI `http://127.0.0.1:8888/callback` and "Web API" ticked.
2. On the **Mac**, run
   `python scripts/spotify_auth.py --client-id <id> --client-secret <secret>`,
   log in, and paste the printed `[spotify]` block into `configs/settings.toml`
   on the Mac and on the Pi.
3. Make the Pi a Spotify speaker with raspotify (a packaged librespot,
   <https://github.com/dtcooper/raspotify>; read its install script before
   piping it to a shell):
   ```bash
   curl -sL https://dtcooper.github.io/raspotify/install.sh | sh
   sudo sed -i 's/^#\?LIBRESPOT_NAME=.*/LIBRESPOT_NAME="kiwi"/' /etc/raspotify/conf
   sudo systemctl restart raspotify
   ```
   Then set `device_name = "kiwi"` in `[spotify]`. While testing on the Mac,
   leave `device_name` empty: commands then control your Spotify app.

**Calls and messages via your Mac.**
1. On the Mac, sign Messages in to your Apple ID and enable **iPhone >
   Settings > Phone > Calls on Other Devices** for the Mac (plus **Text
   Message Forwarding** for SMS to non-iPhones).
2. Start the bridge on the Mac with a secret of your choice:
   `python scripts/mac_phone_bridge.py --token <secret>` (add `--dry-run`
   first to test without sending anything).
3. In the device's `configs/settings.toml`, set `[phone] bridge_url`
   (`http://<mac-name>.local:8765` from the Pi, `http://127.0.0.1:8765` on
   the Mac), `bridge_token = "<secret>"`, and `contacts = { Mom = "+63..." }`.

Messages send automatically. Calls open macOS's call prompt, where one click
on **Call** is needed (macOS doesn't allow fully automatic calls). The first
message triggers a macOS prompt allowing Terminal to control Messages. The
Mac must be awake. Other phone routes are in TODO.md.

**Speaker.** Plug in a USB speaker, or pair a Bluetooth one over SSH with
`bluetoothctl` (`scan on`, `pair <MAC>`, `trust <MAC>`, `connect <MAC>`).
Either becomes the default output, and VOLUME_UP/DOWN controls it.

## 9. Start on boot (optional)

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/vcm.service <<'UNIT'
[Unit]
Description=VCM voice pipeline (Hey Kiwi)
After=sound.target vcm-home.service

[Service]
WorkingDirectory=%h/quielq-vcm
ExecStart=%h/quielq-vcm/.venv/bin/python scripts/vcm_listen.py --server http://127.0.0.1:8000
Restart=always

[Install]
WantedBy=default.target
UNIT
cat > ~/.config/systemd/user/vcm-home.service <<'UNIT'
[Unit]
Description=VCM home server and dashboard
After=network-online.target

[Service]
WorkingDirectory=%h/quielq-vcm
ExecStart=%h/quielq-vcm/.venv/bin/python -m vcm.home.server
Restart=always

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now vcm-home vcm
sudo loginctl enable-linger $USER      # keep user services running without an SSH login
journalctl --user -u vcm -f             # follow the output
```

## Updating

```bash
scripts/deploy_pi.sh <username>@kiwi.local   # from the laptop; or on the Pi:
cd ~/quielq-vcm && git pull          # or rsync from the laptop, as in step 4
systemctl --user restart vcm-home vcm   # if you set up step 9
```

## Troubleshooting

| Symptom | Check |
|---|---|
| `PortAudioError: Error querying device` | `libportaudio2` installed (step 3)? Pass `--device` with the index from `sd.query_devices()`. |
| Never wakes | Run with `--show-scores`. If scores stay low even up close, the mic level is too low (step 5). |
| Wakes on everything | Raise `--wake-threshold`; check that the value matches EXPERIMENTS.md's recommendation. |
| Slow, or results lag | `vcgencmd get_throttled` (anything other than `0x0` means under-voltage or heat throttling); `vcgencmd measure_temp`; use the official 27 W power supply. |
| `Illegal instruction` on import | Make sure the OS is 64-bit: `uname -m` should print `aarch64`. |
