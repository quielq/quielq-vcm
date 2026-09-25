# Deploying the VCM to a Raspberry Pi 5 (headless, SSH only)

This gets the voice pipeline (**"Hey Kiwi" wake word → command → intent +
slot value**) running on a Raspberry Pi 5 with **no monitor or keyboard**:
everything happens over SSH from your laptop. For the rest of the device
(pushbutton wiring, Sense HAT, lights, TTS, music), see
[TESTING.md Part 3](TESTING.md#part-3--raspberry-pi-5-hardware-setup-and-testing).

What runs on the Pi is two small ONNX files, int8-quantized, with no torch:

| Model | File | Size | Job |
|---|---|---:|---|
| Wake word | `models/kiwi_wakeword.int8.onnx` | ~95 KB | Always on: scores a 1.5 s window every 0.1 s |
| Intent + slots | `models/vcm_intent.int8.onnx` | ~300 KB | Runs once per command: 20 intents + TIMER/ALARM/BRIGHTNESS/COLOR values |

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
sudo apt install -y git python3-venv libportaudio2 alsa-utils tmux
```

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
scp models/vcm_intent.int8.onnx models/kiwi_wakeword.int8.onnx <username>@kiwi.local:~/quielq-vcm/models/
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
- `--wake-threshold` sets the trigger level. Use the one suggested by
  `scripts/evaluate_wakeword.py`, recorded in EXPERIMENTS.md.
- `--trigger button` skips the wake word and uses the GPIO 17 pushbutton
  instead (push-to-talk).

**Testing on a laptop** works the same way. The same script runs with the
Mac's built-in microphone:
```bash
.venv/bin/python scripts/vcm_listen.py
```
The Mac needs `onnxruntime` in its environment (`pip install -e ".[deploy]"`).

## 8. Field-test the wake word

Two numbers matter, and both can be measured over SSH:
- **False rejects:** say "Hey Kiwi" 20 times, at about 1 m and 3 m, and
  count the misses.
- **False wake-ups:** leave `vcm_listen.py` running for an hour with a
  podcast or TV on nearby, then count the `[wake word` lines. Target: at
  most 1 per hour.

Compare these with the offline estimates in EXPERIMENTS.md. If false
wake-ups come mostly from one kind of audio, save a sample (step 5's
`arecord`) and add it to the wake-word training negatives.

## 9. Start on boot (optional)

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/vcm.service <<'UNIT'
[Unit]
Description=VCM voice pipeline (Hey Kiwi)
After=sound.target

[Service]
WorkingDirectory=%h/quielq-vcm
ExecStart=%h/quielq-vcm/.venv/bin/python scripts/vcm_listen.py
Restart=always

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now vcm
sudo loginctl enable-linger $USER      # keep user services running without an SSH login
journalctl --user -u vcm -f             # follow the output
```

## Updating

```bash
cd ~/quielq-vcm && git pull          # or rsync from the laptop, as in step 4
systemctl --user restart vcm         # if you set up step 9
```

## Troubleshooting

| Symptom | Check |
|---|---|
| `PortAudioError: Error querying device` | `libportaudio2` installed (step 3)? Pass `--device` with the index from `sd.query_devices()`. |
| Never wakes | Run with `--show-scores`. If scores stay low even up close, the mic level is too low (step 5). |
| Wakes on everything | Raise `--wake-threshold`; check that the value matches EXPERIMENTS.md's recommendation. |
| Slow, or results lag | `vcgencmd get_throttled` (anything other than `0x0` means under-voltage or heat throttling); `vcgencmd measure_temp`; use the official 27 W power supply. |
| `Illegal instruction` on import | Make sure the OS is 64-bit: `uname -m` should print `aarch64`. |
