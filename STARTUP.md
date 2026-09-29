# Starting the VCM (Mac + Raspberry Pi)

The short version of bringing the whole system up: what runs on which
device, in what order. [DEPLOYMENT.md](DEPLOYMENT.md) has the full
explanations, and [the troubleshooting table](#troubleshooting) below
covers the problems hit so far.

**This setup:** Pi at `raspberrypi.local`, user `quielq`, USB mic
"USB PnP Sound Device" (ALSA card 2), repo at `~/quielq-vcm` on both
machines.

## What runs where

| Device | Runs | Why |
|---|---|---|
| **Pi** | `vcm.home.server` (actions + dashboard, port 8000) | Acts on commands, speaks replies, serves the dashboard |
| **Pi** | `scripts/vcm_listen.py` | Listens for "Hey Kiwi", recognizes the command, sends it to the home server |
| **Mac** | `scripts/mac_phone_bridge.py` (port 8765) | Places calls and sends messages through your iPhone. Only needed for CALL / MESSAGE |
| **Mac** | a browser | The dashboard at http://raspberrypi.local:8000 |
| **Mac** | `scripts/deploy_pi.sh` | Copies code, models and settings to the Pi |
| **iPhone** | nothing | Settings only: *Phone → Calls on Other Devices* on, for the Mac |

A spoken command flows: mic → **Pi** `vcm_listen.py` → **Pi** home server
→ the action (lights, timer, weather...) or, for calls and messages, → **Mac**
bridge → **iPhone**.

## One-time setup (done)

Listed so it can be repeated on a new Pi or Mac.

1. **Mac → Pi SSH key**, so SSH and deploys don't ask for a password. On
   the Mac, typing the Pi's password once:
   ```bash
   ssh-copy-id raspberrypi.local
   ```
   `~/.ssh/config` on the Mac:
   ```
   Host raspberrypi.local
     HostName raspberrypi.local
     User quielq
   ```
2. **Settings** in `configs/settings.toml` on the Mac (gitignored):
   - `[weather] api_key`: a free key from
     <https://home.openweathermap.org/api_keys>.
   - `[phone]`:
     - `bridge_url = "http://<Mac LocalHostName>.local:8765"`
       (`scutil --get LocalHostName`), which works from both machines;
     - `bridge_token`: any secret (`openssl rand -hex 16`);
     - `contacts = { Mom = "+639..." }`, with `default_contact` one of
       those names.
   - `[spotify]`: optional, not set up yet (DEPLOYMENT.md 8b).
3. **iPhone:** *Settings → Phone → Calls on Other Devices* → on for the
   Mac. Optionally *Messages → Text Message Forwarding*. **Mac:** Messages
   signed in with the same Apple ID.
4. **Deploy to the Pi** from the Mac. This installs the packages (including
   `espeak-ng`), copies the code, models and settings, builds the venv and
   runs the benchmark:
   ```bash
   scripts/deploy_pi.sh raspberrypi.local
   ```

## Every time: start it up

**1. Mac: the phone bridge** (skip it if you don't need calls or messages).
Keep this terminal open and the Mac awake:
```bash
cd ~/quielq-vcm && .venv/bin/python scripts/mac_phone_bridge.py --token <bridge_token>
```
Add `--dry-run` to print calls and messages instead of sending them.

**2. Pi: the home server and the voice loop** in one tmux session, so they
survive the SSH connection dropping:
```bash
ssh raspberrypi.local
tmux new -s kiwi                       # or: tmux attach -t kiwi, if it's already running
cd ~/quielq-vcm && .venv/bin/python -m vcm.home.server
```
Press `Ctrl-b c` for a second tmux window, then:
```bash
cd ~/quielq-vcm && .venv/bin/python scripts/vcm_listen.py --server http://127.0.0.1:8000 --show-scores
```
- **tmux:** `Ctrl-b n` switches windows, `Ctrl-b d` detaches (both keep
  running), and `Ctrl-b` followed by `&` closes a window.
- **No `--device` needed:** the Pi's default input is the USB mic,
  already at 16 kHz. `--device "USB PnP"` also works: it records the raw
  mic at 48 kHz and resamples.

**Intent model choice.** Two intent models ship, and `--intent-model`
picks one (Experiments 34 and 36, EXPERIMENTS.md):

| File | Intent | Slot values (timer / alarm / brightness / color / temperature / reminder) |
|---|---:|---|
| `models/vcm_intent.onnx` (default: Experiment 36 seed 1) | 84.8% | 74 / 98 / 75 / 87 / 100 / 100% |
| `models/vcm_intent_frozen.onnx` (Experiment 31 frozen + slot heads) | 85.5% | 68 / 90 / 68 / 84 / – / – |

Temperature (18 / 22 / 26 degrees) and reminder (drink water / study /
exercise) values are measured on synthetic clips only; try them on your
own voice.

To compare with the previous default:
```bash
cd ~/quielq-vcm && .venv/bin/python scripts/vcm_listen.py --intent-model models/vcm_intent_frozen.onnx --server http://127.0.0.1:8000 --show-scores
```

**3. Mac: open the dashboard** at http://raspberrypi.local:8000.

**4. Say "Hey Kiwi"**, then a command. You can go straight into it
("Hey Kiwi, stop"); speech during the chime is kept. With `--show-scores`,
the wake score jumps toward 1.00 when you say it. The wake threshold is
0.6, and 0.4 while music plays on the Pi (from Kiwi or any app), since
"Hey Kiwi" scores lower over music (`--wake-threshold 0.7` /
`--noisy-wake-threshold 0.5` if it wakes by itself too often),
and any music playing on the Pi drops by about 12 dB while Kiwi listens.

To run everything at boot instead, without steps 2–3, use
`scripts/deploy_pi.sh raspberrypi.local --services` (DEPLOYMENT.md step 9).

## Test checklist

| Say | Expect |
|---|---|
| "set a timer for five minutes" | TIMER `5m`; a timer appears on the dashboard |
| "turn the lights blue" / "set brightness to 60 percent" | COLOR `blue` / BRIGHTNESS `60%`; the virtual lamp changes |
| "turn off the lights" | LIGHT_OFF |
| "what time is it" / "what's the weather" | The spoken or dashboard reply with the time / Quezon City weather |
| "turn the volume up" | VOLUME_UP (needs a speaker on the Pi) |
| "remind me to buy milk" | CREATE_REMINDER; appears on the dashboard |
| "call Mom" / "message Mom" | The Mac bridge terminal prints it; the Mac shows the call prompt (click **Call**) |
| "play some music" | Spotify plays on the Pi's speaker |
| "pause" → "play music" | Pauses, then continues from the same spot |
| "stop the music" → "play music" | Stops, then starts the song from the beginning |

**Three phrasings per command.** From the author's 137 saved live
commands (current model; transcribed offline with Whisper): "right / tried"
counts a command only if it was acted on (right intent, confidence >= 0.6).
*Untested* = not said live yet. Avoid the last column; those phrasings stay
in training because the class benchmark uses them.

| Command | Phrasing 1 | Phrasing 2 | Phrasing 3 | Avoid |
|---|---|---|---|---|
| PLAY_MUSIC | Play music (7/7) | Start the music (3/3) | Play some music (1/1) | |
| STOP | Stop the music (4/4, with "Stop music") | Stop (6/8) | Stop playing music *untested* | |
| PAUSE | Pause (1/1) | Pause the music (1/1) | Pause song (1/1) | Pause audio (1/3) |
| NEXT | Next song (1/1) | Skip song (1/1) | Go to the next song (1/1) | bare "Next" (fails offline too) |
| VOLUME_UP | Volume up (3/3, with "Turn the volume up") | Increase the volume (1/1) | Turn the volume up | |
| VOLUME_DOWN | Volume down (2/2, with "Turn the volume down") | Decrease the volume (1/1) | Lower the volume (1/1) | |
| WEATHER | Weather (1/1) | What's the weather (1/1) | Tell me the weather (1/1) | |
| TIME | Time (2/2) | What time is it (1/1) | Tell me the time (1/1) | |
| LIGHT_ON | Lights on (1/1) | Turn on the lights (2/3) | Switch on the lights *untested* | Power on the lights (0/4) |
| LIGHT_OFF | Lights off (2/2) | Switch off the lights (2/2) | Turn off the lights (1/1) | Kill the lights (0/1) |
| BRIGHTNESS | Brightness to 60 percent (5/7; both misses were cut off, fixed) | Adjust brightness to 60 percent (4/5) | Set the brightness to 60 percent *untested* | Brightness level 60 percent (0/2) |
| COLOR | Color red (3/4) | Change color to red (1/1) | Set the lights to red *untested* | |
| TEMPERATURE | Temperature 18 degrees (1/1) | Change the temperature to 22 degrees (1/1) | Set the temperature to 26 degrees (2/3) | |
| ALARM | Set an alarm for 7 AM (6/6) | Wake me up at 6 AM *untested* | Alarm 8 AM *untested* | |
| TIMER | Set a timer for 5 minutes *untested* | Timer 30 seconds *untested* | Countdown for 1 minute *untested* | |
| CREATE_REMINDER | Reminder to drink water (3/3) | Remind me to study (2/2) | Create a reminder to exercise (2/2) | "run" (not a value yet) |
| LIST_REMINDERS | Reminders (2/2) | Show my reminders *untested* | List my reminders *untested* | |
| CALL | Make a call (1/1) | Call (2/4) | *none reliable yet* | Make a phone call (0/2), Call mom (0/2) |
| MESSAGE | Send a message (1/1) | Message *untested* | Send my message *untested* | |

Most counts are 1-3 tries, so treat them as a first pass. CALL is the weak
one: only "Make a call" worked reliably (Experiment 37, on hold, adds the
author's recordings for it).

The dashboard's **Simulate a command** box runs the same actions without
speaking. If a command works there but not by voice, the problem is
recognition, not the action.

## Demo day

Everything that went wrong in testing, and what now prevents it:

| What happened | Prevention |
|---|---|
| After a reboot, nothing ran | `scripts/deploy_pi.sh raspberrypi.local --services`: Kiwi starts on boot and restarts 3 s after any crash |
| The USB mic dropped out; the soundbar's mic became the input and Kiwi went deaf | WirePlumber rules (`deploy/wireplumber/51-kiwi-audio.lua`): the USB mic is always preferred, the soundbar's mic is disabled. The listener also exits (and restarts) if the mic stops sending audio or the input changes |
| Music and Kiwi's replies stuck near silent | Ducking fixed (-12 dB, not -36), Kiwi's replies never ducked, ducks undo themselves after 20 s, and stream volumes aren't remembered between streams |
| A running listener lost its mic after an audio restart and never noticed | The listener now exits and restarts; `kiwi_doctor.py` flags a listener that isn't capturing |

**The day before:**
1. `scripts/deploy_pi.sh raspberrypi.local --services` (installs everything
   above; then Kiwi runs without SSH).
2. Reboot the Pi (`ssh raspberrypi.local sudo reboot`) and check it comes
   back by itself: `.venv/bin/python scripts/kiwi_doctor.py --beep` should
   say READY.
3. Set up the venue network: add a phone hotspot to the Pi in advance (on
   the Pi: `sudo nmcli dev wifi connect "<hotspot name>" password "<password>"`),
   since venue Wi-Fi often blocks device-to-device traffic. Without internet
   only Spotify and weather stop; everything else runs on the Pi.
4. Tape the mic and soundbar USB plugs down (the mic dropout wasn't power:
   `vcgencmd get_throttled` stayed `0x0`). Use the official 27 W supply.

**At the venue, before presenting:**
1. `ssh raspberrypi.local 'cd ~/quielq-vcm && .venv/bin/python scripts/kiwi_doctor.py --beep --fix'` → READY.
2. If the mic shows clipping when you speak at normal distance, lower its gain:
   `amixer -c 2 sset Mic 12` (0-16; 16 is the maximum).
3. Say the phrases from the table above, not the ones in the "Avoid" column.
4. Backup if voice fails in a loud room: the dashboard's **Simulate a
   command** box runs every action.

## Stop

- **Pi:** `tmux attach -t kiwi`, then `Ctrl-c` in each window. Or kill the
  whole session: `tmux kill-session -t kiwi`.
- **Mac:** `Ctrl-c` in the bridge terminal.

## Update the Pi after changes

Merge to master on GitHub, then on the Mac:
```bash
cd ~/quielq-vcm && git checkout master && git pull
scripts/deploy_pi.sh raspberrypi.local              # add --settings if configs/settings.toml changed
```
Then restart both Pi windows (or `systemctl --user restart vcm-home vcm` if
you used `--services`). If you'd rather `git pull` on the Pi itself, don't
edit files there first: local edits block the pull. Make changes on the Mac
and merge them.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Permission denied (publickey,password)` | Run `ssh-copy-id raspberrypi.local` on the Mac (one-time setup, step 1). |
| `Invalid sample rate [PaErrorCode -9997]` | The mic's raw device can't do 16 kHz. Fixed in `vcm_listen.py` (it resamples); `git pull` / redeploy. Or drop `--device`. |
| `arecord`: no such device | The card number changed (it's 2 now). Check `arecord -l`; `vcm_listen.py` selects the mic by name, so it isn't affected. |
| Wake score stays low even up close | Mic level too low: `alsamixer -c 2`, `F4` for capture, raise it. |
| Kiwi lags, the Pi's CPU is at 100%, Spotify stalls (`Throughput ... lower than minimum`) | numpy's OpenBLAS was spinning a thread per core. `vcm_listen.py` now limits it to one; redeploy. `top -H` should show the listener at a few %. |
| Static or knocking in recordings | Level too high (clipping), the mic in a blue USB 3 port (use black USB 2), or undervoltage (`vcgencmd get_throttled` should print `0x0`). |
| No spoken replies | No speaker set up: `pactl list short sinks` shows only `auto_null`. Plug in a USB speaker or pair a Bluetooth one (DEPLOYMENT.md 8b). |
| "home server unreachable" | Start `vcm.home.server` in the other tmux window first. |
| CALL / MESSAGE only "simulated" or failing | Bridge not running on the Mac, Mac asleep, `bridge_url` / `bridge_token` mismatch, or macOS blocked incoming connections (allow Python in the firewall prompt). |
| Weather "isn't set up" / HTTP 401 | `api_key` missing, or a new key not active yet (up to ~2 h). |
| Dashboard doesn't load from the Mac | Home server not running, or Mac and Pi on different networks (`ping raspberrypi.local`). |
| `git pull` on the Pi fails | Local edits or untracked copies of tracked files on the Pi. Compare them with master, then `git restore` / remove them. Changes belong on the Mac. |

More: [DEPLOYMENT.md § Troubleshooting](DEPLOYMENT.md#troubleshooting).
