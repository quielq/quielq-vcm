# Runbook: starting, testing and demoing the VCM

Day-to-day operation of the finished system: what runs on which device,
in what order, how to check it, and what to do on demo day. [DEPLOYMENT.md](DEPLOYMENT.md) has the full
explanations, and [the troubleshooting table](#troubleshooting) below
covers the problems hit so far.

**This setup:** Pi at `raspberrypi.local`, user `quielq`, USB mic
"MUSIC-BOOST USB Microphone MB-306" (replaced the C-Media "USB PnP Sound
Device" on 2026-10-03; any USB mic works), repo at `~/quielq-vcm` on both
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
   - `[spotify]`: client id, secret and refresh token from
     `scripts/spotify_auth.py`, and `device_name = "kiwi"` (DEPLOYMENT.md 8b).
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

**Services installed (`deploy_pi.sh --services`, the setup for the demo)?**
Then the Pi side starts by itself at boot, so skip step 2 below and **don't
also start `vcm.home.server` or `vcm_listen.py` by hand**: two copies fight
over port 8000 and the mic. Instead:

| To | Run on the Pi |
|---|---|
| See what the listener hears and does | `journalctl --user -u vcm -f` |
| See the home server's log | `journalctl --user -u vcm-home -f` |
| Restart both (e.g. after an update) | `systemctl --user restart vcm-home vcm` |
| Stop both (to run them by hand for testing) | `systemctl --user stop vcm vcm-home` |
| Check everything | `cd ~/quielq-vcm && .venv/bin/python scripts/kiwi_doctor.py --beep --fix` |

The services run the listener with `--save-commands ~/kiwi_commands` and
the default thresholds (0.6, and 0.4 over music).

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

**Intent model choice.** `--intent-model` picks the file. Accuracy is on
the class test set (all clips / real speech); slot accuracy on its real
speech.

| File | Size | Intent | Slot values, real speech (timer / alarm / temperature / brightness / color / reminder) |
|---|---:|---:|---|
| `models/vcm_intent.onnx` (default: Experiment 43b seed 1) | 1.46 MB | 95.50% / 78.64% | 100 / 90 / 79 / 96 / 82 / 89% |
| `models/vcm_intent_small.onnx` (Experiment 43c seed 0) | 722 KB | 94.53% / 76.06% | |

Slot values are the schema's three per command: 10 s / 30 s / 1 min,
6:00 AM / 8:00 AM / 9:00 PM, 18 / 22 / 26 degrees, 20 / 60 / 100 percent,
red / blue / green, drink water / study / exercise. Other values ("5
minutes", "7 AM") are not in the model's vocabulary. The demo benchmark is
the 93 Option B phrases; the model gets 99% of them right on test.

**3. Mac: open the dashboard** at http://raspberrypi.local:8000. On an
Android phone, which can't open `.local` names, use the Pi's IP address
instead (`ssh raspberrypi.local hostname -I`; it changes with the network).

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
| "start a timer for 30 seconds" | TIMER `30s`; a timer appears on the dashboard |
| "set color to blue" / "adjust brightness to 60 percent" | COLOR `blue` / BRIGHTNESS `60%`; the virtual lamp changes |
| "turn on the lights" / "shut off the lights" | LIGHT_ON / LIGHT_OFF |
| "what time is it" / "what's the weather" | The spoken or dashboard reply with the time / Quezon City weather |
| "set an alarm for 8:00 AM" | ALARM `8:00 AM`; the alarm appears on the dashboard |
| "set the temperature to 22 degrees" | TEMPERATURE `22 degrees`; the thermostat changes |
| "turn the volume up" | VOLUME_UP (needs a speaker on the Pi) |
| "remind me to drink water" / "list my reminders" | CREATE_REMINDER `drink water`, shown on the dashboard / the reminders read aloud |
| "make a call" / "send a message" | The Mac bridge terminal prints it; the Mac shows the call prompt (click **Call**) |
| "play some music" | Spotify plays on the Pi's speaker |
| "pause" → "play music" | Pauses, then continues from the same spot |
| "stop" → "play music" | Stops, then starts the song from the beginning |

**Three phrasings per command.** These are the class schema's (Option B),
the phrasings the class benchmark uses; on the test set the model gets
99.2% of clips with exactly this wording right. Other wordings of the same
command work less often (73% on test), so use these in a demo.

| Command | Phrasing 1 | Phrasing 2 | Phrasing 3 |
|---|---|---|---|
| PLAY_MUSIC | Play music | Start music | Play some music |
| WEATHER | Weather | What's the weather? | Tell me the weather |
| TIME | Time | What time is it? | Tell me the time |
| LIGHT_ON | Lights on | Power on the lights | Turn on the lights |
| LIGHT_OFF | Lights out | Kill the lights | Shut off the lights |
| PAUSE | Pause | Pause audio | Pause song |
| STOP | Stop | Stop playing | End playback |
| NEXT | Next song | Skip song | Play next song |
| VOLUME_UP | Volume up | Increase the volume | Turn the volume up |
| VOLUME_DOWN | Volume down | Lower the volume | Turn the volume down |
| CALL | Call | Make a call | Make a phone call |
| MESSAGE | Message | Send a message | Send my message |
| LIST_REMINDERS | Reminders | Show my reminders | List my reminders |
| TIMER | Timer {duration} | Countdown for {duration} | Start a timer for {duration} |
| ALARM | Alarm {time} | Wake me up at {time} | Set an alarm for {time} |
| TEMPERATURE | Temperature {degrees} | Change the temperature to {degrees} | Set the temperature to {degrees} |
| BRIGHTNESS | Brightness {percent} | Adjust brightness to {percent} | Brightness level {percent} |
| COLOR | Change color to {color} | Switch color to {color} | Set color to {color} |
| CREATE_REMINDER | Reminder {task} | Remind me to {task} | Create a reminder to {task} |

With {duration} 10 seconds / 30 seconds / 1 minute, {time} 6:00 AM / 8:00
AM / 9:00 PM, {degrees} 18 / 22 / 26 degrees, {percent} 20 / 60 / 100
percent, {color} red / blue / green, {task} drink water / study / exercise.
The weakest commands on real speech are CALL, PLAY_MUSIC, WEATHER and
MESSAGE ([TESTING.md](TESTING.md#intent-accuracy-shipped-model-modelsvcm_intentonnx)).

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

**What `kiwi_doctor.py` checks.** Each line is OK, WARN or FAIL, and the
last line is READY or the list of what to fix:

| Line | FAIL / WARN means | Fix |
|---|---|---|
| power, temperature | Undervoltage or overheating | The official 27 W supply; airflow |
| microphone device | The input isn't a USB mic (or is the soundbar's) | Replug it; the WirePlumber rules pick it again |
| speaker device, speaker volume | Not the soundbar, muted, or under 50% | Replug it; `pactl set-sink-volume @DEFAULT_SINK@ 100%` |
| stream '...' | An app's sound is stuck turned down (a duck that never ended) | Run with `--fix` |
| microphone signal | Silent (unplugged, muted) or clipping | Check the cable; `amixer -c 2 sset Mic 12` if clipping |
| speaker output (`--beep`) | The test beep didn't reach the soundbar | Check the speaker device and volume lines |
| service vcm-home / vcm | Not running, or run by hand (won't survive a reboot) | `deploy_pi.sh --services` |
| listener hears the mic | Running but deaf (its mic connection was cut) | `systemctl --user restart vcm` |
| home server | The dashboard server doesn't answer | `systemctl --user restart vcm-home` |
| models | A model file is missing or broken | Redeploy |
| internet, Spotify, weather | Offline, or a key rejected (HTTP 401) | Hotspot; a new key in `configs/settings.toml`, then `deploy_pi.sh --settings` |

It plays a short beep only with `--beep`, and changes nothing without `--fix`.

**At the venue, before presenting:**
1. `ssh raspberrypi.local 'cd ~/quielq-vcm && .venv/bin/python scripts/kiwi_doctor.py --beep --fix'` → READY.
2. If the mic shows clipping when you speak at normal distance, lower its gain:
   `amixer -c 2 sset Mic 12` (0-16; 16 is the maximum).
3. Say the phrases from the table above, with the schema's slot values.
4. Dismiss any old alarm or timer banner on the dashboard.
5. Backup if voice fails in a loud room: the dashboard's **Simulate a
   command** box runs every action.

## Stop

- **Pi, with services:** `systemctl --user stop vcm vcm-home` (they start
  again at the next boot; `systemctl --user disable vcm vcm-home` stops that).
- **Pi, by hand:** `tmux attach -t kiwi`, then `Ctrl-c` in each window. Or kill the
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
| Weather "isn't set up" / HTTP 401 | `api_key` missing, a new key not active yet (up to ~2 h), or the key was regenerated or disabled in the OpenWeatherMap account: make a new one, then `deploy_pi.sh --settings`. |
| Dashboard doesn't load | Home server not running; the device on another network; the Pi's IP changed (a bookmarked IP is stale: `ssh raspberrypi.local hostname -I`); or an Android phone, which can't open `raspberrypi.local` (use the IP). An old tab: reload. |
| The listener's wake score freezes (e.g. at 0.01) | It stopped receiving audio: its mic connection was cut (an audio restart, a mic replug). Current code exits and, with services, restarts in 3 s. By hand: `Ctrl-c` and start it again. `kiwi_doctor.py` shows "listener hears the mic: FAIL". |
| Kiwi stops hearing after the mic is replugged | Another input became the default. The WirePlumber rules (`deploy/wireplumber/51-kiwi-audio.lua`, installed by `deploy_pi.sh`) keep the USB mic as the input and disable the soundbar's mic. Check: `pactl get-default-source`. |
| Music or Kiwi's replies nearly silent | A duck that never ended. Fixed in code (#26, #27); to reset now: `kiwi_doctor.py --fix`, or restart raspotify for Spotify. |
| An old "It's 5:00 AM..." alarm banner stays on the dashboard | A test alarm that rang and was never stopped: click **Dismiss**. |
| `git pull` on the Pi fails | Local edits or untracked copies of tracked files on the Pi. Compare them with master, then `git restore` / remove them. Changes belong on the Mac. |

More: [DEPLOYMENT.md § Troubleshooting](DEPLOYMENT.md#troubleshooting).
