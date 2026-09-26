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

**3. Mac: open the dashboard** at http://raspberrypi.local:8000.

**4. Say "Hey Kiwi"**, pause briefly, then a command. With
`--show-scores`, the wake score jumps toward 1.00 when you say it.

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
| "play some music" | Replies that music isn't set up (Spotify not configured yet) |

The dashboard's **Simulate a command** box runs the same actions without
speaking. If a command works there but not by voice, the problem is
recognition, not the action.

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
| Static or knocking in recordings | Level too high (clipping), the mic in a blue USB 3 port (use black USB 2), or undervoltage (`vcgencmd get_throttled` should print `0x0`). |
| No spoken replies | No speaker set up: `pactl list short sinks` shows only `auto_null`. Plug in a USB speaker or pair a Bluetooth one (DEPLOYMENT.md 8b). |
| "home server unreachable" | Start `vcm.home.server` in the other tmux window first. |
| CALL / MESSAGE only "simulated" or failing | Bridge not running on the Mac, Mac asleep, `bridge_url` / `bridge_token` mismatch, or macOS blocked incoming connections (allow Python in the firewall prompt). |
| Weather "isn't set up" / HTTP 401 | `api_key` missing, or a new key not active yet (up to ~2 h). |
| Dashboard doesn't load from the Mac | Home server not running, or Mac and Pi on different networks (`ping raspberrypi.local`). |
| `git pull` on the Pi fails | Local edits or untracked copies of tracked files on the Pi. Compare them with master, then `git restore` / remove them. Changes belong on the Mac. |

More: [DEPLOYMENT.md § Troubleshooting](DEPLOYMENT.md#troubleshooting).
