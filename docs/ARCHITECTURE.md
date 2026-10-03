# Architecture

How the Voice Command Model (VCM) is built, as shipped. The original
planning document is archived as
[archive/original_architecture_review.md](archive/original_architecture_review.md);
what changed from that plan, and why, is in [archive/AUDIT.md](archive/AUDIT.md).

Courses: AI222 (Supervised Learning) and AI231 (ML Operations), UP Diliman.

## 1. The assignment

> ASR models are not desirable for on-device computing because of footprint.
> The goal is to build a tiny Voice Command Model (VCM) that can understand
> the most common commands humans tell their smart devices.

Target commands, ranked from Amazon Alexa / Google Home usage data:

1. Play music
2. Ask a question (weather, time)
3. Lights on/off
4. Dim / color lights
5. Set a timer
6. Set an alarm
7. Adjust the thermostat
8. Media control: pause, stop, next, volume
9. Reminders and lists
10. Calls and messaging

Tasks: build a dataset (collective), build and train a VCM (individual),
design a benchmark (collective), validate the VCM (individual), and build a
real-world demo that runs on a Raspberry Pi 4/5 (individual).

Constraints: tiny enough to run in real time on a Raspberry Pi; standalone,
with no cloud models; no LLM; a wake word instead of a button. From the
adviser meeting: only the listed commands are needed, not full ASR; the
device may use the internet for *actions* (weather, Spotify) but not for
*recognition*; the class discussed a target of under 3% error.

## 2. System overview

```mermaid
flowchart LR
    MIC[USB microphone] --> WAKE
    subgraph PI[Raspberry Pi 5]
        WAKE["Wake word<br/>kiwi_wakeword.onnx, 107 KB<br/>every 100 ms"] -->|"Hey Kiwi"| REC["Record the command<br/>until 0.6 s of quiet"]
        REC --> FEAT["Log-mel features<br/>numpy"]
        FEAT --> MODEL["Intent + slot model<br/>vcm_intent.onnx, 1.46 MB"]
        MODEL -->|"confidence ≥ 0.6"| HOME["Home server<br/>vcm.home.server :8000"]
        MODEL -->|"< 0.6"| REPEAT["Please repeat"]
        HOME --> ACT["Actions: lamp, thermostat, timers,<br/>alarms, reminders, music, volume"]
        HOME --> TTS[Spoken reply]
        HOME --> DASH[Web dashboard]
    end
    TTS --> SPK[USB speaker]
    HOME -.->|internet| API["Weather API, Spotify"]
    HOME -.->|home Wi-Fi| MAC["Mac phone bridge"] -.-> PHONE[iPhone: calls, messages]
```

Recognition (everything up to the model's output) runs on the Pi with no
network. Only actions use the network: the weather API, Spotify, and the
Mac that relays calls and messages to the iPhone.

Two processes run on the Pi as systemd user services, restarted within 3 s
if they exit:

| Service | Program | Job |
|---|---|---|
| `vcm.service` | `scripts/vcm_listen.py` | Microphone → wake word → command recording → intent + slot model → HTTP POST to the home server |
| `vcm-home.service` | `python -m vcm.home.server` | Runs the action, speaks the reply, keeps state (`data/home_state.json`), fires timers and alarms, serves the dashboard |

The listener also lowers ("ducks") any music by 12 dB while it listens, and
uses a lower wake threshold (0.4 instead of 0.6) while music plays.

## 3. Key decisions

| Decision | Choice | Why |
|---|---|---|
| Problem framing | Closed-set intent classification with slot values; no transcript | The assignment rules out ASR on the device. 19 intents cover the 10 command categories. |
| Model | CRNN, 108K parameters, audio → intent + 6 slot heads | DS-CNN couldn't see whole words (70% real speech); the CRNN reached 85% ([MODEL.md](MODEL.md)) |
| Slot values | One classification head per slotted intent, over a fixed vocabulary | Small, no transcript needed; covers the class schema's values plus common extras |
| Wake word | "Hey Kiwi", a separate 25K-parameter CRNN | Fewest sound-alikes among 11 candidates; a separate tiny model is cheap enough to run always |
| Export and runtime | fp32 ONNX + ONNX Runtime + numpy features | int8 lost 6.8 accuracy points; numpy features avoid librosa (~330 MB → ~90 MB peak memory) |
| Benchmark | Real-speech test accuracy (+ macro, per class, confusable pairs), slot accuracy, wake-word misses and false wake-ups per hour, latency and memory on the Pi | Synthetic clips score 94–99% and hide real-speech errors; plain accuracy hides per-class failures |
| Data | Real speech where it exists (SLURP, FSC, Timers and Such, Snips), class-shared synthetic speech where it doesn't | Three intents have no public real recordings ([DATASET.md](DATASET.md)) |
| Actions | A home server with a virtual lamp and thermostat on a web dashboard | Every command shows a visible result without extra smart-home hardware |
| Not shipped | ASR cascade (Whisper + text classifier), 90.6% accurate | ~280× the disk and ~150× the latency ([FOOTPRINT.md](FOOTPRINT.md)) |

## 4. Commands: intents, slots and actions

| Category | Intents | Slot value | What happens (`vcm/home/dispatcher.py`) |
|---|---|---|---|
| 1. Music | PLAY_MUSIC | — | Spotify on the Pi (raspotify); local files via mpv as a fallback |
| 2. Questions | WEATHER, TIME | — | OpenWeatherMap; system clock; spoken |
| 3. Lights | LIGHT_ON, LIGHT_OFF | — | Virtual lamp on the dashboard (plus a Xiaomi bulb, if configured) |
| 4. Dim / color | BRIGHTNESS, COLOR | 20 / 60 / 100 percent; red / blue / green | Virtual lamp |
| 5. Timer | TIMER | 10 s / 30 s / 1 min | Scheduled; rings until "stop" |
| 6. Alarm | ALARM | 6:00 AM / 8:00 AM / 9:00 PM | Scheduled; rings until "stop" |
| 7. Thermostat | TEMPERATURE | 18 / 22 / 26 degrees | Simulated thermostat; room temperature from the Sense HAT |
| 8. Media control | PAUSE, STOP, NEXT, VOLUME_UP, VOLUME_DOWN | — | Spotify / mpv; speaker volume |
| 9. Reminders | CREATE_REMINDER, LIST_REMINDERS | drink water / study / exercise | Stored and shown on the dashboard; listed aloud |
| 10. Calls / messages | CALL, MESSAGE | — | Through the Mac bridge to the iPhone (default contact); simulated if the bridge is off |
| — | `OUT_OF_SCOPE` (`unknown_background` in models before Exp 37) | — | Nothing |

## 5. Hardware (as built)

| Item | Role |
|---|---|
| Raspberry Pi 5, 8 GB (Cytron kit), 32 GB microSD, Raspberry Pi OS 64-bit | Runs everything |
| USB microphone (MUSIC-BOOST MB-306; any USB mic) | Input |
| USB soundbar (Dell AC511) | Replies, alarms, music |
| Sense HAT | Room temperature for the thermostat |
| GPIO 17 pushbutton (optional) | Push-to-talk instead of the wake word (`--trigger button`) |
| MacBook (optional) | Runs `scripts/mac_phone_bridge.py`, which places calls and sends iMessages/SMS through the paired iPhone |

WirePlumber rules (`deploy/wireplumber/51-kiwi-audio.lua`) pin the USB mic
as the input and the soundbar as the output, so a replugged device can't
steal either role.

## 6. Software stack

| Layer | Used |
|---|---|
| Training | PyTorch on UP's DGX (A100), shared node; [TRAINING.md](TRAINING.md) |
| Dataset tooling | Hugging Face `datasets`, `remotezip`, faster-whisper (QA and transcripts only), Chatterbox TTS (synthetic clips) |
| Device runtime | Python 3.11, numpy, ONNX Runtime, sounddevice (PortAudio), requests |
| Voice replies | Piper (natural voice) or espeak-ng; macOS `say` on the laptop |
| Home server | Python standard library HTTP server + one static HTML page, updated live over server-sent events |
| Integrations | Spotify Web API + raspotify, OpenWeatherMap, Mac bridge (AppleScript → FaceTime/Messages) |
| Deployment | `scripts/deploy_pi.sh` (rsync + venv over SSH), systemd user services, `scripts/kiwi_doctor.py` preflight check |

Hardware-specific code sits behind small interfaces in `vcm/hal/`
(`get_button()`, `read_temperature()`), with a Raspberry Pi implementation
and a laptop fallback chosen at start-up, so the same code runs on both.

## 7. Open questions

- **"No attention layers."** The original constraint list said no
  attention/transformer layers. The CRNN's attention pooling is a single
  linear scorer over time (129 parameters), not self-attention. If the rule
  is meant literally, average pooling is a drop-in replacement to test.
- **Synthetic data.** The class proceeded on the assumption that synthetic
  (voice-cloned) data is allowed; it is about 35% of the training data.
- **Recording classmates.** A tool to record real CALL, NEXT and
  LIST_REMINDERS commands from classmates is built but on hold until the
  adviser approves collecting personal voice data.
