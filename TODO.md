# To-do

Current state (see EXPERIMENTS.md): the deployable model is the CRNN
(Experiment 31, 96K params, 85.1% real-speech test accuracy, 98–99% on
the class schema phrasings). The course asks for a wake word instead of
a button, and the target device is a Raspberry Pi 5 with a model under
1 MB.

## In progress

- [ ] **Experiment 32: intent + slot values.** Slot heads for TIMER
  duration, ALARM time, BRIGHTNESS percent and COLOR (`vcm/slots.py`),
  the schema values plus common extras. Needs the `slots2` synthetic
  batch and the Snips re-split. Code done; DGX run pending.
- [ ] **Experiment 33: "Hey Kiwi" wake word.** Separate ~25K-param
  detector (`vcm/wakeword/`), measured by false rejects and false
  wake-ups per hour. Code done; data generation and training on the DGX
  pending.
- [ ] **Deployment**: export both models to int8 ONNX, commit them to
  `models/`, then follow DEPLOYMENT.md on the Pi (benchmark, field-test
  the wake word).

## Actions and dashboard (built: `vcm/home/`, DEPLOYMENT.md step 8b)

Working now: all 19 intents act (virtual lamp/thermostat, timers, alarms,
reminders, clock, weather, Spotify + local music, speaker volume,
calls/messages through the Mac), with a live web dashboard. To do:

- [ ] Try it with real accounts: Spotify (Premium + raspotify on the Pi),
  OpenWeatherMap key, the Mac phone bridge with a real contact.
- [ ] Run the home server and voice loop together on the Pi (systemd, step 9).

### Phone / mobile (testing through the Mac for now)

- [ ] **Bluetooth hands-free on the Pi (HFP)**: pair the iPhone to the Pi
  like a car kit (BlueZ + oFono/PipeWire) and dial real calls with audio
  through the Pi's mic and speaker. Works without the Mac, but Linux
  Bluetooth setup is fiddly, and iPhones don't allow sending SMS this way.
- [ ] **Phone notification route**: the Pi sends a push notification
  (e.g. ntfy) that opens the call/message on the iPhone with one tap. No
  Mac needed; reliable.
- [ ] **iOS Shortcuts route**: a Shortcut on the iPhone triggered by the
  device (via a notification or an HTTP automation) that sends the message
  without the Mac.
- [ ] Contacts beyond the default ("call Mom" only for now); needs a
  contact slot in the model (below).
- [ ] Dashboard as an installable phone web app (PWA icon, offline shell).

### Tech debt

- [ ] **Model gaps that limit actions** (need a DGX run each):
  - TEMPERATURE has no direction or value ("turn up the heat" and "turn it
    down" share one label), so voice can only *report* the temperature;
    the dashboard buttons change the target. Fix: a TEMPERATURE slot
    (up / down / 16–30 °C) from the FSC transcripts plus schema phrasings.
  - CREATE_REMINDER can't capture its text (closed-set model). Fix: a slot
    head for the schema's reminder tasks, plus saving the spoken audio as
    a playable voice note on the dashboard. Until then, text is typed on
    the dashboard.
  - CALL/MESSAGE have no contact or message content: always the default
    contact and default message.
- [ ] Reminders have no due time, so they're never announced; only timers
  and alarms fire. Alarms are one-shot (no repeat days).
- [ ] The home server and dashboard have **no authentication**: anyone on
  the same network can control them. Add a token or a login before using
  it outside a home network.
- [ ] Remove the old pipeline: `vcm/dispatch.py`, `vcm/taxonomy.py` (the
  10-category labels), the stub model in `vcm/inference/model.py`, and
  `vcm/main.py` are superseded by `vcm_listen.py` + `vcm/home/`. Rewire or
  delete them and their tests.
- [ ] `pyproject.toml` base dependencies still include training/desktop
  libraries (librosa, pynput, python-miio, soundfile); the Pi installs with
  `--no-deps` from `requirements-pi.txt` to avoid them. Split them into
  extras so a plain install is minimal.
- [ ] Spoken replies use `say` (Mac) / espeak-ng (Pi); the Piper TTS
  backend is still a stub.
- [ ] The real Xiaomi bulb only gets on/off/brightness from the
  dispatcher; color isn't sent to it yet.
- [ ] Spotify: "play <song>" isn't possible (no song slot), only
  resume/default playlist. STOP maps to pause (Spotify has no stop).

## Next

- [ ] Make the Experiment 32 model the default in `scripts/demo_infer.py`
  (still defaults to the old DS-CNN) and refresh README / MODEL.md §10.
- [ ] Integrate into `vcm/main.py`: `dispatch.py` still uses the older
  10-category taxonomy, so the 20 intents and slot values need mapping to
  actions (timer length, alarm time, color, brightness).
- [ ] Ask the adviser whether "no attention/transformer layers" rules out
  CRNN's attention pooling (129 params per head). If so, swap in plain
  pooling or dilated convs.
- [ ] Diagnose the live-test COLOR → VOLUME_UP confusion (needs `--debug`
  recordings of the phrases used).

- [ ] Measure on a Pi Zero 2 W if one is available (`scripts/benchmark_pi.py`).
  The original Pi Zero (ARMv6) would need a pure-numpy model runtime.

## Accuracy toward 90% (real speech)

- [ ] Label audit: list clips where the ASR-cascade confidently disagrees
  with the label; review and fix.
- [ ] Second targeted-phrasing round for the free-form SLURP classes
  (COLOR 55%, CREATE_REMINDER 64%, BRIGHTNESS 73% real speech).
- [ ] Capacity check: CRNN at ~200–300K params (3 seeds).
- [ ] Real recordings from classmates for CALL / NEXT / LIST_REMINDERS
  (tool built: `scripts/record_real_examples.py`; waiting on adviser OK).
- [ ] Watch the BRIGHTNESS real-speech dip from Experiment 31 (−2.6pp).

## Done recently

- [x] Pi runtime slimmed to numpy + onnxruntime + sounddevice: numpy
  log-mel (`vcm/audio/dsp.py`, matches librosa) cut peak memory from
  ~330 MB to ~90 MB, so a 512 MB Pi Zero 2 W is viable.
- [x] Wake word chosen: "Hey Kiwi" (EXPERIMENTS.md, "Wake-word selection").
- [x] Snips speaker leakage: splits now by speaker
  (`scripts/resplit_snips_by_speaker.py`); applying it to the DGX
  manifest is part of Experiment 32.
- [x] ONNX export + int8 quantization + ONNX Runtime inference
  (`scripts/export_onnx.py`, `vcm/deploy/`); verified identical to
  PyTorch, 270 KB int8 for Experiment 31, no torch needed on the Pi.
- [x] Silence gate on the loudest 300 ms (PR #4).
- [x] Reject threshold "please repeat" at 0.6 confidence.
