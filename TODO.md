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

- [x] Wake word chosen: "Hey Kiwi" (EXPERIMENTS.md, "Wake-word selection").
- [x] Snips speaker leakage: splits now by speaker
  (`scripts/resplit_snips_by_speaker.py`); applying it to the DGX
  manifest is part of Experiment 32.
- [x] ONNX export + int8 quantization + ONNX Runtime inference
  (`scripts/export_onnx.py`, `vcm/deploy/`); verified identical to
  PyTorch, 270 KB int8 for Experiment 31, no torch needed on the Pi.
- [x] Silence gate on the loudest 300 ms (PR #4).
- [x] Reject threshold "please repeat" at 0.6 confidence.
