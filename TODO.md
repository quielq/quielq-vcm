# To-do

Current state (see EXPERIMENTS.md): the deployable model is the CRNN
(Experiment 34: Experiment 31's encoder frozen + slot heads, 107K params,
426 KB fp32, 85.5% real-speech test accuracy, slot values 68–90%) with the
"Hey Kiwi" wake word v2 (25K params, 107 KB). The course asks for a wake word instead of
a button, and the target device is a Raspberry Pi 5 with a model under
1 MB.

## In progress

- [x] **Experiment 34**: frozen slot heads keep intent at 85.48% (slot
  acc 77.6%); wake word v2 catches all 10 real takes at 0.95 (1.34 false
  wake-ups/h). Models in `models/`, report in `reports/exp34_report.md`.
- [x] **Experiment 32** (intent + slots in one model): slot values work
  (ALARM 98%, COLOR 87%), but intent fell 3.5pp and int8 cost 6.8pp more,
  so fp32 ships. Superseded by Experiment 34.
- [x] **Experiment 33** ("Hey Kiwi"): 6.7% / 16.4% false rejects (clean /
  noisy) at 0.67 false wake-ups per hour, threshold 0.95.
- [ ] **Deployment**: copy `models/*.onnx` (fp32) to the Pi and follow
  DEPLOYMENT.md (benchmark, field-test the wake word at 0.95 vs 0.98).
- [ ] **Joint slots at weight 0.3, 2 more seeds** (DGX): one seed matched
  Experiment 32's slot accuracy (83.4%) at 84.8% intent. If the seeds
  hold at ~85%, ship it instead of the frozen model (+6 points slots).

## Next

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

- [x] `scripts/demo_infer.py` defaults to `models/vcm_intent.onnx` (prints
  slot values); README / MODEL.md §10 refreshed for the CRNN.
- [x] Wake-word evaluation's noise column is reproducible (seeded
  `add_noise`).

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
