# Audit: what we considered and didn't ship

This file records the options, experiments, hardware and code this project
tried or planned but did not keep in the final version. Each entry says what
it was, what happened, and where the evidence is. The shipped system is
described in the [main README](../../README.md) and
[ARCHITECTURE.md](../ARCHITECTURE.md).

Everything here is still in git history. Legacy code is also kept, unrun,
in [legacy_code/](legacy_code/).

## 1. Model architectures

| Option | What happened | Evidence |
|---|---|---|
| **DS-CNN** (Zhang et al. 2017, "Hello Edge"), 24–26K params | Baseline for Experiments 1–27. It plateaued at ~75% val and 68–71% real-speech test. Its receptive field is ~240 ms, shorter than a word like "temperature", so it can't use word order. Replaced by the CRNN (+11 points on real speech). | EXPERIMENTS.md Exp 1, 7, 11–27, 28 |
| **BC-ResNet** (Kim et al. 2021) | The original plan's recommended architecture. It lost to DS-CNN at every size tried (44–59% at 10K params, 67.5% at 26K vs DS-CNN's 72.4%). | Exp 3–6, 10 |
| TC-ResNet, MatchboxNet | Surveyed in the original plan, never implemented. The CRNN result made them unnecessary. | [original_model_plan.md](original_model_plan.md) §2 |
| **ASR cascade**: faster-whisper `base` + TF-IDF/logistic regression | Best accuracy measured: **90.6%** real-speech test vs. our 84.8%. Not allowed by the assignment ("ASR models are not desirable for on-device computing because of footprint"). It needs ~149 MB of models, a 478–696 MB memory peak and 440–950 ms per command, and still needs a separate wake word. Kept only as the reference point in [FOOTPRINT.md](../FOOTPRINT.md). | Exp 26 |
| Knowledge distillation from the cascade into DS-CNN | +0.3 points overall, with large per-class swings (PLAY_MUSIC +16, VOLUME_DOWN −12). Not a clean win; superseded by the CRNN. The `--distill-weight` flag stays in `train.py`, off by default. | Exp 27 |
| Auxiliary word-level CTC loss on Whisper transcripts | −0.7 points on real speech, worse as the weight grew. The `--ctc-weight` flag stays, off by default. | Exp 30 |
| Intent + slots trained jointly at slot weight 1.0 | Slot values worked but intent fell 3.5 points. Weight 0.3 fixed this and ships. | Exp 32, 34 |
| Slot heads on a frozen intent encoder | Kept intent at 85.5%, but slot values were ~6 points weaker and too unreliable in live tests. Shipped for a day, then replaced by joint training. Kept as `models/vcm_intent_frozen.onnx` for comparison. | Exp 34 |
| CRNN at ~200–300K params | Planned capacity check, not run. | — |

## 2. Training techniques tried and dropped

| Technique | Result | Evidence |
|---|---|---|
| SpecAugment (random time/frequency masking) | Hurt twice (−2 and −8 points): random time masks erase the one word that separates "volume up" from "volume down". Replaced by waveform augmentation. | Exp 2, 8 |
| Confusable-pair loss at alpha 1.0 and 3.0; COLOR/BRIGHTNESS as a third group | 1.0 barely helped, 3.0 overshot, the third group traded one confusion for another. alpha 2.0 on two groups is kept in the final recipe. | Exp 14–17 |
| Focal loss, class-balanced weights | Never improved on the starting checkpoint. | Exp 18–19 |
| Capping examples per class (`--max-per-class`) | Worse overall than inverse-frequency class weights. | Exp 20–23 |
| Dropout + weight decay + label smoothing together | Broad regression on the small DS-CNN (it was underfitting). | Exp 24 |
| Targeted (non-random) time masking | Parked: needs word-level alignments the pipeline doesn't have. | EXPERIMENTS.md "Parked" |

## 3. Deployment choices dropped

| Option | Why not | Evidence |
|---|---|---|
| **int8 quantization** (the original plan) | Cost 6.8 points of real-speech accuracy to save 134 KB, and was no faster on the CPU. Both models ship as fp32 ONNX (539 KB total). | Exp 32, 33 |
| Quantization-aware training | Only needed if int8 was needed; fp32 fits the 1 MB budget. | — |
| TFLite / XNNPACK, ncnn | ONNX Runtime was fast enough (9.9 ms per command on the Pi 5), so no second export path. | [FOOTPRINT.md](../FOOTPRINT.md) |
| librosa on the device | Pulled in numba, LLVM and scipy: +186 MB of memory, ~330 MB peak. Replaced by a numpy re-implementation (`vcm/audio/dsp.py`, tested to match librosa): ~90 MB peak. | FOOTPRINT.md |
| Original Pi Zero / Zero W (ARMv6) | No onnxruntime builds exist for 32-bit ARM. | DEPLOYMENT.md |

## 4. Interaction design

| Option | What happened |
|---|---|
| **Push-to-talk button** (the original plan) | The course requires a wake word. "Hey Kiwi" replaced it. The GPIO button still works as an option (`vcm_listen.py --trigger button`). |
| Wake-word candidates | Cory, Ellie, Corinne, Orly, Nini, Kernel, Conan, Crayon, Cronin and Carina were scored for false-trigger risk. Cory would have fired on ~1 in 12 commands ("increase", "decrease", "create"). Kiwi had the fewest everyday sound-alikes. See EXPERIMENTS.md "Wake-word selection". |
| Wake-word v1 (Experiment 33) | Its Whisper-based QA threw away more than half of the positive clips. v2 (Experiment 34) kept every plausible clip plus the author's recordings, halving missed wakes. |
| Wake threshold 0.95 (offline choice) | Too strict in a real room. Lowered to 0.6 after live tests on the Pi, and 0.4 while music plays. |

## 5. Data sources not used

| Source | Why not |
|---|---|
| Mozilla Common Voice (Filipino-accented subset) | Read sentences, not commands; never integrated. |
| MASSIVE | Text-only beyond English, and its English audio is SLURP, which we already use. |
| "Indian EmoSpeech Command Dataset" | No verifiable source was ever found. |
| A classmate's CosyVoice voice-cloned batch | Never pulled in or QA-screened. |
| SLURP `lists_query` → LIST_REMINDERS | Dropped after an audit: 196 of 197 sentences were shopping lists or playlists, not reminders. |
| Real recordings from classmates for CALL / NEXT / LIST_REMINDERS | Tool built (`legacy_code/scripts/record_real_examples.py`) but on hold pending the adviser's approval for collecting personal voice data. Never run. |
| Reminder value "call home" | A stale value in the schema. The class recordings say "exercise", so Experiment 36 retrained with "exercise". |

## 6. Hardware dropped from the original plan

| Item | Why |
|---|---|
| SIM800L GSM module | 2G-only, and the Philippines is shutting down 2G by end of 2026. Calls go through the iPhone instead (via the Mac bridge). |
| TP-Link Tapo bulb and plug | Not needed: the dashboard's virtual lamp and thermostat show every light and temperature command. A Xiaomi bulb can still be driven through `[xiaomi]` in the settings. |
| Fan on a smart plug as the thermostat | Replaced by a simulated thermostat; the Sense HAT supplies the room temperature. |
| DHT22 temperature sensor | Backordered; the Sense HAT covers it. |
| USB speakerphone, UPS HAT, display | Portability polish (the original "Tier 2"), not needed for the graded work. |

The original plan, with its hardware research and wiring diagrams, is kept
as [original_architecture_review.md](original_architecture_review.md).

## 7. Former TODO list, resolved

`TODO.md` was deleted in the cleanup for review. Every item went to one of
three places:

**Done** (no longer open):
- Deploy both models to the Pi, benchmark, field-test the wake word.
- Run the home server and the voice loop as services on the Pi (`deploy_pi.sh --services`).
- Joint slot weight 0.3 on more seeds (Experiments 35–36 ran 3 seeds each).
- TEMPERATURE and CREATE_REMINDER slot values (Experiments 35–36).
- A natural voice: Piper TTS works and runs on the Pi (the TODO said it was a stub).
- Real Spotify, weather and phone-bridge accounts, configured on the Pi.
- Remove the old pipeline (`main.py`, `dispatch.py`, `taxonomy.py`, the stub model): moved to [legacy_code/](legacy_code/).

**Superseded** (no longer relevant):
- Integrate the 20 intents into `vcm/main.py`: the home server (`vcm/home/`) replaced `main.py`.
- Diagnose one live COLOR → VOLUME_UP confusion: not pursued as a
  separate item. COLOR's real-speech accuracy is covered by the open
  accuracy work below.

**Still open**: listed under "Future enhancements" in the
[README](../../README.md#future-enhancements), including the BRIGHTNESS
real-speech decline first seen in Experiment 31 (75.5% → 69.1% by
Experiment 36).

## 8. Legacy code in `legacy_code/`

The first version of this repo (before any model was trained) was a
push-to-talk skeleton with a stub classifier and a 10-category taxonomy. The
final system replaced it with `scripts/vcm_listen.py` (wake word → model)
and `vcm/home/` (actions and dashboard). These files are kept for reference.
They are **not importable from the package** and their tests are not run.

| File | What it was | Replaced by |
|---|---|---|
| `src/vcm/main.py` | Push-to-talk loop: button → record → stub model → dispatch | `scripts/vcm_listen.py` |
| `src/vcm/dispatch.py`, `src/vcm/taxonomy.py` | 10-category labels mapped to actions | `vcm/home/dispatcher.py`, 19 intents from `vcm/dataset/sources/dataset_schema.py` |
| `src/vcm/inference/` | Stub and random "models" used before training | `vcm/deploy/runtime.py` (ONNX Runtime) |
| `src/vcm/actions/{alarms,calls,clock,reminders,thermostat,timers}.py` | Action functions for the old dispatcher | `vcm/home/` (scheduler, state, phone bridge) |
| `tests/test_dispatch.py`, `tests/test_inference_stub.py` | Tests for the above | — |
| `scripts/demo_infer.py` | Push-to-talk demo for a checkpoint or ONNX file | `scripts/vcm_listen.py` (`--trigger button` for push-to-talk, `--intent-model` for another model) |
| `scripts/demo_infer_cascade.py` | Push-to-talk demo of the ASR cascade | `scripts/measure_footprint.py asr` measures the cascade |
| `scripts/record_real_examples.py` | Recording tool for real CALL/NEXT/LIST_REMINDERS clips | On hold (section 5) |

`vcm/actions/` keeps the modules the home server still uses: `weather`,
`lights` (optional Xiaomi bulb), `music` and `media_control` (local files via
mpv).
