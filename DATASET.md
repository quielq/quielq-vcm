# Dataset Development — How to Reproduce This

This walks through every dataset-side step in this repo, in order, so
anyone (classmates included) can reproduce the exact same output from a
fresh clone. See [VCM_Architecture_Review.md](VCM_Architecture_Review.md)
Section 9 for the full research/rationale behind these choices — this
doc only covers running the code.

## Setup (same as the main README)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 1. The taxonomy (Mark Andrian Macalalad's fixed-vs-slotted schema)

[`src/vcm/dataset/sources/dataset_schema.py`](src/vcm/dataset/sources/dataset_schema.py)
holds the 19-label taxonomy (13 fixed-phrase intents, 6 slotted) as plain
Python data, captured from the live "Dataset Schema" sheet's richest
table ("Option B"). It expands into 93 phrases total.

```bash
python -c "from vcm.dataset.sources.dataset_schema import generate_phrases; print(len(generate_phrases()))"
```
**Expect**: `93`.

To regenerate the CSV export at `data/dataset_schema/dataset_schema.csv`
(already committed, only needed if the taxonomy changes):
```bash
python -c "
from pathlib import Path
from vcm.dataset.sources.dataset_schema import export_csv
export_csv(Path('data/dataset_schema/dataset_schema.csv'))
"
```
**Expect**: a 94-line CSV (93 phrases + header). See
[`data/dataset_schema/README.md`](data/dataset_schema/README.md) for the
column format.

## 2. SLURP coverage check (real data, real numbers)

```bash
python scripts/slurp_coverage.py
```
**What it does**: downloads SLURP's `train/devel/test.jsonl` (~13MB of
annotation text, no audio) from `pswietojanski/slurp` on GitHub into
`data/external/slurp/` (gitignored, so this doesn't bloat the repo — the
script re-downloads if that folder isn't there), then prints a table of
how many real SLURP sentences/recordings map to each of the 19 taxonomy
labels, using the empirically-verified mapping in
[`src/vcm/dataset/sources/slurp.py`](src/vcm/dataset/sources/slurp.py).

**Expect** (verified output, reproduced exactly by running the command
above from a clean checkout):
```
Loaded 16521 sentences / 72396 recordings

Label               sentences  recordings  matched intents
PLAY_MUSIC                911        3883                1
WEATHER                   834        3412                1
TIME                      490        2558                1
LIGHT_ON                   30         126                1
LIGHT_OFF                 213        1091                2
PAUSE                       0           0                0
STOP                        0           0                0
NEXT                        0           0                0
VOLUME_UP                 135        1059                1
VOLUME_DOWN                71         587                1
CALL                        0           0                0
MESSAGE                   523        1925                1
LIST_REMINDERS            296        1308                1
TIMER                       0           0                0
ALARM                     253        1503                1
TEMPERATURE                 0           0                0
BRIGHTNESS                229        1234                4
COLOR                     183         751                1
CREATE_REMINDER           234        1005                1

Zero SLURP coverage: PAUSE, STOP, NEXT, CALL, TIMER, TEMPERATURE
```
Takes under a minute on a normal connection (13MB download + parsing
16,521 JSON lines).

**If the mapping looks wrong for a label**: the raw SLURP→taxonomy
mapping is the `LABEL_MAPPING` dict at the top of `sources/slurp.py`,
built by inspecting every one of SLURP's 93 actual `intent` values (not
guessed from the paper) — edit it there and re-run the script.

## 3. FSC coverage check (needs your own file — not automated here)

Unlike SLURP, Fluent Speech Commands isn't freely downloadable without a
Kaggle account (or a Fluent.ai license request). If you have
`train_data.csv` (columns: `path, speakerId, transcription, action,
object, location`):

```bash
python -c "
from vcm.dataset.sources.fsc import load_records, coverage_report
records = load_records('path/to/train_data.csv')
print(coverage_report(records))
"
```
The `LABEL_MAPPING` in
[`src/vcm/dataset/sources/fsc.py`](src/vcm/dataset/sources/fsc.py) is
currently a **placeholder** (only 4 of 19 labels mapped) — it needs
verifying against real `(action, object)` values the same way SLURP's
was, once someone has the actual file. If you run this for real, please
share the numbers, that placeholder mapping should get replaced with a
verified one.

## 4. Synthetic-audio QA gate

[`src/vcm/dataset/qa/synthetic_check.py`](src/vcm/dataset/qa/synthetic_check.py)
generalizes Anthony Navarez's `simple-audio-transcriber` transcribe-and-
compare pattern. It needs a real transcriber backend to run for real
(`FasterWhisperTranscriber`, requires `pip install faster-whisper`, which
downloads an ASR model on first use — not a default dependency here since
it's optional QA tooling, not part of the deployed model):

```python
from pathlib import Path
from vcm.dataset.qa.synthetic_check import FasterWhisperTranscriber, screen_batch

transcriber = FasterWhisperTranscriber()  # pulls a Whisper model on first run
pairs = [
    (Path("synthetic/lights_off_01.wav"), "Lights off"),
    (Path("synthetic/play_music_03.wav"), "Play a song"),
]
results = screen_batch(pairs, transcriber, threshold=0.8)
for r in results:
    print(r.audio_path, r.confidence, "PASS" if r.passed else "REVIEW")
```
Anything below the threshold is the "route to human spot-check" bucket
from Anthony's original write-up, not automatically discarded.

## Manifest format, for combining sources later

[`src/vcm/dataset/manifest.py`](src/vcm/dataset/manifest.py) is the
common row shape (`audio_path, label, source, is_synthetic, speaker_id,
split`) every source above eventually normalizes into, via a
per-source label-mapping file (`apply_label_mapping()`). Not exercised
end-to-end yet, no source has real audio wired up in this repo, but the
tests in `tests/test_dataset_manifest.py` show the intended usage.

## Tests

```bash
python -m pytest
```
42 tests total; the dataset-specific ones are `tests/test_dataset_*.py`,
`tests/test_slurp_coverage.py`, `tests/test_fsc_coverage.py`, and
`tests/test_synthetic_check.py` — all pure logic against synthetic
fixtures, no network or real data files required to pass.
