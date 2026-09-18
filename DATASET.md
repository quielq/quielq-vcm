# Dataset Development — How to Reproduce This

This walks through every dataset-side step in this repo, in order, so
anyone (classmates included) can reproduce the exact same output from a
fresh clone. See [VCM_Architecture_Review.md](VCM_Architecture_Review.md)
Section 9 for the candidate-source research and strategic rationale —
this doc only covers running the code and current status.

Dataset development is a **collective effort**. See "Acknowledgments"
at the end of this document for how this pipeline builds on work shared
across the class.

## Setup (same as the main README)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Current status

| Label coverage | Source |
|---|---|
| 13 of 19 labels | Real audio (SLURP) |
| All 19 labels | Synthetic, QA-filtered audio (class-shared "Option B" dataset) |
| 6 labels with zero real coverage | PAUSE, STOP, NEXT, CALL, TIMER, TEMPERATURE — synthetic-only for now |

Running `scripts/build_manifest.py` (step 5 below) combines both into
one training-ready manifest.

## 1. The taxonomy (class-shared fixed-vs-slotted schema)

[`src/vcm/dataset/sources/dataset_schema.py`](src/vcm/dataset/sources/dataset_schema.py)
holds the 19-label taxonomy (13 fixed-phrase intents, 6 slotted) as plain
Python data, captured from the class's shared taxonomy sheet's richest
table ("Option B" phrasing richness — not to be confused with the
"Option B" *dataset* in step 4 below, same source naming, different
artifact). It expands into 93 phrases total.

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

### What the three columns mean

- **`sentences`** — a "sentence" in SLURP is one unique text+intent
  annotation, a single prompt someone was asked to say (e.g. `"wake me
  up at ten"`, labeled `alarm_set`). This column is a count of *distinct
  prompts* mapped to that taxonomy label.
- **`recordings`** — each sentence was recorded multiple times, usually
  by different crowdworkers, often in paired mic setups (a close-mic
  `-headset` take plus a room-mic take of the same prompt). This column
  counts *actual audio files*, always ≥ the sentence count. Concretely:
  one SLURP sentence (`slurp_id 9024`, text `"event"`, intent
  `calendar_set`) has 9 separate recordings in the raw data — that's
  one sentence contributing 9 to a `recordings` total.
- **`matched intents`** — SLURP has its own internal vocabulary of 93
  intents, finer-grained (and messier) than this taxonomy's 19 labels.
  This is how many of SLURP's raw intents got mapped onto one canonical
  label. Most are 1-to-1; `LIGHT_OFF` is 2 because SLURP has two
  overlapping intents for it (legacy duplicate, e.g. "turn off the
  light" vs. "turn off lamp"), and `BRIGHTNESS` is 4 for the same reason
  across dim-up/dim-down and two legacy naming schemes (e.g. "dim the
  lights" vs. "the lights are too bright"). See `LABEL_MAPPING` in
  `sources/slurp.py` for exactly which raw intents feed each label.

**A phrasing-looseness caveat worth knowing before trusting this data
blindly**: SLURP's crowdsourced sentences aren't clean scripted commands
the way this project's own taxonomy phrases are — e.g. `"open clock"` is
labeled `alarm_set` (→ `ALARM`), and `"olly brighten the lights"` maps to
`BRIGHTNESS`. Real, spoken-in-the-wild phrasing is part of why SLURP is
useful (Section 9 already notes its sentences run long/natural), but it
means a "covered" label here doesn't guarantee phrasing anywhere close to
this project's own command set.

**If the mapping looks wrong for a label**: the raw SLURP→taxonomy
mapping is the `LABEL_MAPPING` dict at the top of `sources/slurp.py`,
built by inspecting every one of SLURP's 93 actual `intent` values (not
guessed from the paper) — edit it there and re-run the script.

## 3. Pulling real SLURP audio

```bash
pip install datasets soundfile
python scripts/fetch_slurp_audio.py
```
**What it does**: streams real audio for the 13 covered labels from a
HuggingFace parquet mirror (`yhfang/slurp_dataset_audio_subset`) filtered
by `slurp_id` against the local metadata from step 2 — no need to
download SLURP's full 3.9GB Zenodo archive. Writes real `.flac` files to
`data/external/slurp_audio/<split>/` plus a `manifest.csv` already in
this project's common schema (see `manifest.py` below). Takes several
minutes; run `scripts/slurp_coverage.py` (step 2) first if
`data/external/slurp/*.jsonl` doesn't exist yet.

## 4. Class-shared synthetic dataset ("Option B")

A classmate-contributed synthetic dataset, publicly shared on GitHub:
17,658 QA-filtered recordings, voice-cloned from 100 real reference
speakers (84 foreign, 16 genuinely Filipino-English), speaker-disjoint
train/val/test split, clean + noisy acoustic conditions, generated
directly against this project's own 19-label taxonomy (so it covers all
19, including every label SLURP has zero coverage for) and already
screened through a classmate-built transcribe-and-compare QA tool (942
of 18,600 originals flagged and excluded).

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/markandrian30/AI231.git /tmp/option_b_repo
cd /tmp/option_b_repo && git sparse-checkout set MEX2/OptionB
mkdir -p <repo-root>/data/external/mark_option_b
mv MEX2/OptionB/manifest.csv <repo-root>/data/external/mark_option_b/
mkdir -p <repo-root>/data/external/mark_option_b/audio
mv MEX2/OptionB/*/  <repo-root>/data/external/mark_option_b/audio/
```
[`src/vcm/dataset/sources/mark_option_b.py`](src/vcm/dataset/sources/mark_option_b.py)
loads it — no label-mapping layer needed, its `intent` column already
matches our 19 canonical labels 1:1 (verified against the real
`manifest.csv`, not assumed).

## 5. Combined manifest

```bash
python scripts/build_manifest.py
```
Combines SLURP (step 3) and the Option B dataset (step 4) into one
`data/dataset_manifest.csv`, in the common schema every source
normalizes into (`audio_path, label, source, is_synthetic, speaker_id,
split` — see `manifest.py`), and prints a real-vs-synthetic breakdown
per label. This is the file a training script should read from.

## 6. FSC coverage check (needs your own file — not automated here)

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
was, once someone has the actual file.

## 7. Synthetic-audio QA gate

[`src/vcm/dataset/qa/synthetic_check.py`](src/vcm/dataset/qa/synthetic_check.py)
generalizes the transcribe-and-compare pattern from the QA tool used to
filter the Option B dataset above, so it can screen output from *any*
generator, not just the one it was built for. It needs a real
transcriber backend to run for real (`FasterWhisperTranscriber`,
requires `pip install faster-whisper`, which downloads an ASR model on
first use — not a default dependency here since it's optional QA
tooling, not part of the deployed model):

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
Anything below the threshold is the "route to human spot-check" bucket,
not automatically discarded.

## Manifest format

[`src/vcm/dataset/manifest.py`](src/vcm/dataset/manifest.py) is the
common row shape (`audio_path, label, source, is_synthetic, speaker_id,
split`) every source normalizes into, via a per-source label-mapping
layer (`apply_label_mapping()`) so no source is hardcoded to one
taxonomy. `scripts/build_manifest.py` (step 5) is the concrete example
of combining sources through it.

## Tests

```bash
python -m pytest
```
46 tests total; the dataset-specific ones are `tests/test_dataset_*.py`,
`tests/test_slurp_coverage.py`, `tests/test_fsc_coverage.py`,
`tests/test_mark_option_b.py`, and `tests/test_synthetic_check.py` — all
pure logic against synthetic fixtures, no network or real data files
required to pass.

## Acknowledgments

This pipeline is built directly on top of work shared across the class,
not developed in isolation. Specifically:

- The **working taxonomy** (Section 1 above) was shared by a classmate
  and is adopted here as-is; this project only added a plain-CSV export
  alongside it for safer downstream tooling.
- The **synthetic dataset** used for the six labels with no real
  coverage (Section 4) was generated and shared by a classmate, using
  a speaker-disjoint voice-cloning methodology that deliberately
  includes real Filipino-English reference speakers.
- The **QA-screening approach** (Section 7) generalizes a
  transcribe-and-compare tool a classmate built and shared, originally
  used to filter the synthetic dataset above.
- A precedented public synthetic-command dataset was identified and
  shared by a classmate as a resource to check before generating new
  synthetic data from scratch, avoiding duplicated effort.
- Several other classmates contributed dataset leads, reference
  implementations, and corpus-sizing corrections that shaped the
  candidate-source research in `VCM_Architecture_Review.md` Section 9.

Names and individual attribution are intentionally kept out of this
document so it stays a stable, factual reference rather than a running
log — but the collective nature of this work is real, and everything
above exists because of it.
