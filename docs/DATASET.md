# Dataset

The final model trains and is tested only on the **class master dataset**
and the final **Dataset Schema (Option B)**. The class agreed on both on
2026-10-01: the schema is the basis for labeling the 19 intents and their
slot values, its 93 phrases are the demo benchmark, and every group's data
is collated into one dataset with one fixed test set.

Dataset development is a **collective effort**. See
[Acknowledgments](#acknowledgments) at the end of this document. The
dataset this project built for itself before the master dataset existed is
documented in [EXPERIMENTS.md, Appendix A](EXPERIMENTS.md#appendix-a-the-projects-own-dataset-experiments-136).

## The class master dataset

- **Where:** [huggingface.co/datasets/airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands),
  revision **`da92a79`** (2026-10-02), pinned in every download command
  (also on the class Google Drive as `ai231-me2-gold-dataset`, and on the
  DGX as a shared `datasets` cache at **`/data/ai231`**, see below).
  Collated by Ailene (`airimonda` on Hugging Face). Audit and
  documentation:
  [claude.ai/artifact/PPNHMWd5rx9qcXTdcukV7s](https://claude.ai/artifact/PPNHMWd5rx9qcXTdcukV7s).
- **Schema:** [`data/dataset_schema/final_dataset_schema.csv`](../data/dataset_schema/final_dataset_schema.csv) (the class sheet,
  Option B), the same as the dataset's `variations.csv`.
  `vcm/dataset/sources/dataset_schema.py` holds it in code and
  [`data/dataset_schema/dataset_schema.csv`](../data/dataset_schema/dataset_schema.csv)
  is its export. A test checks all 93 phrases match.
- **Classes:** the 19 schema commands plus `OUT_OF_SCOPE` (noise,
  Filipino speech, near-miss requests and general speech). We keep
  OUT_OF_SCOPE as the dataset defines it: a class the model learns and the
  device never acts on ([TESTING.md](TESTING.md#out-of-scope-how-it-is-handled-and-tested)).
- **Wake word:** the dataset has no "hey kiwi" clips. The wake word's
  positives are our own (synthetic voices and the author's recordings);
  all of its negatives (commands, out-of-scope speech, noise, numerals)
  come from this dataset.
- **Slot values:** exactly the schema's 3 per slotted command (10 s / 30 s
  / 1 min; 6:00 AM / 8:00 AM / 9:00 PM; 18 / 22 / 26 degrees; 20 / 60 /
  100 percent; red / blue / green; drink water / study / exercise). The
  model's slot heads have 3 classes each (`vcm/slots.py`).
- **Audio:** 16 kHz mono 16-bit WAV.
- **`supplemental_synth`:** 5,856 more synthetic clips of the group's
  voices that are in no split, each tagged with the split its voice
  belongs to (train 3,983, test 1,612, holdout 261). Only train-voice
  clips may be added to training; the final model uses them.

## Splits

The dataset's own splits are used as published. No speaker or synthetic
voice is in more than one split. We only add a validation split, carved
out of train by speaker, so that picking the best epoch or setting never
looks at test.

| Split | Clips | Real | Synthetic | Filipino voices | Out of scope | Speakers | Use |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 9,285 | 2,019 | 7,266 | 708 | 251 | 331 | Training |
| val (ours, from train) | 1,448 | 340 | 1,108 | 5 | 19 | 46 | Choosing epochs and settings |
| **test** (class-fixed) | **4,443** | 824 | 3,619 | 203 | 76 | 144 | **Headline result** |
| holdout (class-fixed) | 202 | 96 | 106 | 87 | 16 | 7 | Raspberry Pi live-test set; also scored offline |
| numerals | 66,390 | all | — | — | all | 2,547 | 1,500 bare numbers sampled as OUT_OF_SCOPE examples |
| supplemental (ours, from `supplemental_synth`) | 3,461 | — | 3,461 | — | — | 60 | Extra synthetic training clips of train voices |

- **Test** has 47 clips per Option B variation ("Message" 43) plus 76
  out-of-scope clips. 81% of it is the group's synthetic voices; 777 clips
  are real people saying a command.
- **Validation** takes 1,448 clips, about 13% of the published train
  split, as whole speakers. Sources with fewer than 5 speakers stay
  in train, so all class recordings (Filipino speakers) and the noise
  clips are used for training. Seed 0, `scripts/build_me2_manifest.py`.
- **Supplemental** keeps only `supplemental_synth` clips whose voice is in
  our train split; the 2,395 others (voices in our val, test or holdout)
  are dropped, so val is the same with or without them.
- Hours: train 6.10 h, test 2.45 h, holdout 0.18 h, numerals 21.8 h,
  supplemental_synth 2.73 h.

Train by source (train + val): group synthetic set 8,305 (+69 synthetic
out-of-scope), SLURP 1,010, class recordings 684, SNIPS 331, Fluent Speech
Commands 256, Common Voice 36, Timers and Such 31, Speech Commands noise 11.
78% of train is synthetic.

## The shared copy on the DGX (`/data/ai231`)

The class uploaded the dataset to the DGX at `/data/ai231`, as a Hugging
Face `datasets` cache (`load_dataset("airimonda/ai231-me2-voice-commands",
cache_dir="/data/ai231")`, the loader in `/data/ai231/gold_dataset.py`). We
compared it clip by clip with our pinned download on 2026-10-02:

| Split | Clips | Audio bytes | Order | Labels and metadata (all 18 columns) |
|---|---:|---|---|---|
| train | 10,733 | identical | identical | identical |
| test | 4,443 | identical | identical | identical |
| holdout | 202 | identical | identical | identical |
| numerals | 66,390 | identical | identical | identical |
| supplemental_synth | 5,856 | identical | identical | identical (added to the shared copy at 17:43) |

Its default splits **are** revision `da92a79` and its `supplemental_synth`
is the one the final model trained on, so building from it gives the same
manifest, slot labels, metadata and audio files as the download (checked),
and every result in this repo holds for it. **The pipeline reads everything
from `/data/ai231`; nothing is downloaded on the DGX.**

The shared copy is the class's revision `6947f13`, which also has
`synthetic_negatives` (1,000 train + 250 test generated out-of-scope clips,
DEMAND / MS-SNSD noise). We don't use it. Hugging Face has a later revision
(`5abbe53`, 2026-10-02 19:32) that adds `supplemental_fil` (14,120
Filipino-accented synthetic clips, train only), which is not in the shared
copy either. **Training stays on the content above. Adding any new class
data is a decision to make first, not something the scripts pick up.**

`scripts/verify_shared_dataset.py` checks any copy against
`data/dataset_schema/me2_fingerprint_da92a79.json` in about 20 seconds. The
fingerprint holds a clip count and a hash over every clip's audio and labels,
per split, including `supplemental_synth`. The script also lists the configs
it doesn't use. If the shared copy is ever refreshed to different content,
the check fails and `scripts/reproduce.sh` falls back to the pinned download.

```bash
python scripts/verify_shared_dataset.py --shared-cache /data/ai231   # expect "identical to the revision every result is on"
```

## Build it

```bash
# 1. Download the pinned revision: train/test/holdout, numerals, supplemental_synth (about 3.6 GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download(
  'airimonda/ai231-me2-voice-commands', repo_type='dataset', revision='da92a79ffde3031d5bb2a25138d9dd7d9f7ed006',
  local_dir='data/me2/hf', allow_patterns=['data/*', 'supplemental_synth/*', 'README.md', 'variations.csv'])"

# 2. Write the audio and this repo's manifest, slot labels and metadata
python scripts/build_me2_manifest.py --numerals --supplemental
```

On the DGX, skip step 1; everything comes from the shared copy:

```bash
python scripts/verify_shared_dataset.py --shared-cache /data/ai231
python scripts/build_me2_manifest.py --shared-cache /data/ai231 --numerals --supplemental
```

Step 2 writes, all under `data/me2/` (gitignored):

| File | What |
|---|---|
| `<split>/audio/*.wav` | every clip, by split |
| `manifest.csv` | `audio_path, label, source, is_synthetic, speaker_id, split` (the common row shape in [`src/vcm/dataset/manifest.py`](../src/vcm/dataset/manifest.py)) |
| `slot_labels.csv` | 10,425 slot labels from the dataset's `slot_value`, in `vcm/slots.py`'s value format |
| `metadata.csv` | accent group, variation, transcript and Whisper check per clip, for evaluation breakdowns |

## Acknowledgments

Ailene (`airimonda`) collated the master dataset and its documentation.
Mark Macalacad shared the Dataset Schema (Option B) and generated the
group's synthetic set. Xela Ubalde shared her recordings. Every classmate
who recorded commands made the real-speech test possible. The public
datasets inside the master dataset are SLURP, Fluent Speech Commands, SNIPS
SLU, Timers and Such, Common Voice, Google Speech Commands, MLEnd and the
Multi-Sensor Voice Command dataset; each keeps its own license (see the
dataset card).
