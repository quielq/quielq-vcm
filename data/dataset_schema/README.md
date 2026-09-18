# Dataset Schema — CSV export

`dataset_schema.csv` is a plain-CSV export of the class's fixed-vs-slotted
command taxonomy (19 intents: 13 fixed-phrase, 6 slotted), one row per
generated phrase (93 total: 13 × 3 phrasing variations + 6 × 3 templates
× 3 slot values).

**Source**: the taxonomy itself was shared in the class's "Dataset
Schema" spreadsheet. This CSV doesn't change the taxonomy at all, it's
the exact same 19 labels and phrasing, just exported to a format that's
simpler to load directly in pandas, a training script, or Excel without
going through Sheets.

<!-- Acknowledgment: this taxonomy was shared by a classmate as part of the
class's collective dataset effort — see DATASET.md's Acknowledgments
section. -->

**Columns**:
- `label` — one of the 19 canonical intent labels (e.g. `LIGHT_ON`, `ALARM`)
- `type` — `fixed` or `slotted`
- `phrase` — the full command text, slot placeholders already filled in
  for slotted intents (e.g. `Set an alarm for 9:00 PM`)
- `slot_value` — the specific value used (e.g. `9:00 PM`), empty for fixed
  intents

**Regenerating it**: this file is generated from
[`src/vcm/dataset/sources/dataset_schema.py`](../../src/vcm/dataset/sources/dataset_schema.py),
which holds the taxonomy as plain Python string literals (captured from
the live sheet's richest table, "Option B"). To regenerate after a
taxonomy update:

```python
from pathlib import Path
from vcm.dataset.sources.dataset_schema import export_csv
export_csv(Path("data/dataset_schema/dataset_schema.csv"))
```
