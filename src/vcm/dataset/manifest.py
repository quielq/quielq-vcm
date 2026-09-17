"""Common manifest schema every dataset source gets normalized into.

Regardless of which taxonomy wins class-wide (Mark Andrian Macalalad's
fixed-vs-slotted schema, the other ~23-class flat draft, or something
else), every source (SLURP, FSC, the Dataset Schema recordings,
synthetic batches) ends up as rows of this one shape, with a per-source
label-mapping file resolving raw labels to whatever the canonical
taxonomy is at build time. This layer is deliberately taxonomy-agnostic
so it doesn't need rework once the class converges.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

FIELDS = ("audio_path", "label", "source", "is_synthetic", "speaker_id", "split")
VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ManifestRow:
    audio_path: str
    label: str
    source: str
    is_synthetic: bool
    speaker_id: str
    split: str

    def __post_init__(self) -> None:
        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {self.split!r}")


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_manifest(path: Path) -> list[ManifestRow]:
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            ManifestRow(
                audio_path=r["audio_path"],
                label=r["label"],
                source=r["source"],
                is_synthetic=r["is_synthetic"] == "True",
                speaker_id=r["speaker_id"],
                split=r["split"],
            )
            for r in reader
        ]


class UnmappedLabelError(KeyError):
    pass


def load_label_mapping(path: Path) -> dict[str, str]:
    """Load a source-label -> canonical-label mapping from a JSON file."""
    return json.loads(Path(path).read_text())


def apply_label_mapping(
    rows: list[ManifestRow], mapping: dict[str, str], on_missing: str = "raise"
) -> list[ManifestRow]:
    """Remap each row's label through `mapping` (source label -> canonical label).

    on_missing: "raise" (default) fails loudly on an unmapped source label
    (catches taxonomy drift early), "drop" silently excludes those rows,
    "keep" leaves the label unmapped.
    """
    if on_missing not in ("raise", "drop", "keep"):
        raise ValueError(f"on_missing must be 'raise', 'drop', or 'keep', got {on_missing!r}")

    remapped = []
    for row in rows:
        if row.label in mapping:
            remapped.append(replace(row, label=mapping[row.label]))
        elif on_missing == "raise":
            raise UnmappedLabelError(
                f"No mapping for source label {row.label!r} (source={row.source!r})"
            )
        elif on_missing == "keep":
            remapped.append(row)
        # "drop": excluded
    return remapped
