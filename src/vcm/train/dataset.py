"""PyTorch Dataset over data/dataset_manifest.csv.

Reuses vcm.dataset.manifest.read_manifest (the same manifest schema
built during dataset development, spanning all 5 combined sources) and
vcm.audio.features.extract_log_mel (the same feature extraction the
live push-to-talk path uses), so training and inference features never
drift apart.

Label space is the class-shared 19-intent taxonomy
(vcm.dataset.sources.dataset_schema.INTENT_LABELS) plus
unknown_background — deliberately NOT vcm.taxonomy.LABELS (the original
10-category taxonomy dispatch.py/main.py still use). Reconciling those
two is a separate, later integration step; this module's label space
is self-contained here and in whatever checkpoint training produces.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset

from vcm.audio.features import extract_log_mel
from vcm.dataset.manifest import ManifestRow, read_manifest
from vcm.dataset.sources.dataset_schema import INTENT_LABELS
from vcm.train.augment import spec_augment

LABELS: tuple[str, ...] = INTENT_LABELS + ("unknown_background",)
LABEL_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(LABELS)}


class ManifestDataset(Dataset):
    def __init__(self, rows: list[ManifestRow], augment: bool = False):
        self.rows = rows
        self.augment = augment

    @classmethod
    def from_csv(cls, csv_path: Path, split: str, augment: bool = False) -> "ManifestDataset":
        rows = [r for r in read_manifest(csv_path) if r.split == split]
        return cls(rows, augment=augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows[idx]
        audio, sample_rate = sf.read(row.audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        features = torch.from_numpy(extract_log_mel(audio, sample_rate=sample_rate))
        if self.augment:
            features = spec_augment(features)
        return features, LABEL_TO_INDEX[row.label]


def class_weights(rows: list[ManifestRow]) -> torch.Tensor:
    """Inverse-frequency class weights for a weighted CrossEntropyLoss.

    Label counts in the real manifest range ~20x (CALL=396 vs
    TEMPERATURE=8,691 in the train split) — worth correcting for from
    the start rather than letting the model ignore rare classes.
    """
    counts = torch.zeros(len(LABELS))
    for row in rows:
        counts[LABEL_TO_INDEX[row.label]] += 1
    counts = counts.clamp(min=1)
    return counts.sum() / (len(LABELS) * counts)
