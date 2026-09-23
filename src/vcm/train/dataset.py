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

import csv
import random
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


def load_distillation_labels(csv_path: Path) -> dict[str, torch.Tensor]:
    """Load scripts/generate_distillation_labels.py's output: audio_path ->
    a (len(LABELS),) teacher probability vector, remapped from the cascade
    classifier's own label order into this module's LABELS order.
    unknown_background always gets probability 0 (the cascade never
    predicts it — it has no transcript to classify)."""
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        teacher_labels = header[1:]  # audio_path, then one column per cascade label
        teacher_idx_in_labels = [LABEL_TO_INDEX[label] for label in teacher_labels]

        result: dict[str, torch.Tensor] = {}
        for row in reader:
            audio_path, probs = row[0], [float(p) for p in row[1:]]
            vec = torch.zeros(len(LABELS))
            for i, p in zip(teacher_idx_in_labels, probs):
                vec[i] = p
            result[audio_path] = vec
    return result


class ManifestDataset(Dataset):
    def __init__(
        self,
        rows: list[ManifestRow],
        augment: bool = False,
        distillation_labels: dict[str, torch.Tensor] | None = None,
    ):
        self.rows = rows
        self.augment = augment
        self.distillation_labels = distillation_labels

    @classmethod
    def from_csv(cls, csv_path: Path, split: str, augment: bool = False) -> "ManifestDataset":
        rows = [r for r in read_manifest(csv_path) if r.split == split]
        return cls(rows, augment=augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        audio, sample_rate = sf.read(row.audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        features = torch.from_numpy(extract_log_mel(audio, sample_rate=sample_rate))
        if self.augment:
            features = spec_augment(features)
        label_idx = LABEL_TO_INDEX[row.label]
        if self.distillation_labels is None:
            return features, label_idx
        # Rows with no teacher available (unknown_background — excluded when
        # generating distillation labels, it has no transcript) get an
        # all-zero vector; the loss treats an all-zero row as "skip the
        # distillation term for this example, use hard-label loss only".
        teacher_probs = self.distillation_labels.get(row.audio_path, torch.zeros(len(LABELS)))
        return features, label_idx, teacher_probs


def cap_per_class(rows: list[ManifestRow], max_per_class: int) -> list[ManifestRow]:
    """Randomly subsample each label down to at most max_per_class rows
    (labels already at or below the cap are returned unchanged). A
    dataset-level alternative to loss reweighting for class imbalance —
    see EXPERIMENTS.md Experiment 20. Deterministic given the caller has
    already seeded `random` (as train.py does via --seed)."""
    by_label: dict[str, list[ManifestRow]] = {}
    for row in rows:
        by_label.setdefault(row.label, []).append(row)
    capped: list[ManifestRow] = []
    for label_rows in by_label.values():
        if len(label_rows) > max_per_class:
            label_rows = random.sample(label_rows, max_per_class)
        capped.extend(label_rows)
    return capped


def class_counts(rows: list[ManifestRow]) -> torch.Tensor:
    """Raw per-class example counts, in LABELS order. Used by class_weights
    below (plain inverse-frequency) and by vcm.train.losses.effective_number_weights
    (the class-balanced alternative, see that function's docstring)."""
    counts = torch.zeros(len(LABELS))
    for row in rows:
        counts[LABEL_TO_INDEX[row.label]] += 1
    return counts.clamp(min=1)


def class_weights(rows: list[ManifestRow]) -> torch.Tensor:
    """Inverse-frequency class weights for a weighted CrossEntropyLoss.

    Label counts in the real manifest range ~20x (CALL=396 vs
    TEMPERATURE=8,691 in the train split) — worth correcting for from
    the start rather than letting the model ignore rare classes.
    """
    counts = class_counts(rows)
    return counts.sum() / (len(LABELS) * counts)
