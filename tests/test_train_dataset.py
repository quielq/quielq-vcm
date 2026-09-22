import numpy as np
import soundfile as sf
import torch

from vcm.dataset.manifest import ManifestRow, write_manifest
from vcm.train.dataset import LABEL_TO_INDEX, LABELS, ManifestDataset, cap_per_class, class_weights


def _write_wav(path, duration_s=1.0, sample_rate=16000, seed=0):
    rng = np.random.default_rng(seed)
    audio = rng.uniform(-0.1, 0.1, size=int(duration_s * sample_rate)).astype("float32")
    sf.write(path, audio, sample_rate)
    return path


def _row(**overrides):
    defaults = dict(
        audio_path="",
        label="PLAY_MUSIC",
        source="test",
        is_synthetic=False,
        speaker_id="spk1",
        split="train",
    )
    defaults.update(overrides)
    return ManifestRow(**defaults)


def test_labels_cover_19_intents_plus_background():
    assert len(LABELS) == 20
    assert LABELS[-1] == "unknown_background"
    assert LABEL_TO_INDEX["PLAY_MUSIC"] == 0
    assert LABEL_TO_INDEX["unknown_background"] == 19


def test_from_csv_filters_by_split(tmp_path):
    wav_a = _write_wav(tmp_path / "a.wav")
    wav_b = _write_wav(tmp_path / "b.wav", seed=1)
    rows = [
        _row(audio_path=str(wav_a), label="LIGHT_ON", split="train"),
        _row(audio_path=str(wav_b), label="LIGHT_OFF", split="val"),
    ]
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(rows, manifest_path)

    train_ds = ManifestDataset.from_csv(manifest_path, split="train")
    val_ds = ManifestDataset.from_csv(manifest_path, split="val")
    assert len(train_ds) == 1
    assert len(val_ds) == 1


def test_getitem_returns_features_and_label_index(tmp_path):
    wav = _write_wav(tmp_path / "clip.wav", duration_s=2.0)
    ds = ManifestDataset([_row(audio_path=str(wav), label="WEATHER")])

    features, label_idx = ds[0]
    assert isinstance(features, torch.Tensor)
    assert features.dtype == torch.float32
    assert features.ndim == 2  # (N_MELS, n_frames)
    assert label_idx == LABEL_TO_INDEX["WEATHER"]


def test_getitem_handles_stereo_audio(tmp_path):
    rng = np.random.default_rng(0)
    stereo = rng.uniform(-0.1, 0.1, size=(16000, 2)).astype("float32")
    wav = tmp_path / "stereo.wav"
    sf.write(wav, stereo, 16000)
    ds = ManifestDataset([_row(audio_path=str(wav), label="CALL")])

    features, label_idx = ds[0]
    assert features.ndim == 2
    assert label_idx == LABEL_TO_INDEX["CALL"]


def test_class_weights_favors_rare_labels():
    rows = [_row(label="PLAY_MUSIC") for _ in range(100)] + [_row(label="CALL") for _ in range(5)]
    weights = class_weights(rows)
    assert weights[LABEL_TO_INDEX["CALL"]] > weights[LABEL_TO_INDEX["PLAY_MUSIC"]]


def test_class_weights_handles_zero_count_label_without_dividing_by_zero():
    rows = [_row(label="PLAY_MUSIC")]
    weights = class_weights(rows)
    assert torch.isfinite(weights).all()


def test_cap_per_class_shrinks_oversized_labels():
    rows = [_row(label="TEMPERATURE") for _ in range(100)] + [_row(label="CALL") for _ in range(5)]
    capped = cap_per_class(rows, max_per_class=20)
    counts = {"TEMPERATURE": 0, "CALL": 0}
    for r in capped:
        counts[r.label] += 1
    assert counts["TEMPERATURE"] == 20
    assert counts["CALL"] == 5  # already under the cap, untouched


def test_cap_per_class_leaves_undersized_labels_unchanged():
    rows = [_row(label="CALL") for _ in range(5)]
    capped = cap_per_class(rows, max_per_class=1000)
    assert len(capped) == 5


def test_cap_per_class_total_matches_sum_of_per_label_caps():
    rows = (
        [_row(label="TEMPERATURE") for _ in range(100)]
        + [_row(label="LIGHT_ON") for _ in range(50)]
        + [_row(label="CALL") for _ in range(5)]
    )
    capped = cap_per_class(rows, max_per_class=30)
    assert len(capped) == 30 + 30 + 5
