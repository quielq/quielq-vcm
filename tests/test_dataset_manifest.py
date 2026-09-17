import pytest

from vcm.dataset.manifest import (
    ManifestRow,
    UnmappedLabelError,
    apply_label_mapping,
    read_manifest,
    write_manifest,
)


def _row(**overrides):
    defaults = dict(
        audio_path="clips/a.wav",
        label="LIGHT_ON",
        source="dataset_schema",
        is_synthetic=False,
        speaker_id="spk01",
        split="train",
    )
    defaults.update(overrides)
    return ManifestRow(**defaults)


def test_manifest_row_rejects_invalid_split():
    with pytest.raises(ValueError):
        _row(split="holdout")


def test_write_read_roundtrip(tmp_path):
    rows = [_row(), _row(audio_path="clips/b.wav", label="LIGHT_OFF", is_synthetic=True)]
    path = tmp_path / "manifest.csv"
    write_manifest(rows, path)
    loaded = read_manifest(path)
    assert loaded == rows


def test_apply_label_mapping_translates_labels():
    rows = [_row(label="iot_hue_lighton")]
    mapping = {"iot_hue_lighton": "LIGHT_ON"}
    remapped = apply_label_mapping(rows, mapping)
    assert remapped[0].label == "LIGHT_ON"
    assert remapped[0].audio_path == rows[0].audio_path  # everything else preserved


def test_apply_label_mapping_raises_on_missing_by_default():
    rows = [_row(label="totally_unmapped")]
    with pytest.raises(UnmappedLabelError):
        apply_label_mapping(rows, {})


def test_apply_label_mapping_drop():
    rows = [_row(label="mapped", audio_path="a"), _row(label="unmapped", audio_path="b")]
    remapped = apply_label_mapping(rows, {"mapped": "LIGHT_ON"}, on_missing="drop")
    assert [r.audio_path for r in remapped] == ["a"]


def test_apply_label_mapping_keep():
    rows = [_row(label="unmapped")]
    remapped = apply_label_mapping(rows, {}, on_missing="keep")
    assert remapped[0].label == "unmapped"


def test_apply_label_mapping_rejects_bad_mode():
    with pytest.raises(ValueError):
        apply_label_mapping([_row()], {}, on_missing="ignore")
