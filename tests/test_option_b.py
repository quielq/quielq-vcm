import csv

from vcm.dataset.sources.option_b import label_counts, load_manifest, to_manifest_rows

FIELDS = ["path", "label", "intent", "speaker", "split", "phrase_id", "variant_id", "transcript", "slot", "slot_value", "duration_sec"]


def _write_manifest(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(**overrides):
    defaults = dict(
        path="CALL/CALL_s1_v1_clean.wav",
        label="CALL",
        intent="CALL",
        speaker="s1",
        split="train",
        phrase_id="v1",
        variant_id="clean",
        transcript="Call",
        slot="",
        slot_value="",
        duration_sec="1.2",
    )
    defaults.update(overrides)
    return defaults


def test_load_manifest(tmp_path):
    csv_path = tmp_path / "manifest.csv"
    _write_manifest(csv_path, [_row()])
    records = load_manifest(csv_path)
    assert len(records) == 1
    assert records[0].intent == "CALL"
    assert records[0].speaker == "s1"


def test_to_manifest_rows_maps_intent_to_label(tmp_path):
    csv_path = tmp_path / "manifest.csv"
    _write_manifest(
        csv_path,
        [_row(path="ALARM_6_00AM/ALARM_6_00AM_s1_v1_clean.wav", label="ALARM_6_00AM", intent="ALARM")],
    )
    records = load_manifest(csv_path)
    rows = to_manifest_rows(records)
    assert rows[0].label == "ALARM"  # canonical label, not the folder-level "ALARM_6_00AM"
    assert rows[0].source == "option_b"
    assert rows[0].is_synthetic is True
    assert rows[0].audio_path == "ALARM_6_00AM/ALARM_6_00AM_s1_v1_clean.wav"


def test_to_manifest_rows_prefixes_audio_root(tmp_path):
    csv_path = tmp_path / "manifest.csv"
    _write_manifest(csv_path, [_row()])
    records = load_manifest(csv_path)
    rows = to_manifest_rows(records, audio_root=tmp_path / "OptionB")
    assert rows[0].audio_path == str(tmp_path / "OptionB" / "CALL/CALL_s1_v1_clean.wav")


def test_label_counts(tmp_path):
    csv_path = tmp_path / "manifest.csv"
    _write_manifest(
        csv_path,
        [
            _row(intent="CALL"),
            _row(intent="CALL", speaker="s2"),
            _row(intent="TIMER", label="TIMER_10s"),
        ],
    )
    records = load_manifest(csv_path)
    assert label_counts(records) == {"CALL": 2, "TIMER": 1}
