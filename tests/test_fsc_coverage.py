import csv

from vcm.dataset.sources.fsc import classify, coverage_report, load_records, to_manifest_rows

FIELDS = ["path", "speakerId", "transcription", "action", "object", "location"]


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(**overrides):
    defaults = dict(
        path="wavs/a.wav",
        speakerId="spk1",
        transcription="turn on the lights",
        action="activate",
        object="lights",
        location="none",
    )
    defaults.update(overrides)
    return defaults


def test_load_records(tmp_path):
    csv_path = tmp_path / "train_data.csv"
    _write_csv(csv_path, [_row()])
    records = load_records(csv_path, split="train")
    assert len(records) == 1
    assert records[0].action == "activate"
    assert records[0].object == "lights"
    assert records[0].speaker_id == "spk1"
    assert records[0].split == "train"


def test_classify_maps_clean_triples():
    from vcm.dataset.sources.fsc import FscRecord

    r = FscRecord(path="a", speaker_id="s", transcription="t", action="activate", object="lights", location="kitchen", split="train")
    assert classify(r) == "LIGHT_ON"


def test_classify_splits_deactivate_music_by_transcription():
    from vcm.dataset.sources.fsc import FscRecord

    pause = FscRecord(path="a", speaker_id="s", transcription="Pause the music", action="deactivate", object="music", location="none", split="train")
    stop = FscRecord(path="b", speaker_id="s", transcription="Turn off the music", action="deactivate", object="music", location="none", split="train")
    assert classify(pause) == "PAUSE"
    assert classify(stop) == "STOP"


def test_classify_drops_unmapped_commands():
    from vcm.dataset.sources.fsc import FscRecord

    r = FscRecord(path="a", speaker_id="s", transcription="Change language to Chinese", action="change language", object="Chinese", location="none", split="train")
    assert classify(r) is None


def test_coverage_report_counts_by_canonical_label(tmp_path):
    csv_path = tmp_path / "train_data.csv"
    _write_csv(
        csv_path,
        [
            _row(path="a", action="activate", object="lights", location="none"),
            _row(path="b", action="activate", object="lights", location="kitchen"),
            _row(path="c", action="deactivate", object="lights", location="none"),
            _row(path="d", action="change language", object="Chinese", location="none"),
        ],
    )
    records = load_records(csv_path, split="train")
    report = coverage_report(records)
    assert report == {"LIGHT_ON": 2, "LIGHT_OFF": 1}


def test_to_manifest_rows_skips_unmapped_and_prefixes_audio_root(tmp_path):
    csv_path = tmp_path / "train_data.csv"
    _write_csv(
        csv_path,
        [
            _row(path="wavs/a.wav", action="activate", object="lights", location="none"),
            _row(path="wavs/b.wav", action="change language", object="Chinese", location="none"),
        ],
    )
    records = load_records(csv_path, split="train")
    rows = to_manifest_rows(records, audio_root=tmp_path / "fsc_root")

    assert len(rows) == 1  # the change-language row is dropped
    assert rows[0].label == "LIGHT_ON"
    assert rows[0].source == "fsc"
    assert rows[0].is_synthetic is False
    assert rows[0].speaker_id == "fsc_spk1"
    assert rows[0].split == "train"
    assert rows[0].audio_path == str(tmp_path / "fsc_root" / "wavs/a.wav")
