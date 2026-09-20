import csv

from vcm.dataset.sources.timers_and_such import coverage_report, load_records

FIELDS = ["path", "semantics", "speakerId", "transcription", "formatted_transcription", "asr_transcription", "WER"]


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(**overrides):
    defaults = dict(
        path="train-real/abc_prompt-1_0.wav",
        semantics="{'intent': 'SetTimer', 'slots': {'hours': 0, 'minutes': 0, 'seconds': 19}}",
        speakerId="spk1",
        transcription="start timer for 19 seconds",
        formatted_transcription="START TIMER FOR NINETEEN SECONDS",
        asr_transcription="START TIMER FOR NINETEEN SECONDS",
        WER="0.0",
    )
    defaults.update(overrides)
    return defaults


def test_load_records_maps_setttimer_and_setalarm(tmp_path):
    _write_csv(
        tmp_path / "train-real.csv",
        [
            _row(),
            _row(path="train-real/def_prompt-2_0.wav", semantics="{'intent': 'SetAlarm', 'slots': {}}"),
        ],
    )
    records = load_records(tmp_path, splits=("train-real",))
    assert len(records) == 2
    assert {r.intent for r in records} == {"SetTimer", "SetAlarm"}
    assert all(r.split == "train" for r in records)


def test_load_records_drops_unmapped_intents(tmp_path):
    _write_csv(
        tmp_path / "train-real.csv",
        [
            _row(),
            _row(path="train-real/ghi_prompt-3_0.wav", semantics="{'intent': 'SimpleMath', 'slots': {}}"),
            _row(path="train-real/jkl_prompt-4_0.wav", semantics="{'intent': 'UnitConversion', 'slots': {}}"),
        ],
    )
    records = load_records(tmp_path, splits=("train-real",))
    assert len(records) == 1
    assert records[0].intent == "SetTimer"


def test_load_records_maps_splits_correctly(tmp_path):
    _write_csv(tmp_path / "train-real.csv", [_row()])
    _write_csv(tmp_path / "dev-real.csv", [_row(path="train-real/x_prompt-5_0.wav")])
    _write_csv(tmp_path / "test-real.csv", [_row(path="train-real/y_prompt-6_0.wav")])
    records = load_records(tmp_path, splits=("train-real", "dev-real", "test-real"))
    splits = {r.split for r in records}
    assert splits == {"train", "val", "test"}


def test_coverage_report_counts_per_label(tmp_path):
    _write_csv(
        tmp_path / "train-real.csv",
        [
            _row(),
            _row(path="train-real/def_prompt-2_0.wav"),
            _row(path="train-real/ghi_prompt-3_0.wav", semantics="{'intent': 'SetAlarm', 'slots': {}}"),
        ],
    )
    records = load_records(tmp_path, splits=("train-real",))
    report = coverage_report(records)
    assert report["TIMER"] == 2
    assert report["ALARM"] == 1
