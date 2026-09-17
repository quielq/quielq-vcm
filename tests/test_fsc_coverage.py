import csv

from vcm.dataset.sources.fsc import coverage_report, load_records


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "speakerId", "transcription", "action", "object", "location"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_load_records(tmp_path):
    csv_path = tmp_path / "train_data.csv"
    _write_csv(
        csv_path,
        [
            {
                "path": "wavs/a.wav",
                "speakerId": "spk1",
                "transcription": "turn on the lights",
                "action": "activate",
                "object": "lights",
                "location": "none",
            }
        ],
    )
    records = load_records(csv_path)
    assert len(records) == 1
    assert records[0].action == "activate"
    assert records[0].object == "lights"


def test_coverage_report_matches_action_object_pairs(tmp_path):
    csv_path = tmp_path / "train_data.csv"
    _write_csv(
        csv_path,
        [
            {"path": "a", "speakerId": "s", "transcription": "t", "action": "activate", "object": "lights", "location": "none"},
            {"path": "b", "speakerId": "s", "transcription": "t", "action": "activate", "object": "lights", "location": "kitchen"},
            {"path": "c", "speakerId": "s", "transcription": "t", "action": "deactivate", "object": "lights", "location": "none"},
        ],
    )
    records = load_records(csv_path)
    report = coverage_report(records, mapping={"LIGHT_ON": [("activate", "lights")], "LIGHT_OFF": [("deactivate", "lights")]})
    assert report == {"LIGHT_ON": 2, "LIGHT_OFF": 1}
