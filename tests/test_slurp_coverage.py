import json

from vcm.dataset.sources.slurp import coverage_report, intent_counts, load_records


def _write_jsonl(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_records_and_intent_counts(tmp_path):
    _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"intent": "play_music", "scenario": "play", "sentence": "play a song", "recordings": [{}, {}]},
            {"intent": "weather_query", "scenario": "weather", "sentence": "what's the weather", "recordings": [{}]},
        ],
    )
    _write_jsonl(
        tmp_path / "devel.jsonl",
        [{"intent": "play_music", "scenario": "play", "sentence": "play music", "recordings": [{}]}],
    )
    records = load_records(tmp_path, split_files=("train.jsonl", "devel.jsonl"))
    assert len(records) == 3
    counts = intent_counts(records)
    assert counts["play_music"] == 2
    assert counts["weather_query"] == 1


def test_coverage_report_sums_sentences_and_recordings(tmp_path):
    _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"intent": "play_music", "scenario": "play", "sentence": "a", "recordings": [{}, {}, {}]},
            {"intent": "iot_hue_lightoff", "scenario": "iot", "sentence": "b", "recordings": [{}]},
            {"intent": "hue_lightoff", "scenario": "iot", "sentence": "c", "recordings": [{}, {}]},
        ],
    )
    records = load_records(tmp_path, split_files=("train.jsonl",))
    report = coverage_report(records, mapping={"PLAY_MUSIC": ["play_music"], "LIGHT_OFF": ["iot_hue_lightoff", "hue_lightoff"]})
    assert report["PLAY_MUSIC"] == {"sentences": 1, "recordings": 3, "matched_intents": 1}
    assert report["LIGHT_OFF"] == {"sentences": 2, "recordings": 3, "matched_intents": 2}


def test_coverage_report_zero_for_unmatched_label(tmp_path):
    _write_jsonl(tmp_path / "train.jsonl", [{"intent": "play_music", "scenario": "play", "sentence": "a", "recordings": [{}]}])
    records = load_records(tmp_path, split_files=("train.jsonl",))
    report = coverage_report(records, mapping={"CALL": []})
    assert report["CALL"] == {"sentences": 0, "recordings": 0, "matched_intents": 0}
