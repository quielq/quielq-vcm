import json

from vcm.dataset.sources.slurp import coverage_report, intent_counts, is_valid_sentence, load_records


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


def test_is_valid_sentence_drops_pure_date_queries_for_time():
    # real SLURP sentence, no time content at all
    assert not is_valid_sentence("TIME", "what date is today")
    assert not is_valid_sentence("TIME", "is today march sixth")


def test_is_valid_sentence_keeps_mixed_date_and_time_for_time():
    # mentions time, so it's kept even though it also mentions date
    assert is_valid_sentence("TIME", "current date and time please")


def test_is_valid_sentence_keeps_real_time_queries_for_time():
    assert is_valid_sentence("TIME", "what time is it in florida")
    assert is_valid_sentence("TIME", "tell me the time now")


def test_is_valid_sentence_drops_known_junk_for_weather():
    assert not is_valid_sentence("WEATHER", "answer email from")
    assert not is_valid_sentence("WEATHER", "food will be given at the exhibition")


def test_is_valid_sentence_keeps_indirect_weather_queries():
    # doesn't say "weather" but is legitimately weather-dependent
    assert is_valid_sentence("WEATHER", "do i need a coat")
    assert is_valid_sentence("WEATHER", "should i bring an umbrella today")


def test_is_valid_sentence_drops_known_junk_for_message():
    assert not is_valid_sentence("MESSAGE", "how it's come to us")


def test_is_valid_sentence_keeps_real_message_queries():
    assert is_valid_sentence("MESSAGE", "send an email to my boss")


def test_is_valid_sentence_passes_through_unfiltered_labels():
    # no known quality issue for this label, so nothing gets excluded
    assert is_valid_sentence("PLAY_MUSIC", "anything at all")
    assert is_valid_sentence("LIST_REMINDERS", "pull up the shopping list")
