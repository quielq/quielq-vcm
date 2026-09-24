import pytest

from vcm.dataset.sources.targeted_synth import PHRASES, normalize_for_qa, schema_phrases, word_error_rate


@pytest.mark.parametrize("label", sorted(PHRASES))
def test_schema_phrasings_come_first_and_unduplicated(label):
    schema = schema_phrases(label)
    assert PHRASES[label][: len(schema)] == schema
    assert len(PHRASES[label]) == len(set(PHRASES[label]))


def test_schema_phrasing_examples():
    assert "stop playing music" in PHRASES["STOP"]
    assert "change the brightness to 60 percent" in PHRASES["BRIGHTNESS"]
    assert "color red" in PHRASES["COLOR"]


@pytest.mark.parametrize(
    "intended, heard",
    [
        ("set a timer for forty five minutes", "Set a timer for 45 minutes."),
        ("brightness one hundred percent", "Brightness 100%."),
        ("timer 10 seconds", "Timer, ten seconds."),
    ],
)
def test_normalize_matches_digit_and_word_forms(intended, heard):
    assert word_error_rate(normalize_for_qa(intended), normalize_for_qa(heard)) == 0.0


def test_stop_heard_as_start_fails_qa_threshold():
    assert word_error_rate(normalize_for_qa("stop music"), normalize_for_qa("Start music.")) > 0.2
