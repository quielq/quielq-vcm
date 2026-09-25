import pytest

from vcm.slots import SLOT_VOCAB, parse_slot, schema_values_covered


def test_every_schema_value_is_in_the_vocabulary():
    assert all(schema_values_covered().values()), schema_values_covered()


def test_vocab_sizes_and_uniqueness():
    for label, vocab in SLOT_VOCAB.items():
        assert len(vocab) == len(set(vocab)), label
    assert len(SLOT_VOCAB["TIMER"]) == 24
    assert len(SLOT_VOCAB["ALARM"]) == 28


@pytest.mark.parametrize(
    "text, value",
    [
        ("Timer 10 seconds", "10s"),
        ("Start a timer for 1 minute", "1m"),
        ("set a timer for five minutes", "5m"),
        ("Set a timer for 45 minutes.", "45m"),
        ("timer for forty five seconds", "45s"),
        ("a minute and a half timer", "1m30s"),
        ("set a timer for an hour and a half", "1h30m"),
        ("countdown for ninety seconds", "1m30s"),
        ("set a timer for half an hour", "30m"),
        ("timer for two hours", "2h"),
        ("timer for 25 minutes", "25m"),
        ("timer for 7 minutes and 30 seconds", None),  # 450s: not in vocab
        ("start a timer", None),
    ],
)
def test_timer(text, value):
    assert parse_slot("TIMER", text) == value


@pytest.mark.parametrize(
    "text, value",
    [
        ("Alarm 6:00 AM", "6:00 AM"),
        ("Wake me up at 8:00 AM", "8:00 AM"),
        ("Set an alarm for 9:00 PM", "9:00 PM"),
        ("wake me up at six thirty a.m.", "6:30 AM"),
        ("set an alarm for 7 in the morning", "7:00 AM"),
        ("alarm at ten at night", "10:00 PM"),
        ("set an alarm for 7:15 am", None),  # out of vocabulary
        ("set an alarm", None),
    ],
)
def test_alarm(text, value):
    assert parse_slot("ALARM", text) == value


@pytest.mark.parametrize(
    "text, value",
    [
        ("Brightness 20 percent", "20%"),
        ("Set the brightness to 100%", "100%"),
        ("dim the lights to seventy five percent", "75%"),
        ("change the lights to one hundred percent", "100%"),
        ("brightness 60", "60%"),
        ("make it brighter", None),
    ],
)
def test_brightness(text, value):
    assert parse_slot("BRIGHTNESS", text) == value


@pytest.mark.parametrize(
    "text, value",
    [
        ("Color Red", "red"),
        ("change the lights to warm white", "warm white"),
        ("set the lights to white", "white"),
        ("make the room purple", "purple"),
        ("change the color", None),
    ],
)
def test_color(text, value):
    assert parse_slot("COLOR", text) == value


def test_non_slotted_intent_has_no_value():
    assert parse_slot("PAUSE", "pause the song") is None
