import pytest

from vcm.dataset.sources.dataset_schema import SLOTTED_INTENTS
from vcm.slots import (
    SLOT_VOCAB,
    parse_alarm_time,
    parse_color,
    parse_duration,
    parse_percent,
    parse_slot,
    schema_values_covered,
)


def test_every_schema_value_is_in_the_vocabulary():
    assert all(schema_values_covered().values()), schema_values_covered()


def test_vocab_is_exactly_the_schema_values():
    for label, vocab in SLOT_VOCAB.items():
        assert len(vocab) == len(set(vocab)) == 3, label
        assert {parse_slot(label, v) for v in SLOTTED_INTENTS[label]["values"]} == set(vocab), label


@pytest.mark.parametrize(
    "text, value",
    [
        ("Timer 10 seconds", "10s"),
        ("Countdown for 30 seconds", "30s"),
        ("Start a timer for 1 minute", "1m"),
        ("start a timer for one minute", "1m"),
        ("set a timer for five minutes", None),  # parses, but not a schema value
        ("start a timer", None),
    ],
)
def test_timer(text, value):
    assert parse_slot("TIMER", text) == value


@pytest.mark.parametrize(
    "text, seconds",
    [
        ("set a timer for five minutes", 300),
        ("timer for forty five seconds", 45),
        ("a minute and a half timer", 90),
        ("set a timer for an hour and a half", 5400),
        ("set a timer for half an hour", 1800),
        ("timer for 7 minutes and 30 seconds", 450),
        ("start a timer", None),
    ],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize(
    "text, value",
    [
        ("Alarm 6:00 AM", "6:00 AM"),
        ("Wake me up at 8:00 AM", "8:00 AM"),
        ("Set an alarm for 9:00 PM", "9:00 PM"),
        ("wake me up at six a.m.", "6:00 AM"),
        ("set an alarm for 9 at night", "9:00 PM"),
        ("set an alarm for 7 in the morning", None),  # not a schema value
        ("set an alarm", None),
    ],
)
def test_alarm(text, value):
    assert parse_slot("ALARM", text) == value


@pytest.mark.parametrize(
    "text, value",
    [
        ("wake me up at six thirty a.m.", "6:30 AM"),
        ("set an alarm for 7 in the morning", "7:00 AM"),
        ("alarm at ten at night", "10:00 PM"),
        ("set an alarm", None),
    ],
)
def test_parse_alarm_time(text, value):
    assert parse_alarm_time(text) == value


@pytest.mark.parametrize(
    "text, value",
    [
        ("Brightness 20 percent", "20%"),
        ("Adjust brightness to 100%", "100%"),
        ("brightness level sixty percent", "60%"),
        ("brightness 60", "60%"),
        ("dim the lights to seventy five percent", None),
        ("make it brighter", None),
    ],
)
def test_brightness(text, value):
    assert parse_slot("BRIGHTNESS", text) == value


def test_parse_percent_outside_schema():
    assert parse_percent("dim the lights to seventy five percent") == 75


@pytest.mark.parametrize(
    "text, value",
    [
        ("Change color to Red", "red"),
        ("Switch color to Blue", "blue"),
        ("set color to green", "green"),
        ("make the room purple", None),
        ("change the color", None),
    ],
)
def test_color(text, value):
    assert parse_slot("COLOR", text) == value


def test_parse_color_outside_schema():
    assert parse_color("change the lights to warm white") == "warm white"
    assert parse_color("set the lights to white") == "white"


def test_temperature_and_reminder():
    assert parse_slot("TEMPERATURE", "Set the temperature to 22 degrees") == "22 degrees"
    assert parse_slot("TEMPERATURE", "temperature twenty six degrees") == "26 degrees"
    assert parse_slot("CREATE_REMINDER", "Remind me to Drink water") == "drink water"
    assert parse_slot("CREATE_REMINDER", "Create a reminder to Study") == "study"


def test_non_slotted_intent_has_no_value():
    assert parse_slot("PAUSE", "pause the song") is None
