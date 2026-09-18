from vcm.dataset.sources.snips_lights import classify

# Real transcripts from the live dataset, spot-checked by hand before
# writing this test (see the module docstring for how classify() was
# validated against all 5,886 real transcripts).
LIGHT_ON_EXAMPLES = [
    "Turn on the changing room lights",
    "Can you turn on the light",
    "Activate all the lights in the entire house.",
    "Switch on my cubicle light please",
]
LIGHT_OFF_EXAMPLES = [
    "Can you turn off the lights please",
    "Switch off the lights",
    "In the kids bedroom turn the lights off.",
    "Be sure the lights are off by the toilet",
]
BRIGHTNESS_EXAMPLES = [
    "Could you increase the brightness of the lights?",
    "Please lower the lights in the room.",
    "Can you make the lights dimmer in the living room?",
    "I want the lights to be dimmed.",
]
COLOR_EXAMPLES = [
    "Make the boardroom lights red",
    "Change the lights to red",
    "set the color of meeting room lights to blue",
    "Turn the backyard lights pink",
]

# The real false positive this classifier was specifically built to avoid
# (an artist name containing a color word, in a music request).
MUSIC_REQUEST_EXAMPLES = [
    "I'd like to listen to White Sea",
    "I'd like to listen to Kodak Black",
    "I'd like to listen to Mon Laferte",
]


def test_light_on_examples():
    for text in LIGHT_ON_EXAMPLES:
        assert classify(text) == "LIGHT_ON", text


def test_light_off_examples():
    for text in LIGHT_OFF_EXAMPLES:
        assert classify(text) == "LIGHT_OFF", text


def test_brightness_examples():
    for text in BRIGHTNESS_EXAMPLES:
        assert classify(text) == "BRIGHTNESS", text


def test_color_examples():
    for text in COLOR_EXAMPLES:
        assert classify(text) == "COLOR", text


def test_music_requests_are_not_misclassified_as_color():
    # Regression test for the exact bug caught during development: "White
    # Sea" (an artist) matching the COLOR word "white".
    for text in MUSIC_REQUEST_EXAMPLES:
        assert classify(text) is None, text


def test_unrelated_text_is_unmatched():
    assert classify("What's the weather like today?") is None
    assert classify("Set an alarm for six AM") is None
    assert classify("") is None
