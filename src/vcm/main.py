"""Orchestration loop: button -> capture -> features -> inference -> dispatch -> TTS.

Run with `python -m vcm.main`. On the Mac, hold the spacebar to talk
(Section 8's push-to-talk mock); on the RPi, hold the physical pushbutton.
"""

from __future__ import annotations

from vcm.audio.capture import record_while_held
from vcm.audio.features import extract_log_mel
from vcm.dispatch import dispatch
from vcm.hal.button import get_button
from vcm.inference.model import load_default_model
from vcm.taxonomy import UNKNOWN_BACKGROUND
from vcm.tts.speak import speak


def run_once(button, model) -> str:
    """Capture one command, classify it, and act on it. Returns the intent label."""
    audio = record_while_held(button)
    features = extract_log_mel(audio)
    label = model.predict(features)
    if label != UNKNOWN_BACKGROUND:
        message = dispatch(label)
        if message:
            speak(message)
    return label


def main() -> None:
    button = get_button()
    model = load_default_model()
    print("VCM ready. Hold the push-to-talk trigger and speak a command (Ctrl+C to quit).")
    while True:
        try:
            label = run_once(button, model)
            print(f"Heard intent: {label}")
        except KeyboardInterrupt:
            print("Exiting.")
            break


if __name__ == "__main__":
    main()
