"""Category 10 (calls and messaging): mocked via TTS (Section 4), no real call.

Real calling is a Tier 2 concern (BlueZ HFP to the iPhone 15, Section 5)
and out of scope here.
"""

from __future__ import annotations

from vcm.tts.speak import speak


def call(contact: str = "Mom") -> None:
    speak(f"Calling {contact}")
