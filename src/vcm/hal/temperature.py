"""Temperature reading for the thermostat command (category 7).

Section 8 explicitly names this as mock-until-hardware-arrives: the mac
backend returns a fixed value from the same function signature the real
Sense HAT read will use. Per Section 4's notes, no sensor-accuracy work
is warranted at Tier 1.
"""

from __future__ import annotations

from vcm.config import get_platform

_MOCK_CELSIUS = 27.0


def read_temperature() -> float:
    """Return the current temperature in Celsius."""
    platform = get_platform()
    if platform == "rpi":
        from sense_hat import SenseHat

        return SenseHat().get_temperature()
    if platform == "mac":
        return _MOCK_CELSIUS
    raise ValueError(f"Unknown platform: {platform!r}")
