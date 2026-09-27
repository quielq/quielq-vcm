"""Temperature reading for the thermostat command (category 7).

Section 8 explicitly names this as mock-until-hardware-arrives: the mac
backend returns a fixed value from the same function signature the real
Sense HAT read will use. Per Section 4's notes, no sensor-accuracy work
is warranted at Tier 1.
"""

from __future__ import annotations

from vcm.config import get_platform

_MOCK_CELSIUS = 27.0


class TemperatureUnavailable(RuntimeError):
    pass


def read_temperature() -> float:
    """Return the current temperature in Celsius."""
    platform = get_platform()
    if platform == "rpi":
        try:
            from sense_hat import SenseHat
        except ImportError as exc:
            raise TemperatureUnavailable("no Sense HAT library (sudo apt install sense-hat)") from exc
        try:
            return SenseHat().get_temperature()
        except OSError as exc:  # library installed but no HAT on the pins / I2C off
            raise TemperatureUnavailable(f"no Sense HAT found: {exc}") from exc
    if platform == "mac":
        return _MOCK_CELSIUS
    raise ValueError(f"Unknown platform: {platform!r}")
