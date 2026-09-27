"""Temperature reading for the thermostat command (category 7).

Section 8 explicitly names this as mock-until-hardware-arrives: the mac
backend returns a fixed value from the same function signature the real
Sense HAT read will use. Per Section 4's notes, no sensor-accuracy work
is warranted at Tier 1.
"""

from __future__ import annotations

from pathlib import Path

from vcm.config import get_platform, load_settings

_MOCK_CELSIUS = 27.0
DEFAULT_SENSOR_FACTOR = 0.85  # from readings on the HAT-on-header Pi 5 (see compensate)


class TemperatureUnavailable(RuntimeError):
    pass


def cpu_temperature() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


def compensate(raw_c: float, cpu_c: float | None, factor: float | None = None) -> float:
    """Remove the CPU's heat from a Sense HAT reading.

    Mounted on the Pi's header, the HAT sits right over the CPU and reads
    well above the room (on a Pi 5: ~50 C with the CPU at ~69 C). The usual
    correction subtracts a share of the CPU-to-HAT difference:
    room = raw - (cpu - raw) / factor. factor comes from [thermostat]
    sensor_factor: calibrate it once against a real thermometer (factor =
    (cpu - raw) / (raw - thermometer)). Expect +-2 C; a 40-pin extension
    cable that moves the HAT away from the CPU is the real fix (then set
    sensor_factor = 0 to turn the correction off).
    """
    factor = load_settings().section("thermostat").get("sensor_factor", DEFAULT_SENSOR_FACTOR) if factor is None else factor
    if not factor or cpu_c is None or cpu_c <= raw_c:
        return raw_c
    return raw_c - (cpu_c - raw_c) / factor


def read_temperature() -> float:
    """Return the current temperature in Celsius."""
    platform = get_platform()
    if platform == "rpi":
        try:
            from sense_hat import SenseHat
        except ImportError as exc:
            raise TemperatureUnavailable("no Sense HAT library (sudo apt install sense-hat)") from exc
        try:
            raw = SenseHat().get_temperature()
        except OSError as exc:  # library installed but no HAT on the pins / I2C off
            raise TemperatureUnavailable(f"no Sense HAT found: {exc}") from exc
        return compensate(raw, cpu_temperature())
    if platform == "mac":
        return _MOCK_CELSIUS
    raise ValueError(f"Unknown platform: {platform!r}")
