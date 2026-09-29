"""Category 7 (adjust thermostat): Xiaomi smart plug driving a fan, plus a temperature read.

Section 4: temperature is a bare-minimum reading, not a sensor-accuracy
project at this tier — see vcm.hal.temperature.
"""

from __future__ import annotations

from vcm.actions._miio import make_device
from vcm.config import load_settings
from vcm.hal.temperature import read_temperature

COMFORT_RANGE_C = (22.0, 26.0)


def _plug():
    xiaomi = load_settings().xiaomi
    return make_device(
        host=xiaomi.get("plug_host", ""),
        token=xiaomi.get("plug_token", ""),
        device_class=xiaomi.get("plug_device_class", "miio.ChuangmiPlug"),
    )


def current_temperature() -> float:
    return read_temperature()


def fan_on() -> None:
    _plug().on()


def fan_off() -> None:
    _plug().off()


def adjust_to_comfort() -> str:
    """Rule-of-thumb thermostat control (Section 4: no sensor-accuracy debates at Tier 1)."""
    temp = current_temperature()
    low, high = COMFORT_RANGE_C
    if temp > high:
        fan_on()
        return f"{temp:.1f}C is above the comfort range, fan turned on"
    if temp < low:
        fan_off()
        return f"{temp:.1f}C is below the comfort range, fan turned off"
    return f"{temp:.1f}C is within the comfort range, no change"
