"""Categories 3-4 (lights on/off, dim/color): Xiaomi bulb via python-miio, local LAN."""

from __future__ import annotations

from vcm.actions._miio import make_device
from vcm.config import load_settings


def _bulb():
    xiaomi = load_settings().xiaomi
    return make_device(
        host=xiaomi.get("bulb_host", ""),
        token=xiaomi.get("bulb_token", ""),
        device_class=xiaomi.get("bulb_device_class", "miio.Yeelight"),
    )


def turn_on() -> None:
    _bulb().on()


def turn_off() -> None:
    _bulb().off()


def set_brightness(percent: int) -> None:
    if not 0 <= percent <= 100:
        raise ValueError(f"brightness percent must be 0-100, got {percent}")
    bulb = _bulb()
    if not hasattr(bulb, "set_brightness"):
        raise NotImplementedError(
            f"{type(bulb).__name__} has no set_brightness(); check the "
            "configured [xiaomi].bulb_device_class matches the real hardware."
        )
    bulb.set_brightness(percent)


def set_color(name_or_hex: str) -> None:
    bulb = _bulb()
    if not hasattr(bulb, "set_rgb") and not hasattr(bulb, "set_hsv"):
        raise NotImplementedError(
            f"{type(bulb).__name__} has no color control; check the "
            "configured [xiaomi].bulb_device_class matches the real hardware."
        )
    if hasattr(bulb, "set_rgb"):
        bulb.set_rgb(_named_or_hex_to_rgb(name_or_hex))


_NAMED_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "warm": (255, 214, 170),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
}


def _named_or_hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lower()
    if value in _NAMED_COLORS:
        return _NAMED_COLORS[value]
    hex_value = value.lstrip("#")
    if len(hex_value) == 6:
        return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    raise ValueError(f"Unrecognized color {value!r}; use a name or a #rrggbb hex value")
