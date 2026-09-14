"""Shared python-miio device instantiation, used by lights.py and thermostat.py.

Section 4 flags the exact Xiaomi bulb family as unconfirmed against
python-miio's supported-device list ("Xiaomi bulb" could be the
Xiaomi/Philips co-branded family or Yeelight-branded), so the device
class is resolved from a dotted path in settings.toml rather than
hardcoded here — the user picks the right one after checking their
actual hardware.
"""

from __future__ import annotations

import importlib


class MiioNotConfigured(RuntimeError):
    pass


def resolve_device_class(dotted_path: str):
    """Import e.g. "miio.Yeelight" and return the class object."""
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Expected a dotted path like 'miio.Yeelight', got {dotted_path!r}")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def make_device(host: str, token: str, device_class: str):
    if not host or not token:
        raise MiioNotConfigured(
            "Missing host/token — run `miiocli cloud` (Section 4) to obtain "
            "the local token, then fill in configs/settings.toml."
        )
    cls = resolve_device_class(device_class)
    return cls(host, token)
