"""Settings loading and platform detection.

The hardware abstraction layer (see vcm.hal) picks its backend based on
`get_platform()`. Everything else (API keys, device tokens, paths) comes
from configs/settings.toml, falling back to configs/settings.example.toml
so the package still imports and tests still run before a user has copied
their own settings file into place.
"""

from __future__ import annotations

import os
import platform as _platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
SETTINGS_PATH = CONFIG_DIR / "settings.toml"
SETTINGS_EXAMPLE_PATH = CONFIG_DIR / "settings.example.toml"

VALID_PLATFORMS = ("mac", "rpi")


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def get_platform() -> str:
    """Return "mac" or "rpi".

    Resolution order: VCM_PLATFORM env var > [platform].override in
    settings.toml > auto-detection from platform.system()/machine().
    """
    env_override = os.environ.get("VCM_PLATFORM")
    if env_override:
        if env_override not in VALID_PLATFORMS:
            raise ValueError(
                f"VCM_PLATFORM={env_override!r} is not one of {VALID_PLATFORMS}"
            )
        return env_override

    settings = _load_toml(SETTINGS_PATH)
    configured = settings.get("platform", {}).get("override")
    if configured:
        if configured not in VALID_PLATFORMS:
            raise ValueError(
                f"[platform].override={configured!r} is not one of {VALID_PLATFORMS}"
            )
        return configured

    system = _platform.system()
    if system == "Darwin":
        return "mac"
    if system == "Linux":
        # Raspberry Pi OS reports its model here; anything else Linux
        # (e.g. this dev sandbox) is treated as "mac"-like for HAL
        # purposes, since it also has no GPIO/Sense HAT hardware.
        model_path = Path("/proc/device-tree/model")
        if model_path.exists():
            try:
                model = model_path.read_text(errors="ignore")
            except OSError:
                model = ""
            if "raspberry pi" in model.lower():
                return "rpi"
        return "mac"
    return "mac"


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any] = field(default_factory=dict)

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})

    @property
    def xiaomi(self) -> dict[str, Any]:
        return self.section("xiaomi")

    @property
    def weather(self) -> dict[str, Any]:
        return self.section("weather")

    @property
    def music(self) -> dict[str, Any]:
        return self.section("music")

    @property
    def reminders(self) -> dict[str, Any]:
        return self.section("reminders")

    @property
    def tts(self) -> dict[str, Any]:
        return self.section("tts")


def load_settings() -> Settings:
    """Load configs/settings.toml, falling back to the example file.

    Falling back keeps imports/tests working with no local setup; any
    action module that needs a real credential (Xiaomi token, weather API
    key) will get an empty string from the example file and should raise
    a clear error at call time rather than silently no-op.
    """
    if SETTINGS_PATH.exists():
        return Settings(raw=_load_toml(SETTINGS_PATH))
    return Settings(raw=_load_toml(SETTINGS_EXAMPLE_PATH))
