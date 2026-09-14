"""Backend-selection tests only — these don't touch real hardware/GPIO/keyboard
listeners, since none of that is available (or meaningful to test) in this
Linux dev sandbox. They confirm vcm.config.get_platform() correctly steers
each HAL module to the right backend class.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

from vcm.hal import button, temperature


@pytest.fixture(autouse=True)
def _clear_platform_override(monkeypatch):
    monkeypatch.delenv("VCM_PLATFORM", raising=False)


def test_get_button_picks_mac_backend(monkeypatch):
    monkeypatch.setenv("VCM_PLATFORM", "mac")
    sentinel = MagicMock(name="mac_button_instance")
    monkeypatch.setattr(button, "_MacSpacebarButton", MagicMock(return_value=sentinel))
    assert button.get_button() is sentinel


def test_get_button_picks_rpi_backend(monkeypatch):
    monkeypatch.setenv("VCM_PLATFORM", "rpi")
    sentinel = MagicMock(name="rpi_button_instance")
    monkeypatch.setattr(button, "_RpiGpioButton", MagicMock(return_value=sentinel))
    assert button.get_button() is sentinel


def test_read_temperature_mac_returns_mock_constant(monkeypatch):
    monkeypatch.setenv("VCM_PLATFORM", "mac")
    assert temperature.read_temperature() == temperature._MOCK_CELSIUS


def test_read_temperature_rpi_uses_sense_hat(monkeypatch):
    monkeypatch.setenv("VCM_PLATFORM", "rpi")

    fake_module = types.ModuleType("sense_hat")

    class _FakeSenseHat:
        def get_temperature(self) -> float:
            return 31.5

    fake_module.SenseHat = _FakeSenseHat
    monkeypatch.setitem(sys.modules, "sense_hat", fake_module)

    assert temperature.read_temperature() == 31.5
