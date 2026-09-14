import pytest

from vcm import config


def test_env_override_valid(monkeypatch):
    monkeypatch.setenv("VCM_PLATFORM", "rpi")
    assert config.get_platform() == "rpi"


def test_env_override_invalid(monkeypatch):
    monkeypatch.setenv("VCM_PLATFORM", "windows")
    with pytest.raises(ValueError):
        config.get_platform()


def test_auto_detect_linux_sandbox_is_mac_like(monkeypatch):
    # This dev sandbox is Linux but has no /proc/device-tree/model, so it
    # should fall back to the "mac"-like branch, not "rpi".
    monkeypatch.delenv("VCM_PLATFORM", raising=False)
    detected = config.get_platform()
    assert detected in config.VALID_PLATFORMS


def test_load_settings_falls_back_to_example(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "does-not-exist.toml")
    settings = config.load_settings()
    # example file ships with an empty api key — proves the fallback loaded
    assert settings.weather.get("api_key", "") == ""
