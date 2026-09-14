"""Category 8 (media control): pause/stop/next/volume against the running mpv instance."""

from __future__ import annotations

from vcm.actions import _mpv


def pause() -> None:
    _mpv.send_command("set_property", "pause", True)


def resume() -> None:
    _mpv.send_command("set_property", "pause", False)


def stop() -> None:
    _mpv.send_command("stop")


def next_track() -> None:
    _mpv.send_command("playlist-next")


def previous_track() -> None:
    _mpv.send_command("playlist-prev")


def volume_up(step: int = 10) -> None:
    _mpv.send_command("add", "volume", step)


def volume_down(step: int = 10) -> None:
    _mpv.send_command("add", "volume", -step)
