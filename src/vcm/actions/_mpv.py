"""Shared mpv JSON-IPC helper, used by both music.py (start/stop playback)
and media_control.py (pause/next/volume against the same running instance).

Kept private (leading underscore) — not part of the dispatch table itself.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

IPC_SOCKET_PATH = Path("/tmp/vcm-mpv.sock")


class MpvUnavailable(RuntimeError):
    pass


def _require_mpv() -> None:
    if shutil.which("mpv") is None:
        raise MpvUnavailable(
            "mpv not found on PATH. Install it (e.g. `brew install mpv` on "
            "the Mac, `apt install mpv` on the RPi/Linux)."
        )


def start(track_path: Path) -> None:
    """Launch mpv on `track_path`, exposing a JSON-IPC socket for control."""
    _require_mpv()
    if IPC_SOCKET_PATH.exists():
        IPC_SOCKET_PATH.unlink()
    subprocess.Popen(
        [
            "mpv",
            f"--input-ipc-server={IPC_SOCKET_PATH}",
            "--no-video",
            "--idle=no",
            str(track_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(50):  # wait up to ~5s for the socket to appear
        if IPC_SOCKET_PATH.exists():
            return
        time.sleep(0.1)
    raise MpvUnavailable("mpv did not create its IPC socket in time")


def send_command(*args: object) -> object:
    """Send a JSON-IPC command to the running mpv instance and return its response."""
    if not IPC_SOCKET_PATH.exists():
        raise MpvUnavailable("no mpv instance is running (call music.play() first)")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(str(IPC_SOCKET_PATH))
        payload = json.dumps({"command": list(args)}) + "\n"
        sock.sendall(payload.encode())
        response = sock.recv(4096)
    return json.loads(response.decode()) if response else None


def is_running() -> bool:
    return IPC_SOCKET_PATH.exists()
