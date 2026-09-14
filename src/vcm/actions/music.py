"""Category 1 (play music): local media library via mpv (Section 5)."""

from __future__ import annotations

from pathlib import Path

from vcm.actions import _mpv
from vcm.config import load_settings

AUDIO_EXTENSIONS = (".mp3", ".m4a", ".flac", ".wav", ".ogg")


def _media_dir() -> Path:
    settings = load_settings()
    return Path(settings.music.get("media_dir", "media"))


def list_tracks() -> list[Path]:
    media_dir = _media_dir()
    if not media_dir.exists():
        return []
    return sorted(
        p for p in media_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )


def play(query: str | None = None) -> Path:
    """Play a track whose filename contains `query` (case-insensitive), or the first track."""
    tracks = list_tracks()
    if not tracks:
        raise FileNotFoundError(
            f"No audio files found under {_media_dir()} — populate the media "
            "directory (see configs/settings.example.toml [music].media_dir)."
        )
    if query:
        matches = [t for t in tracks if query.lower() in t.name.lower()]
        track = matches[0] if matches else tracks[0]
    else:
        track = tracks[0]
    _mpv.start(track)
    return track
