"""Spotify playback control via the Spotify Web API (needs Spotify Premium).

The speaker is a Spotify Connect device: on the Pi, raspotify (librespot)
makes it appear in Spotify as e.g. "kiwi"; when testing on the Mac, the
Spotify desktop app is the device. This module only sends play / pause /
next / volume to that device, and never streams audio itself.

Credentials live in configs/settings.toml [spotify]: client_id and
client_secret from the Spotify developer dashboard, and a refresh_token
from scripts/spotify_auth.py (one-time browser login). DEPLOYMENT.md has
the steps.
"""

from __future__ import annotations

import time

import requests

API = "https://api.spotify.com/v1"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SCOPES = "user-modify-playback-state user-read-playback-state user-read-currently-playing"


class SpotifyError(RuntimeError):
    pass


class SpotifyClient:
    def __init__(self, client_id: str, client_secret: str, refresh_token: str, device_name: str = "", default_uri: str = ""):
        if not (client_id and client_secret and refresh_token):
            raise SpotifyError("Spotify is not configured: set [spotify] client_id, client_secret, refresh_token")
        self.client_id, self.client_secret, self.refresh_token = client_id, client_secret, refresh_token
        self.device_name, self.default_uri = device_name, default_uri
        self._token, self._expires = "", 0.0

    @classmethod
    def from_settings(cls, section: dict) -> "SpotifyClient":
        return cls(
            section.get("client_id", ""),
            section.get("client_secret", ""),
            section.get("refresh_token", ""),
            section.get("device_name", ""),
            section.get("default_uri", ""),
        )

    def _access_token(self) -> str:
        if time.time() < self._expires - 60:
            return self._token
        r = requests.post(
            TOKEN_URL,
            data={"grant_type": "refresh_token", "refresh_token": self.refresh_token},
            auth=(self.client_id, self.client_secret),
            timeout=10,
        )
        if r.status_code != 200:
            raise SpotifyError(f"token refresh failed ({r.status_code}): {r.text[:200]}")
        body = r.json()
        self._token, self._expires = body["access_token"], time.time() + body.get("expires_in", 3600)
        return self._token

    def _call(self, method: str, path: str, **kwargs) -> requests.Response:
        headers = {"Authorization": f"Bearer {self._access_token()}"}
        return requests.request(method, API + path, headers=headers, timeout=10, **kwargs)

    def device_id(self) -> str | None:
        """The configured device (by name), else the currently active one."""
        r = self._call("GET", "/me/player/devices")
        if r.status_code != 200:
            raise SpotifyError(f"listing devices failed ({r.status_code})")
        devices = r.json().get("devices", [])
        for d in devices:
            if self.device_name and d["name"].lower() == self.device_name.lower():
                return d["id"]
        active = [d for d in devices if d.get("is_active")]
        return active[0]["id"] if active else (devices[0]["id"] if devices else None)

    def _player(self, method: str, path: str, **kwargs) -> None:
        device = self.device_id()
        if device is None:
            raise SpotifyError(f"no Spotify device found (is '{self.device_name or 'a Spotify app'}' running?)")
        params = {**kwargs.pop("params", {}), "device_id": device}
        r = self._call(method, path, params=params, **kwargs)
        if r.status_code not in (200, 202, 204):
            reason = "Spotify Premium required" if r.status_code == 403 else r.text[:200]
            raise SpotifyError(f"{method} {path} failed ({r.status_code}): {reason}")

    def play(self, uri: str | None = None) -> None:
        """Resume, or start `uri` (or the configured default playlist) if given."""
        uri = uri or self.default_uri
        body = ({"uris": [uri]} if uri.startswith("spotify:track:") else {"context_uri": uri}) if uri else None
        self._player("PUT", "/me/player/play", json=body)

    def pause(self) -> None:
        self._player("PUT", "/me/player/pause")

    def next(self) -> None:
        self._player("POST", "/me/player/next")

    def set_volume(self, percent: int) -> None:
        self._player("PUT", "/me/player/volume", params={"volume_percent": max(0, min(100, int(percent)))})

    def now_playing(self) -> str | None:
        r = self._call("GET", "/me/player/currently-playing")
        if r.status_code != 200 or not r.content:
            return None
        item = r.json().get("item") or {}
        artists = ", ".join(a["name"] for a in item.get("artists", []))
        return f"{item.get('name')} by {artists}" if item.get("name") else None
