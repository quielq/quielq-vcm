"""Text-to-speech output — the device's sole output modality (no display, by design).

Section 5 targets Piper/espeak-ng on-device for the actual RPi build.
Section 8 calls out macOS's built-in `say` as fine for quick ad hoc
testing on the Mac, so it's the zero-install default here; espeak-ng is
wired up as the cross-platform option (works in this Linux sandbox too,
if installed) and Piper is left as a documented follow-up once a voice
model is chosen.
"""

from __future__ import annotations

import shutil
import subprocess

from vcm.config import load_settings


class TTSBackendUnavailable(RuntimeError):
    pass


def _speak_mac_say(text: str) -> None:
    if shutil.which("say") is None:
        raise TTSBackendUnavailable(
            "`say` not found — this backend only works on macOS. "
            "Set [tts].backend to 'espeak_ng' elsewhere."
        )
    subprocess.run(["say", text], check=True)


def _speak_espeak_ng(text: str) -> None:
    if shutil.which("espeak-ng") is None:
        raise TTSBackendUnavailable(
            "espeak-ng not found on PATH. Install it (e.g. `apt install "
            "espeak-ng` on the RPi/Linux) or switch [tts].backend to 'mac_say'."
        )
    subprocess.run(["espeak-ng", text], check=True)


def _speak_piper(text: str) -> None:
    raise TTSBackendUnavailable(
        "Piper backend not wired up yet — needs a downloaded voice model "
        "(Section 5). Use 'mac_say' or 'espeak_ng' for now."
    )


_BACKENDS = {
    "mac_say": _speak_mac_say,
    "espeak_ng": _speak_espeak_ng,
    "piper": _speak_piper,
}


def speak(text: str, backend: str | None = None) -> None:
    """Speak `text` aloud using the configured (or explicitly given) backend."""
    chosen = backend or load_settings().tts.get("backend", "mac_say")
    if chosen not in _BACKENDS:
        raise ValueError(f"Unknown TTS backend {chosen!r}; choose from {list(_BACKENDS)}")
    _BACKENDS[chosen](text)
