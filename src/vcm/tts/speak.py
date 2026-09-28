"""Text-to-speech output — the device's sole output modality (no display, by design).

Section 5 targets Piper/espeak-ng on-device for the actual RPi build.
Section 8 calls out macOS's built-in `say` as fine for quick ad hoc
testing on the Mac, so it's the zero-install default here; espeak-ng is
the zero-setup option on the Pi but sounds robotic. Piper is the natural-
sounding neural voice (pip install piper-tts, plus a downloaded voice
model: DEPLOYMENT.md), fast enough on a Pi 5 to reply in well under a second.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from vcm.config import REPO_ROOT, load_settings

DEFAULT_PIPER_VOICE = "models/tts/en_US-lessac-medium.onnx"
_piper_voices: dict = {}


class TTSBackendUnavailable(RuntimeError):
    pass


def _speak_mac_say(text: str) -> None:
    if shutil.which("say") is None:
        raise TTSBackendUnavailable(
            "`say` not found — this backend only works on macOS. "
            "Set [tts].backend to 'espeak_ng' elsewhere."
        )
    voice = load_settings().tts.get("mac_voice", "")  # e.g. "Samantha"; `say -v '?'` lists them
    subprocess.run(["say", *(["-v", voice] if voice else []), text], check=True)


def _speak_espeak_ng(text: str) -> None:
    if shutil.which("espeak-ng") is None:
        raise TTSBackendUnavailable(
            "espeak-ng not found on PATH. Install it (e.g. `apt install "
            "espeak-ng` on the RPi/Linux) or switch [tts].backend to 'mac_say'."
        )
    subprocess.run(["espeak-ng", text], check=True)


_CLOCK_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?:\s*([AaPp])(?:\.[Mm]\.|[Mm]\b))?(?!\w)")


def _spoken_clock(match: re.Match) -> str:
    hour, minute, meridiem = int(match.group(1)), int(match.group(2)), match.group(3)
    suffix = f" {meridiem.upper()}M" if meridiem else ""
    if minute == 0 and hour == 12 and meridiem:
        return "midnight" if meridiem.upper() == "A" else "noon"
    if minute == 0:
        return f"{hour}{suffix}" if meridiem else f"{hour} o'clock"
    return f"{hour} {'oh ' if minute < 10 else ''}{minute}{suffix}"


def speakable(text: str) -> str:
    """Rewrite clock times the way people say them, since TTS engines read
    "5:00 PM" digit by digit: "5:00 PM" -> "5 PM", "10:45 AM" -> "10 45 AM",
    "6:05 PM" -> "6 oh 5 PM"."""
    return _CLOCK_RE.sub(_spoken_clock, text)


def play_wav(path: str) -> None:
    if platform.system() == "Darwin":
        player = ["afplay"]
    else:
        player = next(([p] for p in ("paplay", "pw-play", "aplay") if shutil.which(p)), None)
        if player is None:
            raise TTSBackendUnavailable("no audio player found (paplay, pw-play or aplay)")
    subprocess.run([*player, path], check=True)


def _speak_piper(text: str) -> None:
    try:
        from piper import PiperVoice, SynthesisConfig
    except ImportError as exc:
        raise TTSBackendUnavailable("Piper not installed: pip install piper-tts") from exc
    tts = load_settings().tts
    model = Path(tts.get("piper_voice", DEFAULT_PIPER_VOICE))
    model = model if model.is_absolute() else REPO_ROOT / model
    if not model.exists():
        raise TTSBackendUnavailable(f"Piper voice not found at {model} (DEPLOYMENT.md has the download step)")
    if model not in _piper_voices:  # loading takes ~1 s; the server keeps it for every reply
        _piper_voices[model] = PiperVoice.load(str(model))
    config = SynthesisConfig(length_scale=1.0 / float(tts.get("piper_speed", 1.0)))
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        with wave.open(f.name, "wb") as wav:
            _piper_voices[model].synthesize_wav(text, wav, syn_config=config)
        play_wav(f.name)


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
    _BACKENDS[chosen](speakable(text))
