#!/usr/bin/env python
"""Preflight check for the Pi before a demo: is everything Kiwi needs working?

    .venv/bin/python scripts/kiwi_doctor.py            # check only
    .venv/bin/python scripts/kiwi_doctor.py --beep     # also play a short beep and confirm it reaches the speaker
    .venv/bin/python scripts/kiwi_doctor.py --fix      # also reset streams left turned down to 100%

Checks power and temperature, the microphone (right device, and actually
hearing sound), the speaker (right device, volume, not muted, streams not
stuck turned down), Kiwi's services and home server, the models, the
internet, Spotify and the weather. Each line is OK, WARN or FAIL, and the
last line says READY or what to fix. Everything it found on the Pi so far
is here: a mic that reconnected as the wrong input, music stuck at -36 dB,
nothing started after a reboot.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import socket
import subprocess
import tempfile
import urllib.request
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MIC = "USB_PnP"  # part of the input device name (deploy/wireplumber/51-kiwi-audio.lua)
EXPECTED_SPEAKER = "SoundBar"
SERVER = "http://127.0.0.1:8000"

results: list[tuple[str, str, str]] = []


def report(status: str, what: str, detail: str = "") -> None:
    results.append((status, what, detail))
    print(f"{status:<5} {what}{': ' + detail if detail else ''}", flush=True)


def run(cmd: list[str], timeout: float = 10) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()


def check_power() -> None:
    if not shutil.which("vcgencmd"):
        return report("WARN", "power", "vcgencmd not found (not a Pi?)")
    throttled = run(["vcgencmd", "get_throttled"]).split("=")[-1]
    temp = float(run(["vcgencmd", "measure_temp"]).split("=")[-1].rstrip("'C"))
    status = "OK" if throttled == "0x0" else "FAIL"
    report(status, "power", f"throttled={throttled}" + ("" if status == "OK" else " (undervoltage or overheating: use the official 27 W supply)"))
    report("OK" if temp < 75 else "WARN", "temperature", f"{temp:.0f} C")


def record_level(source: str, seconds: float = 2.0) -> tuple[float, float]:
    with tempfile.NamedTemporaryFile(suffix=".raw") as f:
        subprocess.run(["timeout", str(seconds), "parecord", f"--device={source}", "--raw", "--format=s16le",
                        "--channels=1", "--rate=16000", f.name], capture_output=True)  # fmt: skip
        data = Path(f.name).read_bytes()
    samples = [int.from_bytes(data[i : i + 2], "little", signed=True) for i in range(0, len(data) - 1, 2)]
    if not samples:
        return 0.0, 0.0
    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768
    return rms, max(abs(s) for s in samples) / 32768


def check_audio(beep: bool, fix: bool) -> None:
    if not shutil.which("pactl"):
        return report("FAIL", "audio", "pactl not found (PipeWire/PulseAudio not running?)")
    source, sink = run(["pactl", "get-default-source"]), run(["pactl", "get-default-sink"])
    report("OK" if EXPECTED_MIC in source and not source.endswith(".monitor") else "FAIL", "microphone device", source or "none")
    report("OK" if EXPECTED_SPEAKER in sink else "FAIL", "speaker device", sink or "none")

    volume = run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
    percent = int(volume.split("/")[1].strip().rstrip("%")) if "/" in volume else 0
    muted = run(["pactl", "get-sink-mute", "@DEFAULT_SINK@"]).endswith("yes")
    report("OK" if percent >= 50 and not muted else "WARN", "speaker volume", f"{percent}%{' MUTED' if muted else ''}")

    for stream in json.loads(run(["pactl", "-f", "json", "list", "sink-inputs"]) or "[]"):
        name = stream["properties"].get("application.name", "?")
        levels = [int(v["value_percent"].rstrip("%")) for v in stream["volume"].values()]
        if levels and min(levels) < 60:
            if fix:
                run(["pactl", "set-sink-input-volume", str(stream["index"]), "100%"])
                report("OK", f"stream '{name}'", f"was at {min(levels)}%, reset to 100%")
            else:
                report("FAIL", f"stream '{name}'", f"at {min(levels)}% (a stuck duck?): run with --fix")

    rms, peak = record_level(source)
    if rms < 0.0005:
        report("FAIL", "microphone signal", f"silent (RMS {rms:.4f}): check the cable and `alsamixer`")
    else:
        report("OK" if peak < 0.99 else "WARN", "microphone signal",
               f"RMS {rms:.4f}, peak {peak:.2f}" + (" (clipping: lower the mic level)" if peak >= 0.99 else ""))  # fmt: skip

    if beep:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            with wave.open(f.name, "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
                w.writeframes(b"".join(int(3000 * math.sin(2 * math.pi * 660 * i / 16000)).to_bytes(2, "little", signed=True) for i in range(12000)))
            player = subprocess.Popen(["paplay", f.name])
            _, played = record_level(f"{sink}.monitor", 1.0)
            player.wait()
        report("OK" if played > 0.02 else "FAIL", "speaker output", f"beep reached the speaker at peak {played:.3f}")


def check_services() -> None:
    for unit in ("vcm-home", "vcm"):
        state = run(["systemctl", "--user", "is-active", unit]) if shutil.which("systemctl") else "?"
        enabled = run(["systemctl", "--user", "is-enabled", unit]) if shutil.which("systemctl") else "?"
        if state == "active":
            report("OK" if enabled == "enabled" else "WARN", f"service {unit}", f"running, {enabled} at boot")
        else:
            running = run(["pgrep", "-f", "vcm.home.server" if unit == "vcm-home" else "vcm_listen.py"])
            report("WARN" if running else "FAIL", f"service {unit}",
                   "running by hand, won't restart after a reboot (deploy_pi.sh --services)" if running else "not running")  # fmt: skip
    try:
        with urllib.request.urlopen(f"{SERVER}/api/state", timeout=3) as r:
            report("OK", "home server", f"{SERVER} answers ({r.status})")
    except OSError as exc:
        report("FAIL", "home server", f"{SERVER}: {exc}")


def check_models() -> None:
    try:
        import onnxruntime

        for name in ("vcm_intent.onnx", "kiwi_wakeword.onnx"):
            meta = onnxruntime.InferenceSession(str(REPO_ROOT / "models" / name), providers=["CPUExecutionProvider"]).get_modelmeta()
            report("OK", f"model {name}", meta.custom_metadata_map.get("source_checkpoint", "?"))
    except Exception as exc:  # noqa: BLE001 - report anything that stops a model loading
        report("FAIL", "models", str(exc))


def check_online() -> None:
    try:
        socket.create_connection(("api.spotify.com", 443), timeout=4).close()
        report("OK", "internet", "api.spotify.com reachable")
    except OSError as exc:
        return report("FAIL", "internet", f"{exc} (Spotify and weather need it; everything else works offline)")
    try:
        from vcm.config import load_settings
        from vcm.home.dispatcher import integrations_from_settings

        x = integrations_from_settings(load_settings())
        if x.spotify:
            playback = x.spotify.playback()
            report("OK", "Spotify", f"reachable, playing={playback['playing']}, volume={playback['volume']}")
        else:
            report("WARN", "Spotify", "not configured")
        if x.weather:
            report("OK", "weather", x.weather())
        else:
            report("WARN", "weather", "not configured")
    except Exception as exc:  # noqa: BLE001 - a failing integration is a finding, not a crash
        report("FAIL", "integrations", str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--beep", action="store_true", help="play a short beep and check it reaches the speaker")
    parser.add_argument("--fix", action="store_true", help="reset audio streams left turned down to 100%%")
    args = parser.parse_args()

    check_power()
    check_audio(args.beep, args.fix)
    check_services()
    check_models()
    check_online()

    failed = [what for status, what, _ in results if status == "FAIL"]
    warned = [what for status, what, _ in results if status == "WARN"]
    print()
    if failed:
        print(f"NOT READY: fix {', '.join(failed)}" + (f" (and check {', '.join(warned)})" if warned else ""))
        raise SystemExit(1)
    print("READY" + (f" (warnings: {', '.join(warned)})" if warned else ""))


if __name__ == "__main__":
    main()
