#!/usr/bin/env python
"""Hands-free VCM loop: "Hey Kiwi" -> command -> intent + slot value.

Runs the same way on the Raspberry Pi (USB mic, over SSH) and on a laptop
(built-in mic), using only the exported ONNX models (no torch):

    python scripts/vcm_listen.py
    python scripts/vcm_listen.py --trigger button     # push-to-talk instead (spacebar / GPIO 17)
    python scripts/vcm_listen.py --server http://127.0.0.1:8000   # act on commands (vcm.home.server)

After the wake word fires, it plays a short chime (--no-chime to turn off),
asks the home server to turn Spotify down while you speak, and records
until you stop talking (0.7 s below the speech level, or 5 s max), then
classifies. Every result prints with
its timing, so latency can be read straight off an SSH session.
"""

from __future__ import annotations

import argparse
import os
import queue
import time
from pathlib import Path

# One math thread, set before numpy loads: OpenBLAS otherwise starts a
# busy-waiting worker per core, which on the Pi took all four cores (~190%
# CPU), made Kiwi lag and starved raspotify. The models are tiny; one is enough.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

from vcm.audio.capture import SAMPLE_RATE
from vcm.audio.features import speech_level
from vcm.audio.resample import StreamResampler, input_rate
from vcm.deploy.runtime import OnnxIntentModel
from vcm.wakeword.detector import WakeWordDetector, onnx_scorer

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK_S = 0.1
END_SILENCE_S = 0.7
MAX_COMMAND_S = 5.0
SILENCE_THRESHOLD = 0.005  # "was anything said?" gate (loudest 300 ms). 0.008 (scripts/demo_infer.py,
# Mac mic) dropped every command from a quieter mic; silent clips peak at 0.0023-0.0048.
SPEECH_OVER_FLOOR = 3.0  # a chunk is speech if 3x louder than the background (room, or music)
REJECT_THRESHOLD = 0.6  # same as scripts/demo_infer.py


def background_floor(levels: "list[float]") -> float:
    """Background level before the wake word: the 20th percentile of the
    chunks 3.0-1.2 s before it fired (so "Hey Kiwi" itself isn't counted)."""
    earlier = levels[-30:-12]
    return float(np.percentile(earlier, 20)) if earlier else 0.0


def record_command(chunks: "queue.Queue[np.ndarray]", lead_in: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Collect mic chunks until END_SILENCE_S of quiet after some speech, or MAX_COMMAND_S.

    Speech means SPEECH_OVER_FLOOR x the background, not a fixed level: with
    a fixed level, music counted as speech and the ducking that follows the
    wake word counted as silence, so recording ended ~1.3 s after "Hey Kiwi"
    unless the command came at once. The floor follows the background down
    as the music ducks (20th percentile of the last 1.5 s). Offline test
    (real "Hey Kiwi" + real commands, music 10 dB under the voice, 0.5-2 s
    pause before the command): 0-24% -> 62-84% of commands heard.
    """
    audio = [lead_in]
    heard_speech, quiet_s, total_s, levels = False, 0.0, 0.0, []
    while total_s < MAX_COMMAND_S:
        chunk = chunks.get()
        audio.append(chunk)
        total_s += len(chunk) / SAMPLE_RATE
        level = speech_level(chunk)
        levels.append(level)
        if len(levels) >= 5:
            floor = min(floor, float(np.percentile(levels[-15:], 20))) if floor else float(np.percentile(levels[-15:], 20))
        loud = level >= max(SILENCE_THRESHOLD, SPEECH_OVER_FLOOR * floor)
        heard_speech |= loud
        quiet_s = 0.0 if loud else quiet_s + len(chunk) / SAMPLE_RATE
        if heard_speech and quiet_s >= END_SILENCE_S:
            break
    return np.concatenate(audio)


def make_chime(rate: int) -> np.ndarray:
    """Two rising soft tones (~0.25 s): the "I'm listening" cue after the wake word."""
    tones = []
    for freq, dur in ((880.0, 0.09), (1320.0, 0.14)):
        t = np.arange(int(dur * rate)) / rate
        envelope = np.minimum(1.0, t / 0.01) * np.exp(-t / (dur / 2.5))  # 10 ms attack, soft decay
        tones.append(0.3 * envelope * np.sin(2 * np.pi * freq * t))
    return np.concatenate(tones).astype("float32")


def post_json(server: str, path: str, payload: dict, timeout: float = 10) -> dict:
    """POST to vcm.home.server. Standard library only (urllib)."""
    import json
    import urllib.request

    request = urllib.request.Request(
        f"{server.rstrip('/')}{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def duck(server: str | None, on: bool) -> None:
    """Turn Spotify down (on) / back up (off) in the background, never blocking listening."""
    import threading

    def run():
        try:
            post_json(server, "/api/duck", {"on": on}, timeout=5)
        except OSError:
            pass

    if server:
        threading.Thread(target=run, daemon=True).start()


def follow_noise(server: str, detector: WakeWordDetector, quiet: float, noisy: float) -> None:
    """Every 2 s, ask the home server whether music is playing (or an alarm
    ringing) and switch the wake threshold: over music, "Hey Kiwi" scores
    lower, so a stricter threshold would miss it."""
    import json
    import threading
    import urllib.request

    def run():
        while True:
            try:
                with urllib.request.urlopen(f"{server.rstrip('/')}/api/noisy", timeout=2) as response:
                    detector.threshold = noisy if json.loads(response.read()).get("noisy") else quiet
            except (OSError, ValueError):
                detector.threshold = quiet
            time.sleep(2)

    threading.Thread(target=run, daemon=True, name="noise-follower").start()


def send_to_server(server: str, intent: str, slot: str | None, confidence: float) -> None:
    """POST the command to vcm.home.server, which acts on it, speaks the
    reply and updates the dashboard."""
    try:
        reply = post_json(server, "/api/command", {"intent": intent, "slot": slot, "confidence": round(confidence, 3), "source": "voice"})
        print(f"  home: {reply.get('reply', '')}")
    except OSError as exc:
        print(f"  (home server unreachable at {server}: {exc})")


def report(model: OnnxIntentModel, audio: np.ndarray, t_end: float, server: str | None = None) -> None:
    level = speech_level(audio)
    if level < SILENCE_THRESHOLD:
        print(f"  (silence, speech level={level:.4f} — skipped)")
        return
    t0 = time.perf_counter()
    pred = model.predict_audio(audio)
    ms = (time.perf_counter() - t0) * 1000
    slot = f"  {pred.slot_value} ({pred.slot_confidence:.2f})" if pred.slot_value else ""
    verdict = "didn't catch that, please repeat" if pred.confidence < REJECT_THRESHOLD else "->"
    print(f"  {verdict} {pred.intent} ({pred.confidence:.2f}){slot}   [model {ms:.0f} ms, {time.perf_counter() - t_end:.2f} s after end of command]")
    if server and pred.confidence >= REJECT_THRESHOLD:
        send_to_server(server, pred.intent, pred.slot_value, pred.confidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--intent-model", type=Path, default=REPO_ROOT / "models/vcm_intent.onnx")
    parser.add_argument("--wake-model", type=Path, default=REPO_ROOT / "models/kiwi_wakeword.onnx")
    parser.add_argument("--wake-threshold", type=float, default=0.95, help="Exp 34: 14%% clean / 22%% noisy / 0 of 10 real-voice false rejects, 1.34 false wake-ups per hour of test speech")
    parser.add_argument(
        "--noisy-wake-threshold",
        type=float,
        default=0.7,
        help="wake threshold while music plays or an alarm rings (needs --server); lower = hears you better over music, more false wake-ups",
    )
    parser.add_argument("--trigger", choices=["wakeword", "button"], default="wakeword")
    parser.add_argument("--device", default=None, help="sounddevice input device (name or index); default: system default")
    parser.add_argument("--show-scores", action="store_true", help="print the wake-word score continuously (tuning)")
    parser.add_argument(
        "--server",
        default=None,
        help="home server URL (python -m vcm.home.server) to act on commands, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument("--no-chime", action="store_true", help="don't play the chime after the wake word")
    args = parser.parse_args()

    import sounddevice as sd

    model = OnnxIntentModel(args.intent_model)
    print(f"intent model: {args.intent_model.name} ({len(model.labels)} intents, slots: {', '.join(model.slot_vocab) or 'none'})")

    if args.trigger == "button":
        from vcm.audio.capture import record_while_held
        from vcm.hal.button import get_button

        button = get_button()
        print("Hold the button (spacebar on a Mac) and speak. Ctrl+C to quit.")
        while True:
            audio = record_while_held(button)
            report(model, audio, time.perf_counter(), args.server)

    detector = WakeWordDetector(onnx_scorer(args.wake_model), threshold=args.wake_threshold)
    if args.server:
        follow_noise(args.server, detector, args.wake_threshold, args.noisy_wake_threshold)
    chunks: queue.Queue[np.ndarray] = queue.Queue()
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    rate = input_rate(sd, device, SAMPLE_RATE)
    convert = StreamResampler(rate, SAMPLE_RATE) if rate != SAMPLE_RATE else (lambda x: x)
    if rate != SAMPLE_RATE:
        print(f"mic records at {rate} Hz (no 16 kHz mode): resampling to {SAMPLE_RATE} Hz")
    stream = sd.InputStream(
        samplerate=rate,
        channels=1,
        dtype="float32",
        blocksize=int(CHUNK_S * rate),
        device=device,
        callback=lambda indata, frames, t, status: chunks.put(convert(indata[:, 0].copy())),
    )
    out_rate = int(sd.query_devices(kind="output")["default_samplerate"])
    chime = None if args.no_chime else make_chime(out_rate)
    print(f'Say "Hey Kiwi", then your command. (wake threshold {args.wake_threshold}; Ctrl+C to quit)')
    levels: list[float] = []  # recent chunk levels, for the background floor
    with stream:
        try:
            while True:
                chunk = chunks.get()
                levels = [*levels[-29:], speech_level(chunk)]
                if detector.feed(chunk):
                    print(f"\n[wake word, score {detector.last_score:.2f}] listening...")
                    t_wake = time.perf_counter()
                    duck(args.server, on=True)
                    if chime is not None:
                        sd.play(chime, out_rate)
                        sd.wait()
                        while not chunks.empty():  # the mic heard the chime: don't treat it as speech
                            chunks.get_nowait()
                    audio = record_command(chunks, lead_in=np.zeros(0, dtype="float32"), floor=background_floor(levels))
                    report(model, audio, time.perf_counter(), args.server)
                    duck(args.server, on=False)
                    print(f"  ({time.perf_counter() - t_wake:.1f} s from wake word to result)")
                    detector.reset()
                    while not chunks.empty():  # drop audio queued while classifying
                        chunks.get_nowait()
                elif args.show_scores:
                    print(f"\rwake score {detector.last_score:.2f} (threshold {detector.threshold})", end="", flush=True)
        except KeyboardInterrupt:
            print("\nExiting.")


if __name__ == "__main__":
    main()
