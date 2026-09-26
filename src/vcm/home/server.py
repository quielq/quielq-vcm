"""The home server: receives recognized commands, runs them, serves the dashboard.

    python -m vcm.home.server [--port 8000] [--no-speak]
    open http://127.0.0.1:8000 (or http://kiwi.local:8000 from another device)

Endpoints:
- POST /api/command {intent, slot?, confidence?, source?} -> {reply}. What
  scripts/vcm_listen.py --server calls after each recognized command; the
  dashboard's "simulate a command" box uses it too.
- POST /api/action {action, ...}: the dashboard's manual controls (lamp,
  thermostat, reminder text, cancelling timers/alarms).
- GET /api/state: the whole state. GET /api/events: the same, pushed live
  (server-sent events) whenever anything changes.
- GET /api/meta: intents and slot vocabularies, for the dashboard.

Standard library only (ThreadingHTTPServer), so it fits beside the voice
pipeline on a 512 MB Pi. It has no authentication: run it on a trusted home
network only (TODO.md).
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vcm.config import REPO_ROOT, load_settings
from vcm.dataset.sources.dataset_schema import INTENT_LABELS
from vcm.home.dispatcher import Dispatcher, Integrations, integrations_from_settings
from vcm.home.scheduler import Scheduler
from vcm.home.state import HomeState, new_id
from vcm.slots import SLOT_VOCAB

STATIC_DIR = Path(__file__).parent / "static"


class Speaker:
    """Speaks replies one at a time on a background thread, so HTTP replies
    return immediately and speech never overlaps."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.queue: queue.Queue[str] = queue.Queue()
        if enabled:
            threading.Thread(target=self._run, daemon=True, name="vcm-speaker").start()

    def say(self, text: str) -> None:
        if self.enabled and text:
            self.queue.put(text)

    def _run(self) -> None:
        from vcm.tts.speak import speak

        while True:
            text = self.queue.get()
            try:
                speak(text)
            except Exception as exc:  # no TTS backend must not kill the server
                print(f"(speech failed: {exc}) {text}")


def apply_action(state: HomeState, action: dict) -> None:
    """Manual changes from the dashboard. Raises ValueError on bad input."""
    kind = action.get("action")
    if kind == "lights":
        fields = {k: action[k] for k in ("on", "brightness", "color") if k in action}
        if "brightness" in fields:
            fields["brightness"] = max(0, min(100, int(fields["brightness"])))
        state.update(lambda d: d["lights"].update(fields))
    elif kind == "thermostat":
        target = max(16, min(30, int(action["target_c"])))
        state.update(lambda d: d["thermostat"].update(target_c=target))
    elif kind in ("reminder_add", "reminder_edit"):
        text = str(action.get("text", "")).strip()[:200]
        if not text:
            raise ValueError("empty reminder")
        if kind == "reminder_add":
            import time

            state.update(lambda d: d["reminders"].append({"id": new_id(), "text": text, "created": time.time()}))
        else:
            state.update(lambda d: [r.update(text=text) for r in d["reminders"] if r["id"] == action["id"]])
    elif kind in ("reminder_delete", "timer_cancel", "alarm_delete", "alert_dismiss"):
        key = {"reminder_delete": "reminders", "timer_cancel": "timers", "alarm_delete": "alarms", "alert_dismiss": "alerts"}[kind]
        state.update(lambda d: d.__setitem__(key, [x for x in d[key] if x["id"] != action.get("id")]))
    else:
        raise ValueError(f"unknown action {kind!r}")


def make_handler(state: HomeState, dispatcher: Dispatcher, speaker: Speaker):
    meta = {"intents": list(INTENT_LABELS), "slots": {k: list(v) for k, v in SLOT_VOCAB.items()}}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, code: int, body) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}") if length else {}

        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                data = (STATIC_DIR / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/state":
                self._json(200, state.snapshot())
            elif path == "/api/meta":
                self._json(200, meta)
            elif path == "/api/events":
                self._events()
            else:
                self._json(404, {"error": "not found"})

        def _events(self) -> None:
            updates: queue.Queue[dict] = queue.Queue()
            unsubscribe = state.subscribe(updates.put)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                snapshot = state.snapshot()
                while True:
                    self.wfile.write(f"data: {json.dumps(snapshot)}\n\n".encode())
                    self.wfile.flush()
                    try:
                        snapshot = updates.get(timeout=15)
                    except queue.Empty:
                        snapshot = state.snapshot()  # keepalive, also refreshes countdown clocks
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                unsubscribe()

        def do_POST(self):  # noqa: N802
            try:
                body = self._body()
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad json"})
            if self.path == "/api/command":
                intent = body.get("intent")
                if intent not in (*INTENT_LABELS, "unknown_background"):
                    return self._json(400, {"error": f"unknown intent {intent!r}"})
                reply = dispatcher.handle(intent, body.get("slot"), body.get("confidence"), body.get("source", "voice"))
                speaker.say(reply)
                return self._json(200, {"reply": reply})
            if self.path == "/api/action":
                try:
                    apply_action(state, body)
                except (ValueError, KeyError, TypeError) as exc:
                    return self._json(400, {"error": str(exc)})
                return self._json(200, {"ok": True})
            self._json(404, {"error": "not found"})

        def log_message(self, *a):
            pass

    return Handler


def build(state_path: Path, speak: bool, integrations: Integrations | None = None):
    state = HomeState(state_path)
    speaker = Speaker(speak)
    dispatcher = Dispatcher(state, integrations or integrations_from_settings(load_settings()))
    scheduler = Scheduler(state, on_alert=speaker.say)
    return state, dispatcher, scheduler, speaker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from other devices on the network")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--state", type=Path, default=REPO_ROOT / "data/home_state.json")
    parser.add_argument("--no-speak", action="store_true", help="don't speak replies (dashboard only)")
    args = parser.parse_args()

    state, dispatcher, scheduler, speaker = build(args.state, speak=not args.no_speak)
    scheduler.start()
    x = dispatcher.x
    configured = [name for name, on in [("spotify", x.spotify), ("phone bridge", x.phone), ("weather", x.weather), ("local music", x.local_music), ("real bulb", x.bulb)] if on]
    print(f"VCM home server on http://{args.host}:{args.port}  (integrations: {', '.join(configured) or 'none, all simulated'})")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, dispatcher, speaker))
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
