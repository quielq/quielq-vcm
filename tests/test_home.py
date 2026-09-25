import json
import threading
import urllib.request
from datetime import datetime
from http.server import ThreadingHTTPServer
from unittest import mock

import pytest

from vcm.dataset.sources.dataset_schema import INTENT_LABELS
from vcm.home.dispatcher import Dispatcher, Integrations
from vcm.home.phone import PhoneBridge
from vcm.home.scheduler import Scheduler, duration_seconds, next_alarm_time, spoken_duration
from vcm.home.server import Speaker, apply_action, make_handler
from vcm.home.spotify import SpotifyClient, SpotifyError
from vcm.home.state import HomeState


class FakeSpotify:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def _do(self, name, *a):
        if self.fail:
            raise SpotifyError("no device")
        self.calls.append((name, *a))

    def play(self, uri=None): self._do("play")
    def pause(self): self._do("pause")
    def next(self): self._do("next")
    def set_volume(self, p): self._do("volume", p)
    def now_playing(self): return "Song A by Artist"


class FakePhone:
    default_contact, default_message = "Mom", "hello"

    def __init__(self):
        self.sent = []

    def call(self): self.sent.append("call"); return "ok"
    def message(self): self.sent.append("message"); return "ok"


def make(tmp_path, **kw):
    state = HomeState(tmp_path / "state.json")
    x = Integrations(get_volume=lambda: 50, change_volume=lambda step: 50 + step, **kw)
    return state, Dispatcher(state, x)


def test_every_intent_has_a_handler(tmp_path):
    state, d = make(tmp_path)
    for intent in INTENT_LABELS:
        reply = d.handle(intent, None)
        assert reply and not reply.startswith("I don't know"), intent
    assert d.handle("unknown_background") == ""
    assert len(state.snapshot()["log"]) == len(INTENT_LABELS) + 1


def test_lights_brightness_color_and_state_persists(tmp_path):
    state, d = make(tmp_path)
    d.handle("LIGHT_ON")
    d.handle("BRIGHTNESS", "60%")
    d.handle("COLOR", "blue")
    assert state.snapshot()["lights"] == {"on": True, "brightness": 60, "color": "blue"}
    d.handle("LIGHT_OFF")
    reloaded = HomeState(tmp_path / "state.json").snapshot()["lights"]
    assert reloaded == {"on": False, "brightness": 60, "color": "blue"}


def test_music_uses_spotify_and_tracks_state(tmp_path):
    spotify = FakeSpotify()
    state, d = make(tmp_path, spotify=spotify)
    assert d.handle("PLAY_MUSIC") == "Playing Song A by Artist."
    d.handle("NEXT")
    d.handle("PAUSE")
    assert [c[0] for c in spotify.calls] == ["play", "next", "pause"]
    assert state.snapshot()["music"]["playing"] is False
    assert d.handle("VOLUME_UP") == "Volume 60 percent."


def test_music_falls_back_to_local_when_spotify_fails(tmp_path):
    local, media = mock.Mock(), mock.Mock()
    local.play.return_value = mock.Mock(stem="local song")
    state, d = make(tmp_path, spotify=FakeSpotify(fail=True), local_music=local, local_media=media)
    assert d.handle("PLAY_MUSIC") == "Playing local song."
    d.handle("PAUSE")
    media.pause.assert_called_once()
    assert state.snapshot()["music"]["source"] == "local"


def test_timer_alarm_and_scheduler_fire(tmp_path):
    state, d = make(tmp_path)
    assert d.handle("TIMER", "1m30s") == "Timer set for 1 minute and 30 seconds."
    d.handle("ALARM", "6:30 AM")
    snap = state.snapshot()
    timer, alarm = snap["timers"][0], snap["alarms"][0]
    alerts = []
    sched = Scheduler(state, on_alert=alerts.append, clock=lambda: max(timer["ends_at"], alarm["next_at"]) + 1)
    fired = sched.tick()
    assert len(fired) == 2 and alerts == fired
    assert state.snapshot()["timers"] == [] and state.snapshot()["alarms"] == []
    assert len(state.snapshot()["alerts"]) == 2
    assert d.handle("TIMER", None).startswith("How long")


def test_reminders_and_dashboard_actions(tmp_path):
    state, d = make(tmp_path)
    d.handle("CREATE_REMINDER")
    rid = state.snapshot()["reminders"][0]["id"]
    apply_action(state, {"action": "reminder_edit", "id": rid, "text": "buy milk"})
    apply_action(state, {"action": "reminder_add", "text": "call home"})
    assert d.handle("LIST_REMINDERS") == "You have 2 reminders: buy milk; call home."
    apply_action(state, {"action": "reminder_delete", "id": rid})
    apply_action(state, {"action": "thermostat", "target_c": 99})
    assert state.snapshot()["thermostat"]["target_c"] == 30
    with pytest.raises(ValueError):
        apply_action(state, {"action": "reminder_add", "text": "   "})


def test_phone_via_bridge_or_simulated(tmp_path):
    phone = FakePhone()
    state, d = make(tmp_path, phone=phone)
    assert d.handle("CALL") == "Calling Mom."
    assert d.handle("MESSAGE") == "Message sent to Mom."
    assert phone.sent == ["call", "message"]
    state2, d2 = make(tmp_path / "b")
    assert "simulated" in d2.handle("CALL")
    assert state2.snapshot()["calls"][0]["status"].startswith("simulated")


def test_failing_integration_does_not_raise(tmp_path):
    state, d = make(tmp_path, weather=mock.Mock(side_effect=RuntimeError("offline")))
    assert d.handle("WEATHER") == "Sorry, that didn't work: offline"


def test_slot_label_helpers():
    assert duration_seconds("1h30m") == 5400 and duration_seconds("45s") == 45
    assert spoken_duration(3600) == "1 hour"
    now = datetime(2026, 9, 25, 7, 0)
    assert next_alarm_time("6:30 AM", now) == datetime(2026, 9, 26, 6, 30)
    assert next_alarm_time("9:00 PM", now) == datetime(2026, 9, 25, 21, 0)
    assert next_alarm_time("12:00 AM", now) == datetime(2026, 9, 26, 0, 0)


def test_server_round_trip(tmp_path):
    state, d = make(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state, d, Speaker(enabled=False)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        def post(path, body):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())

        assert post("/api/command", {"intent": "LIGHT_ON", "confidence": 0.98})["reply"] == "Lights on."
        with urllib.request.urlopen(base + "/api/state") as r:
            assert json.loads(r.read())["lights"]["on"] is True
        with urllib.request.urlopen(base + "/") as r:
            assert b"Kiwi Home" in r.read()
        with urllib.request.urlopen(base + "/api/meta") as r:
            assert "TIMER" in json.loads(r.read())["slots"]
        with pytest.raises(urllib.error.HTTPError):
            post("/api/command", {"intent": "NOT_AN_INTENT"})
    finally:
        server.shutdown()


def test_spotify_client_requests(monkeypatch):
    calls = []

    def fake_request(method, url, **kw):
        calls.append((method, url, kw.get("params"), kw.get("json")))
        body = {"devices": [{"id": "d1", "name": "Kiwi", "is_active": False}]} if url.endswith("/devices") else {}
        return mock.Mock(status_code=200 if url.endswith("/devices") else 204, json=lambda: body, text="", content=b"")

    monkeypatch.setattr("vcm.home.spotify.requests.post", lambda *a, **k: mock.Mock(status_code=200, json=lambda: {"access_token": "t", "expires_in": 3600}))
    monkeypatch.setattr("vcm.home.spotify.requests.request", fake_request)
    client = SpotifyClient("id", "secret", "refresh", device_name="kiwi", default_uri="spotify:playlist:abc")
    client.play()
    client.set_volume(150)
    plays = [c for c in calls if c[1].endswith("/play")]
    assert plays[0][2]["device_id"] == "d1" and plays[0][3] == {"context_uri": "spotify:playlist:abc"}
    assert [c for c in calls if c[1].endswith("/volume")][0][2]["volume_percent"] == 100
    with pytest.raises(SpotifyError):
        SpotifyClient("", "", "")


def test_phone_bridge_client_and_mac_commands(monkeypatch):
    sent = []
    monkeypatch.setattr(
        "vcm.home.phone.requests.post",
        lambda url, json, headers, timeout: sent.append((url, json, headers)) or mock.Mock(status_code=200, json=lambda: {"status": "ok"}),
    )
    bridge = PhoneBridge("http://mac:8765", "secret", {"Mom": "+639171234567"}, "Mom", "hi")
    bridge.message()
    assert sent[0][0] == "http://mac:8765/message" and sent[0][1]["number"] == "+639171234567"
    assert sent[0][2]["X-Bridge-Token"] == "secret"

    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("bridge", Path(__file__).parents[1] / "scripts/mac_phone_bridge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cmd = mod.message_command("+63917", 'hi" & do shell script "x', "iMessage")
    assert cmd[0] == "osascript" and cmd[3] == "+63917" and cmd[4].startswith("hi")  # text passed as argv, not spliced
    assert mod.call_command("+63 917-123") == ["open", "tel:+63917123"]
