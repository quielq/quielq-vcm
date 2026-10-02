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
    def stop(self): self._do("stop")
    def next(self): self._do("next")
    def set_volume(self, p): self._do("volume", p)
    def now_playing(self): return "Song A by Artist"
    def playback(self): return {"playing": True, "volume": 70}


class FakePhone:
    default_contact, default_message = "Mom", "hello"

    def __init__(self):
        self.sent = []

    def call(self): self.sent.append("call"); return "ok"
    def message(self): self.sent.append("message"); return "ok"


def make(tmp_path, **kw):
    state = HomeState(tmp_path / "state.json")
    x = Integrations(**{"get_volume": lambda: 50, "change_volume": lambda step: 50 + step,
                        "duck_streams": lambda: [], "restore_streams": lambda saved: None,
                        "other_audio_playing": lambda: False, **kw})
    return state, Dispatcher(state, x)


def test_every_intent_has_a_handler(tmp_path):
    state, d = make(tmp_path)
    for intent in INTENT_LABELS:
        reply = d.handle(intent, None)
        assert reply and not reply.startswith("I don't know"), intent
    assert d.handle("OUT_OF_SCOPE") == ""
    assert d.handle("unknown_background") == ""  # models before Exp 37
    assert len(state.snapshot()["log"]) == len(INTENT_LABELS) + 2


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


def test_stop_rewinds_and_pause_keeps_the_place(tmp_path):
    spotify = FakeSpotify()
    state, d = make(tmp_path, spotify=spotify)
    d.handle("PLAY_MUSIC")
    d.handle("STOP")
    d.handle("PAUSE")
    assert [c[0] for c in spotify.calls] == ["play", "stop", "pause"]

    client = SpotifyClient("id", "secret", "refresh")
    with mock.patch.object(client, "_player") as player:
        client.stop()
    assert [c.args for c in player.call_args_list] == [("PUT", "/me/player/pause"), ("PUT", "/me/player/seek")]
    assert player.call_args_list[1].kwargs == {"params": {"position_ms": 0}}


def test_local_music_play_resumes_after_pause_but_restarts_after_stop(tmp_path):
    calls = []

    class Media:
        def pause(self): calls.append("pause")
        def resume(self): calls.append("resume")
        def stop(self): calls.append("stop")

    class Library:
        def play(self):
            calls.append("play")
            return Path("song.mp3")

    state, d = make(tmp_path, local_media=Media(), local_music=Library())
    d.handle("PLAY_MUSIC")
    d.handle("PAUSE")
    d.handle("PLAY_MUSIC")  # continue where it paused
    d.handle("STOP")
    d.handle("PLAY_MUSIC")  # from the start
    assert calls == ["play", "pause", "resume", "stop", "play"]


def test_duck_lowers_and_restores_spotify(tmp_path):
    spotify = FakeSpotify()
    state, d = make(tmp_path, spotify=spotify, change_volume=lambda step: None)
    d.duck(True)
    d.duck(True)  # a second wake word while ducked must not save the ducked level
    d.duck(False)
    assert spotify.calls == [("volume", 15), ("volume", 70)]
    spotify.calls.clear()
    d.duck(True)
    d.handle("VOLUME_UP")  # while ducked: becomes the level restored afterwards
    d.duck(False)
    assert spotify.calls == [("volume", 15), ("volume", 60)]


def test_duck_lowers_and_restores_other_audio_streams(tmp_path):
    calls = []
    state, d = make(tmp_path, duck_streams=lambda: calls.append("duck") or [(7, [65536, 65536])],
                    restore_streams=lambda saved: calls.append(("restore", saved)))
    d.duck(True)
    d.duck(True)  # already ducked: don't duck the ducked level again
    d.duck(False)
    d.duck(False)
    assert calls == ["duck", ("restore", [(7, [65536, 65536])])]


def test_a_duck_that_is_never_ended_undoes_itself(tmp_path, monkeypatch):
    from vcm.home import dispatcher as dispatcher_module

    monkeypatch.setattr(dispatcher_module, "DUCK_TIMEOUT_S", 0.05)
    calls = []
    state, d = make(tmp_path, duck_streams=lambda: calls.append("duck") or [(7, [65536])],
                    restore_streams=lambda saved: calls.append("restore"))
    d.duck(True)  # and the listener never sends duck(False)
    __import__("time").sleep(0.3)
    assert calls == ["duck", "restore"]


def test_duck_streams_skips_kiwis_own_sounds():
    from vcm.home import volume

    streams = [
        {"index": 1, "properties": {"application.process.binary": "librespot"}, "volume": {"front-left": {"value": 65536}, "front-right": {"value": 32768}}},
        {"index": 2, "properties": {"application.process.binary": "python3.11"}, "volume": {"mono": {"value": 65536}}},
        {"index": 3, "properties": {"application.process.binary": "paplay"}, "volume": {"mono": {"value": 65536}}},
        {"index": 5, "properties": {"application.process.binary": "pacat", "application.name": "paplay"}, "volume": {"mono": {"value": 65536}}},
        {"index": 4, "properties": {"application.process.binary": "mpv"}, "volume": {"mono": {"value": 40000}}},
    ]
    ran = []

    def fake_run(cmd):
        ran.append(cmd)
        return json.dumps(streams) if cmd[:3] == ["pactl", "-f", "json"] else ""

    with mock.patch.object(volume, "_run", side_effect=fake_run), mock.patch.object(volume.platform, "system", return_value="Linux"), \
            mock.patch.object(volume.shutil, "which", return_value="/usr/bin/pactl"):
        saved = volume.duck_streams(0.25)
        volume.restore_streams(saved)
    assert saved == [(1, [65536, 32768]), (4, [40000])]
    # PipeWire volume is cubic: -12 dB (gain 0.25) is raw x 0.63, not x 0.25 (-36 dB)
    assert ["pactl", "set-sink-input-volume", "1", "41285", "20642"] in ran
    assert ["pactl", "set-sink-input-volume", "4", "25198"] in ran
    assert ["pactl", "set-sink-input-volume", "1", "65536", "32768"] in ran
    assert not any(c[2:3] in (["2"], ["3"], ["5"]) for c in ran if c[1] == "set-sink-input-volume")


def test_next_waits_for_spotify_to_switch_tracks():
    client = SpotifyClient("id", "secret", "refresh")
    items = iter([{"id": "old"}, {"id": "old"}, {"id": "new"}])
    with mock.patch.object(client, "_current_item", side_effect=lambda: next(items)), \
            mock.patch.object(client, "_player") as player, mock.patch("vcm.home.spotify.time.sleep"):
        client.next()
    player.assert_called_once_with("POST", "/me/player/next")
    assert next(items, None) is None  # polled until the new track showed up


def test_temperature_without_sensor_still_reports_thermostat(tmp_path):
    state, d = make(tmp_path, temperature=mock.Mock(side_effect=RuntimeError("no Sense HAT")))
    assert d.handle("TEMPERATURE") == "The thermostat is set to 24 degrees."


def test_thermostat_simulation_drifts_room_to_target(tmp_path):
    state, d = make(tmp_path, temperature=lambda: 27.0)
    assert d.handle("TEMPERATURE") == "It's 27 degrees in the room. Cooling to 24."
    clock = [0.0]
    sched = Scheduler(state, on_alert=lambda text, kind: None, clock=lambda: clock[0])
    for _ in range(200):
        clock[0] += 2.0
        sched.tick()
    assert state.snapshot()["thermostat"]["room_c"] == 24
    assert d.handle("TEMPERATURE") == "It's 24 degrees, right at the thermostat setting."


def test_temperature_slot_sets_the_thermostat(tmp_path):
    state, d = make(tmp_path, temperature=lambda: 27.0)
    assert d.handle("TEMPERATURE", "22 degrees") == "Thermostat set to 22 degrees. Cooling from 27."
    d.handle("TEMPERATURE", "up")
    assert state.snapshot()["thermostat"]["target_c"] == 23
    d.handle("TEMPERATURE", "45")
    assert state.snapshot()["thermostat"]["target_c"] == 30


def test_reminder_slot_fills_the_dashboard_text(tmp_path):
    state, d = make(tmp_path)
    assert d.handle("CREATE_REMINDER", "drink water") == "Okay, I'll remind you to drink water."
    assert state.snapshot()["reminders"][0]["text"] == "Drink water"


def test_stop_silences_a_ringing_alarm_before_touching_music(tmp_path):
    ringer, spotify = mock.Mock(), FakeSpotify()
    ringer.stop.return_value = True
    state, d = make(tmp_path, ringer=ringer, spotify=spotify)
    assert d.handle("STOP") == "Okay."
    assert spotify.calls == []
    ringer.stop.return_value = False
    d.handle("STOP")
    assert spotify.calls == [("stop",)]  # pause + rewind (SpotifyClient.stop)


def test_ringer_rings_until_stopped():
    from vcm.home.ringer import Ringer

    plays, said = [], []
    ringer = Ringer(play=lambda path: (plays.append(path), __import__("time").sleep(0.01)), say=said.append)
    ringer.ring("Your timer is done.", "timer")
    __import__("time").sleep(0.1)
    assert ringer.ringing and ringer.stop() and not ringer.ringing
    assert plays[0].endswith("timer.wav") and said == ["Your timer is done."]


def test_spoken_clock_times():
    from vcm.tts.speak import speakable

    assert speakable("Alarm set for 5:00 PM today.") == "Alarm set for 5 PM today."
    assert speakable("It's 10:45 AM.") == "It's 10 45 AM."
    assert speakable("It's 6:05 PM, Monday") == "It's 6 oh 5 PM, Monday"
    assert speakable("at 12:00 AM") == "at midnight"


def test_sense_hat_cpu_heat_compensation():
    from vcm.hal.temperature import compensate

    assert round(compensate(50.8, 68.6, factor=0.85), 1) == 29.9
    assert compensate(25.0, None, factor=0.85) == 25.0  # no CPU reading: raw
    assert compensate(25.0, 60.0, factor=0) == 25.0  # correction off


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
    sched = Scheduler(state, on_alert=lambda text, kind: alerts.append(text), clock=lambda: max(timer["ends_at"], alarm["next_at"]) + 1)
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


def test_music_from_any_app_counts_as_noisy_but_kiwis_sounds_dont():
    from vcm.home import volume

    def check(streams):
        with mock.patch.object(volume, "_run", return_value=json.dumps(streams)), \
                mock.patch.object(volume.platform, "system", return_value="Linux"), \
                mock.patch.object(volume.shutil, "which", return_value="/usr/bin/pactl"):
            return volume.other_audio_playing()

    librespot = {"properties": {"application.process.binary": "librespot"}, "corked": False}
    assert check([librespot])
    assert not check([{**librespot, "corked": True}])  # paused
    assert not check([{"properties": {"application.process.binary": "pacat", "application.name": "paplay"}, "corked": False}])
    assert not check([])
