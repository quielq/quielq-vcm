import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vcm_listen  # noqa: E402


def test_chime_audio_kept_only_when_still_talking():
    during = [np.full(1600, 0.01, dtype="float32")] * 3
    talking = 0.2 * np.sin(np.linspace(0, 400, 1600)).astype("float32")
    quiet = np.full(1600, 0.001, dtype="float32")
    assert len(vcm_listen.chime_lead_in(during, talking, floor=0.002)) == 4800
    assert len(vcm_listen.chime_lead_in(during, quiet, floor=0.002)) == 0
    assert len(vcm_listen.chime_lead_in([], talking, floor=0.002)) == 0


def test_soft_end_of_a_phrase_is_not_taken_for_silence():
    import queue

    rng = np.random.default_rng(0)
    loud = [0.2 * rng.standard_normal(1600).astype("float32") for _ in range(5)]   # "brightness"
    soft = [0.012 * rng.standard_normal(1600).astype("float32") for _ in range(8)]  # "...to sixty percent"
    quiet = [0.002 * rng.standard_normal(1600).astype("float32") for _ in range(20)]
    q = queue.Queue()
    for c in loud + soft + quiet:
        q.put(c)
    audio = vcm_listen.record_command(q, np.zeros(0, dtype="float32"), floor=0.005)  # 3x floor = 0.015 > soft
    assert len(audio) >= (5 + 8) * 1600  # the soft part is kept
    assert len(audio) <= (5 + 8 + 10) * 1600  # and it still ends after ~0.9 s of quiet


def test_model_hears_the_command_not_the_music_after_it():
    rng = np.random.default_rng(1)
    sec = lambda level, s: (level * rng.standard_normal(int(16000 * s))).astype("float32")
    stop = sec(0.2, 0.5)  # "stop"
    audio = np.concatenate([sec(0.002, 0.3), stop, sec(0.002, 0.9), sec(0.1, 0.4), sec(0.002, 0.5), sec(0.1, 0.4)])  # then music bursts
    span = vcm_listen.first_speech_span(audio)
    assert 0.5 <= len(span) / 16000 <= 1.0  # "stop" + margins, no bursts
    long_cmd = np.concatenate([sec(0.002, 0.3), sec(0.2, 0.8), sec(0.002, 0.3), sec(0.15, 0.8), sec(0.002, 1.0)])
    assert len(vcm_listen.first_speech_span(long_cmd)) / 16000 >= 1.9  # a short pause inside a phrase is kept


def test_duck_requests_reach_the_server_in_order(monkeypatch):
    import time

    sent = []

    def slow_first(server, path, payload, timeout=10):
        if not sent:
            time.sleep(0.2)  # the "on" request is slow (e.g. the Spotify API)
        sent.append(payload["on"])
        return {}

    monkeypatch.setattr(vcm_listen, "post_json", slow_first)
    monkeypatch.setattr(vcm_listen, "_duck_requests", None)
    vcm_listen.duck("http://x", True)
    vcm_listen.duck("http://x", False)
    deadline = time.time() + 2
    while len(sent) < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert sent == [True, False]


def test_listener_exits_when_the_mic_goes_silent(monkeypatch):
    import queue

    import pytest

    monkeypatch.setattr(vcm_listen, "MIC_TIMEOUT_S", 0.05)
    with pytest.raises(vcm_listen.MicLost):
        vcm_listen.next_chunk(queue.Queue())


def test_listener_refuses_a_speaker_monitor_as_its_microphone(monkeypatch):
    import pytest

    monkeypatch.setattr(vcm_listen, "default_input", lambda: "alsa_output.usb-Dell_SoundBar.analog-stereo.monitor")
    with pytest.raises(vcm_listen.MicLost):
        vcm_listen.guard_input(None)
    vcm_listen.guard_input("USB PnP")  # an explicit --device isn't second-guessed


def test_result_line_says_what_the_device_does():
    """One JSON line per command for the class benchmark (vcm-benchmark): the
    command and slot when acted on, OUT_OF_SCOPE when it is not."""
    import json

    from vcm.deploy.runtime import Prediction

    acted = Prediction("TIMER", 0.93)
    acted.slot_value, acted.slot_confidence = "30s", 0.99
    line = json.loads(vcm_listen.result_line(acted, 9.84, 1500.4))
    assert line == {"intent": "TIMER", "slot": "30s", "confidence": 0.93, "model_intent": "TIMER",
                    "infer_ms": 9.8, "audio_ms": 1500}
    unsure = json.loads(vcm_listen.result_line(Prediction("STOP", 0.27), 10.0, 800.0))
    assert unsure["intent"] == "OUT_OF_SCOPE" and unsure["model_intent"] == "STOP" and "slot" not in unsure
    assert json.loads(vcm_listen.result_line(Prediction("unknown_background", 0.9), 9.0, 900.0))["intent"] == "OUT_OF_SCOPE"
    assert json.loads(vcm_listen.result_line(None, 0.0, 700.0)) == {"intent": "OUT_OF_SCOPE", "infer_ms": 0.0, "audio_ms": 700}
