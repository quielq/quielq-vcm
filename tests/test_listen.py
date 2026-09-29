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
