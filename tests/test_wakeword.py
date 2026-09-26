import numpy as np

from vcm.audio.capture import SAMPLE_RATE
from vcm.wakeword import HOP_S, WINDOW_S
from vcm.wakeword.data import KIND_LABEL, _place
from vcm.wakeword.detector import WakeWordDetector, triggers, window_scores


def test_triggers_need_consecutive_hops_and_respect_refractory():
    scores = np.array([0.1, 0.95, 0.2, 0.95, 0.96, 0.97, 0.99, 0.1] + [0.0] * 30 + [0.95, 0.95])
    fired = triggers(scores, threshold=0.9, consecutive=2, refractory_s=2.0)
    assert fired == [4, 39]  # single 0.95 at 1 ignored; one trigger per burst; fires again after 2 s


def test_window_scores_counts_hops():
    audio = np.zeros(int(3.0 * SAMPLE_RATE), dtype="float32")
    scores = window_scores(audio, lambda f: np.full(len(f), 0.5))
    expected = int((3.0 - WINDOW_S) / HOP_S) + 1
    assert len(scores) == expected


def test_streaming_detector_matches_offline_windows():
    # A scorer that fires when the window contains sound: the live detector,
    # fed in odd-sized chunks, must fire exactly once for one burst. (Features
    # are normalized to each window's own peak, so "sound" shows up as spread
    # across frequencies/time, not as a higher maximum.)
    loud = lambda f: np.array([float(f[0].std() > 0.1)])  # noqa: E731
    det = WakeWordDetector(loud, threshold=0.5, consecutive=2, refractory_s=2.0)
    audio = np.concatenate([np.zeros(SAMPLE_RATE), 0.5 * np.random.default_rng(0).standard_normal(SAMPLE_RATE), np.zeros(SAMPLE_RATE)]).astype("float32")
    fired = [det.feed(audio[i : i + 1234]) for i in range(0, len(audio), 1234)]
    assert sum(fired) == 1


def test_place_pads_or_crops_to_window():
    import random

    n = int(WINDOW_S * SAMPLE_RATE)
    assert len(_place(np.ones(100, "float32"), n, random.Random(0))) == n
    assert len(_place(np.ones(n * 2, "float32"), n, random.Random(0))) == n
    assert KIND_LABEL["wake"] == 1 and KIND_LABEL["partial"] == 0
