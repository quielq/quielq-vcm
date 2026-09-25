import numpy as np

from vcm.audio.features import N_MELS, WINDOW_S, extract_log_mel
from vcm.audio.capture import SAMPLE_RATE


def test_extract_log_mel_shape_and_dtype_on_short_clip():
    short_audio = np.random.uniform(-0.1, 0.1, size=int(SAMPLE_RATE * 0.5)).astype(
        "float32"
    )
    features = extract_log_mel(short_audio)
    assert features.dtype == np.float32
    assert features.shape[0] == N_MELS


def test_extract_log_mel_fixed_shape_regardless_of_input_length():
    short = np.random.uniform(-0.1, 0.1, size=int(SAMPLE_RATE * 0.3)).astype("float32")
    long = np.random.uniform(-0.1, 0.1, size=int(SAMPLE_RATE * (WINDOW_S + 1))).astype(
        "float32"
    )
    assert extract_log_mel(short).shape == extract_log_mel(long).shape


def test_extract_log_mel_handles_silence():
    silence = np.zeros(int(SAMPLE_RATE * WINDOW_S), dtype="float32")
    features = extract_log_mel(silence)
    assert np.isfinite(features).all()


def test_extract_log_mel_is_roughly_normalized():
    # Not exactly zero-mean/unit-variance for any single clip (the
    # normalization constants are dataset-level, not per-clip), but a
    # real-ish clip shouldn't be wildly outside a modest range either —
    # catches a normalization constant that's badly wrong (e.g. swapped
    # mean/std, or applied twice) without requiring a real audio fixture.
    rng = np.random.default_rng(0)
    audio = rng.uniform(-0.2, 0.2, size=int(SAMPLE_RATE * WINDOW_S)).astype("float32")
    features = extract_log_mel(audio)
    assert -6.0 < features.mean() < 6.0
    assert 0.1 < features.std() < 10.0


def test_speech_level_ignores_surrounding_silence():
    # The same 0.5s of speech-like signal must score the same whether the
    # push-to-talk hold around it was short or long (whole-clip RMS doesn't).
    from vcm.audio.features import speech_level

    rng = np.random.default_rng(0)
    speech = (0.05 * rng.standard_normal(int(SAMPLE_RATE * 0.5))).astype("float32")
    short = np.concatenate([np.zeros(int(SAMPLE_RATE * 0.2), "float32"), speech])
    long = np.concatenate([np.zeros(int(SAMPLE_RATE * 3.0), "float32"), speech, np.zeros(SAMPLE_RATE, "float32")])
    assert abs(speech_level(short) - speech_level(long)) < 1e-3
    assert speech_level(long) > 0.04


def test_speech_level_handles_empty_and_tiny_clips():
    from vcm.audio.features import speech_level

    assert speech_level(np.zeros(0, "float32")) == 0.0
    assert speech_level(np.full(100, 0.1, "float32")) > 0.09
