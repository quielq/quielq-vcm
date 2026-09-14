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
