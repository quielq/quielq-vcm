import random

import numpy as np

from vcm.audio.capture import SAMPLE_RATE
from vcm.audio.features import extract_log_mel, trim_silence
from vcm.train.wave_augment import add_noise, add_reverb, augment_waveform, change_speed, shift_start


def _tone(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32")


def test_add_noise_hits_requested_snr():
    rng = np.random.default_rng(0)
    audio = _tone()
    noise = rng.standard_normal(SAMPLE_RATE // 2).astype("float32")  # shorter -> tiled
    mixed = add_noise(audio, noise, snr_db=10.0)
    added = mixed - audio
    snr = 10 * np.log10(np.mean(audio**2) / np.mean(added**2))
    assert mixed.shape == audio.shape
    assert abs(snr - 10.0) < 0.1


def test_change_speed_changes_length_inversely():
    audio = _tone(1.0)
    assert len(change_speed(audio, 1.1)) == round(len(audio) / 1.1)
    assert len(change_speed(audio, 0.9)) == round(len(audio) / 0.9)


def test_add_reverb_keeps_length_and_peak():
    audio = _tone()
    wet = add_reverb(audio, SAMPLE_RATE, rt60_s=0.5)
    assert wet.shape == audio.shape
    assert np.isclose(np.max(np.abs(wet)), np.max(np.abs(audio)), rtol=1e-3)


def test_shift_start_only_prepends_silence():
    random.seed(0)
    audio = _tone(0.5)
    shifted = shift_start(audio, SAMPLE_RATE, max_shift_s=0.3)
    pad = len(shifted) - len(audio)
    assert 0 <= pad <= int(0.3 * SAMPLE_RATE)
    assert not shifted[:pad].any()
    assert np.array_equal(shifted[pad:], audio)


def test_augment_waveform_output_is_finite_float32():
    random.seed(1)
    out = augment_waveform(_tone(), SAMPLE_RATE, noise_bank=[np.random.randn(SAMPLE_RATE).astype("float32")])
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_trim_silence_removes_padding_but_keeps_margin():
    speech = _tone(1.0)
    padded = np.concatenate([np.zeros(SAMPLE_RATE, "float32"), speech, np.zeros(SAMPLE_RATE, "float32")])
    trimmed = trim_silence(padded, SAMPLE_RATE)
    # librosa trims at frame granularity (2048-sample frames), plus the 0.1s margin each side
    assert len(speech) <= len(trimmed) < len(speech) + int(0.5 * SAMPLE_RATE)
    assert len(trimmed) < len(padded) - SAMPLE_RATE


def test_extract_log_mel_window_s_sets_frame_count():
    audio = _tone(1.0)
    assert extract_log_mel(audio, window_s=5.0).shape[1] > extract_log_mel(audio, window_s=3.0).shape[1]


def test_extract_log_mel_trim_handles_pure_silence():
    features = extract_log_mel(np.zeros(SAMPLE_RATE, "float32"), trim=True)
    assert np.isfinite(features).all()


def test_add_noise_with_rng_is_reproducible():
    import random

    audio = np.ones(1000, dtype="float32")
    noise = np.random.default_rng(0).standard_normal(5000).astype("float32")
    a = add_noise(audio, noise, 10.0, random.Random(3))
    b = add_noise(audio, noise, 10.0, random.Random(3))
    assert np.array_equal(a, b)
