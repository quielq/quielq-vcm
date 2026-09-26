import numpy as np

from vcm.audio.resample import StreamResampler


def tone(freq, rate, seconds=1.0):
    t = np.arange(int(rate * seconds)) / rate
    return np.sin(2 * np.pi * freq * t).astype("float32")


def test_48k_to_16k_keeps_speech_band_and_length():
    out = StreamResampler(48000, 16000)(tone(1000, 48000))
    assert abs(len(out) - 16000) <= 1
    spectrum = np.abs(np.fft.rfft(out[200:]))
    peak_hz = np.argmax(spectrum) * 16000 / len(out[200:])
    assert abs(peak_hz - 1000) < 5
    assert 0.9 < np.abs(out[500:]).max() < 1.1


def test_removes_content_above_new_nyquist():
    out = StreamResampler(48000, 16000)(tone(12000, 48000))  # would alias to 4 kHz
    assert np.abs(out[200:]).max() < 0.05


def test_chunked_matches_whole_for_44100():
    x = tone(700, 44100) + 0.3 * tone(3000, 44100)
    whole = StreamResampler(44100, 16000)(x)
    r = StreamResampler(44100, 16000)
    chunked = np.concatenate([r(c) for c in np.array_split(x, 37)])
    assert abs(len(chunked) - len(whole)) <= 1
    n = min(len(chunked), len(whole))
    assert np.allclose(chunked[:n], whole[:n], atol=1e-4)
