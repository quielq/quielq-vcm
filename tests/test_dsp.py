"""vcm.audio.dsp must keep matching librosa exactly: every model was
trained on librosa features, and the device computes them with dsp."""

import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from vcm.audio import dsp  # noqa: E402


@pytest.mark.parametrize("sr", [8000, 16000, 22050])
def test_melspectrogram_and_db_match_librosa(sr):
    rng = np.random.default_rng(sr)
    y = (0.1 * rng.standard_normal(sr * 2)).astype("float32")
    y[: sr // 2] *= 0.01  # quiet lead-in, like a real push-to-talk clip
    ref = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=40)
    got = dsp.melspectrogram(y, sr, 400, 160, 40)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() / ref.max() < 1e-5
    assert np.abs(dsp.power_to_db_ref_max(got) - librosa.power_to_db(ref, ref=np.max)).max() < 1e-2


@pytest.mark.parametrize("n", [0, 300, 16000, 40000])
def test_trim_bounds_match_librosa(n):
    rng = np.random.default_rng(n)
    y = np.zeros(n + 16000, dtype="float32")
    y[8000 : 8000 + n] = 0.2 * rng.standard_normal(n)
    _, (start, end) = librosa.effects.trim(y, top_db=30)
    assert dsp.trim_bounds(y, 30) == (start, end)


def test_mel_filters_match_librosa():
    assert np.allclose(dsp.mel_filters(16000, 400, 40), librosa.filters.mel(sr=16000, n_fft=400, n_mels=40), atol=1e-7)
