import numpy as np
import pytest

from vcm.inference.model import RandomIntentModel, StubIntentModel, load_default_model
from vcm.taxonomy import LABELS, PLAY_MUSIC, UNKNOWN_BACKGROUND


def test_default_model_is_stub_and_returns_unknown_by_default():
    model = load_default_model()
    dummy_features = np.zeros((40, 10), dtype="float32")
    assert model.predict(dummy_features) == UNKNOWN_BACKGROUND


def test_stub_model_rejects_unknown_label():
    with pytest.raises(ValueError):
        StubIntentModel(fixed_label="not_a_real_label")


def test_stub_model_returns_configured_label():
    model = StubIntentModel(fixed_label=PLAY_MUSIC)
    assert model.predict(np.zeros((40, 10))) == PLAY_MUSIC


def test_random_model_always_returns_a_known_label():
    model = RandomIntentModel()
    for _ in range(20):
        assert model.predict(np.zeros((40, 10))) in LABELS
