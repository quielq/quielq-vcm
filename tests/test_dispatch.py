import pytest

from vcm.dispatch import DISPATCH, dispatch
from vcm.taxonomy import LABELS, UNKNOWN_BACKGROUND


def test_every_label_has_a_handler():
    assert set(DISPATCH.keys()) == set(LABELS)


def test_unknown_background_is_a_safe_noop():
    assert dispatch(UNKNOWN_BACKGROUND) == ""


def test_dispatch_rejects_unregistered_label():
    with pytest.raises(KeyError):
        dispatch("not_a_real_label")
