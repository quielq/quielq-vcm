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
