"""Streaming wake-word detection: score a sliding window every HOP_S and
fire when the "wake" probability stays above a threshold.

The same trigger logic serves the live microphone loop (WakeWordDetector)
and offline evaluation (window_scores + triggers over a long recording),
so the false-wake-up rate measured offline is the one the device has.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from vcm.audio.capture import SAMPLE_RATE
from vcm.audio.features import extract_log_mel
from vcm.wakeword import HOP_S, WINDOW_S

# Fire only if CONSECUTIVE successive windows (0.1 s apart) score above the
# threshold: one noisy window can't trigger, and the phrase stays in the
# window for several hops. Then ignore the next REFRACTORY_S, so one
# utterance fires once.
CONSECUTIVE = 2
REFRACTORY_S = 2.0

ScoreFn = Callable[[np.ndarray], np.ndarray]  # (batch, 40, frames) features -> (batch,) wake probability


def onnx_scorer(path) -> ScoreFn:
    from vcm.deploy.runtime import OnnxIntentModel

    model = OnnxIntentModel(path)
    wake = model.labels.index("wake")
    return lambda features: model.probabilities(features)["intent"][:, wake]


def window_scores(audio: np.ndarray, score: ScoreFn, batch: int = 256) -> np.ndarray:
    """Wake probability for every HOP_S-spaced WINDOW_S window of `audio`."""
    n, hop = int(WINDOW_S * SAMPLE_RATE), int(HOP_S * SAMPLE_RATE)
    starts = range(0, max(len(audio) - n, 0) + 1, hop)
    out, buf = [], []
    for s in starts:
        buf.append(extract_log_mel(audio[s : s + n], SAMPLE_RATE, window_s=WINDOW_S, trim=False))
        if len(buf) == batch:
            out.append(score(np.stack(buf)))
            buf = []
    if buf:
        out.append(score(np.stack(buf)))
    return np.concatenate(out) if out else np.zeros(0)


def triggers(scores: np.ndarray, threshold: float, consecutive: int = CONSECUTIVE, refractory_s: float = REFRACTORY_S) -> list[int]:
    """Hop indices where the detector fires."""
    fired, run, blocked_until = [], 0, -1
    refractory = int(refractory_s / HOP_S)
    for i, s in enumerate(scores):
        run = run + 1 if s >= threshold else 0
        if run >= consecutive and i > blocked_until:
            fired.append(i)
            blocked_until, run = i + refractory, 0
    return fired


class WakeWordDetector:
    """Feed microphone audio in any chunk size; `feed` returns True when the
    wake phrase is detected."""

    def __init__(self, score: ScoreFn, threshold: float, consecutive: int = CONSECUTIVE, refractory_s: float = REFRACTORY_S):
        self.score, self.threshold, self.consecutive = score, threshold, consecutive
        self.refractory_hops = int(refractory_s / HOP_S)
        self.n, self.hop = int(WINDOW_S * SAMPLE_RATE), int(HOP_S * SAMPLE_RATE)
        self.reset()

    def reset(self) -> None:
        self.buffer = np.zeros(self.n, dtype="float32")
        self.pending = np.zeros(0, dtype="float32")
        self.run = 0
        self.cooldown = 0
        self.last_score = 0.0

    def feed(self, chunk: np.ndarray) -> bool:
        self.pending = np.concatenate([self.pending, chunk.astype("float32").reshape(-1)])
        fired = False
        while len(self.pending) >= self.hop:
            self.buffer = np.concatenate([self.buffer[self.hop :], self.pending[: self.hop]])
            self.pending = self.pending[self.hop :]
            features = extract_log_mel(self.buffer, SAMPLE_RATE, window_s=WINDOW_S, trim=False)
            self.last_score = float(self.score(features[None])[0])
            self.cooldown = max(0, self.cooldown - 1)
            self.run = self.run + 1 if self.last_score >= self.threshold else 0
            if self.run >= self.consecutive and self.cooldown == 0:
                fired, self.run, self.cooldown = True, 0, self.refractory_hops
        return fired
