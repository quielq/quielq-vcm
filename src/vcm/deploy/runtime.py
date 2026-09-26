"""ONNX Runtime inference for the intent/slot model and the wake-word
detector. Dependencies: onnxruntime and numpy (feature extraction is
numpy-only, vcm.audio.dsp). This is what runs on the Raspberry Pi.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vcm.audio.capture import SAMPLE_RATE
from vcm.audio.features import extract_log_mel


def _session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    # Tiny model: one thread is fastest and leaves the other cores free
    # (the wake-word detector runs continuously alongside it).
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _metadata(session) -> dict[str, str]:
    return session.get_modelmeta().custom_metadata_map


@dataclass
class Prediction:
    intent: str
    confidence: float
    slot_value: str | None = None
    slot_confidence: float | None = None


class OnnxIntentModel:
    def __init__(self, path: Path):
        self.session = _session(path)
        meta = _metadata(self.session)
        self.labels = json.loads(meta["labels"])
        self.feature_config = json.loads(meta["feature_config"])
        self.slot_vocab = json.loads(meta.get("slot_vocab", "{}"))
        self.output_names = [o.name for o in self.session.get_outputs()]

    def probabilities(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """features: (40, n_frames) or (batch, 40, n_frames) -> {output name: probs}."""
        if features.ndim == 2:
            features = features[None]
        outputs = self.session.run(None, {"features": features.astype("float32")})
        return dict(zip(self.output_names, outputs))

    def predict_audio(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Prediction:
        probs = self.probabilities(extract_log_mel(audio, sample_rate, **self.feature_config))
        intent_probs = probs["intent"][0]
        k = int(intent_probs.argmax())
        prediction = Prediction(self.labels[k], float(intent_probs[k]))
        slot_key = f"slot_{prediction.intent}"
        if slot_key in probs:
            slot_probs = probs[slot_key][0]
            j = int(slot_probs.argmax())
            prediction.slot_value = self.slot_vocab[prediction.intent][j]
            prediction.slot_confidence = float(slot_probs[j])
        return prediction
