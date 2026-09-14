"""Intent classification interface.

No trained model exists yet (dataset + DGX training, Section 5/9, are a
separate later workstream). `StubIntentModel` lets everything downstream
of inference (dispatch, actions, main loop) be built and tested now; a
real TFLite/ONNX-backed model swaps in later behind the same
`predict(features) -> str` signature without touching callers.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

import numpy as np

from vcm.taxonomy import LABELS, UNKNOWN_BACKGROUND


class IntentModel(Protocol):
    def predict(self, features: np.ndarray) -> str: ...


class StubIntentModel:
    """Deterministic-by-default stand-in with no real classification behavior."""

    def __init__(self, fixed_label: str = UNKNOWN_BACKGROUND) -> None:
        if fixed_label not in LABELS:
            raise ValueError(f"{fixed_label!r} is not in vcm.taxonomy.LABELS")
        self._fixed_label = fixed_label

    def predict(self, features: np.ndarray) -> str:
        return self._fixed_label


class RandomIntentModel:
    """Uniform-random label, useful for exercising the full dispatch path in dev."""

    def predict(self, features: np.ndarray) -> str:
        return random.choice(LABELS)


class TFLiteIntentModel:
    """Real backend, wired up once a trained+quantized model exists (Section 5).

    Not used by default yet — instantiate explicitly once
    `models/vcm.tflite` (or an equivalent export path) exists.
    """

    def __init__(self, model_path: Path) -> None:
        import tflite_runtime.interpreter as tflite  # type: ignore[import-not-found]

        self._interpreter = tflite.Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        self._input_index = self._interpreter.get_input_details()[0]["index"]
        self._output_index = self._interpreter.get_output_details()[0]["index"]

    def predict(self, features: np.ndarray) -> str:
        batched = features[np.newaxis, ...].astype("float32")
        self._interpreter.set_tensor(self._input_index, batched)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_index)
        return LABELS[int(np.argmax(output))]


def load_default_model() -> IntentModel:
    """Return the model to use until a trained one is exported (see class docstring)."""
    return StubIntentModel()
