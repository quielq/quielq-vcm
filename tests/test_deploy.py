import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from vcm.deploy.export import export_checkpoint, quantize_int8  # noqa: E402
from vcm.deploy.runtime import OnnxIntentModel  # noqa: E402
from vcm.slots import SLOT_VOCAB  # noqa: E402
from vcm.train.architectures import CRNN  # noqa: E402
from vcm.train.dataset import LABELS  # noqa: E402
from vcm.train.train import MODELS  # noqa: E402


def _checkpoint(tmp_path, slots: bool):
    kwargs = {"num_classes": len(LABELS)}
    if slots:
        kwargs["slot_sizes"] = {k: len(v) for k, v in SLOT_VOCAB.items()}
    model = CRNN(**kwargs).eval()
    path = tmp_path / "m.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "crnn",
            "model_kwargs": kwargs,
            "labels": LABELS,
            "feature_config": {"window_s": 1.0, "trim": True},
            "slot_vocab": {k: list(v) for k, v in SLOT_VOCAB.items()} if slots else None,
            "val_acc": 0.5,
        },
        path,
    )
    return model, path


@pytest.mark.parametrize("slots", [False, True])
def test_onnx_matches_torch_and_carries_metadata(tmp_path, slots):
    model, ckpt = _checkpoint(tmp_path, slots)
    export_checkpoint(ckpt, tmp_path / "m.onnx", MODELS)
    quantize_int8(tmp_path / "m.onnx", tmp_path / "m.int8.onnx")
    onnx_model = OnnxIntentModel(tmp_path / "m.onnx")
    assert onnx_model.labels == list(LABELS)
    assert onnx_model.feature_config == {"window_s": 1.0, "trim": True}
    x = np.random.default_rng(0).standard_normal((3, 40, 101)).astype("float32")
    with torch.no_grad():
        expected = torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()
    got = onnx_model.probabilities(x)
    assert np.allclose(got["intent"], expected, atol=1e-5)
    assert (set(k for k in got if k.startswith("slot_")) == {f"slot_{k}" for k in SLOT_VOCAB}) == slots
    int8 = OnnxIntentModel(tmp_path / "m.int8.onnx")
    assert int8.labels == list(LABELS) and int8.probabilities(x)["intent"].shape == (3, len(LABELS))
    pred = onnx_model.predict_audio(np.zeros(8000, dtype="float32"))
    assert pred.intent in LABELS and (pred.slot_value is None or pred.intent in SLOT_VOCAB)
