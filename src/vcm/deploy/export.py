"""Export a trained checkpoint to ONNX, then quantize it to int8.

Everything inference needs travels inside the .onnx file as metadata
(labels, feature settings, slot vocabularies), so the Raspberry Pi needs
only onnxruntime + numpy + librosa: no torch, no checkpoint, no repo
config to keep in sync.

The graph takes one log-mel feature matrix and returns softmax
probabilities: `intent` (1, n_labels) plus one `slot_<INTENT>` output per
slot head, if the model has them.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class _ProbsWrapper(nn.Module):
    def __init__(self, model: nn.Module, slot_intents: list[str]):
        super().__init__()
        self.model = model
        self.slot_intents = slot_intents

    def forward(self, features: torch.Tensor):
        if self.slot_intents:
            logits, slot_logits = self.model.forward_with_slots(features)
            return (torch.softmax(logits, dim=1), *(torch.softmax(slot_logits[k], dim=1) for k in self.slot_intents))
        return torch.softmax(self.model(features), dim=1)


def export_checkpoint(ckpt_path: Path, out_path: Path, models: dict, n_frames: int | None = None) -> dict:
    """Write `out_path` (fp32 ONNX). Returns the metadata written."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = models[ckpt["model_name"]](**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    feature_config = ckpt.get("feature_config") or {"window_s": 3.0, "trim": False}
    if n_frames is None:
        from vcm.audio.features import HOP_LENGTH
        from vcm.audio.capture import SAMPLE_RATE

        n_frames = int(feature_config["window_s"] * SAMPLE_RATE) // HOP_LENGTH + 1
    slot_vocab = ckpt.get("slot_vocab") or {}
    slot_intents = list(slot_vocab)

    dummy = torch.zeros(1, 40, n_frames)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _ProbsWrapper(model, slot_intents),
        (dummy,),
        str(out_path),
        input_names=["features"],
        output_names=["intent", *(f"slot_{k}" for k in slot_intents)],
        # Batch is variable so evaluation can score the test set in batches;
        # on the device it's always 1.
        dynamic_axes={name: {0: "batch"} for name in ["features", "intent", *(f"slot_{k}" for k in slot_intents)]},
        opset_version=17,
        dynamo=False,
    )

    import onnx

    metadata = {
        "labels": json.dumps(list(ckpt["labels"])),
        "feature_config": json.dumps(feature_config),
        "slot_vocab": json.dumps(slot_vocab),
        "model_name": ckpt["model_name"],
        "source_checkpoint": Path(ckpt_path).name,
        "val_acc": f"{ckpt.get('val_acc', float('nan')):.4f}",
    }
    graph = onnx.load(str(out_path))
    for key, value in metadata.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(graph, str(out_path))
    return metadata


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    """Dynamic int8 quantization: weights stored as int8, activations
    quantized on the fly. No calibration data needed; covers Conv, MatMul
    and GRU weights. Metadata is carried over."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    src, dst = onnx.load(str(fp32_path)), onnx.load(str(int8_path))
    if not dst.metadata_props:
        for prop in src.metadata_props:
            entry = dst.metadata_props.add()
            entry.key, entry.value = prop.key, prop.value
        onnx.save(dst, str(int8_path))
