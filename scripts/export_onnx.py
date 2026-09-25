#!/usr/bin/env python
"""Export a trained checkpoint to ONNX (fp32) and int8, for the Raspberry Pi
(DEPLOYMENT.md). Works for the intent/slot model and the wake-word model.

Usage:
    python scripts/export_onnx.py checkpoints/exp32_crnn_slots_s0.pt --out models/vcm_intent
    -> models/vcm_intent.onnx (fp32) and models/vcm_intent.int8.onnx

Afterwards, score the int8 file on the same test rows as the checkpoint:
    python scripts/evaluate_checkpoint.py models/vcm_intent.int8.onnx --manifest ... --split test
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from vcm.deploy.export import export_checkpoint, quantize_int8
from vcm.train.train import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="output path without extension")
    args = parser.parse_args()

    fp32 = args.out.with_suffix(".onnx")
    int8 = args.out.with_name(args.out.name + ".int8.onnx")
    metadata = export_checkpoint(args.checkpoint, fp32, MODELS)
    quantize_int8(fp32, int8)
    print(f"{fp32}: {os.path.getsize(fp32) / 1024:.0f} KB (fp32)")
    print(f"{int8}: {os.path.getsize(int8) / 1024:.0f} KB (int8 weights; GRU stays fp32, no int8 GRU kernel in ONNX Runtime)")
    print(f"metadata: {metadata}")


if __name__ == "__main__":
    main()
