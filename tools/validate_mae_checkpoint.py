"""Validate that a Phase-B MAE checkpoint loads and forwards correctly."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from dual_yolo_mae.model import MAEBackbone
from dual_yolo_mae.utils import load_config


DEFAULT_STRIDES = (8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MAE checkpoint loading and forward pass.")
    parser.add_argument("--config", type=Path, required=True, help="Path to dual_yolo_mae/config.yaml")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run on")
    return parser.parse_args()


def build_target_shapes(input_size: int, strides: Sequence[int]) -> list[tuple[int, int]]:
    return [(input_size // s, input_size // s) for s in strides]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    mae_cfg = model_cfg.get("mae", {})

    checkpoint = mae_cfg.get("checkpoint")
    output_dims = mae_cfg.get("output_dims", (144, 144, 144))
    freeze = mae_cfg.get("freeze", True)
    input_size = int(model_cfg.get("input_size", 1024))

    backbone = MAEBackbone(
        checkpoint=checkpoint,
        output_dims=output_dims,
        freeze=freeze,
    ).to(device)

    dummy = torch.zeros((1, 3, 1024, 1024), device=device)
    target_shapes = build_target_shapes(input_size, DEFAULT_STRIDES)

    with torch.no_grad():
        outputs = backbone(dummy, target_shapes)

    print("MAE feature outputs:")
    for idx, feat in enumerate(outputs, start=1):
        print(f"Scale {idx}: shape={tuple(feat.shape)}")

    print("----- MAE Validation Summary -----")
    print("MAE checkpoint loaded successfully.")
    print("MAE forward pass completed.")
    print("----------------------------------")


if __name__ == "__main__":
    main()
