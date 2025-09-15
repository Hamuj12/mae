"""Utility for programmatically composing a YOLO model with the MAE backbone."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

from ultralytics_mae.nn.backbones.mae_vit import MaeViTBackbone
from ultralytics_mae.nn.necks.mae_fpn import MaeSimpleFPN


def _assign_sequential(modules: Iterable[torch.nn.Module]) -> torch.nn.Sequential:
    """Attach YOLO-specific metadata to modules before wrapping them in ``Sequential``."""
    wrapped = []
    for idx, module in enumerate(modules):
        module.i = idx  # type: ignore[attr-defined]
        module.f = -1  # type: ignore[attr-defined]
        module.type = module.__class__.__name__  # type: ignore[attr-defined]
        module.np = sum(p.numel() for p in module.parameters())  # type: ignore[attr-defined]
        wrapped.append(module)
    return torch.nn.Sequential(*wrapped)


def _build_model(opt: argparse.Namespace) -> YOLO:
    """Construct a YOLO detection model that uses the MAE encoder and FPN."""
    yolo = YOLO(opt.base)
    det_model = yolo.model

    backbone = MaeViTBackbone(
        ckpt_path=opt.ckpt,
        embed_dim=opt.embed_dim,
        patch_size=opt.patch_size,
        freeze=opt.freeze,
    )

    neck = MaeSimpleFPN(in_channels=backbone.out_channels, out_channels=opt.neck_out)

    base_detect = det_model.model[-1]
    nc = opt.nc if opt.nc is not None else getattr(base_detect, "nc", None)
    if nc is None:
        nc = 80
    detect = Detect(nc=nc, ch=[opt.neck_out] * 3)

    stride_p4 = opt.patch_size
    if stride_p4 < 2 or stride_p4 % 2 != 0:
        raise ValueError("patch size must be an even value >= 2 to build P3/P4/P5 feature maps")
    stride_p3 = stride_p4 // 2
    stride_p5 = stride_p4 * 2
    detect.stride = torch.tensor([stride_p3, stride_p4, stride_p5], dtype=torch.float32)
    det_model.stride = detect.stride

    detect.bias_init()

    det_model.model = _assign_sequential((backbone, neck, detect))
    det_model.save = []
    det_model.yaml["nc"] = nc
    det_model.nc = nc
    det_model.names = {i: f"{i}" for i in range(nc)}
    if hasattr(det_model, "args"):
        det_model.args["nc"] = nc
    yolo.overrides["nc"] = nc

    return yolo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to MAE encoder weights (.pt or .ckpt)")
    parser.add_argument("--out", type=str, required=True, help="Output path for the composed YOLO model (.pt)")
    parser.add_argument("--base", type=str, default="yolov8n.yaml", help="Base YOLO model definition")
    parser.add_argument("--embed-dim", type=int, default=768, help="MAE embed dimension")
    parser.add_argument("--patch-size", type=int, default=16, help="MAE patch size")
    parser.add_argument("--neck-out", type=int, default=256, help="FPN output channels")
    parser.add_argument("--nc", type=int, help="Number of detection classes")
    parser.add_argument("--freeze", action="store_true", help="Freeze MAE backbone parameters")
    opt = parser.parse_args()

    model = _build_model(opt)

    out_path = Path(opt.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Saved MAE+YOLO model to {out_path}")


if __name__ == "__main__":
    main()
