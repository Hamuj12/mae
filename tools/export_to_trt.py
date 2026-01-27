#!/usr/bin/env python3
"""
Export a YOLOv8 .pt model to TensorRT .engine using Ultralytics.

Ultralytics supports:
    model.export(format="engine")

Docs:
- Export: https://docs.ultralytics.com/modes/export/
- TensorRT integration: https://docs.ultralytics.com/integrations/tensorrt/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True, help="Path to YOLOv8 .pt weights")
    p.add_argument("--imgsz", type=int, default=1024, help="Square input size for export (e.g., 1024)")
    p.add_argument(
        "--half",
        action="store_true",
        help="Enable FP16 export (recommended on NVIDIA GPUs that support it)",
    )
    p.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic shapes (can be slower; fixed shapes often fastest)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="0",
        help="Export device. Examples: '0' or 'cpu'. (Ultralytics uses this for export.)",
    )
    p.add_argument(
        "--engine-out",
        type=str,
        default=None,
        help="Optional output path for engine. If omitted, Ultralytics chooses a default name next to weights.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)

    # Export directly to TensorRT engine
    export_kwargs = {
        "format": "engine",
        "imgsz": args.imgsz,
        "device": args.device,
        "half": bool(args.half),
        "dynamic": bool(args.dynamic),
    }

    # If user wants a specific engine filename, set project/name so it lands where we want.
    # Ultralytics export API varies slightly by version; safest is to export first then rename if needed.
    print(f"Exporting TensorRT engine with args: {export_kwargs}")
    out = model.export(**export_kwargs)  # returns path-like (varies by version)
    print(f"Ultralytics export returned: {out}")

    if args.engine_out:
        engine_out = Path(args.engine_out)
        engine_out.parent.mkdir(parents=True, exist_ok=True)

        # Try to locate the engine file:
        # Many versions create <weights_stem>.engine in the same folder as weights.
        weights_path = Path(args.weights)
        default_engine = weights_path.with_suffix(".engine")

        # If export returned a path, prefer it.
        candidate = None
        try:
            candidate = Path(str(out))
        except Exception:
            candidate = None

        src_engine = None
        if candidate and candidate.exists() and candidate.suffix == ".engine":
            src_engine = candidate
        elif default_engine.exists():
            src_engine = default_engine
        else:
            # fallback: search near weights
            hits = list(weights_path.parent.glob("*.engine"))
            if hits:
                hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                src_engine = hits[0]

        if src_engine is None:
            raise FileNotFoundError("Could not find the generated .engine file after export.")

        if src_engine.resolve() != engine_out.resolve():
            engine_out.write_bytes(src_engine.read_bytes())
            print(f"Copied engine to: {engine_out}")
        else:
            print(f"Engine already at: {engine_out}")


if __name__ == "__main__":
    main()
    
# PYTHONPATH=. python3 export_to_trt.py \
#   --weights /home/hm25936/mae/runs/yolov8_fda/baseline/weights/best.pt \
#   --imgsz 1024 \
#   --half \
#   --device 0 \
#   --engine-out yolov8_fda_fp16.engine