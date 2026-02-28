"""TensorRT engine export helper for YOLO models"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    # keep module importable on systems that only run ONNX inference
    YOLO    = None


def _resolve_engine_path(export_out, model_path: Path) -> Path:
    """Resolve generated engine path from Ultralytics export output"""

    # ultralytics return type can vary by version
    candidate       = None
    try:
        candidate = Path(str(export_out))
    except Exception:
        candidate   = None

    if candidate is not None and candidate.suffix == '.engine' and candidate.exists():
        return candidate

    # common export path when ultralytics does not return explicit location
    default_engine  = model_path.with_suffix('.engine')
    if default_engine.exists():
        return default_engine

    # final fallback for older versions that drop engine in model directory
    hits    = list(model_path.parent.glob('*.engine'))
    if hits:
        hits.sort(key = lambda p: p.stat().st_mtime, reverse = True)
        return hits[0]

    raise FileNotFoundError('Unable to locate generated .engine after export')


def export_tensorrt_engine(
                            model_path: str,
                            engine_out: str | None = None,
                            imgsz: int = 1024,
                            device: str = '0',
                            half: bool = True,
                            dynamic: bool = False,
                        ) -> Path:
    """
    Export TensorRT engine from YOLO model path

    Supports .pt directly
    """

    if YOLO is None:
        raise ImportError('ultralytics is required for TensorRT export')

    model_path_obj = Path(str(model_path)).expanduser().resolve()
    if not model_path_obj.is_file():
        raise FileNotFoundError(f'model file not found: {model_path_obj}')

    # build YOLO wrapper and ask it to export TensorRT engine
    # let ultralytics handle conversion path for .pt and supported .onnx
    model       = YOLO(str(model_path_obj))
    export_out  = model.export(
                                format = 'engine',
                                imgsz = int(imgsz),
                                device = str(device),
                                half = bool(half),
                                dynamic = bool(dynamic),
                            )

    src_engine  = _resolve_engine_path(export_out = export_out, model_path = model_path_obj)

    if engine_out is None:
        return src_engine

    # optional copy to explicit output path for downstream tooling
    engine_out_obj = Path(str(engine_out)).expanduser().resolve()
    engine_out_obj.parent.mkdir(parents = True, exist_ok = True)

    if src_engine.resolve() != engine_out_obj.resolve():
        shutil.copy2(src_engine, engine_out_obj)

    return engine_out_obj


def _parse_args() -> argparse.Namespace:
    # CLI is intentionally minimal for edge-device scripting
    parser = argparse.ArgumentParser(description = 'Export TensorRT engine from YOLO model path')
    parser.add_argument('--model-path', type = str, required = True, help = 'Path to source model (.pt or .onnx)')
    parser.add_argument('--engine-out', type = str, default = None, help = 'Optional explicit .engine output path')
    parser.add_argument('--imgsz', type = int, default = 1024)
    parser.add_argument('--device', type = str, default = '0')
    parser.add_argument('--half', action = 'store_true', help = 'Enable FP16 export')
    parser.add_argument('--dynamic', action = 'store_true', help = 'Enable dynamic shape export')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # execute export and print final resolved output path
    out = export_tensorrt_engine(
                                    model_path = args.model_path,
                                    engine_out = args.engine_out,
                                    imgsz = args.imgsz,
                                    device = args.device,
                                    half = bool(args.half),
                                    dynamic = bool(args.dynamic),
                                )
    print(f'Engine export complete: {out}')


if __name__ == '__main__':
    main()


# converting .pt to .engine with ultralytics export helper
# mae-bbox-export-trt \
#   --model-path /path/to/model.pt \
#   --engine-out /path/to/model.engine \
#   --imgsz 1024 \
#   --device 0 \
#   --half
# example:
# mae-bbox-export-trt \
#   --model-path /home/saa4743/agnc_repos/mae/bbox_inference/test_artifacts/best_fda.pt \
#   --engine-out /home/saa4743/agnc_repos/mae/bbox_inference/test_results/best_fda.engine \
#   --imgsz 1024 \
#   --device 0 \
#   --half


