#!/usr/bin/env python3
"""Inference utility for the plain YOLOv8 baseline model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from dual_yolo_mae import utils

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "The 'ultralytics' package is required for inference. Install it with 'pip install ultralytics'."
    ) from exc

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a YOLOv8 baseline model")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained YOLOv8 weights (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML or root directory with metadata")
    parser.add_argument("--input", type=str, required=True, help="Image file or directory for inference")
    parser.add_argument("--output", type=str, default="baseline_predictions", help="Directory to store outputs")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device for inference (e.g. 'cuda:0', 'cpu', or leave unset for auto)",
    )
    return parser.parse_args()


def load_class_names(data_arg: str) -> List[str]:
    """Load class name metadata from a dataset YAML or directory."""

    path = Path(data_arg)
    data_cfg = None
    if path.is_file():
        data_cfg = utils.load_config(path)
    elif path.is_dir():
        for candidate in ("data.yaml", "dataset.yaml"):
            candidate_path = path / candidate
            if candidate_path.exists():
                data_cfg = utils.load_config(candidate_path)
                break
    if not data_cfg:
        utils.LOGGER.warning("Could not locate class names metadata for %s", data_arg)
        return []

    names = data_cfg.get("names")
    if isinstance(names, dict):
        try:
            ordered_keys = sorted(names.keys(), key=lambda k: int(k))
        except ValueError:
            ordered_keys = sorted(names.keys())
        return [str(names[key]) for key in ordered_keys]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]

    utils.LOGGER.warning("Dataset metadata did not contain a 'names' entry: %s", data_arg)
    return []


def gather_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        images = sorted([p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
        if not images:
            raise FileNotFoundError(f"No images found in directory {path}")
        return images
    raise FileNotFoundError(f"Input path {path} does not exist")


def save_annotation(result, image_path: Path, output_dir: Path) -> None:
    array = result.plot()  # BGR numpy array from Ultralytics
    if array.ndim != 3:
        raise ValueError("Unexpected result array shape from Ultralytics plot()")
    rgb_array = array[..., ::-1]
    annotated = Image.fromarray(np.ascontiguousarray(rgb_array))
    save_path = output_dir / f"{image_path.stem}_pred.png"
    annotated.save(save_path)
    utils.LOGGER.info("Saved predictions for %s to %s", image_path.name, save_path)


def main() -> None:
    args = parse_args()
    utils.setup_logging()

    class_names = load_class_names(args.data)
    output_dir = utils.ensure_dir(args.output)
    utils.LOGGER.info("Loading weights from %s", args.weights)
    utils.LOGGER.info("Saving annotated predictions to %s", output_dir)

    model = YOLO(args.weights)
    if class_names:
        name_mapping = {idx: name for idx, name in enumerate(class_names)}
        model.model.names = name_mapping
        model.names = name_mapping
        utils.LOGGER.info("Loaded %d class names", len(class_names))
    else:
        utils.LOGGER.info("Using default class names embedded in the model weights")

    image_paths = gather_images(Path(args.input))
    utils.LOGGER.info("Running inference on %d images from %s", len(image_paths), args.input)

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            conf=float(args.conf),
            iou=float(args.iou),
            device=args.device,
            save=False,
            verbose=False,
        )
        if not results:
            utils.LOGGER.warning("Model returned no results for %s", image_path)
            continue
        save_annotation(results[0], image_path, output_dir)


if __name__ == "__main__":
    main()
