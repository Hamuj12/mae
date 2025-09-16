#!/usr/bin/env python3
"""Minimal training entry point for an unmodified YOLOv8n baseline."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict

from dual_yolo_mae import utils


try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "The 'ultralytics' package is required to train the baseline model. "
        "Install it with 'pip install ultralytics'."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8n baseline model")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset YAML file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Training image size")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (e.g. 'cuda:0', 'cpu', or leave unset for Ultralytics defaults)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/yolov8_baseline",
        help="Directory where training artifacts and final weights are stored",
    )
    return parser.parse_args()


def collect_train_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    train_kwargs: Dict[str, Any] = {
        "data": args.data,
        "epochs": int(args.epochs),
        "batch": int(args.batch_size),
        "imgsz": int(args.img_size),
        "project": str(args.output),
        "name": "baseline",
        "exist_ok": True,
    }
    if args.device:
        train_kwargs["device"] = args.device
    return train_kwargs


def resolve_weights_path(output_dir: Path) -> Path:
    """Locate the weight checkpoint produced by Ultralytics training."""
    trainer_save_dir = output_dir / "baseline"
    weights_dir = trainer_save_dir / "weights"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"
    if best_path.exists():
        return best_path
    if last_path.exists():
        return last_path
    raise FileNotFoundError(
        f"Expected Ultralytics to save weights under {weights_dir}, but no checkpoints were found."
    )


def main() -> None:
    args = parse_args()
    utils.setup_logging()

    output_dir = utils.ensure_dir(args.output)
    utils.LOGGER.info("Training baseline YOLOv8n with outputs in %s", output_dir)
    utils.LOGGER.info("Using dataset YAML at %s", args.data)

    model = YOLO("yolov8n.pt")
    train_kwargs = collect_train_kwargs(args)
    utils.LOGGER.info(
        "Calling YOLO.train with arguments: %s",
        {k: v for k, v in train_kwargs.items() if k not in {"data"}},
    )
    model.train(**train_kwargs)

    try:
        weights_src = resolve_weights_path(output_dir)
    except FileNotFoundError as exc:
        utils.LOGGER.error("Unable to locate training weights: %s", exc)
        raise

    final_weights = output_dir / "yolov8n_baseline.pt"
    shutil.copy2(weights_src, final_weights)
    utils.LOGGER.info("Copied final baseline weights to %s", final_weights)


if __name__ == "__main__":
    main()
