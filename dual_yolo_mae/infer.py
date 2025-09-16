"""Inference helper for the dual-backbone YOLO + MAE model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from dual_yolo_mae.model import DualBackboneYOLO
from dual_yolo_mae import utils


def resolve_device(requested: Optional[str]) -> torch.device:
    """Resolve the runtime device, respecting CUDA availability."""

    if requested is None or requested.lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(requested)
    except (TypeError, RuntimeError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid device specification: {requested}") from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        utils.LOGGER.warning(
            "CUDA requested for inference but is unavailable. Falling back to CPU."
        )
        return torch.device("cpu")

    return device


def load_image(path: Path, img_size: int) -> tuple[torch.Tensor, Image.Image, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size  # (width, height)
    resized = image.resize((img_size, img_size))
    tensor = transforms.ToTensor()(resized)
    return tensor, image, (original_size[1], original_size[0])


def draw_detections(image: Image.Image, detections: List[dict], class_names: List[str] | None = None) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        score = det["score"]
        label_idx = det["label"]
        label_name = class_names[label_idx] if class_names and 0 <= label_idx < len(class_names) else str(label_idx)
        caption = f"{label_name}: {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text_bbox = draw.textbbox((x1, y1), caption)
        draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill="red")
        draw.text((x1, y1), caption, fill="white")
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the Dual YOLO + MAE model")
    parser.add_argument("--config", type=str, default="dual_yolo_mae/config.yaml", help="Path to config YAML")
    parser.add_argument("--weights", type=str, required=True, help="Path to model checkpoint (.pt or .ckpt)")
    parser.add_argument("--input", type=str, required=True, help="Image file or directory")
    parser.add_argument("--output", type=str, default="predictions", help="Directory to store annotated results")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold override for NMS")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (e.g. 'cuda:0', 'cpu', or leave unset for auto)",
    )
    return parser.parse_args()


def gather_images(path: Path) -> List[Path]:
    if path.is_dir():
        images = sorted([p for p in path.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".bmp"}])
        if not images:
            raise FileNotFoundError(f"No images found in directory {path}")
        return images
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Input path {path} does not exist")


def load_weights(model: DualBackboneYOLO, weights_path: Path) -> None:
    state = torch.load(weights_path, map_location="cpu")
    if "state_dict" in state:
        state = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
    model.load_state_dict(state, strict=False)


def main() -> None:
    args = parse_args()
    utils.setup_logging()
    config = utils.load_config(args.config)

    inference_cfg = config.get("inference", {})

    model = DualBackboneYOLO(config)
    load_weights(model, Path(args.weights))
    model.eval()

    conf_threshold = (
        float(args.conf)
        if args.conf is not None
        else float(inference_cfg.get("conf_threshold", 0.25))
    )
    iou_threshold = (
        float(args.iou)
        if args.iou is not None
        else float(inference_cfg.get("iou_threshold", 0.45))
    )

    device = resolve_device(args.device or inference_cfg.get("device"))
    utils.LOGGER.info(
        "Running inference on device %s with conf=%.2f, iou=%.2f",
        device,
        conf_threshold,
        iou_threshold,
    )
    model.to(device)

    img_size = int(config.get("model", {}).get("input_size", 640))
    class_names = config.get("dataset", {}).get("class_names")

    image_paths = gather_images(Path(args.input))
    output_dir = utils.ensure_dir(args.output)

    utils.LOGGER.info("Running inference on %d images", len(image_paths))

    with torch.no_grad():
        for image_path in image_paths:
            tensor, original_image, original_hw = load_image(image_path, img_size)
            tensor = tensor.unsqueeze(0).to(device)
            preds = model(tensor)
            detections = model.decode_predictions(
                preds,
                image_sizes=[original_hw],
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )[0]

            annotated = draw_detections(original_image.copy(), detections, class_names)
            save_path = output_dir / f"{image_path.stem}_detections.png"
            annotated.save(save_path)
            utils.LOGGER.info("Saved predictions for %s to %s", image_path.name, save_path)


if __name__ == "__main__":
    main()
