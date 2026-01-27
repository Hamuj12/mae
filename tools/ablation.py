"""Ablation study comparing YOLOv8 baseline vs FDA-trained YOLOv8 vs dual-backbone variant."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from dual_yolo_mae import utils  # noqa: E402
from dual_yolo_mae.model import DualBackboneYOLO  # noqa: E402

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "The 'ultralytics' package is required to run the ablation study. Install it with 'pip install ultralytics'."
    ) from exc


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TO_TENSOR = transforms.ToTensor()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare YOLOv8 baseline vs FDA-YOLO vs dual-backbone model"
    )
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline YOLOv8 weights (.pt)")
    parser.add_argument(
        "--fda",
        type=str,
        default=None,
        help="Optional: Path to YOLOv8 weights trained on synthetic+FDA dataset (.pt)",
    )
    parser.add_argument("--modified", type=str, required=True, help="Path to dual-backbone weights (.ckpt/.pt)")
    parser.add_argument("--config", type=str, required=True, help="Dual-backbone configuration YAML")
    parser.add_argument("--dataset", type=str, required=True, help="Root of YOLO-format dataset")
    parser.add_argument("--n", type=int, default=8, help="Number of random test images for visualization")
    parser.add_argument("--output", type=str, default="ablation_outputs", help="Directory to store results")
    parser.add_argument("--conf", type=float, default=None, help="Optional override for confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="Optional override for IoU threshold")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device (e.g. 'cuda:0', 'cpu', or leave unset for automatic selection)",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> Tuple[torch.device, str]:
    if requested is None or requested.lower() == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        try:
            device = torch.device(requested)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"Invalid device specification: {requested}") from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            utils.LOGGER.warning("CUDA requested but is unavailable. Falling back to CPU.")
            device = torch.device("cpu")

    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        device_str = f"{device.type}:{index}"
    else:
        device_str = device.type
    return device, device_str


def gather_images(root: Path) -> List[Path]:
    test_dir = root / "images" / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test image directory not found: {test_dir}")
    images = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    if not images:
        raise RuntimeError(f"No test images found in {test_dir}")
    return images


def load_ground_truth(label_path: Path) -> List[Dict[str, Sequence[float]]]:
    boxes: List[Dict[str, Sequence[float]]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Malformed label in {label_path} on line {line_num}: expected 5 values")
            cls_id = int(float(parts[0]))
            x_c, y_c, w, h = map(float, parts[1:])
            boxes.append({"label": cls_id, "xywh": [x_c, y_c, w, h]})
    return boxes


def xywh_to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x_c, y_c, w, h = box
    x1 = max(0.0, x_c - w / 2.0)
    y1 = max(0.0, y_c - h / 2.0)
    x2 = min(1.0, x_c + w / 2.0)
    y2 = min(1.0, y_c + h / 2.0)
    return x1, y1, x2, y2


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def match_and_error(preds: List[Dict], gts: List[Dict]) -> Tuple[float, int]:
    total_sq = 0.0
    components = 0
    used_preds: set[int] = set()

    for gt in gts:
        gt_label = gt["label"]
        gt_box = gt["xywh"]
        best_idx = None
        best_iou = 0.0
        for idx, pred in enumerate(preds):
            if idx in used_preds:
                continue
            if int(pred["label"]) != gt_label:
                continue
            iou = compute_iou(pred["xywh"], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx is not None:
            pred_box = preds[best_idx]["xywh"]
            diff = np.array(gt_box) - np.array(pred_box)
            total_sq += float(np.dot(diff, diff))
            components += 4
            used_preds.add(best_idx)
        else:
            diff = np.array(gt_box)
            total_sq += float(np.dot(diff, diff))
            components += 4

    return total_sq, components


def predict_yolo(
    model: YOLO,
    image_path: Path,
    conf: float,
    iou: float,
    device_str: str | None,
) -> List[Dict]:
    """
    Run Ultralytics YOLO predict and return a standardized list of detections.
    Uses result.boxes.xywhn/conf/cls which are part of Ultralytics Results API.
    """
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        device=device_str,
        save=False,
        verbose=False,
    )
    if not results:
        return []

    result = results[0]
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    xywhn = boxes.xywhn.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    predictions: List[Dict] = []
    for idx in range(len(classes)):
        xywh = np.clip(xywhn[idx], 0.0, 1.0)
        predictions.append(
            {
                "label": int(classes[idx]),
                "score": float(confs[idx]),
                "xyxy": xyxy[idx].tolist(),
                "xywh": xywh.tolist(),
            }
        )
    return predictions


def predict_modified(
    model: DualBackboneYOLO,
    image_path: Path,
    device: torch.device,
    img_size: int,
    conf: float,
    iou: float,
    pil_image: Image.Image | None = None,
) -> List[Dict]:
    image = pil_image if pil_image is not None else Image.open(image_path).convert("RGB")
    original_w, original_h = image.size
    resized = image.resize((img_size, img_size))
    tensor = TO_TENSOR(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor)
        detections = model.decode_predictions(
            preds,
            image_sizes=[(original_h, original_w)],
            conf_threshold=conf,
            iou_threshold=iou,
        )[0]

    formatted: List[Dict] = []
    for det in detections:
        x1, y1, x2, y2 = map(float, det["box"])
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        x_c = (x1 + width / 2.0) / original_w
        y_c = (y1 + height / 2.0) / original_h
        w_norm = min(1.0, max(0.0, width / original_w))
        h_norm = min(1.0, max(0.0, height / original_h))
        x_c = min(1.0, max(0.0, x_c))
        y_c = min(1.0, max(0.0, y_c))
        xywh = [x_c, y_c, w_norm, h_norm]
        formatted.append(
            {
                "label": int(det["label"]),
                "score": float(det["score"]),
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "xywh": xywh,
            }
        )
    return formatted


def load_dual_weights(model: DualBackboneYOLO, weights_path: Path) -> None:
    state = torch.load(weights_path, map_location="cpu")
    if "state_dict" in state:
        state = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
    model.load_state_dict(state, strict=False)


def resolve_class_names(config: Dict, dataset_root: Path) -> List[str]:
    class_names = config.get("dataset", {}).get("class_names")
    if class_names:
        return [str(name) for name in class_names]

    for candidate in ("data.yaml", "dataset.yaml", "dataset_fda.yaml"):
        candidate_path = dataset_root / candidate
        if candidate_path.exists():
            data_cfg = utils.load_config(candidate_path)
            names = data_cfg.get("names")
            if isinstance(names, dict):
                try:
                    ordered = sorted(names.keys(), key=lambda k: int(k))
                except ValueError:
                    ordered = sorted(names.keys())
                return [str(names[key]) for key in ordered]
            if isinstance(names, (list, tuple)):
                return [str(name) for name in names]

    return []


def draw_detections(ax, image_array: np.ndarray, detections: List[Dict], class_names: List[str], title: str) -> None:
    ax.imshow(image_array)
    ax.set_title(title)
    ax.axis("off")

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

        label_idx = int(det["label"])
        label_name = (
            class_names[label_idx]
            if class_names and 0 <= label_idx < len(class_names)
            else str(label_idx)
        )
        caption = f"{label_name} {det['score']:.2f}"
        ax.text(
            x1,
            max(y1 - 5, 0),
            caption,
            fontsize=9,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, pad=2),
        )


def create_comparison_plot(
    image_path: Path,
    image_array: np.ndarray,
    baseline_preds: List[Dict],
    fda_preds: Optional[List[Dict]],
    modified_preds: List[Dict],
    class_names: List[str],
    output_dir: Path,
) -> None:
    has_fda = fda_preds is not None
    ncols = 4 if has_fda else 3

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if ncols == 3:
        ax0, ax1, ax2 = axes
        ax0.imshow(image_array)
        ax0.set_title("Input")
        ax0.axis("off")
        draw_detections(ax1, image_array, baseline_preds, class_names, "YOLOv8 Baseline")
        draw_detections(ax2, image_array, modified_preds, class_names, "Dual-Backbone YOLO")
    else:
        ax0, ax1, ax2, ax3 = axes
        ax0.imshow(image_array)
        ax0.set_title("Input")
        ax0.axis("off")
        draw_detections(ax1, image_array, baseline_preds, class_names, "YOLOv8 Baseline")
        draw_detections(ax2, image_array, fda_preds or [], class_names, "YOLOv8 (FDA-trained)")
        draw_detections(ax3, image_array, modified_preds, class_names, "Dual-Backbone YOLO")

    fig.tight_layout()
    save_path = output_dir / f"{image_path.stem}_comparison.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    utils.LOGGER.info("Saved comparison figure to %s", save_path)


def evaluate_dataset(
    image_paths: Sequence[Path],
    labels_root: Path,
    predictor: Callable[[Path], List[Dict]],
) -> Tuple[float, int]:
    total_error = 0.0
    total_components = 0
    for image_path in image_paths:
        label_path = labels_root / f"{image_path.stem}.txt"
        if not label_path.exists():
            utils.LOGGER.warning("Missing label file for %s. Skipping.", image_path.name)
            continue
        gts = load_ground_truth(label_path)
        preds = predictor(image_path)
        error, components = match_and_error(preds, gts)
        total_error += error
        total_components += components
    return total_error, total_components


def main() -> None:
    args = parse_args()
    utils.setup_logging()

    dataset_root = Path(args.dataset)
    output_dir = utils.ensure_dir(args.output)
    config = utils.load_config(args.config)

    utils.LOGGER.info("Dataset root: %s", dataset_root)
    utils.LOGGER.info("Outputs will be saved to %s", output_dir)

    device, device_str = resolve_device(args.device or config.get("inference", {}).get("device"))
    inference_cfg = config.get("inference", {})
    conf_threshold = float(args.conf if args.conf is not None else inference_cfg.get("conf_threshold", 0.25))
    iou_threshold = float(args.iou if args.iou is not None else inference_cfg.get("iou_threshold", 0.45))

    utils.LOGGER.info("Using device %s with conf=%.2f and iou=%.2f", device, conf_threshold, iou_threshold)

    class_names = resolve_class_names(config, dataset_root)
    if class_names:
        utils.LOGGER.info("Resolved %d class names", len(class_names))
    else:
        utils.LOGGER.info("No class names found; falling back to numeric labels")

    # ----------------------
    # Load models
    # ----------------------
    utils.LOGGER.info("Loading baseline YOLO weights from %s", args.baseline)
    baseline_model = YOLO(args.baseline)
    if class_names:
        mapping = {idx: name for idx, name in enumerate(class_names)}
        baseline_model.model.names = mapping

    fda_model = None
    if args.fda:
        utils.LOGGER.info("Loading FDA YOLO weights from %s", args.fda)
        fda_model = YOLO(args.fda)
        if class_names:
            mapping = {idx: name for idx, name in enumerate(class_names)}
            fda_model.model.names = mapping

    utils.LOGGER.info("Loading dual-backbone weights from %s", args.modified)
    modified_model = DualBackboneYOLO(config)
    load_dual_weights(modified_model, Path(args.modified))
    modified_model.to(device)
    modified_model.eval()

    img_size = int(config.get("model", {}).get("input_size", 640))
    all_images = gather_images(dataset_root)

    random.seed(42)
    sample_count = min(int(args.n), len(all_images))
    sampled_images = random.sample(all_images, sample_count)
    utils.LOGGER.info("Generating visual comparisons for %d images", sample_count)

    # ----------------------
    # Predictor wrappers
    # ----------------------
    def baseline_predictor(path: Path) -> List[Dict]:
        return predict_yolo(baseline_model, path, conf_threshold, iou_threshold, device_str)

    def fda_predictor(path: Path) -> List[Dict]:
        assert fda_model is not None
        return predict_yolo(fda_model, path, conf_threshold, iou_threshold, device_str)

    def modified_predictor(path: Path, pil_image: Image.Image | None = None) -> List[Dict]:
        return predict_modified(
            modified_model,
            path,
            device,
            img_size,
            conf_threshold,
            iou_threshold,
            pil_image=pil_image,
        )

    # ----------------------
    # Visualization
    # ----------------------
    for image_path in sampled_images:
        with Image.open(image_path) as img:
            pil_image = img.convert("RGB")
        image_array = np.array(pil_image)

        baseline_preds = baseline_predictor(image_path)
        fda_preds = fda_predictor(image_path) if fda_model is not None else None
        modified_preds = modified_predictor(image_path, pil_image=pil_image)

        create_comparison_plot(
            image_path=image_path,
            image_array=image_array,
            baseline_preds=baseline_preds,
            fda_preds=fda_preds,
            modified_preds=modified_preds,
            class_names=class_names,
            output_dir=output_dir,
        )
        pil_image.close()

    # ----------------------
    # Metric evaluation
    # ----------------------
    labels_root = dataset_root / "labels" / "test"
    if not labels_root.exists():
        raise FileNotFoundError(f"Test label directory not found: {labels_root}")

    utils.LOGGER.info("Evaluating models across %d test images", len(all_images))

    baseline_error, baseline_components = evaluate_dataset(all_images, labels_root, baseline_predictor)
    baseline_mse = baseline_error / baseline_components if baseline_components else float("nan")
    utils.LOGGER.info("Baseline YOLO average box MSE: %.6f", baseline_mse)

    fda_mse = None
    if fda_model is not None:
        fda_error, fda_components = evaluate_dataset(all_images, labels_root, fda_predictor)
        fda_mse = fda_error / fda_components if fda_components else float("nan")
        utils.LOGGER.info("FDA YOLO average box MSE: %.6f", float(fda_mse))

    modified_error, modified_components = evaluate_dataset(
        all_images, labels_root, lambda p: modified_predictor(p, pil_image=None)
    )
    modified_mse = modified_error / modified_components if modified_components else float("nan")
    utils.LOGGER.info("Dual-backbone average box MSE: %.6f", modified_mse)

    summary_path = output_dir / "mse_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Baseline YOLO MSE: {baseline_mse:.6f}\n")
        if fda_mse is not None:
            f.write(f"FDA YOLO MSE: {float(fda_mse):.6f}\n")
        f.write(f"Dual-backbone MSE: {modified_mse:.6f}\n")

    utils.LOGGER.info("Wrote MSE summary to %s", summary_path)


if __name__ == "__main__":
    main()


# PYTHONPATH=. python tools/ablation.py --baseline /home/hm25936/mae/runs/yolov8_baseline/baseline/weights/best.pt --modified /home/hm25936/mae/outputs/sweeps/phase1_v2/p1v2_e100_b24_lr0p0005_wd0010_T1p0/checkpoints/phase1-epoch=49.ckpt --config /home/hm25936/mae/configs/phase1_template.yaml --dataset /home/hm25936/datasets_for_yolo/midLighting_rmag_5m_to_100m --n 15 --output /home/hm25936/mae/ablation_results --device 'cuda:0'
# PYTHONPATH=. python tools/ablation.py --baseline /home/hm25936/mae/runs/yolov8_baseline/baseline/weights/best.pt --modified /home/hm25936/mae/outputs/sweeps/phase1_v2/p1v2_e100_b24_lr0p0005_wd0010_T1p0/checkpoints/phase1-epoch=49.ckpt --config /home/hm25936/mae/configs/phase1_template.yaml --dataset /home/hm25936/datasets_for_yolo/run_K --n 15 --output /home/hm25936/mae/ablation_results_real --device 'cuda:0'
# PYTHONPATH=. python tools/ablation.py \
#   --baseline /home/hm25936/mae/runs/yolov8_baseline/baseline/weights/best.pt \
#   --fda /home/hm25936/mae/runs/yolov8_fda/baseline/weights/best.pt \
#   --modified /home/hm25936/mae/outputs/sweeps/phase1_v2/p1v2_e100_b24_lr0p0005_wd0010_T1p0/checkpoints/phase1-epoch=49.ckpt \
#   --config /home/hm25936/mae/configs/phase1_template.yaml \
#   --dataset /home/hm25936/datasets_for_yolo/lab_images_6000 \
#   --n 15 \
#   --output /home/hm25936/mae/ablation_results_fda \
#   --device cuda:0