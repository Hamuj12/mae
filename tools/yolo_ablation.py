#!/usr/bin/env python3
"""
YOLO-only ablation script:
- Compare N YOLO models passed via repeatable --model name=path
- Create dynamic side-by-side visualization grids (Input + N model panels)
- Compute unlabeled confidence stats over a configurable number of images
- Save a confidence histogram per model + summary.txt

Usage:
  PYTHONPATH=. python tools/ablation.py \
    --model baseline=/path/best.pt \
    --model fda=/path/fda.pt \
    --model fda_mix=/path/fda_mix.pt \
    --dataset /path/to/yolo_dataset_root \
    --output /path/to/outdir \
    --device cuda:0 \
    --conf 0.25 \
    --iou 0.45 \
    --n 12 \
    --eval-images 500 \
    --hist-bins 30
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from dual_yolo_mae import utils  # noqa: E402

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'ultralytics' package is required to run the ablation study. Install it with 'pip install ultralytics'."
    ) from exc


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# -----------------------------
# Args / config
# -----------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path


def parse_model_specs(specs: List[str]) -> List[ModelSpec]:
    """
    Parse repeated --model entries of the form:
      name=/path/to/weights.pt
    """
    out: List[ModelSpec] = []
    seen = set()
    for s in specs:
        if "=" not in s:
            raise ValueError(f"Bad --model '{s}'. Expected name=/path/to/weights.pt")
        name, p = s.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Bad --model '{s}': empty name")
        if name in seen:
            raise ValueError(f"Duplicate model name '{name}'. Names must be unique.")
        path = Path(p).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist for '{name}': {path}")
        out.append(ModelSpec(name=name, path=path))
        seen.add(name)
    if not out:
        raise ValueError("At least one --model must be provided.")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple YOLO models with unlabeled confidence stats.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeatable. Model spec: name=/path/to/weights.pt (e.g., --model baseline=... --model fda=...)",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Root of YOLO-format dataset")
    parser.add_argument("--output", type=str, default="ablation_outputs", help="Directory to store results")
    parser.add_argument("--n", type=int, default=8, help="Number of random test images for visualization plots")
    parser.add_argument(
        "--eval-images",
        type=int,
        default=500,
        help="Number of images to use for confidence statistics (unlabeled eval)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--hist-bins", type=int, default=30, help="Histogram bins for max_conf distribution")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device (e.g. 'cuda:0', 'cpu', or leave unset for auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for choosing sample/eval images",
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


# -----------------------------
# Dataset / predictions
# -----------------------------
def gather_images(dataset_root: Path) -> List[Path]:
    test_dir = dataset_root / "images" / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test image directory not found: {test_dir}")
    images = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    if not images:
        raise RuntimeError(f"No test images found in {test_dir}")
    return images


def predict_yolo(
    model: YOLO,
    image_path: Path,
    conf: float,
    iou: float,
    device_str: str,
) -> List[Dict]:
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

    preds: List[Dict] = []
    for i in range(len(classes)):
        preds.append(
            {
                "label": int(classes[i]),
                "score": float(confs[i]),
                "xyxy": xyxy[i].tolist(),
                "xywh": np.clip(xywhn[i], 0.0, 1.0).tolist(),
            }
        )
    return preds


# -----------------------------
# Visualization
# -----------------------------
def draw_detections(ax, image_array: np.ndarray, detections: List[Dict], title: str) -> None:
    ax.imshow(image_array)
    ax.set_title(title)
    ax.axis("off")
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x1,
            max(y1 - 5, 0),
            f"{det['score']:.2f}",
            fontsize=9,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, pad=2),
        )


def create_comparison_plot(
    image_path: Path,
    image_array: np.ndarray,
    per_model_preds: List[Tuple[str, List[Dict]]],
    output_dir: Path,
) -> None:
    # columns: Input + one per model
    ncols = 1 + len(per_model_preds)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(image_array)
    axes[0].set_title("Input")
    axes[0].axis("off")

    for idx, (name, preds) in enumerate(per_model_preds, start=1):
        draw_detections(axes[idx], image_array, preds, title=name)

    fig.tight_layout()
    save_path = output_dir / f"{image_path.stem}_comparison.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    utils.LOGGER.info("Saved comparison figure to %s", save_path)


# -----------------------------
# Unlabeled confidence stats
# -----------------------------
@dataclass
class ConfStats:
    n_images: int
    n_with_det: int
    mean_dets_per_image: float
    mean_max_conf: float
    median_max_conf: float
    p90_max_conf: float
    p99_max_conf: float


def compute_conf_stats(
    model: YOLO,
    image_paths: Sequence[Path],
    conf: float,
    iou: float,
    device_str: str,
) -> Tuple[ConfStats, List[float], List[int]]:
    """
    Returns:
      - summary stats
      - list of max_conf per image (0.0 if no dets)
      - list of det_count per image
    """
    max_confs: List[float] = []
    det_counts: List[int] = []

    for p in image_paths:
        preds = predict_yolo(model, p, conf=conf, iou=iou, device_str=device_str)
        if preds:
            scores = [d["score"] for d in preds]
            max_confs.append(float(max(scores)))
            det_counts.append(len(preds))
        else:
            max_confs.append(0.0)
            det_counts.append(0)

    arr = np.array(max_confs, dtype=np.float32)
    n_with_det = int(np.sum(arr > 0.0))
    stats = ConfStats(
        n_images=len(image_paths),
        n_with_det=n_with_det,
        mean_dets_per_image=float(np.mean(det_counts)),
        mean_max_conf=float(np.mean(arr)),
        median_max_conf=float(np.median(arr)),
        p90_max_conf=float(np.percentile(arr, 90)),
        p99_max_conf=float(np.percentile(arr, 99)),
    )
    return stats, max_confs, det_counts


def save_histogram(values: List[float], bins: int, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("max_conf per image")
    plt.ylabel("count")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    utils.setup_logging()

    dataset_root = Path(args.dataset)
    output_dir = utils.ensure_dir(args.output)

    device, device_str = resolve_device(args.device)
    utils.LOGGER.info("Device: %s (%s)", device, device_str)
    utils.LOGGER.info("Dataset root: %s", dataset_root)
    utils.LOGGER.info("Output dir: %s", output_dir)
    utils.LOGGER.info("conf=%.3f iou=%.3f", float(args.conf), float(args.iou))

    model_specs = parse_model_specs(args.model)

    # Load images once
    all_images = gather_images(dataset_root)

    rng = random.Random(int(args.seed))
    n_vis = min(int(args.n), len(all_images))
    n_eval = min(int(args.eval_images), len(all_images))

    vis_images = rng.sample(all_images, n_vis) if n_vis > 0 else []
    eval_images = rng.sample(all_images, n_eval) if n_eval > 0 else []

    utils.LOGGER.info("Loaded %d models", len(model_specs))
    utils.LOGGER.info("Visualization images: %d | Eval images: %d", len(vis_images), len(eval_images))

    # Load YOLO models
    models: List[Tuple[str, YOLO]] = []
    for spec in model_specs:
        utils.LOGGER.info("Loading model '%s' from %s", spec.name, spec.path)
        m = YOLO(str(spec.path))
        models.append((spec.name, m))

    # ----------------------
    # Visualization
    # ----------------------
    if vis_images:
        utils.LOGGER.info("Generating dynamic comparison plots...")
        for image_path in vis_images:
            with Image.open(image_path) as img:
                pil_image = img.convert("RGB")
            image_array = np.array(pil_image)

            per_model_preds: List[Tuple[str, List[Dict]]] = []
            for name, m in models:
                preds = predict_yolo(m, image_path, conf=float(args.conf), iou=float(args.iou), device_str=device_str)
                per_model_preds.append((name, preds))

            create_comparison_plot(
                image_path=image_path,
                image_array=image_array,
                per_model_preds=per_model_preds,
                output_dir=output_dir,
            )
            pil_image.close()

    # ----------------------
    # Unlabeled confidence stats
    # ----------------------
    summary_lines: List[str] = []
    summary_lines.append("YOLO Ablation (Unlabeled Confidence Stats)")
    summary_lines.append(f"dataset={dataset_root}")
    summary_lines.append(f"conf={float(args.conf):.3f} iou={float(args.iou):.3f}")
    summary_lines.append(f"eval_images={len(eval_images)}  hist_bins={int(args.hist_bins)}")
    summary_lines.append("")

    utils.LOGGER.info("Computing confidence stats (unlabeled)...")
    for name, m in models:
        stats, max_confs, det_counts = compute_conf_stats(
            m,
            eval_images,
            conf=float(args.conf),
            iou=float(args.iou),
            device_str=device_str,
        )

        # Save per-model histogram of max_conf
        hist_path = output_dir / f"hist_maxconf_{name}.png"
        save_histogram(
            max_confs,
            bins=int(args.hist_bins),
            title=f"max_conf distribution ({name})",
            out_path=hist_path,
        )

        # Save quick text block
        det_rate = 100.0 * stats.n_with_det / max(1, stats.n_images)
        block = [
            f"[{name}]",
            f"  images:              {stats.n_images}",
            f"  det_rate (max_conf>0): {det_rate:.1f}%",
            f"  mean dets/image:     {stats.mean_dets_per_image:.3f}",
            f"  max_conf mean/med:   {stats.mean_max_conf:.3f} / {stats.median_max_conf:.3f}",
            f"  max_conf p90/p99:    {stats.p90_max_conf:.3f} / {stats.p99_max_conf:.3f}",
            f"  hist:               {hist_path.name}",
            "",
        ]
        summary_lines.extend(block)

        utils.LOGGER.info(
            "%s | det_rate=%.1f%% mean_max=%.3f med_max=%.3f p90=%.3f p99=%.3f",
            name,
            det_rate,
            stats.mean_max_conf,
            stats.median_max_conf,
            stats.p90_max_conf,
            stats.p99_max_conf,
        )

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    utils.LOGGER.info("Wrote summary to %s", summary_path)

    utils.LOGGER.info("Done.")


if __name__ == "__main__":
    main()

# Example:
# PYTHONPATH=. python tools/yolo_ablation.py \
#   --model baseline=/home/hm25936/mae/runs/yolov8_baseline/baseline/weights/best.pt \
#   --model fda=/home/hm25936/mae/runs/yolov8_fda/baseline/weights/best.pt \
#   --model fda_mix=/home/hm25936/mae/runs/yolov8_fda_mix/baseline/weights/best.pt \
#   --dataset /home/hm25936/datasets_for_yolo/lab_images_6000 \
#   --n 12 \
#   --eval-images 500 \
#   --hist-bins 30 \
#   --output /home/hm25936/mae/yolo_ablation_runs/fda_mix_resize_100e_lab \
#   --device cuda:0

# PYTHONPATH=. python tools/yolo_ablation.py \
#   --model baseline=/home/hm25936/mae/runs/yolov8_baseline/baseline/weights/best.pt \
#   --model fda=/home/hm25936/mae/runs/yolov8_fda/baseline/weights/best.pt \
#   --model fda_mix=/home/hm25936/mae/runs/yolov8_fda_mix/baseline/weights/best.pt \
#   --dataset /home/hm25936/datasets_for_yolo/soho \
#   --n 12 \
#   --eval-images 500 \
#   --hist-bins 30 \
#   --output /home/hm25936/mae/yolo_ablation_runs/fda_mix_resize_100e_real \
#   --device cuda:0