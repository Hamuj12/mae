#!/usr/bin/env python3
"""
YOLO-only ablation script (multi-model, multi-dataset):

- Compare N YOLO models passed via repeatable --model name=path
- Evaluate M YOLO-root datasets passed via repeatable --dataset name=path
- Dynamic side-by-side visualization grids: Input + one panel per model
- Unlabeled confidence stats using TOP-1 detection only:
    * top1_det_rate@conf  (fraction of images with top1_conf >= conf_threshold)
    * top1_conf mean/median/p90/p99/std
    * mean ± 2σ and mean ± 3σ (simple spread summary)
- Histogram of top1_conf per model per dataset
- summary.txt written per dataset + a global summary.txt

Optional (for later):
- If you provide --neg-dataset name=path (background-only frames), the script will
  also compute "FP proxy rate" as fraction of negatives with top1_conf >= conf.

All datasets are assumed to be YOLO-root style with images/test.

Example:
  PYTHONPATH=. python tools/yolo_ablation.py \
    --model baseline=/path/best.pt \
    --model fda=/path/fda.pt \
    --model fda_mix=/path/fda_mix.pt \
    --dataset lab=/path/to/lab_yolo_root \
    --dataset real=/path/to/real_yolo_root \
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
from typing import Dict, List, Sequence, Tuple, Optional

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


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path


def _parse_named_path_specs(specs: List[str], flag_name: str) -> List[Tuple[str, Path]]:
    """
    Parse repeated args of the form:
      name=/path/to/thing
    """
    out: List[Tuple[str, Path]] = []
    seen = set()
    for s in specs:
        if "=" not in s:
            raise ValueError(f"Bad {flag_name} '{s}'. Expected name=/path")
        name, p = s.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Bad {flag_name} '{s}': empty name")
        if name in seen:
            raise ValueError(f"Duplicate name '{name}' in {flag_name}. Names must be unique.")
        path = Path(p).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{flag_name} path does not exist for '{name}': {path}")
        out.append((name, path))
        seen.add(name)
    return out


def parse_model_specs(specs: List[str]) -> List[ModelSpec]:
    pairs = _parse_named_path_specs(specs, "--model")
    if not pairs:
        raise ValueError("At least one --model must be provided.")
    return [ModelSpec(name=n, path=p) for n, p in pairs]


def parse_dataset_specs(specs: List[str], flag: str) -> List[DatasetSpec]:
    pairs = _parse_named_path_specs(specs, flag)
    if not pairs:
        raise ValueError(f"At least one {flag} must be provided.")
    return [DatasetSpec(name=n, root=p) for n, p in pairs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple YOLO models over multiple datasets (unlabeled stats).")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeatable. Model spec: name=/path/to/weights.pt",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Repeatable. Dataset spec: name=/path/to/yolo_root (expects images/test)",
    )
    parser.add_argument(
        "--neg-dataset",
        action="append",
        default=None,
        help="Optional repeatable. Negative/background-only dataset spec: name=/path/to/yolo_root (expects images/test)",
    )
    parser.add_argument("--output", type=str, default="ablation_outputs", help="Directory to store results")
    parser.add_argument("--n", type=int, default=8, help="Number of random test images per dataset for visualization")
    parser.add_argument(
        "--eval-images",
        type=int,
        default=500,
        help="Number of images per dataset used for confidence statistics",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--hist-bins", type=int, default=30, help="Histogram bins for top1_conf distribution")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device (e.g. 'cuda:0', 'cpu', or leave unset for auto)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for choosing sample/eval images")
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
    confs = boxes.conf.cpu().numpy()

    preds: List[Dict] = []
    for i in range(len(confs)):
        preds.append({"score": float(confs[i]), "xyxy": xyxy[i].tolist()})
    return preds


def top1_detection(preds: List[Dict]) -> Optional[Dict]:
    if not preds:
        return None
    return max(preds, key=lambda d: float(d["score"]))


# -----------------------------
# Visualization
# -----------------------------
def draw_top1(ax, image_array: np.ndarray, top1: Optional[Dict], title: str) -> None:
    ax.imshow(image_array)
    ax.set_title(title)
    ax.axis("off")
    if top1 is None:
        return
    x1, y1, x2, y2 = top1["xyxy"]
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
    ax.add_patch(rect)
    ax.text(
        x1,
        max(y1 - 5, 0),
        f"{top1['score']:.2f}",
        fontsize=9,
        color="white",
        bbox=dict(facecolor="black", alpha=0.6, pad=2),
    )


def create_comparison_plot(
    image_path: Path,
    image_array: np.ndarray,
    per_model_top1: List[Tuple[str, Optional[Dict]]],
    output_dir: Path,
) -> None:
    # columns: Input + one per model
    ncols = 1 + len(per_model_top1)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(image_array)
    axes[0].set_title("Input")
    axes[0].axis("off")

    for idx, (name, top1) in enumerate(per_model_top1, start=1):
        draw_top1(axes[idx], image_array, top1, title=name)

    fig.tight_layout()
    save_path = output_dir / f"{image_path.stem}_comparison.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    utils.LOGGER.info("Saved comparison figure to %s", save_path)


# -----------------------------
# Confidence stats (unlabeled)
# -----------------------------
@dataclass
class ConfStats:
    n_images: int
    n_top1_ge_conf: int
    det_rate_at_conf: float
    mean_top1_conf: float
    median_top1_conf: float
    std_top1_conf: float
    p90_top1_conf: float
    p99_top1_conf: float
    mean_minus_2sigma: float
    mean_plus_2sigma: float
    mean_minus_3sigma: float
    mean_plus_3sigma: float


def compute_conf_stats_top1(
    model: YOLO,
    image_paths: Sequence[Path],
    conf_threshold: float,
    iou: float,
    device_str: str,
) -> Tuple[ConfStats, List[float]]:
    """
    Computes TOP-1 confidence stats.
    For each image:
      - top1_conf = max detection confidence, or 0.0 if no detections
    Then:
      - det_rate_at_conf = fraction with top1_conf >= conf_threshold
    """
    top1_confs: List[float] = []

    for p in image_paths:
        preds = predict_yolo(model, p, conf=0.001, iou=iou, device_str=device_str)
        # ^ Use a tiny conf here so we can compute top1_conf meaningfully; thresholding is applied later.
        #   This avoids "conf" changing the distribution by truncation.
        t1 = top1_detection(preds)
        top1_confs.append(float(t1["score"]) if t1 is not None else 0.0)

    arr = np.array(top1_confs, dtype=np.float32)
    n = int(arr.size)
    n_ge = int(np.sum(arr >= float(conf_threshold)))
    mean_v = float(np.mean(arr)) if n else float("nan")
    std_v = float(np.std(arr)) if n else float("nan")

    stats = ConfStats(
        n_images=n,
        n_top1_ge_conf=n_ge,
        det_rate_at_conf=(n_ge / n) if n else float("nan"),
        mean_top1_conf=mean_v,
        median_top1_conf=float(np.median(arr)) if n else float("nan"),
        std_top1_conf=std_v,
        p90_top1_conf=float(np.percentile(arr, 90)) if n else float("nan"),
        p99_top1_conf=float(np.percentile(arr, 99)) if n else float("nan"),
        mean_minus_2sigma=mean_v - 2.0 * std_v if n else float("nan"),
        mean_plus_2sigma=mean_v + 2.0 * std_v if n else float("nan"),
        mean_minus_3sigma=mean_v - 3.0 * std_v if n else float("nan"),
        mean_plus_3sigma=mean_v + 3.0 * std_v if n else float("nan"),
    )
    return stats, top1_confs


def save_histogram(values: List[float], bins: int, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("top1_conf per image")
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

    output_root = utils.ensure_dir(args.output)
    device, device_str = resolve_device(args.device)
    utils.LOGGER.info("Device: %s (%s)", device, device_str)

    model_specs = parse_model_specs(args.model)
    dataset_specs = parse_dataset_specs(args.dataset, "--dataset")
    neg_dataset_specs = parse_dataset_specs(args.neg_dataset, "--neg-dataset") if args.neg_dataset else []

    rng = random.Random(int(args.seed))

    # Load YOLO models once
    models: List[Tuple[str, YOLO]] = []
    for spec in model_specs:
        utils.LOGGER.info("Loading model '%s' from %s", spec.name, spec.path)
        models.append((spec.name, YOLO(str(spec.path))))

    global_lines: List[str] = []
    global_lines.append("YOLO Ablation (Top-1 Confidence Stats)")
    global_lines.append(f"device={device_str}")
    global_lines.append(f"conf_threshold={float(args.conf):.3f} iou={float(args.iou):.3f}")
    global_lines.append(f"n_vis_per_dataset={int(args.n)} eval_images_per_dataset={int(args.eval_images)} bins={int(args.hist_bins)}")
    global_lines.append("")
    global_lines.append("Datasets:")
    for ds in dataset_specs:
        global_lines.append(f"  - {ds.name}: {ds.root}")
    if neg_dataset_specs:
        global_lines.append("Negative datasets:")
        for ds in neg_dataset_specs:
            global_lines.append(f"  - {ds.name}: {ds.root}")
    global_lines.append("")

    # Evaluate each dataset independently; write outputs under output_root/<dataset_name>/
    for ds in dataset_specs:
        ds_out = output_root / ds.name
        ds_out.mkdir(parents=True, exist_ok=True)

        utils.LOGGER.info("==== Dataset: %s (%s) ====", ds.name, ds.root)
        all_images = gather_images(ds.root)
        n_vis = min(int(args.n), len(all_images))
        n_eval = min(int(args.eval_images), len(all_images))

        vis_images = rng.sample(all_images, n_vis) if n_vis > 0 else []
        eval_images = rng.sample(all_images, n_eval) if n_eval > 0 else []

        # --- Visualization
        if vis_images:
            utils.LOGGER.info("Generating comparison plots for dataset '%s' (n=%d)", ds.name, len(vis_images))
            for image_path in vis_images:
                with Image.open(image_path) as img:
                    pil = img.convert("RGB")
                image_array = np.array(pil)

                per_model_top1: List[Tuple[str, Optional[Dict]]] = []
                for name, m in models:
                    preds = predict_yolo(m, image_path, conf=0.001, iou=float(args.iou), device_str=device_str)
                    per_model_top1.append((name, top1_detection(preds)))

                create_comparison_plot(image_path, image_array, per_model_top1, output_dir=ds_out)
                pil.close()

        # --- Stats + histograms
        ds_lines: List[str] = []
        ds_lines.append(f"Dataset: {ds.name}")
        ds_lines.append(f"root: {ds.root}")
        ds_lines.append(f"eval_images: {len(eval_images)}")
        ds_lines.append(f"conf_threshold (for det_rate): {float(args.conf):.3f}")
        ds_lines.append(f"iou: {float(args.iou):.3f}")
        ds_lines.append("")

        for name, m in models:
            stats, top1_confs = compute_conf_stats_top1(
                m,
                eval_images,
                conf_threshold=float(args.conf),
                iou=float(args.iou),
                device_str=device_str,
            )

            hist_path = ds_out / f"hist_top1conf_{name}.png"
            save_histogram(
                top1_confs,
                bins=int(args.hist_bins),
                title=f"top1_conf distribution ({name}) — {ds.name}",
                out_path=hist_path,
            )

            block = [
                f"[{name}]",
                f"  det_rate@conf:       {100.0 * stats.det_rate_at_conf:.1f}%  ({stats.n_top1_ge_conf}/{stats.n_images})",
                f"  top1_conf mean/med:  {stats.mean_top1_conf:.3f} / {stats.median_top1_conf:.3f}",
                f"  top1_conf std:       {stats.std_top1_conf:.3f}",
                f"  top1_conf p90/p99:   {stats.p90_top1_conf:.3f} / {stats.p99_top1_conf:.3f}",
                f"  mean ± 2σ:           [{stats.mean_minus_2sigma:.3f}, {stats.mean_plus_2sigma:.3f}]",
                f"  mean ± 3σ:           [{stats.mean_minus_3sigma:.3f}, {stats.mean_plus_3sigma:.3f}]",
                f"  hist:                {hist_path.name}",
                "",
            ]
            ds_lines.extend(block)

        ds_summary_path = ds_out / "summary.txt"
        ds_summary_path.write_text("\n".join(ds_lines), encoding="utf-8")
        utils.LOGGER.info("Wrote dataset summary to %s", ds_summary_path)

        # Append to global summary
        global_lines.append(f"==== {ds.name} ====")
        global_lines.extend(ds_lines[4:])  # skip repeated header lines for brevity

    # Optional negative datasets: compute FP proxy rate per model
    if neg_dataset_specs:
        global_lines.append("")
        global_lines.append("==== Negative datasets (FP proxy rates) ====")
        global_lines.append("Interpretation: fraction of negative images with top1_conf >= conf_threshold")
        global_lines.append("")

        for nds in neg_dataset_specs:
            nds_out = output_root / f"neg_{nds.name}"
            nds_out.mkdir(parents=True, exist_ok=True)

            utils.LOGGER.info("==== Negative Dataset: %s (%s) ====", nds.name, nds.root)
            all_images = gather_images(nds.root)
            n_eval = min(int(args.eval_images), len(all_images))
            eval_images = rng.sample(all_images, n_eval) if n_eval > 0 else []

            neg_lines: List[str] = []
            neg_lines.append(f"Negative dataset: {nds.name}")
            neg_lines.append(f"root: {nds.root}")
            neg_lines.append(f"eval_images: {len(eval_images)}")
            neg_lines.append(f"conf_threshold: {float(args.conf):.3f}")
            neg_lines.append("")

            for name, m in models:
                stats, top1_confs = compute_conf_stats_top1(
                    m,
                    eval_images,
                    conf_threshold=float(args.conf),
                    iou=float(args.iou),
                    device_str=device_str,
                )
                fp_rate = stats.det_rate_at_conf  # on negatives, det_rate is FP proxy

                hist_path = nds_out / f"hist_top1conf_{name}.png"
                save_histogram(
                    top1_confs,
                    bins=int(args.hist_bins),
                    title=f"top1_conf distribution ({name}) — NEG:{nds.name}",
                    out_path=hist_path,
                )

                neg_lines.extend(
                    [
                        f"[{name}]",
                        f"  FP_proxy_rate@conf:  {100.0 * fp_rate:.1f}%  ({stats.n_top1_ge_conf}/{stats.n_images})",
                        f"  top1_conf mean/med:  {stats.mean_top1_conf:.3f} / {stats.median_top1_conf:.3f}",
                        f"  top1_conf p90/p99:   {stats.p90_top1_conf:.3f} / {stats.p99_top1_conf:.3f}",
                        f"  hist:                {hist_path.name}",
                        "",
                    ]
                )

            (nds_out / "summary.txt").write_text("\n".join(neg_lines), encoding="utf-8")
            global_lines.append(f"---- NEG:{nds.name} ----")
            global_lines.extend(neg_lines[5:])

    global_summary = output_root / "summary.txt"
    global_summary.write_text("\n".join(global_lines), encoding="utf-8")
    utils.LOGGER.info("Wrote global summary to %s", global_summary)
    utils.LOGGER.info("Done.")


if __name__ == "__main__":
    main()

# Example:
# PYTHONPATH=. python tools/yolo_ablation.py \
#   --model baseline=/home/hm25936/mae/runs/yolov8_baseline/baseline/weights/best.pt \
#   --model fda=/home/hm25936/mae/runs/yolov8_fda/baseline/weights/best.pt \
#   --model fda_mix=/home/hm25936/mae/runs/yolov8_fda_mix/baseline/weights/best.pt \
#   --dataset lab=/home/hm25936/datasets_for_yolo/lab_images_6000 \
#   --dataset real=/home/hm25936/datasets_for_yolo/soho \
#   --output /home/hm25936/mae/yolo_ablation_runs/compare_models_multi_ds \
#   --device cuda:0 \
#   --conf 0.25 \
#   --iou 0.45 \
#   --n 12 \
#   --eval-images 500 \
#   --hist-bins 30
#
# Optional negatives later:
#   --neg-dataset empty=/home/.../empty_frames_yolo_root