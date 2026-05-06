#!/usr/bin/env python3
"""
YOLO-only ablation script (multi-model, multi-dataset).

Primary goals:
- Compare YOLO models passed via repeatable --model name=path
- Evaluate YOLO-root datasets passed via repeatable --dataset name=path
- Preserve the original unlabeled top-1 confidence statistics
- Produce slide-ready summaries and figures in addition to the original text dumps
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from dual_yolo_mae import utils  # noqa: E402

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'ultralytics' package is required to run the ablation study. Install it with 'pip install ultralytics'."
    ) from exc


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

PREFERRED_MODEL_ORDER = [
    "baseline",
    "fda",
    "fda_mix",
    "odad_baseline",
    "odad_1500",
    "odad_3000",
    "odad_4500",
    "odad_final",
]
ODAD_PROGRESS_ORDER = ["odad_baseline", "odad_1500", "odad_3000", "odad_4500", "odad_final"]
DEFAULT_SLIDE_SUBSET = ["baseline", "fda", "fda_mix", "odad_final"]

MODEL_DISPLAY_NAMES = {
    "baseline": "Baseline",
    "fda": "FDA",
    "fda_mix": "FDA-mix",
    "odad_baseline": "ODAD base",
    "odad_1500": "ODAD 1.5k",
    "odad_3000": "ODAD 3.0k",
    "odad_4500": "ODAD 4.5k",
    "odad_final": "ODAD final",
}

MODEL_COLOR_OVERRIDES = {
    "baseline": "#4C566A",
    "fda": "#2563EB",
    "fda_mix": "#059669",
    "odad_baseline": "#D97706",
    "odad_1500": "#F59E0B",
    "odad_3000": "#FBBF24",
    "odad_4500": "#FCD34D",
    "odad_final": "#B91C1C",
}
FALLBACK_COLORS = [
    "#0F766E",
    "#7C3AED",
    "#EA580C",
    "#1D4ED8",
    "#BE123C",
    "#65A30D",
    "#C2410C",
    "#0EA5E9",
]

FIG_DPI = 300

PAIRED_FAILURE_CATEGORIES = [
    "fda_mix_good_odad_bad",
    "odad_good_fda_mix_bad",
    "baseline_bad_odad_good",
    "odad_regression_final_bad_earlier_good",
    "odad_high_conf_wrong_or_weird",
    "odad_progression_success",
]


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


@dataclass
class EvalResult:
    dataset_name: str
    dataset_label: str
    dataset_root: Path
    model_name: str
    model_label: str
    stats: ConfStats
    top1_confs: List[float]
    hist_path: Path


@dataclass
class Top1PredictionRecord:
    image_path: Path
    image_index: int
    image_width: int
    image_height: int
    model_name: str
    top1_conf: float
    top1_xyxy: Optional[Tuple[float, float, float, float]]
    top1_cls: Optional[int]
    has_detection: bool
    box_area_frac: float
    border_touching: bool
    tiny_box: bool
    large_box: bool


@dataclass
class LabelRecord:
    image_path: Path
    gt_boxes_xyxy: List[Tuple[float, float, float, float]]
    gt_cls: List[int]


@dataclass
class LabelAwarePredictionRecord:
    top1_iou: float
    top1_center_error_frac: float
    top1_correct_at_iou: bool
    high_conf_wrong: bool
    false_negative_proxy: bool


@dataclass
class TemporalPredictionRecord:
    prev_iou: float
    center_shift_frac: float
    area_ratio: float
    confidence_delta: float
    box_jump: bool
    confidence_collapse: bool


def _parse_named_path_specs(specs: List[str], flag_name: str) -> List[Tuple[str, Path]]:
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
    ordered = sort_named_items([ModelSpec(name=n, path=p) for n, p in pairs], key_fn=lambda s: s.name)
    return ordered


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
    parser.add_argument(
        "--slide-subset-model",
        action="append",
        default=None,
        help="Optional repeatable model names used for compact slide plots. Defaults to baseline/fda/fda_mix/odad_final when present.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for choosing sample/eval images")
    parser.add_argument(
        "--reliability-mode",
        action="store_true",
        help="Enable reliability metrics, temporal metrics, paired failure analysis, and qualitative galleries.",
    )
    parser.add_argument(
        "--ordered-eval",
        action="store_true",
        help="Use images in sorted sequence order instead of random sampling; needed for temporal reliability.",
    )
    parser.add_argument(
        "--use-labels",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Whether to compute label-aware correctness using YOLO labels if available.",
    )
    parser.add_argument("--gallery-max", type=int, default=24, help="Maximum examples per qualitative gallery category.")
    parser.add_argument(
        "--gallery-models",
        type=str,
        default="baseline,fda_mix,odad_1500,odad_3000,odad_4500,odad_final",
        help="Comma-separated model names to include in qualitative gallery strips.",
    )
    parser.add_argument(
        "--good-conf",
        type=float,
        default=0.75,
        help="Confidence threshold for proxy good detections in unlabeled gallery selection.",
    )
    parser.add_argument(
        "--bad-conf",
        type=float,
        default=0.25,
        help="Confidence threshold for proxy bad detections / misses.",
    )
    parser.add_argument(
        "--correct-iou",
        type=float,
        default=0.50,
        help="IoU threshold for label-aware correct top-1 detection.",
    )
    parser.add_argument(
        "--bad-iou",
        type=float,
        default=0.20,
        help="IoU threshold below which a prediction is considered badly localized for failure galleries.",
    )
    parser.add_argument(
        "--large-box-frac",
        type=float,
        default=0.30,
        help="Box area fraction above which a prediction is considered unusually large.",
    )
    parser.add_argument(
        "--tiny-box-frac",
        type=float,
        default=0.001,
        help="Box area fraction below which a prediction is considered unusually tiny.",
    )
    parser.add_argument(
        "--border-margin-frac",
        type=float,
        default=0.02,
        help="Margin used to flag border-touching boxes.",
    )
    parser.add_argument(
        "--box-jump-center-frac",
        type=float,
        default=0.20,
        help="Normalized center shift threshold for temporal box-jump events.",
    )
    parser.add_argument(
        "--box-jump-area-ratio",
        type=float,
        default=2.5,
        help="Symmetric area ratio threshold for temporal box-size jump events.",
    )
    return parser.parse_args()


def configure_slide_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.2,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.8,
            "lines.markersize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 10.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


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


def sort_named_items(items: Sequence, key_fn) -> List:
    preferred_index = {name: idx for idx, name in enumerate(PREFERRED_MODEL_ORDER)}
    return sorted(
        items,
        key=lambda item: (
            preferred_index.get(key_fn(item), len(preferred_index) + 100),
            str(key_fn(item)),
        ),
    )


def display_name(raw_name: str) -> str:
    if raw_name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[raw_name]
    text = raw_name.replace("_", " ").replace("-", " ").strip()
    return " ".join(part.capitalize() for part in text.split())


def dataset_display_name(raw_name: str) -> str:
    text = raw_name.replace("_", " ").replace("-", " ").strip()
    return " ".join(part.capitalize() for part in text.split())


def build_model_color_map(model_names: Sequence[str]) -> Dict[str, str]:
    colors: Dict[str, str] = {}
    fallback_idx = 0
    for name in model_names:
        if name in MODEL_COLOR_OVERRIDES:
            colors[name] = MODEL_COLOR_OVERRIDES[name]
        else:
            colors[name] = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx += 1
    return colors


def resolve_slide_subset(model_names: Sequence[str], requested_subset: Optional[Sequence[str]]) -> List[str]:
    if requested_subset:
        subset = [name for name in requested_subset if name in model_names]
        missing = [name for name in requested_subset if name not in model_names]
        if missing:
            utils.LOGGER.warning("Ignoring missing --slide-subset-model names: %s", ", ".join(missing))
        if len(subset) >= 2:
            return subset
    default_subset = [name for name in DEFAULT_SLIDE_SUBSET if name in model_names]
    if len(default_subset) >= 2:
        return default_subset
    return list(model_names[: min(4, len(model_names))])


def metric_from_stats(stats: ConfStats, metric_name: str) -> float:
    return float(getattr(stats, metric_name))


def format_float(value: Optional[float], digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def format_signed(value: Optional[float], digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:+.{digits}f}"


def format_percent_pp(value: Optional[float], digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:+.{digits}f} pp"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def parse_csv_names(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def sanitize_filename(text: str, max_len: int = 80) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)[:max_len].strip("_") or "item"


def prediction_is_weird(record: Top1PredictionRecord) -> bool:
    return bool(record.tiny_box or record.large_box or record.border_touching)


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float32)
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_median(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float32)
    return float(np.median(arr)) if arr.size else float("nan")


def safe_rate(flags: Sequence[bool]) -> float:
    return float(np.mean(np.asarray(flags, dtype=np.float32))) if flags else float("nan")


def box_area_xyxy(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def clipped_box_area_frac(box: Tuple[float, float, float, float], image_width: int, image_height: int) -> float:
    x1, y1, x2, y2 = box
    cx1 = min(max(float(x1), 0.0), float(image_width))
    cy1 = min(max(float(y1), 0.0), float(image_height))
    cx2 = min(max(float(x2), 0.0), float(image_width))
    cy2 = min(max(float(y2), 0.0), float(image_height))
    denom = max(1.0, float(image_width) * float(image_height))
    return box_area_xyxy((cx1, cy1, cx2, cy2)) / denom


def box_iou_xyxy(
    a: Optional[Tuple[float, float, float, float]],
    b: Optional[Tuple[float, float, float, float]],
) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(float(ax1), float(bx1))
    iy1 = max(float(ay1), float(by1))
    ix2 = min(float(ax2), float(bx2))
    iy2 = min(float(ay2), float(by2))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area_xyxy(a) + box_area_xyxy(b) - inter
    return inter / union if union > 0.0 else 0.0


def center_error_frac(
    pred_box: Optional[Tuple[float, float, float, float]],
    gt_box: Optional[Tuple[float, float, float, float]],
    image_width: int,
    image_height: int,
) -> float:
    if pred_box is None or gt_box is None:
        return float("nan")
    px = 0.5 * (pred_box[0] + pred_box[2])
    py = 0.5 * (pred_box[1] + pred_box[3])
    gx = 0.5 * (gt_box[0] + gt_box[2])
    gy = 0.5 * (gt_box[1] + gt_box[3])
    diag = max(1.0, sqrt(float(image_width) ** 2 + float(image_height) ** 2))
    return sqrt((px - gx) ** 2 + (py - gy) ** 2) / diag


def center_shift_frac(prev: Top1PredictionRecord, curr: Top1PredictionRecord) -> float:
    if prev.top1_xyxy is None or curr.top1_xyxy is None:
        return float("nan")
    px = 0.5 * (prev.top1_xyxy[0] + prev.top1_xyxy[2])
    py = 0.5 * (prev.top1_xyxy[1] + prev.top1_xyxy[3])
    cx = 0.5 * (curr.top1_xyxy[0] + curr.top1_xyxy[2])
    cy = 0.5 * (curr.top1_xyxy[1] + curr.top1_xyxy[3])
    diag = max(1.0, sqrt(float(curr.image_width) ** 2 + float(curr.image_height) ** 2))
    return sqrt((cx - px) ** 2 + (cy - py) ** 2) / diag


def symmetric_area_ratio(prev_area: float, curr_area: float) -> float:
    if prev_area <= 0.0 or curr_area <= 0.0:
        return float("nan")
    return max(prev_area / curr_area, curr_area / prev_area)


def get_image_sizes(image_paths: Sequence[Path]) -> Dict[Path, Tuple[int, int]]:
    sizes: Dict[Path, Tuple[int, int]] = {}
    for p in image_paths:
        with Image.open(p) as img:
            sizes[p] = img.size
    return sizes


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
    classes = boxes.cls.cpu().numpy() if boxes.cls is not None else [None] * len(confs)

    preds: List[Dict] = []
    for i in range(len(confs)):
        cls_value = None if classes[i] is None else int(classes[i])
        preds.append({"score": float(confs[i]), "xyxy": xyxy[i].tolist(), "cls": cls_value})
    return preds


def top1_detection(preds: List[Dict]) -> Optional[Dict]:
    if not preds:
        return None
    return max(preds, key=lambda d: float(d["score"]))


def collect_top1_prediction_records(
    model_name: str,
    model: YOLO,
    image_paths: Sequence[Path],
    image_sizes: Mapping[Path, Tuple[int, int]],
    iou: float,
    device_str: str,
    large_box_frac: float,
    tiny_box_frac: float,
    border_margin_frac: float,
) -> List[Top1PredictionRecord]:
    records: List[Top1PredictionRecord] = []
    total = len(image_paths)
    for image_index, p in enumerate(image_paths):
        image_width, image_height = image_sizes[p]
        preds = predict_yolo(model, p, conf=0.001, iou=iou, device_str=device_str)
        t1 = top1_detection(preds)

        top1_conf = 0.0
        top1_xyxy: Optional[Tuple[float, float, float, float]] = None
        top1_cls: Optional[int] = None
        box_area_frac = 0.0
        border_touching = False
        tiny_box = False
        large_box = False

        if t1 is not None:
            top1_conf = float(t1["score"])
            top1_xyxy = tuple(float(v) for v in t1["xyxy"])
            top1_cls = None if t1.get("cls") is None else int(t1["cls"])
            box_area_frac = clipped_box_area_frac(top1_xyxy, image_width, image_height)
            margin_x = float(border_margin_frac) * float(image_width)
            margin_y = float(border_margin_frac) * float(image_height)
            x1, y1, x2, y2 = top1_xyxy
            border_touching = bool(
                x1 <= margin_x
                or y1 <= margin_y
                or x2 >= float(image_width) - margin_x
                or y2 >= float(image_height) - margin_y
            )
            tiny_box = bool(box_area_frac <= float(tiny_box_frac))
            large_box = bool(box_area_frac >= float(large_box_frac))

        records.append(
            Top1PredictionRecord(
                image_path=p,
                image_index=image_index,
                image_width=image_width,
                image_height=image_height,
                model_name=model_name,
                top1_conf=top1_conf,
                top1_xyxy=top1_xyxy,
                top1_cls=top1_cls,
                has_detection=t1 is not None,
                box_area_frac=box_area_frac,
                border_touching=border_touching,
                tiny_box=tiny_box,
                large_box=large_box,
            )
        )
        if total >= 100 and ((image_index + 1) % 100 == 0 or image_index + 1 == total):
            utils.LOGGER.info("Reliability predictions '%s': %d/%d", model_name, image_index + 1, total)
    return records


def compute_conf_stats_from_records(
    records: Sequence[Top1PredictionRecord],
    conf_threshold: float,
) -> Tuple[ConfStats, List[float]]:
    top1_confs = [float(record.top1_conf) for record in records]
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


def label_path_for_image(dataset_root: Path, image_path: Path) -> Path:
    images_root = dataset_root / "images" / "test"
    labels_root = dataset_root / "labels" / "test"
    try:
        rel_path = image_path.relative_to(images_root)
    except ValueError:
        rel_path = Path(image_path.name)
    return labels_root / rel_path.with_suffix(".txt")


def parse_yolo_label_file(
    image_path: Path,
    label_path: Path,
    image_width: int,
    image_height: int,
) -> LabelRecord:
    gt_boxes: List[Tuple[float, float, float, float]] = []
    gt_cls: List[int] = []
    if not label_path.exists():
        return LabelRecord(image_path=image_path, gt_boxes_xyxy=gt_boxes, gt_cls=gt_cls)

    for line_num, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            utils.LOGGER.warning("Skipping malformed YOLO label line %s:%d", label_path, line_num)
            continue
        try:
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = [float(v) for v in parts[1:5]]
        except ValueError:
            utils.LOGGER.warning("Skipping non-numeric YOLO label line %s:%d", label_path, line_num)
            continue
        x1 = (xc - bw / 2.0) * float(image_width)
        y1 = (yc - bh / 2.0) * float(image_height)
        x2 = (xc + bw / 2.0) * float(image_width)
        y2 = (yc + bh / 2.0) * float(image_height)
        gt_boxes.append(
            (
                min(max(x1, 0.0), float(image_width)),
                min(max(y1, 0.0), float(image_height)),
                min(max(x2, 0.0), float(image_width)),
                min(max(y2, 0.0), float(image_height)),
            )
        )
        gt_cls.append(cls_id)

    return LabelRecord(image_path=image_path, gt_boxes_xyxy=gt_boxes, gt_cls=gt_cls)


def load_label_records(
    dataset_root: Path,
    image_paths: Sequence[Path],
    image_sizes: Mapping[Path, Tuple[int, int]],
    use_labels: str,
) -> Tuple[bool, Dict[Path, LabelRecord], str]:
    if use_labels == "no":
        return False, {}, "Label-aware metrics disabled by --use-labels=no."

    labels_root = dataset_root / "labels" / "test"
    if not labels_root.exists():
        message = f"YOLO labels not found at {labels_root}; using proxy reliability metrics only."
        if use_labels == "yes":
            raise FileNotFoundError(message)
        return False, {}, message

    records: Dict[Path, LabelRecord] = {}
    label_files_found = 0
    total_boxes = 0
    for image_path in image_paths:
        image_width, image_height = image_sizes[image_path]
        label_path = label_path_for_image(dataset_root, image_path)
        if label_path.exists():
            label_files_found += 1
        record = parse_yolo_label_file(image_path, label_path, image_width, image_height)
        records[image_path] = record
        total_boxes += len(record.gt_boxes_xyxy)

    if label_files_found == 0 or total_boxes == 0:
        message = (
            f"Found {labels_root}, but no usable YOLO labels matched the evaluated images; "
            "using proxy reliability metrics only."
        )
        if use_labels == "yes":
            raise RuntimeError(message)
        return False, {}, message

    message = (
        f"Using YOLO labels from {labels_root} "
        f"({label_files_found}/{len(image_paths)} label files, {total_boxes} boxes). "
        "Top-1 predictions are matched to same-class GT boxes when class IDs are available."
    )
    return True, records, message


def compute_label_aware_records(
    records: Sequence[Top1PredictionRecord],
    label_records: Mapping[Path, LabelRecord],
    correct_iou: float,
    conf_threshold: float,
    good_conf: float,
) -> List[LabelAwarePredictionRecord]:
    out: List[LabelAwarePredictionRecord] = []
    for record in records:
        labels = label_records.get(record.image_path)
        if labels is None or not labels.gt_boxes_xyxy:
            out.append(
                LabelAwarePredictionRecord(
                    top1_iou=float("nan"),
                    top1_center_error_frac=float("nan"),
                    top1_correct_at_iou=False,
                    high_conf_wrong=False,
                    false_negative_proxy=False,
                )
            )
            continue

        best_iou = 0.0
        best_gt: Optional[Tuple[float, float, float, float]] = None
        if record.has_detection and record.top1_xyxy is not None:
            candidate_indices = list(range(len(labels.gt_boxes_xyxy)))
            if record.top1_cls is not None:
                same_class = [idx for idx, cls_id in enumerate(labels.gt_cls) if cls_id == record.top1_cls]
                candidate_indices = same_class
            if candidate_indices:
                for idx in candidate_indices:
                    gt_box = labels.gt_boxes_xyxy[idx]
                    iou_value = box_iou_xyxy(record.top1_xyxy, gt_box)
                    if iou_value > best_iou:
                        best_iou = iou_value
                        best_gt = gt_box
            elif labels.gt_boxes_xyxy:
                best_gt = max(labels.gt_boxes_xyxy, key=lambda gt: box_iou_xyxy(record.top1_xyxy, gt))
        elif labels.gt_boxes_xyxy:
            best_gt = labels.gt_boxes_xyxy[0]

        center_error = center_error_frac(record.top1_xyxy, best_gt, record.image_width, record.image_height)
        if not np.isfinite(center_error) and labels.gt_boxes_xyxy:
            center_error = 1.0
        correct = bool(best_iou >= float(correct_iou))
        out.append(
            LabelAwarePredictionRecord(
                top1_iou=best_iou,
                top1_center_error_frac=center_error,
                top1_correct_at_iou=correct,
                high_conf_wrong=bool(record.top1_conf >= float(good_conf) and not correct),
                false_negative_proxy=bool(not (record.top1_conf >= float(conf_threshold) and correct)),
            )
        )
    return out


# -----------------------------
# Visualization
# -----------------------------
def draw_top1(ax, image_array: np.ndarray, top1: Optional[Dict], title: str) -> None:
    ax.imshow(image_array)
    ax.set_title(title, fontweight="semibold")
    ax.axis("off")
    if top1 is None:
        ax.text(
            0.03,
            0.05,
            "No detection",
            transform=ax.transAxes,
            fontsize=11,
            color="white",
            bbox=dict(facecolor="black", alpha=0.65, pad=3),
        )
        return

    x1, y1, x2, y2 = top1["xyxy"]
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3.0, edgecolor="#00C853", facecolor="none")
    ax.add_patch(rect)
    ax.text(
        x1,
        max(y1 - 8, 0),
        f"{top1['score']:.2f}",
        fontsize=11,
        color="white",
        bbox=dict(facecolor="black", alpha=0.65, pad=3),
    )


def create_comparison_plot(
    image_path: Path,
    image_array: np.ndarray,
    dataset_label: str,
    per_model_top1: List[Tuple[str, Optional[Dict]]],
    output_dir: Path,
) -> Path:
    n_panels = 1 + len(per_model_top1)
    if n_panels <= 4:
        ncols = n_panels
    elif n_panels <= 6:
        ncols = 3
    else:
        ncols = 4
    nrows = ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 4.5 * nrows),
        constrained_layout=True,
    )
    flat_axes = np.atleast_1d(axes).ravel()

    flat_axes[0].imshow(image_array)
    flat_axes[0].set_title("Input", fontweight="semibold")
    flat_axes[0].axis("off")

    for idx, (name, top1) in enumerate(per_model_top1, start=1):
        draw_top1(flat_axes[idx], image_array, top1, title=display_name(name))

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(f"{dataset_label}: {image_path.name}", fontsize=20, fontweight="bold")
    save_path = output_dir / f"qual_compare_{image_path.stem}.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    utils.LOGGER.info("Saved comparison figure to %s", save_path)
    return save_path


def save_histogram(
    values: List[float],
    bins: int,
    conf_threshold: float,
    title: str,
    out_path: Path,
) -> None:
    arr = np.asarray(values, dtype=np.float32)
    mean_v = float(np.mean(arr)) if arr.size else float("nan")
    median_v = float(np.median(arr)) if arr.size else float("nan")

    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    ax.hist(arr, bins=bins, color="#5B8FF9", alpha=0.9, edgecolor="white", linewidth=0.8)
    ax.axvline(conf_threshold, color="black", linestyle="--", linewidth=1.7)
    if np.isfinite(mean_v):
        ax.axvline(mean_v, color="#B91C1C", linestyle="-", linewidth=2.3, label=f"Mean {mean_v:.3f}")
    if np.isfinite(median_v):
        ax.axvline(median_v, color="#0F766E", linestyle="-.", linewidth=2.3, label=f"Median {median_v:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xlabel("Top-1 confidence")
    ax.set_ylabel("Images")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="upper left")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_cdf_plot(
    dataset_label: str,
    results: Sequence[EvalResult],
    conf_threshold: float,
    colors: Mapping[str, str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.0), constrained_layout=True)
    for result in results:
        arr = np.sort(np.asarray(result.top1_confs, dtype=np.float32))
        if arr.size == 0:
            continue
        y = np.arange(1, arr.size + 1, dtype=np.float32) / float(arr.size)
        ax.plot(arr, y, label=result.model_label, color=colors[result.model_name])

    ax.axvline(conf_threshold, color="black", linestyle="--", linewidth=1.7)
    ax.text(
        min(conf_threshold + 0.015, 0.92),
        0.07,
        rf"$\tau_c={conf_threshold:.2f}$",
        fontsize=12,
        color="black",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{dataset_label}: top-1 confidence CDF")
    ax.set_xlabel("Top-1 confidence")
    ax.set_ylabel("CDF")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="lower right", ncol=2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _set_metric_ylim(ax, values: Sequence[float], use_percent_axis: bool) -> None:
    valid = [float(v) for v in values if np.isfinite(v)]
    if not valid:
        return
    upper = max(valid)
    if use_percent_axis:
        ax.set_ylim(0.0, max(8.0, min(104.0, upper * 1.10 + 2.0)))
    else:
        if upper <= 1.0:
            ax.set_ylim(0.0, max(0.08, min(1.02, upper * 1.10 + 0.04)))
        else:
            ax.set_ylim(0.0, max(1.0, upper * 1.10 + 1.0))


def _annotate_bars(ax, bars, values: Sequence[float], use_percent_axis: bool) -> None:
    for bar, value in zip(bars, values):
        label = f"{value:.1f}%" if use_percent_axis else f"{value:.3f}"
        y_offset = 1.0 if use_percent_axis or value > 1.0 else 0.015
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=0,
        )


def save_grouped_metric_bar(
    dataset_specs: Sequence[DatasetSpec],
    results_by_dataset: Mapping[str, List[EvalResult]],
    model_names: Sequence[str],
    colors: Mapping[str, str],
    metric_name: str,
    title: str,
    ylabel: str,
    out_path: Path,
    use_percent_axis: bool = False,
) -> None:
    dataset_names = [ds.name for ds in dataset_specs]
    lookup = {ds_name: {result.model_name: result for result in results_by_dataset[ds_name]} for ds_name in dataset_names}

    fig, ax = plt.subplots(figsize=(12.0 if len(dataset_names) == 1 else 13.0, 7.2), constrained_layout=True)

    if len(dataset_names) == 1:
        ds_name = dataset_names[0]
        ordered_results = [lookup[ds_name][model_name] for model_name in model_names if model_name in lookup[ds_name]]
        x = np.arange(len(ordered_results), dtype=np.float32)
        values = []
        bar_colors = []
        labels = []
        for result in ordered_results:
            value = metric_from_stats(result.stats, metric_name)
            value = value * 100.0 if use_percent_axis else value
            values.append(value)
            bar_colors.append(colors[result.model_name])
            labels.append(result.model_label)

        bars = ax.bar(x, values, color=bar_colors, width=0.72)
        ax.set_xticks(x, labels, rotation=24, ha="right")
        ax.set_title(f"{results_by_dataset[ds_name][0].dataset_label}: {title}")
        _set_metric_ylim(ax, values, use_percent_axis=use_percent_axis)
        _annotate_bars(ax, bars, values, use_percent_axis=use_percent_axis)
    else:
        x = np.arange(len(dataset_names), dtype=np.float32)
        width = min(0.82 / max(len(model_names), 1), 0.16)
        total_width = width * len(model_names)
        for idx, model_name in enumerate(model_names):
            offsets = x - total_width / 2.0 + width / 2.0 + idx * width
            values = []
            for ds_name in dataset_names:
                value = metric_from_stats(lookup[ds_name][model_name].stats, metric_name)
                values.append(value * 100.0 if use_percent_axis else value)
            ax.bar(offsets, values, width=width, color=colors[model_name], label=display_name(model_name))

        ax.set_xticks(x, [results_by_dataset[ds_name][0].dataset_label for ds_name in dataset_names])
        ax.set_title(title)
        flat_values = []
        for ds_name in dataset_names:
            for model_name in model_names:
                value = metric_from_stats(lookup[ds_name][model_name].stats, metric_name)
                flat_values.append(value * 100.0 if use_percent_axis else value)
        _set_metric_ylim(ax, flat_values, use_percent_axis=use_percent_axis)
        legend_cols = min(4, max(1, ceil(len(model_names) / 2)))
        ax.legend(frameon=False, loc="upper center", ncol=legend_cols)

    ax.set_ylabel(ylabel)
    if use_percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ax.grid(axis="y", alpha=0.22)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_odad_checkpoint_trend(
    dataset_specs: Sequence[DatasetSpec],
    results_by_dataset: Mapping[str, List[EvalResult]],
    colors: Mapping[str, str],
    out_path: Path,
) -> Optional[Path]:
    dataset_names = [ds.name for ds in dataset_specs]
    lookup = {ds_name: {result.model_name: result for result in results_by_dataset[ds_name]} for ds_name in dataset_names}
    ordered_models = [name for name in ODAD_PROGRESS_ORDER if any(name in lookup[ds_name] for ds_name in dataset_names)]
    if len(ordered_models) < 2:
        return None

    metric_specs = [
        ("det_rate_at_conf", r"Det. rate @ $\tau_c$", True),
        ("mean_top1_conf", "Mean top-1 confidence", False),
        ("median_top1_conf", "Median top-1 confidence", False),
    ]
    nrows = len(dataset_names)
    fig, axes = plt.subplots(
        nrows,
        len(metric_specs),
        figsize=(6.2 * len(metric_specs), 4.7 * nrows),
        constrained_layout=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for row_idx, ds_name in enumerate(dataset_names):
        ds_lookup = lookup[ds_name]
        present_models = [name for name in ordered_models if name in ds_lookup]
        x = np.arange(len(present_models), dtype=np.float32)

        for col_idx, (metric_name, panel_title, use_percent_axis) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            values = []
            for name in present_models:
                value = metric_from_stats(ds_lookup[name].stats, metric_name)
                values.append(value * 100.0 if use_percent_axis else value)

            ax.plot(x, values, color="#334155", linewidth=2.2, alpha=0.95, zorder=1)
            ax.scatter(
                x,
                values,
                s=90,
                c=[colors[name] for name in present_models],
                edgecolors="white",
                linewidths=1.2,
                zorder=2,
            )
            for point_x, value in zip(x, values):
                label = f"{value:.1f}%" if use_percent_axis else f"{value:.3f}"
                y_offset = 1.2 if use_percent_axis else 0.02
                ax.text(point_x, value + y_offset, label, ha="center", va="bottom", fontsize=10)

            ax.set_xticks(x, [display_name(name) for name in present_models], rotation=22, ha="right")
            if row_idx == 0:
                ax.set_title(panel_title)
            if col_idx == 0:
                ax.text(
                    -0.34,
                    0.5,
                    results_by_dataset[ds_name][0].dataset_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=14,
                    fontweight="bold",
                )
            _set_metric_ylim(ax, values, use_percent_axis=use_percent_axis)
            if use_percent_axis:
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))

    fig.suptitle("ODAD checkpoint progression", fontsize=20, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_precision_style_rank_plot(
    dataset_specs: Sequence[DatasetSpec],
    results_by_dataset: Mapping[str, List[EvalResult]],
    subset_model_names: Sequence[str],
    colors: Mapping[str, str],
    conf_threshold: float,
    out_path: Path,
) -> Optional[Path]:
    if len(subset_model_names) < 2:
        return None

    dataset_names = [ds.name for ds in dataset_specs]
    lookup = {ds_name: {result.model_name: result for result in results_by_dataset[ds_name]} for ds_name in dataset_names}
    nrows = len(dataset_names)
    fig, axes = plt.subplots(nrows, 1, figsize=(11.0, 5.0 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, ds_name in zip(axes, dataset_names):
        ds_lookup = lookup[ds_name]
        for model_name in subset_model_names:
            if model_name not in ds_lookup:
                continue
            arr = np.sort(np.asarray(ds_lookup[model_name].top1_confs, dtype=np.float32))[::-1]
            if arr.size == 0:
                continue
            x = np.linspace(0.0, 100.0, arr.size, endpoint=True)
            ax.plot(x, arr, color=colors[model_name], label=display_name(model_name))

        ax.axhline(conf_threshold, color="black", linestyle="--", linewidth=1.6)
        ax.text(1.0, min(conf_threshold + 0.03, 0.96), rf"$\tau_c={conf_threshold:.2f}$", fontsize=11, color="black")
        ax.set_xlim(0.0, 100.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{results_by_dataset[ds_name][0].dataset_label}: sorted top-1 confidence")
        ax.set_xlabel("Image rank percentile")
        ax.set_ylabel("Top-1 confidence")
        ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Compact model ranking view", fontsize=20, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_grouped_row_metric_bar(
    dataset_specs: Sequence[DatasetSpec],
    rows: Sequence[Mapping[str, object]],
    model_names: Sequence[str],
    colors: Mapping[str, str],
    metric_name: str,
    title: str,
    ylabel: str,
    out_path: Path,
    use_percent_axis: bool = False,
) -> Optional[Path]:
    if not rows:
        return None
    dataset_names = [ds.name for ds in dataset_specs if any(row["dataset"] == ds.name for row in rows)]
    if not dataset_names:
        return None
    lookup = {
        ds_name: {str(row["model"]): row for row in rows if row["dataset"] == ds_name}
        for ds_name in dataset_names
    }

    fig, ax = plt.subplots(figsize=(12.0 if len(dataset_names) == 1 else 13.0, 7.2), constrained_layout=True)
    if len(dataset_names) == 1:
        ds_name = dataset_names[0]
        ordered_rows = [lookup[ds_name][name] for name in model_names if name in lookup[ds_name]]
        x = np.arange(len(ordered_rows), dtype=np.float32)
        values = []
        bar_colors = []
        labels = []
        for row in ordered_rows:
            value = safe_float(row.get(metric_name))
            values.append(value * 100.0 if use_percent_axis else value)
            bar_colors.append(colors[str(row["model"])])
            labels.append(str(row.get("model_label", display_name(str(row["model"])))))

        bars = ax.bar(x, values, color=bar_colors, width=0.72)
        ax.set_xticks(x, labels, rotation=24, ha="right")
        ax.set_title(f"{dataset_display_name(ds_name)}: {title}")
        _set_metric_ylim(ax, values, use_percent_axis=use_percent_axis)
        _annotate_bars(ax, bars, values, use_percent_axis=use_percent_axis)
    else:
        x = np.arange(len(dataset_names), dtype=np.float32)
        width = min(0.82 / max(len(model_names), 1), 0.16)
        total_width = width * len(model_names)
        for idx, model_name in enumerate(model_names):
            offsets = x - total_width / 2.0 + width / 2.0 + idx * width
            values = []
            for ds_name in dataset_names:
                row = lookup[ds_name].get(model_name)
                value = safe_float(row.get(metric_name)) if row is not None else float("nan")
                values.append(value * 100.0 if use_percent_axis else value)
            ax.bar(offsets, values, width=width, color=colors[model_name], label=display_name(model_name))
        ax.set_xticks(x, [dataset_display_name(ds_name) for ds_name in dataset_names])
        ax.set_title(title)
        flat_values = []
        for ds_name in dataset_names:
            for model_name in model_names:
                row = lookup[ds_name].get(model_name)
                if row is not None:
                    value = safe_float(row.get(metric_name))
                    flat_values.append(value * 100.0 if use_percent_axis else value)
        _set_metric_ylim(ax, flat_values, use_percent_axis=use_percent_axis)
        ax.legend(frameon=False, loc="upper center", ncol=min(4, max(1, ceil(len(model_names) / 2))))

    ax.set_ylabel(ylabel)
    if use_percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ax.grid(axis="y", alpha=0.22)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_checkpoint_reliability_trend(
    dataset_specs: Sequence[DatasetSpec],
    reliability_rows: Sequence[Mapping[str, object]],
    temporal_rows: Sequence[Mapping[str, object]],
    label_rows: Sequence[Mapping[str, object]],
    colors: Mapping[str, str],
    out_path: Path,
) -> Optional[Path]:
    if not reliability_rows:
        return None
    dataset_names = [ds.name for ds in dataset_specs if any(row["dataset"] == ds.name for row in reliability_rows)]
    reliability_lookup = {
        ds_name: {str(row["model"]): row for row in reliability_rows if row["dataset"] == ds_name}
        for ds_name in dataset_names
    }
    temporal_lookup = {
        ds_name: {str(row["model"]): row for row in temporal_rows if row["dataset"] == ds_name}
        for ds_name in dataset_names
    }
    label_lookup = {
        ds_name: {str(row["model"]): row for row in label_rows if row["dataset"] == ds_name}
        for ds_name in dataset_names
    }
    ordered_models = [
        name
        for name in ODAD_PROGRESS_ORDER
        if any(name in reliability_lookup.get(ds_name, {}) for ds_name in dataset_names)
    ]
    if len(ordered_models) < 2:
        return None

    metric_specs: List[Tuple[str, str, str, bool]] = [
        ("reliability", "det_rate_at_conf", r"Det. rate @ $\tau_c$", True),
    ]
    if label_rows:
        metric_specs.append(("label", "top1_correct_rate_at_iou", r"Correct @ IoU", True))
    metric_specs.extend(
        [
            ("reliability", "weird_box_rate", "Weird-box rate", True),
            ("temporal", "box_jump_rate", "Box-jump rate", True),
        ]
    )

    nrows = len(dataset_names)
    fig, axes = plt.subplots(
        nrows,
        len(metric_specs),
        figsize=(5.5 * len(metric_specs), 4.7 * nrows),
        constrained_layout=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for row_idx, ds_name in enumerate(dataset_names):
        present_models = [name for name in ordered_models if name in reliability_lookup[ds_name]]
        x = np.arange(len(present_models), dtype=np.float32)
        for col_idx, (source, metric_name, panel_title, use_percent_axis) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            values = []
            for name in present_models:
                source_lookup = {
                    "reliability": reliability_lookup,
                    "temporal": temporal_lookup,
                    "label": label_lookup,
                }[source]
                row = source_lookup.get(ds_name, {}).get(name)
                value = safe_float(row.get(metric_name)) if row is not None else float("nan")
                values.append(value * 100.0 if use_percent_axis else value)
            ax.plot(x, values, color="#334155", linewidth=2.2, alpha=0.95, zorder=1)
            ax.scatter(
                x,
                values,
                s=90,
                c=[colors[name] for name in present_models],
                edgecolors="white",
                linewidths=1.2,
                zorder=2,
            )
            for point_x, value in zip(x, values):
                if not np.isfinite(value):
                    continue
                label = f"{value:.1f}%" if use_percent_axis else f"{value:.3f}"
                y_offset = 1.2 if use_percent_axis else 0.02
                ax.text(point_x, value + y_offset, label, ha="center", va="bottom", fontsize=10)
            ax.set_xticks(x, [display_name(name) for name in present_models], rotation=22, ha="right")
            if row_idx == 0:
                ax.set_title(panel_title)
            if col_idx == 0:
                ax.text(
                    -0.34,
                    0.5,
                    dataset_display_name(ds_name),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=14,
                    fontweight="bold",
                )
            _set_metric_ylim(ax, values, use_percent_axis=use_percent_axis)
            if use_percent_axis:
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))

    fig.suptitle("ODAD checkpoint reliability progression", fontsize=20, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_gallery_font(size: int, bold: bool = False):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    fill: str,
    width: int,
    dash: int = 8,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if length <= 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pos = 0.0
    while pos < length:
        end_pos = min(pos + dash, length)
        if int(pos / dash) % 2 == 0:
            draw.line(
                (x1 + dx * pos, y1 + dy * pos, x1 + dx * end_pos, y1 + dy * end_pos),
                fill=fill,
                width=width,
            )
        pos += dash


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: Tuple[float, float, float, float],
    fill: str,
    width: int = 2,
    dash: int = 8,
) -> None:
    x1, y1, x2, y2 = box
    draw_dashed_line(draw, (x1, y1), (x2, y1), fill=fill, width=width, dash=dash)
    draw_dashed_line(draw, (x2, y1), (x2, y2), fill=fill, width=width, dash=dash)
    draw_dashed_line(draw, (x2, y2), (x1, y2), fill=fill, width=width, dash=dash)
    draw_dashed_line(draw, (x1, y2), (x1, y1), fill=fill, width=width, dash=dash)


def scaled_box(
    box: Tuple[float, float, float, float],
    scale: float,
    y_offset: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 * scale, y1 * scale + y_offset, x2 * scale, y2 * scale + y_offset)


def box_style_for_record(
    record: Top1PredictionRecord,
    label_aware: Optional[LabelAwarePredictionRecord],
    good_conf: float,
    bad_conf: float,
    bad_iou: float,
) -> str:
    if not record.has_detection:
        return "#DC2626"
    if label_aware is not None and np.isfinite(label_aware.top1_iou):
        if label_aware.top1_correct_at_iou:
            return "#16A34A"
        if label_aware.top1_iou < float(bad_iou) or label_aware.high_conf_wrong or prediction_is_weird(record):
            return "#DC2626"
        return "#2563EB"
    if prediction_is_weird(record) or record.top1_conf < float(bad_conf):
        return "#DC2626"
    if record.top1_conf >= float(good_conf):
        return "#16A34A"
    return "#2563EB"


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font,
    fill: str = "white",
    bg: str = "black",
) -> None:
    bbox = draw.textbbox(xy, text, font=font)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg)
    draw.text(xy, text, font=font, fill=fill)


def render_gallery_panel(
    image: Image.Image,
    title: str,
    record: Optional[Top1PredictionRecord],
    label_record: Optional[LabelRecord],
    label_aware: Optional[LabelAwarePredictionRecord],
    good_conf: float,
    bad_conf: float,
    bad_iou: float,
    is_input: bool = False,
) -> Image.Image:
    panel_w = 320
    header_h = 50
    image_w, image_h = image.size
    scale = panel_w / max(1.0, float(image_w))
    panel_img_h = max(1, int(round(float(image_h) * scale)))
    resized = image.resize((panel_w, panel_img_h), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (panel_w, header_h + panel_img_h), "#F8FAFC")
    panel.paste(resized, (0, header_h))
    draw = ImageDraw.Draw(panel)
    title_font = load_gallery_font(14, bold=True)
    small_font = load_gallery_font(12, bold=False)
    draw.rectangle((0, 0, panel_w, header_h), fill="#F1F5F9")
    draw.text((8, 5), title, font=title_font, fill="#0F172A")

    metric_text = "input"
    if record is not None and not is_input:
        if record.has_detection:
            metric_text = f"conf {record.top1_conf:.2f}"
            if label_aware is not None and np.isfinite(label_aware.top1_iou):
                metric_text += f" | IoU {label_aware.top1_iou:.2f}"
            if prediction_is_weird(record):
                metric_text += " | weird"
        else:
            metric_text = "no detection"
    draw.text((8, 27), metric_text, font=small_font, fill="#334155")

    if label_record is not None:
        for gt_box in label_record.gt_boxes_xyxy:
            draw_dashed_rectangle(draw, scaled_box(gt_box, scale, header_h), fill="#FACC15", width=2, dash=8)

    if record is not None and not is_input and record.top1_xyxy is not None:
        color = box_style_for_record(record, label_aware, good_conf=good_conf, bad_conf=bad_conf, bad_iou=bad_iou)
        xyxy = scaled_box(record.top1_xyxy, scale, header_h)
        draw.rectangle(xyxy, outline=color, width=4)
        label = f"{record.top1_conf:.2f}"
        if label_aware is not None and np.isfinite(label_aware.top1_iou):
            label += f" / {label_aware.top1_iou:.2f}"
        draw_text_box(draw, (xyxy[0] + 4, max(header_h + 4, xyxy[1] - 18)), label, small_font, bg=color)

    return panel


def save_gallery_strip(
    image_path: Path,
    category: str,
    reason: str,
    records_by_model: Mapping[str, Sequence[Top1PredictionRecord]],
    label_record: Optional[LabelRecord],
    label_aware_by_model: Mapping[str, Sequence[LabelAwarePredictionRecord]],
    gallery_model_names: Sequence[str],
    output_path: Path,
    good_conf: float,
    bad_conf: float,
    bad_iou: float,
) -> Path:
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    image_index = next(iter(records_by_model.values()))[0].image_index if records_by_model else 0
    for records in records_by_model.values():
        for record in records:
            if record.image_path == image_path:
                image_index = record.image_index
                break

    panels: List[Image.Image] = [
        render_gallery_panel(
            image=image,
            title="Input",
            record=None,
            label_record=label_record,
            label_aware=None,
            good_conf=good_conf,
            bad_conf=bad_conf,
            bad_iou=bad_iou,
            is_input=True,
        )
    ]

    for model_name in gallery_model_names:
        records = records_by_model.get(model_name)
        if records is None:
            continue
        record = next((r for r in records if r.image_path == image_path), None)
        if record is None:
            continue
        label_aware_records = label_aware_by_model.get(model_name)
        label_aware = None
        if label_aware_records is not None:
            label_aware = label_aware_records[record.image_index]
        panels.append(
            render_gallery_panel(
                image=image,
                title=display_name(model_name),
                record=record,
                label_record=label_record,
                label_aware=label_aware,
                good_conf=good_conf,
                bad_conf=bad_conf,
                bad_iou=bad_iou,
            )
        )

    title_h = 54
    width = sum(panel.width for panel in panels)
    height = title_h + max(panel.height for panel in panels)
    strip = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(strip)
    title_font = load_gallery_font(16, bold=True)
    small_font = load_gallery_font(12, bold=False)
    compact_reason = reason if len(reason) <= 180 else reason[:177] + "..."
    draw.text((10, 7), f"{category} | idx {image_index} | {image_path.name}", font=title_font, fill="#0F172A")
    draw.text((10, 30), compact_reason, font=small_font, fill="#334155")

    x_offset = 0
    for panel in panels:
        strip.paste(panel, (x_offset, title_h))
        x_offset += panel.width

    output_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(output_path)
    return output_path


def select_paired_failure_examples(
    dataset_name: str,
    image_paths: Sequence[Path],
    records_by_model: Mapping[str, Sequence[Top1PredictionRecord]],
    label_aware_by_model: Mapping[str, Sequence[LabelAwarePredictionRecord]],
    labels_available: bool,
    good_conf: float,
    bad_conf: float,
    correct_iou: float,
    bad_iou: float,
) -> Dict[str, List[Dict[str, object]]]:
    selected: Dict[str, List[Dict[str, object]]] = {category: [] for category in PAIRED_FAILURE_CATEGORIES}

    def has_model(name: str) -> bool:
        return name in records_by_model

    def rec(name: str, idx: int) -> Top1PredictionRecord:
        return records_by_model[name][idx]

    def label_aware(name: str, idx: int) -> Optional[LabelAwarePredictionRecord]:
        values = label_aware_by_model.get(name)
        return values[idx] if values is not None else None

    def finite_iou(name: str, idx: int) -> Optional[float]:
        la = label_aware(name, idx)
        if la is None or not np.isfinite(la.top1_iou):
            return None
        return float(la.top1_iou)

    def add(category: str, idx: int, reason: str, score: float) -> None:
        selected[category].append(
            {
                "dataset": dataset_name,
                "category": category,
                "image_path": image_paths[idx],
                "image_index": idx,
                "reason": reason,
                "score": score,
            }
        )

    earlier_odad = [name for name in ODAD_PROGRESS_ORDER if name != "odad_final" and has_model(name)]

    for idx, _image_path in enumerate(image_paths):
        if has_model("fda_mix") and has_model("odad_final"):
            fda_mix = rec("fda_mix", idx)
            odad_final = rec("odad_final", idx)
            fda_iou = finite_iou("fda_mix", idx) if labels_available else None
            odad_iou = finite_iou("odad_final", idx) if labels_available else None
            if fda_iou is not None and odad_iou is not None:
                if fda_iou >= correct_iou and odad_iou < bad_iou:
                    add(
                        "fda_mix_good_odad_bad",
                        idx,
                        f"FDA-mix IoU {fda_iou:.2f} >= {correct_iou:.2f}; ODAD final IoU {odad_iou:.2f} < {bad_iou:.2f}",
                        fda_iou - odad_iou,
                    )
            elif fda_mix.top1_conf >= good_conf and (odad_final.top1_conf < bad_conf or prediction_is_weird(odad_final)):
                add(
                    "fda_mix_good_odad_bad",
                    idx,
                    f"FDA-mix conf {fda_mix.top1_conf:.2f} >= {good_conf:.2f}; ODAD final conf {odad_final.top1_conf:.2f}"
                    f" or weird={int(prediction_is_weird(odad_final))}",
                    fda_mix.top1_conf - odad_final.top1_conf + (0.25 if prediction_is_weird(odad_final) else 0.0),
                )

            if fda_iou is not None and odad_iou is not None:
                if odad_iou >= correct_iou and fda_iou < bad_iou:
                    add(
                        "odad_good_fda_mix_bad",
                        idx,
                        f"ODAD final IoU {odad_iou:.2f} >= {correct_iou:.2f}; FDA-mix IoU {fda_iou:.2f} < {bad_iou:.2f}",
                        odad_iou - fda_iou,
                    )
            elif odad_final.top1_conf >= good_conf and fda_mix.top1_conf < 0.50:
                add(
                    "odad_good_fda_mix_bad",
                    idx,
                    f"ODAD final conf {odad_final.top1_conf:.2f} >= {good_conf:.2f}; FDA-mix conf {fda_mix.top1_conf:.2f} < 0.50",
                    odad_final.top1_conf - fda_mix.top1_conf,
                )

        if has_model("baseline") and has_model("odad_final"):
            baseline = rec("baseline", idx)
            odad_final = rec("odad_final", idx)
            baseline_iou = finite_iou("baseline", idx) if labels_available else None
            odad_iou = finite_iou("odad_final", idx) if labels_available else None
            if baseline_iou is not None and odad_iou is not None:
                if odad_iou >= correct_iou and baseline_iou < bad_iou:
                    add(
                        "baseline_bad_odad_good",
                        idx,
                        f"ODAD final IoU {odad_iou:.2f} >= {correct_iou:.2f}; baseline IoU {baseline_iou:.2f} < {bad_iou:.2f}",
                        odad_iou - baseline_iou,
                    )
            elif odad_final.top1_conf >= good_conf and baseline.top1_conf < bad_conf:
                add(
                    "baseline_bad_odad_good",
                    idx,
                    f"ODAD final conf {odad_final.top1_conf:.2f} >= {good_conf:.2f}; baseline conf {baseline.top1_conf:.2f} < {bad_conf:.2f}",
                    odad_final.top1_conf - baseline.top1_conf,
                )

        if has_model("odad_final") and earlier_odad:
            odad_final = rec("odad_final", idx)
            final_iou = finite_iou("odad_final", idx) if labels_available else None
            earlier_iou_pairs = [
                (name, finite_iou(name, idx))
                for name in earlier_odad
                if finite_iou(name, idx) is not None
            ]
            if final_iou is not None and earlier_iou_pairs:
                best_name, best_iou = max(earlier_iou_pairs, key=lambda item: float(item[1]))
                if best_iou is not None and best_iou >= correct_iou and final_iou < bad_iou:
                    add(
                        "odad_regression_final_bad_earlier_good",
                        idx,
                        f"{display_name(best_name)} IoU {best_iou:.2f} >= {correct_iou:.2f}; ODAD final IoU {final_iou:.2f} < {bad_iou:.2f}",
                        float(best_iou) - final_iou,
                    )
            else:
                best_name, best_record = max(
                    ((name, rec(name, idx)) for name in earlier_odad),
                    key=lambda item: item[1].top1_conf,
                )
                if best_record.top1_conf >= good_conf and odad_final.top1_conf < bad_conf:
                    add(
                        "odad_regression_final_bad_earlier_good",
                        idx,
                        f"{display_name(best_name)} conf {best_record.top1_conf:.2f} >= {good_conf:.2f}; ODAD final conf {odad_final.top1_conf:.2f} < {bad_conf:.2f}",
                        best_record.top1_conf - odad_final.top1_conf,
                    )

            if final_iou is not None:
                if odad_final.top1_conf >= good_conf and final_iou < bad_iou:
                    add(
                        "odad_high_conf_wrong_or_weird",
                        idx,
                        f"ODAD final conf {odad_final.top1_conf:.2f} >= {good_conf:.2f}; IoU {final_iou:.2f} < {bad_iou:.2f}",
                        odad_final.top1_conf + (bad_iou - final_iou),
                    )
            elif odad_final.top1_conf >= good_conf and prediction_is_weird(odad_final):
                add(
                    "odad_high_conf_wrong_or_weird",
                    idx,
                    f"ODAD final conf {odad_final.top1_conf:.2f} >= {good_conf:.2f}; weird box",
                    odad_final.top1_conf + 0.25,
                )

        if has_model("odad_1500") and has_model("odad_final"):
            odad_1500 = rec("odad_1500", idx)
            odad_final = rec("odad_final", idx)
            iou_1500 = finite_iou("odad_1500", idx) if labels_available else None
            final_iou = finite_iou("odad_final", idx) if labels_available else None
            if iou_1500 is not None and final_iou is not None:
                if iou_1500 < bad_iou and final_iou >= correct_iou:
                    add(
                        "odad_progression_success",
                        idx,
                        f"ODAD 1.5k IoU {iou_1500:.2f} < {bad_iou:.2f}; ODAD final IoU {final_iou:.2f} >= {correct_iou:.2f}",
                        final_iou - iou_1500,
                    )
            elif odad_1500.top1_conf < bad_conf and odad_final.top1_conf >= good_conf:
                add(
                    "odad_progression_success",
                    idx,
                    f"ODAD 1.5k conf {odad_1500.top1_conf:.2f} < {bad_conf:.2f}; ODAD final conf {odad_final.top1_conf:.2f} >= {good_conf:.2f}",
                    odad_final.top1_conf - odad_1500.top1_conf,
                )

    for category in PAIRED_FAILURE_CATEGORIES:
        selected[category].sort(key=lambda item: float(item["score"]), reverse=True)
    return selected


def save_paired_failure_galleries(
    dataset_name: str,
    dataset_label: str,
    selected_by_category: Mapping[str, Sequence[Mapping[str, object]]],
    records_by_model: Mapping[str, Sequence[Top1PredictionRecord]],
    label_records: Mapping[Path, LabelRecord],
    label_aware_by_model: Mapping[str, Sequence[LabelAwarePredictionRecord]],
    gallery_model_names: Sequence[str],
    output_root: Path,
    gallery_max: int,
    good_conf: float,
    bad_conf: float,
    bad_iou: float,
) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    gallery_root = output_root / "reliability_galleries"
    summary_rows: List[Dict[str, object]] = []
    index_rows_by_category: Dict[str, List[Dict[str, object]]] = {category: [] for category in PAIRED_FAILURE_CATEGORIES}
    index_models = []
    for name in list(gallery_model_names) + ["baseline", "fda_mix", "odad_1500", "odad_3000", "odad_4500", "odad_final"]:
        if name in records_by_model and name not in index_models:
            index_models.append(name)

    for category in PAIRED_FAILURE_CATEGORIES:
        category_dir = gallery_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        examples = list(selected_by_category.get(category, []))
        saved = 0
        for rank, example in enumerate(examples[: max(0, int(gallery_max))], start=1):
            image_path = Path(example["image_path"])
            file_stem = sanitize_filename(f"{dataset_name}_{rank:03d}_{image_path.stem}")
            save_path = category_dir / f"{file_stem}.png"
            label_record = label_records.get(image_path)
            save_gallery_strip(
                image_path=image_path,
                category=category,
                reason=str(example["reason"]),
                records_by_model=records_by_model,
                label_record=label_record,
                label_aware_by_model=label_aware_by_model,
                gallery_model_names=gallery_model_names,
                output_path=save_path,
                good_conf=good_conf,
                bad_conf=bad_conf,
                bad_iou=bad_iou,
            )
            saved += 1
            row: Dict[str, object] = {
                "dataset": dataset_name,
                "dataset_label": dataset_label,
                "category": category,
                "image_index": example["image_index"],
                "image_path": str(image_path),
                "reason": example["reason"],
                "score": f"{float(example['score']):.4f}",
                "gallery_path": str(save_path),
            }
            for model_name in index_models:
                record = records_by_model[model_name][int(example["image_index"])]
                row[f"{model_name}_conf"] = f"{record.top1_conf:.4f}"
                row[f"{model_name}_weird"] = int(prediction_is_weird(record))
                label_aware_records = label_aware_by_model.get(model_name)
                if label_aware_records is not None:
                    iou_value = label_aware_records[int(example["image_index"])].top1_iou
                    row[f"{model_name}_iou"] = "" if not np.isfinite(iou_value) else f"{iou_value:.4f}"
                else:
                    row[f"{model_name}_iou"] = ""
            index_rows_by_category[category].append(row)

        summary_rows.append(
            {
                "dataset": dataset_name,
                "dataset_label": dataset_label,
                "category": category,
                "n_selected": len(examples),
                "n_saved": saved,
                "gallery_dir": str(category_dir),
                "index_csv": str(category_dir / "index.csv"),
            }
        )

    return summary_rows, index_rows_by_category


def write_gallery_indices(
    gallery_index_rows_by_category: Mapping[str, Sequence[Mapping[str, object]]],
    output_root: Path,
) -> None:
    gallery_root = output_root / "reliability_galleries"
    for category in PAIRED_FAILURE_CATEGORIES:
        category_dir = gallery_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        rows = list(gallery_index_rows_by_category.get(category, []))
        if rows:
            fieldnames: List[str] = []
            for row in rows:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
        else:
            fieldnames = ["dataset", "category", "image_index", "image_path", "reason", "score", "gallery_path"]
        with (category_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


# -----------------------------
# Confidence stats (unlabeled)
# -----------------------------
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


def aggregate_box_reliability(
    dataset_name: str,
    dataset_label: str,
    model_name: str,
    model_label: str,
    records: Sequence[Top1PredictionRecord],
    conf_threshold: float,
    good_conf: float,
) -> Dict[str, object]:
    n = len(records)
    detected = [record for record in records if record.has_detection]
    area_values = [record.box_area_frac for record in detected]
    weird_flags = [prediction_is_weird(record) for record in records]
    high_conf_flags = [record.top1_conf >= float(good_conf) for record in records]
    high_conf_weird_flags = [
        record.top1_conf >= float(good_conf) and prediction_is_weird(record) for record in records
    ]

    return {
        "dataset": dataset_name,
        "dataset_label": dataset_label,
        "model": model_name,
        "model_label": model_label,
        "n_images": n,
        "n_detected": len(detected),
        "det_rate_at_conf": safe_rate([record.top1_conf >= float(conf_threshold) for record in records]),
        "no_detection_rate": safe_rate([not record.has_detection for record in records]),
        "low_conf_rate": safe_rate([record.top1_conf < float(conf_threshold) for record in records]),
        "high_conf_rate": safe_rate(high_conf_flags),
        "mean_box_area_frac": safe_mean(area_values),
        "median_box_area_frac": safe_median(area_values),
        "tiny_box_rate": safe_rate([record.tiny_box for record in records]),
        "large_box_rate": safe_rate([record.large_box for record in records]),
        "border_touch_rate": safe_rate([record.border_touching for record in records]),
        "weird_box_rate": safe_rate(weird_flags),
        "high_conf_weird_box_rate": safe_rate(high_conf_weird_flags),
    }


def aggregate_label_aware_reliability(
    dataset_name: str,
    dataset_label: str,
    model_name: str,
    model_label: str,
    records: Sequence[Top1PredictionRecord],
    label_aware_records: Sequence[LabelAwarePredictionRecord],
    label_records: Mapping[Path, LabelRecord],
    correct_iou: float,
    good_conf: float,
) -> Dict[str, object]:
    pairs = [
        (record, label_aware)
        for record, label_aware in zip(records, label_aware_records)
        if label_records.get(record.image_path) is not None and label_records[record.image_path].gt_boxes_xyxy
    ]
    ious = [label_aware.top1_iou for _, label_aware in pairs]
    center_errors = [label_aware.top1_center_error_frac for _, label_aware in pairs]
    correct_flags = [label_aware.top1_correct_at_iou for _, label_aware in pairs]
    high_conf_wrong_flags = [label_aware.high_conf_wrong for _, label_aware in pairs]
    false_negative_flags = [label_aware.false_negative_proxy for _, label_aware in pairs]
    high_conf_wrong_or_weird = [
        bool(label_aware.high_conf_wrong or (record.top1_conf >= float(good_conf) and prediction_is_weird(record)))
        for record, label_aware in pairs
    ]
    correct_col = f"top1_correct_rate_at_iou_{float(correct_iou):.2f}"
    correct_rate = safe_rate(correct_flags)
    return {
        "dataset": dataset_name,
        "dataset_label": dataset_label,
        "model": model_name,
        "model_label": model_label,
        "n_labeled_images": len(pairs),
        "top1_correct_rate_at_iou": correct_rate,
        correct_col: correct_rate,
        "mean_top1_iou": safe_mean(ious),
        "median_top1_iou": safe_median(ious),
        "mean_center_error_frac": safe_mean(center_errors),
        "false_negative_rate_at_conf": safe_rate(false_negative_flags),
        "high_conf_wrong_rate": safe_rate(high_conf_wrong_flags),
        "high_conf_wrong_or_weird_rate": safe_rate(high_conf_wrong_or_weird),
    }


def compute_temporal_prediction_records(
    records: Sequence[Top1PredictionRecord],
    good_conf: float,
    bad_conf: float,
    box_jump_center_frac: float,
) -> List[TemporalPredictionRecord]:
    temporal_records: List[TemporalPredictionRecord] = []
    for idx, record in enumerate(records):
        if idx == 0:
            temporal_records.append(
                TemporalPredictionRecord(
                    prev_iou=float("nan"),
                    center_shift_frac=float("nan"),
                    area_ratio=float("nan"),
                    confidence_delta=float("nan"),
                    box_jump=False,
                    confidence_collapse=False,
                )
            )
            continue

        prev = records[idx - 1]
        prev_iou = (
            box_iou_xyxy(prev.top1_xyxy, record.top1_xyxy)
            if prev.top1_xyxy is not None and record.top1_xyxy is not None
            else float("nan")
        )
        shift = center_shift_frac(prev, record)
        area_ratio = symmetric_area_ratio(prev.box_area_frac, record.box_area_frac)
        temporal_records.append(
            TemporalPredictionRecord(
                prev_iou=prev_iou,
                center_shift_frac=shift,
                area_ratio=area_ratio,
                confidence_delta=record.top1_conf - prev.top1_conf,
                box_jump=bool(np.isfinite(shift) and shift >= float(box_jump_center_frac)),
                confidence_collapse=bool(prev.top1_conf >= float(good_conf) and record.top1_conf <= float(bad_conf)),
            )
        )
    return temporal_records


def failure_streak_lengths(flags: Sequence[bool]) -> List[int]:
    lengths: List[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current > 0:
            lengths.append(current)
            current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def aggregate_temporal_reliability(
    dataset_name: str,
    dataset_label: str,
    model_name: str,
    model_label: str,
    records: Sequence[Top1PredictionRecord],
    temporal_records: Sequence[TemporalPredictionRecord],
    conf_threshold: float,
    box_jump_area_ratio: float,
    label_aware_records: Optional[Sequence[LabelAwarePredictionRecord]] = None,
    label_records: Optional[Mapping[Path, LabelRecord]] = None,
) -> Dict[str, object]:
    transitions = list(temporal_records[1:])
    failure_flags: List[bool] = []
    for idx, record in enumerate(records):
        label_aware = label_aware_records[idx] if label_aware_records is not None else None
        label_record = label_records.get(record.image_path) if label_records is not None else None
        if label_aware is not None and label_record is not None and label_record.gt_boxes_xyxy:
            failure_flags.append(bool((not label_aware.top1_correct_at_iou) or prediction_is_weird(record)))
        else:
            failure_flags.append(bool((not record.has_detection) or record.top1_conf < float(conf_threshold) or prediction_is_weird(record)))

    streaks = failure_streak_lengths(failure_flags)
    good_flags: List[bool] = []
    for idx, record in enumerate(records):
        temporal_record = temporal_records[idx] if idx < len(temporal_records) else None
        has_box_jump = bool(temporal_record.box_jump) if temporal_record is not None else False
        good_flags.append(
            bool(
                record.has_detection
                and record.top1_conf >= float(conf_threshold)
                and not prediction_is_weird(record)
                and not has_box_jump
            )
        )
    good_streaks = failure_streak_lengths(good_flags)
    area_jump_flags = [
        bool(np.isfinite(record.area_ratio) and record.area_ratio >= float(box_jump_area_ratio))
        for record in transitions
    ]

    return {
        "dataset": dataset_name,
        "dataset_label": dataset_label,
        "model": model_name,
        "model_label": model_label,
        "n_images": len(records),
        "n_transitions": len(transitions),
        "n_valid_box_transitions": int(sum(np.isfinite(record.prev_iou) for record in transitions)),
        "mean_consecutive_box_iou": safe_mean([record.prev_iou for record in transitions]),
        "median_consecutive_box_iou": safe_median([record.prev_iou for record in transitions]),
        "mean_center_shift_frac": safe_mean([record.center_shift_frac for record in transitions]),
        "box_jump_rate": safe_rate([record.box_jump for record in transitions]),
        "area_jump_rate": safe_rate(area_jump_flags),
        "confidence_collapse_rate": safe_rate([record.confidence_collapse for record in transitions]),
        "failure_streak_count": len(streaks),
        "max_failure_streak": max(streaks) if streaks else 0,
        "mean_failure_streak_len": safe_mean([float(length) for length in streaks]),
        "failure_rate": safe_rate(failure_flags),
        "failure_basis": "label_correct_or_weird" if label_aware_records is not None else "proxy_conf_or_weird",
        "good_streak_count": len(good_streaks),
        "max_good_streak": max(good_streaks) if good_streaks else 0,
        "mean_good_streak_len": safe_mean([float(length) for length in good_streaks]),
        "good_rate": safe_rate(good_flags),
        "good_basis": "proxy_conf_not_weird_not_box_jump",
    }


def write_csv_rows(rows: Sequence[Mapping[str, object]], out_path: Path, fieldnames: Sequence[str]) -> Optional[Path]:
    if not rows:
        return None
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def write_reliability_summary_txt(
    reliability_rows: Sequence[Mapping[str, object]],
    temporal_rows: Sequence[Mapping[str, object]],
    label_rows: Sequence[Mapping[str, object]],
    label_notes: Sequence[str],
    out_path: Path,
) -> Path:
    lines: List[str] = ["Reliability summary", ""]
    lines.extend(["Label status:"] + [f"  - {note}" for note in label_notes] + [""])

    dataset_names = sorted({str(row["dataset"]) for row in reliability_rows})
    for dataset_name in dataset_names:
        lines.append(f"Dataset: {dataset_name}")
        ds_reliability = [row for row in reliability_rows if row["dataset"] == dataset_name]
        ds_temporal = [row for row in temporal_rows if row["dataset"] == dataset_name]
        ds_labels = [row for row in label_rows if row["dataset"] == dataset_name]
        temporal_lookup = {row["model"]: row for row in ds_temporal}
        label_lookup = {row["model"]: row for row in ds_labels}
        for row in ds_reliability:
            temporal = temporal_lookup.get(row["model"], {})
            label = label_lookup.get(row["model"], {})
            pieces = [
                f"  {row['model_label']}:",
                f"det@conf={100.0 * safe_float(row.get('det_rate_at_conf')):.1f}%",
                f"weird={100.0 * safe_float(row.get('weird_box_rate')):.1f}%",
                f"high-conf weird={100.0 * safe_float(row.get('high_conf_weird_box_rate')):.1f}%",
                f"box-jump={100.0 * safe_float(temporal.get('box_jump_rate')):.1f}%",
                f"max streak={safe_float(temporal.get('max_failure_streak'), 0.0):.0f}",
                f"max good={safe_float(temporal.get('max_good_streak'), 0.0):.0f}",
            ]
            if label:
                pieces.insert(2, f"correct={100.0 * safe_float(label.get('top1_correct_rate_at_iou')):.1f}%")
                pieces.append(f"high-conf wrong={100.0 * safe_float(label.get('high_conf_wrong_rate')):.1f}%")
            lines.append(" ".join(pieces))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# -----------------------------
# Slide summaries / reports
# -----------------------------
def write_slide_summary_csv(
    dataset_specs: Sequence[DatasetSpec],
    results_by_dataset: Mapping[str, List[EvalResult]],
    out_path: Path,
) -> Path:
    fieldnames = [
        "dataset",
        "model",
        "det_rate_at_conf_pct",
        "mean_top1_conf",
        "median_top1_conf",
        "p90_top1_conf",
        "p99_top1_conf",
        "std_top1_conf",
        "delta_det_rate_vs_baseline_pp",
        "delta_mean_top1_conf_vs_baseline",
        "delta_det_rate_vs_odad_baseline_pp",
        "delta_mean_top1_conf_vs_odad_baseline",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ds in dataset_specs:
            ds_results = results_by_dataset[ds.name]
            ds_lookup = {result.model_name: result for result in ds_results}
            baseline = ds_lookup.get("baseline")
            odad_base = ds_lookup.get("odad_baseline")

            for result in ds_results:
                det_rate = result.stats.det_rate_at_conf * 100.0
                baseline_det_delta = None
                baseline_mean_delta = None
                odad_det_delta = None
                odad_mean_delta = None

                if baseline is not None:
                    baseline_det_delta = 100.0 * (result.stats.det_rate_at_conf - baseline.stats.det_rate_at_conf)
                    baseline_mean_delta = result.stats.mean_top1_conf - baseline.stats.mean_top1_conf
                if odad_base is not None:
                    odad_det_delta = 100.0 * (result.stats.det_rate_at_conf - odad_base.stats.det_rate_at_conf)
                    odad_mean_delta = result.stats.mean_top1_conf - odad_base.stats.mean_top1_conf

                writer.writerow(
                    {
                        "dataset": result.dataset_name,
                        "model": result.model_name,
                        "det_rate_at_conf_pct": f"{det_rate:.2f}",
                        "mean_top1_conf": f"{result.stats.mean_top1_conf:.4f}",
                        "median_top1_conf": f"{result.stats.median_top1_conf:.4f}",
                        "p90_top1_conf": f"{result.stats.p90_top1_conf:.4f}",
                        "p99_top1_conf": f"{result.stats.p99_top1_conf:.4f}",
                        "std_top1_conf": f"{result.stats.std_top1_conf:.4f}",
                        "delta_det_rate_vs_baseline_pp": "" if baseline_det_delta is None else f"{baseline_det_delta:.2f}",
                        "delta_mean_top1_conf_vs_baseline": "" if baseline_mean_delta is None else f"{baseline_mean_delta:.4f}",
                        "delta_det_rate_vs_odad_baseline_pp": "" if odad_det_delta is None else f"{odad_det_delta:.2f}",
                        "delta_mean_top1_conf_vs_odad_baseline": "" if odad_mean_delta is None else f"{odad_mean_delta:.4f}",
                    }
                )

    return out_path


def write_slide_summary_tex(
    dataset_specs: Sequence[DatasetSpec],
    results_by_dataset: Mapping[str, List[EvalResult]],
    conf_threshold: float,
    out_path: Path,
) -> Path:
    lines: List[str] = [
        "% Auto-generated by tools/yolo_ablation.py",
        "% Compact slide table focused on the most presentation-friendly metrics.",
        "",
    ]

    for ds in dataset_specs:
        ds_results = results_by_dataset[ds.name]
        ds_label = ds_results[0].dataset_label
        lines.extend(
            [
                "\\begin{table}[t]",
                "\\centering",
                "\\small",
                f"\\caption{{{latex_escape(ds_label)} top-1 confidence summary at $\\tau_c={conf_threshold:.2f}$.}}",
                "\\begin{tabular}{lrrrrr}",
                "\\hline",
                "Model & Det. @ $\\tau_c$ (\\%) & Mean & Median & P90 & Std \\\\",
                "\\hline",
            ]
        )
        for result in ds_results:
            lines.append(
                f"{latex_escape(result.model_label)}"
                f" & {100.0 * result.stats.det_rate_at_conf:.1f}"
                f" & {result.stats.mean_top1_conf:.3f}"
                f" & {result.stats.median_top1_conf:.3f}"
                f" & {result.stats.p90_top1_conf:.3f}"
                f" & {result.stats.std_top1_conf:.3f} \\\\"
            )
        lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_odad_delta_summary(
    dataset_specs: Sequence[DatasetSpec],
    results_by_dataset: Mapping[str, List[EvalResult]],
    out_path: Path,
) -> Path:
    lines: List[str] = ["ODAD delta summary", ""]

    for ds in dataset_specs:
        ds_results = results_by_dataset[ds.name]
        ds_lookup = {result.model_name: result for result in ds_results}
        lines.append(f"Dataset: {ds_results[0].dataset_label}")

        baseline = ds_lookup.get("baseline")
        odad_base = ds_lookup.get("odad_baseline")
        odad_final = ds_lookup.get("odad_final")
        fda = ds_lookup.get("fda")
        fda_mix = ds_lookup.get("fda_mix")

        if odad_final is None:
            lines.append("  ODAD final checkpoint is not available in this run.")
            lines.append("")
            continue

        if baseline is not None:
            lines.append(
                "  ODAD final vs baseline: "
                f"det_rate {format_percent_pp(100.0 * (odad_final.stats.det_rate_at_conf - baseline.stats.det_rate_at_conf))}, "
                f"mean {format_signed(odad_final.stats.mean_top1_conf - baseline.stats.mean_top1_conf)}, "
                f"median {format_signed(odad_final.stats.median_top1_conf - baseline.stats.median_top1_conf)}"
            )
        if odad_base is not None:
            lines.append(
                "  ODAD final vs odad_baseline: "
                f"det_rate {format_percent_pp(100.0 * (odad_final.stats.det_rate_at_conf - odad_base.stats.det_rate_at_conf))}, "
                f"mean {format_signed(odad_final.stats.mean_top1_conf - odad_base.stats.mean_top1_conf)}, "
                f"median {format_signed(odad_final.stats.median_top1_conf - odad_base.stats.median_top1_conf)}"
            )

        if fda is not None:
            lines.append(
                "  Gap to FDA: "
                f"det_rate {format_percent_pp(100.0 * (odad_final.stats.det_rate_at_conf - fda.stats.det_rate_at_conf))}, "
                f"mean {format_signed(odad_final.stats.mean_top1_conf - fda.stats.mean_top1_conf)}, "
                f"median {format_signed(odad_final.stats.median_top1_conf - fda.stats.median_top1_conf)}"
            )
        if fda_mix is not None:
            lines.append(
                "  Gap to FDA-mix: "
                f"det_rate {format_percent_pp(100.0 * (odad_final.stats.det_rate_at_conf - fda_mix.stats.det_rate_at_conf))}, "
                f"mean {format_signed(odad_final.stats.mean_top1_conf - fda_mix.stats.mean_top1_conf)}, "
                f"median {format_signed(odad_final.stats.median_top1_conf - fda_mix.stats.median_top1_conf)}"
            )

        present_progress = [name for name in ODAD_PROGRESS_ORDER if name in ds_lookup]
        if len(present_progress) >= 2:
            lines.append("  Checkpoint-to-checkpoint deltas:")
            for prev_name, curr_name in zip(present_progress[:-1], present_progress[1:]):
                prev = ds_lookup[prev_name]
                curr = ds_lookup[curr_name]
                lines.append(
                    f"    {curr.model_label} - {prev.model_label}: "
                    f"det_rate {format_percent_pp(100.0 * (curr.stats.det_rate_at_conf - prev.stats.det_rate_at_conf))}, "
                    f"mean {format_signed(curr.stats.mean_top1_conf - prev.stats.mean_top1_conf)}"
                )

        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    utils.setup_logging()
    configure_slide_style()

    output_root = Path(args.output).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    device, device_str = resolve_device(args.device)
    utils.LOGGER.info("Device: %s (%s)", device, device_str)

    model_specs = parse_model_specs(args.model)
    dataset_specs = parse_dataset_specs(args.dataset, "--dataset")
    neg_dataset_specs = parse_dataset_specs(args.neg_dataset, "--neg-dataset") if args.neg_dataset else []

    rng = random.Random(int(args.seed))
    model_names = [spec.name for spec in model_specs]
    model_colors = build_model_color_map(model_names)
    slide_subset_names = resolve_slide_subset(model_names, args.slide_subset_model)
    requested_gallery_model_names = parse_csv_names(args.gallery_models)
    missing_gallery_models = [name for name in requested_gallery_model_names if name not in model_names]
    if missing_gallery_models and args.reliability_mode:
        utils.LOGGER.warning("Ignoring missing --gallery-models names: %s", ", ".join(missing_gallery_models))
    gallery_model_names = [name for name in requested_gallery_model_names if name in model_names]
    if args.reliability_mode and not gallery_model_names:
        gallery_model_names = list(model_names)
    ordered_eval = bool(args.ordered_eval or args.reliability_mode)
    if args.reliability_mode and not args.ordered_eval:
        utils.LOGGER.warning("--reliability-mode enabled; forcing sorted ordered evaluation for temporal metrics.")

    models: List[Tuple[str, YOLO]] = []
    for spec in model_specs:
        utils.LOGGER.info("Loading model '%s' from %s", spec.name, spec.path)
        models.append((spec.name, YOLO(str(spec.path))))

    global_lines: List[str] = []
    global_lines.append("YOLO Ablation (Top-1 Confidence Stats)")
    global_lines.append(f"device={device_str}")
    global_lines.append(f"conf_threshold={float(args.conf):.3f} iou={float(args.iou):.3f}")
    global_lines.append(f"n_vis_per_dataset={int(args.n)} eval_images_per_dataset={int(args.eval_images)} bins={int(args.hist_bins)}")
    if args.reliability_mode:
        global_lines.append("reliability_mode=true")
        global_lines.append(f"ordered_eval={ordered_eval}")
        global_lines.append(f"use_labels={args.use_labels}")
        global_lines.append(f"gallery_models={','.join(gallery_model_names)}")
    global_lines.append("")
    global_lines.append("Datasets:")
    for ds in dataset_specs:
        global_lines.append(f"  - {ds.name}: {ds.root}")
    if neg_dataset_specs:
        global_lines.append("Negative datasets:")
        for ds in neg_dataset_specs:
            global_lines.append(f"  - {ds.name}: {ds.root}")
    global_lines.append("")

    results_by_dataset: Dict[str, List[EvalResult]] = {}
    cdf_paths: List[Path] = []
    reliability_summary_rows: List[Dict[str, object]] = []
    label_aware_summary_rows: List[Dict[str, object]] = []
    temporal_summary_rows: List[Dict[str, object]] = []
    paired_failure_summary_rows: List[Dict[str, object]] = []
    gallery_index_rows_by_category: Dict[str, List[Dict[str, object]]] = {
        category: [] for category in PAIRED_FAILURE_CATEGORIES
    }
    label_notes: List[str] = []

    for ds in dataset_specs:
        ds_out = output_root / ds.name
        ds_out.mkdir(parents=True, exist_ok=True)
        ds_label = dataset_display_name(ds.name)

        utils.LOGGER.info("==== Dataset: %s (%s) ====", ds.name, ds.root)
        all_images = gather_images(ds.root)
        n_vis = min(int(args.n), len(all_images))
        n_eval = min(int(args.eval_images), len(all_images))

        vis_images = rng.sample(all_images, n_vis) if n_vis > 0 else []
        if ordered_eval:
            eval_images = all_images[:n_eval]
        else:
            eval_images = rng.sample(all_images, n_eval) if n_eval > 0 else []

        if vis_images:
            utils.LOGGER.info("Generating qualitative comparison plots for dataset '%s' (n=%d)", ds.name, len(vis_images))
            for image_path in vis_images:
                with Image.open(image_path) as img:
                    image_array = np.array(img.convert("RGB"))

                per_model_top1: List[Tuple[str, Optional[Dict]]] = []
                for name, m in models:
                    preds = predict_yolo(m, image_path, conf=0.001, iou=float(args.iou), device_str=device_str)
                    per_model_top1.append((name, top1_detection(preds)))

                create_comparison_plot(
                    image_path=image_path,
                    image_array=image_array,
                    dataset_label=ds_label,
                    per_model_top1=per_model_top1,
                    output_dir=ds_out,
                )

        ds_lines: List[str] = []
        ds_lines.append(f"Dataset: {ds_label}")
        ds_lines.append(f"root: {ds.root}")
        ds_lines.append(f"eval_images: {len(eval_images)}")
        ds_lines.append(f"conf_threshold (for det_rate): {float(args.conf):.3f}")
        ds_lines.append(f"iou: {float(args.iou):.3f}")
        ds_lines.append("")

        image_sizes: Dict[Path, Tuple[int, int]] = {}
        labels_available = False
        label_records: Dict[Path, LabelRecord] = {}
        label_aware_by_model: Dict[str, List[LabelAwarePredictionRecord]] = {}
        reliability_records_by_model: Dict[str, List[Top1PredictionRecord]] = {}
        if args.reliability_mode:
            image_sizes = get_image_sizes(eval_images)
            labels_available, label_records, label_note = load_label_records(
                ds.root,
                eval_images,
                image_sizes,
                use_labels=str(args.use_labels),
            )
            label_notes.append(f"{ds.name}: {label_note}")
            if labels_available:
                utils.LOGGER.info(label_note)
            else:
                utils.LOGGER.warning(label_note)
            ds_lines.extend(["Reliability mode:", f"  labels:              {label_note}", ""])

        ds_results: List[EvalResult] = []
        for name, m in models:
            if args.reliability_mode:
                records = collect_top1_prediction_records(
                    model_name=name,
                    model=m,
                    image_paths=eval_images,
                    image_sizes=image_sizes,
                    iou=float(args.iou),
                    device_str=device_str,
                    large_box_frac=float(args.large_box_frac),
                    tiny_box_frac=float(args.tiny_box_frac),
                    border_margin_frac=float(args.border_margin_frac),
                )
                reliability_records_by_model[name] = records
                stats, top1_confs = compute_conf_stats_from_records(records, conf_threshold=float(args.conf))
                reliability_summary_rows.append(
                    aggregate_box_reliability(
                        dataset_name=ds.name,
                        dataset_label=ds_label,
                        model_name=name,
                        model_label=display_name(name),
                        records=records,
                        conf_threshold=float(args.conf),
                        good_conf=float(args.good_conf),
                    )
                )
            else:
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
                conf_threshold=float(args.conf),
                title=f"{ds_label}: top-1 confidence histogram ({display_name(name)})",
                out_path=hist_path,
            )

            result = EvalResult(
                dataset_name=ds.name,
                dataset_label=ds_label,
                dataset_root=ds.root,
                model_name=name,
                model_label=display_name(name),
                stats=stats,
                top1_confs=top1_confs,
                hist_path=hist_path,
            )
            ds_results.append(result)

            block = [
                f"[{display_name(name)}]",
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

        if args.reliability_mode:
            temporal_records_by_model: Dict[str, List[TemporalPredictionRecord]] = {}
            for name in model_names:
                if name not in reliability_records_by_model:
                    continue
                records = reliability_records_by_model[name]
                if labels_available:
                    label_aware_records = compute_label_aware_records(
                        records=records,
                        label_records=label_records,
                        correct_iou=float(args.correct_iou),
                        conf_threshold=float(args.conf),
                        good_conf=float(args.good_conf),
                    )
                    label_aware_by_model[name] = label_aware_records
                    label_aware_summary_rows.append(
                        aggregate_label_aware_reliability(
                            dataset_name=ds.name,
                            dataset_label=ds_label,
                            model_name=name,
                            model_label=display_name(name),
                            records=records,
                            label_aware_records=label_aware_records,
                            label_records=label_records,
                            correct_iou=float(args.correct_iou),
                            good_conf=float(args.good_conf),
                        )
                    )

                temporal_records = compute_temporal_prediction_records(
                    records,
                    good_conf=float(args.good_conf),
                    bad_conf=float(args.bad_conf),
                    box_jump_center_frac=float(args.box_jump_center_frac),
                )
                temporal_records_by_model[name] = temporal_records
                temporal_summary = aggregate_temporal_reliability(
                    dataset_name=ds.name,
                    dataset_label=ds_label,
                    model_name=name,
                    model_label=display_name(name),
                    records=records,
                    temporal_records=temporal_records,
                    conf_threshold=float(args.conf),
                    box_jump_area_ratio=float(args.box_jump_area_ratio),
                    label_aware_records=label_aware_by_model.get(name),
                    label_records=label_records if labels_available else None,
                )
                temporal_summary_rows.append(temporal_summary)
                for reliability_row in reversed(reliability_summary_rows):
                    if reliability_row["dataset"] == ds.name and reliability_row["model"] == name:
                        for key in (
                            "good_streak_count",
                            "max_good_streak",
                            "mean_good_streak_len",
                            "good_rate",
                            "good_basis",
                        ):
                            reliability_row[key] = temporal_summary.get(key)
                        break

            selected_by_category = select_paired_failure_examples(
                dataset_name=ds.name,
                image_paths=eval_images,
                records_by_model=reliability_records_by_model,
                label_aware_by_model=label_aware_by_model,
                labels_available=labels_available,
                good_conf=float(args.good_conf),
                bad_conf=float(args.bad_conf),
                correct_iou=float(args.correct_iou),
                bad_iou=float(args.bad_iou),
            )
            paired_rows, category_index_rows = save_paired_failure_galleries(
                dataset_name=ds.name,
                dataset_label=ds_label,
                selected_by_category=selected_by_category,
                records_by_model=reliability_records_by_model,
                label_records=label_records,
                label_aware_by_model=label_aware_by_model,
                gallery_model_names=gallery_model_names,
                output_root=output_root,
                gallery_max=int(args.gallery_max),
                good_conf=float(args.good_conf),
                bad_conf=float(args.bad_conf),
                bad_iou=float(args.bad_iou),
            )
            paired_failure_summary_rows.extend(paired_rows)
            for category, rows in category_index_rows.items():
                gallery_index_rows_by_category[category].extend(rows)

        results_by_dataset[ds.name] = ds_results

        cdf_path = output_root / f"cdf_top1conf_{ds.name}.png"
        save_cdf_plot(
            dataset_label=ds_label,
            results=ds_results,
            conf_threshold=float(args.conf),
            colors=model_colors,
            out_path=cdf_path,
        )
        cdf_paths.append(cdf_path)
        ds_lines.extend(
            [
                "Slide-ready artifacts:",
                f"  cdf:                 {cdf_path.name}",
                "",
            ]
        )

        ds_summary_path = ds_out / "summary.txt"
        ds_summary_path.write_text("\n".join(ds_lines), encoding="utf-8")
        utils.LOGGER.info("Wrote dataset summary to %s", ds_summary_path)

        global_lines.append(f"==== {ds_label} ====")
        global_lines.extend(ds_lines[2:])

    if results_by_dataset:
        det_rate_bar_path = output_root / "grouped_bar_det_rate.png"
        mean_conf_bar_path = output_root / "grouped_bar_mean_conf.png"
        save_grouped_metric_bar(
            dataset_specs=dataset_specs,
            results_by_dataset=results_by_dataset,
            model_names=model_names,
            colors=model_colors,
            metric_name="det_rate_at_conf",
            title=rf"Detection rate @ $\tau_c={float(args.conf):.2f}$",
            ylabel=rf"Detection rate @ $\tau_c$",
            out_path=det_rate_bar_path,
            use_percent_axis=True,
        )
        save_grouped_metric_bar(
            dataset_specs=dataset_specs,
            results_by_dataset=results_by_dataset,
            model_names=model_names,
            colors=model_colors,
            metric_name="mean_top1_conf",
            title="Mean top-1 confidence",
            ylabel="Mean top-1 confidence",
            out_path=mean_conf_bar_path,
            use_percent_axis=False,
        )

        odad_trend_path = save_odad_checkpoint_trend(
            dataset_specs=dataset_specs,
            results_by_dataset=results_by_dataset,
            colors=model_colors,
            out_path=output_root / "odad_checkpoint_trend.png",
        )
        rank_plot_path = save_precision_style_rank_plot(
            dataset_specs=dataset_specs,
            results_by_dataset=results_by_dataset,
            subset_model_names=slide_subset_names,
            colors=model_colors,
            conf_threshold=float(args.conf),
            out_path=output_root / "precision_style_rank_plot.png",
        )

        slide_csv_path = write_slide_summary_csv(
            dataset_specs=dataset_specs,
            results_by_dataset=results_by_dataset,
            out_path=output_root / "slide_summary.csv",
        )
        slide_tex_path = write_slide_summary_tex(
            dataset_specs=dataset_specs,
            results_by_dataset=results_by_dataset,
            conf_threshold=float(args.conf),
            out_path=output_root / "slide_summary.tex",
        )
        odad_delta_path = write_odad_delta_summary(
            dataset_specs=dataset_specs,
            results_by_dataset=results_by_dataset,
            out_path=output_root / "odad_delta_summary.txt",
        )

        global_lines.append("")
        global_lines.append("==== Slide-ready artifacts ====")
        global_lines.append(f"  grouped_bar_det_rate.png")
        global_lines.append(f"  grouped_bar_mean_conf.png")
        for cdf_path in cdf_paths:
            global_lines.append(f"  {cdf_path.name}")
        if odad_trend_path is not None:
            global_lines.append(f"  {odad_trend_path.name}")
        if rank_plot_path is not None:
            global_lines.append(f"  {rank_plot_path.name}")
        global_lines.append(f"  {slide_csv_path.name}")
        global_lines.append(f"  {slide_tex_path.name}")
        global_lines.append(f"  {odad_delta_path.name}")

        if args.reliability_mode:
            reliability_fieldnames = [
                "dataset",
                "dataset_label",
                "model",
                "model_label",
                "n_images",
                "n_detected",
                "det_rate_at_conf",
                "no_detection_rate",
                "low_conf_rate",
                "high_conf_rate",
                "mean_box_area_frac",
                "median_box_area_frac",
                "tiny_box_rate",
                "large_box_rate",
                "border_touch_rate",
                "weird_box_rate",
                "high_conf_weird_box_rate",
                "good_streak_count",
                "max_good_streak",
                "mean_good_streak_len",
                "good_rate",
                "good_basis",
            ]
            temporal_fieldnames = [
                "dataset",
                "dataset_label",
                "model",
                "model_label",
                "n_images",
                "n_transitions",
                "n_valid_box_transitions",
                "mean_consecutive_box_iou",
                "median_consecutive_box_iou",
                "mean_center_shift_frac",
                "box_jump_rate",
                "area_jump_rate",
                "confidence_collapse_rate",
                "failure_streak_count",
                "max_failure_streak",
                "mean_failure_streak_len",
                "failure_rate",
                "failure_basis",
                "good_streak_count",
                "max_good_streak",
                "mean_good_streak_len",
                "good_rate",
                "good_basis",
            ]
            label_fieldnames = [
                "dataset",
                "dataset_label",
                "model",
                "model_label",
                "n_labeled_images",
                "top1_correct_rate_at_iou",
                f"top1_correct_rate_at_iou_{float(args.correct_iou):.2f}",
                "mean_top1_iou",
                "median_top1_iou",
                "mean_center_error_frac",
                "false_negative_rate_at_conf",
                "high_conf_wrong_rate",
                "high_conf_wrong_or_weird_rate",
            ]
            paired_fieldnames = [
                "dataset",
                "dataset_label",
                "category",
                "n_selected",
                "n_saved",
                "gallery_dir",
                "index_csv",
            ]

            reliability_csv_path = write_csv_rows(
                reliability_summary_rows,
                output_root / "reliability_summary.csv",
                reliability_fieldnames,
            )
            temporal_csv_path = write_csv_rows(
                temporal_summary_rows,
                output_root / "temporal_reliability_summary.csv",
                temporal_fieldnames,
            )
            label_csv_path = write_csv_rows(
                label_aware_summary_rows,
                output_root / "label_aware_summary.csv",
                label_fieldnames,
            )
            paired_csv_path = write_csv_rows(
                paired_failure_summary_rows,
                output_root / "paired_failure_summary.csv",
                paired_fieldnames,
            )
            write_gallery_indices(gallery_index_rows_by_category, output_root)
            reliability_txt_path = write_reliability_summary_txt(
                reliability_rows=reliability_summary_rows,
                temporal_rows=temporal_summary_rows,
                label_rows=label_aware_summary_rows,
                label_notes=label_notes,
                out_path=output_root / "reliability_summary.txt",
            )

            weird_plot_path = save_grouped_row_metric_bar(
                dataset_specs=dataset_specs,
                rows=reliability_summary_rows,
                model_names=model_names,
                colors=model_colors,
                metric_name="weird_box_rate",
                title="Weird-box rate",
                ylabel="Weird-box rate",
                out_path=output_root / "reliability_bar_weird_box_rate.png",
                use_percent_axis=True,
            )
            high_conf_wrong_plot_path = None
            if label_aware_summary_rows:
                high_conf_wrong_plot_path = save_grouped_row_metric_bar(
                    dataset_specs=dataset_specs,
                    rows=label_aware_summary_rows,
                    model_names=model_names,
                    colors=model_colors,
                    metric_name="high_conf_wrong_rate",
                    title="High-confidence wrong prediction rate",
                    ylabel="High-conf wrong rate",
                    out_path=output_root / "reliability_bar_high_conf_wrong_rate.png",
                    use_percent_axis=True,
                )
            box_jump_plot_path = save_grouped_row_metric_bar(
                dataset_specs=dataset_specs,
                rows=temporal_summary_rows,
                model_names=model_names,
                colors=model_colors,
                metric_name="box_jump_rate",
                title="Temporal box-jump rate",
                ylabel="Box-jump rate",
                out_path=output_root / "temporal_box_jump_rate.png",
                use_percent_axis=True,
            )
            checkpoint_reliability_path = save_checkpoint_reliability_trend(
                dataset_specs=dataset_specs,
                reliability_rows=reliability_summary_rows,
                temporal_rows=temporal_summary_rows,
                label_rows=label_aware_summary_rows,
                colors=model_colors,
                out_path=output_root / "checkpoint_reliability_trend.png",
            )
            failure_streak_plot_path = save_grouped_row_metric_bar(
                dataset_specs=dataset_specs,
                rows=temporal_summary_rows,
                model_names=model_names,
                colors=model_colors,
                metric_name="max_failure_streak",
                title="Max failure streak by model",
                ylabel="Max failure streak (frames)",
                out_path=output_root / "failure_streaks_by_model.png",
                use_percent_axis=False,
            )

            global_lines.append("")
            global_lines.append("==== Reliability artifacts ====")
            for path in [
                reliability_csv_path,
                reliability_txt_path,
                label_csv_path,
                temporal_csv_path,
                paired_csv_path,
                weird_plot_path,
                high_conf_wrong_plot_path,
                box_jump_plot_path,
                checkpoint_reliability_path,
                failure_streak_plot_path,
            ]:
                if path is not None:
                    global_lines.append(f"  {Path(path).name}")
            global_lines.append("  reliability_galleries/")

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
            nds_label = dataset_display_name(nds.name)

            neg_lines: List[str] = []
            neg_lines.append(f"Negative dataset: {nds_label}")
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
                fp_rate = stats.det_rate_at_conf

                hist_path = nds_out / f"hist_top1conf_{name}.png"
                save_histogram(
                    top1_confs,
                    bins=int(args.hist_bins),
                    conf_threshold=float(args.conf),
                    title=f"{nds_label}: top-1 confidence histogram ({display_name(name)})",
                    out_path=hist_path,
                )

                neg_lines.extend(
                    [
                        f"[{display_name(name)}]",
                        f"  FP_proxy_rate@conf:  {100.0 * fp_rate:.1f}%  ({stats.n_top1_ge_conf}/{stats.n_images})",
                        f"  top1_conf mean/med:  {stats.mean_top1_conf:.3f} / {stats.median_top1_conf:.3f}",
                        f"  top1_conf p90/p99:   {stats.p90_top1_conf:.3f} / {stats.p99_top1_conf:.3f}",
                        f"  hist:                {hist_path.name}",
                        "",
                    ]
                )

            (nds_out / "summary.txt").write_text("\n".join(neg_lines), encoding="utf-8")
            global_lines.append(f"---- NEG:{nds_label} ----")
            global_lines.extend(neg_lines[5:])

    global_summary = output_root / "summary.txt"
    global_summary.write_text("\n".join(global_lines), encoding="utf-8")
    utils.LOGGER.info("Wrote global summary to %s", global_summary)
    utils.LOGGER.info("Done.")


if __name__ == "__main__":
    main()


# PYTHONPATH=. /home/hm25936/miniforge3/envs/gpu_test/bin/python3 tools/yolo_ablation.py \
# --model baseline=/home/hm25936/mae/runs/yolov8_baseline/baseline/weights/best.pt \
# --model fda=/home/hm25936/mae/runs/yolov8_fda/baseline/weights/best.pt \
# --model fda_mix=/home/hm25936/mae/runs/yolov8_fda_mix/baseline/weights/best.pt \
# --model odad_1500=/home/hm25936/mae/odad/online_adapt_topk2_full/checkpoints/student_frame_001500.pt \
# --model odad_3000=/home/hm25936/mae/odad/online_adapt_topk2_full/checkpoints/student_frame_003000.pt \
# --model odad_4500=/home/hm25936/mae/odad/online_adapt_topk2_full/checkpoints/student_frame_004500.pt \
# --model odad_final=/home/hm25936/mae/odad/online_adapt_topk2_full/student_final.pt \
# --dataset lab=/home/hm25936/datasets_for_yolo/lab_images_6000 \
# --output /home/hm25936/mae/yolo_ablation_runs/compare_models_odad_topk2_slide_ready \
# --device cuda:0 \
# --conf 0.25 \
# --iou 0.45 \
# --n 12 \
# --eval-images 500 \
# --hist-bins 30
