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
from math import ceil
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
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
        ax.set_ylim(0.0, max(0.08, min(1.02, upper * 1.10 + 0.04)))


def _annotate_bars(ax, bars, values: Sequence[float], use_percent_axis: bool) -> None:
    for bar, value in zip(bars, values):
        label = f"{value:.1f}%" if use_percent_axis else f"{value:.3f}"
        y_offset = 1.0 if use_percent_axis else 0.015
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

    results_by_dataset: Dict[str, List[EvalResult]] = {}
    cdf_paths: List[Path] = []

    for ds in dataset_specs:
        ds_out = output_root / ds.name
        ds_out.mkdir(parents=True, exist_ok=True)
        ds_label = dataset_display_name(ds.name)

        utils.LOGGER.info("==== Dataset: %s (%s) ====", ds.name, ds.root)
        all_images = gather_images(ds.root)
        n_vis = min(int(args.n), len(all_images))
        n_eval = min(int(args.eval_images), len(all_images))

        vis_images = rng.sample(all_images, n_vis) if n_vis > 0 else []
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

        ds_results: List[EvalResult] = []
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