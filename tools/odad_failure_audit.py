#!/usr/bin/env python3
"""Failure audit and confidence-threshold sweep for ODAD YOLO checkpoints."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from dual_yolo_mae import utils  # noqa: E402
from yolo_ablation import (  # noqa: E402
    Top1PredictionRecord,
    aggregate_proxy_stream_reliability,
    build_model_color_map,
    collect_top1_prediction_records,
    compute_temporal_prediction_records,
    display_name,
    failure_streak_lengths,
    gather_images,
    get_image_sizes,
    load_gallery_font,
    prediction_is_weird,
    proxy_states_for_records,
    resolve_device,
    safe_float,
    save_gallery_strip,
    sanitize_filename,
)


AUDIT_CATEGORIES = [
    "fda_mix_good_membank_bad",
    "membank_good_fda_mix_bad",
    "current_odad_good_membank_bad",
    "adapter_bad_membank_good",
    "membank_low_conf_candidate",
    "membank_no_detection_or_tiny_conf",
    "membank_weird_or_border",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="Repeatable name=/path/to/weights.pt")
    parser.add_argument("--dataset", required=True, help="YOLO dataset root with images/test")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--device", default=None)
    parser.add_argument("--infer-conf", type=float, default=0.001, help="Accepted for run metadata; top-1 pass uses 0.001.")
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--eval-images", type=int, default=0, help="0 means all images")
    parser.add_argument("--ordered-eval", action="store_true", help="Keep sorted stream order; default for this tool.")
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--reference-model", default="membank_prev")
    parser.add_argument("--gallery-max", type=int, default=24)
    parser.add_argument("--manual-audit-count", type=int, default=80)
    parser.add_argument("--good-conf", type=float, default=0.75)
    parser.add_argument("--bad-conf", type=float, default=0.25)
    parser.add_argument("--large-box-frac", type=float, default=0.30)
    parser.add_argument("--tiny-box-frac", type=float, default=0.001)
    parser.add_argument("--border-margin-frac", type=float, default=0.02)
    parser.add_argument("--box-jump-center-frac", type=float, default=0.20)
    parser.add_argument("--bad-iou", type=float, default=0.20, help="Renderer compatibility only.")
    return parser.parse_args()


def parse_models(specs: Sequence[str]) -> Dict[str, Path]:
    models: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Bad --model '{spec}', expected name=/path/to/weights.pt")
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist for {name}: {path}")
        models[name.strip()] = path
    return models


def parse_thresholds(text: str) -> List[float]:
    values = sorted({float(part.strip()) for part in text.split(",") if part.strip()})
    if not values:
        raise ValueError("--thresholds must contain at least one value")
    return values


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def model_state_maps(
    records_by_model: Mapping[str, Sequence[Top1PredictionRecord]],
    temporal_by_model: Mapping[str, Sequence],
    threshold: float,
    good_conf: float,
) -> Dict[str, List[Dict[str, bool]]]:
    return {
        name: proxy_states_for_records(
            records_by_model[name],
            temporal_by_model[name],
            conf_threshold=threshold,
            good_conf=good_conf,
        )
        for name in records_by_model
    }


def add_sweep_aliases(row: Dict[str, object], threshold: float) -> Dict[str, object]:
    row = dict(row)
    row["confidence_threshold"] = threshold
    row["det_rate"] = row.get("det_rate_at_conf")
    row["weird_rate"] = row.get("weird_box_rate")
    row["bad_rate"] = row.get("bad_frame_rate")
    row["p95_bad_streak"] = row.get("p95_bad_streak_len")
    row["bad_streaks_over_50"] = row.get("bad_streaks_gt_50")
    return row


def plot_det_vs_weird(rows: Sequence[Mapping[str, object]], model_names: Sequence[str], out_path: Path) -> Path:
    colors = build_model_color_map(model_names)
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for name in model_names:
        mr = [row for row in rows if row["model"] == name]
        xs = [safe_float(row.get("det_rate")) * 100.0 for row in mr]
        ys = [safe_float(row.get("weird_rate")) * 100.0 for row in mr]
        ax.plot(xs, ys, marker="o", label=display_name(name), color=colors.get(name))
        for row, x, y in zip(mr, xs, ys):
            if float(row["confidence_threshold"]) in {0.15, 0.20, 0.25}:
                ax.text(x + 0.15, y + 0.15, f"{float(row['confidence_threshold']):.2f}", fontsize=8)
    ax.set_xlabel("Detection rate (%)")
    ax.set_ylabel("Weird-box rate (%)")
    ax.set_title("Detection vs weird-box operating points")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_tradeoff(rows: Sequence[Mapping[str, object]], reference_model: str, out_path: Path) -> Path:
    mr = [row for row in rows if row["model"] == reference_model]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    x = [float(row["confidence_threshold"]) for row in mr]
    for key, label in [
        ("det_rate", "det"),
        ("weird_rate", "weird"),
        ("box_jump_rate", "box jump"),
        ("bad_rate", "bad"),
    ]:
        ax.plot(x, [safe_float(row.get(key)) * 100.0 for row in mr], marker="o", label=label)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"{display_name(reference_model)} threshold tradeoff")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_streaks(rows: Sequence[Mapping[str, object]], reference_model: str, out_path: Path) -> Path:
    mr = [row for row in rows if row["model"] == reference_model]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    x = [float(row["confidence_threshold"]) for row in mr]
    for key, label in [("max_bad_streak", "max bad"), ("max_good_streak", "max good")]:
        ax.plot(x, [safe_float(row.get(key)) for row in mr], marker="o", label=label)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Frames")
    ax.set_title(f"{display_name(reference_model)} streak lengths")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def state_good(states: Mapping[str, Sequence[Mapping[str, bool]]], model: str, idx: int) -> bool:
    return bool(states.get(model, [])[idx].get("good")) if model in states else False


def category_reason(category: str, ref: Top1PredictionRecord, other: Optional[Top1PredictionRecord], threshold: float) -> str:
    other_part = "" if other is None else f"; compare conf={other.top1_conf:.3f} weird={int(prediction_is_weird(other))}"
    return (
        f"threshold={threshold:.2f}; membank conf={ref.top1_conf:.3f} "
        f"weird={int(prediction_is_weird(ref))} border={int(ref.border_touching)} "
        f"tiny={int(ref.tiny_box)} large={int(ref.large_box)}{other_part}"
    )


def build_paired_audit_rows(
    records_by_model: Mapping[str, Sequence[Top1PredictionRecord]],
    temporal_by_model: Mapping[str, Sequence],
    thresholds: Sequence[float],
    reference_model: str,
    good_conf: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ref_records = records_by_model[reference_model]
    audit_thresholds = [t for t in thresholds if round(t, 2) in {0.15, 0.20, 0.25}]
    for threshold in sorted(audit_thresholds, reverse=True):
        states = model_state_maps(records_by_model, temporal_by_model, threshold=threshold, good_conf=good_conf)
        for idx, ref in enumerate(ref_records):
            ref_good = state_good(states, reference_model, idx)
            category_models = [
                ("fda_mix_good_membank_bad", "fda_mix", state_good(states, "fda_mix", idx) and not ref_good),
                ("membank_good_fda_mix_bad", "fda_mix", ref_good and not state_good(states, "fda_mix", idx)),
                ("current_odad_good_membank_bad", "current_odad", state_good(states, "current_odad", idx) and not ref_good),
                ("adapter_bad_membank_good", "adapter_base", ref_good and not state_good(states, "adapter_base", idx)),
            ]
            for category, compare_model, include in category_models:
                if include and compare_model in records_by_model:
                    other = records_by_model[compare_model][idx]
                    rows.append(make_audit_row(category, threshold, ref, other, compare_model))

            if ref.top1_conf < 0.25 and ref.top1_conf >= threshold and not prediction_is_weird(ref):
                rows.append(make_audit_row("membank_low_conf_candidate", threshold, ref, None, ""))
            if (not ref.has_detection) or ref.top1_conf < 0.10:
                rows.append(make_audit_row("membank_no_detection_or_tiny_conf", threshold, ref, None, ""))
            if prediction_is_weird(ref):
                rows.append(make_audit_row("membank_weird_or_border", threshold, ref, None, ""))
    return rows


def make_audit_row(
    category: str,
    threshold: float,
    ref: Top1PredictionRecord,
    other: Optional[Top1PredictionRecord],
    compare_model: str,
) -> Dict[str, object]:
    return {
        "threshold": f"{threshold:.2f}",
        "category": category,
        "frame": ref.image_index,
        "image_path": str(ref.image_path),
        "compare_model": compare_model,
        "membank_conf": f"{ref.top1_conf:.4f}",
        "membank_weird": int(prediction_is_weird(ref)),
        "membank_border": int(ref.border_touching),
        "membank_tiny": int(ref.tiny_box),
        "membank_large": int(ref.large_box),
        "compare_conf": "" if other is None else f"{other.top1_conf:.4f}",
        "compare_weird": "" if other is None else int(prediction_is_weird(other)),
        "reason": category_reason(category, ref, other, threshold),
    }


def select_diverse_rows(rows: Sequence[Mapping[str, object]], max_count: int) -> List[Mapping[str, object]]:
    if max_count <= 0 or not rows:
        return []
    ordered = sorted(rows, key=lambda r: (str(r["category"]), float(r["threshold"]), int(r["frame"])))
    if len(ordered) <= max_count:
        return list(ordered)
    selected: List[Mapping[str, object]] = []
    by_category: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in ordered:
        by_category[str(row["category"])].append(row)
    quota = max(1, max_count // max(1, len(by_category)))
    for cat_rows in by_category.values():
        step = max(1, len(cat_rows) // quota)
        selected.extend(cat_rows[::step][:quota])
    if len(selected) < max_count:
        seen = {(row["category"], row["threshold"], row["frame"]) for row in selected}
        for row in ordered:
            key = (row["category"], row["threshold"], row["frame"])
            if key not in seen:
                selected.append(row)
                seen.add(key)
            if len(selected) >= max_count:
                break
    return selected[:max_count]


def make_contact_sheet(strip_paths: Sequence[Path], out_path: Path, columns: int = 2) -> Optional[Path]:
    paths = [p for p in strip_paths if p.exists()]
    if not paths:
        return None
    images = [Image.open(p).convert("RGB") for p in paths]
    width = max(img.width for img in images)
    height = max(img.height for img in images)
    rows = int(np.ceil(len(images) / max(1, columns)))
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for idx, img in enumerate(images):
        x = (idx % columns) * width
        y = (idx // columns) * height
        sheet.paste(img, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    for img in images:
        img.close()
    return out_path


def write_galleries(
    audit_rows: Sequence[Mapping[str, object]],
    records_by_model: Mapping[str, Sequence[Top1PredictionRecord]],
    model_names: Sequence[str],
    output_root: Path,
    gallery_max: int,
    good_conf: float,
    bad_conf: float,
    bad_iou: float,
) -> List[Path]:
    out_paths: List[Path] = []
    root = output_root / "failure_gallery_contact_sheets"
    for category in AUDIT_CATEGORIES:
        cat_rows = [row for row in audit_rows if row["category"] == category]
        selected = select_diverse_rows(cat_rows, gallery_max)
        strip_paths: List[Path] = []
        for rank, row in enumerate(selected, start=1):
            image_path = Path(str(row["image_path"]))
            strip_path = root / category / f"{rank:03d}_frame_{int(row['frame']):06d}.png"
            save_gallery_strip(
                image_path=image_path,
                category=category,
                reason=str(row["reason"]),
                records_by_model=records_by_model,
                label_record=None,
                label_aware_by_model={},
                gallery_model_names=model_names,
                output_path=strip_path,
                good_conf=good_conf,
                bad_conf=bad_conf,
                bad_iou=bad_iou,
            )
            strip_paths.append(strip_path)
        sheet = make_contact_sheet(strip_paths, root / category / "contact_sheet.png", columns=1)
        if sheet is not None:
            out_paths.append(sheet)

    low_conf_rows = [row for row in audit_rows if row["category"] == "membank_low_conf_candidate"]
    low_root = output_root / "low_conf_correct_candidate_gallery"
    selected_low = select_diverse_rows(low_conf_rows, gallery_max)
    low_strips: List[Path] = []
    for rank, row in enumerate(selected_low, start=1):
        strip_path = low_root / f"{rank:03d}_frame_{int(row['frame']):06d}.png"
        save_gallery_strip(
            image_path=Path(str(row["image_path"])),
            category="membank_low_conf_candidate",
            reason=str(row["reason"]),
            records_by_model=records_by_model,
            label_record=None,
            label_aware_by_model={},
            gallery_model_names=model_names,
            output_path=strip_path,
            good_conf=good_conf,
            bad_conf=bad_conf,
            bad_iou=bad_iou,
        )
        low_strips.append(strip_path)
    sheet = make_contact_sheet(low_strips, low_root / "contact_sheet.png", columns=1)
    if sheet is not None:
        out_paths.append(sheet)
    return out_paths


def write_manual_audit_sheet(
    audit_rows: Sequence[Mapping[str, object]],
    records_by_model: Mapping[str, Sequence[Top1PredictionRecord]],
    output_root: Path,
    count: int,
) -> Path:
    selected = select_diverse_rows(audit_rows, count)
    rows: List[Dict[str, object]] = []
    for row in selected:
        idx = int(row["frame"])
        out = {
            "frame": idx,
            "image_path": row["image_path"],
            "category": row["category"],
            "fda_conf": conf_for(records_by_model, "fda_mix", idx),
            "current_odad_conf": conf_for(records_by_model, "current_odad", idx),
            "adapter_conf": conf_for(records_by_model, "adapter_base", idx),
            "membank_conf": conf_for(records_by_model, "membank_prev", idx),
            "source_anchor_conf": conf_for(records_by_model, "source_anchor", idx),
            "suggested_reason": row["reason"],
            "manual_label_blank": "",
        }
        rows.append(out)
    return write_csv(
        output_root / "manual_audit_sheet.csv",
        rows,
        [
            "frame",
            "image_path",
            "category",
            "fda_conf",
            "current_odad_conf",
            "adapter_conf",
            "membank_conf",
            "source_anchor_conf",
            "suggested_reason",
            "manual_label_blank",
        ],
    )


def conf_for(records_by_model: Mapping[str, Sequence[Top1PredictionRecord]], model: str, idx: int) -> str:
    if model not in records_by_model or idx >= len(records_by_model[model]):
        return ""
    return f"{records_by_model[model][idx].top1_conf:.4f}"


def write_summary(
    output_root: Path,
    sweep_rows: Sequence[Mapping[str, object]],
    audit_rows: Sequence[Mapping[str, object]],
    reference_model: str,
) -> Path:
    counts = Counter(str(row["category"]) for row in audit_rows)
    ref_rows = {round(float(row["confidence_threshold"]), 2): row for row in sweep_rows if row["model"] == reference_model}
    lines = ["Paired failure summary", ""]
    lines.append("Memory-bank threshold points:")
    for threshold in [0.25, 0.20, 0.15, 0.10, 0.05]:
        row = ref_rows.get(round(threshold, 2))
        if row:
            lines.append(
                f"  conf {threshold:.2f}: det={100*safe_float(row.get('det_rate')):.1f}% "
                f"weird={100*safe_float(row.get('weird_rate')):.1f}% "
                f"jump={100*safe_float(row.get('box_jump_rate')):.1f}% "
                f"bad={100*safe_float(row.get('bad_rate')):.1f}% "
                f"max_bad={safe_float(row.get('max_bad_streak'), 0):.0f}"
            )
    lines.extend(["", "Paired category counts:"])
    for category in AUDIT_CATEGORIES:
        lines.append(f"  {category}: {counts.get(category, 0)}")
    path = output_root / "paired_failure_summary.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_recommendation(output_root: Path, rows: Sequence[Mapping[str, object]], reference_model: str) -> Path:
    ref_rows = [row for row in rows if row["model"] == reference_model]
    candidates = [
        row
        for row in ref_rows
        if safe_float(row.get("det_rate")) >= 0.87
        and safe_float(row.get("weird_rate")) <= 0.12
        and safe_float(row.get("box_jump_rate")) <= 0.05
        and safe_float(row.get("max_bad_streak"), 9999.0) <= 100
    ]
    if candidates:
        best = max(candidates, key=lambda r: (safe_float(r.get("det_rate")), -safe_float(r.get("bad_rate"))))
        diagnosis = (
            f"Recommended operating point: conf={float(best['confidence_threshold']):.2f}. "
            "Lowering the threshold recovers the detection target while keeping proxy reliability within limits, "
            "so the main issue looks like confidence calibration/operating threshold rather than capability."
        )
    else:
        best = max(ref_rows, key=lambda r: (safe_float(r.get("det_rate")) - safe_float(r.get("bad_rate")))) if ref_rows else None
        if best is None:
            diagnosis = "No recommendation: no reference-model rows were produced."
        else:
            diagnosis = (
                f"No threshold met all deployability limits. Best observed compromise by det-bad score was "
                f"conf={float(best['confidence_threshold']):.2f}; remaining issue is not solved by threshold alone."
            )
    path = output_root / "operating_point_recommendation.txt"
    path.write_text(diagnosis + "\n")
    return path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    model_paths = parse_models(args.model)
    thresholds = parse_thresholds(args.thresholds)
    if args.reference_model not in model_paths:
        raise ValueError(f"--reference-model {args.reference_model} is not among --model names")

    _, device_str = resolve_device(args.device)
    image_paths = gather_images(Path(args.dataset).expanduser())
    if int(args.eval_images) > 0:
        image_paths = image_paths[: int(args.eval_images)]
    image_sizes = get_image_sizes(image_paths)

    records_by_model: Dict[str, List[Top1PredictionRecord]] = {}
    temporal_by_model: Dict[str, List] = {}
    for name, path in model_paths.items():
        utils.LOGGER.info("Loading %s from %s", name, path)
        model = YOLO(str(path))
        records = collect_top1_prediction_records(
            model_name=name,
            model=model,
            image_paths=image_paths,
            image_sizes=image_sizes,
            iou=float(args.iou),
            device_str=device_str,
            large_box_frac=float(args.large_box_frac),
            tiny_box_frac=float(args.tiny_box_frac),
            border_margin_frac=float(args.border_margin_frac),
        )
        records_by_model[name] = records
        temporal_by_model[name] = compute_temporal_prediction_records(
            records,
            good_conf=float(args.good_conf),
            bad_conf=float(args.bad_conf),
            box_jump_center_frac=float(args.box_jump_center_frac),
        )

    sweep_rows: List[Dict[str, object]] = []
    for threshold in thresholds:
        for name in model_paths:
            row = aggregate_proxy_stream_reliability(
                dataset_name="lab",
                dataset_label="Lab",
                model_name=name,
                model_label=display_name(name),
                records=records_by_model[name],
                temporal_records=temporal_by_model[name],
                conf_threshold=float(threshold),
                good_conf=float(args.good_conf),
            )
            sweep_rows.append(add_sweep_aliases(row, threshold=float(threshold)))

    sweep_fields = [
        "dataset",
        "model",
        "confidence_threshold",
        "n_images",
        "det_rate",
        "weird_rate",
        "high_conf_weird_rate",
        "box_jump_rate",
        "bad_rate",
        "max_bad_streak",
        "p95_bad_streak",
        "bad_streaks_over_50",
        "max_good_streak",
        "mean_top1_conf",
        "median_top1_conf",
        "low_conf_rate",
        "no_detection_rate",
    ]
    write_csv(output_root / "threshold_sweep_summary.csv", sweep_rows, sweep_fields)
    plot_det_vs_weird(sweep_rows, list(model_paths), output_root / "threshold_sweep_plot_det_vs_weird.png")
    plot_tradeoff(sweep_rows, args.reference_model, output_root / "threshold_sweep_plot_tradeoff.png")
    plot_streaks(sweep_rows, args.reference_model, output_root / "threshold_sweep_plot_streaks.png")

    audit_rows = build_paired_audit_rows(
        records_by_model=records_by_model,
        temporal_by_model=temporal_by_model,
        thresholds=thresholds,
        reference_model=args.reference_model,
        good_conf=float(args.good_conf),
    )
    write_csv(
        output_root / "paired_failure_audit.csv",
        audit_rows,
        [
            "threshold",
            "category",
            "frame",
            "image_path",
            "compare_model",
            "membank_conf",
            "membank_weird",
            "membank_border",
            "membank_tiny",
            "membank_large",
            "compare_conf",
            "compare_weird",
            "reason",
        ],
    )
    write_summary(output_root, sweep_rows, audit_rows, args.reference_model)
    write_recommendation(output_root, sweep_rows, args.reference_model)
    write_manual_audit_sheet(audit_rows, records_by_model, output_root, int(args.manual_audit_count))
    write_galleries(
        audit_rows,
        records_by_model=records_by_model,
        model_names=list(model_paths),
        output_root=output_root,
        gallery_max=int(args.gallery_max),
        good_conf=float(args.good_conf),
        bad_conf=float(args.bad_conf),
        bad_iou=float(args.bad_iou),
    )
    (output_root / "run_metadata.txt").write_text(
        "\n".join(
            [
                f"dataset={Path(args.dataset).expanduser()}",
                f"n_images={len(image_paths)}",
                f"device={device_str}",
                f"infer_conf={float(args.infer_conf):.4f}",
                f"iou={float(args.iou):.3f}",
                f"thresholds={','.join(f'{t:.2f}' for t in thresholds)}",
                f"models={','.join(model_paths)}",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
