#!/usr/bin/env python3
"""Static and adaptive confidence-threshold policy eval for ODAD YOLO checkpoints."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from dual_yolo_mae import utils  # noqa: E402
from yolo_ablation import (  # noqa: E402
    Top1PredictionRecord,
    box_iou_xyxy,
    build_model_color_map,
    center_shift_frac,
    collect_top1_prediction_records,
    compute_temporal_prediction_records,
    display_name,
    failure_streak_lengths,
    gather_images,
    get_image_sizes,
    load_gallery_font,
    prediction_is_weird,
    resolve_device,
    safe_float,
    safe_mean,
    safe_percentile,
    safe_rate,
    save_gallery_strip,
    sanitize_filename,
    symmetric_area_ratio,
)


STATIC_POLICY_THRESHOLDS = {
    "static_025": 0.25,
    "static_020": 0.20,
    "static_015": 0.15,
    "static_010": 0.10,
    "static_005": 0.05,
}


@dataclass
class PolicyDecision:
    accepted: bool
    threshold_used: float
    low_conf_accept: bool
    reason: str
    temporal_iou: float = float("nan")
    center_shift_frac: float = float("nan")
    area_ratio: float = float("nan")
    box_jump: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="YOLO dataset root with images/test")
    parser.add_argument("--weights", required=True, help="Model weights to evaluate")
    parser.add_argument("--model-name", default="membank_bank16_top2_teacherroi_full")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--infer-conf", type=float, default=0.001, help="Recorded for metadata; top-1 pass uses 0.001.")
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--eval-images", type=int, default=0, help="0 means all images")
    parser.add_argument("--ordered-eval", action="store_true", help="Keep sorted stream order; default behavior.")
    parser.add_argument(
        "--policies",
        default="static_025,static_020,static_015,static_010,static_005,"
        "adaptive_025_to_015_stable,adaptive_025_to_005_strict,hysteresis_policy",
    )
    parser.add_argument("--gallery-max", type=int, default=24)
    parser.add_argument("--manual-review-count", type=int, default=120)
    parser.add_argument("--good-conf", type=float, default=0.75)
    parser.add_argument("--bad-conf", type=float, default=0.25)
    parser.add_argument("--large-box-frac", type=float, default=0.30)
    parser.add_argument("--tiny-box-frac", type=float, default=0.001)
    parser.add_argument("--border-margin-frac", type=float, default=0.02)
    parser.add_argument("--box-jump-center-frac", type=float, default=0.20)
    parser.add_argument("--stable-center-frac", type=float, default=0.20)
    parser.add_argument("--strict-center-frac", type=float, default=0.14)
    parser.add_argument("--stable-min-iou", type=float, default=0.05)
    parser.add_argument("--strict-min-iou", type=float, default=0.10)
    parser.add_argument("--stable-area-ratio", type=float, default=4.0)
    parser.add_argument("--strict-area-ratio", type=float, default=2.5)
    return parser.parse_args()


def parse_csv_names(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def record_pair_motion(prev: Optional[Top1PredictionRecord], curr: Top1PredictionRecord) -> Tuple[float, float, float]:
    if prev is None:
        return float("nan"), float("nan"), float("nan")
    iou = box_iou_xyxy(prev.top1_xyxy, curr.top1_xyxy)
    shift = center_shift_frac(prev, curr)
    area_ratio = symmetric_area_ratio(prev.box_area_frac, curr.box_area_frac)
    return iou, shift, area_ratio


def finite_le(value: float, limit: float) -> bool:
    return bool(np.isfinite(value) and value <= float(limit))


def finite_ge(value: float, limit: float) -> bool:
    return bool(np.isfinite(value) and value >= float(limit))


def temporal_gate(
    prev_accepted: Optional[Top1PredictionRecord],
    curr: Top1PredictionRecord,
    min_iou: float,
    max_center: float,
    max_area_ratio: float,
) -> Tuple[bool, float, float, float, str]:
    if prev_accepted is None:
        return False, float("nan"), float("nan"), float("nan"), "no_previous_accepted_box"
    iou, shift, area_ratio = record_pair_motion(prev_accepted, curr)
    gates = [
        finite_ge(iou, min_iou),
        finite_le(shift, max_center),
        finite_le(area_ratio, max_area_ratio),
    ]
    reason = f"prev_iou={iou:.3f}; center_shift={shift:.3f}; area_ratio={area_ratio:.2f}"
    return bool(all(gates)), iou, shift, area_ratio, reason


def next_context_gate(
    records: Sequence[Top1PredictionRecord],
    idx: int,
    curr: Top1PredictionRecord,
    min_iou: float,
    max_center: float,
    max_area_ratio: float,
) -> Tuple[bool, str]:
    if idx + 1 >= len(records):
        return False, "no_next_frame"
    nxt = records[idx + 1]
    if (not nxt.has_detection) or prediction_is_weird(nxt) or nxt.top1_conf < 0.15:
        return False, f"next_unstable(conf={nxt.top1_conf:.3f}, weird={int(prediction_is_weird(nxt))})"
    iou, shift, area_ratio = record_pair_motion(curr, nxt)
    passed = finite_ge(iou, min_iou) and finite_le(shift, max_center) and finite_le(area_ratio, max_area_ratio)
    return passed, f"next_iou={iou:.3f}; next_center_shift={shift:.3f}; next_area_ratio={area_ratio:.2f}"


def static_decisions(records: Sequence[Top1PredictionRecord], threshold: float, box_jump_center_frac: float) -> List[PolicyDecision]:
    decisions: List[PolicyDecision] = []
    prev_accepted: Optional[Top1PredictionRecord] = None
    for record in records:
        accepted = bool(record.has_detection and record.top1_conf >= float(threshold))
        iou, shift, area_ratio = record_pair_motion(prev_accepted, record) if accepted else (float("nan"), float("nan"), float("nan"))
        box_jump = bool(accepted and prev_accepted is not None and finite_le(0.0, shift) and shift >= float(box_jump_center_frac))
        reason = f"conf {record.top1_conf:.3f} >= {threshold:.2f}" if accepted else f"conf {record.top1_conf:.3f} < {threshold:.2f}"
        decisions.append(
            PolicyDecision(
                accepted=accepted,
                threshold_used=float(threshold),
                low_conf_accept=bool(accepted and record.top1_conf < 0.25),
                reason=reason,
                temporal_iou=iou,
                center_shift_frac=shift,
                area_ratio=area_ratio,
                box_jump=box_jump,
            )
        )
        if accepted:
            prev_accepted = record
    return decisions


def adaptive_decisions(
    records: Sequence[Top1PredictionRecord],
    policy: str,
    box_jump_center_frac: float,
    stable_min_iou: float,
    strict_min_iou: float,
    stable_center_frac: float,
    strict_center_frac: float,
    stable_area_ratio: float,
    strict_area_ratio: float,
) -> List[PolicyDecision]:
    decisions: List[PolicyDecision] = []
    prev_accepted: Optional[Top1PredictionRecord] = None
    stable_run = 0
    in_stable_segment = False
    for idx, record in enumerate(records):
        accepted = False
        threshold_used = 0.25
        low_conf_accept = False
        reason = ""
        motion_iou = float("nan")
        motion_shift = float("nan")
        motion_area = float("nan")

        if record.has_detection and record.top1_conf >= 0.25:
            accepted = True
            reason = f"base_accept(conf={record.top1_conf:.3f})"
        elif policy == "adaptive_025_to_015_stable" and 0.15 <= record.top1_conf < 0.25:
            threshold_used = 0.15
            if prediction_is_weird(record):
                reason = "reject_low_conf_weird_or_border"
            else:
                gate, motion_iou, motion_shift, motion_area, gate_reason = temporal_gate(
                    prev_accepted, record, stable_min_iou, stable_center_frac, stable_area_ratio
                )
                accepted = gate
                reason = ("stable_low_conf_accept; " if gate else "stable_low_conf_reject; ") + gate_reason
        elif policy == "adaptive_025_to_005_strict" and 0.05 <= record.top1_conf < 0.25:
            threshold_used = 0.05
            if prediction_is_weird(record):
                reason = "reject_low_conf_weird_or_border"
            else:
                gate, motion_iou, motion_shift, motion_area, gate_reason = temporal_gate(
                    prev_accepted, record, strict_min_iou, strict_center_frac, strict_area_ratio
                )
                next_gate, next_reason = next_context_gate(
                    records, idx, record, strict_min_iou, strict_center_frac, strict_area_ratio
                )
                accepted = bool(gate and next_gate)
                reason = ("strict_low_conf_accept; " if accepted else "strict_low_conf_reject; ") + gate_reason + "; " + next_reason
        elif policy == "hysteresis_policy" and 0.15 <= record.top1_conf < 0.25:
            threshold_used = 0.15
            if (not in_stable_segment) or prediction_is_weird(record):
                reason = f"hysteresis_reject(in_stable={int(in_stable_segment)}, weird={int(prediction_is_weird(record))})"
            else:
                gate, motion_iou, motion_shift, motion_area, gate_reason = temporal_gate(
                    prev_accepted, record, stable_min_iou, stable_center_frac, stable_area_ratio
                )
                accepted = gate
                reason = ("hysteresis_low_conf_accept; " if gate else "hysteresis_low_conf_reject; ") + gate_reason
        else:
            reason = f"below_policy_floor(conf={record.top1_conf:.3f})"

        if accepted:
            if not np.isfinite(motion_shift):
                motion_iou, motion_shift, motion_area = record_pair_motion(prev_accepted, record)
            box_jump = bool(prev_accepted is not None and np.isfinite(motion_shift) and motion_shift >= float(box_jump_center_frac))
            low_conf_accept = bool(record.top1_conf < 0.25)
        else:
            box_jump = False

        decisions.append(
            PolicyDecision(
                accepted=accepted,
                threshold_used=threshold_used,
                low_conf_accept=low_conf_accept,
                reason=reason,
                temporal_iou=motion_iou,
                center_shift_frac=motion_shift,
                area_ratio=motion_area,
                box_jump=box_jump,
            )
        )

        if accepted:
            if prediction_is_weird(record) or box_jump:
                stable_run = 0
                in_stable_segment = False
            else:
                stable_run += 1
                in_stable_segment = stable_run >= 3
            prev_accepted = record
        else:
            stable_run = 0
            in_stable_segment = False
    return decisions


def evaluate_policy(
    policy: str,
    records: Sequence[Top1PredictionRecord],
    decisions: Sequence[PolicyDecision],
    raw_box_jump_flags: Sequence[bool],
    good_conf: float,
) -> Dict[str, object]:
    accepted_flags = [decision.accepted for decision in decisions]
    low_conf_accept_flags = [decision.low_conf_accept for decision in decisions]
    raw_weird_flags = [prediction_is_weird(record) for record in records]
    accepted_weird_flags = [decision.accepted and prediction_is_weird(record) for record, decision in zip(records, decisions)]
    high_conf_weird_flags = [
        decision.accepted and record.top1_conf >= float(good_conf) and prediction_is_weird(record)
        for record, decision in zip(records, decisions)
    ]
    box_jump_flags = [bool(flag) for flag in raw_box_jump_flags]
    bad_flags = [
        (not decision.accepted) or prediction_is_weird(record) or bool(raw_jump)
        for record, decision, raw_jump in zip(records, decisions, box_jump_flags)
    ]
    good_flags = [
        decision.accepted and (not prediction_is_weird(record)) and (not bool(raw_jump))
        for record, decision, raw_jump in zip(records, decisions, box_jump_flags)
    ]
    bad_streaks = failure_streak_lengths(bad_flags)
    good_streaks = failure_streak_lengths(good_flags)
    accepted_confs = [record.top1_conf for record, decision in zip(records, decisions) if decision.accepted]
    low_conf_accept_weird = [
        prediction_is_weird(record) for record, decision in zip(records, decisions) if decision.low_conf_accept
    ]
    low_conf_accept_jump = [
        bool(raw_jump) for decision, raw_jump in zip(decisions, box_jump_flags) if decision.low_conf_accept
    ]
    jump_denominator = box_jump_flags[1:] if len(box_jump_flags) > 1 else box_jump_flags
    return {
        "policy": policy,
        "n_images": len(records),
        "accepted_count": int(sum(accepted_flags)),
        "det_rate": safe_rate(accepted_flags),
        "weird_rate": safe_rate(raw_weird_flags),
        "accepted_weird_rate": safe_rate(accepted_weird_flags),
        "high_conf_weird_rate": safe_rate(high_conf_weird_flags),
        "box_jump_rate": safe_rate(jump_denominator),
        "bad_rate": safe_rate(bad_flags),
        "good_rate": safe_rate(good_flags),
        "max_bad_streak": int(max(bad_streaks) if bad_streaks else 0),
        "p95_bad_streak": safe_percentile([float(v) for v in bad_streaks], 95),
        "bad_streaks_over_50": int(sum(length > 50 for length in bad_streaks)),
        "max_good_streak": int(max(good_streaks) if good_streaks else 0),
        "mean_conf_accepted": safe_mean(accepted_confs),
        "fraction_low_conf_accepts": safe_rate(low_conf_accept_flags),
        "low_conf_accept_weird_rate": safe_rate(low_conf_accept_weird),
        "low_conf_accept_jump_rate": safe_rate(low_conf_accept_jump),
    }


def build_decisions_for_policy(
    policy: str,
    records: Sequence[Top1PredictionRecord],
    args: argparse.Namespace,
) -> List[PolicyDecision]:
    if policy in STATIC_POLICY_THRESHOLDS:
        return static_decisions(records, STATIC_POLICY_THRESHOLDS[policy], float(args.box_jump_center_frac))
    if policy in {"adaptive_025_to_015_stable", "adaptive_025_to_005_strict", "hysteresis_policy"}:
        return adaptive_decisions(
            records,
            policy=policy,
            box_jump_center_frac=float(args.box_jump_center_frac),
            stable_min_iou=float(args.stable_min_iou),
            strict_min_iou=float(args.strict_min_iou),
            stable_center_frac=float(args.stable_center_frac),
            strict_center_frac=float(args.strict_center_frac),
            stable_area_ratio=float(args.stable_area_ratio),
            strict_area_ratio=float(args.strict_area_ratio),
        )
    raise ValueError(f"Unknown policy '{policy}'")


def plot_tradeoff(rows: Sequence[Mapping[str, object]], out_path: Path) -> Path:
    policies = [str(row["policy"]) for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for key, label in [
        ("det_rate", "det"),
        ("weird_rate", "weird"),
        ("accepted_weird_rate", "accepted weird"),
        ("box_jump_rate", "box jump"),
        ("bad_rate", "bad"),
    ]:
        ax.plot(x, [safe_float(row.get(key)) * 100.0 for row in rows], marker="o", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=25, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title("ODAD confidence policy tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_streaks(rows: Sequence[Mapping[str, object]], out_path: Path) -> Path:
    policies = [str(row["policy"]) for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(x, [safe_float(row.get("max_bad_streak")) for row in rows], marker="o", label="max bad")
    ax.plot(x, [safe_float(row.get("p95_bad_streak")) for row in rows], marker="o", label="p95 bad")
    ax.plot(x, [safe_float(row.get("max_good_streak")) for row in rows], marker="o", label="max good")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=25, ha="right")
    ax.set_ylabel("Frames")
    ax.set_title("ODAD confidence policy streaks")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def make_contact_sheet(strip_paths: Sequence[Path], out_path: Path, columns: int = 1) -> Optional[Path]:
    paths = [p for p in strip_paths if p.exists()]
    if not paths:
        return None
    images = [Image.open(p).convert("RGB") for p in paths]
    width = max(img.width for img in images)
    height = max(img.height for img in images)
    rows = int(np.ceil(len(images) / max(1, columns)))
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for idx, img in enumerate(images):
        sheet.paste(img, ((idx % columns) * width, (idx // columns) * height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    for img in images:
        img.close()
    return out_path


def diverse_indices(indices: Sequence[int], max_count: int) -> List[int]:
    ordered = sorted(set(int(idx) for idx in indices))
    if max_count <= 0 or len(ordered) <= max_count:
        return ordered[: max(0, max_count)]
    step = max(1, len(ordered) // max_count)
    selected = ordered[::step][:max_count]
    if len(selected) < max_count:
        seen = set(selected)
        for idx in ordered:
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
            if len(selected) >= max_count:
                break
    return selected


def write_galleries(
    records: Sequence[Top1PredictionRecord],
    decisions_by_policy: Mapping[str, Sequence[PolicyDecision]],
    raw_box_jump_flags: Sequence[bool],
    output_root: Path,
    model_name: str,
    gallery_max: int,
    good_conf: float,
    bad_conf: float,
) -> None:
    records_by_model = {model_name: list(records)}
    low_root = output_root / "low_conf_accept_gallery"
    fail_root = output_root / "policy_failure_gallery"
    for policy, decisions in decisions_by_policy.items():
        low_indices = [
            idx
            for idx, (record, decision) in enumerate(zip(records, decisions))
            if decision.low_conf_accept and not prediction_is_weird(record) and not bool(raw_box_jump_flags[idx])
        ]
        fail_indices = [
            idx
            for idx, (record, decision) in enumerate(zip(records, decisions))
            if (not decision.accepted) or prediction_is_weird(record) or bool(raw_box_jump_flags[idx])
        ]
        for root, category, indices in [
            (low_root, "low_conf_accept", low_indices),
            (fail_root, "policy_failure", fail_indices),
        ]:
            strip_paths: List[Path] = []
            for rank, idx in enumerate(diverse_indices(indices, gallery_max), start=1):
                record = records[idx]
                decision = decisions[idx]
                strip_path = root / sanitize_filename(policy) / f"{rank:03d}_frame_{idx:06d}.png"
                reason = (
                    f"policy={policy}; accepted={int(decision.accepted)}; low_conf={int(decision.low_conf_accept)}; "
                    f"conf={record.top1_conf:.3f}; weird={int(prediction_is_weird(record))}; "
                    f"jump={int(bool(raw_box_jump_flags[idx]))}; gate_jump={int(decision.box_jump)}; {decision.reason}"
                )
                save_gallery_strip(
                    image_path=record.image_path,
                    category=category,
                    reason=reason,
                    records_by_model=records_by_model,
                    label_record=None,
                    label_aware_by_model={},
                    gallery_model_names=[model_name],
                    output_path=strip_path,
                    good_conf=good_conf,
                    bad_conf=bad_conf,
                    bad_iou=0.20,
                )
                strip_paths.append(strip_path)
            make_contact_sheet(strip_paths, root / sanitize_filename(policy) / "contact_sheet.png", columns=1)


def write_manual_review_sheet(
    records: Sequence[Top1PredictionRecord],
    decisions_by_policy: Mapping[str, Sequence[PolicyDecision]],
    output_root: Path,
    count: int,
) -> Path:
    focus_policies = [
        policy
        for policy in decisions_by_policy
        if policy == "static_015" or policy.startswith("adaptive_") or policy == "hysteresis_policy"
    ]
    rows: List[Dict[str, object]] = []
    for policy in focus_policies:
        for idx, (record, decision) in enumerate(zip(records, decisions_by_policy[policy])):
            if not decision.low_conf_accept:
                continue
            xyxy = record.top1_xyxy or (float("nan"), float("nan"), float("nan"), float("nan"))
            rows.append(
                {
                    "policy": policy,
                    "frame": idx,
                    "image_path": str(record.image_path),
                    "conf": f"{record.top1_conf:.4f}",
                    "x1": f"{xyxy[0]:.2f}",
                    "y1": f"{xyxy[1]:.2f}",
                    "x2": f"{xyxy[2]:.2f}",
                    "y2": f"{xyxy[3]:.2f}",
                    "box_area_frac": f"{record.box_area_frac:.6f}",
                    "weird": int(prediction_is_weird(record)),
                    "border": int(record.border_touching),
                    "tiny": int(record.tiny_box),
                    "large": int(record.large_box),
                    "threshold_used": f"{decision.threshold_used:.2f}",
                    "reason_accepted": decision.reason,
                    "manual_label_blank": "",
                }
            )
    rows = sorted(rows, key=lambda r: (str(r["policy"]), int(r["frame"])))
    selected_rows = [rows[idx] for idx in diverse_indices(list(range(len(rows))), count)]
    return write_csv(
        output_root / "policy_manual_review_sheet.csv",
        selected_rows,
        [
            "policy",
            "frame",
            "image_path",
            "conf",
            "x1",
            "y1",
            "x2",
            "y2",
            "box_area_frac",
            "weird",
            "border",
            "tiny",
            "large",
            "threshold_used",
            "reason_accepted",
            "manual_label_blank",
        ],
    )


def write_summary(output_root: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    lines = ["ODAD threshold policy summary", ""]
    lines.append(
        "policy, det, weird, accepted_weird, jump, bad, max_bad, p95_bad, low_conf_accepts, low_conf_weird, low_conf_jump"
    )
    for row in rows:
        lines.append(
            f"{row['policy']}: det={100*safe_float(row.get('det_rate')):.1f}% "
            f"weird={100*safe_float(row.get('weird_rate')):.1f}% "
            f"accepted_weird={100*safe_float(row.get('accepted_weird_rate')):.1f}% "
            f"jump={100*safe_float(row.get('box_jump_rate')):.1f}% "
            f"bad={100*safe_float(row.get('bad_rate')):.1f}% "
            f"max_bad={safe_float(row.get('max_bad_streak'), 0):.0f} "
            f"p95_bad={safe_float(row.get('p95_bad_streak'), 0):.1f} "
            f"low_accept={100*safe_float(row.get('fraction_low_conf_accepts')):.1f}% "
            f"low_weird={100*safe_float(row.get('low_conf_accept_weird_rate')):.1f}% "
            f"low_jump={100*safe_float(row.get('low_conf_accept_jump_rate')):.1f}%"
        )
    path = output_root / "policy_summary.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_recommendation(output_root: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    candidates = [
        row
        for row in rows
        if safe_float(row.get("det_rate")) >= 0.87
        and safe_float(row.get("weird_rate")) <= 0.10
        and safe_float(row.get("box_jump_rate")) <= 0.05
        and safe_float(row.get("max_bad_streak"), 9999.0) <= 100
    ]
    strong = [
        row
        for row in candidates
        if safe_float(row.get("det_rate")) >= 0.90 and safe_float(row.get("max_bad_streak"), 9999.0) <= 80
    ]
    static_015 = next((row for row in rows if row["policy"] == "static_015"), None)
    adaptive_candidates = [row for row in candidates if str(row["policy"]).startswith("adaptive_") or row["policy"] == "hysteresis_policy"]
    if adaptive_candidates and static_015 is not None:
        best_adaptive = max(adaptive_candidates, key=lambda r: (safe_float(r.get("det_rate")), -safe_float(r.get("bad_rate"))))
        if safe_float(best_adaptive.get("det_rate")) > safe_float(static_015.get("det_rate")) + 1e-6:
            best = best_adaptive
            diagnosis = "Recommend adaptive thresholding because it beats static_015 on detection while preserving reliability gates."
        else:
            best = static_015
            diagnosis = "Recommend static_015 because adaptive policies did not beat its detection/reliability tradeoff."
    elif candidates:
        best = max(candidates, key=lambda r: (safe_float(r.get("det_rate")), -safe_float(r.get("bad_rate"))))
        diagnosis = "Recommend the best policy meeting minimum reliability criteria."
    else:
        best = max(rows, key=lambda r: (safe_float(r.get("det_rate")) - safe_float(r.get("bad_rate")))) if rows else None
        diagnosis = "No policy met all minimum criteria; threshold policy alone is not deployment-ready."

    lines = ["Operating point recommendation", ""]
    if best is not None:
        lines.append(
            f"recommended_policy={best['policy']} det={safe_float(best.get('det_rate')):.4f} "
            f"weird={safe_float(best.get('weird_rate')):.4f} box_jump={safe_float(best.get('box_jump_rate')):.4f} "
            f"bad={safe_float(best.get('bad_rate')):.4f} max_bad={safe_float(best.get('max_bad_streak'), 0):.0f}"
        )
    lines.append(f"minimum_criteria_met={int(bool(candidates))}")
    lines.append(f"strong_criteria_met={int(bool(strong))}")
    lines.append(diagnosis)
    path = output_root / "operating_point_recommendation.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_metadata(output_root: Path, args: argparse.Namespace, n_images: int, device_str: str, policies: Sequence[str]) -> Path:
    path = output_root / "run_metadata.txt"
    path.write_text(
        "\n".join(
            [
                f"dataset={Path(args.dataset).expanduser()}",
                f"weights={Path(args.weights).expanduser()}",
                f"model_name={args.model_name}",
                f"n_images={n_images}",
                f"device={device_str}",
                f"infer_conf={float(args.infer_conf):.4f}",
                f"iou={float(args.iou):.3f}",
                f"policies={','.join(policies)}",
            ]
        )
        + "\n"
    )
    return path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    weights = Path(args.weights).expanduser()
    if not weights.exists():
        raise FileNotFoundError(f"Weights path does not exist: {weights}")
    policies = parse_csv_names(args.policies)

    _, device_str = resolve_device(args.device)
    image_paths = gather_images(Path(args.dataset).expanduser())
    if int(args.eval_images) > 0:
        image_paths = image_paths[: int(args.eval_images)]
    image_sizes = get_image_sizes(image_paths)

    utils.LOGGER.info("Loading %s from %s", args.model_name, weights)
    model = YOLO(str(weights))
    records = collect_top1_prediction_records(
        model_name=str(args.model_name),
        model=model,
        image_paths=image_paths,
        image_sizes=image_sizes,
        iou=float(args.iou),
        device_str=device_str,
        large_box_frac=float(args.large_box_frac),
        tiny_box_frac=float(args.tiny_box_frac),
        border_margin_frac=float(args.border_margin_frac),
    )
    raw_temporal_records = compute_temporal_prediction_records(
        records,
        good_conf=float(args.good_conf),
        bad_conf=float(args.bad_conf),
        box_jump_center_frac=float(args.box_jump_center_frac),
    )
    raw_box_jump_flags = [bool(record.box_jump) for record in raw_temporal_records]

    decisions_by_policy: Dict[str, List[PolicyDecision]] = {}
    summary_rows: List[Dict[str, object]] = []
    for policy in policies:
        decisions = build_decisions_for_policy(policy, records, args)
        decisions_by_policy[policy] = decisions
        summary_rows.append(
            evaluate_policy(
                policy,
                records,
                decisions,
                raw_box_jump_flags=raw_box_jump_flags,
                good_conf=float(args.good_conf),
            )
        )

    fields = [
        "policy",
        "n_images",
        "accepted_count",
        "det_rate",
        "weird_rate",
        "accepted_weird_rate",
        "high_conf_weird_rate",
        "box_jump_rate",
        "bad_rate",
        "max_bad_streak",
        "p95_bad_streak",
        "bad_streaks_over_50",
        "max_good_streak",
        "mean_conf_accepted",
        "fraction_low_conf_accepts",
        "low_conf_accept_weird_rate",
        "low_conf_accept_jump_rate",
    ]
    write_csv(output_root / "policy_summary.csv", summary_rows, fields)
    write_summary(output_root, summary_rows)
    write_recommendation(output_root, summary_rows)
    plot_tradeoff(summary_rows, output_root / "policy_tradeoff_plot.png")
    plot_streaks(summary_rows, output_root / "policy_streak_plot.png")
    write_galleries(
        records,
        decisions_by_policy=decisions_by_policy,
        raw_box_jump_flags=raw_box_jump_flags,
        output_root=output_root,
        model_name=str(args.model_name),
        gallery_max=int(args.gallery_max),
        good_conf=float(args.good_conf),
        bad_conf=float(args.bad_conf),
    )
    write_manual_review_sheet(records, decisions_by_policy, output_root, int(args.manual_review_count))
    write_metadata(output_root, args, len(records), device_str, policies)


if __name__ == "__main__":
    main()
