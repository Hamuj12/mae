#!/usr/bin/env python3
"""Compact adapter sweep summaries from ODAD output directories."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def parse_labeled_path(value: str) -> Tuple[Optional[str], Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip() or None, Path(path)
    return None, Path(value)


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def read_summary(path: Path) -> Dict[str, str]:
    summary_path = path / "summary.txt" if path.is_dir() else path
    data: Dict[str, str] = {}
    if not summary_path.exists():
        data["missing_summary"] = str(summary_path)
        return data
    for raw_line in summary_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def read_reliability_rows(path: Path) -> List[Dict[str, str]]:
    if path.is_dir():
        for name in ("full_stream_reliability_summary.csv", "reliability_summary.csv"):
            candidate = path / name
            if candidate.exists():
                path = candidate
                break
    if not path.exists() or path.is_dir():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def run_label(default_label: Optional[str], path: Path, summary: Mapping[str, str]) -> str:
    if default_label:
        return default_label
    layers = summary.get("adapter_layers", "")
    reduction = summary.get("adapter_reduction", "")
    scale = summary.get("adapter_scale", "")
    if layers and layers != "n/a":
        bits = [f"l{layers.replace(',', '_l')}"]
        if reduction:
            bits.append(f"r{reduction}")
        if scale:
            bits.append(f"s{safe_float(scale):g}")
        if summary.get("adapter_train_detect_head") == "1":
            bits.append("plus_head")
        return "_".join(bits)
    return path.name


def pct(value: object) -> str:
    number = safe_float(value)
    return "n/a" if not math.isfinite(number) else f"{100.0 * number:.1f}"


def num(value: object, digits: int = 1) -> str:
    number = safe_float(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def integer(value: object) -> str:
    number = safe_float(value)
    return "n/a" if not math.isfinite(number) else str(int(round(number)))


def choose_candidate_row(rows: Sequence[Mapping[str, str]], preferred_model: Optional[str]) -> Optional[Mapping[str, str]]:
    if not rows:
        return None
    if preferred_model:
        for row in rows:
            if str(row.get("model", "")) == preferred_model:
                return row
    for row in rows:
        model = str(row.get("model", ""))
        if model not in {"fda_mix", "current_odad"}:
            return row
    return rows[-1]


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact markdown table for adapter sweep outputs.")
    parser.add_argument("--runs", nargs="+", required=True, help="online_adapt output dirs. Use label=path to set config.")
    parser.add_argument(
        "--reliability",
        nargs="*",
        default=[],
        help="Optional yolo_ablation output dirs/CSVs. Use label=path to map to a run label.",
    )
    parser.add_argument("--candidate-model", default=None, help="Preferred reliability model row, e.g. adapter_candidate.")
    args = parser.parse_args()

    reliability_by_label: Dict[str, Mapping[str, str]] = {}
    unlabeled_reliability: List[Mapping[str, str]] = []
    for value in args.reliability:
        label, path = parse_labeled_path(value)
        row = choose_candidate_row(read_reliability_rows(path), args.candidate_model)
        if row is None:
            continue
        if label:
            reliability_by_label[label] = row
        else:
            unlabeled_reliability.append(row)

    table_rows: List[List[str]] = []
    for idx, value in enumerate(args.runs):
        explicit_label, path = parse_labeled_path(value)
        summary = read_summary(path)
        label = run_label(explicit_label, path, summary)
        rel = reliability_by_label.get(label)
        if rel is None and len(args.runs) == 1 and unlabeled_reliability:
            rel = unlabeled_reliability[0]
        elif rel is None and idx < len(unlabeled_reliability):
            rel = unlabeled_reliability[idx]

        table_rows.append(
            [
                label,
                integer(summary.get("accepted_frames")),
                integer(summary.get("optimizer_update_steps")),
                integer(summary.get("trainable_params")),
                integer(summary.get("memory_adapter_updates")),
                integer(summary.get("memory_bank_active_slots")),
                integer(summary.get("memory_bank_writes")),
                integer(summary.get("memory_bank_replacements")),
                integer(summary.get("memory_bank_duplicate_skips")),
                summary.get("memory_slot_scale_bin_counts", "n/a"),
                summary.get("memory_slot_conf_bin_counts", "n/a"),
                num(summary.get("memory_retrieval_top1_sim"), 3),
                num(summary.get("memory_retrieval_mean_topk_sim"), 3),
                num(summary.get("memory_slot_mean_pairwise_sim"), 3),
                num(summary.get("memory_slot_max_pairwise_sim"), 3),
                summary.get("memory_adapter_initialized", "n/a"),
                num(summary.get("memory_adapter_mean_norm"), 3),
                integer(summary.get("memory_adapter_trainable_params")),
                num(summary.get("mean_memory_conditioning_norm"), 3),
                num(summary.get("mean_source_memory_loss_updates"), 3),
                integer(summary.get("source_memory_valid_entries")),
                integer(summary.get("source_memory_skipped_updates")),
                num(summary.get("source_memory_mean_pos_sim"), 3),
                num(summary.get("source_memory_mean_neg_sim"), 3),
                num(summary.get("source_memory_margin"), 3),
                integer(summary.get("source_memory_projection_params")),
                integer(summary.get("coverage_entries_added")),
                integer(summary.get("coverage_entries_sampled")),
                integer(summary.get("coverage_region_entries_skipped")),
                num(summary.get("mean_coverage_region_student_conf"), 3),
                num(summary.get("mean_coverage_region_conf_gap"), 3),
                num(summary.get("mean_detection_loss_updates"), 3),
                num(summary.get("mean_coverage_loss_updates"), 3),
                summary.get("coverage_loss_type", "n/a"),
                num(summary.get("coverage_weight"), 2),
                num(summary.get("mean_update_latency_ms"), 1),
                num(summary.get("mean_pre_update_sync_latency_ms"), 1),
                num(summary.get("mean_batch_build_latency_ms"), 1),
                num(summary.get("mean_update_forward_latency_ms"), 1),
                num(summary.get("mean_update_loss_latency_ms"), 1),
                num(summary.get("mean_update_backward_step_latency_ms"), 1),
                num(summary.get("peak_cuda_reserved_mb"), 0),
                num(summary.get("final_checkpoint_size_mb"), 3),
                summary.get("checkpoint_reload_status", "n/a"),
                pct(rel.get("det_rate_at_conf") if rel else None),
                pct(rel.get("weird_box_rate") if rel else None),
                pct((rel.get("high_conf_weird_rate") or rel.get("high_conf_weird_box_rate")) if rel else None),
                pct(rel.get("box_jump_rate") if rel else None),
                integer(rel.get("max_bad_streak") if rel else None),
                integer(rel.get("max_good_streak") if rel else None),
            ]
        )

    print(
        markdown_table(
            [
                "config",
                "accepted",
                "updates",
                "params",
                "mem upd",
                "slots",
                "writes",
                "replace",
                "dup skip",
                "scale bins",
                "conf bins",
                "top1 sim",
                "topk sim",
                "slot mean",
                "slot max",
                "mem init",
                "mem norm",
                "mem params",
                "mem cond",
                "src loss",
                "src valid",
                "src skip",
                "pos sim",
                "neg sim",
                "margin",
                "proj params",
                "cov added",
                "cov sampled",
                "cov skip",
                "reg stud",
                "reg gap",
                "det loss",
                "cov loss",
                "cov type",
                "cov w",
                "upd ms",
                "pre-sync",
                "batch",
                "fwd",
                "loss",
                "bwd+step",
                "peak MB",
                "ckpt MB",
                "reload",
                "Det",
                "weird",
                "HC weird",
                "jump",
                "max bad",
                "max good",
            ],
            table_rows,
        )
    )


if __name__ == "__main__":
    main()
