#!/usr/bin/env python3
"""Build compact ODAD threshold-policy reporting artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "yolo_ablation_runs" / "membank_threshold_policy_eval"
OUT_DIR = ROOT / "odad" / "reporting" / "threshold_policy_summary"


COMPARISON_ROWS = [
    {
        "model": "FDA-mix",
        "policy": "static_025",
        "conf": "0.25",
        "det_rate": 0.982,
        "weird_rate": 0.215,
        "box_jump_rate": 0.019,
        "bad_rate": 0.235,
        "max_bad_streak": 41,
        "note": "High detection baseline; reliability proxy failures remain elevated.",
    },
    {
        "model": "Current ODAD",
        "policy": "static_025",
        "conf": "0.25",
        "det_rate": 0.870,
        "weird_rate": 0.363,
        "box_jump_rate": 0.091,
        "bad_rate": 0.459,
        "max_bad_streak": 198,
        "note": "Existing ODAD baseline at default threshold.",
    },
    {
        "model": "Adapter Base",
        "policy": "static_025",
        "conf": "0.25",
        "det_rate": 0.844840407371521,
        "weird_rate": 0.06310319155454636,
        "box_jump_rate": 0.031185030937194824,
        "bad_rate": 0.19465479254722595,
        "max_bad_streak": 71,
        "note": "Adapter baseline from reliability_adapter_l18_l21_full full-stream summary.",
    },
]


STATIC_CONF = {
    "static_025": "0.25",
    "static_020": "0.20",
    "static_015": "0.15",
    "static_010": "0.10",
    "static_005": "0.05",
}


def pct(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{100.0 * float(value):.1f}%"


def tex_pct(value: float) -> str:
    return pct(value).replace("%", r"\%")


def compact_number(value: float) -> str:
    if pd.isna(value):
        return "--"
    return str(int(round(float(value))))


def load_policy_rows() -> pd.DataFrame:
    policy = pd.read_csv(POLICY_DIR / "policy_summary.csv")
    policy = policy[policy["policy"].isin(STATIC_CONF)].copy()
    policy["model"] = "Memory-bank ODAD"
    policy["conf"] = policy["policy"].map(STATIC_CONF)
    policy["note"] = policy["policy"].map(
        {
            "static_025": "Default threshold; under-confident for memory-bank ODAD.",
            "static_020": "Intermediate threshold in static sweep.",
            "static_015": "Recommended reportable/deployable operating point.",
            "static_010": "Higher-recall static point; not selected as main report point.",
            "static_005": "Aggressive/manual-review candidate only.",
        }
    )
    keep = [
        "model",
        "policy",
        "conf",
        "det_rate",
        "weird_rate",
        "box_jump_rate",
        "bad_rate",
        "max_bad_streak",
        "note",
    ]
    return policy[keep]


def build_table() -> pd.DataFrame:
    rows = pd.concat([pd.DataFrame(COMPARISON_ROWS), load_policy_rows()], ignore_index=True)
    order = {
        ("FDA-mix", "0.25"): 0,
        ("Current ODAD", "0.25"): 1,
        ("Adapter Base", "0.25"): 2,
        ("Memory-bank ODAD", "0.25"): 3,
        ("Memory-bank ODAD", "0.20"): 4,
        ("Memory-bank ODAD", "0.15"): 5,
        ("Memory-bank ODAD", "0.10"): 6,
        ("Memory-bank ODAD", "0.05"): 7,
    }
    rows["_order"] = [order.get((r.model, r.conf), 99) for r in rows.itertuples()]
    return rows.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def write_csv(rows: pd.DataFrame) -> None:
    rows.to_csv(OUT_DIR / "odad_threshold_policy_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)


def write_markdown(rows: pd.DataFrame) -> None:
    display = rows.copy()
    for col in ["det_rate", "weird_rate", "box_jump_rate", "bad_rate"]:
        display[col] = display[col].map(pct)
    display["max_bad_streak"] = display["max_bad_streak"].map(compact_number)
    columns = [
        "model",
        "conf",
        "det_rate",
        "weird_rate",
        "box_jump_rate",
        "bad_rate",
        "max_bad_streak",
        "note",
    ]
    headers = ["Model", "Conf.", "Det.", "Weird", "Jump", "Bad", "Max Bad", "Note"]
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display[columns].itertuples(index=False):
        table_lines.append("| " + " | ".join(str(value) for value in row) + " |")
    md = [
        "# ODAD Threshold Policy Summary",
        "",
        "**Recommended report point:** Memory-bank ODAD at static confidence threshold 0.15.",
        "",
        "At conf=0.15, memory-bank ODAD reaches 88.4% detection with 6.2% weird-box rate, 3.0% box-jump rate, 16.0% bad-frame proxy rate, and max bad streak 51.",
        "",
        "Interpretation: memory-bank ODAD is mainly under-confident at the default 0.25 threshold. Static threshold calibration recovers detection while preserving substantially better proxy reliability than current ODAD. Adaptive policies were cleaner by proxy on low-confidence accepts, but did not beat static 0.15 on the overall detection/reliability tradeoff. Static 0.05 remains an aggressive/manual-review candidate.",
        "",
        "\n".join(table_lines),
        "",
        "Proxy metrics are not label-aware accuracy. They summarize detection continuity, weird boxes, box jumps, and bad streaks over the ordered lab-image stream.",
        "",
    ]
    (OUT_DIR / "odad_threshold_policy_summary.md").write_text("\n".join(md), encoding="utf-8")


def write_tex(rows: pd.DataFrame) -> None:
    key = rows[
        (rows["model"].isin(["FDA-mix", "Current ODAD", "Adapter Base", "Memory-bank ODAD"]))
        & (
            ((rows["model"] != "Memory-bank ODAD") & (rows["conf"] == "0.25"))
            | ((rows["model"] == "Memory-bank ODAD") & (rows["conf"].isin(["0.25", "0.15", "0.05"])))
        )
    ].copy()
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Conf. & Det. & Weird & Jump & Max Bad \\",
        r"\midrule",
    ]
    for row in key.itertuples(index=False):
        fields = [
            row.model,
            row.conf,
            tex_pct(row.det_rate),
            tex_pct(row.weird_rate),
            tex_pct(row.box_jump_rate),
            compact_number(row.max_bad_streak),
        ]
        if row.model == "Memory-bank ODAD" and row.conf == "0.15":
            fields = [rf"\textbf{{{field}}}" for field in fields]
        lines.append(" & ".join(fields) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (OUT_DIR / "odad_key_metrics_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_table_png(rows: pd.DataFrame) -> None:
    key = rows[
        (rows["model"].isin(["FDA-mix", "Current ODAD", "Adapter Base", "Memory-bank ODAD"]))
        & (
            ((rows["model"] != "Memory-bank ODAD") & (rows["conf"] == "0.25"))
            | ((rows["model"] == "Memory-bank ODAD") & (rows["conf"].isin(["0.25", "0.15", "0.05"])))
        )
    ].copy()
    cell_text = []
    for row in key.itertuples(index=False):
        cell_text.append(
            [
                row.model,
                row.conf,
                pct(row.det_rate),
                pct(row.weird_rate),
                pct(row.box_jump_rate),
                pct(row.bad_rate),
                compact_number(row.max_bad_streak),
            ]
        )
    fig, ax = plt.subplots(figsize=(9, 2.45))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=["Model", "Conf.", "Det.", "Weird", "Jump", "Bad", "Max Bad"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.28, 0.09, 0.11, 0.11, 0.11, 0.11, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#777777")
        if r == 0:
            cell.set_facecolor("#e9ecef")
            cell.set_text_props(weight="bold")
        elif key.iloc[r - 1]["model"] == "Memory-bank ODAD" and key.iloc[r - 1]["conf"] == "0.15":
            cell.set_facecolor("#fff3bf")
            cell.set_text_props(weight="bold")
    fig.savefig(OUT_DIR / "odad_key_metrics_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_tradeoff_plot(rows: pd.DataFrame) -> None:
    mem = rows[rows["model"] == "Memory-bank ODAD"].copy()
    mem["conf_float"] = mem["conf"].astype(float)
    mem = mem.sort_values("conf_float", ascending=False)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(mem["conf_float"], 100 * mem["det_rate"], marker="o", color="#1f77b4", label="Detection")
    ax1.plot(mem["conf_float"], 100 * mem["bad_rate"], marker="s", color="#d62728", label="Bad proxy")
    ax1.set_xlabel("Static confidence threshold")
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Memory-bank ODAD threshold sweep")
    ax1.grid(True, alpha=0.25)
    ax1.invert_xaxis()
    ax1.axvline(0.15, color="#2ca02c", linestyle="--", linewidth=1.6, label="Recommended 0.15")
    ax2 = ax1.twinx()
    ax2.plot(mem["conf_float"], mem["max_bad_streak"], marker="^", color="#9467bd", label="Max bad streak")
    ax2.set_ylabel("Max bad streak")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="center right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "odad_threshold_tradeoff_clean.png", dpi=220)
    plt.close(fig)


def write_notes() -> None:
    notes = """ODAD reporting notes

Source of truth:
- yolo_ablation_runs/membank_threshold_policy_eval/policy_summary.csv
- yolo_ablation_runs/membank_threshold_policy_eval/operating_point_recommendation.txt

Recommended report point:
- Memory-bank ODAD, static confidence threshold 0.15.
- det=88.4%, weird=6.2%, box_jump=3.0%, bad=16.0%, max_bad_streak=51.

Use caveat:
- Weird, jump, bad, and streak metrics are proxy reliability metrics over the ordered lab-image stream, not label-aware accuracy.
- Static 0.05 improves detection/bad proxy numerically but is reserved for manual review because of occasional flare/overexposed-region boxes.
"""
    (OUT_DIR / "odad_reporting_notes.txt").write_text(notes, encoding="utf-8")


def write_cleanup_recommendations() -> None:
    cleanup = """ODAD artifact cleanup recommendations

Do not delete important baselines automatically. Prefer archiving older negative branches after the report is finalized.

Minimal active ODAD folders to keep:
- odad/online_adapt_topk2_full
- odad/adapter_l18_l21_full
- odad/membank_bank16_top2_teacherroi_full
- odad/source_topksim_w005_full
- odad/shadow_manager_full
- odad/shadow_manager_odad_active_full_diagnostic

Minimal active yolo_ablation_runs folders to keep:
- yolo_ablation_runs/membank_threshold_policy_eval
- yolo_ablation_runs/failure_audit_membank_operating_point
- yolo_ablation_runs/reliability_membank_bank16_top2_teacherroi_full
- yolo_ablation_runs/reliability_adapter_l18_l21_full
- yolo_ablation_runs/reliability_shadow_streak_fullstream_gpu
- yolo_ablation_runs/compare_models_odad_topk2_slide_ready

Optional historical negative-branch folders to archive:
- ODAD experiments superseded by memory-bank threshold calibration.
- Track-aware or multi-track branches that reduced detection and are no longer report candidates.
- Earlier shadow-manager diagnostics not referenced by the current report.

Clearly safe smoke/unit folders to delete only after manual inspection:
- Short max-frame smoke outputs.
- Probe folders created only to validate script wiring.
- Empty or failed plotting scratch directories.

Suggested approach:
- Move archive candidates to a timestamped archive directory outside the active report path.
- Keep summaries, CSVs, plots, and final weights for every branch cited in slides or reports.
- Do not run rm -rf from this recommendation file without a manual path-by-path review.
"""
    (OUT_DIR / "artifact_cleanup_recommendations.txt").write_text(cleanup, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_table()
    write_csv(rows)
    write_markdown(rows)
    write_tex(rows)
    write_table_png(rows)
    write_tradeoff_plot(rows)
    write_notes()
    write_cleanup_recommendations()


if __name__ == "__main__":
    main()
