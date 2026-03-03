"""Visualization and I/O utilities for bbox detections"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .types import DetectionResult
from .types import invalid_detection


def _rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    # axis-aligned rectangle overlap check in pixel coordinates
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return (inter_x2 > inter_x1) and (inter_y2 > inter_y1)


def _label_block_size(
                        lines: Sequence[str],
                        font,
                        scale: float,
                        thickness: int,
                        pad_x: int,
                        pad_y: int,
                        line_gap: int,
                    ):
    # compute text block geometry so placement can avoid bbox region
    widths = []
    heights = []
    baselines = []

    for line in lines:
        (w, h), b = cv2.getTextSize(str(line), font, scale, thickness)
        widths.append(int(w))
        heights.append(int(h))
        baselines.append(int(b))

    block_w = max(widths) + 2 * pad_x
    block_h = sum(h + b for h, b in zip(heights, baselines)) + max(0, len(lines) - 1) * line_gap + 2 * pad_y
    return int(block_w), int(block_h), heights, baselines


def draw_detection_overlay(
                            image_bgr: np.ndarray,
                            det: DetectionResult,
                            bbox_color: tuple[int, int, int] = (40, 220, 40),
                            text_color: tuple[int, int, int] = (230, 230, 230),
                        ) -> np.ndarray:
    """Draw bbox + text block (confidence + inference ms) with overlap-aware text placement"""

    # copy to avoid modifying caller-owned image buffer
    draw = np.asarray(image_bgr).copy()
    if draw.ndim != 3:
        raise ValueError('Expected HxWx3 image for overlay')

    img_h       = int(draw.shape[0])
    img_w       = int(draw.shape[1])
    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = 0.58
    thickness   = 2
    pad_x       = 8
    pad_y       = 6
    line_gap    = 6

    infer_txt = f'infer={float(det.inference_ms):.2f}ms' if np.isfinite(float(det.inference_ms)) else 'infer=n/a'
    if bool(det.valid):
        # valid detection block includes class id and confidence
        info_lines = [
                        f'class={int(det.class_id)} conf={float(det.score):.3f}',
                        infer_txt,
                    ]
    else:
        # invalid detection still carries timing info for diagnostics
        info_lines = [
                        'no detection',
                        infer_txt,
                    ]

    bbox_rect = None
    if bool(det.valid):
        # draw bbox first and reserve this region for overlap checks
        x1, y1, x2, y2 = np.asarray(det.bbox_xyxy, dtype = float).reshape(4,)
        x1 = int(np.clip(round(x1), 0, max(0, img_w - 1)))
        y1 = int(np.clip(round(y1), 0, max(0, img_h - 1)))
        x2 = int(np.clip(round(x2), 0, max(0, img_w - 1)))
        y2 = int(np.clip(round(y2), 0, max(0, img_h - 1)))

        cv2.rectangle(draw, (x1, y1), (x2, y2), bbox_color, 2)
        bbox_rect = (
                        float(min(x1, x2)),
                        float(min(y1, y2)),
                        float(max(x1, x2)),
                        float(max(y1, y2)),
                    )

    block_w, block_h, heights, baselines = _label_block_size(
                                                                lines = info_lines,
                                                                font = font,
                                                                scale = font_scale,
                                                                thickness = thickness,
                                                                pad_x = pad_x,
                                                                pad_y = pad_y,
                                                                line_gap = line_gap,
                                                            )

    candidates = [
                    (12, 12),
                    (12, max(12, img_h - block_h - 12)),
                    (max(12, img_w - block_w - 12), 12),
                    (max(12, img_w - block_w - 12), max(12, img_h - block_h - 12)),
                ]

    # choose first label location that does not overlap bbox region
    block_x, block_y = candidates[0]
    for cx, cy in candidates:
        rect = (float(cx), float(cy), float(cx + block_w), float(cy + block_h))
        if bbox_rect is None or not _rects_overlap(rect, bbox_rect):
            block_x, block_y = cx, cy
            break

    cv2.rectangle(
                    draw,
                    (int(block_x), int(block_y)),
                    (int(block_x + block_w), int(block_y + block_h)),
                    (22, 22, 22),
                    -1,
                )
    cv2.rectangle(
                    draw,
                    (int(block_x), int(block_y)),
                    (int(block_x + block_w), int(block_y + block_h)),
                    (95, 95, 95),
                    1,
                )

    cursor_y = int(block_y + pad_y)
    for line, h, b in zip(info_lines, heights, baselines):
        cursor_y += int(h)
        cv2.putText(
                        draw,
                        str(line),
                        (int(block_x + pad_x), int(cursor_y)),
                        font,
                        font_scale,
                        text_color,
                        thickness,
                        cv2.LINE_AA,
                    )
        cursor_y += int(b + line_gap)

    return draw


def _record_from_detection(det: DetectionResult) -> dict:
    # flatten dataclass into stable primitive fields for CSV/JSON interchange
    x1, y1, x2, y2 = np.asarray(det.bbox_xyxy, dtype = float).reshape(4,)
    cx, cy, bw, bh = np.asarray(det.bbox_xywh_norm, dtype = float).reshape(4,)
    image_stem = Path(str(det.image_path)).stem if str(det.image_path) else ''

    return {
                'image_path': str(det.image_path),
                'image_stem': image_stem,
                'valid': int(bool(det.valid)),
                'class_id': int(det.class_id),
                'score': float(det.score),
                'inference_ms': float(det.inference_ms),
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2),
                'cx_norm': float(cx),
                'cy_norm': float(cy),
                'w_norm': float(bw),
                'h_norm': float(bh),
                'backend': str(det.backend),
                'detector_name': str(det.detector_name),
                'model_path': str(det.model_path),
            }


def write_detections_csv(csv_path: str | Path, detections: Sequence[DetectionResult]) -> Path:
    """Write detection rows to CSV"""

    csv_path_obj = Path(csv_path).expanduser().resolve()
    csv_path_obj.parent.mkdir(parents = True, exist_ok = True)

    fieldnames = [
                    'image_path', 'image_stem',
                    'valid', 'class_id', 'score', 'inference_ms',
                    'x1', 'y1', 'x2', 'y2',
                    'cx_norm', 'cy_norm', 'w_norm', 'h_norm',
                    'backend', 'detector_name', 'model_path',
                ]

    with csv_path_obj.open('w', newline = '', encoding = 'utf-8') as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()

        # one row per image inference
        for det in detections:
            writer.writerow(_record_from_detection(det))

    return csv_path_obj


def write_detections_json(json_path: str | Path, detections: Sequence[DetectionResult]) -> Path:
    """Write detection rows to JSON"""

    json_path_obj = Path(json_path).expanduser().resolve()
    json_path_obj.parent.mkdir(parents = True, exist_ok = True)

    rows = [_record_from_detection(det) for det in detections]
    with json_path_obj.open('w', encoding = 'utf-8') as f:
        json.dump(rows, f, indent = 2)

    return json_path_obj


def _detection_from_row(row: dict) -> DetectionResult:
    # rebuild DetectionResult from persisted CSV row
    image_path = str(row.get('image_path', ''))
    det = invalid_detection(
                            backend = str(row.get('backend', '')),
                            detector_name = str(row.get('detector_name', '')),
                            model_path = str(row.get('model_path', '')),
                        )

    det.valid = bool(int(float(row.get('valid', 0))))
    det.class_id = int(float(row.get('class_id', -1)))
    det.score = float(row.get('score', 0.0))
    det.inference_ms = float(row.get('inference_ms', np.nan))
    det.image_path = image_path
    # preserve exact serialized fields for reproducible re-visualization
    det.bbox_xyxy = np.array(
                            [
                                float(row.get('x1', np.nan)),
                                float(row.get('y1', np.nan)),
                                float(row.get('x2', np.nan)),
                                float(row.get('y2', np.nan)),
                            ],
                            dtype = float,
                        )
    det.bbox_xywh_norm = np.array(
                                    [
                                        float(row.get('cx_norm', np.nan)),
                                        float(row.get('cy_norm', np.nan)),
                                        float(row.get('w_norm', np.nan)),
                                        float(row.get('h_norm', np.nan)),
                                    ],
                                    dtype = float,
                                )
    return det


def load_detections_csv(csv_path: str | Path) -> list[DetectionResult]:
    """Load detection records from CSV"""

    csv_path_obj = Path(csv_path).expanduser().resolve()
    if not csv_path_obj.is_file():
        raise FileNotFoundError(f'csv file not found: {csv_path_obj}')

    detections = []
    with csv_path_obj.open('r', newline = '', encoding = 'utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            detections.append(_detection_from_row(row))

    return detections


def _save_topk_panel(detections: list[DetectionResult], output_path: Path, top_k: int) -> None:
    # prioritize high-confidence valid detections for quick visual check
    valid = [det for det in detections if bool(det.valid)]
    ranked = sorted(valid, key = lambda d: float(d.score), reverse = True)
    if not ranked:
        # fallback so panel still renders when no valid detections exist
        ranked = sorted(detections, key = lambda d: float(d.score), reverse = True)

    top = ranked[:max(1, int(top_k))]
    n = len(top)
    # fixed figure sizing keeps per-frame text readable
    fig, axes = plt.subplots(1, n, figsize = (6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, det in zip(axes, top):
        image_bgr = cv2.imread(str(det.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            ax.set_title(f'missing: {Path(str(det.image_path)).name}')
            ax.axis('off')
            continue

        draw = draw_detection_overlay(image_bgr = image_bgr, det = det)
        ax.imshow(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
        ax.set_title(
                        f'{Path(str(det.image_path)).stem}\n'
                        f'valid={int(bool(det.valid))} class={int(det.class_id)}\n'
                        f'score={float(det.score):.3f} infer={float(det.inference_ms):.2f}ms'
                    )
        ax.axis('off')

    fig.suptitle('Top Bounding-Box Detections')
    # tight layout keeps text and images legible in exported panel
    fig.tight_layout()
    fig.savefig(output_path, dpi = 180)
    plt.close(fig)


def save_visualizations_from_csv(csv_path: str | Path, output_dir: str | Path, top_k: int = 3) -> dict:
    """Build per-image bbox overlays and top-k panel from CSV detections"""

    # load detection rows produced by offline_test logging
    detections = load_detections_csv(csv_path)
    output_dir_obj = Path(output_dir).expanduser().resolve()
    output_dir_obj.mkdir(parents = True, exist_ok = True)

    bbox_viz_dir = output_dir_obj / 'bbox_viz'
    bbox_viz_dir.mkdir(parents = True, exist_ok = True)

    # count successful image writes for run summary
    written = 0

    # write one overlay image per CSV row
    for det in detections:
        if not str(det.image_path):
            continue

        image_bgr = cv2.imread(str(det.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        draw = draw_detection_overlay(image_bgr = image_bgr, det = det)
        out_path = bbox_viz_dir / f'{Path(str(det.image_path)).stem}_bbox.png'
        if cv2.imwrite(str(out_path), draw):
            written += 1

    top_panel = output_dir_obj / 'top_detections.png'
    if len(detections) > 0:
        _save_topk_panel(detections = detections, output_path = top_panel, top_k = int(top_k))

    return {
                'num_rows': int(len(detections)),
                'num_visualizations_written': int(written),
                'bbox_viz_dir': str(bbox_viz_dir),
                'top_panel_path': str(top_panel),
            }


def _parse_args() -> argparse.Namespace:
    # CLI targets post-run visualization from existing detection CSV
    parser = argparse.ArgumentParser(description = 'Create bbox visualizations from detections CSV')
    parser.add_argument('--csv-path', type = str, required = True)
    parser.add_argument('--output-dir', type = str, required = True)
    parser.add_argument('--top-k', type = int, default = 3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    info = save_visualizations_from_csv(
                                        csv_path = args.csv_path,
                                        output_dir = args.output_dir,
                                        top_k = int(args.top_k),
                                    )
    print('Visualization complete')
    print(json.dumps(info, indent = 2))


if __name__ == '__main__':
    main()
