#!/usr/bin/env python3
"""
Phase-0: Stream evaluation harness for sequential YOLO-root datasets (no learning).

What it does
- Reads frames in order from a YOLO-root dataset: <dataset_root>/images/test
- Runs YOLO inference per frame
- Logs per-frame:
    * top1_conf (max confidence)
    * top1_box_xyxy (pixel coords)
    * det_flag
    * IoU w/ previous top1 box (temporal stability)
    * center displacement (px) vs previous (jitter)
    * inference latency (ms)
- Writes:
    * CSV: stream_eval.csv
    * Plots:
        - top1_conf vs frame
        - det_rate over time (rolling mean)
        - temporal IoU vs frame
        - latency vs frame
        - histograms for confidence + latency
    * MP4: overlay of top1 box per frame (optional)

Assumptions
- Your dataset is sequential: filenames sort in frame order.
- Input can be any resolution. Ultralytics handles letterbox internally.

Example:
  python3 stream_eval.py \
    --model /path/to/best.pt \
    --dataset /path/to/yolo_root \
    --output /path/to/outdir \
    --device cuda:0 \
    --conf 0.25 \
    --iou 0.45 \
    --max-frames 6000 \
    --make-mp4 \
    --mp4-every 2 \
    --mp4-fps 12
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover
    raise ImportError("Please install imageio: pip install imageio") from exc

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install ultralytics: pip install ultralytics") from exc


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# -------------------------
# Helpers
# -------------------------
def list_test_images(dataset_root: Path) -> List[Path]:
    test_dir = dataset_root / "images" / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Expected YOLO-root dataset at {dataset_root}, missing: {test_dir}")
    imgs = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    if not imgs:
        raise RuntimeError(f"No images found under {test_dir}")
    return imgs


def xyxy_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

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


def box_center(a: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = a
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    out = np.empty_like(x, dtype=np.float32)
    s = 0.0
    q: List[float] = []
    for i, v in enumerate(x):
        q.append(float(v))
        s += float(v)
        if len(q) > window:
            s -= q.pop(0)
        out[i] = s / len(q)
    return out


def try_load_font(size: int = 16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def draw_overlay(
    img_rgb: Image.Image,
    top1_xyxy: Optional[Tuple[float, float, float, float]],
    top1_conf: float,
    frame_idx: int,
    latency_ms: Optional[float],
) -> Image.Image:
    out = img_rgb.copy()
    draw = ImageDraw.Draw(out)
    font = try_load_font(16)

    header = f"frame={frame_idx}  conf={top1_conf:.3f}"
    if latency_ms is not None and not math.isnan(latency_ms):
        header += f"  lat={latency_ms:.1f}ms"

    draw.rectangle([(0, 0), (out.size[0], 28)], fill=(0, 0, 0))
    draw.text((6, 6), header, fill=(255, 255, 255), font=font)

    if top1_xyxy is not None:
        x1, y1, x2, y2 = top1_xyxy
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)

        label = f"{top1_conf:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        y_top = max(0, y1 - th - 6)
        draw.rectangle([(x1, y_top), (x1 + tw + 10, y1)], fill=(0, 0, 0))
        draw.text((x1 + 5, y_top + 2), label, fill=(255, 255, 255), font=font)

    return out


# -------------------------
# Core inference
# -------------------------
@dataclass
class FrameRecord:
    frame: int
    path: str
    width: int
    height: int
    det: int
    top1_conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    iou_prev: float
    center_dx: float
    center_dy: float
    center_dist: float
    latency_ms: float


def predict_top1(
    model: YOLO,
    image_path: Path,
    device: str,
    conf: float,
    iou: float,
) -> Tuple[Optional[Tuple[float, float, float, float]], float, float]:
    """
    Returns (top1_xyxy, top1_conf, latency_ms).
    If no detection: (None, 0.0, latency_ms)
    """
    use_cuda_timing = device.startswith("cuda") and torch.cuda.is_available()
    starter = ender = None
    if use_cuda_timing:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()

    t0 = time.time()
    results = model.predict(
        source=str(image_path),
        device=device,
        conf=conf,
        iou=iou,
        verbose=False,
        save=False,
    )

    if use_cuda_timing:
        ender.record()
        torch.cuda.synchronize()
        latency_ms = float(starter.elapsed_time(ender))
    else:
        latency_ms = (time.time() - t0) * 1000.0

    if not results:
        return None, 0.0, latency_ms

    r = results[0]
    boxes = r.boxes
    if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
        return None, 0.0, latency_ms

    confs = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()

    idx = int(np.argmax(confs))
    top_conf = float(confs[idx])
    x1, y1, x2, y2 = map(float, xyxy[idx].tolist())
    return (x1, y1, x2, y2), top_conf, latency_ms


# -------------------------
# Plotting
# -------------------------
def save_plot_line(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(10, 4))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_plot_hist(values: np.ndarray, bins: int, title: str, xlabel: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream evaluation for sequential YOLO-root datasets (Phase-0).")
    p.add_argument("--model", type=str, required=True, help="Path to YOLO weights (.pt or .engine)")
    p.add_argument("--dataset", type=str, required=True, help="YOLO-root dataset path (expects images/test)")
    p.add_argument("--output", type=str, default="stream_eval_out", help="Output directory")
    p.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames, else limit to first N frames")
    p.add_argument("--warmup", type=int, default=10, help="Warmup frames before logging")
    p.add_argument("--roll", type=int, default=50, help="Rolling window for det-rate plot")
    p.add_argument("--hist-bins", type=int, default=30, help="Histogram bins for top1_conf and latency")
    p.add_argument("--seed", type=int, default=0, help="Unused, kept for symmetry")

    # MP4 only
    p.add_argument("--make-mp4", action="store_true", help="Write an overlay MP4")
    p.add_argument("--mp4-every", type=int, default=1, help="Use every k-th frame in MP4")
    p.add_argument("--mp4-max", type=int, default=0, help="Max frames in MP4, 0 = no limit")
    p.add_argument("--mp4-fps", type=int, default=12, help="MP4 fps")
    p.add_argument("--mp4-scale", type=float, default=1.0, help="Downscale MP4 frames by this factor")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset)
    images = list_test_images(dataset_root)
    if args.max_frames and args.max_frames > 0:
        images = images[: int(args.max_frames)]

    model = YOLO(args.model)

    # Warmup
    warmup_n = max(0, int(args.warmup))
    if warmup_n > 0:
        for i in range(min(warmup_n, len(images))):
            _ = model.predict(
                source=str(images[i]),
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
                save=False,
            )

    records: List[FrameRecord] = []
    mp4_frames: List[np.ndarray] = []

    prev_box: Optional[Tuple[float, float, float, float]] = None
    prev_center: Optional[Tuple[float, float]] = None

    for idx, img_path in enumerate(images):
        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB")
            w, h = im_rgb.size

        box, top_conf, latency_ms = predict_top1(
            model=model,
            image_path=img_path,
            device=args.device,
            conf=float(args.conf),
            iou=float(args.iou),
        )

        det = 1 if box is not None else 0

        if det and prev_box is not None:
            iou_prev = float(xyxy_iou(box, prev_box))
        else:
            iou_prev = float("nan")

        if det:
            cx, cy = box_center(box)
            if prev_center is not None:
                dx = float(cx - prev_center[0])
                dy = float(cy - prev_center[1])
                dist = float(math.sqrt(dx * dx + dy * dy))
            else:
                dx = dy = dist = float("nan")
        else:
            dx = dy = dist = float("nan")

        records.append(
            FrameRecord(
                frame=idx,
                path=str(img_path),
                width=int(w),
                height=int(h),
                det=int(det),
                top1_conf=float(top_conf),
                x1=float(box[0]) if box else float("nan"),
                y1=float(box[1]) if box else float("nan"),
                x2=float(box[2]) if box else float("nan"),
                y2=float(box[3]) if box else float("nan"),
                iou_prev=iou_prev,
                center_dx=dx,
                center_dy=dy,
                center_dist=dist,
                latency_ms=float(latency_ms),
            )
        )

        if det:
            prev_box = box
            prev_center = box_center(box)
        else:
            prev_box = None
            prev_center = None

        if args.make_mp4:
            use_mp4 = (idx % max(1, int(args.mp4_every)) == 0)
            under_mp4_cap = (int(args.mp4_max) <= 0) or (len(mp4_frames) < int(args.mp4_max))
            if use_mp4 and under_mp4_cap:
                with Image.open(img_path) as im:
                    overlay = draw_overlay(im.convert("RGB"), box, float(top_conf), idx, float(latency_ms))

                if float(args.mp4_scale) != 1.0:
                    new_w = max(1, int(round(overlay.size[0] * float(args.mp4_scale))))
                    new_h = max(1, int(round(overlay.size[1] * float(args.mp4_scale))))
                    overlay = overlay.resize((new_w, new_h), Image.Resampling.BILINEAR)

                mp4_frames.append(np.array(overlay))

    # Write CSV
    csv_path = out_dir / "stream_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "path",
                "width",
                "height",
                "det",
                "top1_conf",
                "x1",
                "y1",
                "x2",
                "y2",
                "iou_prev",
                "center_dx",
                "center_dy",
                "center_dist",
                "latency_ms",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.frame,
                    r.path,
                    r.width,
                    r.height,
                    r.det,
                    f"{r.top1_conf:.6f}",
                    f"{r.x1:.3f}",
                    f"{r.y1:.3f}",
                    f"{r.x2:.3f}",
                    f"{r.y2:.3f}",
                    f"{r.iou_prev:.6f}" if not math.isnan(r.iou_prev) else "",
                    f"{r.center_dx:.3f}" if not math.isnan(r.center_dx) else "",
                    f"{r.center_dy:.3f}" if not math.isnan(r.center_dy) else "",
                    f"{r.center_dist:.3f}" if not math.isnan(r.center_dist) else "",
                    f"{r.latency_ms:.3f}",
                ]
            )

    # Arrays for plots
    frames = np.array([r.frame for r in records], dtype=np.int32)
    top1_conf = np.array([r.top1_conf for r in records], dtype=np.float32)
    det = np.array([r.det for r in records], dtype=np.float32)
    iou_prev = np.array([r.iou_prev for r in records], dtype=np.float32)
    latency = np.array([r.latency_ms for r in records], dtype=np.float32)

    save_plot_line(
        frames,
        top1_conf,
        "Top-1 Confidence vs Frame",
        "frame",
        "top1_conf",
        out_dir / "plot_top1_conf.png",
    )
    save_plot_hist(
        top1_conf,
        int(args.hist_bins),
        "Top-1 Confidence Histogram",
        "top1_conf",
        out_dir / "hist_top1_conf.png",
    )

    det_roll = rolling_mean(det, int(args.roll))
    save_plot_line(
        frames,
        det_roll,
        f"Detection Rate (rolling mean, window={args.roll})",
        "frame",
        "det_rate",
        out_dir / "plot_det_rate_roll.png",
    )

    iou_defined = np.where(np.isfinite(iou_prev), iou_prev, np.nan)
    save_plot_line(
        frames,
        iou_defined,
        "Temporal IoU (top-1 box vs previous)",
        "frame",
        "IoU",
        out_dir / "plot_iou_prev.png",
    )

    save_plot_line(
        frames,
        latency,
        "Inference Latency vs Frame",
        "frame",
        "latency_ms",
        out_dir / "plot_latency_ms.png",
    )
    save_plot_hist(
        latency,
        int(args.hist_bins),
        "Latency Histogram",
        "latency_ms",
        out_dir / "hist_latency_ms.png",
    )

    # Summary
    det_rate = float(np.mean(det)) if len(det) else float("nan")
    mean_conf = float(np.mean(top1_conf)) if len(top1_conf) else float("nan")
    med_conf = float(np.median(top1_conf)) if len(top1_conf) else float("nan")
    mean_latency = float(np.mean(latency)) if len(latency) else float("nan")
    p90_latency = float(np.percentile(latency, 90)) if len(latency) else float("nan")
    mean_iou_prev = float(np.nanmean(iou_prev)) if np.any(np.isfinite(iou_prev)) else float("nan")
    mean_center_dist = (
        float(np.nanmean(np.array([r.center_dist for r in records], dtype=np.float32)))
        if any(not math.isnan(r.center_dist) for r in records)
        else float("nan")
    )

    summary_path = out_dir / "summary.txt"
    summary_lines = [
        "Stream Eval Summary (Phase-0)",
        f"model={args.model}",
        f"dataset={dataset_root}",
        f"frames={len(records)}",
        f"conf={float(args.conf):.3f}  iou={float(args.iou):.3f}",
        "",
        f"det_rate (top1 exists) = {100.0 * det_rate:.1f}%",
        f"top1_conf mean/median  = {mean_conf:.3f} / {med_conf:.3f}",
        f"latency mean/p90 (ms)  = {mean_latency:.2f} / {p90_latency:.2f}",
        f"temporal IoU mean      = {mean_iou_prev:.3f}" if not math.isnan(mean_iou_prev) else "temporal IoU mean      = n/a",
        f"center jitter mean(px) = {mean_center_dist:.2f}" if not math.isnan(mean_center_dist) else "center jitter mean(px) = n/a",
        "",
        "Outputs:",
        f"  csv: {csv_path.name}",
        "  plots: plot_top1_conf.png, plot_det_rate_roll.png, plot_iou_prev.png, plot_latency_ms.png",
        "  hists: hist_top1_conf.png, hist_latency_ms.png",
    ]
    if args.make_mp4:
        summary_lines.append("  mp4: stream_overlay.mp4")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    if args.make_mp4 and mp4_frames:
        mp4_path = out_dir / "stream_overlay.mp4"
        with imageio.get_writer(mp4_path, fps=int(args.mp4_fps), codec="libx264", quality=7) as writer:
            for fr in mp4_frames:
                writer.append_data(fr)

    print(f"Done. Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()


# Example:
# PYTHONPATH=. python odad/stream_eval.py \
#   --model /home/hm25936/mae/runs/yolov8_baseline/baseline/weights/best.pt \
#   --dataset /home/hm25936/datasets_for_yolo/lab_images_6000 \
#   --output /home/hm25936/mae/odad/baseline_stream_test \
#   --device cuda:0 \
#   --conf 0.25 \
#   --iou 0.45 \
#   --max-frames 6000 \
#   --make-mp4 \
#   --mp4-every 2 \
#   --mp4-fps 12 \
#   --mp4-scale 0.75