#!/usr/bin/env python3
"""
Phase-1: Online teacher-student adaptation scaffold for sequential YOLO-root datasets.

What it does
- Reads frames in order from a YOLO-root dataset: <dataset_root>/images/test
- Maintains:
    * student YOLO model (trainable, head-only by default)
    * teacher YOLO model (EMA of student, eval-only)
- For each frame:
    1) Teacher predicts top-1 pseudo label on the original image
    2) Apply gating:
         - confidence threshold
         - bbox area sanity
         - temporal IoU gate (optional)
    3) Student predicts on the original image (for logging / visualization)
    4) If pseudo label accepted:
         - apply appearance-only strong augmentation to the image
         - letterbox to imgsz
         - convert teacher pseudo box from original coords -> normalized xywh in letterboxed coords
         - run one gradient step on student using Ultralytics detection loss
         - update teacher via EMA
    5) Student predicts again on the original image (post-update) for logging / visualization
- Writes:
    * CSV: adapt_log.csv
    * Plots:
        - teacher_conf vs frame
        - student_pre_conf vs frame
        - student_post_conf vs frame
        - update mask vs frame
        - loss vs frame
        - teacher temporal IoU vs frame
        - student temporal IoU vs frame
    * MP4: 3-panel video (input / teacher / student-post)

Notes
- This is intentionally conservative:
    * top-1 pseudo label only
    * one gradient step per accepted frame
    * head-only updates by default
    * appearance-only student augmentation (no geometry changes)
- This is a first-pass scaffold. Ultralytics internals are version-sensitive, so the loss import path
  may need minor adjustment depending on your installed version.

Example:
  PYTHONPATH=. python odad/online_adapt.py \
    --weights /home/hm25936/mae/runs/yolov8_baseline/baseline/weights/best.pt \
    --dataset /home/hm25936/datasets_for_yolo/lab_images_6000 \
    --output /home/hm25936/mae/odad/online_adapt_lab \
    --device cuda:0 \
    --imgsz 1024 \
    --teacher-conf-thresh 0.70 \
    --temporal-iou-gate 0.30 \
    --lr 1e-4 \
    --ema-decay 0.999 \
    --make-mp4 \
    --mp4-every 2 \
    --mp4-fps 12
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

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

# Loss import varies by ultralytics version
LOSS_IMPORT_ERROR = None
LossClass = None
for _candidate in (
    "ultralytics.utils.loss:v8DetectionLoss",
    "ultralytics.yolo.v8.detect.train:Loss",
):
    try:
        module_name, class_name = _candidate.split(":")
        module = __import__(module_name, fromlist=[class_name])
        LossClass = getattr(module, class_name)
        break
    except Exception as exc:  # pragma: no cover
        LOSS_IMPORT_ERROR = exc

if LossClass is None:  # pragma: no cover
    raise ImportError(
        "Unable to import Ultralytics detection loss. "
        "Tried ultralytics.utils.loss:v8DetectionLoss and ultralytics.yolo.v8.detect.train:Loss."
    ) from LOSS_IMPORT_ERROR


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# -------------------------
# Data classes
# -------------------------
@dataclass
class Top1Det:
    conf: float
    xyxy: Tuple[float, float, float, float]


@dataclass
class AdaptRecord:
    frame: int
    path: str
    width: int
    height: int

    teacher_det: int
    teacher_conf: float
    teacher_x1: float
    teacher_y1: float
    teacher_x2: float
    teacher_y2: float
    teacher_iou_prev: float

    student_pre_det: int
    student_pre_conf: float
    student_pre_x1: float
    student_pre_y1: float
    student_pre_x2: float
    student_pre_y2: float
    student_pre_iou_prev: float

    student_post_det: int
    student_post_conf: float
    student_post_x1: float
    student_post_y1: float
    student_post_x2: float
    student_post_y2: float
    student_post_iou_prev: float

    pseudo_accepted: int
    update_applied: int
    loss_total: float
    loss_box: float
    loss_cls: float
    loss_dfl: float
    latency_teacher_ms: float
    latency_student_pre_ms: float
    latency_student_post_ms: float


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


def try_load_font(size: int = 16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def draw_panel(
    img_rgb: Image.Image,
    title: str,
    top1: Optional[Top1Det],
    frame_idx: int,
    extra_text: str = "",
) -> Image.Image:
    out = img_rgb.copy()
    draw = ImageDraw.Draw(out)
    font = try_load_font(16)

    header = f"{title} | frame={frame_idx}"
    if extra_text:
        header += f" | {extra_text}"

    draw.rectangle([(0, 0), (out.size[0], 28)], fill=(0, 0, 0))
    draw.text((6, 6), header, fill=(255, 255, 255), font=font)

    if top1 is not None:
        x1, y1, x2, y2 = top1.xyxy
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
        label = f"{top1.conf:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        y_top = max(0, y1 - th - 6)
        draw.rectangle([(x1, y_top), (x1 + tw + 10, y1)], fill=(0, 0, 0))
        draw.text((x1 + 5, y_top + 2), label, fill=(255, 255, 255), font=font)

    return out


def make_triptych(
    img_rgb: Image.Image,
    frame_idx: int,
    teacher_top1: Optional[Top1Det],
    student_top1: Optional[Top1Det],
    update_applied: bool,
    pseudo_accepted: bool,
    loss_total: float,
) -> Image.Image:
    w, h = img_rgb.size
    left = draw_panel(img_rgb, "Input", None, frame_idx)
    mid = draw_panel(
        img_rgb,
        "Teacher",
        teacher_top1,
        frame_idx,
        extra_text=f"accepted={int(pseudo_accepted)}",
    )
    right = draw_panel(
        img_rgb,
        "Student(post)",
        student_top1,
        frame_idx,
        extra_text=f"update={int(update_applied)} loss={loss_total:.3f}" if update_applied else f"update={int(update_applied)}",
    )

    out = Image.new("RGB", (w * 3, h), color=(0, 0, 0))
    out.paste(left, (0, 0))
    out.paste(mid, (w, 0))
    out.paste(right, (2 * w, 0))
    return out


def to_numpy_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def letterbox_image(
    img_rgb: Image.Image,
    size: int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, float, int, int]:
    """
    Returns:
      letterboxed_img, gain, pad_left, pad_top
    """
    w0, h0 = img_rgb.size
    gain = min(size / w0, size / h0)
    new_w = int(round(w0 * gain))
    new_h = int(round(h0 * gain))

    resized = img_rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (size, size), color=color)

    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    out.paste(resized, (pad_left, pad_top))
    return out, gain, pad_left, pad_top


def strong_augment(img_rgb: Image.Image, rng: random.Random) -> Image.Image:
    """
    Appearance-only augmentation. No geometric transforms, so bbox stays valid in original coords.
    """
    out = img_rgb.copy()

    # Brightness
    b = 0.75 + 0.5 * rng.random()   # [0.75, 1.25]
    out = ImageEnhance.Brightness(out).enhance(b)

    # Contrast
    c = 0.75 + 0.5 * rng.random()
    out = ImageEnhance.Contrast(out).enhance(c)

    # Blur (occasionally)
    if rng.random() < 0.35:
        radius = 0.5 + 1.5 * rng.random()
        out = out.filter(ImageFilter.GaussianBlur(radius=radius))

    # Additive Gaussian noise
    arr = np.array(out).astype(np.float32)
    if rng.random() < 0.5:
        sigma = rng.uniform(2.0, 8.0)
        noise = rng.normalvariate(0.0, 1.0)
        # vectorized noise
        arr += np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr, 0.0, 255.0)

    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def xyxy_original_to_norm_xywh_letterboxed(
    box_xyxy: Tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
    size: int,
    gain: float,
    pad_left: int,
    pad_top: int,
) -> Tuple[float, float, float, float]:
    """
    Convert original-image xyxy (pixels) to normalized xywh in the letterboxed square image.
    """
    x1, y1, x2, y2 = box_xyxy
    x1_l = x1 * gain + pad_left
    y1_l = y1 * gain + pad_top
    x2_l = x2 * gain + pad_left
    y2_l = y2 * gain + pad_top

    x_c = ((x1_l + x2_l) / 2.0) / size
    y_c = ((y1_l + y2_l) / 2.0) / size
    w = max(0.0, x2_l - x1_l) / size
    h = max(0.0, y2_l - y1_l) / size

    return (
        min(1.0, max(0.0, x_c)),
        min(1.0, max(0.0, y_c)),
        min(1.0, max(0.0, w)),
        min(1.0, max(0.0, h)),
    )


def pil_to_model_tensor(img_rgb: Image.Image) -> torch.Tensor:
    """
    RGB PIL -> float tensor [1,3,H,W] in [0,1]
    """
    arr = np.array(img_rgb).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


def top1_from_results(results) -> Tuple[Optional[Top1Det], float]:
    """
    Parse Ultralytics results list into Top1Det and latency_ms placeholder (returned separately elsewhere).
    """
    if not results:
        return None, 0.0
    r = results[0]
    boxes = r.boxes
    if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
        return None, 0.0
    confs = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()
    idx = int(np.argmax(confs))
    conf = float(confs[idx])
    x1, y1, x2, y2 = map(float, xyxy[idx].tolist())
    return Top1Det(conf=conf, xyxy=(x1, y1, x2, y2)), conf


def predict_top1_wrapper(
    yolo_wrapper: YOLO,
    source,
    device: str,
    conf: float,
    iou: float,
) -> Tuple[Optional[Top1Det], float]:
    """
    Wrapper-based top1 inference for teacher/student visualization + pseudo-label extraction.
    source can be a path or numpy RGB array.
    Returns top1 and latency_ms.
    """
    use_cuda_timing = device.startswith("cuda") and torch.cuda.is_available()
    starter = ender = None
    if use_cuda_timing:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()

    t0 = time.time()
    results = yolo_wrapper.predict(
        source=source,
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

    top1, _ = top1_from_results(results)
    return top1, latency_ms


def area_fraction(box: Tuple[float, float, float, float], width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    img_area = max(1.0, float(width * height))
    return float(box_area / img_area)


def near_border(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
    margin_frac: float,
) -> bool:
    if margin_frac <= 0:
        return False
    x1, y1, x2, y2 = box
    mx = margin_frac * width
    my = margin_frac * height
    return (x1 < mx) or (y1 < my) or (x2 > (width - mx)) or (y2 > (height - my))


def should_accept_pseudo(
    top1: Optional[Top1Det],
    prev_teacher_box: Optional[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
    conf_thresh: float,
    min_area_frac: float,
    max_area_frac: float,
    border_margin_frac: float,
    temporal_iou_gate: float,
) -> bool:
    if top1 is None:
        return False
    if top1.conf < conf_thresh:
        return False

    af = area_fraction(top1.xyxy, img_w, img_h)
    if af < min_area_frac or af > max_area_frac:
        return False

    if near_border(top1.xyxy, img_w, img_h, border_margin_frac):
        return False

    if prev_teacher_box is not None and temporal_iou_gate > 0:
        if xyxy_iou(top1.xyxy, prev_teacher_box) < temporal_iou_gate:
            return False

    return True


def freeze_for_head_only(student_model: nn.Module) -> None:
    for p in student_model.parameters():
        p.requires_grad = False
    # Ultralytics DetectionModel stores layers in .model
    detect_module = student_model.model[-1]
    for p in detect_module.parameters():
        p.requires_grad = True


def update_teacher_ema(teacher_model: nn.Module, student_model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        t_state = teacher_model.state_dict()
        s_state = student_model.state_dict()
        for k, v_t in t_state.items():
            v_s = s_state[k]
            if torch.is_floating_point(v_t):
                v_t.mul_(decay).add_(v_s.detach(), alpha=(1.0 - decay))
            else:
                v_t.copy_(v_s)


def safe_loss_items_to_dict(loss_items) -> Dict[str, float]:
    """
    Ultralytics loss_items can be a tensor/list/tuple depending on version.
    Convention is often [box, cls, dfl].
    """
    out = {"box": float("nan"), "cls": float("nan"), "dfl": float("nan")}
    try:
        if isinstance(loss_items, torch.Tensor):
            vals = loss_items.detach().cpu().flatten().tolist()
        elif isinstance(loss_items, (list, tuple)):
            vals = [float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v) for v in loss_items]
        else:
            return out

        if len(vals) > 0:
            out["box"] = float(vals[0])
        if len(vals) > 1:
            out["cls"] = float(vals[1])
        if len(vals) > 2:
            out["dfl"] = float(vals[2])
    except Exception:
        pass
    return out


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online teacher-student adaptation for sequential YOLO-root datasets.")
    p.add_argument("--weights", type=str, required=True, help="Path to YOLO weights (.pt)")
    p.add_argument("--dataset", type=str, required=True, help="YOLO-root dataset path (expects images/test)")
    p.add_argument("--output", type=str, default="online_adapt_out", help="Output directory")
    p.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    p.add_argument("--imgsz", type=int, default=1024, help="Student training image size (letterbox target)")
    p.add_argument("--teacher-conf-thresh", type=float, default=0.70, help="Pseudo-label acceptance threshold")
    p.add_argument("--infer-conf", type=float, default=0.001, help="Inference conf for teacher/student top1 extraction")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames, else limit to first N frames")
    p.add_argument("--warmup", type=int, default=10, help="Warmup frames before logging/adaptation")
    p.add_argument("--lr", type=float, default=1e-4, help="Student optimizer learning rate")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    p.add_argument("--weight-decay", type=float, default=5e-4, help="SGD weight decay")
    p.add_argument("--ema-decay", type=float, default=0.999, help="Teacher EMA decay per accepted update")
    p.add_argument("--grad-clip", type=float, default=10.0, help="Gradient clipping max norm")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")

    # Pseudo-label gating
    p.add_argument("--min-area-frac", type=float, default=0.001, help="Min accepted bbox area fraction")
    p.add_argument("--max-area-frac", type=float, default=0.80, help="Max accepted bbox area fraction")
    p.add_argument("--border-margin-frac", type=float, default=0.02, help="Reject boxes too close to image border")
    p.add_argument("--temporal-iou-gate", type=float, default=0.30, help="Require teacher IoU(prev,current) >= gate (0 disables)")

    # Plots
    p.add_argument("--hist-bins", type=int, default=30, help="Histogram bins")
    p.add_argument("--roll", type=int, default=50, help="Rolling window for update-rate plot")

    # MP4
    p.add_argument("--make-mp4", action="store_true", help="Write an adaptation overlay MP4")
    p.add_argument("--mp4-every", type=int, default=1, help="Use every k-th frame in MP4")
    p.add_argument("--mp4-max", type=int, default=0, help="Max frames in MP4, 0 = no limit")
    p.add_argument("--mp4-fps", type=int, default=12, help="MP4 fps")
    p.add_argument("--mp4-scale", type=float, default=0.75, help="Downscale MP4 frames by this factor")
    return p.parse_args()


# -------------------------
# Main
# -------------------------
def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset)
    images = list_test_images(dataset_root)
    if args.max_frames and args.max_frames > 0:
        images = images[: int(args.max_frames)]

    # Two wrappers so .predict() remains easy for both teacher and student
    student_yolo = YOLO(args.weights)
    teacher_yolo = YOLO(args.weights)

    student_model = student_yolo.model
    teacher_model = teacher_yolo.model

    student_model.to(args.device)
    teacher_model.to(args.device)

    # Conservative adaptation: head only
    freeze_for_head_only(student_model)

    optim_params = [p for p in student_model.parameters() if p.requires_grad]
    if not optim_params:
        raise RuntimeError("No trainable parameters found after head-only freezing.")
    optimizer = torch.optim.SGD(
        optim_params,
        lr=float(args.lr),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
    )

    criterion = LossClass(student_model)

    # Warmup
    warmup_n = max(0, int(args.warmup))
    if warmup_n > 0:
        for i in range(min(warmup_n, len(images))):
            _ = teacher_yolo.predict(
                source=str(images[i]),
                device=args.device,
                conf=args.infer_conf,
                iou=args.iou,
                verbose=False,
                save=False,
            )

    records: List[AdaptRecord] = []
    mp4_frames: List[np.ndarray] = []

    prev_teacher_box: Optional[Tuple[float, float, float, float]] = None
    prev_student_pre_box: Optional[Tuple[float, float, float, float]] = None
    prev_student_post_box: Optional[Tuple[float, float, float, float]] = None

    rng = random.Random(int(args.seed))

    for idx, img_path in enumerate(images):
        with Image.open(img_path) as im:
            img_rgb = im.convert("RGB")
            w, h = img_rgb.size

        # -------------------------
        # Teacher prediction on original image
        # -------------------------
        teacher_model.eval()
        teacher_top1, teacher_lat_ms = predict_top1_wrapper(
            teacher_yolo,
            source=str(img_path),
            device=args.device,
            conf=float(args.infer_conf),
            iou=float(args.iou),
        )

        teacher_iou_prev = (
            xyxy_iou(teacher_top1.xyxy, prev_teacher_box)
            if (teacher_top1 is not None and prev_teacher_box is not None)
            else float("nan")
        )

        pseudo_accepted = should_accept_pseudo(
            top1=teacher_top1,
            prev_teacher_box=prev_teacher_box,
            img_w=w,
            img_h=h,
            conf_thresh=float(args.teacher_conf_thresh),
            min_area_frac=float(args.min_area_frac),
            max_area_frac=float(args.max_area_frac),
            border_margin_frac=float(args.border_margin_frac),
            temporal_iou_gate=float(args.temporal_iou_gate),
        )

        # -------------------------
        # Student prediction BEFORE update (on original image)
        # -------------------------
        student_model.eval()
        student_pre_top1, student_pre_lat_ms = predict_top1_wrapper(
            student_yolo,
            source=str(img_path),
            device=args.device,
            conf=float(args.infer_conf),
            iou=float(args.iou),
        )

        student_pre_iou_prev = (
            xyxy_iou(student_pre_top1.xyxy, prev_student_pre_box)
            if (student_pre_top1 is not None and prev_student_pre_box is not None)
            else float("nan")
        )

        # -------------------------
        # Optional adaptation step
        # -------------------------
        update_applied = False
        loss_total = float("nan")
        loss_box = float("nan")
        loss_cls = float("nan")
        loss_dfl = float("nan")

        if pseudo_accepted and teacher_top1 is not None:
            # Strong appearance-only augmentation on original image
            student_aug_rgb = strong_augment(img_rgb, rng)
            letterboxed, gain, pad_left, pad_top = letterbox_image(student_aug_rgb, int(args.imgsz))
            img_tensor = pil_to_model_tensor(letterboxed).to(args.device)

            # Convert teacher original-image box -> normalized xywh in student letterboxed coords
            x_c, y_c, bw, bh = xyxy_original_to_norm_xywh_letterboxed(
                box_xyxy=teacher_top1.xyxy,
                orig_w=w,
                orig_h=h,
                size=int(args.imgsz),
                gain=gain,
                pad_left=pad_left,
                pad_top=pad_top,
            )

            batch = {
                "img": img_tensor,
                "batch_idx": torch.tensor([0], dtype=torch.long, device=args.device),
                "cls": torch.tensor([[0.0]], dtype=torch.float32, device=args.device),
                "bboxes": torch.tensor([[x_c, y_c, bw, bh]], dtype=torch.float32, device=args.device),
            }

            student_model.train()
            optimizer.zero_grad(set_to_none=True)
            preds = student_model(img_tensor)
            loss, loss_items = criterion(preds, batch)
            loss.backward()

            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(optim_params, float(args.grad_clip))

            optimizer.step()
            update_teacher_ema(teacher_model, student_model, decay=float(args.ema_decay))
            update_applied = True

            loss_total = float(loss.detach().cpu())
            loss_dict = safe_loss_items_to_dict(loss_items)
            loss_box = loss_dict["box"]
            loss_cls = loss_dict["cls"]
            loss_dfl = loss_dict["dfl"]

        # -------------------------
        # Student prediction AFTER update (on original image)
        # -------------------------
        student_model.eval()
        student_post_top1, student_post_lat_ms = predict_top1_wrapper(
            student_yolo,
            source=str(img_path),
            device=args.device,
            conf=float(args.infer_conf),
            iou=float(args.iou),
        )

        student_post_iou_prev = (
            xyxy_iou(student_post_top1.xyxy, prev_student_post_box)
            if (student_post_top1 is not None and prev_student_post_box is not None)
            else float("nan")
        )

        # Log
        records.append(
            AdaptRecord(
                frame=idx,
                path=str(img_path),
                width=int(w),
                height=int(h),

                teacher_det=int(teacher_top1 is not None),
                teacher_conf=float(teacher_top1.conf) if teacher_top1 is not None else 0.0,
                teacher_x1=float(teacher_top1.xyxy[0]) if teacher_top1 is not None else float("nan"),
                teacher_y1=float(teacher_top1.xyxy[1]) if teacher_top1 is not None else float("nan"),
                teacher_x2=float(teacher_top1.xyxy[2]) if teacher_top1 is not None else float("nan"),
                teacher_y2=float(teacher_top1.xyxy[3]) if teacher_top1 is not None else float("nan"),
                teacher_iou_prev=float(teacher_iou_prev),

                student_pre_det=int(student_pre_top1 is not None),
                student_pre_conf=float(student_pre_top1.conf) if student_pre_top1 is not None else 0.0,
                student_pre_x1=float(student_pre_top1.xyxy[0]) if student_pre_top1 is not None else float("nan"),
                student_pre_y1=float(student_pre_top1.xyxy[1]) if student_pre_top1 is not None else float("nan"),
                student_pre_x2=float(student_pre_top1.xyxy[2]) if student_pre_top1 is not None else float("nan"),
                student_pre_y2=float(student_pre_top1.xyxy[3]) if student_pre_top1 is not None else float("nan"),
                student_pre_iou_prev=float(student_pre_iou_prev),

                student_post_det=int(student_post_top1 is not None),
                student_post_conf=float(student_post_top1.conf) if student_post_top1 is not None else 0.0,
                student_post_x1=float(student_post_top1.xyxy[0]) if student_post_top1 is not None else float("nan"),
                student_post_y1=float(student_post_top1.xyxy[1]) if student_post_top1 is not None else float("nan"),
                student_post_x2=float(student_post_top1.xyxy[2]) if student_post_top1 is not None else float("nan"),
                student_post_y2=float(student_post_top1.xyxy[3]) if student_post_top1 is not None else float("nan"),
                student_post_iou_prev=float(student_post_iou_prev),

                pseudo_accepted=int(pseudo_accepted),
                update_applied=int(update_applied),
                loss_total=float(loss_total),
                loss_box=float(loss_box),
                loss_cls=float(loss_cls),
                loss_dfl=float(loss_dfl),
                latency_teacher_ms=float(teacher_lat_ms),
                latency_student_pre_ms=float(student_pre_lat_ms),
                latency_student_post_ms=float(student_post_lat_ms),
            )
        )

        # Update prevs
        prev_teacher_box = teacher_top1.xyxy if teacher_top1 is not None else None
        prev_student_pre_box = student_pre_top1.xyxy if student_pre_top1 is not None else None
        prev_student_post_box = student_post_top1.xyxy if student_post_top1 is not None else None

        # MP4
        if args.make_mp4:
            use_mp4 = (idx % max(1, int(args.mp4_every)) == 0)
            under_mp4_cap = (int(args.mp4_max) <= 0) or (len(mp4_frames) < int(args.mp4_max))
            if use_mp4 and under_mp4_cap:
                triptych = make_triptych(
                    img_rgb=img_rgb,
                    frame_idx=idx,
                    teacher_top1=teacher_top1,
                    student_top1=student_post_top1,
                    update_applied=bool(update_applied),
                    pseudo_accepted=bool(pseudo_accepted),
                    loss_total=0.0 if math.isnan(loss_total) else float(loss_total),
                )

                if float(args.mp4_scale) != 1.0:
                    new_w = max(1, int(round(triptych.size[0] * float(args.mp4_scale))))
                    new_h = max(1, int(round(triptych.size[1] * float(args.mp4_scale))))
                    triptych = triptych.resize((new_w, new_h), Image.Resampling.BILINEAR)

                mp4_frames.append(np.array(triptych))

    # -------------------------
    # Write CSV
    # -------------------------
    csv_path = out_dir / "adapt_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame", "path", "width", "height",
                "teacher_det", "teacher_conf", "teacher_x1", "teacher_y1", "teacher_x2", "teacher_y2", "teacher_iou_prev",
                "student_pre_det", "student_pre_conf", "student_pre_x1", "student_pre_y1", "student_pre_x2", "student_pre_y2", "student_pre_iou_prev",
                "student_post_det", "student_post_conf", "student_post_x1", "student_post_y1", "student_post_x2", "student_post_y2", "student_post_iou_prev",
                "pseudo_accepted", "update_applied",
                "loss_total", "loss_box", "loss_cls", "loss_dfl",
                "latency_teacher_ms", "latency_student_pre_ms", "latency_student_post_ms",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.frame, r.path, r.width, r.height,
                    r.teacher_det, f"{r.teacher_conf:.6f}",
                    f"{r.teacher_x1:.3f}" if not math.isnan(r.teacher_x1) else "",
                    f"{r.teacher_y1:.3f}" if not math.isnan(r.teacher_y1) else "",
                    f"{r.teacher_x2:.3f}" if not math.isnan(r.teacher_x2) else "",
                    f"{r.teacher_y2:.3f}" if not math.isnan(r.teacher_y2) else "",
                    f"{r.teacher_iou_prev:.6f}" if not math.isnan(r.teacher_iou_prev) else "",

                    r.student_pre_det, f"{r.student_pre_conf:.6f}",
                    f"{r.student_pre_x1:.3f}" if not math.isnan(r.student_pre_x1) else "",
                    f"{r.student_pre_y1:.3f}" if not math.isnan(r.student_pre_y1) else "",
                    f"{r.student_pre_x2:.3f}" if not math.isnan(r.student_pre_x2) else "",
                    f"{r.student_pre_y2:.3f}" if not math.isnan(r.student_pre_y2) else "",
                    f"{r.student_pre_iou_prev:.6f}" if not math.isnan(r.student_pre_iou_prev) else "",

                    r.student_post_det, f"{r.student_post_conf:.6f}",
                    f"{r.student_post_x1:.3f}" if not math.isnan(r.student_post_x1) else "",
                    f"{r.student_post_y1:.3f}" if not math.isnan(r.student_post_y1) else "",
                    f"{r.student_post_x2:.3f}" if not math.isnan(r.student_post_x2) else "",
                    f"{r.student_post_y2:.3f}" if not math.isnan(r.student_post_y2) else "",
                    f"{r.student_post_iou_prev:.6f}" if not math.isnan(r.student_post_iou_prev) else "",

                    r.pseudo_accepted, r.update_applied,
                    f"{r.loss_total:.6f}" if not math.isnan(r.loss_total) else "",
                    f"{r.loss_box:.6f}" if not math.isnan(r.loss_box) else "",
                    f"{r.loss_cls:.6f}" if not math.isnan(r.loss_cls) else "",
                    f"{r.loss_dfl:.6f}" if not math.isnan(r.loss_dfl) else "",
                    f"{r.latency_teacher_ms:.3f}",
                    f"{r.latency_student_pre_ms:.3f}",
                    f"{r.latency_student_post_ms:.3f}",
                ]
            )

    # -------------------------
    # Plots
    # -------------------------
    frames = np.array([r.frame for r in records], dtype=np.int32)
    teacher_conf = np.array([r.teacher_conf for r in records], dtype=np.float32)
    student_pre_conf = np.array([r.student_pre_conf for r in records], dtype=np.float32)
    student_post_conf = np.array([r.student_post_conf for r in records], dtype=np.float32)
    updates = np.array([r.update_applied for r in records], dtype=np.float32)
    pseudo_accept = np.array([r.pseudo_accepted for r in records], dtype=np.float32)
    teacher_iou_prev = np.array([r.teacher_iou_prev for r in records], dtype=np.float32)
    student_post_iou_prev = np.array([r.student_post_iou_prev for r in records], dtype=np.float32)
    loss_total = np.array([r.loss_total for r in records], dtype=np.float32)

    save_plot_line(frames, teacher_conf, "Teacher top-1 Confidence vs Frame", "frame", "conf", out_dir / "plot_teacher_conf.png")
    save_plot_line(frames, student_pre_conf, "Student PRE top-1 Confidence vs Frame", "frame", "conf", out_dir / "plot_student_pre_conf.png")
    save_plot_line(frames, student_post_conf, "Student POST top-1 Confidence vs Frame", "frame", "conf", out_dir / "plot_student_post_conf.png")

    save_plot_line(
        frames,
        pseudo_accept,
        "Pseudo-label Accepted vs Frame",
        "frame",
        "accepted",
        out_dir / "plot_pseudo_accept.png",
    )
    save_plot_line(
        frames,
        updates,
        "Update Applied vs Frame",
        "frame",
        "update",
        out_dir / "plot_update_applied.png",
    )

    updates_roll = np.convolve(updates, np.ones(max(1, int(args.roll))) / max(1, int(args.roll)), mode="same")
    save_plot_line(
        frames,
        updates_roll,
        f"Update Rate (rolling mean, window={args.roll})",
        "frame",
        "update_rate",
        out_dir / "plot_update_rate_roll.png",
    )

    loss_defined = np.where(np.isfinite(loss_total), loss_total, np.nan)
    save_plot_line(frames, loss_defined, "Adaptation Loss vs Frame", "frame", "loss", out_dir / "plot_loss_total.png")
    save_plot_hist(loss_total[np.isfinite(loss_total)], int(args.hist_bins), "Adaptation Loss Histogram", "loss", out_dir / "hist_loss_total.png")

    teacher_iou_defined = np.where(np.isfinite(teacher_iou_prev), teacher_iou_prev, np.nan)
    student_iou_defined = np.where(np.isfinite(student_post_iou_prev), student_post_iou_prev, np.nan)
    save_plot_line(frames, teacher_iou_defined, "Teacher Temporal IoU vs Frame", "frame", "IoU", out_dir / "plot_teacher_iou_prev.png")
    save_plot_line(frames, student_iou_defined, "Student POST Temporal IoU vs Frame", "frame", "IoU", out_dir / "plot_student_post_iou_prev.png")

    save_plot_hist(teacher_conf, int(args.hist_bins), "Teacher top-1 Confidence Histogram", "conf", out_dir / "hist_teacher_conf.png")
    save_plot_hist(student_post_conf, int(args.hist_bins), "Student POST top-1 Confidence Histogram", "conf", out_dir / "hist_student_post_conf.png")

    # -------------------------
    # Summary
    # -------------------------
    n_frames = len(records)
    n_pseudo = int(np.sum(pseudo_accept))
    n_updates = int(np.sum(updates))
    mean_teacher_conf = float(np.mean(teacher_conf)) if n_frames else float("nan")
    mean_student_post_conf = float(np.mean(student_post_conf)) if n_frames else float("nan")
    mean_loss = float(np.nanmean(loss_total)) if np.any(np.isfinite(loss_total)) else float("nan")
    mean_teacher_iou = float(np.nanmean(teacher_iou_prev)) if np.any(np.isfinite(teacher_iou_prev)) else float("nan")
    mean_student_iou = float(np.nanmean(student_post_iou_prev)) if np.any(np.isfinite(student_post_iou_prev)) else float("nan")

    summary_path = out_dir / "adapt_summary.txt"
    summary_lines = [
        "Online Adaptation Summary (Phase-1)",
        f"weights={args.weights}",
        f"dataset={dataset_root}",
        f"frames={n_frames}",
        f"imgsz={int(args.imgsz)}",
        f"teacher_conf_thresh={float(args.teacher_conf_thresh):.3f}",
        f"infer_conf={float(args.infer_conf):.3f}",
        f"iou={float(args.iou):.3f}",
        f"lr={float(args.lr):.2e}",
        f"ema_decay={float(args.ema_decay):.6f}",
        "",
        f"pseudo accepted: {n_pseudo}/{n_frames} ({100.0*n_pseudo/max(1,n_frames):.1f}%)",
        f"updates applied: {n_updates}/{n_frames} ({100.0*n_updates/max(1,n_frames):.1f}%)",
        f"mean teacher conf:       {mean_teacher_conf:.3f}",
        f"mean student post conf:  {mean_student_post_conf:.3f}",
        f"mean adaptation loss:    {mean_loss:.4f}" if not math.isnan(mean_loss) else "mean adaptation loss:    n/a",
        f"mean teacher temporal IoU: {mean_teacher_iou:.3f}" if not math.isnan(mean_teacher_iou) else "mean teacher temporal IoU: n/a",
        f"mean student temporal IoU: {mean_student_iou:.3f}" if not math.isnan(mean_student_iou) else "mean student temporal IoU: n/a",
        "",
        "Outputs:",
        f"  csv: {csv_path.name}",
        "  plots: plot_teacher_conf.png, plot_student_pre_conf.png, plot_student_post_conf.png, plot_pseudo_accept.png, plot_update_applied.png, plot_update_rate_roll.png, plot_loss_total.png, plot_teacher_iou_prev.png, plot_student_post_iou_prev.png",
        "  hists: hist_teacher_conf.png, hist_student_post_conf.png, hist_loss_total.png",
    ]
    if args.make_mp4:
        summary_lines.append("  mp4: adapt_overlay.mp4")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    # -------------------------
    # MP4
    # -------------------------
    if args.make_mp4 and mp4_frames:
        mp4_path = out_dir / "adapt_overlay.mp4"
        with imageio.get_writer(mp4_path, fps=int(args.mp4_fps), codec="libx264", quality=7) as writer:
            for fr in mp4_frames:
                writer.append_data(fr)

    print(f"Done. Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()


# Example:
# PYTHONPATH=. python odad/online_adapt.py \
#   --weights /home/hm25936/mae/runs/yolov8_baseline/baseline/weights/best.pt \
#   --dataset /home/hm25936/datasets_for_yolo/lab_images_6000 \
#   --output /home/hm25936/mae/odad/online_adapt_lab \
#   --device cuda:0 \
#   --imgsz 1024 \
#   --teacher-conf-thresh 0.70 \
#   --infer-conf 0.001 \
#   --temporal-iou-gate 0.30 \
#   --lr 1e-4 \
#   --ema-decay 0.999 \
#   --make-mp4 \
#   --mp4-every 2 \
#   --mp4-fps 12 \
#   --mp4-scale 0.75