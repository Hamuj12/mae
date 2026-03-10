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
from typing import Any, Dict, List, Optional, Tuple

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

from types import SimpleNamespace

try:
    from ultralytics.nn.modules import Detect, Segment
except Exception:  # pragma: no cover
    Detect = None
    Segment = None

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
    student_delta_conf: float
    student_pre_post_iou: float
    student_pre_post_center_shift: float

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

    return Image.fromarray(arr.astype(np.uint8))


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


def unwrap_core_and_layers(student_model: nn.Module) -> Tuple[nn.Module, List[nn.Module]]:
    """
    Unwrap potential YOLO wrapper/core nesting:
      - if student_model is a YOLO wrapper, student_model.model is the core DetectionModel
      - if student_model is already a DetectionModel, its .model is the layer sequence
    """
    maybe_core = getattr(student_model, "model", None)
    core_model = maybe_core if (maybe_core is not None and hasattr(maybe_core, "yaml")) else student_model
    layer_seq = getattr(core_model, "model", core_model)

    if isinstance(layer_seq, (nn.ModuleList, nn.Sequential, list, tuple)):
        layers = list(layer_seq)
    else:
        layers = list(layer_seq.children()) if isinstance(layer_seq, nn.Module) else []

    if not layers:
        raise RuntimeError("Unable to locate YOLO module sequence from loaded student model.")
    return core_model, layers


def is_detect_or_segment_module(module: nn.Module) -> bool:
    if Detect is not None and isinstance(module, Detect):
        return True
    if Segment is not None and isinstance(module, Segment):
        return True
    return module.__class__.__name__ in {"Detect", "Segment"}


def find_head_idx(layers: List[nn.Module]) -> int:
    head_idx = -1
    for idx, module in enumerate(layers):
        if is_detect_or_segment_module(module):
            head_idx = idx
    if head_idx < 0:
        raise RuntimeError("Unable to identify detection head (Detect/Segment) in loaded student model.")
    return head_idx


def resolve_neck_start_idx(core_model: nn.Module, neck_start_idx_override: int) -> Optional[int]:
    if int(neck_start_idx_override) >= 0:
        return int(neck_start_idx_override)

    yaml_cfg = getattr(core_model, "yaml", None)
    if isinstance(yaml_cfg, dict):
        backbone = yaml_cfg.get("backbone")
        if isinstance(backbone, (list, tuple)):
            return len(backbone)
    return None


def compute_unfrozen_indices(
    update_scope: str,
    neck_start_idx: Optional[int],
    head_idx: int,
    n_layers: int,
) -> List[int]:
    if update_scope == "head_only":
        return [head_idx]

    if neck_start_idx is None:
        raise RuntimeError(
            "Unable to infer neck_start_idx from model YAML. "
            "Pass --neck-start-idx >= 0 to use neck+head updates."
        )
    if neck_start_idx < 0 or neck_start_idx >= n_layers:
        raise RuntimeError(
            f"Invalid neck_start_idx={neck_start_idx}; expected range [0, {n_layers - 1}] for this model."
        )
    if neck_start_idx > head_idx:
        raise RuntimeError(
            f"Invalid update region: neck_start_idx={neck_start_idx} is after head_idx={head_idx}."
        )
    return list(range(neck_start_idx, head_idx + 1))


def apply_freeze_policy(student_model: nn.Module, layers: List[nn.Module], unfrozen_indices: List[int]) -> None:
    for p in student_model.parameters():
        p.requires_grad = False
    for idx in unfrozen_indices:
        for p in layers[idx].parameters():
            p.requires_grad = True


def module_grad_l2_norm(module: nn.Module) -> float:
    has_params = False
    sq_sum = 0.0
    for p in module.parameters():
        has_params = True
        if p.grad is not None:
            g = p.grad.detach().float()
            sq_sum += float(torch.sum(g * g).item())
    if not has_params:
        return 0.0
    return math.sqrt(sq_sum)


def clone_module_params(module: nn.Module) -> List[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def module_param_delta_l2_norm(module: nn.Module, params_before: List[torch.Tensor]) -> float:
    params_after = list(module.parameters())
    if not params_after:
        return 0.0
    if len(params_after) != len(params_before):
        raise RuntimeError("Module parameter structure changed during optimizer step.")

    sq_sum = 0.0
    for p_before, p_after in zip(params_before, params_after):
        delta = (p_after.detach() - p_before).float()
        sq_sum += float(torch.sum(delta * delta).item())
    return math.sqrt(sq_sum)


def tensor_l2_norm(tensor: Optional[torch.Tensor]) -> float:
    if tensor is None:
        return 0.0
    t = tensor.detach().float()
    return math.sqrt(float(torch.sum(t * t).item()))


def params_grad_l2_norm(params: List[torch.nn.Parameter]) -> float:
    sq_sum = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach().float()
            sq_sum += float(torch.sum(g * g).item())
    return math.sqrt(sq_sum)


def retain_grad_tensors(obj: Any, prefix: str, sink: List[Tuple[str, torch.Tensor]]) -> None:
    if isinstance(obj, torch.Tensor):
        if obj.requires_grad:
            obj.retain_grad()
            sink.append((prefix, obj))
        return

    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            retain_grad_tensors(item, f"{prefix}.{i}", sink)
        return

    if isinstance(obj, dict):
        for key, item in obj.items():
            retain_grad_tensors(item, f"{prefix}.{key}", sink)


def collect_selected_trainable_params(
    student_model: nn.Module,
    layers: List[nn.Module],
    unfrozen_indices: List[int],
) -> List[Dict[str, object]]:
    """
    Source of truth is student_model.named_parameters() after freeze policy.
    Map each trainable parameter object back to unfrozen module index via module graph identity.
    """
    unfrozen_set = set(int(i) for i in unfrozen_indices)
    param_to_module_idxs: Dict[int, List[int]] = {}
    for idx, module in enumerate(layers):
        for p in module.parameters():
            pid = id(p)
            if pid not in param_to_module_idxs:
                param_to_module_idxs[pid] = []
            param_to_module_idxs[pid].append(idx)

    selected: List[Dict[str, object]] = []
    seen_param_ids = set()
    for name, param in student_model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen_param_ids:
            continue
        idx_candidates = [i for i in param_to_module_idxs.get(pid, []) if i in unfrozen_set]
        if not idx_candidates:
            continue
        module_idx = min(idx_candidates)
        selected.append(
            {
                "name": name,
                "param": param,
                "module_idx": int(module_idx),
                "module_name": layers[module_idx].__class__.__name__,
                "numel": int(param.numel()),
            }
        )
        seen_param_ids.add(pid)

    return selected


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
    p.add_argument(
        "--update-scope",
        type=str,
        default="head_only",
        choices=["head_only", "neck_head"],
        help="Student trainable region: head_only or neck_head",
    )
    p.add_argument(
        "--neck-start-idx",
        type=int,
        default=-1,
        help="Manual neck start module index override; >=0 disables YAML auto-detection",
    )
    p.add_argument("--ema-decay", type=float, default=0.999, help="Teacher EMA decay per accepted update")
    p.add_argument("--grad-clip", type=float, default=10.0, help="Gradient clipping max norm")
    p.add_argument("--debug-first-update", action="store_true", help="Verbose diagnostics on the first accepted update")
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

    core_model, layers = unwrap_core_and_layers(student_model)
    head_idx = find_head_idx(layers)
    neck_start_idx = resolve_neck_start_idx(core_model, int(args.neck_start_idx))
    unfrozen_indices = compute_unfrozen_indices(
        update_scope=str(args.update_scope),
        neck_start_idx=neck_start_idx,
        head_idx=head_idx,
        n_layers=len(layers),
    )
    apply_freeze_policy(student_model, layers, unfrozen_indices)

    if neck_start_idx is None:
        raise RuntimeError(
            "Unable to infer neck_start_idx from model YAML. "
            "Pass --neck-start-idx >= 0 to provide a manual override."
        )

    unfrozen_desc = [f"{i}:{layers[i].__class__.__name__}" for i in unfrozen_indices]
    trainable_param_count = int(sum(p.numel() for p in student_model.parameters() if p.requires_grad))
    print(f"[adapt] update_scope={args.update_scope}")
    print(f"[adapt] neck_start_idx={neck_start_idx}")
    print(f"[adapt] head_idx={head_idx}")
    print(f"[adapt] unfrozen_modules={unfrozen_desc}")
    print(f"[adapt] trainable_params={trainable_param_count}")
    unfrozen_module_tags = [f"m{i}_{layers[i].__class__.__name__}" for i in unfrozen_indices]

    selected_trainable_params = collect_selected_trainable_params(student_model, layers, unfrozen_indices)
    selected_trainable_param_ids = {id(item["param"]) for item in selected_trainable_params}
    selected_trainable_by_module_idx: Dict[int, List[Dict[str, object]]] = {}
    for item in selected_trainable_params:
        mod_idx = int(item["module_idx"])
        if mod_idx not in selected_trainable_by_module_idx:
            selected_trainable_by_module_idx[mod_idx] = []
        selected_trainable_by_module_idx[mod_idx].append(item)

    unmapped_trainable_names = [
        name
        for name, param in student_model.named_parameters()
        if param.requires_grad and id(param) not in selected_trainable_param_ids
    ]
    if unmapped_trainable_names:
        raise RuntimeError(
            "Found trainable student_model parameters not mapped to unfrozen module indices: "
            + ", ".join(unmapped_trainable_names[:10])
        )

    optimizer_module_param_counts: List[Dict[str, object]] = []
    for mod_idx in unfrozen_indices:
        module = layers[mod_idx]
        module_params = list(module.parameters())
        module_total_numel = int(sum(p.numel() for p in module_params))
        module_optimizer_numel = int(
            sum(int(item["numel"]) for item in selected_trainable_by_module_idx.get(int(mod_idx), []))
        )

        optimizer_module_param_counts.append(
            {
                "idx": int(mod_idx),
                "name": module.__class__.__name__,
                "module_total_numel": module_total_numel,
                "optimizer_numel": module_optimizer_numel,
            }
        )
        if module_total_numel > 0 and module_optimizer_numel == 0:
            raise RuntimeError(
                f"Unfrozen parameterized module idx={mod_idx} ({module.__class__.__name__}) has zero optimizer parameters."
            )

    optim_params = [item["param"] for item in selected_trainable_params]
    if not optim_params:
        raise RuntimeError("No trainable parameters found after applying freeze policy.")
    optimizer_total_param_count = int(sum(int(item["numel"]) for item in selected_trainable_params))
    print("[adapt] optimizer_module_param_counts:")
    for item in optimizer_module_param_counts:
        optimizer_member = 1 if (int(item["module_total_numel"]) == 0 or int(item["optimizer_numel"]) > 0) else 0
        print(
            "[adapt]   "
            f"idx={item['idx']} class={item['name']} "
            f"module_total_params={item['module_total_numel']} "
            f"optimizer_params={item['optimizer_numel']} "
            f"optimizer_member={optimizer_member}"
        )
    print(f"[adapt] optimizer_total_params={optimizer_total_param_count}")

    optimizer = torch.optim.SGD(
        optim_params,
        lr=float(args.lr),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
    )

    # Ultralytics loss expects model.args / criterion.hyp to have attribute access
    # and to contain at least box / cls / dfl gains.
    if isinstance(student_model.args, dict):
        hyp_dict = dict(student_model.args)
    elif isinstance(student_model.args, SimpleNamespace):
        hyp_dict = vars(student_model.args).copy()
    else:
        hyp_dict = vars(student_model.args).copy() if hasattr(student_model.args, "__dict__") else {}

    # Set safe defaults if missing
    hyp_dict.setdefault("box", 7.5)
    hyp_dict.setdefault("cls", 0.5)
    hyp_dict.setdefault("dfl", 1.5)

    student_model.args = SimpleNamespace(**hyp_dict)
    criterion = LossClass(student_model)
    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing_optimizer_names = [
        str(item["name"]) for item in selected_trainable_params if id(item["param"]) not in optimizer_param_ids
    ]
    if missing_optimizer_names:
        raise RuntimeError(
            "Some selected trainable student_model params are not in optimizer param groups: "
            + ", ".join(missing_optimizer_names[:10])
        )
    first_update_debug_done = False
    first_update_debug_lines: List[str] = []
    first_update_debug_records: List[Dict[str, object]] = []
    first_update_debug_path = out_dir / "first_update_debug.txt"
    first_update_debug_csv_path = out_dir / "first_update_debug.csv"
    first_update_activation_debug_lines: List[str] = []
    first_update_activation_debug_path = out_dir / "first_update_activation_debug.txt"

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
    update_debug_records: List[Dict[str, object]] = []
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
        update_grad_norms = [float("nan")] * len(unfrozen_indices)
        update_delta_norms = [float("nan")] * len(unfrozen_indices)

        if pseudo_accepted and teacher_top1 is not None:
            # Ultralytics predict() can force model params to requires_grad=False.
            # Re-apply selected trainability scope before each adaptation step.
            apply_freeze_policy(student_model, layers, unfrozen_indices)

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

            # IMPORTANT: explicitly re-enable grad mode after Ultralytics predict()
            with torch.inference_mode(False):
                with torch.enable_grad():
                    student_model.train()
                    optimizer.zero_grad(set_to_none=True)
                    debug_this_update = bool(args.debug_first_update and (not first_update_debug_done))
                    activation_target_indices = [12, 15, 18, 21]
                    activation_module_tensor_refs: Dict[int, List[Tuple[str, torch.Tensor]]] = {}
                    detect_input_tensor_refs: List[Tuple[str, torch.Tensor]] = []
                    detect_output_tensor_refs: List[Tuple[str, torch.Tensor]] = []
                    activation_hook_handles = []
                    if debug_this_update:
                        def make_activation_hook(mod_idx: int):
                            def _hook(_module, _inputs, output):
                                refs: List[Tuple[str, torch.Tensor]] = []
                                retain_grad_tensors(output, f"layer{mod_idx}", refs)
                                activation_module_tensor_refs[mod_idx] = refs
                            return _hook

                        for mod_idx in activation_target_indices:
                            if 0 <= mod_idx < len(layers):
                                activation_hook_handles.append(layers[mod_idx].register_forward_hook(make_activation_hook(mod_idx)))

                        def detect_io_hook(_module, inputs, output):
                            detect_input_tensor_refs.clear()
                            detect_output_tensor_refs.clear()
                            if isinstance(inputs, tuple) and len(inputs) > 0:
                                retain_grad_tensors(inputs[0], "detect_in", detect_input_tensor_refs)
                            else:
                                retain_grad_tensors(inputs, "detect_in", detect_input_tensor_refs)
                            retain_grad_tensors(output, "detect_out", detect_output_tensor_refs)

                        if 0 <= head_idx < len(layers):
                            activation_hook_handles.append(layers[head_idx].register_forward_hook(detect_io_hook))

                    try:
                        preds = student_model(img_tensor)
                    finally:
                        for handle in activation_hook_handles:
                            handle.remove()

                    # helpful one-time debug if needed
                    # print("grad enabled:", torch.is_grad_enabled())
                    # print("student training:", student_model.training)
                    # if isinstance(preds, (list, tuple)):
                    #     print("pred types:", [type(p) for p in preds])
                    #     for i, p in enumerate(preds):
                    #         if isinstance(p, torch.Tensor):
                    #             print(f"pred[{i}].requires_grad =", p.requires_grad)

                    def unpack_loss_pair(loss_out_obj):
                        if not (isinstance(loss_out_obj, (list, tuple)) and len(loss_out_obj) == 2):
                            raise RuntimeError(f"Unexpected loss output type: {type(loss_out_obj)}")
                        x, y = loss_out_obj
                        x_ok = isinstance(x, torch.Tensor) and (x.requires_grad or x.grad_fn is not None)
                        y_ok = isinstance(y, torch.Tensor) and (y.requires_grad or y.grad_fn is not None)
                        if x_ok:
                            return x, y
                        if y_ok:
                            return y, x
                        return None, None

                    # Prefer model-native loss API when available (more version-robust).
                    if hasattr(student_model, "loss"):
                        loss_out = student_model.loss(batch, preds=preds)
                    else:
                        loss_out = criterion(preds, batch)

                    loss, loss_items = unpack_loss_pair(loss_out)

                    # Fallback: some Ultralytics versions return detached tensors for the direct criterion path.
                    if loss is None and hasattr(student_model, "loss"):
                        loss_out = student_model.loss(batch)
                        loss, loss_items = unpack_loss_pair(loss_out)

                    if loss is None:
                        if isinstance(loss_out, (list, tuple)) and len(loss_out) == 2:
                            a, b = loss_out
                            raise RuntimeError(
                                "Loss outputs are both detached. "
                                f"a.requires_grad={getattr(a, 'requires_grad', None)}, "
                                f"b.requires_grad={getattr(b, 'requires_grad', None)}"
                            )
                        raise RuntimeError(f"Unexpected loss output type: {type(loss_out)}")

                    # Some Ultralytics versions return a vector [box, cls, dfl].
                    # Backprop requires a scalar.
                    loss_scalar = loss.sum() if isinstance(loss, torch.Tensor) and loss.ndim > 0 else loss
                    loss_scalar.backward()
                    params_before_step_by_idx = {}
                    for mod_pos, mod_idx in enumerate(unfrozen_indices):
                        module = layers[mod_idx]
                        update_grad_norms[mod_pos] = module_grad_l2_norm(module)
                        params_before_step_by_idx[mod_idx] = clone_module_params(module)
                    debug_param_records_pre: List[Dict[str, object]] = []
                    global_grad_before_clip = float("nan")
                    global_grad_after_clip = float("nan")
                    optimizer_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float("nan")
                    activation_module_grad_summary: Dict[int, Dict[str, object]] = {}
                    detect_input_grad_summary: List[Dict[str, object]] = []
                    detect_output_grad_summary: List[Dict[str, object]] = []
                    detect_branch_grad_summary: Dict[str, Dict[str, float]] = {}
                    if debug_this_update:
                        global_grad_before_clip = params_grad_l2_norm(optim_params)
                        for selected_entry in selected_trainable_params:
                            param = selected_entry["param"]
                            in_optimizer = bool(id(param) in optimizer_param_ids)
                            if not in_optimizer:
                                raise RuntimeError(
                                    "Selected debug parameter is not present in optimizer params: "
                                    f"{selected_entry['name']} (module idx {selected_entry['module_idx']})."
                                )
                            debug_param_records_pre.append(
                                {
                                    "param_name": str(selected_entry["name"]),
                                    "module_idx": int(selected_entry["module_idx"]),
                                    "module_name": str(selected_entry["module_name"]),
                                    "shape": tuple(param.shape),
                                    "param_obj": param,
                                    "pre_tensor": param.detach().clone(),
                                    "grad_norm": tensor_l2_norm(param.grad),
                                    "optimizer_member": int(in_optimizer),
                                }
                            )

                        for mod_idx in activation_target_indices:
                            refs = activation_module_tensor_refs.get(mod_idx, [])
                            sq_sum = 0.0
                            n_nonzero = 0
                            for _, tensor_ref in refs:
                                grad_norm = tensor_l2_norm(tensor_ref.grad)
                                sq_sum += float(grad_norm * grad_norm)
                                if grad_norm > 1e-12:
                                    n_nonzero += 1
                            activation_module_grad_summary[int(mod_idx)] = {
                                "module_name": layers[mod_idx].__class__.__name__ if 0 <= mod_idx < len(layers) else "n/a",
                                "num_tensors": len(refs),
                                "num_nonzero": int(n_nonzero),
                                "grad_l2": math.sqrt(sq_sum),
                            }

                        for name, tensor_ref in detect_input_tensor_refs:
                            detect_input_grad_summary.append(
                                {
                                    "name": str(name),
                                    "shape": tuple(tensor_ref.shape),
                                    "grad_l2": float(tensor_l2_norm(tensor_ref.grad)),
                                }
                            )
                        for name, tensor_ref in detect_output_tensor_refs:
                            detect_output_grad_summary.append(
                                {
                                    "name": str(name),
                                    "shape": tuple(tensor_ref.shape),
                                    "grad_l2": float(tensor_l2_norm(tensor_ref.grad)),
                                }
                            )

                        detect_prefix = f"model.{head_idx}."
                        for selected_entry in selected_trainable_params:
                            if int(selected_entry["module_idx"]) != int(head_idx):
                                continue
                            local_name = str(selected_entry["name"])
                            if local_name.startswith(detect_prefix):
                                local_name = local_name[len(detect_prefix):]
                            parts = local_name.split(".")
                            if len(parts) >= 2 and parts[0] in {"cv2", "cv3"} and parts[1].isdigit():
                                branch_key = f"{parts[0]}.{parts[1]}"
                            elif parts:
                                branch_key = parts[0]
                            else:
                                branch_key = "unknown"

                            if branch_key not in detect_branch_grad_summary:
                                detect_branch_grad_summary[branch_key] = {
                                    "num_params": 0.0,
                                    "num_nonzero_grad": 0.0,
                                    "sum_grad_l2": 0.0,
                                    "max_grad_l2": 0.0,
                                }
                            stat = detect_branch_grad_summary[branch_key]
                            grad_norm = float(tensor_l2_norm(selected_entry["param"].grad))
                            stat["num_params"] += 1.0
                            stat["sum_grad_l2"] += grad_norm
                            if grad_norm > 1e-12:
                                stat["num_nonzero_grad"] += 1.0
                            stat["max_grad_l2"] = max(float(stat["max_grad_l2"]), grad_norm)

                    if float(args.grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(optim_params, float(args.grad_clip))
                    if debug_this_update:
                        global_grad_after_clip = params_grad_l2_norm(optim_params)

                    optimizer.step()
                    for mod_pos, mod_idx in enumerate(unfrozen_indices):
                        update_delta_norms[mod_pos] = module_param_delta_l2_norm(
                            layers[mod_idx],
                            params_before_step_by_idx[mod_idx],
                        )
                    if debug_this_update:
                        debug_lines = [
                            "First Accepted Update Debug",
                            f"frame={idx}",
                            f"optimizer_lr={optimizer_lr:.8e}",
                            f"global_grad_l2_before_clip={global_grad_before_clip:.8e}",
                            f"global_grad_l2_after_clip={global_grad_after_clip:.8e}",
                            "",
                            f"num_selected_trainable_params={len(debug_param_records_pre)}",
                            "",
                        ]
                        if not debug_param_records_pre:
                            debug_lines.append("No selected trainable parameters available for first-update diagnostics.")

                        first_update_debug_records = []
                        for item in debug_param_records_pre:
                            param = item["param_obj"]
                            pre_tensor = item["pre_tensor"]
                            post_tensor = param.detach()
                            delta = (post_tensor - pre_tensor).float()
                            delta_norm = tensor_l2_norm(delta)
                            max_abs_delta = float(torch.max(torch.abs(delta)).item()) if delta.numel() > 0 else 0.0
                            first_update_debug_records.append(
                                {
                                    "param_name": str(item["param_name"]),
                                    "module_idx": int(item["module_idx"]),
                                    "module_name": str(item["module_name"]),
                                    "shape": tuple(item["shape"]),
                                    "grad_l2": float(item["grad_norm"]),
                                    "delta_l2": float(delta_norm),
                                    "max_abs_delta": float(max_abs_delta),
                                    "optimizer_member": int(item["optimizer_member"]),
                                }
                            )

                        eps = 1e-12
                        top_grad = sorted(first_update_debug_records, key=lambda r: float(r["grad_l2"]), reverse=True)[:20]
                        top_delta = sorted(first_update_debug_records, key=lambda r: float(r["delta_l2"]), reverse=True)[:20]

                        debug_lines.append("Top 20 Parameters by grad_l2")
                        for rank, rec in enumerate(top_grad, start=1):
                            debug_lines.append(
                                f"{rank:02d} grad_l2={float(rec['grad_l2']):.8e} delta_l2={float(rec['delta_l2']):.8e} "
                                f"module_idx={int(rec['module_idx'])} class={rec['module_name']} name={rec['param_name']}"
                            )
                        debug_lines.append("")
                        debug_lines.append("Top 20 Parameters by delta_l2")
                        for rank, rec in enumerate(top_delta, start=1):
                            debug_lines.append(
                                f"{rank:02d} delta_l2={float(rec['delta_l2']):.8e} grad_l2={float(rec['grad_l2']):.8e} "
                                f"module_idx={int(rec['module_idx'])} class={rec['module_name']} name={rec['param_name']}"
                            )

                        per_module: Dict[Tuple[int, str], Dict[str, float]] = {}
                        for rec in first_update_debug_records:
                            key = (int(rec["module_idx"]), str(rec["module_name"]))
                            if key not in per_module:
                                per_module[key] = {
                                    "num_params": 0.0,
                                    "num_nonzero_grad": 0.0,
                                    "num_nonzero_delta": 0.0,
                                    "sum_grad_l2": 0.0,
                                    "sum_delta_l2": 0.0,
                                    "max_delta_l2": 0.0,
                                }
                            stat = per_module[key]
                            grad_v = float(rec["grad_l2"])
                            delta_v = float(rec["delta_l2"])
                            stat["num_params"] += 1.0
                            stat["sum_grad_l2"] += grad_v
                            stat["sum_delta_l2"] += delta_v
                            if grad_v > eps:
                                stat["num_nonzero_grad"] += 1.0
                            if delta_v > eps:
                                stat["num_nonzero_delta"] += 1.0
                            stat["max_delta_l2"] = max(float(stat["max_delta_l2"]), delta_v)

                        debug_lines.append("")
                        debug_lines.append("Per-Module Aggregation (selected optimizer-bound params only)")
                        for key in sorted(per_module.keys(), key=lambda x: x[0]):
                            mod_idx, mod_name = key
                            stat = per_module[key]
                            n = max(1.0, stat["num_params"])
                            debug_lines.append(
                                f"module_idx={mod_idx} class={mod_name} "
                                f"num_params={int(stat['num_params'])} "
                                f"num_nonzero_grad={int(stat['num_nonzero_grad'])} "
                                f"num_nonzero_delta={int(stat['num_nonzero_delta'])} "
                                f"mean_grad_l2={float(stat['sum_grad_l2'])/n:.8e} "
                                f"mean_delta_l2={float(stat['sum_delta_l2'])/n:.8e} "
                                f"max_delta_l2={float(stat['max_delta_l2']):.8e}"
                            )

                        activation_debug_lines = [
                            "First Accepted Update Activation Debug",
                            f"frame={idx}",
                            f"head_idx={head_idx}",
                            "",
                            "Activation Grad Norms at Selected Neck Modules",
                        ]
                        for mod_idx in activation_target_indices:
                            if mod_idx not in activation_module_grad_summary:
                                activation_debug_lines.append(f"module_idx={mod_idx} status=not_captured")
                                continue
                            mod_stat = activation_module_grad_summary[mod_idx]
                            activation_debug_lines.append(
                                f"module_idx={mod_idx} class={mod_stat['module_name']} "
                                f"num_tensors={int(mod_stat['num_tensors'])} "
                                f"num_nonzero={int(mod_stat['num_nonzero'])} "
                                f"activation_grad_l2={float(mod_stat['grad_l2']):.8e}"
                            )

                        activation_debug_lines.append("")
                        activation_debug_lines.append("Detect Input Activation Grad Norms")
                        if not detect_input_grad_summary:
                            activation_debug_lines.append("detect_inputs=not_accessible_or_no_grad_tensors")
                        for rec in detect_input_grad_summary:
                            activation_debug_lines.append(
                                f"name={rec['name']} shape={rec['shape']} grad_l2={float(rec['grad_l2']):.8e}"
                            )

                        activation_debug_lines.append("")
                        activation_debug_lines.append("Detect Output Activation Grad Norms")
                        if not detect_output_grad_summary:
                            activation_debug_lines.append("detect_outputs=not_accessible_or_no_grad_tensors")
                        for rec in detect_output_grad_summary:
                            activation_debug_lines.append(
                                f"name={rec['name']} shape={rec['shape']} grad_l2={float(rec['grad_l2']):.8e}"
                            )

                        activation_debug_lines.append("")
                        activation_debug_lines.append("Detect Branch Parameter Grad Summary")
                        if not detect_branch_grad_summary:
                            activation_debug_lines.append("detect_branch_summary=unavailable")
                        for branch_key in sorted(detect_branch_grad_summary.keys()):
                            stat = detect_branch_grad_summary[branch_key]
                            n_params = max(1.0, float(stat["num_params"]))
                            activation_debug_lines.append(
                                f"branch={branch_key} "
                                f"num_params={int(stat['num_params'])} "
                                f"num_nonzero_grad={int(stat['num_nonzero_grad'])} "
                                f"mean_grad_l2={float(stat['sum_grad_l2'])/n_params:.8e} "
                                f"max_grad_l2={float(stat['max_grad_l2']):.8e}"
                            )

                        first_update_debug_lines = debug_lines
                        first_update_activation_debug_lines = activation_debug_lines
                        for line in debug_lines:
                            print(f"[first-update-debug] {line}")
                        for line in activation_debug_lines:
                            print(f"[first-update-activation-debug] {line}")
                        first_update_debug_done = True

            # teacher follows student after successful update
            update_teacher_ema(teacher_model, student_model, decay=float(args.ema_decay))
            update_applied = True

            loss_total = float(loss_scalar.detach().cpu())
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

        student_pre_conf_val = float(student_pre_top1.conf) if student_pre_top1 is not None else 0.0
        student_post_conf_val = float(student_post_top1.conf) if student_post_top1 is not None else 0.0
        student_delta_conf = student_post_conf_val - student_pre_conf_val

        if student_pre_top1 is not None and student_post_top1 is not None:
            student_pre_post_iou = xyxy_iou(student_pre_top1.xyxy, student_post_top1.xyxy)
            pre_cx, pre_cy = box_center(student_pre_top1.xyxy)
            post_cx, post_cy = box_center(student_post_top1.xyxy)
            student_pre_post_center_shift = math.sqrt((post_cx - pre_cx) ** 2 + (post_cy - pre_cy) ** 2)
        else:
            student_pre_post_iou = float("nan")
            student_pre_post_center_shift = float("nan")

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
                student_pre_conf=student_pre_conf_val,
                student_pre_x1=float(student_pre_top1.xyxy[0]) if student_pre_top1 is not None else float("nan"),
                student_pre_y1=float(student_pre_top1.xyxy[1]) if student_pre_top1 is not None else float("nan"),
                student_pre_x2=float(student_pre_top1.xyxy[2]) if student_pre_top1 is not None else float("nan"),
                student_pre_y2=float(student_pre_top1.xyxy[3]) if student_pre_top1 is not None else float("nan"),
                student_pre_iou_prev=float(student_pre_iou_prev),

                student_post_det=int(student_post_top1 is not None),
                student_post_conf=student_post_conf_val,
                student_post_x1=float(student_post_top1.xyxy[0]) if student_post_top1 is not None else float("nan"),
                student_post_y1=float(student_post_top1.xyxy[1]) if student_post_top1 is not None else float("nan"),
                student_post_x2=float(student_post_top1.xyxy[2]) if student_post_top1 is not None else float("nan"),
                student_post_y2=float(student_post_top1.xyxy[3]) if student_post_top1 is not None else float("nan"),
                student_post_iou_prev=float(student_post_iou_prev),
                student_delta_conf=float(student_delta_conf),
                student_pre_post_iou=float(student_pre_post_iou),
                student_pre_post_center_shift=float(student_pre_post_center_shift),

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
        update_debug_records.append(
            {
                "frame": idx,
                "update_applied": int(update_applied),
                "grad_norms": list(update_grad_norms),
                "delta_norms": list(update_delta_norms),
            }
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
                "student_delta_conf", "student_pre_post_iou", "student_pre_post_center_shift",
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
                    f"{r.student_delta_conf:.6f}",
                    f"{r.student_pre_post_iou:.6f}" if not math.isnan(r.student_pre_post_iou) else "",
                    f"{r.student_pre_post_center_shift:.6f}" if not math.isnan(r.student_pre_post_center_shift) else "",

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

    update_debug_csv_path = out_dir / "update_debug.csv"
    with update_debug_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["frame", "update_applied"]
        header.extend([f"grad_{tag}" for tag in unfrozen_module_tags])
        header.extend([f"delta_{tag}" for tag in unfrozen_module_tags])
        writer.writerow(header)

        for rec in update_debug_records:
            row = [int(rec["frame"]), int(rec["update_applied"])]
            grad_vals = rec["grad_norms"]
            delta_vals = rec["delta_norms"]
            row.extend([f"{v:.8e}" if math.isfinite(v) else "" for v in grad_vals])
            row.extend([f"{v:.8e}" if math.isfinite(v) else "" for v in delta_vals])
            writer.writerow(row)

    update_debug_summary_path = out_dir / "update_debug_summary.txt"
    applied_updates = [r for r in update_debug_records if int(r["update_applied"]) == 1]
    debug_lines = [
        "Update Debug Summary",
        f"frames_total={len(update_debug_records)}",
        f"updates_applied={len(applied_updates)}",
        "",
    ]
    for mod_pos, mod_idx in enumerate(unfrozen_indices):
        mod_name = layers[mod_idx].__class__.__name__
        grad_vals = [
            float(r["grad_norms"][mod_pos])
            for r in applied_updates
            if math.isfinite(float(r["grad_norms"][mod_pos]))
        ]
        delta_vals = [
            float(r["delta_norms"][mod_pos])
            for r in applied_updates
            if math.isfinite(float(r["delta_norms"][mod_pos]))
        ]
        mean_grad = float(np.mean(grad_vals)) if grad_vals else float("nan")
        mean_delta = float(np.mean(delta_vals)) if delta_vals else float("nan")
        debug_lines.append(
            (
                f"module_idx={mod_idx} class={mod_name} "
                f"mean_grad_l2={mean_grad:.8e} "
                f"mean_delta_l2={mean_delta:.8e} "
                f"samples={len(grad_vals)}"
            )
        )
    update_debug_summary_path.write_text("\n".join(debug_lines), encoding="utf-8")
    if args.debug_first_update:
        with first_update_debug_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "param_name",
                    "module_idx",
                    "module_name",
                    "shape",
                    "grad_l2",
                    "delta_l2",
                    "max_abs_delta",
                    "optimizer_member",
                ]
            )
            for rec in first_update_debug_records:
                writer.writerow(
                    [
                        rec["param_name"],
                        int(rec["module_idx"]),
                        rec["module_name"],
                        str(tuple(rec["shape"])),
                        f"{float(rec['grad_l2']):.8e}",
                        f"{float(rec['delta_l2']):.8e}",
                        f"{float(rec['max_abs_delta']):.8e}",
                        int(rec["optimizer_member"]),
                    ]
                )
        if not first_update_debug_lines:
            first_update_debug_lines = [
                "First Accepted Update Debug",
                "No accepted update encountered; detailed first-update diagnostics were not collected.",
            ]
        first_update_debug_path.write_text("\n".join(first_update_debug_lines), encoding="utf-8")
        if not first_update_activation_debug_lines:
            first_update_activation_debug_lines = [
                "First Accepted Update Activation Debug",
                "No accepted update encountered; activation-flow diagnostics were not collected.",
            ]
        first_update_activation_debug_path.write_text("\n".join(first_update_activation_debug_lines), encoding="utf-8")

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
    student_delta_conf = np.array([r.student_delta_conf for r in records], dtype=np.float32)
    student_pre_post_iou = np.array([r.student_pre_post_iou for r in records], dtype=np.float32)
    student_pre_post_center_shift = np.array([r.student_pre_post_center_shift for r in records], dtype=np.float32)
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
    save_plot_line(frames, student_delta_conf, "Student POST-PRE Confidence Delta vs Frame", "frame", "delta_conf", out_dir / "plot_student_delta_conf.png")
    save_plot_hist(student_delta_conf, int(args.hist_bins), "Student POST-PRE Confidence Delta Histogram", "delta_conf", out_dir / "hist_student_delta_conf.png")
    save_plot_line(
        frames,
        np.where(np.isfinite(student_pre_post_iou), student_pre_post_iou, np.nan),
        "Student PRE-POST IoU vs Frame",
        "frame",
        "IoU",
        out_dir / "plot_student_pre_post_iou.png",
    )
    save_plot_line(
        frames,
        np.where(np.isfinite(student_pre_post_center_shift), student_pre_post_center_shift, np.nan),
        "Student PRE-POST Center Shift vs Frame",
        "frame",
        "pixels",
        out_dir / "plot_student_pre_post_center_shift.png",
    )

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
    mean_delta_conf_all = float(np.mean(student_delta_conf)) if n_frames else float("nan")
    update_mask = updates == 1
    mean_delta_conf_updated_only = float(np.mean(student_delta_conf[update_mask])) if np.any(update_mask) else float("nan")
    improved_updated_fraction = (
        float(np.mean(student_delta_conf[update_mask] > 0.0)) if np.any(update_mask) else float("nan")
    )
    mean_pre_post_iou = float(np.nanmean(student_pre_post_iou)) if np.any(np.isfinite(student_pre_post_iou)) else float("nan")
    mean_pre_post_center_shift = (
        float(np.nanmean(student_pre_post_center_shift))
        if np.any(np.isfinite(student_pre_post_center_shift))
        else float("nan")
    )

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
        f"mean student delta conf (all frames): {mean_delta_conf_all:.6f}" if not math.isnan(mean_delta_conf_all) else "mean student delta conf (all frames): n/a",
        f"mean student delta conf (updated frames): {mean_delta_conf_updated_only:.6f}" if not math.isnan(mean_delta_conf_updated_only) else "mean student delta conf (updated frames): n/a",
        f"fraction updated frames improved: {improved_updated_fraction:.6f}" if not math.isnan(improved_updated_fraction) else "fraction updated frames improved: n/a",
        f"mean student pre-post IoU: {mean_pre_post_iou:.6f}" if not math.isnan(mean_pre_post_iou) else "mean student pre-post IoU: n/a",
        f"mean student pre-post center shift: {mean_pre_post_center_shift:.6f}" if not math.isnan(mean_pre_post_center_shift) else "mean student pre-post center shift: n/a",
        "",
        "Outputs:",
        f"  csv: {csv_path.name}",
        f"  update_debug_csv: {update_debug_csv_path.name}",
        f"  update_debug_summary: {update_debug_summary_path.name}",
        "  plots: plot_teacher_conf.png, plot_student_pre_conf.png, plot_student_post_conf.png, plot_pseudo_accept.png, plot_update_applied.png, plot_update_rate_roll.png, plot_loss_total.png, plot_teacher_iou_prev.png, plot_student_post_iou_prev.png, plot_student_delta_conf.png, plot_student_pre_post_iou.png, plot_student_pre_post_center_shift.png",
        "  hists: hist_teacher_conf.png, hist_student_post_conf.png, hist_loss_total.png, hist_student_delta_conf.png",
    ]
    if args.debug_first_update:
        summary_lines.append(f"  first_update_debug: {first_update_debug_path.name}")
        summary_lines.append(f"  first_update_debug_csv: {first_update_debug_csv_path.name}")
        summary_lines.append(f"  first_update_activation_debug: {first_update_activation_debug_path.name}")
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
#   --output /home/hm25936/mae/odad/online_adapt_lab_neck_head_update \
#   --device cuda:0 \
#   --imgsz 1024 \
#   --teacher-conf-thresh 0.80 \
#   --infer-conf 0.001 \
#   --temporal-iou-gate 0.50 \
#   --lr 3e-4 \
#   --ema-decay 0.999 \
#   --update-scope neck_head \
#   --debug-first-update \
#   --max-frames 300 \
#   --make-mp4 \
#   --mp4-every 2 \
#   --mp4-fps 12 \
#   --mp4-scale 0.75
