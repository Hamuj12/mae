#!/usr/bin/env python3
"""Online teacher-student adaptation with buffered replay updates.

This baseline keeps a conservative pseudo-label gate and an EMA teacher, but
updates the student from mini-batches sampled from a temporal replay buffer.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install ultralytics: pip install ultralytics") from exc

try:
    from ultralytics.nn.modules import Detect, Segment
except Exception:  # pragma: no cover
    Detect = None
    Segment = None

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


@dataclass
class Top1Det:
    conf: float
    cls_id: int
    xyxy: Tuple[float, float, float, float]


@dataclass
class MemoryEntry:
    frame_idx: int
    path: str
    width: int
    height: int
    pseudo_box: Tuple[float, float, float, float]
    pseudo_cls: int
    teacher_conf: float


@dataclass
class FrameLog:
    frame: int
    path: str
    teacher_conf: float
    accepted: int
    num_pseudo_boxes_used: int
    buffer_size: int
    update_applied: int
    updates_this_frame: int
    batch_size_used: int
    det_loss: float
    teacher_latency_ms: float
    student_post_conf: float
    student_post_latency_ms: float
    update_latency_ms: float
    frame_latency_ms: float


class ReplayBuffer:
    def __init__(self, max_size: int, rng: random.Random) -> None:
        self._entries: Deque[MemoryEntry] = deque(maxlen=max(1, int(max_size)))
        self._rng = rng

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)

    def sample(self, batch_size: int, mode: str) -> List[MemoryEntry]:
        if not self._entries:
            return []
        k = min(max(1, int(batch_size)), len(self._entries))
        entries = list(self._entries)
        if mode == "recent":
            return entries[-k:]
        if mode == "random":
            idxs = self._rng.sample(range(len(entries)), k=k)
            return [entries[i] for i in idxs]
        raise RuntimeError(f"Unsupported buffer sample mode: {mode}")


def list_test_images(dataset_root: Path) -> List[Path]:
    test_dir = dataset_root / "images" / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Expected YOLO-root dataset at {dataset_root}, missing: {test_dir}")
    images = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    if not images:
        raise RuntimeError(f"No images found under {test_dir}")
    return images


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
    return 0.0 if union <= 0.0 else float(inter_area / union)


def area_fraction(box: Tuple[float, float, float, float], width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    img_area = max(1.0, float(width * height))
    return float(box_area / img_area)


def near_border(box: Tuple[float, float, float, float], width: int, height: int, margin_frac: float) -> bool:
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


def letterbox_image(
    img_rgb: Image.Image,
    size: int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, float, int, int]:
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
    out = img_rgb.copy()

    brightness = 0.75 + 0.5 * rng.random()
    contrast = 0.75 + 0.5 * rng.random()
    out = ImageEnhance.Brightness(out).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)

    if rng.random() < 0.35:
        radius = 0.5 + 1.5 * rng.random()
        out = out.filter(ImageFilter.GaussianBlur(radius=radius))

    if rng.random() < 0.5:
        sigma = rng.uniform(2.0, 8.0)
        arr = np.array(out).astype(np.float32)
        arr += np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr, 0.0, 255.0)
        out = Image.fromarray(arr.astype(np.uint8))

    return out


def xyxy_original_to_norm_xywh_letterboxed(
    box_xyxy: Tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
    size: int,
    gain: float,
    pad_left: int,
    pad_top: int,
) -> Tuple[float, float, float, float]:
    del orig_w, orig_h
    x1, y1, x2, y2 = box_xyxy

    x1_l = x1 * gain + pad_left
    y1_l = y1 * gain + pad_top
    x2_l = x2 * gain + pad_left
    y2_l = y2 * gain + pad_top

    x_c = ((x1_l + x2_l) / 2.0) / size
    y_c = ((y1_l + y2_l) / 2.0) / size
    bw = max(0.0, x2_l - x1_l) / size
    bh = max(0.0, y2_l - y1_l) / size

    return (
        min(1.0, max(0.0, x_c)),
        min(1.0, max(0.0, y_c)),
        min(1.0, max(0.0, bw)),
        min(1.0, max(0.0, bh)),
    )


def pil_to_model_tensor(img_rgb: Image.Image) -> torch.Tensor:
    arr = np.array(img_rgb).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def top1_from_results(results: Any) -> Optional[Top1Det]:
    if not results:
        return None
    r = results[0]
    boxes = r.boxes
    if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
        return None

    confs = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls_vals = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else None

    idx = int(np.argmax(confs))
    cls_id = int(cls_vals[idx]) if cls_vals is not None else 0
    x1, y1, x2, y2 = map(float, xyxy[idx].tolist())
    return Top1Det(conf=float(confs[idx]), cls_id=cls_id, xyxy=(x1, y1, x2, y2))


def predict_top1_wrapper(
    yolo_wrapper: YOLO,
    source: Any,
    device: str,
    conf: float,
    iou: float,
) -> Tuple[Optional[Top1Det], float]:
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

    return top1_from_results(results), latency_ms


def unwrap_core_and_layers(model: nn.Module) -> Tuple[nn.Module, List[nn.Module]]:
    maybe_core = getattr(model, "model", None)
    core_model = maybe_core if (maybe_core is not None and hasattr(maybe_core, "yaml")) else model
    layer_seq = getattr(core_model, "model", core_model)

    if isinstance(layer_seq, (nn.ModuleList, nn.Sequential, list, tuple)):
        layers = list(layer_seq)
    else:
        layers = list(layer_seq.children()) if isinstance(layer_seq, nn.Module) else []

    if not layers:
        raise RuntimeError("Unable to locate YOLO module sequence from loaded model.")
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
        raise RuntimeError("Unable to identify detection head (Detect/Segment).")
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
        raise RuntimeError(f"Invalid neck_start_idx={neck_start_idx}; expected range [0, {n_layers - 1}].")
    if neck_start_idx > head_idx:
        raise RuntimeError(f"Invalid update region: neck_start_idx={neck_start_idx} is after head_idx={head_idx}.")
    return list(range(neck_start_idx, head_idx + 1))


def apply_freeze_policy(model: nn.Module, layers: List[nn.Module], unfrozen_indices: List[int]) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for idx in unfrozen_indices:
        for p in layers[idx].parameters():
            p.requires_grad = True


def update_teacher_ema(teacher_model: nn.Module, student_model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        t_state = teacher_model.state_dict()
        s_state = student_model.state_dict()
        for key, teacher_val in t_state.items():
            student_val = s_state[key]
            if torch.is_floating_point(teacher_val):
                teacher_val.mul_(decay).add_(student_val.detach(), alpha=(1.0 - decay))
            else:
                teacher_val.copy_(student_val)


def safe_loss_items_to_dict(loss_items: Any) -> Dict[str, float]:
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


def unpack_loss_pair(loss_out_obj: Any) -> Tuple[Optional[torch.Tensor], Any]:
    if not (isinstance(loss_out_obj, (list, tuple)) and len(loss_out_obj) == 2):
        return None, None
    x, y = loss_out_obj
    x_ok = isinstance(x, torch.Tensor) and (x.requires_grad or x.grad_fn is not None)
    y_ok = isinstance(y, torch.Tensor) and (y.requires_grad or y.grad_fn is not None)
    if x_ok:
        return x, y
    if y_ok:
        return y, x
    return None, None


def build_training_batch(
    entries: Sequence[MemoryEntry],
    imgsz: int,
    device: str,
    rng: random.Random,
) -> Dict[str, torch.Tensor]:
    if not entries:
        raise RuntimeError("build_training_batch called with empty entries")

    img_tensors: List[torch.Tensor] = []
    batch_idx_vals: List[int] = []
    cls_vals: List[List[float]] = []
    box_vals: List[List[float]] = []

    for i, entry in enumerate(entries):
        with Image.open(entry.path) as im:
            img_rgb = im.convert("RGB")

        aug_rgb = strong_augment(img_rgb, rng)
        letterboxed, gain, pad_left, pad_top = letterbox_image(aug_rgb, imgsz)
        img_tensors.append(pil_to_model_tensor(letterboxed))

        x_c, y_c, bw, bh = xyxy_original_to_norm_xywh_letterboxed(
            box_xyxy=entry.pseudo_box,
            orig_w=entry.width,
            orig_h=entry.height,
            size=imgsz,
            gain=gain,
            pad_left=pad_left,
            pad_top=pad_top,
        )
        batch_idx_vals.append(i)
        cls_vals.append([float(entry.pseudo_cls)])
        box_vals.append([x_c, y_c, bw, bh])

    img = torch.cat(img_tensors, dim=0).to(device)
    batch = {
        "img": img,
        "batch_idx": torch.tensor(batch_idx_vals, dtype=torch.long, device=device),
        "cls": torch.tensor(cls_vals, dtype=torch.float32, device=device),
        "bboxes": torch.tensor(box_vals, dtype=torch.float32, device=device),
    }
    return batch


def run_detection_update(
    student_model: nn.Module,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    optim_params: Sequence[torch.nn.Parameter],
    batch: Dict[str, torch.Tensor],
    grad_clip: float,
) -> Tuple[float, Dict[str, float]]:
    with torch.inference_mode(False):
        with torch.enable_grad():
            student_model.train()
            optimizer.zero_grad(set_to_none=True)
            preds = student_model(batch["img"])

            if hasattr(student_model, "loss"):
                loss_out = student_model.loss(batch, preds=preds)
            else:
                loss_out = criterion(preds, batch)

            loss, loss_items = unpack_loss_pair(loss_out)

            if loss is None and hasattr(student_model, "loss"):
                fallback = student_model.loss(batch)
                loss, loss_items = unpack_loss_pair(fallback)

            if loss is None:
                raise RuntimeError(f"Unable to obtain differentiable detection loss; output type={type(loss_out)}")

            loss_scalar = loss.sum() if isinstance(loss, torch.Tensor) and loss.ndim > 0 else loss
            loss_scalar.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(optim_params, grad_clip)
            optimizer.step()

    return float(loss_scalar.detach().cpu()), safe_loss_items_to_dict(loss_items)


def try_load_font(size: int = 16) -> ImageFont.ImageFont:
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

    if top1 is not None:
        x1, y1, x2, y2 = top1.xyxy
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1 + 4, max(0, y1 - 20)), f"c={top1.conf:.2f} cls={top1.cls_id}", fill=(0, 255, 0), font=font)

    draw.rectangle([0, 0, out.size[0], 28], fill=(0, 0, 0))
    draw.text((8, 6), f"f={frame_idx} {title}", fill=(255, 255, 255), font=font)

    if extra_text:
        draw.rectangle([0, out.size[1] - 24, out.size[0], out.size[1]], fill=(0, 0, 0))
        draw.text((8, out.size[1] - 20), extra_text, fill=(255, 255, 255), font=font)

    return out


def make_triptych(
    img_rgb: Image.Image,
    frame_idx: int,
    teacher_top1: Optional[Top1Det],
    student_top1: Optional[Top1Det],
    accepted: bool,
    update_applied: bool,
    buffer_size: int,
    det_loss: float,
) -> Image.Image:
    w, h = img_rgb.size
    left = draw_panel(
        img_rgb,
        "Input",
        None,
        frame_idx,
        extra_text=f"accepted={int(accepted)} update={int(update_applied)} buf={buffer_size}",
    )
    mid = draw_panel(img_rgb, "Teacher", teacher_top1, frame_idx)
    right = draw_panel(
        img_rgb,
        "Student(post)",
        student_top1,
        frame_idx,
        extra_text=(f"loss={det_loss:.4f}" if math.isfinite(det_loss) else "loss=n/a"),
    )

    out = Image.new("RGB", (w * 3, h), color=(0, 0, 0))
    out.paste(left, (0, 0))
    out.paste(mid, (w, 0))
    out.paste(right, (2 * w, 0))
    return out


def make_progress_iter(total: int):
    if tqdm is None:
        return None
    return tqdm(total=total, desc="Online adapt", dynamic_ncols=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online teacher-student adaptation with buffered replay updates.")
    p.add_argument("--weights", type=str, required=True, help="Path to YOLO weights (.pt)")
    p.add_argument("--dataset", type=str, required=True, help="YOLO-root dataset path (expects images/test)")
    p.add_argument("--output", type=str, default="online_adapt_out", help="Output directory")
    p.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    p.add_argument("--imgsz", type=int, default=1024, help="Training image size (letterbox target)")

    p.add_argument("--teacher-conf-thresh", type=float, default=0.70, help="Pseudo-label acceptance threshold")
    p.add_argument("--infer-conf", type=float, default=0.001, help="Inference conf for teacher/student top1")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames, else first N frames")
    p.add_argument("--warmup", type=int, default=10, help="Warmup teacher forwards before adaptation")

    p.add_argument("--lr", type=float, default=1e-4, help="Student optimizer learning rate")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    p.add_argument("--weight-decay", type=float, default=5e-4, help="SGD weight decay")
    p.add_argument("--ema-decay", type=float, default=0.999, help="Teacher EMA decay")
    p.add_argument("--grad-clip", type=float, default=10.0, help="Gradient clipping max norm")

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
        help="Manual neck start index override; >=0 disables YAML auto-detection",
    )

    p.add_argument("--buffer-size", type=int, default=32, help="Replay buffer capacity")
    p.add_argument("--update-batch-size", type=int, default=4, help="Mini-batch size sampled from replay buffer")
    p.add_argument(
        "--min-buffer-before-update",
        type=int,
        default=4,
        help="Minimum number of buffered entries required before updates are allowed",
    )
    p.add_argument(
        "--buffer-sample-mode",
        type=str,
        default="recent",
        choices=["recent", "random"],
        help="Replay sampling strategy",
    )
    p.add_argument("--max-updates-per-frame", type=int, default=1, help="Max optimizer updates per stream frame")

    p.add_argument("--min-area-frac", type=float, default=0.001, help="Min accepted bbox area fraction")
    p.add_argument("--max-area-frac", type=float, default=0.80, help="Max accepted bbox area fraction")
    p.add_argument("--border-margin-frac", type=float, default=0.02, help="Reject boxes too close to border")
    p.add_argument("--temporal-iou-gate", type=float, default=0.30, help="Require teacher IoU(prev,current) >= gate")

    p.add_argument("--seed", type=int, default=0, help="RNG seed")

    p.add_argument("--make-mp4", action="store_true", help="Write adaptation overlay MP4")
    p.add_argument("--mp4-every", type=int, default=1, help="Use every k-th frame in MP4")
    p.add_argument("--mp4-max", type=int, default=0, help="Max frames in MP4, 0 = no limit")
    p.add_argument("--mp4-fps", type=int, default=12, help="MP4 fps")
    p.add_argument("--mp4-scale", type=float, default=0.75, help="Downscale MP4 frames by this factor")

    return p.parse_args()


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
    if int(args.max_frames) > 0:
        images = images[: int(args.max_frames)]

    student_yolo = YOLO(args.weights)
    teacher_yolo = YOLO(args.weights)
    student_model = student_yolo.model
    teacher_model = teacher_yolo.model

    student_model.to(args.device)
    teacher_model.to(args.device)

    core_model, layers = unwrap_core_and_layers(student_model)
    _, teacher_layers = unwrap_core_and_layers(teacher_model)

    head_idx = find_head_idx(layers)
    teacher_head_idx = find_head_idx(teacher_layers)
    if head_idx != teacher_head_idx:
        raise RuntimeError(f"Student/teacher head mismatch: student={head_idx}, teacher={teacher_head_idx}")

    neck_start_idx = resolve_neck_start_idx(core_model, int(args.neck_start_idx))
    unfrozen_indices = compute_unfrozen_indices(
        update_scope=str(args.update_scope),
        neck_start_idx=neck_start_idx,
        head_idx=head_idx,
        n_layers=len(layers),
    )
    apply_freeze_policy(student_model, layers, unfrozen_indices)

    optim_params = [p for p in student_model.parameters() if p.requires_grad]
    if not optim_params:
        raise RuntimeError("No trainable parameters found after freeze policy")

    optimizer = torch.optim.SGD(
        optim_params,
        lr=float(args.lr),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
    )

    if isinstance(student_model.args, dict):
        hyp_dict = dict(student_model.args)
    elif isinstance(student_model.args, SimpleNamespace):
        hyp_dict = vars(student_model.args).copy()
    else:
        hyp_dict = vars(student_model.args).copy() if hasattr(student_model.args, "__dict__") else {}

    hyp_dict.setdefault("box", 7.5)
    hyp_dict.setdefault("cls", 0.5)
    hyp_dict.setdefault("dfl", 1.5)
    student_model.args = SimpleNamespace(**hyp_dict)
    criterion = LossClass(student_model)

    warmup_n = max(0, int(args.warmup))
    if warmup_n > 0:
        for i in range(min(warmup_n, len(images))):
            _ = teacher_yolo.predict(
                source=str(images[i]),
                device=args.device,
                conf=float(args.infer_conf),
                iou=float(args.iou),
                verbose=False,
                save=False,
            )

    rng = random.Random(int(args.seed))
    buffer = ReplayBuffer(max_size=int(args.buffer_size), rng=rng)

    logs: List[FrameLog] = []
    mp4_frames: List[np.ndarray] = []

    accepted_frames = 0
    updated_frames = 0
    total_optimizer_updates = 0
    update_losses: List[float] = []
    buffer_sizes_on_updates: List[int] = []

    prev_teacher_box: Optional[Tuple[float, float, float, float]] = None

    progress = make_progress_iter(len(images))
    for idx, img_path in enumerate(images):
        frame_t0 = time.time()

        with Image.open(img_path) as im:
            img_rgb = im.convert("RGB")
            w, h = img_rgb.size

        teacher_model.eval()
        teacher_top1, teacher_lat_ms = predict_top1_wrapper(
            teacher_yolo,
            source=str(img_path),
            device=str(args.device),
            conf=float(args.infer_conf),
            iou=float(args.iou),
        )

        accepted = should_accept_pseudo(
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

        if accepted and teacher_top1 is not None:
            accepted_frames += 1
            buffer.add(
                MemoryEntry(
                    frame_idx=idx,
                    path=str(img_path),
                    width=w,
                    height=h,
                    pseudo_box=teacher_top1.xyxy,
                    pseudo_cls=int(teacher_top1.cls_id),
                    teacher_conf=float(teacher_top1.conf),
                )
            )

        updates_this_frame = 0
        batch_size_used = 0
        num_pseudo_boxes_used = 0
        last_det_loss = float("nan")
        update_latency_ms = 0.0

        if len(buffer) >= int(args.min_buffer_before_update):
            for _ in range(max(1, int(args.max_updates_per_frame))):
                sampled = buffer.sample(batch_size=int(args.update_batch_size), mode=str(args.buffer_sample_mode))
                if not sampled:
                    break

                apply_freeze_policy(student_model, layers, unfrozen_indices)

                update_t0 = time.time()
                batch = build_training_batch(
                    entries=sampled,
                    imgsz=int(args.imgsz),
                    device=str(args.device),
                    rng=rng,
                )
                det_loss, _loss_items = run_detection_update(
                    student_model=student_model,
                    criterion=criterion,
                    optimizer=optimizer,
                    optim_params=optim_params,
                    batch=batch,
                    grad_clip=float(args.grad_clip),
                )
                update_latency_ms += (time.time() - update_t0) * 1000.0

                update_teacher_ema(
                    teacher_model=teacher_model,
                    student_model=student_model,
                    decay=float(args.ema_decay),
                )

                updates_this_frame += 1
                total_optimizer_updates += 1

                last_det_loss = float(det_loss)
                update_losses.append(float(det_loss))
                batch_size_used = len(sampled)
                num_pseudo_boxes_used += len(sampled)
                buffer_sizes_on_updates.append(len(buffer))

        if updates_this_frame > 0:
            updated_frames += 1

        student_model.eval()
        student_post_top1, student_post_lat_ms = predict_top1_wrapper(
            student_yolo,
            source=str(img_path),
            device=str(args.device),
            conf=float(args.infer_conf),
            iou=float(args.iou),
        )

        frame_latency_ms = (time.time() - frame_t0) * 1000.0

        logs.append(
            FrameLog(
                frame=idx,
                path=str(img_path),
                teacher_conf=float(teacher_top1.conf) if teacher_top1 is not None else 0.0,
                accepted=int(accepted),
                num_pseudo_boxes_used=int(num_pseudo_boxes_used),
                buffer_size=len(buffer),
                update_applied=int(updates_this_frame > 0),
                updates_this_frame=int(updates_this_frame),
                batch_size_used=int(batch_size_used),
                det_loss=float(last_det_loss),
                teacher_latency_ms=float(teacher_lat_ms),
                student_post_conf=float(student_post_top1.conf) if student_post_top1 is not None else 0.0,
                student_post_latency_ms=float(student_post_lat_ms),
                update_latency_ms=float(update_latency_ms),
                frame_latency_ms=float(frame_latency_ms),
            )
        )

        prev_teacher_box = teacher_top1.xyxy if teacher_top1 is not None else None

        if args.make_mp4:
            use_mp4 = (idx % max(1, int(args.mp4_every)) == 0)
            under_cap = (int(args.mp4_max) <= 0) or (len(mp4_frames) < int(args.mp4_max))
            if use_mp4 and under_cap:
                triptych = make_triptych(
                    img_rgb=img_rgb,
                    frame_idx=idx,
                    teacher_top1=teacher_top1,
                    student_top1=student_post_top1,
                    accepted=bool(accepted),
                    update_applied=bool(updates_this_frame > 0),
                    buffer_size=len(buffer),
                    det_loss=float(last_det_loss),
                )

                if float(args.mp4_scale) != 1.0:
                    new_w = max(1, int(round(triptych.size[0] * float(args.mp4_scale))))
                    new_h = max(1, int(round(triptych.size[1] * float(args.mp4_scale))))
                    triptych = triptych.resize((new_w, new_h), Image.Resampling.BILINEAR)

                mp4_frames.append(np.array(triptych))

        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                {
                    "accepted": accepted_frames,
                    "updates": total_optimizer_updates,
                    "buf": len(buffer),
                    "last_bs": batch_size_used,
                },
                refresh=False,
            )

    if progress is not None:
        progress.close()

    csv_path = out_dir / "adapt_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "path",
                "teacher_top1_conf",
                "accepted",
                "num_pseudo_boxes_used",
                "buffer_size",
                "update_applied",
                "updates_this_frame",
                "batch_size_used",
                "det_loss",
                "teacher_latency_ms",
                "student_post_conf",
                "student_post_latency_ms",
                "update_latency_ms",
                "frame_latency_ms",
            ]
        )
        for row in logs:
            writer.writerow(
                [
                    row.frame,
                    row.path,
                    f"{row.teacher_conf:.6f}",
                    row.accepted,
                    row.num_pseudo_boxes_used,
                    row.buffer_size,
                    row.update_applied,
                    row.updates_this_frame,
                    row.batch_size_used,
                    f"{row.det_loss:.6f}" if math.isfinite(row.det_loss) else "",
                    f"{row.teacher_latency_ms:.3f}",
                    f"{row.student_post_conf:.6f}",
                    f"{row.student_post_latency_ms:.3f}",
                    f"{row.update_latency_ms:.3f}",
                    f"{row.frame_latency_ms:.3f}",
                ]
            )

    teacher_conf_vals = np.array([r.teacher_conf for r in logs], dtype=np.float32)
    student_post_conf_vals = np.array([r.student_post_conf for r in logs], dtype=np.float32)

    mean_teacher_conf = float(np.mean(teacher_conf_vals)) if logs else float("nan")
    mean_student_post_conf = float(np.mean(student_post_conf_vals)) if logs else float("nan")
    mean_update_loss = float(np.mean(update_losses)) if update_losses else float("nan")
    mean_buffer_size_updates = float(np.mean(buffer_sizes_on_updates)) if buffer_sizes_on_updates else float("nan")

    summary_path = out_dir / "summary.txt"
    summary_lines = [
        "Online Adaptation Summary",
        f"weights={args.weights}",
        f"dataset={dataset_root}",
        f"total_frames={len(logs)}",
        f"accepted_frames={accepted_frames}",
        f"updates_applied={updated_frames}",
        f"optimizer_update_steps={total_optimizer_updates}",
        f"mean_teacher_conf={mean_teacher_conf:.6f}" if math.isfinite(mean_teacher_conf) else "mean_teacher_conf=n/a",
        (
            f"mean_student_post_conf={mean_student_post_conf:.6f}"
            if math.isfinite(mean_student_post_conf)
            else "mean_student_post_conf=n/a"
        ),
        f"mean_detection_loss_updates={mean_update_loss:.6f}" if math.isfinite(mean_update_loss) else "mean_detection_loss_updates=n/a",
        (
            f"mean_buffer_size_during_updates={mean_buffer_size_updates:.3f}"
            if math.isfinite(mean_buffer_size_updates)
            else "mean_buffer_size_during_updates=n/a"
        ),
        "",
        "buffer_config:",
        f"  buffer_size={int(args.buffer_size)}",
        f"  update_batch_size={int(args.update_batch_size)}",
        f"  min_buffer_before_update={int(args.min_buffer_before_update)}",
        f"  buffer_sample_mode={args.buffer_sample_mode}",
        f"  max_updates_per_frame={int(args.max_updates_per_frame)}",
        "",
        "outputs:",
        f"  csv={csv_path.name}",
        f"  summary={summary_path.name}",
    ]
    if args.make_mp4:
        summary_lines.append("  mp4=adapt_overlay.mp4")

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    if args.make_mp4:
        if imageio is None:
            print("[warn] imageio is not available, skipping MP4 export")
        elif mp4_frames:
            mp4_path = out_dir / "adapt_overlay.mp4"
            with imageio.get_writer(mp4_path, fps=int(args.mp4_fps), codec="libx264", quality=7) as writer:
                for frame in mp4_frames:
                    writer.append_data(frame)

    print(f"Done. Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()


# PYTHONPATH=. python odad/online_adapt.py \
#   --weights /home/hm25936/mae/runs/yolov8_baseline/baseline/weights/best.pt \
#   --dataset /home/hm25936/datasets_for_yolo/lab_images_6000 \
#   --output /home/hm25936/mae/odad/online_adapt_buffered_neck_head_full \
#   --device cuda:0 \
#   --imgsz 1024 \
#   --teacher-conf-thresh 0.80 \
#   --infer-conf 0.001 \
#   --iou 0.45 \
#   --lr 3e-4 \
#   --ema-decay 0.999 \
#   --update-scope neck_head \
#   --buffer-size 32 \
#   --update-batch-size 4 \
#   --min-buffer-before-update 4 \
#   --buffer-sample-mode recent \
#   --max-updates-per-frame 1 \
#   --temporal-iou-gate 0.50 \
#   --make-mp4 \
#   --mp4-every 2 \
#   --mp4-fps 12 \
#   --mp4-scale 0.75