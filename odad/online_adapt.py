#!/usr/bin/env python3
"""Online teacher-student ODAD with persist2 replay gating and memory-aware top-k teacher selection."""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:
    from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION
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
    rank: int = 1


@dataclass
class ReplayEntry:
    frame_idx: int
    path: str
    width: int
    height: int
    pseudo_box: Tuple[float, float, float, float]
    pseudo_cls: int


@dataclass
class FrameLog:
    frame: int
    path: str
    teacher_conf: float
    accepted: int
    accepted_final: int
    passed_base_gate: int
    passed_motion_gate: int
    passed_persistence_gate: int
    teacher_num_candidates: int
    teacher_selected_rank: int
    teacher_selected_score: float
    teacher_selected_score_conf: float
    teacher_selected_score_temporal: float
    teacher_selected_score_memory: float
    teacher_candidate_memory_size_active: int
    persistence_count: int
    temporal_iou: float
    persistence_iou: float
    center_shift_frac: float
    area_ratio: float
    num_pseudo_boxes_used: int
    buffer_size: int
    update_event_triggered: int
    update_applied: int
    updates_this_frame: int
    batch_size_used: int
    det_loss: float
    total_loss: float
    teacher_latency_ms: float
    student_post_conf: float
    student_post_latency_ms: float
    update_latency_ms: float
    frame_latency_ms: float


@dataclass
class PersistenceState:
    cls_id: int
    xyxy: Tuple[float, float, float, float]
    count: int


@dataclass
class CandidateMemoryState:
    frame_idx: int
    cls_id: int
    box_xyxy: Tuple[float, float, float, float]
    conf: float
    state_vector: np.ndarray


@dataclass
class CandidateSelectionResult:
    selected: Optional[Top1Det]
    num_candidates: int
    selected_rank: int
    selected_score: float
    score_conf: float
    score_temporal: float
    score_memory: float
    memory_size_active: int
    selected_state_vector: Optional[np.ndarray]


class ReplayBuffer:
    def __init__(self, max_size: int, rng: random.Random) -> None:
        self._entries: Deque[ReplayEntry] = deque(maxlen=max(1, int(max_size)))
        self._rng = rng

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entry: ReplayEntry) -> None:
        self._entries.append(entry)

    def sample(self, batch_size: int, mode: str) -> List[ReplayEntry]:
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


def evaluate_detection_sanity(
    top1: Optional[Top1Det],
    img_w: int,
    img_h: int,
    conf_thresh: float,
    min_area_frac: float,
    max_area_frac: float,
    border_margin_frac: float,
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
    return True


def evaluate_base_gate(
    top1: Optional[Top1Det],
    prev_teacher_box: Optional[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
    conf_thresh: float,
    min_area_frac: float,
    max_area_frac: float,
    border_margin_frac: float,
    temporal_iou_gate: float,
) -> Tuple[bool, float]:
    temporal_iou = float("nan")
    if not evaluate_detection_sanity(
        top1=top1,
        img_w=img_w,
        img_h=img_h,
        conf_thresh=conf_thresh,
        min_area_frac=min_area_frac,
        max_area_frac=max_area_frac,
        border_margin_frac=border_margin_frac,
    ):
        return False, temporal_iou

    if prev_teacher_box is not None and temporal_iou_gate > 0:
        temporal_iou = xyxy_iou(top1.xyxy, prev_teacher_box)
        if temporal_iou < temporal_iou_gate:
            return False, temporal_iou

    return True, temporal_iou


def box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def box_area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def center_shift_fraction(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> float:
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    dx = (ax - bx) / max(1.0, float(img_w))
    dy = (ay - by) / max(1.0, float(img_h))
    return float(math.sqrt(dx * dx + dy * dy))


def symmetric_area_ratio(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    larger = max(area_a, area_b)
    smaller = max(min(area_a, area_b), 1e-12)
    return float(larger / smaller) if larger > 0.0 else 1.0


def evaluate_motion_gate(
    top1: Optional[Top1Det],
    prev_teacher_box: Optional[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
    max_center_shift_frac: float,
    max_area_ratio: float,
    enabled: bool,
) -> Tuple[bool, float, float]:
    center_shift_frac = float("nan")
    area_ratio = float("nan")
    if top1 is None:
        return False, center_shift_frac, area_ratio
    if prev_teacher_box is None or not enabled:
        return True, center_shift_frac, area_ratio

    center_shift_frac = center_shift_fraction(top1.xyxy, prev_teacher_box, img_w=img_w, img_h=img_h)
    area_ratio = symmetric_area_ratio(top1.xyxy, prev_teacher_box)

    if max_center_shift_frac > 0 and center_shift_frac > max_center_shift_frac:
        return False, center_shift_frac, area_ratio
    if max_area_ratio > 0 and area_ratio > max_area_ratio:
        return False, center_shift_frac, area_ratio
    return True, center_shift_frac, area_ratio


def update_persistence_state(
    state: Optional[PersistenceState],
    top1: Optional[Top1Det],
    candidate_valid: bool,
    persistence_frames: int,
    persistence_iou: float,
) -> Tuple[Optional[PersistenceState], int, float, bool]:
    required_frames = max(1, int(persistence_frames))
    if top1 is None or not candidate_valid:
        return None, 0, float("nan"), False

    if required_frames <= 1:
        next_state = PersistenceState(
            cls_id=int(top1.cls_id),
            xyxy=top1.xyxy,
            count=1,
        )
        return next_state, 1, float("nan"), True

    persistence_overlap = float("nan")
    if state is None or int(state.cls_id) != int(top1.cls_id):
        count = 1
    else:
        persistence_overlap = xyxy_iou(top1.xyxy, state.xyxy)
        count = state.count + 1 if persistence_overlap >= persistence_iou else 1

    next_state = PersistenceState(
        cls_id=int(top1.cls_id),
        xyxy=top1.xyxy,
        count=int(count),
    )
    return next_state, int(count), persistence_overlap, bool(count >= required_frames)


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


def teacher_candidates_from_results(
    results: Any,
    topk: int,
    conf_floor: float,
    allow_top1_fallback: bool,
) -> Tuple[Optional[Top1Det], List[Top1Det]]:
    if not results:
        return None, []
    result = results[0]
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
        return None, []

    confs = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls_vals = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else None
    order = np.argsort(-confs)
    if order.size == 0:
        return None, []

    def build_det(idx: int, rank: int) -> Top1Det:
        cls_id = int(cls_vals[idx]) if cls_vals is not None else 0
        x1, y1, x2, y2 = map(float, xyxy[idx].tolist())
        return Top1Det(conf=float(confs[idx]), cls_id=cls_id, xyxy=(x1, y1, x2, y2), rank=int(rank))

    raw_top1 = build_det(int(order[0]), rank=1)
    candidates: List[Top1Det] = []
    limit = max(1, int(topk))
    for rank, idx in enumerate(order.tolist(), start=1):
        idx_int = int(idx)
        if float(confs[idx_int]) < float(conf_floor):
            break
        candidates.append(build_det(idx_int, rank=rank))
        if len(candidates) >= limit:
            break

    if allow_top1_fallback and raw_top1 is not None and not candidates:
        candidates = [raw_top1]
    return raw_top1, candidates


def top1_from_results(results: Any) -> Optional[Top1Det]:
    raw_top1, _ = teacher_candidates_from_results(
        results=results,
        topk=1,
        conf_floor=-1.0,
        allow_top1_fallback=False,
    )
    return raw_top1


def extract_feature_tensor(output: Any) -> Optional[torch.Tensor]:
    tensors: List[torch.Tensor] = []

    def visit(node: Any) -> None:
        if isinstance(node, torch.Tensor) and node.ndim == 4:
            tensors.append(node)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                visit(item)
            return
        if isinstance(node, dict):
            for item in node.values():
                visit(item)

    visit(output)
    if not tensors:
        return None
    return max(tensors, key=lambda tensor: int(tensor.shape[-2]) * int(tensor.shape[-1]))


class LayerFeatureTap:
    def __init__(self, module: nn.Module) -> None:
        self._last_feature: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        self._last_feature = extract_feature_tensor(output)

    def clear(self) -> None:
        self._last_feature = None

    def get(self) -> Optional[torch.Tensor]:
        return self._last_feature

    def close(self) -> None:
        self._handle.remove()


def predict_results_with_latency(
    yolo_wrapper: YOLO,
    source: Any,
    device: str,
    conf: float,
    iou: float,
) -> Tuple[Any, float]:
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

    return results, latency_ms


def predict_top1_wrapper(
    yolo_wrapper: YOLO,
    source: Any,
    device: str,
    conf: float,
    iou: float,
) -> Tuple[Optional[Top1Det], float]:
    results, latency_ms = predict_results_with_latency(
        yolo_wrapper=yolo_wrapper,
        source=source,
        device=device,
        conf=conf,
        iou=iou,
    )
    return top1_from_results(results), latency_ms


def predict_teacher_candidates_wrapper(
    yolo_wrapper: YOLO,
    source: Any,
    device: str,
    conf: float,
    iou: float,
    topk: int,
    conf_floor: float,
    allow_top1_fallback: bool,
    feature_tap: Optional[LayerFeatureTap] = None,
) -> Tuple[Optional[Top1Det], List[Top1Det], Optional[torch.Tensor], float]:
    if feature_tap is not None:
        feature_tap.clear()
    results, latency_ms = predict_results_with_latency(
        yolo_wrapper=yolo_wrapper,
        source=source,
        device=device,
        conf=conf,
        iou=iou,
    )
    raw_top1, candidates = teacher_candidates_from_results(
        results=results,
        topk=topk,
        conf_floor=conf_floor,
        allow_top1_fallback=allow_top1_fallback,
    )
    return raw_top1, candidates, feature_tap.get() if feature_tap is not None else None, latency_ms


def compute_temporal_consistency_score(
    box_xyxy: Tuple[float, float, float, float],
    prev_box_xyxy: Optional[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
    max_center_shift_frac: float,
    max_area_ratio: float,
) -> Tuple[float, float, float, float]:
    if prev_box_xyxy is None:
        return 0.0, float("nan"), float("nan"), float("nan")

    iou = xyxy_iou(box_xyxy, prev_box_xyxy)
    center_shift = center_shift_fraction(box_xyxy, prev_box_xyxy, img_w=img_w, img_h=img_h)
    area_ratio = symmetric_area_ratio(box_xyxy, prev_box_xyxy)

    shift_factor = 1.0
    if max_center_shift_frac > 0:
        shift_factor = max(0.0, 1.0 - max(0.0, center_shift / max_center_shift_frac))

    area_factor = 1.0
    if max_area_ratio > 0:
        if max_area_ratio <= 1.0:
            area_factor = 1.0 if area_ratio <= max_area_ratio else 0.0
        else:
            area_budget = max(1e-6, max_area_ratio - 1.0)
            area_factor = max(0.0, 1.0 - max(0.0, area_ratio - 1.0) / area_budget)

    temporal_score = float(iou * shift_factor * area_factor)
    return temporal_score, float(iou), float(center_shift), float(area_ratio)


def box_xyxy_to_norm_state(
    box_xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    center_x = ((x1 + x2) * 0.5) / max(1.0, float(img_w))
    center_y = ((y1 + y2) * 0.5) / max(1.0, float(img_h))
    box_w = max(0.0, x2 - x1) / max(1.0, float(img_w))
    box_h = max(0.0, y2 - y1) / max(1.0, float(img_h))
    return np.array(
        [
            min(1.0, max(0.0, center_x)),
            min(1.0, max(0.0, center_y)),
            min(1.0, max(0.0, box_w)),
            min(1.0, max(0.0, box_h)),
        ],
        dtype=np.float32,
    )


def normalize_state_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec.astype(np.float32, copy=True)
    return (vec / norm).astype(np.float32, copy=False)


def project_box_to_feature_map(
    box_xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    feat_w: int,
    feat_h: int,
) -> Tuple[int, int, int, int]:
    gain = min(float(feat_w) / max(1.0, float(img_w)), float(feat_h) / max(1.0, float(img_h)))
    pad_left = 0.5 * max(0.0, float(feat_w) - float(img_w) * gain)
    pad_top = 0.5 * max(0.0, float(feat_h) - float(img_h) * gain)

    x1, y1, x2, y2 = box_xyxy
    fx1 = x1 * gain + pad_left
    fy1 = y1 * gain + pad_top
    fx2 = x2 * gain + pad_left
    fy2 = y2 * gain + pad_top

    x1i = max(0, min(feat_w - 1, int(math.floor(fx1))))
    y1i = max(0, min(feat_h - 1, int(math.floor(fy1))))
    x2i = max(x1i + 1, min(feat_w, int(math.ceil(fx2))))
    y2i = max(y1i + 1, min(feat_h, int(math.ceil(fy2))))
    return x1i, y1i, x2i, y2i


def extract_candidate_feature_embedding(
    feature_map: Optional[torch.Tensor],
    box_xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    min_box_frac: float,
    output_dim: int = 8,
) -> np.ndarray:
    if feature_map is None or feature_map.ndim != 4 or feature_map.shape[0] <= 0:
        return np.zeros(output_dim, dtype=np.float32)
    if area_fraction(box_xyxy, width=img_w, height=img_h) < float(min_box_frac):
        return np.zeros(output_dim, dtype=np.float32)

    fmap = feature_map[0]
    feat_h = int(fmap.shape[-2])
    feat_w = int(fmap.shape[-1])
    if feat_h <= 0 or feat_w <= 0:
        return np.zeros(output_dim, dtype=np.float32)

    x1i, y1i, x2i, y2i = project_box_to_feature_map(
        box_xyxy=box_xyxy,
        img_w=img_w,
        img_h=img_h,
        feat_w=feat_w,
        feat_h=feat_h,
    )
    roi = fmap[:, y1i:y2i, x1i:x2i]
    if roi.numel() == 0:
        return np.zeros(output_dim, dtype=np.float32)

    pooled = roi.mean(dim=(1, 2))
    if pooled.numel() > output_dim:
        pooled = F.adaptive_avg_pool1d(pooled.view(1, 1, -1), output_dim).view(-1)
    elif pooled.numel() < output_dim:
        pooled = F.pad(pooled, (0, output_dim - int(pooled.numel())))
    return pooled.detach().float().cpu().numpy().astype(np.float32, copy=False)


def build_candidate_state_vector(
    candidate: Top1Det,
    img_w: int,
    img_h: int,
    feature_map: Optional[torch.Tensor],
    use_feature: bool,
    min_box_frac: float,
) -> np.ndarray:
    geom_and_conf = np.concatenate(
        [
            box_xyxy_to_norm_state(candidate.xyxy, img_w=img_w, img_h=img_h),
            np.array([float(candidate.conf)], dtype=np.float32),
        ],
        axis=0,
    )
    geom_and_conf = normalize_state_vector(geom_and_conf)
    if not use_feature:
        return geom_and_conf

    feature_vec = extract_candidate_feature_embedding(
        feature_map=feature_map,
        box_xyxy=candidate.xyxy,
        img_w=img_w,
        img_h=img_h,
        min_box_frac=min_box_frac,
    )
    feature_vec = normalize_state_vector(feature_vec)
    combined = np.concatenate(
        [
            geom_and_conf * math.sqrt(0.5),
            feature_vec * math.sqrt(0.5),
        ],
        axis=0,
    )
    return normalize_state_vector(combined)


def compute_candidate_memory_score(
    candidate: Top1Det,
    candidate_state_vector: np.ndarray,
    memory_states: Sequence[CandidateMemoryState],
    current_frame_idx: int,
    memory_score_mode: str,
) -> float:
    if not memory_states:
        return 0.0

    similarities: List[float] = []
    weighted_pairs: List[Tuple[float, float]] = []
    for entry in memory_states:
        if int(entry.cls_id) != int(candidate.cls_id):
            sim = 0.0
        else:
            sim = float(np.clip(np.dot(candidate_state_vector, entry.state_vector), -1.0, 1.0))
            sim = 0.5 * (sim + 1.0)
        similarities.append(sim)
        age = max(0, int(current_frame_idx) - int(entry.frame_idx))
        weighted_pairs.append((sim, 1.0 / float(1 + age)))

    mode = str(memory_score_mode)
    if mode == "max":
        return float(max(similarities))
    if mode == "top2_mean":
        top_vals = sorted(similarities, reverse=True)[:2]
        return float(np.mean(top_vals)) if top_vals else 0.0
    if mode == "recency_mean":
        weight_sum = float(sum(weight for _sim, weight in weighted_pairs))
        if weight_sum <= 0.0:
            return 0.0
        return float(sum(sim * weight for sim, weight in weighted_pairs) / weight_sum)
    raise RuntimeError(f"Unsupported teacher candidate memory score mode: {memory_score_mode}")


def score_teacher_candidate(
    candidate: Top1Det,
    prev_reference_box: Optional[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
    max_center_shift_frac: float,
    max_area_ratio: float,
    score_mode: str,
    conf_weight: float,
    temporal_weight: float,
    memory_weight: float,
    memory_term: float,
) -> Tuple[float, float, float, float]:
    mode = str(score_mode)
    conf_term = float(candidate.conf)
    temporal_term = 0.0
    if mode in {"conf_temporal", "conf_temporal_memory"}:
        temporal_term, _iou, _center_shift, _area_ratio = compute_temporal_consistency_score(
            box_xyxy=candidate.xyxy,
            prev_box_xyxy=prev_reference_box,
            img_w=img_w,
            img_h=img_h,
            max_center_shift_frac=max_center_shift_frac,
            max_area_ratio=max_area_ratio,
        )

    total = float(conf_weight) * conf_term
    if mode in {"conf_temporal", "conf_temporal_memory"}:
        total += float(temporal_weight) * float(temporal_term)
    if mode == "conf_temporal_memory":
        total += float(memory_weight) * float(memory_term)
    elif mode not in {"conf_only", "conf_temporal"}:  # pragma: no cover
        raise RuntimeError(f"Unsupported teacher candidate score mode: {score_mode}")
    return float(total), float(conf_term), float(temporal_term), float(memory_term)


def select_teacher_candidate(
    candidates: Sequence[Top1Det],
    prev_reference_box: Optional[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
    max_center_shift_frac: float,
    max_area_ratio: float,
    score_mode: str,
    conf_weight: float,
    temporal_weight: float,
    memory_weight: float,
    min_score: float,
    current_frame_idx: int,
    memory_states: Sequence[CandidateMemoryState],
    memory_score_mode: str,
    memory_feature_map: Optional[torch.Tensor],
    memory_use_feature: bool,
    memory_min_box_frac: float,
) -> CandidateSelectionResult:
    if not candidates:
        return CandidateSelectionResult(
            selected=None,
            num_candidates=0,
            selected_rank=0,
            selected_score=float("nan"),
            score_conf=float("nan"),
            score_temporal=float("nan"),
            score_memory=float("nan"),
            memory_size_active=len(memory_states),
            selected_state_vector=None,
        )

    best_candidate: Optional[Top1Det] = None
    best_total = float("-inf")
    best_conf_term = float("nan")
    best_temporal_term = float("nan")
    best_memory_term = float("nan")
    best_state_vector: Optional[np.ndarray] = None
    memory_enabled = str(score_mode) == "conf_temporal_memory"

    for candidate in candidates:
        candidate_state_vector = build_candidate_state_vector(
            candidate=candidate,
            img_w=img_w,
            img_h=img_h,
            feature_map=memory_feature_map,
            use_feature=memory_use_feature,
            min_box_frac=memory_min_box_frac,
        )
        memory_term = (
            compute_candidate_memory_score(
                candidate=candidate,
                candidate_state_vector=candidate_state_vector,
                memory_states=memory_states,
                current_frame_idx=current_frame_idx,
                memory_score_mode=memory_score_mode,
            )
            if memory_enabled
            else float("nan")
        )
        total, conf_term, temporal_term, memory_term = score_teacher_candidate(
            candidate=candidate,
            prev_reference_box=prev_reference_box,
            img_w=img_w,
            img_h=img_h,
            max_center_shift_frac=max_center_shift_frac,
            max_area_ratio=max_area_ratio,
            score_mode=score_mode,
            conf_weight=conf_weight,
            temporal_weight=temporal_weight,
            memory_weight=memory_weight,
            memory_term=memory_term,
        )
        if (
            best_candidate is None
            or total > best_total
            or (
                math.isclose(total, best_total, rel_tol=0.0, abs_tol=1e-12)
                and (candidate.conf > best_candidate.conf or candidate.rank < best_candidate.rank)
            )
        ):
            best_candidate = candidate
            best_total = float(total)
            best_conf_term = float(conf_term)
            best_temporal_term = float(temporal_term)
            best_memory_term = float(memory_term)
            best_state_vector = candidate_state_vector

    if best_candidate is None or best_total < float(min_score):
        return CandidateSelectionResult(
            selected=None,
            num_candidates=len(candidates),
            selected_rank=0,
            selected_score=float("nan"),
            score_conf=float("nan"),
            score_temporal=float("nan"),
            score_memory=float("nan"),
            memory_size_active=len(memory_states),
            selected_state_vector=None,
        )

    return CandidateSelectionResult(
        selected=best_candidate,
        num_candidates=len(candidates),
        selected_rank=int(best_candidate.rank),
        selected_score=float(best_total),
        score_conf=float(best_conf_term),
        score_temporal=float(best_temporal_term),
        score_memory=float(best_memory_term),
        memory_size_active=len(memory_states),
        selected_state_vector=best_state_vector,
    )


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


def param_id_set_for_indices(layers: List[nn.Module], indices: Sequence[int]) -> set[int]:
    out: set[int] = set()
    for idx in indices:
        for param in layers[idx].parameters():
            out.add(id(param))
    return out


def apply_freeze_policy(model: nn.Module, layers: List[nn.Module], unfrozen_indices: Sequence[int]) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for idx in unfrozen_indices:
        for param in layers[idx].parameters():
            param.requires_grad = True


def update_teacher_ema(teacher_model: nn.Module, student_model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        teacher_state = teacher_model.state_dict()
        student_state = student_model.state_dict()
        for key, teacher_val in teacher_state.items():
            student_val = student_state[key]
            if torch.is_floating_point(teacher_val):
                teacher_val.mul_(decay).add_(student_val.detach(), alpha=(1.0 - decay))
            else:
                teacher_val.copy_(student_val)


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
    target_entries: Sequence[ReplayEntry],
    imgsz: int,
    device: str,
    rng: random.Random,
) -> Dict[str, torch.Tensor]:
    if not target_entries:
        raise RuntimeError("build_training_batch called with empty entries")

    img_tensors: List[torch.Tensor] = []
    batch_idx_vals: List[int] = []
    cls_vals: List[List[float]] = []
    box_vals: List[List[float]] = []

    for i, entry in enumerate(target_entries):
        with Image.open(entry.path) as im:
            img_rgb = im.convert("RGB")
            img_w, img_h = img_rgb.size

        aug_rgb = strong_augment(img_rgb, rng)
        letterboxed, gain, pad_left, pad_top = letterbox_image(aug_rgb, imgsz)
        img_tensors.append(pil_to_model_tensor(letterboxed))

        x_c, y_c, bw, bh = xyxy_original_to_norm_xywh_letterboxed(
            box_xyxy=entry.pseudo_box,
            orig_w=img_w,
            orig_h=img_h,
            size=imgsz,
            gain=gain,
            pad_left=pad_left,
            pad_top=pad_top,
        )
        batch_idx_vals.append(i)
        cls_vals.append([float(entry.pseudo_cls)])
        box_vals.append([x_c, y_c, bw, bh])

    return {
        "img": torch.cat(img_tensors, dim=0).to(device),
        "batch_idx": torch.tensor(batch_idx_vals, dtype=torch.long, device=device),
        "cls": torch.tensor(cls_vals, dtype=torch.float32, device=device),
        "bboxes": torch.tensor(box_vals, dtype=torch.float32, device=device),
    }


def run_detection_update(
    student_model: nn.Module,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    optim_params: Sequence[torch.nn.Parameter],
    batch: Dict[str, torch.Tensor],
    grad_clip: float,
) -> Tuple[float, float]:
    with torch.inference_mode(False):
        with torch.enable_grad():
            student_model.train()
            optimizer.zero_grad(set_to_none=True)
            preds = student_model(batch["img"])

            if hasattr(student_model, "loss"):
                loss_out = student_model.loss(batch, preds=preds)
            else:
                loss_out = criterion(preds, batch)

            loss, _loss_items = unpack_loss_pair(loss_out)
            if loss is None:
                fallback = criterion(preds, batch)
                loss, _loss_items = unpack_loss_pair(fallback)

            if loss is None and hasattr(student_model, "loss"):
                fallback = student_model.loss(batch)
                loss, _loss_items = unpack_loss_pair(fallback)

            if loss is None:
                raise RuntimeError(
                    f"Unable to obtain differentiable detection loss; output type={type(loss_out)}"
                )

            det_loss = loss.sum() if isinstance(loss, torch.Tensor) and loss.ndim > 0 else loss
            det_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(optim_params, grad_clip)
            optimizer.step()

    det_loss_value = float(det_loss.detach().cpu())
    return det_loss_value, det_loss_value


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
    loss_value: float,
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
        extra_text=(f"loss={loss_value:.4f}" if math.isfinite(loss_value) else "loss=n/a"),
    )

    out = Image.new("RGB", (w * 3, h), color=(0, 0, 0))
    out.paste(left, (0, 0))
    out.paste(mid, (w, 0))
    out.paste(right, (2 * w, 0))
    return out


def save_selected_rank_example_image(
    out_dir: Path,
    frame_idx: int,
    img_rgb: Image.Image,
    raw_top1: Top1Det,
    selected_candidate: Optional[Top1Det],
    selected_score: float,
    score_conf: float,
    score_temporal: float,
    score_memory: float,
    score_mode: str,
) -> Path:
    diag_dir = out_dir / "selected_rank_gt1"
    diag_dir.mkdir(parents=True, exist_ok=True)

    out = img_rgb.copy()
    draw = ImageDraw.Draw(out)
    font = try_load_font(18)

    x1, y1, x2, y2 = raw_top1.xyxy
    draw.rectangle([x1, y1, x2, y2], outline=(255, 96, 96), width=4)
    draw.text(
        (x1 + 4, max(0, y1 - 24)),
        f"top1 conf={raw_top1.conf:.3f} cls={raw_top1.cls_id}",
        fill=(255, 96, 96),
        font=font,
    )
    if selected_candidate is not None:
        sx1, sy1, sx2, sy2 = selected_candidate.xyxy
        draw.rectangle([sx1, sy1, sx2, sy2], outline=(80, 255, 120), width=4)
        draw.text(
            (sx1 + 4, min(out.size[1] - 20, sy2 + 4)),
            f"rank={selected_candidate.rank} conf={selected_candidate.conf:.3f}",
            fill=(80, 255, 120),
            font=font,
        )

    draw.rectangle([0, 0, out.size[0], 30], fill=(0, 0, 0))
    selected_rank_text = str(selected_candidate.rank) if selected_candidate is not None else "n/a"
    draw.text((8, 6), f"selected rank>1 frame={frame_idx} rank={selected_rank_text}", fill=(255, 255, 255), font=font)

    footer_top = max(0, out.size[1] - 64)
    draw.rectangle([0, footer_top, out.size[0], out.size[1]], fill=(0, 0, 0))
    footer = f"score={selected_score:.3f} conf={score_conf:.3f} temporal={score_temporal:.3f}"
    draw.text((8, footer_top + 8), footer, fill=(255, 255, 255), font=font)
    memory_footer = (
        f"memory={score_memory:.3f}" if math.isfinite(score_memory) else "memory=n/a"
    )
    draw.text((8, footer_top + 26), memory_footer, fill=(255, 255, 255), font=font)
    draw.text((8, footer_top + 44), f"mode={score_mode}", fill=(255, 255, 255), font=font)

    score_tag = f"{selected_score:.3f}".replace(".", "p")
    rank_tag = selected_rank_text
    out_path = diag_dir / f"selected_rank_{int(frame_idx):06d}_r{rank_tag}_{score_tag}.jpg"
    out.save(out_path, quality=95)
    return out_path


def make_progress_iter(total: int):
    if tqdm is None:
        return None
    return tqdm(total=total, desc="Online adapt", dynamic_ncols=True)


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    if window <= 1:
        return x.astype(np.float32).copy()
    out = np.empty_like(x, dtype=np.float32)
    running_sum = 0.0
    queue: Deque[float] = deque()
    for i, value in enumerate(x.astype(np.float32)):
        queue.append(float(value))
        running_sum += float(value)
        if len(queue) > window:
            running_sum -= queue.popleft()
        out[i] = running_sum / float(len(queue))
    return out


def save_plot_lines(
    x: np.ndarray,
    ys: Sequence[np.ndarray],
    labels: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(10, 4))
    finite_series = [y for y in ys if y.size > 0 and np.any(np.isfinite(y))]
    if x.size == 0 or not finite_series:
        plt.title(title)
        plt.text(0.5, 0.5, "no data", ha="center", va="center", transform=plt.gca().transAxes)
        plt.axis("off")
    else:
        for y, label in zip(ys, labels):
            plt.plot(x, y, label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if len(labels) > 1:
            plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_plot_hist(values: np.ndarray, bins: int, title: str, xlabel: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 5))
    finite_vals = values[np.isfinite(values)] if values.size else np.array([], dtype=np.float32)
    if finite_vals.size == 0:
        plt.title(title)
        plt.text(0.5, 0.5, "no data", ha="center", va="center", transform=plt.gca().transAxes)
        plt.axis("off")
    else:
        plt.hist(finite_vals, bins=max(5, int(bins)))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("count")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def mean_or_nan(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def rolling_mean_ignore_nan(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    if window <= 1:
        return x.astype(np.float32).copy()
    out = np.full_like(x, np.nan, dtype=np.float32)
    running_sum = 0.0
    valid_count = 0
    queue: Deque[float] = deque()
    for i, value in enumerate(x.astype(np.float32)):
        value_f = float(value)
        queue.append(value_f)
        if math.isfinite(value_f):
            running_sum += value_f
            valid_count += 1
        if len(queue) > window:
            dropped = float(queue.popleft())
            if math.isfinite(dropped):
                running_sum -= dropped
                valid_count -= 1
        if valid_count > 0:
            out[i] = running_sum / float(valid_count)
    return out


def format_metric(name: str, value: float, fmt: str = ".6f", prefix: str = "") -> str:
    return f"{prefix}{name}={value:{fmt}}" if math.isfinite(value) else f"{prefix}{name}=n/a"


def resolve_update_schedule(
    max_updates_per_frame: int,
    update_every_frames: int,
    updates_per_event: int,
) -> Tuple[int, int]:
    cadence = max(1, int(update_every_frames))
    steps_per_event = max(1, int(updates_per_event))
    legacy_steps = max(1, int(max_updates_per_frame))
    if steps_per_event == 1 and legacy_steps != 1:
        steps_per_event = legacy_steps
    return cadence, steps_per_event


def should_trigger_update_event(frame_idx: int, update_every_frames: int) -> bool:
    cadence = max(1, int(update_every_frames))
    return cadence <= 1 or ((int(frame_idx) + 1) % cadence == 0)


def strip_runtime_hooks(model: nn.Module) -> None:
    for module in model.modules():
        for attr in (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
            "_state_dict_hooks",
            "_state_dict_pre_hooks",
            "_load_state_dict_pre_hooks",
            "_load_state_dict_post_hooks",
        ):
            hook_map = getattr(module, attr, None)
            if hook_map is not None:
                hook_map.clear()


def save_student_weights_checkpoint(
    yolo_wrapper: YOLO,
    student_model: nn.Module,
    out_path: Path,
    checkpoint_type: str,
    frame_idx: Optional[int] = None,
) -> Path:
    base_ckpt = getattr(yolo_wrapper, "ckpt", None)
    ckpt = dict(base_ckpt) if isinstance(base_ckpt, dict) else {}

    export_model = deepcopy(student_model).to("cpu").eval()
    strip_runtime_hooks(export_model)
    if hasattr(export_model, "args"):
        model_args = getattr(export_model, "args")
        if isinstance(model_args, dict):
            export_model.args = dict(model_args)
        elif isinstance(model_args, SimpleNamespace):
            export_model.args = vars(model_args).copy()
        elif hasattr(model_args, "__dict__"):
            export_model.args = vars(model_args).copy()
    if hasattr(export_model, "criterion"):
        export_model.criterion = None
    export_model.half()
    for param in export_model.parameters():
        param.requires_grad = False

    ckpt.update(
        {
            "epoch": -1,
            "best_fitness": None,
            "model": export_model,
            "ema": None,
            "updates": None,
            "optimizer": None,
            "date": datetime.now().isoformat(),
            "version": ULTRALYTICS_VERSION,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
    )
    ckpt["odad_adaptation"] = {
        "checkpoint_type": str(checkpoint_type),
        "frame_idx": None if frame_idx is None else int(frame_idx),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(ckpt, out_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save adapted student weights to {out_path}: {exc}") from exc
    return out_path


def save_final_student_weights(yolo_wrapper: YOLO, student_model: nn.Module, out_path: Path) -> Path:
    return save_student_weights_checkpoint(
        yolo_wrapper=yolo_wrapper,
        student_model=student_model,
        out_path=out_path,
        checkpoint_type="final",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online teacher-student ODAD with persist2 replay gating and memory-aware top-k teacher selection."
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--dataset", type=str, required=True, help="YOLO-root dataset path (expects images/test)")
    parser.add_argument("--output", type=str, default="online_adapt_out", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--imgsz", type=int, default=1024, help="Training image size (letterbox target)")

    parser.add_argument("--teacher-conf-thresh", type=float, default=0.80, help="Pseudo-label acceptance threshold")
    parser.add_argument(
        "--infer-conf",
        "--infer_conf",
        dest="infer_conf",
        type=float,
        default=0.001,
        help="Inference conf for teacher/student top1",
    )
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames, else first N frames")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup teacher forwards before adaptation")

    parser.add_argument("--lr", type=float, default=1e-4, help="Student optimizer learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="SGD weight decay")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="Teacher EMA decay")
    parser.add_argument("--grad-clip", type=float, default=10.0, help="Gradient clipping max norm")

    parser.add_argument(
        "--update-scope",
        type=str,
        default="head_only",
        choices=["head_only", "neck_head"],
        help="Student trainable region: head_only or neck_head",
    )
    parser.add_argument(
        "--neck-start-idx",
        type=int,
        default=-1,
        help="Manual neck start index override; >=0 disables YAML auto-detection",
    )

    parser.add_argument("--buffer-size", type=int, default=32, help="Replay buffer capacity")
    parser.add_argument("--update-batch-size", type=int, default=4, help="Mini-batch size sampled from replay buffer")
    parser.add_argument(
        "--min-buffer-before-update",
        type=int,
        default=4,
        help="Minimum number of buffered entries required before updates are allowed",
    )
    parser.add_argument(
        "--buffer-sample-mode",
        type=str,
        default="recent",
        choices=["recent", "random"],
        help="Replay sampling strategy",
    )
    parser.add_argument(
        "--max-updates-per-frame",
        type=int,
        default=1,
        help="Legacy compatibility knob for the old per-frame update path; values >1 act as updates-per-event when that flag stays at 1.",
    )
    parser.add_argument(
        "--update-every-frames",
        type=int,
        default=1,
        help="Run optimizer updates only every N stream frames once the buffer is warm. 1 preserves current behavior.",
    )
    parser.add_argument(
        "--updates-per-event",
        type=int,
        default=1,
        help="Number of optimizer steps to run when an update event is triggered.",
    )

    parser.add_argument("--min-area-frac", type=float, default=0.001, help="Min accepted bbox area fraction")
    parser.add_argument("--max-area-frac", type=float, default=0.80, help="Max accepted bbox area fraction")
    parser.add_argument("--border-margin-frac", type=float, default=0.02, help="Reject boxes too close to border")
    parser.add_argument("--temporal-iou-gate", type=float, default=0.50, help="Require teacher IoU(prev,current) >= gate")
    parser.add_argument(
        "--persistence-frames",
        type=int,
        default=2,
        help="Number of consecutive stable frames required before a pseudo-label is accepted.",
    )
    parser.add_argument(
        "--persistence-iou",
        type=float,
        default=0.50,
        help="Minimum IoU between consecutive teacher boxes for persistence tracking.",
    )
    parser.add_argument(
        "--max-center-shift-frac",
        type=float,
        default=0.20,
        help="Maximum normalized center shift allowed between consecutive teacher boxes.",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=2.5,
        help="Maximum allowed ratio between consecutive box areas.",
    )
    parser.add_argument(
        "--teacher-topk",
        type=int,
        default=1,
        help="Number of teacher detections to consider per frame. 1 preserves current behavior.",
    )
    parser.add_argument(
        "--teacher-candidate-conf-floor",
        type=float,
        default=0.25,
        help="Minimum confidence required for a teacher detection to enter the candidate set.",
    )
    parser.add_argument(
        "--teacher-candidate-score-mode",
        type=str,
        default="conf_temporal",
        choices=["conf_only", "conf_temporal", "conf_temporal_memory"],
        help="How to score top-k teacher candidates.",
    )
    parser.add_argument(
        "--teacher-candidate-conf-weight",
        type=float,
        default=1.0,
        help="Weight for the confidence term in candidate scoring.",
    )
    parser.add_argument(
        "--teacher-candidate-temporal-weight",
        type=float,
        default=1.0,
        help="Weight for the temporal consistency term in candidate scoring.",
    )
    parser.add_argument(
        "--teacher-candidate-memory-weight",
        type=float,
        default=0.5,
        help="Weight for the temporal-memory score term when memory scoring is enabled.",
    )
    parser.add_argument(
        "--teacher-candidate-memory-size",
        type=int,
        default=8,
        help="Number of recent selected candidate states kept in temporal memory.",
    )
    parser.add_argument(
        "--teacher-candidate-memory-score-mode",
        type=str,
        default="top2_mean",
        choices=["max", "top2_mean", "recency_mean"],
        help="How to aggregate similarity to recent memory states.",
    )
    parser.add_argument(
        "--teacher-candidate-memory-layer",
        type=int,
        default=21,
        help="Feature layer used for optional object-centric candidate embeddings.",
    )
    parser.add_argument(
        "--teacher-candidate-memory-use-feature",
        action="store_true",
        help="If enabled, include a compact object-centric feature embedding in the candidate memory state.",
    )
    parser.add_argument(
        "--teacher-candidate-memory-min-box-frac",
        type=float,
        default=0.01,
        help="Minimum normalized box size for reliable object-centric embedding extraction if feature memory is enabled.",
    )
    parser.add_argument(
        "--teacher-candidate-min-score",
        type=float,
        default=0.0,
        help="Optional minimum final candidate score required before selection.",
    )

    parser.add_argument(
        "--save-checkpoints-every",
        type=int,
        default=0,
        help="If >0, save student checkpoints every N frames.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")

    parser.add_argument("--make-mp4", action="store_true", help="Write adaptation overlay MP4")
    parser.add_argument("--mp4-every", type=int, default=1, help="Use every k-th frame in MP4")
    parser.add_argument("--mp4-max", type=int, default=0, help="Max frames in MP4, 0 = no limit")
    parser.add_argument("--mp4-fps", type=int, default=12, help="MP4 fps")
    parser.add_argument("--mp4-scale", type=float, default=0.75, help="Downscale MP4 frames by this factor")
    parser.add_argument(
        "--save-final-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the final adapted student weights at the end of the run.",
    )
    parser.add_argument(
        "--final-weights-name",
        type=str,
        default="student_final.pt",
        help="Filename for the saved final adapted student checkpoint.",
    )

    return parser.parse_args()


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
    if not images:
        raise RuntimeError("No images available after applying --max-frames.")

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
    teacher_topk = max(1, int(args.teacher_topk))
    teacher_candidate_score_mode = str(args.teacher_candidate_score_mode)
    teacher_candidate_memory_size = max(1, int(args.teacher_candidate_memory_size))
    teacher_candidate_memory_score_mode = str(args.teacher_candidate_memory_score_mode)
    memory_score_enabled = teacher_candidate_score_mode == "conf_temporal_memory"
    memory_use_feature = bool(args.teacher_candidate_memory_use_feature) and memory_score_enabled
    if memory_use_feature:
        memory_layer_idx = int(args.teacher_candidate_memory_layer)
        if memory_layer_idx < 0 or memory_layer_idx >= len(teacher_layers):
            raise RuntimeError(
                f"Invalid --teacher-candidate-memory-layer={memory_layer_idx}; expected range [0, {len(teacher_layers) - 1}]."
            )
        teacher_feature_tap: Optional[LayerFeatureTap] = LayerFeatureTap(teacher_layers[memory_layer_idx])
    else:
        teacher_feature_tap = None

    neck_start_idx = resolve_neck_start_idx(core_model, int(args.neck_start_idx))
    unfrozen_indices = compute_unfrozen_indices(
        update_scope=str(args.update_scope),
        neck_start_idx=neck_start_idx,
        head_idx=head_idx,
        n_layers=len(layers),
    )

    if str(args.update_scope) == "head_only":
        head_module_name = layers[head_idx].__class__.__name__
        if head_module_name != "Detect":
            raise RuntimeError(
                f"update_scope=head_only expects Detect head, found {head_module_name} at idx={head_idx}."
            )

    apply_freeze_policy(
        model=student_model,
        layers=layers,
        unfrozen_indices=unfrozen_indices,
    )
    expected_trainable_param_ids = param_id_set_for_indices(layers, unfrozen_indices)
    actual_trainable_param_ids = {id(param) for param in student_model.parameters() if param.requires_grad}
    if expected_trainable_param_ids != actual_trainable_param_ids:
        raise RuntimeError(
            "Freeze policy mismatch: actual trainable parameters do not match expected trainable modules."
        )

    optim_params = [param for param in student_model.parameters() if param.requires_grad]
    if not optim_params:
        raise RuntimeError("No trainable parameters found after freeze policy.")

    optimizer = torch.optim.SGD(
        optim_params,
        lr=float(args.lr),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
    )

    trainable_params = int(sum(param.numel() for param in optim_params))
    unfrozen_modules = [f"{idx}:{layers[idx].__class__.__name__}" for idx in unfrozen_indices]
    persistence_frames = max(1, int(args.persistence_frames))
    update_every_frames, updates_per_event = resolve_update_schedule(
        max_updates_per_frame=int(args.max_updates_per_frame),
        update_every_frames=int(args.update_every_frames),
        updates_per_event=int(args.updates_per_event),
    )
    rng = random.Random(int(args.seed))
    candidate_memory: Deque[CandidateMemoryState] = deque(maxlen=teacher_candidate_memory_size)

    summary_path = out_dir / "summary.txt"
    startup_lines = [
        "startup:",
        f"  update_scope={args.update_scope}",
        f"  neck_start_idx={neck_start_idx if neck_start_idx is not None else 'n/a'}",
        f"  head_idx={head_idx}",
        f"  teacher_conf_thresh={float(args.teacher_conf_thresh):.3f}",
        f"  temporal_iou_gate={float(args.temporal_iou_gate):.3f}",
        f"  persistence_frames={persistence_frames}",
        f"  persistence_iou={float(args.persistence_iou):.3f}",
        f"  max_center_shift_frac={float(args.max_center_shift_frac):.3f}",
        f"  max_area_ratio={float(args.max_area_ratio):.3f}",
        f"  teacher_topk={teacher_topk}",
        f"  teacher_candidate_conf_floor={float(args.teacher_candidate_conf_floor):.3f}",
        f"  teacher_candidate_score_mode={teacher_candidate_score_mode}",
        f"  teacher_candidate_conf_weight={float(args.teacher_candidate_conf_weight):.3f}",
        f"  teacher_candidate_temporal_weight={float(args.teacher_candidate_temporal_weight):.3f}",
        f"  teacher_candidate_memory_weight={float(args.teacher_candidate_memory_weight):.3f}",
        f"  teacher_candidate_memory_size={teacher_candidate_memory_size}",
        f"  teacher_candidate_memory_score_mode={teacher_candidate_memory_score_mode}",
        f"  teacher_candidate_memory_use_feature={int(memory_use_feature)}",
        f"  teacher_candidate_memory_layer={int(args.teacher_candidate_memory_layer)}",
        f"  teacher_candidate_memory_min_box_frac={float(args.teacher_candidate_memory_min_box_frac):.4f}",
        f"  teacher_candidate_min_score={float(args.teacher_candidate_min_score):.3f}",
        f"  update_every_frames={update_every_frames}",
        f"  updates_per_event={updates_per_event}",
        f"  max_updates_per_frame_legacy={max(1, int(args.max_updates_per_frame))}",
        f"  save_checkpoints_every={max(0, int(args.save_checkpoints_every))}",
        f"  save_final_weights={int(bool(args.save_final_weights))}",
        f"  final_weights_name={args.final_weights_name}",
        f"  unfrozen_modules=[{', '.join(unfrozen_modules)}]",
        f"  trainable_params={trainable_params}",
    ]

    print("Startup configuration:")
    for line in startup_lines[1:]:
        print(line)

    summary_path.write_text(
        "\n".join(
            [
                "Online Adaptation Summary",
                "",
                *startup_lines,
                "",
                "status=running",
            ]
        ),
        encoding="utf-8",
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

    buffer = ReplayBuffer(max_size=int(args.buffer_size), rng=rng)

    logs: List[FrameLog] = []
    mp4_frames: List[np.ndarray] = []

    accepted_frames = 0
    updated_frames = 0
    number_of_update_events = 0
    total_optimizer_updates = 0
    update_losses: List[float] = []
    total_losses: List[float] = []
    buffer_sizes_on_updates: List[int] = []
    batch_sizes_on_updates: List[int] = []
    checkpoint_paths: List[Path] = []
    selected_rank_example_paths: List[Path] = []
    max_selected_rank_examples = 8
    selected_rank_gt1_frames = 0

    prev_selected_box: Optional[Tuple[float, float, float, float]] = None
    persistence_state: Optional[PersistenceState] = None
    motion_gate_enabled = persistence_frames > 1

    progress = make_progress_iter(len(images))
    for idx, img_path in enumerate(images):
        frame_t0 = time.time()

        with Image.open(img_path) as im:
            img_rgb = im.convert("RGB")
            w, h = img_rgb.size

        teacher_model.eval()
        raw_teacher_top1, teacher_candidates, teacher_feature_map, teacher_lat_ms = predict_teacher_candidates_wrapper(
            yolo_wrapper=teacher_yolo,
            source=str(img_path),
            device=str(args.device),
            conf=float(args.infer_conf),
            iou=float(args.iou),
            topk=teacher_topk,
            conf_floor=float(args.teacher_candidate_conf_floor),
            allow_top1_fallback=teacher_topk <= 1,
            feature_tap=teacher_feature_tap,
        )
        selection = select_teacher_candidate(
            candidates=teacher_candidates,
            prev_reference_box=prev_selected_box,
            img_w=w,
            img_h=h,
            max_center_shift_frac=float(args.max_center_shift_frac),
            max_area_ratio=float(args.max_area_ratio),
            score_mode=teacher_candidate_score_mode,
            conf_weight=float(args.teacher_candidate_conf_weight),
            temporal_weight=float(args.teacher_candidate_temporal_weight),
            memory_weight=float(args.teacher_candidate_memory_weight),
            min_score=float(args.teacher_candidate_min_score),
            current_frame_idx=idx,
            memory_states=list(candidate_memory),
            memory_score_mode=teacher_candidate_memory_score_mode,
            memory_feature_map=teacher_feature_map,
            memory_use_feature=memory_use_feature,
            memory_min_box_frac=float(args.teacher_candidate_memory_min_box_frac),
        )
        teacher_top1 = selection.selected

        if selection.selected_rank > 1:
            selected_rank_gt1_frames += 1
            if len(selected_rank_example_paths) < max_selected_rank_examples and raw_teacher_top1 is not None:
                selected_rank_example_paths.append(
                    save_selected_rank_example_image(
                        out_dir=out_dir,
                        frame_idx=idx,
                        img_rgb=img_rgb,
                        raw_top1=raw_teacher_top1,
                        selected_candidate=teacher_top1,
                        selected_score=float(selection.selected_score),
                        score_conf=float(selection.score_conf),
                        score_temporal=float(selection.score_temporal),
                        score_memory=float(selection.score_memory),
                        score_mode=teacher_candidate_score_mode,
                    )
                )

        passed_base_gate, temporal_iou = evaluate_base_gate(
            top1=teacher_top1,
            prev_teacher_box=prev_selected_box,
            img_w=w,
            img_h=h,
            conf_thresh=float(args.teacher_conf_thresh),
            min_area_frac=float(args.min_area_frac),
            max_area_frac=float(args.max_area_frac),
            border_margin_frac=float(args.border_margin_frac),
            temporal_iou_gate=float(args.temporal_iou_gate),
        )
        passed_motion_gate, center_shift_frac, area_ratio = evaluate_motion_gate(
            top1=teacher_top1,
            prev_teacher_box=prev_selected_box,
            img_w=w,
            img_h=h,
            max_center_shift_frac=float(args.max_center_shift_frac),
            max_area_ratio=float(args.max_area_ratio),
            enabled=motion_gate_enabled,
        )
        persistence_state, persistence_count, persistence_iou, passed_persistence_gate = update_persistence_state(
            state=persistence_state,
            top1=teacher_top1,
            candidate_valid=bool(passed_base_gate and passed_motion_gate),
            persistence_frames=persistence_frames,
            persistence_iou=float(args.persistence_iou),
        )
        accepted = bool(passed_base_gate and passed_motion_gate and passed_persistence_gate)
        accepted_final = bool(accepted)
        if (
            teacher_top1 is not None
            and selection.selected_state_vector is not None
            and passed_base_gate
            and passed_motion_gate
        ):
            candidate_memory.append(
                CandidateMemoryState(
                    frame_idx=idx,
                    cls_id=int(teacher_top1.cls_id),
                    box_xyxy=teacher_top1.xyxy,
                    conf=float(teacher_top1.conf),
                    state_vector=selection.selected_state_vector.copy(),
                )
            )
        if accepted_final and teacher_top1 is not None:
            accepted_frames += 1
            buffer.add(
                ReplayEntry(
                    frame_idx=idx,
                    path=str(img_path),
                    width=w,
                    height=h,
                    pseudo_box=teacher_top1.xyxy,
                    pseudo_cls=int(teacher_top1.cls_id),
                )
            )

        updates_this_frame = 0
        batch_size_used = 0
        num_pseudo_boxes_used = 0
        last_det_loss = float("nan")
        last_total_loss = float("nan")
        update_latency_ms = 0.0
        buffer_warm = len(buffer) >= int(args.min_buffer_before_update)
        update_event_triggered = int(buffer_warm and should_trigger_update_event(idx, update_every_frames))

        if update_event_triggered:
            number_of_update_events += 1
            # Replay admission stays continuous; only optimizer steps are cadence-gated.
            for _ in range(updates_per_event):
                target_entries = buffer.sample(batch_size=int(args.update_batch_size), mode=str(args.buffer_sample_mode))
                if not target_entries:
                    break

                apply_freeze_policy(
                    model=student_model,
                    layers=layers,
                    unfrozen_indices=unfrozen_indices,
                )

                update_t0 = time.time()
                batch = build_training_batch(
                    target_entries=target_entries,
                    imgsz=int(args.imgsz),
                    device=str(args.device),
                    rng=rng,
                )
                det_loss, total_loss = run_detection_update(
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
                last_total_loss = float(total_loss)
                update_losses.append(float(det_loss))
                total_losses.append(float(total_loss))
                target_count_step = len(target_entries)
                batch_size_used = target_count_step
                num_pseudo_boxes_used += target_count_step
                buffer_sizes_on_updates.append(len(buffer))
                batch_sizes_on_updates.append(target_count_step)

        if updates_this_frame > 0:
            updated_frames += 1

        student_model.eval()
        student_post_top1, student_post_lat_ms = predict_top1_wrapper(
            yolo_wrapper=student_yolo,
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
                accepted_final=int(accepted_final),
                passed_base_gate=int(passed_base_gate),
                passed_motion_gate=int(passed_motion_gate),
                passed_persistence_gate=int(passed_persistence_gate),
                teacher_num_candidates=int(selection.num_candidates),
                teacher_selected_rank=int(selection.selected_rank),
                teacher_selected_score=float(selection.selected_score),
                teacher_selected_score_conf=float(selection.score_conf),
                teacher_selected_score_temporal=float(selection.score_temporal),
                teacher_selected_score_memory=float(selection.score_memory),
                teacher_candidate_memory_size_active=int(selection.memory_size_active),
                persistence_count=int(persistence_count),
                temporal_iou=float(temporal_iou),
                persistence_iou=float(persistence_iou),
                center_shift_frac=float(center_shift_frac),
                area_ratio=float(area_ratio),
                num_pseudo_boxes_used=int(num_pseudo_boxes_used),
                buffer_size=len(buffer),
                update_event_triggered=int(update_event_triggered),
                update_applied=int(updates_this_frame > 0),
                updates_this_frame=int(updates_this_frame),
                batch_size_used=int(batch_size_used),
                det_loss=float(last_det_loss),
                total_loss=float(last_total_loss),
                teacher_latency_ms=float(teacher_lat_ms),
                student_post_conf=float(student_post_top1.conf) if student_post_top1 is not None else 0.0,
                student_post_latency_ms=float(student_post_lat_ms),
                update_latency_ms=float(update_latency_ms),
                frame_latency_ms=float(frame_latency_ms),
            )
        )

        prev_selected_box = teacher_top1.xyxy if teacher_top1 is not None else None

        if args.make_mp4:
            use_mp4 = idx % max(1, int(args.mp4_every)) == 0
            under_cap = int(args.mp4_max) <= 0 or len(mp4_frames) < int(args.mp4_max)
            if use_mp4 and under_cap:
                triptych = make_triptych(
                    img_rgb=img_rgb,
                    frame_idx=idx,
                    teacher_top1=teacher_top1,
                    student_top1=student_post_top1,
                    accepted=accepted_final,
                    update_applied=bool(updates_this_frame > 0),
                    buffer_size=len(buffer),
                    loss_value=float(last_total_loss if math.isfinite(last_total_loss) else last_det_loss),
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
                    "rank2+": selected_rank_gt1_frames,
                    "bs": batch_size_used,
                },
                refresh=False,
            )

        if int(args.save_checkpoints_every) > 0 and (idx + 1) % int(args.save_checkpoints_every) == 0:
            checkpoint_path = save_student_weights_checkpoint(
                yolo_wrapper=student_yolo,
                student_model=student_model,
                out_path=out_dir / "checkpoints" / f"student_frame_{idx + 1:06d}.pt",
                checkpoint_type="intermediate",
                frame_idx=idx,
            )
            checkpoint_paths.append(checkpoint_path)

    if progress is not None:
        progress.close()

    csv_path = out_dir / "adapt_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "path",
                "teacher_conf",
                "accepted",
                "accepted_final",
                "passed_base_gate",
                "passed_motion_gate",
                "passed_persistence_gate",
                "teacher_num_candidates",
                "teacher_selected_rank",
                "teacher_selected_score",
                "teacher_selected_score_conf",
                "teacher_selected_score_temporal",
                "teacher_selected_score_memory",
                "teacher_candidate_memory_size_active",
                "persistence_count",
                "temporal_iou",
                "persistence_iou",
                "center_shift_frac",
                "area_ratio",
                "num_pseudo_boxes_used",
                "buffer_size",
                "update_event_triggered",
                "update_applied",
                "updates_this_frame",
                "batch_size_used",
                "det_loss",
                "total_loss",
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
                    row.accepted_final,
                    row.passed_base_gate,
                    row.passed_motion_gate,
                    row.passed_persistence_gate,
                    row.teacher_num_candidates,
                    row.teacher_selected_rank,
                    f"{row.teacher_selected_score:.6f}" if math.isfinite(row.teacher_selected_score) else "",
                    (
                        f"{row.teacher_selected_score_conf:.6f}"
                        if math.isfinite(row.teacher_selected_score_conf)
                        else ""
                    ),
                    (
                        f"{row.teacher_selected_score_temporal:.6f}"
                        if math.isfinite(row.teacher_selected_score_temporal)
                        else ""
                    ),
                    (
                        f"{row.teacher_selected_score_memory:.6f}"
                        if math.isfinite(row.teacher_selected_score_memory)
                        else ""
                    ),
                    row.teacher_candidate_memory_size_active,
                    row.persistence_count,
                    f"{row.temporal_iou:.6f}" if math.isfinite(row.temporal_iou) else "",
                    f"{row.persistence_iou:.6f}" if math.isfinite(row.persistence_iou) else "",
                    f"{row.center_shift_frac:.6f}" if math.isfinite(row.center_shift_frac) else "",
                    f"{row.area_ratio:.6f}" if math.isfinite(row.area_ratio) else "",
                    row.num_pseudo_boxes_used,
                    row.buffer_size,
                    row.update_event_triggered,
                    row.update_applied,
                    row.updates_this_frame,
                    row.batch_size_used,
                    f"{row.det_loss:.6f}" if math.isfinite(row.det_loss) else "",
                    f"{row.total_loss:.6f}" if math.isfinite(row.total_loss) else "",
                    f"{row.teacher_latency_ms:.3f}",
                    f"{row.student_post_conf:.6f}",
                    f"{row.student_post_latency_ms:.3f}",
                    f"{row.update_latency_ms:.3f}",
                    f"{row.frame_latency_ms:.3f}",
                ]
            )

    frames = np.array([row.frame for row in logs], dtype=np.int32)
    teacher_conf_vals = np.array([row.teacher_conf for row in logs], dtype=np.float32)
    accepted_vals = np.array([row.accepted for row in logs], dtype=np.float32)
    accepted_final_vals = np.array([row.accepted_final for row in logs], dtype=np.float32)
    student_post_conf_vals = np.array([row.student_post_conf for row in logs], dtype=np.float32)
    update_event_triggered_vals = np.array([row.update_event_triggered for row in logs], dtype=np.float32)
    update_applied_vals = np.array([row.update_applied for row in logs], dtype=np.float32)
    buffer_size_vals = np.array([row.buffer_size for row in logs], dtype=np.float32)
    teacher_num_candidates_vals = np.array([row.teacher_num_candidates for row in logs], dtype=np.float32)
    teacher_selected_rank_vals = np.array([row.teacher_selected_rank for row in logs], dtype=np.float32)
    teacher_selected_score_vals = np.array([row.teacher_selected_score for row in logs], dtype=np.float32)
    teacher_selected_score_memory_vals = np.array([row.teacher_selected_score_memory for row in logs], dtype=np.float32)
    teacher_candidate_memory_size_vals = np.array(
        [row.teacher_candidate_memory_size_active for row in logs], dtype=np.float32
    )
    batch_size_used_vals = np.array([row.batch_size_used for row in logs], dtype=np.float32)
    updates_this_frame_vals = np.array([row.updates_this_frame for row in logs], dtype=np.float32)
    det_loss_vals = np.array([row.det_loss for row in logs], dtype=np.float32)
    update_latency_vals = np.array([row.update_latency_ms for row in logs], dtype=np.float32)

    conf_gap_vals = student_post_conf_vals - teacher_conf_vals
    roll_window = max(5, min(100, max(1, len(logs) // 10)))

    save_plot_lines(
        frames,
        [teacher_conf_vals, student_post_conf_vals],
        ["teacher_conf", "student_post_conf"],
        "Teacher vs Student Confidence",
        "frame",
        "confidence",
        out_dir / "plot_teacher_vs_student_conf.png",
    )
    save_plot_lines(
        frames,
        [conf_gap_vals],
        ["student_post_conf - teacher_conf"],
        "Confidence Gap vs Frame",
        "frame",
        "conf_gap",
        out_dir / "plot_conf_gap.png",
    )
    save_plot_lines(
        frames,
        [rolling_mean(conf_gap_vals, roll_window)],
        [f"rolling_mean(window={roll_window})"],
        "Rolling Confidence Gap",
        "frame",
        "conf_gap",
        out_dir / "plot_conf_gap_roll.png",
    )
    save_plot_lines(
        frames,
        [rolling_mean(teacher_conf_vals, roll_window)],
        [f"rolling_mean(window={roll_window})"],
        "Rolling Teacher Confidence",
        "frame",
        "confidence",
        out_dir / "plot_teacher_conf_roll.png",
    )
    save_plot_lines(
        frames,
        [rolling_mean(student_post_conf_vals, roll_window)],
        [f"rolling_mean(window={roll_window})"],
        "Rolling Student Confidence",
        "frame",
        "confidence",
        out_dir / "plot_student_conf_roll.png",
    )
    save_plot_lines(
        frames,
        [rolling_mean(accepted_final_vals, roll_window)],
        [f"rolling_mean(window={roll_window})"],
        "Rolling Acceptance Rate",
        "frame",
        "accept_rate",
        out_dir / "plot_accept_rate_roll.png",
    )
    save_plot_lines(
        frames,
        [rolling_mean(updates_this_frame_vals, roll_window)],
        [f"rolling_mean(window={roll_window})"],
        "Rolling Update Count",
        "frame",
        "updates_per_frame",
        out_dir / "plot_update_count_roll.png",
    )
    save_plot_lines(
        frames,
        [accepted_vals, accepted_final_vals, update_event_triggered_vals, update_applied_vals],
        ["accepted_base", "accepted_final", "update_event_triggered", "update_applied"],
        "Accept and Update Flags",
        "frame",
        "flag",
        out_dir / "plot_accept_update_flags.png",
    )
    save_plot_lines(
        frames,
        [buffer_size_vals],
        ["buffer_size"],
        "Replay Buffer Size",
        "frame",
        "entries",
        out_dir / "plot_buffer_size.png",
    )
    save_plot_lines(
        frames,
        [teacher_selected_rank_vals],
        ["teacher_selected_rank"],
        "Selected Teacher Rank",
        "frame",
        "rank",
        out_dir / "plot_selected_rank.png",
    )
    save_plot_lines(
        frames,
        [rolling_mean_ignore_nan(teacher_selected_score_vals, roll_window)],
        [f"rolling_mean(window={roll_window})"],
        "Rolling Selected Candidate Score",
        "frame",
        "score",
        out_dir / "plot_selected_score_roll.png",
    )
    save_plot_lines(
        frames,
        [rolling_mean_ignore_nan(teacher_selected_score_memory_vals, roll_window)],
        [f"rolling_mean(window={roll_window})"],
        "Rolling Selected Memory Score",
        "frame",
        "memory_score",
        out_dir / "plot_selected_memory_score_roll.png",
    )
    save_plot_lines(
        frames,
        [batch_size_used_vals],
        ["batch_size_used"],
        "Batch Size Used",
        "frame",
        "batch_size",
        out_dir / "plot_batch_size_used.png",
    )

    update_mask = update_applied_vals > 0.5
    save_plot_lines(
        frames[update_mask],
        [det_loss_vals[update_mask]],
        ["det_loss"],
        "Detection Loss on Update Frames",
        "frame",
        "det_loss",
        out_dir / "plot_det_loss.png",
    )
    save_plot_hist(conf_gap_vals, 40, "Confidence Gap Histogram", "conf_gap", out_dir / "hist_conf_gap.png")
    save_plot_hist(
        update_latency_vals[update_mask],
        40,
        "Update Latency Histogram",
        "update_latency_ms",
        out_dir / "hist_update_latency.png",
    )

    mean_teacher_conf = float(np.mean(teacher_conf_vals)) if logs else float("nan")
    mean_student_post_conf = float(np.mean(student_post_conf_vals)) if logs else float("nan")
    mean_update_loss = mean_or_nan(update_losses)
    mean_total_loss_updates = mean_or_nan(total_losses)
    mean_buffer_size_updates = mean_or_nan(buffer_sizes_on_updates)

    mean_conf_gap = float(np.mean(conf_gap_vals)) if logs else float("nan")
    median_conf_gap = float(np.median(conf_gap_vals)) if logs else float("nan")
    fraction_frames_student_gt_teacher = float(np.mean(conf_gap_vals > 0.0)) if logs else float("nan")
    fraction_frames_student_lt_teacher = float(np.mean(conf_gap_vals < 0.0)) if logs else float("nan")
    mean_accept_rate_roll = float(np.mean(rolling_mean(accepted_final_vals, roll_window))) if logs else float("nan")
    mean_batch_size_used_on_updates = mean_or_nan(batch_sizes_on_updates)
    mean_updates_per_event = (
        float(np.mean(updates_this_frame_vals[update_event_triggered_vals > 0.5]))
        if np.any(update_event_triggered_vals > 0.5)
        else float("nan")
    )
    selected_mask = teacher_selected_rank_vals > 0.5
    mean_selected_rank = (
        float(np.mean(teacher_selected_rank_vals[selected_mask]))
        if np.any(selected_mask)
        else float("nan")
    )
    fraction_selected_rank_gt1 = (
        float(np.mean(teacher_selected_rank_vals[selected_mask] > 1.0))
        if np.any(selected_mask)
        else float("nan")
    )
    mean_selected_score = (
        mean_or_nan(teacher_selected_score_vals[selected_mask].tolist())
        if np.any(selected_mask)
        else float("nan")
    )
    mean_selected_memory_score = (
        mean_or_nan(teacher_selected_score_memory_vals[selected_mask].tolist())
        if np.any(selected_mask)
        else float("nan")
    )
    mean_num_candidates = float(np.mean(teacher_num_candidates_vals)) if logs else float("nan")
    mean_memory_size_active = float(np.mean(teacher_candidate_memory_size_vals)) if logs else float("nan")

    final_weights_path: Optional[Path] = None
    if bool(args.save_final_weights):
        student_model.eval()
        final_weights_name = Path(str(args.final_weights_name)).name
        if not final_weights_name.endswith(".pt"):
            raise RuntimeError(
                f"--final-weights-name must end with .pt for YOLO reload compatibility, got: {args.final_weights_name}"
            )
        final_weights_path = save_final_student_weights(
            yolo_wrapper=student_yolo,
            student_model=student_model,
            out_path=out_dir / final_weights_name,
        )
        print(f"Saved final adapted student weights to: {final_weights_path}")

    summary_lines = [
        "Online Adaptation Summary",
        "",
        *startup_lines,
        "",
        f"weights={args.weights}",
        f"dataset={dataset_root}",
        f"total_frames={len(logs)}",
        f"accepted_frames={accepted_frames}",
        f"updates_applied={updated_frames}",
        f"number_of_update_events={number_of_update_events}",
        f"optimizer_update_steps={total_optimizer_updates}",
        format_metric("mean_teacher_conf", mean_teacher_conf),
        format_metric("mean_student_post_conf", mean_student_post_conf),
        format_metric("mean_detection_loss_updates", mean_update_loss),
        format_metric("mean_total_loss_updates", mean_total_loss_updates),
        format_metric("mean_buffer_size_during_updates", mean_buffer_size_updates, ".3f"),
        "",
        "teacher_vs_student:",
        format_metric("mean_conf_gap", mean_conf_gap, prefix="  "),
        format_metric("median_conf_gap", median_conf_gap, prefix="  "),
        format_metric("fraction_frames_student_gt_teacher", fraction_frames_student_gt_teacher, prefix="  "),
        format_metric("fraction_frames_student_lt_teacher", fraction_frames_student_lt_teacher, prefix="  "),
        format_metric("mean_accept_rate_roll", mean_accept_rate_roll, prefix="  "),
        format_metric("mean_batch_size_used_on_updates", mean_batch_size_used_on_updates, prefix="  "),
        "",
        "reliability_gates:",
        f"  teacher_conf_thresh={float(args.teacher_conf_thresh):.3f}",
        f"  temporal_iou_gate={float(args.temporal_iou_gate):.3f}",
        f"  persistence_frames={persistence_frames}",
        f"  persistence_iou={float(args.persistence_iou):.3f}",
        f"  max_center_shift_frac={float(args.max_center_shift_frac):.3f}",
        f"  max_area_ratio={float(args.max_area_ratio):.3f}",
        f"  number_of_checkpoints_saved={len(checkpoint_paths)}",
        "",
        "teacher_candidate_selection:",
        f"  teacher_topk={teacher_topk}",
        f"  teacher_candidate_conf_floor={float(args.teacher_candidate_conf_floor):.3f}",
        f"  teacher_candidate_score_mode={teacher_candidate_score_mode}",
        f"  teacher_candidate_conf_weight={float(args.teacher_candidate_conf_weight):.3f}",
        f"  teacher_candidate_temporal_weight={float(args.teacher_candidate_temporal_weight):.3f}",
        f"  teacher_candidate_memory_weight={float(args.teacher_candidate_memory_weight):.3f}",
        f"  teacher_candidate_memory_size={teacher_candidate_memory_size}",
        f"  teacher_candidate_memory_score_mode={teacher_candidate_memory_score_mode}",
        f"  teacher_candidate_memory_use_feature={int(memory_use_feature)}",
        f"  teacher_candidate_min_score={float(args.teacher_candidate_min_score):.3f}",
        format_metric("mean_num_candidates", mean_num_candidates, prefix="  "),
        format_metric("mean_selected_rank", mean_selected_rank, prefix="  "),
        format_metric("fraction_selected_rank_gt1", fraction_selected_rank_gt1, prefix="  "),
        format_metric("mean_selected_score", mean_selected_score, prefix="  "),
        format_metric("mean_selected_memory_score", mean_selected_memory_score, prefix="  "),
        format_metric("mean_memory_size_active", mean_memory_size_active, prefix="  "),
        f"  selected_rank_gt1_frames={selected_rank_gt1_frames}",
        *(
            [f"  selected_rank_gt1_examples_saved={len(selected_rank_example_paths)}"]
            if selected_rank_example_paths
            else []
        ),
        "",
        "update_schedule:",
        f"  update_every_frames={update_every_frames}",
        f"  updates_per_event={updates_per_event}",
        format_metric("mean_updates_per_event", mean_updates_per_event, prefix="  "),
        f"  number_of_update_events={number_of_update_events}",
        "",
        "buffer_config:",
        f"  buffer_size={int(args.buffer_size)}",
        f"  update_batch_size={int(args.update_batch_size)}",
        f"  min_buffer_before_update={int(args.min_buffer_before_update)}",
        f"  buffer_sample_mode={args.buffer_sample_mode}",
        f"  max_updates_per_frame_legacy={max(1, int(args.max_updates_per_frame))}",
        "",
        "outputs:",
        f"  csv={csv_path.name}",
        f"  summary={summary_path.name}",
        *([f"  final_student_weights={final_weights_path.name}"] if final_weights_path is not None else []),
        "  plot_teacher_vs_student_conf.png",
        "  plot_conf_gap.png",
        "  plot_conf_gap_roll.png",
        "  plot_teacher_conf_roll.png",
        "  plot_student_conf_roll.png",
        "  plot_accept_rate_roll.png",
        "  plot_update_count_roll.png",
        "  plot_accept_update_flags.png",
        "  plot_buffer_size.png",
        "  plot_selected_rank.png",
        "  plot_selected_score_roll.png",
        "  plot_selected_memory_score_roll.png",
        "  plot_batch_size_used.png",
        "  plot_det_loss.png",
        "  hist_conf_gap.png",
        "  hist_update_latency.png",
        *(
            [f"  selected_rank_gt1_dir=selected_rank_gt1 ({len(selected_rank_example_paths)} saved)"]
            if selected_rank_example_paths
            else []
        ),
    ]
    if checkpoint_paths:
        summary_lines.extend(
            [
                "",
                "intermediate_checkpoints:",
                *[f"  {path.relative_to(out_dir)}" for path in checkpoint_paths],
            ]
        )
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

    if teacher_feature_tap is not None:
        teacher_feature_tap.close()

    print(f"Done. Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()


# PYTHONPATH=. /home/hm25936/miniforge3/envs/gpu_test/bin/python3 odad/online_adapt.py \
#   --weights /home/hm25936/mae/runs/yolov8_baseline/baseline/weights/best.pt \
#   --dataset /home/hm25936/datasets_for_yolo/lab_images_6000 \
#   --output /home/hm25936/mae/odad/online_adapt_topk2_full \
#   --device cuda:0 \
#   --imgsz 1024 \
#   --teacher-conf-thresh 0.80 \
#   --infer-conf 0.001 \
#   --iou 0.45 \
#   --lr 3e-4 \
#   --ema-decay 0.999 \
#   --update-scope head_only \
#   --buffer-size 32 \
#   --update-batch-size 4 \
#   --min-buffer-before-update 4 \
#   --buffer-sample-mode recent \
#   --max-updates-per-frame 1 \
#   --update-every-frames 1 \
#   --updates-per-event 1 \
#   --temporal-iou-gate 0.50 \
#   --persistence-frames 2 \
#   --persistence-iou 0.50 \
#   --max-center-shift-frac 0.20 \
#   --max-area-ratio 2.5 \
#   --teacher-topk 2 \
#   --teacher-candidate-conf-floor 0.25 \
#   --teacher-candidate-score-mode conf_temporal \
#   --teacher-candidate-conf-weight 1.0 \
#   --teacher-candidate-temporal-weight 1.0 \
#   --teacher-candidate-min-score 0.0 \
#   --save-checkpoints-every 500 \
#   --save-final-weights \
#   --final-weights-name student_final.pt \
#   --make-mp4 \
#   --mp4-every 2 \
#   --mp4-fps 12 \
#   --mp4-scale 0.75
