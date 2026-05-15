#!/usr/bin/env python3
"""Clean top-k teacher-student ODAD with persist2 replay gating."""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
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

from odad.adapters import (  # noqa: E402
    AdaptedLayer,
    AdapterSpec,
    adapter_debug_stats,
    adapter_param_count,
    assert_matching_adapter_state,
    attach_residual_adapters,
    apply_adapter_freeze_policy,
    freeze_frozen_batchnorm_stats,
    memory_adapter_modules,
    memory_adapter_param_count,
    memory_adapter_stats,
    parse_adapter_layers,
    set_memory_bank_slots,
    snapshot_adapter_params,
    sync_memory_context_from_student,
)

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
class CoverageEntry:
    frame_idx: int
    path: str
    width: int
    height: int
    teacher_box_xyxy: Tuple[float, float, float, float]
    teacher_cls: int
    teacher_conf: float
    reason_not_accepted: str


@dataclass
class CoverageLossStats:
    entries_skipped: int = 0
    mean_student_conf: float = float("nan")
    mean_teacher_conf: float = float("nan")
    mean_conf_gap: float = float("nan")


@dataclass
class SourceMemoryStats:
    loss: float = float("nan")
    valid_entries: int = 0
    skipped: int = 0
    mean_pos_sim: float = float("nan")
    mean_neg_sim: float = float("nan")
    margin: float = float("nan")


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
    coverage_buffer_size: int
    coverage_entries_added: int
    coverage_entries_sampled: int
    coverage_loss: float
    coverage_region_entries_skipped: int
    coverage_region_mean_student_conf: float
    coverage_region_mean_teacher_conf: float
    coverage_region_mean_conf_gap: float
    source_memory_loss: float
    source_memory_valid_entries: int
    source_memory_mean_pos_sim: float
    source_memory_mean_neg_sim: float
    source_memory_margin: float
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
class CandidateSelectionResult:
    selected: Optional[Top1Det]
    num_candidates: int
    selected_rank: int
    selected_score: float
    score_conf: float
    score_temporal: float
    reject_reason: str = ""


@dataclass
class MemoryBankSlotMeta:
    slot_id: int
    frame_idx: int
    path: str
    width: int
    height: int
    box_xyxy: Tuple[float, float, float, float]
    teacher_conf: float
    area_frac: float
    scale_bin: str
    conf_bin: str
    write_source: str
    write_reason: str
    write_count: int
    memory_norm: float


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


class CoverageBuffer:
    def __init__(self, max_size: int, rng: random.Random) -> None:
        self._entries: Deque[CoverageEntry] = deque(maxlen=max(1, int(max_size)))
        self._rng = rng

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entry: CoverageEntry) -> None:
        self._entries.append(entry)

    def sample(self, batch_size: int, mode: str = "recent") -> List[CoverageEntry]:
        if not self._entries:
            return []
        k = min(max(1, int(batch_size)), len(self._entries))
        entries = list(self._entries)
        if mode == "recent":
            return entries[-k:]
        if mode == "random":
            idxs = self._rng.sample(range(len(entries)), k=k)
            return [entries[i] for i in idxs]
        raise RuntimeError(f"Unsupported coverage buffer sample mode: {mode}")


class MemoryAdapterState:
    def __init__(
        self,
        model: nn.Module,
        source_layer: nn.Module,
        memory_dim: int,
        ema: float,
        min_conf: float,
        min_area_frac: float,
        imgsz: int,
        device: str,
    ) -> None:
        self.model = model
        self.source_layer = source_layer
        self.memory_dim = max(1, int(memory_dim))
        self.ema = min(max(float(ema), 0.0), 0.9999)
        self.min_conf = float(min_conf)
        self.min_area_frac = float(min_area_frac)
        self.imgsz = max(1, int(imgsz))
        self.device = str(device)
        self.memory = torch.zeros(self.memory_dim, device=self.device)
        self.initialized = False
        self.updates = 0
        self.last_update_frame = -1
        self.norm_history: List[float] = []
        self.update_history: List[int] = []
        self._feature: Optional[torch.Tensor] = None
        self._hook = self.source_layer.register_forward_hook(self._capture_feature)

    def close(self) -> None:
        self._hook.remove()

    def _capture_feature(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            self._feature = output.detach()

    def record_frame(self) -> None:
        self.norm_history.append(float(self.memory.detach().float().norm().cpu()) if self.initialized else 0.0)
        self.update_history.append(int(self.updates))

    def stats(self) -> Dict[str, float]:
        adapter_stats = memory_adapter_stats(self.model)
        return {
            "memory_adapter_updates": float(self.updates),
            "memory_adapter_initialized": float(int(self.initialized)),
            "memory_adapter_mean_norm": float(adapter_stats.get("memory_adapter_mean_norm", float("nan"))),
            "memory_adapter_last_update_frame": float(self.last_update_frame),
            "mean_memory_conditioning_norm": float(adapter_stats.get("mean_memory_conditioning_norm", float("nan"))),
        }

    def maybe_update(
        self,
        frame_idx: int,
        det: Optional[Top1Det],
        img_w: int,
        img_h: int,
        img_path: Optional[Path] = None,
        img_rgb: Optional[Image.Image] = None,
    ) -> bool:
        del img_path, img_rgb
        if det is None:
            return False
        if float(det.conf) < self.min_conf:
            return False
        area_frac = box_area(det.xyxy) / max(1.0, float(img_w * img_h))
        if area_frac < self.min_area_frac:
            return False
        feature = self._feature
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4 or int(feature.shape[0]) < 1:
            return False
        pooled = pool_feature_for_original_box(
            feature=feature[0],
            box_xyxy=det.xyxy,
            orig_w=int(img_w),
            orig_h=int(img_h),
        )
        source_adapter = first_memory_source_adapter(self.model)
        if source_adapter is None:
            raise RuntimeError("Memory adapter source layer must be one of the adapted memory layers for v1.")
        with torch.no_grad():
            embedding = source_adapter.project_memory_feature(pooled)
            embedding = torch.nn.functional.normalize(embedding.float(), dim=0, eps=1e-6)
            if not self.initialized:
                self.memory = embedding.to(device=self.memory.device)
                self.initialized = True
            else:
                self.memory.mul_(self.ema).add_(embedding.to(device=self.memory.device), alpha=(1.0 - self.ema))
                self.memory = torch.nn.functional.normalize(self.memory, dim=0, eps=1e-6)
            for adapter in memory_adapter_modules(self.model):
                adapter.set_memory_context(self.memory, initialized=True)
            self.updates += 1
            self.last_update_frame = int(frame_idx)
        return True


class MemoryBankAdapterState:
    def __init__(
        self,
        model: nn.Module,
        source_layer: nn.Module,
        memory_dim: int,
        bank_size: int,
        topk: int,
        query_mode: str,
        write_policy: str,
        diversity_thresh: float,
        duplicate_thresh: float,
        quality_margin: float,
        balance_scale_bins: bool,
        balance_conf_bins: bool,
        stable_medium_write: bool,
        stable_conf_min: float,
        stable_iou_min: float,
        retrieval_temp: float,
        min_conf: float,
        min_area_frac: float,
        imgsz: int,
        device: str,
        out_dir: Path,
        debug_save: bool,
        debug_max_images: int,
        debug_every: int,
        debug_topk_examples: int,
    ) -> None:
        self.model = model
        self.source_layer = source_layer
        self.memory_dim = max(1, int(memory_dim))
        self.bank_size = max(1, int(bank_size))
        self.topk = max(1, int(topk))
        self.query_mode = str(query_mode)
        self.write_policy = str(write_policy)
        self.diversity_thresh = float(diversity_thresh)
        self.duplicate_thresh = float(duplicate_thresh)
        self.quality_margin = max(0.0, float(quality_margin))
        self.balance_scale_bins = bool(balance_scale_bins)
        self.balance_conf_bins = bool(balance_conf_bins)
        self.stable_medium_write = bool(stable_medium_write)
        self.stable_conf_min = float(stable_conf_min)
        self.stable_iou_min = float(stable_iou_min)
        self.retrieval_temp = max(1e-6, float(retrieval_temp))
        self.min_conf = float(min_conf)
        self.min_area_frac = float(min_area_frac)
        self.imgsz = max(1, int(imgsz))
        self.device = str(device)
        self.slots = torch.zeros(self.bank_size, self.memory_dim, device=self.device)
        self.slot_metas: List[Optional[MemoryBankSlotMeta]] = [None for _ in range(self.bank_size)]
        self.slot_write_counts: List[int] = [0 for _ in range(self.bank_size)]
        self.active_slots = 0
        self.writes = 0
        self.appends = 0
        self.replacements = 0
        self.duplicate_skips = 0
        self.low_quality_skips = 0
        self.stable_medium_writes = 0
        self.decision_counts: Counter[str] = Counter()
        self.last_update_frame = -1
        self.last_retrieval: Dict[str, float] = {
            "memory_retrieval_top1_sim": float("nan"),
            "memory_retrieval_mean_topk_sim": float("nan"),
            "memory_retrieval_top1_slot": -1.0,
            "memory_retrieval_entropy": float("nan"),
            "memory_bank_context_norm": 0.0,
        }
        self.timeline_rows: List[Dict[str, float]] = []
        self.debug_save = bool(debug_save)
        self.debug_max_images = max(0, int(debug_max_images))
        self.debug_every = max(1, int(debug_every))
        self.debug_topk_examples = max(1, int(debug_topk_examples))
        self.debug_images_saved = 0
        self.out_dir = out_dir
        self.debug_root = out_dir / "memory_debug"
        self.write_dir = self.debug_root / "memory_writes"
        self.skipped_write_dir = self.debug_root / "memory_skipped_writes"
        self.retrieval_dir = self.debug_root / "retrieval_examples"
        if self.debug_save:
            self.write_dir.mkdir(parents=True, exist_ok=True)
            self.skipped_write_dir.mkdir(parents=True, exist_ok=True)
            self.retrieval_dir.mkdir(parents=True, exist_ok=True)
        self._feature: Optional[torch.Tensor] = None
        self._hook = self.source_layer.register_forward_hook(self._capture_feature)

    def close(self) -> None:
        self._hook.remove()

    def _capture_feature(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            self._feature = output.detach()

    def _source_adapter(self):
        source_adapter = first_memory_source_adapter(self.model)
        if source_adapter is None:
            raise RuntimeError("Memory adapter source layer must be one of the adapted memory layers.")
        return source_adapter

    def _embedding_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embedding = self._source_adapter().project_memory_feature(pooled)
            return torch.nn.functional.normalize(embedding.float(), dim=0, eps=1e-6).to(device=self.slots.device)

    def _current_feature(self) -> Optional[torch.Tensor]:
        feature = self._feature
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4 or int(feature.shape[0]) < 1:
            return None
        return feature[0]

    def _roi_embedding(self, det: Optional[Top1Det], img_w: int, img_h: int) -> Optional[torch.Tensor]:
        feature = self._current_feature()
        if feature is None or det is None:
            return None
        pooled = pool_feature_for_original_box(
            feature=feature,
            box_xyxy=det.xyxy,
            orig_w=int(img_w),
            orig_h=int(img_h),
        )
        return self._embedding_from_pooled(pooled)

    def _query_embedding(self, det: Optional[Top1Det], img_w: int, img_h: int) -> Optional[torch.Tensor]:
        feature = self._current_feature()
        if feature is None:
            return None
        if self.query_mode == "global_gap":
            pooled = feature.float().mean(dim=(1, 2))
            return self._embedding_from_pooled(pooled)
        if self.query_mode in {"teacher_roi", "student_roi"}:
            return self._roi_embedding(det=det, img_w=img_w, img_h=img_h)
        raise RuntimeError(f"Unsupported memory bank query mode: {self.query_mode}")

    def _pairwise_stats(self) -> Tuple[float, float]:
        if self.active_slots <= 1:
            return float("nan"), float("nan")
        slot_mat = torch.nn.functional.normalize(self.slots[: self.active_slots].float(), dim=1, eps=1e-6)
        sims = slot_mat @ slot_mat.t()
        mask = ~torch.eye(self.active_slots, dtype=torch.bool, device=sims.device)
        vals = sims[mask]
        return float(vals.mean().cpu()), float(vals.max().cpu())

    @staticmethod
    def _scale_bin(area_frac: float) -> str:
        if float(area_frac) < 0.01:
            return "small"
        if float(area_frac) < 0.05:
            return "medium"
        return "large"

    @staticmethod
    def _conf_bin(conf: float) -> str:
        return "medium" if float(conf) < 0.80 else "high"

    def _slot_bin_counts(self, attr: str) -> Counter[str]:
        counts: Counter[str] = Counter()
        for meta in self.slot_metas[: self.active_slots]:
            if meta is not None:
                counts[str(getattr(meta, attr))] += 1
        return counts

    @staticmethod
    def _format_counts(counts: Counter[str], keys: Sequence[str]) -> str:
        return "|".join(f"{key}:{int(counts.get(key, 0))}" for key in keys)

    def _slot_utility(self, idx: int, embedding: torch.Tensor) -> float:
        meta = self.slot_metas[int(idx)]
        quality = float(meta.teacher_conf) if meta is not None else 0.0
        age = max(0, int(self.last_update_frame) - int(meta.frame_idx)) if meta is not None else self.bank_size
        sim = float((torch.nn.functional.normalize(self.slots[int(idx)].float(), dim=0, eps=1e-6) @ embedding.float()).cpu())
        return quality - 0.05 * min(age / 500.0, 1.0) + 0.10 * sim

    def _overrepresented_slot(
        self,
        embedding: torch.Tensor,
        scale_bin: str,
        conf_bin: str,
    ) -> Optional[int]:
        candidates = list(range(self.active_slots))
        if not candidates:
            return None
        scale_counts = self._slot_bin_counts("scale_bin")
        conf_counts = self._slot_bin_counts("conf_bin")
        target_scale_count = int(scale_counts.get(scale_bin, 0))
        target_conf_count = int(conf_counts.get(conf_bin, 0))

        def pressure(idx: int) -> Tuple[int, float]:
            meta = self.slot_metas[idx]
            if meta is None:
                return (0, 0.0)
            score = 0
            if self.balance_scale_bins and int(scale_counts.get(meta.scale_bin, 0)) > target_scale_count:
                score += 2
            if self.balance_conf_bins and int(conf_counts.get(meta.conf_bin, 0)) > target_conf_count:
                score += 1
            return (score, -self._slot_utility(idx, embedding))

        best_idx = max(candidates, key=pressure)
        return int(best_idx) if pressure(best_idx)[0] > 0 else None

    def _select_replacement_slot(
        self,
        embedding: torch.Tensor,
        teacher_conf: float,
        scale_bin: str,
        conf_bin: str,
    ) -> Tuple[Optional[int], str]:
        if self.active_slots < self.bank_size:
            if self.write_policy in {"fifo", "diverse_fifo"}:
                return int(self.active_slots), "append"
            if self.active_slots < 4:
                return int(self.active_slots), "append"
            active = torch.nn.functional.normalize(self.slots[: self.active_slots].float(), dim=1, eps=1e-6)
            max_sim = float(torch.max(active @ embedding.float()).cpu())
            if max_sim >= self.duplicate_thresh:
                return None, "skip_duplicate"
            return int(self.active_slots), "append"

        active = torch.nn.functional.normalize(self.slots[: self.active_slots].float(), dim=1, eps=1e-6)
        sims = active @ embedding.float()
        max_sim, max_idx = torch.max(sims, dim=0)
        if self.write_policy == "fifo":
            return self._oldest_slot(), "replace_fifo"
        if self.write_policy == "diverse_fifo":
            if float(max_sim.cpu()) >= self.diversity_thresh:
                return None, "skip_duplicate"
            return self._oldest_slot(), "replace_diverse"
        if self.write_policy in {"diversity_reservoir", "hard_example_aware", "scale_conf_balanced"}:
            max_sim_f = float(max_sim.cpu())
            near_dup = max_sim_f >= self.duplicate_thresh
            similar = max_sim_f >= self.diversity_thresh
            nearest_idx = int(max_idx.item())
            nearest_meta = self.slot_metas[nearest_idx]
            nearest_quality = float(nearest_meta.teacher_conf) if nearest_meta is not None else 0.0
            if near_dup:
                if float(teacher_conf) >= nearest_quality + self.quality_margin:
                    return nearest_idx, "replace_quality"
                return None, "skip_duplicate"
            if self.write_policy == "scale_conf_balanced":
                balanced_slot = self._overrepresented_slot(
                    embedding=embedding,
                    scale_bin=scale_bin,
                    conf_bin=conf_bin,
                )
                if balanced_slot is not None:
                    return int(balanced_slot), "replace_balanced"
            if not similar:
                utilities = [self._slot_utility(idx, embedding) for idx in range(self.active_slots)]
                return int(np.argmin(np.array(utilities, dtype=np.float32))), "replace_diverse"
            return None, "skip_duplicate"
        raise RuntimeError(f"Unsupported memory bank write policy: {self.write_policy}")

    def _oldest_slot(self) -> int:
        best_idx = 0
        best_frame = float("inf")
        for idx, meta in enumerate(self.slot_metas):
            frame = float(meta.frame_idx) if meta is not None else float("-inf")
            if frame < best_frame:
                best_frame = frame
                best_idx = idx
        return int(best_idx)

    def maybe_update(
        self,
        frame_idx: int,
        det: Optional[Top1Det],
        img_w: int,
        img_h: int,
        img_path: Optional[Path] = None,
        img_rgb: Optional[Image.Image] = None,
        stable_medium: bool = False,
        temporal_quality: float = float("nan"),
    ) -> bool:
        if det is None:
            self.low_quality_skips += 1
            self.decision_counts["skip_low_conf"] += 1
            return False
        is_stable_medium = bool(stable_medium)
        min_conf = self.stable_conf_min if is_stable_medium else self.min_conf
        if float(det.conf) < float(min_conf):
            self.low_quality_skips += 1
            self.decision_counts["skip_low_conf"] += 1
            self._save_skipped_write_debug(
                frame_idx=int(frame_idx),
                det=det,
                reason="skip_low_conf",
                img_rgb=img_rgb,
                detail=f"conf={float(det.conf):.3f} min={float(min_conf):.3f}",
            )
            return False
        area_frac = box_area(det.xyxy) / max(1.0, float(img_w * img_h))
        if area_frac < self.min_area_frac:
            self.low_quality_skips += 1
            self.decision_counts["skip_low_area"] += 1
            self._save_skipped_write_debug(
                frame_idx=int(frame_idx),
                det=det,
                reason="skip_low_area",
                img_rgb=img_rgb,
                detail=f"area={float(area_frac):.4f} min={float(self.min_area_frac):.4f}",
            )
            return False
        if is_stable_medium:
            stable_quality = float(temporal_quality) if math.isfinite(float(temporal_quality)) else 0.0
            if not self.stable_medium_write or stable_quality < self.stable_iou_min:
                self.low_quality_skips += 1
                self.decision_counts["skip_unstable_medium"] += 1
                self._save_skipped_write_debug(
                    frame_idx=int(frame_idx),
                    det=det,
                    reason="skip_unstable_medium",
                    img_rgb=img_rgb,
                    detail=f"temporal={stable_quality:.3f} min={float(self.stable_iou_min):.3f}",
                )
                return False
        elif self.write_policy == "hard_example_aware" and float(det.conf) < self.min_conf:
            return False
        embedding = self._roi_embedding(det=det, img_w=img_w, img_h=img_h)
        if embedding is None:
            return False
        scale_bin = self._scale_bin(float(area_frac))
        conf_bin = self._conf_bin(float(det.conf))
        slot_idx, reason = self._select_replacement_slot(
            embedding=embedding,
            teacher_conf=float(det.conf),
            scale_bin=scale_bin,
            conf_bin=conf_bin,
        )
        if slot_idx is None:
            if reason == "skip_duplicate":
                self.duplicate_skips += 1
            else:
                self.low_quality_skips += 1
            self.decision_counts[reason] += 1
            self._save_skipped_write_debug(
                frame_idx=int(frame_idx),
                det=det,
                reason=reason,
                img_rgb=img_rgb,
                detail=f"scale={scale_bin} conf_bin={conf_bin}",
            )
            return False
        if self.active_slots >= self.bank_size:
            self.replacements += 1
        else:
            self.active_slots += 1
            self.appends += 1
        self.slots[slot_idx].copy_(embedding)
        self.slot_write_counts[slot_idx] += 1
        self.writes += 1
        self.decision_counts[reason] += 1
        if is_stable_medium:
            self.stable_medium_writes += 1
        self.last_update_frame = int(frame_idx)
        path_text = str(img_path) if img_path is not None else ""
        self.slot_metas[slot_idx] = MemoryBankSlotMeta(
            slot_id=int(slot_idx),
            frame_idx=int(frame_idx),
            path=path_text,
            width=int(img_w),
            height=int(img_h),
            box_xyxy=tuple(float(v) for v in det.xyxy),
            teacher_conf=float(det.conf),
            area_frac=float(area_frac),
            scale_bin=scale_bin,
            conf_bin=conf_bin,
            write_source="stable_medium" if is_stable_medium else "accepted",
            write_reason=reason,
            write_count=int(self.slot_write_counts[slot_idx]),
            memory_norm=float(embedding.detach().float().norm().cpu()),
        )
        self._save_write_debug(
            frame_idx=int(frame_idx),
            slot_idx=int(slot_idx),
            det=det,
            area_frac=float(area_frac),
            reason=reason,
            img_rgb=img_rgb,
        )
        return True

    def read_current(
        self,
        frame_idx: int,
        det: Optional[Top1Det],
        img_w: int,
        img_h: int,
        img_path: Optional[Path] = None,
        img_rgb: Optional[Image.Image] = None,
    ) -> None:
        if self.active_slots <= 0:
            for adapter in memory_adapter_modules(self.model):
                adapter.clear_memory_context()
            self.last_retrieval = {
                "memory_retrieval_top1_sim": float("nan"),
                "memory_retrieval_mean_topk_sim": float("nan"),
                "memory_retrieval_top1_slot": -1.0,
                "memory_retrieval_entropy": float("nan"),
                "memory_bank_context_norm": 0.0,
            }
            return
        query = self._query_embedding(det=det, img_w=img_w, img_h=img_h)
        if query is None:
            return
        active = torch.nn.functional.normalize(self.slots[: self.active_slots].float(), dim=1, eps=1e-6)
        sims = active @ query.float()
        k = min(self.topk, self.active_slots)
        top_vals, top_idxs = torch.topk(sims, k=k)
        weights = torch.softmax(top_vals / self.retrieval_temp, dim=0)
        context = (self.slots[top_idxs] * weights.view(-1, 1)).sum(dim=0)
        context = torch.nn.functional.normalize(context.float(), dim=0, eps=1e-6)
        for adapter in memory_adapter_modules(self.model):
            adapter.set_memory_context(context, initialized=True)
        set_memory_bank_slots(
            self.model,
            self.slots[: self.active_slots].detach(),
            active_slots=int(self.active_slots),
            initialized=True,
        )
        entropy = float((-(weights * torch.log(weights.clamp_min(1e-12))).sum() / math.log(max(2, k))).cpu())
        self.last_retrieval = {
            "memory_retrieval_top1_sim": float(top_vals[0].cpu()),
            "memory_retrieval_mean_topk_sim": float(top_vals.mean().cpu()),
            "memory_retrieval_top1_slot": float(top_idxs[0].cpu()),
            "memory_retrieval_entropy": entropy,
            "memory_bank_context_norm": float(context.detach().float().norm().cpu()),
        }
        save_periodic_debug = self.debug_save and int(frame_idx) % self.debug_every == 0
        if save_periodic_debug:
            self._save_retrieval_debug(
                frame_idx=int(frame_idx),
                det=det,
                img_rgb=img_rgb,
                top_idxs=[int(v) for v in top_idxs.detach().cpu().tolist()],
                top_vals=[float(v) for v in top_vals.detach().cpu().tolist()],
            )

    def record_frame(self) -> None:
        mean_pair, max_pair = self._pairwise_stats()
        scale_counts = self._slot_bin_counts("scale_bin")
        conf_counts = self._slot_bin_counts("conf_bin")
        self.timeline_rows.append(
            {
                "frame": float(len(self.timeline_rows)),
                "memory_bank_active_slots": float(self.active_slots),
                "memory_bank_writes": float(self.writes),
                "memory_bank_appends": float(self.appends),
                "memory_bank_replacements": float(self.replacements),
                "memory_bank_duplicate_skips": float(self.duplicate_skips),
                "memory_bank_low_quality_skips": float(self.low_quality_skips),
                "memory_bank_stable_medium_writes": float(self.stable_medium_writes),
                "memory_slot_scale_small": float(scale_counts.get("small", 0)),
                "memory_slot_scale_medium": float(scale_counts.get("medium", 0)),
                "memory_slot_scale_large": float(scale_counts.get("large", 0)),
                "memory_slot_conf_medium": float(conf_counts.get("medium", 0)),
                "memory_slot_conf_high": float(conf_counts.get("high", 0)),
                "memory_retrieval_top1_sim": float(self.last_retrieval["memory_retrieval_top1_sim"]),
                "memory_retrieval_mean_topk_sim": float(self.last_retrieval["memory_retrieval_mean_topk_sim"]),
                "memory_retrieval_top1_slot": float(self.last_retrieval["memory_retrieval_top1_slot"]),
                "memory_retrieval_entropy": float(self.last_retrieval["memory_retrieval_entropy"]),
                "memory_slot_mean_pairwise_sim": float(mean_pair),
                "memory_slot_max_pairwise_sim": float(max_pair),
                "memory_bank_context_norm": float(self.last_retrieval["memory_bank_context_norm"]),
                "memory_debug_images_saved": float(self.debug_images_saved),
            }
        )

    def stats(self) -> Dict[str, float]:
        adapter_stats = memory_adapter_stats(self.model)
        mean_pair, max_pair = self._pairwise_stats()
        scale_counts = self._slot_bin_counts("scale_bin")
        conf_counts = self._slot_bin_counts("conf_bin")
        return {
            "memory_adapter_updates": float(self.writes),
            "memory_adapter_initialized": float(int(self.active_slots > 0)),
            "memory_adapter_mean_norm": float(adapter_stats.get("memory_adapter_mean_norm", float("nan"))),
            "memory_adapter_last_update_frame": float(self.last_update_frame),
            "mean_memory_conditioning_norm": float(adapter_stats.get("mean_memory_conditioning_norm", float("nan"))),
            "memory_bank_enabled": 1.0,
            "memory_bank_size": float(self.bank_size),
            "memory_bank_active_slots": float(self.active_slots),
            "memory_bank_writes": float(self.writes),
            "memory_bank_appends": float(self.appends),
            "memory_bank_replacements": float(self.replacements),
            "memory_bank_duplicate_skips": float(self.duplicate_skips),
            "memory_bank_low_quality_skips": float(self.low_quality_skips),
            "memory_bank_stable_medium_writes": float(self.stable_medium_writes),
            "memory_slot_scale_bin_counts": self._format_counts(scale_counts, ("small", "medium", "large")),
            "memory_slot_conf_bin_counts": self._format_counts(conf_counts, ("medium", "high")),
            "memory_write_decision_counts": self._format_counts(
                self.decision_counts,
                (
                    "append",
                    "replace_fifo",
                    "replace_diverse",
                    "replace_balanced",
                    "replace_quality",
                    "skip_duplicate",
                    "skip_low_conf",
                    "skip_low_area",
                    "skip_unstable_medium",
                ),
            ),
            "memory_retrieval_top1_sim": float(self.last_retrieval["memory_retrieval_top1_sim"]),
            "memory_retrieval_mean_topk_sim": float(self.last_retrieval["memory_retrieval_mean_topk_sim"]),
            "memory_retrieval_top1_slot": float(self.last_retrieval["memory_retrieval_top1_slot"]),
            "memory_retrieval_entropy": float(self.last_retrieval["memory_retrieval_entropy"]),
            "memory_slot_mean_pairwise_sim": float(mean_pair),
            "memory_slot_max_pairwise_sim": float(max_pair),
            "memory_debug_images_saved": float(self.debug_images_saved),
        }

    def _save_write_debug(
        self,
        frame_idx: int,
        slot_idx: int,
        det: Top1Det,
        area_frac: float,
        reason: str,
        img_rgb: Optional[Image.Image],
    ) -> None:
        if not self.debug_save or img_rgb is None or self.debug_images_saved >= self.debug_max_images:
            return
        out = img_rgb.copy()
        draw = ImageDraw.Draw(out)
        font = try_load_font(16)
        x1, y1, x2, y2 = det.xyxy
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 120), width=4)
        draw.rectangle([0, 0, out.size[0], 52], fill=(0, 0, 0))
        lines = [
            f"frame={frame_idx} slot={slot_idx} {reason} conf={det.conf:.3f} area={area_frac:.4f}",
            f"policy={self.write_policy} active={self.active_slots}/{self.bank_size} norm={float(self.slots[slot_idx].float().norm().cpu()):.3f}",
        ]
        for i, text in enumerate(lines):
            draw.text((8, 6 + i * 22), text, fill=(255, 255, 255), font=font)
        out_path = self.write_dir / f"write_{frame_idx:06d}_slot{slot_idx:02d}.jpg"
        out.save(out_path, quality=92)
        self.debug_images_saved += 1

    def _save_skipped_write_debug(
        self,
        frame_idx: int,
        det: Top1Det,
        reason: str,
        img_rgb: Optional[Image.Image],
        detail: str = "",
    ) -> None:
        if not self.debug_save or img_rgb is None or self.debug_images_saved >= self.debug_max_images:
            return
        out = img_rgb.copy()
        draw = ImageDraw.Draw(out)
        font = try_load_font(16)
        x1, y1, x2, y2 = det.xyxy
        draw.rectangle([x1, y1, x2, y2], outline=(255, 170, 0), width=4)
        draw.rectangle([0, 0, out.size[0], 74], fill=(0, 0, 0))
        lines = [
            f"frame={frame_idx} skipped={reason}",
            f"conf={float(det.conf):.3f} policy={self.write_policy}",
            detail[:90],
        ]
        for i, text in enumerate(lines):
            if text:
                draw.text((8, 6 + i * 22), text, fill=(255, 255, 255), font=font)
        out_path = self.skipped_write_dir / f"skip_{frame_idx:06d}_{reason}.jpg"
        out.save(out_path, quality=92)
        self.debug_images_saved += 1

    def _slot_crop(self, slot_idx: int, fallback_size: Tuple[int, int] = (220, 180)) -> Image.Image:
        meta = self.slot_metas[int(slot_idx)]
        if meta is None or not meta.path:
            return Image.new("RGB", fallback_size, color=(32, 32, 32))
        path = Path(meta.path)
        if not path.exists():
            return Image.new("RGB", fallback_size, color=(32, 32, 32))
        with Image.open(path) as im:
            img = im.convert("RGB")
        x1, y1, x2, y2 = meta.box_xyxy
        crop = img.crop(
            (
                max(0, int(math.floor(x1))),
                max(0, int(math.floor(y1))),
                min(img.size[0], int(math.ceil(x2))),
                min(img.size[1], int(math.ceil(y2))),
            )
        )
        if crop.size[0] <= 0 or crop.size[1] <= 0:
            crop = img
        crop.thumbnail(fallback_size)
        out = Image.new("RGB", fallback_size, color=(18, 18, 18))
        out.paste(crop, ((fallback_size[0] - crop.size[0]) // 2, (fallback_size[1] - crop.size[1]) // 2))
        return out

    def _save_retrieval_debug(
        self,
        frame_idx: int,
        det: Optional[Top1Det],
        img_rgb: Optional[Image.Image],
        top_idxs: Sequence[int],
        top_vals: Sequence[float],
    ) -> None:
        if not self.debug_save or img_rgb is None or self.debug_images_saved >= self.debug_max_images:
            return
        panel_w, panel_h = 260, 220
        n_slots = min(len(top_idxs), self.debug_topk_examples)
        out = Image.new("RGB", (panel_w * (1 + n_slots), panel_h), color=(0, 0, 0))
        current = img_rgb.copy()
        if det is not None:
            draw = ImageDraw.Draw(current)
            draw.rectangle(det.xyxy, outline=(0, 255, 120), width=4)
        current.thumbnail((panel_w, panel_h - 34))
        current_panel = Image.new("RGB", (panel_w, panel_h), color=(18, 18, 18))
        current_panel.paste(current, ((panel_w - current.size[0]) // 2, 28 + (panel_h - 34 - current.size[1]) // 2))
        draw = ImageDraw.Draw(current_panel)
        font = try_load_font(14)
        draw.text((8, 6), f"current frame={frame_idx}", fill=(255, 255, 255), font=font)
        out.paste(current_panel, (0, 0))
        for rank, (slot_idx, sim) in enumerate(zip(top_idxs[:n_slots], top_vals[:n_slots]), start=1):
            panel = self._slot_crop(int(slot_idx), fallback_size=(panel_w, panel_h - 34))
            slot_panel = Image.new("RGB", (panel_w, panel_h), color=(18, 18, 18))
            slot_panel.paste(panel, (0, 28))
            draw = ImageDraw.Draw(slot_panel)
            draw.text((8, 6), f"top{rank} slot={slot_idx} sim={sim:.3f}", fill=(255, 255, 255), font=font)
            out.paste(slot_panel, (rank * panel_w, 0))
        out_path = self.retrieval_dir / f"retrieval_{frame_idx:06d}.jpg"
        out.save(out_path, quality=92)
        self.debug_images_saved += 1

    def finalize_outputs(self) -> Optional[Path]:
        if not self.debug_save:
            return None
        self.debug_root.mkdir(parents=True, exist_ok=True)
        self._write_memory_summary_csv()
        self._save_slots_contact_sheet()
        self._save_similarity_heatmap()
        self._save_timeline_plot()
        self._save_required_series_plots()
        return self.debug_root

    def _write_memory_summary_csv(self) -> None:
        out_path = self.debug_root / "memory_summary.csv"
        fieldnames = [
            "frame",
            "memory_bank_active_slots",
            "memory_bank_writes",
            "memory_bank_appends",
            "memory_bank_replacements",
            "memory_bank_duplicate_skips",
            "memory_bank_low_quality_skips",
            "memory_bank_stable_medium_writes",
            "memory_slot_scale_small",
            "memory_slot_scale_medium",
            "memory_slot_scale_large",
            "memory_slot_conf_medium",
            "memory_slot_conf_high",
            "memory_retrieval_top1_sim",
            "memory_retrieval_mean_topk_sim",
            "memory_retrieval_top1_slot",
            "memory_retrieval_entropy",
            "memory_slot_mean_pairwise_sim",
            "memory_slot_max_pairwise_sim",
            "memory_bank_context_norm",
            "memory_debug_images_saved",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.timeline_rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def _save_slots_contact_sheet(self) -> None:
        out_path = self.debug_root / "memory_slots_final.png"
        cols = 4
        rows = max(1, int(math.ceil(max(1, self.bank_size) / cols)))
        panel_w, panel_h = 260, 230
        sheet = Image.new("RGB", (cols * panel_w, rows * panel_h), color=(0, 0, 0))
        font = try_load_font(14)
        pair_sims = None
        if self.active_slots > 1:
            slot_mat = torch.nn.functional.normalize(self.slots[: self.active_slots].float(), dim=1, eps=1e-6)
            pair_sims = (slot_mat @ slot_mat.t()).detach().cpu().numpy()
        for idx in range(self.bank_size):
            panel = Image.new("RGB", (panel_w, panel_h), color=(18, 18, 18))
            draw = ImageDraw.Draw(panel)
            meta = self.slot_metas[idx] if idx < len(self.slot_metas) else None
            if meta is not None:
                crop = self._slot_crop(idx, fallback_size=(panel_w, panel_h - 70))
                panel.paste(crop, (0, 42))
                nearest = float("nan")
                if pair_sims is not None and idx < self.active_slots:
                    vals = np.delete(pair_sims[idx], idx)
                    nearest = float(np.max(vals)) if vals.size else float("nan")
                draw.text((8, 6), f"slot={idx} f={meta.frame_idx} conf={meta.teacher_conf:.3f}", fill=(255, 255, 255), font=font)
                draw.text((8, 24), f"{meta.scale_bin}/{meta.conf_bin} {meta.write_reason} near={nearest:.3f}", fill=(255, 255, 255), font=font)
                draw.text((8, panel_h - 22), f"age={self.last_update_frame - meta.frame_idx} writes={meta.write_count} src={meta.write_source}", fill=(255, 255, 255), font=font)
            else:
                draw.text((8, 8), f"slot={idx} empty", fill=(180, 180, 180), font=font)
            sheet.paste(panel, ((idx % cols) * panel_w, (idx // cols) * panel_h))
        sheet.save(out_path)
        self.debug_images_saved += 1

    def _save_similarity_heatmap(self) -> None:
        out_path = self.debug_root / "memory_similarity_heatmap.png"
        fig = plt.figure(figsize=(6, 5))
        if self.active_slots <= 0:
            plt.text(0.5, 0.5, "no active slots", ha="center", va="center", transform=plt.gca().transAxes)
            plt.axis("off")
        else:
            slot_mat = torch.nn.functional.normalize(self.slots[: self.active_slots].float(), dim=1, eps=1e-6)
            sims = (slot_mat @ slot_mat.t()).detach().cpu().numpy()
            plt.imshow(sims, vmin=-1.0, vmax=1.0, cmap="viridis")
            plt.colorbar(label="cosine")
            plt.title("Memory Slot Similarity")
            plt.xlabel("slot")
            plt.ylabel("slot")
        plt.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        self.debug_images_saved += 1

    def _timeline_array(self, key: str) -> np.ndarray:
        return np.array([float(row.get(key, float("nan"))) for row in self.timeline_rows], dtype=np.float32)

    def _save_timeline_plot(self) -> None:
        out_path = self.debug_root / "memory_timeline.png"
        frames = self._timeline_array("frame")
        save_plot_lines(
            frames,
            [
                self._timeline_array("memory_bank_writes"),
                self._timeline_array("memory_bank_active_slots"),
                self._timeline_array("memory_bank_duplicate_skips"),
                self._timeline_array("memory_bank_replacements"),
                self._timeline_array("memory_retrieval_top1_sim"),
                self._timeline_array("memory_retrieval_mean_topk_sim"),
                self._timeline_array("memory_retrieval_top1_slot"),
                self._timeline_array("memory_bank_context_norm"),
            ],
            ["writes", "active_slots", "dup_skips", "replacements", "top1_sim", "mean_topk_sim", "top1_slot", "context_norm"],
            "Memory Timeline",
            "frame",
            "value",
            out_path,
        )

    def _save_required_series_plots(self) -> None:
        frames = self._timeline_array("frame")
        for root in (self.debug_root, self.out_dir):
            save_plot_lines(
                frames,
                [self._timeline_array("memory_bank_active_slots")],
                ["active_slots"],
                "Memory Bank Active Slots",
                "frame",
                "slots",
                root / "plot_memory_bank_active_slots.png",
            )
            save_plot_lines(
                frames,
                [
                    self._timeline_array("memory_retrieval_top1_sim"),
                    self._timeline_array("memory_retrieval_mean_topk_sim"),
                ],
                ["top1_sim", "mean_topk_sim"],
                "Memory Retrieval Similarity",
                "frame",
                "cosine",
                root / "plot_memory_retrieval_similarity.png",
            )
            save_plot_lines(
                frames,
                [self._timeline_array("memory_retrieval_entropy")],
                ["retrieval_entropy"],
                "Memory Retrieval Entropy",
                "frame",
                "entropy",
                root / "plot_memory_retrieval_entropy.png",
            )
            save_plot_lines(
                frames,
                [
                    self._timeline_array("memory_bank_appends"),
                    self._timeline_array("memory_bank_replacements"),
                    self._timeline_array("memory_bank_duplicate_skips"),
                    self._timeline_array("memory_bank_low_quality_skips"),
                    self._timeline_array("memory_bank_stable_medium_writes"),
                ],
                ["appends", "replacements", "duplicate_skips", "low_quality_skips", "stable_medium"],
                "Memory Write Decisions",
                "frame",
                "count",
                root / "plot_memory_bank_writes.png",
            )


class SourceMemoryAnchorState:
    def __init__(
        self,
        model: nn.Module,
        source_layer: nn.Module,
        source_memory_path: str,
        weight: float,
        loss_type: str,
        temp: float,
        topk_pos: int,
        neg_k: int,
        save_debug: bool,
        out_dir: Path,
        max_debug_images: int,
    ) -> None:
        self.model = model
        self.source_layer = source_layer
        self.source_memory_path = str(source_memory_path)
        self.weight = max(0.0, float(weight))
        self.loss_type = str(loss_type)
        if self.loss_type not in {"topk_sim", "infonce"}:
            raise RuntimeError(f"Unsupported source memory loss type: {self.loss_type}")
        self.temp = max(1e-6, float(temp))
        self.topk_pos = max(1, int(topk_pos))
        self.neg_k = max(1, int(neg_k))
        self.save_debug = bool(save_debug)
        self.debug_max_images = max(0, int(max_debug_images))
        self.debug_images_saved = 0
        self.debug_dir = out_dir / "memory_debug" / "source_memory_retrieval_examples"
        if self.save_debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        payload = torch.load(self.source_memory_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Source memory file must contain a dict, got {type(payload).__name__}.")
        vectors = payload.get("vectors", payload.get("embeddings"))
        if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2:
            raise RuntimeError("Source memory file must contain a 2D tensor under 'vectors' or 'embeddings'.")
        self.source_vectors = torch.nn.functional.normalize(vectors.float(), dim=1, eps=1e-6)
        self.source_metas = payload.get("metas", payload.get("metadata", []))
        if not isinstance(self.source_metas, list):
            self.source_metas = []
        self._feature: Optional[torch.Tensor] = None
        self._hook = self.source_layer.register_forward_hook(self._capture_feature)
        self.timeline_rows: List[Dict[str, float]] = []
        self.losses: List[float] = []
        self.valid_counts: List[int] = []
        self.pos_sims: List[float] = []
        self.neg_sims: List[float] = []
        self.margins: List[float] = []
        self.skipped_updates = 0

    def close(self) -> None:
        self._hook.remove()

    def _capture_feature(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            self._feature = output

    def _projection_adapter(self):
        source_adapter = first_memory_source_adapter(self.model)
        if source_adapter is None:
            raise RuntimeError("Source memory loss requires a memory source adapter projector.")
        return source_adapter

    @staticmethod
    def _pool_feature_for_norm_xywh(feature: torch.Tensor, box_xywh: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 3:
            raise RuntimeError(f"Expected CHW feature tensor, got {tuple(feature.shape)}.")
        _channels, feat_h, feat_w = feature.shape
        cx, cy, bw, bh = [float(v) for v in box_xywh.detach().float().cpu().tolist()]
        x1 = (cx - 0.5 * bw) * feat_w
        x2 = (cx + 0.5 * bw) * feat_w
        y1 = (cy - 0.5 * bh) * feat_h
        y2 = (cy + 0.5 * bh) * feat_h
        ix1 = max(0, min(int(math.floor(x1)), feat_w - 1))
        ix2 = max(ix1 + 1, min(int(math.ceil(x2)), feat_w))
        iy1 = max(0, min(int(math.floor(y1)), feat_h - 1))
        iy2 = max(iy1 + 1, min(int(math.ceil(y2)), feat_h))
        return feature[:, iy1:iy2, ix1:ix2].float().mean(dim=(1, 2))

    def compute(
        self,
        batch: Dict[str, torch.Tensor],
        target_entries: Sequence[ReplayEntry],
        frame_idx: int,
    ) -> Tuple[Optional[torch.Tensor], SourceMemoryStats]:
        feature = self._feature
        active = self.source_vectors.to(device=feature.device if isinstance(feature, torch.Tensor) else "cpu")
        if self.weight <= 0.0 or int(active.shape[0]) < max(1, self.topk_pos):
            self.skipped_updates += 1
            return None, self._record(frame_idx, SourceMemoryStats(skipped=1))
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
            self.skipped_updates += 1
            return None, self._record(frame_idx, SourceMemoryStats(skipped=1))
        bboxes = batch.get("bboxes")
        batch_idx = batch.get("batch_idx")
        if not isinstance(bboxes, torch.Tensor) or not isinstance(batch_idx, torch.Tensor):
            self.skipped_updates += 1
            return None, self._record(frame_idx, SourceMemoryStats(skipped=1))

        adapter = self._projection_adapter()
        active = active.to(device=feature.device, dtype=feature.dtype).detach()
        active = torch.nn.functional.normalize(active.float(), dim=1, eps=1e-6)
        if active.shape[0] < 2:
            self.skipped_updates += 1
            return None, self._record(frame_idx, SourceMemoryStats(skipped=1))

        losses: List[torch.Tensor] = []
        mean_pos_vals: List[torch.Tensor] = []
        mean_neg_vals: List[torch.Tensor] = []
        debug_payload: Optional[Tuple[ReplayEntry, List[int], List[float], List[int], List[float]]] = None
        for row_idx in range(int(bboxes.shape[0])):
            img_idx = int(batch_idx[row_idx].detach().cpu())
            if img_idx < 0 or img_idx >= int(feature.shape[0]) or img_idx >= len(target_entries):
                continue
            pooled = self._pool_feature_for_norm_xywh(feature[img_idx], bboxes[row_idx])
            q = adapter.project_memory_feature(pooled)
            q = torch.nn.functional.normalize(q.float(), dim=0, eps=1e-6)
            sims = active @ q
            pos_k = min(self.topk_pos, int(sims.numel()) - 1)
            neg_k = min(self.neg_k, max(1, int(sims.numel()) - pos_k))
            if pos_k <= 0 or neg_k <= 0:
                continue
            pos_vals, pos_idxs = torch.topk(sims, k=pos_k, largest=True)
            neg_vals, neg_idxs = torch.topk(sims, k=neg_k, largest=False)
            if self.loss_type == "topk_sim":
                entry_loss = 1.0 - pos_vals.mean()
            else:
                pos_logits = pos_vals / self.temp
                all_logits = torch.cat([pos_vals, neg_vals], dim=0) / self.temp
                entry_loss = -torch.logsumexp(pos_logits, dim=0) + torch.logsumexp(all_logits, dim=0)
            if torch.isfinite(entry_loss):
                losses.append(entry_loss)
                mean_pos_vals.append(pos_vals.mean())
                mean_neg_vals.append(neg_vals.mean())
                if debug_payload is None:
                    debug_payload = (
                        target_entries[img_idx],
                        [int(v) for v in pos_idxs.detach().cpu().tolist()],
                        [float(v) for v in pos_vals.detach().cpu().tolist()],
                        [int(v) for v in neg_idxs.detach().cpu().tolist()],
                        [float(v) for v in neg_vals.detach().cpu().tolist()],
                    )

        if not losses:
            self.skipped_updates += 1
            return None, self._record(frame_idx, SourceMemoryStats(skipped=1))

        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            self.skipped_updates += 1
            return None, self._record(frame_idx, SourceMemoryStats(skipped=1))
        pos_mean_t = torch.stack(mean_pos_vals).mean()
        neg_mean_t = torch.stack(mean_neg_vals).mean()
        stats = SourceMemoryStats(
            loss=float(loss.detach().cpu()),
            valid_entries=len(losses),
            skipped=0,
            mean_pos_sim=float(pos_mean_t.detach().cpu()),
            mean_neg_sim=float(neg_mean_t.detach().cpu()),
            margin=float((pos_mean_t - neg_mean_t).detach().cpu()),
        )
        if debug_payload is not None:
            self._save_debug_example(frame_idx, *debug_payload)
        return loss, self._record(frame_idx, stats)

    def _record(self, frame_idx: int, stats: SourceMemoryStats) -> SourceMemoryStats:
        self.timeline_rows.append(
            {
                "frame": float(frame_idx),
                "source_memory_loss": float(stats.loss),
                "source_memory_valid_entries": float(stats.valid_entries),
                "source_memory_skipped": float(stats.skipped),
                "source_memory_mean_pos_sim": float(stats.mean_pos_sim),
                "source_memory_mean_neg_sim": float(stats.mean_neg_sim),
                "source_memory_margin": float(stats.margin),
            }
        )
        if math.isfinite(float(stats.loss)):
            self.losses.append(float(stats.loss))
        if int(stats.valid_entries) > 0:
            self.valid_counts.append(int(stats.valid_entries))
        if math.isfinite(float(stats.mean_pos_sim)):
            self.pos_sims.append(float(stats.mean_pos_sim))
        if math.isfinite(float(stats.mean_neg_sim)):
            self.neg_sims.append(float(stats.mean_neg_sim))
        if math.isfinite(float(stats.margin)):
            self.margins.append(float(stats.margin))
        return stats

    def stats(self) -> Dict[str, float]:
        return {
            "source_memory_loss": float(self.losses[-1]) if self.losses else float("nan"),
            "mean_source_memory_loss_updates": mean_or_nan(self.losses),
            "source_memory_valid_entries": float(sum(self.valid_counts)),
            "source_memory_skipped_updates": float(self.skipped_updates),
            "source_memory_mean_pos_sim": mean_or_nan(self.pos_sims),
            "source_memory_mean_neg_sim": mean_or_nan(self.neg_sims),
            "source_memory_margin": mean_or_nan(self.margins),
            "source_memory_debug_images_saved": float(self.debug_images_saved),
        }

    def finalize_outputs(self, out_dir: Path) -> None:
        frames = np.array([float(row["frame"]) for row in self.timeline_rows], dtype=np.float32)
        loss_vals = np.array([float(row["source_memory_loss"]) for row in self.timeline_rows], dtype=np.float32)
        pos_vals = np.array([float(row["source_memory_mean_pos_sim"]) for row in self.timeline_rows], dtype=np.float32)
        neg_vals = np.array([float(row["source_memory_mean_neg_sim"]) for row in self.timeline_rows], dtype=np.float32)
        margin_vals = np.array([float(row["source_memory_margin"]) for row in self.timeline_rows], dtype=np.float32)
        for root in (out_dir, out_dir / "memory_debug"):
            root.mkdir(parents=True, exist_ok=True)
            save_plot_lines(
                frames,
                [loss_vals],
                ["source_memory_loss"],
                "Source Memory Anchor Loss",
                "frame",
                "loss",
                root / "plot_source_memory_loss.png",
            )
            save_plot_lines(
                frames,
                [pos_vals, neg_vals],
                ["pos_sim", "neg_sim"],
                "Source Memory Positive/Negative Similarity",
                "frame",
                "cosine",
                root / "plot_source_memory_pos_neg_sim.png",
            )
            save_plot_lines(
                frames,
                [margin_vals],
                ["pos_minus_neg"],
                "Source Memory Margin",
                "frame",
                "cosine",
                root / "plot_source_memory_margin.png",
            )

    def _save_debug_example(
        self,
        frame_idx: int,
        entry: ReplayEntry,
        pos_idxs: Sequence[int],
        pos_vals: Sequence[float],
        neg_idxs: Sequence[int],
        neg_vals: Sequence[float],
    ) -> None:
        if not self.save_debug or self.debug_images_saved >= self.debug_max_images:
            return
        path = Path(entry.path)
        if not path.exists():
            return
        with Image.open(path) as im:
            img = im.convert("RGB")
        x1, y1, x2, y2 = entry.pseudo_box
        current = img.crop(
            (
                max(0, int(math.floor(x1))),
                max(0, int(math.floor(y1))),
                min(img.size[0], int(math.ceil(x2))),
                min(img.size[1], int(math.ceil(y2))),
            )
        )
        panel_w, panel_h = 230, 205
        n_slots = len(pos_idxs) + len(neg_idxs)
        out = Image.new("RGB", (panel_w * (1 + n_slots), panel_h), color=(0, 0, 0))
        font = try_load_font(13)

        def make_panel(crop: Image.Image, title: str, fill: Tuple[int, int, int]) -> Image.Image:
            crop = crop.copy()
            crop.thumbnail((panel_w, panel_h - 34))
            panel = Image.new("RGB", (panel_w, panel_h), color=(18, 18, 18))
            panel.paste(crop, ((panel_w - crop.size[0]) // 2, 30 + (panel_h - 34 - crop.size[1]) // 2))
            draw = ImageDraw.Draw(panel)
            draw.rectangle([0, 0, panel_w - 1, panel_h - 1], outline=fill, width=3)
            draw.text((8, 7), title[:34], fill=(255, 255, 255), font=font)
            return panel

        out.paste(make_panel(current, f"query frame={entry.frame_idx}", (255, 255, 255)), (0, 0))
        col = 1
        for rank, (slot_idx, sim) in enumerate(zip(pos_idxs, pos_vals), start=1):
            out.paste(
                make_panel(self._source_slot_crop(int(slot_idx)), f"pos{rank} slot={slot_idx} sim={sim:.3f}", (0, 220, 120)),
                (col * panel_w, 0),
            )
            col += 1
        for rank, (slot_idx, sim) in enumerate(zip(neg_idxs, neg_vals), start=1):
            out.paste(
                make_panel(self._source_slot_crop(int(slot_idx)), f"neg{rank} slot={slot_idx} sim={sim:.3f}", (255, 120, 0)),
                (col * panel_w, 0),
            )
            col += 1
        out_path = self.debug_dir / f"source_memory_{frame_idx:06d}_entry{entry.frame_idx:06d}.jpg"
        out.save(out_path, quality=92)
        self.debug_images_saved += 1

    def _source_slot_crop(self, slot_idx: int, fallback_size: Tuple[int, int] = (220, 180)) -> Image.Image:
        if slot_idx < 0 or slot_idx >= len(self.source_metas):
            return Image.new("RGB", fallback_size, color=(32, 32, 32))
        meta = self.source_metas[int(slot_idx)]
        if not isinstance(meta, dict):
            return Image.new("RGB", fallback_size, color=(32, 32, 32))
        crop_path = str(meta.get("crop_path", ""))
        path = Path(crop_path) if crop_path else Path(str(meta.get("path", meta.get("image_path", ""))))
        if not path.exists():
            return Image.new("RGB", fallback_size, color=(32, 32, 32))
        with Image.open(path) as im:
            img = im.convert("RGB")
        if not crop_path:
            box = meta.get("box_xyxy", meta.get("xyxy"))
            if isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, x2, y2 = [float(v) for v in box]
                img = img.crop(
                    (
                        max(0, int(math.floor(x1))),
                        max(0, int(math.floor(y1))),
                        min(img.size[0], int(math.ceil(x2))),
                        min(img.size[1], int(math.ceil(y2))),
                    )
                )
        img.thumbnail(fallback_size)
        out = Image.new("RGB", fallback_size, color=(18, 18, 18))
        out.paste(img, ((fallback_size[0] - img.size[0]) // 2, (fallback_size[1] - img.size[1]) // 2))
        return out


def first_memory_source_adapter(model: nn.Module):
    for adapter in memory_adapter_modules(model):
        if getattr(adapter, "memory_projector", None) is not None:
            return adapter
    return None


def pool_feature_for_original_box(
    feature: torch.Tensor,
    box_xyxy: Tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
) -> torch.Tensor:
    if feature.ndim != 3:
        raise RuntimeError(f"Expected CHW feature tensor, got {tuple(feature.shape)}.")
    _channels, feat_h, feat_w = feature.shape
    x1, y1, x2, y2 = box_xyxy
    ow = max(1.0, float(orig_w))
    oh = max(1.0, float(orig_h))
    if ow >= oh:
        nx1 = x1 / ow
        nx2 = x2 / ow
        ny1 = y1 / ow + (1.0 - oh / ow) * 0.5
        ny2 = y2 / ow + (1.0 - oh / ow) * 0.5
    else:
        nx1 = x1 / oh + (1.0 - ow / oh) * 0.5
        nx2 = x2 / oh + (1.0 - ow / oh) * 0.5
        ny1 = y1 / oh
        ny2 = y2 / oh
    ix1 = max(0, min(int(math.floor(nx1 * feat_w)), feat_w - 1))
    ix2 = max(ix1 + 1, min(int(math.ceil(nx2 * feat_w)), feat_w))
    iy1 = max(0, min(int(math.floor(ny1 * feat_h)), feat_h - 1))
    iy2 = max(iy1 + 1, min(int(math.ceil(ny2 * feat_h)), feat_h))
    return feature[:, iy1:iy2, ix1:ix2].float().mean(dim=(1, 2))


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
) -> Tuple[Optional[Top1Det], List[Top1Det], float]:
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
    return raw_top1, candidates, latency_ms


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
) -> Tuple[float, float, float]:
    mode = str(score_mode)
    conf_term = float(candidate.conf)
    temporal_term = 0.0
    if mode == "conf_temporal":
        temporal_term, _iou, _center_shift, _area_ratio = compute_temporal_consistency_score(
            box_xyxy=candidate.xyxy,
            prev_box_xyxy=prev_reference_box,
            img_w=img_w,
            img_h=img_h,
            max_center_shift_frac=max_center_shift_frac,
            max_area_ratio=max_area_ratio,
        )
    elif mode != "conf_only":  # pragma: no cover
        raise RuntimeError(f"Unsupported teacher candidate score mode: {score_mode}")

    total = float(conf_weight) * conf_term
    if mode == "conf_temporal":
        total += float(temporal_weight) * float(temporal_term)
    return float(total), float(conf_term), float(temporal_term)


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
    min_score: float,
) -> CandidateSelectionResult:
    if not candidates:
        return CandidateSelectionResult(
            selected=None,
            num_candidates=0,
            selected_rank=0,
            selected_score=float("nan"),
            score_conf=float("nan"),
            score_temporal=float("nan"),
            reject_reason="no_candidates",
        )

    best_candidate: Optional[Top1Det] = None
    best_total = float("-inf")
    best_conf_term = float("nan")
    best_temporal_term = float("nan")

    for candidate in candidates:
        total, conf_term, temporal_term = score_teacher_candidate(
            candidate=candidate,
            prev_reference_box=prev_reference_box,
            img_w=img_w,
            img_h=img_h,
            max_center_shift_frac=max_center_shift_frac,
            max_area_ratio=max_area_ratio,
            score_mode=score_mode,
            conf_weight=conf_weight,
            temporal_weight=temporal_weight,
        )
        if float(total) < float(min_score):
            continue

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

    if best_candidate is None:
        return CandidateSelectionResult(
            selected=None,
            num_candidates=len(candidates),
            selected_rank=0,
            selected_score=float("nan"),
            score_conf=float("nan"),
            score_temporal=float("nan"),
            reject_reason="no_selectable_candidate",
        )

    return CandidateSelectionResult(
        selected=best_candidate,
        num_candidates=len(candidates),
        selected_rank=int(best_candidate.rank),
        selected_score=float(best_total),
        score_conf=float(best_conf_term),
        score_temporal=float(best_temporal_term),
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
            if "memory_context" in key:
                teacher_val.copy_(student_val)
            elif torch.is_floating_point(teacher_val):
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


def build_coverage_image_batch(
    coverage_entries: Sequence[CoverageEntry],
    imgsz: int,
    device: str,
    rng: random.Random,
) -> Dict[str, torch.Tensor]:
    if not coverage_entries:
        raise RuntimeError("build_coverage_image_batch called with empty entries")

    img_tensors: List[torch.Tensor] = []
    teacher_confs: List[float] = []
    batch_idx_vals: List[int] = []
    cls_vals: List[List[float]] = []
    box_vals: List[List[float]] = []
    for entry in coverage_entries:
        with Image.open(entry.path) as im:
            img_rgb = im.convert("RGB")
            img_w, img_h = img_rgb.size

        aug_rgb = strong_augment(img_rgb, rng)
        letterboxed, gain, pad_left, pad_top = letterbox_image(aug_rgb, imgsz)
        img_tensors.append(pil_to_model_tensor(letterboxed))
        teacher_confs.append(float(entry.teacher_conf))
        x_c, y_c, bw, bh = xyxy_original_to_norm_xywh_letterboxed(
            box_xyxy=entry.teacher_box_xyxy,
            orig_w=img_w,
            orig_h=img_h,
            size=imgsz,
            gain=gain,
            pad_left=pad_left,
            pad_top=pad_top,
        )
        batch_idx_vals.append(len(batch_idx_vals))
        cls_vals.append([float(entry.teacher_cls)])
        box_vals.append([x_c, y_c, bw, bh])

    return {
        "img": torch.cat(img_tensors, dim=0).to(device),
        "teacher_conf": torch.tensor(teacher_confs, dtype=torch.float32, device=device),
        "batch_idx": torch.tensor(batch_idx_vals, dtype=torch.long, device=device),
        "cls": torch.tensor(cls_vals, dtype=torch.float32, device=device),
        "bboxes": torch.tensor(box_vals, dtype=torch.float32, device=device),
    }


def model_num_classes(model: nn.Module) -> int:
    for module in reversed(list(model.modules())):
        if is_detect_or_segment_module(module) and hasattr(module, "nc"):
            return max(1, int(getattr(module, "nc")))
    return 1


def raw_prediction_class_logits(preds: Any, num_classes: int) -> Optional[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    if isinstance(preds, torch.Tensor):
        tensors = [preds]
    elif isinstance(preds, (list, tuple)):
        tensors = [p for p in preds if isinstance(p, torch.Tensor) and p.ndim >= 3]
        if not tensors:
            for item in preds:
                nested = raw_prediction_class_logits(item, num_classes=num_classes)
                if nested is not None:
                    return nested
    if not tensors:
        return None

    cls_chunks: List[torch.Tensor] = []
    nc = max(1, int(num_classes))
    for pred in tensors:
        if pred.ndim == 4:
            if pred.shape[1] < nc:
                continue
            cls_logits = pred[:, -nc:, :, :].reshape(pred.shape[0], nc, -1)
            cls_chunks.append(cls_logits)
        elif pred.ndim == 3:
            if pred.shape[1] >= nc:
                cls_chunks.append(pred[:, -nc:, :])
            elif pred.shape[2] >= nc:
                cls_chunks.append(pred[:, :, -nc:].transpose(1, 2))

    if not cls_chunks:
        return None
    return torch.cat(cls_chunks, dim=2)


def _grid_centers_for_hw(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / max(1, int(height))
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / max(1, int(width))
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=0)


def _split_flat_prediction_centers(num_preds: int, image_hw: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    img_h, img_w = image_hw
    centers: List[torch.Tensor] = []
    total = 0
    for stride in (8, 16, 32):
        grid_h = max(1, int(math.ceil(float(img_h) / float(stride))))
        grid_w = max(1, int(math.ceil(float(img_w) / float(stride))))
        total += grid_h * grid_w
        centers.append(_grid_centers_for_hw(grid_h, grid_w, device=device, dtype=dtype))
    if total != int(num_preds):
        return None
    return torch.cat(centers, dim=1)


def raw_prediction_class_logits_and_centers(
    preds: Any,
    num_classes: int,
    image_hw: Tuple[int, int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    tensors: List[torch.Tensor] = []
    if isinstance(preds, torch.Tensor):
        tensors = [preds]
    elif isinstance(preds, (list, tuple)):
        tensors = [p for p in preds if isinstance(p, torch.Tensor) and p.ndim >= 3]
        if not tensors:
            for item in preds:
                cls_logits, centers = raw_prediction_class_logits_and_centers(
                    item,
                    num_classes=num_classes,
                    image_hw=image_hw,
                )
                if cls_logits is not None and centers is not None:
                    return cls_logits, centers
    if not tensors:
        return None, None

    cls_chunks: List[torch.Tensor] = []
    center_chunks: List[torch.Tensor] = []
    nc = max(1, int(num_classes))
    for pred in tensors:
        if pred.ndim == 4:
            if pred.shape[1] < nc:
                continue
            cls_chunks.append(pred[:, -nc:, :, :].reshape(pred.shape[0], nc, -1))
            center_chunks.append(
                _grid_centers_for_hw(
                    height=int(pred.shape[2]),
                    width=int(pred.shape[3]),
                    device=pred.device,
                    dtype=pred.dtype,
                )
            )
        elif pred.ndim == 3:
            cls_logits: Optional[torch.Tensor] = None
            num_preds = 0
            if pred.shape[1] >= nc:
                cls_logits = pred[:, -nc:, :]
                num_preds = int(pred.shape[2])
            elif pred.shape[2] >= nc:
                cls_logits = pred[:, :, -nc:].transpose(1, 2)
                num_preds = int(pred.shape[1])
            if cls_logits is None:
                continue
            centers = _split_flat_prediction_centers(
                num_preds=num_preds,
                image_hw=image_hw,
                device=pred.device,
                dtype=pred.dtype,
            )
            if centers is None:
                continue
            cls_chunks.append(cls_logits)
            center_chunks.append(centers)

    if not cls_chunks:
        return None, None
    return torch.cat(cls_chunks, dim=2), torch.cat(center_chunks, dim=1)


def differentiable_top_conf_from_raw_preds(preds: Any, num_classes: int) -> torch.Tensor:
    cls_logits = raw_prediction_class_logits(preds, num_classes=num_classes)
    if cls_logits is None:
        raise RuntimeError("Unable to extract differentiable class logits from raw YOLO predictions.")
    return cls_logits.sigmoid().flatten(1).amax(dim=1)


def compute_coverage_aux_loss(
    student_model: nn.Module,
    coverage_imgs: torch.Tensor,
    teacher_confs: torch.Tensor,
    num_classes: int,
    margin: float,
    max_loss: float,
) -> torch.Tensor:
    preds = student_model(coverage_imgs)
    student_top_conf = differentiable_top_conf_from_raw_preds(preds, num_classes=num_classes)
    target_conf = torch.clamp(teacher_confs - float(margin), min=0.0, max=1.0)
    per_entry = torch.relu(target_conf - student_top_conf).pow(2)
    return torch.clamp(per_entry, max=max(0.0, float(max_loss))).mean()


def compute_coverage_region_consistency_loss(
    student_model: nn.Module,
    coverage_imgs: torch.Tensor,
    teacher_confs: torch.Tensor,
    teacher_boxes_xywh: torch.Tensor,
    teacher_classes: torch.Tensor,
    num_classes: int,
    margin: float,
    max_loss: float,
    region_expand: float,
    min_candidates: int,
    center_radius_frac: float,
    use_cls: bool,
) -> Tuple[torch.Tensor, CoverageLossStats]:
    preds = student_model(coverage_imgs)
    image_hw = (int(coverage_imgs.shape[-2]), int(coverage_imgs.shape[-1]))
    cls_logits, centers = raw_prediction_class_logits_and_centers(
        preds,
        num_classes=num_classes,
        image_hw=image_hw,
    )
    if cls_logits is None or centers is None:
        zero = coverage_imgs.sum() * 0.0
        return zero, CoverageLossStats(entries_skipped=int(coverage_imgs.shape[0]))

    nc = max(1, int(num_classes))
    confs = cls_logits.sigmoid()
    if bool(use_cls):
        cls_idx = teacher_classes.reshape(-1).long().clamp(min=0, max=nc - 1)
        gather_idx = cls_idx.view(-1, 1, 1).expand(-1, 1, confs.shape[2])
        pred_confs = confs.gather(1, gather_idx).squeeze(1)
    else:
        pred_confs = confs.amax(dim=1)

    boxes = teacher_boxes_xywh.to(device=coverage_imgs.device, dtype=coverage_imgs.dtype)
    cx = boxes[:, 0].clamp(0.0, 1.0)
    cy = boxes[:, 1].clamp(0.0, 1.0)
    bw = boxes[:, 2].clamp(min=0.0, max=1.0) * (1.0 + max(0.0, float(region_expand)))
    bh = boxes[:, 3].clamp(min=0.0, max=1.0) * (1.0 + max(0.0, float(region_expand)))
    x1 = (cx - 0.5 * bw).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * bh).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * bw).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * bh).clamp(0.0, 1.0)

    centers_x = centers[0].view(1, -1)
    centers_y = centers[1].view(1, -1)
    region_mask = (
        (centers_x >= x1.view(-1, 1))
        & (centers_x <= x2.view(-1, 1))
        & (centers_y >= y1.view(-1, 1))
        & (centers_y <= y2.view(-1, 1))
    )
    if float(center_radius_frac) > 0.0:
        dx = centers_x - cx.view(-1, 1)
        dy = centers_y - cy.view(-1, 1)
        radius_sq = float(center_radius_frac) * float(center_radius_frac)
        region_mask = region_mask | ((dx * dx + dy * dy) <= radius_sq)

    per_entry_losses: List[torch.Tensor] = []
    student_region_confs: List[torch.Tensor] = []
    teacher_region_confs: List[torch.Tensor] = []
    skipped = 0
    min_k = max(0, int(min_candidates))
    for batch_idx in range(pred_confs.shape[0]):
        mask = region_mask[batch_idx]
        if int(mask.sum().item()) < min_k and min_k > 0:
            dx = centers_x[0] - cx[batch_idx]
            dy = centers_y[0] - cy[batch_idx]
            nearest_k = min(min_k, pred_confs.shape[1])
            nearest_idx = torch.topk(-(dx * dx + dy * dy), k=nearest_k).indices
            mask = mask.clone()
            mask[nearest_idx] = True
        if not bool(mask.any().item()):
            skipped += 1
            continue
        student_conf = pred_confs[batch_idx][mask].amax()
        target_conf = torch.clamp(teacher_confs[batch_idx] - float(margin), min=0.0, max=1.0)
        per_entry_losses.append(torch.relu(target_conf - student_conf).pow(2))
        student_region_confs.append(student_conf.detach())
        teacher_region_confs.append(teacher_confs[batch_idx].detach())

    if not per_entry_losses:
        zero = coverage_imgs.sum() * 0.0
        return zero, CoverageLossStats(entries_skipped=skipped)

    per_entry = torch.stack(per_entry_losses)
    loss = torch.clamp(per_entry, max=max(0.0, float(max_loss))).mean()
    student_vals = torch.stack(student_region_confs).float()
    teacher_vals = torch.stack(teacher_region_confs).float()
    stats = CoverageLossStats(
        entries_skipped=int(skipped),
        mean_student_conf=float(student_vals.mean().cpu()),
        mean_teacher_conf=float(teacher_vals.mean().cpu()),
        mean_conf_gap=float((student_vals - teacher_vals).mean().cpu()),
    )
    return loss, stats


def normalize_coverage_loss_type(loss_type: str) -> str:
    aliases = {
        "pred_conf_margin": "global_pred_conf_margin",
        "teacher_student_top1": "global_pred_conf_margin",
        "box_region_conf": "weak_box_fallback",
    }
    return aliases.get(str(loss_type), str(loss_type))


def coverage_reject_reason(
    selection: CandidateSelectionResult,
    passed_base_gate: bool,
    passed_motion_gate: bool,
    passed_persistence_gate: bool,
) -> str:
    if selection.selected is None:
        return selection.reject_reason or "no_selected_candidate"
    reasons: List[str] = []
    if not passed_base_gate:
        reasons.append("base_gate")
    if not passed_motion_gate:
        reasons.append("motion_gate")
    if not passed_persistence_gate:
        reasons.append("persistence_gate")
    return "+".join(reasons) if reasons else "not_accepted"


def should_store_coverage_entry(
    candidate: Optional[Top1Det],
    img_w: int,
    img_h: int,
    accepted: bool,
    conf_min: float,
    conf_max: float,
    min_area_frac: float,
    max_area_frac: float,
    border_margin_frac: float,
) -> bool:
    if accepted or candidate is None:
        return False
    if candidate.conf < float(conf_min) or candidate.conf > float(conf_max):
        return False
    return evaluate_detection_sanity(
        top1=candidate,
        img_w=img_w,
        img_h=img_h,
        conf_thresh=float(conf_min),
        min_area_frac=float(min_area_frac),
        max_area_frac=float(max_area_frac),
        border_margin_frac=float(border_margin_frac),
    )


def run_detection_update(
    student_model: nn.Module,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    optim_params: Sequence[torch.nn.Parameter],
    batch: Dict[str, torch.Tensor],
    grad_clip: float,
    target_entries: Optional[Sequence[ReplayEntry]] = None,
    frame_idx: int = -1,
    source_memory_state: Optional[SourceMemoryAnchorState] = None,
    coverage_batch: Optional[Dict[str, torch.Tensor]] = None,
    coverage_weight: float = 0.0,
    coverage_loss_type: str = "pred_conf_margin",
    coverage_margin: float = 0.05,
    coverage_max_loss: float = 2.0,
    coverage_region_expand: float = 0.25,
    coverage_region_min_candidates: int = 1,
    coverage_region_center_radius_frac: float = 0.10,
    coverage_region_use_cls: bool = True,
    num_classes: int = 1,
    train_mode_callback: Optional[Callable[[nn.Module], None]] = None,
    update_timing: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, float, CoverageLossStats, SourceMemoryStats]:
    def _sync() -> None:
        img = batch.get("img")
        if isinstance(img, torch.Tensor) and img.is_cuda:
            torch.cuda.synchronize(img.device)

    with torch.inference_mode(False):
        with torch.enable_grad():
            student_model.train()
            if train_mode_callback is not None:
                train_mode_callback(student_model)
            optimizer.zero_grad(set_to_none=True)

            _sync()
            t0 = time.perf_counter()
            preds = student_model(batch["img"])
            _sync()
            if update_timing is not None:
                update_timing["forward_ms"] = (time.perf_counter() - t0) * 1000.0

            _sync()
            t0 = time.perf_counter()
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
            coverage_loss = det_loss.new_tensor(0.0)
            coverage_stats = CoverageLossStats()
            source_memory_loss = det_loss.new_tensor(0.0)
            source_memory_stats = SourceMemoryStats()
            if coverage_batch is not None and float(coverage_weight) > 0:
                coverage_imgs = coverage_batch.get("img")
                teacher_confs = coverage_batch.get("teacher_conf")
                if not isinstance(coverage_imgs, torch.Tensor) or not isinstance(teacher_confs, torch.Tensor):
                    raise RuntimeError("Coverage batch is missing differentiable tensors.")
                normalized_loss_type = normalize_coverage_loss_type(str(coverage_loss_type))
                if normalized_loss_type == "global_pred_conf_margin":
                    coverage_loss = compute_coverage_aux_loss(
                        student_model=student_model,
                        coverage_imgs=coverage_imgs,
                        teacher_confs=teacher_confs,
                        num_classes=int(num_classes),
                        margin=float(coverage_margin),
                        max_loss=float(coverage_max_loss),
                    )
                elif normalized_loss_type == "region_conf_margin":
                    teacher_boxes = coverage_batch.get("bboxes")
                    teacher_classes = coverage_batch.get("cls")
                    if not isinstance(teacher_boxes, torch.Tensor) or not isinstance(teacher_classes, torch.Tensor):
                        raise RuntimeError("Coverage batch is missing teacher boxes/classes for region loss.")
                    coverage_loss, coverage_stats = compute_coverage_region_consistency_loss(
                        student_model=student_model,
                        coverage_imgs=coverage_imgs,
                        teacher_confs=teacher_confs,
                        teacher_boxes_xywh=teacher_boxes,
                        teacher_classes=teacher_classes,
                        num_classes=int(num_classes),
                        margin=float(coverage_margin),
                        max_loss=float(coverage_max_loss),
                        region_expand=float(coverage_region_expand),
                        min_candidates=int(coverage_region_min_candidates),
                        center_radius_frac=float(coverage_region_center_radius_frac),
                        use_cls=bool(coverage_region_use_cls),
                    )
                elif normalized_loss_type == "weak_box_fallback":
                    fallback = criterion(student_model(coverage_imgs), coverage_batch)
                    weak_box_loss, _weak_items = unpack_loss_pair(fallback)
                    if weak_box_loss is None:
                        raise RuntimeError("Unable to obtain differentiable weak-box coverage loss.")
                    coverage_loss = weak_box_loss.sum() if weak_box_loss.ndim > 0 else weak_box_loss
                    coverage_loss = torch.clamp(coverage_loss, max=max(0.0, float(coverage_max_loss)))
                else:  # pragma: no cover
                    raise RuntimeError(f"Unsupported coverage loss type: {coverage_loss_type}")
            if (
                source_memory_state is not None
                and target_entries is not None
                and float(source_memory_state.weight) > 0.0
            ):
                contrastive_loss, source_memory_stats = source_memory_state.compute(
                    batch=batch,
                    target_entries=target_entries,
                    frame_idx=int(frame_idx),
                )
                if contrastive_loss is not None:
                    source_memory_loss = contrastive_loss
            _sync()
            if update_timing is not None:
                update_timing["loss_ms"] = (time.perf_counter() - t0) * 1000.0

            total_loss = (
                det_loss
                + float(coverage_weight) * coverage_loss
                + float(source_memory_state.weight if source_memory_state is not None else 0.0)
                * source_memory_loss
            )
            _sync()
            t0 = time.perf_counter()
            total_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(optim_params, grad_clip)
            optimizer.step()
            _sync()
            if update_timing is not None:
                update_timing["backward_step_ms"] = (time.perf_counter() - t0) * 1000.0

    det_loss_value = float(det_loss.detach().cpu())
    total_loss_value = float(total_loss.detach().cpu())
    coverage_loss_value = float(coverage_loss.detach().cpu())
    return det_loss_value, total_loss_value, coverage_loss_value, coverage_stats, source_memory_stats


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

    footer_top = max(0, out.size[1] - 48)
    draw.rectangle([0, footer_top, out.size[0], out.size[1]], fill=(0, 0, 0))
    footer = f"score={selected_score:.3f} conf={score_conf:.3f} temporal={score_temporal:.3f}"
    draw.text((8, footer_top + 8), footer, fill=(255, 255, 255), font=font)
    draw.text((8, footer_top + 28), f"mode={score_mode}", fill=(255, 255, 255), font=font)

    score_tag = f"{selected_score:.3f}".replace(".", "p")
    rank_tag = selected_rank_text
    out_path = diag_dir / f"selected_rank_{int(frame_idx):06d}_r{rank_tag}_{score_tag}.jpg"
    out.save(out_path, quality=95)
    return out_path


def make_progress_iter(total: int, disabled: bool = False):
    if disabled or tqdm is None:
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


def resolve_adaptation_mode(args: argparse.Namespace) -> str:
    if bool(getattr(args, "adapter_enable", False)):
        return "adapter"
    requested = str(getattr(args, "adaptation_mode", "head_only"))
    if requested == "adapter":
        return "adapter"
    if requested == "neck_head":
        return "neck_head"
    return str(getattr(args, "update_scope", "head_only"))


def path_size_mb(path: Optional[Path]) -> float:
    if path is None or not path.exists():
        return float("nan")
    return float(path.stat().st_size) / (1024.0 * 1024.0)


def cuda_resource_metrics(device: str) -> Dict[str, float]:
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return {
            "peak_cuda_allocated_mb": float("nan"),
            "peak_cuda_reserved_mb": float("nan"),
            "current_cuda_allocated_mb": float("nan"),
            "current_cuda_reserved_mb": float("nan"),
        }
    dev = torch.device(str(device))
    return {
        "peak_cuda_allocated_mb": float(torch.cuda.max_memory_allocated(dev) / (1024.0 * 1024.0)),
        "peak_cuda_reserved_mb": float(torch.cuda.max_memory_reserved(dev) / (1024.0 * 1024.0)),
        "current_cuda_allocated_mb": float(torch.cuda.memory_allocated(dev) / (1024.0 * 1024.0)),
        "current_cuda_reserved_mb": float(torch.cuda.memory_reserved(dev) / (1024.0 * 1024.0)),
    }


def synchronize_cuda_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(str(device)))


def adapter_specs_text(adapter_specs: Sequence[AdapterSpec]) -> List[str]:
    return [
        (
            f"{spec.layer_idx}:ResidualConvAdapter("
            f"C={spec.in_channels}, hidden={spec.hidden_channels}, params={spec.param_count}, scale={spec.scale:g})"
        )
        for spec in adapter_specs
    ]


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
    adapter_layers: List[int] = []
    adapter_specs_meta = getattr(student_model, "odad_adapter_specs", [])
    try:
        _export_core, export_layers = unwrap_core_and_layers(export_model)
        adapter_layers = [
            int(idx)
            for idx, module in enumerate(export_layers)
            if isinstance(module, AdaptedLayer)
        ]
    except Exception:
        adapter_layers = []
    ckpt["odad_adaptation"] = {
        "checkpoint_type": str(checkpoint_type),
        "frame_idx": None if frame_idx is None else int(frame_idx),
        "adaptation_mode": str(getattr(student_model, "odad_adaptation_mode", "head_only")),
        "adapter_layers": adapter_layers,
        "adapter_param_count": int(adapter_param_count(export_model)),
        "adapter_train_detect_head": int(bool(getattr(student_model, "odad_adapter_train_detect_head", False))),
        "adapter_specs": adapter_specs_meta,
        "memory_adapter_enabled": int(bool(getattr(student_model, "odad_memory_adapter_enabled", False))),
        "memory_adapter_dim": int(getattr(student_model, "odad_memory_adapter_dim", 0)),
        "memory_adapter_source_layer": int(getattr(student_model, "odad_memory_adapter_source_layer", -1)),
        "memory_adapter_conditioning": str(getattr(student_model, "odad_memory_adapter_conditioning", "n/a")),
        "memory_adapter_param_count": int(memory_adapter_param_count(export_model)),
        "memory_adapter_bank_enabled": int(bool(getattr(student_model, "odad_memory_adapter_bank_enabled", False))),
        "memory_adapter_bank_size": int(getattr(student_model, "odad_memory_adapter_bank_size", 0)),
        "memory_adapter_topk": int(getattr(student_model, "odad_memory_adapter_topk", 0)),
        "memory_adapter_slot_dim": int(getattr(student_model, "odad_memory_adapter_slot_dim", 0)),
        "source_memory_enabled": int(bool(getattr(student_model, "odad_source_memory_enabled", False))),
        "source_memory_weight": float(getattr(student_model, "odad_source_memory_weight", 0.0)),
        "source_memory_temp": float(getattr(student_model, "odad_source_memory_temp", 0.0)),
        "source_memory_layer": int(getattr(student_model, "odad_source_memory_layer", -1)),
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
        description="Clean online teacher-student ODAD with top-k confidence+temporal selection."
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
        "--adaptation-mode",
        type=str,
        default="head_only",
        choices=["head_only", "neck_head", "adapter"],
        help="High-level adaptation mode. head_only/neck_head preserve current behavior; adapter enables lightweight residual adapters.",
    )
    parser.add_argument(
        "--adapter-enable",
        action="store_true",
        help="Enable lightweight residual adapters. Equivalent to --adaptation-mode adapter.",
    )
    parser.add_argument(
        "--adapter-layers",
        type=str,
        default="21",
        help="Comma-separated YOLO layer indices where residual adapters are inserted.",
    )
    parser.add_argument(
        "--adapter-reduction",
        type=int,
        default=8,
        help="Bottleneck reduction ratio for adapter hidden channels.",
    )
    parser.add_argument(
        "--adapter-min-channels",
        type=int,
        default=8,
        help="Minimum hidden channels in adapter bottleneck.",
    )
    parser.add_argument(
        "--adapter-scale",
        type=float,
        default=1.0,
        help="Residual adapter scale.",
    )
    parser.add_argument(
        "--adapter-train-detect-head",
        action="store_true",
        help="If enabled, train adapters plus Detect head. Default false for pure adapter-only updates.",
    )
    parser.add_argument(
        "--adapter-save-debug",
        action="store_true",
        help="Log adapter parameter norms and update deltas.",
    )
    parser.add_argument(
        "--memory-adapter-enable",
        action="store_true",
        help="Enable object-memory-conditioned adapters.",
    )
    parser.add_argument(
        "--memory-adapter-bank-enable",
        action="store_true",
        default=False,
        help="Enable multi-slot object memory bank for memory-conditioned adapters.",
    )
    parser.add_argument(
        "--memory-adapter-bank-size",
        type=int,
        default=16,
        help="Number of memory slots.",
    )
    parser.add_argument(
        "--memory-adapter-topk",
        type=int,
        default=2,
        help="Number of memory slots retrieved per read.",
    )
    parser.add_argument(
        "--memory-adapter-slot-dim",
        type=int,
        default=32,
        help="Dimension of each memory slot.",
    )
    parser.add_argument(
        "--memory-adapter-query-mode",
        type=str,
        choices=["global_gap", "teacher_roi", "student_roi"],
        default="global_gap",
        help="How to compute current query embedding for memory retrieval.",
    )
    parser.add_argument(
        "--memory-adapter-write-policy",
        type=str,
        choices=["fifo", "diverse_fifo", "diversity_reservoir", "scale_conf_balanced", "hard_example_aware"],
        default="diverse_fifo",
        help="Memory slot replacement strategy.",
    )
    parser.add_argument(
        "--memory-adapter-diversity-thresh",
        type=float,
        default=0.85,
        help="Similarity threshold for considering a new memory candidate distinct enough to write.",
    )
    parser.add_argument(
        "--memory-adapter-duplicate-thresh",
        type=float,
        default=0.95,
        help="If a new embedding is above this similarity to an existing slot, treat as duplicate unless quality is much higher.",
    )
    parser.add_argument(
        "--memory-adapter-quality-margin",
        type=float,
        default=0.05,
        help="Minimum teacher confidence improvement needed to replace a near-duplicate slot.",
    )
    parser.add_argument(
        "--memory-adapter-balance-scale-bins",
        action="store_true",
        default=False,
        help="Encourage memory slots across small/medium/large target area bins.",
    )
    parser.add_argument(
        "--memory-adapter-balance-conf-bins",
        action="store_true",
        default=False,
        help="Encourage memory slots across medium/high confidence bins.",
    )
    parser.add_argument(
        "--memory-adapter-stable-medium-write",
        action="store_true",
        default=False,
        help="Allow writes from temporally stable medium-confidence candidates, not hard pseudo-label training.",
    )
    parser.add_argument(
        "--memory-adapter-stable-conf-min",
        type=float,
        default=0.60,
        help="Minimum confidence for stable medium-confidence memory writes.",
    )
    parser.add_argument(
        "--memory-adapter-stable-iou-min",
        type=float,
        default=0.55,
        help="Minimum temporal IoU/persistence quality for stable medium-confidence memory writes.",
    )
    parser.add_argument(
        "--memory-adapter-retrieval-temp",
        type=float,
        default=0.10,
        help="Temperature for softmax weighting of retrieved memory slots.",
    )
    parser.add_argument(
        "--memory-adapter-dim",
        type=int,
        default=32,
        help="Dimension of compact object memory vector.",
    )
    parser.add_argument(
        "--memory-adapter-source-layer",
        type=int,
        default=21,
        help="Feature layer used to extract object memory embeddings.",
    )
    parser.add_argument(
        "--memory-adapter-ema",
        type=float,
        default=0.95,
        help="EMA decay for object memory updates.",
    )
    parser.add_argument(
        "--memory-adapter-min-conf",
        type=float,
        default=0.80,
        help="Minimum accepted teacher confidence required to update object memory.",
    )
    parser.add_argument(
        "--memory-adapter-min-area-frac",
        type=float,
        default=0.001,
        help="Minimum accepted box area fraction for memory update.",
    )
    parser.add_argument(
        "--memory-adapter-update-on-accepted-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only update object memory from accepted pseudo-labels.",
    )
    parser.add_argument(
        "--memory-adapter-conditioning",
        type=str,
        choices=["film", "concat"],
        default="film",
        help="Conditioning mechanism for memory adapter.",
    )
    parser.add_argument(
        "--memory-adapter-disable-conditioning-if-supported",
        action="store_true",
        default=False,
        help="Keep memory writes/projection active but skip memory reads into adapter conditioning.",
    )
    parser.add_argument(
        "--memory-adapter-save-debug",
        action="store_true",
        default=False,
        help="Save compact memory diagnostics, not full tensors.",
    )
    parser.add_argument(
        "--memory-debug-save",
        action="store_true",
        default=False,
        help="Save visual memory diagnostics.",
    )
    parser.add_argument(
        "--memory-debug-max-images",
        type=int,
        default=64,
        help="Maximum memory write/retrieval debug images to save.",
    )
    parser.add_argument(
        "--memory-debug-every",
        type=int,
        default=100,
        help="Save retrieval debug visuals every N frames when enabled.",
    )
    parser.add_argument(
        "--memory-debug-topk-examples",
        type=int,
        default=3,
        help="Number of retrieved slots to show in retrieval contact sheets.",
    )
    parser.add_argument(
        "--source-memory-enable",
        action="store_true",
        default=False,
        help="Enable source object memory anchor during online adaptation.",
    )
    parser.add_argument(
        "--source-memory-path",
        type=str,
        default="",
        help="Path to precomputed source_memory.pt.",
    )
    parser.add_argument(
        "--source-memory-weight",
        type=float,
        default=0.05,
        help="Weight for source anchor loss.",
    )
    parser.add_argument(
        "--source-memory-loss-type",
        type=str,
        choices=["topk_sim", "infonce"],
        default="topk_sim",
        help="Source anchor loss type.",
    )
    parser.add_argument(
        "--source-memory-temp",
        type=float,
        default=0.10,
        help="Temperature for InfoNCE-style source memory loss.",
    )
    parser.add_argument(
        "--source-memory-topk-pos",
        type=int,
        default=4,
        help="Number of nearest source memory slots used as positives.",
    )
    parser.add_argument(
        "--source-memory-neg-k",
        type=int,
        default=8,
        help="Number of source negatives for InfoNCE if enabled.",
    )
    parser.add_argument(
        "--source-memory-layer",
        type=int,
        default=21,
        help="Feature layer used for source-anchor student ROI pooling.",
    )
    parser.add_argument(
        "--source-memory-debug-save",
        action="store_true",
        default=False,
        help="Save compact source-memory retrieval diagnostics.",
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
        help="Legacy compatibility knob; values >1 act as updates-per-event when that flag stays at 1.",
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
        default=2,
        help="Number of teacher detections to consider per frame.",
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
        choices=["conf_only", "conf_temporal"],
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
        "--teacher-candidate-min-score",
        type=float,
        default=0.0,
        help="Optional minimum base candidate score required before selection.",
    )
    parser.add_argument(
        "--coverage-aux-enable",
        dest="coverage_aux_enable",
        action="store_true",
        help="Enable coverage-preserving consistency loss using plausible-but-unaccepted teacher candidates.",
    )
    parser.add_argument("--coverage-buffer-size", type=int, default=64, help="Capacity for coverage buffer.")
    parser.add_argument(
        "--coverage-batch-size",
        type=int,
        default=2,
        help="Number of coverage entries sampled per update event.",
    )
    parser.add_argument("--coverage-weight", type=float, default=0.10, help="Weight on coverage consistency loss.")
    parser.add_argument(
        "--coverage-candidate-conf-min",
        type=float,
        default=0.25,
        help="Minimum teacher top1/candidate confidence to store a coverage entry.",
    )
    parser.add_argument(
        "--coverage-candidate-conf-max",
        type=float,
        default=0.80,
        help="Maximum teacher confidence for coverage entries; accepted high-confidence pseudo-labels remain normal replay.",
    )
    parser.add_argument(
        "--coverage-loss-type",
        type=str,
        default="pred_conf_margin",
        choices=[
            "pred_conf_margin",
            "global_pred_conf_margin",
            "region_conf_margin",
            "weak_box_fallback",
            "box_region_conf",
            "teacher_student_top1",
        ],
        help="Coverage consistency loss type. pred_conf_margin is kept as an alias for global_pred_conf_margin.",
    )
    parser.add_argument(
        "--coverage-margin",
        type=float,
        default=0.05,
        help="Allow student confidence to be slightly below teacher before applying penalty.",
    )
    parser.add_argument(
        "--coverage-max-loss",
        type=float,
        default=2.0,
        help="Clamp coverage loss to avoid noisy weak candidates dominating updates.",
    )
    parser.add_argument(
        "--coverage-region-expand",
        type=float,
        default=0.25,
        help="Expand teacher box by this fraction before selecting student cells for region consistency.",
    )
    parser.add_argument(
        "--coverage-region-min-candidates",
        type=int,
        default=1,
        help="Minimum number of regional cells; nearest cells are used when the box region is empty.",
    )
    parser.add_argument(
        "--coverage-region-center-radius-frac",
        type=float,
        default=0.10,
        help="Optional normalized radius around teacher box center for regional candidate selection.",
    )
    parser.add_argument(
        "--coverage-region-use-cls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use teacher class when computing regional confidence if class logits are available.",
    )
    parser.add_argument(
        "--coverage-region-objectness-weight",
        type=float,
        default=1.0,
        help="Reserved for detectors with explicit objectness; YOLOv8 class confidence is used directly.",
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
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output.")
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
    num_classes = model_num_classes(student_model)
    coverage_enabled = bool(args.coverage_aux_enable)

    if bool(args.memory_adapter_bank_enable):
        args.memory_adapter_enable = True
        args.memory_adapter_dim = int(args.memory_adapter_slot_dim)
    if bool(args.source_memory_enable):
        args.memory_adapter_bank_enable = True
        args.memory_adapter_enable = True
        args.memory_adapter_dim = int(args.memory_adapter_slot_dim)
    if bool(args.memory_adapter_enable):
        args.adapter_enable = True
    adaptation_mode = resolve_adaptation_mode(args)
    args.adaptation_mode = adaptation_mode
    if adaptation_mode in {"head_only", "neck_head"}:
        args.update_scope = adaptation_mode
    if adaptation_mode == "adapter":
        args.adapter_enable = True

    neck_start_idx = resolve_neck_start_idx(core_model, int(args.neck_start_idx))
    adapter_enabled = adaptation_mode == "adapter"
    adapter_layer_indices: List[int] = []
    adapter_specs: List[AdapterSpec] = []
    adapter_trainable_params = 0
    memory_bank_enabled = bool(args.memory_adapter_bank_enable and adapter_enabled)
    memory_adapter_enabled = bool(args.memory_adapter_enable and adapter_enabled)
    memory_adapter_trainable_params = 0
    memory_state: Optional[Any] = None
    source_memory_state: Optional[SourceMemoryAnchorState] = None
    train_mode_callback: Optional[Callable[[nn.Module], None]] = None

    if adapter_enabled:
        adapter_layer_indices = parse_adapter_layers(str(args.adapter_layers))
        if memory_adapter_enabled and int(args.memory_adapter_source_layer) not in adapter_layer_indices:
            raise RuntimeError(
                "--memory-adapter-source-layer must be included in --adapter-layers for v1 "
                f"(source={int(args.memory_adapter_source_layer)}, layers={adapter_layer_indices})."
            )
        if bool(args.source_memory_enable):
            if not memory_bank_enabled:
                raise RuntimeError("--source-memory-enable requires --memory-adapter-bank-enable.")
            if int(args.source_memory_layer) != int(args.memory_adapter_source_layer):
                raise RuntimeError(
                    "MemXFormer-lite v1 reuses the memory source projector, so "
                    "--source-memory-layer must match --memory-adapter-source-layer."
                )
            if int(args.source_memory_layer) not in adapter_layer_indices:
                raise RuntimeError("--source-memory-layer must be included in --adapter-layers.")
        layers, adapter_specs = attach_residual_adapters(
            core_model=core_model,
            layers=layers,
            adapter_layers=adapter_layer_indices,
            imgsz=int(args.imgsz),
            device=str(args.device),
            reduction=int(args.adapter_reduction),
            min_channels=int(args.adapter_min_channels),
            scale=float(args.adapter_scale),
            memory_enable=memory_adapter_enabled,
            memory_dim=int(args.memory_adapter_slot_dim if memory_bank_enabled else args.memory_adapter_dim),
            memory_conditioning=str(args.memory_adapter_conditioning),
            memory_source_layer=int(args.memory_adapter_source_layer) if memory_adapter_enabled else None,
            memory_bank_size=int(args.memory_adapter_bank_size),
        )
        _, teacher_layers = unwrap_core_and_layers(teacher_model)
        teacher_layers, _teacher_specs = attach_residual_adapters(
            core_model=teacher_model,
            layers=teacher_layers,
            adapter_layers=adapter_layer_indices,
            imgsz=int(args.imgsz),
            device=str(args.device),
            reduction=int(args.adapter_reduction),
            min_channels=int(args.adapter_min_channels),
            scale=float(args.adapter_scale),
            memory_enable=memory_adapter_enabled,
            memory_dim=int(args.memory_adapter_slot_dim if memory_bank_enabled else args.memory_adapter_dim),
            memory_conditioning=str(args.memory_adapter_conditioning),
            memory_source_layer=int(args.memory_adapter_source_layer) if memory_adapter_enabled else None,
            memory_bank_size=int(args.memory_adapter_bank_size),
        )
        teacher_model.load_state_dict(student_model.state_dict(), strict=True)
        assert_matching_adapter_state(student_model, teacher_model)
        adapter_trainable_params = adapter_param_count(student_model)
        memory_adapter_trainable_params = memory_adapter_param_count(student_model)
        if memory_adapter_enabled:
            source_idx = int(args.memory_adapter_source_layer)
            if memory_bank_enabled:
                memory_state = MemoryBankAdapterState(
                    model=student_model,
                    source_layer=teacher_layers[source_idx],
                    memory_dim=int(args.memory_adapter_slot_dim),
                    bank_size=int(args.memory_adapter_bank_size),
                    topk=int(args.memory_adapter_topk),
                    query_mode=str(args.memory_adapter_query_mode),
                    write_policy=str(args.memory_adapter_write_policy),
                    diversity_thresh=float(args.memory_adapter_diversity_thresh),
                    duplicate_thresh=float(args.memory_adapter_duplicate_thresh),
                    quality_margin=float(args.memory_adapter_quality_margin),
                    balance_scale_bins=bool(args.memory_adapter_balance_scale_bins),
                    balance_conf_bins=bool(args.memory_adapter_balance_conf_bins),
                    stable_medium_write=bool(args.memory_adapter_stable_medium_write),
                    stable_conf_min=float(args.memory_adapter_stable_conf_min),
                    stable_iou_min=float(args.memory_adapter_stable_iou_min),
                    retrieval_temp=float(args.memory_adapter_retrieval_temp),
                    min_conf=float(args.memory_adapter_min_conf),
                    min_area_frac=float(args.memory_adapter_min_area_frac),
                    imgsz=int(args.imgsz),
                    device=str(args.device),
                    out_dir=out_dir,
                    debug_save=bool(args.memory_debug_save),
                    debug_max_images=int(args.memory_debug_max_images),
                    debug_every=int(args.memory_debug_every),
                    debug_topk_examples=int(args.memory_debug_topk_examples),
                )
            else:
                memory_state = MemoryAdapterState(
                    model=student_model,
                    source_layer=teacher_layers[source_idx],
                    memory_dim=int(args.memory_adapter_dim),
                    ema=float(args.memory_adapter_ema),
                    min_conf=float(args.memory_adapter_min_conf),
                    min_area_frac=float(args.memory_adapter_min_area_frac),
                    imgsz=int(args.imgsz),
                    device=str(args.device),
                )
            if bool(args.source_memory_enable):
                if not str(args.source_memory_path):
                    raise RuntimeError("--source-memory-enable requires --source-memory-path.")
                source_idx = int(args.source_memory_layer)
                source_memory_state = SourceMemoryAnchorState(
                    model=student_model,
                    source_layer=layers[source_idx],
                    source_memory_path=str(args.source_memory_path),
                    weight=float(args.source_memory_weight),
                    loss_type=str(args.source_memory_loss_type),
                    temp=float(args.source_memory_temp),
                    topk_pos=int(args.source_memory_topk_pos),
                    neg_k=int(args.source_memory_neg_k),
                    save_debug=bool(args.source_memory_debug_save),
                    out_dir=out_dir,
                    max_debug_images=int(args.memory_debug_max_images),
                )
        unfrozen_indices = [head_idx] if bool(args.adapter_train_detect_head) else []
        train_mode_callback = freeze_frozen_batchnorm_stats
    else:
        unfrozen_indices = compute_unfrozen_indices(
            update_scope=str(args.update_scope),
            neck_start_idx=neck_start_idx,
            head_idx=head_idx,
            n_layers=len(layers),
        )

    if str(args.update_scope) == "head_only" and not adapter_enabled:
        head_module_name = layers[head_idx].__class__.__name__
        if head_module_name != "Detect":
            raise RuntimeError(
                f"update_scope=head_only expects Detect head, found {head_module_name} at idx={head_idx}."
            )

    if adapter_enabled:
        expected_trainable_param_ids = apply_adapter_freeze_policy(
            model=student_model,
            layers=layers,
            head_idx=head_idx,
            train_detect_head=bool(args.adapter_train_detect_head),
        )
    else:
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
    total_param_tensors = sum(1 for _name, _param in student_model.named_parameters())
    frozen_param_tensors = sum(1 for _name, param in student_model.named_parameters() if not param.requires_grad)
    frozen_param_numel = int(sum(param.numel() for param in student_model.parameters() if not param.requires_grad))
    optimizer_param_tensors = len(optim_params)
    optimizer_param_numel = int(sum(param.numel() for param in optim_params))
    non_optimizer_trainable_tensors = sum(
        1 for param in student_model.parameters() if param.requires_grad and id(param) not in expected_trainable_param_ids
    )
    if adapter_enabled:
        unfrozen_modules = adapter_specs_text(adapter_specs)
        if bool(args.adapter_train_detect_head):
            unfrozen_modules.append(f"{head_idx}:{layers[head_idx].__class__.__name__}")
    else:
        unfrozen_modules = [f"{idx}:{layers[idx].__class__.__name__}" for idx in unfrozen_indices]
    student_model.odad_adaptation_mode = adaptation_mode
    student_model.odad_adapter_train_detect_head = bool(args.adapter_train_detect_head)
    student_model.odad_adapter_specs = [spec.__dict__.copy() for spec in adapter_specs]
    student_model.odad_memory_adapter_enabled = bool(memory_adapter_enabled)
    student_model.odad_memory_adapter_dim = int(args.memory_adapter_dim)
    student_model.odad_memory_adapter_source_layer = int(args.memory_adapter_source_layer)
    student_model.odad_memory_adapter_conditioning = str(args.memory_adapter_conditioning)
    student_model.odad_memory_adapter_bank_enabled = bool(memory_bank_enabled)
    student_model.odad_memory_adapter_bank_size = int(args.memory_adapter_bank_size)
    student_model.odad_memory_adapter_topk = int(args.memory_adapter_topk)
    student_model.odad_memory_adapter_slot_dim = int(args.memory_adapter_slot_dim)
    student_model.odad_source_memory_enabled = bool(args.source_memory_enable)
    student_model.odad_source_memory_weight = float(args.source_memory_weight)
    student_model.odad_source_memory_temp = float(args.source_memory_temp)
    student_model.odad_source_memory_layer = int(args.source_memory_layer)
    persistence_frames = max(1, int(args.persistence_frames))
    update_every_frames, updates_per_event = resolve_update_schedule(
        max_updates_per_frame=int(args.max_updates_per_frame),
        update_every_frames=int(args.update_every_frames),
        updates_per_event=int(args.updates_per_event),
    )
    rng = random.Random(int(args.seed))

    summary_path = out_dir / "summary.txt"
    startup_lines = [
        "startup:",
        f"  adaptation_mode={adaptation_mode}",
        f"  update_scope={args.update_scope}",
        f"  neck_start_idx={neck_start_idx if neck_start_idx is not None else 'n/a'}",
        f"  head_idx={head_idx}",
        f"  adapter_enabled={int(adapter_enabled)}",
        f"  adapter_layers={','.join(str(v) for v in adapter_layer_indices) if adapter_layer_indices else 'n/a'}",
        f"  adapter_reduction={int(args.adapter_reduction)}",
        f"  adapter_min_channels={int(args.adapter_min_channels)}",
        f"  adapter_scale={float(args.adapter_scale):.3f}",
        f"  adapter_train_detect_head={int(bool(args.adapter_train_detect_head))}",
        f"  memory_adapter_enabled={int(memory_adapter_enabled)}",
        f"  memory_bank_enabled={int(memory_bank_enabled)}",
        f"  memory_bank_size={int(args.memory_adapter_bank_size)}",
        f"  memory_bank_topk={int(args.memory_adapter_topk)}",
        f"  memory_bank_slot_dim={int(args.memory_adapter_slot_dim)}",
        f"  memory_bank_query_mode={args.memory_adapter_query_mode}",
        f"  memory_bank_write_policy={args.memory_adapter_write_policy}",
        f"  memory_bank_diversity_thresh={float(args.memory_adapter_diversity_thresh):.3f}",
        f"  memory_bank_duplicate_thresh={float(args.memory_adapter_duplicate_thresh):.3f}",
        f"  memory_bank_quality_margin={float(args.memory_adapter_quality_margin):.3f}",
        f"  memory_bank_balance_scale_bins={int(bool(args.memory_adapter_balance_scale_bins))}",
        f"  memory_bank_balance_conf_bins={int(bool(args.memory_adapter_balance_conf_bins))}",
        f"  memory_bank_stable_medium_write={int(bool(args.memory_adapter_stable_medium_write))}",
        f"  memory_bank_stable_conf_min={float(args.memory_adapter_stable_conf_min):.3f}",
        f"  memory_bank_stable_iou_min={float(args.memory_adapter_stable_iou_min):.3f}",
        f"  memory_bank_retrieval_temp={float(args.memory_adapter_retrieval_temp):.3f}",
        f"  memory_adapter_dim={int(args.memory_adapter_dim)}",
        f"  memory_adapter_source_layer={int(args.memory_adapter_source_layer)}",
        f"  memory_adapter_ema={float(args.memory_adapter_ema):.3f}",
        f"  memory_adapter_min_conf={float(args.memory_adapter_min_conf):.3f}",
        f"  memory_adapter_min_area_frac={float(args.memory_adapter_min_area_frac):.6f}",
        f"  memory_adapter_update_on_accepted_only={int(bool(args.memory_adapter_update_on_accepted_only))}",
        f"  memory_adapter_conditioning={args.memory_adapter_conditioning}",
        f"  memory_adapter_disable_conditioning_if_supported={int(bool(args.memory_adapter_disable_conditioning_if_supported))}",
        f"  source_memory_enabled={int(bool(args.source_memory_enable))}",
        f"  source_memory_path={args.source_memory_path}",
        f"  source_memory_loss_type={args.source_memory_loss_type}",
        f"  source_memory_weight={float(args.source_memory_weight):.3f}",
        f"  source_memory_temp={float(args.source_memory_temp):.3f}",
        f"  source_memory_topk_pos={int(args.source_memory_topk_pos)}",
        f"  source_memory_neg_k={int(args.source_memory_neg_k)}",
        f"  source_memory_layer={int(args.source_memory_layer)}",
        f"  source_memory_debug_save={int(bool(args.source_memory_debug_save))}",
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
        f"  teacher_candidate_min_score={float(args.teacher_candidate_min_score):.3f}",
        f"  coverage_aux_enable={int(coverage_enabled)}",
        f"  coverage_buffer_size={int(args.coverage_buffer_size)}",
        f"  coverage_batch_size={int(args.coverage_batch_size)}",
        f"  coverage_weight={float(args.coverage_weight):.3f}",
        f"  coverage_candidate_conf_min={float(args.coverage_candidate_conf_min):.3f}",
        f"  coverage_candidate_conf_max={float(args.coverage_candidate_conf_max):.3f}",
        f"  coverage_loss_type={args.coverage_loss_type}",
        f"  coverage_margin={float(args.coverage_margin):.3f}",
        f"  coverage_max_loss={float(args.coverage_max_loss):.3f}",
        f"  coverage_region_expand={float(args.coverage_region_expand):.3f}",
        f"  coverage_region_min_candidates={int(args.coverage_region_min_candidates)}",
        f"  coverage_region_center_radius_frac={float(args.coverage_region_center_radius_frac):.3f}",
        f"  coverage_region_use_cls={int(bool(args.coverage_region_use_cls))}",
        f"  coverage_region_objectness_weight={float(args.coverage_region_objectness_weight):.3f}",
        f"  update_every_frames={update_every_frames}",
        f"  updates_per_event={updates_per_event}",
        f"  max_updates_per_frame_legacy={max(1, int(args.max_updates_per_frame))}",
        f"  save_checkpoints_every={max(0, int(args.save_checkpoints_every))}",
        f"  save_final_weights={int(bool(args.save_final_weights))}",
        f"  final_weights_name={args.final_weights_name}",
        f"  unfrozen_modules=[{', '.join(unfrozen_modules)}]",
        f"  adapter_trainable_params={adapter_trainable_params}",
        f"  memory_adapter_trainable_params={memory_adapter_trainable_params}",
        f"  trainable_params={trainable_params}",
        f"  total_param_tensors={total_param_tensors}",
        f"  frozen_param_tensors={frozen_param_tensors}",
        f"  frozen_param_numel={frozen_param_numel}",
        f"  optimizer_param_tensors={optimizer_param_tensors}",
        f"  optimizer_param_numel={optimizer_param_numel}",
        f"  non_optimizer_trainable_tensors={non_optimizer_trainable_tensors}",
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
    coverage_buffer = CoverageBuffer(max_size=int(args.coverage_buffer_size), rng=rng)

    logs: List[FrameLog] = []
    mp4_frames: List[np.ndarray] = []
    adapter_debug_rows: List[Dict[str, float]] = []
    pre_update_sync_latencies_ms: List[float] = []
    batch_build_latencies_ms: List[float] = []
    update_forward_latencies_ms: List[float] = []
    update_loss_latencies_ms: List[float] = []
    update_backward_step_latencies_ms: List[float] = []
    memory_update_flags: List[int] = []
    memory_norm_values: List[float] = []
    memory_conditioning_norm_values: List[float] = []

    accepted_frames = 0
    updated_frames = 0
    number_of_update_events = 0
    total_optimizer_updates = 0
    update_losses: List[float] = []
    total_losses: List[float] = []
    coverage_losses: List[float] = []
    coverage_sample_counts: List[float] = []
    coverage_buffer_sizes_on_updates: List[int] = []
    coverage_region_skipped_counts: List[float] = []
    coverage_region_student_confs: List[float] = []
    coverage_region_teacher_confs: List[float] = []
    coverage_region_conf_gaps: List[float] = []
    source_memory_losses: List[float] = []
    source_memory_valid_counts: List[int] = []
    source_memory_pos_sims: List[float] = []
    source_memory_neg_sims: List[float] = []
    source_memory_margins: List[float] = []
    buffer_sizes_on_updates: List[int] = []
    batch_sizes_on_updates: List[int] = []
    checkpoint_paths: List[Path] = []
    selected_rank_example_paths: List[Path] = []
    max_selected_rank_examples = 8
    selected_rank_gt1_frames = 0
    coverage_entries_added_total = 0
    coverage_entries_sampled_total = 0
    coverage_reason_counts: Counter[str] = Counter()

    prev_selected_box: Optional[Tuple[float, float, float, float]] = None
    persistence_state: Optional[PersistenceState] = None
    motion_gate_enabled = persistence_frames > 1

    progress = make_progress_iter(len(images), disabled=bool(args.no_progress))
    for idx, img_path in enumerate(images):
        frame_t0 = time.time()

        with Image.open(img_path) as im:
            img_rgb = im.convert("RGB")
            w, h = img_rgb.size

        teacher_model.eval()
        raw_teacher_top1, teacher_candidates, teacher_lat_ms = predict_teacher_candidates_wrapper(
            yolo_wrapper=teacher_yolo,
            source=str(img_path),
            device=str(args.device),
            conf=float(args.infer_conf),
            iou=float(args.iou),
            topk=teacher_topk,
            conf_floor=float(args.teacher_candidate_conf_floor),
            allow_top1_fallback=teacher_topk <= 1,
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
            min_score=float(args.teacher_candidate_min_score),
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
        coverage_entries_added_this_frame = 0
        memory_updated_this_frame = 0
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
            if memory_state is not None:
                memory_updated_this_frame = int(
                    memory_state.maybe_update(
                        idx,
                        teacher_top1,
                        img_w=w,
                        img_h=h,
                        img_path=img_path,
                        img_rgb=img_rgb,
                    )
                )
                sync_memory_context_from_student(teacher_model, student_model)
        elif coverage_enabled and should_store_coverage_entry(
            candidate=teacher_top1,
            img_w=w,
            img_h=h,
            accepted=accepted_final,
            conf_min=float(args.coverage_candidate_conf_min),
            conf_max=float(args.coverage_candidate_conf_max),
            min_area_frac=float(args.min_area_frac),
            max_area_frac=float(args.max_area_frac),
            border_margin_frac=float(args.border_margin_frac),
        ):
            assert teacher_top1 is not None
            reason_not_accepted = coverage_reject_reason(
                selection=selection,
                passed_base_gate=passed_base_gate,
                passed_motion_gate=passed_motion_gate,
                passed_persistence_gate=passed_persistence_gate,
            )
            coverage_buffer.add(
                CoverageEntry(
                    frame_idx=idx,
                    path=str(img_path),
                    width=w,
                    height=h,
                    teacher_box_xyxy=teacher_top1.xyxy,
                    teacher_cls=int(teacher_top1.cls_id),
                    teacher_conf=float(teacher_top1.conf),
                    reason_not_accepted=reason_not_accepted,
                )
            )
            coverage_reason_counts[reason_not_accepted] += 1
            coverage_entries_added_this_frame = 1
            coverage_entries_added_total += 1

        stable_memory_quality = max(
            float(temporal_iou) if math.isfinite(float(temporal_iou)) else 0.0,
            float(persistence_iou) if math.isfinite(float(persistence_iou)) else 0.0,
        )
        if (
            memory_state is not None
            and hasattr(memory_state, "stable_medium_write")
            and bool(getattr(memory_state, "stable_medium_write"))
            and not accepted_final
            and not memory_updated_this_frame
            and teacher_top1 is not None
            and float(teacher_top1.conf) >= float(args.memory_adapter_stable_conf_min)
            and stable_memory_quality >= float(args.memory_adapter_stable_iou_min)
        ):
            memory_updated_this_frame = int(
                memory_state.maybe_update(
                    idx,
                    teacher_top1,
                    img_w=w,
                    img_h=h,
                    img_path=img_path,
                    img_rgb=img_rgb,
                    stable_medium=True,
                    temporal_quality=stable_memory_quality,
                )
            )
            sync_memory_context_from_student(teacher_model, student_model)

        if (
            memory_state is not None
            and not bool(args.memory_adapter_update_on_accepted_only)
            and not memory_updated_this_frame
            and teacher_top1 is not None
        ):
            memory_updated_this_frame = int(
                memory_state.maybe_update(
                    idx,
                    teacher_top1,
                    img_w=w,
                    img_h=h,
                    img_path=img_path,
                    img_rgb=img_rgb,
                )
            )
            sync_memory_context_from_student(teacher_model, student_model)

        if (
            memory_state is not None
            and hasattr(memory_state, "read_current")
            and not bool(args.memory_adapter_disable_conditioning_if_supported)
        ):
            memory_state.read_current(
                frame_idx=idx,
                det=teacher_top1,
                img_w=w,
                img_h=h,
                img_path=img_path,
                img_rgb=img_rgb,
            )
            sync_memory_context_from_student(teacher_model, student_model)

        updates_this_frame = 0
        batch_size_used = 0
        num_pseudo_boxes_used = 0
        last_det_loss = float("nan")
        last_total_loss = float("nan")
        last_coverage_loss = float("nan")
        last_coverage_stats = CoverageLossStats()
        last_source_memory_stats = SourceMemoryStats()
        coverage_entries_sampled_this_frame = 0
        update_latency_ms = 0.0
        buffer_warm = len(buffer) >= int(args.min_buffer_before_update)
        update_event_triggered = int(buffer_warm and should_trigger_update_event(idx, update_every_frames))

        if update_event_triggered:
            number_of_update_events += 1
            for _ in range(updates_per_event):
                target_entries = buffer.sample(
                    batch_size=int(args.update_batch_size),
                    mode=str(args.buffer_sample_mode),
                )
                if not target_entries:
                    break

                if adapter_enabled:
                    apply_adapter_freeze_policy(
                        model=student_model,
                        layers=layers,
                        head_idx=head_idx,
                        train_detect_head=bool(args.adapter_train_detect_head),
                    )
                else:
                    apply_freeze_policy(
                        model=student_model,
                        layers=layers,
                        unfrozen_indices=unfrozen_indices,
                    )

                sync_t0 = time.perf_counter()
                synchronize_cuda_device(str(args.device))
                pre_update_sync_latencies_ms.append((time.perf_counter() - sync_t0) * 1000.0)
                update_t0 = time.perf_counter()
                batch_t0 = time.perf_counter()
                batch = build_training_batch(
                    target_entries=target_entries,
                    imgsz=int(args.imgsz),
                    device=str(args.device),
                    rng=rng,
                )
                coverage_entries: List[CoverageEntry] = []
                coverage_batch: Optional[Dict[str, torch.Tensor]] = None
                if coverage_enabled and len(coverage_buffer) > 0 and int(args.coverage_batch_size) > 0:
                    coverage_entries = coverage_buffer.sample(
                        batch_size=int(args.coverage_batch_size),
                        mode=str(args.buffer_sample_mode),
                    )
                    if coverage_entries:
                        coverage_batch = build_coverage_image_batch(
                            coverage_entries=coverage_entries,
                            imgsz=int(args.imgsz),
                            device=str(args.device),
                            rng=rng,
                        )
                if isinstance(batch.get("img"), torch.Tensor) and batch["img"].is_cuda:
                    torch.cuda.synchronize(batch["img"].device)
                batch_build_latencies_ms.append((time.perf_counter() - batch_t0) * 1000.0)
                adapter_snapshot = (
                    snapshot_adapter_params(student_model)
                    if adapter_enabled and bool(args.adapter_save_debug)
                    else None
                )
                update_timing: Dict[str, float] = {}
                det_loss, total_loss, coverage_loss, coverage_stats, source_memory_stats = run_detection_update(
                    student_model=student_model,
                    criterion=criterion,
                    optimizer=optimizer,
                    optim_params=optim_params,
                    batch=batch,
                    grad_clip=float(args.grad_clip),
                    target_entries=target_entries,
                    frame_idx=idx,
                    source_memory_state=source_memory_state if bool(args.source_memory_enable) else None,
                    coverage_batch=coverage_batch,
                    coverage_weight=float(args.coverage_weight) if coverage_enabled else 0.0,
                    coverage_loss_type=str(args.coverage_loss_type),
                    coverage_margin=float(args.coverage_margin),
                    coverage_max_loss=float(args.coverage_max_loss),
                    coverage_region_expand=float(args.coverage_region_expand),
                    coverage_region_min_candidates=int(args.coverage_region_min_candidates),
                    coverage_region_center_radius_frac=float(args.coverage_region_center_radius_frac),
                    coverage_region_use_cls=bool(args.coverage_region_use_cls),
                    num_classes=num_classes,
                    train_mode_callback=train_mode_callback,
                    update_timing=update_timing,
                )
                synchronize_cuda_device(str(args.device))
                update_latency_ms += (time.perf_counter() - update_t0) * 1000.0
                if "forward_ms" in update_timing:
                    update_forward_latencies_ms.append(float(update_timing["forward_ms"]))
                if "loss_ms" in update_timing:
                    update_loss_latencies_ms.append(float(update_timing["loss_ms"]))
                if "backward_step_ms" in update_timing:
                    update_backward_step_latencies_ms.append(float(update_timing["backward_step_ms"]))
                if adapter_snapshot is not None:
                    debug_stats = adapter_debug_stats(student_model, adapter_snapshot)
                    norm_sq = sum(
                        float(value) * float(value)
                        for key, value in debug_stats.items()
                        if key.endswith(".norm")
                    )
                    delta_sq = sum(
                        float(value) * float(value)
                        for key, value in debug_stats.items()
                        if key.endswith(".delta_norm")
                    )
                    adapter_debug_rows.append(
                        {
                            "frame": float(idx),
                            "optimizer_update": float(total_optimizer_updates + 1),
                            "adapter_param_norm": float(math.sqrt(norm_sq)),
                            "adapter_delta_norm": float(math.sqrt(delta_sq)),
                        }
                    )

                update_teacher_ema(
                    teacher_model=teacher_model,
                    student_model=student_model,
                    decay=float(args.ema_decay),
                )
                if memory_state is not None:
                    sync_memory_context_from_student(teacher_model, student_model)

                updates_this_frame += 1
                total_optimizer_updates += 1

                last_det_loss = float(det_loss)
                last_total_loss = float(total_loss)
                last_coverage_loss = float(coverage_loss)
                last_coverage_stats = coverage_stats
                last_source_memory_stats = source_memory_stats
                update_losses.append(float(det_loss))
                total_losses.append(float(total_loss))
                coverage_losses.append(float(coverage_loss))
                if math.isfinite(float(source_memory_stats.loss)):
                    source_memory_losses.append(float(source_memory_stats.loss))
                if int(source_memory_stats.valid_entries) > 0:
                    source_memory_valid_counts.append(int(source_memory_stats.valid_entries))
                if math.isfinite(float(source_memory_stats.mean_pos_sim)):
                    source_memory_pos_sims.append(float(source_memory_stats.mean_pos_sim))
                if math.isfinite(float(source_memory_stats.mean_neg_sim)):
                    source_memory_neg_sims.append(float(source_memory_stats.mean_neg_sim))
                if math.isfinite(float(source_memory_stats.margin)):
                    source_memory_margins.append(float(source_memory_stats.margin))
                target_count_step = len(target_entries)
                coverage_count_step = len(coverage_entries)
                batch_size_used = target_count_step
                num_pseudo_boxes_used += target_count_step
                coverage_entries_sampled_this_frame += coverage_count_step
                coverage_entries_sampled_total += coverage_count_step
                coverage_sample_counts.append(float(coverage_count_step))
                buffer_sizes_on_updates.append(len(buffer))
                coverage_buffer_sizes_on_updates.append(len(coverage_buffer))
                batch_sizes_on_updates.append(target_count_step)
                coverage_region_skipped_counts.append(float(coverage_stats.entries_skipped))
                if math.isfinite(float(coverage_stats.mean_student_conf)):
                    coverage_region_student_confs.append(float(coverage_stats.mean_student_conf))
                if math.isfinite(float(coverage_stats.mean_teacher_conf)):
                    coverage_region_teacher_confs.append(float(coverage_stats.mean_teacher_conf))
                if math.isfinite(float(coverage_stats.mean_conf_gap)):
                    coverage_region_conf_gaps.append(float(coverage_stats.mean_conf_gap))

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
        if memory_state is not None:
            memory_state.record_frame()
            memory_update_flags.append(int(memory_updated_this_frame))
            current_memory_stats = memory_state.stats()
            memory_norm_values.append(float(current_memory_stats["memory_adapter_mean_norm"]))
            memory_conditioning_norm_values.append(float(current_memory_stats["mean_memory_conditioning_norm"]))

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
                coverage_buffer_size=len(coverage_buffer),
                coverage_entries_added=int(coverage_entries_added_this_frame),
                coverage_entries_sampled=int(coverage_entries_sampled_this_frame),
                coverage_loss=float(last_coverage_loss),
                coverage_region_entries_skipped=int(last_coverage_stats.entries_skipped),
                coverage_region_mean_student_conf=float(last_coverage_stats.mean_student_conf),
                coverage_region_mean_teacher_conf=float(last_coverage_stats.mean_teacher_conf),
                coverage_region_mean_conf_gap=float(last_coverage_stats.mean_conf_gap),
                source_memory_loss=float(last_source_memory_stats.loss),
                source_memory_valid_entries=int(last_source_memory_stats.valid_entries),
                source_memory_mean_pos_sim=float(last_source_memory_stats.mean_pos_sim),
                source_memory_mean_neg_sim=float(last_source_memory_stats.mean_neg_sim),
                source_memory_margin=float(last_source_memory_stats.margin),
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
                    "cov": len(coverage_buffer),
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
                "coverage_buffer_size",
                "coverage_entries_added",
                "coverage_entries_sampled",
                "coverage_loss",
                "coverage_region_entries_skipped",
                "coverage_region_mean_student_conf",
                "coverage_region_mean_teacher_conf",
                "coverage_region_mean_conf_gap",
                "source_memory_loss",
                "source_memory_valid_entries",
                "source_memory_mean_pos_sim",
                "source_memory_mean_neg_sim",
                "source_memory_margin",
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
                    f"{row.teacher_selected_score_conf:.6f}" if math.isfinite(row.teacher_selected_score_conf) else "",
                    f"{row.teacher_selected_score_temporal:.6f}" if math.isfinite(row.teacher_selected_score_temporal) else "",
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
                    row.coverage_buffer_size,
                    row.coverage_entries_added,
                    row.coverage_entries_sampled,
                    f"{row.coverage_loss:.6f}" if math.isfinite(row.coverage_loss) else "",
                    row.coverage_region_entries_skipped,
                    (
                        f"{row.coverage_region_mean_student_conf:.6f}"
                        if math.isfinite(row.coverage_region_mean_student_conf)
                        else ""
                    ),
                    (
                        f"{row.coverage_region_mean_teacher_conf:.6f}"
                        if math.isfinite(row.coverage_region_mean_teacher_conf)
                        else ""
                    ),
                    (
                        f"{row.coverage_region_mean_conf_gap:.6f}"
                        if math.isfinite(row.coverage_region_mean_conf_gap)
                        else ""
                    ),
                    f"{row.source_memory_loss:.6f}" if math.isfinite(row.source_memory_loss) else "",
                    row.source_memory_valid_entries,
                    (
                        f"{row.source_memory_mean_pos_sim:.6f}"
                        if math.isfinite(row.source_memory_mean_pos_sim)
                        else ""
                    ),
                    (
                        f"{row.source_memory_mean_neg_sim:.6f}"
                        if math.isfinite(row.source_memory_mean_neg_sim)
                        else ""
                    ),
                    (
                        f"{row.source_memory_margin:.6f}"
                        if math.isfinite(row.source_memory_margin)
                        else ""
                    ),
                    f"{row.teacher_latency_ms:.3f}",
                    f"{row.student_post_conf:.6f}",
                    f"{row.student_post_latency_ms:.3f}",
                    f"{row.update_latency_ms:.3f}",
                    f"{row.frame_latency_ms:.3f}",
                ]
            )

    adapter_debug_path: Optional[Path] = None
    if adapter_debug_rows:
        adapter_debug_path = out_dir / "adapter_debug.csv"
        with adapter_debug_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame",
                    "optimizer_update",
                    "adapter_param_norm",
                    "adapter_delta_norm",
                ],
            )
            writer.writeheader()
            for row in adapter_debug_rows:
                writer.writerow(row)

    memory_debug_path: Optional[Path] = None
    if memory_state is not None and bool(args.memory_adapter_save_debug):
        memory_debug_path = out_dir / "memory_adapter_debug.csv"
        with memory_debug_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "memory_update", "memory_norm", "memory_conditioning_norm"])
            for frame, update_flag, norm_value, cond_norm in zip(
                [row.frame for row in logs],
                memory_update_flags,
                memory_norm_values,
                memory_conditioning_norm_values,
            ):
                writer.writerow(
                    [
                        int(frame),
                        int(update_flag),
                        f"{float(norm_value):.6f}" if math.isfinite(float(norm_value)) else "",
                        f"{float(cond_norm):.6f}" if math.isfinite(float(cond_norm)) else "",
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
    batch_size_used_vals = np.array([row.batch_size_used for row in logs], dtype=np.float32)
    updates_this_frame_vals = np.array([row.updates_this_frame for row in logs], dtype=np.float32)
    det_loss_vals = np.array([row.det_loss for row in logs], dtype=np.float32)
    coverage_buffer_size_vals = np.array([row.coverage_buffer_size for row in logs], dtype=np.float32)
    coverage_loss_vals = np.array([row.coverage_loss for row in logs], dtype=np.float32)
    coverage_region_conf_gap_vals = np.array([row.coverage_region_mean_conf_gap for row in logs], dtype=np.float32)
    source_memory_loss_vals = np.array([row.source_memory_loss for row in logs], dtype=np.float32)
    source_memory_pos_sim_vals = np.array([row.source_memory_mean_pos_sim for row in logs], dtype=np.float32)
    source_memory_neg_sim_vals = np.array([row.source_memory_mean_neg_sim for row in logs], dtype=np.float32)
    source_memory_margin_vals = np.array([row.source_memory_margin for row in logs], dtype=np.float32)
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
        "updates_this_frame",
        out_dir / "plot_update_count_roll.png",
    )
    save_plot_lines(
        frames,
        [accepted_vals, update_applied_vals],
        ["accepted", "update_applied"],
        "Acceptance and Update Flags",
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
        "Selected Teacher Candidate Rank",
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
    save_plot_lines(
        frames[update_mask],
        [coverage_loss_vals[update_mask]],
        ["coverage_loss"],
        "Coverage Loss on Update Frames",
        "frame",
        "coverage_loss",
        out_dir / "plot_coverage_loss.png",
    )
    save_plot_lines(
        frames[update_mask],
        [coverage_loss_vals[update_mask]],
        ["coverage_region_loss"],
        "Coverage Region Loss on Update Frames",
        "frame",
        "coverage_region_loss",
        out_dir / "plot_coverage_region_loss.png",
    )
    save_plot_lines(
        frames[update_mask],
        [coverage_region_conf_gap_vals[update_mask]],
        ["student_region_conf - teacher_conf"],
        "Coverage Region Confidence Gap",
        "frame",
        "conf_gap",
        out_dir / "plot_coverage_region_conf_gap.png",
    )
    save_plot_lines(
        frames,
        [coverage_buffer_size_vals],
        ["coverage_buffer_size"],
        "Coverage Buffer Size",
        "frame",
        "entries",
        out_dir / "plot_coverage_buffer_size.png",
    )
    if memory_state is not None:
        memory_norm_vals = np.array(memory_norm_values, dtype=np.float32)
        memory_update_vals = np.array(memory_update_flags, dtype=np.float32)
        save_plot_lines(
            frames,
            [memory_norm_vals],
            ["memory_norm"],
            "Memory Adapter Norm",
            "frame",
            "norm",
            out_dir / "plot_memory_adapter_norm.png",
        )
        save_plot_lines(
            frames,
            [memory_update_vals],
            ["memory_update"],
            "Memory Adapter Updates",
            "frame",
            "flag",
            out_dir / "plot_memory_adapter_updates.png",
        )
    if source_memory_state is not None:
        save_plot_lines(
            frames,
            [source_memory_loss_vals],
            ["contrastive_loss"],
            "Memory Contrastive Loss",
            "frame",
            "loss",
            out_dir / "plot_source_memory_loss.png",
        )
        save_plot_lines(
            frames,
            [source_memory_pos_sim_vals, source_memory_neg_sim_vals],
            ["pos_sim", "neg_sim"],
            "Memory Contrastive Positive/Negative Similarity",
            "frame",
            "cosine",
            out_dir / "plot_source_memory_pos_neg_sim.png",
        )
        save_plot_lines(
            frames,
            [source_memory_margin_vals],
            ["pos_minus_neg"],
            "Memory Contrastive Margin",
            "frame",
            "cosine",
            out_dir / "plot_source_memory_margin.png",
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
    mean_coverage_loss_updates = mean_or_nan(coverage_losses)
    mean_coverage_samples_updates = mean_or_nan(coverage_sample_counts)
    mean_buffer_size_updates = mean_or_nan(buffer_sizes_on_updates)
    mean_coverage_buffer_size_updates = mean_or_nan(coverage_buffer_sizes_on_updates)
    mean_coverage_region_entries_skipped = mean_or_nan(coverage_region_skipped_counts)
    total_coverage_region_entries_skipped = int(sum(float(v) for v in coverage_region_skipped_counts))
    mean_coverage_region_student_conf = mean_or_nan(coverage_region_student_confs)
    mean_coverage_region_teacher_conf = mean_or_nan(coverage_region_teacher_confs)
    mean_coverage_region_conf_gap = mean_or_nan(coverage_region_conf_gaps)

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
    mean_num_candidates = float(np.mean(teacher_num_candidates_vals)) if logs else float("nan")
    memory_debug_root: Optional[Path] = None
    if memory_state is not None and hasattr(memory_state, "finalize_outputs"):
        memory_debug_root = memory_state.finalize_outputs()
    if source_memory_state is not None:
        source_memory_state.finalize_outputs(out_dir)

    final_memory_stats = memory_state.stats() if memory_state is not None else {}
    final_source_memory_stats = (
        source_memory_state.stats() if source_memory_state is not None else {}
    )
    memory_adapter_updates = int(final_memory_stats.get("memory_adapter_updates", 0.0))
    memory_adapter_initialized = int(final_memory_stats.get("memory_adapter_initialized", 0.0))
    memory_adapter_mean_norm = float(final_memory_stats.get("memory_adapter_mean_norm", float("nan")))
    memory_adapter_last_update_frame = int(final_memory_stats.get("memory_adapter_last_update_frame", -1.0))
    memory_bank_active_slots = int(final_memory_stats.get("memory_bank_active_slots", 0.0))
    memory_bank_writes = int(final_memory_stats.get("memory_bank_writes", 0.0))
    memory_bank_appends = int(final_memory_stats.get("memory_bank_appends", 0.0))
    memory_bank_replacements = int(final_memory_stats.get("memory_bank_replacements", 0.0))
    memory_bank_duplicate_skips = int(final_memory_stats.get("memory_bank_duplicate_skips", 0.0))
    memory_bank_low_quality_skips = int(final_memory_stats.get("memory_bank_low_quality_skips", 0.0))
    memory_bank_stable_medium_writes = int(final_memory_stats.get("memory_bank_stable_medium_writes", 0.0))
    memory_slot_scale_bin_counts = str(final_memory_stats.get("memory_slot_scale_bin_counts", "small:0|medium:0|large:0"))
    memory_slot_conf_bin_counts = str(final_memory_stats.get("memory_slot_conf_bin_counts", "medium:0|high:0"))
    memory_write_decision_counts = str(final_memory_stats.get("memory_write_decision_counts", ""))
    memory_retrieval_top1_sim = float(final_memory_stats.get("memory_retrieval_top1_sim", float("nan")))
    memory_retrieval_mean_topk_sim = float(final_memory_stats.get("memory_retrieval_mean_topk_sim", float("nan")))
    memory_retrieval_top1_slot = int(final_memory_stats.get("memory_retrieval_top1_slot", -1.0))
    memory_retrieval_entropy = float(final_memory_stats.get("memory_retrieval_entropy", float("nan")))
    memory_slot_mean_pairwise_sim = float(final_memory_stats.get("memory_slot_mean_pairwise_sim", float("nan")))
    memory_slot_max_pairwise_sim = float(final_memory_stats.get("memory_slot_max_pairwise_sim", float("nan")))
    memory_debug_images_saved = int(final_memory_stats.get("memory_debug_images_saved", 0.0))
    mean_memory_conditioning_norm = (
        mean_or_nan(memory_conditioning_norm_values)
        if memory_conditioning_norm_values
        else float(final_memory_stats.get("mean_memory_conditioning_norm", float("nan")))
    )
    source_memory_projection_params = 0
    source_adapter = first_memory_source_adapter(student_model)
    if source_adapter is not None and getattr(source_adapter, "memory_projector", None) is not None:
        source_memory_projection_params = int(
            sum(param.numel() for param in source_adapter.memory_projector.parameters())
        )
    source_memory_loss = float(final_source_memory_stats.get("source_memory_loss", float("nan")))
    mean_source_memory_loss_updates = float(
        final_source_memory_stats.get(
            "mean_source_memory_loss_updates",
            mean_or_nan(source_memory_losses),
        )
    )
    source_memory_valid_entries = int(
        final_source_memory_stats.get(
            "source_memory_valid_entries",
            float(sum(source_memory_valid_counts)),
        )
    )
    source_memory_skipped_updates = int(
        final_source_memory_stats.get("source_memory_skipped_updates", 0.0)
    )
    source_memory_mean_pos_sim = float(
        final_source_memory_stats.get(
            "source_memory_mean_pos_sim",
            mean_or_nan(source_memory_pos_sims),
        )
    )
    source_memory_mean_neg_sim = float(
        final_source_memory_stats.get(
            "source_memory_mean_neg_sim",
            mean_or_nan(source_memory_neg_sims),
        )
    )
    source_memory_margin = float(
        final_source_memory_stats.get(
            "source_memory_margin",
            mean_or_nan(source_memory_margins),
        )
    )
    source_memory_debug_images_saved = int(
        final_source_memory_stats.get("source_memory_debug_images_saved", 0.0)
    )

    final_weights_path: Optional[Path] = None
    checkpoint_reload_status = "not_checked"
    checkpoint_reload_error = ""
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
        try:
            reload_yolo = YOLO(str(final_weights_path))
            del reload_yolo
            checkpoint_reload_status = "ok"
        except Exception as exc:
            checkpoint_reload_status = "failed"
            checkpoint_reload_error = str(exc)

    update_latencies_positive = [float(v) for v in update_latency_vals.tolist() if float(v) > 0.0 and math.isfinite(float(v))]
    resource_metrics = cuda_resource_metrics(str(args.device))
    final_checkpoint_size_mb = path_size_mb(final_weights_path)

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
        f"adaptation_mode={adaptation_mode}",
        f"memory_adapter_enabled={int(memory_adapter_enabled)}",
        f"memory_bank_enabled={int(memory_bank_enabled)}",
        f"memory_bank_size={int(args.memory_adapter_bank_size)}",
        f"memory_write_policy={args.memory_adapter_write_policy}",
        f"memory_bank_active_slots={memory_bank_active_slots}",
        f"memory_bank_writes={memory_bank_writes}",
        f"memory_bank_writes_total={memory_bank_writes}",
        f"memory_bank_appends={memory_bank_appends}",
        f"memory_bank_replacements={memory_bank_replacements}",
        f"memory_bank_duplicate_skips={memory_bank_duplicate_skips}",
        f"memory_bank_low_quality_skips={memory_bank_low_quality_skips}",
        f"memory_bank_stable_medium_writes={memory_bank_stable_medium_writes}",
        f"memory_slot_scale_bin_counts={memory_slot_scale_bin_counts}",
        f"memory_slot_conf_bin_counts={memory_slot_conf_bin_counts}",
        f"memory_write_decision_counts={memory_write_decision_counts}",
        format_metric("memory_retrieval_top1_sim", memory_retrieval_top1_sim),
        format_metric("memory_retrieval_mean_topk_sim", memory_retrieval_mean_topk_sim),
        f"memory_retrieval_top1_slot={memory_retrieval_top1_slot}",
        format_metric("memory_retrieval_entropy", memory_retrieval_entropy),
        format_metric("memory_slot_mean_pairwise_sim", memory_slot_mean_pairwise_sim),
        format_metric("memory_slot_max_pairwise_sim", memory_slot_max_pairwise_sim),
        f"memory_debug_images_saved={memory_debug_images_saved}",
        f"source_memory_enabled={int(bool(args.source_memory_enable))}",
        f"source_memory_path={args.source_memory_path}",
        f"source_memory_slots={int(source_memory_state.source_vectors.shape[0]) if source_memory_state is not None else 0}",
        f"source_memory_loss_type={args.source_memory_loss_type}",
        format_metric("source_memory_weight", float(args.source_memory_weight)),
        format_metric("source_memory_temp", float(args.source_memory_temp)),
        format_metric("source_memory_loss", source_memory_loss),
        format_metric("mean_source_memory_loss_updates", mean_source_memory_loss_updates),
        f"source_memory_valid_entries={source_memory_valid_entries}",
        f"source_memory_skipped_updates={source_memory_skipped_updates}",
        format_metric("source_memory_mean_pos_sim", source_memory_mean_pos_sim),
        format_metric("source_memory_mean_neg_sim", source_memory_mean_neg_sim),
        format_metric("source_memory_margin", source_memory_margin),
        f"source_memory_projection_params={source_memory_projection_params}",
        f"source_memory_debug_images_saved={source_memory_debug_images_saved}",
        f"memory_adapter_dim={int(args.memory_adapter_dim)}",
        f"memory_adapter_source_layer={int(args.memory_adapter_source_layer)}",
        f"memory_adapter_updates={memory_adapter_updates}",
        f"memory_adapter_initialized={memory_adapter_initialized}",
        format_metric("memory_adapter_mean_norm", memory_adapter_mean_norm),
        f"memory_adapter_last_update_frame={memory_adapter_last_update_frame}",
        f"adapter_trainable_params={adapter_trainable_params}",
        f"adapter_total_trainable_params={adapter_trainable_params}",
        f"memory_adapter_trainable_params={memory_adapter_trainable_params}",
        f"trainable_params={trainable_params}",
        f"total_param_tensors={total_param_tensors}",
        f"frozen_param_tensors={frozen_param_tensors}",
        f"frozen_param_numel={frozen_param_numel}",
        f"optimizer_param_tensors={optimizer_param_tensors}",
        f"optimizer_param_numel={optimizer_param_numel}",
        f"non_optimizer_trainable_tensors={non_optimizer_trainable_tensors}",
        format_metric("mean_teacher_conf", mean_teacher_conf),
        format_metric("mean_student_post_conf", mean_student_post_conf),
        format_metric("mean_detection_loss_updates", mean_update_loss),
        format_metric("mean_total_loss_updates", mean_total_loss_updates),
        format_metric("mean_coverage_loss_updates", mean_coverage_loss_updates),
        format_metric("mean_coverage_region_student_conf", mean_coverage_region_student_conf),
        format_metric("mean_coverage_region_teacher_conf", mean_coverage_region_teacher_conf),
        format_metric("mean_coverage_region_conf_gap", mean_coverage_region_conf_gap),
        format_metric("mean_memory_conditioning_norm", mean_memory_conditioning_norm),
        format_metric("mean_buffer_size_during_updates", mean_buffer_size_updates, ".3f"),
        format_metric("mean_coverage_buffer_size_during_updates", mean_coverage_buffer_size_updates, ".3f"),
        "",
        "resource_metrics:",
        format_metric("mean_update_latency_ms", mean_or_nan(update_latencies_positive), ".3f", prefix="  "),
        format_metric("mean_pre_update_sync_latency_ms", mean_or_nan(pre_update_sync_latencies_ms), ".3f", prefix="  "),
        format_metric("mean_batch_build_latency_ms", mean_or_nan(batch_build_latencies_ms), ".3f", prefix="  "),
        format_metric("mean_update_forward_latency_ms", mean_or_nan(update_forward_latencies_ms), ".3f", prefix="  "),
        format_metric("mean_update_loss_latency_ms", mean_or_nan(update_loss_latencies_ms), ".3f", prefix="  "),
        format_metric("mean_update_backward_step_latency_ms", mean_or_nan(update_backward_step_latencies_ms), ".3f", prefix="  "),
        format_metric("peak_cuda_allocated_mb", resource_metrics["peak_cuda_allocated_mb"], ".1f", prefix="  "),
        format_metric("peak_cuda_reserved_mb", resource_metrics["peak_cuda_reserved_mb"], ".1f", prefix="  "),
        format_metric("final_checkpoint_size_mb", final_checkpoint_size_mb, ".3f", prefix="  "),
        f"  checkpoint_reload_status={checkpoint_reload_status}",
        f"  checkpoint_reload_ok={int(checkpoint_reload_status == 'ok')}",
        *([f"  checkpoint_reload_error={checkpoint_reload_error}"] if checkpoint_reload_error else []),
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
        f"  teacher_candidate_min_score={float(args.teacher_candidate_min_score):.3f}",
        format_metric("mean_num_candidates", mean_num_candidates, prefix="  "),
        format_metric("mean_selected_rank", mean_selected_rank, prefix="  "),
        format_metric("fraction_selected_rank_gt1", fraction_selected_rank_gt1, prefix="  "),
        format_metric("mean_selected_score", mean_selected_score, prefix="  "),
        f"  selected_rank_gt1_frames={selected_rank_gt1_frames}",
        *(
            [f"  selected_rank_gt1_examples_saved={len(selected_rank_example_paths)}"]
            if selected_rank_example_paths
            else []
        ),
        "",
        "coverage_aux:",
        f"  coverage_aux_enable={int(coverage_enabled)}",
        f"  coverage_loss_type={args.coverage_loss_type}",
        f"  coverage_weight={float(args.coverage_weight):.3f}",
        f"  coverage_margin={float(args.coverage_margin):.3f}",
        f"  coverage_max_loss={float(args.coverage_max_loss):.3f}",
        f"  coverage_region_expand={float(args.coverage_region_expand):.3f}",
        f"  coverage_region_min_candidates={int(args.coverage_region_min_candidates)}",
        f"  coverage_region_center_radius_frac={float(args.coverage_region_center_radius_frac):.3f}",
        f"  coverage_region_use_cls={int(bool(args.coverage_region_use_cls))}",
        f"  coverage_region_objectness_weight={float(args.coverage_region_objectness_weight):.3f}",
        f"  coverage_candidate_conf_min={float(args.coverage_candidate_conf_min):.3f}",
        f"  coverage_candidate_conf_max={float(args.coverage_candidate_conf_max):.3f}",
        f"  coverage_buffer_size={len(coverage_buffer)}",
        f"  coverage_entries_added={coverage_entries_added_total}",
        f"  coverage_entries_sampled={coverage_entries_sampled_total}",
        f"  coverage_region_entries_skipped={total_coverage_region_entries_skipped}",
        format_metric("mean_coverage_loss_updates", mean_coverage_loss_updates, prefix="  "),
        format_metric("mean_coverage_samples_updates", mean_coverage_samples_updates, prefix="  "),
        format_metric("mean_coverage_buffer_size_during_updates", mean_coverage_buffer_size_updates, ".3f", prefix="  "),
        format_metric("mean_coverage_region_entries_skipped", mean_coverage_region_entries_skipped, ".3f", prefix="  "),
        format_metric("mean_coverage_region_student_conf", mean_coverage_region_student_conf, prefix="  "),
        format_metric("mean_coverage_region_teacher_conf", mean_coverage_region_teacher_conf, prefix="  "),
        format_metric("mean_coverage_region_conf_gap", mean_coverage_region_conf_gap, prefix="  "),
        f"  coverage_reason_counts={dict(sorted(coverage_reason_counts.items()))}",
        "",
        "memory_adapter:",
        f"  memory_adapter_enabled={int(memory_adapter_enabled)}",
        f"  memory_bank_enabled={int(memory_bank_enabled)}",
        f"  memory_bank_size={int(args.memory_adapter_bank_size)}",
        f"  memory_adapter_topk={int(args.memory_adapter_topk)}",
        f"  memory_adapter_slot_dim={int(args.memory_adapter_slot_dim)}",
        f"  memory_adapter_query_mode={args.memory_adapter_query_mode}",
        f"  memory_adapter_write_policy={args.memory_adapter_write_policy}",
        f"  memory_write_policy={args.memory_adapter_write_policy}",
        f"  memory_adapter_diversity_thresh={float(args.memory_adapter_diversity_thresh):.3f}",
        f"  memory_adapter_duplicate_thresh={float(args.memory_adapter_duplicate_thresh):.3f}",
        f"  memory_adapter_quality_margin={float(args.memory_adapter_quality_margin):.3f}",
        f"  memory_adapter_balance_scale_bins={int(bool(args.memory_adapter_balance_scale_bins))}",
        f"  memory_adapter_balance_conf_bins={int(bool(args.memory_adapter_balance_conf_bins))}",
        f"  memory_adapter_stable_medium_write={int(bool(args.memory_adapter_stable_medium_write))}",
        f"  memory_adapter_stable_conf_min={float(args.memory_adapter_stable_conf_min):.3f}",
        f"  memory_adapter_stable_iou_min={float(args.memory_adapter_stable_iou_min):.3f}",
        f"  memory_adapter_retrieval_temp={float(args.memory_adapter_retrieval_temp):.3f}",
        f"  memory_adapter_dim={int(args.memory_adapter_dim)}",
        f"  memory_adapter_source_layer={int(args.memory_adapter_source_layer)}",
        f"  memory_adapter_ema={float(args.memory_adapter_ema):.3f}",
        f"  memory_adapter_min_conf={float(args.memory_adapter_min_conf):.3f}",
        f"  memory_adapter_min_area_frac={float(args.memory_adapter_min_area_frac):.6f}",
        f"  memory_adapter_update_on_accepted_only={int(bool(args.memory_adapter_update_on_accepted_only))}",
        f"  memory_adapter_conditioning={args.memory_adapter_conditioning}",
        f"  memory_adapter_disable_conditioning_if_supported={int(bool(args.memory_adapter_disable_conditioning_if_supported))}",
        f"  memory_adapter_updates={memory_adapter_updates}",
        f"  memory_adapter_initialized={memory_adapter_initialized}",
        format_metric("memory_adapter_mean_norm", memory_adapter_mean_norm, prefix="  "),
        f"  memory_adapter_last_update_frame={memory_adapter_last_update_frame}",
        f"  memory_adapter_trainable_params={memory_adapter_trainable_params}",
        format_metric("mean_memory_conditioning_norm", mean_memory_conditioning_norm, prefix="  "),
        f"  memory_bank_active_slots={memory_bank_active_slots}",
        f"  memory_bank_writes={memory_bank_writes}",
        f"  memory_bank_appends={memory_bank_appends}",
        f"  memory_bank_replacements={memory_bank_replacements}",
        f"  memory_bank_duplicate_skips={memory_bank_duplicate_skips}",
        f"  memory_bank_low_quality_skips={memory_bank_low_quality_skips}",
        f"  memory_bank_stable_medium_writes={memory_bank_stable_medium_writes}",
        f"  memory_slot_scale_bin_counts={memory_slot_scale_bin_counts}",
        f"  memory_slot_conf_bin_counts={memory_slot_conf_bin_counts}",
        f"  memory_write_decision_counts={memory_write_decision_counts}",
        format_metric("memory_retrieval_top1_sim", memory_retrieval_top1_sim, prefix="  "),
        format_metric("memory_retrieval_mean_topk_sim", memory_retrieval_mean_topk_sim, prefix="  "),
        f"  memory_retrieval_top1_slot={memory_retrieval_top1_slot}",
        format_metric("memory_retrieval_entropy", memory_retrieval_entropy, prefix="  "),
        format_metric("memory_slot_mean_pairwise_sim", memory_slot_mean_pairwise_sim, prefix="  "),
        format_metric("memory_slot_max_pairwise_sim", memory_slot_max_pairwise_sim, prefix="  "),
        f"  memory_debug_images_saved={memory_debug_images_saved}",
        f"  source_memory_enabled={int(bool(args.source_memory_enable))}",
        f"  source_memory_path={args.source_memory_path}",
        f"  source_memory_slots={int(source_memory_state.source_vectors.shape[0]) if source_memory_state is not None else 0}",
        f"  source_memory_loss_type={args.source_memory_loss_type}",
        format_metric("source_memory_weight", float(args.source_memory_weight), prefix="  "),
        format_metric("source_memory_temp", float(args.source_memory_temp), prefix="  "),
        f"  source_memory_topk_pos={int(args.source_memory_topk_pos)}",
        f"  source_memory_neg_k={int(args.source_memory_neg_k)}",
        f"  source_memory_layer={int(args.source_memory_layer)}",
        f"  source_memory_debug_save={int(bool(args.source_memory_debug_save))}",
        format_metric("source_memory_loss", source_memory_loss, prefix="  "),
        format_metric("mean_source_memory_loss_updates", mean_source_memory_loss_updates, prefix="  "),
        f"  source_memory_valid_entries={source_memory_valid_entries}",
        f"  source_memory_skipped_updates={source_memory_skipped_updates}",
        format_metric("source_memory_mean_pos_sim", source_memory_mean_pos_sim, prefix="  "),
        format_metric("source_memory_mean_neg_sim", source_memory_mean_neg_sim, prefix="  "),
        format_metric("source_memory_margin", source_memory_margin, prefix="  "),
        f"  source_memory_projection_params={source_memory_projection_params}",
        f"  source_memory_debug_images_saved={source_memory_debug_images_saved}",
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
        *([f"  adapter_debug={adapter_debug_path.name}"] if adapter_debug_path is not None else []),
        *([f"  memory_adapter_debug={memory_debug_path.name}"] if memory_debug_path is not None else []),
        *([f"  memory_debug={memory_debug_root.relative_to(out_dir)}"] if memory_debug_root is not None else []),
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
        "  plot_batch_size_used.png",
        "  plot_det_loss.png",
        "  plot_coverage_loss.png",
        "  plot_coverage_region_loss.png",
        "  plot_coverage_region_conf_gap.png",
        "  plot_coverage_buffer_size.png",
        *(["  plot_memory_adapter_norm.png", "  plot_memory_adapter_updates.png"] if memory_state is not None else []),
        *(
            [
                "  plot_source_memory_loss.png",
                "  plot_source_memory_pos_neg_sim.png",
                "  plot_source_memory_margin.png",
            ]
            if source_memory_state is not None
            else []
        ),
        *(
            [
                "  memory_debug/memory_writes/",
                "  memory_debug/memory_skipped_writes/",
                "  memory_debug/retrieval_examples/",
                *(
                    ["  memory_debug/source_memory_retrieval_examples/"]
                    if source_memory_state is not None and bool(args.source_memory_debug_save)
                    else []
                ),
                "  memory_debug/memory_slots_final.png",
                "  memory_debug/memory_similarity_heatmap.png",
                "  memory_debug/memory_timeline.png",
                "  memory_debug/memory_summary.csv",
                *(
                    [
                        "  memory_debug/plot_source_memory_loss.png",
                        "  memory_debug/plot_source_memory_pos_neg_sim.png",
                        "  memory_debug/plot_source_memory_margin.png",
                    ]
                    if source_memory_state is not None
                    else []
                ),
            ]
            if memory_debug_root is not None
            else []
        ),
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

    if memory_state is not None:
        memory_state.close()
    if source_memory_state is not None:
        source_memory_state.close()

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
