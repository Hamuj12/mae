#!/usr/bin/env python3
"""Shadow ODAD manager v1.

The manager simulates deployment with one stable active YOLO model serving every
frame while a separate clean top-k ODAD learner adapts in the background.
Promotion is label-free and compares active versus shadow on the same recent
rolling frame window before swapping checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install ultralytics: pip install ultralytics") from exc

from odad.online_adapt import (  # noqa: E402
    FrameLog,
    LossClass,
    PersistenceState,
    ReplayBuffer,
    ReplayEntry,
    Top1Det,
    apply_freeze_policy,
    build_training_batch,
    center_shift_fraction,
    compute_unfrozen_indices,
    evaluate_base_gate,
    evaluate_motion_gate,
    find_head_idx,
    format_metric,
    list_test_images,
    mean_or_nan,
    near_border,
    param_id_set_for_indices,
    predict_teacher_candidates_wrapper,
    predict_top1_wrapper,
    resolve_neck_start_idx,
    resolve_update_schedule,
    run_detection_update,
    save_plot_lines,
    save_student_weights_checkpoint,
    select_teacher_candidate,
    should_trigger_update_event,
    symmetric_area_ratio,
    unwrap_core_and_layers,
    update_persistence_state,
    update_teacher_ema,
)


@dataclass
class ShadowLearnerConfig:
    device: str
    imgsz: int
    infer_conf: float
    iou: float
    teacher_conf_thresh: float
    lr: float
    momentum: float
    weight_decay: float
    ema_decay: float
    grad_clip: float
    update_scope: str
    neck_start_idx: int
    buffer_size: int
    update_batch_size: int
    min_buffer_before_update: int
    buffer_sample_mode: str
    max_updates_per_frame: int
    update_every_frames: int
    updates_per_event: int
    min_area_frac: float
    max_area_frac: float
    border_margin_frac: float
    temporal_iou_gate: float
    persistence_frames: int
    persistence_iou: float
    max_center_shift_frac: float
    max_area_ratio: float
    teacher_topk: int
    teacher_candidate_conf_floor: float
    teacher_candidate_score_mode: str
    teacher_candidate_conf_weight: float
    teacher_candidate_temporal_weight: float
    teacher_candidate_min_score: float


@dataclass
class PredictionRecord:
    frame: int
    path: str
    model: str
    width: int
    height: int
    top1_conf: float
    top1_cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    has_detection: bool
    box_area_frac: float
    tiny_box: bool
    large_box: bool
    border_touching: bool
    box_jump: bool
    center_shift_frac: float
    area_ratio: float
    latency_ms: float


@dataclass
class ShadowStepResult:
    log: FrameLog
    teacher_top1: Optional[Top1Det]
    student_top1: Optional[Top1Det]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential shadow-learning ODAD deployment manager.")
    parser.add_argument("--active-weights", type=str, required=True, help="Initial active model weights used for stable inference.")
    parser.add_argument(
        "--shadow-init-weights",
        type=str,
        default="",
        help="Initial shadow learner weights. If omitted, start shadow from active weights.",
    )
    parser.add_argument("--dataset", type=str, required=True, help="YOLO-root dataset path with images/test stream.")
    parser.add_argument("--output", type=str, default="shadow_odad_out", help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=1024, help="YOLO image size.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames.")
    parser.add_argument("--eval-window", type=int, default=500, help="Number of recent frames used for active-vs-shadow promotion evaluation.")
    parser.add_argument("--promotion-every-frames", type=int, default=500, help="Evaluate shadow for promotion every N stream frames.")
    parser.add_argument("--shadow-update-start-frame", type=int, default=0, help="Frame index after which shadow updates are allowed.")
    parser.add_argument("--max-promotions", type=int, default=3, help="Maximum number of active-model promotions in a run.")
    parser.add_argument(
        "--rollback-on-failures",
        type=int,
        default=2,
        help="Reset shadow from active after this many consecutive failed promotion checks. 0 disables.",
    )
    parser.add_argument("--enable-update-throttle", action="store_true", help="Pause shadow updates when recent reliability proxies are poor.")
    parser.add_argument("--throttle-weird-rate", type=float, default=0.35, help="Pause updates if recent weird-box rate exceeds this.")
    parser.add_argument("--throttle-box-jump-rate", type=float, default=0.15, help="Pause updates if recent box-jump rate exceeds this.")
    parser.add_argument("--min-promotion-det-rate", type=float, default=0.95, help="Minimum candidate Det.@conf required for promotion.")
    parser.add_argument(
        "--promotion-det-margin",
        type=float,
        default=-0.01,
        help="Candidate det rate may be this much below active if reliability improves. Negative allows small drop.",
    )
    parser.add_argument(
        "--max-promotion-latency-ms",
        type=float,
        default=0.0,
        help="If >0, reject promotion if candidate active inference latency exceeds this.",
    )
    parser.add_argument(
        "--max-promotion-memory-mb",
        type=float,
        default=0.0,
        help="If >0, reject promotion if peak memory exceeds this.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for promotion reliability metrics.")
    parser.add_argument("--infer-conf", type=float, default=0.001, help="Low confidence threshold for top-1 distribution.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU.")

    parser.add_argument("--teacher-conf-thresh", type=float, default=0.80, help="Shadow pseudo-label confidence threshold.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Shadow student optimizer learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Shadow optimizer momentum.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Shadow optimizer weight decay.")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="Shadow teacher EMA decay.")
    parser.add_argument("--grad-clip", type=float, default=10.0, help="Shadow gradient clipping max norm.")
    parser.add_argument(
        "--update-scope",
        type=str,
        default="head_only",
        choices=["head_only", "neck_head"],
        help="Shadow trainable region.",
    )
    parser.add_argument("--neck-start-idx", type=int, default=-1, help="Manual neck start index override.")
    parser.add_argument("--buffer-size", type=int, default=32, help="Shadow replay buffer capacity.")
    parser.add_argument("--update-batch-size", type=int, default=4, help="Shadow replay mini-batch size.")
    parser.add_argument("--min-buffer-before-update", type=int, default=4, help="Minimum buffer entries before updates.")
    parser.add_argument("--buffer-sample-mode", choices=["recent", "random"], default="recent", help="Replay sampling mode.")
    parser.add_argument("--max-updates-per-frame", type=int, default=1, help="Legacy compatibility update count.")
    parser.add_argument("--update-every-frames", type=int, default=1, help="Shadow update cadence in stream frames.")
    parser.add_argument("--updates-per-event", type=int, default=1, help="Shadow optimizer steps per update event.")
    parser.add_argument("--min-area-frac", type=float, default=0.001, help="Min accepted pseudo-box area fraction.")
    parser.add_argument("--max-area-frac", type=float, default=0.80, help="Max accepted pseudo-box area fraction.")
    parser.add_argument("--border-margin-frac", type=float, default=0.02, help="Reject pseudo-boxes too close to border.")
    parser.add_argument("--temporal-iou-gate", type=float, default=0.50, help="Teacher temporal IoU gate.")
    parser.add_argument("--persistence-frames", type=int, default=2, help="Consecutive stable frames required for pseudo-labels.")
    parser.add_argument("--persistence-iou", type=float, default=0.50, help="Persistence IoU threshold.")
    parser.add_argument("--max-center-shift-frac", type=float, default=0.20, help="Max allowed center shift for shadow gates.")
    parser.add_argument("--max-area-ratio", type=float, default=2.5, help="Max allowed consecutive box area ratio.")
    parser.add_argument("--teacher-topk", type=int, default=2, help="Number of teacher detections considered by shadow.")
    parser.add_argument("--teacher-candidate-conf-floor", type=float, default=0.25, help="Min confidence for teacher candidate set.")
    parser.add_argument(
        "--teacher-candidate-score-mode",
        choices=["conf_only", "conf_temporal"],
        default="conf_temporal",
        help="How to score top-k teacher candidates.",
    )
    parser.add_argument("--teacher-candidate-conf-weight", type=float, default=1.0, help="Teacher candidate confidence weight.")
    parser.add_argument("--teacher-candidate-temporal-weight", type=float, default=1.0, help="Teacher candidate temporal weight.")
    parser.add_argument("--teacher-candidate-min-score", type=float, default=0.0, help="Optional minimum teacher candidate score.")

    parser.add_argument("--large-box-frac", type=float, default=0.30, help="Reliability weird-box large area threshold.")
    parser.add_argument("--tiny-box-frac", type=float, default=0.001, help="Reliability weird-box tiny area threshold.")
    parser.add_argument("--reliability-border-margin-frac", type=float, default=0.02, help="Reliability border-touch margin.")
    parser.add_argument("--box-jump-center-frac", type=float, default=0.20, help="Reliability box-jump center shift threshold.")
    parser.add_argument("--good-conf", type=float, default=0.75, help="High-confidence threshold for high-conf weird metrics.")
    parser.add_argument("--warmup", type=int, default=0, help="Shadow teacher warmup frames.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output.")
    return parser.parse_args()


def shadow_config_from_args(args: argparse.Namespace) -> ShadowLearnerConfig:
    return ShadowLearnerConfig(
        device=str(args.device),
        imgsz=int(args.imgsz),
        infer_conf=float(args.infer_conf),
        iou=float(args.iou),
        teacher_conf_thresh=float(args.teacher_conf_thresh),
        lr=float(args.lr),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        ema_decay=float(args.ema_decay),
        grad_clip=float(args.grad_clip),
        update_scope=str(args.update_scope),
        neck_start_idx=int(args.neck_start_idx),
        buffer_size=int(args.buffer_size),
        update_batch_size=int(args.update_batch_size),
        min_buffer_before_update=int(args.min_buffer_before_update),
        buffer_sample_mode=str(args.buffer_sample_mode),
        max_updates_per_frame=int(args.max_updates_per_frame),
        update_every_frames=int(args.update_every_frames),
        updates_per_event=int(args.updates_per_event),
        min_area_frac=float(args.min_area_frac),
        max_area_frac=float(args.max_area_frac),
        border_margin_frac=float(args.border_margin_frac),
        temporal_iou_gate=float(args.temporal_iou_gate),
        persistence_frames=max(1, int(args.persistence_frames)),
        persistence_iou=float(args.persistence_iou),
        max_center_shift_frac=float(args.max_center_shift_frac),
        max_area_ratio=float(args.max_area_ratio),
        teacher_topk=max(1, int(args.teacher_topk)),
        teacher_candidate_conf_floor=float(args.teacher_candidate_conf_floor),
        teacher_candidate_score_mode=str(args.teacher_candidate_score_mode),
        teacher_candidate_conf_weight=float(args.teacher_candidate_conf_weight),
        teacher_candidate_temporal_weight=float(args.teacher_candidate_temporal_weight),
        teacher_candidate_min_score=float(args.teacher_candidate_min_score),
    )


class ShadowODADLearner:
    def __init__(self, weights: str, config: ShadowLearnerConfig, seed: int = 0) -> None:
        self.weights = str(weights)
        self.config = config
        self.rng = random.Random(int(seed))
        self.student_yolo = YOLO(self.weights)
        self.teacher_yolo = YOLO(self.weights)
        self.student_model = self.student_yolo.model
        self.teacher_model = self.teacher_yolo.model
        self.student_model.to(config.device)
        self.teacher_model.to(config.device)

        self.core_model, self.layers = unwrap_core_and_layers(self.student_model)
        _, teacher_layers = unwrap_core_and_layers(self.teacher_model)
        self.head_idx = find_head_idx(self.layers)
        teacher_head_idx = find_head_idx(teacher_layers)
        if self.head_idx != teacher_head_idx:
            raise RuntimeError(f"Shadow student/teacher head mismatch: student={self.head_idx}, teacher={teacher_head_idx}")

        self.neck_start_idx = resolve_neck_start_idx(self.core_model, int(config.neck_start_idx))
        self.unfrozen_indices = compute_unfrozen_indices(
            update_scope=config.update_scope,
            neck_start_idx=self.neck_start_idx,
            head_idx=self.head_idx,
            n_layers=len(self.layers),
        )
        if config.update_scope == "head_only" and self.layers[self.head_idx].__class__.__name__ != "Detect":
            raise RuntimeError(
                f"update_scope=head_only expects Detect head, found {self.layers[self.head_idx].__class__.__name__}."
            )

        apply_freeze_policy(self.student_model, self.layers, self.unfrozen_indices)
        expected_trainable = param_id_set_for_indices(self.layers, self.unfrozen_indices)
        actual_trainable = {id(param) for param in self.student_model.parameters() if param.requires_grad}
        if expected_trainable != actual_trainable:
            raise RuntimeError("Shadow freeze policy mismatch.")
        self.optim_params = [param for param in self.student_model.parameters() if param.requires_grad]
        if not self.optim_params:
            raise RuntimeError("No shadow trainable parameters found.")
        self.optimizer = torch.optim.SGD(
            self.optim_params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        self.trainable_params = int(sum(param.numel() for param in self.optim_params))
        self.unfrozen_modules = [f"{idx}:{self.layers[idx].__class__.__name__}" for idx in self.unfrozen_indices]

        if isinstance(self.student_model.args, dict):
            hyp_dict = dict(self.student_model.args)
        elif isinstance(self.student_model.args, SimpleNamespace):
            hyp_dict = vars(self.student_model.args).copy()
        else:
            hyp_dict = vars(self.student_model.args).copy() if hasattr(self.student_model.args, "__dict__") else {}
        hyp_dict.setdefault("box", 7.5)
        hyp_dict.setdefault("cls", 0.5)
        hyp_dict.setdefault("dfl", 1.5)
        self.student_model.args = SimpleNamespace(**hyp_dict)
        self.criterion = LossClass(self.student_model)

        self.update_every_frames, self.updates_per_event = resolve_update_schedule(
            max_updates_per_frame=config.max_updates_per_frame,
            update_every_frames=config.update_every_frames,
            updates_per_event=config.updates_per_event,
        )
        self.buffer = ReplayBuffer(max_size=config.buffer_size, rng=self.rng)
        self.prev_selected_box: Optional[Tuple[float, float, float, float]] = None
        self.persistence_state: Optional[PersistenceState] = None
        self.motion_gate_enabled = config.persistence_frames > 1
        self.accepted_frames = 0
        self.updated_frames = 0
        self.update_events = 0
        self.optimizer_updates = 0

    def warmup(self, images: Sequence[Path], warmup_n: int) -> None:
        for img_path in list(images)[: max(0, int(warmup_n))]:
            _ = self.teacher_yolo.predict(
                source=str(img_path),
                device=self.config.device,
                conf=self.config.infer_conf,
                iou=self.config.iou,
                verbose=False,
                save=False,
            )

    def step(self, img_path: Path, frame_idx: int, allow_update: bool = True) -> ShadowStepResult:
        cfg = self.config
        frame_t0 = time.time()
        with Image.open(img_path) as im:
            w, h = im.size

        self.teacher_model.eval()
        raw_teacher_top1, teacher_candidates, teacher_lat_ms = predict_teacher_candidates_wrapper(
            yolo_wrapper=self.teacher_yolo,
            source=str(img_path),
            device=cfg.device,
            conf=cfg.infer_conf,
            iou=cfg.iou,
            topk=cfg.teacher_topk,
            conf_floor=cfg.teacher_candidate_conf_floor,
            allow_top1_fallback=cfg.teacher_topk <= 1,
        )
        del raw_teacher_top1
        selection = select_teacher_candidate(
            candidates=teacher_candidates,
            prev_reference_box=self.prev_selected_box,
            img_w=w,
            img_h=h,
            max_center_shift_frac=cfg.max_center_shift_frac,
            max_area_ratio=cfg.max_area_ratio,
            score_mode=cfg.teacher_candidate_score_mode,
            conf_weight=cfg.teacher_candidate_conf_weight,
            temporal_weight=cfg.teacher_candidate_temporal_weight,
            min_score=cfg.teacher_candidate_min_score,
        )
        teacher_top1 = selection.selected

        passed_base_gate, temporal_iou = evaluate_base_gate(
            top1=teacher_top1,
            prev_teacher_box=self.prev_selected_box,
            img_w=w,
            img_h=h,
            conf_thresh=cfg.teacher_conf_thresh,
            min_area_frac=cfg.min_area_frac,
            max_area_frac=cfg.max_area_frac,
            border_margin_frac=cfg.border_margin_frac,
            temporal_iou_gate=cfg.temporal_iou_gate,
        )
        passed_motion_gate, center_shift_frac, area_ratio = evaluate_motion_gate(
            top1=teacher_top1,
            prev_teacher_box=self.prev_selected_box,
            img_w=w,
            img_h=h,
            max_center_shift_frac=cfg.max_center_shift_frac,
            max_area_ratio=cfg.max_area_ratio,
            enabled=self.motion_gate_enabled,
        )
        self.persistence_state, persistence_count, persistence_iou, passed_persistence_gate = update_persistence_state(
            state=self.persistence_state,
            top1=teacher_top1,
            candidate_valid=bool(passed_base_gate and passed_motion_gate),
            persistence_frames=cfg.persistence_frames,
            persistence_iou=cfg.persistence_iou,
        )
        accepted = bool(passed_base_gate and passed_motion_gate and passed_persistence_gate)
        if accepted and teacher_top1 is not None:
            self.accepted_frames += 1
            self.buffer.add(
                ReplayEntry(
                    frame_idx=int(frame_idx),
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
        buffer_warm = len(self.buffer) >= cfg.min_buffer_before_update
        update_event_triggered = int(
            bool(allow_update)
            and buffer_warm
            and should_trigger_update_event(frame_idx, self.update_every_frames)
        )
        if update_event_triggered:
            self.update_events += 1
            for _ in range(self.updates_per_event):
                target_entries = self.buffer.sample(
                    batch_size=cfg.update_batch_size,
                    mode=cfg.buffer_sample_mode,
                )
                if not target_entries:
                    break

                apply_freeze_policy(self.student_model, self.layers, self.unfrozen_indices)
                update_t0 = time.time()
                batch = build_training_batch(
                    target_entries=target_entries,
                    imgsz=cfg.imgsz,
                    device=cfg.device,
                    rng=self.rng,
                )
                det_loss, total_loss = run_detection_update(
                    student_model=self.student_model,
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    optim_params=self.optim_params,
                    batch=batch,
                    grad_clip=cfg.grad_clip,
                )
                update_latency_ms += (time.time() - update_t0) * 1000.0
                update_teacher_ema(
                    teacher_model=self.teacher_model,
                    student_model=self.student_model,
                    decay=cfg.ema_decay,
                )
                updates_this_frame += 1
                self.optimizer_updates += 1
                last_det_loss = float(det_loss)
                last_total_loss = float(total_loss)
                batch_size_used = len(target_entries)
                num_pseudo_boxes_used += len(target_entries)
        if updates_this_frame > 0:
            self.updated_frames += 1

        self.student_model.eval()
        student_post_top1, student_post_lat_ms = predict_top1_wrapper(
            yolo_wrapper=self.student_yolo,
            source=str(img_path),
            device=cfg.device,
            conf=cfg.infer_conf,
            iou=cfg.iou,
        )
        frame_latency_ms = (time.time() - frame_t0) * 1000.0
        self.prev_selected_box = teacher_top1.xyxy if teacher_top1 is not None else None

        log = FrameLog(
            frame=int(frame_idx),
            path=str(img_path),
            teacher_conf=float(teacher_top1.conf) if teacher_top1 is not None else 0.0,
            accepted=int(accepted),
            accepted_final=int(accepted),
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
            buffer_size=len(self.buffer),
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
        return ShadowStepResult(log=log, teacher_top1=teacher_top1, student_top1=student_post_top1)

    def save_checkpoint(self, out_path: Path, checkpoint_type: str, frame_idx: int) -> Path:
        self.student_model.eval()
        return save_student_weights_checkpoint(
            yolo_wrapper=self.student_yolo,
            student_model=self.student_model,
            out_path=out_path,
            checkpoint_type=checkpoint_type,
            frame_idx=frame_idx,
        )


def path_size_mb(path: Optional[Path]) -> float:
    if path is None or not path.exists():
        return float("nan")
    return float(path.stat().st_size) / (1024.0 * 1024.0)


def percentile_or_nan(values: Sequence[float], q: float) -> float:
    finite = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float32)
    return float(np.percentile(finite, q)) if finite.size else float("nan")


def streak_lengths(flags: Sequence[bool]) -> List[int]:
    lengths: List[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current > 0:
            lengths.append(current)
            current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def clipped_box_area_frac(box: Tuple[float, float, float, float], width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    cx1 = min(max(float(x1), 0.0), float(width))
    cy1 = min(max(float(y1), 0.0), float(height))
    cx2 = min(max(float(x2), 0.0), float(width))
    cy2 = min(max(float(y2), 0.0), float(height))
    area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    return float(area / max(1.0, float(width) * float(height)))


def prediction_record_from_top1(
    frame_idx: int,
    path: Path,
    model_name: str,
    width: int,
    height: int,
    top1: Optional[Top1Det],
    latency_ms: float,
    prev_record: Optional[PredictionRecord],
    args: argparse.Namespace,
) -> PredictionRecord:
    top1_conf = 0.0
    top1_cls = -1
    xyxy = (float("nan"), float("nan"), float("nan"), float("nan"))
    has_detection = top1 is not None
    box_area_frac = 0.0
    tiny_box = False
    large_box = False
    border_touching = False
    if top1 is not None:
        top1_conf = float(top1.conf)
        top1_cls = int(top1.cls_id)
        xyxy = tuple(float(v) for v in top1.xyxy)
        box_area_frac = clipped_box_area_frac(xyxy, width, height)
        tiny_box = bool(box_area_frac <= float(args.tiny_box_frac))
        large_box = bool(box_area_frac >= float(args.large_box_frac))
        border_touching = bool(near_border(xyxy, width, height, float(args.reliability_border_margin_frac)))

    center_shift = float("nan")
    area_ratio = float("nan")
    box_jump = False
    if (
        prev_record is not None
        and has_detection
        and prev_record.has_detection
        and math.isfinite(prev_record.x1)
    ):
        prev_box = (prev_record.x1, prev_record.y1, prev_record.x2, prev_record.y2)
        center_shift = center_shift_fraction(xyxy, prev_box, img_w=width, img_h=height)
        area_ratio = symmetric_area_ratio(xyxy, prev_box)
        box_jump = bool(math.isfinite(center_shift) and center_shift >= float(args.box_jump_center_frac))

    return PredictionRecord(
        frame=int(frame_idx),
        path=str(path),
        model=str(model_name),
        width=int(width),
        height=int(height),
        top1_conf=float(top1_conf),
        top1_cls=int(top1_cls),
        x1=float(xyxy[0]),
        y1=float(xyxy[1]),
        x2=float(xyxy[2]),
        y2=float(xyxy[3]),
        has_detection=bool(has_detection),
        box_area_frac=float(box_area_frac),
        tiny_box=bool(tiny_box),
        large_box=bool(large_box),
        border_touching=bool(border_touching),
        box_jump=bool(box_jump),
        center_shift_frac=float(center_shift),
        area_ratio=float(area_ratio),
        latency_ms=float(latency_ms),
    )


def reliability_metrics(records: Sequence[PredictionRecord], args: argparse.Namespace) -> Dict[str, float]:
    n = len(records)
    if n == 0:
        return {
            "n_frames": 0,
            "det_rate_at_conf": float("nan"),
            "mean_top1_conf": float("nan"),
            "median_top1_conf": float("nan"),
            "weird_box_rate": float("nan"),
            "high_conf_weird_rate": float("nan"),
            "box_jump_rate": float("nan"),
            "max_bad_streak": 0,
            "max_good_streak": 0,
            "mean_good_streak_len": float("nan"),
            "num_good_streaks": 0,
            "mean_inference_latency_ms": float("nan"),
            "p90_inference_latency_ms": float("nan"),
        }
    weird_flags = [bool(r.tiny_box or r.large_box or r.border_touching) for r in records]
    high_conf_weird = [bool(r.top1_conf >= float(args.good_conf) and weird) for r, weird in zip(records, weird_flags)]
    box_jump_flags = [bool(r.box_jump) for r in records]
    det_flags = [bool(r.top1_conf >= float(args.conf)) for r in records]
    bad_flags = [
        bool((not r.has_detection) or r.top1_conf < float(args.conf) or weird or r.box_jump)
        for r, weird in zip(records, weird_flags)
    ]
    good_flags = [
        bool(r.has_detection and r.top1_conf >= float(args.conf) and (not weird) and (not r.box_jump))
        for r, weird in zip(records, weird_flags)
    ]
    bad_streaks = streak_lengths(bad_flags)
    good_streaks = streak_lengths(good_flags)
    latencies = [r.latency_ms for r in records]
    confs = np.asarray([r.top1_conf for r in records], dtype=np.float32)
    return {
        "n_frames": float(n),
        "det_rate_at_conf": float(np.mean(np.asarray(det_flags, dtype=np.float32))),
        "mean_top1_conf": float(np.mean(confs)),
        "median_top1_conf": float(np.median(confs)),
        "weird_box_rate": float(np.mean(np.asarray(weird_flags, dtype=np.float32))),
        "high_conf_weird_rate": float(np.mean(np.asarray(high_conf_weird, dtype=np.float32))),
        "box_jump_rate": float(np.mean(np.asarray(box_jump_flags, dtype=np.float32))) if n > 1 else 0.0,
        "max_bad_streak": float(max(bad_streaks) if bad_streaks else 0),
        "max_good_streak": float(max(good_streaks) if good_streaks else 0),
        "mean_good_streak_len": mean_or_nan([float(v) for v in good_streaks]),
        "num_good_streaks": float(len(good_streaks)),
        "mean_inference_latency_ms": mean_or_nan(latencies),
        "p90_inference_latency_ms": percentile_or_nan(latencies, 90),
    }


def promotion_score(metrics: Mapping[str, float]) -> float:
    return float(
        float(metrics.get("det_rate_at_conf", 0.0))
        - 0.5 * float(metrics.get("weird_box_rate", 0.0))
        - 0.75 * float(metrics.get("box_jump_rate", 0.0))
        - 0.5 * float(metrics.get("high_conf_weird_rate", 0.0))
        - 0.002 * float(metrics.get("max_bad_streak", 0.0))
        + 0.001 * float(metrics.get("max_good_streak", 0.0))
    )


def evaluate_model_window(
    model_name: str,
    yolo_wrapper: YOLO,
    image_paths: Sequence[Path],
    image_sizes: Mapping[Path, Tuple[int, int]],
    args: argparse.Namespace,
) -> Tuple[List[PredictionRecord], Dict[str, float]]:
    records: List[PredictionRecord] = []
    prev_record: Optional[PredictionRecord] = None
    for frame_offset, img_path in enumerate(image_paths):
        top1, latency_ms = predict_top1_wrapper(
            yolo_wrapper=yolo_wrapper,
            source=str(img_path),
            device=str(args.device),
            conf=float(args.infer_conf),
            iou=float(args.iou),
        )
        width, height = image_sizes[img_path]
        record = prediction_record_from_top1(
            frame_idx=frame_offset,
            path=img_path,
            model_name=model_name,
            width=width,
            height=height,
            top1=top1,
            latency_ms=latency_ms,
            prev_record=prev_record,
            args=args,
        )
        records.append(record)
        prev_record = record
    metrics = reliability_metrics(records, args)
    metrics["promotion_score"] = promotion_score(metrics)
    return records, metrics


def cuda_resource_metrics(device: str) -> Dict[str, float]:
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return {
            "peak_cuda_allocated_mb": float("nan"),
            "peak_cuda_reserved_mb": float("nan"),
            "current_cuda_allocated_mb": float("nan"),
            "current_cuda_reserved_mb": float("nan"),
        }
    dev = torch.device(device)
    return {
        "peak_cuda_allocated_mb": float(torch.cuda.max_memory_allocated(dev) / (1024.0 * 1024.0)),
        "peak_cuda_reserved_mb": float(torch.cuda.max_memory_reserved(dev) / (1024.0 * 1024.0)),
        "current_cuda_allocated_mb": float(torch.cuda.memory_allocated(dev) / (1024.0 * 1024.0)),
        "current_cuda_reserved_mb": float(torch.cuda.memory_reserved(dev) / (1024.0 * 1024.0)),
    }


def should_promote(
    active_metrics: Mapping[str, float],
    shadow_metrics: Mapping[str, float],
    resource_metrics: Mapping[str, float],
    promotions_so_far: int,
    args: argparse.Namespace,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    active_score = promotion_score(active_metrics)
    shadow_score = promotion_score(shadow_metrics)
    shadow_det = float(shadow_metrics["det_rate_at_conf"])
    active_det = float(active_metrics["det_rate_at_conf"])

    relaxed_floor = (
        float(args.promotion_det_margin) < 0.0
        and shadow_det >= active_det + float(args.promotion_det_margin)
        and shadow_score > active_score
    )
    checks = [
        (
            promotions_so_far < int(args.max_promotions),
            f"max_promotions_reached({promotions_so_far}/{int(args.max_promotions)})",
        ),
        (
            shadow_det >= float(args.min_promotion_det_rate) or relaxed_floor,
            (
                "det_floor"
                f"(shadow={shadow_det:.4f}, min={float(args.min_promotion_det_rate):.4f}, relaxed={int(relaxed_floor)})"
            ),
        ),
        (
            shadow_det >= active_det + float(args.promotion_det_margin),
            f"det_margin(shadow={shadow_det:.4f}, active={active_det:.4f}, margin={float(args.promotion_det_margin):.4f})",
        ),
        (
            float(shadow_metrics["weird_box_rate"]) <= float(active_metrics["weird_box_rate"]) + 1e-12,
            (
                "weird_not_improved"
                f"(shadow={float(shadow_metrics['weird_box_rate']):.4f}, active={float(active_metrics['weird_box_rate']):.4f})"
            ),
        ),
        (
            float(shadow_metrics["box_jump_rate"]) <= float(active_metrics["box_jump_rate"]) + 1e-12,
            (
                "box_jump_not_improved"
                f"(shadow={float(shadow_metrics['box_jump_rate']):.4f}, active={float(active_metrics['box_jump_rate']):.4f})"
            ),
        ),
        (
            float(shadow_metrics["max_bad_streak"]) <= float(active_metrics["max_bad_streak"]) + 1e-12,
            (
                "max_bad_streak_not_improved"
                f"(shadow={float(shadow_metrics['max_bad_streak']):.0f}, active={float(active_metrics['max_bad_streak']):.0f})"
            ),
        ),
        (
            float(shadow_metrics["max_good_streak"]) >= float(active_metrics["max_good_streak"]) - 1e-12
            or shadow_score > active_score,
            (
                "max_good_streak_or_score"
                f"(shadow_good={float(shadow_metrics['max_good_streak']):.0f}, active_good={float(active_metrics['max_good_streak']):.0f})"
            ),
        ),
        (
            shadow_score > active_score,
            f"score_not_improved(shadow={shadow_score:.6f}, active={active_score:.6f})",
        ),
    ]
    if float(args.max_promotion_latency_ms) > 0.0:
        checks.append(
            (
                float(shadow_metrics["mean_inference_latency_ms"]) <= float(args.max_promotion_latency_ms),
                (
                    "latency_budget"
                    f"(shadow={float(shadow_metrics['mean_inference_latency_ms']):.3f}, max={float(args.max_promotion_latency_ms):.3f})"
                ),
            )
        )
    if float(args.max_promotion_memory_mb) > 0.0:
        peak_reserved = float(resource_metrics.get("peak_cuda_reserved_mb", float("nan")))
        checks.append(
            (
                math.isfinite(peak_reserved) and peak_reserved <= float(args.max_promotion_memory_mb),
                f"memory_budget(peak_reserved={peak_reserved:.1f}, max={float(args.max_promotion_memory_mb):.1f})",
            )
        )

    for passed, fail_reason in checks:
        if not passed:
            reasons.append(fail_reason)
    return len(reasons) == 0, reasons


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_metrics_row(frame: int, model_name: str, metrics: Mapping[str, float], checkpoint: str) -> Dict[str, object]:
    row: Dict[str, object] = {
        "frame": int(frame),
        "model": model_name,
        "checkpoint": checkpoint,
    }
    for key, value in metrics.items():
        row[key] = value
    return row


def summarize_resource_row(
    frame: int,
    event: str,
    active_latencies: Sequence[float],
    shadow_logs: Sequence[FrameLog],
    checkpoint_path: Optional[Path],
    trainable_params: int,
    device: str,
) -> Dict[str, object]:
    active_mean = mean_or_nan(active_latencies)
    update_latencies = [row.update_latency_ms for row in shadow_logs if row.update_latency_ms > 0.0]
    teacher_latencies = [row.teacher_latency_ms for row in shadow_logs]
    student_latencies = [row.student_post_latency_ms for row in shadow_logs]
    resource = cuda_resource_metrics(device)
    return {
        "frame": int(frame),
        "event": event,
        "mean_active_inference_latency_ms": active_mean,
        "p90_active_inference_latency_ms": percentile_or_nan(active_latencies, 90),
        "estimated_fps": (1000.0 / active_mean) if math.isfinite(active_mean) and active_mean > 0.0 else float("nan"),
        "mean_shadow_teacher_latency_ms": mean_or_nan(teacher_latencies),
        "mean_shadow_student_latency_ms": mean_or_nan(student_latencies),
        "mean_shadow_update_latency_ms": mean_or_nan(update_latencies),
        "peak_cuda_allocated_mb": resource["peak_cuda_allocated_mb"],
        "peak_cuda_reserved_mb": resource["peak_cuda_reserved_mb"],
        "current_cuda_allocated_mb": resource["current_cuda_allocated_mb"],
        "current_cuda_reserved_mb": resource["current_cuda_reserved_mb"],
        "checkpoint_size_mb": path_size_mb(checkpoint_path),
        "trainable_params": int(trainable_params),
    }


def save_manager_plots(
    out_dir: Path,
    manager_rows: Sequence[Mapping[str, object]],
    window_rows: Sequence[Mapping[str, object]],
    resource_rows: Sequence[Mapping[str, object]],
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    event_frames = np.asarray(
        [int(row["frame"]) for row in window_rows if row["model"] == "active"],
        dtype=np.int32,
    )
    active_rows = [row for row in window_rows if row["model"] == "active"]
    shadow_rows = [row for row in window_rows if row["model"] == "shadow"]

    def arr(rows: Sequence[Mapping[str, object]], key: str) -> np.ndarray:
        return np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=np.float32)

    save_plot_lines(
        event_frames,
        [arr(active_rows, "promotion_score"), arr(shadow_rows, "promotion_score")],
        ["active", "shadow"],
        "Promotion Score",
        "frame",
        "score",
        plots_dir / "plot_promotion_score.png",
    )
    save_plot_lines(
        event_frames,
        [arr(active_rows, "det_rate_at_conf"), arr(shadow_rows, "det_rate_at_conf")],
        ["active", "shadow"],
        "Active vs Shadow Det Rate",
        "frame",
        "det_rate",
        plots_dir / "plot_active_vs_shadow_det_rate.png",
    )
    save_plot_lines(
        event_frames,
        [arr(active_rows, "weird_box_rate"), arr(shadow_rows, "weird_box_rate")],
        ["active", "shadow"],
        "Active vs Shadow Weird Rate",
        "frame",
        "weird_rate",
        plots_dir / "plot_active_vs_shadow_weird_rate.png",
    )
    save_plot_lines(
        event_frames,
        [arr(active_rows, "box_jump_rate"), arr(shadow_rows, "box_jump_rate")],
        ["active", "shadow"],
        "Active vs Shadow Box-Jump Rate",
        "frame",
        "box_jump_rate",
        plots_dir / "plot_active_vs_shadow_box_jump.png",
    )
    save_plot_lines(
        event_frames,
        [
            arr(active_rows, "max_bad_streak"),
            arr(shadow_rows, "max_bad_streak"),
            arr(active_rows, "max_good_streak"),
            arr(shadow_rows, "max_good_streak"),
        ],
        ["active_bad", "shadow_bad", "active_good", "shadow_good"],
        "Good and Bad Streaks",
        "frame",
        "frames",
        plots_dir / "plot_good_bad_streaks.png",
    )

    frames = np.asarray([int(row["frame"]) for row in manager_rows], dtype=np.int32)
    active_latency = np.asarray([float(row["active_latency_ms"]) for row in manager_rows], dtype=np.float32)
    update_latency = np.asarray([float(row["shadow_update_latency_ms"]) for row in manager_rows], dtype=np.float32)
    save_plot_lines(
        frames,
        [active_latency, update_latency],
        ["active_infer", "shadow_update"],
        "Latency",
        "frame",
        "ms",
        plots_dir / "plot_latency.png",
    )
    resource_frames = np.asarray([int(row["frame"]) for row in resource_rows], dtype=np.int32)
    save_plot_lines(
        resource_frames,
        [
            np.asarray([float(row.get("peak_cuda_allocated_mb", float("nan"))) for row in resource_rows], dtype=np.float32),
            np.asarray([float(row.get("peak_cuda_reserved_mb", float("nan"))) for row in resource_rows], dtype=np.float32),
        ],
        ["allocated", "reserved"],
        "CUDA Memory",
        "frame",
        "MB",
        plots_dir / "plot_memory.png",
    )


def load_yolo(weights: str, device: str) -> YOLO:
    model = YOLO(weights)
    model.model.to(device)
    model.model.eval()
    return model


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def maybe_progress(total: int, disabled: bool):
    if disabled or tqdm is None:
        return None
    return tqdm(total=total, desc="Shadow ODAD", dynamic_ncols=True)


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    out_dir = Path(args.output)
    checkpoints_dir = out_dir / "checkpoints"
    promoted_dir = out_dir / "promoted"
    for path in [out_dir, checkpoints_dir, promoted_dir, out_dir / "plots"]:
        path.mkdir(parents=True, exist_ok=True)

    images = list_test_images(Path(args.dataset))
    if int(args.max_frames) > 0:
        images = images[: int(args.max_frames)]
    if not images:
        raise RuntimeError("No images available after applying --max-frames.")
    image_sizes: Dict[Path, Tuple[int, int]] = {}
    for img_path in images:
        with Image.open(img_path) as im:
            image_sizes[img_path] = im.size

    active_weights = str(args.active_weights)
    shadow_init_weights = str(args.shadow_init_weights or active_weights)
    active_yolo = load_yolo(active_weights, str(args.device))
    shadow_cfg = shadow_config_from_args(args)
    shadow = ShadowODADLearner(shadow_init_weights, shadow_cfg, seed=int(args.seed))
    shadow.warmup(images, int(args.warmup))
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(str(args.device)))

    rolling_window: deque[Path] = deque(maxlen=max(1, int(args.eval_window)))
    recent_shadow_records: deque[PredictionRecord] = deque(maxlen=max(20, min(100, int(args.eval_window))))
    manager_rows: List[Dict[str, object]] = []
    promotion_rows: List[Dict[str, object]] = []
    window_metric_rows: List[Dict[str, object]] = []
    resource_rows: List[Dict[str, object]] = []
    active_frame_latencies: List[float] = []
    shadow_logs: List[FrameLog] = []
    active_prev_record: Optional[PredictionRecord] = None
    shadow_recent_prev_record: Optional[PredictionRecord] = None
    promotions = 0
    rollbacks = 0
    consecutive_failures = 0
    last_candidate_checkpoint: Optional[Path] = None
    last_promoted_checkpoint: Optional[Path] = None

    progress = maybe_progress(len(images), bool(args.no_progress))
    for idx, img_path in enumerate(images):
        width, height = image_sizes[img_path]
        active_top1, active_latency_ms = predict_top1_wrapper(
            yolo_wrapper=active_yolo,
            source=str(img_path),
            device=str(args.device),
            conf=float(args.infer_conf),
            iou=float(args.iou),
        )
        active_record = prediction_record_from_top1(
            frame_idx=idx,
            path=img_path,
            model_name="active_stream",
            width=width,
            height=height,
            top1=active_top1,
            latency_ms=active_latency_ms,
            prev_record=active_prev_record,
            args=args,
        )
        active_prev_record = active_record
        active_frame_latencies.append(float(active_latency_ms))
        rolling_window.append(img_path)

        update_paused = False
        throttle_metrics: Dict[str, float] = {}
        if bool(args.enable_update_throttle) and len(recent_shadow_records) >= min(20, recent_shadow_records.maxlen or 20):
            throttle_metrics = reliability_metrics(list(recent_shadow_records), args)
            update_paused = bool(
                float(throttle_metrics["weird_box_rate"]) > float(args.throttle_weird_rate)
                or float(throttle_metrics["box_jump_rate"]) > float(args.throttle_box_jump_rate)
            )
        allow_shadow_update = bool(idx >= int(args.shadow_update_start_frame) and not update_paused)
        shadow_step = shadow.step(img_path=img_path, frame_idx=idx, allow_update=allow_shadow_update)
        shadow_logs.append(shadow_step.log)

        shadow_record = prediction_record_from_top1(
            frame_idx=idx,
            path=img_path,
            model_name="shadow_stream",
            width=width,
            height=height,
            top1=shadow_step.student_top1,
            latency_ms=shadow_step.log.student_post_latency_ms,
            prev_record=shadow_recent_prev_record,
            args=args,
        )
        shadow_recent_prev_record = shadow_record
        recent_shadow_records.append(shadow_record)

        manager_rows.append(
            {
                "frame": idx,
                "path": str(img_path),
                "active_conf": active_record.top1_conf,
                "active_latency_ms": active_latency_ms,
                "shadow_teacher_conf": shadow_step.log.teacher_conf,
                "shadow_student_conf": shadow_step.log.student_post_conf,
                "shadow_teacher_latency_ms": shadow_step.log.teacher_latency_ms,
                "shadow_student_latency_ms": shadow_step.log.student_post_latency_ms,
                "shadow_update_latency_ms": shadow_step.log.update_latency_ms,
                "shadow_update_applied": shadow_step.log.update_applied,
                "shadow_updates_this_frame": shadow_step.log.updates_this_frame,
                "shadow_buffer_size": shadow_step.log.buffer_size,
                "shadow_accepted": shadow_step.log.accepted_final,
                "update_paused": int(update_paused),
                "throttle_weird_rate": throttle_metrics.get("weird_box_rate", float("nan")),
                "throttle_box_jump_rate": throttle_metrics.get("box_jump_rate", float("nan")),
                "active_weights": active_weights,
            }
        )

        should_eval = (
            len(rolling_window) >= min(int(args.eval_window), len(images))
            and int(args.promotion_every_frames) > 0
            and (idx + 1) % int(args.promotion_every_frames) == 0
        )
        if should_eval:
            candidate_checkpoint = shadow.save_checkpoint(
                checkpoints_dir / f"shadow_frame_{idx + 1:06d}.pt",
                checkpoint_type="shadow_candidate",
                frame_idx=idx,
            )
            last_candidate_checkpoint = candidate_checkpoint
            window_paths = list(rolling_window)
            active_window_records, active_metrics = evaluate_model_window(
                model_name="active",
                yolo_wrapper=active_yolo,
                image_paths=window_paths,
                image_sizes=image_sizes,
                args=args,
            )
            del active_window_records
            shadow_window_records, shadow_metrics = evaluate_model_window(
                model_name="shadow",
                yolo_wrapper=shadow.student_yolo,
                image_paths=window_paths,
                image_sizes=image_sizes,
                args=args,
            )
            del shadow_window_records
            resource = summarize_resource_row(
                frame=idx + 1,
                event="promotion_eval",
                active_latencies=active_frame_latencies,
                shadow_logs=shadow_logs,
                checkpoint_path=candidate_checkpoint,
                trainable_params=shadow.trainable_params,
                device=str(args.device),
            )
            resource_rows.append(resource)
            active_metrics["promotion_score"] = promotion_score(active_metrics)
            shadow_metrics["promotion_score"] = promotion_score(shadow_metrics)
            window_metric_rows.append(flatten_metrics_row(idx + 1, "active", active_metrics, active_weights))
            window_metric_rows.append(flatten_metrics_row(idx + 1, "shadow", shadow_metrics, str(candidate_checkpoint)))

            promote, fail_reasons = should_promote(
                active_metrics=active_metrics,
                shadow_metrics=shadow_metrics,
                resource_metrics=resource,
                promotions_so_far=promotions,
                args=args,
            )
            event = "promote" if promote else "reject"
            promoted_checkpoint = ""
            if promote:
                promoted_path = shadow.save_checkpoint(
                    promoted_dir / f"active_frame_{idx + 1:06d}.pt",
                    checkpoint_type="promoted_active",
                    frame_idx=idx,
                )
                last_promoted_checkpoint = promoted_path
                promoted_checkpoint = str(promoted_path)
                active_weights = str(promoted_path)
                del active_yolo
                active_yolo = load_yolo(active_weights, str(args.device))
                promotions += 1
                consecutive_failures = 0
                del shadow
                cleanup_cuda()
                shadow = ShadowODADLearner(active_weights, shadow_cfg, seed=int(args.seed) + idx + promotions)
                shadow.warmup(window_paths[-min(len(window_paths), int(args.warmup)) :], int(args.warmup))
                recent_shadow_records.clear()
                shadow_recent_prev_record = None
            else:
                consecutive_failures += 1
                rollback_now = bool(
                    int(args.rollback_on_failures) > 0
                    and consecutive_failures >= int(args.rollback_on_failures)
                    and promotions < int(args.max_promotions)
                )
                if rollback_now:
                    event = "rollback"
                    rollbacks += 1
                    consecutive_failures = 0
                    del shadow
                    cleanup_cuda()
                    shadow = ShadowODADLearner(active_weights, shadow_cfg, seed=int(args.seed) + idx + rollbacks)
                    shadow.warmup(window_paths[-min(len(window_paths), int(args.warmup)) :], int(args.warmup))
                    recent_shadow_records.clear()
                    shadow_recent_prev_record = None

            promotion_rows.append(
                {
                    "frame": idx + 1,
                    "event": event,
                    "promoted": int(promote),
                    "active_score": active_metrics["promotion_score"],
                    "shadow_score": shadow_metrics["promotion_score"],
                    "active_det_rate": active_metrics["det_rate_at_conf"],
                    "shadow_det_rate": shadow_metrics["det_rate_at_conf"],
                    "active_weird_rate": active_metrics["weird_box_rate"],
                    "shadow_weird_rate": shadow_metrics["weird_box_rate"],
                    "active_box_jump_rate": active_metrics["box_jump_rate"],
                    "shadow_box_jump_rate": shadow_metrics["box_jump_rate"],
                    "active_max_bad_streak": active_metrics["max_bad_streak"],
                    "shadow_max_bad_streak": shadow_metrics["max_bad_streak"],
                    "active_max_good_streak": active_metrics["max_good_streak"],
                    "shadow_max_good_streak": shadow_metrics["max_good_streak"],
                    "candidate_checkpoint": str(candidate_checkpoint),
                    "promoted_checkpoint": promoted_checkpoint,
                    "fail_reasons": ";".join(fail_reasons),
                    "consecutive_failures": consecutive_failures,
                    "promotions": promotions,
                    "rollbacks": rollbacks,
                }
            )

        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                {
                    "promote": promotions,
                    "rollback": rollbacks,
                    "buf": shadow_step.log.buffer_size,
                    "upd": shadow_step.log.update_applied,
                },
                refresh=False,
            )

    if progress is not None:
        progress.close()

    resource_rows.append(
        summarize_resource_row(
            frame=len(images),
            event="final",
            active_latencies=active_frame_latencies,
            shadow_logs=shadow_logs,
            checkpoint_path=last_candidate_checkpoint,
            trainable_params=shadow.trainable_params,
            device=str(args.device),
        )
    )

    write_csv(
        out_dir / "manager_log.csv",
        manager_rows,
        [
            "frame",
            "path",
            "active_conf",
            "active_latency_ms",
            "shadow_teacher_conf",
            "shadow_student_conf",
            "shadow_teacher_latency_ms",
            "shadow_student_latency_ms",
            "shadow_update_latency_ms",
            "shadow_update_applied",
            "shadow_updates_this_frame",
            "shadow_buffer_size",
            "shadow_accepted",
            "update_paused",
            "throttle_weird_rate",
            "throttle_box_jump_rate",
            "active_weights",
        ],
    )
    write_csv(
        out_dir / "promotion_events.csv",
        promotion_rows,
        [
            "frame",
            "event",
            "promoted",
            "active_score",
            "shadow_score",
            "active_det_rate",
            "shadow_det_rate",
            "active_weird_rate",
            "shadow_weird_rate",
            "active_box_jump_rate",
            "shadow_box_jump_rate",
            "active_max_bad_streak",
            "shadow_max_bad_streak",
            "active_max_good_streak",
            "shadow_max_good_streak",
            "candidate_checkpoint",
            "promoted_checkpoint",
            "fail_reasons",
            "consecutive_failures",
            "promotions",
            "rollbacks",
        ],
    )
    metric_fields = [
        "frame",
        "model",
        "checkpoint",
        "n_frames",
        "det_rate_at_conf",
        "mean_top1_conf",
        "median_top1_conf",
        "weird_box_rate",
        "high_conf_weird_rate",
        "box_jump_rate",
        "max_bad_streak",
        "max_good_streak",
        "mean_good_streak_len",
        "num_good_streaks",
        "mean_inference_latency_ms",
        "p90_inference_latency_ms",
        "promotion_score",
    ]
    write_csv(out_dir / "active_vs_shadow_window_metrics.csv", window_metric_rows, metric_fields)
    write_csv(
        out_dir / "resource_metrics.csv",
        resource_rows,
        [
            "frame",
            "event",
            "mean_active_inference_latency_ms",
            "p90_active_inference_latency_ms",
            "estimated_fps",
            "mean_shadow_teacher_latency_ms",
            "mean_shadow_student_latency_ms",
            "mean_shadow_update_latency_ms",
            "peak_cuda_allocated_mb",
            "peak_cuda_reserved_mb",
            "current_cuda_allocated_mb",
            "current_cuda_reserved_mb",
            "checkpoint_size_mb",
            "trainable_params",
        ],
    )
    save_manager_plots(out_dir, manager_rows, window_metric_rows, resource_rows)

    final_resource = resource_rows[-1]
    last_eval_rows = window_metric_rows[-2:] if len(window_metric_rows) >= 2 else []
    summary_lines = [
        "Shadow ODAD Manager Summary",
        "",
        "architecture:",
        "  active_model=stable inference model used on every stream frame",
        "  shadow_model=clean top-k2 ODAD learner updated in background",
        "  promotion=label-free same-window active-vs-shadow reliability/resource gate",
        "",
        f"active_initial_weights={args.active_weights}",
        f"shadow_initial_weights={shadow_init_weights}",
        f"active_final_weights={active_weights}",
        f"dataset={args.dataset}",
        f"frames={len(images)}",
        f"eval_window={int(args.eval_window)}",
        f"promotion_every_frames={int(args.promotion_every_frames)}",
        f"promotions={promotions}",
        f"rollbacks={rollbacks}",
        f"last_candidate_checkpoint={last_candidate_checkpoint if last_candidate_checkpoint is not None else 'n/a'}",
        f"last_promoted_checkpoint={last_promoted_checkpoint if last_promoted_checkpoint is not None else 'n/a'}",
        "",
        "resource_metrics:",
        format_metric("mean_active_inference_latency_ms", float(final_resource["mean_active_inference_latency_ms"]), ".3f", prefix="  "),
        format_metric("p90_active_inference_latency_ms", float(final_resource["p90_active_inference_latency_ms"]), ".3f", prefix="  "),
        format_metric("estimated_fps", float(final_resource["estimated_fps"]), ".3f", prefix="  "),
        format_metric("mean_shadow_teacher_latency_ms", float(final_resource["mean_shadow_teacher_latency_ms"]), ".3f", prefix="  "),
        format_metric("mean_shadow_student_latency_ms", float(final_resource["mean_shadow_student_latency_ms"]), ".3f", prefix="  "),
        format_metric("mean_shadow_update_latency_ms", float(final_resource["mean_shadow_update_latency_ms"]), ".3f", prefix="  "),
        format_metric("peak_cuda_allocated_mb", float(final_resource["peak_cuda_allocated_mb"]), ".1f", prefix="  "),
        format_metric("peak_cuda_reserved_mb", float(final_resource["peak_cuda_reserved_mb"]), ".1f", prefix="  "),
        format_metric("checkpoint_size_mb", float(final_resource["checkpoint_size_mb"]), ".3f", prefix="  "),
        f"  trainable_params={int(final_resource['trainable_params'])}",
        "",
        "promotion_rule:",
        "  score = det_rate - 0.5*weird - 0.75*box_jump - 0.5*high_conf_weird - 0.002*max_bad_streak + 0.001*max_good_streak",
        f"  min_promotion_det_rate={float(args.min_promotion_det_rate):.3f}",
        f"  promotion_det_margin={float(args.promotion_det_margin):.3f}",
        f"  max_promotion_latency_ms={float(args.max_promotion_latency_ms):.3f}",
        f"  max_promotion_memory_mb={float(args.max_promotion_memory_mb):.3f}",
        "",
        "outputs:",
        "  manager_log.csv",
        "  promotion_events.csv",
        "  active_vs_shadow_window_metrics.csv",
        "  resource_metrics.csv",
        "  checkpoints/",
        "  promoted/",
        "  plots/",
    ]
    if last_eval_rows:
        summary_lines.extend(["", "last_active_vs_shadow_window:"])
        for row in last_eval_rows:
            summary_lines.append(
                "  "
                f"{row['model']}: det={float(row['det_rate_at_conf']):.4f} "
                f"weird={float(row['weird_box_rate']):.4f} "
                f"high_conf_weird={float(row['high_conf_weird_rate']):.4f} "
                f"box_jump={float(row['box_jump_rate']):.4f} "
                f"max_bad={float(row['max_bad_streak']):.0f} "
                f"max_good={float(row['max_good_streak']):.0f} "
                f"score={float(row['promotion_score']):.6f}"
            )
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Done. Wrote shadow manager outputs to: {out_dir}")


if __name__ == "__main__":
    main()
