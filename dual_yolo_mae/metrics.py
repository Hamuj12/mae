"""Metric utilities for Phase 1 training."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


def compute_box_mse(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> float:
    """Compute MSE between predicted and target boxes (xywh, normalised).

    Boxes are matched greedily by IoU; if no predictions are available the
    function returns ``float("nan")`` to signal the absence of matches.
    """

    if target_boxes.numel() == 0:
        return float("nan")
    if pred_boxes.numel() == 0:
        return float("nan")

    pred_xyxy = _xywh_to_xyxy(pred_boxes)
    target_xyxy = _xywh_to_xyxy(target_boxes)

    ious = _box_iou(pred_xyxy, target_xyxy)
    if ious.numel() == 0:
        return float("nan")

    mse_vals: List[float] = []
    assigned_preds: set[int] = set()
    for tgt_idx in range(target_boxes.shape[0]):
        iou_row = ious[:, tgt_idx]
        best_idx = int(torch.argmax(iou_row).item())
        if best_idx in assigned_preds:
            continue
        assigned_preds.add(best_idx)
        mse = torch.mean((pred_boxes[best_idx] - target_boxes[tgt_idx]) ** 2).item()
        mse_vals.append(mse)

    if not mse_vals:
        return float("nan")
    return float(sum(mse_vals) / len(mse_vals))


def compute_gate_histograms(gates: Iterable[np.ndarray], bins: int = 20) -> np.ndarray:
    """Compute histogram counts for gate activations."""

    flat: List[float] = []
    for g in gates:
        flat.extend(g.astype(float).ravel().tolist())
    if not flat:
        return np.array([])
    counts, _ = np.histogram(flat, bins=bins, range=(0.0, 1.0))
    return counts


def outputs_to_boxes(
    preds: Sequence[torch.Tensor],
    conf_threshold: float = 0.25,
    device: str | torch.device | None = None,
) -> List[torch.Tensor]:
    """Convert raw YOLO outputs into xywh predictions on [0, 1]."""

    outputs: List[torch.Tensor] = []
    batch_size = preds[0].shape[0]
    for batch_idx in range(batch_size):
        boxes: List[torch.Tensor] = []
        for scale_pred in preds:
            logits = scale_pred[batch_idx].permute(1, 2, 0)
            probs = torch.sigmoid(logits)
            obj = probs[..., 0]
            xywh = probs[..., 1:5]
            cls_scores, _ = torch.max(probs[..., 5:], dim=-1)
            conf = cls_scores * obj
            mask = conf > conf_threshold
            if mask.sum() == 0:
                continue
            boxes.append(xywh[mask])
        if boxes:
            outputs.append(torch.cat(boxes, dim=0).to(device) if device is not None else torch.cat(boxes, dim=0))
        else:
            outputs.append(torch.empty((0, 4), device=device or preds[0].device))
    return outputs


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = boxes.unbind(-1)
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


__all__ = ["compute_box_mse", "compute_gate_histograms", "outputs_to_boxes"]
