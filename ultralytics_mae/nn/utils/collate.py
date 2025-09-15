"""Collate functions for custom datasets."""
from __future__ import annotations

from typing import Dict, List

import torch


def orbitgen_collate_fn(batch: List[Dict]):
    """Collate a batch of OrbitGen samples for YOLO training."""
    imgs = torch.stack([b["img"] for b in batch], 0)
    masks = None
    if batch[0].get("mask") is not None:
        masks = torch.stack([b["mask"] for b in batch], 0)

    bboxes = []
    keypoints = []
    metas = []
    for i, sample in enumerate(batch):
        n = sample["bboxes"].shape[0]
        if n:
            batch_idx = torch.full((n, 1), i, dtype=sample["bboxes"].dtype)
            bboxes.append(torch.cat([batch_idx, sample["bboxes"]], dim=1))
            if sample["keypoints"].numel():
                keypoints.append(sample["keypoints"])
        metas.append(sample["meta"])

    bbox_tensor = torch.cat(bboxes, 0) if bboxes else torch.zeros((0, 6), dtype=torch.float32)
    if keypoints:
        kpt_tensor = torch.cat(keypoints, 0)
    else:
        kpt_tensor = torch.zeros((0, batch[0]["keypoints"].shape[1] if batch[0]["keypoints"].ndim > 1 else 0, 3), dtype=torch.float32)

    out = {"img": imgs, "bboxes": bbox_tensor, "keypoints": kpt_tensor, "meta": metas}
    if masks is not None:
        out["mask"] = masks
    return out
