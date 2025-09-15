"""Custom detection head wrapper for MAE-based YOLO models."""
from __future__ import annotations

import torch

from ultralytics.nn.modules.head import Detect


class MaeDetect(Detect):
    """Thin wrapper around :class:`~ultralytics.nn.modules.head.Detect` with explicit channel metadata."""

    def __init__(self, nc: int, ch: list[int], stride: list[int] | None = None) -> None:
        super().__init__(nc=nc, ch=tuple(ch))
        if stride is not None:
            stride_tensor = torch.as_tensor(stride, dtype=torch.float32)
            if stride_tensor.numel() != self.nl:
                raise ValueError(
                    f"stride list must match number of detection layers ({self.nl}); got {stride_tensor.numel()} entries"
                )
            self.stride = stride_tensor
