"""Additional modules used to integrate MAE features with Ultralytics."""
from __future__ import annotations

from .mae_detect import MaeDetect
from .mae_adapter import (
    check_divisible,
    interpolate_pos_embed,
    tokens_to_feature_map,
)

__all__ = [
    "MaeDetect",
    "check_divisible",
    "interpolate_pos_embed",
    "tokens_to_feature_map",
]
