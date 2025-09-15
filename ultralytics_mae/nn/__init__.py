"""MAE extensions for Ultralytics YOLO models."""
from __future__ import annotations

from .backbones import MaeViTBackbone
from .modules import MaeDetect
from .necks import MaeSimpleFPN

__all__ = ["MaeViTBackbone", "MaeSimpleFPN", "MaeDetect", "register_ultralytics_modules"]


def register_ultralytics_modules() -> None:
    """Register custom MAE modules with Ultralytics' YAML parser."""
    try:
        from ultralytics.nn import tasks
    except Exception:  # pragma: no cover - Ultralytics optional during docs/tests
        return

    tasks.MaeViTBackbone = MaeViTBackbone
    tasks.MaeSimpleFPN = MaeSimpleFPN
    tasks.MaeDetect = MaeDetect
