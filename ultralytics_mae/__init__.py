"""Ultralytics MAE integration utilities."""
from __future__ import annotations

from .nn import MaeDetect, MaeSimpleFPN, MaeViTBackbone, register_ultralytics_modules

__all__ = ["MaeDetect", "MaeSimpleFPN", "MaeViTBackbone", "register_ultralytics_modules"]

# Ensure the MAE modules are registered as soon as the package is imported.
register_ultralytics_modules()
