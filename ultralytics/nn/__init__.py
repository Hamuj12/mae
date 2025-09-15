"""Wrapper for :mod:`ultralytics.nn` that adds MAE modules to YOLO's registry."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from types import ModuleType


def _load_base_nn() -> ModuleType:
    """Load the installed ``ultralytics.nn`` package."""
    search_paths = sys.path[1:]
    spec = importlib.machinery.PathFinder.find_spec(__name__, search_paths)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to locate the installed 'ultralytics.nn' package")

    module = importlib.util.module_from_spec(spec)
    sys.modules[__name__] = module
    spec.loader.exec_module(module)
    return module


def _register_mae_modules() -> None:
    try:
        from ultralytics_mae.nn import register_ultralytics_modules
    except Exception:  # pragma: no cover - fallback if MAE extras unavailable
        return

    register_ultralytics_modules()


_base_nn = _load_base_nn()
_register_mae_modules()

globals().update(_base_nn.__dict__)
