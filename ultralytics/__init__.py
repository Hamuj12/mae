"""Local wrapper around the installed Ultralytics package with MAE registry patches."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from types import ModuleType


def _load_base_package() -> ModuleType:
    """Load the real ``ultralytics`` package from site-packages."""
    search_paths = sys.path[1:]  # skip current working directory
    spec = importlib.machinery.PathFinder.find_spec(__name__, search_paths)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to locate the installed 'ultralytics' package")

    module = importlib.util.module_from_spec(spec)
    sys.modules[__name__] = module  # ensure relative imports inside package succeed
    spec.loader.exec_module(module)
    return module


_base_pkg = _load_base_package()


def _register_mae_modules() -> None:
    """Inject MAE components into Ultralytics' registry if available."""
    try:
        from ultralytics_mae.nn import register_ultralytics_modules
    except Exception:  # pragma: no cover - defensive against optional dependency issues
        return

    register_ultralytics_modules()


_register_mae_modules()

# Re-export public attributes from the real package
globals().update(_base_pkg.__dict__)
