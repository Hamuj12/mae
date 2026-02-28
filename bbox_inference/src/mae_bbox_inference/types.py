"""Shared dataclasses and record helpers for bbox inference"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class DetectionResult:
    """Single-image bbox detection result"""

    # validity and class metadata
    # valid marks whether detector produced a usable bbox
    valid: bool
    class_id: int              = -1
    score: float               = 0.0

    # bbox formats
    # bbox_xyxy uses absolute pixel coordinates
    bbox_xyxy: np.ndarray      = field(default_factory = lambda: np.array([np.nan, np.nan, np.nan, np.nan], dtype = float))
    # bbox_xywh_norm uses center-width-height normalized by image size
    bbox_xywh_norm: np.ndarray = field(default_factory = lambda: np.array([np.nan, np.nan, np.nan, np.nan], dtype = float))

    # runtime provenance
    # these fields make logs self-describing across models and runs
    backend: str               = ''
    detector_name: str         = ''
    inference_ms: float        = float('nan')
    image_path: str            = ''
    model_path: str            = ''


def invalid_detection(backend: str, detector_name: str, model_path: str = '') -> DetectionResult:
    """Create a default invalid detection result"""

    # canonical invalid payload used across failed reads and empty detections
    # NaN bbox values make invalid rows easy to filter in pandas/numpy
    return DetectionResult(
                            valid = False,
                            class_id = -1,
                            score = 0.0,
                            bbox_xyxy = np.array([np.nan, np.nan, np.nan, np.nan], dtype = float),
                            bbox_xywh_norm = np.array([np.nan, np.nan, np.nan, np.nan], dtype = float),
                            backend = str(backend),
                            detector_name = str(detector_name),
                            model_path = str(model_path),
                        )
