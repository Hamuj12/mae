"""Shared dataclasses and record helpers for detection and segmentation inference"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class DetectionResult:
    """Single-image detection/segmentation result"""

    # validity and class metadata
    # valid marks whether detector produced a usable primary prediction
    valid: bool
    class_id: int = -1
    score: float = 0.0
    model_task: str = 'detect'

    # bbox formats
    # bbox_xyxy uses absolute pixel coordinates
    bbox_xyxy: np.ndarray = field(default_factory = lambda: np.array([np.nan, np.nan, np.nan, np.nan], dtype = float))
    # bbox_xywh_norm uses center-width-height normalized by image size
    bbox_xywh_norm: np.ndarray = field(default_factory = lambda: np.array([np.nan, np.nan, np.nan, np.nan], dtype = float))

    # optional segmentation payload (single selected instance)
    has_mask: bool = False
    mask_area_ratio: float = float('nan')
    # segment_xy uses absolute pixel coordinates; shape is [N, 2] for polygon vertices
    segment_xy: np.ndarray = field(default_factory = lambda: np.empty((0, 2), dtype = float))
    # segment_xy_norm uses normalized [0, 1] coordinates; shape is [N, 2]
    segment_xy_norm: np.ndarray = field(default_factory = lambda: np.empty((0, 2), dtype = float))

    # runtime provenance
    # these fields make logs self-describing across models and runs
    backend: str = ''
    detector_name: str = ''
    inference_ms: float = float('nan')
    image_path: str = ''
    model_path: str = ''


def invalid_detection(backend: str, detector_name: str, model_path: str = '', model_task: str = 'detect') -> DetectionResult:
    """Create a default invalid detection result"""

    # canonical invalid payload used across failed reads and empty detections
    # NaN bbox values make invalid rows easy to filter in pandas/numpy
    return DetectionResult(
        valid = False,
        class_id = -1,
        score = 0.0,
        model_task = str(model_task or 'detect'),
        bbox_xyxy = np.array([np.nan, np.nan, np.nan, np.nan], dtype = float),
        bbox_xywh_norm = np.array([np.nan, np.nan, np.nan, np.nan], dtype = float),
        has_mask = False,
        mask_area_ratio = float('nan'),
        segment_xy = np.empty((0, 2), dtype = float),
        segment_xy_norm = np.empty((0, 2), dtype = float),
        backend = str(backend),
        detector_name = str(detector_name),
        model_path = str(model_path),
    )
