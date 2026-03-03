"""Public package interface for boxinfer"""

from __future__ import annotations

from typing import TYPE_CHECKING

# TYPE_CHECKING imports avoid importing heavy runtime deps on package import
if TYPE_CHECKING:
    from .detector import YoloBBoxDetector
    from .export import export_tensorrt_engine
    from .offline_test import run_offline_bbox_test
    from .offline_test import OfflineTestConfig
    from .timing import TimingSummary
    from .timing import summarize_latencies
    from .types import DetectionResult
    from .visualization import save_visualizations_from_csv
    from .visualization import write_detections_csv
    from .visualization import write_detections_json



__all__ = [
            # runtime detector + model export + offline test API
            'YoloBBoxDetector',
            'export_tensorrt_engine',
            'run_offline_bbox_test',
            'OfflineTestConfig', 

            # timing + typed result payload
            'TimingSummary',
            'summarize_latencies',
            'DetectionResult',

            # logging and overlay helpers
            'write_detections_csv',
            'write_detections_json',
            'save_visualizations_from_csv',
        ]


def __getattr__(name: str):
    # Lazy attribute loading keeps import overhead low and optional deps optional
    if name == 'YoloBBoxDetector':
        # defer ultralytics/onnxruntime import until detector is requested
        from .detector import YoloBBoxDetector
        return YoloBBoxDetector

    if name == 'export_tensorrt_engine':
        # exporter has separate runtime dependency surface
        from .export import export_tensorrt_engine
        return export_tensorrt_engine

    if name == 'run_offline_bbox_test':
        # offline runner is only needed in test/eval flows
        from .offline_test import run_offline_bbox_test
        return run_offline_bbox_test

    if name == 'OfflineTestConfig':
        from .offline_test import OfflineTestConfig
        return OfflineTestConfig

    if name == 'TimingSummary':
        from .timing import TimingSummary
        return TimingSummary

    if name == 'summarize_latencies':
        from .timing import summarize_latencies
        return summarize_latencies

    if name == 'DetectionResult':
        from .types import DetectionResult
        return DetectionResult

    if name == 'write_detections_csv':
        from .visualization import write_detections_csv
        return write_detections_csv

    if name == 'write_detections_json':
        from .visualization import write_detections_json
        return write_detections_json

    if name == 'save_visualizations_from_csv':
        from .visualization import save_visualizations_from_csv
        return save_visualizations_from_csv

    # preserve normal attribute error semantics for unknown names
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
