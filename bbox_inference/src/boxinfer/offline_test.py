"""Standalone offline detect/segment test runner that logs CSV/JSON and visualizations"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from .detector import YoloBBoxDetector
from .timing import summarize_latencies
from .visualization import save_visualizations_from_csv
from .visualization import write_detections_csv
from .visualization import write_detections_json


@dataclass
class OfflineTestConfig:
    """Config for standalone offline detection/segmentation test"""

    # detector backend and input/output locations
    backend: str = 'onnx'
    dataset_dir: str = './datasets'
    model_path: str = './models/model.onnx'
    output_dir: str = './outputs'

    # task determines detector parsing mode
    model_task: str = 'detect'

    # dataset frame pattern
    image_prefix: str = 'image_'
    image_suffix: str = '.png'

    # detector thresholds and runtime knobs
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    input_size: int | tuple[int, int] = 1024
    class_whitelist: list[int] | None = None
    prefer_cuda: bool = True
    device: str = 'cuda:0'

    # warmup and visualization settings
    warmup_iters: int = 3
    top_k: int = 3


def _parse_input_size(input_vals: list[int]) -> int | tuple[int, int]:
    vals = [int(v) for v in input_vals]
    if len(vals) == 1:
        return int(vals[0])
    if len(vals) == 2:
        return (int(vals[0]), int(vals[1]))
    raise ValueError(f'--input-size expects one value (S) or two values (H W), got: {vals}')


def _collect_images(dataset_dir: Path, image_prefix: str, image_suffix: str) -> list[Path]:
    # deterministic ordering by filename
    image_paths = sorted(dataset_dir.glob(f'{image_prefix}*{image_suffix}'))
    if not image_paths:
        raise RuntimeError(f'No images found: {dataset_dir}/{image_prefix}*{image_suffix}')
    return image_paths


def run_offline_test(cfg: OfflineTestConfig) -> dict:
    """Run offline detect/segment test end-to-end"""

    # resolve and validate all incoming paths once
    dataset_dir = Path(str(cfg.dataset_dir)).expanduser().resolve()
    model_path = Path(str(cfg.model_path)).expanduser().resolve()
    output_root = Path(str(cfg.output_dir)).expanduser().resolve()

    if not dataset_dir.is_dir():
        raise ValueError(f'dataset directory not found: {dataset_dir}')
    if not model_path.is_file():
        raise ValueError(f'model not found: {model_path}')

    task_name = str(cfg.model_task).strip().lower() or 'detect'
    if task_name not in {'detect', 'segment', 'pose', 'classify', 'obb'}:
        raise ValueError(
            f'unsupported model_task: {task_name}; expected one of '
            f'{{detect, segment, pose, classify, obb}}'
        )

    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    run_dir = output_root / timestamp
    run_dir.mkdir(parents = True, exist_ok = True)

    # frame order is determined by lexical sort of filenames
    image_paths = _collect_images(
        dataset_dir = dataset_dir,
        image_prefix = str(cfg.image_prefix),
        image_suffix = str(cfg.image_suffix),
    )

    # construct backend detector object with runtime config
    detector = YoloBBoxDetector(
        model_path = str(model_path),
        backend = str(cfg.backend),
        detector_name = f'mae_{task_name}_offline',
        conf_threshold = float(cfg.conf_threshold),
        iou_threshold = float(cfg.iou_threshold),
        input_size = cfg.input_size,
        class_whitelist = cfg.class_whitelist,
        prefer_cuda = bool(cfg.prefer_cuda),
        device = str(cfg.device),
        model_task = task_name,
    )

    # warmup is optional and not included in main timing summary below
    warmup_iters = int(cfg.warmup_iters)
    if warmup_iters > 0:
        first_img = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
        if first_img is None:
            raise RuntimeError(f'Failed to read first image for warmup: {image_paths[0]}')
        warmup_info = detector.warmup(image_bgr = first_img, warmup_iters = warmup_iters)
    else:
        warmup_info = detector.warmup(image_bgr = None, warmup_iters = 0)

    detections = []
    latencies = []

    # primary inference loop for all images in dataset
    for image_path in image_paths:
        det = detector.infer_image_path(str(image_path))
        detections.append(det)
        latencies.append(float(det.inference_ms))

    artifact_prefix = str(task_name)

    # write machine-readable logs first, then render visual products from CSV
    csv_path = write_detections_csv(run_dir / f'{artifact_prefix}_detections.csv', detections)
    json_path = write_detections_json(run_dir / f'{artifact_prefix}_detections.json', detections)

    viz_info = save_visualizations_from_csv(
        csv_path = csv_path,
        output_dir = run_dir,
        top_k = int(cfg.top_k),
    )

    timing = summarize_latencies(latencies)
    num_valid = int(sum(int(bool(det.valid)) for det in detections))
    num_mask = int(sum(int(bool(getattr(det, 'has_mask', False))) for det in detections))

    # summarize run outputs for quick integration checks
    summary = {
        'run_dir': str(run_dir),
        'dataset_dir': str(dataset_dir),
        'model_path': str(model_path),
        'backend': str(cfg.backend),
        'model_task': str(task_name),
        'num_images': int(len(detections)),
        'num_valid': int(num_valid),
        'num_with_mask': int(num_mask),
        'warmup': warmup_info,
        'timing_excludes_warmup': True,
        'timing': {
            'count': int(timing.count),
            'mean_ms': float(timing.mean_ms),
            'min_ms': float(timing.min_ms),
            'p50_ms': float(timing.p50_ms),
            'p90_ms': float(timing.p90_ms),
            'p99_ms': float(timing.p99_ms),
            'max_ms': float(timing.max_ms),
        },
        'csv_path': str(csv_path),
        'json_path': str(json_path),
        'visualization': viz_info,
    }

    with (run_dir / 'run_summary.json').open('w', encoding = 'utf-8') as f:
        json.dump(summary, f, indent = 2)

    return summary


def run_offline_bbox_test(cfg: OfflineTestConfig) -> dict:
    """Backward-compatible alias for historical API name."""
    return run_offline_test(cfg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = 'Standalone YOLO offline test runner (detect/segment)')

    # core runtime inputs
    parser.add_argument('--backend', type = str, choices = ['onnx', 'tensorrt'], default = 'onnx')
    parser.add_argument('--dataset-dir', type = str, required = True)
    parser.add_argument('--model-path', type = str, required = True)
    parser.add_argument('--output-dir', type = str, default = './outputs')
    parser.add_argument(
        '--model-task',
        type = str,
        choices = ['detect', 'segment', 'pose', 'classify', 'obb'],
        default = 'detect',
    )

    # dataset naming pattern for frame discovery
    parser.add_argument('--image-prefix', type = str, default = 'image_')
    parser.add_argument('--image-suffix', type = str, default = '.png')

    # detection and backend configuration
    parser.add_argument('--conf-threshold', type = float, default = 0.25)
    parser.add_argument('--iou-threshold', type = float, default = 0.45)
    parser.add_argument(
        '--input-size',
        type = int,
        nargs = '+',
        default = [1024],
        help = 'Image size as one value (S) or two values (H W).',
    )
    parser.add_argument('--class-whitelist', nargs = '*', type = int, default = None)
    parser.add_argument('--no-prefer-cuda', action = 'store_true')
    parser.add_argument('--device', type = str, default = 'cuda:0')

    parser.add_argument('--warmup-iters', type = int, default = 3)
    parser.add_argument('--top-k', type = int, default = 3)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # map argparse values into typed config payload
    cfg = OfflineTestConfig(
        backend = args.backend,
        dataset_dir = args.dataset_dir,
        model_path = args.model_path,
        output_dir = args.output_dir,
        model_task = args.model_task,
        image_prefix = args.image_prefix,
        image_suffix = args.image_suffix,
        conf_threshold = args.conf_threshold,
        iou_threshold = args.iou_threshold,
        input_size = _parse_input_size(args.input_size),
        class_whitelist = args.class_whitelist,
        prefer_cuda = not bool(args.no_prefer_cuda),
        device = args.device,
        warmup_iters = args.warmup_iters,
        top_k = args.top_k,
    )

    # run full pipeline and print JSON summary for shell logs
    summary = run_offline_test(cfg)
    print(f'Offline test complete (task={cfg.model_task})')
    print(json.dumps(summary, indent = 2))


if __name__ == '__main__':
    main()


# runtime:
# boxinfer-offline-test \
#   --backend onnx \
#   --model-task detect \
#   --dataset-dir /path/to/dataset \
#   --model-path /path/to/model.onnx \
#   --output-dir /path/to/output
#
# TensorRT segmentation example:
# boxinfer-offline-test \
#   --backend tensorrt \
#   --model-task segment \
#   --dataset-dir /home/hm25936/mae/bbox_inference/test_artifacts \
#   --model-path /home/hm25936/mae/runs/yolov8_seg_fda_mix/best.engine \
#   --input-size 3000 4096 \
#   --device cuda:0 \
#   --output-dir /home/hm25936/mae/bbox_inference/test_results
