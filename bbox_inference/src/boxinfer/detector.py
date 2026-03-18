"""Detector object supporting ONNX and TensorRT backends"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort
import pdb

from .timing import summarize_latencies
from .timing import time_call
from .types import DetectionResult
from .types import invalid_detection

try:
    from ultralytics import YOLO
except Exception:
    # allow ONNX-only use when ultralytics is not installed
    YOLO = None


def _shape_dim_to_int(dim) -> int | None:
    # ONNX shape dims may be symbolic or negative in some exports
    if isinstance(dim, (int, np.integer)) and int(dim) > 0:
        return int(dim)
    return None


def _xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    # model-native center box format to corner format
    return np.array([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype = float)


def _xyxy_to_xywh_norm(xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    # convert pixel bbox to normalized center format for lightweight logging
    x1, y1, x2, y2 = np.asarray(xyxy, dtype = float).reshape(4,)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    if img_w <= 0 or img_h <= 0:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype = float)
    return np.array([cx / img_w, cy / img_h, w / img_w, h / img_h], dtype = float)


def _parse_input_size(input_size) -> tuple[int, int]:
    """
    Parse detector input size as (height, width).
    Supported forms:
    - int -> square (size, size)
    - tuple/list with 2 values -> (height, width)
    """
    if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
        h = int(max(32, int(input_size[0])))
        w = int(max(32, int(input_size[1])))
        return h, w

    size = int(max(32, int(input_size)))
    return size, size


def _validate_preprocessed_shape(image_bgr: np.ndarray, exp_h: int, exp_w: int) -> tuple[int, int]:
    """Validate that a preprocessed image already matches detector input size."""
    img_h, img_w = image_bgr.shape[:2]
    if int(img_h) != int(exp_h) or int(img_w) != int(exp_w):
        raise ValueError(
                            f'preprocessed_input=True but image shape is ({int(img_h)}, {int(img_w)}); '
                            f'expected ({int(exp_h)}, {int(exp_w)})'
                        )
    return int(img_h), int(img_w)


def letterbox_image(
                        image_bgr: np.ndarray,
                        dst_w: int,
                        dst_h: int,
                        pad_value: int = 114,
                    ) -> tuple[np.ndarray, float, float, float]:
    # resize with aspect ratio preserved, then pad to model input size
    src_h, src_w = image_bgr.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError('Invalid input image shape')

    scale = min(float(dst_w) / float(src_w), float(dst_h) / float(src_h))
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation = cv2.INTER_LINEAR)

    pad_w = dst_w - new_w
    pad_h = dst_h - new_h
    left = int(np.floor(pad_w / 2.0))
    right = int(np.ceil(pad_w / 2.0))
    top = int(np.floor(pad_h / 2.0))
    bottom = int(np.ceil(pad_h / 2.0))

    out = cv2.copyMakeBorder(
                                resized,
                                top,
                                bottom,
                                left,
                                right,
                                cv2.BORDER_CONSTANT,
                                value = (pad_value, pad_value, pad_value),
                            )
    return out, scale, float(left), float(top)


def rescale_camera_matrix_for_letterbox(
                                            K_src: np.ndarray,
                                            src_w: int,
                                            src_h: int,
                                            dst_w: int,
                                            dst_h: int,
                                        ) -> tuple[np.ndarray, float, float, float]:
    """
    Rescale camera intrinsics for letterboxed image coordinates.

    Returns:
    - K_lb: 3x3 intrinsics after resize+pad
    - scale: uniform resize scale
    - pad_x: left pad in pixels
    - pad_y: top pad in pixels
    """
    K = np.asarray(K_src, dtype = float).reshape(3, 3)
    sw = int(src_w)
    sh = int(src_h)
    dw = int(dst_w)
    dh = int(dst_h)
    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        raise ValueError(
                            f'invalid source/destination image size for K letterbox rescale: '
                            f'src=({sw}, {sh}), dst=({dw}, {dh})'
                        )

    # use same resize/pad policy as letterbox_image()
    scale = min(float(dw) / float(sw), float(dh) / float(sh))
    new_w = int(round(sw * scale))
    new_h = int(round(sh * scale))
    pad_w = dw - new_w
    pad_h = dh - new_h
    pad_x = float(int(np.floor(pad_w / 2.0)))
    pad_y = float(int(np.floor(pad_h / 2.0)))

    # x_lb = scale * x_src + pad_x, y_lb = scale * y_src + pad_y
    A = np.array(
                    [
                        [scale, 0.0, pad_x],
                        [0.0, scale, pad_y],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype = float,
                )
    K_lb = A @ K
    return K_lb, float(scale), pad_x, pad_y


class YoloBBoxDetector:
    """Unified detector object that returns bbox, score, class, and inference timing"""

    def __init__(
                    self,
                    model_path: str,
                    backend: str = 'onnx',
                    detector_name: str = 'yolo_bbox',
                    conf_threshold: float = 0.25,
                    iou_threshold: float = 0.45,
                    input_size = 1024,
                    class_whitelist: Optional[List[int]] = None,
                    prefer_cuda: bool = True,
                    device: str = 'cuda:0',
                    preprocessed_input: bool = False,
                ) -> None:
        # resolve once so logs carry canonical absolute path
        self._model_path = str(Path(str(model_path)).expanduser().resolve())
        if not Path(self._model_path).is_file():
            raise FileNotFoundError(f'model file not found: {self._model_path}')

        # supported execution backends
        self._backend = str(backend).strip().lower()
        if self._backend not in {'onnx', 'tensorrt'}:
            raise ValueError(f'unsupported backend: {self._backend}')

        self._detector_name   = str(detector_name)
        self._conf_thr        = float(conf_threshold)
        self._iou_thr         = float(iou_threshold)
        self._requested_input_h, self._requested_input_w = _parse_input_size(input_size)
        self._input_h, self._input_w = int(self._requested_input_h), int(self._requested_input_w)
        self._class_whitelist = set(class_whitelist or [])
        self._prefer_cuda     = bool(prefer_cuda)
        self._device          = str(device).strip() or 'cuda:0'
        self._preprocessed_input = bool(preprocessed_input)

        # backend runtime state is initialized lazily in init helpers
        self._session       = None
        self._model         = None
        self._input_name    = None
        self._output_name   = None
        self._input_h       = None
        self._input_w       = None

        # initialize backend-specific runtime once
        if self._backend == 'onnx':
            self._init_onnx()
        else:
            self._init_tensorrt()

    def _init_onnx(self) -> None:
        if Path(self._model_path).suffix.lower() != '.onnx':
            raise ValueError(f'ONNX backend expects a .onnx model, got: {self._model_path}')

        # prefer CUDA provider when available, otherwise CPU fallback
        providers       = ['CPUExecutionProvider']
        if self._prefer_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers   = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # providers   = ['CUDAExecutionProvider']
            
        # pdb.set_trace()
        self._session     = ort.InferenceSession(self._model_path, providers = providers)
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # use fixed model input size when present, otherwise requested input_size
        model_input_shape = self._session.get_inputs()[0].shape
        model_h = _shape_dim_to_int(model_input_shape[2]) if len(model_input_shape) >= 4 else None
        model_w = _shape_dim_to_int(model_input_shape[3]) if len(model_input_shape) >= 4 else None
        if model_h is not None and model_w is not None:
            self._input_h = int(model_h)
            self._input_w = int(model_w)
            if (self._input_h, self._input_w) != (int(self._requested_input_h), int(self._requested_input_w)):
                raise ValueError(
                                    f'ONNX model input is fixed at ({self._input_h}, {self._input_w}) '
                                    f'but requested input_size is ({int(self._requested_input_h)}, {int(self._requested_input_w)}). '
                                    f'Use matching config or re-export ONNX with the desired/static or dynamic input shape.'
                                )
        else:
            self._input_h = int(self._requested_input_h)
            self._input_w = int(self._requested_input_w)

    def _init_tensorrt(self) -> None:
        if Path(self._model_path).suffix.lower() != '.engine':
            raise ValueError(f'TensorRT backend expects a .engine model, got: {self._model_path}')

        if YOLO is None:
            raise ImportError('ultralytics is required for TensorRT backend')

        # TensorRT engines are configured against the requested detector input size.
        self._input_h = int(self._requested_input_h)
        self._input_w = int(self._requested_input_w)

        # set task explicitly to avoid warning when metadata is not embedded in engine
        self._model = YOLO(self._model_path, task = 'detect')
        if not self._prefer_cuda:
            self._device = 'cpu'

    def infer(self, image_bgr: np.ndarray, image_path: str = '') -> DetectionResult:
        """Run one inference and return detection + latency"""

        # guard for empty images and keep output contract stable
        if image_bgr is None or image_bgr.size == 0:
            det = invalid_detection(self._backend, self._detector_name, self._model_path)
            det.image_path = str(image_path)
            return det

        # time only the backend predict path
        if self._backend == 'onnx':
            det, infer_ms = time_call(self._detect_onnx, image_bgr)
        else:
            det, infer_ms = time_call(self._detect_tensorrt, image_bgr)

        det.inference_ms = float(infer_ms)
        det.image_path   = str(image_path)
        det.model_path   = self._model_path
        return det

    def infer_image_path(self, image_path: str) -> DetectionResult:
        """Read image from disk and run detection"""

        # resolve here so CSV contains absolute image paths
        image_path = str(Path(str(image_path)).expanduser().resolve())
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            det = invalid_detection(self._backend, self._detector_name, self._model_path)
            det.image_path = image_path
            return det

        return self.infer(image_bgr = image_bgr, image_path = image_path)

    def warmup(self, image_bgr: np.ndarray | None, warmup_iters: int = 3) -> dict:
        """Warm up model memory/graph by running a few inferences"""

        # allow explicit warmup disable with warmup_iters <= 0
        runs      = max(0, int(warmup_iters))
        latencies = []

        if runs == 0:
            summary = summarize_latencies(latencies)
            return {
                        'runs': 0,
                        'latencies_ms': [],
                        'timing': {
                                    'count': int(summary.count),
                                    'mean_ms': float(summary.mean_ms),
                                    'min_ms': float(summary.min_ms),
                                    'p50_ms': float(summary.p50_ms),
                                    'p90_ms': float(summary.p90_ms),
                                    'p99_ms': float(summary.p99_ms),
                                    'max_ms': float(summary.max_ms),
                                },
                    }

        # execute a few forward passes to stabilize first-run latency
        for _ in range(runs):
            det = self.infer(image_bgr = image_bgr)
            latencies.append(float(det.inference_ms))

        summary = summarize_latencies(latencies)
        return {
                    'runs': int(runs),
                    # raw per-run values are useful for quick startup diagnostics
                    'latencies_ms': [float(v) for v in latencies],
                    'timing': {
                                'count': int(summary.count),
                                'mean_ms': float(summary.mean_ms),
                                'min_ms': float(summary.min_ms),
                                'p50_ms': float(summary.p50_ms),
                                'p90_ms': float(summary.p90_ms),
                                'p99_ms': float(summary.p99_ms),
                                'max_ms': float(summary.max_ms),
                            },
                }

    def _detect_onnx(self, image_bgr: np.ndarray) -> DetectionResult:
        img_h, img_w = image_bgr.shape[:2]

        # either preprocess inside detector or trust caller-provided detector-sized input
        if self._preprocessed_input:
            _validate_preprocessed_shape(image_bgr, exp_h = int(self._input_h), exp_w = int(self._input_w))
            lb_img = image_bgr
            scale = 1.0
            pad_x = 0.0
            pad_y = 0.0
        else:
            lb_img, scale, pad_x, pad_y = letterbox_image(
                                                                image_bgr = image_bgr,
                                                                dst_w = int(self._input_w),
                                                                dst_h = int(self._input_h),
                                                            )
        rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)
        # model expects NCHW float32 in [0, 1]
        inp = rgb.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))
        inp = np.expand_dims(inp, axis = 0)

        # raw output shape varies by model export and opset
        raw     = self._session.run([self._output_name], {self._input_name: inp})[0]
        preds   = np.asarray(raw)
        if preds.ndim == 3:
            preds   = preds[0]
        if preds.ndim != 2:
            return invalid_detection(self._backend, self._detector_name, self._model_path)
        if preds.shape[0] < preds.shape[1]:
            preds   = preds.T
        if preds.shape[1] < 5:
            return invalid_detection(self._backend, self._detector_name, self._model_path)

        boxes_xyxy  = []
        scores      = []
        classes     = []

        # decode candidate rows and apply confidence/class filters
        for row in preds:
            vals    = np.asarray(row, dtype = float).reshape(-1)
            if vals.size < 5:
                continue

            cx, cy, w, h    = vals[:4]
            cls_scores      = vals[4:]
            if cls_scores.size == 0:
                continue

            if cls_scores.size == 1:
                class_id    = 0
                score       = float(cls_scores[0])
            else:
                class_id    = int(np.argmax(cls_scores))
                score       = float(cls_scores[class_id])

            if not np.isfinite(score) or score < self._conf_thr:
                continue
            if self._class_whitelist and class_id not in self._class_whitelist:
                continue

            boxes_xyxy.append(_xywh_to_xyxy(cx, cy, w, h))
            scores.append(score)
            classes.append(class_id)

        # no candidates survived thresholds
        if not boxes_xyxy:
            return invalid_detection(self._backend, self._detector_name, self._model_path)

        # run NMS in xywh format expected by cv2.dnn.NMSBoxes
        nms_boxes = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box
            nms_boxes.append([float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))])

        keep = cv2.dnn.NMSBoxes(
                                    bboxes = nms_boxes,
                                    scores = scores,
                                    score_threshold = self._conf_thr,
                                    nms_threshold = self._iou_thr,
                                )
        if keep is None or len(keep) == 0:
            return invalid_detection(self._backend, self._detector_name, self._model_path)

        # map selected bbox back from letterboxed space to original image space
        keep_idx    = int(np.asarray(keep).reshape(-1)[0])
        box_lb      = np.asarray(boxes_xyxy[keep_idx], dtype = float).reshape(4,)
        box         = box_lb.copy()
        box[0]      = (box[0] - pad_x) / max(scale, 1e-9)
        box[1]      = (box[1] - pad_y) / max(scale, 1e-9)
        box[2]      = (box[2] - pad_x) / max(scale, 1e-9)
        box[3]      = (box[3] - pad_y) / max(scale, 1e-9)

        box[0]      = float(np.clip(box[0], 0.0, img_w - 1.0))
        box[1]      = float(np.clip(box[1], 0.0, img_h - 1.0))
        box[2]      = float(np.clip(box[2], 0.0, img_w - 1.0))
        box[3]      = float(np.clip(box[3], 0.0, img_h - 1.0))

        # return both pixel bbox and normalized bbox for downstream logging
        return DetectionResult(
                                valid = True,
                                class_id = int(classes[keep_idx]),
                                score = float(scores[keep_idx]),
                                bbox_xyxy = box.astype(float),
                                bbox_xywh_norm = _xyxy_to_xywh_norm(box, img_w = img_w, img_h = img_h),
                                backend = self._backend,
                                detector_name = self._detector_name,
                                model_path = self._model_path,
                            )

    def _detect_tensorrt(self, image_bgr: np.ndarray) -> DetectionResult:
        img_h, img_w    = image_bgr.shape[:2]
        if self._preprocessed_input:
            img_h, img_w = _validate_preprocessed_shape(image_bgr, exp_h = int(self._input_h), exp_w = int(self._input_w))

        classes_arg     = sorted(self._class_whitelist) if self._class_whitelist else None
        imgsz_arg       = int(self._input_h) if int(self._input_h) == int(self._input_w) else (int(self._input_h), int(self._input_w))

        # ultralytics still owns TensorRT runtime preprocessing; when preprocessed_input=True
        # we enforce that the incoming image already matches detector input geometry.
        results = self._model.predict(
                                        source = image_bgr,
                                        imgsz = imgsz_arg,
                                        conf = float(self._conf_thr),
                                        iou = float(self._iou_thr),
                                        device = self._device,
                                        classes = classes_arg,
                                        verbose = False,
                                    )
        if results is None or len(results) == 0:
            return invalid_detection(self._backend, self._detector_name, self._model_path)

        res     = results[0]
        boxes   = getattr(res, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return invalid_detection(self._backend, self._detector_name, self._model_path)

        xyxy    = np.asarray(boxes.xyxy.cpu().numpy()).reshape(-1, 4).astype(float)
        conf    = np.asarray(boxes.conf.cpu().numpy()).reshape(-1).astype(float)
        cls     = np.asarray(boxes.cls.cpu().numpy()).reshape(-1).astype(int)
        if xyxy.shape[0] == 0:
            return invalid_detection(self._backend, self._detector_name, self._model_path)

        valid   = np.isfinite(conf)
        if self._class_whitelist:
            valid   = valid & np.isin(cls, list(self._class_whitelist))
        valid   = valid & (conf >= self._conf_thr)
        if not np.any(valid):
            return invalid_detection(self._backend, self._detector_name, self._model_path)

        # choose highest confidence candidate after class + score filtering
        cand_idx    = np.where(valid)[0]
        keep_idx    = int(cand_idx[np.argmax(conf[cand_idx])])
        box         = xyxy[keep_idx].copy()
        box[0]      = float(np.clip(box[0], 0.0, img_w - 1.0))
        box[1]      = float(np.clip(box[1], 0.0, img_h - 1.0))
        box[2]      = float(np.clip(box[2], 0.0, img_w - 1.0))
        box[3]      = float(np.clip(box[3], 0.0, img_h - 1.0))

        # TensorRT path returns same payload shape as ONNX path
        return DetectionResult(
                                valid = True,
                                class_id = int(cls[keep_idx]),
                                score = float(conf[keep_idx]),
                                bbox_xyxy = box.astype(float),
                                bbox_xywh_norm = _xyxy_to_xywh_norm(box, img_w = img_w, img_h = img_h),
                                backend = self._backend,
                                detector_name = self._detector_name,
                                model_path = self._model_path,
                            )
