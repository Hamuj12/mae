from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxMetadata:
    scale: float
    pad_left: int
    pad_top: int
    output_size: int
    original_width: int
    original_height: int


def letterbox(image: np.ndarray, size: int, color: Tuple[int, int, int] = (114, 114, 114)) -> tuple[np.ndarray, LetterboxMetadata]:
    """Resize with unchanged aspect ratio and pad to a square canvas."""
    if image.ndim != 3:
        raise ValueError('Expected HxWxC image for letterbox.')

    h0, w0 = image.shape[:2]
    if h0 <= 0 or w0 <= 0:
        raise ValueError('Image dimensions must be > 0.')

    r = min(size / w0, size / h0)
    w1, h1 = int(round(w0 * r)), int(round(h0 * r))

    resized = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_LINEAR)

    dw = size - w1
    dh = size - h1
    pad_left = int(dw // 2)
    pad_right = int(dw - pad_left)
    pad_top = int(dh // 2)
    pad_bottom = int(dh - pad_top)

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    meta = LetterboxMetadata(
        scale=r,
        pad_left=pad_left,
        pad_top=pad_top,
        output_size=size,
        original_width=w0,
        original_height=h0,
    )
    return padded, meta


def map_xyxy_to_original(xyxy: np.ndarray, meta: LetterboxMetadata) -> np.ndarray:
    """Map letterboxed [x1,y1,x2,y2] detections back to original image pixels."""
    if xyxy.size == 0:
        return xyxy

    mapped = xyxy.astype(np.float32).copy()
    mapped[:, [0, 2]] = (mapped[:, [0, 2]] - float(meta.pad_left)) / float(meta.scale)
    mapped[:, [1, 3]] = (mapped[:, [1, 3]] - float(meta.pad_top)) / float(meta.scale)

    mapped[:, [0, 2]] = np.clip(mapped[:, [0, 2]], 0.0, float(meta.original_width - 1))
    mapped[:, [1, 3]] = np.clip(mapped[:, [1, 3]], 0.0, float(meta.original_height - 1))
    return mapped
