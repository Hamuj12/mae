#!/usr/bin/env python3
"""
Quick sanity-check visualizer for a YOLOv8 segmentation dataset.

For a few random images, it creates:
- image with raw mask overlay
- image with YOLO polygon overlay
- side-by-side comparison

Assumes:
- dataset_root/images/<split>/*.png|jpg...
- dataset_root/masks/<split>/<same_stem>.png
- dataset_root/labels/<split>/<same_stem>.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify YOLO segmentation dataset visually.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset root")
    parser.add_argument("--split", type=str, default="train", help="Split to sample from")
    parser.add_argument("--num-samples", type=int, default=3, help="How many random samples")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="seg_verify_outputs",
        help="Where to save verification images",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def load_image_paths(images_dir: Path) -> List[Path]:
    return sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    )


def overlay_mask(image_bgr: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    binary = mask_gray > 0

    color = np.zeros_like(out)
    color[:, :, 1] = 255  # green

    alpha = 0.35
    out[binary] = cv2.addWeighted(out, 1 - alpha, color, alpha, 0)[binary]

    # outline for visibility
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    return out


def parse_yolo_seg_txt(label_path: Path, w: int, h: int) -> List[np.ndarray]:
    polygons: List[np.ndarray] = []
    if not label_path.exists():
        return polygons

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return polygons

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue  # class + at least 3 points
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0:
            continue

        pts = []
        for i in range(0, len(coords), 2):
            x = int(round(coords[i] * w))
            y = int(round(coords[i + 1] * h))
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            pts.append([x, y])

        if len(pts) >= 3:
            polygons.append(np.array(pts, dtype=np.int32))

    return polygons


def overlay_polygons(image_bgr: np.ndarray, polygons: List[np.ndarray]) -> np.ndarray:
    out = image_bgr.copy()
    if not polygons:
        return out

    fill = out.copy()
    for poly in polygons:
        cv2.fillPoly(fill, [poly], (255, 0, 0))   # blue fill
        cv2.polylines(out, [poly], isClosed=True, color=(0, 0, 255), thickness=2)  # red edge

    out = cv2.addWeighted(fill, 0.25, out, 0.75, 0)
    return out


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    return out


def make_panel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])

    def pad_to_h(img: np.ndarray, h_out: int) -> np.ndarray:
        if img.shape[0] == h_out:
            return img
        pad = h_out - img.shape[0]
        return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    left = pad_to_h(left, h)
    right = pad_to_h(right, h)
    return np.hstack([left, right])


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    dataset = Path(args.dataset).expanduser().resolve()
    images_dir = dataset / "images" / args.split
    masks_dir = dataset / "masks" / args.split
    labels_dir = dataset / "labels" / args.split
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = load_image_paths(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    samples = random.sample(image_paths, k=min(args.num_samples, len(image_paths)))

    print(f"[INFO] Writing outputs to: {output_dir}")
    for idx, img_path in enumerate(samples):
        stem = img_path.stem
        mask_path = masks_dir / f"{stem}.png"
        label_path = labels_dir / f"{stem}.txt"

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        h, w = image.shape[:2]

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[WARN] Could not read mask: {mask_path}")
                mask_overlay = image.copy()
            else:
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_overlay = overlay_mask(image, mask)
        else:
            print(f"[WARN] Missing mask: {mask_path}")
            mask_overlay = image.copy()

        polygons = parse_yolo_seg_txt(label_path, w, h)
        poly_overlay = overlay_polygons(image, polygons)

        left = add_label(mask_overlay, "Raw mask overlay")
        right = add_label(poly_overlay, "YOLO polygon overlay")
        panel = make_panel(left, right)

        out_path = output_dir / f"{idx:02d}_{stem}_verify.png"
        cv2.imwrite(str(out_path), panel)

        print(f"[OK] {img_path.name}")
        print(f"     mask:  {mask_path.name if mask_path.exists() else 'MISSING'}")
        print(f"     label: {label_path.name if label_path.exists() else 'MISSING'}")
        print(f"     polys: {len(polygons)}")
        print(f"     saved: {out_path}")

    print("[DONE] Verification images written.")
    

if __name__ == "__main__":
    main()
    
# PYTHONPATH=. python tools/verify_yolo_seg_dataset.py \
#   --dataset /home/hm25936/datasets_for_yolo/midLighting_rmag_5m_to_100m_FDA_MIX_SEG \
#   --split train \
#   --num-samples 3 \
#   --output-dir /home/hm25936/datasets_for_yolo/seg_verify_outputs \
#   --seed 42