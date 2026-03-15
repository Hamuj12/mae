#!/usr/bin/env python3
"""
Copy a dataset, duplicate per-image mask PNGs to match augmented image names,
and generate YOLOv8 segmentation labels (*.txt) from binary masks.

Assumptions:
- Image files are named like:
    image_00000.png
    image_00000_fda_lab_v1.png
    image_00000_fda_real_v0.png
- Base masks are named like:
    mask_00000.png
- A mask applies to all image variants sharing the same numeric ID.
- Single-class segmentation (class 0), foreground is any nonzero pixel.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXT = ".png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLOv8 segmentation dataset from image+mask folders.")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source dataset root (contains images/, labels/, masks/, dataset yaml, etc.)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Destination dataset root to create",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test", "train_fda"],
        help="Splits to process if present under images/",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="YOLO class id to write for foreground objects",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=10.0,
        help="Ignore contours smaller than this many pixels",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete destination if it already exists",
    )
    parser.add_argument(
        "--keep-old-labels",
        action="store_true",
        help="Keep existing labels in dst/labels instead of replacing them",
    )
    return parser.parse_args()


def copy_dataset(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst}\n"
                "Use --overwrite to replace it."
            )
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def extract_numeric_id(stem: str) -> Optional[str]:
    """
    Extract trailing numeric group after image_/mask_ prefix.

    Examples:
      image_00000               -> 00000
      image_00000_fda_lab_v1    -> 00000
      mask_00000                -> 00000
    """
    m = re.match(r"^(?:image|mask)_(\d+)(?:.*)?$", stem)
    return m.group(1) if m else None


def list_images(split_dir: Path) -> List[Path]:
    if not split_dir.exists():
        return []
    return sorted([p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def find_mask_for_id(mask_dir: Path, numeric_id: str) -> Optional[Path]:
    """
    Prefer exact mask_00000.* match, otherwise any mask file with same numeric id.
    """
    exact_candidates = [
        mask_dir / f"mask_{numeric_id}.png",
        mask_dir / f"mask_{numeric_id}.jpg",
        mask_dir / f"mask_{numeric_id}.jpeg",
    ]
    for p in exact_candidates:
        if p.exists():
            return p

    for p in mask_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if extract_numeric_id(p.stem) == numeric_id:
            return p
    return None


def duplicate_masks_for_split(images_dir: Path, masks_dir: Path) -> int:
    """
    For each image file, create a mask PNG with the same stem in masks_dir.
    """
    images = list_images(images_dir)
    if not images:
        return 0

    masks_dir.mkdir(parents=True, exist_ok=True)
    created = 0

    for img_path in images:
        numeric_id = extract_numeric_id(img_path.stem)
        if numeric_id is None:
            print(f"[WARN] Could not parse numeric id from image name: {img_path.name}")
            continue

        src_mask = find_mask_for_id(masks_dir, numeric_id)
        if src_mask is None:
            print(f"[WARN] No source mask found for image {img_path.name} (id={numeric_id})")
            continue

        dst_mask = masks_dir / f"{img_path.stem}{MASK_EXT}"
        if src_mask.resolve() == dst_mask.resolve():
            # already the base mask with same name somehow
            continue

        shutil.copy2(src_mask, dst_mask)
        created += 1

    return created


def contour_to_yolo_segment(contour: np.ndarray, w: int, h: int) -> Optional[str]:
    """
    Convert contour Nx1x2 or Nx2 to a YOLO seg line payload of normalized xy pairs.
    """
    contour = contour.reshape(-1, 2)
    if contour.shape[0] < 3:
        return None

    xs = np.clip(contour[:, 0].astype(np.float32) / w, 0.0, 1.0)
    ys = np.clip(contour[:, 1].astype(np.float32) / h, 0.0, 1.0)

    coords = []
    for x, y in zip(xs, ys):
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")
    return " ".join(coords)


def write_yolo_seg_label_from_mask(
    mask_path: Path,
    label_path: Path,
    class_id: int,
    min_area: float,
) -> None:
    """
    Foreground = any nonzero pixel.
    Writes one line per connected contour:
      class x1 y1 x2 y2 ... xn yn
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines: List[str] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        seg = contour_to_yolo_segment(contour, w, h)
        if seg is None:
            continue

        lines.append(f"{class_id} {seg}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
        # else: keep empty file for no-object image


def generate_labels_for_split(
    images_dir: Path,
    masks_dir: Path,
    labels_dir: Path,
    class_id: int,
    min_area: float,
) -> int:
    images = list_images(images_dir)
    if not images:
        return 0

    labels_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for img_path in images:
        mask_path = masks_dir / f"{img_path.stem}{MASK_EXT}"
        if not mask_path.exists():
            print(f"[WARN] Missing duplicated mask for {img_path.name}: expected {mask_path.name}")
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        write_yolo_seg_label_from_mask(
            mask_path=mask_path,
            label_path=label_path,
            class_id=class_id,
            min_area=min_area,
        )
        written += 1

    return written


def remove_cache_files(dataset_root: Path) -> None:
    for p in dataset_root.rglob("*.cache"):
        p.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    copy_dataset(src, dst, overwrite=args.overwrite)

    if not args.keep_old_labels:
        labels_root = dst / "labels"
        if labels_root.exists():
            shutil.rmtree(labels_root)
        labels_root.mkdir(parents=True, exist_ok=True)

    total_masks_created = 0
    total_labels_written = 0

    for split in args.splits:
        images_dir = dst / "images" / split
        masks_dir = dst / "masks" / split
        labels_dir = dst / "labels" / split

        if not images_dir.exists():
            print(f"[INFO] Skipping split '{split}' (no images dir)")
            continue
        if not masks_dir.exists():
            print(f"[INFO] Skipping split '{split}' (no masks dir)")
            continue

        print(f"[INFO] Processing split: {split}")
        created = duplicate_masks_for_split(images_dir, masks_dir)
        written = generate_labels_for_split(
            images_dir=images_dir,
            masks_dir=masks_dir,
            labels_dir=labels_dir,
            class_id=args.class_id,
            min_area=args.min_area,
        )
        total_masks_created += created
        total_labels_written += written
        print(f"[INFO]   duplicated mask PNGs: {created}")
        print(f"[INFO]   wrote YOLO seg labels: {written}")

    remove_cache_files(dst)

    print(f"[DONE] New dataset created at: {dst}")
    print(f"[DONE] Total duplicated mask PNGs: {total_masks_created}")
    print(f"[DONE] Total YOLO seg labels written: {total_labels_written}")


if __name__ == "__main__":
    main()
    
# PYTHONPATH=. python tools/prepare_yolo_seg_dataset.py \
#   --src /home/hm25936/datasets_for_yolo/midLighting_rmag_5m_to_100m_FDA_MIX \
#   --dst /home/hm25936/datasets_for_yolo/midLighting_rmag_5m_to_100m_FDA_MIX_SEG \
#   --splits train val test train_fda \
#   --class-id 0 \
#   --min-area 10 \
#   --overwrite