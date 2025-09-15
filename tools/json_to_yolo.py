#!/usr/bin/env python3
"""Convert OrbitGen JSON metadata annotations into YOLO label files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_yaml", type=Path, required=True, help="Path to dataset YAML file")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write YOLO labels")
    return parser.parse_args()


def load_dataset_config(yaml_path: Path) -> Tuple[Path, Dict[str, Path]]:
    """Return dataset root and resolved image directories for each split."""
    with yaml_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    dataset_root = Path(cfg.get("path", yaml_path.parent))
    splits: Dict[str, Path] = {}
    for split in ("train", "val", "test"):
        split_dir = cfg.get(split)
        if not split_dir:
            continue
        split_path = Path(split_dir)
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        splits[split] = split_path
    return dataset_root, splits


def candidate_meta_paths(
    meta_dirs: Iterable[Path], relative_path: Path, stem: str
) -> Iterable[Path]:
    """Yield potential metadata file locations for an image."""
    for meta_dir in meta_dirs:
        if not meta_dir:
            continue
        if not meta_dir.exists():
            continue
        rel_parent = relative_path.parent
        yield meta_dir / rel_parent / f"meta_{stem}.json"
        yield meta_dir / rel_parent / f"{stem}.json"
        yield meta_dir / f"meta_{stem}.json"
        yield meta_dir / f"{stem}.json"


def resolve_meta_file(
    dataset_root: Path, split: str, relative_path: Path, stem: str
) -> Optional[Path]:
    """Find the metadata JSON path corresponding to an image if it exists."""
    meta_root = dataset_root / "meta"
    meta_dirs = []
    split_meta = meta_root / split
    if split_meta.exists():
        meta_dirs.append(split_meta)
    meta_dirs.append(meta_root)

    seen = set()
    for candidate in candidate_meta_paths(meta_dirs, relative_path, stem):
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def extract_bbox_coordinates(bbox_entry: object) -> Optional[Tuple[float, float, float, float]]:
    """Return ``(xmin, ymin, xmax, ymax)`` if available in *bbox_entry*."""
    if isinstance(bbox_entry, dict):
        keys = {k.lower(): k for k in bbox_entry.keys()}
        if {"xmin", "ymin", "xmax", "ymax"}.issubset(keys):
            return (
                float(bbox_entry[keys["xmin"]]),
                float(bbox_entry[keys["ymin"]]),
                float(bbox_entry[keys["xmax"]]),
                float(bbox_entry[keys["ymax"]]),
            )
        if {"x", "y", "w", "h"}.issubset(keys):
            x = float(bbox_entry[keys["x"]])
            y = float(bbox_entry[keys["y"]])
            w = float(bbox_entry[keys["w"]])
            h = float(bbox_entry[keys["h"]])
            return x, y, x + w, y + h
        if "bbox" in keys:
            return extract_bbox_coordinates(bbox_entry[keys["bbox"]])
    elif isinstance(bbox_entry, (list, tuple)) and len(bbox_entry) >= 4:
        xmin, ymin, xmax, ymax = bbox_entry[:4]
        return float(xmin), float(ymin), float(xmax), float(ymax)
    return None


def determine_class_id(
    bbox_entry: object,
    key: object,
    class_map: Dict[str, int],
) -> int:
    """Infer a class id from the bounding box entry or dictionary key."""
    candidates: List[object] = []
    if isinstance(bbox_entry, dict):
        for field in ("class_id", "category_id", "cls", "label"):
            if field in bbox_entry:
                candidates.append(bbox_entry[field])
    if key not in (None, ""):
        candidates.append(key)

    for candidate in candidates:
        if isinstance(candidate, (int, float)):
            return int(candidate)
        if isinstance(candidate, str):
            if candidate.isdigit():
                return int(candidate)
            return class_map.setdefault(candidate, len(class_map))
    return 0


def normalize_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """Convert absolute coordinates to normalized YOLO ``xc, yc, w, h``."""
    xc = ((xmin + xmax) / 2.0) / width
    yc = ((ymin + ymax) / 2.0) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return xc, yc, w, h


def process_split(
    split: str,
    image_dir: Path,
    dataset_root: Path,
    output_root: Path,
    class_map: Dict[str, int],
) -> Tuple[int, int]:
    """Convert metadata for a single dataset split."""
    converted = 0
    skipped = 0

    if not image_dir.exists():
        print(f"[WARN] Image directory for split '{split}' does not exist: {image_dir}")
        return converted, skipped

    output_dir = output_root / "labels" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_files:
        print(f"[WARN] No images found for split '{split}' in {image_dir}")

    for img_path in sorted(image_files):
        relative_path = img_path.relative_to(image_dir)
        stem = img_path.stem
        meta_path = resolve_meta_file(dataset_root, split, relative_path, stem)
        if meta_path is None:
            skipped += 1
            continue

        try:
            with meta_path.open("r") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Failed to read metadata for {img_path}: {exc}")
            skipped += 1
            continue

        bboxes = metadata.get("bboxes", {})
        if isinstance(bboxes, dict):
            bbox_items = bboxes.items()
        elif isinstance(bboxes, list):
            bbox_items = enumerate(bboxes)
        else:
            bbox_items = []

        with Image.open(img_path) as img:
            width, height = img.size

        label_lines = []
        for key, bbox_entry in bbox_items:
            coords = extract_bbox_coordinates(bbox_entry)
            if coords is None:
                continue
            xmin, ymin, xmax, ymax = coords
            if xmax <= xmin or ymax <= ymin:
                continue
            xc, yc, bw, bh = normalize_bbox(xmin, ymin, xmax, ymax, width, height)
            class_id = determine_class_id(bbox_entry, key, class_map)
            label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        label_path = output_dir / relative_path.with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(label_lines))
        converted += 1

    print(f"Split '{split}': converted {converted}, skipped {skipped}")
    return converted, skipped


def main() -> None:
    args = parse_args()
    dataset_root, splits = load_dataset_config(args.data_yaml)
    class_map: Dict[str, int] = {}

    total_converted = 0
    total_skipped = 0

    for split, image_dir in splits.items():
        converted, skipped = process_split(split, image_dir, dataset_root, args.output, class_map)
        total_converted += converted
        total_skipped += skipped

    print(f"Total converted: {total_converted}, skipped: {total_skipped}")


if __name__ == "__main__":
    main()
