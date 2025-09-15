#!/usr/bin/env python3
"""Convert OrbitGen JSON metadata annotations into YOLO label files."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_yaml", type=Path, required=True, help="Path to dataset YAML file")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write YOLO labels")
    return parser.parse_args()


def load_dataset_config(yaml_path: Path) -> Tuple[Path, Dict[str, Path]]:
    with yaml_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    dataset_root = Path(cfg.get("path", yaml_path.parent))
    splits = {s: dataset_root / cfg[s] for s in ("train", "val", "test") if cfg.get(s)}
    return dataset_root, splits


def normalize_bbox(xmin, ymin, xmax, ymax, width, height):
    xc = ((xmin + xmax) / 2.0) / width
    yc = ((ymin + ymax) / 2.0) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return xc, yc, w, h


def process_split(split: str, image_dir: Path, meta_dir: Path, output_root: Path):
    converted, skipped = 0, 0
    output_dir = output_root / "labels" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(image_dir.glob("*.png")):
        stem = img_path.stem  # e.g. "image_00000"

        # Match corresponding metadata: replace "image_" with "meta_"
        if stem.startswith("image_"):
            meta_name = "meta_" + stem.split("image_")[1] + ".json"
        else:
            meta_name = stem + ".json"

        meta_path = meta_dir / meta_name

        if not meta_path.exists():
            print(f"[SKIP] No meta for {img_path.name} (expected {meta_name})")
            skipped += 1
            continue

        with meta_path.open("r") as f:
            metadata = json.load(f)

        bboxes = metadata.get("bboxes", {})
        if not bboxes:
            print(f"[SKIP] No bboxes in {meta_path.name}")
            skipped += 1
            continue

        with Image.open(img_path) as img:
            w, h = img.size

        label_lines = []
        for class_id, bbox in enumerate(bboxes.values()):
            xmin, ymin = bbox["xmin"], bbox["ymin"]
            xmax, ymax = bbox["xmax"], bbox["ymax"]
            xc, yc, bw, bh = normalize_bbox(xmin, ymin, xmax, ymax, w, h)
            label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        label_path = output_dir / f"{stem}.txt"
        label_path.write_text("\n".join(label_lines))
        converted += 1

    print(f"Split '{split}': converted {converted}, skipped {skipped}")
    return converted, skipped


def main():
    args = parse_args()
    dataset_root, splits = load_dataset_config(args.data_yaml)

    total_converted, total_skipped = 0, 0
    for split, image_dir in splits.items():
        meta_dir = dataset_root / "meta" / split
        c, s = process_split(split, image_dir, meta_dir, args.output)
        total_converted += c
        total_skipped += s
    print(f"Total converted: {total_converted}, skipped: {total_skipped}")


if __name__ == "__main__":
    main()