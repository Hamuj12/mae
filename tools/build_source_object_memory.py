#!/usr/bin/env python3
"""Build a compact labeled source-object memory for source-anchored ODAD."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from odad.adapters import attach_residual_adapters, parse_adapter_layers
from odad.online_adapt import (
    SUPPORTED_EXTENSIONS,
    first_memory_source_adapter,
    pool_feature_for_original_box,
    try_load_font,
    unwrap_core_and_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute a labeled source object memory bank.")
    parser.add_argument("--weights", required=True, type=str, help="Source-trained YOLO weights.")
    parser.add_argument("--dataset", required=True, type=str, help="YOLO dataset root with images/ and labels/.")
    parser.add_argument("--output", required=True, type=str, help="Output source_memory.pt path.")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--imgsz", default=1024, type=int)
    parser.add_argument("--layer", default=21, type=int, help="Feature layer used for object memory.")
    parser.add_argument("--slot-dim", default=32, type=int)
    parser.add_argument("--memory-size", default=64, type=int)
    parser.add_argument("--max-images", default=2000, type=int)
    parser.add_argument("--diversity-thresh", default=0.85, type=float)
    parser.add_argument("--adapter-layers", default="18,21", type=str)
    parser.add_argument("--adapter-reduction", default=8, type=int)
    parser.add_argument("--adapter-min-channels", default=8, type=int)
    parser.add_argument("--adapter-scale", default=1.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def image_roots(dataset: Path) -> List[Path]:
    roots = [dataset / "images" / split for split in ("train", "val", "test")]
    roots.extend([dataset / "images", dataset])
    seen = set()
    out: List[Path] = []
    for root in roots:
        if root.exists() and root not in seen:
            seen.add(root)
            out.append(root)
    return out


def list_images(dataset: Path, max_images: int, rng: random.Random) -> List[Path]:
    images: List[Path] = []
    for root in image_roots(dataset):
        images.extend(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if images:
            break
    images = sorted(set(images))
    if max_images > 0 and len(images) > max_images:
        rng.shuffle(images)
        images = sorted(images[:max_images])
    if not images:
        raise RuntimeError(f"No source images found under {dataset}.")
    return images


def label_path_for_image(dataset: Path, image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        rel = Path(*parts[idx + 1 :]).with_suffix(".txt")
        return dataset / "labels" / rel
    return dataset / "labels" / image_path.with_suffix(".txt").name


def read_yolo_labels(path: Path, width: int, height: int) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    if not path.exists():
        return []
    rows: List[Tuple[int, Tuple[float, float, float, float]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = [float(v) for v in parts[1:5]]
        x1 = (cx - 0.5 * bw) * width
        y1 = (cy - 0.5 * bh) * height
        x2 = (cx + 0.5 * bw) * width
        y2 = (cy + 0.5 * bh) * height
        rows.append((cls_id, (x1, y1, x2, y2)))
    return rows


def scale_bin(area_frac: float) -> str:
    if area_frac < 0.01:
        return "small"
    if area_frac < 0.05:
        return "medium"
    return "large"


def select_slot(vectors: torch.Tensor, slots: List[torch.Tensor], limit: int, diversity_thresh: float) -> bool:
    if len(slots) < limit:
        return True
    active = torch.stack(slots)
    sim = float((active @ vectors.float()).max().cpu())
    return sim < float(diversity_thresh)


def save_contact_sheet(output: Path, metas: Sequence[Dict[str, Any]]) -> None:
    cols = 4
    panel_w, panel_h = 260, 220
    rows = max(1, int(math.ceil(max(1, len(metas)) / cols)))
    sheet = Image.new("RGB", (cols * panel_w, rows * panel_h), color=(0, 0, 0))
    font = try_load_font(14)
    for idx, meta in enumerate(metas):
        panel = Image.new("RGB", (panel_w, panel_h), color=(18, 18, 18))
        draw = ImageDraw.Draw(panel)
        path = Path(str(meta["path"]))
        if path.exists():
            with Image.open(path) as im:
                img = im.convert("RGB")
            x1, y1, x2, y2 = [float(v) for v in meta["box_xyxy"]]
            crop = img.crop((max(0, int(x1)), max(0, int(y1)), min(img.size[0], int(x2)), min(img.size[1], int(y2))))
            crop.thumbnail((panel_w, panel_h - 50))
            panel.paste(crop, ((panel_w - crop.size[0]) // 2, 38))
        draw.text((8, 6), f"slot={idx} cls={meta['class_id']} {meta['scale_bin']}", fill=(255, 255, 255), font=font)
        draw.text((8, 24), f"area={float(meta['area_frac']):.4f}", fill=(255, 255, 255), font=font)
        sheet.paste(panel, ((idx % cols) * panel_w, (idx // cols) * panel_h))
    sheet.save(output)


def save_heatmap(output: Path, vectors: torch.Tensor) -> None:
    fig = plt.figure(figsize=(6, 5))
    if int(vectors.shape[0]) <= 0:
        plt.text(0.5, 0.5, "no source slots", ha="center", va="center", transform=plt.gca().transAxes)
        plt.axis("off")
    else:
        sims = (vectors.float() @ vectors.float().t()).cpu().numpy()
        plt.imshow(sims, vmin=-1.0, vmax=1.0, cmap="viridis")
        plt.colorbar(label="cosine")
        plt.title("Source Memory Similarity")
        plt.xlabel("slot")
        plt.ylabel("slot")
    plt.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    rng = random.Random(int(args.seed))

    dataset = Path(args.dataset)
    out_path = Path(args.output)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.weights)
    model = yolo.model.to(args.device).eval()
    core_model, layers = unwrap_core_and_layers(model)
    adapter_layers = parse_adapter_layers(str(args.adapter_layers))
    if int(args.layer) not in adapter_layers:
        adapter_layers.append(int(args.layer))
    layers, _specs = attach_residual_adapters(
        core_model=core_model,
        layers=layers,
        adapter_layers=adapter_layers,
        imgsz=int(args.imgsz),
        device=str(args.device),
        reduction=int(args.adapter_reduction),
        min_channels=int(args.adapter_min_channels),
        scale=float(args.adapter_scale),
        memory_enable=True,
        memory_dim=int(args.slot_dim),
        memory_conditioning="film",
        memory_source_layer=int(args.layer),
        memory_bank_size=max(1, int(args.memory_size)),
    )
    source_adapter = first_memory_source_adapter(model)
    if source_adapter is None:
        raise RuntimeError("Unable to attach source memory projector.")

    captured: Dict[str, Optional[torch.Tensor]] = {"feature": None}

    def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            captured["feature"] = output.detach()

    handle = layers[int(args.layer)].register_forward_hook(hook)
    slots: List[torch.Tensor] = []
    metas: List[Dict[str, Any]] = []
    images_seen = 0
    labels_seen = 0
    try:
        for image_path in list_images(dataset, int(args.max_images), rng):
            with Image.open(image_path) as im:
                width, height = im.size
            labels = read_yolo_labels(label_path_for_image(dataset, image_path), width, height)
            if not labels:
                continue
            images_seen += 1
            captured["feature"] = None
            _ = yolo.predict(
                source=str(image_path),
                device=str(args.device),
                imgsz=int(args.imgsz),
                conf=0.001,
                verbose=False,
                save=False,
            )
            feature = captured["feature"]
            if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
                continue
            for cls_id, box_xyxy in labels:
                labels_seen += 1
                area_frac = max(0.0, (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1])) / max(1.0, width * height)
                pooled = pool_feature_for_original_box(feature[0], box_xyxy, width, height)
                with torch.no_grad():
                    vec = source_adapter.project_memory_feature(pooled)
                    vec = torch.nn.functional.normalize(vec.float(), dim=0, eps=1e-6).cpu()
                if select_slot(vec, slots, int(args.memory_size), float(args.diversity_thresh)):
                    if len(slots) < int(args.memory_size):
                        slots.append(vec)
                        metas.append(
                            {
                                "path": str(image_path),
                                "class_id": int(cls_id),
                                "box_xyxy": [float(v) for v in box_xyxy],
                                "area_frac": float(area_frac),
                                "scale_bin": scale_bin(area_frac),
                            }
                        )
                    else:
                        replace_idx = int(rng.randrange(len(slots)))
                        slots[replace_idx] = vec
                        metas[replace_idx] = {
                            "path": str(image_path),
                            "class_id": int(cls_id),
                            "box_xyxy": [float(v) for v in box_xyxy],
                            "area_frac": float(area_frac),
                            "scale_bin": scale_bin(area_frac),
                        }
                if len(slots) >= int(args.memory_size):
                    active = torch.stack(slots)
                    if float((active @ vec).max().cpu()) < float(args.diversity_thresh):
                        break
            if len(slots) >= int(args.memory_size):
                continue
    finally:
        handle.remove()

    if not slots:
        raise RuntimeError("No labeled source object embeddings were collected.")
    vectors = torch.stack(slots)
    payload = {
        "vectors": vectors,
        "metas": metas,
        "slot_dim": int(vectors.shape[1]),
        "layer": int(args.layer),
        "weights": str(args.weights),
        "dataset": str(dataset),
        "adapter_layers": ",".join(str(v) for v in adapter_layers),
        "adapter_reduction": int(args.adapter_reduction),
        "adapter_scale": float(args.adapter_scale),
        "images_seen": int(images_seen),
        "labels_seen": int(labels_seen),
    }
    torch.save(payload, out_path)

    csv_path = out_dir / "source_memory_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slot", "path", "class_id", "box_xyxy", "area_frac", "scale_bin"])
        writer.writeheader()
        for idx, meta in enumerate(metas):
            row = dict(meta)
            row["slot"] = idx
            writer.writerow(row)
    save_contact_sheet(out_dir / "source_memory_slots.png", metas)
    save_heatmap(out_dir / "source_memory_similarity_heatmap.png", vectors)
    print(
        f"saved={out_path} slots={int(vectors.shape[0])} slot_dim={int(vectors.shape[1])} "
        f"images_seen={images_seen} labels_seen={labels_seen}"
    )


if __name__ == "__main__":
    main()
