#!/usr/bin/env python3
"""
Build a YOLO detection dataset with FDA-augmented synthetic images.

You have:
- Source (labeled): synthetic dataset with YOLO txt labels
- Target(s) (unlabeled): one or more image pools used only as "style donors"

We generate:
- New dataset root (copied from synthetic):
    images/{train,valid,test}
    labels/{train,valid,test}
    masks/{train,valid,test}
- FDA outputs (generated for a chosen split, typically train):
    images/<split>_fda
    labels/<split>_fda
    masks/<split>_fda
- Optionally merge <split>_fda into images/<split> etc.

FDA method follows paper idea: swap low-frequency Fourier amplitude of source with target.
Ref (paper): https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf
Repo: https://github.com/YanchaoYang/FDA

Key extensions:
- Multiple target donor pools via repeated --targets
- K variants per synthetic image via --variants-per-image
- Optional preprocessing of target donors:
    --target-preprocess resize|letterbox
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm
import yaml


# -----------------------------
# FDA core (paper-style)
# -----------------------------
def _fft2(img: np.ndarray) -> np.ndarray:
    return np.fft.fft2(img, axes=(0, 1))


def _ifft2(freq: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(freq, axes=(0, 1))


def fda_source_to_target(src_bgr: np.ndarray, trg_bgr: np.ndarray, beta: float) -> np.ndarray:
    """
    Apply FDA: replace low-frequency amplitude of src with that of trg.
    Keeps src phase, so geometry tends to remain stable.

    src_bgr, trg_bgr: uint8 BGR images (H,W,3)
    beta: fraction of low-frequency region swapped (typical small: 0.01~0.1)
    """
    assert src_bgr.ndim == 3 and src_bgr.shape[2] == 3
    assert trg_bgr.ndim == 3 and trg_bgr.shape[2] == 3

    src = src_bgr.astype(np.float32)
    trg = trg_bgr.astype(np.float32)

    src_f = _fft2(src)
    trg_f = _fft2(trg)

    src_amp, src_phase = np.abs(src_f), np.angle(src_f)
    trg_amp = np.abs(trg_f)

    h, w = src.shape[:2]
    b = int(min(h, w) * beta)
    if b < 1:
        return src_bgr

    src_amp_shift = np.fft.fftshift(src_amp, axes=(0, 1))
    trg_amp_shift = np.fft.fftshift(trg_amp, axes=(0, 1))

    c_h, c_w = h // 2, w // 2
    h1, h2 = c_h - b, c_h + b
    w1, w2 = c_w - b, c_w + b

    src_amp_shift[h1:h2, w1:w2, :] = trg_amp_shift[h1:h2, w1:w2, :]

    src_amp_ = np.fft.ifftshift(src_amp_shift, axes=(0, 1))
    src_f_ = src_amp_ * np.exp(1j * src_phase)

    src_rec = _ifft2(src_f_)
    src_rec = np.real(src_rec)
    src_rec = np.clip(src_rec, 0, 255).astype(np.uint8)
    return src_rec


# -----------------------------
# Utilities
# -----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def resize_to_match(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    if img.shape[0] == th and img.shape[1] == tw:
        return img
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)


def letterbox_to_square(
    img_bgr: np.ndarray,
    side: int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """
    Letterbox image to (side, side) without distortion (preserve aspect ratio + pad).
    Returns BGR uint8.
    """
    h0, w0 = img_bgr.shape[:2]
    if h0 == 0 or w0 == 0:
        raise ValueError("Invalid image with zero dimension")

    r = min(side / w0, side / h0)
    w1 = int(round(w0 * r))
    h1 = int(round(h0 * r))

    resized = cv2.resize(img_bgr, (w1, h1), interpolation=cv2.INTER_AREA)

    dw = side - w1
    dh = side - h1
    left = dw // 2
    right = dw - left
    top = dh // 2
    bottom = dh - top

    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    assert out.shape[0] == side and out.shape[1] == side, f"Letterbox produced {out.shape[:2]} not {(side, side)}"
    return out


def write_image(path: Path, img_bgr: np.ndarray, quality: int = 95) -> None:
    ensure_dir(path.parent)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    else:
        cv2.imwrite(str(path), img_bgr)


def corresponding_label_path(img_path: Path, images_root: Path, labels_root: Path) -> Path:
    rel = img_path.relative_to(images_root)
    return labels_root / rel.with_suffix(".txt")


def corresponding_mask_path(img_path: Path, images_root: Path, masks_root: Path) -> Path:
    rel = img_path.relative_to(images_root)
    return masks_root / rel


@dataclass(frozen=True)
class TargetSource:
    name: str
    root: Path
    weight: float
    images: List[Path]


def parse_target_specs(specs: List[str]) -> List[TargetSource]:
    """
    Parse repeated --targets entries of the form:
      NAME=/path/to/images[:weight]
    Examples:
      lab=/data/lab_images:0.7
      real=/data/real_images:0.3
      lab=/data/lab_images   (defaults weight=1.0)
    """
    out: List[TargetSource] = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"Bad target spec '{s}'. Expected NAME=/path[:weight].")
        name, rhs = s.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Bad target spec '{s}': empty NAME")

        if ":" in rhs:
            path_str, w_str = rhs.rsplit(":", 1)
            weight = float(w_str)
        else:
            path_str, weight = rhs, 1.0

        root = Path(path_str)
        if not root.exists():
            raise FileNotFoundError(f"Target root does not exist: {root}")

        imgs = list_images(root)
        if not imgs:
            raise RuntimeError(f"No images found for target '{name}' in {root}")

        out.append(TargetSource(name=name, root=root, weight=float(weight), images=imgs))

    total = sum(t.weight for t in out)
    if total <= 0:
        raise ValueError("Sum of target weights must be > 0")

    # normalize weights
    normalized = [
        TargetSource(t.name, t.root, t.weight / total, t.images)
        for t in out
    ]
    return normalized


def sample_target(targets: List[TargetSource]) -> TargetSource:
    r = random.random()
    acc = 0.0
    for t in targets:
        acc += t.weight
        if r <= acc:
            return t
    return targets[-1]


@dataclass
class BuildConfig:
    synthetic_root: Path
    out_root: Path
    beta: float = 0.05
    seed: int = 123
    apply_to_split: str = "train"
    jpg_quality: int = 95
    make_merged_train: bool = True

    # New:
    variants_per_image: int = 1
    target_specs: Optional[List[str]] = None
    lab_root: Optional[Path] = None  # backward-compat only (used if no target_specs)
    target_preprocess: str = "resize"  # 'resize' or 'letterbox'
    letterbox_color: Tuple[int, int, int] = (114, 114, 114)


def preprocess_target(
    trg_img_bgr: np.ndarray,
    target_hw: Tuple[int, int],
    mode: str,
    pad_color: Tuple[int, int, int],
) -> np.ndarray:
    th, tw = target_hw
    if mode == "resize":
        return resize_to_match(trg_img_bgr, target_hw)
    if mode == "letterbox":
        if th != tw:
            # For our use-case src is typically square; if not, keep behavior explicit.
            raise ValueError(f"letterbox preprocess expects square target, got {target_hw}")
        return letterbox_to_square(trg_img_bgr, side=th, color=pad_color)
    raise ValueError(f"Unknown target_preprocess mode: {mode}")


def build_dataset(cfg: BuildConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    syn_images_root = cfg.synthetic_root / "images"
    syn_labels_root = cfg.synthetic_root / "labels"
    syn_masks_root = cfg.synthetic_root / "masks"

    assert syn_images_root.exists(), f"Missing {syn_images_root}"
    assert syn_labels_root.exists(), f"Missing {syn_labels_root}"
    assert syn_masks_root.exists(), f"Missing {syn_masks_root}"

    # Resolve target pools
    targets: List[TargetSource]
    if cfg.target_specs and len(cfg.target_specs) > 0:
        targets = parse_target_specs(cfg.target_specs)
    else:
        if cfg.lab_root is None:
            raise ValueError("Provide either --targets (recommended) or --lab-root (legacy).")
        if not cfg.lab_root.exists():
            raise FileNotFoundError(f"Lab root does not exist: {cfg.lab_root}")
        lab_images = list_images(cfg.lab_root)
        if not lab_images:
            raise RuntimeError(f"No lab images found in: {cfg.lab_root}")
        targets = [TargetSource(name="lab", root=cfg.lab_root, weight=1.0, images=lab_images)]

    # 1) Copy synthetic dataset to out_root
    print(f"[1/4] Copying synthetic dataset -> {cfg.out_root}")
    ensure_dir(cfg.out_root)
    copy_tree(cfg.synthetic_root / "images", cfg.out_root / "images")
    copy_tree(cfg.synthetic_root / "labels", cfg.out_root / "labels")
    copy_tree(cfg.synthetic_root / "masks", cfg.out_root / "masks")

    # 2) Generate FDA images for chosen split
    split = cfg.apply_to_split
    src_split_dir = cfg.synthetic_root / "images" / split
    if not src_split_dir.exists():
        raise RuntimeError(f"Split folder not found: {src_split_dir}")

    out_fda_img_dir = cfg.out_root / "images" / f"{split}_fda"
    out_fda_lbl_dir = cfg.out_root / "labels" / f"{split}_fda"
    out_fda_msk_dir = cfg.out_root / "masks" / f"{split}_fda"

    ensure_dir(out_fda_img_dir)
    ensure_dir(out_fda_lbl_dir)
    ensure_dir(out_fda_msk_dir)

    src_imgs = list_images(src_split_dir)
    if not src_imgs:
        raise RuntimeError(f"No synthetic images in: {src_split_dir}")

    if cfg.variants_per_image < 1:
        raise ValueError("--variants-per-image must be >= 1")

    target_str = ", ".join([f"{t.name}:{t.weight:.2f} (n={len(t.images)})" for t in targets])
    print(
        f"[2/4] Generating FDA for split='{split}' | N={len(src_imgs)} | beta={cfg.beta} | "
        f"variants_per_image={cfg.variants_per_image} | target_preprocess={cfg.target_preprocess} | "
        f"targets=[{target_str}]"
    )

    for src_path in tqdm(src_imgs, desc=f"FDA {split}"):
        src_img = read_bgr(src_path)
        target_hw = (src_img.shape[0], src_img.shape[1])

        # label path once (same for all variants)
        src_lbl = corresponding_label_path(src_path, syn_images_root, syn_labels_root)
        if not src_lbl.exists():
            raise RuntimeError(f"Missing label for {src_path}: {src_lbl}")

        # optional mask
        src_msk = corresponding_mask_path(src_path, syn_images_root, syn_masks_root)
        has_mask = src_msk.exists()

        for v in range(cfg.variants_per_image):
            chosen = sample_target(targets)
            trg_path = random.choice(chosen.images)

            trg_img = read_bgr(trg_path)
            trg_img_pp = preprocess_target(
                trg_img_bgr=trg_img,
                target_hw=target_hw,
                mode=cfg.target_preprocess,
                pad_color=cfg.letterbox_color,
            )

            out_img = fda_source_to_target(src_img, trg_img_pp, beta=cfg.beta)

            out_stem = f"{src_path.stem}_fda_{chosen.name}_v{v}"
            out_img_path = out_fda_img_dir / (out_stem + src_path.suffix.lower())
            write_image(out_img_path, out_img, quality=cfg.jpg_quality)

            out_lbl_path = out_fda_lbl_dir / (out_stem + ".txt")
            shutil.copy2(src_lbl, out_lbl_path)

            if has_mask:
                out_msk_path = out_fda_msk_dir / (out_stem + src_msk.suffix.lower())
                shutil.copy2(src_msk, out_msk_path)

    # 3) Optionally merge train + train_fda into train
    if cfg.make_merged_train and split == "train":
        print("[3/4] Merging train_fda into train (images/labels/masks)")
        merged_img_dir = cfg.out_root / "images" / "train"
        merged_lbl_dir = cfg.out_root / "labels" / "train"
        merged_msk_dir = cfg.out_root / "masks" / "train"

        ensure_dir(merged_img_dir)
        ensure_dir(merged_lbl_dir)
        ensure_dir(merged_msk_dir)

        for p in out_fda_img_dir.iterdir():
            shutil.copy2(p, merged_img_dir / p.name)
        for p in out_fda_lbl_dir.iterdir():
            shutil.copy2(p, merged_lbl_dir / p.name)
        for p in out_fda_msk_dir.iterdir():
            shutil.copy2(p, merged_msk_dir / p.name)

    # 4) Write new dataset YAML
    print("[4/4] Writing dataset YAML")
    data_yaml = {
        "path": str(cfg.out_root),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": 1,
        "names": ["soho"],
    }
    yaml_path = cfg.out_root / "dataset_fda.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print("\nDone.")
    print(f"New dataset root: {cfg.out_root}")
    print(f"YAML: {yaml_path}")
    if cfg.make_merged_train and split == "train":
        print("Train images include FDA copies (merged).")
    else:
        print(f"FDA images are in: images/{split}_fda (not merged).")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-root", type=str, required=True, help="Synthetic labeled dataset root")

    # Backward-compat: single target pool
    ap.add_argument(
        "--lab-root",
        type=str,
        default=None,
        help="Legacy: single unlabeled target folder. Prefer using repeated --targets instead.",
    )

    # New: multiple target pools
    ap.add_argument(
        "--targets",
        type=str,
        action="append",
        default=None,
        help=(
            "Repeatable. Target donor pools: NAME=/path/to/images[:weight]. "
            "Example: --targets lab=/path/lab:0.7 --targets real=/path/real:0.3"
        ),
    )

    ap.add_argument("--out-root", type=str, required=True, help="Output dataset root")
    ap.add_argument("--beta", type=float, default=0.05, help="FDA low-frequency swap ratio (e.g., 0.01, 0.05, 0.1)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    ap.add_argument("--jpg-quality", type=int, default=95)
    ap.add_argument(
        "--no-merge-train",
        action="store_true",
        help="If set, keep train_fda separate and do NOT merge into images/train",
    )

    ap.add_argument(
        "--variants-per-image",
        type=int,
        default=1,
        help="How many FDA variants to generate per synthetic image (e.g., 2 => 2x dataset size).",
    )

    ap.add_argument(
        "--target-preprocess",
        type=str,
        default="resize",
        choices=["resize", "letterbox"],
        help="How to bring target donor images to source resolution before FDA: resize or letterbox.",
    )

    ap.add_argument(
        "--letterbox-color",
        type=str,
        default="114,114,114",
        help="BGR pad color for letterbox mode, as 'B,G,R' (default 114,114,114).",
    )

    return ap.parse_args()


def parse_bgr_triplet(s: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected B,G,R triplet like '114,114,114', got '{s}'")
    b, g, r = (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])))
    return (b, g, r)


def main() -> None:
    args = parse_args()

    letterbox_color = parse_bgr_triplet(args.letterbox_color)

    cfg = BuildConfig(
        synthetic_root=Path(args.synthetic_root),
        out_root=Path(args.out_root),
        beta=float(args.beta),
        seed=int(args.seed),
        apply_to_split=str(args.split),
        jpg_quality=int(args.jpg_quality),
        make_merged_train=not bool(args.no_merge_train),
        variants_per_image=int(args.variants_per_image),
        target_specs=args.targets,
        lab_root=Path(args.lab_root) if args.lab_root else None,
        target_preprocess=str(args.target_preprocess),
        letterbox_color=letterbox_color,
    )
    build_dataset(cfg)


if __name__ == "__main__":
    main()

# -----------------------------
# Examples
# -----------------------------
# 1) Single target donor pool (legacy behavior):
# python3 build_fda_yolo_dataset_multi.py \
#   --synthetic-root /home/hm25936/datasets_for_yolo/midLighting_rmag_5m_to_100m \
#   --lab-root /home/hm25936/datasets_for_yolo/lab_images_6000 \
#   --out-root /home/hm25936/datasets_for_yolo/midLighting_rmag_5m_to_100m_FDA \
#   --beta 0.05 \
#   --variants-per-image 1 \
#   --target-preprocess resize
#
# 2) Multiple target donor pools + two variants per synthetic image:
# PYTHONPATH=. python tools/build_fda_yolo_dataset_multi.py \
#   --synthetic-root /home/hm25936/datasets_for_yolo/midLighting_rmag_5m_to_100m \
#   --out-root /home/hm25936/datasets_for_yolo/midLighting_rmag_5m_to_100m_FDA_MIX \
#   --targets lab=/home/hm25936/datasets_for_yolo/lab_images_6000/images/test:0.5 \
#   --targets real=/home/hm25936/datasets_for_yolo/soho/images/test:0.5 \
#   --variants-per-image 2 \
#   --beta 0.05 \
#   --target-preprocess resize
#
# 3) Bias sampling toward lab donors (70/30):
#   --targets lab=/.../lab:0.7 --targets real=/.../real:0.3
#
# Training remains the same (Ultralytics):
# PYTHONPATH=. python yolo/train_yolov8.py \
#   --data /home/hm25936/datasets_for_yolo/midLighting_rmag_5m_to_100m_FDA_MIX/dataset_fda.yaml \
#   --epochs 100 --batch-size 16 --img-size 1024 --device 0,1,2,3 --output runs/yolov8_fda_mix