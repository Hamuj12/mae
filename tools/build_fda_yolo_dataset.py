#!/usr/bin/env python3
"""
Build a YOLO detection dataset with FDA-augmented synthetic images.

You have:
- Source (labeled): synthetic dataset with YOLO txt labels
- Target (unlabeled): lab images used only as "style donors"

We generate:
- New dataset root
  images/train, labels/train, masks/train   (copied from synthetic)
  images/valid, labels/valid, masks/valid   (copied)
  images/test,  labels/test,  masks/test    (copied)
  images/train_fda, labels/train_fda, masks/train_fda (FDA images + copied labels)
And optionally merge train + train_fda into a single train folder if desired.

FDA method follows paper idea: swap low-frequency Fourier amplitude of source with target.
Ref: https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf
Repo: https://github.com/YanchaoYang/FDA
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
    # img: HxW or HxWxC float32
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

    # Convert to float32
    src = src_bgr.astype(np.float32)
    trg = trg_bgr.astype(np.float32)

    # FFT
    src_f = _fft2(src)
    trg_f = _fft2(trg)

    # amplitude + phase
    src_amp, src_phase = np.abs(src_f), np.angle(src_f)
    trg_amp = np.abs(trg_f)

    h, w = src.shape[:2]
    b = int(min(h, w) * beta)
    if b < 1:
        return src_bgr  # no-op if too small

    # shift so low-freq is centered
    src_amp_shift = np.fft.fftshift(src_amp, axes=(0, 1))
    trg_amp_shift = np.fft.fftshift(trg_amp, axes=(0, 1))

    c_h, c_w = h // 2, w // 2
    h1, h2 = c_h - b, c_h + b
    w1, w2 = c_w - b, c_w + b

    # swap the center low-freq amplitude region
    src_amp_shift[h1:h2, w1:w2, :] = trg_amp_shift[h1:h2, w1:w2, :]

    # inverse shift
    src_amp_ = np.fft.ifftshift(src_amp_shift, axes=(0, 1))

    # reconstruct complex FFT: A * exp(i*phase)
    src_f_ = src_amp_ * np.exp(1j * src_phase)

    # iFFT
    src_rec = _ifft2(src_f_)
    src_rec = np.real(src_rec)

    # clip to uint8
    src_rec = np.clip(src_rec, 0, 255).astype(np.uint8)
    return src_rec


# -----------------------------
# Utilities
# -----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
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


def write_image(path: Path, img_bgr: np.ndarray, quality: int = 95) -> None:
    ensure_dir(path.parent)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    else:
        cv2.imwrite(str(path), img_bgr)


def corresponding_label_path(img_path: Path, images_root: Path, labels_root: Path) -> Path:
    """
    Given .../images/train/abc.jpg -> .../labels/train/abc.txt
    """
    rel = img_path.relative_to(images_root)
    return labels_root / rel.with_suffix(".txt")


def corresponding_mask_path(img_path: Path, images_root: Path, masks_root: Path) -> Path:
    """
    Given .../images/train/abc.jpg -> .../masks/train/abc.jpg (same suffix as mask files if needed)
    """
    rel = img_path.relative_to(images_root)
    return masks_root / rel  # same filename/suffix


@dataclass
class BuildConfig:
    synthetic_root: Path
    lab_root: Path
    out_root: Path
    beta: float = 0.05
    seed: int = 123
    apply_to_split: str = "train"  # only FDA-transform train by default
    jpg_quality: int = 95
    make_merged_train: bool = True  # merge train + train_fda into images/train etc.


def build_dataset(cfg: BuildConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    syn_images = cfg.synthetic_root / "images"
    syn_labels = cfg.synthetic_root / "labels"
    syn_masks = cfg.synthetic_root / "masks"

    assert syn_images.exists(), f"Missing {syn_images}"
    assert syn_labels.exists(), f"Missing {syn_labels}"
    assert syn_masks.exists(), f"Missing {syn_masks}"

    lab_images = list_images(cfg.lab_root)
    if len(lab_images) == 0:
        raise RuntimeError(f"No lab images found in: {cfg.lab_root}")

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
    if len(src_imgs) == 0:
        raise RuntimeError(f"No synthetic images in: {src_split_dir}")

    print(f"[2/4] Generating FDA for split='{split}' | N={len(src_imgs)} | beta={cfg.beta}")
    for src_path in tqdm(src_imgs, desc=f"FDA {split}"):
        # choose random lab donor
        trg_path = random.choice(lab_images)

        src_img = read_bgr(src_path)  # 1024x1024
        trg_img = read_bgr(trg_path)  # 1936x1216

        # resize target to match source resolution (FDA expects same H,W)
        trg_img_rs = resize_to_match(trg_img, (src_img.shape[0], src_img.shape[1]))

        out_img = fda_source_to_target(src_img, trg_img_rs, beta=cfg.beta)

        # output filename: keep same base name with suffix _fda
        out_name = src_path.stem + "_fda" + src_path.suffix.lower()
        out_img_path = out_fda_img_dir / out_name
        write_image(out_img_path, out_img, quality=cfg.jpg_quality)

        # copy label unchanged
        src_lbl = corresponding_label_path(src_path, cfg.synthetic_root / "images", cfg.synthetic_root / "labels")
        if not src_lbl.exists():
            raise RuntimeError(f"Missing label for {src_path}: {src_lbl}")
        out_lbl_path = out_fda_lbl_dir / (src_path.stem + "_fda.txt")
        shutil.copy2(src_lbl, out_lbl_path)

        # copy mask if exists (optional)
        src_msk = corresponding_mask_path(src_path, cfg.synthetic_root / "images", cfg.synthetic_root / "masks")
        if src_msk.exists():
            out_msk_path = out_fda_msk_dir / (src_path.stem + "_fda" + src_msk.suffix.lower())
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

        # move/copy fda files into train alongside originals
        for p in out_fda_img_dir.iterdir():
            shutil.copy2(p, merged_img_dir / p.name)
        for p in out_fda_lbl_dir.iterdir():
            shutil.copy2(p, merged_lbl_dir / p.name)
        for p in out_fda_msk_dir.iterdir():
            shutil.copy2(p, merged_msk_dir / p.name)

        # (optional) keep train_fda folders too; harmless.

    # 4) Write new dataset YAML
    print("[4/4] Writing dataset YAML")
    data_yaml = {
        "path": str(cfg.out_root),
        "train": "images/train",
        "val": "images/valid",  # your dataset uses 'valid'
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
    print(f"Train images now include FDA copies: {cfg.make_merged_train and split=='train'}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-root", type=str, required=True, help="Synthetic labeled dataset root")
    ap.add_argument("--lab-root", type=str, required=True, help="Folder containing unlabeled lab images")
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
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BuildConfig(
        synthetic_root=Path(args.synthetic_root),
        lab_root=Path(args.lab_root),
        out_root=Path(args.out_root),
        beta=float(args.beta),
        seed=int(args.seed),
        apply_to_split=str(args.split),
        jpg_quality=int(args.jpg_quality),
        make_merged_train=not bool(args.no_merge_train),
    )
    build_dataset(cfg)


if __name__ == "__main__":
    main()
    
# python3 build_fda_yolo_dataset.py \
#   --synthetic-root /home/hm25936/datasets_for_yolo/midLighting_rmag_5m_to_100m \
#   --lab-root /home/hm25936/datasets_for_yolo/lab_images_6000 \
#   --out-root /home/hm25936/datasets_for_yolo/midLighting_rmag_5m_to_100m_FDA \
#   --beta 0.05

# PYTHONPATH=. python yolo/train_yolov8.py \
#   --data /home/hm25936/datasets_for_yolo/midLighting_rmag_5m_to_100m_FDA/dataset_fda.yaml \
#   --epochs 100 \
#   --batch-size 16 \
#   --img-size 1024 \
#   --device 0,1,2,3 \
#   --output runs/yolov8_fda