#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SANITY_ROOT="${ROOT_DIR}/scripts/.sanity"
DATASET_DIR="${SANITY_ROOT}/toy_imagenet"
YOLO_DIR="${SANITY_ROOT}/toy_yolo"
PRETRAIN_OUT="${SANITY_ROOT}/pretrain"
LINPROBE_OUT="${SANITY_ROOT}/linprobe"
YOLO_CONFIG="${SANITY_ROOT}/dual_config.yaml"

mkdir -p "${SANITY_ROOT}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MAE_ROOT="${ROOT_DIR}"
export SANITY_DATASET_DIR="${DATASET_DIR}"
export SANITY_YOLO_DIR="${YOLO_DIR}"

python - <<'PY'
import os
import random
import shutil
from pathlib import Path

root = Path(os.environ["MAE_ROOT"])
sanity_dir = Path(os.environ["SANITY_DATASET_DIR"])
yolo_dir = Path(os.environ["SANITY_YOLO_DIR"])
source_images = sorted((root / "test_imgs").glob("*"))
if not source_images:
    raise SystemExit("No test images found for sanity dataset generation.")

# Prepare classification dataset
for split in ("train", "val"):
    class_dir = sanity_dir / split / "class0"
    class_dir.mkdir(parents=True, exist_ok=True)

random.seed(0)
train_images = source_images * 4  # duplicate to ensure enough samples
val_images = source_images * 2
for idx, img in enumerate(train_images):
    shutil.copy(img, sanity_dir / "train" / "class0" / f"img_{idx:03d}{img.suffix}")
for idx, img in enumerate(val_images):
    shutil.copy(img, sanity_dir / "val" / "class0" / f"img_{idx:03d}{img.suffix}")

# Prepare YOLO dataset with synthetic boxes
for split in ("train", "val"):
    (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

num_train = 50
num_val = 10

def write_label(path: Path):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("0 0.5 0.5 0.6 0.6\n")

for idx in range(num_train):
    src = source_images[idx % len(source_images)]
    dst = yolo_dir / "images" / "train" / f"img_{idx:03d}{src.suffix}"
    shutil.copy(src, dst)
    write_label(yolo_dir / "labels" / "train" / f"img_{idx:03d}.txt")

for idx in range(num_val):
    src = source_images[idx % len(source_images)]
    dst = yolo_dir / "images" / "val" / f"img_{idx:03d}{src.suffix}"
    shutil.copy(src, dst)
    write_label(yolo_dir / "labels" / "val" / f"img_{idx:03d}.txt")
PY

python - <<'PY'
import os
from utils import wandb_utils
os.environ.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))
run = wandb_utils.init_wandb("mae-sanity", "dry-run", {"phase": "sanity"}, resume="never")
wandb_utils.log_metrics(step=0, wandb_dry_run=1.0)
if run is not None:
    run.finish()
print("W&B offline dry run succeeded.")
PY

python "${ROOT_DIR}/main_pretrain.py" \
  --data_path "${DATASET_DIR}" \
  --epochs 1 --mask_ratio 0.75 --blr 0.0015 \
  --batch_size 4 --accum_iter 1 --world_size 1 \
  --device cpu --num_workers 0 \
  --output_dir "${PRETRAIN_OUT}" \
  --log_dir "${PRETRAIN_OUT}/logs" \
  --wandb_project "mae-sanity" \
  --run_name "sanity-pretrain" \
  --probe_data_path "${DATASET_DIR}" \
  --no_auto_probe

CKPT_PATH="${PRETRAIN_OUT}/checkpoint-0.pth"
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Expected checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi

python "${ROOT_DIR}/main_linprobe.py" \
  --data_path "${DATASET_DIR}" \
  --epochs 1 --batch_size 4 --accum_iter 1 \
  --device cpu --num_workers 0 \
  --finetune "${CKPT_PATH}" \
  --output_dir "${LINPROBE_OUT}" \
  --log_dir "${LINPROBE_OUT}/logs" \
  --wandb_project "mae-sanity" \
  --run_name "sanity-linprobe"

cat <<CFG > "${YOLO_CONFIG}"
model:
  num_classes: 1
  input_size: 640
  mae:
    checkpoint: "${CKPT_PATH}"
    output_dims: [64, 64, 64]
    freeze: true
  yolo:
    weights: "${ROOT_DIR}/yolo11n.pt"
    freeze: true
  fusion:
    channels: [64, 64, 64]

dataset:
  path: "${YOLO_DIR}"
  train_split: train
  val_split: val
  class_names: ["dummy"]
  augment: false

training:
  epochs: 1
  batch_size: 1
  lr: 5e-4
  weight_decay: 0.0
  num_workers: 0
  precision: 32
  accelerator: cpu
  devices: 1
  checkpoint_dir: "${SANITY_ROOT}/dual_ckpts"
  gradient_clip_val: 0.0
  seed: 0

logging:
  wandb_project: "mae-sanity"
  run_name: "sanity-dual"
CFG

python "${ROOT_DIR}/dual_yolo_mae/train.py" --config "${YOLO_CONFIG}"

echo "Sanity checks completed successfully."
