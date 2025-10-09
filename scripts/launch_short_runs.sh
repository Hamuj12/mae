#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-/path/to/imagenet_like_corpus}
PROBE_DATA_PATH=${PROBE_DATA_PATH:-${DATA_ROOT}/probes_mini}
WAND_PROJECT=${WANDB_PROJECT:-mae-space}
WANDB_MODE=${WANDB_MODE:-online}
export WANDB_MODE

mkdir -p "${ROOT_DIR}/outputs"

echo "Launching MAE sweeps with DATA_ROOT=${DATA_ROOT}" >&2
echo "Linear probe dataset: ${PROBE_DATA_PATH}" >&2

declare -a pids=()

python "${ROOT_DIR}/main_pretrain.py" \
  --model mae_vit_base_patch16 --mask_ratio 0.75 \
  --epochs 200 --batch_size 128 --accum_iter 2 --blr 1.5e-3 \
  --weight_decay 0.05 --warmup_epochs 20 \
  --output_dir "${ROOT_DIR}/outputs/A_baseline" \
  --data_path "${DATA_ROOT}/general_150k" \
  --wandb_project "${WAND_PROJECT}" --run_name "A_vitb16_m75_dec512x8_normFalse" \
  --probe_data_path "${PROBE_DATA_PATH}" \
  --wandb_resume auto \
  --world_size 1 \
  --num_workers 8 \
  --device cuda \
  &
pids+=($!)

python "${ROOT_DIR}/main_pretrain.py" \
  --model mae_vit_base_patch16 --mask_ratio 0.75 \
  --norm_pix_loss \
  --epochs 200 --batch_size 96 --accum_iter 2 --blr 1.2e-3 \
  --weight_decay 0.05 --warmup_epochs 20 \
  --output_dir "${ROOT_DIR}/outputs/B_normed" \
  --data_path "${DATA_ROOT}/general_150k" \
  --wandb_project "${WAND_PROJECT}" --run_name "B_vitb16_m75_dec512x8_normTrue" \
  --probe_data_path "${PROBE_DATA_PATH}" \
  --wandb_resume auto \
  --world_size 1 \
  --num_workers 8 \
  --device cuda \
  &
pids+=($!)

python "${ROOT_DIR}/main_pretrain.py" \
  --model mae_vit_base_patch16 --mask_ratio 0.65 \
  --norm_pix_loss \
  --epochs 200 --batch_size 96 --accum_iter 1 --blr 1.2e-3 \
  --weight_decay 0.05 --warmup_epochs 20 \
  --output_dir "${ROOT_DIR}/outputs/C_mask65" \
  --data_path "${DATA_ROOT}/general_150k" \
  --wandb_project "${WAND_PROJECT}" --run_name "C_vitb16_m65_dec512x8_normTrue" \
  --probe_data_path "${PROBE_DATA_PATH}" \
  --wandb_resume auto \
  --world_size 1 \
  --num_workers 8 \
  --device cuda \
  &
pids+=($!)

trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done' INT TERM

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "All short-run jobs completed." >&2
