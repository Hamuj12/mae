#!/usr/bin/env python3
"""
Phase_C_tmux.py

Final long MAE pretraining run ("Phase C") in a tmux session.

- Uses the best config from Phase A/B analysis.
- Single job, long horizon (default: 1000 epochs).
- Runs with 4 GPUs via torchrun.
- Periodic auto linear probes for monitoring only.

Usage example:
  DATA_ROOT=/path/to/train \
  PROBE_DATA=/path/to/probe \
  WANDB_PROJECT=mae-space \
  python Phase_C_tmux.py

If you're already inside tmux, AUTO_ATTACH won't re-attach.
"""

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# ----------------- knobs -----------------
SESSION_NAME   = f"phaseC_{time.strftime('%m%d_%H%M')}"
DRY_RUN        = False          # True => print commands only, don't launch tmux
GPUS_PER_JOB   = 4              # torchrun --nproc_per_node
BATCH_PER_GPU  = 32
TOTAL_EPOCHS   = 1000           # <- main horizon; bump if you want (e.g. 1200/1600)
PROBE_EVERY_K  = 50
PROBE_EPOCHS   = 10

AUTO_ATTACH    = True
INSIDE_TMUX    = bool(os.environ.get("TMUX"))

MODEL = "mae_vit_base_patch16"

# Best hyperparameters from Phase B analysis
BEST_MASK = 0.85
BEST_BLR  = 1.6e-3
BEST_WD   = 0.10
BEST_WU   = 40

# Optional: if you ever want to RESUME instead of fresh:
# export PHASEC_RESUME=/path/to/checkpoint-XXX.pth
RESUME_PATH = os.environ.get("PHASEC_RESUME", "").strip()

# ------------- env inputs ----------------
DATA_ROOT     = os.environ.get(
    "DATA_ROOT",
    "/home/hm25936/mae_datasets/midLighting_rmag_5m_to_100m_background_only",
)
PROBE_DATA    = os.environ.get(
    "PROBE_DATA",
    "/home/hm25936/mae_datasets/probe-space-split",
)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "mae-space")

# Base folders
OUT_BASE = Path("outputs") / "phaseC"
OUT_BASE.mkdir(parents=True, exist_ok=True)

# ------------- command builder -----------
def build_cmd(mask, blr, wd, wu, epochs, outdir, run_name):
    cmd = [
        "torchrun", f"--nproc_per_node={GPUS_PER_JOB}", "main_pretrain.py",
        "--model", MODEL,
        "--mask_ratio", str(mask),
        "--blr", str(blr),
        "--weight_decay", str(wd),
        "--warmup_epochs", str(wu),
        "--epochs", str(epochs),
        "--batch_size", str(BATCH_PER_GPU),
        "--accum_iter", "1",
        "--data_path", DATA_ROOT,
        "--output_dir", str(outdir),
        "--wandb_project", WANDB_PROJECT,
        "--run_name", run_name,
        "--wandb_resume", "never",
        "--probe_data_path", PROBE_DATA,
        "--probe_every_k", str(PROBE_EVERY_K),
        "--probe_epochs", str(PROBE_EPOCHS),
    ]

    # Optional resume: if PHASEC_RESUME is set, chain it in.
    if RESUME_PATH:
        cmd += ["--resume", RESUME_PATH]
        # Note: we let main_pretrain infer start_epoch from the checkpoint.

    return " ".join(shlex.quote(x) for x in cmd)

def nice_name(mask, blr, wd, wu, epochs, prefix="phaseC"):
    mr = f"mr{int(mask * 100):02d}"
    bl = f"blr{blr:.4g}".replace("-", "m").replace(".", "p")
    ww = f"wd{int(wd * 1000):03d}"
    wu_ = f"wu{wu}"
    return f"{prefix}_{mr}_{bl}_{ww}_{wu_}_e{epochs}"

# ------------- build job list ------------
run_name = nice_name(BEST_MASK, BEST_BLR, BEST_WD, BEST_WU, TOTAL_EPOCHS)
out_dir  = OUT_BASE / run_name
cmd      = build_cmd(BEST_MASK, BEST_BLR, BEST_WD, BEST_WU, TOTAL_EPOCHS, out_dir, run_name)

jobs = [(run_name, cmd)]

# ------------- tmux helpers --------------
def tmux(*args):
    return subprocess.run(["tmux", *args], check=True)

def start_tmux_sequential(session, jobs):
    # one window; here it's just a single job, but keep the pattern
    chain = "set -e\n"
    for i, (title, c) in enumerate(jobs, start=1):
        chain += f'echo "=== [{i}/{len(jobs)}] {title} START ==="\n'
        chain += f"{c}\n"
        chain += f'echo "=== [{i}/{len(jobs)}] {title} DONE ==="\n'
    chain += 'echo "All Phase C jobs finished."\nexec bash\n'

    tmux("new-session", "-d", "-s", session, "bash")
    tmux("send-keys", "-t", f"{session}:0", chain, "C-m")
    tmux("rename-window", "-t", f"{session}:0", "phaseC")

# ------------- launch --------------------
print(f"\nSession: {SESSION_NAME}")
print(f"WANDB_PROJECT={WANDB_PROJECT}")
print(f"DATA_ROOT={DATA_ROOT}")
print(f"PROBE_DATA={PROBE_DATA}")
print(f"RESUME_PATH={RESUME_PATH or '(fresh run)'}")
print(f"Jobs: {len(jobs)}")
for idx, (title, c) in enumerate(jobs, 1):
    print(f"\n[{idx}] {title}\n{c}")

if DRY_RUN:
    print("\nDRY_RUN=True -> not launching tmux.")
    sys.exit(0)

try:
    start_tmux_sequential(SESSION_NAME, jobs)
    print(f"\nStarted tmux session '{SESSION_NAME}'.")
    print(f"Attach later with:\n  tmux attach -t {SESSION_NAME}\n")

    if AUTO_ATTACH and not INSIDE_TMUX:
        try:
            tmux("attach-session", "-t", SESSION_NAME)
        except subprocess.CalledProcessError as e:
            print("Non-fatal: tmux attach failed (likely already inside tmux).")
            print(e)
    else:
        print("Not auto-attaching (already in tmux or AUTO_ATTACH=False).")
except subprocess.CalledProcessError as e:
    print("tmux returned an error (often harmless when already inside tmux):")
    print(e)