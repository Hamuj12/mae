#!/usr/bin/env python3
# Phase_B_top3_tmux.py
#
# Launch 3 long Phase B runs (your top-3 Phase A configs).
# - Defaults to sequential dispatch in a single tmux session/window.
# - Uses 4 GPUs per job (change GPUS_PER_JOB if you want parallelization).
#
# Usage (example):
#   DATA_ROOT=/path/to/train \
#   PROBE_DATA=/path/to/probe \
#   WANDB_PROJECT=mae-space \
#   python Phase_B_top3_tmux.py
#
# Tip: If you’re already inside tmux, AUTO_ATTACH will not attach again.

import os, shlex, subprocess, sys, time
from pathlib import Path

# ----------------- knobs -----------------
SESSION_NAME = f"phaseB_top3_{time.strftime('%m%d_%H%M')}"
DRY_RUN = False               # True => print only, do not launch tmux
PARALLEL = False              # False => run all jobs sequentially in one window
GPUS_PER_JOB = 4              # torchrun --nproc_per_node
BATCH_PER_GPU = 32            # keep consistent with your earlier runs
PROBE_EVERY_K = 50
PROBE_EPOCHS = 10             # <— per your request
TOTAL_EPOCHS = 400

AUTO_ATTACH = True
INSIDE_TMUX = bool(os.environ.get("TMUX"))

MODEL = "mae_vit_base_patch16"

# ----- Top-3 from Phase A (as selected) -----
# 1) phaseA_mr85_blr0p0016_wd010_wu40
TOP3 = [
    dict(mask=0.85, blr=1.6e-3, wd=0.10, wu=40, tag="A1_mr85_blr0p0016_wd010_wu40"),
    # 2) phaseA_mr85_blr0p0016_wd010_wu20
    dict(mask=0.85, blr=1.6e-3, wd=0.10, wu=20, tag="A2_mr85_blr0p0016_wd010_wu20"),
    # 3) phaseA_mr85_blr0p0012_wd010_wu20
    dict(mask=0.85, blr=1.2e-3, wd=0.10, wu=20, tag="A3_mr85_blr0p0012_wd010_wu20"),
]

# ------------- env inputs ----------------
DATA_ROOT     = os.environ.get("DATA_ROOT",     "/home/hm25936/mae_datasets/midLighting_rmag_5m_to_100m_background_only")
PROBE_DATA    = os.environ.get("PROBE_DATA",    "/home/hm25936/mae_datasets/probe-space-split")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "mae-space")

# Base folders
OUT_BASE = Path("outputs") / "phaseB_top3"
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
    return " ".join(shlex.quote(x) for x in cmd)

def nice_name(mask, blr, wd, wu, epochs, prefix="phaseBtop3"):
    mr = f"mr{int(mask*100):02d}"
    bl = f"blr{blr:.4g}".replace("-", "m").replace(".", "p")
    ww = f"wd{int(wd*1000):03d}"
    wu_ = f"wu{wu}"
    return f"{prefix}_{mr}_{bl}_{ww}_{wu_}_e{epochs}"

# ------------- build job list ------------
jobs = []
for cfg in TOP3:
    out = OUT_BASE / cfg["tag"]
    run = nice_name(cfg["mask"], cfg["blr"], cfg["wd"], cfg["wu"], TOTAL_EPOCHS)
    cmd = build_cmd(cfg["mask"], cfg["blr"], cfg["wd"], cfg["wu"], TOTAL_EPOCHS, out, run)
    jobs.append((cfg["tag"], cmd))

# ------------- tmux helpers --------------
def tmux(*args):
    return subprocess.run(["tmux", *args], check=True)

def start_tmux_sequential(session, jobs):
    # one window; chain jobs sequentially
    chain = "set -e\n"
    for i,(title,cmd) in enumerate(jobs):
        chain += f'echo "=== [{i+1}/{len(jobs)}] {title} START ==="\n'
        chain += f"{cmd}\n"
        chain += f'echo "=== [{i+1}/{len(jobs)}] {title} DONE ==="\n'
    chain += 'echo "All Phase B top-3 jobs finished."\nexec bash\n'
    tmux("new-session", "-d", "-s", session, "bash")
    tmux("send-keys", "-t", f"{session}:0", chain, "C-m")
    tmux("rename-window", "-t", f"{session}:0", "phaseB_top3_seq")

def start_tmux_parallel(session, jobs):
    tmux("new-session", "-d", "-s", session, "bash")
    tmux("rename-window", "-t", f"{session}:0", jobs[0][0])
    tmux("send-keys", "-t", f"{session}:0", jobs[0][1], "C-m")
    for i,(title,cmd) in enumerate(jobs[1:], start=1):
        tmux("new-window", "-t", session, "-n", title, "bash")
        tmux("send-keys", "-t", f"{session}:{i}", cmd, "C-m")

# ------------- launch --------------------
print(f"\nSession: {SESSION_NAME}")
print(f"WANDB_PROJECT={WANDB_PROJECT}")
print(f"DATA_ROOT={DATA_ROOT}")
print(f"PROBE_DATA={PROBE_DATA}")
print(f"Jobs: {len(jobs)}")
for idx,(title,cmd) in enumerate(jobs,1):
    print(f"\n[{idx}] {title}\n{cmd}")

if DRY_RUN:
    print("\nDRY_RUN=True -> not launching tmux.")
    sys.exit(0)

try:
    if PARALLEL:
        # WARNING: with GPUS_PER_JOB=4 you likely can’t run these in parallel on a 4-GPU box.
        start_tmux_parallel(SESSION_NAME, jobs)
    else:
        start_tmux_sequential(SESSION_NAME, jobs)
    print(f"\nStarted tmux session '{SESSION_NAME}'.")
    print(f"Attach later with:\n  tmux attach -t {SESSION_NAME}\n")
    # Only auto-attach if NOT already inside tmux and flag is enabled
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