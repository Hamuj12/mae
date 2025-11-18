#!/usr/bin/env python3
# Phase_B_tmux.py
#
# Launch Phase B MAE pretraining runs in tmux.
# - One long run for the top config (400 epochs, 4 GPUs).
# - A small local sweep (200 epochs) around the winner.
#
# Usage:
#   DATA_ROOT=/path/to/imagenet \
#   PROBE_DATA=/path/to/probe_set \
#   WANDB_PROJECT=YourProject \
#   python Phase_B_tmux.py
#
# Notes:
# - Default is SEQUENTIAL dispatch (safe when GPUS_PER_JOB == total GPUs).
# - Set DRY_RUN=True to just print the tmux plan and commands.

import os, shlex, subprocess, time
from pathlib import Path

# ----------------- knobs -----------------
SESSION_NAME = f"phaseB_{time.strftime('%m%d_%H%M')}"
DRY_RUN = False               # True => print only, do not launch tmux
PARALLEL = False              # False => run all jobs sequentially in one tmux window
GPUS_PER_JOB = 4              # torchrun --nproc_per_node
BATCH_PER_GPU = 32            # matches your previous runs
PROBE_EVERY_K = 50
PROBE_EPOCHS = 5

AUTO_ATTACH = True  # attach only if we're NOT already inside tmux
INSIDE_TMUX = bool(os.environ.get("TMUX"))

# Phase A winner (from your results)
TOP_MASK = 0.85
TOP_BLR  = 1.6e-3
TOP_WD   = 0.10
TOP_WU   = 40

# Local-sweep options (tight box around the winner)
MASK_CANDIDATES = [0.85]             # you can add [0.80, 0.90] if you want
BLR_CANDIDATES  = [1.4e-3, 1.6e-3, 1.8e-3]
WD_CANDIDATES   = [0.08, 0.10, 0.12]
WU_CANDIDATES   = [30, 40, 50]

# Include top-2/top-3 seeds as variants (optional)
INCLUDE_TOP3_SEEDS = False
TOP2 = dict(mask=0.80, blr=1.2e-3, wd=0.10, wu=40)  # adjust if you want
TOP3 = dict(mask=0.85, blr=1.2e-3, wd=0.05, wu=40)  # adjust if you want

# ------------- env inputs ----------------
DATA_ROOT    = os.environ.get("DATA_ROOT",    "/home/hm25936/mae_datasets/midLighting_rmag_5m_to_100m_background_only")
PROBE_DATA   = os.environ.get("PROBE_DATA",   "/home/hm25936/mae_datasets/probe-space-split")
WANDB_PROJECT= os.environ.get("WANDB_PROJECT", "mae-sweeps")

MODEL = "mae_vit_base_patch16"

# Base folders
OUT_BASE = Path("outputs") / "phaseB"
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

# ------------- build job list ------------
jobs = []

# B1: long run for the best config
long_out = OUT_BASE / "top1_long"
long_run = f"phaseB_mr{int(TOP_MASK*100):02d}_blr{TOP_BLR:.4g}_wd{int(TOP_WD*1000):03d}_wu{TOP_WU}_e400"
cmd_long = build_cmd(TOP_MASK, TOP_BLR, TOP_WD, TOP_WU, 400, long_out, long_run)
jobs.append(("B1_top1_long", cmd_long))

# B2: local sweep around the winner (200 epochs)
def nice_name(mask, blr, wd, wu):
    mr = f"mr{int(mask*100):02d}"
    bl = f"blr{blr:.4g}".replace("-", "m").replace(".", "p")
    ww = f"wd{int(wd*1000):03d}"
    wu_ = f"wu{wu}"
    return f"phaseB_{mr}_{bl}_{ww}_{wu_}_e200"

variants = []
for m in MASK_CANDIDATES:
    for b in BLR_CANDIDATES:
        for w in WD_CANDIDATES:
            for wu in WU_CANDIDATES:
                # keep the sweep compact; only vary one or two dims at a time
                # here we restrict to combos close to the top (simple heuristic)
                if (abs(m - TOP_MASK) <= 0.05) and (abs(b - TOP_BLR) <= TOP_BLR*0.25) and (abs(w - TOP_WD) <= 0.04):
                    variants.append((m,b,w,wu))

# de-duplicate and pick a manageable set (~6–8)
seen = set()
compact = []
for v in variants:
    key = (round(v[0],3), round(v[1],6), round(v[2],3), int(v[3]))
    if key in seen:
        continue
    seen.add(key)
    compact.append(v)
    if len(compact) >= 8:
        break

for (m,b,w,wu) in compact:
    out = OUT_BASE / "locals" / f"{int(m*100):02d}_{b:.4g}_{int(w*1000):03d}_{wu}"
    run_name = nice_name(m,b,w,wu)
    jobs.append((f"B2_{run_name}", build_cmd(m,b,w,wu,200,out,run_name)))

# Optional: seed a couple of variants around #2/#3
if INCLUDE_TOP3_SEEDS:
    for seed in (TOP2, TOP3):
        m,b,w,wu = seed["mask"], seed["blr"], seed["wd"], seed["wu"]
        out = OUT_BASE / "seeds" / f"{int(m*100):02d}_{b:.4g}_{int(w*1000):03d}_{wu}"
        run_name = nice_name(m,b,w,wu)
        jobs.append((f"B2_seed_{run_name}", build_cmd(m,b,w,wu,200,out,run_name)))

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
    chain += 'echo "All Phase B jobs finished."\nexec bash\n'
    tmux("new-session", "-d", "-s", session, "bash")
    tmux("send-keys", "-t", f"{session}:0", chain, "C-m")
    tmux("rename-window", "-t", f"{session}:0", "phaseB_seq")

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
        start_tmux_parallel(SESSION_NAME, jobs)
    else:
        start_tmux_sequential(SESSION_NAME, jobs)
    print(f"\nStarted tmux session '{SESSION_NAME}'.")
    print(f"Attach later with:\n  tmux attach -t {SESSION_NAME}\n")
    # Only auto-attach if NOT already inside tmux and flag is enabled
    if AUTO_ATTACH and not INSIDE_TMUX:
        tmux("attach-session", "-t", SESSION_NAME)
    else:
        print("Not auto-attaching (either already inside tmux or AUTO_ATTACH=False).")
except subprocess.CalledProcessError as e:
    # Don't crash the launcher if attach fails (e.g., we're in tmux)
    print("tmux returned an error (often harmless when already inside tmux):")
    print(e)