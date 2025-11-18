#!/usr/bin/env python3
"""
Phase_A_tmux.py
Launch a Phase-A sweep in tmux, two runs at a time (each uses 2 GPUs).

Grid:
- mask_ratio      : [0.70, 0.75, 0.80, 0.85]
- blr (base LR)   : [9e-4, 1.2e-3, 1.6e-3]
- weight_decay    : [0.05, 0.10]
- warmup_epochs   : [20, 40]

Defaults:
- epochs          : 120 (shorter “A/B” phase; lock best for 200 later)
- probe_every_k   : 30
- probe_epochs    : 2
- batch_size/gpu  : 32 (safe default for 24GB class GPUs, adjust if you want)
- model           : mae_vit_base_patch16

You’ll get tmux sessions like:  phaseA_mr080_blr1e-3_wd005_wu40
Outputs go under:              outputs/sweeps/phaseA/<run_name>/
"""

import itertools
import os
import shlex
import subprocess
import time
from pathlib import Path

# ------------------- USER CONFIG -------------------

# Preview only (no tmux is started). Set to False to actually launch.
# DRY_RUN = True
DRY_RUN = False

# Two concurrent jobs total, each job uses 2 GPUs -> fits a 4-GPU node.
MAX_CONCURRENT_SLOTS = 2
GPU_SLOTS = [
    (0, 1),  # slot 0 uses GPUs 0 & 1
    (2, 3),  # slot 1 uses GPUs 2 & 3
]

# Torchrun master port base (unique per job to avoid collisions)
MASTER_PORT_BASE = 29500

# Training hyperparams for Phase A
EPOCHS = 120
PROBE_EVERY_K = 30
PROBE_EPOCHS = 2
BATCH_SIZE_PER_GPU = 32

MODEL = "mae_vit_base_patch16"

# Sweep grid (Phase A)
mask_ratios   = [0.70, 0.75, 0.80, 0.85]
base_lrs      = [9.0e-4, 1.2e-3, 1.6e-3]
weight_decays = [0.05, 0.10]
warmups       = [20, 40]

# Project & data (from env; change if you prefer hard-coding)
DATA_ROOT     = os.environ.get("DATA_ROOT", "/home/hm25936/mae_datasets/midLighting_rmag_5m_to_100m_background_only")
PROBE_DATA    = os.environ.get("PROBE_DATA", "/home/hm25936/mae_datasets/probe-space-split")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "mae-sweeps")

# Where to save outputs/logs
OUT_ROOT = Path("outputs/sweeps/phaseA")

# ---------------------------------------------------

def tmux_has_session(name: str) -> bool:
    r = subprocess.run(["tmux", "has-session", "-t", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0

def tmux_kill_session(name: str) -> None:
    subprocess.run(["tmux", "kill-session", "-t", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def tmux_new_session(name: str, command: str, cwd: Path | None = None, env: dict | None = None) -> None:
    # Build env prefix string for tmux
    env_prefix = ""
    if env:
        parts = [f'{k}={shlex.quote(str(v))}' for k, v in env.items()]
        env_prefix = " ".join(parts) + " "
    full_cmd = env_prefix + command
    args = ["tmux", "new-session", "-d", "-s", name, full_cmd]
    subprocess.check_call(args, cwd=str(cwd) if cwd else None)

def format_run_name(mr, blr, wd, wu) -> str:
    mr_s  = f"mr{int(round(mr*100)):02d}"
    wd_s  = f"wd{int(round(wd*100)):03d}"
    # Pretty BLR (1.2e-3 -> 1p2e-3)
    blr_s = f"blr{str(blr).replace('.', 'p')}"
    wu_s  = f"wu{wu}"
    return f"phaseA_{mr_s}_{blr_s}_{wd_s}_{wu_s}"

def build_command(run_name: str, out_dir: Path, mr, blr, wd, wu, master_port: int) -> str:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "train.out"

    cmd_parts = [
        "torchrun",
        f"--nproc_per_node=2",
        f"--master_port={master_port}",
        "main_pretrain.py",
        "--model", MODEL,
        "--mask_ratio", str(mr),
        "--norm_pix_loss",
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE_PER_GPU),
        "--accum_iter", "1",
        "--blr", str(blr),
        "--weight_decay", str(wd),
        "--warmup_epochs", str(wu),
        "--data_path", shlex.quote(DATA_ROOT),
        "--output_dir", shlex.quote(str(out_dir)),
        "--wandb_project", shlex.quote(WANDB_PROJECT),
        "--run_name", shlex.quote(run_name),
        "--wandb_resume", "never",
        "--probe_data_path", shlex.quote(PROBE_DATA),
        "--probe_every_k", str(PROBE_EVERY_K),
        "--probe_at_epochs", '""',
        # keep probes lightweight
        "--probe_epochs", str(PROBE_EPOCHS),
    ]

    # Stream logs to file (and keep tmux pane output)
    cmd = " ".join(cmd_parts) + f" |& tee -a {shlex.quote(str(logfile))}"
    return cmd

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(mask_ratios, base_lrs, weight_decays, warmups))
    print(f"[info] Total runs in Phase A grid: {len(grid)}")
    print(f"[info] DRY_RUN={DRY_RUN}  |  concurrent slots={MAX_CONCURRENT_SLOTS}  |  GPU_SLOTS={GPU_SLOTS}\n")

    # Build per-slot queues
    slot_queues = [[] for _ in range(MAX_CONCURRENT_SLOTS)]
    for i, (mr, blr, wd, wu) in enumerate(grid):
        slot_idx = i % MAX_CONCURRENT_SLOTS
        run_name = format_run_name(mr, blr, wd, wu)
        out_dir  = OUT_ROOT / run_name
        master_port = MASTER_PORT_BASE + i  # unique port per job

        slot_queues[slot_idx].append({
            "index": i,
            "mr": mr, "blr": blr, "wd": wd, "wu": wu,
            "run_name": run_name,
            "out_dir": out_dir,
            "master_port": master_port,
        })

    # One active session per slot at a time; launch next when previous ends
    active = [None] * MAX_CONCURRENT_SLOTS

    def launch(slot_idx: int, job: dict):
        g0, g1 = GPU_SLOTS[slot_idx]
        env = {
            "CUDA_VISIBLE_DEVICES": f"{g0},{g1}",
            "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
            "WANDB_PROJECT": WANDB_PROJECT,
            # Optional grouping at the project level:
            "WANDB_RUN_GROUP": "phaseA",
        }
        run_name   = job["run_name"]
        out_dir    = job["out_dir"]
        master_port = job["master_port"]
        cmd = build_command(run_name, out_dir, job["mr"], job["blr"], job["wd"], job["wu"], master_port)
        session = run_name

        print(f"[slot {slot_idx}] -> {session}  (GPUs {g0},{g1})")
        print(f"  tmux new-session -d -s {session}  (cwd={Path.cwd()})")
        print(f"  ENV: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}  WANDB_PROJECT={WANDB_PROJECT}")
        print(f"  CMD: {cmd}\n")

        if not DRY_RUN:
            # avoid collision if session exists (e.g., rerun)
            if tmux_has_session(session):
                print(f"  [warn] session {session} already exists; killing it.")
                tmux_kill_session(session)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "logs").mkdir(parents=True, exist_ok=True)
            tmux_new_session(session, cmd, cwd=Path.cwd(), env=env)

        return session

    # Start one per slot
    for s in range(MAX_CONCURRENT_SLOTS):
        if slot_queues[s]:
            job = slot_queues[s].pop(0)
            if not DRY_RUN:
                active[s] = launch(s, job)
            else:
                launch(s, job)
                active[s] = f"(dry)_{job['run_name']}"

    if DRY_RUN:
        print("[DRY_RUN] Planned all sessions; not monitoring tmux. Set DRY_RUN=False to actually launch.")
        return

    # Monitor: when a session finishes, start next job on that slot
    print("[info] Monitoring tmux sessions; launching queued jobs as slots free up...")
    while any(active) or any(slot_queues):
        for s in range(MAX_CONCURRENT_SLOTS):
            sess = active[s]
            if sess is None:
                # idle slot, try to launch next
                if slot_queues[s]:
                    job = slot_queues[s].pop(0)
                    active[s] = launch(s, job)
                continue
            # check if session is still alive
            if not tmux_has_session(sess):
                print(f"[slot {s}] session {sess} ended.")
                active[s] = None
                if slot_queues[s]:
                    job = slot_queues[s].pop(0)
                    active[s] = launch(s, job)
        time.sleep(30)

    print("\n[done] All Phase-A runs have been launched and completed (per tmux session exits).")

if __name__ == "__main__":
    main()
    
# ENV variables to set before run!
# export DATA_ROOT=/home/hm25936/mae_datasets/midLighting_rmag_5m_to_100m_background_only
# export PROBE_DATA=/home/hm25936/mae_datasets/probe-space-split
# export WANDB_PROJECT=mae-sweeps