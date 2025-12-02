#!/usr/bin/env python3
"""
Phase1_v3_tmux.py
Hyperparameter sweep v3 for the Dual YOLO+MAE model after major patches:

- MAE unfrozen & trainable
- YOLO-style grid offsets
- GIoU/CIoU loss replacing SmoothL1
- Updated decoding pipeline

Sweep focuses narrowly on the “good region” from v2 to validate whether
the new head/loss/MAE changes break the mAP50 ~0.64 ceiling.

Sweep size: 18 runs.
"""

import itertools
import os
import shlex
import subprocess
import time
from pathlib import Path

# ------------------- USER CONFIG -------------------

DRY_RUN = False   # Set True to preview commands without launching tmux

MAX_CONCURRENT_SLOTS = 2            # 2 concurrent jobs
GPU_SLOTS = [(0, 1), (2, 3)]        # Slot -> GPU pair
MASTER_PORT_BASE = 29900            # Base for torchrun ports

EPOCHS = 100

# ------------------- Phase 1 v3 grid -------------------
# Keep batch=24 (proved best). Explore LR/WD/T again but smaller.
BATCH_SIZES   = [24]
LRS           = [3e-4, 5e-4, 7e-4]
WEIGHT_DECAYS = [8e-4, 1e-3]
FUSION_TEMPS  = [0.7, 1.0, 1.3]

CONFIG_PATH   = "configs/phase1_template.yaml"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "dual-yolo-phase1")
RUN_GROUP     = "phase1_sweep_v3"

OUT_ROOT = Path("outputs/sweeps/phase1_v3")

# ------------------- TMUX HELPERS -------------------

def tmux_has_session(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0

def tmux_kill_session(name: str):
    subprocess.run(["tmux", "kill-session", "-t", name],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

def tmux_new_session(name: str, command: str, cwd=None, env=None):
    env_prefix = ""
    if env:
        env_prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k,v in env.items()) + " "
    full_cmd = env_prefix + command
    subprocess.check_call(
        ["tmux", "new-session", "-d", "-s", name, full_cmd],
        cwd=str(cwd) if cwd else None
    )

# ------------------- RUN NAMING & COMMAND -------------------

def format_run_name(batch, lr, wd, temp):
    lr_s = f"lr{str(lr).replace('.', 'p')}"
    wd_int = int(round(wd * 1e4))     # e.g., 0.001 -> 10
    wd_s = f"wd{wd_int:04d}"
    t_s  = f"T{str(temp).replace('.', 'p')}"
    return f"p1v3_e{EPOCHS}_b{batch}_{lr_s}_{wd_s}_{t_s}"

def build_command(run_name, out_dir, batch, lr, wd, temp, master_port):
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "train.out"

    cmd_parts = [
        "PYTHONPATH=. torchrun",
        f"--nproc_per_node=2",
        f"--master_port={master_port}",
        "dual_yolo_mae/train_phase1.py",
        "--config", shlex.quote(CONFIG_PATH),
        "--epochs", str(EPOCHS),
        "--batch", str(batch),
        "--lr", str(lr),
        "--weight_decay", str(wd),
        "--fusion_temp", str(temp),
        "--early_stop_patience", "10",
        "--early_stop_min_delta", "0.001",
        "--run_name", shlex.quote(run_name),
        "--output", shlex.quote(str(out_dir)),
    ]
    return " ".join(cmd_parts) + f" |& tee -a {shlex.quote(str(logfile))}"

# ------------------- MAIN LOGIC -------------------

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(BATCH_SIZES, LRS, WEIGHT_DECAYS, FUSION_TEMPS))
    print(f"[info] Phase-1 v3: {len(grid)} runs")
    print(f"[info] DRY_RUN={DRY_RUN}")

    slot_queues = [[] for _ in range(MAX_CONCURRENT_SLOTS)]

    for i, (batch, lr, wd, temp) in enumerate(grid):
        slot = i % MAX_CONCURRENT_SLOTS
        run_name    = format_run_name(batch, lr, wd, temp)
        out_dir     = OUT_ROOT / run_name
        master_port = MASTER_PORT_BASE + i

        slot_queues[slot].append({
            "batch": batch,
            "lr": lr,
            "wd": wd,
            "temp": temp,
            "run_name": run_name,
            "out_dir": out_dir,
            "master_port": master_port,
        })

    active = [None] * MAX_CONCURRENT_SLOTS

    def launch(slot_idx, job):
        g0, g1 = GPU_SLOTS[slot_idx]
        env = {
            "CUDA_VISIBLE_DEVICES": f"{g0},{g1}",
            "WANDB_PROJECT": WANDB_PROJECT,
            "WANDB_RUN_GROUP": RUN_GROUP,
            "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
        }
        cmd = build_command(
            job["run_name"],
            job["out_dir"],
            job["batch"],
            job["lr"],
            job["wd"],
            job["temp"],
            job["master_port"]
        )

        print(f"[slot {slot_idx}] Launching: {job['run_name']} on GPUs {g0},{g1}")
        print("CMD:", cmd)

        if not DRY_RUN:
            if tmux_has_session(job["run_name"]):
                tmux_kill_session(job["run_name"])
            job["out_dir"].mkdir(parents=True, exist_ok=True)
            (job["out_dir"] / "logs").mkdir(parents=True, exist_ok=True)
            tmux_new_session(job["run_name"], cmd, cwd=Path.cwd(), env=env)

        return job["run_name"]

    # Kick off one job per slot
    for s in range(MAX_CONCURRENT_SLOTS):
        if slot_queues[s]:
            active[s] = launch(s, slot_queues[s].pop(0))

    if DRY_RUN:
        print("[DRY_RUN] Finished preview.")
        return

    print("[info] Monitoring sessions...")
    while any(active) or any(slot_queues):
        for s in range(MAX_CONCURRENT_SLOTS):
            sess = active[s]
            if sess and not tmux_has_session(sess):
                print(f"[slot {s}] Session {sess} finished.")
                active[s] = None
                if slot_queues[s]:
                    active[s] = launch(s, slot_queues[s].pop(0))
        time.sleep(30)

    print("[done] All Phase 1 v3 runs complete.")

if __name__ == "__main__":
    main()