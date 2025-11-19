#!/usr/bin/env python3
"""
Phase1_tmux.py
Launch a Phase-1 sweep for Dual YOLO + MAE in tmux, two runs at a time (each uses 2 GPUs).

Grid:
- epochs        : [50, 100]
- lr            : [5e-4, 1e-3, 2e-3]
- weight_decay  : [0.01, 0.05]

This sweeps Phase-1 hyperparameters with:
- MAE encoder frozen (Phase 1 only).
- YOLO backbone + fusion + head trainable.
- Dataset / splits / batch_size taken from configs/phase1_template.yaml.

You’ll get tmux sessions like:
    p1_e050_lr5e-4_wd010

Outputs go under:
    outputs/sweeps/phase1/<run_name>/

W&B:
- Project name taken from config["logging"]["wandb_project"] if set,
  or from env var WANDB_PROJECT (default: "dual-yolo-phase1-sweep").
"""

import itertools
import os
import shlex
import subprocess
import time
from pathlib import Path

# ------------------- USER CONFIG -------------------

# Preview only (no tmux is started). Set to False to actually launch.
DRY_RUN = True  # <-- flip to False when you're happy

# Two concurrent jobs total, each job uses 2 GPUs -> fits a 4-GPU node.
MAX_CONCURRENT_SLOTS = 2
GPU_SLOTS = [
    (0, 1),  # slot 0 uses GPUs 0 & 1
    (2, 3),  # slot 1 uses GPUs 2 & 3
]

# Torchrun master port base (unique per job to avoid collisions)
MASTER_PORT_BASE = 29600

# Phase-1 config path (the YAML you used for train_phase1.py)
PHASE1_CONFIG = "configs/phase1_template.yaml"

# Sweep grid (Phase 1)
EPOCHS_LIST = [50, 100]
LRS = [5.0e-4, 1.0e-3, 2.0e-3]
WEIGHT_DECAYS = [0.01, 0.05]

# Where to save outputs/logs
OUT_ROOT = Path("outputs/sweeps/phase1")

# W&B project (can also be set inside config.yaml)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "dual-yolo-phase1-sweep")

# ---------------------------------------------------


def tmux_has_session(name: str) -> bool:
    r = subprocess.run(["tmux", "has-session", "-t", name],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    return r.returncode == 0


def tmux_kill_session(name: str) -> None:
    subprocess.run(["tmux", "kill-session", "-t", name],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def tmux_new_session(name: str, command: str,
                     cwd: Path | None = None,
                     env: dict | None = None) -> None:
    # Build env prefix string for tmux
    env_prefix = ""
    if env:
        parts = [f"{k}={shlex.quote(str(v))}" for k, v in env.items()]
        env_prefix = " ".join(parts) + " "
    full_cmd = env_prefix + command
    args = ["tmux", "new-session", "-d", "-s", name, full_cmd]
    subprocess.check_call(args, cwd=str(cwd) if cwd else None)


def format_run_name(epochs: int, lr: float, wd: float) -> str:
    # epochs: 50 -> e050, 100 -> e100
    e_s = f"e{epochs:03d}"
    # lr: 0.001 -> lr0p001, 0.002 -> lr0p002, etc.
    lr_s = f"lr{str(lr).replace('.', 'p')}"
    # wd: 0.01 -> wd010, 0.05 -> wd050
    wd_s = f"wd{int(round(wd * 1000)):03d}"
    return f"p1_{e_s}_{lr_s}_{wd_s}"


def build_command(run_name: str,
                  out_dir: Path,
                  epochs: int,
                  lr: float,
                  wd: float,
                  master_port: int) -> str:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "train.out"

    cmd_parts = [
        "torchrun",
        "--nproc_per_node=2",
        f"--master_port={master_port}",
        "dual_yolo_mae/train_phase1.py",
        "--config", PHASE1_CONFIG,
        "--epochs", str(epochs),
        "--lr", str(lr),
        # batch size is taken from the YAML unless you want to override:
        # "--batch", "8",
        "--freeze_mae", "true",
        "--run_name", run_name,
        "--output", str(out_dir),
    ]

    # Stream logs to file (and keep tmux pane output)
    cmd = " ".join(cmd_parts) + f" |& tee -a {shlex.quote(str(logfile))}"
    return cmd


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(EPOCHS_LIST, LRS, WEIGHT_DECAYS))
    print(f"[info] Total runs in Phase 1 grid: {len(grid)}")
    print(f"[info] DRY_RUN={DRY_RUN}  |  concurrent slots={MAX_CONCURRENT_SLOTS}  |  GPU_SLOTS={GPU_SLOTS}\n")

    # Build per-slot queues (round-robin)
    slot_queues = [[] for _ in range(MAX_CONCURRENT_SLOTS)]
    for i, (epochs, lr, wd) in enumerate(grid):
        slot_idx = i % MAX_CONCURRENT_SLOTS
        run_name = format_run_name(epochs, lr, wd)
        out_dir = OUT_ROOT / run_name
        master_port = MASTER_PORT_BASE + i  # unique port per job

        slot_queues[slot_idx].append({
            "index": i,
            "epochs": epochs,
            "lr": lr,
            "wd": wd,
            "run_name": run_name,
            "out_dir": out_dir,
            "master_port": master_port,
        })

    active: list[str | None] = [None] * MAX_CONCURRENT_SLOTS

    def launch(slot_idx: int, job: dict) -> str:
        g0, g1 = GPU_SLOTS[slot_idx]
        env = {
            "CUDA_VISIBLE_DEVICES": f"{g0},{g1}",
            "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
            "WANDB_PROJECT": WANDB_PROJECT,
            # Optional grouping at the project level:
            "WANDB_RUN_GROUP": "phase1_sweep",
        }
        run_name = job["run_name"]
        out_dir = job["out_dir"]
        master_port = job["master_port"]
        cmd = build_command(
            run_name,
            out_dir,
            job["epochs"],
            job["lr"],
            job["wd"],
            master_port,
        )
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
                active[s] = f"(dry)_{job['run_name']}"
                launch(s, job)

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

    print("\n[done] All Phase-1 runs have been launched and completed (per tmux session exits).")


if __name__ == "__main__":
    main()