#!/usr/bin/env python3
"""
Phase1_v2_tmux.py
Hyperparameter sweep v2 for Phase-1 dual-backbone YOLO training (frozen MAE).

Refined grid around the best region found in v1:

- epochs          : 100 (fixed)
- batch_size      : [24, 32]          (global batch in your Lightning config)
- lr              : [3e-4, 5e-4, 7e-4]
- weight_decay    : [8e-4, 1e-3, 1.5e-3]
- fusion_temp     : [0.7, 1.0, 1.3]   (gate temperature; 1.0 == baseline)

Each run:
- Uses dual_yolo_mae/train_phase1.py with overrides: --epochs, --batch, --lr,
  --weight_decay, --fusion_temp
- Uses the same MAE checkpoint and dataset specified in configs/phase1_template.yaml
- Logs to W&B project: dual-yolo-phase1 (can be overridden via env)

Launch style:
- 2 concurrent tmux sessions total
- Each session uses 2 GPUs via CUDA_VISIBLE_DEVICES
- torchrun is used with --nproc_per_node=2

Set DRY_RUN=True first to inspect commands before actually launching.
"""

import itertools
import os
import shlex
import subprocess
import time
from pathlib import Path

# ------------------- USER CONFIG -------------------

# Preview only (no tmux is started). Set to False to actually launch.
DRY_RUN = False  # flip to False once you're happy with the commands

# Two concurrent jobs total, each job uses 2 GPUs.
MAX_CONCURRENT_SLOTS = 2
GPU_SLOTS = [
    (0, 1),  # slot 0 uses GPUs 0 & 1
    (2, 3),  # slot 1 uses GPUs 2 & 3
]

# Torchrun master port base (unique per job to avoid collisions)
MASTER_PORT_BASE = 29700

# Training hyperparams for Phase 1
EPOCHS = 100

# ---- Refined sweep grid (v2) ----
BATCH_SIZES   = [24, 32]
LRS           = [3e-4, 5e-4, 7e-4]
WEIGHT_DECAYS = [8e-4, 1e-3, 1.5e-3]
FUSION_TEMPS  = [0.7, 1.0, 1.3]

# Config & logging
CONFIG_PATH   = "configs/phase1_template.yaml"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "dual-yolo-phase1")
RUN_GROUP     = "phase1_sweep_v2"

# Where to save outputs/logs
OUT_ROOT = Path("outputs/sweeps/phase1_v2")

# ------------------- TMUX HELPERS -------------------

def tmux_has_session(name: str) -> bool:
    r = subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return r.returncode == 0


def tmux_kill_session(name: str) -> None:
    subprocess.run(
        ["tmux", "kill-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def tmux_new_session(
    name: str,
    command: str,
    cwd: Path | None = None,
    env: dict | None = None,
) -> None:
    env_prefix = ""
    if env:
        parts = [f'{k}={shlex.quote(str(v))}' for k, v in env.items()]
        env_prefix = " ".join(parts) + " "
    full_cmd = env_prefix + command
    args = ["tmux", "new-session", "-d", "-s", name, full_cmd]
    subprocess.check_call(args, cwd=str(cwd) if cwd else None)

# ------------------- RUN NAMING & COMMAND -------------------

def format_run_name(batch: int, lr: float, wd: float, temp: float) -> str:
    # e.g., p1v2_e100_b24_lr3e-4_wd0008_T0p7
    lr_s = f"lr{str(lr).replace('.', 'p')}"      # 0.0005 -> lr0p0005
    wd_int = int(round(wd * 1e4))               # 8e-4 -> 8, 1e-3 -> 10, 1.5e-3 -> 15
    wd_s = f"wd{wd_int:04d}"
    t_s = f"T{str(temp).replace('.', 'p')}"     # 0.7 -> T0p7
    return f"p1v2_e{EPOCHS}_b{batch}_{lr_s}_{wd_s}_{t_s}"


def build_command(
    run_name: str,
    out_dir: Path,
    batch: int,
    lr: float,
    weight_decay: float,
    fusion_temp: float,
    master_port: int,
) -> str:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "train.out"

    # train_phase1.py CLI:
    #   --config       : YAML config
    #   --epochs       : override epochs
    #   --batch        : override batch_size
    #   --lr           : override learning rate
    #   --weight_decay : override weight decay
    #   --fusion_temp  : override fusion gate temperature (scalar)
    #   --run_name     : W&B run name
    #   --output       : output dir
    cmd_parts = [
        "PYTHONPATH=. torchrun",
        f"--nproc_per_node=2",
        f"--master_port={master_port}",
        "dual_yolo_mae/train_phase1.py",
        "--config", shlex.quote(CONFIG_PATH),
        "--epochs", str(EPOCHS),
        "--batch", str(batch),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--fusion_temp", str(fusion_temp),
        "--early_stop_patience", "10",
        "--early_stop_min_delta", "0.001",
        "--run_name", shlex.quote(run_name),
        "--output", shlex.quote(str(out_dir)),
    ]

    cmd = " ".join(cmd_parts) + f" |& tee -a {shlex.quote(str(logfile))}"
    return cmd

# ------------------- MAIN LAUNCH LOGIC -------------------

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(BATCH_SIZES, LRS, WEIGHT_DECAYS, FUSION_TEMPS))
    print(f"[info] Total runs in Phase-1 v2 grid: {len(grid)}")
    print(f"[info] DRY_RUN={DRY_RUN}  |  concurrent slots={MAX_CONCURRENT_SLOTS}  |  GPU_SLOTS={GPU_SLOTS}\n")

    # Distribute runs round-robin into slot queues
    slot_queues = [[] for _ in range(MAX_CONCURRENT_SLOTS)]
    for i, (batch, lr, wd, temp) in enumerate(grid):
        slot_idx = i % MAX_CONCURRENT_SLOTS
        run_name    = format_run_name(batch, lr, wd, temp)
        out_dir     = OUT_ROOT / run_name
        master_port = MASTER_PORT_BASE + i

        slot_queues[slot_idx].append({
            "index": i,
            "batch": batch,
            "lr": lr,
            "wd": wd,
            "temp": temp,
            "run_name": run_name,
            "out_dir": out_dir,
            "master_port": master_port,
        })

    active = [None] * MAX_CONCURRENT_SLOTS

    def launch(slot_idx: int, job: dict):
        g0, g1 = GPU_SLOTS[slot_idx]
        env = {
            "CUDA_VISIBLE_DEVICES": f"{g0},{g1}",
            "WANDB_PROJECT": WANDB_PROJECT,
            "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
            "WANDB_RUN_GROUP": RUN_GROUP,
        }
        run_name    = job["run_name"]
        out_dir     = job["out_dir"]
        master_port = job["master_port"]
        cmd         = build_command(
            run_name,
            out_dir,
            job["batch"],
            job["lr"],
            job["wd"],
            job["temp"],
            master_port,
        )
        session     = run_name

        print(f"[slot {slot_idx}] -> {session}  (GPUs {g0},{g1})")
        print(f"  tmux new-session -d -s {session}  (cwd={Path.cwd()})")
        print(f"  ENV: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}  WANDB_PROJECT={WANDB_PROJECT}")
        print(f"  CMD: {cmd}\n")

        if not DRY_RUN:
            if tmux_has_session(session):
                print(f"  [warn] session {session} already exists; killing it.")
                tmux_kill_session(session)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "logs").mkdir(parents=True, exist_ok=True)
            tmux_new_session(session, cmd, cwd=Path.cwd(), env=env)

        return session

    # Start one job per slot initially
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

    # Monitor tmux sessions and launch queued jobs as slots free up
    print("[info] Monitoring tmux sessions; launching queued jobs as slots free up...")
    while any(active) or any(slot_queues):
        for s in range(MAX_CONCURRENT_SLOTS):
            sess = active[s]
            if sess is None:
                if slot_queues[s]:
                    job = slot_queues[s].pop(0)
                    active[s] = launch(s, job)
                continue
            if not tmux_has_session(sess):
                print(f"[slot {s}] session {sess} ended.")
                active[s] = None
                if slot_queues[s]:
                    job = slot_queues[s].pop(0)
                    active[s] = launch(s, job)
        time.sleep(30)

    print("\n[done] All Phase-1 v2 runs have been launched and completed (per tmux session exits).")

if __name__ == "__main__":
    main()