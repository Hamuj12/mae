# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
# Fix for numpy issue with recent versions
import numpy as np
if not hasattr(np, "float"):
    np.float = float
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from util.datasets import build_dataset
from utils.wandb_utils import init_wandb, log_metrics

import timm

# assert timm.__version__ == "0.3.2"  # version check
print("Using timm", timm.__version__)
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch

import torch.distributed as dist

def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        # device_ids kwarg is optional; basic barrier is enough
        dist.barrier()
        
def parse_probe_epochs(s: str):
    if not s:
        return set()
    return {int(x.strip()) for x in s.split(',') if x.strip().isdigit()}

def should_probe(epoch1_based: int, every_k: int, explicit: set[int]) -> bool:
    if epoch1_based in explicit:
        return True
    if every_k and epoch1_based % every_k == 0:
        return True
    return False


def _run_auto_linprobe(epoch, ckpt_path, args):
    """Launch a tiny linear-probe job and return top-1 accuracy, or None."""
    if args.no_auto_probe or not args.probe_data_path:
        return None
    if not os.path.exists(args.probe_data_path):
        print(f"[auto-probe] Skipping – dataset not found: {args.probe_data_path}")
        return None

    probe_dir = Path(args.output_dir) / "auto_probe" / f"epoch{epoch:03d}"
    probe_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "main_linprobe.py",
        # IMPORTANT: train the head (no --eval), and match the MAE backbone size
        "--model", "vit_base_patch16",
        "--finetune", str(ckpt_path),
        "--data_path", str(args.probe_data_path),
        "--epochs", str(args.probe_epochs),
        "--batch_size", str(args.probe_batch_size),
        "--output_dir", str(probe_dir),
        "--log_dir", str(probe_dir / "logs"),
    ]
    # if args.wandb_project:
    #     cmd += ["--wandb_project", args.wandb_project]
    # if args.run_name:
    #     cmd += ["--run_name", f"{args.run_name}-probe@{epoch}"]

    # --- child env: strip DDP/elastic, force single-process, CPU-only ---
    env = os.environ.copy()

    # Strip DDP/elastic noise so the child is single-process.
    for k in ["RANK","LOCAL_RANK","WORLD_SIZE","MASTER_ADDR","MASTER_PORT",
            "GROUP_RANK","LOCAL_WORLD_SIZE","TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS","TORCHELASTIC_RUN_ID"]:
        env.pop(k, None)

    # Force NO wandb in the child.
    env["WANDB_MODE"] = "disabled"       # <-- key line
    env["WANDB_SILENT"] = "true"
    env.setdefault("OMP_NUM_THREADS", "1")

    # Stick to rank-0’s GPU (or GPU 0) for the probe.
    if hasattr(args, "gpu") and isinstance(args.gpu, int) and args.gpu >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        env["CUDA_VISIBLE_DEVICES"] = "0"
        
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("WANDB_MODE", env.get("WANDB_MODE", "offline"))
    env.setdefault("WANDB_SILENT", "true")

    # Add ultra-safe dataloader settings to the probe
    cmd += ["--num_workers", "0", "--no_pin_mem"]
    # Also avoid accidental W&B resume collisions
    # cmd += ["--wandb_resume", "never", "--run_name", f"{args.run_name}-probe@{epoch}-{int(time.time())}"]

    # Drop simple markers for debugging
    start_mark = probe_dir / "_STARTED"
    done_mark  = probe_dir / "_DONE"
    fail_mark  = probe_dir / "_FAILED"
    stdout_file = probe_dir / "stdout.txt"
    stderr_file = probe_dir / "stderr.txt"
    start_mark.write_text("")

    try:
        completed = subprocess.run(
            cmd, check=True, capture_output=True, text=True, env=env, timeout=300  # 5 min cap
        )
        stdout_file.write_text(completed.stdout or "")
        stderr_file.write_text(completed.stderr or "")
        done_mark.write_text("")
    except subprocess.TimeoutExpired as exc:
        stdout_file.write_text((exc.stdout or ""))
        stderr_file.write_text((exc.stderr or ""))
        fail_mark.write_text("TIMEOUT")
        print(f"[auto-probe] Timeout at epoch {epoch} after {exc.timeout}s.")
        return None
    except subprocess.CalledProcessError as exc:
        stdout_file.write_text(exc.stdout or "")
        stderr_file.write_text(exc.stderr or "")
        fail_mark.write_text("CALLED_PROCESS_ERROR")
        print(f"[auto-probe] Linear probe failed at epoch {epoch}: {exc}")
        return None

    # --- Parse validation metrics from child stdout ---
    out_text = (completed.stdout or "") + "\n" + (completed.stderr or "")

    acc1 = acc5 = valloss = None

    # First try the common single-line summary: "* Acc@1 68.7 Acc@5 95.7 loss 6.78"
    m_all = re.search(
        r"\*?\s*Acc@1\s+([0-9.]+)\s+Acc@5\s+([0-9.]+)\s+loss\s+([0-9.]+)",
        out_text, flags=re.IGNORECASE
    )
    if m_all:
        acc1   = float(m_all.group(1))
        acc5   = float(m_all.group(2))
        valloss = float(m_all.group(3))
    else:
        # Fall back to individual matches
        m1    = re.search(r"Acc@1\s+([0-9.]+)", out_text, flags=re.IGNORECASE)
        m5    = re.search(r"Acc@5\s+([0-9.]+)", out_text, flags=re.IGNORECASE)
        mloss = re.search(r"\bloss\b\s*[:=]?\s*([0-9.]+)", out_text, flags=re.IGNORECASE)
        if m1:    acc1 = float(m1.group(1))
        if m5:    acc5 = float(m5.group(1))
        if mloss: valloss = float(mloss.group(1))

    if acc1 is None:
        print(f"[auto-probe] Warning: could not parse Acc@1 for epoch {epoch}")

    return {"acc1": acc1, "acc5": acc5, "val_loss": valloss}


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data for validation when no split dirs are present')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Logging & diagnostics
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Optional W&B project for experiment tracking.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Friendly name for the W&B run.')
    parser.add_argument('--wandb_resume', type=str, default='auto',
                        choices=['auto', 'allow', 'never'],
                        help='Resume behaviour passed to wandb.init.')
    parser.add_argument('--probe_data_path', type=str, default=None,
                        help='Optional dataset for auto linear-probe evaluation.')
    parser.add_argument('--probe_batch_size', type=int, default=128,
                        help='Batch size for the auto linear-probe runs.')
    parser.add_argument('--probe_epochs', type=int, default=2,
                        help='Number of epochs for the auto linear-probe runs.')
    parser.add_argument('--no_auto_probe', action='store_true',
                        help='Disable automatic linear probing at epochs 50/100.')
    parser.add_argument('--probe_every_k', type=int, default=50,
                        help='Run auto linear probe every K epochs (1-based). 0 disables.')
    parser.add_argument('--probe_at_epochs', type=str, default='',
                        help='Comma-separated list of 1-based epochs to probe (e.g., "50,100,150,200").')
    parser.add_argument('--probe_on_resume', action='store_true',
                        help='Run one-off probe immediately after resume on latest checkpoint.')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if args.wandb_project is None and os.environ.get("WANDB_PROJECT"):
        args.wandb_project = os.environ["WANDB_PROJECT"]
    if args.run_name is None and os.environ.get("WANDB_RUN_NAME"):
        args.run_name = os.environ["WANDB_RUN_NAME"]

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = build_dataset(is_train=True, args=args, transform=transform_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    patch_size = getattr(model_without_ddp.patch_embed, "patch_size", (args.input_size,))
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    decoder_depth = len(getattr(model_without_ddp, "decoder_blocks", []))
    decoder_width = getattr(getattr(model_without_ddp, "decoder_embed", None), "out_features", None)
    wandb_config = {
        "mask_ratio": args.mask_ratio,
        "patch_size": patch_size,
        "decoder_depth": decoder_depth,
        "decoder_width": decoder_width,
        "norm_pix_loss": args.norm_pix_loss,
        "batch_size": args.batch_size,
        "base_lr": args.blr,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "epochs": args.epochs,
        "accum_iter": args.accum_iter,
        "amp": True,
        "seed": args.seed,
        "world_size": args.world_size,
        "probe_data_path": args.probe_data_path,
    }
    
    # rank-0 creates the *only* parent run
    if misc.is_main_process():
        parent = init_wandb(
            args.wandb_project,
            args.run_name,
            wandb_config,
            resume=args.wandb_resume,
            # parent has no group; children will join its group name
        )
        # let children (probes) know how to group themselves
        os.environ["WANDB_PARENT_GROUP"] = args.run_name or "mae-pretrain"
    else:
        # hard-disable W&B on non-main ranks
        os.environ.setdefault("WANDB_MODE", "disabled")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    try:
        # timm >= 0.6/0.9
        from timm.optim.optim_factory import param_groups_weight_decay
        param_groups = param_groups_weight_decay(model_without_ddp, args.weight_decay)
    except Exception:
        # Fallback compatible with torch>=1.10 and 2.x
        def add_weight_decay_fallback(model, weight_decay=0.05, skip_list=()):
            decay, no_decay = [], []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim < 2 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return [
                {'params': no_decay, 'weight_decay': 0.0},
                {'params': decay, 'weight_decay': weight_decay},
            ]
        param_groups = add_weight_decay_fallback(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Parse explicit probe epochs once
    explicit_epochs = parse_probe_epochs(args.probe_at_epochs)

    # Optional: one-off probe on the latest checkpoint immediately after resume.
    if args.resume and args.probe_on_resume and misc.is_main_process() and args.probe_data_path:
        outs = Path(args.output_dir)
        latest = max(outs.glob("checkpoint-*.pth"), key=lambda p: p.stat().st_mtime, default=None)
        if latest is not None:
            print(f"[auto-probe] One-off probe after resume on: {latest.name}")
            ddp_barrier()
            acc = _run_auto_linprobe(args.start_epoch, latest, args)
            ddp_barrier()
            if acc is not None:
                log_metrics(step=args.start_epoch, **{"probe/acc1": acc})

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        epoch1 = epoch + 1
        probe_now = should_probe(epoch1, args.probe_every_k, explicit_epochs)

        probe_metrics = None
        if probe_now:
            if misc.is_main_process() and args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                )
            ddp_barrier()

            ckpt_path = None
            if misc.is_main_process():
                candidate = Path(args.output_dir) / f"checkpoint-{epoch}.pth"
                if candidate.exists():
                    ckpt_path = candidate
                else:
                    outs = Path(args.output_dir)
                    ckpt_path = max(outs.glob("checkpoint-*.pth"),
                                    key=lambda p: p.stat().st_mtime, default=None)
            ddp_barrier()

            if misc.is_main_process() and ckpt_path is not None:
                probe_metrics = _run_auto_linprobe(epoch1, ckpt_path, args)
            ddp_barrier()

        # ---- W&B logging (parent run only) ----
        if misc.is_main_process():
            metrics = {
                "train/loss": train_stats.get('loss'),
                "train/lr":   train_stats.get('lr'),
                "epoch":      epoch1,  # optional, handy for panels
            }
            if probe_now and probe_metrics:
                # LOG THE SAME KEYS EVERY TIME A PROBE HAPPENS
                if probe_metrics.get("val_loss") is not None:
                    metrics["probe/val_loss"] = probe_metrics["val_loss"]
                if probe_metrics.get("acc1") is not None:
                    metrics["probe/acc1"] = probe_metrics["acc1"]
                if probe_metrics.get("acc5") is not None:
                    metrics["probe/acc5"] = probe_metrics["acc5"]

            # one series per key; x-axis is the step (epoch1)
            log_metrics(step=epoch1, **metrics)

        # ---- text log (keep it aligned with W&B keys) ----
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if probe_now and probe_metrics:
            if probe_metrics.get("val_loss") is not None:
                log_stats["probe_val_loss"] = probe_metrics["val_loss"]
            if probe_metrics.get("acc1") is not None:
                log_stats["probe_acc1"] = probe_metrics["acc1"]
            if probe_metrics.get("acc5") is not None:
                log_stats["probe_acc5"] = probe_metrics["acc5"]

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
