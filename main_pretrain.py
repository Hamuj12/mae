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
import numpy as np
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

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch


def _run_auto_linprobe(epoch, ckpt_path, args):
    """Launch a tiny linear-probe job and return the top-1 accuracy."""
    if args.no_auto_probe:
        return None
    if not args.probe_data_path:
        return None
    if not os.path.exists(args.probe_data_path):
        print(f"[auto-probe] Skipping – dataset not found: {args.probe_data_path}")
        return None

    probe_dir = Path(args.output_dir) / "auto_probe" / f"epoch{epoch:03d}"
    probe_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "main_linprobe.py",
        "--eval",
        "--finetune", str(ckpt_path),
        "--data_path", str(args.probe_data_path),
        "--epochs", str(args.probe_epochs),
        "--batch_size", str(args.probe_batch_size),
        "--output_dir", str(probe_dir),
        "--log_dir", str(probe_dir / "logs"),
    ]
    if args.wandb_project:
        cmd.extend(["--wandb_project", args.wandb_project])
    if args.run_name:
        cmd.extend(["--run_name", f"{args.run_name}-probe@{epoch}"])
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", env.get("WANDB_MODE", "offline"))
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"[auto-probe] Linear probe failed at epoch {epoch}: {exc}")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)
        return None

    pattern = re.compile(r"Acc@1\\s+([0-9]+\.?[0-9]*)")
    match = pattern.search(completed.stdout + "\n" + completed.stderr)
    if not match:
        print(f"[auto-probe] Unable to parse accuracy from output at epoch {epoch}.")
        return None
    return float(match.group(1))


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
    init_wandb(
        args.wandb_project,
        args.run_name,
        wandb_config,
        resume=args.wandb_resume,
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

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
        ckpt_path = None
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs or (epoch + 1) in {50, 100}):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            candidate = Path(args.output_dir) / f"checkpoint-{epoch}.pth"
            if candidate.exists():
                ckpt_path = candidate

        probe_acc = None
        if misc.is_main_process() and (epoch + 1) in {50, 100} and ckpt_path is not None:
            probe_acc = _run_auto_linprobe(epoch + 1, ckpt_path, args)

        if misc.is_main_process():
            metrics = {
                "train/loss": train_stats.get('loss'),
                "train/lr": train_stats.get('lr'),
            }
            if (epoch + 1) in {50, 100}:
                metrics[f"val_rec_loss@{epoch + 1}"] = train_stats.get('loss')
            if probe_acc is not None:
                metrics[f"lin_probe_acc@{epoch + 1}"] = probe_acc
            log_metrics(step=epoch + 1, **metrics)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        if (epoch + 1) in {50, 100}:
            log_stats[f'val_rec_loss@{epoch + 1}'] = train_stats.get('loss')
        if probe_acc is not None:
            log_stats[f'lin_probe_acc@{epoch + 1}'] = probe_acc

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
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
