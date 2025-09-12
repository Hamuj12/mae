from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Ensure project root is on path when running as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mae.lightning.data.background_dataset import BackgroundOnlyDataset
from mae.lightning.models.mae_lightning import MAELightning
from mae.lightning.utils.common import log_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAE with Lightning")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(args.seed)
    log_config(cfg)

    train_ds = BackgroundOnlyDataset(cfg.data.train_dir, cfg.data.image_size)
    val_ds = BackgroundOnlyDataset(cfg.data.val_dir, cfg.data.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    model = MAELightning(
        mask_ratio=cfg.optim.mask_ratio,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.betas),
        warmup_epochs=cfg.optim.warmup_epochs,
        max_epochs=cfg.optim.max_epochs,
        batch_size=cfg.data.batch_size,
    )

    trainer = pl.Trainer(max_epochs=cfg.optim.max_epochs, **cfg.trainer)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
