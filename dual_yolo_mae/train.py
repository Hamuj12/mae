"""Training script for the dual-backbone YOLO + MAE detector."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from dual_yolo_mae.model import DualBackboneYOLO
from dual_yolo_mae import utils


class DualYoloMaeModule(pl.LightningModule):
    """Lightning wrapper around :class:`DualBackboneYOLO`."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.model = DualBackboneYOLO(config)
        self.training_cfg = config.get("training", {})
        self.lr = float(self.training_cfg.get("lr", 1e-4))
        self.weight_decay = float(self.training_cfg.get("weight_decay", 0.0))

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.device)
        targets = utils.move_targets_to_device(targets, self.device)
        preds = self.model(images)
        loss_dict = self.model.compute_loss(preds, targets)
        self.log("train/loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("train/box", loss_dict["loss_box"], on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train/obj", loss_dict["loss_obj"], on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train/cls", loss_dict["loss_cls"], on_step=False, on_epoch=True, batch_size=images.size(0))
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.device)
        targets = utils.move_targets_to_device(targets, self.device)
        preds = self.model(images)
        loss_dict = self.model.compute_loss(preds, targets)
        self.log("val/loss", loss_dict["loss"], prog_bar=True, batch_size=images.size(0))
        self.log("val/box", loss_dict["loss_box"], batch_size=images.size(0))
        self.log("val/obj", loss_dict["loss_obj"], batch_size=images.size(0))
        self.log("val/cls", loss_dict["loss_cls"], batch_size=images.size(0))
        return loss_dict["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        epochs = int(self.training_cfg.get("epochs", 50))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }


def create_dataloaders(config: Dict) -> tuple[DataLoader, Optional[DataLoader]]:
    dataset_cfg = config.get("dataset", {})
    train_dataset = utils.YOLODataset(
        root=dataset_cfg.get("path", "dataset"),
        split=dataset_cfg.get("train_split", "train"),
        img_size=int(config.get("model", {}).get("input_size", 640)),
        class_names=dataset_cfg.get("class_names"),
        augment=dataset_cfg.get("augment", False),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 4)),
        shuffle=True,
        num_workers=int(config.get("training", {}).get("num_workers", 4)),
        pin_memory=True,
        collate_fn=utils.yolo_collate_fn,
        drop_last=False,
    )

    val_split = dataset_cfg.get("val_split")
    if not val_split:
        return train_loader, None

    try:
        val_dataset = utils.YOLODataset(
            root=dataset_cfg.get("path", "dataset"),
            split=val_split,
            img_size=int(config.get("model", {}).get("input_size", 640)),
            class_names=dataset_cfg.get("class_names"),
            augment=False,
        )
    except FileNotFoundError:
        utils.LOGGER.warning("Validation split '%s' not found. Skipping validation.", val_split)
        return train_loader, None

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 4)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 4)),
        pin_memory=True,
        collate_fn=utils.yolo_collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Dual YOLO + MAE detector")
    parser.add_argument("--config", type=str, default="dual_yolo_mae/config.yaml", help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utils.setup_logging()
    config = utils.load_config(args.config)
    utils.LOGGER.info("Loaded configuration from %s", args.config)
    for line in utils.format_config_for_logging(config):
        utils.LOGGER.info(line)

    pl.seed_everything(int(config.get("training", {}).get("seed", 42)))

    train_loader, val_loader = create_dataloaders(config)
    if val_loader is None:
        utils.LOGGER.info(
            "No validation split configured. Only training metrics will be logged."
        )
    else:
        utils.LOGGER.info(
            "Validation split active with %d batches per epoch.", len(val_loader)
        )
    module = DualYoloMaeModule(config)

    checkpoint_dir = utils.ensure_dir(config.get("training", {}).get("checkpoint_dir", "checkpoints"))
    logger = pl.loggers.TensorBoardLogger(save_dir=str(checkpoint_dir), name="logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="dual-yolo-mae-{epoch:02d}",
        save_top_k=1,
        monitor="val/loss" if val_loader is not None else "train/loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=int(config.get("training", {}).get("epochs", 50)),
        accelerator=config.get("training", {}).get("accelerator", "auto"),
        devices=config.get("training", {}).get("devices", "auto"),
        precision=config.get("training", {}).get("precision", 32),
        default_root_dir=str(checkpoint_dir),
        gradient_clip_val=float(config.get("training", {}).get("gradient_clip_val", 0.0)),
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume)

    final_path = Path(checkpoint_dir) / "model.pt"
    module.cpu()
    torch.save(module.model.state_dict(), final_path)
    utils.LOGGER.info("Saved final model weights to %s", final_path)


if __name__ == "__main__":
    main()
