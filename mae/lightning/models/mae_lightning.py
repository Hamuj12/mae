from __future__ import annotations

import math
from typing import Any, Tuple

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from models_mae import mae_vit_base_patch16


class MAELightning(pl.LightningModule):
    """PyTorch Lightning module wrapping the official MAE model."""

    def __init__(
        self,
        mask_ratio: float = 0.75,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        betas: Tuple[float, float] = (0.9, 0.95),
        warmup_epochs: int = 10,
        max_epochs: int = 200,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = mae_vit_base_patch16()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x, mask_ratio=self.hparams.mask_ratio)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, _, _ = self(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss, _, _ = self(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Any:
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(epoch: int) -> float:
            warmup = self.hparams.warmup_epochs
            max_epochs = self.hparams.max_epochs
            if epoch < warmup:
                return float(epoch + 1) / float(warmup)
            progress = (epoch - warmup) / max(1, max_epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
