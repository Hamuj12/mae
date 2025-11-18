"""Lightning callbacks for Phase 1 dual-backbone training."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl

from dual_yolo_mae.metrics import compute_gate_histograms
from dual_yolo_mae.utils_wandb import log_metrics


class GateHistogramCallback(pl.Callback):
    """Sample gate activations and log lightweight histograms."""

    def __init__(self, interval: int = 200, max_samples: int = 5000) -> None:
        super().__init__()
        self.interval = interval
        self.max_samples = max_samples
        self._gate_buffers: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._hooks = []

    def _make_hook(self, scale_idx: int):
        def hook(_module, _inp, output):
            data = output.detach().flatten().cpu().numpy()
            if data.size > self.max_samples:
                data = np.random.choice(data, size=self.max_samples, replace=False)
            self._gate_buffers[scale_idx].append(data)

        return hook

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        fusion_layers = getattr(getattr(pl_module, "model", pl_module), "fusion_layers", [])
        for idx, fusion in enumerate(fusion_layers):
            if hasattr(fusion, "gate"):
                handle = fusion.gate.register_forward_hook(self._make_hook(idx))
                self._hooks.append(handle)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx) -> None:
        if trainer.global_step == 0 or trainer.global_step % self.interval != 0:
            return
        if not self._gate_buffers:
            return
        metrics = {}
        for idx, values in list(self._gate_buffers.items()):
            hist = compute_gate_histograms(values)
            if hist.size:
                metrics[f"gate_hist/scale{idx}"] = hist
        if metrics:
            log_metrics(metrics, step=int(trainer.global_step))
        self._gate_buffers.clear()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._gate_buffers.clear()


class BestCheckpointCallback(pl.Callback):
    """Track the best validation metric and log it to W&B."""

    def __init__(self, monitor: str = "val/mAP50", mode: str = "max") -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_value = None
        self.best_epoch = None

    def _is_better(self, current, best) -> bool:
        if best is None:
            return True
        if self.mode == "max":
            return current > best
        return current < best

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor]
        try:
            current_value = float(current.detach().cpu().item()) if hasattr(current, "detach") else float(current)
        except Exception:
            return
        if self._is_better(current_value, self.best_value):
            self.best_value = current_value
            self.best_epoch = trainer.current_epoch
            log_metrics({f"best/{self.monitor}": current_value, "best/epoch": trainer.current_epoch})

