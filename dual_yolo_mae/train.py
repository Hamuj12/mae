"""Training script for the dual-backbone YOLO + MAE detector."""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from dual_yolo_mae.model import DualBackboneYOLO
from dual_yolo_mae import utils
from utils.wandb_utils import init_wandb, log_metrics


class DualYoloMaeModule(pl.LightningModule):
    """Lightning wrapper around :class:`DualBackboneYOLO`."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.model = DualBackboneYOLO(config)
        self.training_cfg = config.get("training", {})
        self.lr = float(self.training_cfg.get("lr", 1e-4))
        self.weight_decay = float(self.training_cfg.get("weight_decay", 0.0))
        self._val_batches: list[tuple[list[Dict], list[Dict]]] = []
        self._val_image_index = 0
        self.map_iou_threshold = float(config.get("metrics", {}).get("map_iou", 0.5))

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
        gate_stats = self.model.get_gate_statistics()
        metrics = {
            "train/loss": loss_dict["loss"].detach(),
            "train/box": loss_dict["loss_box"],
            "train/obj": loss_dict["loss_obj"],
            "train/cls": loss_dict["loss_cls"],
        }
        if gate_stats:
            for idx, stats in enumerate(gate_stats):
                metrics[f"gate/scale{idx}_mean"] = stats.get("mean")
                metrics[f"gate/scale{idx}_std"] = stats.get("std")
        log_metrics(step=int(self.global_step), **metrics)
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
        metrics = {
            "val/loss": loss_dict["loss"],
            "val/box": loss_dict["loss_box"],
            "val/obj": loss_dict["loss_obj"],
            "val/cls": loss_dict["loss_cls"],
        }
        log_metrics(step=int(self.global_step), **metrics)
        image_sizes = [tuple(map(int, t["orig_size"].detach().cpu().flip(0).tolist())) for t in targets]
        detections = self.model.decode_predictions(preds, image_sizes)
        cpu_targets: list[Dict] = []
        for tgt in targets:
            width, height = tgt["orig_size"].detach().cpu().tolist()
            boxes = tgt["boxes"].detach().cpu()
            labels = tgt["labels"].detach().cpu()
            x_c = boxes[:, 0] * width
            y_c = boxes[:, 1] * height
            w = boxes[:, 2] * width
            h = boxes[:, 3] * height
            x1 = x_c - w / 2
            y1 = y_c - h / 2
            x2 = x_c + w / 2
            y2 = y_c + h / 2
            cpu_targets.append(
                {
                    "boxes": torch.stack([x1, y1, x2, y2], dim=-1),
                    "labels": labels,
                }
            )
        self._val_batches.append((detections, cpu_targets))
        return loss_dict["loss"]

    def on_validation_epoch_start(self) -> None:
        self._val_batches = []
        self._val_image_index = 0

    @staticmethod
    def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
        lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        return inter / (union + 1e-6)

    def _compute_map50(self) -> Optional[float]:
        if not self._val_batches:
            return None

        per_class_dets: Dict[int, list] = defaultdict(list)
        gt_boxes: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        gt_counts: Dict[int, int] = defaultdict(int)

        for detections, targets in self._val_batches:
            for det_list, tgt in zip(detections, targets):
                image_id = self._val_image_index
                self._val_image_index += 1
                boxes = tgt["boxes"]
                labels = tgt["labels"]
                for cls_idx, box in zip(labels.tolist(), boxes):
                    gt_counts[cls_idx] += 1
                    if image_id in gt_boxes[cls_idx]:
                        gt_boxes[cls_idx][image_id] = torch.vstack([gt_boxes[cls_idx][image_id], box])
                    else:
                        gt_boxes[cls_idx][image_id] = box.unsqueeze(0)
                for det in det_list:
                    per_class_dets[det["label"]].append((det["score"], torch.tensor(det["box"]), image_id))

        aps: list[float] = []
        for cls_idx, dets in per_class_dets.items():
            total_gt = gt_counts.get(cls_idx, 0)
            if total_gt == 0:
                continue
            dets.sort(key=lambda x: x[0], reverse=True)
            assigned: Dict[int, list[bool]] = {}
            tp = []
            fp = []
            for score, box_list, image_id in dets:
                box = torch.tensor(box_list, dtype=torch.float32).unsqueeze(0)
                class_gts = gt_boxes[cls_idx].get(image_id)
                if class_gts is None or class_gts.numel() == 0:
                    tp.append(0.0)
                    fp.append(1.0)
                    continue
                if image_id not in assigned:
                    assigned[image_id] = [False] * class_gts.shape[0]
                ious = self._box_iou(box, class_gts)
                best_iou, best_idx = torch.max(ious.squeeze(0), dim=0)
                if best_iou.item() >= self.map_iou_threshold and not assigned[image_id][best_idx]:
                    assigned[image_id][best_idx] = True
                    tp.append(1.0)
                    fp.append(0.0)
                else:
                    tp.append(0.0)
                    fp.append(1.0)
            if not tp:
                aps.append(0.0)
                continue
            tp_tensor = torch.tensor(tp)
            fp_tensor = torch.tensor(fp)
            tp_cum = torch.cumsum(tp_tensor, dim=0)
            fp_cum = torch.cumsum(fp_tensor, dim=0)
            recalls = tp_cum / max(total_gt, 1e-6)
            precisions = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-6)
            precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
            recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
            for i in range(precisions.numel() - 1, 0, -1):
                precisions[i - 1] = torch.maximum(precisions[i - 1], precisions[i])
            idx = torch.nonzero(recalls[1:] != recalls[:-1], as_tuple=False).squeeze(-1)
            ap = torch.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]).item()
            aps.append(ap)

        if not aps:
            return 0.0
        return float(sum(aps) / len(aps))

    def on_validation_epoch_end(self) -> None:
        map50 = self._compute_map50()
        if map50 is not None:
            self.log("val/mAP50", map50, prog_bar=True)
            log_metrics(step=int(self.global_step), **{"val/mAP50": map50})

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

    logging_cfg = config.get("logging", {})
    wandb_project = logging_cfg.get("wandb_project") or os.environ.get("WANDB_PROJECT")
    run_name = logging_cfg.get("run_name") or os.environ.get("WANDB_RUN_NAME")
    wandb_resume = logging_cfg.get("wandb_resume", "auto")
    init_wandb(wandb_project, run_name, config, resume=wandb_resume)
    log_metrics(step=0, resume_path=args.resume, dataset=config.get("dataset", {}).get("path"))

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
