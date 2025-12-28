"""Phase 1 training script: freeze MAE, train YOLO neck/head/fusion."""
# Debug recipe (rank0-only logging):
# torchrun --nproc_per_node=2 dual_yolo_mae/train_phase1.py --config <cfg.yaml> \
#   --no-freeze-mae --debug-train-dynamics --debug-every-n-steps 1 --limit_train_batches 0.01
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from dual_yolo_mae import utils
from dual_yolo_mae.callbacks import BestCheckpointCallback, GateHistogramCallback
from dual_yolo_mae.metrics import compute_box_mse, outputs_to_boxes
from dual_yolo_mae.model import DualBackboneYOLO
from dual_yolo_mae.utils_wandb import finish_run, init_wandb, log_metrics


class DualYoloPhase1Module(pl.LightningModule):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model = DualBackboneYOLO(config)
        self.training_cfg = config.get("training", {})
        self.lr = float(self.training_cfg.get("lr", 1e-3))
        self.weight_decay = float(self.training_cfg.get("weight_decay", 0.0))
        self.map_iou_threshold = float(config.get("metrics", {}).get("map_iou", 0.5))
        self.conf_threshold = float(config.get("metrics", {}).get("conf_threshold", 0.25))
        self._val_batches: list[tuple[list[Dict], list[Dict]]] = []
        self._val_box_mses: List[float] = []
        self._val_image_index = 0
        debug_cfg = config.get("debug", {})
        self.debug_train_dynamics = self._parse_debug_flag(
            debug_cfg.get("train_dynamics", False),
            env_var="DEBUG_TRAIN_DYNAMICS",
        )
        self.debug_every_n_steps = self._parse_debug_int(
            debug_cfg.get("every_n_steps", 0),
            env_var="DEBUG_EVERY_N_STEPS",
        )
        self._debug_params: Dict[str, tuple[str, torch.nn.Parameter]] = {}
        self._debug_param_snapshots: Dict[str, torch.Tensor] = {}
        self._debug_logged_startup = False
        self._last_cuda_mem_snapshot: Dict[str, float] = {}

    def forward(self, x: torch.Tensor):
        return self.model(x)

    @staticmethod
    def _parse_debug_flag(config_value: bool, env_var: str) -> bool:
        env_value = os.getenv(env_var)
        if env_value is None:
            return bool(config_value)
        return env_value.strip().lower() in {"1", "true", "yes", "y"}

    @staticmethod
    def _parse_debug_int(config_value: int, env_var: str) -> int:
        env_value = os.getenv(env_var)
        if env_value is None:
            return int(config_value)
        try:
            return int(env_value)
        except ValueError:
            return int(config_value)

    def _should_debug_step(self) -> bool:
        if not self.debug_train_dynamics:
            return False
        every_n = int(self.debug_every_n_steps)
        if every_n > 0:
            return self.global_step % every_n == 0
        return self.global_step == 0

    def _log_debug(self, message: str, *args) -> None:
        if not self.trainer or not self.trainer.is_global_zero:
            return
        utils.LOGGER.info(message, *args)

    def _summarize_module_params(self, module: torch.nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def _count_optimizer_params(self, optimizer: torch.optim.Optimizer, module: torch.nn.Module) -> int:
        module_param_ids = {id(p) for p in module.parameters()}
        total = 0
        for group in optimizer.param_groups:
            for param in group["params"]:
                if id(param) in module_param_ids:
                    total += param.numel()
        return total

    def _setup_debug_params(self) -> None:
        if self._debug_params:
            return

        def pick_first_param(module: torch.nn.Module) -> Optional[tuple[str, torch.nn.Parameter]]:
            for name, param in module.named_parameters():
                return name, param
            return None

        def get_module(attr: str) -> Optional[torch.nn.Module]:
            return getattr(self.model, attr, None)

        candidates = {
            "mae_encoder": getattr(get_module("mae_backbone"), "encoder", None),
            "mae_proj": getattr(get_module("mae_backbone"), "projections", None),
            "fusion": get_module("fusion_layers"),
            "head": get_module("detection_head"),
        }
        for label, module in candidates.items():
            if module is None:
                continue
            picked = pick_first_param(module)
            if picked is None:
                continue
            name, param = picked
            self._debug_params[label] = (name, param)

    def _log_cuda_memory(self, tag: str) -> None:
        if not self.debug_train_dynamics:
            return
        if not torch.cuda.is_available():
            return
        if not self.trainer or not self.trainer.is_global_zero:
            return
        snapshots: Dict[str, float] = {}
        for device_idx in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024**3)
            snapshots[f"cuda:{device_idx}.allocated_gb"] = allocated
            snapshots[f"cuda:{device_idx}.reserved_gb"] = reserved
            snapshots[f"cuda:{device_idx}.max_allocated_gb"] = max_allocated
        self._last_cuda_mem_snapshot = snapshots
        details = " | ".join(f"{k}={v:.2f}" for k, v in snapshots.items())
        self._log_debug("[debug] CUDA memory %s: %s", tag, details)

    def on_fit_start(self) -> None:
        if not self.debug_train_dynamics or self._debug_logged_startup:
            return
        self._setup_debug_params()
        mae_backbone = getattr(self.model, "mae_backbone", None)
        encoder = getattr(mae_backbone, "encoder", None) if mae_backbone is not None else None
        model_lines = [
            "[debug] ===== Train Dynamics Summary =====",
            f"[debug] MAE freeze={getattr(mae_backbone, 'freeze', None)}",
            f"[debug] Model training={self.model.training}",
            f"[debug] mae_backbone.training={getattr(mae_backbone, 'training', None)}",
            f"[debug] mae_encoder.training={getattr(encoder, 'training', None)}",
            f"[debug] yolo_backbone.training={getattr(getattr(self.model, 'yolo_backbone', None), 'training', None)}",
            f"[debug] fusion.training={getattr(getattr(self.model, 'fusion_layers', None), 'training', None)}",
            f"[debug] head.training={getattr(getattr(self.model, 'detection_head', None), 'training', None)}",
            "[debug] ================================",
        ]
        for line in model_lines:
            self._log_debug("%s", line)
        total_params, total_trainable = self._summarize_module_params(self.model)
        self._log_debug(
            "[debug] Params total=%d trainable=%d",
            total_params,
            total_trainable,
        )
        summaries = {
            "mae_encoder": encoder,
            "mae_proj": getattr(mae_backbone, "projections", None),
            "yolo": getattr(self.model, "yolo_backbone", None),
            "fusion": getattr(self.model, "fusion_layers", None),
            "head": getattr(self.model, "detection_head", None),
        }
        for name, module in summaries.items():
            if module is None:
                continue
            total, trainable = self._summarize_module_params(module)
            self._log_debug(
                "[debug] Module %s params total=%d trainable=%d",
                name,
                total,
                trainable,
            )

        optimizer = None
        if self.trainer and self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
        if optimizer is not None:
            total_opt_params = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
            self._log_debug("[debug] Optimizer params total=%d", total_opt_params)
            for idx, group in enumerate(optimizer.param_groups):
                group_total = sum(p.numel() for p in group["params"])
                group_trainable = sum(p.numel() for p in group["params"] if p.requires_grad)
                self._log_debug(
                    "[debug] Optimizer group %d params total=%d trainable=%d lr=%s",
                    idx,
                    group_total,
                    group_trainable,
                    group.get("lr"),
                )
            if mae_backbone is not None:
                mae_opt_params = self._count_optimizer_params(optimizer, mae_backbone)
                self._log_debug("[debug] Optimizer MAE params total=%d", mae_opt_params)
                if not mae_backbone.freeze and mae_opt_params == 0:
                    self._log_debug(
                        "[debug][WARN] MAE is unfrozen but optimizer has 0 MAE params.",
                    )
            for name, module in summaries.items():
                if module is None:
                    continue
                module_opt_params = self._count_optimizer_params(optimizer, module)
                self._log_debug("[debug] Optimizer module %s params total=%d", name, module_opt_params)
        if mae_backbone is not None:
            _, mae_trainable = self._summarize_module_params(mae_backbone)
            if not mae_backbone.freeze and mae_trainable == 0:
                self._log_debug(
                    "[debug][WARN] MAE is unfrozen but has 0 trainable params.",
                )

        for label, (name, param) in self._debug_params.items():
            self._log_debug(
                "[debug] Track param %s: %s requires_grad=%s shape=%s",
                label,
                name,
                param.requires_grad,
                tuple(param.shape),
            )

        self._log_cuda_memory("fit_start")
        self._debug_logged_startup = True

    def on_after_backward(self) -> None:
        if not self._should_debug_step():
            return
        self._setup_debug_params()
        for label, (name, param) in self._debug_params.items():
            grad = param.grad
            if grad is None:
                grad_norm = None
            else:
                grad_norm = float(grad.detach().float().norm().item())
            self._log_debug(
                "[debug] Grad norm %s (%s): %s",
                label,
                name,
                "None" if grad_norm is None else f"{grad_norm:.6f}",
            )

    def on_before_optimizer_step(self, optimizer) -> None:
        if not self._should_debug_step():
            return
        self._setup_debug_params()
        self._debug_param_snapshots = {
            label: param.detach().float().clone()
            for label, (_, param) in self._debug_params.items()
        }

    def on_after_optimizer_step(self, optimizer) -> None:
        if not self._should_debug_step() or not self._debug_param_snapshots:
            return
        mae_backbone = getattr(self.model, "mae_backbone", None)
        for label, (name, param) in self._debug_params.items():
            before = self._debug_param_snapshots.get(label)
            if before is None:
                continue
            delta = (param.detach().float() - before).norm().item()
            self._log_debug(
                "[debug] Weight delta %s (%s): %.6f",
                label,
                name,
                float(delta),
            )
            if mae_backbone is not None and label.startswith("mae"):
                if mae_backbone.freeze and delta > 0:
                    self._log_debug(
                        "[debug][WARN] MAE is frozen but %s (%s) changed (delta=%.6f).",
                        label,
                        name,
                        float(delta),
                    )
                if not mae_backbone.freeze and param.requires_grad and delta == 0:
                    self._log_debug(
                        "[debug][WARN] MAE is unfrozen but %s (%s) had zero delta.",
                        label,
                        name,
                    )
        self._debug_param_snapshots = {}

    def training_step(self, batch, batch_idx):
        if self.global_step == 0:
            self._log_cuda_memory("train_step_0")
        images, targets = batch
        images = images.to(self.device)
        targets = utils.move_targets_to_device(targets, self.device)
        preds = self.model(images)
        loss_dict = self.model.compute_loss(preds, targets)

        # Keep Lightning metrics for progress bar / epoch aggregation
        self.log(
            "train/loss",
            loss_dict["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )
        self.log("train/obj", loss_dict["loss_obj"], on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train/cls", loss_dict["loss_cls"], on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train/box", loss_dict["loss_box"], on_step=False, on_epoch=True, batch_size=images.size(0))

        # No direct W&B logging here; we'll log once per epoch instead
        return loss_dict["loss"]
    
    def on_train_epoch_end(self) -> None:
        """Log epoch-aggregated training metrics to W&B."""
        if self.trainer is None:
            return
        cbm = self.trainer.callback_metrics
        payload = {"epoch": int(self.current_epoch)}
        for key in ["train/loss", "train/obj", "train/cls", "train/box"]:
            if key in cbm:
                try:
                    payload[key] = float(cbm[key])
                except Exception:
                    pass
        if len(payload) > 1:
            log_metrics(payload, step=int(self.current_epoch))

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

        image_sizes = [tuple(map(int, t["orig_size"].detach().cpu().flip(0).tolist())) for t in targets]
        detections = self.model.decode_predictions(preds, image_sizes, conf_threshold=self.conf_threshold)
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
            cpu_targets.append({"boxes": torch.stack([x1, y1, x2, y2], dim=-1), "labels": labels})

        pred_boxes = outputs_to_boxes(preds, conf_threshold=self.conf_threshold, device=self.device)
        for p, t in zip(pred_boxes, targets):
            mse_val = compute_box_mse(p, t["boxes"].to(self.device))
            if not (mse_val != mse_val):  # filter NaN
                self._val_box_mses.append(mse_val)

        self._val_batches.append((detections, cpu_targets))
        return loss_dict["loss"]

    def on_validation_epoch_start(self) -> None:
        self._val_batches = []
        self._val_box_mses = []
        self._val_image_index = 0

    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
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

        per_class_dets: Dict[int, list] = {}
        gt_boxes: Dict[int, Dict[int, torch.Tensor]] = {}
        gt_counts: Dict[int, int] = {}

        for detections, targets in self._val_batches:
            for det_list, tgt in zip(detections, targets):
                image_id = self._val_image_index
                self._val_image_index += 1
                boxes = tgt["boxes"]
                labels = tgt["labels"]
                for cls_idx, box in zip(labels.tolist(), boxes):
                    gt_counts[cls_idx] = gt_counts.get(cls_idx, 0) + 1
                    gt_boxes.setdefault(cls_idx, {})
                    if image_id in gt_boxes[cls_idx]:
                        gt_boxes[cls_idx][image_id] = torch.vstack([gt_boxes[cls_idx][image_id], box])
                    else:
                        gt_boxes[cls_idx][image_id] = box.unsqueeze(0)
                for det in det_list:
                    per_class_dets.setdefault(det["label"], []).append((det["score"], torch.as_tensor(det["box"]), image_id))

        aps: list[float] = []
        for cls_idx, dets in per_class_dets.items():
            total_gt = gt_counts.get(cls_idx, 0)
            if total_gt == 0:
                continue
            dets.sort(key=lambda x: x[0], reverse=True)
            assigned: Dict[int, List[bool]] = {}
            tp: List[float] = []
            fp: List[float] = []
            for score, box_list, image_id in dets:
                box = torch.as_tensor(box_list, dtype=torch.float32).unsqueeze(0)
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
        box_mse = float(sum(self._val_box_mses) / len(self._val_box_mses)) if self._val_box_mses else float("nan")
        self.log("val/box_mse", box_mse, prog_bar=False)
        log_payload = {}
        if map50 is not None:
            log_payload["val/mAP50"] = map50
        if box_mse == box_mse:  # not NaN
            log_payload["val/box_mse"] = box_mse
        if self.trainer is not None:
            cbm = self.trainer.callback_metrics
            for key in ["val/loss", "val/box", "val/obj", "val/cls"]:
                if key in cbm:
                    try:
                        log_payload[key] = float(cbm[key])
                    except Exception:
                        pass
            log_payload["epoch"] = int(self.current_epoch)
        if log_payload:
            log_metrics(log_payload, step=int(self.current_epoch))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        epochs = int(self.training_cfg.get("epochs", 20))
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
    parser = argparse.ArgumentParser(description="Phase 1 training for Dual YOLO + MAE")
    parser.add_argument("--config", type=str, default="configs/phase1_template.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help="Early stopping patience in epochs (disabled if None)",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum improvement in val/mAP50 to count as progress",
    )
    parser.add_argument(
        "--fusion_temp",
        type=float,
        default=None,
        help="Override fusion gate temperature (scalar)",
    )
    parser.add_argument("--weight_decay", type=float, default=None, help="Override weight decay")  # <---
    parser.add_argument(
        "--precision_override",
        type=str,
        default=None,
        help="Override trainer precision (e.g. bf16-mixed, 16-mixed, 32-true).",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=None,
        help="Override gradient accumulation steps.",
    )
    parser.add_argument(
        "--gradient_checkpoint_mae",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable gradient checkpointing for MAE backbone.",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=float,
        default=None,
        help="Limit train batches (int for count, float for fraction).",
    )
    parser.add_argument(
        "--freeze_mae",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Freeze MAE encoder (default: respect config)",
    )
    parser.add_argument(
        "--debug_train_dynamics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log MAE/YOLO parameter, gradient, and update diagnostics.",
    )
    parser.add_argument(
        "--debug_every_n_steps",
        type=int,
        default=0,
        help="Debug logging frequency in steps (0 = only first step).",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Optional W&B run name")
    parser.add_argument("--output", type=str, default="outputs/phase1", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    return parser.parse_args()


def apply_overrides(config: Dict, args: argparse.Namespace) -> Dict:
    cfg = dict(config)
    cfg.setdefault("training", {})
    cfg.setdefault("model", {})
    cfg["model"].setdefault("mae", {})
    cfg["model"].setdefault("yolo", {})
    cfg["model"].setdefault("fusion", {})
    cfg.setdefault("debug", {})

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch is not None:
        cfg["training"]["batch_size"] = args.batch
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.fusion_temp is not None:
        cfg["model"]["fusion"]["temperature"] = float(args.fusion_temp)
    if args.weight_decay is not None:
        cfg["training"]["weight_decay"] = args.weight_decay   # <---
    if args.precision_override is not None:
        cfg["training"]["precision"] = args.precision_override
    if args.accumulate_grad_batches is not None:
        cfg["training"]["accumulate_grad_batches"] = int(args.accumulate_grad_batches)
    if args.limit_train_batches is not None:
        cfg["training"]["limit_train_batches"] = args.limit_train_batches

    if args.freeze_mae is not None:
        cfg["model"]["mae"]["freeze"] = bool(args.freeze_mae)
    if args.gradient_checkpoint_mae is not None:
        cfg["model"]["mae"]["gradient_checkpointing"] = bool(args.gradient_checkpoint_mae)
    cfg["model"]["yolo"]["freeze"] = False
    if args.debug_train_dynamics is not None:
        cfg["debug"]["train_dynamics"] = bool(args.debug_train_dynamics)
    if args.debug_every_n_steps is not None:
        cfg["debug"]["every_n_steps"] = int(args.debug_every_n_steps)
    return cfg


def main() -> None:
    args = parse_args()
    utils.setup_logging()
    config = utils.load_config(args.config)
    config = apply_overrides(config, args)
    output_dir = utils.ensure_dir(args.output)
    utils.LOGGER.info("Loaded configuration from %s", args.config)
    for line in utils.format_config_for_logging(config):
        utils.LOGGER.info(line)

    pl.seed_everything(int(config.get("training", {}).get("seed", 42)))

    wandb_run = init_wandb(config, args.run_name, output_dir)
    if wandb_run:
        fusion_temp = (
            config.get("model", {})
            .get("fusion", {})
            .get("temperature", None)
        )
        payload = {"phase": "phase1", "resume": bool(args.resume)}
        if fusion_temp is not None:
            payload["fusion/temperature"] = fusion_temp
        log_metrics(payload, step=0)

    train_loader, val_loader = create_dataloaders(config)
    module = DualYoloPhase1Module(config)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="phase1-{epoch:02d}",
        save_last=True,
        save_top_k=1,
        monitor="val/mAP50",
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    best_tracker = BestCheckpointCallback(monitor="val/mAP50", mode="max")
    gate_hist = GateHistogramCallback(
        interval=max(1, int(config.get("logging", {}).get("gate_interval", 200)))
    )

    callbacks = [checkpoint_callback, lr_monitor, best_tracker, gate_hist]

    # Optional early stopping on val/mAP50
    if args.early_stop_patience is not None and args.early_stop_patience > 0:
        early_stop_cb = EarlyStopping(
            monitor="val/mAP50",
            mode="max",
            patience=int(args.early_stop_patience),
            min_delta=float(args.early_stop_min_delta),
            verbose=True,
        )
        callbacks.append(early_stop_cb)

    trainer = pl.Trainer(
        max_epochs=int(config.get("training", {}).get("epochs", 20)),
        accelerator=config.get("training", {}).get("accelerator", "auto"),
        devices=config.get("training", {}).get("devices", "auto"),
        precision=config.get("training", {}).get("precision", 32),
        default_root_dir=str(output_dir),
        gradient_clip_val=float(config.get("training", {}).get("gradient_clip_val", 0.0)),
        accumulate_grad_batches=config.get("training", {}).get("accumulate_grad_batches", 1),
        limit_train_batches=config.get("training", {}).get("limit_train_batches", 1.0),
        logger=True,
        callbacks=callbacks,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume)
    finish_run()


if __name__ == "__main__":
    main()
