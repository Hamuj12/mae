"""Dual-backbone YOLO detector that fuses MAE and YOLOv8 features."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from models_mae import mae_vit_base_patch16_dec512d8b
from dual_yolo_mae.utils import load_ultralytics_model


class MAEBackbone(nn.Module):
    """Wrapper around the MAE encoder that exposes convolutional feature maps."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        output_dims: Sequence[int] = (144, 144, 144),
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = mae_vit_base_patch16_dec512d8b()
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            key = "model" if "model" in state else "state_dict"
            missing, unexpected = self.encoder.load_state_dict(state[key], strict=False)
            if missing:
                print(f"[MAEBackbone] Missing keys while loading checkpoint: {missing}")
            if unexpected:
                print(f"[MAEBackbone] Unexpected keys while loading checkpoint: {unexpected}")
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        self.embed_dim = self.encoder.patch_embed.proj.out_channels
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.embed_dim, dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.SiLU(inplace=True),
                )
                for dim in output_dims
            ]
        )

    def forward(self, x: torch.Tensor, target_shapes: Sequence[Tuple[int, int]]) -> List[torch.Tensor]:
        base = self._encode(x)
        outputs: List[torch.Tensor] = []
        for proj, shape in zip(self.projections, target_shapes):
            resized = base if base.shape[-2:] == shape else F.interpolate(base, size=shape, mode="bilinear", align_corners=False)
            outputs.append(proj(resized))
        return outputs

    @torch.no_grad()
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed[:, 1:, :]
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        x = x[:, 1:, :]
        side = int(math.sqrt(x.shape[1]))
        x = x.transpose(1, 2).contiguous().view(B, self.embed_dim, side, side)
        return x


class YOLOBackbone(nn.Module):
    """Feature extractor that wraps a pretrained YOLOv8 model."""

    def __init__(
        self,
        weights: str,
        freeze: bool = True,
        dummy_input: int = 640,
    ) -> None:
        super().__init__()
        self.yolo = load_ultralytics_model(weights).model
        if freeze:
            for param in self.yolo.parameters():
                param.requires_grad = False
            self.yolo.eval()
        self.strides = self.yolo.stride.tolist()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, dummy_input, dummy_input)
            preds, feats = self.yolo(dummy)
            self.output_channels = [f.shape[1] for f in feats]
        self.num_scales = len(self.output_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        _, feats = self.yolo(x)
        return feats


class DetectionHead(nn.Module):
    """Simple convolutional detection head."""

    def __init__(self, in_channels: Sequence[int], num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(ch, self.num_outputs, kernel_size=1),
                )
                for ch in in_channels
            ]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [layer(feat) for layer, feat in zip(self.layers, features)]


@dataclass
class LossWeights:
    box: float = 5.0
    obj: float = 1.0
    cls: float = 1.0


class DualBackboneYOLO(nn.Module):
    """Detector that fuses YOLOv8 and MAE representations."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        model_cfg = config.get("model", config)
        mae_cfg = model_cfg.get("mae", {})
        yolo_cfg = model_cfg.get("yolo", {})
        fusion_cfg = model_cfg.get("fusion", {})

        self.num_classes = int(model_cfg.get("num_classes", 80))
        self.scale_ranges = model_cfg.get(
            "scale_ranges", [[0.0, 0.07], [0.07, 0.15], [0.15, 1.0]]
        )
        self.loss_weights = LossWeights(**model_cfg.get("loss_weights", {}))

        mae_output_dims = mae_cfg.get("output_dims", [144, 144, 144])
        self.mae_backbone = MAEBackbone(
            checkpoint=mae_cfg.get("checkpoint"),
            output_dims=mae_output_dims,
            freeze=mae_cfg.get("freeze", True),
        )

        self.yolo_backbone = YOLOBackbone(
            weights=yolo_cfg.get("weights", "yolov8n.pt"),
            freeze=yolo_cfg.get("freeze", True),
            dummy_input=model_cfg.get("input_size", 640),
        )

        fusion_dims = fusion_cfg.get("channels", self.yolo_backbone.output_channels)
        if len(fusion_dims) != self.yolo_backbone.num_scales:
            raise ValueError("Fusion channel list must match number of feature scales from YOLO.")

        self.fusion_layers = nn.ModuleList()
        for yolo_ch, mae_ch, fused_ch in zip(
            self.yolo_backbone.output_channels, mae_output_dims, fusion_dims
        ):
            self.fusion_layers.append(
                nn.Sequential(
                    nn.Conv2d(yolo_ch + mae_ch, fused_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(fused_ch),
                    nn.SiLU(inplace=True),
                )
            )

        self.detection_head = DetectionHead(fusion_dims, self.num_classes)
        self.bce = nn.BCEWithLogitsLoss()
        self.reg_loss = nn.SmoothL1Loss()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        yolo_feats = self.yolo_backbone(x)
        spatial_shapes = [feat.shape[-2:] for feat in yolo_feats]
        mae_feats = self.mae_backbone(x, spatial_shapes)
        fused_feats = [
            fusion(torch.cat([y, m], dim=1))
            for fusion, y, m in zip(self.fusion_layers, yolo_feats, mae_feats)
        ]
        return self.detection_head(fused_feats)

    def select_scale(self, box_size: float) -> int:
        for idx, (low, high) in enumerate(self.scale_ranges):
            if low <= box_size < high:
                return idx
        return len(self.scale_ranges) - 1

    def build_targets(
        self, preds: Sequence[torch.Tensor], targets: List[Dict]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        device = preds[0].device
        obj_targets: List[torch.Tensor] = []
        box_targets: List[torch.Tensor] = []
        cls_targets: List[torch.Tensor] = []

        for pred in preds:
            B, _, H, W = pred.shape
            obj_targets.append(torch.zeros((B, H, W), device=device))
            box_targets.append(torch.zeros((B, H, W, 4), device=device))
            cls_targets.append(torch.zeros((B, H, W, self.num_classes), device=device))

        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"].to(device)
            labels = target["labels"].to(device)
            if boxes.numel() == 0:
                continue
            for box, label in zip(boxes, labels):
                x_c, y_c, w, h = box.tolist()
                scale_idx = self.select_scale(max(w, h))
                grid_h, grid_w = obj_targets[scale_idx].shape[1:]
                gi = min(int(x_c * grid_w), grid_w - 1)
                gj = min(int(y_c * grid_h), grid_h - 1)
                obj_targets[scale_idx][batch_idx, gj, gi] = 1.0
                box_targets[scale_idx][batch_idx, gj, gi] = torch.tensor(
                    [x_c, y_c, w, h], device=device
                )
                cls_targets[scale_idx][batch_idx, gj, gi, int(label.item())] = 1.0

        return obj_targets, box_targets, cls_targets

    def compute_loss(
        self, preds: Sequence[torch.Tensor], targets: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        obj_targets, box_targets, cls_targets = self.build_targets(preds, targets)
        loss_obj = torch.tensor(0.0, device=preds[0].device)
        loss_box = torch.tensor(0.0, device=preds[0].device)
        loss_cls = torch.tensor(0.0, device=preds[0].device)

        for pred, obj_t, box_t, cls_t in zip(preds, obj_targets, box_targets, cls_targets):
            pred = pred.permute(0, 2, 3, 1)
            obj_logit = pred[..., 0]
            box_logit = pred[..., 1:5]
            cls_logit = pred[..., 5:]

            loss_obj = loss_obj + self.bce(obj_logit, obj_t)

            positive = obj_t > 0.5
            if positive.any():
                box_pred = torch.sigmoid(box_logit[positive])
                loss_box = loss_box + self.reg_loss(box_pred, box_t[positive])
                loss_cls = loss_cls + self.bce(cls_logit[positive], cls_t[positive])

        total = (
            self.loss_weights.box * loss_box
            + self.loss_weights.obj * loss_obj
            + self.loss_weights.cls * loss_cls
        )
        return {
            "loss": total,
            "loss_box": loss_box.detach(),
            "loss_obj": loss_obj.detach(),
            "loss_cls": loss_cls.detach(),
        }

    def decode_predictions(
        self,
        preds: Sequence[torch.Tensor],
        image_sizes: Sequence[Tuple[int, int]],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[List[Dict[str, float]]]:
        results: List[List[Dict[str, float]]] = []
        batch_size = preds[0].shape[0]
        for batch_idx in range(batch_size):
            boxes_accum: List[torch.Tensor] = []
            scores_accum: List[torch.Tensor] = []
            labels_accum: List[torch.Tensor] = []
            for scale_pred in preds:
                logits = scale_pred[batch_idx].permute(1, 2, 0)
                probs = torch.sigmoid(logits)
                obj = probs[..., 0]
                box = probs[..., 1:5]
                cls_probs = probs[..., 5:]
                cls_scores, cls_idx = torch.max(cls_probs, dim=-1)
                conf = cls_scores * obj
                mask = conf > conf_threshold
                if mask.sum() == 0:
                    continue
                h_img, w_img = image_sizes[batch_idx]
                selected_box = box[mask]
                selected_conf = conf[mask]
                selected_cls = cls_idx[mask]

                x_c = selected_box[:, 0] * w_img
                y_c = selected_box[:, 1] * h_img
                widths = selected_box[:, 2] * w_img
                heights = selected_box[:, 3] * h_img
                x1 = x_c - widths / 2
                y1 = y_c - heights / 2
                x2 = x_c + widths / 2
                y2 = y_c + heights / 2
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)

                boxes_accum.append(boxes)
                scores_accum.append(selected_conf)
                labels_accum.append(selected_cls)

            if not boxes_accum:
                results.append([])
                continue

            boxes = torch.cat(boxes_accum, dim=0)
            scores = torch.cat(scores_accum, dim=0)
            labels = torch.cat(labels_accum, dim=0)

            keep = nms(boxes, scores, iou_threshold)
            keep_boxes = boxes[keep].cpu().tolist()
            keep_scores = scores[keep].cpu().tolist()
            keep_labels = labels[keep].cpu().tolist()

            detections = []
            for box_coords, score, label in zip(keep_boxes, keep_scores, keep_labels):
                detections.append(
                    {
                        "box": box_coords,
                        "score": float(score),
                        "label": int(label),
                    }
                )
            results.append(detections)
        return results
