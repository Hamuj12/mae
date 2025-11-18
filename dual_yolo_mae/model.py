"""Dual-backbone YOLO detector that fuses MAE and YOLOv8 features."""
from __future__ import annotations


import argparse
import torch.serialization
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from models_mae import mae_vit_base_patch16_dec512d8b
from dual_yolo_mae.utils import load_ultralytics_model

# Fix for numpy issue with recent versions
import numpy as np
if not hasattr(np, "float"):
    np.float = float
    
# add near top of model.py
def _interpolate_pos_embed_2d(pos_embed: torch.Tensor, new_hw: tuple[int,int]) -> torch.Tensor:
    # pos_embed: [1, 1+N, C] (with cls at index 0). We’ll return same shape for new grid.
    cls_pos = pos_embed[:, :1, :]          # [1, 1, C]
    patch_pos = pos_embed[:, 1:, :]        # [1, N, C]
    C = patch_pos.shape[-1]
    N = patch_pos.shape[1]
    old_side = int(round(N ** 0.5))
    patch_pos_2d = patch_pos.reshape(1, old_side, old_side, C).permute(0, 3, 1, 2)  # [1,C,H,W]
    new_h, new_w = new_hw
    patch_pos_2d_resized = torch.nn.functional.interpolate(
        patch_pos_2d, size=(new_h, new_w), mode="bicubic", align_corners=False
    )
    patch_pos_new = patch_pos_2d_resized.permute(0, 2, 3, 1).reshape(1, new_h * new_w, C)  # [1, Nnew, C]
    return torch.cat([cls_pos, patch_pos_new], dim=1)

class GatedFusion(nn.Module):
    def __init__(self, yolo_ch: int, mae_ch: int, fused_ch: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(yolo_ch + mae_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(yolo_ch + mae_ch, fused_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(fused_ch),
            nn.SiLU(inplace=True),
        )
        self.out_channels = fused_ch   # <--- Expose this
        self.last_gate_stats: Dict[str, torch.Tensor] = {}

    def forward(self, yolo_f, mae_f):
        gate_in = torch.cat([yolo_f, mae_f], dim=1)
        g = self.gate(gate_in)
        with torch.no_grad():
            self.last_gate_stats = {
                "mean": g.mean().detach(),
                "std": g.std(unbiased=False).detach(),
            }
        fused = torch.cat([yolo_f, g * mae_f], dim=1)
        return self.proj(fused)

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

        # Infer the MAE encoder's native input size (e.g., 224x224 for ViT-B/16)
        enc_size = getattr(self.encoder.patch_embed, "img_size", 224)
        if isinstance(enc_size, (list, tuple)):
            self.mae_in_size = (int(enc_size[0]), int(enc_size[1]))
        else:
            self.mae_in_size = (int(enc_size), int(enc_size))

        self.load_pretrained(checkpoint)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        self.embed_dim = self.encoder.patch_embed.proj.out_channels
        self.output_dims = list(output_dims)
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

    def load_pretrained(self, checkpoint: Optional[str]) -> None:
        if not checkpoint:
            print("[MAEBackbone] No checkpoint provided; using default initialization.")
            return

        try:
            # Allow argparse.Namespace stored inside the checkpoint (we trust our own runs).
            torch.serialization.add_safe_globals([argparse.Namespace])

            # PyTorch 2.6+: default weights_only=True breaks older-style checkpoints.
            # Explicitly disable weights_only, with a fallback for older torch versions.
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            except TypeError:
                # Older PyTorch that doesn’t know weights_only
                state = torch.load(checkpoint, map_location="cpu")

        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unable to load MAE checkpoint from {checkpoint}: {exc}") from exc

        if "state_dict" in state:
            state = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
        elif "model" in state:
            state = state["model"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint at {checkpoint} does not contain a state_dict-compatible mapping.")

        adaptation_notes: List[str] = []
        self._adapt_pos_embed(state, adaptation_notes)
        self._adapt_patch_embed(state, adaptation_notes)

        try:
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        except RuntimeError as exc:  # noqa: BLE001
            print(f"[MAEBackbone] Warning: encountered issue while loading checkpoint: {exc}")
            missing, unexpected = (), ()

        if adaptation_notes:
            print("[MAEBackbone] Applied checkpoint adaptations:")
            for note in adaptation_notes:
                print(f"- {note}")

        print("----- MAE Checkpoint Diagnostics -----")
        print("Missing keys:")
        if missing:
            for key in missing:
                print(f"- {key}")
        else:
            print("- (none)")

        print("Unexpected keys:")
        if unexpected:
            for key in unexpected:
                print(f"- {key}")
        else:
            print("- (none)")
        print("-------------------------------------")

    def _adapt_pos_embed(self, state: Dict[str, torch.Tensor], notes: List[str]) -> None:
        if "pos_embed" not in state:
            return

        ckpt_pos = state["pos_embed"]
        target_pos = self.encoder.pos_embed
        if ckpt_pos.shape == target_pos.shape:
            return

        num_extra = ckpt_pos.shape[1] - 1
        target_num_extra = target_pos.shape[1] - 1
        old_side = int(round(math.sqrt(num_extra)))
        new_side = int(round(math.sqrt(target_num_extra)))
        if old_side * old_side != num_extra or new_side * new_side != target_num_extra:
            notes.append(
                f"pos_embed shape {tuple(ckpt_pos.shape)} incompatible with target {tuple(target_pos.shape)}; key dropped."
            )
            state.pop("pos_embed")
            return

        cls_pos = ckpt_pos[:, :1, :]
        patch_pos = ckpt_pos[:, 1:, :].reshape(1, old_side, old_side, -1).permute(0, 3, 1, 2)
        resized = F.interpolate(patch_pos, size=(new_side, new_side), mode="bicubic", align_corners=False)
        patch_pos_new = resized.permute(0, 2, 3, 1).reshape(1, new_side * new_side, -1)
        state["pos_embed"] = torch.cat([cls_pos, patch_pos_new], dim=1)
        notes.append(f"Interpolated pos_embed from {old_side}x{old_side} to {new_side}x{new_side}.")

    def _adapt_patch_embed(self, state: Dict[str, torch.Tensor], notes: List[str]) -> None:
        weight_key = "patch_embed.proj.weight"
        if weight_key not in state:
            return

        ckpt_weight = state[weight_key]
        target_weight = self.encoder.patch_embed.proj.weight
        if ckpt_weight.shape == target_weight.shape:
            return

        same_out_in = (
            ckpt_weight.shape[0] == target_weight.shape[0]
            and ckpt_weight.shape[1] == target_weight.shape[1]
        )
        if same_out_in:
            resized = F.interpolate(
                ckpt_weight,
                size=target_weight.shape[2:],
                mode="bicubic",
                align_corners=False,
            )
            state[weight_key] = resized
            notes.append(
                "Resized patch_embed.proj.weight from "
                f"{tuple(ckpt_weight.shape[2:])} to {tuple(target_weight.shape[2:])}."
            )
        else:
            notes.append(
                "Removed patch_embed.proj.weight due to incompatible channel dimensions: "
                f"{tuple(ckpt_weight.shape)} -> {tuple(target_weight.shape)}."
            )
            state.pop(weight_key)

    def forward(self, x: torch.Tensor, target_shapes: Sequence[Tuple[int, int]]) -> List[torch.Tensor]:
        base = self._encode(x)
        outputs: List[torch.Tensor] = []
        if len(target_shapes) != len(self.projections):
            raise ValueError(
                "Number of target feature shapes must match MAE projection layers."
            )
        for proj, shape in zip(self.projections, target_shapes):
            resized = base if base.shape[-2:] == shape else F.interpolate(base, size=shape, mode="bilinear", align_corners=False)
            outputs.append(proj(resized))
        return outputs

    @torch.no_grad()
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        # Dynamically reset timm's internal img_size so the assertion passes
        self.encoder.patch_embed.img_size = (H, W)

        # Compute patch embeddings (shape: [B, N, C])
        x = self.encoder.patch_embed(x)
        side_h = H // self.encoder.patch_embed.patch_size[0]
        side_w = W // self.encoder.patch_embed.patch_size[1]

        # Interpolate positional embeddings to the new grid
        pos_embed = _interpolate_pos_embed_2d(self.encoder.pos_embed, (side_h, side_w))  # [1, 1+Nnew, C]

        # Add pos embeddings and cls token
        x = x + pos_embed[:, 1:, :]
        cls_tokens = (self.encoder.cls_token + pos_embed[:, :1, :]).expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Standard ViT forward through transformer blocks
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)

        # Drop cls token and reshape to [B, C, H/patch, W/patch]
        x = x[:, 1:, :]
        x = x.transpose(1, 2).contiguous().view(B, self.embed_dim, side_h, side_w)

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
        self.loss_weights = LossWeights(**model_cfg.get("loss_weights", {}))

        self.yolo_backbone = YOLOBackbone(
            weights=yolo_cfg.get("weights", "yolov8n.pt"),
            freeze=yolo_cfg.get("freeze", True),
            dummy_input=model_cfg.get("input_size", 640),
        )

        fusion_dims_cfg = fusion_cfg.get("channels")
        if fusion_dims_cfg is None:
            fusion_dims = list(self.yolo_backbone.output_channels)
        else:
            fusion_dims = [int(dim) for dim in fusion_dims_cfg]
        if len(fusion_dims) != self.yolo_backbone.num_scales:
            raise ValueError("Fusion channel list must match number of feature scales from YOLO.")

        mae_output_dims_cfg = mae_cfg.get("output_dims")
        if mae_output_dims_cfg is None:
            mae_output_dims = list(fusion_dims)
        else:
            mae_output_dims = [int(dim) for dim in mae_output_dims_cfg]
        if len(mae_output_dims) != self.yolo_backbone.num_scales:
            raise ValueError("MAE output dims must match number of feature scales from YOLO.")

        self.mae_backbone = MAEBackbone(
            checkpoint=mae_cfg.get("checkpoint"),
            output_dims=mae_output_dims,
            freeze=mae_cfg.get("freeze", True),
        )

        self.scale_ranges = [tuple(map(float, r)) for r in model_cfg.get(
            "scale_ranges", [[0.0, 0.07], [0.07, 0.15], [0.15, 1.0]]
        )]
        if len(self.scale_ranges) != self.yolo_backbone.num_scales:
            raise ValueError("scale_ranges length must match number of detection scales.")
        for low, high in self.scale_ranges:
            if high <= low:
                raise ValueError("Each scale range must have high > low.")

        self.fusion_layers = nn.ModuleList(
            [GatedFusion(yolo_ch, mae_ch, fused_ch)
            for yolo_ch, mae_ch, fused_ch in zip(
                self.yolo_backbone.output_channels, mae_output_dims, fusion_dims)]
        )

        self.fusion_channels = [layer.out_channels for layer in self.fusion_layers]
        self.detection_head = DetectionHead(self.fusion_channels, self.num_classes)
        self.bce = nn.BCEWithLogitsLoss()
        self.reg_loss = nn.SmoothL1Loss()
        self._last_gate_statistics: List[Dict[str, torch.Tensor]] = []

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        yolo_feats = self.yolo_backbone(x)
        spatial_shapes = [feat.shape[-2:] for feat in yolo_feats]
        mae_feats = self.mae_backbone(x, spatial_shapes)
        if len(mae_feats) != len(yolo_feats):
            raise RuntimeError("MAE and YOLO backbones returned mismatched feature scales.")
        self._last_gate_statistics = []
        fused_feats = []
        for fusion, y_feat, m_feat in zip(self.fusion_layers, yolo_feats, mae_feats):
            fused_feats.append(fusion(y_feat, m_feat))
            stats = getattr(fusion, "last_gate_stats", None)
            if stats:
                self._last_gate_statistics.append({
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                })
        return self.detection_head(fused_feats)

    def get_gate_statistics(self) -> List[Dict[str, torch.Tensor]]:
        return self._last_gate_statistics

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
