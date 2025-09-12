import os
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from ..modules.mae_adapter import (
    check_divisible,
    interpolate_pos_embed,
    tokens_to_feature_map,
)


class SimplePatchEmbed(nn.Module):
    """Patch embedding that supports arbitrary image sizes."""

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MaeViTBackbone(nn.Module):
    """MAE ViT-B/16 encoder as backbone producing stride-16 feature map."""

    def __init__(
        self,
        ckpt_path: str | None = None,
        embed_dim: int = 768,
        patch_size: int = 16,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # ViT encoder definition
        self.patch_embed = SimplePatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        num_patches = (224 // patch_size) * (224 // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=12,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
                for _ in range(12)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        if ckpt_path and os.path.isfile(ckpt_path):
            self._load_from_ckpt(ckpt_path)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        self.out_channels = embed_dim

    def _load_from_ckpt(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        new_state = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("model.model.encoder."):
                new_state[k[len("model.model.encoder."):]] = v
            elif k.startswith("encoder."):
                new_state[k[len("encoder."):]] = v
        missing, unexpected = self.load_state_dict(new_state, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

    def forward(self, x: torch.Tensor) -> dict:
        B, _, H, W = x.shape
        check_divisible(H, W, self.patch_size)
        H_feat, W_feat = H // self.patch_size, W // self.patch_size

        x = self.patch_embed(x)
        # Add positional embedding
        pos_embed = interpolate_pos_embed(self.pos_embed, H_feat, W_feat)
        if pos_embed.shape[1] == x.shape[1] + 1:
            cls_tokens = self.cls_token + pos_embed[:, :1]
            x = x + pos_embed[:, 1:]
            x = torch.cat((cls_tokens.expand(B, -1, -1), x), dim=1)
        else:
            x = x + pos_embed
            x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = x[:, 1:, :]  # remove cls token
        feat = tokens_to_feature_map(x, H_feat, W_feat)
        return {"p4": feat}
