import torch
import torch.nn.functional as F


def interpolate_pos_embed(pos_embed: torch.Tensor, H_feat: int, W_feat: int) -> torch.Tensor:
    """Interpolate positional embeddings to match feature map size.

    Args:
        pos_embed: Positional embeddings of shape [1, N, C] with optional CLS token.
        H_feat: Target feature map height.
        W_feat: Target feature map width.

    Returns:
        Resized positional embeddings with shape [1, H_feat * W_feat (+1), C].
    """
    if pos_embed.ndim != 3:
        raise ValueError("pos_embed must have shape [1, N, C]")

    cls_pos = None
    if pos_embed.shape[1] != H_feat * W_feat:
        cls_pos = pos_embed[:, :1]
        pos_tokens = pos_embed[:, 1:]
    else:
        pos_tokens = pos_embed

    n = pos_tokens.shape[1]
    orig_size = int(n**0.5)
    if orig_size * orig_size != n:
        raise ValueError("positional embedding has non-square grid")

    pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=(H_feat, W_feat), mode="bicubic", align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, H_feat * W_feat, -1)

    if cls_pos is not None:
        pos_tokens = torch.cat((cls_pos, pos_tokens), dim=1)
    return pos_tokens


def tokens_to_feature_map(x_tokens: torch.Tensor, H_feat: int, W_feat: int) -> torch.Tensor:
    """Convert token sequence to spatial feature map."""
    B, L, C = x_tokens.shape
    if L != H_feat * W_feat:
        raise ValueError("Token length does not match target dimensions")
    return x_tokens.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)


def check_divisible(img_h: int, img_w: int, patch: int = 16) -> None:
    """Ensure image dimensions are divisible by patch size."""
    if img_h % patch != 0 or img_w % patch != 0:
        raise ValueError(f"Image size ({img_h}, {img_w}) not divisible by patch size {patch}")
