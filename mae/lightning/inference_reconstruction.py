from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from matplotlib import cm
import torch
from PIL import Image
from torchvision import transforms, utils

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mae.lightning.models.mae_lightning import MAELightning  # noqa: E402

# Fix for numpy issue with recent versions
import numpy as np
if not hasattr(np, "float"):
    np.float = float


def load_images(input_dir: Path) -> Iterable[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in exts:
        files.extend(sorted(input_dir.rglob(ext)))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct images with a trained MAE model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save reconstructions")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Mask ratio for MAE")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MAELightning.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    files = load_images(input_dir)
    if not files:
        raise RuntimeError(f"No images found in {input_dir}")

    for path in files:
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            loss, pred, mask = model.model(tensor, mask_ratio=args.mask_ratio)

        patches = model.model.patchify(tensor)
        mask = mask.unsqueeze(-1)  # [1, L, 1]
        masked = patches * (1 - mask)
        img_masked = model.model.unpatchify(masked)
        recon_patches = pred * mask + masked
        recon = model.model.unpatchify(recon_patches)

        # Compute heatmap (per-pixel squared error averaged over channels)
        # error = ((recon - tensor) ** 2).mean(dim=1, keepdim=True)  # [1,1,H,W]
        # error_norm = error / (error.max() + 1e-8)

        # Save side-by-side reconstruction
        grid = torch.cat([tensor, img_masked, recon], dim=3)  # concatenate along width
        grid_path = output_dir / f"{path.stem}_recon.png"
        utils.save_image(grid, grid_path)

        # Save heatmap overlay
        # heatmap_path = output_dir / f"{path.stem}_heatmap.png"
        # orig = transforms.ToPILImage()(tensor[0].cpu())
        # err = error_norm[0, 0].cpu().numpy()
        # heatmap = cm.get_cmap("inferno")(err)
        # heatmap[..., 3] = 0.6  # set alpha channel
        # heatmap_img = Image.fromarray((heatmap * 255).astype("uint8"))
        # overlay = Image.alpha_composite(orig.convert("RGBA"), heatmap_img)
        # overlay.save(heatmap_path)


if __name__ == "__main__":
    main()
