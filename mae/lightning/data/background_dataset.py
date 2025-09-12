from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BackgroundOnlyDataset(Dataset):
    """Dataset that loads images from a directory without labels.

    All ``.png`` and ``.jpg`` files under ``root`` will be loaded. Each image
    undergoes a series of augmentations and is returned as a tensor in
    ``[C, H, W]`` format within ``[0, 1]`` range.
    """

    def __init__(self, root: str | Path, image_size: int = 224) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"Root directory {root} does not exist")

        # Collect image file paths
        exts = ("*.png", "*.jpg", "*.jpeg")
        self.files: List[Path] = []
        for ext in exts:
            self.files.extend(sorted(self.root.rglob(ext)))
        if not self.files:
            raise RuntimeError(f"No images found in {root}")

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.05, hue=0.02
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)
