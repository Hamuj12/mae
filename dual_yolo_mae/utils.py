"""Utility helpers for the Dual YOLO + MAE reference implementation."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import site
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import yaml

LOGGER = logging.getLogger("dual_yolo_mae")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a simple console logger."""
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)


def load_config(path: str | Path) -> Dict:
    """Load a YAML configuration file."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


_ULTRALYTICS_MODULE = None


def _load_official_ultralytics_module():
    """Import the pip-installed ultralytics package even if a local copy exists."""
    global _ULTRALYTICS_MODULE
    if _ULTRALYTICS_MODULE is not None:
        return _ULTRALYTICS_MODULE

    for package_dir in site.getsitepackages():
        spec = importlib.machinery.PathFinder.find_spec("ultralytics", [package_dir])
        if spec is not None:
            module = importlib.util.module_from_spec(spec)
            # Ensure subsequent imports reuse this module instance.
            sys.modules["ultralytics"] = module
            spec.loader.exec_module(module)  # type: ignore[arg-type]
            _ULTRALYTICS_MODULE = module
            LOGGER.info("Loaded official ultralytics package from %s", package_dir)
            return module

    raise ImportError(
        "Unable to locate the pip-installed 'ultralytics' package. Install it via 'pip install ultralytics'."
    )


def load_ultralytics_model(weights: str | Path):
    """Load a pretrained YOLO model from the official ultralytics package."""
    module = _load_official_ultralytics_module()
    YOLO = getattr(module, "YOLO")
    return YOLO(str(weights))


class YOLODataset(Dataset):
    """Minimal dataset that reads images and YOLO-format labels."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size: int = 640,
        class_names: Optional[Sequence[str]] = None,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.class_names = list(class_names) if class_names is not None else None
        self.augment = augment

        image_dir = self.root / "images" / split
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        self.image_paths = sorted(
            [p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found in {image_dir}")

        self.label_dir = self.root / "labels" / split
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_labels(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        label_path = self.label_dir / f"{path.stem}.txt"
        if not label_path.exists():
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        boxes: List[List[float]] = []
        labels: List[int] = []
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                x_c, y_c, w, h = map(float, parts[1:])
                boxes.append([x_c, y_c, w, h])
                labels.append(cls_id)

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        if self.img_size is not None:
            image = image.resize((self.img_size, self.img_size))

        image_tensor = self.to_tensor(image)
        boxes, labels = self._load_labels(image_path)

        target = {
            "boxes": boxes,
            "labels": labels,
            "orig_size": torch.tensor(original_size, dtype=torch.float32),
            "path": str(image_path),
        }
        return image_tensor, target


def yolo_collate_fn(batch: Iterable[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    images: List[torch.Tensor] = []
    targets: List[Dict] = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return torch.stack(images, dim=0), targets


def move_targets_to_device(targets: List[Dict], device: torch.device) -> List[Dict]:
    moved: List[Dict] = []
    for target in targets:
        moved.append(
            {
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device),
                "orig_size": target["orig_size"].to(device),
                "path": target["path"],
            }
        )
    return moved


def format_config_for_logging(cfg: Dict, prefix: str = "") -> List[str]:
    lines: List[str] = []
    for key, value in cfg.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(format_config_for_logging(value, prefix + "  "))
        else:
            lines.append(f"{prefix}{key}: {value}")
    return lines
