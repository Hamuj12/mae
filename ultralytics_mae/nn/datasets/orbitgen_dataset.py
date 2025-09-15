"""OrbitGen dataset loader for Ultralytics models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OrbitGenDataset(Dataset):
    """Dataset for OrbitGen-generated samples.

    Each sample is expected to have an RGB image, optional mask, and a JSON
    metadata file containing bounding boxes and keypoints. Bounding boxes are
    returned in YOLO ``xywh`` format normalized to ``[0, 1]``. Keypoints are
    returned as ``[x, y, v]`` triplets also normalized to ``[0, 1]``.
    """

    def __init__(self, root_dir: str | Path, split: str = "train", img_size: int = 640) -> None:
        self.root = Path(root_dir)
        self.split = split
        self.img_size = img_size

        # e.g. root/images/train, root/masks/train, root/meta/train
        self.img_dir = self.root / "images" / split
        self.mask_dir = self.root / "masks" / split
        self.meta_dir = self.root / "meta" / split

        self.images = sorted(self.img_dir.glob("*.png"))
        if not self.images:
            raise RuntimeError(f"No images found in {self.img_dir}. Check dataset path and split.")

    def __len__(self) -> int:  # noqa: D401 - standard Dataset API
        return len(self.images)

    def _load_meta(self, stem: str) -> Dict[str, Any]:
        """Load metadata JSON for an item, handling optional ``meta_`` prefix."""
        meta_path = self.meta_dir / f"meta_{stem}.json"
        if not meta_path.exists():
            meta_path = self.meta_dir / f"{stem}.json"
        with meta_path.open() as f:
            return json.load(f)

    def __getitem__(self, index: int) -> Dict[str, Any]:  # noqa: D401 - standard Dataset API
        img_path = self.images[index]
        stem = img_path.stem

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img = img.resize((self.img_size, self.img_size))
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        mask_tensor = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
        mask_path = self.mask_dir / f"{stem}.png"
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size))
            mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0

        meta = self._load_meta(stem)
        bboxes = []
        for cls_id, bbox in enumerate(meta.get("bboxes", {}).values()):
            xmin, ymin = bbox["xmin"], bbox["ymin"]
            xmax, ymax = bbox["xmax"], bbox["ymax"]
            xc = ((xmin + xmax) / 2) / w
            yc = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            bboxes.append([float(cls_id), xc, yc, bw, bh])
        bbox_tensor = (
            torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 5), dtype=torch.float32)
        )

        kp_list = meta.get("keypoints", [])
        vis_list = meta.get("keypoint_visibility", [])
        kpts = []
        for i, (x, y) in enumerate(kp_list):
            v = float(vis_list[i]) if i < len(vis_list) else 1.0
            if x > 1 or y > 1:  # assume pixel coordinates if >1
                x /= w
                y /= h
            kpts.append([x, y, v])
        if kpts:
            kpt_tensor = torch.tensor([kpts], dtype=torch.float32)
        else:
            kpt_tensor = torch.zeros((bbox_tensor.shape[0], 0, 3), dtype=torch.float32)

        return {
            "img": img_tensor,
            "bboxes": bbox_tensor,
            "keypoints": kpt_tensor,
            "mask": mask_tensor,
            "meta": meta,
        }