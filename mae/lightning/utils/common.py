from __future__ import annotations

import random

import numpy as np
import torch
from omegaconf import OmegaConf


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_config(cfg) -> None:
    """Pretty print configuration to stdout."""
    print(OmegaConf.to_yaml(cfg))
