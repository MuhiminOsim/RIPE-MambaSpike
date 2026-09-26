"""Reproducibility helpers."""

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int, deterministic: bool = False):
    """Seed Python, NumPy and Torch.

    ``deterministic`` also pins cuDNN to deterministic kernels, which costs
    throughput and is off by default.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
