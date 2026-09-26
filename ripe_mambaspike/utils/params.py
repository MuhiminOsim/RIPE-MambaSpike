"""Parameter accounting."""

import copy

import torch

__all__ = ["count_parameters", "deployed_parameters"]


def count_parameters(model, trainable_only: bool = False) -> int:
    """Total parameter count of ``model``."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


@torch.no_grad()
def deployed_parameters(model) -> int:
    """Parameter count after reparameterization fusion.

    Fuses a copy, so the model passed in is left training-ready.
    """
    clone = copy.deepcopy(model)
    clone.eval()
    clone.fuse_model()
    return count_parameters(clone)
