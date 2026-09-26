"""Optimizer construction."""

import torch

__all__ = ["build_optimizer", "split_param_groups"]

_NO_DECAY_KEYS = (".log", ".alpha", ".delta", ".b", "bias", "v_threshold")


def split_param_groups(model):
    """Separate parameters that should not be weight-decayed.

    Neuron time constants, thresholds, TDM coefficients, norms and biases are
    exempt: decaying them pulls the dynamics toward a degenerate regime rather
    than regularizing capacity.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.ndim <= 1 or any(key in name for key in _NO_DECAY_KEYS):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def build_optimizer(model, lr: float, weight_decay: float = 0.06):
    """AdamW with the no-decay group split out."""
    decay, no_decay = split_param_groups(model)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )
