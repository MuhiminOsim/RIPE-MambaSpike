"""Learning-rate and surrogate-sharpness schedules."""

import math

__all__ = ["CosineWarmup", "surrogate_alpha", "continuous_weight"]


class CosineWarmup:
    """Linear warm-up followed by cosine decay, stepped once per epoch.

    The step counter is part of the object's state so a resumed run picks the
    schedule up where it stopped rather than restarting the warm-up.
    """

    def __init__(self, optimizer, warmup: int, total: int, lr_init: float, lr_min: float):
        self.optimizer = optimizer
        self.warmup = warmup
        self.total = total
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.step_idx = 0

    def step(self) -> float:
        self.step_idx += 1
        if self.step_idx <= self.warmup:
            lr = self.lr_min + (self.lr_init - self.lr_min) * (self.step_idx / self.warmup)
        else:
            progress = (self.step_idx - self.warmup) / max(1, self.total - self.warmup)
            lr = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self):
        return {"step_idx": self.step_idx}

    def load_state_dict(self, state):
        self.step_idx = state.get("step_idx", 0)


def surrogate_alpha(epoch: int, total_epochs: int, start: float = 2.0, end: float = 4.0) -> float:
    """Anneal surrogate sharpness, widening the gradient early and narrowing it late.

    The end point is per-benchmark: 4.0 on DVS-Gesture, N-MNIST and
    DailyDVS-200, and 3.0 on CIFAR10-DVS and N-Caltech101.
    """
    return start + (end - start) * (epoch / max(1, total_epochs - 1))


def continuous_weight(epoch: int, total_epochs: int, floor: float = 0.1, start: float = 0.5) -> float:
    """Weight on the relaxed forward pass, decayed toward ``floor``."""
    return max(floor, start * (1.0 - epoch / max(1, total_epochs)))
