"""Temporal normalization layers for spiking feature maps."""

import torch
import torch.nn as nn

__all__ = ["BNTT2d", "TemporalSharedBN", "make_temporal_norm"]


class BNTT2d(nn.Module):
    """Batch Normalization Through Time.

    Holds one ``nn.BatchNorm2d`` per timestep index, so each step is
    normalized by statistics gathered at that step alone.

    Shape:
        input / output ``(B, T, C, H, W)``
    """

    def __init__(self, num_features: int, T: int):
        super().__init__()
        self.T = T
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features) for _ in range(T)])

    def forward(self, x):
        out = [self.bns[t](x[:, t]) for t in range(self.T)]
        return torch.stack(out, dim=1)


class TemporalSharedBN(nn.Module):
    """tdBN-style normalization: one BatchNorm pooled over batch and time.

    Statistics are gathered over ``B * T`` rather than ``B``, which makes the
    layer invariant to time-reversal augmentation and gives less noisy
    estimates. ``T`` is accepted for interface parity with :class:`BNTT2d`
    and ignored.
    """

    def __init__(self, num_features: int, T: int = None):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        B, T, C, H, W = x.shape
        return self.bn(x.reshape(B * T, C, H, W)).view(B, T, C, H, W)


def make_temporal_norm(norm_type: str, num_features: int, T: int) -> nn.Module:
    """Build a temporal norm layer. ``'tdbn'`` selects :class:`TemporalSharedBN`,
    anything else selects :class:`BNTT2d`."""
    if norm_type == "tdbn":
        return TemporalSharedBN(num_features, T)
    return BNTT2d(num_features, T)
