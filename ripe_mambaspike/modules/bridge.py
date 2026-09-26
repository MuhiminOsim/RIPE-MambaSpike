"""Spike-to-token bridge (MR-S3A).

The bridge is where the parameter cost of a spiking/state-space hybrid is
usually decided. Projecting a flattened feature map into the sequence model
makes the projection quadratic in sensor resolution; pooling every stage to a
common grid and projecting at a fixed model width does not, so no state-space
parameter tracks ``H x W``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SpikeToRate", "MRS3ABridge"]


class SpikeToRate(nn.Module):
    """Convert a spike train to a running rate and its increment.

    Emits ``2C`` channels: the cumulative firing rate up to each step, and the
    step-to-step change in that rate. The rate carries the accumulated
    evidence, the increment carries what the current step added.

    Shape: ``(B, T, C, H, W)`` in, ``(B, T, 2C, H, W)`` out.
    """

    def forward(self, spikes):
        B, T, C, H, W = spikes.shape
        t_idx = torch.arange(1, T + 1, device=spikes.device).view(1, -1, 1, 1, 1)
        rates = torch.cumsum(spikes, dim=1) / t_idx
        rates_prev = F.pad(rates[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
        return torch.cat([rates, rates - rates_prev], dim=2)


class MRS3ABridge(nn.Module):
    """Multi-Resolution Spiking-State-Space Aggregation.

    Every stage's rate map is pooled to the final stage's grid and projected
    to a common width ``d_model`` by its own ``3x3`` convolution, then the
    scales are combined by softmax-normalized learnable weights. Because
    ``d_model`` is fixed and pooling is adaptive, the parameter count depends
    on the stage widths only, never on the input resolution.

    Args:
        c1, c2, c3: channel counts of the three spiking stages.
        d_model: output token width.
        c4: optional fourth stage's channel count.
    """

    def __init__(self, c1, c2, c3, d_model, c4=None):
        super().__init__()
        self.proj1 = self._make_proj(2 * c1, d_model)
        self.proj2 = self._make_proj(2 * c2, d_model)
        self.proj3 = self._make_proj(2 * c3, d_model)
        self.n_scales = 3 if c4 is None else 4
        self.scale_weights = nn.Parameter(torch.ones(self.n_scales))
        if c4 is not None:
            self.proj4 = self._make_proj(2 * c4, d_model)

    @staticmethod
    def _make_proj(c_in, d_model):
        return nn.Sequential(
            nn.Conv2d(c_in, d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
        )

    @staticmethod
    def _pool(x, size):
        B, T = x.shape[0], x.shape[1]
        return F.adaptive_avg_pool2d(x.reshape(B * T, *x.shape[2:]), size)

    def forward(self, s1, s2, s3, s4=None):
        target = s3 if s4 is None else s4
        B, T = target.shape[0], target.shape[1]
        grid = target.shape[-2:]

        w = F.softmax(self.scale_weights, dim=0)
        out = w[0] * self.proj1(self._pool(s1, grid))
        out = out + w[1] * self.proj2(self._pool(s2, grid))
        if s4 is None:
            out = out + w[2] * self.proj3(s3.reshape(B * T, *s3.shape[2:]))
        else:
            out = out + w[2] * self.proj3(self._pool(s3, grid))
            out = out + w[3] * self.proj4(s4.reshape(B * T, *s4.shape[2:]))
        return out.view(B, T, -1, *grid)
