"""
Mamba-Spike with a POOLED interface -- the interface-swap control.
============================================================================
This is the "minimal fix" baseline requested in review: Mamba-Spike with
*only* its SNN->SSM interface replaced, and nothing else changed.

The released model builds its interface as

    h_out, w_out   = H // 4, W // 4
    spike_features = spiking_channels * h_out * w_out
    input_proj     = nn.Linear(spike_features, d_model)          # <-- quadratic in H,W

so `input_proj` alone is 33,554,688 parameters at 128x128 and 102,760,704 at
224x224. This variant inserts an adaptive average pool to a fixed pool_grid x
pool_grid grid before the projection:

    input_proj     = nn.Linear(spiking_channels * pool_grid**2, d_model)

which is resolution-independent, i.e. the standard remedy (pool/patchify to a
fixed grid before projecting) that predates this work.

Everything else -- SpikingFrontEnd, SpikeToActivation, MambaBlock, the norm and
the classifier -- is IMPORTED from mamba_spike_real, not reimplemented, so the
only difference between this model and the released one is the interface layer.
That is what makes the comparison a control rather than a reimplementation.

Choice of pool_grid: larger keeps more spatial detail and is the *stronger*
baseline. pool_grid=8 is the default here deliberately -- it matches the grid
the released model already uses at 34x34 (34//4 = 8), so it is the baseline's
own operating point rather than a number chosen to make it look bad.

Usage:
    from models.mamba_spike_pooled import create_pooled_cifar10dvs
    model = create_pooled_cifar10dvs(num_classes=10, pool_grid=8)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mamba_spike_real import (
    SpikingFrontEnd,
    SpikeToActivation,
    MambaBlock,
)


class MambaSpikePooled(nn.Module):
    """Mamba-Spike with the flatten-then-project interface replaced by
    adaptive-pool-then-project. All other components are the released ones."""

    def __init__(
        self,
        input_channels: int = 2,
        input_size: Tuple[int, int] = (128, 128),
        num_classes: int = 10,
        spiking_channels: int = 128,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        beta: float = 0.97,
        pool_1d: bool = False,
        pool_grid: int = 8,
    ):
        super().__init__()
        if pool_1d:
            raise NotImplementedError(
                "pool_1d collapses the spatial grid the pooled interface acts on; "
                "the control is defined for the 2-D front-end only."
            )

        self.spiking_frontend = SpikingFrontEnd(
            in_channels=input_channels,
            hidden_channels=32,
            out_channels=spiking_channels,
            beta=beta,
            pool_1d=pool_1d,
        )

        # Spatial grid the front-end actually emits. Retained so forward() can
        # un-flatten SpikeToActivation's output before pooling.
        self.spiking_channels = spiking_channels
        self.h_out = input_size[0] // 4
        self.w_out = input_size[1] // 4
        self.pool_grid = pool_grid

        self.spike_to_activation = SpikeToActivation(time_window=5)

        # THE ONLY ARCHITECTURAL CHANGE: fixed-size projection.
        self.input_proj = nn.Linear(spiking_channels * pool_grid * pool_grid, d_model)

        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) -> logits: (B, num_classes)"""
        spikes, _ = self.spiking_frontend(x)
        activations = self.spike_to_activation(spikes)      # (B, T, C*h_out*w_out)

        B, T, _ = activations.shape
        # SpikeToActivation ends on a transpose, so the result is not
        # contiguous; reshape (not view) is required here.
        a = activations.reshape(B * T, self.spiking_channels, self.h_out, self.w_out)
        a = F.adaptive_avg_pool2d(a, (self.pool_grid, self.pool_grid))
        a = a.reshape(B, T, -1)                             # (B, T, C*pool_grid^2)

        x = self.input_proj(a)
        for block in self.mamba_blocks:
            x = block(x)
        x = self.norm(x.mean(dim=1))
        return self.classifier(x)


# --- factories mirroring mamba_spike_real's, with pool_grid added -----------

def create_pooled_cifar10dvs(num_classes: int = 10, pool_grid: int = 8):
    return MambaSpikePooled(
        input_channels=2, input_size=(128, 128), num_classes=num_classes,
        spiking_channels=128, d_model=256, n_layers=6, d_state=16, beta=0.97,
        pool_grid=pool_grid,
    )


def create_pooled_ncaltech101(num_classes: int = 101, pool_grid: int = 8):
    # Same widths as the CIFAR10-DVS config; N-Caltech101 also runs at 128^2.
    return MambaSpikePooled(
        input_channels=2, input_size=(128, 128), num_classes=num_classes,
        spiking_channels=128, d_model=256, n_layers=6, d_state=16, beta=0.97,
        pool_grid=pool_grid,
    )


def create_pooled_dvsgesture(num_classes: int = 11, pool_grid: int = 8):
    return MambaSpikePooled(
        input_channels=2, input_size=(128, 128), num_classes=num_classes,
        spiking_channels=128, d_model=256, n_layers=6, d_state=16, beta=0.97,
        pool_grid=pool_grid,
    )


def create_pooled_nmnist(num_classes: int = 10, pool_grid: int = 8):
    # At 34^2 the released model's own grid is 34//4 = 8, so pool_grid=8 is a
    # no-op here: the pooled and released interfaces are identical in size.
    return MambaSpikePooled(
        input_channels=2, input_size=(34, 34), num_classes=num_classes,
        spiking_channels=64, d_model=128, n_layers=4, d_state=16, beta=0.97,
        pool_grid=pool_grid,
    )
