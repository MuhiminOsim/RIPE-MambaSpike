"""
Mamba-Spike with real mamba_ssm backend.
Drop-in replacement for mamba_spike.py — same SpikingFrontEnd and SpikeToActivation,
but MambaBlock uses mamba_ssm.Mamba instead of the hand-rolled Python SSM.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import snntorch as snn
from snntorch import surrogate
from typing import Optional, Tuple

from mamba_ssm import Mamba as MambaSSM


class MambaBlock(nn.Module):
    """Mamba block using real mamba_ssm kernel (CUDA-fused, not a Python loop)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class SpikingFrontEnd(nn.Module):
    """Spiking front-end: LIF neurons with recurrent connections and spatial pooling."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        beta: float = 0.97,
        spike_grad: Optional[object] = None,
        use_recurrent: bool = True,
        pool_1d: bool = False,
    ):
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)
        self.use_recurrent = use_recurrent

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        if use_recurrent:
            self.recurrent1 = nn.Conv2d(hidden_channels, hidden_channels, 1)
            self.recurrent2 = nn.Conv2d(hidden_channels, hidden_channels, 1)
            self.recurrent3 = nn.Conv2d(out_channels, out_channels, 1)

        self.pool = nn.MaxPool2d((2, 1) if pool_1d else 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, C, H, W)  →  spikes: (B, T, C_out, H/4, W/4)"""
        if len(x.shape) == 4:
            batch_size, time_steps, c, h = x.shape
            x = x.unsqueeze(-1)
            is_1d = True
        else:
            batch_size, time_steps, _, _, _ = x.shape
            is_1d = False

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk1_prev = spk2_prev = spk3_prev = None
        spk_rec = []

        for t in range(time_steps):
            x_t = x[:, t]
            cur1 = self.pool(self.conv1(x_t))
            if self.use_recurrent and spk1_prev is not None:
                cur1 = cur1 + self.recurrent1(spk1_prev)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_prev = spk1

            cur2 = self.pool(self.conv2(spk1))
            if self.use_recurrent and spk2_prev is not None:
                cur2 = cur2 + self.recurrent2(spk2_prev)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_prev = spk2

            cur3 = self.conv3(spk2)
            if self.use_recurrent and spk3_prev is not None:
                cur3 = cur3 + self.recurrent3(spk3_prev)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_prev = spk3
            spk_rec.append(spk3)

        spikes = torch.stack(spk_rec, dim=1)
        if is_1d:
            spikes = spikes.squeeze(-1)
        return spikes, mem3


class SpikeToActivation(nn.Module):
    """Convert spike tensor to continuous activations via windowed firing rate."""

    def __init__(self, time_window: int = 5):
        super().__init__()
        self.time_window = time_window

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        if len(spikes.shape) == 4:
            batch_size, time_steps, channels, height = spikes.shape
        else:
            batch_size, time_steps, channels, height, width = spikes.shape

        spikes_flat = spikes.view(batch_size, time_steps, -1)
        padded = F.pad(spikes_flat, (0, 0, self.time_window - 1, 0)).transpose(1, 2)
        windows = padded.unfold(dimension=2, size=self.time_window, step=1)
        spike_counts = windows.sum(dim=-1).transpose(1, 2)
        return spike_counts.float() / self.time_window


class MambaSpike(nn.Module):
    """Mamba-Spike with real mamba_ssm backbone."""

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
    ):
        super().__init__()

        self.spiking_frontend = SpikingFrontEnd(
            in_channels=input_channels,
            hidden_channels=32,
            out_channels=spiking_channels,
            beta=beta,
            pool_1d=pool_1d,
        )

        h_out = input_size[0] // 4
        w_out = input_size[1] // 4 if not pool_1d else input_size[1]
        spike_features = spiking_channels * h_out * w_out

        self.spike_to_activation = SpikeToActivation(time_window=5)
        self.input_proj = nn.Linear(spike_features, d_model)

        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) → logits: (B, num_classes)"""
        spikes, _ = self.spiking_frontend(x)
        activations = self.spike_to_activation(spikes)
        x = self.input_proj(activations)
        for block in self.mamba_blocks:
            x = block(x)
        x = self.norm(x.mean(dim=1))
        return self.classifier(x)


def create_mamba_spike_cifar10dvs(num_classes: int = 10) -> MambaSpike:
    return MambaSpike(
        input_channels=2,
        input_size=(128, 128),
        num_classes=num_classes,
        spiking_channels=128,
        d_model=256,
        n_layers=6,
        d_state=16,
        beta=0.97,
    )


def create_mamba_spike_nmnist(num_classes: int = 10) -> MambaSpike:
    return MambaSpike(
        input_channels=2,
        input_size=(34, 34),
        num_classes=num_classes,
        spiking_channels=64,
        d_model=128,
        n_layers=4,
        d_state=16,
        beta=0.97,
    )


def create_mamba_spike_dvsgesture(num_classes: int = 11) -> MambaSpike:
    return MambaSpike(
        input_channels=2,
        input_size=(128, 128),
        num_classes=num_classes,
        spiking_channels=128,
        d_model=256,
        n_layers=6,
        d_state=16,
        beta=0.97,
    )
