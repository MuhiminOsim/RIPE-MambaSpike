"""Spiking front-end stage."""

import torch
import torch.nn as nn

from ..layers.attention import DSMPA
from ..layers.neurons import build_neuron
from ..layers.normalization import make_temporal_norm
from ..layers.reparam import RepConvBlock
from .tdm import shift_prev, temporal_decoupled_modulation

__all__ = ["SpikingRepStage"]


class SpikingRepStage(nn.Module):
    """One stage of the reparameterized spiking front-end.

    The stage runs ``TDM -> RepConv -> temporal norm -> neuron`` twice, with
    DS-MPA between the two neurons. The attention reads queries and keys from
    the first neuron's membrane potential and values from its previous-step
    spikes, then modulates the second convolution's output.

    Args:
        in_channels: input channels.
        out_channels: output channels.
        T: number of timesteps.
        stride: stride of the first convolution; the second is always 1.
        neuron_type: ``'lif'``, ``'silif'`` or ``'csilif'``.
        v_threshold: initial firing threshold.
        norm_type: ``'bntt'`` or ``'tdbn'``.

    Returns:
        ``(spikes, potentials, attention, spike_rate)``.
    """

    def __init__(self, in_channels, out_channels, T, stride=2, neuron_type="lif",
                 v_threshold=1.0, norm_type="bntt"):
        super().__init__()
        self.conv1 = RepConvBlock(in_channels, out_channels, stride=stride)
        self.bntt1 = make_temporal_norm(norm_type, out_channels, T)
        self.plif1 = build_neuron(neuron_type, out_channels, v_threshold=v_threshold)

        self.conv2 = RepConvBlock(out_channels, out_channels, stride=1)
        self.bntt2 = make_temporal_norm(norm_type, out_channels, T)
        self.plif2 = build_neuron(neuron_type, out_channels, v_threshold=v_threshold)

        self.ds_mpa = DSMPA(out_channels)

        self.alpha1 = nn.Parameter(torch.zeros(in_channels))
        self.alpha2 = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, continuous_mode=False):
        B, T, C, H, W = x.shape
        self.plif1.continuous_mode = continuous_mode
        self.plif2.continuous_mode = continuous_mode

        x = temporal_decoupled_modulation(x, self.alpha1)
        h = self.conv1(x.view(B * T, C, H, W))
        h = self.bntt1(h.view(B, T, *h.shape[1:]))

        S1, U1 = self.plif1(h)
        S1_prev = shift_prev(S1)

        g = temporal_decoupled_modulation(S1, self.alpha2)
        g = self.conv2(g.view(B * T, *S1.shape[2:]))
        X2 = self.bntt2(g.view(B, T, *g.shape[1:]))

        X2, A_t = self.ds_mpa(U1, S1_prev, X2)
        S2, U2 = self.plif2(X2)

        spike_rate = S1.mean(dtype=torch.float32) + S2.mean(dtype=torch.float32)
        return S2, U2, A_t, spike_rate
