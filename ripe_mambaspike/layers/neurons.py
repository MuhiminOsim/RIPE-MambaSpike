"""Spiking neuron models.

All neurons share one interface:

    spikes, membrane = neuron(x)          # x: (B, T, C, H, W)

``spikes`` is binary under the default discrete mode and real-valued under
``continuous_mode``, which replaces the Heaviside by a sigmoid so a second,
differentiable forward pass can be taken for surrogate-gradient consistency.
Returning the membrane potential alongside the spikes is what lets DS-MPA
read queries and keys from the potential while values come from spikes.
"""

import math

import torch
import torch.nn as nn

from .surrogate import ATanSurrogate

__all__ = ["SpikeRateMonitor", "LIFNeuron", "SiLIFNeuron", "CSiLIFNeuron", "SPIKING_NEURONS"]


class SpikeRateMonitor:
    """Sync-free exponential moving average of the firing rate.

    The EMA is kept in a float64 device scalar and only transferred to the
    host when :attr:`spike_rate` is read, so logging a rate costs no
    device synchronization inside the training step.
    """

    @property
    def spike_rate(self) -> float:
        r = getattr(self, "_rate_ema", None)
        return 0.0 if r is None else r.item()

    @spike_rate.setter
    def spike_rate(self, value):
        if value is None:
            self._rate_ema = None
        elif isinstance(value, torch.Tensor):
            self._rate_ema = value.detach().double()
        else:
            self._rate_ema = torch.tensor(float(value), dtype=torch.float64)

    @torch.no_grad()
    def _rate_update(self, out_spikes):
        m = self._ema_momentum
        cur = out_spikes.mean(dtype=torch.float32).double()
        if self._rate_ema is None:
            self._rate_ema = torch.zeros((), dtype=torch.float64, device=cur.device)
        elif self._rate_ema.device != cur.device:
            self._rate_ema = self._rate_ema.to(cur.device)
        self._rate_ema = m * self._rate_ema + (1 - m) * cur


class _BaseNeuron(SpikeRateMonitor, nn.Module):
    def __init__(self):
        super().__init__()
        self._rate_ema = None
        self._ema_momentum = 0.99
        self.alpha = 2.0
        self.continuous_mode = False

    def _fire(self, drive):
        if self.continuous_mode:
            return torch.sigmoid(self.alpha * drive)
        return ATanSurrogate.apply(drive, self.alpha)

    def _finalize(self, spikes, potentials, dtype):
        out_spikes = torch.stack(spikes, dim=1).to(dtype)
        out_u = torch.stack(potentials, dim=1).to(dtype)
        if not self.continuous_mode:
            with torch.no_grad():
                self._rate_update(out_spikes)
        return out_spikes, out_u


class LIFNeuron(_BaseNeuron):
    """Leaky integrate-and-fire with a fixed decay and a soft reset."""

    def __init__(self, num_channels=None, v_threshold: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.beta = beta
        self.v_th = v_threshold

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        mem = torch.zeros(B, C, H, W, device=x_seq.device, dtype=x_seq.dtype)
        spikes, potentials = [], []
        for t in range(T):
            mem = self.beta * mem + x_seq[:, t]
            potentials.append(mem)
            spike = self._fire(mem - self.v_th)
            mem = mem - spike * self.v_th
            spikes.append(spike)
        return self._finalize(spikes, potentials, x_seq.dtype)


class SiLIFNeuron(_BaseNeuron):
    """Selective LIF with a learnable per-channel decay and threshold.

    The decay is parameterized as ``beta = exp(-exp(alpha_log) * exp(delta_log))``
    so it stays in ``(0, 1)`` without clamping, and the input is scaled by
    ``1 - beta`` to keep the steady-state response independent of the decay.
    """

    def __init__(self, num_channels: int, v_threshold: float = 1.0):
        super().__init__()
        self.alpha_log = nn.Parameter(torch.Tensor(num_channels))
        self.delta_log = nn.Parameter(torch.Tensor(num_channels))
        self.v_threshold = nn.Parameter(torch.ones(num_channels) * v_threshold)
        nn.init.uniform_(self.alpha_log, math.log(10.0), math.log(250.0))
        nn.init.constant_(self.delta_log, math.log(0.004))

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        alpha_eff = torch.clamp(self.alpha_log, min=math.log(1.0))
        beta = torch.exp(-torch.exp(alpha_eff) * torch.exp(self.delta_log)).view(1, C, 1, 1)
        input_scale = 1.0 - beta
        v_th = self.v_threshold.view(1, C, 1, 1)

        mem = torch.zeros(B, C, H, W, device=x_seq.device, dtype=x_seq.dtype)
        spikes, potentials = [], []
        for t in range(T):
            mem = beta * mem + input_scale * x_seq[:, t]
            potentials.append(mem)
            spike = self._fire(mem - v_th)
            mem = mem - spike * v_th
            spikes.append(spike)
        return self._finalize(spikes, potentials, x_seq.dtype)


class CSiLIFNeuron(_BaseNeuron):
    """Complex-state selective LIF.

    The membrane state is complex, discretized from a continuous-time pole
    ``-exp(log_log_alpha) + i * alpha_img`` by a learnable step ``exp(log_dt)``.
    The oscillatory component gives the first stage a longer, frequency-selective
    memory than a real-valued decay can provide at the same parameter cost.
    """

    def __init__(self, num_channels: int, v_threshold: float = 1.0):
        super().__init__()
        self.log_log_alpha = nn.Parameter(torch.log(25.0 * torch.ones(num_channels)))
        self.alpha_img = nn.Parameter(math.pi * torch.ones(num_channels))
        self.log_dt = nn.Parameter(torch.full((num_channels,), math.log(0.004)))
        self.b = nn.Parameter(torch.ones(num_channels))
        self.v_threshold = nn.Parameter(torch.ones(num_channels) * v_threshold)
        self.reset_factor = 0.5

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        alpha_cont = -torch.exp(self.log_log_alpha) + 1j * self.alpha_img
        alpha = torch.exp(alpha_cont * torch.exp(self.log_dt)).view(1, C, 1, 1)
        b = self.b.view(1, C, 1, 1)
        v_th = self.v_threshold.view(1, C, 1, 1)

        mem = torch.zeros(B, C, H, W, device=x_seq.device, dtype=torch.cfloat)
        st = torch.zeros(B, C, H, W, device=x_seq.device, dtype=x_seq.dtype)
        spikes, potentials = [], []
        for t in range(T):
            mem = alpha * (mem - self.reset_factor * st) + b * x_seq[:, t]
            U = 2 * mem.real
            potentials.append(U)
            st = self._fire(U - v_th)
            spikes.append(st)
        return self._finalize(spikes, potentials, x_seq.dtype)


SPIKING_NEURONS = (LIFNeuron, SiLIFNeuron, CSiLIFNeuron)

NEURON_REGISTRY = {
    "lif": LIFNeuron,
    "silif": SiLIFNeuron,
    "csilif": CSiLIFNeuron,
}


def build_neuron(neuron_type: str, num_channels: int, v_threshold: float = 1.0):
    """Instantiate a neuron by name (``'lif'``, ``'silif'`` or ``'csilif'``)."""
    cls = NEURON_REGISTRY.get(neuron_type, LIFNeuron)
    return cls(num_channels, v_threshold=v_threshold)
