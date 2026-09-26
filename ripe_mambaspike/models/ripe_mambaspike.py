"""RIPE-MambaSpike: resolution-independent spiking/state-space network."""

import math

import torch
import torch.nn as nn

from ..layers.neurons import SPIKING_NEURONS
from ..layers.normalization import make_temporal_norm
from ..layers.reparam import RepConvBlock
from ..modules.bridge import MRS3ABridge, SpikeToRate
from ..modules.frontend import SpikingRepStage
from ..modules.hierarchy import HierarchicalMambaStack
from ..modules.pruning import gather_tokens, select_tokens, spike_latency_scores

__all__ = ["RIPEMambaSpike"]


class RIPEMambaSpike(nn.Module):
    """Reparameterized spiking front-end joined to a hierarchical Mamba stack.

    The network runs three (optionally four) reparameterized spiking stages,
    converts each stage's spikes to firing rates, fuses them at a fixed token
    width through the MR-S3A bridge, selects tokens by spike latency, and
    scans them with a hierarchical bidirectional state-space stack.

    No parameter in the bridge or the stack depends on ``H x W``, so the
    deployed parameter count is constant across input resolutions.

    Args:
        num_classes: classifier outputs.
        stage_channels: three or four spiking stage widths.
        hier_dims: width at each hierarchy level.
        hier_pool_ratios: token pooling before each level; the first must be 1.
        d_bridge: bridge token width; defaults to ``hier_dims[0]``.
        drop_path_rate: maximum stochastic-depth rate in the stack.
        T: timesteps.
        top_k: tokens retained per timestep; ``0`` keeps all of them.
        v_threshold: initial neuron threshold.
        norm_type: ``'bntt'`` or ``'tdbn'``.
        sgc: run the second, continuous forward pass during training.
        pos_enc: ``'none'`` or ``'fourier'``.
        pe_bands: Fourier bands when ``pos_enc='fourier'``.
        head_dim: widen the penultimate feature to this width before the
            classifier. A ``d``-dimensional feature cannot form the optimal
            simplex for ``C`` classes once ``d`` is close to ``C``, so the
            many-class benchmarks insert this projection. ``None`` keeps the
            single linear head.

    Shape:
        input ``(B, T, 2, H, W)``, output ``(B, T, num_classes)``.

    In training mode ``forward`` returns
    ``(logits_discrete, logits_continuous, spike_rate)``; in eval mode it
    returns the logits alone.
    """

    def __init__(self, num_classes=10, stage_channels=(32, 64, 128),
                 hier_dims=(32, 64, 128), hier_pool_ratios=(1, 2, 2),
                 d_bridge=None, drop_path_rate=0.1, T=16, top_k=256,
                 v_threshold=1.0, norm_type="bntt", sgc=True,
                 pos_enc="none", pe_bands=8, head_dim=None):
        super().__init__()
        self.T = T
        self.top_k = top_k
        self.sgc = sgc
        self.pos_enc = pos_enc
        self.pe_bands = pe_bands

        stage_channels = list(stage_channels)
        if len(stage_channels) == 4:
            c1, c2, c3, c4 = stage_channels
        else:
            c1, c2, c3 = stage_channels
            c4 = None
        self.n_stages = 4 if c4 is not None else 3

        self.entry_bntt = make_temporal_norm(norm_type, 2, T)
        self.stage1 = SpikingRepStage(2, c1, T, stride=2, neuron_type="csilif",
                                      v_threshold=v_threshold, norm_type=norm_type)
        self.stage2 = SpikingRepStage(c1, c2, T, stride=2, neuron_type="silif",
                                      v_threshold=v_threshold, norm_type=norm_type)
        self.stage3 = SpikingRepStage(c2, c3, T, stride=2, neuron_type="silif",
                                      v_threshold=v_threshold, norm_type=norm_type)
        if c4 is not None:
            self.stage4 = SpikingRepStage(c3, c4, T, stride=2, neuron_type="silif",
                                          v_threshold=v_threshold, norm_type=norm_type)

        d_bridge = hier_dims[0] if d_bridge is None else d_bridge
        self.d_bridge = d_bridge

        self.spike_bridge = SpikeToRate()
        self.ms_bridge = MRS3ABridge(c1, c2, c3, d_bridge, c4=c4)
        self.bridge_norm = nn.LayerNorm(d_bridge)

        self.alpha_p = nn.Parameter(torch.tensor(1.0))
        self.beta_p = nn.Parameter(torch.tensor(1.0))
        self.A_t_proj = nn.Linear(c3 if c4 is None else c4, d_bridge)

        if pos_enc == "fourier":
            self.pe_proj = nn.Linear(4 * pe_bands, d_bridge)
            nn.init.zeros_(self.pe_proj.weight)
            nn.init.zeros_(self.pe_proj.bias)
            self.register_buffer(
                "pe_freqs", 2.0 ** torch.arange(pe_bands).float() * math.pi, persistent=False
            )

        self.hier_stack = HierarchicalMambaStack(
            d_in=d_bridge, T=T, k_in=top_k, d_stages=hier_dims,
            pool_ratios=hier_pool_ratios, drop_path_rate=drop_path_rate,
        )
        if head_dim and head_dim > hier_dims[-1]:
            self.classifier = nn.Sequential(
                nn.Linear(hier_dims[-1], head_dim),
                nn.LayerNorm(head_dim),
                nn.GELU(),
                nn.Linear(head_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(hier_dims[-1], num_classes)

    def set_alpha(self, alpha: float):
        """Set the surrogate-gradient sharpness on every spiking neuron."""
        for m in self.modules():
            if isinstance(m, SPIKING_NEURONS):
                m.alpha = alpha

    def fuse_model(self):
        """Fuse every reparameterized block. Call once before inference."""
        for m in self.modules():
            if isinstance(m, RepConvBlock):
                m.fuse()

    def get_spike_rates(self):
        """Return the per-neuron firing-rate EMAs, keyed by module name."""
        return {name: m.spike_rate for name, m in self.named_modules()
                if isinstance(m, SPIKING_NEURONS)}

    def _positional_encoding(self, idx, H_s, W_s, dtype):
        gy = (idx // W_s).float() / max(1, H_s - 1) * 2.0 - 1.0
        gx = (idx % W_s).float() / max(1, W_s - 1) * 2.0 - 1.0
        f = self.pe_freqs.view(1, 1, -1)
        feats = torch.cat([
            torch.sin(gx.unsqueeze(-1) * f), torch.cos(gx.unsqueeze(-1) * f),
            torch.sin(gy.unsqueeze(-1) * f), torch.cos(gy.unsqueeze(-1) * f),
        ], dim=-1)
        return self.pe_proj(feats.to(dtype))

    def _forward_pass(self, x, continuous_mode=False):
        B, T = x.shape[0], x.shape[1]
        x = self.entry_bntt(x)

        s1, _, _, r1 = self.stage1(x, continuous_mode)
        s2, _, _, r2 = self.stage2(s1, continuous_mode)
        s3, _, a3, r3 = self.stage3(s2, continuous_mode)

        if self.n_stages == 4:
            s4, _, a4, r4 = self.stage4(s3, continuous_mode)
            spike_rate = (r1 + r2 + r3 + r4) / 8.0
            s_last, a_last = s4, a4
        else:
            s4 = None
            spike_rate = (r1 + r2 + r3) / 6.0
            s_last, a_last = s3, a3

        f1, f2, f3 = self.spike_bridge(s1), self.spike_bridge(s2), self.spike_bridge(s3)
        f4 = self.spike_bridge(s4) if s4 is not None else None
        tokens = self.ms_bridge(f1, f2, f3, f4)
        H_s, W_s = tokens.shape[-2], tokens.shape[-1]

        scores = spike_latency_scores(s_last, self.alpha_p, self.beta_p)
        idx, k = select_tokens(scores, self.top_k, H_s * W_s)

        seq = gather_tokens(tokens.view(B, T, tokens.shape[2], H_s * W_s), idx, k)
        seq = self.bridge_norm(seq)

        attn = gather_tokens(a_last.view(B, T, a_last.shape[2], H_s * W_s), idx, k)
        seq = seq + self.A_t_proj(attn)

        if self.pos_enc == "fourier":
            pe = self._positional_encoding(idx, H_s, W_s, seq.dtype)
            d = seq.shape[-1]
            seq = seq + pe.unsqueeze(1).expand(B, T, k, d).reshape(B, T * k, d)

        h = self.hier_stack(seq)
        final_k = h.shape[1] // T
        out = h.view(B, T, final_k, h.shape[2]).mean(dim=2)
        logits = self.classifier(out)

        if self.training and not continuous_mode:
            return logits, spike_rate
        return logits

    def forward(self, x):
        if not self.training:
            return self._forward_pass(x, continuous_mode=False)

        logits_d, spike_rate = self._forward_pass(x, continuous_mode=False)
        if not self.sgc:
            return logits_d, logits_d.detach(), spike_rate

        bns = [m for m in self.modules() if isinstance(m, nn.BatchNorm2d)]
        momentums = [m.momentum for m in bns]
        for m in bns:
            m.momentum = 0
        try:
            logits_c = self._forward_pass(x, continuous_mode=True)
        finally:
            for m, momentum in zip(bns, momentums):
                m.momentum = momentum
        return logits_d, logits_c, spike_rate
