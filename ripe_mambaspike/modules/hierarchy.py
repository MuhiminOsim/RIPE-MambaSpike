"""Hierarchical Mamba stack: narrow and long, then wide and short."""

import torch
import torch.nn as nn

from .mamba import GatedBidirectionalMambaBlock

__all__ = ["TokenPoolProject", "HierarchicalMambaStack"]


class TokenPoolProject(nn.Module):
    """Transition between hierarchy levels.

    Pools adjacent tokens along the spatial axis of the flattened ``T * k``
    sequence, then projects the channel dimension. Pooling happens within a
    timestep, never across timesteps, so temporal structure survives the
    transition intact.

    Args:
        d_in: input width.
        d_out: output width.
        T: number of timesteps, used to locate the spatial axis.
        pool_ratio: how many adjacent tokens each output token covers.

    Shape: ``(B, T * k, d_in)`` in, ``(B, T * k / pool_ratio, d_out)`` out.
    """

    def __init__(self, d_in, d_out, T, pool_ratio):
        super().__init__()
        self.T = T
        self.pool_ratio = pool_ratio
        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        B, L, d = x.shape
        k = L // self.T

        remainder = k % self.pool_ratio
        if remainder != 0:
            k_trim = k - remainder
            x = x.view(B, self.T, k, d)[:, :, :k_trim, :].reshape(B, self.T * k_trim, d)
            k = k_trim

        x = x.view(B, self.T, k // self.pool_ratio, self.pool_ratio, d).mean(dim=3)
        x = x.reshape(B, self.T * (k // self.pool_ratio), d)
        return self.norm(self.proj(x))


class HierarchicalMambaStack(nn.Module):
    """Stack of bidirectional scans over a shrinking token budget.

    Each level halves (or more) the token count and widens the channel
    dimension, so the deep levels see a short sequence at full width and the
    shallow ones see a long sequence cheaply. The parameter count is set by
    ``d_stages`` alone, never by the input resolution.

    Args:
        d_in: width of the bridged tokens.
        T: number of timesteps.
        k_in: retained tokens per timestep entering the stack.
        d_stages: width at each level.
        pool_ratios: pooling applied before each level; the first must be 1.
        drop_path_rate: maximum stochastic-depth rate, staged linearly.
    """

    def __init__(self, d_in, T, k_in, d_stages=(32, 64, 128),
                 pool_ratios=(1, 2, 2), drop_path_rate=0.1):
        super().__init__()
        if len(d_stages) != len(pool_ratios):
            raise ValueError("d_stages and pool_ratios must have the same length")
        if pool_ratios[0] != 1:
            raise ValueError("pool_ratios[0] must be 1")

        self.T = T
        self.k_in = k_in
        self.d_stages = tuple(d_stages)
        self.pool_ratios = tuple(pool_ratios)

        self.input_proj = nn.Identity() if d_in == d_stages[0] else nn.Linear(d_in, d_stages[0])
        self.input_norm = nn.LayerNorm(d_stages[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(d_stages))]
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        current_k = k_in
        for i, d in enumerate(d_stages):
            if i == 0:
                self.transitions.append(nn.Identity())
            else:
                self.transitions.append(TokenPoolProject(d_stages[i - 1], d, T, pool_ratios[i]))
                current_k //= pool_ratios[i]
            self.blocks.append(GatedBidirectionalMambaBlock(d, drop_path=dpr[i]))

        self.final_k = current_k
        self.d_final = d_stages[-1]

    def forward(self, x):
        x = self.input_norm(self.input_proj(x))
        for transition, block in zip(self.transitions, self.blocks):
            x = block(transition(x))
        return x
