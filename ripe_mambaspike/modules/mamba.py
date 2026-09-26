"""Bidirectional selective state-space block."""

import torch
import torch.nn as nn
from timm.layers import DropPath

from mamba_ssm import Mamba

__all__ = ["GatedBidirectionalMambaBlock"]


class GatedBidirectionalMambaBlock(nn.Module):
    """Pre-norm residual block with a forward and a reverse selective scan.

    Event streams are not causal in the way language is: evidence arriving
    late in a clip disambiguates what happened early. Running a second scan
    over the reversed sequence and merging the two by a learned linear map
    gives every position access to both directions at linear cost.

    Shape: ``(B, L, d_model)`` in and out.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.merge_proj = nn.Linear(2 * d_model, d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        normed = self.norm(x)
        y_fwd = self.mamba_fwd(normed)
        y_bwd = self.mamba_bwd(normed.flip(1)).flip(1)
        merged = self.merge_proj(torch.cat([y_fwd, y_bwd], dim=-1))
        return x + self.drop_path(merged)
