"""Dual-Stream Membrane-Potential Attention (DS-MPA)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DSMPA"]


class DSMPA(nn.Module):
    """Linear attention between the membrane potential and prior spikes.

    Queries and keys are read from the membrane potential ``U``, which is
    real-valued and therefore carries sub-threshold information that the spike
    train has discarded. Values are read from the previous step's spikes
    ``S_prev``, keeping the value stream binary.

    The feature map is ``phi(x) = elu(x) + 1``, which is strictly positive and
    non-saturating, so the normalizer cannot collapse the way a sigmoid gate
    does once it saturates. With ``phi >= 0`` the output of each row is a
    convex combination of the rows of ``V`` contracted toward zero, so the
    result stays inside the convex hull of its values.

    Cost is ``O(N * C^2)`` against ``O(N^2 * C)`` for softmax attention, which
    is what keeps the module affordable at the front-end's token counts.

    Shape:
        ``U``, ``S_prev``, ``X`` are ``(B, T, C, H, W)``; returns the modulated
        ``X`` and the attention map, both the same shape.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.W_Q = nn.Conv2d(channels, channels, 1, bias=False)
        self.W_K = nn.Conv2d(channels, channels, 1, bias=False)
        self.W_V = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.ones(1, 1, channels, 1, 1))

    def forward(self, U, S_prev, X):
        B, T, C, H, W = U.shape
        N = H * W

        U_flat = U.view(B * T, C, H, W)
        S_flat = S_prev.view(B * T, C, H, W)

        Q = self.W_Q(U_flat).view(B * T, C, N).permute(0, 2, 1)
        K = self.W_K(U_flat).view(B * T, C, N).permute(0, 2, 1)
        V = self.W_V(S_flat).view(B * T, C, N).permute(0, 2, 1)

        phi_Q = F.elu(Q) + 1.0
        phi_K = F.elu(K) + 1.0

        KV = torch.matmul(phi_K.permute(0, 2, 1), V)
        A = torch.matmul(phi_Q, KV)

        phi_K_sum = phi_K.sum(dim=1, keepdim=True)
        denom = torch.sum(phi_Q * phi_K_sum, dim=-1, keepdim=True)
        A = A / (denom + 1e-5)

        A = A.permute(0, 2, 1).view(B, T, C, H, W)
        return X + self.gamma * (A * X), A
