"""Temporal Decoupled Modulation (TDM)."""

import torch

__all__ = ["temporal_decoupled_modulation", "shift_prev"]


def shift_prev(x_seq):
    """Return ``x`` delayed by one step along time, zero-padded at ``t = 0``.

    Shape: ``(B, T, C, H, W)`` in and out.
    """
    return torch.cat([torch.zeros_like(x_seq[:, :1]), x_seq[:, :-1]], dim=1)


def temporal_decoupled_modulation(x_seq, alpha):
    """Per-channel first-order temporal filter.

    Computes ``(1 + alpha) * x_t - alpha * x_{t-1}``, one learnable scalar
    ``alpha`` per input channel. At ``alpha = 0`` the layer is the identity,
    so it can be inserted anywhere without changing the function at
    initialization; as ``alpha`` grows the layer emphasizes the discrete
    temporal derivative and suppresses the static component.

    Being linear, it commutes with the convolution that follows it, which is
    what allows the dual-kernel accumulate-only reparameterization in
    :meth:`~ripe_mambaspike.layers.reparam.RepConvBlock.fuse_dual_kernel`.

    Args:
        x_seq: ``(B, T, C, H, W)``.
        alpha: ``(C,)`` per-channel coefficients.
    """
    a = alpha.view(1, 1, -1, 1, 1)
    return (1.0 + a) * x_seq - a * shift_prev(x_seq)
