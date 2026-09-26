"""Spike-latency token selection."""

import torch

__all__ = ["spike_latency_scores", "select_tokens"]


def spike_latency_scores(spikes, alpha_p, beta_p):
    """Rank spatial positions by activity and by how early they first fire.

    The score is ``alpha_p * (total spikes) + beta_p * exp(-t_first)``, so a
    position is salient if it fires a lot, fires early, or both. Positions
    that never fire receive the smallest possible latency term.

    Args:
        spikes: ``(B, T, C, H, W)``.
        alpha_p, beta_p: scalar weights.

    Returns:
        ``(B, H * W)`` scores.
    """
    B, T = spikes.shape[0], spikes.shape[1]
    total = spikes.sum(dim=(1, 2))
    fired = (spikes.sum(dim=2) > 0).float()
    t_idx = torch.arange(1, T + 1, device=spikes.device).view(1, T, 1, 1).float()
    t_first = (fired * t_idx).masked_fill(fired == 0, 999.0).min(dim=1)[0]
    scores = alpha_p * total + beta_p * torch.exp(-t_first)
    return scores.view(B, -1)


def select_tokens(scores, top_k, n_positions):
    """Return sorted indices of the ``top_k`` highest-scoring positions.

    ``top_k`` of ``0`` or ``None`` keeps every position. Indices are sorted so
    the scan still sees tokens in spatial order after selection.
    """
    k = n_positions if top_k in (0, None) else min(top_k, n_positions)
    _, idx = torch.topk(scores, k, dim=1)
    idx, _ = torch.sort(idx, dim=1)
    return idx, k


def gather_tokens(x, idx, k):
    """Gather ``(B, T, C, N)`` at ``idx`` and flatten to ``(B, T * k, C)``."""
    B, T, C = x.shape[0], x.shape[1], x.shape[2]
    expand = idx.view(B, 1, 1, k).expand(B, T, C, k)
    out = torch.gather(x, 3, expand)
    return out.permute(0, 1, 3, 2).reshape(B, T * k, C)
