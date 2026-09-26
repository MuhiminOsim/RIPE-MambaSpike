"""Training objectives."""

import torch
import torch.nn.functional as F

__all__ = ["soft_cross_entropy", "tet_loss", "sgc_loss"]


def soft_cross_entropy(pred, target):
    """Cross-entropy that accepts either hard labels or a soft distribution."""
    if target.ndim == 1:
        return F.cross_entropy(pred, target)
    return (-target * F.log_softmax(pred, dim=-1)).sum(-1).mean()


def tet_loss(logits, targets, lamb: float = 1e-2, criterion=None):
    """Temporal Efficient Training loss.

    Applies the classification loss at every timestep rather than to the
    time-averaged logits, so no step is free to drift, and adds an MSE term
    pulling each step toward the temporal mean to keep the trajectory stable.

    Args:
        logits: ``(B, T, num_classes)``.
        targets: hard labels ``(B,)`` or soft targets ``(B, num_classes)``.
        lamb: weight of the temporal-consistency term.
        criterion: per-step loss; defaults to :func:`soft_cross_entropy`.
    """
    criterion = soft_cross_entropy if criterion is None else criterion
    T = logits.shape[1]
    loss_ce = sum(criterion(logits[:, t], targets) for t in range(T)) / T
    mean_logits = logits.mean(dim=1, keepdim=True)
    loss_mse = F.mse_loss(logits, mean_logits.expand_as(logits))
    return loss_ce + lamb * loss_mse


def sgc_loss(logits_discrete, logits_continuous):
    """Surrogate-gradient consistency between the spiking and relaxed passes."""
    return F.mse_loss(logits_discrete, logits_continuous)
