"""Surrogate gradients for the non-differentiable spike function."""

import math
import torch

__all__ = ["ATanSurrogate", "heaviside"]


class ATanSurrogate(torch.autograd.Function):
    """Heaviside forward, arctangent-derivative backward.

    The backward pass substitutes ``d/dx arctan(pi * alpha * x / 2)`` for the
    Dirac delta, giving a bounded, non-vanishing gradient around threshold.
    """

    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).float()

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = alpha / (2.0 * (1.0 + (math.pi * alpha * x / 2.0).pow(2)))
        return grad_output * grad, None


def heaviside(x, alpha: float = 2.0):
    """Spike function with an ATan surrogate gradient."""
    return ATanSurrogate.apply(x, alpha)
