"""BatchNorm recalibration on clean data.

Training runs under Mixup, CutMix and NDA, so the BatchNorm running statistics
converge to the distribution of augmented batches, which is not the
distribution seen at test time. Before evaluating, the statistics are
re-estimated from a small number of un-augmented training batches and then
restored, leaving the training trajectory untouched.

This reads only the training split. No test data is used, so it is a
calibration step rather than test-time adaptation.
"""

import torch
import torch.nn as nn

__all__ = ["bn_modules", "snapshot_bn", "restore_bn", "recalibrate_bn"]


def bn_modules(model):
    """Every ``BatchNorm2d`` in ``model``."""
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


def snapshot_bn(model):
    """Copy the running statistics so they can be restored afterwards."""
    return {
        name: buf.detach().clone()
        for name, buf in model.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }


def restore_bn(model, snapshot):
    """Write a snapshot taken by :func:`snapshot_bn` back into ``model``."""
    buffers = dict(model.named_buffers())
    for name, value in snapshot.items():
        if name in buffers:
            buffers[name].copy_(value)


@torch.no_grad()
def recalibrate_bn(model, loader, device, n_batches: int = 50, amp: bool = True):
    """Re-estimate BatchNorm statistics from ``n_batches`` clean batches.

    ``momentum=None`` makes each layer accumulate a cumulative average over the
    batches seen, rather than an exponential one, so the estimate does not
    depend on batch order. The module is left in eval mode.
    """
    mods = bn_modules(model)
    for m in mods:
        m.reset_running_stats()
        m.momentum = None

    model.train()
    autocast = torch.amp.autocast(
        "cuda", enabled=amp and device.type == "cuda", dtype=torch.bfloat16
    )
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        with autocast:
            model._forward_pass(x.to(device, non_blocking=True), continuous_mode=False)

    for m in mods:
        m.momentum = 0.1
    model.eval()
