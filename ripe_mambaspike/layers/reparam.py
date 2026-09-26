"""Structural reparameterization.

:class:`RepConvBlock` trains as a seven-branch residual convolution and
collapses at inference to a single ``3x3`` convolution with bias, so the
multi-branch capacity costs nothing at deployment.

Branches: ``3x3``, ``1x1``, ``1x3``, ``3x1``, a ``1x1 -> 3x3`` sequence, and,
when the shapes allow it, identity and average-pool branches. Each carries its
own BatchNorm, which is absorbed into the fused kernel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RepConvBlock"]


class RepConvBlock(nn.Module):
    """Multi-branch convolution that fuses to one ``3x3`` kernel.

    Args:
        in_channels: input channel count.
        out_channels: output channel count.
        stride: spatial stride, shared by every branch.

    Call :meth:`fuse` before inference. It is an explicit step, never
    automatic, so a training graph is never silently replaced.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.is_fused = False

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn3x3 = nn.BatchNorm2d(out_channels)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, (1, 3), stride=stride, padding=(0, 1), bias=False)
        self.bn1x3 = nn.BatchNorm2d(out_channels)
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, (3, 1), stride=stride, padding=(1, 0), bias=False)
        self.bn3x1 = nn.BatchNorm2d(out_channels)
        self.conv1x1_seq = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)
        self.conv3x3_seq = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn_seq = nn.BatchNorm2d(out_channels)

        self.has_identity = in_channels == out_channels and stride == 1
        if self.has_identity:
            self.bn_id = nn.BatchNorm2d(out_channels)
            self.avgpool = nn.AvgPool2d(3, stride=stride, padding=1)
            self.bn_avg = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.is_fused:
            return self.fused_conv(x)
        out = self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x))
        out = out + self.bn1x3(self.conv1x3(x)) + self.bn3x1(self.conv3x1(x))
        out = out + self.bn_seq(self.conv3x3_seq(self.conv1x1_seq(x)))
        if self.has_identity:
            out = out + self.bn_id(x) + self.bn_avg(self.avgpool(x))
        return out

    @staticmethod
    def _absorb_bn(weight, bn):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return weight * scale.reshape(-1, 1, 1, 1), -bn.running_mean * scale + bn.bias

    def _fuse_conv_bn(self, conv, bn):
        return self._absorb_bn(conv.weight, bn)

    @torch.no_grad()
    def fuse(self):
        """Collapse every branch into ``self.fused_conv`` and drop the rest."""
        if self.is_fused:
            return
        w3, b3 = self._fuse_conv_bn(self.conv3x3, self.bn3x3)
        w1, b1 = self._fuse_conv_bn(self.conv1x1, self.bn1x1)
        w13, b13 = self._fuse_conv_bn(self.conv1x3, self.bn1x3)
        w31, b31 = self._fuse_conv_bn(self.conv3x1, self.bn3x1)

        w_seq_merged = F.conv2d(self.conv3x3_seq.weight, self.conv1x1_seq.weight.permute(1, 0, 2, 3))
        w_seq, b_seq = self._absorb_bn(w_seq_merged, self.bn_seq)

        w1 = F.pad(w1, [1, 1, 1, 1])
        w13 = F.pad(w13, [0, 0, 1, 1])
        w31 = F.pad(w31, [1, 1, 0, 0])

        w_fused = w3 + w1 + w13 + w31 + w_seq
        b_fused = b3 + b1 + b13 + b31 + b_seq

        if self.has_identity:
            id_w = torch.zeros_like(w3)
            for i in range(self.in_channels):
                id_w[i, i, 1, 1] = 1.0
            w_id, b_id = self._absorb_bn(id_w, self.bn_id)
            avg_w = torch.zeros_like(w3)
            for i in range(self.out_channels):
                avg_w[i, i, :, :] = 1.0 / 9.0
            w_avg, b_avg = self._absorb_bn(avg_w, self.bn_avg)
            w_fused = w_fused + w_id + w_avg
            b_fused = b_fused + b_id + b_avg

        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, 3, stride=self.stride, padding=1, bias=True
        ).to(w_fused.device)
        self.fused_conv.weight.data = w_fused
        self.fused_conv.bias.data = b_fused
        self.is_fused = True

        for attr in ("conv3x3", "bn3x3", "conv1x1", "bn1x1", "conv1x3", "bn1x3",
                     "conv3x1", "bn3x1", "conv1x1_seq", "conv3x3_seq", "bn_seq",
                     "bn_id", "avgpool", "bn_avg"):
            if hasattr(self, attr):
                delattr(self, attr)

    @torch.no_grad()
    def fuse_dual_kernel(self, alpha):
        """Split the fused kernel for accumulate-only inference.

        Given the paired TDM coefficient ``alpha``, writes

            ``W_a = W* . diag(1 + alpha)``   ``W_b = W* . diag(alpha)``

        so that ``fused_conv(TDM(x_t, x_prev)) == conv(W_a, x_t) - conv(W_b, x_prev) + b*``.
        Both operands stay binary, so one dense multiply-accumulate pass over a
        four-valued TDM output becomes two accumulate-only passes.
        Requires :meth:`fuse` first.
        """
        if not self.is_fused:
            raise RuntimeError("fuse() must be called before fuse_dual_kernel()")
        w = self.fused_conv.weight
        a = alpha.reshape(1, -1, 1, 1).to(device=w.device, dtype=w.dtype)
        self.register_buffer("w_a", w * (1.0 + a), persistent=False)
        self.register_buffer("w_b", w * a, persistent=False)
        self.dual_kernel_ready = True

    def forward_ac(self, x_t, x_prev):
        """Accumulate-only forward. ``x_prev`` must be zeros at ``t = 0``."""
        if not getattr(self, "dual_kernel_ready", False):
            raise RuntimeError("fuse_dual_kernel() must be called before forward_ac()")
        out_t = F.conv2d(x_t, self.w_a, bias=None, stride=self.stride, padding=1)
        out_prev = F.conv2d(x_prev, self.w_b, bias=None, stride=self.stride, padding=1)
        return out_t - out_prev + self.fused_conv.bias.view(1, -1, 1, 1)
