"""
Neuromorphic Data Augmentation (NDA) for RepMambaSNN-V5
========================================================
Adapted from Li, Kim, Panda — "Neuromorphic Data Augmentation for Training
Spiking Neural Networks", ECCV 2022 — to operate directly on the
(T, 2, H, W) event-frame tensor produced by your CIFARDVSDataset.

THREE CRITICAL DESIGN DECISIONS (each fixes a common failure mode)
------------------------------------------------------------------
1. Same spatial parameters across all T frames. The op is sampled once per
   sample and applied identically to every timestep, preserving temporal
   coherence so the network can still learn motion across T.
2. Same parameters applied to both polarity channels. ON/OFF channels are
   geometrically locked, so the event-correlation structure stays intact.
3. Roll (cyclic shift) instead of zero-pad translate. Avoids the
   "no-events border" artifact that the network would otherwise learn as a
   positional cue.

USAGE
-----
    from nda import NDA, polarity_flip, time_reversal

    # in CIFARDVSDataset.__init__:
    self.nda = NDA(n_ops=2, magnitude=0.5) if is_train else None

    # in CIFARDVSDataset.__getitem__ (after temporal handling):
    if self.is_train:
        if random.random() > 0.5: data = torch.flip(data, dims=[3])  # H-flip
        if random.random() > 0.5: data = polarity_flip(data)
        if random.random() > 0.5: data = time_reversal(data)
        data = self.nda(data)
"""

import math
import random
import torch
import torch.nn.functional as F


class NDA:
    """
    RandAugment-style policy for (T, 2, H, W) event-frame tensors.

    Args
    ----
    n_ops : int
        Number of operations sampled per call (default 2, matches NDA paper).
    magnitude : float in [0, 1]
        Global magnitude scale. 0.5 is a safe starting point for 128x128.
        If accuracy regresses early in training, drop to 0.3. If overfitting
        persists, raise to 0.7.
    """

    def __init__(self, n_ops=2, magnitude=0.5,
                 shear_max=0.3, roll_max_frac=0.25,
                 rotate_max_deg=15.0,
                 cutout_max_frac=0.25, cutout_t_max=2):
        self.n_ops = n_ops
        self.magnitude = magnitude
        self.shear_max = shear_max
        self.roll_max_frac = roll_max_frac
        self.rotate_max_deg = rotate_max_deg
        self.cutout_max_frac = cutout_max_frac
        self.cutout_t_max = cutout_t_max

        self.ops = [
            self._identity,
            self._shear_x,
            self._shear_y,
            self._roll_x,
            self._roll_y,
            self._rotate,
            self._cutout_spatial,
            self._cutout_temporal,
        ]

    def __call__(self, x):
        """
        x : (T, 2, H, W) float tensor
        returns : same shape, same dtype, same device
        """
        assert x.ndim == 4, f"Expected (T, C, H, W), got {tuple(x.shape)}"
        for _ in range(self.n_ops):
            op = random.choice(self.ops)
            x = op(x)
        return x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_affine(self, x, theta_2x3):
        """
        Apply a single 2x3 affine to every (T, C) slice with SHARED params.
        Uses grid_sample with zeros padding for affine ops; cutout/roll
        bypass grid_sample to keep event density correct.
        """
        T, C, H, W = x.shape
        x_flat = x.reshape(T * C, 1, H, W)
        theta = theta_2x3.to(x.dtype).to(x.device)
        theta = theta.unsqueeze(0).expand(T * C, -1, -1).contiguous()
        grid = F.affine_grid(theta, list(x_flat.shape), align_corners=False)
        out = F.grid_sample(x_flat, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=False)
        return out.reshape(T, C, H, W)

    def _sample_signed(self, max_val):
        """Uniform sample in [-max_val * magnitude, +max_val * magnitude]."""
        return (random.random() * 2 - 1) * max_val * self.magnitude

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _identity(self, x):
        return x

    def _shear_x(self, x):
        s = self._sample_signed(self.shear_max)
        theta = torch.tensor([[1.0, s, 0.0],
                              [0.0, 1.0, 0.0]])
        return self._apply_affine(x, theta)

    def _shear_y(self, x):
        s = self._sample_signed(self.shear_max)
        theta = torch.tensor([[1.0, 0.0, 0.0],
                              [s,   1.0, 0.0]])
        return self._apply_affine(x, theta)

    def _roll_x(self, x):
        # Cyclic shift along W; preserves total event count exactly.
        _, _, _, W = x.shape
        shift = int(self._sample_signed(self.roll_max_frac * W))
        return torch.roll(x, shifts=shift, dims=3)

    def _roll_y(self, x):
        _, _, H, _ = x.shape
        shift = int(self._sample_signed(self.roll_max_frac * H))
        return torch.roll(x, shifts=shift, dims=2)

    def _rotate(self, x):
        deg = self._sample_signed(self.rotate_max_deg)
        rad = math.radians(deg)
        cos, sin = math.cos(rad), math.sin(rad)
        theta = torch.tensor([[cos, -sin, 0.0],
                              [sin,  cos, 0.0]])
        return self._apply_affine(x, theta)

    def _cutout_spatial(self, x):
        # Mask one rectangle to zero. Same rectangle across (T, C).
        T, C, H, W = x.shape
        max_h = max(8, int(self.cutout_max_frac * H * self.magnitude))
        max_w = max(8, int(self.cutout_max_frac * W * self.magnitude))
        h_size = random.randint(8, max_h)
        w_size = random.randint(8, max_w)
        h_start = random.randint(0, H - h_size)
        w_start = random.randint(0, W - w_size)
        x = x.clone()
        x[:, :, h_start:h_start + h_size, w_start:w_start + w_size] = 0.0
        return x

    def _cutout_temporal(self, x):
        # Mask one or more contiguous timesteps to zero.
        T = x.shape[0]
        max_t = max(1, int(self.cutout_t_max * self.magnitude))
        if max_t >= T:
            return x
        t_size = random.randint(1, max_t)
        t_start = random.randint(0, T - t_size)
        x = x.clone()
        x[t_start:t_start + t_size] = 0.0
        return x


# ----------------------------------------------------------------------
# Standalone label-preserving symmetries for event data
# ----------------------------------------------------------------------

def polarity_flip(x):
    """Swap ON/OFF event channels: x[:, 0] <-> x[:, 1]. Free symmetry on
    static-image-on-panning-monitor datasets like CIFAR10-DVS."""
    return x.flip(1)


def time_reversal(x):
    """Reverse temporal order. On CIFAR10-DVS this is also a near-symmetry
    because the recordings are static images under a panning DVS — direction
    of pan is nuisance variation."""
    return x.flip(0)


# ----------------------------------------------------------------------
# Sanity check / visualization
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Synthesize a fake event tensor with known event positions.
    T, C, H, W = 10, 2, 128, 128
    x = torch.zeros(T, C, H, W)
    x[3, 0, 60:70, 60:70] = 1.0   # ON events at frame 3, center
    x[5, 1, 30:40, 30:40] = 1.0   # OFF events at frame 5, top-left

    n_events_in = x.sum().item()
    print(f"Input  : shape={tuple(x.shape)}, events={n_events_in:.0f}")

    nda = NDA(n_ops=2, magnitude=0.5)

    for trial in range(5):
        x_aug = nda(x)
        n_events_out = x_aug.sum().item()
        # Roll preserves count exactly. Affine ops use bilinear interp
        # so event mass is conserved up to small interpolation drift.
        # Cutout reduces count.
        print(f"Trial {trial}: shape={tuple(x_aug.shape)}, "
              f"events={n_events_out:.1f}, "
              f"min={x_aug.min().item():.3f}, "
              f"max={x_aug.max().item():.3f}")
        assert x_aug.shape == x.shape, f"Shape changed: {x_aug.shape}"

    # Verify temporal consistency: with fixed seed, the same sample should
    # produce the same op trajectory.
    print("\nTemporal-consistency check (same op should be applied to all T):")
    random.seed(0)
    nda2 = NDA(n_ops=1, magnitude=1.0)
    # Force a roll_x op for clarity:
    nda2.ops = [nda2._roll_x]
    x_rolled = nda2(x)
    # Find shift in W: argmax along W of frame 3, channel 0
    orig_col = x[3, 0].sum(dim=0).argmax().item()
    new_col  = x_rolled[3, 0].sum(dim=0).argmax().item()
    shift_f3 = (new_col - orig_col) % W
    orig_col5 = x[5, 1].sum(dim=0).argmax().item()
    new_col5  = x_rolled[5, 1].sum(dim=0).argmax().item()
    shift_f5  = (new_col5 - orig_col5) % W
    print(f"  Frame 3 shift: {shift_f3}, Frame 5 shift: {shift_f5}")
    assert shift_f3 == shift_f5, "Temporal coherence broken!"
    print("  PASS: same shift across timesteps.")

    # Polarity flip / time reversal sanity
    x_pflip = polarity_flip(x)
    assert torch.allclose(x_pflip[:, 0], x[:, 1])
    assert torch.allclose(x_pflip[:, 1], x[:, 0])
    print("Polarity flip: PASS")

    x_rev = time_reversal(x)
    assert torch.allclose(x_rev[0], x[T-1])
    print("Time reversal: PASS")

    print("\nNDA self-test passed.")
