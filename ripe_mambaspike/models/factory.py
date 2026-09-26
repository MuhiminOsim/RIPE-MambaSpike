"""Per-benchmark model builders.

Every entry fixes only what the benchmark requires: the class count, the
number of timesteps, the retained-token budget and, for DailyDVS-200, a wider
front-end. The bridge and stack widths are shared, which is what keeps the
deployed parameter count flat across sensor resolutions.
"""

from .ripe_mambaspike import RIPEMambaSpike

__all__ = ["BENCHMARKS", "build_model", "list_benchmarks"]


BENCHMARKS = {
    "dvsgesture": dict(
        num_classes=11, stage_channels=(32, 64, 128), hier_dims=(32, 64, 128),
        hier_pool_ratios=(1, 2, 2), T=16, top_k=64, drop_path_rate=0.1,
    ),
    "cifar10dvs": dict(
        num_classes=10, stage_channels=(32, 64, 128), hier_dims=(32, 64, 128),
        hier_pool_ratios=(1, 2, 2), T=10, top_k=128, drop_path_rate=0.2,
    ),
    "ncaltech101": dict(
        num_classes=101, stage_channels=(32, 64, 128), hier_dims=(32, 64, 128),
        hier_pool_ratios=(1, 2, 2), T=10, top_k=128, drop_path_rate=0.2,
    ),
    "nmnist": dict(
        num_classes=10, stage_channels=(32, 64, 128), hier_dims=(32, 64, 128),
        hier_pool_ratios=(1, 2, 2), T=10, top_k=64, drop_path_rate=0.1,
    ),
    "dailydvs200": dict(
        num_classes=200, stage_channels=(96, 192, 384), hier_dims=(96, 192, 384),
        hier_pool_ratios=(1, 2, 2), T=10, top_k=0, drop_path_rate=0.2,
        head_dim=1024,
    ),
    "dailydvs200_lite": dict(
        num_classes=200, stage_channels=(64, 128, 256), hier_dims=(64, 128, 256),
        hier_pool_ratios=(1, 2, 2), T=10, top_k=0, drop_path_rate=0.2,
        head_dim=1024,
    ),
}


def list_benchmarks():
    """Names accepted by :func:`build_model`."""
    return sorted(BENCHMARKS)


def build_model(benchmark: str, **overrides) -> RIPEMambaSpike:
    """Build the model configured for ``benchmark``.

    Keyword overrides are applied on top of the stored configuration, so a
    training script can change ``T`` or ``top_k`` without restating the rest.
    """
    if benchmark not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {benchmark!r}; choose from {list_benchmarks()}")
    cfg = dict(BENCHMARKS[benchmark])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return RIPEMambaSpike(**cfg)
