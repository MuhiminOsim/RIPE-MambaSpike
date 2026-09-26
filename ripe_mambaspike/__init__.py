"""RIPE-MambaSpike.

Resolution-independent spiking/state-space interfaces for parameter-efficient
event-based vision.

Layout:
    ``layers``   primitive building blocks (neurons, surrogates, norms,
                 reparameterized convolutions, linear attention)
    ``modules``  architectural components (TDM, front-end stage, MR-S3A
                 bridge, token pruning, bidirectional scan, hierarchy)
    ``models``   the full network and per-benchmark builders
    ``data``     event datasets and neuromorphic augmentation
    ``engine``   losses, schedules, optimizer, checkpointing, training loop
    ``utils``    seeding and parameter accounting
"""

__version__ = "1.0.0"

from .models import RIPEMambaSpike, build_model, list_benchmarks

__all__ = ["RIPEMambaSpike", "build_model", "list_benchmarks", "__version__"]
