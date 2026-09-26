"""Architectural modules: TDM, front-end, bridge, pruning, sequence stack."""

from .bridge import MRS3ABridge, SpikeToRate
from .frontend import SpikingRepStage
from .hierarchy import HierarchicalMambaStack, TokenPoolProject
from .mamba import GatedBidirectionalMambaBlock
from .pruning import gather_tokens, select_tokens, spike_latency_scores
from .tdm import shift_prev, temporal_decoupled_modulation

__all__ = [
    "temporal_decoupled_modulation", "shift_prev",
    "SpikingRepStage",
    "SpikeToRate", "MRS3ABridge",
    "spike_latency_scores", "select_tokens", "gather_tokens",
    "GatedBidirectionalMambaBlock",
    "TokenPoolProject", "HierarchicalMambaStack",
]
