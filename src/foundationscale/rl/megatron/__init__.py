"""Megatron-Core RL lane: rungs 0-2 public surface.

This package is importable without Megatron-Core, Megatron-Bridge, or a GPU.
The stable rungs 0-1 pieces -- vocab-parallel token logprobs, global loss
denominators, and lane config refusal rules -- are re-exported eagerly because
they are pure torch/python. Pipeline wiring (``pp_step``), the torchrun
driver (``driver``) and rung 2's rollout helpers (``online``) are intentionally
lazy: they import megatron/bridge/transformers only inside functions, and refuse
by name when touched without them.

WHAT IS CLAIMED: importing this package never initializes distributed and
never imports megatron. WHAT IS NOT CLAIMED: that the pp/driver entrypoints
exist in the rungs 0-1 slice.
"""

from __future__ import annotations

from foundationscale.rl.megatron.lane_config import (
    TE_CP_SUPPORTED,
    MegatronLaneConfig,
)
from foundationscale.rl.megatron.logprobs import (
    check_softcap_owner,
    check_vocab_shard,
    softcap,
    vocab_parallel_token_logprobs,
)
from foundationscale.rl.megatron.normalization import (
    GlobalDenominators,
    compute_denominators,
    normalized_loss,
)

__all__ = (
    "GlobalDenominators",
    "MegatronLaneConfig",
    "TE_CP_SUPPORTED",
    "check_softcap_owner",
    "check_vocab_shard",
    "compute_denominators",
    "normalized_loss",
    "softcap",
    "vocab_parallel_token_logprobs",
)
