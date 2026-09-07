"""Reinforcement-learning contracts (Phase 3, stage 1).

Stage 1 of the staging in ``validation_campaigns/nemo_rl_baseline/PHASE3_DESIGN.md``
section 9: the data contract, the loss contract with supervised fine-tuning only,
and the bridge that turns one measured step into an ``ObjectiveGateContext``.

The remaining contracts named in design section 3 -- ``Algorithm``,
``RolloutSource``, ``AdvantageFn``, ``PolicyPair``, ``WeightSync`` -- are later
stages and are deliberately absent. Nothing here has been benchmarked against the
reference implementation; design section 10 states what is not established.
"""

from __future__ import annotations

from foundationscale.rl.interfaces import (
    BatchRefusal,
    ExperienceBatch,
    ForwardFn,
    LossFn,
    LossOutput,
    SFTLoss,
    SupervisionRefusal,
    build_objective_gate_context,
)

__all__ = (
    "BatchRefusal",
    "ExperienceBatch",
    "ForwardFn",
    "LossFn",
    "LossOutput",
    "SFTLoss",
    "SupervisionRefusal",
    "build_objective_gate_context",
)
