"""Reinforcement-learning contracts (Phase 3, stages 1-2).

Design section 9 of ``validation_campaigns/nemo_rl_baseline/PHASE3_DESIGN.md``:

* stage 1 -- the data contract, the loss contract with supervised fine-tuning
  only, and the bridge that turns one measured step into an
  ``ObjectiveGateContext``;
* stage 2 -- ``PolicyPair``'s reference role and ``DPOLoss``, carrying the two
  declaration conditions the section-7 item-4 measurement attached to it.

The remaining contracts named in design section 3 -- ``Algorithm``,
``RolloutSource``, ``AdvantageFn``, ``WeightSync`` -- are later stages and are
deliberately absent. Nothing here has been benchmarked against the reference
implementation; design section 10 states what is not established.

The public import path is this package. ``SFTLoss`` moved from ``interfaces``
to ``losses`` when stage 2 split contracts from implementations, and that move
is invisible from here on purpose.
"""

from __future__ import annotations

from foundationscale.rl.interfaces import (
    BatchRefusal,
    ExperienceBatch,
    ForwardFn,
    LossConfigRefusal,
    LossDeclaration,
    LossFn,
    LossOutput,
    SupervisionRefusal,
    build_objective_gate_context,
)
from foundationscale.rl.losses import DPOLoss, SFTLoss
from foundationscale.rl.policy import PolicyPair, PolicyRoleRefusal

__all__ = (
    "BatchRefusal",
    "DPOLoss",
    "ExperienceBatch",
    "ForwardFn",
    "LossConfigRefusal",
    "LossDeclaration",
    "LossFn",
    "LossOutput",
    "PolicyPair",
    "PolicyRoleRefusal",
    "SFTLoss",
    "SupervisionRefusal",
    "build_objective_gate_context",
)
