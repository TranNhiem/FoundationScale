"""Reinforcement-learning contracts (Phase 3, stages 1-2).

Design section 9 of ``validation_campaigns/nemo_rl_baseline/PHASE3_DESIGN.md``:

* stage 1 -- the data contract, the loss contract with supervised fine-tuning
  only, and the bridge that turns one measured step into an
  ``ObjectiveGateContext``;
* stage 2 -- ``PolicyPair``'s reference role and ``DPOLoss``, carrying the two
  declaration conditions the section-7 item-4 measurement attached to it;
* stage 3a -- ``RolloutSource`` and ``AdvantageFn``, the two contracts that
  need no distributed measurement to specify.

Section 3's remaining contracts -- ``Algorithm`` and ``WeightSync`` -- are
stage 3b and deliberately absent, because both are shaped by the section-7
item-3 weight-transfer measurement, which has not been taken. Writing them
first would be asserting a cost model rather than measuring one. Nothing here
has been benchmarked against the reference implementation; design section 10
states what is not established.

The public import path is this package. ``SFTLoss`` moved from ``interfaces``
to ``losses`` when stage 2 split contracts from implementations, and that move
is invisible from here on purpose.
"""

from __future__ import annotations

from foundationscale.rl.advantage import (
    AdvantageConfigRefusal,
    AdvantageFn,
    AdvantageRefusal,
    AdvantageResult,
    GeneralisedAdvantageEstimation,
    GroupNormalisedAdvantage,
    LeaveOneOutAdvantage,
    RewardStats,
)
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
from foundationscale.rl.rollout import (
    CapabilityRefusal,
    RolloutSource,
    SourceCapabilities,
    check_capabilities,
    verify_generated,
)

__all__ = (
    "AdvantageConfigRefusal",
    "AdvantageFn",
    "AdvantageRefusal",
    "AdvantageResult",
    "BatchRefusal",
    "CapabilityRefusal",
    "DPOLoss",
    "ExperienceBatch",
    "ForwardFn",
    "GeneralisedAdvantageEstimation",
    "GroupNormalisedAdvantage",
    "LeaveOneOutAdvantage",
    "LossConfigRefusal",
    "LossDeclaration",
    "LossFn",
    "LossOutput",
    "PolicyPair",
    "PolicyRoleRefusal",
    "RewardStats",
    "RolloutSource",
    "SFTLoss",
    "SourceCapabilities",
    "SupervisionRefusal",
    "build_objective_gate_context",
    "check_capabilities",
    "verify_generated",
)
