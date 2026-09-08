"""Reinforcement-learning contracts for Phase 3.

Design section 9 of ``validation_campaigns/nemo_rl_baseline/PHASE3_DESIGN.md``:

* stage 1 -- the data contract, the loss contract with supervised fine-tuning
  only, and the bridge that turns one measured step into an
  ``ObjectiveGateContext``;
* stage 2 -- ``PolicyPair``'s reference role and ``DPOLoss``, carrying the two
  declaration conditions the section-7 item-4 measurement attached to it;
* stage 3a -- ``RolloutSource`` and ``AdvantageFn``, the two contracts that
  need no distributed measurement to specify.

* stage 3b -- ``WeightSync``, the sibling edge that moves weights from the
  training view to the generation view.

``WeightSync`` could only be specified once the section-7 item-3
weight-transfer measurement was taken. It has been, and it returned
CLEAR_WITH_ABSTENTIONS: no single transport wins -- resharding and a
collective beat a full copy per byte at the large end, the full copy wins at
the small per-call floor, and one cell was withheld for spread over
tolerance. So the contract fixes no transport; the realization chooses one at
construction and the report evidences which one ran. Deliberately not
restated here: the per-transport ratios, which are a measurement artifact and
belong in the campaign record, not in a docstring that no gate re-measures.

Section 3's one remaining contract is ``Algorithm``, the policy-gradient
binding that consumes all of the above. Nothing here has been benchmarked
against the reference implementation; design section 10 states what is not
established.

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
from foundationscale.rl.weightsync import (
    SyncCapabilities,
    SyncCapabilityRefusal,
    SyncReport,
    SyncReportRefusal,
    WeightSync,
    check_sync_capabilities,
    verify_sync,
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
    "SyncCapabilities",
    "SyncCapabilityRefusal",
    "SyncReport",
    "SyncReportRefusal",
    "WeightSync",
    "build_objective_gate_context",
    "check_capabilities",
    "check_sync_capabilities",
    "verify_generated",
    "verify_sync",
)
