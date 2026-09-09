"""Reinforcement-learning contracts for Phase 3.

Design section 9 of ``validation_campaigns/nemo_rl_baseline/PHASE3_DESIGN.md``:

* stage 1 -- the data contract, the loss contract with supervised fine-tuning
  only, and the bridge that turns one measured step into an
  ``ObjectiveGateContext``;
* stage 2 -- ``PolicyPair``'s reference role and ``DPOLoss``, carrying the two
  declaration conditions the section-7 item-4 measurement attached to it;
* stage 3a -- ``RolloutSource`` and ``AdvantageFn``, the two contracts that
  need no distributed measurement to specify;
* stage 3b -- ``WeightSync``, the sibling edge that moves weights from the
  training view to the generation view;
* stage 3c -- ``Algorithm``, the policy-gradient binding that consumes all of
  the above, with ``AlgorithmRequirements`` as its declaration and
  ``StepReport`` as what one step evidences;
* stage 3d -- the registry that resolves a name to a binding, and
  ``GRPOAlgorithm``, the first concrete binding to pass through it. The
  registry is the seam the remaining algorithm families plug into: a family
  adds a module and one line in the registry's default install, not an edit
  to the contracts above. The install lives in the registry rather than in
  each family's module import so that ``reset_algorithm_registry`` restores
  the same set every time;
* stage 3e -- the policy-gradient family ``RLOOAlgorithm``,
  ``ReinforceBaselineAlgorithm`` and ``ReinforcePlusPlusAlgorithm``, the first
  use of that seam by something other than the binding it was designed
  around. The three differ only in how a reward becomes an advantage, so they
  are the cheapest available test of whether the seam is a seam or a shape
  fitted to GRPO. It held: no contract in stages 1--3c changed to admit them;
* stage 3f -- the PPO objectives: ``PPOClippedPolicyLoss``,
  ``ValueFunctionLoss`` and the KL-control trio
  (``KLCoefficientController`` with its fixed and adaptive realizations,
  consumed by ``KLPenaltyLoss``), alongside
  ``LearnedValueAdvantageEstimation`` and its sibling seam
  ``TemporalAdvantageFn`` in ``advantage``. PPO is the first family whose
  advantage estimator needs a per-token VALUE input, which is why the temporal
  seam is a sibling of ``AdvantageFn`` rather than a subtype of it: both extra
  parameters are required, so substitutability runs the wrong way. It is also
  the first family with state that survives a step -- the adaptive KL
  coefficient -- and that state is threaded the frozen way, by ``update()``
  returning a NEW controller the caller rebinds, never by mutating one.

``WeightSync`` could only be specified once the section-7 item-3
weight-transfer measurement was taken. It has been, and it returned
CLEAR_WITH_ABSTENTIONS: no single transport wins -- resharding and a
collective beat a full copy per byte at the large end, the full copy wins at
the small per-call floor, and one cell was withheld for spread over
tolerance. So the contract fixes no transport; the realization chooses one at
construction and the report evidences which one ran. Deliberately not
restated here: the per-transport ratios, which are a measurement artifact and
belong in the campaign record, not in a docstring that no gate re-measures.

``Algorithm`` declares ONCE, in ``AlgorithmRequirements``, and two gates read
that declaration: ``check_algorithm_wiring`` compares it against what setup was
HANDED -- including the wired loss's own ``declaration()``, because those are
one countable stated in two objects -- and ``verify_step`` compares it against
what each step OBSERVED. Section 3's contract set is now complete. Nothing here
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
    LearnedValueAdvantageEstimation,
    LeaveOneOutAdvantage,
    RewardStats,
    TemporalAdvantageFn,
)
from foundationscale.rl.algorithm import (
    Algorithm,
    AlgorithmRequirements,
    AlgorithmSemantics,
    AlgorithmWiringRefusal,
    StepReport,
    StepReportRefusal,
    check_algorithm_wiring,
    check_role_map,
    verify_step,
)
from foundationscale.rl.grpo import (
    GRPOAlgorithm,
    GRPOPolicyLoss,
    check_grpo_requirements,
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
from foundationscale.rl.policy_gradient import (
    ReinforceBaselineAlgorithm,
    ReinforceBaselineLoss,
    ReinforcePlusPlusAlgorithm,
    ReinforcePlusPlusLoss,
    RLOOAlgorithm,
    RLOOPolicyLoss,
    check_reinforce_baseline_requirements,
    check_reinforce_pp_requirements,
    check_rloo_requirements,
)
from foundationscale.rl.ppo_objectives import (
    AdaptiveKLController,
    FixedKLCoefficient,
    KLCoefficientController,
    KLPenaltyLoss,
    PPOClippedPolicyLoss,
    ValueFunctionLoss,
)
from foundationscale.rl.registry import (
    AlgorithmRegistryRefusal,
    available_algorithm_names,
    lookup_algorithm,
    register_algorithm,
    reset_algorithm_registry,
)
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
    "AdaptiveKLController",
    "AdvantageConfigRefusal",
    "AdvantageFn",
    "AdvantageRefusal",
    "AdvantageResult",
    "Algorithm",
    "AlgorithmRegistryRefusal",
    "AlgorithmRequirements",
    "AlgorithmSemantics",
    "AlgorithmWiringRefusal",
    "BatchRefusal",
    "CapabilityRefusal",
    "DPOLoss",
    "ExperienceBatch",
    "FixedKLCoefficient",
    "ForwardFn",
    "GRPOAlgorithm",
    "GRPOPolicyLoss",
    "GeneralisedAdvantageEstimation",
    "GroupNormalisedAdvantage",
    "KLCoefficientController",
    "KLPenaltyLoss",
    "LearnedValueAdvantageEstimation",
    "LeaveOneOutAdvantage",
    "LossConfigRefusal",
    "LossDeclaration",
    "LossFn",
    "LossOutput",
    "PPOClippedPolicyLoss",
    "PolicyPair",
    "PolicyRoleRefusal",
    "RLOOAlgorithm",
    "RLOOPolicyLoss",
    "ReinforceBaselineAlgorithm",
    "ReinforceBaselineLoss",
    "ReinforcePlusPlusAlgorithm",
    "ReinforcePlusPlusLoss",
    "RewardStats",
    "RolloutSource",
    "SFTLoss",
    "SourceCapabilities",
    "StepReport",
    "StepReportRefusal",
    "SupervisionRefusal",
    "SyncCapabilities",
    "SyncCapabilityRefusal",
    "SyncReport",
    "SyncReportRefusal",
    "TemporalAdvantageFn",
    "ValueFunctionLoss",
    "WeightSync",
    "available_algorithm_names",
    "build_objective_gate_context",
    "check_algorithm_wiring",
    "check_capabilities",
    "check_grpo_requirements",
    "check_reinforce_baseline_requirements",
    "check_reinforce_pp_requirements",
    "check_role_map",
    "check_rloo_requirements",
    "check_sync_capabilities",
    "lookup_algorithm",
    "register_algorithm",
    "reset_algorithm_registry",
    "verify_generated",
    "verify_step",
    "verify_sync",
)
