"""Tests for the stage-1 RL contracts: the batch, the SFT loss, and the gate bridge."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from collections.abc import Mapping
from typing import Any, cast

import pytest

from foundationscale.gates.core import REGISTRY, Lifecycle, run_event
from foundationscale.gates.objective_gates import (
    LossComponent,
    ValueProvenance,
    fingerprint_hparams,
)
from foundationscale.rl import (
    BatchRefusal,
    ExperienceBatch,
    LossOutput,
    SFTLoss,
    SupervisionRefusal,
    build_objective_gate_context,
)

# Per MODULE, not per package: stage 2 split implementations out of
# ``interfaces`` into ``losses`` and ``policy``, and a torch import can creep
# into any of them. Asserting over the package alone would leave all but one
# module in no denominator -- and the modules that hold the arithmetic are
# exactly the ones that would have dropped out.
#
# This dict is a DENOMINATOR, not a snapshot of the modules that existed when
# it was written. It stayed at three entries when stage 3a added ``rollout``
# and ``advantage``: both are torch-free today, but nothing held them there,
# which is the same "grew without its declaration growing" shape this suite
# exists to catch. ``test_every_rl_module_is_in_the_torch_free_denominator``
# below refuses if any ``foundationscale.rl`` module is missing an entry, so
# the next module to land cannot repeat it.
TORCH_FREE_MODULES = {
    "foundationscale.rl.advantage": (
        "AdvantageConfigRefusal",
        "AdvantageFn",
        "AdvantageRefusal",
        "AdvantageResult",
        "GeneralisedAdvantageEstimation",
        "GroupNormalisedAdvantage",
        "LearnedValueAdvantageEstimation",
        "LeaveOneOutAdvantage",
        "RewardStats",
        "TemporalAdvantageFn",
    ),
    "foundationscale.rl.algorithm": (
        "Algorithm",
        "AlgorithmRequirements",
        "AlgorithmSemantics",
        "AlgorithmWiringRefusal",
        "StepReport",
        "StepReportRefusal",
        "check_algorithm_wiring",
        "check_role_map",
        "verify_step",
    ),
    "foundationscale.rl.corpus": (
        "Sample",
        "extract_mcq_gold",
        "load_sharegpt",
    ),
    "foundationscale.rl.group_policy": (
        "SequenceObjective",
        "SequencePolicyAlgorithm",
        "check_group_policy_requirements",
        "dapo_algorithm",
        "dr_grpo_algorithm",
        "gspo_algorithm",
    ),
    "foundationscale.rl.group_policy_objectives": (
        "DAPOLoss",
        "DrGRPOLoss",
        "GSPOLoss",
    ),
    "foundationscale.rl.grpo": (
        "GRPOAlgorithm",
        "GRPOPolicyLoss",
        "check_grpo_requirements",
    ),
    "foundationscale.rl.reward_model": (
        "RewardModelLoss",
        "check_reward_model_requirements",
    ),
    "foundationscale.rl.rewards": ("MCQLetterReward",),
    "foundationscale.rl.prompt_surface": (
        "PromptSurface",
        "chat_template_or_refuse",
        "encode_prompts",
        "resolve_prompt_surface",
    ),
    "foundationscale.rl.registry": (
        "AlgorithmRegistryRefusal",
        "available_algorithm_names",
        "lookup_algorithm",
        "register_algorithm",
        "reset_algorithm_registry",
    ),
    "foundationscale.rl.interfaces": (
        "BatchRefusal",
        "ExperienceBatch",
        "ForwardFn",
        "LossConfigRefusal",
        "LossDeclaration",
        "LossFn",
        "LossOutput",
        "SupervisionRefusal",
        "build_objective_gate_context",
    ),
    "foundationscale.rl.losses": ("DPOLoss", "SFTLoss"),
    "foundationscale.rl.online_objectives": (
        "BestOfNLoss",
        "IterativeDPOLoss",
        "OnlineDPOLoss",
        "RAFTLoss",
    ),
    "foundationscale.rl.policy": ("PolicyPair", "PolicyRoleRefusal"),
    "foundationscale.rl.policy_gradient": (
        "RLOOAlgorithm",
        "RLOOPolicyLoss",
        "ReinforceBaselineAlgorithm",
        "ReinforceBaselineLoss",
        "ReinforcePlusPlusAlgorithm",
        "ReinforcePlusPlusLoss",
        "check_reinforce_baseline_requirements",
        "check_reinforce_pp_requirements",
        "check_rloo_requirements",
    ),
    "foundationscale.rl.ppo": (
        "PPOAlgorithm",
        "PPOCompositeLoss",
        "check_ppo_requirements",
    ),
    "foundationscale.rl.preference": (
        "PreferenceAlgorithm",
        "PreferenceObjective",
        "check_preference_requirements",
        "cpo_algorithm",
        "dpo_algorithm",
        "ipo_algorithm",
        "kto_algorithm",
        "orpo_algorithm",
        "simpo_algorithm",
    ),
    "foundationscale.rl.preference_objectives": (
        "CPOLoss",
        "IPOLoss",
        "KTOLoss",
        "ORPOLoss",
        "SimPOLoss",
    ),
    "foundationscale.rl.ppo_objectives": (
        "AdaptiveKLController",
        "FixedKLCoefficient",
        "KLCoefficientController",
        "KLPenaltyLoss",
        "PPOClippedPolicyLoss",
        "ValueFunctionLoss",
    ),
    "foundationscale.rl.rollout": (
        "CapabilityRefusal",
        "RolloutSource",
        "SourceCapabilities",
        "check_capabilities",
        "verify_generated",
    ),
    "foundationscale.rl.value_head": (
        "ValueCapabilities",
        "ValueCapabilityRefusal",
        "ValueEstimateRefusal",
        "ValueHead",
        "check_value_capabilities",
        "verify_estimates",
    ),
    "foundationscale.rl.weightsync": (
        "SyncCapabilities",
        "SyncCapabilityRefusal",
        "SyncReport",
        "SyncReportRefusal",
        "WeightSync",
        "check_sync_capabilities",
        "verify_sync",
    ),
    "foundationscale.rl.torch_backend": ("TensorPolicyLoss",),
    "foundationscale.rl.trainer": (
        "RLTrainConfig",
        "RLTrainer",
        "TrainerRefusal",
    ),
}

# ``torch_backend`` and ``trainer`` are in the set ABOVE, and the reason is
# worth stating because it is counter-intuitive: both of them USE torch
# heavily, and both are nonetheless torch-free AT IMPORT. The packaging
# census (tests/packaging/test_emitter_torch_free_imports.py) forbids an
# import-time torch import anywhere under src/, so every torch use in the
# plane is a function-local import. The distinction those two modules carry
# is CALL-time, not import-time: importing them succeeds without torch and
# calling them raises. So there is ONE denominator here, not two -- a
# second "torch-dependent" set would today be empty, and a parametrised
# sweep over an empty set is a vacuous pass wearing a measurement's clothes.


def _measured_loss() -> LossOutput:
    component = LossComponent(name="sft_loss", weight=1.0, observed=True, contribution=0.25)
    return LossOutput(loss=0.25, components=(component,))


# The sweeps below vary the step-0 record and nothing else, so the objective is
# DECLARED by default. Leaving it None would make `objective.declared` block on
# every arm, and each negative arm's `report.ok is False` would then hold even
# with the drift removed -- a confounded assertion that proves nothing about the
# gate under test. `source` must be one of the gate's known provenance classes
# ({"cli", "config", "default", "env"}); a config field is "config".
_DECLARED_OBJECTIVE = ValueProvenance(name="objective", value="sft", source="config", recorded=True)


def _gate_context(
    *,
    current_hparams: Mapping[str, Any],
    step0_fingerprint: str | None,
    step0_hparams: Mapping[str, Any] | None,
    objective: ValueProvenance | None = _DECLARED_OBJECTIVE,
) -> Any:
    return build_objective_gate_context(
        _measured_loss(),
        objective=objective,
        declared_components=("sft_loss",),
        current_hparams=current_hparams,
        step0_fingerprint=step0_fingerprint,
        step0_hparams=step0_hparams,
    )


def _result_by_id(report: Any, gate_id: str) -> Any:
    # Gate result order is not part of the contract; select by id.
    for result in report.results:
        if result.gate_id == gate_id:
            return result
    msg = f"no gate result with gate_id {gate_id!r}"
    raise AssertionError(msg)


def test_missing_required_columns_message_counts_and_lists() -> None:
    with pytest.raises(BatchRefusal) as excinfo:
        ExperienceBatch(
            columns={"tokens": [1, 2]},
            required=("tokens", "loss_mask", "weights"),
        )
    message = str(excinfo.value)
    assert "2 of 3" in message
    assert "'loss_mask'" in message
    assert "'weights'" in message
    assert "('tokens',)" in message


def test_misaligned_columns_message_names_both_columns_and_counts() -> None:
    with pytest.raises(BatchRefusal) as excinfo:
        ExperienceBatch(columns={"tokens": [1, 2, 3], "loss_mask": [1, 1]})
    message = str(excinfo.value)
    assert "column 'loss_mask' has 2 rows" in message
    assert "column 'tokens' has 3 rows" in message


def test_unsized_column_is_refused_with_column_name() -> None:
    def rows() -> Any:
        yield [1, 2]
        yield [3, 4]

    columns: dict[str, Any] = {"generated": rows()}
    with pytest.raises(BatchRefusal) as excinfo:
        ExperienceBatch(columns=columns)
    message = str(excinfo.value)
    assert "'generated'" in message
    assert "no row count" in message


def test_columns_are_frozen_to_tuples_at_construction() -> None:
    tokens = [10, 20, 30]
    batch = ExperienceBatch(columns={"tokens": tokens})
    assert isinstance(batch.columns["tokens"], tuple)
    tokens.append(40)
    assert batch.columns["tokens"] == (10, 20, 30)
    assert len(batch) == 3


def test_len_is_row_count_and_empty_batch_has_length_zero() -> None:
    batch = ExperienceBatch(columns={"a": [1, 2], "b": [3, 4]})
    assert len(batch) == 2
    assert len(ExperienceBatch(columns={})) == 0


def test_slice_takes_same_rows_from_every_column_and_keeps_required() -> None:
    batch = ExperienceBatch(
        columns={"tokens": [10, 20, 30, 40], "loss_mask": [0, 1, 0, 1]},
        required=("tokens",),
    )
    sliced = batch[1:3]
    assert sliced.column("tokens") == (20, 30)
    assert sliced.column("loss_mask") == (1, 0)
    assert sliced.required == ("tokens",)


def test_int_index_raises_typeerror_pointing_at_column_access() -> None:
    batch = ExperienceBatch(columns={"tokens": [10, 20]})
    bad_index: Any = 0
    with pytest.raises(TypeError) as excinfo:
        batch[bad_index]
    message = str(excinfo.value)
    assert "slice" in message
    assert ".column(name)" in message


def test_absent_column_keyerror_lists_carried_columns() -> None:
    batch = ExperienceBatch(columns={"tokens": [1], "loss_mask": [1]})
    with pytest.raises(KeyError) as excinfo:
        batch.column("absent")
    message = str(excinfo.value)
    assert "'absent'" in message
    assert "'tokens'" in message
    assert "'loss_mask'" in message


def test_all_zero_mask_raises_supervision_refusal_not_zero_loss() -> None:
    # Returning 0.0 here would report a perfect loss for a batch that taught
    # the model nothing, so the loss must refuse instead of reporting.
    batch = ExperienceBatch(columns={"loss_mask": [[0, 0], [0, 0]]})
    log_probs = [[-0.1, -0.2], [-0.3, -0.4]]
    with pytest.raises(SupervisionRefusal) as excinfo:
        SFTLoss()(lambda _batch: log_probs, batch)
    message = str(excinfo.value)
    assert "0 supervised tokens" in message
    assert "2 batch rows" in message


def test_sft_loss_matches_hand_computed_value() -> None:
    # Supervised entries: (row 0, pos 1) -0.2, (row 0, pos 2) -0.3,
    # (row 1, pos 2) -0.6 -> 3 supervised tokens, sum -1.1, so the loss is
    # -(-1.1 / 3) == 1.1 / 3 == 0.3666666666666667.
    batch = ExperienceBatch(columns={"loss_mask": [[0, 1, 1], [0, 0, 1]]})
    log_probs = [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]]
    output = SFTLoss()(lambda _batch: log_probs, batch)
    assert output.loss == pytest.approx(1.1 / 3)
    (component,) = output.components
    assert component.name == "sft_loss"
    assert component.weight == 1.0
    assert component.observed is True
    assert component.contribution == pytest.approx(1.1 / 3)


def test_weight_scales_loss_and_is_recorded_on_component() -> None:
    batch = ExperienceBatch(columns={"loss_mask": [[0, 1, 1], [0, 0, 1]]})
    log_probs = [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]]
    output = SFTLoss(weight=0.5)(lambda _batch: log_probs, batch)
    assert output.loss == pytest.approx(0.5 * (1.1 / 3))
    (component,) = output.components
    assert component.weight == 0.5
    assert component.contribution == output.loss


def test_sft_declaration_states_one_component_and_a_declared_metric_abstention() -> None:
    declaration = SFTLoss().declaration()
    assert declaration.components == ("sft_loss",)
    assert declaration.metrics == ()
    # Asserting the empty tuple alone cannot tell a DECLARED abstention from an
    # oversight -- both are `()`. Feeding it to the plane can: with nothing
    # declared and nothing observed, objective.metrics answers SKIP, so the
    # absence stays in the sweep's denominator instead of reading as coverage.
    step0_hparams = {"lr": 1e-5}
    ctx = build_objective_gate_context(
        _measured_loss(),
        objective=_DECLARED_OBJECTIVE,
        declared_components=declaration.components,
        declared_metrics=declaration.metrics,
        current_hparams=dict(step0_hparams),
        step0_fingerprint=fingerprint_hparams(step0_hparams),
        step0_hparams=step0_hparams,
    )
    report = run_event(REGISTRY, Lifecycle.STEP_ZERO, ctx, missing_ctx="report-skip")
    assert _result_by_id(report, "objective.metrics").verdict.name == "SKIP"


def test_sft_declared_and_observed_component_names_cannot_diverge() -> None:
    # Both the declaration and the computed component read the ONE field
    # component_name, so renaming it cannot leave a declared-but-unobserved
    # component behind -- the state LossComponentCoverageGate blocks on.
    loss_fn = SFTLoss(component_name="policy_loss")
    batch = ExperienceBatch(columns={"loss_mask": [[1, 1]]})
    output = loss_fn(lambda _batch: [[-0.5, -0.25]], batch)
    declared = set(loss_fn.declaration().components)
    assert declared == {"policy_loss"}
    assert {component.name for component in output.components} == declared


def test_fractional_mask_entry_is_refused_with_row_and_position() -> None:
    batch = ExperienceBatch(columns={"loss_mask": [[1, 0.5]]})
    with pytest.raises(BatchRefusal) as excinfo:
        SFTLoss()(lambda _batch: [[-0.1, -0.2]], batch)
    message = str(excinfo.value)
    assert "row 0" in message
    assert "position 1" in message
    assert "0.5" in message


def test_foreign_scalar_and_bool_mask_entries_are_admitted() -> None:
    # At the loop seam a mask arrives as a foreign scalar (one framework's
    # 0-d array element); refusing it by Python class would refuse a
    # well-formed mask, so admission is by VALUE via float(). True/False
    # are admitted as 1/0.
    class _ForeignOne:
        def __float__(self) -> float:
            return 1.0

    batch = ExperienceBatch(
        columns={"loss_mask": [[_ForeignOne(), True], [False, 0]]},
    )
    log_probs = [[-0.2, -0.4], [-0.6, -0.8]]
    output = SFTLoss()(lambda _batch: log_probs, batch)
    # Supervised: the foreign 1.0 and True -> sum -0.2 + -0.4 over 2 tokens.
    assert output.loss == pytest.approx(0.3)


def test_forward_returning_wrong_row_count_is_refused() -> None:
    batch = ExperienceBatch(columns={"loss_mask": [[1], [1]]})
    with pytest.raises(BatchRefusal) as excinfo:
        SFTLoss()(lambda _batch: [[-0.5]], batch)
    message = str(excinfo.value)
    assert "returned 1 per-token rows" in message
    assert "batch of 2 rows" in message


def test_forward_returning_wrong_token_count_is_refused() -> None:
    batch = ExperienceBatch(columns={"loss_mask": [[1, 1], [1, 1]]})
    log_probs = [[-0.1, -0.2], [-0.3, -0.4, -0.5]]
    with pytest.raises(BatchRefusal) as excinfo:
        SFTLoss()(lambda _batch: log_probs, batch)
    message = str(excinfo.value)
    assert "row 1" in message
    assert "returned 3 per-token log-probabilities" in message
    assert "mask has 2 entries" in message


def test_missing_mask_column_is_refused_naming_column_and_present() -> None:
    batch = ExperienceBatch(columns={"tokens": [[1, 2]]})
    with pytest.raises(BatchRefusal) as excinfo:
        SFTLoss()(lambda _batch: [[-0.1, -0.2]], batch)
    message = str(excinfo.value)
    assert "'loss_mask'" in message
    assert "'tokens'" in message


def test_forward_fn_is_called_exactly_once_per_evaluation() -> None:
    calls: list[ExperienceBatch] = []

    def forward(batch: ExperienceBatch) -> list[list[float]]:
        calls.append(batch)
        return [[-0.5, -0.25]]

    batch = ExperienceBatch(columns={"loss_mask": [[1, 1]]})
    SFTLoss()(forward, batch)
    assert len(calls) == 1
    assert calls[0] is batch


def test_components_are_coerced_to_tuple_from_list() -> None:
    component = LossComponent(name="sft_loss", weight=1.0, observed=True, contribution=0.5)
    components: Any = [component]
    output = LossOutput(loss=0.5, components=components)
    assert isinstance(output.components, tuple)
    assert output.components == (component,)


def test_absent_contribution_round_trips_as_none_not_zero() -> None:
    unmeasured = LossComponent(name="kl", weight=1.0, observed=False, contribution=None)
    measured_zero = LossComponent(name="sft_loss", weight=1.0, observed=True, contribution=0.0)
    output = LossOutput(loss=0.0, components=(unmeasured, measured_zero))
    assert output.components[0].contribution is None
    # A measured 0.0 is a real observation and stays distinct from None.
    assert output.components[1].contribution is not None
    assert output.components[1].contribution == 0.0


def test_step0_record_parameters_are_required_with_no_default() -> None:
    # The point of the missing defaults: a caller with no step-0 record must
    # state the absence explicitly rather than have it defaulted away.
    build = cast(Any, build_objective_gate_context)
    with pytest.raises(TypeError) as excinfo:
        build(
            _measured_loss(),
            objective=None,
            declared_components=("sft_loss",),
            current_hparams={"lr": 1e-5},
        )
    message = str(excinfo.value)
    assert "step0_fingerprint" in message
    assert "step0_hparams" in message


def test_gate_sweep_passes_when_current_hparams_match_step0() -> None:
    # Positive control: this PASS is evidence only because the must-fire
    # test below shows the same instrument producing FAIL on drift.
    step0_hparams = {"lr": 1e-5, "kl_coef": 0.04}
    ctx = _gate_context(
        current_hparams=dict(step0_hparams),
        step0_fingerprint=fingerprint_hparams(step0_hparams),
        step0_hparams=step0_hparams,
    )
    report = run_event(REGISTRY, Lifecycle.STEP_ZERO, ctx, missing_ctx="report-skip")
    assert report.ok is True
    drift = _result_by_id(report, "objective.hparam_drift")
    assert drift.verdict.name == "PASS"


def test_gate_sweep_fails_when_kl_coef_drifts_from_step0() -> None:
    step0_hparams = {"lr": 1e-5, "kl_coef": 0.04}
    ctx = _gate_context(
        current_hparams={"lr": 1e-5, "kl_coef": 0.0},
        step0_fingerprint=fingerprint_hparams(step0_hparams),
        step0_hparams=step0_hparams,
    )
    report = run_event(REGISTRY, Lifecycle.STEP_ZERO, ctx, missing_ctx="report-skip")
    assert report.ok is False
    drift = _result_by_id(report, "objective.hparam_drift")
    assert drift.verdict.name == "FAIL"
    assert "kl_coef" in drift.detail


def test_gate_sweep_fails_when_step0_record_is_stated_absent() -> None:
    # This FAIL is the gate behaving correctly: hyperparameters with nothing
    # recorded to compare against are unfalsifiable, and the bridge declines
    # to invent a record on the caller's behalf.
    ctx = _gate_context(
        current_hparams={"lr": 1e-5, "kl_coef": 0.04},
        step0_fingerprint=None,
        step0_hparams=None,
    )
    report = run_event(REGISTRY, Lifecycle.STEP_ZERO, ctx, missing_ctx="report-skip")
    assert report.ok is False
    drift = _result_by_id(report, "objective.hparam_drift")
    assert drift.verdict.name == "FAIL"


def test_bridge_passes_none_objective_through_as_none() -> None:
    ctx = _gate_context(
        objective=None,
        current_hparams={"lr": 1e-5},
        step0_fingerprint=None,
        step0_hparams=None,
    )
    assert ctx.objective is None


def test_bridge_passes_loss_components_through_unchanged() -> None:
    loss = _measured_loss()
    ctx = build_objective_gate_context(
        loss,
        objective=None,
        declared_components=("sft_loss",),
        current_hparams={"lr": 1e-5},
        step0_fingerprint=None,
        step0_hparams=None,
    )
    assert tuple(ctx.components) == loss.components
    for reached, original in zip(ctx.components, loss.components, strict=True):
        assert reached is original


@pytest.mark.parametrize("module_name", sorted(TORCH_FREE_MODULES))
def test_module_imports_with_torch_absent(module_name: str) -> None:
    class _TorchBlocker:
        def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
            if name == "torch" or name.startswith("torch."):
                msg = f"torch is blocked for this test ({name})"
                raise ImportError(msg)
            return None

    blocker = _TorchBlocker()
    saved = sys.modules.pop(module_name, None)
    sys.meta_path.insert(0, cast(Any, blocker))
    try:
        module = importlib.import_module(module_name)
        for public_name in TORCH_FREE_MODULES[module_name]:
            assert hasattr(module, public_name), public_name
    finally:
        # Restore both the finder chain and sys.modules exactly: a leaked
        # finder or a stale module entry would make every later test in the
        # session order-dependent.
        sys.meta_path.remove(blocker)
        sys.modules.pop(module_name, None)
        if saved is not None:
            sys.modules[module_name] = saved


def test_every_rl_module_is_in_the_torch_free_denominator() -> None:
    # The sweep above can only measure what TORCH_FREE_MODULES lists, so the
    # dict is itself a claim: "these are all the modules in the package".
    # Measure that claim against the package rather than trusting it. Stage 3a
    # landed two modules and this dict did not grow; a module that is torch-free
    # by luck and in no denominator is not a covered module.
    package = importlib.import_module("foundationscale.rl")
    present = {
        f"foundationscale.rl.{info.name}"
        for info in pkgutil.iter_modules(package.__path__)
        if not info.name.startswith("_")
    }
    assert present, (
        "found 0 submodules of foundationscale.rl: the enumeration is broken, "
        "and an empty denominator makes this control vacuously pass -- which "
        "is the exact failure it exists to catch"
    )
    declared = set(TORCH_FREE_MODULES)
    unmeasured = sorted(present - declared)
    assert not unmeasured, (
        f"{len(unmeasured)} of {len(present)} foundationscale.rl modules sit in "
        f"no torch-free denominator: {', '.join(unmeasured)}. Add each to "
        f"TORCH_FREE_MODULES with the public names it must expose under a "
        f"blocked torch import. A module that USES torch still belongs here: "
        f"the packaging census forbids an import-time torch import under "
        f"src/, so every such module imports torch inside its functions and "
        f"is importable without it. Being torch-free today is not being "
        f"held there"
    )
    stale = sorted(declared - present)
    assert not stale, (
        f"{len(stale)} of {len(declared)} declared modules do not exist: "
        f"{', '.join(stale)}. A declaration naming a module that is gone "
        f"reports coverage it cannot deliver"
    )
