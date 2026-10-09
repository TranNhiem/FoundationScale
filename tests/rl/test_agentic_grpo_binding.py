"""The ``agentic_grpo`` binding: registry, declarations, lane refusals.

WHAT IS CLAIMED: the registry resolves ``agentic_grpo`` to a fresh binding;
``check_algorithm_wiring`` grades exactly the rollout-owning role map the
family declares; the five semantics fields read as the objective declares
them; the objective carries NO ``kl_weight`` field under any configuration
and demands no reference column; and the Megatron lane refuses
``prompt_mean`` with the named lane in its message.

WHAT IS NOT CLAIMED: that the objective arithmetic is re-verified here (it
is pinned in ``test_prompt_mean.py``), or that any run has been trained.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from foundationscale.rl.advantage import GroupNormalisedAdvantage, SessionGroupAdvantage
from foundationscale.rl.algorithm import AlgorithmWiringRefusal, check_algorithm_wiring
from foundationscale.rl.group_policy import SequencePolicyAlgorithm, agentic_grpo_algorithm
from foundationscale.rl.group_policy_objectives import AgenticGRPOLoss
from foundationscale.rl.interfaces import LossConfigRefusal
from foundationscale.rl.megatron.pp_step import loss_unit
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.registry import lookup_algorithm, reset_algorithm_registry
from foundationscale.rl.torch_backend import TensorPolicyLoss


class _RolloutSource:
    def rollout(self) -> None:
        return None


def _objective() -> AgenticGRPOLoss:
    return AgenticGRPOLoss(
        group_size=2,
        advantage_fn=SessionGroupAdvantage(min_group_size=2),
    )


def test_registry_lookup_returns_a_fresh_agentic_grpo_binding() -> None:
    reset_algorithm_registry()
    first = lookup_algorithm("agentic_grpo")
    second = lookup_algorithm("agentic_grpo")
    assert first.requirements().name == "agentic_grpo"
    assert first.supplied() is None
    assert second.supplied() is None
    assert first is not second
    reset_algorithm_registry()


def test_registry_binding_carries_the_objective_the_factory_constructs() -> None:
    algorithm = agentic_grpo_algorithm()
    assert isinstance(algorithm, SequencePolicyAlgorithm)
    assert isinstance(algorithm._objective, AgenticGRPOLoss)
    assert isinstance(algorithm._objective.advantage_fn, SessionGroupAdvantage)
    # The factory's estimator and objective must not disagree about group
    # size -- and the same check runs in AgenticGRPOLoss.__post_init__.
    assert algorithm._objective.advantage_fn.min_group_size == algorithm._objective.group_size
    assert algorithm._objective.group_size == 2


def test_check_algorithm_wiring_consumes_the_declared_role_map() -> None:
    # The requirement record declares rollout_source and advantage_fn True
    # and weight_sync / reference_policy False, so exactly those two roles
    # are consumed and the mirror in setup compares against the same pair.
    objective = _objective()
    algorithm = SequencePolicyAlgorithm(name="agentic_grpo", objective=objective)
    consumed = check_algorithm_wiring(
        algorithm.requirements(),
        policy_pair=PolicyPair(train_view=object(), generate_view=object()),
        loss_fn=objective,
        supplied={
            "rollout_source": _RolloutSource(),
            "advantage_fn": objective.advantage_fn,
            "weight_sync": None,
        },
        origin="test_agentic_grpo",
    )
    assert set(consumed) == {"advantage_fn", "rollout_source"}


def test_setup_wires_through_the_handshake_and_records_supplied() -> None:
    objective = _objective()
    algorithm = SequencePolicyAlgorithm(name="agentic_grpo", objective=objective)
    algorithm.setup(
        policy_pair=PolicyPair(train_view=object(), generate_view=object()),
        loss_fn=objective,
        dataloader=iter(()),
        config={"policy_logprob_column": "logprobs"},
        rollout_source=_RolloutSource(),
        advantage_fn=objective.advantage_fn,
    )
    supplied = algorithm.supplied()
    assert supplied is not None
    assert set(supplied) == {
        "policy_pair",
        "loss_fn",
        "dataloader",
        "rollout_source",
        "advantage_fn",
    }


def test_setup_refuses_a_foreign_advantage_estimator() -> None:
    objective = _objective()
    algorithm = SequencePolicyAlgorithm(name="agentic_grpo", objective=objective)
    with pytest.raises(AlgorithmWiringRefusal, match="2 distinct objects"):
        algorithm.setup(
            policy_pair=PolicyPair(train_view=object(), generate_view=object()),
            loss_fn=objective,
            dataloader=iter(()),
            config={"policy_logprob_column": "logprobs"},
            rollout_source=_RolloutSource(),
            advantage_fn=SessionGroupAdvantage(min_group_size=2),
        )


def test_semantics_fields_are_the_declared_agentic_values() -> None:
    algorithm = agentic_grpo_algorithm()
    semantics = algorithm.semantics()
    objective = algorithm._objective
    assert semantics == objective.semantics()
    assert semantics.group_size == 2
    assert semantics.ratio_scope == "token"
    assert semantics.kl_estimator is None
    assert semantics.clip_bounds == objective.clip_bounds
    assert semantics.reference_free is True
    assert algorithm.requirements().requires == {
        "rollout_source": True,
        "advantage_fn": True,
        "weight_sync": False,
        "reference_policy": False,
    }


def test_the_objective_declaraxes_prompt_mean_and_no_dynamic_sampling() -> None:
    objective = _objective()
    assert objective.reduction == "prompt_mean"
    assert objective.ratio_scope == "token"
    assert objective.expects_dynamic_sampling is False
    assert objective.clip_bounds == (0.8, 1.2)
    assert objective.declaration().components == ("agentic_grpo_policy_loss",)
    assert objective.declaration().metrics == ()


def test_the_objective_carries_no_kl_field_under_any_configuration() -> None:
    objective = _objective()
    field_names = tuple(field.name for field in dataclasses.fields(objective))
    assert "kl_weight" not in field_names
    assert "kl_component_name" not in field_names
    assert "reference_logprob_column" not in field_names
    # Absent, not falsy: the kernel's optional read is the only reader, and
    # it must fall back to 0.0 rather than find a field pretending to hold a
    # configured KL weight.
    assert not hasattr(objective, "kl_weight")
    assert getattr(objective, "kl_weight", 0.0) == 0.0
    assert objective.required_columns == (
        "prompt_ids",
        "rewards",
        "loss_mask",
        "old_logprobs",
    )


def test_construction_refuses_a_foreign_advantage_estimator() -> None:
    with pytest.raises(LossConfigRefusal, match="SessionGroupAdvantage"):
        AgenticGRPOLoss(advantage_fn=GroupNormalisedAdvantage())  # type: ignore[arg-type]


def test_construction_refuses_a_disagreeing_group_size() -> None:
    with pytest.raises(LossConfigRefusal, match="disagrees with 1 estimator minimum"):
        AgenticGRPOLoss(
            group_size=3,
            advantage_fn=SessionGroupAdvantage(min_group_size=2),
        )


def test_megatron_lane_refuses_prompt_mean_with_the_named_lane() -> None:
    with pytest.raises(
        ValueError,
        match=r"prompt_mean is implemented on the HF/FSDP2 lane only in this release",
    ):
        loss_unit(TensorPolicyLoss(objective=_objective()))


def test_megatron_lane_still_maps_the_three_supported_units() -> None:
    # The control that keeps the refusal above specific: the other three
    # reductions keep resolving to their design-section-3 unit.
    stub = lambda reduction: SimpleNamespace(  # noqa: E731
        objective=SimpleNamespace(reduction=reduction)
    )
    assert loss_unit(stub("token_mean")) == "token"
    assert loss_unit(stub("sequence_mean")) == "sequence"
    assert loss_unit(stub("constant")) == "dr_grpo"
