"""Adversarial tests for the preference binding's setup wiring and refusal behaviour."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from foundationscale.rl.algorithm import AlgorithmSemantics, AlgorithmWiringRefusal
from foundationscale.rl.interfaces import ExperienceBatch, ForwardFn
from foundationscale.rl.losses import DPOLoss
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.preference import (
    PreferenceAlgorithm,
    PreferenceObjective,
    check_preference_requirements,
)
from foundationscale.rl.preference_objectives import KTOLoss, ORPOLoss


@dataclass
class _StructuralPreferenceObjective:
    """A protocol-conforming objective whose schema can be varied independently."""

    required_columns: tuple[str, ...]
    reference_free: bool = True
    semantics_overrides: Mapping[str, object] = field(default_factory=dict)

    def declaration(self) -> Any:
        # Pairedness must come from this fake's schema rather than from its
        # concrete class, so it deliberately carries a valid shipped declaration.
        return DPOLoss().declaration()

    def semantics(self) -> AlgorithmSemantics:
        semantic_fields: dict[str, object] = {"reference_free": self.reference_free}
        semantic_fields.update(self.semantics_overrides)
        return AlgorithmSemantics(**semantic_fields)

    def __call__(self, _forward_fn: ForwardFn, _batch: ExperienceBatch) -> Any:
        # Setup must never invoke an objective; calling this would make a
        # supposedly setup-only test data-dependent.
        raise AssertionError("setup must not price the objective")


def _pair() -> PolicyPair:
    return PolicyPair(train_view=object(), generate_view=None, references={})


def _paired_config() -> dict[str, str]:
    return {
        "policy_chosen_logprob_column": "chosen_policy_logprobs",
        "policy_rejected_logprob_column": "rejected_policy_logprobs",
    }


def _unpaired_config() -> dict[str, str]:
    return {"policy_logprob_column": "policy_logprobs"}


def _setup_kwargs(
    objective: PreferenceObjective,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    dataloader: list[ExperienceBatch] = []
    return {
        "policy_pair": _pair(),
        "loss_fn": objective,
        "dataloader": dataloader,
        "config": config,
    }


def test_check_preference_requirements_returns_the_sorted_consumed_roles() -> None:
    consumed = check_preference_requirements(
        requires={
            "policy_pair": True,
            "reference_policy": False,
            "rollout_source": False,
        },
        supplied={
            "policy_pair": object(),
            "reference_policy": None,
            "rollout_source": None,
        },
        origin="preference-under-test",
    )

    assert consumed == ("policy_pair",)


def test_check_preference_requirements_refuses_an_empty_role_map() -> None:
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="0 roles appear in either map for preference-under-test",
    ) as excinfo:
        check_preference_requirements(
            requires={},
            supplied={},
            origin="preference-under-test",
        )

    assert "preference-under-test" in str(excinfo.value)


def test_check_preference_requirements_refuses_a_map_declaring_no_required_roles() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match="required") as excinfo:
        check_preference_requirements(
            requires={"reference_policy": False},
            supplied={},
            origin="preference-under-test",
        )

    assert "preference-under-test" in str(excinfo.value)


def test_check_preference_requirements_refuses_an_absent_required_role() -> None:
    with pytest.raises(
        AlgorithmWiringRefusal,
        match=r"1 of 1 required inputs absent for preference-under-test: \('policy_pair',\)",
    ) as excinfo:
        check_preference_requirements(
            requires={"policy_pair": True, "reference_policy": False},
            supplied={},
            origin="preference-under-test",
        )

    message = str(excinfo.value)
    assert message.startswith("field supplied")
    assert "1 of 1 required inputs absent for preference-under-test" in message


def test_check_preference_requirements_refuses_a_none_required_role() -> None:
    # Presence in the mapping is not consumption: only a value distinct from
    # None can satisfy the declared role.
    with pytest.raises(
        AlgorithmWiringRefusal,
        match=r"1 of 1 required inputs absent for preference-under-test: \('policy_pair',\)",
    ) as excinfo:
        check_preference_requirements(
            requires={"policy_pair": True},
            supplied={"policy_pair": None},
            origin="preference-under-test",
        )

    message = str(excinfo.value)
    assert "1 supplied keys" in message
    assert "preference-under-test" in message


def test_check_preference_requirements_refuses_an_undeclared_supplied_role() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match="mystery_role") as excinfo:
        check_preference_requirements(
            requires={"policy_pair": True},
            supplied={"policy_pair": object(), "mystery_role": object()},
            origin="preference-under-test",
        )

    assert "preference-under-test" in str(excinfo.value)


def test_check_preference_requirements_refuses_a_role_declared_unconsumed() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match="rollout_source") as excinfo:
        check_preference_requirements(
            requires={"policy_pair": True, "rollout_source": False},
            supplied={
                "policy_pair": object(),
                "rollout_source": object(),
            },
            origin="preference-under-test",
        )

    assert "preference-under-test" in str(excinfo.value)


def test_reference_anchored_setup_records_requires_and_supplied_around_wiring() -> None:
    objective = DPOLoss()
    algorithm = PreferenceAlgorithm(name="dpo", objective=objective)
    requires_before = algorithm.requires()

    assert requires_before == {
        "policy_pair": True,
        "loss_fn": True,
        "dataloader": True,
        "advantage_fn": False,
        "reference_policy": False,
        "rollout_source": False,
        "weight_sync": False,
    }
    assert algorithm.supplied() is None

    pair = _pair()
    dataloader: list[ExperienceBatch] = []
    algorithm.setup(
        policy_pair=pair,
        loss_fn=objective,
        dataloader=dataloader,
        config=_paired_config(),
    )
    supplied = algorithm.supplied()

    assert algorithm.requires() is requires_before
    assert supplied is not None
    assert supplied == {
        "policy_pair": pair,
        "loss_fn": objective,
        "dataloader": dataloader,
    }


def test_setup_wires_a_reference_free_paired_objective_without_a_batch() -> None:
    objective = ORPOLoss()
    algorithm = PreferenceAlgorithm(name="orpo", objective=objective)
    pair = _pair()
    dataloader: list[ExperienceBatch] = []

    algorithm.setup(
        policy_pair=pair,
        loss_fn=objective,
        dataloader=dataloader,
        config=_paired_config(),
    )

    assert algorithm.supplied() == {
        "policy_pair": pair,
        "loss_fn": objective,
        "dataloader": dataloader,
    }


def test_setup_wires_kto_with_the_unpaired_config_key() -> None:
    objective = KTOLoss()
    algorithm = PreferenceAlgorithm(name="kto", objective=objective)
    pair = _pair()
    dataloader: list[ExperienceBatch] = []

    algorithm.setup(
        policy_pair=pair,
        loss_fn=objective,
        dataloader=dataloader,
        config=_unpaired_config(),
    )

    assert algorithm.supplied() == {
        "policy_pair": pair,
        "loss_fn": objective,
        "dataloader": dataloader,
    }


def test_schema_with_one_chosen_and_one_rejected_column_selects_paired_setup() -> None:
    objective = _StructuralPreferenceObjective(
        required_columns=("chosen_scores", "rejected_scores"),
    )
    algorithm = PreferenceAlgorithm(name="custom-paired", objective=objective)

    algorithm.setup(**_setup_kwargs(objective, config=_paired_config()))

    assert algorithm.supplied() is not None


def test_schema_with_neither_preference_side_selects_unpaired_setup() -> None:
    objective = _StructuralPreferenceObjective(required_columns=("flagged_completions",))
    algorithm = PreferenceAlgorithm(name="custom-unpaired", objective=objective)

    algorithm.setup(**_setup_kwargs(objective, config=_unpaired_config()))

    assert algorithm.supplied() is not None


@pytest.mark.parametrize(
    ("required_columns", "chosen_count", "rejected_count"),
    [
        (("chosen_scores",), 1, 0),
        (("rejected_scores",), 0, 1),
    ],
    ids=["chosen-only", "rejected-only"],
)
def test_a_half_declared_preference_schema_is_refused_not_guessed(
    required_columns: tuple[str, ...],
    chosen_count: int,
    rejected_count: int,
) -> None:
    # The stub is fully protocol-conforming so the refusal cannot be explained
    # by the constructor's structural smoke test; only the half schema can cause it.
    objective = _StructuralPreferenceObjective(required_columns=required_columns)

    with pytest.raises(AlgorithmWiringRefusal, match="1 of 2 sides is absent") as excinfo:
        PreferenceAlgorithm(name="half-declared", objective=objective)

    message = str(excinfo.value)
    assert f"names {chosen_count} chosen and {rejected_count} rejected column(s)" in message
    assert "pairedness decides the forward closure's shape" in message


def test_setup_refuses_a_missing_policy_pair_within_the_common_handshake() -> None:
    objective = DPOLoss()
    algorithm = PreferenceAlgorithm(name="dpo", objective=objective)
    kwargs = _setup_kwargs(objective, config=_paired_config())
    kwargs["policy_pair"] = None

    with pytest.raises(
        AlgorithmWiringRefusal,
        match="wired dpo against a NoneType rather than a PolicyPair",
    ):
        algorithm.setup(**kwargs)


@pytest.mark.parametrize(
    "surplus_role",
    ["advantage_fn", "rollout_source", "weight_sync"],
)
def test_setup_refuses_each_component_declared_unconsumed(surplus_role: str) -> None:
    objective = DPOLoss()
    algorithm = PreferenceAlgorithm(name="dpo", objective=objective)
    kwargs = _setup_kwargs(objective, config=_paired_config())
    kwargs[surplus_role] = object()

    with pytest.raises(AlgorithmWiringRefusal, match=surplus_role):
        algorithm.setup(**kwargs)


def test_setup_refuses_a_pair_carrying_a_reference_even_for_an_anchored_objective() -> None:
    objective = DPOLoss()
    algorithm = PreferenceAlgorithm(name="dpo", objective=objective)
    kwargs = _setup_kwargs(objective, config=_paired_config())
    kwargs["policy_pair"] = PolicyPair(
        train_view=object(),
        generate_view=None,
        references={"reference_policy": object()},
    )

    with pytest.raises(AlgorithmWiringRefusal, match="reference_policy"):
        algorithm.setup(**kwargs)


def test_setup_refuses_a_loss_that_is_not_the_carried_objective_instance() -> None:
    objective = DPOLoss()
    algorithm = PreferenceAlgorithm(name="dpo", objective=objective)
    kwargs = _setup_kwargs(objective, config=_paired_config())
    kwargs["loss_fn"] = DPOLoss()

    with pytest.raises(AlgorithmWiringRefusal, match="2 distinct objects") as excinfo:
        algorithm.setup(**kwargs)

    assert "1 declared objective role" in str(excinfo.value)


def test_setup_refuses_the_algorithm_after_one_successful_wiring() -> None:
    objective = DPOLoss()
    algorithm = PreferenceAlgorithm(name="dpo", objective=objective)
    kwargs = _setup_kwargs(objective, config=_paired_config())

    algorithm.setup(**kwargs)
    with pytest.raises(AlgorithmWiringRefusal, match="already wired") as excinfo:
        algorithm.setup(**kwargs)

    assert "1 of 1 dpo algorithm objects" in str(excinfo.value)


def test_setup_refuses_a_non_iterable_dataloader_and_chains_the_type_error() -> None:
    objective = DPOLoss()
    algorithm = PreferenceAlgorithm(name="dpo", objective=objective)
    kwargs = _setup_kwargs(objective, config=_paired_config())
    kwargs["dataloader"] = object()

    with pytest.raises(AlgorithmWiringRefusal, match=r"was not iterable \(object\)") as excinfo:
        algorithm.setup(**kwargs)

    assert "1 of 1 mandatory step inputs" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, TypeError)


def test_setup_refuses_a_config_that_is_not_a_mapping() -> None:
    objective = DPOLoss()
    algorithm = PreferenceAlgorithm(name="dpo", objective=objective)
    kwargs = _setup_kwargs(objective, config=_paired_config())
    kwargs["config"] = {
        "policy_chosen_logprob_column",
        "policy_rejected_logprob_column",
    }

    with pytest.raises(
        AlgorithmWiringRefusal,
        match="1 of 1 setup configurations must implement Mapping",
    ) as excinfo:
        algorithm.setup(**kwargs)

    assert "field config=" in str(excinfo.value)


@pytest.mark.parametrize(
    ("config", "refused_key"),
    [
        ({}, "policy_chosen_logprob_column"),
        ({"policy_chosen_logprob_column": 42}, "policy_chosen_logprob_column"),
        ({"policy_chosen_logprob_column": ""}, "policy_chosen_logprob_column"),
        (
            {"policy_chosen_logprob_column": "chosen_policy_logprobs"},
            "policy_rejected_logprob_column",
        ),
        (
            {
                "policy_chosen_logprob_column": "chosen_policy_logprobs",
                "policy_rejected_logprob_column": 42,
            },
            "policy_rejected_logprob_column",
        ),
        (
            {
                "policy_chosen_logprob_column": "chosen_policy_logprobs",
                "policy_rejected_logprob_column": "",
            },
            "policy_rejected_logprob_column",
        ),
    ],
    ids=[
        "chosen-absent",
        "chosen-non-string",
        "chosen-empty-string",
        "rejected-absent",
        "rejected-non-string",
        "rejected-empty-string",
    ],
)
def test_setup_refuses_each_unusable_paired_config_column(
    config: dict[str, Any],
    refused_key: str,
) -> None:
    # The empty-string legs separate value-validation from type-validation:
    # they pass isinstance(str) but do not name a batch column.
    objective = DPOLoss()
    algorithm = PreferenceAlgorithm(name="dpo", objective=objective)

    with pytest.raises(AlgorithmWiringRefusal, match=refused_key) as excinfo:
        algorithm.setup(**_setup_kwargs(objective, config=config))

    message = str(excinfo.value)
    assert "0 of 1 required config values names a non-empty batch column for dpo" in message
    assert "current-policy token log-probabilities cannot be attributed" in message


# The singular KTO key is independently guarded from the paired loop above.
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"policy_logprob_column": 42},
        {"policy_logprob_column": ""},
    ],
    ids=["absent", "non-string", "empty-string"],
)
def test_setup_refuses_an_unusable_unpaired_config_column(config: dict[str, Any]) -> None:
    objective = KTOLoss()
    algorithm = PreferenceAlgorithm(name="kto", objective=objective)

    with pytest.raises(AlgorithmWiringRefusal, match="policy_logprob_column") as excinfo:
        algorithm.setup(**_setup_kwargs(objective, config=config))

    message = str(excinfo.value)
    assert "0 of 1 required config values names a non-empty batch column for kto" in message
    assert "current-policy token log-probabilities cannot be attributed" in message


@pytest.mark.parametrize(
    ("field_name", "loss_value"),
    [
        ("group_size", 4),
        ("ratio_scope", "sequence"),
        ("kl_estimator", "k3"),
        ("clip_bounds", (0.9, 1.1)),
        ("reference_free", False),
    ],
    ids=["group-size", "ratio-scope", "kl-estimator", "clip-bounds", "reference-free"],
)
def test_setup_refuses_each_loss_side_semantics_disagreement(
    field_name: str,
    loss_value: object,
) -> None:
    # The overrides let one objective instance carry two records: the record
    # measured into the algorithm at construction and a changed record measured
    # back from the loss at wiring. Identity alone must not authorise the pair.
    objective = _StructuralPreferenceObjective(
        required_columns=("chosen_scores", "rejected_scores"),
        reference_free=True,
        semantics_overrides={field_name: loss_value},
    )
    algorithm = PreferenceAlgorithm(name="custom", objective=objective)

    with pytest.raises(AlgorithmWiringRefusal, match=field_name) as excinfo:
        algorithm.setup(**_setup_kwargs(objective, config=_paired_config()))

    message = str(excinfo.value)
    assert "algorithm declares" in message
    assert "loss declares" in message
    assert "1 of 5 semantics fields disagrees" in message
