"""Construction tests for the preference family: the objective-slot refusal and the six factories.

Measures only what happens before any wiring: ``PreferenceAlgorithm.__init__`` (the
``PreferenceObjective`` smoke test, the half-declared pair-schema refusal, and the verbatim
recording of ``name``), and the six thin factories (freshness, bound name, carried objective
class, default and overridden knob threading, and the derived requirements/semantics records).
``setup()``, ``step()`` and ``check_preference_requirements`` are measured by a sibling module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from foundationscale.rl.algorithm import AlgorithmWiringRefusal
from foundationscale.rl.losses import DPOLoss
from foundationscale.rl.preference import (
    PreferenceAlgorithm,
    cpo_algorithm,
    dpo_algorithm,
    ipo_algorithm,
    kto_algorithm,
    orpo_algorithm,
    simpo_algorithm,
)
from foundationscale.rl.preference_objectives import (
    CPOLoss,
    IPOLoss,
    KTOLoss,
    ORPOLoss,
    SimPOLoss,
)


class _ObjectiveWithoutReferenceFree:
    """Structural fake exposing four of the five members the protocol demands."""

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ()

    def declaration(self) -> Any:
        raise AssertionError("the slot smoke test must fire before any member is read")

    def semantics(self) -> Any:
        raise AssertionError("the slot smoke test must fire before any member is read")

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a structurally refused objective is never priced")


class _HalfSchemaObjective:
    """Conforming shell whose declared pair schema names only the chosen side."""

    @property
    def reference_free(self) -> bool:
        return False

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("chosen_sequence_logps",)

    def declaration(self) -> Any:
        raise AssertionError("the pairedness refusal must fire before the declaration is read")

    def semantics(self) -> Any:
        raise AssertionError("the pairedness refusal must fire before semantics is read")

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a structurally refused objective is never priced")


def test_init_refuses_a_non_objective_and_names_every_absent_member() -> None:
    # object() exposes none of the five protocol members, so the refusal must
    # enumerate all five against the total denominator; a refusal that named
    # fewer would understate how far the offered object is from conforming.
    with pytest.raises(
        AlgorithmWiringRefusal,
        match=r"1 of 1 objective slots must carry a PreferenceObjective",
    ) as excinfo:
        PreferenceAlgorithm(name="rogue", objective=object())
    message = str(excinfo.value)
    assert "object does not expose" in message
    for member in (
        "__call__",
        "declaration",
        "reference_free",
        "required_columns",
        "semantics",
    ):
        assert member in message
    assert "of the 5 required members" in message


def test_init_names_precisely_the_one_member_a_near_conforming_object_lacks() -> None:
    # A caller's hand-rolled objective typically misses one member, not all five;
    # if the diagnostic stopped enumerating honestly it would list members the
    # object actually carries, and the caller would fix the wrong thing.
    with pytest.raises(
        AlgorithmWiringRefusal,
        match=r"\('reference_free',\)",
    ) as excinfo:
        PreferenceAlgorithm(name="rogue", objective=_ObjectiveWithoutReferenceFree())
    message = str(excinfo.value)
    assert "_ObjectiveWithoutReferenceFree does not expose" in message
    assert "'required_columns'" not in message
    assert "'declaration'" not in message


def test_init_refuses_a_schema_that_declares_only_the_chosen_half() -> None:
    # Pairedness is derived from the objective's own schema, never its class, so
    # a schema naming chosen columns without rejected ones must refuse: guessing
    # pairedness here would silently shape an unpaired forward closure around a
    # paired objective.
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="1 of 2 sides is absent",
    ) as excinfo:
        PreferenceAlgorithm(name="rogue", objective=_HalfSchemaObjective())
    message = str(excinfo.value)
    assert "objective=_HalfSchemaObjective" in message
    assert "names 1 chosen and 0 rejected column(s)" in message
    assert "cannot be inferred from half a declaration" in message


def test_init_records_the_given_name_in_the_requirements_verbatim() -> None:
    # name is the one statement __init__ cannot derive from the objective -- the
    # losses carry no family name -- so a custom binding must see its own name in
    # the denominator that the gates measure, not an invented or formatted one.
    algorithm = PreferenceAlgorithm(name="dpo-nightly", objective=DPOLoss())
    assert algorithm.requirements().name == "dpo-nightly"


_FACTORY_CASES = (
    (dpo_algorithm, "dpo", DPOLoss, {"beta": 0.1, "weight": 1.0, "sft_weight": 0.0}, False),
    (ipo_algorithm, "ipo", IPOLoss, {"tau": 1.0, "weight": 1.0}, False),
    (
        kto_algorithm,
        "kto",
        KTOLoss,
        {"beta": 0.1, "weight": 1.0, "lambda_desirable": 1.0, "lambda_undesirable": 1.0},
        False,
    ),
    (orpo_algorithm, "orpo", ORPOLoss, {"lambda_": 1.0}, True),
    (simpo_algorithm, "simpo", SimPOLoss, {"beta": 1.0, "gamma": 0.0, "weight": 1.0}, True),
    (cpo_algorithm, "cpo", CPOLoss, {"beta": 0.1, "lambda_": 1.0}, True),
)


@pytest.mark.parametrize(
    ("factory", "expected_name", "objective_type", "default_knobs", "reference_free"),
    _FACTORY_CASES,
    ids=("dpo", "ipo", "kto", "orpo", "simpo", "cpo"),
)
def test_factory_binds_a_fresh_algorithm_carrying_a_default_knobbed_objective(
    factory: Callable[[], PreferenceAlgorithm],
    expected_name: str,
    objective_type: type,
    default_knobs: dict[str, Any],
    reference_free: bool,
) -> None:
    # Two calls must not share state: the registry registers the factory itself,
    # so a shared objective between calls would leak one run's wiring into
    # another's pricing without either caller ever seeing it.
    first = factory()
    second = factory()
    assert first is not second
    assert first.requirements().name == expected_name
    first_objective = first._objective
    assert first_objective is not second._objective
    # The factory surface claims an objective of exactly the shipped class built
    # with the factories' own defaults, so a re-tuned default or a substituted
    # class is precisely what this leg exists to notice. The carried objective
    # has no public reader; the slot is read where the claim lives.
    assert type(first_objective) is objective_type
    for knob, expected in default_knobs.items():
        assert getattr(first_objective, knob) == expected
    assert first.semantics().reference_free is reference_free


_OVERRIDE_CASES = (
    (dpo_algorithm, {"beta": 0.7, "weight": 2.0, "sft_weight": 0.25}),
    (ipo_algorithm, {"tau": 0.25, "weight": 2.0}),
    (
        kto_algorithm,
        {"beta": 0.25, "weight": 2.0, "lambda_desirable": 0.5, "lambda_undesirable": 1.5},
    ),
    (orpo_algorithm, {"lambda_": 2.5}),
    (simpo_algorithm, {"beta": 2.0, "gamma": 0.4, "weight": 0.5}),
    (cpo_algorithm, {"beta": 0.3, "lambda_": 0.5}),
)


@pytest.mark.parametrize(
    ("factory", "overrides"),
    _OVERRIDE_CASES,
    ids=("dpo", "ipo", "kto", "orpo", "simpo", "cpo"),
)
def test_factory_threads_every_keyword_argument_into_the_carried_objective(
    factory: Callable[..., PreferenceAlgorithm],
    overrides: dict[str, Any],
) -> None:
    carried = factory(**overrides)._objective
    for knob, expected in overrides.items():
        # Each knob must arrive verbatim and off the default path: a factory
        # that silently substituted its default here would price a tuned run
        # with hyperparameters the caller never chose.
        assert getattr(carried, knob) == expected


_DECLARATION_CASES = (
    (dpo_algorithm, "dpo", False),
    (ipo_algorithm, "ipo", False),
    (kto_algorithm, "kto", False),
    (orpo_algorithm, "orpo", True),
    (simpo_algorithm, "simpo", True),
    (cpo_algorithm, "cpo", True),
)


@pytest.mark.parametrize(
    ("factory", "expected_name", "reference_free"),
    _DECLARATION_CASES,
    ids=("dpo", "ipo", "kto", "orpo", "simpo", "cpo"),
)
def test_requirements_and_semantics_are_derived_from_the_carried_objective(
    factory: Callable[[], PreferenceAlgorithm],
    expected_name: str,
    reference_free: bool,
) -> None:
    algorithm = factory()
    requirements = algorithm.requirements()
    assert requirements.name == expected_name
    # The offline role map must name every unconsumed role with False --
    # reference_policy included, even for the anchored three, which consume
    # reference scores from columns and never a model -- because an unnamed role
    # is an ungraded role at the wiring gate.
    assert dict(requirements.requires) == {
        "advantage_fn": False,
        "reference_policy": False,
        "rollout_source": False,
        "weight_sync": False,
    }
    # The binding claims its declared components and metrics come straight off
    # the objective's own declaration so the two cannot drift; measuring them
    # against that declaration here is the only place drift could be caught at
    # construction time.
    declaration = algorithm._objective.declaration()
    assert requirements.declared_components == declaration.components
    assert requirements.declared_metrics == tuple(metric.name for metric in declaration.metrics)
    semantics = algorithm.semantics()
    assert semantics is requirements.semantics
    # reference_free is False for exactly the three anchored objectives and True
    # for ORPO, SimPO and CPO; the other four fields must abstain rather than
    # pose measured values on seams this family does not constrain.
    assert semantics.reference_free is reference_free
    assert semantics.group_size is None
    assert semantics.ratio_scope is None
    assert semantics.kl_estimator is None
    assert semantics.clip_bounds is None
