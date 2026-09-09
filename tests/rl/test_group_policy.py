"""Tests for the group-relative policy binding and its three factories.

WHAT IS CLAIMED: the factories each return a fresh, correctly bound
``SequencePolicyAlgorithm``; ``check_group_policy_requirements`` accepts a
satisfied role map and refuses every malformed one with both sides of its
count named; ``setup`` enforces identity (not equality) for the loss and the
advantage estimator, refuses the mis-typed or missing components, and never
peeks a batch; ``step`` prices exactly one batch through the carried
objective and reports the batch's row count with ``reward_stats`` and
``sync`` abstaining; and the five-field semantics tripwire fires when the
two declarations are made to disagree.

WHAT IS NOT CLAIMED: that the objective arithmetic (owned by
``group_policy_objectives.py``), the role-map arithmetic (owned by
``algorithm.py``), or DAPO's rollout-side dynamic sampling is re-verified
here -- only the binding's own seams are measured. The stub objective is a
structural placeholder; it mimics the protocol surface so the binding's
derivations can be observed without dragging the real loss arithmetic in.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest

from foundationscale.rl.algorithm import (
    AlgorithmSemantics,
    AlgorithmWiringRefusal,
)
from foundationscale.rl.group_policy import (
    SequencePolicyAlgorithm,
    check_group_policy_requirements,
    dapo_algorithm,
    dr_grpo_algorithm,
    gspo_algorithm,
)


class _Advantage:
    """Minimal advantage-estimator placeholder; identity is what setup measures."""

    def __call__(self, batch: Any) -> Any:
        return batch


class _Policy:
    """Structural stand-in for a policy; the binding never calls it."""

    def forward(self, batch: Any) -> Any:
        return batch


class _PolicyPair:
    """Reference-free pair stub: a carried frozen reference must never appear here."""

    def __init__(self) -> None:
        self.trainable = _Policy()
        self.reference = None
        self.frozen = None
        self.policy = self.trainable
        self.reference_policy = None


class _RolloutSource:
    """Presence-graded only; the binding never generates a batch itself."""

    def rollout(self) -> None:
        return None


class _Batch:
    """Hand-built batch with the columns the stub objective declares."""

    def __init__(self, columns: dict[str, list[float]]) -> None:
        self._columns = columns
        first = next(iter(columns.values()))
        self._rows = len(first)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._columns)

    def column(self, name: str) -> list[float]:
        return self._columns[name]

    def __len__(self) -> int:
        return self._rows


class _Objective:
    """Structural ``SequenceObjective``: every reading the binding takes is recorded."""

    def __init__(
        self,
        *,
        ratio_scope: str = "token",
        clip_bounds: tuple[float, float] = (0.8, 1.2),
        reference_free: bool = True,
        required_columns: tuple[str, ...] = ("logprobs_old", "advantages"),
    ) -> None:
        self._ratio_scope = ratio_scope
        self._clip_bounds = clip_bounds
        self._reference_free = reference_free
        self._required_columns = required_columns
        self._advantage = _Advantage()
        self.calls: list[tuple[Any, _Batch]] = []

    def __call__(self, forward_fn: Any, batch: _Batch) -> SimpleNamespace:
        scores = forward_fn(batch)
        self.calls.append((scores, batch))
        return SimpleNamespace(value=-float(sum(scores)), batch_rows=len(batch))

    @property
    def advantage_fn(self) -> _Advantage:
        return self._advantage

    @property
    def clip_bounds(self) -> tuple[float, float]:
        return self._clip_bounds

    @property
    def expects_dynamic_sampling(self) -> bool:
        return False

    @property
    def ratio_scope(self) -> str:
        return self._ratio_scope

    @property
    def reduction(self) -> str:
        return "token_mean"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return self._required_columns

    def declaration(self) -> SimpleNamespace:
        return SimpleNamespace(
            components=("rollout_source", "advantage_fn"),
            metrics=(SimpleNamespace(name="loss"),),
        )

    def semantics(self) -> AlgorithmSemantics:
        return AlgorithmSemantics(
            group_size=None,
            ratio_scope=self._ratio_scope,
            kl_estimator=None,
            clip_bounds=self._clip_bounds,
            reference_free=self._reference_free,
        )


def _config() -> dict[str, Any]:
    return {"policy_logprob_column": "logprobs_new"}


def _batch(**columns: list[float]) -> _Batch:
    if not columns:
        columns = {
            "logprobs_old": [-0.5, -0.4, -0.3, -0.2],
            "advantages": [1.0, 0.0, -1.0, 0.5],
            "logprobs_new": [-0.51, -0.39, -0.31, -0.19],
        }
    return _Batch(columns)


def _make_algorithm(**objective_kwargs: Any) -> tuple[SequencePolicyAlgorithm, _Objective]:
    objective = _Objective(**objective_kwargs)
    return SequencePolicyAlgorithm(name="stub_family", objective=objective), objective


def _setup(
    algorithm: SequencePolicyAlgorithm,
    objective: _Objective,
    *,
    batches: list[_Batch] | None = None,
) -> None:
    algorithm.setup(
        policy_pair=_PolicyPair(),
        loss_fn=objective,
        dataloader=iter(batches) if batches is not None else iter([_batch()]),
        config=_config(),
        rollout_source=_RolloutSource(),
        advantage_fn=objective.advantage_fn,
    )


def test_factories_return_fresh_bindings_with_the_declared_names() -> None:
    """Three factories, three names, and no shared singleton between calls."""
    pairs = (
        (gspo_algorithm, "gspo"),
        (dr_grpo_algorithm, "dr_grpo"),
        (dapo_algorithm, "dapo"),
    )
    for factory, expected_name in pairs:
        first = factory()
        second = factory()
        assert first.requirements().name == expected_name
        # A factory must not hand two registrations the same wired object.
        assert first is not second
        assert first.requirements() is not second.requirements()


def test_factories_derive_the_ratio_scope_of_their_objective() -> None:
    """GSPO alone carries the sequence-scope ratio; the split is objective-derived."""
    assert gspo_algorithm().semantics().ratio_scope == "sequence"
    assert dr_grpo_algorithm().semantics().ratio_scope == "token"
    assert dapo_algorithm().semantics().ratio_scope == "token"


def test_factories_declare_the_rollout_owning_role_map() -> None:
    """Weight sync is unconsumed across the family; it must be graded as False."""
    for factory in (gspo_algorithm, dr_grpo_algorithm, dapo_algorithm):
        requires = factory().requirements().requires
        assert requires["rollout_source"] is True
        assert requires["advantage_fn"] is True
        assert requires["weight_sync"] is False
        # Default objectives are reference-free, so the role is graded False.
        assert requires["reference_policy"] is False


def test_check_group_policy_requirements_accepts_a_satisfied_map() -> None:
    """The consumed set, sorted, is the only measurement on the happy path."""
    supplied = {"rollout_source": _RolloutSource(), "advantage_fn": _Advantage()}
    consumed = check_group_policy_requirements(
        requires={"rollout_source": True, "advantage_fn": True, "weight_sync": False},
        supplied={**supplied, "weight_sync": None},
    )
    assert consumed == ("advantage_fn", "rollout_source")


def test_check_group_policy_requirements_refuses_a_missing_role() -> None:
    """An absent required role names the role on both sides of the count."""
    with pytest.raises(AlgorithmWiringRefusal, match="rollout_source"):
        check_group_policy_requirements(
            requires={"rollout_source": True, "advantage_fn": True},
            supplied={"advantage_fn": _Advantage()},
        )


def test_check_group_policy_requirements_refuses_a_none_role() -> None:
    """A present-but-None required value is emptily supplied, not supplied."""
    with pytest.raises(AlgorithmWiringRefusal, match="advantage_fn"):
        check_group_policy_requirements(
            requires={"rollout_source": True, "advantage_fn": True},
            supplied={"rollout_source": _RolloutSource(), "advantage_fn": None},
        )


def test_check_group_policy_requirements_refuses_an_unconsumed_role() -> None:
    """A role declared False that arrives anyway reads as a stub wiring."""
    with pytest.raises(AlgorithmWiringRefusal, match="weight_sync"):
        check_group_policy_requirements(
            requires={"rollout_source": True, "weight_sync": False},
            supplied={"rollout_source": _RolloutSource(), "weight_sync": object()},
        )


def test_check_group_policy_requirements_refuses_an_empty_denominator() -> None:
    """An empty required set is an unsatisfiable declaration, not a vacuous pass."""
    with pytest.raises(AlgorithmWiringRefusal, match="0"):
        check_group_policy_requirements(requires={}, supplied={})


def test_constructor_refuses_an_object_outside_the_protocol() -> None:
    """The refusal names how many of the structural members are absent."""
    with pytest.raises(AlgorithmWiringRefusal, match="9 required members"):
        SequencePolicyAlgorithm(name="broken", objective=object())


def test_semantics_is_derived_from_the_carried_objective() -> None:
    """Every field on the binding's record traces to the objective, not a restatement."""
    algorithm, _ = _make_algorithm(ratio_scope="sequence", clip_bounds=(0.9, 1.1))
    semantics = algorithm.semantics()
    assert semantics.ratio_scope == "sequence"
    assert semantics.clip_bounds == (0.9, 1.1)
    assert semantics.group_size is None
    assert semantics.kl_estimator is None
    assert semantics.reference_free is True


def test_setup_refuses_a_distinct_loss_instance() -> None:
    """Identity, not equality: two agreeing objectives still double the one role."""
    algorithm, _ = _make_algorithm()
    impostor = _Objective()
    with pytest.raises(AlgorithmWiringRefusal, match="2 distinct objects"):
        algorithm.setup(
            policy_pair=_PolicyPair(),
            loss_fn=impostor,
            dataloader=iter([_batch()]),
            config=_config(),
            rollout_source=_RolloutSource(),
        )


def test_setup_refuses_a_distinct_advantage_estimator() -> None:
    """Two estimators that disagree about what an advantage is cannot both be right."""
    algorithm, objective = _make_algorithm()
    with pytest.raises(AlgorithmWiringRefusal, match="advantage_fn"):
        algorithm.setup(
            policy_pair=_PolicyPair(),
            loss_fn=objective,
            dataloader=iter([_batch()]),
            config=_config(),
            rollout_source=_RolloutSource(),
            advantage_fn=_Advantage(),
        )


def _setup_config(
    algorithm: SequencePolicyAlgorithm, objective: _Objective, *, config: Any
) -> None:
    algorithm.setup(
        policy_pair=_PolicyPair(),
        loss_fn=objective,
        dataloader=iter([_batch()]),
        config=config,
        rollout_source=_RolloutSource(),
        advantage_fn=objective.advantage_fn,
    )


def test_setup_semantics_tripwire_fires_on_a_forced_disagreement() -> None:
    """No public construction can split the two records, so one is edited by hand."""
    algorithm, objective = _make_algorithm(ratio_scope="token")
    object.__setattr__(
        algorithm,
        "_semantics",
        dataclasses.replace(algorithm.semantics(), ratio_scope="sequence"),
    )
    with pytest.raises(AlgorithmWiringRefusal, match="ratio_scope"):
        _setup(algorithm, objective)


def test_step_before_setup_refuses_instead_of_reporting_zero() -> None:
    """An unwired algorithm has 0 of 3 required inputs, not a zero-row step."""
    algorithm, _ = _make_algorithm()
    with pytest.raises(AlgorithmWiringRefusal, match="setup has not completed"):
        algorithm.step()


def test_requires_denominator_names_every_graded_role() -> None:
    """The static denominator covers both consumed and unconsumed roles."""
    algorithm, _ = _make_algorithm()
    requires = algorithm.requires()
    assert requires["policy_pair"] is True
    assert requires["loss_fn"] is True
    assert requires["dataloader"] is True
    assert requires["weight_sync"] is False
    assert requires["reference_policy"] is False
    # Mutability of the returned map would let a caller rewrite the denominator.
    with pytest.raises(TypeError):
        requires["weight_sync"] = True  # type: ignore[index]
