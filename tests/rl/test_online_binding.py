"""Binding tests for the online family: factories, derivation, and the contradiction refusal.

WHAT THIS MODULE PROVES: the four factories are zero-argument callable and bind a fresh
``OnlineAlgorithm`` carrying the shipped objective with the contract's default knobs, the
contract's ``RowsUnit`` and the contract's name; pairedness is derived from the objective's
own declared ``chosen_``/``rejected_`` schema (paired for OnlineDPO and IterativeDPO,
unpaired for RAFT and Best-of-N), never from the concrete class; the rows_unit/pairedness
contradiction refusal fires and names both sides; ``requirements()`` declares
``rollout_source`` False with every other optional role named False; a supplied-but-
unconsumed role is refused at setup; a full setup->step round trip reports ``rows`` as the
batch length on BOTH the paired and the unpaired arm, with the StepReport carrying the
objective's own declared components and metrics; and an object missing any of the five
protocol members is refused at construction with the absent members named.

WHAT THIS MODULE DOES NOT PROVE: the loss arithmetic of the four shipped objectives (their
own suites own that); that any sample actually came from the current policy -- only the
producer knows that, and the binding's role map honestly says so; or the exact prose of any
refusal beyond the contract-mandated denominator and named sides.

CONTROL ARMS: every detector leg is paired with a leg that must NOT fire -- matched
rows_unit/pairedness combinations construct cleanly beside both mismatch directions; the
paired forward-shape leg is mirrored by an unpaired leg that must NOT zip; the four real
round trips stand beside wrong-config refusals proving the derived pairedness drove key
selection; and the Best-of-N leg's ``rows == 2`` against ONE priced winner separates the
declared batch-length denominator from the SAMPLED_COMPLETION selection count.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from foundationscale.rl.algorithm import (
    AlgorithmSemantics,
    AlgorithmWiringRefusal,
    StepReport,
)
from foundationscale.rl.interfaces import (
    ExperienceBatch,
    ForwardFn,
    LossComponent,
    LossOutput,
    MetricObservation,
)
from foundationscale.rl.online import (
    OnlineAlgorithm,
    RowsUnit,
    best_of_n_algorithm,
    iterative_dpo_algorithm,
    online_dpo_algorithm,
    raft_algorithm,
)
from foundationscale.rl.online_objectives import (
    BestOfNLoss,
    IterativeDPOLoss,
    OnlineDPOLoss,
    RAFTLoss,
)
from foundationscale.rl.policy import PolicyPair

_PAIRED_CONFIG: dict[str, str] = {
    "policy_chosen_logprob_column": "policy_chosen",
    "policy_rejected_logprob_column": "policy_rejected",
}
_UNPAIRED_CONFIG: dict[str, str] = {"policy_logprob_column": "policy_single"}


class _FakeOnlineObjective:
    """Structural ``OnlineObjective`` fake whose schema declares its own pairedness.

    The binding reads only what the protocol declares -- ``required_columns`` for
    pairedness and the batch schema, ``__call__`` for the price -- so a fake that
    records what the binding's forward closure returns exercises the derivation
    without re-pricing the shipped arithmetic. ``declaration()`` is delegated to a
    real shipped loss so construction sees a genuine component/metric record. The
    returned ``LossOutput`` carries one real component and one real metric because
    ``StepReport`` refuses a step whose objective decomposed into zero components.
    """

    def __init__(
        self,
        *,
        anchor: Any,
        reference_free: bool,
        required_columns: tuple[str, ...],
    ) -> None:
        self._anchor = anchor
        self._reference_free = reference_free
        self._required_columns = required_columns
        self.forward_calls: list[Any] = []
        self.last_output: LossOutput | None = None

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        self.forward_calls.append(forward_fn(batch))
        price = 0.5 * len(self.forward_calls)
        self.last_output = LossOutput(
            loss=price,
            components=(
                LossComponent(
                    name="online_objective",
                    weight=1.0,
                    observed=True,
                    contribution=price,
                ),
            ),
            metrics=(
                MetricObservation(
                    name="batches_priced",
                    value=float(len(self.forward_calls)),
                ),
            ),
        )
        return self.last_output

    def declaration(self) -> Any:
        return self._anchor.declaration()

    @property
    def reference_free(self) -> bool:
        return self._reference_free

    @property
    def required_columns(self) -> tuple[str, ...]:
        return self._required_columns

    def semantics(self) -> AlgorithmSemantics:
        return AlgorithmSemantics(reference_free=self._reference_free)


class _ObjectiveWithoutRequiredColumns:
    """Structural fake exposing four of the five members the protocol demands."""

    @property
    def reference_free(self) -> bool:
        return True

    def declaration(self) -> Any:
        raise AssertionError("the slot smoke test must fire before any member is read")

    def semantics(self) -> Any:
        raise AssertionError("the slot smoke test must fire before any member is read")

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a structurally refused objective is never priced")


class _HalfSchemaOnlineObjective:
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


def _pair() -> PolicyPair:
    return PolicyPair(train_view=object(), generate_view=None, references={})


def _wire(
    objective: _FakeOnlineObjective,
    *,
    name: str,
    rows_unit: RowsUnit,
    config: dict[str, str],
    dataloader: list[ExperienceBatch],
) -> OnlineAlgorithm:
    algorithm = OnlineAlgorithm(name=name, objective=objective, rows_unit=rows_unit)
    algorithm.setup(
        policy_pair=_pair(),
        loss_fn=objective,
        dataloader=dataloader,
        config=config,
    )
    return algorithm


def test_rows_unit_declares_exactly_the_three_contract_members() -> None:
    # The enum IS the explicit declaration of what one row counts, which the
    # schema cannot state; any fourth unit would change the family silently.
    assert {member.value for member in RowsUnit} == {
        "preference_pair",
        "surviving_completion",
        "sampled_completion",
    }
    assert RowsUnit.PREFERENCE_PAIR.value == "preference_pair"
    assert RowsUnit.SURVIVING_COMPLETION.value == "surviving_completion"
    assert RowsUnit.SAMPLED_COMPLETION.value == "sampled_completion"


def test_init_refuses_a_non_objective_and_names_every_absent_member() -> None:
    # object() exposes none of the five protocol members; a refusal that named
    # fewer would understate how far the offered object is from conforming. The
    # check is a structural smoke test -- @runtime_checkable measures member
    # PRESENCE only -- so the honest thing to measure is exactly that.
    with pytest.raises(
        AlgorithmWiringRefusal,
        match=r"1 of 1 objective slots must carry",
    ) as excinfo:
        OnlineAlgorithm(
            name="rogue",
            objective=object(),
            rows_unit=RowsUnit.PREFERENCE_PAIR,
        )
    message = str(excinfo.value)
    assert "OnlineObjective" in message
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
    # If the diagnostic stopped enumerating honestly it would list members the
    # object actually carries, and the caller would fix the wrong thing.
    with pytest.raises(
        AlgorithmWiringRefusal,
        match=r"\('required_columns',\)",
    ) as excinfo:
        OnlineAlgorithm(
            name="rogue",
            objective=_ObjectiveWithoutRequiredColumns(),
            rows_unit=RowsUnit.SURVIVING_COMPLETION,
        )
    message = str(excinfo.value)
    assert "_ObjectiveWithoutRequiredColumns does not expose" in message
    assert "'reference_free'" not in message
    assert "'declaration'" not in message
    assert "of the 5 required members" in message


def test_init_refuses_a_half_declared_pair_schema_before_any_contradiction_check() -> None:
    # Pairedness is derived from the objective's own schema, never its class, so
    # a schema naming the chosen side without the rejected side is refused rather
    # than guessed -- and it must be refused BEFORE the rows_unit comparison can
    # run, because there is no derived pairedness to compare yet.
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="1 of 2 sides is absent",
    ) as excinfo:
        OnlineAlgorithm(
            name="rogue",
            objective=_HalfSchemaOnlineObjective(),
            rows_unit=RowsUnit.PREFERENCE_PAIR,
        )
    message = str(excinfo.value)
    assert "names 1 chosen and 0 rejected column(s)" in message


_FACTORY_CASES = (
    (
        online_dpo_algorithm,
        "online_dpo",
        OnlineDPOLoss,
        {"beta": 0.1, "weight": 1.0},
        RowsUnit.PREFERENCE_PAIR,
        False,
    ),
    (
        iterative_dpo_algorithm,
        "iterative_dpo",
        IterativeDPOLoss,
        {"beta": 0.1, "weight": 1.0},
        RowsUnit.PREFERENCE_PAIR,
        False,
    ),
    (
        raft_algorithm,
        "raft",
        RAFTLoss,
        {"weight": 1.0},
        RowsUnit.SURVIVING_COMPLETION,
        True,
    ),
    (
        best_of_n_algorithm,
        "best_of_n",
        BestOfNLoss,
        {"weight": 1.0},
        RowsUnit.SAMPLED_COMPLETION,
        True,
    ),
)


@pytest.mark.parametrize(
    ("factory", "expected_name", "objective_type", "default_knobs", "unit", "reference_free"),
    _FACTORY_CASES,
    ids=("online_dpo", "iterative_dpo", "raft", "best_of_n"),
)
def test_factory_binds_a_fresh_algorithm_carrying_the_declared_unit_and_objective(
    factory: Callable[[], OnlineAlgorithm],
    expected_name: str,
    objective_type: type,
    default_knobs: dict[str, Any],
    unit: RowsUnit,
    reference_free: bool,
) -> None:
    # Calling with zero arguments IS the registry's call shape: registry.py
    # registers the factory itself, so every knob must default and two calls
    # must not share the objective, or one run's wiring leaks into another's
    # pricing. The carried slots have no public reader; they are read where
    # the claim lives, exactly as the preference exemplar reads _objective.
    first = factory()
    second = factory()
    assert isinstance(first, OnlineAlgorithm)
    assert first is not second
    assert first.requirements().name == expected_name
    first_objective = first._objective
    assert first_objective is not second._objective
    assert type(first_objective) is objective_type
    for knob, expected in default_knobs.items():
        assert getattr(first_objective, knob) == expected
    assert first.rows_unit is unit
    assert second.rows_unit is unit
    assert first.semantics().reference_free is reference_free


_OVERRIDE_CASES = (
    (online_dpo_algorithm, {"beta": 0.7, "weight": 2.0}, "online_dpo"),
    (iterative_dpo_algorithm, {"beta": 0.25, "weight": 2.0}, "iterative_dpo"),
    (raft_algorithm, {"weight": 0.5}, "raft"),
    (best_of_n_algorithm, {"weight": 0.5}, "best_of_n"),
)


@pytest.mark.parametrize(
    ("factory", "overrides", "expected_name"),
    _OVERRIDE_CASES,
    ids=("online_dpo", "iterative_dpo", "raft", "best_of_n"),
)
def test_factory_threads_every_keyword_argument_and_keeps_its_bound_name_and_unit(
    factory: Callable[..., OnlineAlgorithm],
    overrides: dict[str, Any],
    expected_name: str,
) -> None:
    # An override must arrive verbatim off the default path -- a factory that
    # silently substituted its default would price a tuned run with
    # hyperparameters the caller never chose -- and the override must not
    # disturb the two things the factory still OWNS: the name and the unit.
    expected_unit = {
        "online_dpo": RowsUnit.PREFERENCE_PAIR,
        "iterative_dpo": RowsUnit.PREFERENCE_PAIR,
        "raft": RowsUnit.SURVIVING_COMPLETION,
        "best_of_n": RowsUnit.SAMPLED_COMPLETION,
    }[expected_name]
    algorithm = factory(**overrides)
    assert algorithm.requirements().name == expected_name
    assert algorithm.rows_unit is expected_unit
    for knob, expected in overrides.items():
        assert getattr(algorithm._objective, knob) == expected


@pytest.mark.parametrize(
    ("factory", "expected_name", "objective_type", "default_knobs", "unit", "reference_free"),
    _FACTORY_CASES,
    ids=("online_dpo", "iterative_dpo", "raft", "best_of_n"),
)
def test_requirements_and_semantics_are_derived_from_the_carried_objective(
    factory: Callable[[], OnlineAlgorithm],
    expected_name: str,
    objective_type: type,
    default_knobs: dict[str, Any],
    unit: RowsUnit,
    reference_free: bool,
) -> None:
    algorithm = factory()
    requirements = algorithm.requirements()
    assert requirements.name == expected_name
    # The online role map names every unconsumed role with False --
    # rollout_source included, because the binding PRICES A BATCH and who
    # produced the batch is upstream; an unnamed role is an ungraded role.
    assert dict(requirements.requires) == {
        "advantage_fn": False,
        "reference_policy": False,
        "rollout_source": False,
        "weight_sync": False,
    }
    # The binding claims its components and metrics come straight off the
    # objective's own declaration; measuring them against that declaration
    # here is the only place drift could be caught at construction time, and
    # a fresh default objective pins the factory's whole translation.
    declaration = algorithm._objective.declaration()
    assert declaration == objective_type().declaration()
    assert requirements.declared_components == declaration.components
    assert requirements.declared_metrics == tuple(metric.name for metric in declaration.metrics)
    semantics = algorithm.semantics()
    assert semantics is requirements.semantics
    assert semantics.reference_free is reference_free
    # The other four fields must abstain rather than pose measured values on
    # seams the online family does not constrain.
    assert semantics.group_size is None
    assert semantics.ratio_scope is None
    assert semantics.kl_estimator is None
    assert semantics.clip_bounds is None


@pytest.mark.parametrize(
    ("rows_unit", "derived_side"),
    [
        (RowsUnit.SURVIVING_COMPLETION, "paired"),
        (RowsUnit.SAMPLED_COMPLETION, "paired"),
    ],
    ids=["paired-schema-with-surviving-unit", "paired-schema-with-sampled-unit"],
)
def test_init_refuses_a_pairing_unit_contradiction_and_names_both_sides(
    rows_unit: RowsUnit,
    derived_side: str,
) -> None:
    # A paired schema with a non-pair rows_unit is a refusal, not a rescue:
    # the schema says one row is a (chosen, rejected) pair while the unit says
    # one row is one completion, and whichever side won silently would misstate
    # every downstream row count. The refusal must name BOTH sides so the
    # caller can see which declaration contradicted which.
    objective = _FakeOnlineObjective(
        anchor=OnlineDPOLoss(),
        reference_free=False,
        required_columns=("chosen_tokens", "rejected_tokens"),
    )
    with pytest.raises(AlgorithmWiringRefusal) as excinfo:
        OnlineAlgorithm(name="mismatched", objective=objective, rows_unit=rows_unit)
    message = str(excinfo.value)
    assert rows_unit.value in message
    assert derived_side in message
    assert RowsUnit.PREFERENCE_PAIR.value in message


def test_init_refuses_an_unpaired_schema_carrying_the_pairing_unit() -> None:
    # The reverse direction must fire too, or the detector is a one-sided
    # string match that happens to pass the legs above.
    objective = _FakeOnlineObjective(
        anchor=RAFTLoss(),
        reference_free=True,
        required_columns=("survivor_mask",),
    )
    with pytest.raises(AlgorithmWiringRefusal) as excinfo:
        OnlineAlgorithm(
            name="mismatched",
            objective=objective,
            rows_unit=RowsUnit.PREFERENCE_PAIR,
        )
    message = str(excinfo.value)
    assert RowsUnit.PREFERENCE_PAIR.value in message
    assert "paired" in message


@pytest.mark.parametrize(
    ("required_columns", "rows_unit"),
    [
        (("chosen_tokens", "rejected_tokens"), RowsUnit.PREFERENCE_PAIR),
        (("survivor_mask",), RowsUnit.SURVIVING_COMPLETION),
        (("sample_mask", "reward"), RowsUnit.SAMPLED_COMPLETION),
    ],
    ids=["paired-with-pair-unit", "unpaired-with-surviving-unit", "unpaired-with-sampled-unit"],
)
def test_init_accepts_every_matching_schema_and_unit_combination(
    required_columns: tuple[str, ...],
    rows_unit: RowsUnit,
) -> None:
    # The control arm of the contradiction detector: the same construction
    # path with consistent declarations must NOT refuse, or the refusal legs
    # above prove nothing -- a check that always fires also passes them.
    objective = _FakeOnlineObjective(
        anchor=OnlineDPOLoss(),
        reference_free=False,
        required_columns=required_columns,
    )
    algorithm = OnlineAlgorithm(name="matched", objective=objective, rows_unit=rows_unit)
    assert algorithm.rows_unit is rows_unit


def test_requirements_declares_rollout_source_unconsumed() -> None:
    # House convention (GRPO states it first): an Algorithm PRICES A BATCH and
    # who produced the batch is upstream, so rollout_source is declared False
    # even though this is the ONLINE family. The binding cannot verify the
    # samples came from the current policy -- only the producer knows that --
    # which is exactly why the role must stay unconsumed rather than half-used.
    requires = online_dpo_algorithm().requirements().requires
    assert requires["rollout_source"] is False
    assert set(requires) == {
        "advantage_fn",
        "reference_policy",
        "rollout_source",
        "weight_sync",
    }


@pytest.mark.parametrize(
    "surplus_role",
    ["advantage_fn", "rollout_source", "weight_sync"],
)
def test_setup_refuses_each_component_declared_unconsumed(surplus_role: str) -> None:
    # Supplied-but-unconsumed is refused, never silently ignored: a rollout
    # source handed to an algorithm that declared it False is a stub that
    # reads as a working online wiring. rollout_source is the load-bearing
    # leg; the other two prove the refusal is the shared role check, not a
    # special case.
    objective = OnlineDPOLoss()
    algorithm = OnlineAlgorithm(
        name="online_dpo",
        objective=objective,
        rows_unit=RowsUnit.PREFERENCE_PAIR,
    )
    kwargs: dict[str, Any] = {
        "policy_pair": _pair(),
        "loss_fn": objective,
        "dataloader": [],
        "config": _PAIRED_CONFIG,
        surplus_role: object(),
    }
    with pytest.raises(AlgorithmWiringRefusal, match=surplus_role):
        algorithm.setup(**kwargs)


def test_requirements_keeps_declaring_rollout_source_unconsumed_after_the_refusal() -> None:
    # A refused supply must not rewrite the declaration: the role map is a
    # statement about what the algorithm CONSUMES, not about what was offered.
    algorithm = online_dpo_algorithm()
    requires_before = dict(algorithm.requirements().requires)
    objective = algorithm._objective
    kwargs: dict[str, Any] = {
        "policy_pair": _pair(),
        "loss_fn": objective,
        "dataloader": [],
        "config": _PAIRED_CONFIG,
        "rollout_source": object(),
    }
    with pytest.raises(AlgorithmWiringRefusal, match="rollout_source"):
        algorithm.setup(**kwargs)
    assert dict(algorithm.requirements().requires) == requires_before
    assert requires_before["rollout_source"] is False


def _paired_fake_objective() -> _FakeOnlineObjective:
    return _FakeOnlineObjective(
        anchor=OnlineDPOLoss(),
        reference_free=False,
        required_columns=("chosen_tokens", "rejected_tokens", "snapshot_margin"),
    )


def _unpaired_fake_objective() -> _FakeOnlineObjective:
    return _FakeOnlineObjective(
        anchor=RAFTLoss(),
        reference_free=True,
        required_columns=("survivor_mask",),
    )


def _fake_paired_batch() -> ExperienceBatch:
    return ExperienceBatch(
        columns={
            "chosen_tokens": [[11, 12], [13]],
            "rejected_tokens": [[21], [22, 23]],
            "snapshot_margin": [0.5, -0.25],
            "policy_chosen": [[-0.1], [-0.2]],
            "policy_rejected": [[-0.9], [-1.1]],
        }
    )


def _fake_unpaired_batch() -> ExperienceBatch:
    return ExperienceBatch(
        columns={
            "survivor_mask": [[1, 1], [1], [1]],
            "policy_single": [[-0.1], [-0.2], [-0.3]],
        }
    )


def test_paired_forward_passes_chosen_then_rejected_and_counts_pair_rows() -> None:
    objective = _paired_fake_objective()
    algorithm = _wire(
        objective,
        name="online_dpo",
        rows_unit=RowsUnit.PREFERENCE_PAIR,
        config=_PAIRED_CONFIG,
        dataloader=[_fake_paired_batch()],
    )
    report = algorithm.step()
    assert len(objective.forward_calls) == 1
    # Exact tuple-of-pairs asserts BOTH order and completeness: a closure
    # that swapped the config-named columns would flip the sign of every
    # online margin the objective prices.
    assert objective.forward_calls[0] == (
        ([-0.1], [-0.9]),
        ([-0.2], [-1.1]),
    )
    # One row is one preference pair, not two completions: two rows carry
    # four sequences between them.
    assert report.rows == 2
    assert report.loss is objective.last_output


def test_unpaired_forward_passes_the_raw_column_and_counts_completion_rows() -> None:
    objective = _unpaired_fake_objective()
    algorithm = _wire(
        objective,
        name="raft",
        rows_unit=RowsUnit.SURVIVING_COMPLETION,
        config=_UNPAIRED_CONFIG,
        dataloader=[_fake_unpaired_batch()],
    )
    report = algorithm.step()
    # The unpaired arm hands the objective the raw column with NO zip: the
    # paired arm's list-of-2-tuples shape here would be a silent re-pairing
    # of independent completions. This leg is the control arm of the paired
    # leg above -- the same wiring code must NOT pair when the schema says
    # unpaired.
    assert objective.forward_calls == [([-0.1], [-0.2], [-0.3])]
    assert report.rows == 3
    assert report.loss is objective.last_output


def _online_dpo_batch() -> ExperienceBatch:
    return ExperienceBatch(
        columns={
            "chosen_loss_mask": [[1, 1], [1]],
            "rejected_loss_mask": [[1], [1, 1]],
            "reference_chosen_logprob": [-2.0, -3.0],
            "reference_rejected_logprob": [-1.0, -0.5],
            "policy_chosen": [[-0.5, -0.6], [-0.7]],
            "policy_rejected": [[-1.5], [-1.6, -1.7]],
        }
    )


def _iterative_dpo_batch() -> ExperienceBatch:
    return ExperienceBatch(
        columns={
            "chosen_loss_mask": [[1, 1], [1]],
            "rejected_loss_mask": [[1], [1, 1]],
            "reference_chosen_logprob": [-2.0, -3.0],
            "reference_rejected_logprob": [-1.0, -0.5],
            "reference_snapshot_id": ["snap-a", "snap-a"],
            "policy_chosen": [[-0.5, -0.6], [-0.7]],
            "policy_rejected": [[-1.5], [-1.6, -1.7]],
        }
    )


def _raft_batch() -> ExperienceBatch:
    return ExperienceBatch(
        columns={
            "completion_loss_mask": [[1, 1], [1]],
            "policy_single": [[-0.2, -0.3], [-0.4]],
        }
    )


def _best_of_n_batch() -> ExperienceBatch:
    # One group of two rows with a unique argmax: exactly ONE winner is
    # priced while the batch carries TWO rows, so rows == 2 separates the
    # declared batch-length denominator from the selection count.
    return ExperienceBatch(
        columns={
            "completion_loss_mask": [[1], [1]],
            "reward": [1.0, 0.5],
            "group_id": ["g", "g"],
            "policy_single": [[-0.2], [-0.3]],
        }
    )


_ROUND_TRIP_CASES = (
    (online_dpo_algorithm, dict(_PAIRED_CONFIG), _online_dpo_batch, 2),
    (iterative_dpo_algorithm, dict(_PAIRED_CONFIG), _iterative_dpo_batch, 2),
    (raft_algorithm, dict(_UNPAIRED_CONFIG), _raft_batch, 2),
    (best_of_n_algorithm, dict(_UNPAIRED_CONFIG), _best_of_n_batch, 2),
)


@pytest.mark.parametrize(
    ("factory", "config", "batch_builder", "expected_rows"),
    _ROUND_TRIP_CASES,
    ids=("online_dpo", "iterative_dpo", "raft", "best_of_n"),
)
def test_factory_round_trip_reports_batch_length_and_declared_components_and_metrics(
    factory: Callable[[], OnlineAlgorithm],
    config: dict[str, str],
    batch_builder: Callable[[], ExperienceBatch],
    expected_rows: int,
) -> None:
    # The full setup->step path with the SHIPPED objective: the StepReport's
    # decomposition must be the objective's own declared components/metrics --
    # read off the requirements the binding derived at construction, and
    # independently off the objective's declaration, so neither side of the
    # derivation seam can drift without one comparison failing.
    algorithm = factory()
    objective = algorithm._objective
    algorithm.setup(
        policy_pair=_pair(),
        loss_fn=objective,
        dataloader=[batch_builder()],
        config=config,
    )
    report = algorithm.step()
    assert isinstance(report, StepReport)
    assert report.step == 0
    assert report.rows == expected_rows
    assert report.rows == len(batch_builder())
    requirements = algorithm.requirements()
    declared_metric_names = tuple(metric.name for metric in objective.declaration().metrics)
    assert requirements.declared_components == objective.declaration().components
    assert requirements.declared_metrics == declared_metric_names
    assert tuple(component.name for component in report.loss.components) == (
        requirements.declared_components
    )
    assert tuple(metric.name for metric in report.loss.metrics) == declared_metric_names
    # No advantage function and no weight sync are wired, so the optional
    # slots abstain rather than reporting fabricated measurements.
    assert report.reward_stats is None
    assert report.sync is None


@pytest.mark.parametrize(
    ("factory", "wrong_config", "refused_key"),
    [
        (
            online_dpo_algorithm,
            _UNPAIRED_CONFIG,
            "policy_chosen_logprob_column",
        ),
        (
            iterative_dpo_algorithm,
            _UNPAIRED_CONFIG,
            "policy_chosen_logprob_column",
        ),
        (
            raft_algorithm,
            _PAIRED_CONFIG,
            "policy_logprob_column",
        ),
        (
            best_of_n_algorithm,
            _PAIRED_CONFIG,
            "policy_logprob_column",
        ),
    ],
    ids=("online_dpo", "iterative_dpo", "raft", "best_of_n"),
)
def test_setup_refuses_the_config_shape_of_the_opposite_pairedness(
    factory: Callable[[], OnlineAlgorithm],
    wrong_config: dict[str, str],
    refused_key: str,
) -> None:
    # The derived pairedness must DRIVE the config-key selection: the two
    # paired members demand the chosen/rejected key pair (never the singular
    # unpaired key) and the two unpaired members demand the singular key
    # (never the pair). The successful legs above could also pass if the
    # binding lazily accepted whichever keys it was handed; these legs are
    # the control arm proving the acceptance was keyed on the derivation.
    algorithm = factory()
    with pytest.raises(AlgorithmWiringRefusal, match=refused_key):
        algorithm.setup(
            policy_pair=_pair(),
            loss_fn=algorithm._objective,
            dataloader=[],
            config=wrong_config,
        )


def test_step_before_setup_refuses_the_unwired_binding() -> None:
    # An unwired binding holds no dataloader and no schema; a zero report
    # here would read as a measured step that never happened.
    algorithm = OnlineAlgorithm(
        name="online_dpo",
        objective=_paired_fake_objective(),
        rows_unit=RowsUnit.PREFERENCE_PAIR,
    )
    with pytest.raises(AlgorithmWiringRefusal, match="setup has not completed"):
        algorithm.step()


def test_setup_refuses_a_loss_that_is_not_the_carried_objective_instance() -> None:
    # loss_fn must BE the carried objective -- identity, not equality; two
    # online objectives that agree everywhere still double the one declared
    # objective role.
    objective = OnlineDPOLoss()
    algorithm = OnlineAlgorithm(
        name="online_dpo",
        objective=objective,
        rows_unit=RowsUnit.PREFERENCE_PAIR,
    )
    with pytest.raises(AlgorithmWiringRefusal, match="2 distinct objects") as excinfo:
        algorithm.setup(
            policy_pair=_pair(),
            loss_fn=OnlineDPOLoss(),
            dataloader=[],
            config=_PAIRED_CONFIG,
        )
    assert "1 declared objective role" in str(excinfo.value)
