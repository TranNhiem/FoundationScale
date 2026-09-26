"""Coverage of the group-policy binding's success paths and late refusals.

Why this file exists
--------------------
The pre-existing ``test_group_policy.py`` pins construction, the factories,
the identity refusals and the tripwire -- but per the measured coverage map
(0c46636, suite-wide), every line of ``setup`` below the five-field
cross-check and every functional line of ``step`` executed in NO test: the
successful handshake, the dataloader/config/column refusals, and all three
of ``step``'s arms. A binding whose happy path is unmeasured can rot in
exactly the place callers depend on it, so these legs drive ``setup`` to
completion and then ``step`` over fabricated batches, and assert the
refusals that guard the unpriceable shapes. The doubles mirror the
neighbouring file's deliberately: recipes stay file-local in this suite so
a change here cannot silently move another file's denominators.

WHAT IS CLAIMED: setup accepts a consistent wiring and records the supplied
map; the refusals for double setup, non-iterable dataloader, non-Mapping
config, and a missing or empty ``policy_logprob_column`` fire with their
field named; and step prices exactly one batch through the objective via the
config-named column, reporting the batch's row count.

WHAT IS NOT CLAIMED: the objective arithmetic (owned by
``group_policy_objectives.py``) or the role-map arithmetic (owned by
``algorithm.py``) -- only the binding's own late seams are measured here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from foundationscale.rl.algorithm import (
    AlgorithmSemantics,
    AlgorithmWiringRefusal,
    StepReportRefusal,
)
from foundationscale.rl.group_policy import SequencePolicyAlgorithm
from foundationscale.rl.interfaces import BatchRefusal, LossComponent, LossOutput
from foundationscale.rl.policy import PolicyPair


class _Advantage:
    """Minimal advantage-estimator placeholder; identity is what setup measures."""

    def __call__(self, batch: Any) -> Any:
        return batch


class _Policy:
    """Structural stand-in for a view; the binding never calls it."""

    def forward(self, batch: Any) -> Any:
        return batch


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

    def __call__(self, forward_fn: Any, batch: _Batch) -> LossOutput:
        scores = forward_fn(batch)
        self.calls.append((scores, batch))
        value = -float(sum(scores))
        return LossOutput(
            loss=value,
            # One real component so the output is honest about what moved the
            # scalar; the binding only forwards it.
            components=(
                LossComponent(name="stub_policy", weight=1.0, observed=True, contribution=value),
            ),
        )

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


def _make_algorithm(
    **objective_kwargs: Any,
) -> tuple[SequencePolicyAlgorithm, _Objective]:
    objective = _Objective(**objective_kwargs)
    return SequencePolicyAlgorithm(name="stub_family", objective=objective), objective


def _setup(
    algorithm: SequencePolicyAlgorithm,
    objective: _Objective,
    *,
    batches: list[_Batch] | None = None,
    dataloader: Any = None,
    config: Any = None,
) -> None:
    # A REAL PolicyPair: the wiring handshake enforces the type by name --
    # "wired against a _PolicyPair rather than a PolicyPair" -- the neighbour
    # file's structural stub cannot cross this seam, which is exactly why the
    # happy path below was unmeasured in-repo before this file.
    payload = (
        iter(batches)
        if batches is not None
        else (dataloader if dataloader is not None else iter([_batch()]))
    )
    algorithm.setup(
        policy_pair=PolicyPair(train_view=_Policy(), generate_view=_Policy()),
        loss_fn=objective,
        dataloader=payload,
        config=_config() if config is None else config,
        rollout_source=_RolloutSource(),
        advantage_fn=objective.advantage_fn,
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


# --- setup: the successful handshake and the arms below the tripwire -------


def test_setup_completes_and_records_the_wired_supplied_map() -> None:
    """The happy path: handshake accepted, supplied() names exactly the wired roles.

    Until this leg existed, no test in the suite ever reached the second half
    of ``setup`` -- the role-map call, the dataloader iterator capture, the
    config schema, the recorded ``required_columns``, the group-policy
    denominator check and the wiring assignment were all unmeasured lines.
    """
    algorithm, objective = _make_algorithm()
    _setup(algorithm, objective)
    supplied = algorithm.supplied()
    assert supplied is not None, "post-setup supplied() must not abstain"
    assert supplied["policy_pair"] is not None
    assert supplied["loss_fn"] is objective
    assert supplied["advantage_fn"] is objective.advantage_fn
    assert "weight_sync" not in supplied, "the family consumes no weight sync"
    with pytest.raises(TypeError):
        supplied["loss_fn"] = None  # type: ignore[index]


def test_setup_twice_refuses_instead_of_replacing_the_measured_mapping() -> None:
    """A second setup would silently rewrite what the first one measured."""
    algorithm, objective = _make_algorithm()
    _setup(algorithm, objective)
    with pytest.raises(AlgorithmWiringRefusal, match="1 of 1 .* already wired"):
        _setup(algorithm, objective)


def test_setup_refuses_a_non_iterable_dataloader() -> None:
    """A dataloader that yields nothing iter()-able is a wiring fault, named by type."""
    algorithm, objective = _make_algorithm()
    with pytest.raises(AlgorithmWiringRefusal, match="not iterable .*object"):
        _setup(algorithm, objective, dataloader=object())


def test_setup_refuses_a_non_mapping_config() -> None:
    """The config must implement Mapping; a bare list cannot name a column."""
    algorithm, objective = _make_algorithm()
    with pytest.raises(AlgorithmWiringRefusal, match="must implement Mapping"):
        _setup(algorithm, objective, config=["policy_logprob_column"])


def test_setup_refuses_a_missing_policy_logprob_column() -> None:
    """Without the column name the current-policy scores cannot be attributed."""
    algorithm, objective = _make_algorithm()
    with pytest.raises(AlgorithmWiringRefusal, match="policy_logprob_column"):
        _setup(algorithm, objective, config={})


def test_setup_refuses_an_empty_policy_logprob_column() -> None:
    """An empty string is not a column name; absence-shaped values are refusals."""
    algorithm, objective = _make_algorithm()
    with pytest.raises(AlgorithmWiringRefusal, match="policy_logprob_column"):
        _setup(algorithm, objective, config={"policy_logprob_column": ""})


# --- step: the three arms --------------------------------------------------


def test_step_prices_one_batch_through_the_config_named_column() -> None:
    """One batch in, one report out: forward reads logprobs_new, rows == len(batch)."""
    algorithm, objective = _make_algorithm()
    first = _batch()
    _setup(algorithm, objective, batches=[first])

    report = algorithm.step()

    assert len(objective.calls) == 1, f"objective invoked {len(objective.calls)} times, expected 1"
    scores, seen_batch = objective.calls[0]
    assert seen_batch is first
    assert scores == first.column("logprobs_new"), (
        "the forward function must read the config-named column, not a hard-coded one"
    )
    assert report.loss.loss == -float(sum(scores)), (
        f"the report carries the objective's own LossOutput; got {report.loss!r}"
    )
    assert len(report.loss.components) == 1 and report.loss.components[0].name == "stub_policy"
    assert report.rows == 4, f"rows={report.rows}: the report owns the batch's row count"
    assert report.reward_stats is None and report.sync is None


def test_step_reports_successive_step_indices() -> None:
    """_next_step advances only on priced steps; the sequence is observable."""
    algorithm, objective = _make_algorithm()
    _setup(algorithm, objective, batches=[_batch(), _batch()])
    first_report = algorithm.step()
    second_report = algorithm.step()
    assert (first_report.step, second_report.step) == (0, 1)


def test_step_on_an_exhausted_dataloader_is_unmeasured_not_zero_rows() -> None:
    """StopIteration converts to StepReportRefusal: an absent batch is not a 0-row step."""
    algorithm, objective = _make_algorithm()
    _setup(algorithm, objective, batches=[_batch()])
    algorithm.step()
    with pytest.raises(StepReportRefusal, match="0 of at least 1 required batches"):
        algorithm.step()


def test_step_refuses_a_batch_missing_a_declared_column() -> None:
    """A mid-stream batch narrower than the handshake schema changes the denominator."""
    algorithm, objective = _make_algorithm()
    narrow = _batch(logprobs_old=[-0.5], logprobs_new=[-0.51])  # no "advantages"
    _setup(algorithm, objective, batches=[_batch(), narrow])
    algorithm.step()
    with pytest.raises(BatchRefusal, match="advantages"):
        algorithm.step()


EOF_MARKER_NOT_NEEDED = True
