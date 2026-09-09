"""Adversarial tests for ``PreferenceAlgorithm.step()``.

Four axes: refusal when the binding is unwired, the per-batch schema
measurement, the forward closure's shape on both the paired and the
unpaired arm, and the ``StepReport`` fields the binding actually populates.
"""

from __future__ import annotations

from typing import Any

import pytest

from foundationscale.rl.algorithm import (
    AlgorithmSemantics,
    AlgorithmWiringRefusal,
    StepReport,
    StepReportRefusal,
)
from foundationscale.rl.interfaces import (
    BatchRefusal,
    ExperienceBatch,
    ForwardFn,
    LossComponent,
    LossOutput,
    MetricObservation,
)
from foundationscale.rl.losses import DPOLoss
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.preference import PreferenceAlgorithm
from foundationscale.rl.preference_objectives import KTOLoss, ORPOLoss

_PAIRED_CONFIG: dict[str, str] = {
    "policy_chosen_logprob_column": "policy_chosen",
    "policy_rejected_logprob_column": "policy_rejected",
}
_UNPAIRED_CONFIG: dict[str, str] = {"policy_logprob_column": "policy_single"}


class _FakeObjective:
    """Structural ``PreferenceObjective`` fake.

    step() measures only what the protocol declares -- required_columns for
    the per-batch schema, pairedness from those column names, and __call__
    for the price -- so a fake that declares its own schema and records what
    the binding's forward closure returns exercises the binding without the
    loss arithmetic, which the objectives' own suites own. ``declaration()``
    is delegated to a real loss so setup's wiring handshake sees a genuine
    component/metric record.

    The returned ``LossOutput`` must carry at least one real component and
    one real metric, because ``StepReport.__post_init__`` refuses a step
    whose objective decomposed into ZERO components -- an empty tuple there
    would place the whole objective outside the coverage gate's denominator.
    A ``LossOutput`` object permits emptiness because a loss is not a step;
    these tests build steps, so the fake must report what a step requires.
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
        # A distinct price per call lets a multi-step test tell reports apart.
        price = 0.5 * len(self.forward_calls)
        self.last_output = LossOutput(
            loss=price,
            components=(
                LossComponent(
                    name="preference_objective",
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


def _anchored_objective() -> _FakeObjective:
    # Stands in for the reference-anchored kind (DPO/IPO/KTO): paired, with
    # one schema column carrying pre-computed reference scores.
    return _FakeObjective(
        anchor=DPOLoss(),
        reference_free=False,
        required_columns=("chosen_tokens", "rejected_tokens", "reference_margin"),
    )


def _reference_free_objective() -> _FakeObjective:
    return _FakeObjective(
        anchor=ORPOLoss(),
        reference_free=True,
        required_columns=("chosen_tokens", "rejected_tokens"),
    )


def _unpaired_objective() -> _FakeObjective:
    # No chosen_/rejected_ prefixes, so the binding must derive UNPAIRED.
    return _FakeObjective(
        anchor=KTOLoss(),
        reference_free=False,
        required_columns=("desirable_flag", "kl_reference"),
    )


def _pair() -> PolicyPair:
    return PolicyPair(train_view=object(), generate_view=None, references={})


def _paired_batch(*, drop: str | None = None) -> ExperienceBatch:
    columns: dict[str, list[Any]] = {
        "chosen_tokens": [[11, 12], [13]],
        "rejected_tokens": [[21], [22, 23]],
        "reference_margin": [0.5, -0.25],
        "policy_chosen": [[-0.1], [-0.2]],
        "policy_rejected": [[-0.9], [-1.1]],
    }
    if drop is not None:
        del columns[drop]
    return ExperienceBatch(columns=columns)


def _unpaired_batch() -> ExperienceBatch:
    return ExperienceBatch(
        columns={
            "desirable_flag": [True, False, True],
            "kl_reference": [0.05, 0.05, 0.05],
            "policy_single": [[-0.1], [-0.2], [-0.3]],
        }
    )


def _wire(
    objective: _FakeObjective,
    *,
    name: str,
    config: dict[str, str],
    dataloader: list[ExperienceBatch],
) -> PreferenceAlgorithm:
    algorithm = PreferenceAlgorithm(name=name, objective=objective)
    algorithm.setup(
        policy_pair=_pair(),
        loss_fn=objective,
        dataloader=dataloader,
        config=config,
    )
    return algorithm


def test_step_before_setup_refuses_the_unwired_binding() -> None:
    # An unwired binding holds no dataloader and no schema; a zero report
    # here would read as a measured step that never happened.
    algorithm = PreferenceAlgorithm(name="dpo", objective=_anchored_objective())
    with pytest.raises(AlgorithmWiringRefusal, match="setup has not completed") as excinfo:
        algorithm.step()
    assert "0 of 3 required inputs are present for dpo" in str(excinfo.value)


def test_step_reports_the_anchored_objective_output_and_the_pair_row_count() -> None:
    objective = _anchored_objective()
    algorithm = _wire(
        objective,
        name="dpo",
        config=_PAIRED_CONFIG,
        dataloader=[_paired_batch()],
    )
    report = algorithm.step()
    assert isinstance(report, StepReport)
    assert report.step == 0
    # One row is one preference pair, not two completions: the batch's two
    # rows carry four sequences between them.
    assert report.rows == 2
    assert report.loss is objective.last_output
    assert report.loss.loss == 0.5
    # The step's report must decompose into at least one component -- the
    # StepReport gate refuses an empty decomposition -- so the fake declares
    # exactly one and the leg pins its name and value, not just its presence.
    assert [(component.name, component.contribution) for component in report.loss.components] == [
        ("preference_objective", 0.5)
    ]
    assert [(metric.name, metric.value) for metric in report.loss.metrics] == [
        ("batches_priced", 1.0)
    ]
    # No advantage function and no weight sync are wired, so both optional
    # slots abstain rather than reporting fabricated measurements.
    assert report.reward_stats is None
    assert report.sync is None


def test_step_builds_a_paired_forward_passing_chosen_then_rejected() -> None:
    objective = _reference_free_objective()
    algorithm = _wire(
        objective,
        name="orpo",
        config=_PAIRED_CONFIG,
        dataloader=[_paired_batch()],
    )
    report = algorithm.step()
    assert len(objective.forward_calls) == 1
    # Exact tuple-of-pairs asserts BOTH order and completeness: a closure
    # that swapped the config-named columns, or returned a single column,
    # would flip the sign of every margin the objective prices.
    assert objective.forward_calls[0] == (
        ([-0.1], [-0.9]),
        ([-0.2], [-1.1]),
    )
    assert report.rows == 2
    assert report.loss is objective.last_output


def test_step_builds_an_unpaired_forward_reading_the_single_column() -> None:
    objective = _unpaired_objective()
    algorithm = _wire(
        objective,
        name="kto",
        config=_UNPAIRED_CONFIG,
        dataloader=[_unpaired_batch()],
    )
    report = algorithm.step()
    # The unpaired arm must hand the objective the raw column -- no zip, no
    # row pairing -- and must not demand the two paired config keys instead.
    # ExperienceBatch.column returns a tuple, and the unpaired closure passes
    # it through unchanged, so what the objective sees IS the column: the
    # paired arm's list-of-2-tuples shape would be a silent re-pairing.
    assert objective.forward_calls == [([-0.1], [-0.2], [-0.3])]
    # Under the unpaired shape one row is one flagged singleton completion.
    assert report.rows == 3


def test_step_numbers_successive_reports_and_prices_each_batch_once() -> None:
    objective = _anchored_objective()
    algorithm = _wire(
        objective,
        name="dpo",
        config=_PAIRED_CONFIG,
        dataloader=[_paired_batch(), _paired_batch()],
    )
    first = algorithm.step()
    second = algorithm.step()
    # Distinct per-call prices prove both batches were priced exactly once
    # and in order -- a binding that replayed the first batch, or skipped a
    # step index, shows up here. The metric carries the same counter, so a
    # fake that rebuilt its output object between calls would diverge.
    assert (first.step, second.step) == (0, 1)
    assert (first.loss.loss, second.loss.loss) == (0.5, 1.0)
    assert (
        first.loss.metrics[0].value,
        second.loss.metrics[0].value,
    ) == (1.0, 2.0)
    assert len(objective.forward_calls) == 2


def test_step_refuses_an_exhausted_dataloader_and_chains_stop_iteration() -> None:
    objective = _anchored_objective()
    algorithm = _wire(
        objective,
        name="dpo",
        config=_PAIRED_CONFIG,
        dataloader=[_paired_batch()],
    )
    algorithm.step()
    with pytest.raises(
        StepReportRefusal,
        match="0 of at least 1 required batches remain for dpo",
    ) as excinfo:
        algorithm.step()
    assert "the step is unmeasured rather than a step with zero rows" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, StopIteration)


@pytest.mark.parametrize(
    "dropped",
    ["reference_margin", "policy_rejected"],
    ids=["objective-column", "policy-column"],
)
def test_step_refuses_a_later_batch_missing_a_declared_column(dropped: str) -> None:
    # Setup records the schema but measures no data, so this per-step check
    # is the only thing refusing a batch whose schema changed mid-stream --
    # a column present in the first batch and absent now is a changed
    # denominator, and pricing it would silently attribute None scores.
    objective = _anchored_objective()
    algorithm = _wire(
        objective,
        name="dpo",
        config=_PAIRED_CONFIG,
        dataloader=[_paired_batch(), _paired_batch(drop=dropped)],
    )
    first = algorithm.step()
    assert first.step == 0
    with pytest.raises(
        BatchRefusal,
        match=rf"1 of 5 required batch columns are absent \({dropped}\)",
    ) as excinfo:
        algorithm.step()
    assert "step 1 of dpo" in str(excinfo.value)
