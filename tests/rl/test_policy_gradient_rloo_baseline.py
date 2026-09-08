"""Hostile-branch tests for the RLOO and REINFORCE-with-baseline family seams."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from foundationscale.rl.advantage import AdvantageRefusal, LeaveOneOutAdvantage
from foundationscale.rl.algorithm import (
    AlgorithmSemantics,
    AlgorithmWiringRefusal,
    StepReport,
    StepReportRefusal,
    verify_step,
)
from foundationscale.rl.interfaces import (
    BatchRefusal,
    LossConfigRefusal,
)
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.policy_gradient import (
    ReinforceBaselineAlgorithm,
    ReinforceBaselineLoss,
    RLOOAlgorithm,
    RLOOPolicyLoss,
)

__all__ = ()


@dataclass(frozen=True, slots=True)
class _Batch:
    """Minimal pure-Python ExperienceBatch stand-in over a column mapping."""

    source: Mapping[str, Any]

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.source)

    def column(self, name: str) -> Any:
        return self.source[name]

    def __len__(self) -> int:
        if not self.source:
            return 0
        return len(next(iter(self.source.values())))


def _forward_with_rows(
    rows: tuple[tuple[float, ...], ...],
) -> Callable[[_Batch], tuple[tuple[float, ...], ...]]:
    def current_logprobs(_batch: _Batch) -> tuple[tuple[float, ...], ...]:
        return rows

    return current_logprobs


def _rloo_batch(
    *,
    prompt_ids: tuple[Any, ...] = ("prompt-a", "prompt-a"),
    rewards: tuple[Any, ...] = (2.0, 4.0),
    loss_mask: tuple[tuple[int, ...], ...] = ((1, 1), (1, 1)),
    old_logprobs: tuple[tuple[float, ...], ...] = ((-1.0, -1.0), (-1.0, -1.0)),
    current_logprobs: tuple[tuple[float, ...], ...] | None = None,
) -> _Batch:
    if current_logprobs is None:
        current_logprobs = old_logprobs
    return _Batch(
        {
            "prompt_ids": prompt_ids,
            "rewards": rewards,
            "loss_mask": loss_mask,
            "old_logprobs": old_logprobs,
            "current_logprobs": current_logprobs,
        }
    )


def _reinforce_batch(*, rewards: tuple[float, float]) -> _Batch:
    return _Batch(
        {
            "rewards": rewards,
            "loss_mask": ((1,), (1,)),
            "current_logprobs": ((-1.0,), (-2.0,)),
        }
    )


class _LyingOfferedEstimator(LeaveOneOutAdvantage):
    """Real estimator instance whose compute lies about the offered count."""

    def compute(self, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(offered=0)


class _ForeignSemanticsLoss(ReinforceBaselineLoss):
    """A correctly typed loss whose semantics declaration disagrees upstream."""

    def semantics(self) -> AlgorithmSemantics:
        return AlgorithmSemantics(
            group_size=1,
            ratio_scope=None,
            kl_estimator=None,
            clip_bounds=None,
            reference_free=True,
        )


def test_rloo_semantics_and_requires_declarations() -> None:
    """The one-line declaration accessors return the constructor's own records.

    WHAT IS CLAIMED: ``semantics()`` reports group size 2, token ratio scope,
    abstained KL and clip seams, and reference-freeness; ``requires()``
    declares the four mandatory-and-optional roles with exactly the booleans
    the constructor wrote.

    WHAT IS NOT CLAIMED: that these declarations agree with any loss; that
    agreement is the setup handshake's business, not this accessor's.
    """
    algorithm = RLOOAlgorithm()
    semantics = algorithm.semantics()
    assert semantics.group_size == 2
    assert semantics.ratio_scope == "token"
    assert semantics.kl_estimator is None
    assert semantics.clip_bounds is None
    assert semantics.reference_free is True
    requires = algorithm.requires()
    assert requires["policy_pair"] is True
    assert requires["loss_fn"] is True
    assert requires["dataloader"] is True
    assert requires["advantage_fn"] is True
    assert requires["rollout_source"] is False
    assert requires["weight_sync"] is False


def test_rloo_post_init_refuses_non_estimator_and_empty_column_name() -> None:
    """Both remaining constructor guards refuse, naming their own field.

    WHAT IS CLAIMED: a foreign advantage object refuses with the estimator
    slot message, and an empty column name refuses with the non-empty-string
    message -- each before any batch is ever read.

    WHAT IS NOT CLAIMED: that a foreign estimator would compute wrong
    numbers; type identity, not behaviour, is what is refused.
    """
    with pytest.raises(LossConfigRefusal, match="must carry that concrete estimator"):
        RLOOPolicyLoss(advantage_fn=object())
    with pytest.raises(LossConfigRefusal, match="non-empty strings"):
        RLOOPolicyLoss(old_logprob_column="")


def test_rloo_call_returns_the_declared_loss_output() -> None:
    """The ``LossFn`` surface call prices the batch and drops the report side.

    Hand computation: advantages (-2, +2), all ratios 1, three supervised
    tokens, so the contribution is -(-2 - 2 + 2) / 3 = 2/3.

    WHAT IS CLAIMED: ``__call__`` returns exactly the ``LossOutput`` half of
    ``compute_with_report`` carrying the one declared component.

    WHAT IS NOT CLAIMED: that the estimator denominator was preserved; the
    protocol call shape intentionally discards it.
    """
    loss_fn = RLOOPolicyLoss()
    batch = _rloo_batch(loss_mask=((1, 1), (1,)), old_logprobs=((-1.0, -1.0), (-1.0,)))
    output = loss_fn(_forward_with_rows(((-1.0, -1.0), (-1.0,))), batch)
    assert output.loss == pytest.approx(2.0 / 3.0)
    assert len(output.components) == 1
    assert output.components[0].name == "rloo_policy_loss"


def test_rloo_mixed_mask_prices_only_supervised_positions() -> None:
    """An unsupervised token position contributes nothing and is skipped.

    Hand computation: advantages (-2, +2); with row 0 masked to one token the
    supervised set is {-2, +2, +2}, totalling 2 over 3 tokens, so the
    contribution is -2/3. The same result holds whether the estimator
    broadcasts its advantage across the masked slot or zeroes it.

    WHAT IS CLAIMED: the loss matches the hand value, which is only possible
    if the masked position was skipped rather than priced.

    WHAT IS NOT CLAIMED: how the estimator encodes the masked slot in its
    weights row; the test is designed so both encodings agree numerically.
    """
    loss_fn = RLOOPolicyLoss()
    batch = _rloo_batch(loss_mask=((1, 0), (1, 1)))
    output, advantage = loss_fn.compute_with_report(
        _forward_with_rows(((-1.0, -1.0), (-1.0, -1.0))),
        batch,
    )
    assert advantage.used == 2
    assert output.loss == pytest.approx(-2.0 / 3.0)


def test_rloo_refuses_missing_column_and_empty_batch() -> None:
    """Both batch-shape guardrails refuse before any estimator call.

    WHAT IS CLAIMED: a batch without the old-logprob column refuses naming
    the absent input, and a zero-row batch with all columns present refuses
    rather than reporting a 0.0 mean over nothing.

    WHAT IS NOT CLAIMED: which refusal fires when both defects are present;
    each test isolates exactly one defect.
    """
    loss_fn = RLOOPolicyLoss()
    missing = _Batch(
        {
            "prompt_ids": ("p", "p"),
            "rewards": (2.0, 4.0),
            "loss_mask": ((1, 1), (1, 1)),
        }
    )
    with pytest.raises(BatchRefusal, match="required inputs absent for rloo"):
        loss_fn.compute_with_report(_forward_with_rows(()), missing)
    empty = _Batch(
        {
            "prompt_ids": (),
            "rewards": (),
            "loss_mask": (),
            "old_logprobs": (),
            "current_logprobs": (),
        }
    )
    with pytest.raises(BatchRefusal, match="0 of at least 1 required batch rows"):
        loss_fn.compute_with_report(_forward_with_rows(()), empty)


def test_rloo_refuses_column_count_mismatch() -> None:
    """A column shorter than the offered batch refuses on the count.

    WHAT IS CLAIMED: one rewards entry against two batch rows refuses with
    the one-offered-denominator message before any grouping happens.

    WHAT IS NOT CLAIMED: that the other three columns behave identically;
    only the rewards leg is exercised here.
    """
    batch = _Batch(
        {
            "prompt_ids": ("p", "p"),
            "rewards": (2.0,),
            "loss_mask": ((1, 1), (1, 1)),
            "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
        }
    )
    with pytest.raises(BatchRefusal, match="must be the one offered denominator"):
        RLOOPolicyLoss().compute_with_report(_forward_with_rows(()), batch)


def test_rloo_refuses_unhashable_prompt_id() -> None:
    """A prompt id that cannot group cannot anchor a leave-one-out baseline.

    WHAT IS CLAIMED: a list prompt id refuses at grouping time, naming the
    row and the unhashable type.

    WHAT IS NOT CLAIMED: that equally unhashable types differ in any way;
    only the list shape is exercised.
    """
    batch = _rloo_batch(prompt_ids=(["a"], ["a"]))
    with pytest.raises(BatchRefusal, match=re.escape("not hashable (list)")):
        RLOOPolicyLoss().compute_with_report(_forward_with_rows(()), batch)


def test_rloo_refuses_prompt_group_disagreeing_with_declared_k() -> None:
    """A prompt group smaller than the declared group size refuses outright.

    WHAT IS CLAIMED: with group_size 2, a singleton prompt group refuses
    before the estimator runs, naming the prompt and the declared K.

    WHAT IS NOT CLAIMED: that the estimator would have noticed on its own;
    the loss performs its own refusal precisely so the estimator never sees
    it.
    """
    batch = _rloo_batch(
        prompt_ids=("a", "a", "b"),
        rewards=(1.0, 2.0, 3.0),
        loss_mask=((1,), (1,), (1,)),
        old_logprobs=((-1.0,), (-1.0,), (-1.0,)),
    )
    with pytest.raises(BatchRefusal, match="disagrees with 1 declared K"):
        RLOOPolicyLoss().compute_with_report(_forward_with_rows(()), batch)


def test_rloo_refuses_estimator_report_anchored_to_wrong_offered_count() -> None:
    """An estimator report whose offered count is not the batch size refuses.

    WHAT IS CLAIMED: a report claiming zero offered rows against a two-row
    batch refuses with the cannot-anchor message -- the defensive lie is
    manufactured by a real estimator subclass so the type guard still passes.

    WHAT IS NOT CLAIMED: that the real estimator can produce this; its compute
    is honest, and the check exists for a corrupted intermediate, which this
    test stands in for.
    """
    loss_fn = RLOOPolicyLoss(advantage_fn=_LyingOfferedEstimator())
    with pytest.raises(BatchRefusal, match="cannot anchor an RLOO step"):
        loss_fn.compute_with_report(
            _forward_with_rows(((-1.0, -1.0), (-1.0, -1.0))),
            _rloo_batch(),
        )


def test_rloo_refuses_forward_row_count_mismatch() -> None:
    """Current log-probabilities must arrive one token row per batch row.

    WHAT IS CLAIMED: a forward pass returning one row for a two-row batch
    refuses before any token is priced.

    WHAT IS NOT CLAIMED: that additional rows would be equally refused; only
    the short side is exercised.
    """
    with pytest.raises(BatchRefusal, match="one token row per batch row is required"):
        RLOOPolicyLoss().compute_with_report(
            _forward_with_rows(((-1.0, -1.0),)),
            _rloo_batch(),
        )


def test_rloo_refuses_token_denominator_disagreement() -> None:
    """Mask, advantage, current, and old lengths must be one shared denominator.

    WHAT IS CLAIMED: an old-logprob row short by one token refuses with the
    four-denominators message, naming the row and all four lengths.

    WHAT IS NOT CLAIMED: which of the four counts is 'right'; the refusal is
    on their disagreement, not on any absolute length.
    """
    batch = _rloo_batch(old_logprobs=((-1.0, -1.0), (-1.0,)))
    with pytest.raises(BatchRefusal, match="all 4 token denominators must be equal"):
        RLOOPolicyLoss().compute_with_report(
            _forward_with_rows(((-1.0, -1.0), (-1.0, -1.0))),
            batch,
        )


def test_rloo_refuses_unrepresentable_token_ratio() -> None:
    """A token ratio that overflows exp() refuses rather than reading as huge.

    WHAT IS CLAIMED: a log-ratio of 801 refuses as not representable before
    any use of the ratio.

    WHAT IS NOT CLAIMED: where the enormous log-probability gap came from;
    the refusal is about representability, not provenance.
    """
    batch = _rloo_batch(loss_mask=((1,), (1,)), old_logprobs=((-1.0,), (-1.0,)))
    with pytest.raises(BatchRefusal, match="is not representable"):
        RLOOPolicyLoss().compute_with_report(
            _forward_with_rows(((800.0,), (-1.0,))),
            batch,
        )


def test_rloo_refuses_zero_supervised_tokens() -> None:
    """An all-masked batch is an unmeasured RLOO loss, never a 0.0.

    WHAT IS CLAIMED: with every mask entry 0 the batch is refused, and the
    refusal comes from the ADVANTAGE estimator, which names the offending row
    and its 0-of-2 supervised positions. That is the guard the operator meets
    first, so it is the guard this test pins.

    WHAT IS NOT CLAIMED: that ``RLOOPolicyLoss``'s own zero-supervision
    tripwire is what fires. It is not, and it cannot be: ``LeaveOneOutAdvantage``
    refuses any all-zero mask row before the loss counts a single token, and any
    row it does admit carries at least one supervised position, which makes the
    loss's ``supervised == 0`` arm unreachable through this public path. That arm
    is a defensive tripwire against a rigged estimator, and it is left uncovered
    deliberately rather than reached by staging a lying estimator -- a test that
    can only fire against a fake is measuring the fake.

    WHAT IS NOT CLAIMED: that a partially masked batch is healthy; partial
    masking is exercised by the happy-path mixed-mask test instead.
    """
    batch = _rloo_batch(loss_mask=((0, 0), (0, 0)))
    with pytest.raises(AdvantageRefusal, match="mask row 0 supervises 0 of 2 positions"):
        RLOOPolicyLoss().compute_with_report(
            _forward_with_rows(((-1.0, -1.0), (-1.0, -1.0))),
            batch,
        )


def test_rloo_setup_refuses_second_wiring() -> None:
    """A completed setup cannot be silently replaced by another call.

    WHAT IS CLAIMED: after one fully successful setup, a second setup refuses
    with the already-wired message, and the first supplied mapping survives.

    WHAT IS NOT CLAIMED: that the second wiring would have been wrong; the
    refusal is about silently replacing a measured mapping.
    """
    algorithm = RLOOAlgorithm()
    loss_fn = RLOOPolicyLoss()
    algorithm.setup(
        policy_pair=PolicyPair(train_view=object()),
        loss_fn=loss_fn,
        dataloader=[_rloo_batch()],
        config={"policy_logprob_column": "current_logprobs"},
        advantage_fn=loss_fn.advantage_fn,
    )
    assert algorithm.supplied() is not None
    with pytest.raises(AlgorithmWiringRefusal, match="was already wired"):
        algorithm.setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=loss_fn,
            dataloader=[_rloo_batch()],
            config={"policy_logprob_column": "current_logprobs"},
            advantage_fn=loss_fn.advantage_fn,
        )


def test_rloo_setup_refuses_foreign_loss_type() -> None:
    """A non-RLOOPolicyLoss in the loss slot refuses before any other leg.

    WHAT IS CLAIMED: an untyped object in the slot refuses naming the carried
    type against the required one.

    WHAT IS NOT CLAIMED: that a correctly typed but misconfigured loss passes;
    the semantics legs refuse those, and are tested separately.
    """
    with pytest.raises(AlgorithmWiringRefusal, match="RLOO requires RLOOPolicyLoss"):
        RLOOAlgorithm().setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=object(),
            dataloader=[_rloo_batch()],
            config={"policy_logprob_column": "current_logprobs"},
            advantage_fn=LeaveOneOutAdvantage(),
        )


def test_rloo_setup_refuses_missing_estimator_role() -> None:
    """The advantage_fn role is required; None is absence, not wiring.

    WHAT IS CLAIMED: omitting the estimator role refuses naming the single
    absent input for rloo.

    WHAT IS NOT CLAIMED: that False would be a valid stand-in; presence must
    be the actual estimator object.
    """
    with pytest.raises(AlgorithmWiringRefusal, match="required inputs absent for rloo"):
        RLOOAlgorithm().setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=RLOOPolicyLoss(),
            dataloader=[_rloo_batch()],
            config={"policy_logprob_column": "current_logprobs"},
        )


def test_rloo_setup_refuses_semantics_disagreement_on_group_size() -> None:
    """Two declarations of group size that disagree refuse at the handshake.

    WHAT IS CLAIMED: an algorithm constructed at the default K=2 wired to a
    loss declaring K=3 refuses on the first semantics field, naming both
    declared values.

    WHAT IS NOT CLAIMED: which side is right; the refusal is on the
    disagreement of two independently constructed declarations.
    """
    loss_fn = RLOOPolicyLoss(
        group_size=3,
        advantage_fn=LeaveOneOutAdvantage(min_group_size=3),
    )
    with pytest.raises(AlgorithmWiringRefusal, match="1 of 5 semantics fields disagrees") as exc:
        RLOOAlgorithm().setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=loss_fn,
            dataloader=[],
            config={},
            advantage_fn=loss_fn.advantage_fn,
        )
    message = str(exc.value)
    assert "algorithm declares 2" in message
    assert "loss declares 3" in message


def test_rloo_setup_refuses_non_iterable_dataloader_and_non_mapping_config() -> None:
    """A dataloader that cannot iterate and a config that is not a map refuse.

    WHAT IS CLAIMED: an integer dataloader refuses with the not-iterable
    message, and an integer config refuses with the must-implement-Mapping
    message -- two separate legs, each isolated.

    WHAT IS NOT CLAIMED: about order dependence; each refusal is reached with
    every earlier leg satisfied.
    """
    algorithm = RLOOAlgorithm()
    loss_fn = RLOOPolicyLoss()
    with pytest.raises(AlgorithmWiringRefusal, match="was not iterable"):
        algorithm.setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=loss_fn,
            dataloader=42,
            config={"policy_logprob_column": "current_logprobs"},
            advantage_fn=loss_fn.advantage_fn,
        )
    with pytest.raises(AlgorithmWiringRefusal, match="must implement Mapping"):
        algorithm.setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=loss_fn,
            dataloader=[_rloo_batch()],
            config=42,
            advantage_fn=loss_fn.advantage_fn,
        )


def test_rloo_setup_refuses_missing_policy_logprob_column_config() -> None:
    """Without a named column, current log-probs cannot reach the loss surface.

    WHAT IS CLAIMED: a config lacking the key refuses with the
    0-of-1-required-config-values message for rloo.

    WHAT IS NOT CLAIMED: that a non-string or empty value differs; only the
    absent-key shape is exercised.
    """
    algorithm = RLOOAlgorithm()
    loss_fn = RLOOPolicyLoss()
    with pytest.raises(AlgorithmWiringRefusal, match="0 of 1 required config values"):
        algorithm.setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=loss_fn,
            dataloader=[_rloo_batch()],
            config={},
            advantage_fn=loss_fn.advantage_fn,
        )


def test_reinforce_baseline_refuses_colliding_and_invalid_declared_names() -> None:
    """The remaining constructor guards refuse, each naming its own field.

    WHAT IS CLAIMED: identical component and metric names refuse with the
    2-names-in-1-denominator message, an empty column name refuses with the
    non-empty-string message, and a non-positive weight refuses with the
    field name and value.

    WHAT IS NOT CLAIMED: that the momentum guards misbehave; the existing
    module already measures those.
    """
    with pytest.raises(
        LossConfigRefusal,
        match="2 declared names would be 1 name in 2 denominators",
    ):
        ReinforceBaselineLoss(component_name="dup", baseline_metric_name="dup")
    with pytest.raises(LossConfigRefusal, match="non-empty strings"):
        ReinforceBaselineLoss(reward_column="")
    with pytest.raises(LossConfigRefusal, match=re.escape("weight=-1.0")):
        ReinforceBaselineLoss(weight=-1.0)


def test_reinforce_baseline_call_uses_batch_mean_as_baseline() -> None:
    """The ``LossFn`` surface call is exactly the abstained-baseline case.

    Hand computation: rewards (2, 4) give mean 3, advantages (-1, +1), and
    tokens (1, -2), so the contribution is 0.5; the fraction-above reading is
    1 of 2 rows, i.e. 0.5.

    WHAT IS CLAIMED: ``__call__`` returns the first half of
    ``compute_with_report`` run with ``baseline=None``, metric included.

    WHAT IS NOT CLAIMED: that any carried state informed the price; the
    protocol shape cannot carry one.
    """
    loss_fn = ReinforceBaselineLoss()
    output = loss_fn(
        _forward_with_rows(((-1.0,), (-2.0,))),
        _reinforce_batch(rewards=(2.0, 4.0)),
    )
    assert output.loss == pytest.approx(0.5)
    assert output.components[0].name == "reinforce_loss"
    assert len(output.metrics) == 1
    assert output.metrics[0].name == "reinforce_baseline_frac_above"
    assert output.metrics[0].value == pytest.approx(0.5)


def test_reinforce_baseline_refuses_boolean_and_non_finite_baseline() -> None:
    """A baseline must be abstention (None) or a finite real number.

    WHAT IS CLAIMED: True refuses with the bool-specific message despite bool
    being an int subclass, and NaN refuses with the finite-number message.

    WHAT IS NOT CLAIMED: that 0.0 is an invalid baseline; a measured zero is
    a measurement and the guard admits it by construction.
    """
    loss_fn = ReinforceBaselineLoss()
    batch = _reinforce_batch(rewards=(2.0, 4.0))
    with pytest.raises(LossConfigRefusal, match="True is not a baseline"):
        loss_fn.compute_with_report(
            _forward_with_rows(((-1.0,), (-2.0,))),
            batch,
            baseline=True,
        )
    with pytest.raises(
        LossConfigRefusal,
        match=re.escape("None (abstain) or a finite real number"),
    ):
        loss_fn.compute_with_report(
            _forward_with_rows(((-1.0,), (-2.0,))),
            batch,
            baseline=math.nan,
        )


def test_reinforce_baseline_refuses_missing_column_and_empty_batch() -> None:
    """Both batch-shape guardrails refuse before any baseline is derived.

    WHAT IS CLAIMED: a batch without the rewards column refuses naming the
    absent input, and a zero-row batch refuses rather than deriving a mean
    over nothing.

    WHAT IS NOT CLAIMED: which refusal wins when both defects hold; each test
    isolates exactly one.
    """
    loss_fn = ReinforceBaselineLoss()
    missing = _Batch({"loss_mask": ((1,), (1,))})
    with pytest.raises(
        BatchRefusal,
        match="required inputs absent for reinforce_baseline",
    ):
        loss_fn.compute_with_report(_forward_with_rows(()), missing, baseline=None)
    empty = _Batch({"rewards": (), "loss_mask": ()})
    with pytest.raises(BatchRefusal, match="0 of at least 1 required batch rows"):
        loss_fn.compute_with_report(_forward_with_rows(()), empty, baseline=None)


def test_reinforce_baseline_refuses_denominator_count_mismatch() -> None:
    """Rewards and masks must each cover the one offered batch length.

    WHAT IS CLAIMED: one mask row against two rewards refuses with the
    both-counts denominator message before any token work.

    WHAT IS NOT CLAIMED: that the mirror-image length failure differs; only
    the short-mask side is exercised.
    """
    batch = _Batch({"rewards": (1.0, 2.0), "loss_mask": ((1,),)})
    with pytest.raises(BatchRefusal, match="both counts must be the one offered denominator"):
        ReinforceBaselineLoss().compute_with_report(
            _forward_with_rows(()),
            batch,
            baseline=None,
        )


def test_reinforce_baseline_refuses_forward_and_token_length_mismatch() -> None:
    """Current log-probs must match the batch rows and the mask's token length.

    WHAT IS CLAIMED: a one-row forward answer for a two-row batch refuses
    with the one-row-per-batch-row message, and a mask longer than the
    current row refuses with the two-denominators message -- separate legs.

    WHAT IS NOT CLAIMED: which token position first disagrees; the refusal is
    on the row-level length pair.
    """
    loss_fn = ReinforceBaselineLoss()
    batch = _reinforce_batch(rewards=(2.0, 4.0))
    with pytest.raises(BatchRefusal, match="one token row per batch row is required"):
        loss_fn.compute_with_report(
            _forward_with_rows(((-1.0,),)),
            batch,
            baseline=None,
        )
    long_mask = _Batch(
        {
            "rewards": (2.0, 4.0),
            "loss_mask": ((1, 1), (1, 1)),
            "current_logprobs": ((-1.0,), (-1.0,)),
        }
    )
    with pytest.raises(BatchRefusal, match="2 token denominators must be equal"):
        loss_fn.compute_with_report(
            _forward_with_rows(((-1.0,), (-1.0,))),
            long_mask,
            baseline=None,
        )


def test_reinforce_baseline_explicit_baseline_is_subtracted_and_reported() -> None:
    """A supplied finite baseline is the value subtracted and the value read.

    Hand computation: baseline 1.0 over rewards (2, 4) gives advantages (1,
    3), tokens (-1, -6), contribution 3.5; the reading is the fraction strictly
    above 1.0, which is 1.0 -- the declared-degenerate endpoint, honestly so,
    because a baseline below every return reduces no variance. The returned
    float is the batch's own mean return 3.0, the value an EMA caller needs.

    WHAT IS CLAIMED: the contribution, the metric value, and the returned
    mean all match the hand computation.

    WHAT IS NOT CLAIMED: that 1.0 was a good baseline; only that it was the
    one used and the one reported.
    """
    output, returns_mean = ReinforceBaselineLoss().compute_with_report(
        _forward_with_rows(((-1.0,), (-2.0,))),
        _reinforce_batch(rewards=(2.0, 4.0)),
        baseline=1.0,
    )
    assert output.loss == pytest.approx(3.5)
    assert output.metrics[0].value == pytest.approx(1.0)
    assert returns_mean == pytest.approx(3.0)


def test_reinforce_baseline_metric_degenerate_zero_endpoint_verifies() -> None:
    """An all-equal batch reads the declared-degenerate 0.0 endpoint.

    Hand computation: rewards (5, 5) give mean 5, zero advantages, a 0.0
    contribution, and no return strictly above the baseline, so the reading
    is 0.0 -- degenerate by declaration, and honest: equality shifts no
    advantage's sign.

    WHAT IS CLAIMED: the contribution is a measured 0.0 (not ``None``), the
    metric reads exactly 0.0, and declaration/observation agreement verifies.

    WHAT IS NOT CLAIMED: that such a batch is a healthy training signal; the
    degenerate endpoint is declared precisely to make that visible.
    """
    loss_fn = ReinforceBaselineLoss()
    output, returns_mean = loss_fn.compute_with_report(
        _forward_with_rows(((-1.0,), (-2.0,))),
        _reinforce_batch(rewards=(5.0, 5.0)),
        baseline=None,
    )
    assert output.loss == pytest.approx(0.0)
    assert output.metrics[0].value == pytest.approx(0.0)
    assert returns_mean == pytest.approx(5.0)
    report = StepReport(step=0, loss=output, rows=2, reward_stats=None)
    assert verify_step(report, ReinforceBaselineAlgorithm().requirements()) == 1


def test_reinforce_baseline_semantics_and_requires_declarations() -> None:
    """The one-line declaration accessors return the constructor's own records.

    WHAT IS CLAIMED: ``semantics()`` declares reference-freeness and abstains
    (``None``) on the four unconstrained seams; ``requires()`` marks exactly
    the three mandatory setup roles True and every optional role False.

    WHAT IS NOT CLAIMED: agreement with any loss object; the setup handshake
    measures that.
    """
    algorithm = ReinforceBaselineAlgorithm()
    semantics = algorithm.semantics()
    assert semantics.group_size is None
    assert semantics.ratio_scope is None
    assert semantics.kl_estimator is None
    assert semantics.clip_bounds is None
    assert semantics.reference_free is True
    requires = algorithm.requires()
    assert requires["policy_pair"] is True
    assert requires["loss_fn"] is True
    assert requires["dataloader"] is True
    assert requires["advantage_fn"] is False
    assert requires["reference_policy"] is False
    assert requires["rollout_source"] is False
    assert requires["weight_sync"] is False


def test_reinforce_baseline_setup_refuses_second_wiring() -> None:
    """A completed setup cannot be silently replaced by another call.

    WHAT IS CLAIMED: after one fully successful setup a second setup refuses
    with the already-wired message.

    WHAT IS NOT CLAIMED: that the supplied mapping was mutated; the refusal
    exists precisely so it cannot be replaced unnoticed.
    """
    algorithm = ReinforceBaselineAlgorithm()
    loss_fn = ReinforceBaselineLoss()
    algorithm.setup(
        policy_pair=PolicyPair(train_view=object()),
        loss_fn=loss_fn,
        dataloader=[_reinforce_batch(rewards=(2.0, 4.0))],
        config={"policy_logprob_column": "current_logprobs"},
    )
    with pytest.raises(AlgorithmWiringRefusal, match="was already wired"):
        algorithm.setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=loss_fn,
            dataloader=[_reinforce_batch(rewards=(2.0, 4.0))],
            config={"policy_logprob_column": "current_logprobs"},
        )


def test_reinforce_baseline_setup_refuses_foreign_loss_type() -> None:
    """A non-ReinforceBaselineLoss in the loss slot refuses before other legs.

    WHAT IS CLAIMED: an RLOOPolicyLoss in the slot refuses naming the carried
    type against the required one.

    WHAT IS NOT CLAIMED: that the momentum or semantics legs would have
    disagreed; the type refusal necessarily fires first.
    """
    with pytest.raises(AlgorithmWiringRefusal, match="requires ReinforceBaselineLoss"):
        ReinforceBaselineAlgorithm().setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=RLOOPolicyLoss(),
            dataloader=[_reinforce_batch(rewards=(2.0, 4.0))],
            config={"policy_logprob_column": "current_logprobs"},
        )


def test_reinforce_baseline_setup_refuses_semantics_disagreement() -> None:
    """Two semantics declarations that disagree refuse at the handshake.

    WHAT IS CLAIMED: a correctly typed loss whose semantics disagree on
    group_size (declaring 1 where the algorithm abstains with None) refuses
    with the 1-of-5 disagreement message, even though type and momentum match.

    WHAT IS NOT CLAIMED: that any honest ReinforceBaselineLoss can produce
    this; the subclass's override manufactures the two-editions situation the
    leg exists to refuse.
    """
    with pytest.raises(AlgorithmWiringRefusal, match="1 of 5 semantics fields disagrees"):
        ReinforceBaselineAlgorithm().setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=_ForeignSemanticsLoss(),
            dataloader=[_reinforce_batch(rewards=(2.0, 4.0))],
            config={"policy_logprob_column": "current_logprobs"},
        )


def test_reinforce_baseline_setup_refuses_non_iterable_dataloader_and_config() -> None:
    """A dataloader that cannot iterate and a config that is not a map refuse.

    WHAT IS CLAIMED: an integer dataloader refuses with the not-iterable
    message, and an integer config refuses with the must-implement-Mapping
    message -- each reached with every earlier leg satisfied.

    WHAT IS NOT CLAIMED: any ordering claim between the two legs; they are
    tested independently.
    """
    loss_fn = ReinforceBaselineLoss()
    with pytest.raises(AlgorithmWiringRefusal, match="was not iterable"):
        ReinforceBaselineAlgorithm().setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=loss_fn,
            dataloader=42,
            config={"policy_logprob_column": "current_logprobs"},
        )
    with pytest.raises(AlgorithmWiringRefusal, match="must implement Mapping"):
        ReinforceBaselineAlgorithm().setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=loss_fn,
            dataloader=[_reinforce_batch(rewards=(2.0, 4.0))],
            config=42,
        )


def test_reinforce_baseline_setup_refuses_missing_policy_logprob_column() -> None:
    """Without a named column, current log-probs cannot reach the loss surface.

    WHAT IS CLAIMED: a config lacking the key refuses with the
    0-of-1-required-config-values message for reinforce_baseline.

    WHAT IS NOT CLAIMED: that a non-string value differs from absence; only
    the absent-key shape is exercised.
    """
    with pytest.raises(AlgorithmWiringRefusal, match="0 of 1 required config values"):
        ReinforceBaselineAlgorithm().setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=ReinforceBaselineLoss(),
            dataloader=[_reinforce_batch(rewards=(2.0, 4.0))],
            config={},
        )


def test_reinforce_baseline_exhausted_dataloader_refuses_second_step() -> None:
    """A wired algorithm prices one batch, seeds, then refuses the empty step.

    Hand computation: rewards (2, 4) give mean 3; the unseeded first step
    resolves its baseline to 3 and seeds the state with 3.

    WHAT IS CLAIMED: the first step succeeds and seeds ``baseline()`` at
    3.0, the supplied mapping is present, and the second step on the
    exhausted one-batch dataloader refuses rather than reporting zero rows.

    WHAT IS NOT CLAIMED: anything about the EMA update; the existing module
    measures that path with a two-batch dataloader.
    """
    algorithm = ReinforceBaselineAlgorithm()
    algorithm.setup(
        policy_pair=PolicyPair(train_view=object()),
        loss_fn=ReinforceBaselineLoss(),
        dataloader=[_reinforce_batch(rewards=(2.0, 4.0))],
        config={"policy_logprob_column": "current_logprobs"},
    )
    first = algorithm.step()
    assert first.step == 0
    assert first.rows == 2
    assert algorithm.baseline() == pytest.approx(3.0)
    assert algorithm.supplied() is not None
    with pytest.raises(
        StepReportRefusal,
        match="0 of at least 1 required batches remain for reinforce_baseline",
    ):
        algorithm.step()
