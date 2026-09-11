"""Adversarial tests for the PPO binding's refusal surface: setup wiring and batch anchors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from foundationscale.rl.advantage import (
    AdvantageFn,
    AdvantageResult,
    GroupNormalisedAdvantage,
    LearnedValueAdvantageEstimation,
    RewardStats,
)
from foundationscale.rl.algorithm import AlgorithmWiringRefusal
from foundationscale.rl.interfaces import BatchRefusal, ExperienceBatch
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.ppo import PPOAlgorithm, PPOCompositeLoss
from foundationscale.rl.ppo_objectives import ValueFunctionLoss
from foundationscale.rl.value_head import ValueCapabilities, ValueHead

# The same hand-computable two-row batch as the happy-path suite: one
# supervised token per row, terminated=True so each row's only delta is
# reward - value == reward - 0.25.
_ESTIMATES: tuple[tuple[float, ...], ...] = ((0.25,), (0.25,))


@dataclass(frozen=True)
class _FixedValueHead:
    # Structural fake for the runtime_checkable ValueHead Protocol, which
    # cannot be instantiated. ``reports_old_values`` is a field because the
    # value-clipping arm of setup demands a head that reports old values.
    estimates: tuple[tuple[float, ...], ...]
    reports_old_values: bool = False

    def estimate(self, _batch: ExperienceBatch) -> list[list[float]]:
        return [list(row) for row in self.estimates]

    def capabilities(self) -> ValueCapabilities:
        return ValueCapabilities(
            granularity="token",
            shares_policy_trunk=False,
            reports_old_values=self.reports_old_values,
        )


class _CompactingAdvantageEstimation(LearnedValueAdvantageEstimation):
    # A SUBCLASS, not a substitute: PPOCompositeLoss.__post_init__ admits it
    # through isinstance, which honours subclasses, so the offered-rows guard
    # in compute_with_report is the only thing between this estimator and a
    # batch silently shrunk by compaction. It computes honestly and then lies
    # about how many rows it was offered.
    def compute(
        self,
        *,
        prompt_ids: Sequence[str],
        rewards: Sequence[float],
        mask: Sequence[Sequence[int]],
        values: Sequence[Sequence[float]],
        terminated: Sequence[bool],
    ) -> AdvantageResult:
        measured = super().compute(
            prompt_ids=prompt_ids,
            rewards=rewards,
            mask=mask,
            values=values,
            terminated=terminated,
        )
        shrunk = measured.offered - 1
        return AdvantageResult(
            weights=measured.weights[:shrunk],
            rewards=RewardStats.over(tuple(float(reward) for reward in rewards[:shrunk])),
            used=shrunk,
            offered=shrunk,
            method=measured.method,
            rows=tuple(range(shrunk)),
        )


def _value_head(*, reports_old_values: bool = False) -> ValueHead:
    return _FixedValueHead(estimates=_ESTIMATES, reports_old_values=reports_old_values)


def _batch() -> ExperienceBatch:
    return ExperienceBatch(
        columns={
            "current_logprobs": [[-0.1], [-0.2]],
            "old_logprobs": [[-0.1], [-0.2]],
            "advantages": [[0.75], [1.75]],
            "returns": [[1.0], [2.0]],
            "prompt_ids": ["prompt-a", "prompt-b"],
            "rewards": [1.0, 2.0],
            "terminated": [True, True],
            "loss_mask": [[True], [True]],
        }
    )


def _pair() -> PolicyPair:
    return PolicyPair(train_view=object(), generate_view=None, references={})


def _current_logprobs(batch: ExperienceBatch) -> Sequence[Any]:
    return batch.column("current_logprobs")


def _setup_kwargs(**overrides: Any) -> dict[str, Any]:
    value_head = _value_head()
    advantage_fn = LearnedValueAdvantageEstimation()
    kwargs: dict[str, Any] = {
        "policy_pair": _pair(),
        "loss_fn": PPOCompositeLoss(value_head=value_head, advantage_fn=advantage_fn),
        "dataloader": [_batch()],
        "config": {"policy_logprob_column": "current_logprobs"},
        "advantage_fn": advantage_fn,
        "value_head": value_head,
    }
    kwargs.update(overrides)
    return kwargs


def _clipped_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    value_head = _value_head(reports_old_values=True)
    advantage_fn = LearnedValueAdvantageEstimation()
    return {
        "policy_pair": _pair(),
        "loss_fn": PPOCompositeLoss(
            value_loss=ValueFunctionLoss(clip_epsilon=0.1),
            value_head=value_head,
            advantage_fn=advantage_fn,
        ),
        "dataloader": [_batch()],
        "config": config,
        "advantage_fn": advantage_fn,
        "value_head": value_head,
    }


def test_call_refuses_an_estimator_whose_offered_count_shrinks_the_batch() -> None:
    # The previous round called this guard unreachable because __post_init__
    # enforces the concrete estimator type. isinstance admits subclasses, so a
    # subclass can compute legally and still report an offered count that
    # disagrees with the batch -- precisely the silent compaction the guard
    # exists to refuse. Both numbers must be named.
    loss_fn = PPOCompositeLoss(
        value_head=_value_head(),
        advantage_fn=_CompactingAdvantageEstimation(),
    )
    with pytest.raises(BatchRefusal, match=r"AdvantageResult\.offered=1") as excinfo:
        loss_fn(_current_logprobs, _batch())
    message = str(excinfo.value)
    assert "the batch contains 2 rows" in message
    assert "1 of 2 offered rows cannot anchor a PPO step" in message


def test_semantics_returns_the_constructed_record_by_identity() -> None:
    kl_free = PPOAlgorithm(with_kl=False)
    semantics = kl_free.semantics()
    assert semantics is kl_free.requirements().semantics
    assert semantics.reference_free is True
    assert semantics.clip_bounds == (0.8, 1.2)
    with_kl = PPOAlgorithm()
    assert with_kl.semantics() is with_kl.requirements().semantics
    assert with_kl.semantics().reference_free is False
    assert with_kl.semantics().kl_estimator == "k3"


def test_setup_refuses_a_loss_that_is_not_a_ppo_composite() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match="PPO requires PPOCompositeLoss") as excinfo:
        PPOAlgorithm(with_kl=False).setup(**_setup_kwargs(loss_fn=object()))
    assert "1 of 1 loss slots carries object" in str(excinfo.value)


def test_setup_refuses_an_absent_setup_side_advantage_fn() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match=r"\{'advantage_fn'\}") as excinfo:
        PPOAlgorithm(with_kl=False).setup(**_setup_kwargs(advantage_fn=None))
    assert "1 of 1 required inputs absent for ppo" in str(excinfo.value)


def test_setup_refuses_a_protocol_satisfying_estimator_that_is_not_learned_value() -> None:
    """The width the annotation admits is narrowed HERE, by measurement (#359/#395).

    WHAT IS CLAIMED: handing ``setup`` an object that genuinely satisfies the
    ``AdvantageFn`` protocol -- ``GroupNormalisedAdvantage``, a shipped binding,
    not a stub -- is refused, and the refusal names the type it received.

    This leg exists because the parameter annotation was WIDENED to the
    protocol's own width so ``PPOAlgorithm`` would be substitutable for
    ``Algorithm`` and could enter the registry. Widening a parameter type
    deletes a static claim; if nothing replaced it, PPO would accept an
    estimator that cannot supply the ``values``/``terminated`` arguments its
    GAE binding requires, and the failure would surface later as a TypeError
    from inside the estimator rather than as a wiring refusal naming the
    field. The runtime guard is the replacement, and this is the measurement
    that it fires.

    The probe is a REAL AdvantageFn rather than a fake for a reason: a stub
    would leave open whether the guard rejects everything unfamiliar or
    rejects precisely the protocol-satisfying-but-wrong-shape case, which is
    the case that the widened annotation newly admits.

    WHAT IS NOT CLAIMED: that the refusal fires before every other setup
    check -- the loss-side composite refuses the same type at construction,
    so this leg wires a VALID loss and corrupts only the setup-side argument.
    """
    wrong_but_valid_protocol_impl = GroupNormalisedAdvantage()
    # Positive control on the premise: the probe really does satisfy the
    # protocol, so this test measures the guard and not a typo. Without this
    # line a renamed method would make the object refusable for the wrong
    # reason and the leg would still pass.
    assert isinstance(wrong_but_valid_protocol_impl, AdvantageFn)

    value_head = _value_head()
    loss_fn = PPOCompositeLoss(
        value_head=value_head, advantage_fn=LearnedValueAdvantageEstimation()
    )
    with pytest.raises(
        AlgorithmWiringRefusal, match="1 of 1 advantage slots must carry a"
    ) as excinfo:
        PPOAlgorithm(with_kl=False).setup(
            policy_pair=_pair(),
            loss_fn=loss_fn,
            dataloader=[_batch()],
            config={"policy_logprob_column": "current_logprobs"},
            advantage_fn=wrong_but_valid_protocol_impl,
            value_head=value_head,
        )
    message = str(excinfo.value)
    assert "field advantage_fn=" in message
    # The type it actually received, not just the type it wanted: a refusal
    # that names only the requirement leaves the operator to guess what was
    # handed over.
    assert "it carries GroupNormalisedAdvantage" in message


def test_setup_refuses_a_distinct_but_equal_setup_side_estimator() -> None:
    # Two separately frozen constructions compare EQUAL by value; only an
    # identity check can tell them apart, and identity is what this role
    # means -- one LearnedValueAdvantageEstimation instance wired through
    # both sides.
    loss_side = LearnedValueAdvantageEstimation()
    setup_side = LearnedValueAdvantageEstimation()
    assert setup_side == loss_side
    value_head = _value_head()
    loss_fn = PPOCompositeLoss(value_head=value_head, advantage_fn=loss_side)
    with pytest.raises(
        AlgorithmWiringRefusal, match="loss-owned estimator are 2 distinct objects"
    ) as excinfo:
        PPOAlgorithm(with_kl=False).setup(
            policy_pair=_pair(),
            loss_fn=loss_fn,
            dataloader=[_batch()],
            config={"policy_logprob_column": "current_logprobs"},
            advantage_fn=setup_side,
            value_head=value_head,
        )
    assert "for 1 declared role" in str(excinfo.value)


def test_setup_refuses_an_absent_setup_side_value_head() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match=r"\{'value_head'\}") as excinfo:
        PPOAlgorithm(with_kl=False).setup(**_setup_kwargs(value_head=None))
    assert "1 of 1 required inputs absent for ppo" in str(excinfo.value)


def test_setup_refuses_a_distinct_but_equal_setup_side_value_head() -> None:
    # Same proof shape as the estimator arm: equal frozen dataclasses, only
    # identity separating them, because one ValueHead instance must be wired
    # through both sides.
    loss_side = _FixedValueHead(estimates=_ESTIMATES)
    setup_side = _FixedValueHead(estimates=_ESTIMATES)
    assert setup_side == loss_side
    advantage_fn = LearnedValueAdvantageEstimation()
    loss_fn = PPOCompositeLoss(value_head=loss_side, advantage_fn=advantage_fn)
    with pytest.raises(
        AlgorithmWiringRefusal, match="loss-owned value head are 2 distinct objects"
    ) as excinfo:
        PPOAlgorithm(with_kl=False).setup(
            policy_pair=_pair(),
            loss_fn=loss_fn,
            dataloader=[_batch()],
            config={"policy_logprob_column": "current_logprobs"},
            advantage_fn=advantage_fn,
            value_head=setup_side,
        )
    assert "for 1 declared role" in str(excinfo.value)


def test_setup_refuses_a_non_iterable_dataloader_and_chains_the_type_error() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match=r"was not iterable \(object\)") as excinfo:
        PPOAlgorithm(with_kl=False).setup(**_setup_kwargs(dataloader=object()))
    message = str(excinfo.value)
    assert "1 of 1 mandatory step inputs" in message
    assert isinstance(excinfo.value.__cause__, TypeError)


def test_setup_refuses_a_config_that_is_not_a_mapping() -> None:
    kwargs = _setup_kwargs(config={"policy_logprob_column", "old_value_column"})
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="1 of 1 setup configurations must implement Mapping",
    ) as excinfo:
        PPOAlgorithm(with_kl=False).setup(**kwargs)
    assert "field config=" in str(excinfo.value)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"policy_logprob_column": 42},
        {"policy_logprob_column": ""},
    ],
    ids=["absent", "non-string", "empty-string"],
)
def test_setup_refuses_a_config_without_a_usable_policy_logprob_column(
    config: dict[str, Any],
) -> None:
    # The empty-string variant is what sees an `or not column` clause being
    # dropped: None and 42 both fail isinstance, but "" passes it.
    with pytest.raises(AlgorithmWiringRefusal, match="policy_logprob_column") as excinfo:
        PPOAlgorithm(with_kl=False).setup(**_setup_kwargs(config=config))
    message = str(excinfo.value)
    assert "0 of 1 required config values names a non-empty batch column for ppo" in message
    assert "current token log-probabilities cannot be attributed" in message


@pytest.mark.parametrize(
    "config",
    [
        {"policy_logprob_column": "current_logprobs"},
        {"policy_logprob_column": "current_logprobs", "old_value_column": 42},
        {"policy_logprob_column": "current_logprobs", "old_value_column": ""},
    ],
    ids=["absent", "non-string", "empty-string"],
)
def test_setup_refuses_an_unusable_old_value_column_under_value_clipping(
    config: dict[str, Any],
) -> None:
    # Reached only when the wired value loss clips (clip_epsilon set) AND the
    # head reports old values; both sides of that gate are wired here.
    with pytest.raises(AlgorithmWiringRefusal, match="old_value_column") as excinfo:
        PPOAlgorithm(with_kl=False, value_clip_epsilon=0.1).setup(**_clipped_kwargs(config))
    message = str(excinfo.value)
    assert "0 of 1 required config values names a non-empty batch column for ppo" in message
    assert "PPO2 value-clipping reference is a static column" in message


def test_setup_refuses_an_old_value_column_disagreement_naming_both_names() -> None:
    config = {
        "policy_logprob_column": "current_logprobs",
        "old_value_column": "stale_values",
    }
    with pytest.raises(AlgorithmWiringRefusal, match=r"config names 'stale_values'") as excinfo:
        PPOAlgorithm(with_kl=False, value_clip_epsilon=0.1).setup(**_clipped_kwargs(config))
    message = str(excinfo.value)
    assert "the wired value loss reads 'old_values'" in message
    assert "1 old-value seam was stated 2 ways" in message
