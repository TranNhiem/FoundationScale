"""Stage-3e policy-gradient family tests at the pure stdlib seam."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from foundationscale.rl.advantage import AdvantageRefusal, LeaveOneOutAdvantage
from foundationscale.rl.algorithm import (
    AlgorithmWiringRefusal,
    StepReport,
    StepReportRefusal,
    check_algorithm_wiring,
    verify_step,
)
from foundationscale.rl.interfaces import (
    BatchRefusal,
    LossConfigRefusal,
    SupervisionRefusal,
)
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.policy_gradient import (
    ReinforceBaselineAlgorithm,
    ReinforceBaselineLoss,
    ReinforcePlusPlusAlgorithm,
    ReinforcePlusPlusLoss,
    RLOOAlgorithm,
    RLOOPolicyLoss,
    check_reinforce_baseline_requirements,
    check_reinforce_pp_requirements,
    check_rloo_requirements,
)
from foundationscale.rl.registry import (
    AlgorithmRegistryRefusal,
    available_algorithm_names,
    lookup_algorithm,
    register_algorithm,
    reset_algorithm_registry,
)

__all__ = ()


@dataclass(frozen=True, slots=True)
class _Batch:
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


def _rloo_batch() -> _Batch:
    return _Batch(
        {
            "prompt_ids": ("prompt-a", "prompt-a"),
            "rewards": (2.0, 4.0),
            "loss_mask": ((1, 1), (1,)),
            "old_logprobs": ((-1.0, -1.0), (-1.0,)),
            "current_logprobs": ((-1.0, -1.0), (-1.0,)),
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


def _reinforce_pp_batch(*, reference_offset: float = 0.0) -> _Batch:
    return _Batch(
        {
            "rewards": (2.0, 4.0),
            "loss_mask": ((1, 1), (1, 1)),
            "old_logprobs": ((-2.2, -1.2), (-1.2, -1.2)),
            "reference_logprobs": (
                (-2.0 - reference_offset, -1.0 - reference_offset),
                (-1.0 - reference_offset, -1.0 - reference_offset),
            ),
            "current_logprobs": ((-2.0, -1.0), (-1.0, -1.0)),
        }
    )


def test_registry_round_trip_returns_fresh_binding_for_all_three() -> None:
    """Every family name resolves to a fresh, structurally valid algorithm.

    WHAT IS CLAIMED: sorted names are exactly grpo plus the three family
    entries, two lookups per name are distinct objects, and each abstains
    (``None``) on wiring before setup.

    WHAT IS NOT CLAIMED: that any module-import side effect installs the
    family; the reset itself reinstalls all four, which is the property
    under test -- a family that registered on its own import would survive
    a reset only if something re-imported it.
    """
    reset_algorithm_registry()
    assert available_algorithm_names() == (
        "grpo",
        "reinforce_baseline",
        "reinforce_pp",
        "rloo",
    )
    for name, cls in (
        ("rloo", RLOOAlgorithm),
        ("reinforce_baseline", ReinforceBaselineAlgorithm),
        ("reinforce_pp", ReinforcePlusPlusAlgorithm),
    ):
        first = lookup_algorithm(name)
        second = lookup_algorithm(name)
        assert isinstance(first, cls)
        assert isinstance(second, cls)
        assert first is not second
        assert first.supplied() is None
    reset_algorithm_registry()


def test_registry_refuses_duplicate_family_name_with_denominators() -> None:
    """A duplicate registration names the occupied key and both counts.

    WHAT IS CLAIMED: the refusal carries the key, the 1-of-1 new
    registration, and the 1-of-4 registered-name denominator.

    WHAT IS NOT CLAIMED: which factory was attempted second; identity is not
    needed to establish the occupied key.
    """
    reset_algorithm_registry()
    with pytest.raises(
        AlgorithmRegistryRefusal,
        match="duplicate registration requested",
    ) as exc_info:
        register_algorithm("rloo", lambda: RLOOAlgorithm())
    message = str(exc_info.value)
    assert "field name='rloo'" in message
    assert "1 of 1 new registrations" in message
    assert "1 of 4 registered names" in message
    reset_algorithm_registry()


def test_family_requirements_refuse_empty_and_all_false_required_sets() -> None:
    """Both vacuous shapes refuse: no roles at all, and no required roles.

    WHAT IS CLAIMED: an empty union and an all-False requirement map are
    both refused as vacuous for each of the three helpers -- a wiring check
    over nothing must never read as coverage.

    WHAT IS NOT CLAIMED: that a partially False map fails; only the
    denominator-empty shapes are measured here.
    """
    reset_algorithm_registry()
    for check, origin in (
        (check_rloo_requirements, "rloo"),
        (check_reinforce_baseline_requirements, "reinforce_baseline"),
        (check_reinforce_pp_requirements, "reinforce_pp"),
    ):
        with pytest.raises(
            AlgorithmWiringRefusal,
            match="0 roles appear in either map",
        ):
            check(requires={}, supplied={})
        with pytest.raises(
            AlgorithmWiringRefusal,
            match=f"0 of 1 declared roles are required for {origin}",
        ):
            check(requires={"rollout_source": False}, supplied={})


def test_rloo_config_refusals_name_field_and_counts() -> None:
    """Invalid RLOO config refuses at construction, naming the field.

    WHAT IS CLAIMED: a group size below 2, a bool group size, and a
    group/estimator size disagreement all refuse before any step, with the
    field name and both sides' values in the message.

    WHAT IS NOT CLAIMED: that K=2 is algorithmically canonical; only the
    invalid and the disagreeing shapes are refused.
    """
    with pytest.raises(LossConfigRefusal, match="group_size=1") as exc_info:
        RLOOPolicyLoss(group_size=1)
    assert "at least 2 samples" in str(exc_info.value)
    with pytest.raises(LossConfigRefusal, match="True is not an RLOO group size"):
        RLOOPolicyLoss(group_size=True)
    with pytest.raises(
        LossConfigRefusal,
        match="1 configured RLOO group size disagrees with 1 estimator minimum",
    ) as exc_info:
        RLOOPolicyLoss(group_size=3)
    message = str(exc_info.value)
    assert "group_size=3" in message
    assert "min_group_size=2" in message


def test_rloo_hand_computed_advantage_and_loss_numbers() -> None:
    """A three-sample leave-one-out batch prices to hand-computed values.

    Hand computation: rewards (1, 2, 6) give leave-one-out baselines
    (4.0, 3.5, 1.5) and advantages (-3.0, -1.5, 4.5). Token ratios are
    exp(0.1), 1, and exp(-0.1) row by row, unclipped.

    WHAT IS CLAIMED: estimator weights, the single component's contribution,
    the total loss, and ``verify_step`` all agree with those values.

    WHAT IS NOT CLAIMED: numerical equivalence to an external RLOO
    implementation.
    """
    loss_fn = RLOOPolicyLoss(
        group_size=3,
        advantage_fn=LeaveOneOutAdvantage(min_group_size=3),
    )
    batch = _Batch(
        {
            "prompt_ids": ("p", "p", "p"),
            "rewards": (1.0, 2.0, 6.0),
            "loss_mask": ((1,), (1,), (1,)),
            "old_logprobs": ((-1.0,), (-1.0,), (-1.0,)),
        }
    )
    output, advantage = loss_fn.compute_with_report(
        lambda b: ((-0.9,), (-1.0,), (-1.1,)),
        batch,
    )
    assert advantage.weights == ((-3.0,), (-1.5,), (4.5,))
    assert advantage.rows == (0, 1, 2)
    assert advantage.used == 3
    assert advantage.offered == 3
    expected = -(-3.0 * math.exp(0.1) - 1.5 + 4.5 * math.exp(-0.1)) / 3.0
    assert output.components[0].name == "rloo_policy_loss"
    assert output.components[0].contribution == pytest.approx(expected)
    assert output.loss == pytest.approx(expected)
    report = StepReport(
        step=0,
        loss=output,
        rows=advantage.used,
        reward_stats=advantage.rewards,
    )
    assert verify_step(report, RLOOAlgorithm().requirements()) == 1


def test_rloo_zero_signal_group_stays_as_measured_zero_not_refused() -> None:
    """Leave-one-out keeps measured zeros where group normalisation excludes.

    WHAT IS CLAIMED: an all-equal-reward group prices without refusal, its
    advantages are genuinely 0.0, both rows stay USED (``used == offered``),
    and the loss is a real 0.0 -- abstention would be ``None``, not this.

    WHAT IS NOT CLAIMED: that the zero is a healthy training signal; only
    that it is a measured one and is not laundered into an exclusion.
    """
    loss_fn = RLOOPolicyLoss()
    batch = _Batch(
        {
            "prompt_ids": ("p", "p"),
            "rewards": (5.0, 5.0),
            "loss_mask": ((1, 1), (1, 1)),
            "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
        }
    )
    output, advantage = loss_fn.compute_with_report(
        lambda b: ((-1.0, -1.0), (-1.0, -1.0)),
        batch,
    )
    assert advantage.used == 2
    assert advantage.offered == 2
    assert advantage.weights == ((0.0, 0.0), (0.0, 0.0))
    assert output.loss == pytest.approx(0.0)


def test_rloo_setup_refuses_second_estimator_instance_for_one_role() -> None:
    """A distinct leave-one-out estimator cannot stand in for the loss-owned one.

    WHAT IS CLAIMED: the refusal fires at setup, before any batch is priced.

    WHAT IS NOT CLAIMED: that the two estimators would have computed
    different numbers; identity, not behaviour, is what is refused.
    """
    algorithm = RLOOAlgorithm()
    loss_fn = RLOOPolicyLoss()
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="2 distinct objects for 1 declared role",
    ):
        algorithm.setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=loss_fn,
            dataloader=[_rloo_batch()],
            config={"policy_logprob_column": "current_logprobs"},
            advantage_fn=LeaveOneOutAdvantage(),
        )


def test_rloo_abstains_wiring_before_refusing_a_step() -> None:
    """Abstention precedes refusal: supplied() is None, then step() refuses.

    WHAT IS CLAIMED: before setup the wiring answer is ``None`` (not an
    empty map, not False), and only the subsequent step is refused with the
    0-of-4 present-inputs message.

    WHAT IS NOT CLAIMED: that the refusal means the wiring failed; it means
    the wiring was never measured.
    """
    algorithm = RLOOAlgorithm()
    assert algorithm.supplied() is None
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="0 of 4 required inputs are present for rloo",
    ):
        algorithm.step()


def test_rloo_full_setup_step_and_exhaustion_refusal() -> None:
    """A wired RLOO step prices, verifies, then refuses an empty dataloader.

    Hand computation: rewards (2, 4) give advantages (-2, +2); ratios are 1
    (current equals old); supervised tokens number 3, so the contribution is
    -(-2 - 2 + 2)/ 3 = 2/3.

    WHAT IS CLAIMED: the wired step, its step index, ``rows == used``, and
    ``verify_step`` agree; a second step on the exhausted one-batch
    dataloader is refused rather than reporting zero rows.

    WHAT IS NOT CLAIMED: rollout freshness or any optimizer behaviour.
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
    report = algorithm.step()
    assert report.step == 0
    assert report.rows == 2
    assert report.loss.components[0].contribution == pytest.approx(2.0 / 3.0)
    assert verify_step(report, algorithm.requirements()) == 1
    assert algorithm.supplied() is not None
    with pytest.raises(
        StepReportRefusal,
        match="0 of at least 1 required batches remain for rloo",
    ):
        algorithm.step()


def test_reinforce_momentum_config_refusals_name_field_and_range() -> None:
    """Every invalid momentum shape refuses at construction.

    WHAT IS CLAIMED: zero, above-one, non-finite, and boolean momenta all
    refuse, each naming ``baseline_momentum`` and the admitted range; the
    algorithm constructor refuses the same values through its temporary
    validation loss.

    WHAT IS NOT CLAIMED: that 0.99 is a good momentum; bounds only.
    """
    for bad in (0.0, 1.5, math.inf, math.nan):
        with pytest.raises(
            LossConfigRefusal,
            match="baseline_momentum=",
        ):
            ReinforceBaselineLoss(baseline_momentum=bad)
    with pytest.raises(LossConfigRefusal, match="True is not a momentum"):
        ReinforceBaselineLoss(baseline_momentum=True)
    with pytest.raises(LossConfigRefusal, match="baseline_momentum=0.0"):
        ReinforceBaselineAlgorithm(baseline_momentum=0.0)


def test_reinforce_abstains_baseline_and_wiring_before_refusing() -> None:
    """The baseline accessor abstains (None) before the first step, and the
    pre-setup step refusal comes only after the abstention reads.

    WHAT IS CLAIMED: ``baseline()`` is ``None`` -- never 0.0 -- before any
    batch, and step refuses with the 0-of-3 message.

    WHAT IS NOT CLAIMED: that a seeded baseline is ever absent; every
    successful step reports its baseline through the metric channel.
    """
    algorithm = ReinforceBaselineAlgorithm()
    assert algorithm.baseline() is None
    assert algorithm.supplied() is None
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="0 of 3 required inputs are present for reinforce_baseline",
    ):
        algorithm.step()


def test_reinforce_two_steps_track_ema_and_report_metric() -> None:
    """Two wired steps exercise seeding, EMA update, and metric reporting.

    Hand computation, momentum 0.5. Step 0: rewards (2, 4) mean 3, unseeded
    baseline resolves to 3, advantages (-1, +1), so tokens are (1, -2) and
    the contribution is 0.5; the seeded state is 3. The declared diagnostic
    is the fraction strictly above the baseline: 1 of 2 rows, so 0.5. Step
    1: rewards (4, 6), baseline used 3.0, advantages (1, 3), tokens (-1,
    -6), contribution 3.5; the state updates to 0.5*3 + 0.5*5 = 4.0, and
    the diagnostic reads 2 of 2 rows above, so 1.0 -- a DECLARED degenerate
    endpoint, and honestly so: a baseline under every return in the batch
    shifts every advantage positive and reduces no variance.

    WHAT IS CLAIMED: both contributions, both metric values, the final EMA
    state, and ``verify_step``'s declaration/observation agreement on the
    declared metric -- including on the step whose reading is degenerate,
    because agreement and health are different questions.

    WHAT IS NOT CLAIMED: variance reduction or any reference-implementation
    equivalence.
    """
    algorithm = ReinforceBaselineAlgorithm(baseline_momentum=0.5)
    loss_fn = ReinforceBaselineLoss(baseline_momentum=0.5)
    algorithm.setup(
        policy_pair=PolicyPair(train_view=object()),
        loss_fn=loss_fn,
        dataloader=[_reinforce_batch(rewards=(2.0, 4.0)), _reinforce_batch(rewards=(4.0, 6.0))],
        config={"policy_logprob_column": "current_logprobs"},
    )
    first = algorithm.step()
    assert first.loss.components[0].contribution == pytest.approx(0.5)
    assert len(first.loss.metrics) == 1
    assert first.loss.metrics[0].name == "reinforce_baseline_frac_above"
    assert first.loss.metrics[0].value == pytest.approx(0.5)
    assert first.rows == 2
    assert algorithm.baseline() == pytest.approx(3.0)
    assert verify_step(first, algorithm.requirements()) == 1
    second = algorithm.step()
    assert second.loss.components[0].contribution == pytest.approx(3.5)
    assert second.loss.metrics[0].value == pytest.approx(1.0)
    assert algorithm.baseline() == pytest.approx(4.0)
    assert verify_step(second, algorithm.requirements()) == 1


def test_reinforce_setup_refuses_momentum_disagreement() -> None:
    """One momentum declared twice, differently, refuses at setup.

    WHAT IS CLAIMED: the message names the field and both declared values
    with its 1-of-1 denominator.

    WHAT IS NOT CLAIMED: which side is right; only that they disagree.
    """
    algorithm = ReinforceBaselineAlgorithm(baseline_momentum=0.5)
    loss_fn = ReinforceBaselineLoss(baseline_momentum=0.9)
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="1 of 1 baseline-momentum declarations disagrees",
    ) as exc_info:
        algorithm.setup(
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=loss_fn,
            dataloader=[_reinforce_batch(rewards=(1.0, 2.0))],
            config={"policy_logprob_column": "current_logprobs"},
        )
    message = str(exc_info.value)
    assert "0.5" in message
    assert "0.9" in message


def test_reinforce_refuses_zero_supervised_tokens() -> None:
    """An all-masked-out batch is unmeasurable, not a 0.0 loss.

    WHAT IS CLAIMED: SupervisionRefusal names the zero supervised count
    against the two offered rows.

    WHAT IS NOT CLAIMED: that any subset of masked positions fails; only
    the fully-masked case is measured.
    """
    loss_fn = ReinforceBaselineLoss()
    batch = _Batch(
        {
            "rewards": (1.0, 2.0),
            "loss_mask": ((0, 0), (0, 0)),
        }
    )
    with pytest.raises(
        SupervisionRefusal,
        match="0 supervised tokens remain across 2 rows",
    ):
        loss_fn(lambda b: ((-1.0, -1.0), (-1.0, -1.0)), batch)


def test_reinforce_pp_config_refusals_name_field_and_range() -> None:
    """Invalid epsilon or penalty weight refuses at construction.

    WHAT IS CLAIMED: zero/one/bool epsilons and a zero kl_beta refuse with
    the field name and admitted range in the message.

    WHAT IS NOT CLAIMED: that the defaults are canonical; bounds only.
    """
    with pytest.raises(LossConfigRefusal, match=r"clip_epsilon=0\.0"):
        ReinforcePlusPlusLoss(clip_epsilon=0.0)
    with pytest.raises(LossConfigRefusal, match=r"clip_epsilon=1\.0"):
        ReinforcePlusPlusLoss(clip_epsilon=1.0)
    with pytest.raises(LossConfigRefusal, match="True is not a clip epsilon"):
        ReinforcePlusPlusLoss(clip_epsilon=True)
    with pytest.raises(LossConfigRefusal, match=r"kl_beta=0\.0"):
        ReinforcePlusPlusLoss(kl_beta=0.0)
    with pytest.raises(LossConfigRefusal, match=r"clip_epsilon=0\.0"):
        ReinforcePlusPlusAlgorithm(clip_epsilon=0.0)


def test_reinforce_pp_refuses_singleton_global_denominator() -> None:
    """A one-sample batch leaves the global spread a manufactured zero.

    WHAT IS CLAIMED: the refusal names the 1-of-2 row shortfall before any
    normalisation.

    WHAT IS NOT CLAIMED: that a two-sample batch with equal penalised
    returns passes; that is the zero-spread refusal measured next.
    """
    loss_fn = ReinforcePlusPlusLoss()
    batch = _Batch(
        {
            "rewards": (1.0,),
            "loss_mask": ((1, 1),),
            "old_logprobs": ((-1.0, -1.0),),
            "reference_logprobs": ((-1.0, -1.0),),
        }
    )
    with pytest.raises(
        BatchRefusal,
        match="1 of at least 2 required batch rows",
    ):
        loss_fn(lambda b: ((-1.0, -1.0),), batch)


def test_reinforce_pp_refuses_zero_global_spread() -> None:
    """Equal penalised returns are a manufactured gradient, not a measured one.

    WHAT IS CLAIMED: zero global spread refuses with AdvantageRefusal naming
    both the sample count and the manufactured-zero reason -- where RLOO's
    leave-one-out keeps measured zeros, a z-score over zero spread has no
    measurement behind it at all.

    WHAT IS NOT CLAIMED: that small-but-nonzero spread is unstable; only
    the exact-zero case is measured here.
    """
    loss_fn = ReinforcePlusPlusLoss()
    batch = _Batch(
        {
            "rewards": (3.0, 3.0),
            "loss_mask": ((1, 1), (1, 1)),
            "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
            "reference_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
        }
    )
    with pytest.raises(
        AdvantageRefusal,
        match="zero spread in the penalised returns",
    ):
        loss_fn.compute_with_report(lambda b: ((-1.0, -1.0), (-1.0, -1.0)), batch)


def test_reinforce_pp_hand_computed_numbers_and_kl_fold() -> None:
    """Penalised returns, global z-scores, and clipping agree by hand.

    Without a reference offset: G = (2, 4), population mean 3, std 1, so
    advantages are (-1, +1); every raw ratio is exp(0.2), clipped to 1.2,
    giving contribution (exp(0.2) - 1.2) / 2. With reference offset 0.1 the
    penalty subtracts 0.04 * (2 * 0.1) from every return, shifting the
    penalised mean to 2.992 while the z-scores -- and thus the loss -- are
    unchanged; the shift proves the penalty went through the reward path.

    WHAT IS CLAIMED: both contributions, both penalised-return statistics,
    and ``verify_step`` on the folded single component.

    WHAT IS NOT CLAIMED: equivalence to any reference Reinforce++ run.
    """
    loss_fn = ReinforcePlusPlusLoss()
    expected = (math.exp(0.2) - 1.2) / 2.0
    plain_output, plain_stats = loss_fn.compute_with_report(
        lambda b: ((-2.0, -1.0), (-1.0, -1.0)),
        _reinforce_pp_batch(reference_offset=0.0),
    )
    assert plain_output.components[0].name == "reinforce_pp_loss"
    assert plain_output.components[0].contribution == pytest.approx(expected)
    assert plain_stats.count == 2
    assert plain_stats.mean == pytest.approx(3.0)
    assert plain_stats.std == pytest.approx(1.0)
    penalised_output, penalised_stats = loss_fn.compute_with_report(
        lambda b: ((-2.0, -1.0), (-1.0, -1.0)),
        _reinforce_pp_batch(reference_offset=0.1),
    )
    assert penalised_stats.mean == pytest.approx(2.992)
    assert penalised_stats.std == pytest.approx(1.0)
    assert penalised_output.loss == pytest.approx(expected)
    report = StepReport(
        step=0,
        loss=plain_output,
        rows=2,
        reward_stats=None,
    )
    assert verify_step(report, ReinforcePlusPlusAlgorithm().requirements()) == 1


def test_reinforce_pp_wiring_refuses_referenceless_pair() -> None:
    """A declared reference role cannot be attested by a referenceless pair.

    WHAT IS CLAIMED: the wiring handshake refuses naming the declared
    reference need against the pair's empty references.

    WHAT IS NOT CLAIMED: converse coverage (an unwanted reference) here --
    that refusal is GRPO-tested surface, and Reinforce++ declares the role
    True.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="declares a reference policy, but the policy pair carries no references",
    ):
        check_algorithm_wiring(
            ReinforcePlusPlusAlgorithm().requirements(),
            policy_pair=PolicyPair(train_view=object()),
            loss_fn=ReinforcePlusPlusLoss(),
            supplied={"rollout_source": None, "advantage_fn": None, "weight_sync": None},
        )


def test_rloo_wiring_refuses_unconsumed_references() -> None:
    """RLOO is reference-free; a loaded reference on the pair is a stub.

    WHAT IS CLAIMED: the both-directions reference check refuses when the
    pair carries a reference the declaration never consumes.

    WHAT IS NOT CLAIMED: that the reference was itself unusable; presence
    alone decides.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="declares no reference policy, but the policy pair carries 1 references",
    ):
        check_algorithm_wiring(
            RLOOAlgorithm().requirements(),
            policy_pair=PolicyPair(train_view=object(), references={"reference": object()}),
            loss_fn=RLOOPolicyLoss(),
            supplied={
                "rollout_source": None,
                "advantage_fn": LeaveOneOutAdvantage(),
                "weight_sync": None,
            },
        )


def test_declarations_and_wiring_consumed_roles_agree_for_all_three() -> None:
    """Algorithm and loss declarations agree, and the wiring check returns
    exactly the consumed-role set each family declares.

    WHAT IS CLAIMED: declared components and declared metric names are the
    same set stated twice, once per object; and the consumed tuples are
    ("advantage_fn",), (), and ("reference_policy",) respectively -- the
    empty middle answer is a measured agreement over a real union, not a
    vacuous pass, because the mandatory roles still sit in that union.

    WHAT IS NOT CLAIMED: that the consumed objects WORK; the only loss-side
    call is ``declaration()``, by design.
    """
    pairs = (
        (RLOOAlgorithm(), RLOOPolicyLoss(), ("advantage_fn",)),
        (ReinforceBaselineAlgorithm(), ReinforceBaselineLoss(), ()),
        (ReinforcePlusPlusAlgorithm(), ReinforcePlusPlusLoss(), ("reference_policy",)),
    )
    for algorithm, loss_fn, expected_consumed in pairs:
        requirements = algorithm.requirements()
        declaration = loss_fn.declaration()
        assert requirements.declared_components == declaration.components
        assert requirements.declared_metrics == tuple(metric.name for metric in declaration.metrics)
        pair = PolicyPair(
            train_view=object(),
            references=(
                {"reference": object()} if expected_consumed == ("reference_policy",) else {}
            ),
        )
        supplied: dict[str, Any] = {
            "rollout_source": None,
            "advantage_fn": None,
            "weight_sync": None,
        }
        if expected_consumed == ("advantage_fn",):
            supplied["advantage_fn"] = loss_fn.advantage_fn
        consumed = check_algorithm_wiring(
            requirements,
            policy_pair=pair,
            loss_fn=loss_fn,
            supplied=supplied,
        )
        assert consumed == expected_consumed
