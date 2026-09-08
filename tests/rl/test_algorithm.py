import math
from typing import Any, cast

import pytest

from foundationscale.gates.objective_gates import (
    LossComponent,
    MetricExpectation,
    MetricObservation,
)
from foundationscale.rl.advantage import RewardStats
from foundationscale.rl.algorithm import (
    Algorithm,
    AlgorithmRequirements,
    AlgorithmWiringRefusal,
    StepReport,
    StepReportRefusal,
    check_algorithm_wiring,
    verify_step,
)
from foundationscale.rl.interfaces import LossDeclaration, LossFn, LossOutput
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.weightsync import SyncReport


def _component(
    name: str,
    *,
    weight: float = 1.0,
    observed: bool = True,
    contribution: float | None = 0.5,
) -> LossComponent:
    return LossComponent(name=name, weight=weight, observed=observed, contribution=contribution)


def _metric(name: str, value: float | None = 0.75) -> MetricObservation:
    return MetricObservation(name=name, value=value)


def _loss(
    *names: str, loss: float = 1.25, metrics: tuple[MetricObservation, ...] = ()
) -> LossOutput:
    return LossOutput(
        loss=loss, components=tuple(_component(name) for name in names), metrics=metrics
    )


def _requirements(**overrides: object) -> AlgorithmRequirements:
    fields: dict[str, object] = {
        "name": "grpo",
        "requires_rollout": True,
        "requires_advantage": True,
        "requires_weight_sync": True,
        "requires_reference_policy": False,
        "declared_components": ("pg_loss",),
        "declared_metrics": (),
    }
    fields.update(overrides)
    return AlgorithmRequirements(**cast(dict[str, Any], fields))


def _report(**overrides: object) -> StepReport:
    fields: dict[str, object] = {
        "step": 3,
        "loss": _loss("pg_loss"),
        "rows": 4,
        "reward_stats": RewardStats(count=4, mean=0.5, std=0.1, minimum=0.0, maximum=1.0),
        "sync": SyncReport(transport="reshard", offered=("w.a",), transferred=("w.a",), skipped=()),
    }
    fields.update(overrides)
    return StepReport(**cast(dict[str, Any], fields))


def _pair(**overrides: object) -> PolicyPair:
    fields: dict[str, object] = {
        "train_view": object(),
        "generate_view": object(),
        "references": {},
    }
    fields.update(overrides)
    return PolicyPair(**cast(dict[str, Any], fields))


def _expectation(name: str, *, low: float = 0.0, high: float = 1.0) -> MetricExpectation:
    return MetricExpectation(name=name, low=low, high=high)


def _loss_fn(*components: str, metrics: tuple[MetricExpectation, ...] = ()) -> LossFn:
    class _Loss:
        def declaration(self) -> LossDeclaration:
            return LossDeclaration(components=components, metrics=metrics)

        def __call__(self, *_args: Any, **_kwargs: Any) -> LossOutput:
            # check_algorithm_wiring only ever calls declaration(); a double
            # whose forward pass "worked" would pin behaviour the contract
            # explicitly does not exercise.
            raise NotImplementedError("the wiring handshake never calls the loss")

    return cast(LossFn, _Loss())


def test_requirements_refuses_a_non_string_name() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match="cannot be attributed"):
        _requirements(name=7)


def test_requirements_refuses_a_whitespace_only_name() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match="an algorithm with no name"):
        _requirements(name="   ")


def test_requirements_refuses_empty_declared_components_as_vacuous() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match="vacuously"):
        _requirements(declared_components=())


def test_requirements_refuses_a_non_name_component_entry_and_names_its_index() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match=r"declared_components\[1\]=0") as excinfo:
        _requirements(declared_components=("pg_loss", 0))
    assert "absence of a name is not a name" in str(excinfo.value)


def test_requirements_refuses_a_repeated_component_name() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match=r"repeats 'pg_loss'"):
        _requirements(declared_components=("pg_loss", "pg_loss"))


def test_requirements_refuses_a_non_name_metric_entry() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match=r"declared_metrics\[0\]=None"):
        _requirements(declared_metrics=(None,))


def test_requirements_refuses_a_repeated_metric_name() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match=r"repeats 'accuracy'"):
        _requirements(declared_metrics=("accuracy", "accuracy"))


def test_requirements_refuses_a_name_declared_as_both_component_and_metric() -> None:
    # A component is a term OF the loss and a metric is a reading the loss
    # does not optimise: one name in both is one quantity in two denominators,
    # and both gates would grade it as covered.
    with pytest.raises(AlgorithmWiringRefusal, match="appear in BOTH") as excinfo:
        _requirements(declared_components=("pg_loss", "kl"), declared_metrics=("kl",))
    assert "kl" in str(excinfo.value)


def test_requirements_refuses_weight_sync_without_rollout() -> None:
    # A sync moves the training view into the generation view, and with no
    # rollout there is no generation view -- the report would describe a
    # transfer to nowhere.
    with pytest.raises(AlgorithmWiringRefusal, match="requires_weight_sync without"):
        _requirements(requires_rollout=False)


def test_requirements_accepts_advantage_without_rollout_for_offline_rl() -> None:
    # Positive control for the asymmetry above: refusing this shape would
    # make offline RL -- advantages over rollouts a previous run generated --
    # unrepresentable.
    requirements = _requirements(requires_rollout=False, requires_weight_sync=False)
    assert requirements.requires_advantage is True
    assert requirements.requires_rollout is False
    assert requirements.requires_weight_sync is False


def test_requirements_accepts_empty_declared_metrics() -> None:
    requirements = _requirements(declared_metrics=())
    assert requirements.declared_metrics == ()


def test_requirements_normalises_list_arguments_to_tuples() -> None:
    requirements = _requirements(declared_components=["pg_loss"], declared_metrics=["acc"])
    assert requirements.declared_components == ("pg_loss",)
    assert requirements.declared_metrics == ("acc",)


def test_step_report_refuses_a_bool_step_because_bool_is_int() -> None:
    with pytest.raises(StepReportRefusal, match="True is not a step index"):
        _report(step=True)


def test_step_report_refuses_a_negative_step() -> None:
    with pytest.raises(StepReportRefusal, match="a step index is an int >= 0"):
        _report(step=-1)


def test_step_report_accepts_step_zero_as_a_real_step() -> None:
    report = _report(step=0)
    assert report.step == 0


def test_step_report_refuses_a_bool_rows() -> None:
    with pytest.raises(StepReportRefusal, match="True is not a row count"):
        _report(rows=True)


def test_step_report_refuses_zero_rows_because_zero_is_not_a_denominator() -> None:
    with pytest.raises(StepReportRefusal, match="not a denominator"):
        _report(rows=0)


def test_step_report_refuses_negative_rows() -> None:
    with pytest.raises(StepReportRefusal, match="not a denominator"):
        _report(rows=-2)


def test_step_report_refuses_a_loss_that_is_not_a_loss_output() -> None:
    with pytest.raises(StepReportRefusal, match="not a LossOutput"):
        _report(loss=object())


def test_step_report_refuses_an_empty_loss_components_tuple() -> None:
    # LossOutput permits the empty tuple because a loss object is not a step;
    # a STEP that reports one has its whole objective outside the coverage
    # gate's denominator.
    with pytest.raises(StepReportRefusal, match="loss.components is EMPTY"):
        _report(loss=_loss())


def test_step_report_refuses_reward_stats_count_that_disagrees_with_rows() -> None:
    # One step would carry two denominators and each gate prices against
    # whichever it reads, so the refusal must state both counts.
    with pytest.raises(StepReportRefusal, match=r"2 of 4") as excinfo:
        _report(reward_stats=RewardStats(count=2, mean=0.5, std=0.1, minimum=0.0, maximum=1.0))
    message = str(excinfo.value)
    assert "2" in message and "4" in message


def test_step_report_accepts_reward_stats_count_equal_to_rows() -> None:
    stats = RewardStats(count=4, mean=0.5, std=0.1, minimum=0.0, maximum=1.0)
    report = _report(reward_stats=stats, rows=4)
    assert report.reward_stats is stats
    assert report.rows == 4


def test_step_report_refuses_reward_stats_that_is_not_a_reward_stats() -> None:
    with pytest.raises(StepReportRefusal, match="summary this contract cannot read"):
        _report(reward_stats=object())


def test_step_report_refuses_a_sync_that_is_not_a_sync_report() -> None:
    with pytest.raises(StepReportRefusal, match="transfer record this contract cannot read"):
        _report(sync=object())


def test_step_report_round_trips_a_nan_loss_because_divergence_must_be_visible() -> None:
    # A diverged step is the single event a step record most needs to carry;
    # a validator that deleted it would make divergence unrepresentable and
    # the report would go quiet at exactly the moment it is needed. The
    # opposite call from SyncReport.seconds: seconds is an instrument
    # reading, loss is the observation.
    report = _report(loss=_loss("pg_loss", loss=math.nan))
    assert math.isnan(report.loss.loss)


def test_step_report_round_trips_an_infinite_loss() -> None:
    report = _report(loss=_loss("pg_loss", loss=math.inf))
    assert report.loss.loss == math.inf


def test_step_report_round_trips_none_reward_stats_and_sync_as_none() -> None:
    report = _report(reward_stats=None, sync=None)
    assert report.reward_stats is None
    assert report.sync is None


def test_step_report_exposes_no_self_grading_property() -> None:
    # A report that grades itself is a detector inside its own denominator;
    # the verdict is the gate plane's, and it must be taken by something
    # that can also say NO.
    report = _report()
    for name in ("ok", "passed", "healthy", "success", "failed"):
        assert not hasattr(report, name), f"StepReport must not expose self-grading {name!r}"


def test_step_report_does_not_restate_the_loss_decomposition() -> None:
    # The report holds the LossOutput itself, so report.loss.components is
    # the one edition; a second copy on the report would be two countables
    # for one quantity -- the drift class this repository keeps filing.
    report = _report()
    for name in ("components", "metrics", "loss_per_row"):
        assert not hasattr(report, name), f"StepReport must not restate {name!r}"


def test_algorithm_protocol_rejects_a_class_with_no_step_method() -> None:
    class _NoStep:
        def requirements(self) -> AlgorithmRequirements:
            return _requirements()

        def setup(self) -> None:
            return None

    assert not isinstance(_NoStep(), Algorithm)


def test_algorithm_protocol_rejects_a_class_with_no_requirements_method() -> None:
    class _NoRequirements:
        def setup(self) -> None:
            return None

        def step(self) -> StepReport:
            return _report()

    assert not isinstance(_NoRequirements(), Algorithm)


def test_algorithm_protocol_accepts_a_class_with_all_three_methods() -> None:
    class _Complete:
        def requirements(self) -> AlgorithmRequirements:
            return _requirements()

        def setup(self) -> None:
            return None

        def step(self) -> StepReport:
            return _report()

    assert isinstance(_Complete(), Algorithm)


def test_algorithm_protocol_accepts_wrong_step_signature_because_presence_only() -> None:
    # @runtime_checkable checks method PRESENCE only, never signatures.
    # Pinning this stops a reader mistaking the isinstance smoke test for
    # the handshake -- check_algorithm_wiring is the handshake.
    class _WrongStepSignature:
        def requirements(self) -> AlgorithmRequirements:
            return _requirements()

        def setup(self) -> None:
            return None

        def step(self, bogus: int, another: str) -> int:
            return 0

    assert isinstance(_WrongStepSignature(), Algorithm)


def test_wiring_refuses_a_required_rollout_source_handed_none() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match="rollout_source is required"):
        check_algorithm_wiring(
            _requirements(),
            policy_pair=_pair(),
            loss_fn=_loss_fn("pg_loss"),
            rollout_source=None,
            advantage_fn=cast(Any, object()),
            weight_sync=cast(Any, object()),
        )


def test_wiring_refuses_a_supplied_rollout_source_the_algorithm_does_not_consume() -> None:
    # A component the algorithm does not consume is absent, never stubbed;
    # a supplied-but-unconsumed component is a stub by another name and it
    # reads as wiring that works.
    requirements = _requirements(
        requires_rollout=False, requires_advantage=False, requires_weight_sync=False
    )
    with pytest.raises(AlgorithmWiringRefusal, match="rollout_source was supplied"):
        check_algorithm_wiring(
            requirements,
            policy_pair=_pair(generate_view=None),
            loss_fn=_loss_fn("pg_loss"),
            rollout_source=cast(Any, object()),
        )


def test_wiring_refuses_a_supplied_advantage_fn_the_algorithm_does_not_consume() -> None:
    requirements = _requirements(
        requires_rollout=False, requires_advantage=False, requires_weight_sync=False
    )
    with pytest.raises(AlgorithmWiringRefusal, match="advantage_fn was supplied"):
        check_algorithm_wiring(
            requirements,
            policy_pair=_pair(generate_view=None),
            loss_fn=_loss_fn("pg_loss"),
            advantage_fn=cast(Any, object()),
        )


def test_wiring_refuses_a_supplied_weight_sync_the_algorithm_does_not_consume() -> None:
    requirements = _requirements(
        requires_rollout=False, requires_advantage=False, requires_weight_sync=False
    )
    with pytest.raises(AlgorithmWiringRefusal, match="weight_sync was supplied"):
        check_algorithm_wiring(
            requirements,
            policy_pair=_pair(generate_view=None),
            loss_fn=_loss_fn("pg_loss"),
            weight_sync=cast(Any, object()),
        )


def test_wiring_names_every_disagreeing_role_in_one_refusal() -> None:
    # A refusal that stops at the first role makes the operator rediscover
    # the same class three times.
    with pytest.raises(AlgorithmWiringRefusal, match=r"2 of 3 component roles") as excinfo:
        check_algorithm_wiring(
            _requirements(),
            policy_pair=_pair(),
            loss_fn=_loss_fn("pg_loss"),
            rollout_source=None,
            advantage_fn=None,
            weight_sync=cast(Any, object()),
        )
    message = str(excinfo.value)
    assert "rollout_source" in message
    assert "advantage_fn" in message


def test_wiring_refuses_reference_requirement_against_a_pair_with_no_references() -> None:
    requirements = _requirements(requires_reference_policy=True)
    with pytest.raises(AlgorithmWiringRefusal, match="carries no references"):
        check_algorithm_wiring(
            requirements,
            policy_pair=_pair(references={}),
            loss_fn=_loss_fn("pg_loss"),
            rollout_source=cast(Any, object()),
            advantage_fn=cast(Any, object()),
            weight_sync=cast(Any, object()),
        )


def test_wiring_refuses_unrequired_references_and_names_the_reference_key() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match="ref_a") as excinfo:
        check_algorithm_wiring(
            _requirements(),
            policy_pair=_pair(references={"ref_a": object()}),
            loss_fn=_loss_fn("pg_loss"),
            rollout_source=cast(Any, object()),
            advantage_fn=cast(Any, object()),
            weight_sync=cast(Any, object()),
        )
    assert "ref_a" in str(excinfo.value)


def test_wiring_refuses_rollout_requirement_against_a_pair_with_no_generate_view() -> None:
    # This is a cross-check between two contracts that neither can make
    # alone: the pair does not know what the algorithm declares, and the
    # algorithm does not know what the pair carries.
    with pytest.raises(AlgorithmWiringRefusal, match="no generation view"):
        check_algorithm_wiring(
            _requirements(),
            policy_pair=_pair(generate_view=None),
            loss_fn=_loss_fn("pg_loss"),
            rollout_source=cast(Any, object()),
            advantage_fn=cast(Any, object()),
            weight_sync=cast(Any, object()),
        )


def test_wiring_returns_the_sorted_consumed_roles_for_grpo() -> None:
    consumed = check_algorithm_wiring(
        _requirements(),
        policy_pair=_pair(),
        loss_fn=_loss_fn("pg_loss"),
        rollout_source=cast(Any, object()),
        advantage_fn=cast(Any, object()),
        weight_sync=cast(Any, object()),
    )
    assert consumed == ("advantage_fn", "rollout_source", "weight_sync")


def test_wiring_includes_reference_policy_in_consumed_roles_when_required() -> None:
    requirements = _requirements(requires_reference_policy=True)
    consumed = check_algorithm_wiring(
        requirements,
        policy_pair=_pair(references={"ref_b": object()}),
        loss_fn=_loss_fn("pg_loss"),
        rollout_source=cast(Any, object()),
        advantage_fn=cast(Any, object()),
        weight_sync=cast(Any, object()),
    )
    assert "reference_policy" in consumed
    assert consumed == (
        "advantage_fn",
        "reference_policy",
        "rollout_source",
        "weight_sync",
    )


def test_wiring_returns_empty_tuple_for_the_degenerate_sft_shape() -> None:
    # SFT must stay degenerate; if it needed stubs to satisfy this
    # handshake the abstraction would be drawn wrong, so this shape needs
    # neither role objects nor a generation view on the pair.
    requirements = _requirements(
        name="sft",
        requires_rollout=False,
        requires_advantage=False,
        requires_weight_sync=False,
        requires_reference_policy=False,
    )
    consumed = check_algorithm_wiring(
        requirements,
        policy_pair=_pair(generate_view=None, references={}),
        loss_fn=_loss_fn("pg_loss"),
    )
    assert consumed == ()


def test_wiring_refuses_a_policy_pair_that_is_not_a_policy_pair() -> None:
    # Every other refusal here names the disagreement it found; reading the
    # views off a foreign object would raise an AttributeError that names
    # nothing.
    with pytest.raises(AlgorithmWiringRefusal, match="rather than a PolicyPair"):
        check_algorithm_wiring(
            _requirements(), policy_pair=cast(Any, object()), loss_fn=_loss_fn("pg_loss")
        )


def test_wiring_accepts_an_unrequired_generation_view() -> None:
    # Positive control for the deliberate asymmetry against unrequired
    # references: a generation view is a VIEW onto weights the pair already
    # holds and one pair is legitimately handed to a sequence of phases,
    # whereas a reference is a SECOND, separately loaded frozen model that
    # nothing would read.
    requirements = _requirements(
        name="sft",
        requires_rollout=False,
        requires_advantage=False,
        requires_weight_sync=False,
        requires_reference_policy=False,
    )
    consumed = check_algorithm_wiring(
        requirements, policy_pair=_pair(references={}), loss_fn=_loss_fn("pg_loss")
    )
    assert consumed == ()


def test_wiring_refuses_a_loss_declaration_that_repeats_a_component() -> None:
    # The loss side of the comparison would have no unambiguous denominator,
    # so no agreement with it could be measured -- this is checked before
    # the set comparison that would hide the repeat.
    with pytest.raises(AlgorithmWiringRefusal, match=r"'pg_loss' twice"):
        check_algorithm_wiring(
            _requirements(),
            policy_pair=_pair(),
            loss_fn=_loss_fn("pg_loss", "pg_loss"),
            rollout_source=cast(Any, object()),
            advantage_fn=cast(Any, object()),
            weight_sync=cast(Any, object()),
        )


def test_wiring_refuses_disagreeing_component_sets_with_both_direction_counts() -> None:
    # These are ONE countable declared twice, in two objects. verify_step
    # would also catch it, but only at step one -- after an allocation has
    # been burned; at setup it costs one method call.
    requirements = _requirements(declared_components=("pg_loss", "kl_loss"))
    with pytest.raises(AlgorithmWiringRefusal, match="different objective") as excinfo:
        check_algorithm_wiring(
            requirements,
            policy_pair=_pair(),
            loss_fn=_loss_fn("pg_loss", "kl_loss", "entropy"),
            rollout_source=cast(Any, object()),
            advantage_fn=cast(Any, object()),
            weight_sync=cast(Any, object()),
        )
    message = str(excinfo.value)
    # Each side is counted against ITS OWN denominator: the algorithm names
    # 2 and none of them are missing from the loss; the loss names 3 and one
    # of those 3 is missing from the algorithm. A shared denominator here
    # would report "1 of 2" and quietly price the loss's extra term against
    # the wrong total.
    assert "0 of 2" in message
    assert "1 of 3" in message
    assert "entropy" in message


def test_wiring_refuses_equal_sized_but_different_component_sets() -> None:
    requirements = _requirements(declared_components=("pg_loss", "kl_loss"))
    with pytest.raises(AlgorithmWiringRefusal, match=r"1 of 2") as excinfo:
        check_algorithm_wiring(
            requirements,
            policy_pair=_pair(),
            loss_fn=_loss_fn("pg_loss", "entropy"),
            rollout_source=cast(Any, object()),
            advantage_fn=cast(Any, object()),
            weight_sync=cast(Any, object()),
        )
    message = str(excinfo.value)
    assert "kl_loss" in message
    assert "entropy" in message


def test_wiring_refuses_a_loss_declaration_that_repeats_a_metric() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match=r"metric 'accuracy' twice"):
        check_algorithm_wiring(
            _requirements(),
            policy_pair=_pair(),
            loss_fn=_loss_fn(
                "pg_loss", metrics=(_expectation("accuracy"), _expectation("accuracy"))
            ),
            rollout_source=cast(Any, object()),
            advantage_fn=cast(Any, object()),
            weight_sync=cast(Any, object()),
        )


def test_wiring_refuses_disagreeing_metric_sets() -> None:
    requirements = _requirements(declared_metrics=("kl",))
    with pytest.raises(AlgorithmWiringRefusal, match="different diagnostics") as excinfo:
        check_algorithm_wiring(
            requirements,
            policy_pair=_pair(),
            loss_fn=_loss_fn("pg_loss", metrics=(_expectation("accuracy"),)),
            rollout_source=cast(Any, object()),
            advantage_fn=cast(Any, object()),
            weight_sync=cast(Any, object()),
        )
    message = str(excinfo.value)
    assert "kl" in message
    assert "accuracy" in message


def test_wiring_passes_when_both_metric_sides_are_empty() -> None:
    consumed = check_algorithm_wiring(
        _requirements(),
        policy_pair=_pair(),
        loss_fn=_loss_fn("pg_loss"),
        rollout_source=cast(Any, object()),
        advantage_fn=cast(Any, object()),
        weight_sync=cast(Any, object()),
    )
    assert consumed == ("advantage_fn", "rollout_source", "weight_sync")


def test_wiring_passes_when_both_metric_sides_name_the_same_metric() -> None:
    requirements = _requirements(declared_metrics=("accuracy",))
    consumed = check_algorithm_wiring(
        requirements,
        policy_pair=_pair(),
        loss_fn=_loss_fn("pg_loss", metrics=(_expectation("accuracy"),)),
        rollout_source=cast(Any, object()),
        advantage_fn=cast(Any, object()),
        weight_sync=cast(Any, object()),
    )
    # Asserting the consumed tuple proves the function ran past the metric
    # cross-check rather than merely not raising somewhere earlier.
    assert consumed == ("advantage_fn", "rollout_source", "weight_sync")


def test_verify_step_refuses_a_duplicate_observed_component_name() -> None:
    # Two entries with one name collapse in any set comparison, so the
    # repeat is checked before the comparison that would hide it.
    report = _report(loss=_loss("pg_loss", "pg_loss"))
    with pytest.raises(StepReportRefusal, match=r"'pg_loss' twice"):
        verify_step(report, _requirements())


def test_verify_step_refuses_mismatched_components_with_both_direction_counts() -> None:
    report = _report(loss=_loss("pg_loss", "entropy"))
    with pytest.raises(StepReportRefusal, match="does not account") as excinfo:
        verify_step(report, _requirements())
    message = str(excinfo.value)
    assert "1 of 2" in message
    assert "0 of 1" in message
    assert "entropy" in message


def test_verify_step_refuses_equal_sized_observed_and_declared_sets_that_differ() -> None:
    # A message built from counts alone would read "observed 2, declared 2"
    # here -- a refusal that states no reason over two genuinely different
    # sets, so the offending NAMES must appear.
    requirements = _requirements(declared_components=("pg_loss", "kl_loss"))
    report = _report(loss=_loss("pg_loss", "entropy"))
    with pytest.raises(StepReportRefusal, match=r"1 of 2") as excinfo:
        verify_step(report, requirements)
    message = str(excinfo.value)
    assert "entropy" in message
    assert "kl_loss" in message


def test_verify_step_refuses_a_duplicate_observed_metric_name() -> None:
    report = _report(loss=_loss("pg_loss", metrics=(_metric("acc"), _metric("acc"))))
    with pytest.raises(StepReportRefusal, match=r"metric 'acc' twice"):
        verify_step(report, _requirements())


def test_verify_step_refuses_observed_metrics_that_differ_from_declared() -> None:
    requirements = _requirements(declared_metrics=("kl",))
    report = _report(loss=_loss("pg_loss", metrics=(_metric("acc"),)))
    with pytest.raises(StepReportRefusal, match="does not account") as excinfo:
        verify_step(report, requirements)
    message = str(excinfo.value)
    assert "kl" in message
    assert "acc" in message


def test_verify_step_passes_when_both_metric_sides_are_empty() -> None:
    assert verify_step(_report(), _requirements()) == 1


def test_verify_step_refuses_reward_stats_when_no_advantage_function() -> None:
    requirements = _requirements(requires_advantage=False)
    with pytest.raises(StepReportRefusal, match="declares no advantage function"):
        verify_step(_report(), requirements)


def test_verify_step_refuses_missing_reward_stats_when_advantage_required() -> None:
    with pytest.raises(StepReportRefusal, match="no reward statistics"):
        verify_step(_report(reward_stats=None), _requirements())


def test_verify_step_refuses_a_reported_sync_when_no_weight_sync_consumed() -> None:
    requirements = _requirements(requires_weight_sync=False)
    with pytest.raises(StepReportRefusal, match="declares no weight sync"):
        verify_step(_report(), requirements)


def test_verify_step_allows_a_missing_sync_between_sync_cadence_steps() -> None:
    # Deliberate asymmetry with the reward-statistics checks: a weight sync
    # happens on a CADENCE, so a step with no sync is a step BETWEEN syncs,
    # not a step that lost one; refusing it would make every cadence other
    # than every-step unrepresentable.
    assert verify_step(_report(sync=None), _requirements()) == 1


def test_verify_step_returns_the_observed_component_count() -> None:
    requirements = _requirements(declared_components=("pg_loss", "kl_loss"))
    report = _report(loss=_loss("pg_loss", "kl_loss"))
    assert verify_step(report, requirements) == 2


def test_verify_step_counts_components_regardless_of_their_order() -> None:
    # Declared and observed name the same two components in OPPOSITE order,
    # so this pins that the comparison is by set and the count is unaffected
    # by ordering.
    #
    # What this test does NOT establish, despite the return being written as
    # len(report.loss.components): that the OBSERVED count is returned rather
    # than the DECLARED one. It cannot. Every path reaching the return has
    # already refused a repeat on both sides and refused set inequality, so
    # the two lengths are equal by construction and no input distinguishes
    # them. Substituting one for the other is an equivalent mutant; a test
    # named for that distinction would be claiming a discrimination the
    # guards above have removed.
    requirements = _requirements(declared_components=("pg_loss", "kl_loss"))
    report = _report(loss=_loss("kl_loss", "pg_loss"))
    assert verify_step(report, requirements) == 2
