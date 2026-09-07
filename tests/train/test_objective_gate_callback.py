"""The objective-gate callback: WHEN STEP_ZERO fires, what it carries, and the
two arms where it does not fire on a real loss.

Driven directly rather than through ``transformers.Trainer``. The callback base
degrades to ``object`` when the extra is absent, so these legs run in a
torch-free environment and pin the dispatch logic without a 90-second model
download standing between a defect and its red.

The four named arms (A/B/C/D) are the birth table measured before the callback
was written -- #239, a gate must land GREEN at birth. Arm A is the positive
control: without a green arm, the failing arms are indistinguishable from a
broken instrument.
"""

from __future__ import annotations

import re
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from foundationscale.train.loop import FoundationScaleObjectiveGate


def _callback(objective: str | None = "sft") -> FoundationScaleObjectiveGate:
    return FoundationScaleObjectiveGate(
        objective=objective,
        learning_rate=5e-5,
        per_device_batch_size=1,
        max_steps=20,
    )


def _control() -> SimpleNamespace:
    return SimpleNamespace(should_training_stop=False)


def _state(global_step: int) -> SimpleNamespace:
    return SimpleNamespace(global_step=global_step)


def _result_by_id(report: Any, gate_id: str) -> Any:
    # Gate result order is not part of the contract; select by id.
    for result in report.results:
        if result.gate_id == gate_id:
            return result
    msg = f"no gate result with gate_id {gate_id!r}"
    raise AssertionError(msg)


def test_arm_a_declared_sft_with_observed_loss_clears_the_whole_sweep() -> None:
    cb = _callback(objective="sft")
    control = _control()
    cb.on_train_begin(SimpleNamespace(), _state(0), control)
    cb.on_log(SimpleNamespace(), _state(10), control, logs={"loss": 2.5})
    assert cb._fired is True
    assert cb.blocked is False
    assert cb.unmeasured is False
    # A gate that fires and lets the run continue is a check that cannot fail,
    # so a clean sweep must leave the run alone.
    assert control.should_training_stop is False
    (report,) = cb.reports
    assert report.ok is True
    # The exact set, not the count: a bare `== 5` would let a sixth gate join the
    # sweep with no verdict asserted for it, which is the shape of a denominator
    # that grows while the claim about it stands still.
    assert {r.gate_id for r in report.results} == {
        "objective.declared",
        "objective.loss_components",
        "objective.metrics",
        "objective.reward_scale",
        "objective.hparam_drift",
    }
    assert len(report.results) == report.registered
    assert _result_by_id(report, "objective.declared").verdict.name == "PASS"
    assert _result_by_id(report, "objective.loss_components").verdict.name == "PASS"
    # SFT declares no reward term, so the reward gate abstains -- a SKIP stays
    # visible in the denominator and is never a PASS.
    assert _result_by_id(report, "objective.reward_scale").verdict.name == "SKIP"
    # Same shape on the metrics axis (#316): SFT emits no diagnostic metric, and
    # the empty case is a DECLARED abstention rather than a free PASS, so the
    # absence of a metric channel stays countable instead of reading as coverage.
    assert _result_by_id(report, "objective.metrics").verdict.name == "SKIP"
    assert _result_by_id(report, "objective.hparam_drift").verdict.name == "PASS"


def test_arm_b_undeclared_objective_blocks_as_vacuous() -> None:
    cb = _callback(objective=None)
    control = _control()
    cb.on_train_begin(SimpleNamespace(), _state(0), control)
    cb.on_log(SimpleNamespace(), _state(10), control, logs={"loss": 2.5})
    assert cb.blocked is True
    assert cb.unmeasured is False
    assert control.should_training_stop is True
    (report,) = cb.reports
    assert report.ok is False
    # An unstated objective is an absence of a fact, not a verified one: the
    # gate abstains, and the abstention itself blocks.
    assert _result_by_id(report, "objective.declared").verdict.name == "VACUOUS"


def test_arm_c_hparam_drift_after_train_begin_blocks_the_run() -> None:
    cb = _callback(objective="sft")
    control = _control()
    cb.on_train_begin(SimpleNamespace(), _state(0), control)
    # Mutate after the step-0 record is taken: the live mapping no longer
    # matches what the run declared at step 0.
    cb.learning_rate = 1e-4
    cb.on_log(SimpleNamespace(), _state(10), control, logs={"loss": 2.5})
    assert cb.blocked is True
    assert control.should_training_stop is True
    (report,) = cb.reports
    assert _result_by_id(report, "objective.hparam_drift").verdict.name == "FAIL"


def test_arm_d_no_loss_ever_logged_fires_the_backstop_as_unmeasured() -> None:
    cb = _callback(objective="sft")
    control = _control()
    cb.on_train_begin(SimpleNamespace(), _state(0), control)
    cb.on_train_end(SimpleNamespace(), _state(20), control)
    assert cb.unmeasured is True
    assert cb.blocked is False
    # A component the instrument never got to read is UNMEASURED, not RED, so
    # the run is not stopped over a missing log line.
    assert control.should_training_stop is False
    (report,) = cb.reports
    assert report.ok is False
    assert _result_by_id(report, "objective.loss_components").verdict.name == "FAIL"


def test_step_zero_fires_exactly_once_across_later_logs_and_train_end() -> None:
    cb = _callback()
    control = _control()
    cb.on_train_begin(SimpleNamespace(), _state(0), control)
    cb.on_log(SimpleNamespace(), _state(10), control, logs={"loss": 2.5})
    cb.on_log(SimpleNamespace(), _state(20), control, logs={"loss": 2.0})
    cb.on_train_end(SimpleNamespace(), _state(20), control)
    # The boolean guard -- not a step comparison -- is what makes "once" hold:
    # a second report would be a contradictory record of the same run.
    assert len(cb.reports) == 1


def test_eval_only_log_neither_dispatches_nor_records_a_loss() -> None:
    cb = _callback()
    control = _control()
    cb.on_train_begin(SimpleNamespace(), _state(0), control)
    cb.on_log(SimpleNamespace(), _state(10), control, logs={"eval_loss": 9.0})
    # An eval-only log is not a training measurement; recording it would let an
    # eval number stand in for the objective's loss at dispatch.
    assert cb.reports == []
    assert cb._last_loss is None


def test_none_logs_are_handled_and_dispatch_nothing() -> None:
    cb = _callback()
    control = _control()
    cb.on_train_begin(SimpleNamespace(), _state(0), control)
    returned = cb.on_log(SimpleNamespace(), _state(10), control, logs=None)
    assert returned is control
    assert cb.reports == []
    assert cb._last_loss is None


def test_loss_log_at_global_step_zero_does_not_dispatch() -> None:
    cb = _callback()
    control = _control()
    cb.on_train_begin(SimpleNamespace(), _state(0), control)
    cb.on_log(SimpleNamespace(), _state(0), control, logs={"loss": 2.5})
    # The guard is `< 1`: at step 0 the logging cadence has produced no real
    # measurement, so firing would grade the cadence knob, not the objective.
    assert cb.reports == []
    assert cb._fired is False
    # The step-0 log must not have consumed the single firing.
    cb.on_log(SimpleNamespace(), _state(1), control, logs={"loss": 2.4})
    assert len(cb.reports) == 1


def test_step0_record_is_immutable_once_train_begins() -> None:
    cb = _callback()
    cb.on_train_begin(SimpleNamespace(), _state(0), _control())
    assert isinstance(cb._step0_hparams, MappingProxyType)
    assert cb._step0_fingerprint is not None
    # hparam_drift compares the live mapping against this record; a writable
    # record would let the run edit what it claims to be checked against.
    record: Any = cb._step0_hparams
    with pytest.raises(TypeError):
        record["learning_rate"] = 1e-4


def test_dispatched_context_carries_live_hparams_against_the_step0_record() -> None:
    cb = _callback(objective="sft")
    control = _control()
    cb.on_train_begin(SimpleNamespace(), _state(0), control)
    cb.learning_rate = 1e-4
    cb.on_log(SimpleNamespace(), _state(10), control, logs={"loss": 2.5})
    (report,) = cb.reports
    drift = _result_by_id(report, "objective.hparam_drift")
    assert drift.verdict.name == "FAIL"
    # The changed-key diff can only name learning_rate if the context carried
    # the LIVE mapping against the frozen step-0 record; passing the snapshot
    # twice would make the comparison reflexive and the gate vacuously PASS.
    assert "learning_rate" in drift.detail


def test_backstop_cannot_be_skipped_when_no_log_ever_arrives() -> None:
    cb = _callback()
    control = _control()
    cb.on_train_begin(SimpleNamespace(), _state(0), control)
    cb.on_train_end(SimpleNamespace(), _state(20), control)
    # A gate that can quietly not run is the vacuous pass this codebase exists
    # to refuse: train-begin with no loss log must still yield exactly one report.
    assert len(cb.reports) == 1


def test_arm_e_undeclared_and_unobserved_blocks_and_reports_both_states(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Arm E: the backstop fires over a run that ALSO declared nothing.

    Not a fifth colour -- a real refusal and an abstention arriving in the same
    sweep, which is the shape a real transformers run produced. It is the arm
    that caught the banner defect: the count was taken over every blocker and
    the names over the refusals alone, so the operator's one line read "2
    blocking (objective.declared)" -- a numerator and a denominator from
    different sets, in the sentence that says what stopped the run.

    A refusal outranks an abstention, so the run stops; the abstention is
    disclosed rather than folded into the blocking count or dropped.
    """
    cb = _callback(objective=None)
    control = _control()
    cb.on_train_begin(SimpleNamespace(), _state(0), control)
    cb.on_train_end(SimpleNamespace(), _state(20), control)
    assert cb.blocked is True
    # NOT unmeasured: something was genuinely refuted, so the run is not
    # entitled to report that it merely could not look.
    assert cb.unmeasured is False
    assert control.should_training_stop is True
    (report,) = cb.reports
    assert _result_by_id(report, "objective.declared").verdict.name == "VACUOUS"
    assert _result_by_id(report, "objective.loss_components").verdict.name == "FAIL"
    assert len(report.blocking) == 2

    line = next(
        ln for ln in capsys.readouterr().out.splitlines() if "[fs:train:objective_gate]" in ln
    )
    assert "1 blocking (objective.declared)" in line, line
    assert "1 abstaining (objective.loss_components)" in line, line


def test_banner_count_and_names_agree_on_every_dispatching_arm(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The stated count equals the number of gate ids stated beside it.

    Written as a sweep rather than per-arm because the defect was not in any one
    arm's verdict -- every verdict was correct -- but in the line that reports
    them, and a line is easy to get right for the arm you are looking at.
    """
    arms: list[tuple[str, str | None, bool]] = [
        ("undeclared, observed", None, True),
        ("declared, unobserved", "sft", False),
        ("undeclared, unobserved", None, False),
    ]
    for name, objective, observed in arms:
        cb = _callback(objective=objective)
        control = _control()
        cb.on_train_begin(SimpleNamespace(), _state(0), control)
        if observed:
            cb.on_log(SimpleNamespace(), _state(10), control, logs={"loss": 2.5})
        cb.on_train_end(SimpleNamespace(), _state(20), control)
        line = next(
            ln for ln in capsys.readouterr().out.splitlines() if "[fs:train:objective_gate]" in ln
        )
        for count_word in ("blocking", "abstaining"):
            claimed = re.search(rf"(\d+) {count_word} \(([^)]*)\)", line)
            if claimed is None:
                continue
            named = [g for g in claimed.group(2).split(", ") if g]
            assert int(claimed.group(1)) == len(named), f"{name}: {line}"


def test_arm_a_is_the_positive_control_for_the_fail_arms() -> None:
    cb = _callback(objective="sft")
    control = _control()
    cb.on_train_begin(SimpleNamespace(), _state(0), control)
    cb.on_log(SimpleNamespace(), _state(10), control, logs={"loss": 2.5})
    (report,) = cb.reports
    # Without a green arm the FAIL arms above are just a broken instrument: the
    # same sweep must be able to PASS a healthy run.
    assert report.ok is True
    assert not report.blocking
