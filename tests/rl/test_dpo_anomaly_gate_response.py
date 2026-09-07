"""PHASE3_DESIGN section 7 item 4: would the objective gates have caught Phase 2's DPO anomaly?

Phase 2 ran NeMo-RL's DPO recipe on `Qwen2.5-0.5B-Instruct` for ten steps. The
loss moved (0.6931 -> 0.6823) but `sft_loss` and `accuracy` both read exactly
`0.0000` at every step. PHASE2_REPRODUCTION section 6 lists that as unexplained,
and PHASE3_DESIGN section 9 makes this measurement the gate on stage 2: the
design claims the gate plane is the right instrument for this class of defect,
and until it is measured the claim is UNMEASURED.

The Phase 2 run's config is on the cluster, not in this repository, so WHY the
readings are zero is not settled here. That is deliberate and it does not block
the measurement: the question section 7 asks is what the GATES return given the
reading, and that can be answered for every hypothesis about what the run
declared without knowing which hypothesis is true. Three exhaust the space --
the term is declared and unweighted, declared and inert, or not declared at all.

The measured answer is split, and the split is the finding:

* `sft_loss` at 0.0000 IS caught -- but only when the objective DECLARES
  `sft_loss` as a loss component (H1, H2). Undeclared, the same reading passes
  four green gates (H3). Declaration is what puts a component in the
  denominator, and NeMo-RL logs `sft_loss` whether or not it is an active term.
* `accuracy` at 0.0000 was caught by NOTHING, and could not be: no field of
  `ObjectiveGateContext` could carry a diagnostic metric. That gap is #316, and
  it is now closed -- the last three tests measure the channel that closed it,
  including the one conditional that remains open.

Arm CONTROL is the positive control (#239). Without a green arm the firing arms
are indistinguishable from a comparator that refuses everything.
"""

from __future__ import annotations

import dataclasses
import inspect

from foundationscale.gates.core import REGISTRY, Lifecycle, run_event
from foundationscale.gates.objective_gates import (
    LossComponent,
    MetricExpectation,
    MetricObservation,
    ObjectiveGateContext,
    ValueProvenance,
    fingerprint_hparams,
)

# The Phase 2 DPO run's recorded hyperparameters (PHASE2_REPRODUCTION section 3.2).
# Held identical on both sides of the drift comparison: this measurement is about
# the loss-component axis, and a drifting fingerprint would add a second red that
# has nothing to do with the reading under test.
_HPARAMS = {"learning_rate": 5e-06, "global_batch_size": 128, "micro_batch_size": 2}
_FINGERPRINT = fingerprint_hparams(_HPARAMS)
_OBJECTIVE = ValueProvenance(name="objective", value="dpo", source="config", recorded=True)

# The healthy half of the reading: the preference term did move.
_PREFERENCE_OK = LossComponent(
    name="preference_loss", weight=1.0, observed=True, contribution=0.6823
)


def _context(
    components: tuple[LossComponent, ...],
    declared: tuple[str, ...],
    *,
    declared_metrics: tuple[MetricExpectation, ...] = (),
    metrics: tuple[MetricObservation, ...] = (),
) -> ObjectiveGateContext:
    # The metric arguments default to empty so the loss-axis arms below read
    # exactly as they did before #316: those arms are about the loss channel, and
    # a metric they never mention must not silently change what they assert.
    return ObjectiveGateContext(
        objective=_OBJECTIVE,
        declared_components=declared,
        components=components,
        uses_rewards=False,
        declared_metrics=declared_metrics,
        metrics=metrics,
        step0_fingerprint=_FINGERPRINT,
        step0_hparams=_HPARAMS,
        current_hparams=_HPARAMS,
        origin="phase2-dpo-step10",
    )


def _sweep(ctx: ObjectiveGateContext) -> object:
    # The production dispatch, not a hand-rolled loop over the gate classes:
    # measuring anything else would answer a question about this test file.
    return run_event(
        REGISTRY, Lifecycle.STEP_ZERO, {ObjectiveGateContext: ctx}, missing_ctx="report-skip"
    )


def _verdict(report: object, gate_id: str) -> str:
    for result in report.results:  # type: ignore[attr-defined]
        if result.gate_id == gate_id:
            return str(result.verdict.name)
    msg = f"no gate result with gate_id {gate_id!r}"
    raise AssertionError(msg)


def _detail(report: object, gate_id: str) -> str:
    for result in report.results:  # type: ignore[attr-defined]
        if result.gate_id == gate_id:
            return str(result.detail)
    msg = f"no gate result with gate_id {gate_id!r}"
    raise AssertionError(msg)


def test_control_a_healthy_dpo_step_passes_the_whole_sweep() -> None:
    """Positive control: a DPO step with both terms live must come back clean."""
    report = _sweep(
        _context(
            (
                _PREFERENCE_OK,
                LossComponent(name="sft_loss", weight=0.1, observed=True, contribution=0.0342),
            ),
            ("preference_loss", "sft_loss"),
        )
    )
    assert report.ok is True  # type: ignore[attr-defined]
    assert _verdict(report, "objective.declared") == "PASS"
    assert _verdict(report, "objective.loss_components") == "PASS"
    # DPO declares no reward term at this stage, so the reward gate abstains. A
    # SKIP stays visible in the denominator and is never counted as a PASS.
    assert _verdict(report, "objective.reward_scale") == "SKIP"
    assert _verdict(report, "objective.hparam_drift") == "PASS"


def test_h1_declared_component_with_zero_weight_is_caught() -> None:
    """`sft_loss` present but weighted 0.0 -- the auxiliary-term-disabled shape."""
    report = _sweep(
        _context(
            (
                _PREFERENCE_OK,
                LossComponent(name="sft_loss", weight=0.0, observed=True, contribution=0.0),
            ),
            ("preference_loss", "sft_loss"),
        )
    )
    assert report.ok is False  # type: ignore[attr-defined]
    assert _verdict(report, "objective.loss_components") == "FAIL"
    assert "weight 0.0" in _detail(report, "objective.loss_components")
    assert "sft_loss" in _detail(report, "objective.loss_components")


def test_h2_declared_component_that_is_weighted_but_inert_is_caught() -> None:
    """`sft_loss` present and weighted, contributing exactly 0.0."""
    report = _sweep(
        _context(
            (
                _PREFERENCE_OK,
                LossComponent(name="sft_loss", weight=1.0, observed=True, contribution=0.0),
            ),
            ("preference_loss", "sft_loss"),
        )
    )
    assert report.ok is False  # type: ignore[attr-defined]
    assert _verdict(report, "objective.loss_components") == "FAIL"
    assert "exactly 0.0" in _detail(report, "objective.loss_components")


def test_h1_and_h2_are_distinguishable_and_not_one_blanket_refusal() -> None:
    """The two firing arms must name DIFFERENT problems.

    A comparator that refuses any zero would fire on both with one reason, and
    would then also be refusing the control. Naming the leg is what makes the
    verdict actionable: "unweighted" sends the operator to the config, "inert"
    sends them to the data.
    """
    zero_weight = _detail(
        _sweep(
            _context(
                (
                    _PREFERENCE_OK,
                    LossComponent(name="sft_loss", weight=0.0, observed=True, contribution=0.0),
                ),
                ("preference_loss", "sft_loss"),
            )
        ),
        "objective.loss_components",
    )
    inert = _detail(
        _sweep(
            _context(
                (
                    _PREFERENCE_OK,
                    LossComponent(name="sft_loss", weight=1.0, observed=True, contribution=0.0),
                ),
                ("preference_loss", "sft_loss"),
            )
        ),
        "objective.loss_components",
    )
    assert zero_weight != inert


def test_h3_the_same_reading_is_invisible_when_the_term_is_not_declared() -> None:
    """The conditional, and the reason section 7 item 4 does not have a yes/no answer.

    Same run, same 0.0000, but the objective declares only the preference term
    and `sft_loss` is a log line. Every gate is green. Nothing here is broken --
    the gate is answering the question it was asked, over the components the
    objective declared -- but it means the Phase 2 reading is caught only if the
    FoundationScale DPO implementation declares `sft_loss` even when the term is
    inactive. That is a requirement on stage 2, recorded in PHASE3_DESIGN.
    """
    report = _sweep(_context((_PREFERENCE_OK,), ("preference_loss",)))
    assert report.ok is True  # type: ignore[attr-defined]
    assert _verdict(report, "objective.loss_components") == "PASS"


def test_the_diagnostic_metric_channel_exists_end_to_end() -> None:
    """#316 CLOSED on the EXISTENCE axis: the channel is present in both layers.

    This test previously asserted the OPPOSITE -- that no field of
    `ObjectiveGateContext` could carry a diagnostic metric -- and it was written
    to go red on exactly this change so the re-statement would be deliberate
    rather than a silent relaxation. It is re-stated here as the positive claim.

    `LossComponent` still REQUIRES a `weight`, and that is the point: a
    diagnostic metric has no weight and is not a term of the loss, so it gets its
    own pair of types rather than being routed through the loss channel, which
    would make the plane assert something false in order to look at it.

    Asserted over the live field sets, so removing the channel again turns this
    red instead of quietly restoring the blindness.
    """
    from foundationscale.rl.interfaces import LossOutput, build_objective_gate_context

    ctx_fields = {f.name for f in dataclasses.fields(ObjectiveGateContext)}
    assert {"declared_metrics", "metrics"} <= ctx_fields
    assert {f.name for f in dataclasses.fields(LossComponent)} == {
        "name",
        "weight",
        "observed",
        "contribution",
    }
    assert {f.name for f in dataclasses.fields(MetricExpectation)} == {
        "name",
        "low",
        "high",
        "degenerate",
    }
    assert {f.name for f in dataclasses.fields(MetricObservation)} == {"name", "value"}
    # The seam, not just the context: a channel the stage-1 bridge cannot fill is
    # a field the production path never populates -- #316's own failure shape.
    assert {f.name for f in dataclasses.fields(LossOutput)} == {"loss", "components", "metrics"}
    assert "declared_metrics" in set(inspect.signature(build_objective_gate_context).parameters)


def test_the_phase_two_accuracy_reading_now_fires() -> None:
    """The `accuracy` half, measured rather than declared unmeasurable.

    DPO `accuracy` at exactly 0.0000 for ten steps says the policy ranked the
    REJECTED response above the chosen one on every pair -- the more alarming of
    Phase 2's two readings, and the one no gate could see before #316.

    The firing is conditional in the same way H1/H2 are, and the condition is
    named rather than hidden: `0.0` is INSIDE the natural [0, 1] range of an
    accuracy, so bounds alone cannot refuse it. What refuses it is the objective
    declaring 0.0 pathological for THIS metric. The plane cannot infer that --
    accuracy pinned at 0.0 is broken, a truncation fraction at 0.0 is ideal --
    so `degenerate` is the algorithm's claim, not the gate's guess.
    """
    accuracy = MetricExpectation(name="accuracy", low=0.0, high=1.0, degenerate=(0.0,))
    report = _sweep(
        _context(
            (_PREFERENCE_OK,),
            ("preference_loss",),
            declared_metrics=(accuracy,),
            metrics=(MetricObservation(name="accuracy", value=0.0),),
        )
    )
    assert report.ok is False  # type: ignore[attr-defined]
    assert _verdict(report, "objective.metrics") == "FAIL"
    assert "accuracy" in _detail(report, "objective.metrics")
    # The loss-side gates stay green on this arm: the two halves of the Phase 2
    # anomaly are caught by different gates, and collapsing them into one refusal
    # would make the verdict unactionable.
    assert _verdict(report, "objective.loss_components") == "PASS"


def test_the_metric_axis_has_no_h3_hole_but_the_degenerate_claim_is_load_bearing() -> None:
    """Two conditionals, measured, because only one of them is closed.

    H3 on the loss axis: an UNDECLARED `sft_loss` at 0.0 passes every gate,
    because declaration is what puts a term in the denominator. The metric axis
    does NOT inherit that hole -- an observed metric the objective never declared
    is itself a refusal, so `accuracy` cannot slip through by going undeclared.

    What it DOES inherit is the weaker conditional above: declared with a range
    but with no `degenerate` reading named, 0.0 is in-range and passes. That is
    the residual, and it is a requirement on stage 2 rather than a defect here.
    """
    observed_only = _sweep(
        _context(
            (_PREFERENCE_OK,),
            ("preference_loss",),
            declared_metrics=(),
            metrics=(MetricObservation(name="accuracy", value=0.0),),
        )
    )
    assert _verdict(observed_only, "objective.metrics") == "FAIL"
    assert "never declared" in _detail(observed_only, "objective.metrics")

    no_degenerate = _sweep(
        _context(
            (_PREFERENCE_OK,),
            ("preference_loss",),
            declared_metrics=(MetricExpectation(name="accuracy", low=0.0, high=1.0),),
            metrics=(MetricObservation(name="accuracy", value=0.0),),
        )
    )
    assert _verdict(no_degenerate, "objective.metrics") == "PASS"
