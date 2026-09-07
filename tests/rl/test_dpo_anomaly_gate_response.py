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
* `accuracy` at 0.0000 is caught by NOTHING, and cannot be: no field of
  `ObjectiveGateContext` can carry a diagnostic metric. See the last test.

Arm CONTROL is the positive control (#239). Without a green arm the two firing
arms are indistinguishable from a comparator that refuses everything.
"""

from __future__ import annotations

import dataclasses
import inspect

from foundationscale.gates.core import REGISTRY, Lifecycle, run_event
from foundationscale.gates.objective_gates import (
    LossComponent,
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
    components: tuple[LossComponent, ...], declared: tuple[str, ...]
) -> ObjectiveGateContext:
    return ObjectiveGateContext(
        objective=_OBJECTIVE,
        declared_components=declared,
        components=components,
        uses_rewards=False,
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


def test_a_diagnostic_metric_has_no_channel_into_the_gate_plane() -> None:
    """The `accuracy` half: structurally unmeasurable, not merely unmeasured.

    DPO `accuracy` at exactly 0.0000 for ten steps says the policy ranked the
    REJECTED response above the chosen one on every pair -- the more alarming of
    Phase 2's two readings. No gate can see it, and the reason is the shape of
    the context rather than the content of any gate: of eleven fields, the only
    named-scalar channel is `components`, and `LossComponent` REQUIRES a
    `weight`. A diagnostic metric has no weight and is not a term of the loss,
    so routing `accuracy` through it would make the plane assert something false
    in order to look at it.

    Asserted over the live field sets rather than a copied list, so adding the
    missing channel turns this test red and it has to be re-stated deliberately.
    """
    from foundationscale.rl.interfaces import LossOutput, build_objective_gate_context

    ctx_fields = {f.name for f in dataclasses.fields(ObjectiveGateContext)}
    assert "metrics" not in ctx_fields
    assert {f.name for f in dataclasses.fields(LossComponent)} == {
        "name",
        "weight",
        "observed",
        "contribution",
    }
    # The bridge stage 1 shipped: neither the loss it reads nor the parameters it
    # accepts carry a metric, so the absence is in the seam, not just the context.
    assert {f.name for f in dataclasses.fields(LossOutput)} == {"loss", "components"}
    assert "metrics" not in set(inspect.signature(build_objective_gate_context).parameters)
