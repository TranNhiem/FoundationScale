"""Report-layer vacuity: an empty report must block on the type, not only on the runner.

The registry runner already synthesizes a blocking marker for an empty sweep; these
tests close the hole one level up, where a consumer that hand-builds, filters, or
merges a :class:`GateReport` could otherwise reduce it to zero results and still get
``ok`` — ``all([])`` restated at the report layer.

Companion tests pin :meth:`GateReport.render` so the head and footer can never again
disagree about how many gates ran.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from foundationscale.gates.core import (
    _EMPTY_SWEEP_GATE_PREFIX,
    Control,
    ControlKind,
    Coverage,
    Gate,
    GateBlocked,
    GateRegistry,
    GateReport,
    GateResult,
    Lifecycle,
    Verdict,
    run_event,
)


class _PresenceGate(Gate):
    """A gate whose only failure mode is vacuity, for building real reports."""

    id = "test.report_vacuity.presence"
    description = "Passes over a non-empty unit list"
    events = (Lifecycle.PROMOTE,)

    def check(self, ctx: Any) -> GateResult:
        units = list(ctx)
        # WHY bare ok() over possibly-zero coverage: it exercises the framework's
        # VACUOUS downgrade, and that downgrade is this gate's MUST_FIRE fixture.
        return self.ok("all units accounted for", Coverage(len(units), "units"))

    def controls(self) -> list[Control]:
        return [
            Control(
                "empty-unit-list",
                ControlKind.MUST_FIRE,
                list,
                note="zero units must downgrade ok() to blocking VACUOUS",
            )
        ]


def test_handbuilt_empty_report_is_not_ok_and_raises() -> None:
    # WHY: GateReport's own docstring blesses hand-built reports, so the type
    # itself must refuse to read one over zero results as success.
    rep = GateReport(event=Lifecycle.PROMOTE, results=(), missing=())
    assert rep.is_vacuous
    assert rep.blocking == ()
    assert not rep.ok
    with pytest.raises(GateBlocked):
        rep.raise_if_blocking()


def test_handbuilt_report_of_only_empty_sweep_marker_is_not_ok() -> None:
    # WHY: a merge that transplants the framework's marker into a fresh report
    # must still block; the marker is a verdict, not an all-clear token.
    marker = GateResult(
        gate_id=f"{_EMPTY_SWEEP_GATE_PREFIX}{Lifecycle.PROMOTE.value}",
        verdict=Verdict.VACUOUS,
        coverage=Coverage.none("gates"),
        detail="no gates ran for promote; transplanted by a report merge",
    )
    rep = GateReport(event=Lifecycle.PROMOTE, results=(marker,), missing=())
    assert rep.is_vacuous
    assert not rep.ok
    with pytest.raises(GateBlocked):
        rep.raise_if_blocking()


def test_event_allow_empty_opt_out_stays_nonblocking() -> None:
    # WHY: declared gateless is the one legitimate extension point; moving the
    # vacuity check onto the type must lift the block for it alone, which means
    # the declaration has to travel on the report.
    reg = GateRegistry(event_allow_empty=(Lifecycle.SAVE,))
    rep = reg.run(Lifecycle.SAVE, ctx=object())
    assert rep.is_vacuous
    assert rep.allow_empty
    assert rep.ok
    rep.raise_if_blocking()


def test_undeclared_empty_registry_sweep_still_blocks() -> None:
    # WHY: the runner's own floor must not regress while the type learns it.
    reg = GateRegistry()
    rep = reg.run(Lifecycle.SAVE, ctx=object())
    assert rep.is_vacuous
    assert not rep.allow_empty
    assert not rep.ok
    with pytest.raises(GateBlocked):
        rep.raise_if_blocking()


def test_run_event_propagates_the_allow_empty_declaration() -> None:
    # WHY: both sweep entry points must stamp the report, or run_event callers
    # would get fail-closed behaviour on events the registry declared gateless.
    reg = GateRegistry(event_allow_empty=(Lifecycle.EXPORT,))
    rep = run_event(reg, Lifecycle.EXPORT, {})
    assert rep.is_vacuous
    assert rep.allow_empty
    assert rep.ok


def test_populated_passing_report_is_still_ok() -> None:
    # WHY negative control: this fix could "work" by making everything block;
    # a real sweep over real units with no defect found must still pass.
    reg = GateRegistry()
    reg.register(_PresenceGate())
    rep = reg.run(Lifecycle.PROMOTE, ctx=["layer-0", "layer-1"])
    assert not rep.is_vacuous
    assert rep.ok
    rep.raise_if_blocking()


def test_report_filtered_down_to_empty_blocks() -> None:
    # WHY: the finding verbatim — re-wrapping a report after filtering its
    # results reopened the hole; the type must close it regardless of provenance.
    reg = GateRegistry()
    reg.register(_PresenceGate())
    rep = reg.run(Lifecycle.PROMOTE, ctx=["layer-0"])
    filtered = GateReport(
        event=rep.event,
        results=tuple(r for r in rep.results if r.gate_id == "no.such.gate"),
        missing=rep.missing,
        registered=rep.registered,
        allow_empty=rep.allow_empty,
    )
    assert filtered.is_vacuous
    assert not filtered.ok
    with pytest.raises(GateBlocked):
        filtered.raise_if_blocking()


def test_to_json_records_allow_empty() -> None:
    # WHY: to_json is how the verdict leaves the process; the new state must be
    # on the wire or every downstream consumer rebuilds the old fail-open type.
    reg = GateRegistry(event_allow_empty=(Lifecycle.SAVE,))
    rep = reg.run(Lifecycle.SAVE, ctx=object())
    payload = json.loads(rep.to_json())
    assert payload["allow_empty"] is True
    assert payload["ok"] is True
    assert payload["results"] == []


def test_render_head_does_not_count_the_empty_sweep_marker() -> None:
    # WHY (finding 2): the footer already excluded the marker; the head counted
    # it, rendering "1 run" directly above "0 gates ran of 0 registered".
    reg = GateRegistry()
    rep = reg.run(Lifecycle.SAVE, ctx=object())
    lines = rep.render().splitlines()
    assert lines[0].startswith("gates @ save: 0 run")
    assert "1 blocking" in lines[0]
    assert "0 gates ran of 0 registered" in lines[-1]


def test_render_head_counts_real_gates_only() -> None:
    # WHY: head and footer must agree when gates genuinely ran, too — the hoisted
    # count is shared, and this pins the non-empty case.
    reg = GateRegistry()
    reg.register(_PresenceGate())
    rep = reg.run(Lifecycle.PROMOTE, ctx=["layer-0"])
    lines = rep.render().splitlines()
    assert lines[0].startswith("gates @ promote: 1 run")
    assert "1 gates ran of 1 registered" in lines[-1]


def test_render_names_vacuity_for_a_handbuilt_empty_report() -> None:
    # WHY: a blocking report with an empty blocking tuple would render a bare
    # dash or read as a near-miss; the head has to name that nothing ran.
    rep = GateReport(event=Lifecycle.PROMOTE, results=(), missing=())
    head = rep.render().splitlines()[0]
    assert head.startswith("gates @ promote: 0 run")
    assert "VACUOUS" in head


def test_render_declared_empty_sweep_is_labelled_not_barely_all_clear() -> None:
    # WHY: "all clear" over zero gates is a claim broader than its evidence even
    # when gateless was declared; the render must carry the declaration.
    reg = GateRegistry(event_allow_empty=(Lifecycle.SAVE,))
    head = reg.run(Lifecycle.SAVE, ctx=object()).render().splitlines()[0]
    assert "gateless by declaration" in head


class _AbstainingGate(Gate):
    """The all-SKIP sweep's building block: declares itself not applicable, always.

    SKIP is a first-class per-gate outcome with a reason; the report-layer
    question is what the AGGREGATE may claim when every gate in it abstained.
    Never verified here — report-level tests only, never registered globally.
    """

    id = "test.report_vacuity.abstains"
    description = "Skips with a reason on every input"
    events = (Lifecycle.PROMOTE,)

    def check(self, ctx: Any) -> GateResult:
        return self.skip("declared not applicable in this environment")

    def controls(self) -> list[Control]:
        return []


def test_all_skip_report_is_not_ok_and_blocks() -> None:
    # WHY: the finding's trace verbatim — SKIP does not block, nothing is
    # missing, and the ids are real gate ids, so is_vacuous (prefix-based,
    # correctly) cannot see it. Zero verified units must not read as success.
    abstaining = _AbstainingGate()
    results = (
        abstaining.skip("no CUDA on this runner"),
        abstaining.skip("no CUDA on this runner"),
    )
    rep = GateReport(event=Lifecycle.PROMOTE, results=results, missing=())
    assert not rep.is_vacuous
    assert rep.is_unverified
    assert not rep.ok
    with pytest.raises(GateBlocked):
        rep.raise_if_blocking()


def test_mixed_pass_and_skip_report_is_still_ok_and_says_so() -> None:
    # WHY negative control: a healthy partial sweep must keep passing — the fix
    # over-rotating into blocking mixed reports would be a regression of the
    # same size as the defect. AND doctrine 5: the headline must state how much
    # of it actually verified.
    reg = GateRegistry()
    reg.register(_PresenceGate())
    reg.register(_AbstainingGate())
    rep = reg.run(Lifecycle.PROMOTE, ctx=["layer-0"])
    assert rep.ok
    assert not rep.is_unverified
    head = rep.render().splitlines()[0]
    assert head.startswith("gates @ promote: 2 run")
    assert "1 of 2 verified; 1 declared SKIP" in head


def test_allow_empty_declaration_does_not_pardon_an_all_skip_sweep() -> None:
    # WHY: event_allow_empty declares "zero gates for this event is deliberate".
    # A gate that ran and abstained is not the absence of a gate; pardoning it
    # with the same flag would widen the declaration beyond its evidence.
    reg = GateRegistry(event_allow_empty=(Lifecycle.PROMOTE,))
    reg.register(_AbstainingGate())
    rep = reg.run(Lifecycle.PROMOTE, ctx=object())
    assert rep.allow_empty
    assert rep.is_unverified
    assert not rep.ok
    with pytest.raises(GateBlocked):
        rep.raise_if_blocking()


def test_all_skip_head_states_zero_verified_of_n_not_all_clear() -> None:
    # WHY: "1 run — all clear" over one abstention is the doctrine-5 defect in
    # the verdict-bearing position of the rendered output.
    reg = GateRegistry()
    reg.register(_AbstainingGate())
    rep = reg.run(Lifecycle.PROMOTE, ctx=object())
    head = rep.render().splitlines()[0]
    assert "all clear" not in head
    assert "0 of 1 verified" in head


def test_to_json_records_unverified_for_abstention_sweeps() -> None:
    # WHY: the verdict leaves the process as JSON; the new state must be on the
    # wire or downstream consumers rebuild the fail-open reading from it.
    reg = GateRegistry()
    reg.register(_AbstainingGate())
    rep = reg.run(Lifecycle.PROMOTE, ctx=object())
    payload = json.loads(rep.to_json())
    assert payload["unverified"] is True
    assert payload["ok"] is False


def test_empty_report_is_vacuous_not_unverified_polarity_preserved() -> None:
    # WHY: is_vacuous's all([])-is-True polarity is load-bearing; is_unverified
    # must NOT also claim the empty report, or declared-gateless (allow_empty)
    # sweeps would suddenly block. Two properties, two shapes, disjoint.
    rep = GateReport(event=Lifecycle.PROMOTE, results=(), missing=())
    assert rep.is_vacuous
    assert not rep.is_unverified
