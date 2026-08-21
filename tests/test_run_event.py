"""Typed-context dispatch: ``run_event`` and ``Gate.context_type``.

Every test here fails against the pre-dispatch tree: either ``run_event`` does not
exist (ImportError at collection), or its contemporary equivalent —
``GateRegistry.run`` with one shared context handed to gates of different families —
dies inside a gate's ``check`` with a raw ``TypeError`` one frame down instead of
returning the named, blocking ERROR these tests assert. Each test docstring names
the single assertion that cannot hold today.
"""

from __future__ import annotations

import json

import pytest

from foundationscale.gates import checkpoint_gates as ckg
from foundationscale.gates import fixtures as fx
from foundationscale.gates.core import (
    Coverage,
    Gate,
    GateRegistry,
    Lifecycle,
    Verdict,
    run_event,
    verify_controls,
)
from foundationscale.verify import parity as pv


class _Alpha:
    """Probe context family A."""

    def __init__(self, marker: object | None = None) -> None:
        self.marker = marker if marker is not None else object()


class _AlphaSub(_Alpha):
    """A subclass of family A, for subtype-dispatch tests."""


class _AlphaGrandchild(_AlphaSub):
    """A second line under family A, for ambiguity tests."""


class _Beta:
    """Probe context family B."""


class _ProbeGate(Gate):
    """Records every context it is handed; the tests dispatch it by family type."""

    id = "probe.base"
    description = "records the context it received"
    events = (Lifecycle.SAVE,)
    context_type = None

    def __init__(self) -> None:
        self.received: list[object] = []

    def check(self, ctx):
        self.received.append(ctx)
        return self.ok("probe received its context", Coverage(1, "probes"))

    def controls(self):
        return []  # probes are dispatched, never CI-verified


class _WantsAlpha(_ProbeGate):
    id = "probe.wants_alpha"
    context_type = _Alpha


class _WantsBeta(_ProbeGate):
    id = "probe.wants_beta"
    context_type = _Beta


class _WantsBetaToo(_ProbeGate):
    id = "probe.wants_beta_too"
    context_type = _Beta


class _LegacyProbe(_ProbeGate):
    """Declares no context_type: legacy broadcast behaviour must be preserved."""

    id = "probe.legacy"


def test_run_event_dispatches_each_gate_its_own_context() -> None:
    """Fails today: ``run_event`` does not exist, and ``registry.run`` with either
    single context would hand it to BOTH gates (one family TypeErrors) — the
    identity assertions on ``received`` could never both hold."""
    alpha_gate, beta_gate = _WantsAlpha(), _WantsBeta()
    reg = GateRegistry()
    reg.register(alpha_gate)
    reg.register(beta_gate)
    alpha, beta = _Alpha(), _Beta()

    report = run_event(reg, Lifecycle.SAVE, {_Alpha: alpha, _Beta: beta})

    assert report.ok
    assert len(report.results) == 2
    assert alpha_gate.received == [alpha]
    assert beta_gate.received == [beta]


def test_missing_context_is_named_blocking_error_not_traceback() -> None:
    """Fails today: no path produces this verdict — the equivalent miswiring used
    to surface, if at all, as an ERROR carrying a raw TypeError/AttributeError."""
    reg = GateRegistry()
    reg.register(_WantsBeta())

    report = run_event(reg, Lifecycle.SAVE, {_Alpha: _Alpha()})

    assert not report.ok
    (result,) = report.results
    assert result.verdict is Verdict.ERROR
    assert result.blocking
    assert result.coverage.checked == 0
    assert result.coverage.unit == "_Beta"
    assert result.detail == (
        "no context of type _Beta supplied for gate probe.wants_beta — unwired, not healthy"
    )
    assert "traceback" not in result.evidence  # wiring is a finding, not a crash


def test_missing_ctx_report_skip_never_reads_as_pass() -> None:
    """Fails today: the mode did not exist; the invariant is that the declared
    abstention is a surfaced SKIP — which never counts as evidence of health."""
    reg = GateRegistry()
    reg.register(_WantsBeta())

    report = run_event(reg, Lifecycle.SAVE, {_Alpha: _Alpha()}, missing_ctx="report-skip")

    (result,) = report.results
    assert result.verdict is Verdict.SKIP
    assert result.verdict is not Verdict.PASS
    assert not result.blocking
    assert report.ok
    assert "established nothing" in result.detail


def test_unknown_missing_ctx_mode_raises() -> None:
    """Fails today: no such parameter — a misspelled abstention must not be
    silently coerced into the blocking default (or worse, into quiet)."""
    reg = GateRegistry()
    with pytest.raises(ValueError, match="missing_ctx"):
        run_event(reg, Lifecycle.SAVE, {}, missing_ctx="ignore")


def test_untyped_gate_refuses_ambiguous_typed_broadcast() -> None:
    """Fails today: a legacy gate in a multi-context sweep must stop instead of
    running against whichever context the caller happened to hold."""
    legacy = _LegacyProbe()
    reg = GateRegistry()
    reg.register(legacy)

    report = run_event(reg, Lifecycle.SAVE, {_Alpha: _Alpha()})

    (result,) = report.results
    assert result.verdict is Verdict.ERROR
    assert "declares no context_type" in result.detail
    assert legacy.received == []  # it must not run — guessing is the defect


def test_single_bare_context_still_broadcasts_to_untyped_gate() -> None:
    """Pins the legacy contract: a bare context broadcast through run_event is
    exactly GateRegistry.run semantics, so existing trainers keep working."""
    legacy = _LegacyProbe()
    reg = GateRegistry()
    reg.register(legacy)
    bare = _Alpha()

    report = run_event(reg, Lifecycle.SAVE, bare)

    assert report.ok
    assert legacy.received == [bare]


def test_bare_context_dispatches_by_isinstance_and_names_the_rest() -> None:
    """Fails today: the unmatched typed gate must get the named ERROR, not a
    TypeError from inside its own check."""
    alpha_gate, beta_gate = _WantsAlpha(), _WantsBeta()
    reg = GateRegistry()
    reg.register(alpha_gate)
    reg.register(beta_gate)

    report = run_event(reg, Lifecycle.SAVE, _Beta())

    assert not report.ok
    by_id = {r.gate_id: r for r in report.results}
    assert by_id["probe.wants_beta"].verdict is Verdict.PASS
    assert by_id["probe.wants_alpha"].verdict is Verdict.ERROR
    assert "no context of type _Alpha supplied" in by_id["probe.wants_alpha"].detail
    assert beta_gate.received and alpha_gate.received == []


def test_unique_subtype_context_match_wins() -> None:
    """A single subclass context satisfies the declared base type."""
    gate = _WantsAlpha()
    reg = GateRegistry()
    reg.register(gate)
    sub = _AlphaSub()

    report = run_event(reg, Lifecycle.SAVE, {_AlphaSub: sub})

    assert report.ok
    assert gate.received == [sub]


def test_ambiguous_subtype_contexts_block() -> None:
    """Two contexts both satisfying the declared type: dispatch must not pick."""
    gate = _WantsAlpha()
    reg = GateRegistry()
    reg.register(gate)

    report = run_event(
        reg, Lifecycle.SAVE, {_AlphaSub: _AlphaSub(), _AlphaGrandchild: _AlphaGrandchild()}
    )

    (result,) = report.results
    assert result.verdict is Verdict.ERROR
    assert "ambiguous" in result.detail
    assert "_AlphaGrandchild" in result.detail
    assert gate.received == []


def test_lazy_factory_contexts_built_per_gate_only_when_consumed() -> None:
    """Fails today: no lazy form existed; the invariant is per-gate materialisation
    and no evaluation of contexts no gate asked for."""
    calls: list[str] = []
    alpha = _Alpha()

    def make_alpha() -> _Alpha:
        calls.append("alpha")
        return alpha

    def make_unused() -> _Beta:
        calls.append("unused")
        raise AssertionError("a context no gate consumes must never be built")

    gate = _WantsAlpha()
    reg = GateRegistry()
    reg.register(gate)

    report = run_event(reg, Lifecycle.SAVE, {_Alpha: make_alpha, _Beta: make_unused})

    assert report.ok
    assert gate.received == [alpha]
    assert calls == ["alpha"]


def test_factory_failure_is_error_verdict_not_an_escape() -> None:
    """A raising context factory is gate-author code: it converts, with traceback."""

    def boom() -> _Alpha:
        raise RuntimeError("factory exploded")

    reg = GateRegistry()
    reg.register(_WantsAlpha())

    report = run_event(reg, Lifecycle.SAVE, {_Alpha: boom})

    (result,) = report.results
    assert result.verdict is Verdict.ERROR
    assert "RuntimeError" in result.detail
    assert "factory exploded" in result.detail
    assert "traceback" in result.evidence


def test_selected_but_unregistered_gate_id_is_missing_not_dropped() -> None:
    """A typo'd selection id must land in ``missing`` — dropping it reads as
    'all selected gates clear' over fewer gates than were asked for."""
    reg = GateRegistry()
    reg.register(_WantsAlpha())

    report = run_event(
        reg,
        Lifecycle.SAVE,
        {_Alpha: _Alpha()},
        gate_ids=("probe.wants_alpha", "probe.ghost"),
    )

    assert report.missing == ("probe.ghost",)
    assert not report.ok
    assert len(report.results) == 1


def test_required_and_excluded_gate_blocks_as_missing() -> None:
    """Contradictory instructions fail closed: required ∩ exclude is missing."""
    reg = GateRegistry()
    reg.register(_WantsAlpha())
    reg.register(_WantsBeta())

    report = run_event(
        reg,
        Lifecycle.SAVE,
        {_Alpha: _Alpha(), _Beta: _Beta()},
        required=("probe.wants_beta",),
        exclude=("probe.wants_beta",),
    )

    assert report.missing == ("probe.wants_beta",)
    assert not report.ok


def test_empty_selection_synthesizes_vacuous_sweep_with_denominator() -> None:
    """The sweep-level all([]): zero gates ran, so the report is blocking VACUOUS
    — and the footer must say 0 ran, not count the marker as a run gate."""
    reg = GateRegistry()
    reg.register(_WantsAlpha())

    report = run_event(reg, Lifecycle.SAVE, {_Alpha: _Alpha()}, gate_ids=("probe.ghost",))

    assert not report.ok
    assert report.missing == ("probe.ghost",)
    (marker,) = report.results
    assert marker.gate_id == "registry.empty_sweep.save"
    assert marker.verdict is Verdict.VACUOUS
    assert report.is_vacuous
    assert "0 gates ran of 1 registered for save" in report.render()


def test_report_footer_and_json_carry_the_sweep_denominator() -> None:
    """Fails today: neither the footer line nor the ``registered`` key existed."""
    reg = GateRegistry()
    reg.register(_WantsAlpha())
    reg.register(_WantsBeta())
    reg.register(_WantsBetaToo())

    report = run_event(
        reg,
        Lifecycle.SAVE,
        {_Alpha: _Alpha(), _Beta: _Beta()},
        gate_ids=("probe.wants_alpha", "probe.wants_beta"),
    )

    assert "2 gates ran of 3 registered for save" in report.render()
    assert json.loads(report.to_json())["registered"] == 3


def test_registry_run_reports_the_registered_denominator_too() -> None:
    """Fails today: no footer existed; broadcast sweeps carry the same doctrine-2
    denominator as dispatched ones."""
    reg = GateRegistry()
    reg.register(_LegacyProbe())

    report = reg.run(Lifecycle.SAVE, object())

    assert "1 gates ran of 1 registered for save" in report.render()


def test_context_type_must_be_a_type_or_none() -> None:
    """Fails today: no validation existed — a nonsense declaration passed silently
    until a sweep tried to dispatch on it."""
    with pytest.raises(TypeError, match="context_type"):

        class _Bad(Gate):
            id = "probe.bad"
            description = "misdeclared context_type"
            events = (Lifecycle.SAVE,)
            context_type = "not-a-type"

            def check(self, ctx):
                return self.ok("never runs", Coverage(1, "probes"))

            def controls(self):
                return []


# ---------------------------------------------------------------------------
# The shipped gates: real fixtures, real code paths (no paraphrase)
# ---------------------------------------------------------------------------


def _torch_or_skip() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")


def _shipped_save_registry() -> GateRegistry:
    reg = GateRegistry()
    reg.register(ckg.ExpertDistinctnessGate())
    reg.register(ckg.ExpertByteVolumeGate())
    reg.register(ckg.SaveCompletenessGate())
    reg.register(pv.WeightParityGate())
    return reg


def test_shipped_families_share_one_save_sweep() -> None:
    """Fails today: these four gates could never run in one event without one of
    them TypeErroring on the other's context."""
    _torch_or_skip()
    reg = _shipped_save_registry()
    # A per-expert (sharded) fixture, not the fused one: a fused/stacked layout
    # now abstains on distinctness because per-expert storage identity is not
    # observable inside a stacked tensor, and this test is about dispatch, not
    # about what the gates can see.
    ckpt_ctx = ckg.ExpertDistinctnessGate().coerce_context(fx.healthy_sharded_moe_ctx())
    assert isinstance(ckpt_ctx, ckg.CheckpointGateContext)

    report = run_event(
        reg,
        Lifecycle.SAVE,
        {
            ckg.CheckpointGateContext: ckpt_ctx,
            pv.ParityGateContext: pv._make_identical_sources_ctx(),
        },
    )

    assert report.ok, [r.render() for r in report.results]
    assert {r.gate_id: r.verdict for r in report.results} == {
        "checkpoint.expert_distinctness": Verdict.PASS,
        "checkpoint.expert_bytes": Verdict.PASS,
        "checkpoint.save_complete": Verdict.PASS,
        "checkpoint.weight_parity": Verdict.PASS,
    }
    assert "4 gates ran of 4 registered for save" in report.render()


def test_first_save_event_runs_composite_and_parity_with_string_event() -> None:
    """The composite first-save gate and parity share FIRST_SAVE; string events
    are accepted."""
    _torch_or_skip()
    reg = GateRegistry()
    reg.register(ckg.FirstSaveGate())
    reg.register(pv.WeightParityGate())

    report = run_event(
        reg,
        "first_save",
        {
            # Sharded, not fused: a stacked composite is legitimately
            # UNDERCOVERED, and this test asks whether the two gates dispatch.
            ckg.CheckpointGateContext: ckg.FirstSaveGate().coerce_context(
                fx.healthy_sharded_moe_ctx()
            ),
            pv.ParityGateContext: pv._make_identical_sources_ctx(),
        },
    )

    assert report.ok, [r.render() for r in report.results]
    assert {r.gate_id for r in report.results} == {
        "checkpoint.first_save",
        "checkpoint.weight_parity",
    }


def test_shipped_gates_declare_their_context_type() -> None:
    """Fails today: the attribute did not exist at all (AttributeError)."""
    for gate_cls, expected in (
        (ckg.ExpertDistinctnessGate, ckg.CheckpointGateContext),
        (ckg.ExpertByteVolumeGate, ckg.CheckpointGateContext),
        (ckg.SaveCompletenessGate, ckg.CheckpointGateContext),
        (ckg.FirstSaveGate, ckg.CheckpointGateContext),
        (pv.WeightParityGate, pv.ParityGateContext),
    ):
        assert gate_cls.context_type is expected


def test_shipped_gates_controls_still_hold_after_dispatch_wiring() -> None:
    """Dispatch wiring must not rot the gates' own controls; this reruns them
    against the declared-type code path (the fixtures ARE the declared types)."""
    _torch_or_skip()
    assert verify_controls(_shipped_save_registry()) == []


def test_integrate_reexports_run_event() -> None:
    """Fails today: the module did not exist (ModuleNotFoundError)."""
    from foundationscale import integrate

    assert integrate.run_event is run_event
