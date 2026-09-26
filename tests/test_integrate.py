"""The integrate facade: gate-context dispatch at package root.

``foundationscale.integrate`` is the import surface a training composition
actually meets: one ``run_event`` call over a typed context map instead of the
broadcast ``GateRegistry.run``, whose one-object-to-every-gate shape degrades
into frame-deep TypeErrors. These tests pin the two halves of that promise:
the names re-exported here ARE the machinery in ``foundationscale.gates.core``
(not a second, drifting implementation), and the dispatch rules that doctrine
depends on -- unwired blocks, empty sweeps block -- fire through this facade.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from foundationscale import integrate
from foundationscale.gates import core


def test_facade_reexports_the_single_implementation() -> None:
    """Identity, not equivalence: a facade copy would drift silently."""
    assert integrate.run_event is core.run_event
    assert integrate.REGISTRY is core.REGISTRY
    assert integrate.GateRegistry is core.GateRegistry
    assert integrate.GateReport is core.GateReport
    assert integrate.GateBlocked is core.GateBlocked
    assert integrate.Lifecycle is core.Lifecycle


class _Ctx:
    def __init__(self, marker: str) -> None:
        self.marker = marker


class _OtherCtx:
    pass


class _MarkerGate(core.Gate):
    """Consumes _Ctx; vacuous when the marker is absent, PASS when present."""

    id: ClassVar[str] = "test.marker_seen"
    description: ClassVar[str] = "dispatch feeds the declared context type"
    events: ClassVar[tuple[core.Lifecycle, ...]] = (core.Lifecycle.SAVE,)
    context_type: ClassVar[type | None] = _Ctx

    def controls(self) -> list[core.Control]:
        return []

    def check(self, ctx: Any) -> core.GateResult:
        if getattr(ctx, "marker", None) == "present":
            return core.GateResult(
                gate_id=self.id,
                verdict=core.Verdict.PASS,
                coverage=core.Coverage(checked=1, unit="markers"),
                detail="marker seen",
            )
        return core.GateResult(
            gate_id=self.id,
            verdict=core.Verdict.VACUOUS,
            coverage=core.Coverage.none("unit"),
            detail="no marker",
        )


def _registry() -> core.GateRegistry:
    """A local registry: identity tests above cover the shared REGISTRY."""
    return core.GateRegistry()


def test_dispatch_feeds_the_declared_context_type() -> None:
    reg = _registry()
    reg.register(_MarkerGate())

    report = integrate.run_event(
        reg, core.Lifecycle.SAVE, {_Ctx: _Ctx("present"), _OtherCtx: _OtherCtx()}
    )

    parsed = {r.gate_id: r for r in report.results}
    assert parsed["test.marker_seen"].verdict is core.Verdict.PASS
    report.raise_if_blocking()


def test_missing_context_blocks_never_guesses() -> None:
    reg = _registry()
    reg.register(_MarkerGate())

    report = integrate.run_event(reg, "save", {_OtherCtx: _OtherCtx()})

    (result,) = report.results
    assert result.verdict is core.Verdict.ERROR
    assert "unwired, not healthy" in result.detail
    with pytest.raises(core.GateBlocked):
        report.raise_if_blocking()


def test_missing_context_report_skip_must_be_asked_for() -> None:
    reg = _registry()
    reg.register(_MarkerGate())

    report = integrate.run_event(reg, "save", {_OtherCtx: _OtherCtx()}, missing_ctx="report-skip")

    (result,) = report.results
    assert result.verdict is core.Verdict.SKIP


def test_misspelled_abstention_mode_raises() -> None:
    reg = _registry()
    reg.register(_MarkerGate())

    with pytest.raises(ValueError, match="missing_ctx"):
        integrate.run_event(reg, "save", {}, missing_ctx="sikp")


def test_empty_sweep_blocks_like_every_other_altitude() -> None:
    """all([]) over zero gates is not a clear, at any altitude."""
    report = integrate.run_event(_registry(), "save", {})

    (result,) = report.results
    assert result.verdict is core.Verdict.VACUOUS
    with pytest.raises(core.GateBlocked):
        report.raise_if_blocking()
