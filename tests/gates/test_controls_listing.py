"""The controls runner's listing must print each gate's dispatch contract.

``verify_controls`` is the framework's own MUST_FIRE/MUST_PASS harness, and the
listing :func:`~foundationscale.gates.controls.main` prints is the one surface
where a human audits the whole gate population at once. A gate that declares no
``context_type`` falls back to untyped broadcast; if the listing printed
nothing for it, a forgotten declaration would be indistinguishable from a
deliberate one — a coverage fact inferred from silence, which is the doctrine
this repository exists to reject. These tests pin the listing as a returned
fact: typed gates name their declared type, untyped gates carry an explicit
marker, and the run prints its typed/untyped denominator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from foundationscale.gates import controls as controls_runner
from foundationscale.gates.core import (
    Control,
    ControlKind,
    Coverage,
    Gate,
    GateRegistry,
    GateResult,
    Lifecycle,
)


@dataclass(frozen=True)
class _ListingCtx:
    """Context type whose ``__name__`` must appear verbatim in the listing."""

    broken: bool = False


class _TypedListingGate(Gate):
    """Declares a ``context_type``; the listing must print the type's name."""

    id = "controls_listing.typed"
    description = "Typed gate pinned by the listing tests"
    events = (Lifecycle.BUILD,)
    context_type = _ListingCtx

    def check(self, ctx: Any) -> GateResult:
        if ctx.broken:
            return self.fail("control injected the defect", Coverage(1, "contexts"))
        return self.ok("context well-formed", Coverage(1, "contexts"))

    def controls(self) -> Sequence[Control]:
        return [
            Control(
                "good-context",
                ControlKind.MUST_PASS,
                lambda: _ListingCtx(broken=False),
                note="a known-good context must not block",
            ),
            Control(
                "broken-context",
                ControlKind.MUST_FIRE,
                lambda: _ListingCtx(broken=True),
                note="injects the defect the gate must flag",
            ),
        ]


class _UntypedListingGate(Gate):
    """Declares no ``context_type``; the listing must print the marker."""

    id = "controls_listing.untyped"
    description = "Untyped gate pinned by the listing tests"
    events = (Lifecycle.BUILD,)
    # context_type deliberately left at None — the legacy untyped-broadcast
    # default whose absence the listing must state, not leave blank.

    def check(self, ctx: Any) -> GateResult:
        if ctx["broken"]:
            return self.fail("control injected the defect", Coverage(1, "contexts"))
        return self.ok("context well-formed", Coverage(1, "contexts"))

    def controls(self) -> Sequence[Control]:
        return [
            Control(
                "good-mapping",
                ControlKind.MUST_PASS,
                lambda: {"broken": False},
                note="a known-good context must not block",
            ),
            Control(
                "broken-mapping",
                ControlKind.MUST_FIRE,
                lambda: {"broken": True},
                note="injects the defect the gate must flag",
            ),
        ]


def test_listing_prints_context_type_or_explicit_marker(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() prints each gate's context_type — declaration or explicit absence.

    WHY a hermetic registry and an INJECTED walk: this test pins the listing's
    rendering, not the shipped modules' import health ( the CI ``controls``
    job asserts that itself), and a real walk would re-register every shipped
    gate into the global registry for the rest of the session. fix27 measured
    how silent that breach is: fix24 moved the walk worker, and the
    ``monkeypatch.setattr(controls_runner, "_import_gate_modules", ...)`` that
    used to stand here went dead — the stub returned its two-tuple into a void
    while main() walked and imported the real tree, and the test reddened only
    through collateral exit-code damage (an off-target provenance finding),
    never through "your stub no longer intercepts". A test whose stub is
    silently ignored is the same defect class as a control whose detector
    cannot fire; the walk is now injected through main()'s ``walk`` parameter,
    which cannot rot silently — deleting it raises TypeError at this call
    site, and bypassing it inside main() reddens
    test_main_consults_the_injected_walk below.

    The injected walk reports the one module that genuinely WAS imported into
    this process before main() ran — this test module, by pytest — not the
    empty list the old stub claimed. The provenance reconciliation is thereby
    satisfied by a FACT (the two doubles below really do come from a module
    the process imported), and the reconciliation's machinery still executes
    end-to-end instead of being fed a population of zero.
    """
    registry = GateRegistry()
    registry.register(_TypedListingGate())
    registry.register(_UntypedListingGate())
    monkeypatch.setattr(controls_runner, "REGISTRY", registry)

    exit_code = controls_runner.main(walk=lambda: ([__name__], [], []))
    out = capsys.readouterr().out

    # Both gates carry working controls, so the run itself is clean — the
    # listing, not the exit path, is the behaviour under test.
    assert exit_code == 0
    lines = out.splitlines()
    typed_lines = [line for line in lines if "controls_listing.typed" in line]
    untyped_lines = [line for line in lines if "controls_listing.untyped" in line]

    # The gate appears in the listing at all — a listing that skips gates
    # would make every assertion below vacuous.
    assert typed_lines, out
    assert untyped_lines, out
    # A declared context_type is printed as a fact...
    assert any("_ListingCtx" in line for line in typed_lines), out
    # ...and an absent declaration is stated explicitly, never left blank.
    assert any("no context_type declared" in line for line in untyped_lines), out
    # The population denominator, so "how typed are we" needs no hand tally.
    assert "context_type declared: 1/2" in out


def test_main_consults_the_injected_walk(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUST_FIRE for the walk seam: a finding only the injected walk could supply.

    The listing test above is the seam's MUST_PASS half — a healthy injected
    walk renders a clean listing with exit 0. This is the half that proves the
    seam cannot be abandoned unnoticed: fix24 moved the walk worker and the
    previous seam (a monkeypatch site) went silently dead, so this control
    exists to fail the day main() stops consulting the walk it was handed, or
    consults it and discards what it returned — the same defect in a
    consulted-costume. Red on the pre-fix27 tree with ``TypeError: main() got
    an unexpected keyword argument 'walk'`` — the seam's absence stated
    plainly, in the fix24 tradition of vocabulary-absence reds, not dressed up
    as a behavioural failure.
    """
    sentinel = "fix27.walk_probe: only an honestly consulted walk could report this"
    consultations = 0

    def probe_walk() -> tuple[list[str], list[str], list[str]]:
        nonlocal consultations
        consultations += 1
        # Claim exactly the module that WAS imported (this one, by pytest): the
        # hermetic registry's double then passes the provenance reconciliation
        # by fact, and the sentinel is the ONLY finding this run may
        # legitimately print.
        return [__name__], [], [sentinel]

    registry = GateRegistry()
    registry.register(_TypedListingGate())
    monkeypatch.setattr(controls_runner, "REGISTRY", registry)
    exit_code = controls_runner.main(walk=probe_walk)
    out = capsys.readouterr().out

    # Consulted exactly once: zero means bypassed (the fix24 shape); two would
    # mean the walk's import side effects were doubled without anyone paying
    # for them.
    assert consultations == 1
    assert exit_code == 1
    assert sentinel in out
    # ...and undiluted: the clean double contributes no finding and the census
    # is quiet over a classified tree, so any header but exactly one means an
    # off-target finding crept into the tally this control owns.
    assert "1 control failure(s):" in out
