"""Robustness tests for ``foundationscale.gates.core.verify_controls`` itself.

Why this suite exists
---------------------
``verify_controls`` is the function this package wires into CI and then trusts:
an empty return is how it says "all controls held". That makes its own failure
modes worth pinning, because two author-side errors used to break that contract
at exactly the wrong moment:

* a gate whose ``controls()`` raises took the whole call down with it, losing
  every finding already collected for every earlier gate;
* an unknown id in ``gate_ids`` raised ``KeyError`` before a single finding was
  returned — and had the id merely been dropped instead, the call would have
  returned ``[]`` over zero units of work, the ``all([])`` incident wearing a
  registry costume.

The property under test is therefore not "does not crash" — a function that
returns early satisfies that — but "every finding that exists comes back".
"""

from __future__ import annotations

from typing import Any

from foundationscale.gates.core import (
    Control,
    ControlKind,
    Coverage,
    Gate,
    GateRegistry,
    GateResult,
    Lifecycle,
    verify_controls,
)

# ---------------------------------------------------------------------------
# Gates. Ids are prefixed ``test.`` and none of these are ever registered into
# the global REGISTRY: every test builds its own GateRegistry.
# ---------------------------------------------------------------------------


class _ControlsExplodeGate(Gate):
    """A gate whose control list cannot even be built."""

    id = "test.controls_explode"
    description = "controls() raises, like a fixture module whose import failed"
    events = (Lifecycle.SAVE,)

    def check(self, ctx: Any) -> GateResult:
        return self.ok("irrelevant — controls never got this far", Coverage(1, "units"))

    def controls(self) -> list[Control]:
        raise RuntimeError("control fixture module failed to import")


class _DeafGate(Gate):
    """Reports success on everything, so its own MUST_FIRE control exposes it.

    This is the useful *second* gate: its ``controls()`` builds fine, so it is
    a sound verification target, but running that control must yield a finding
    (the gate does not block on its own defective fixture). A test that merely
    survives alongside a broken gate proves nothing unless this gate's finding
    demonstrably comes back too.
    """

    id = "test.deaf"
    description = "Passes even on a deliberately defective input"
    events = (Lifecycle.SAVE,)

    def check(self, ctx: Any) -> GateResult:
        return self.ok("looks fine to me", Coverage(1, "checkpoints"))

    def controls(self) -> list[Control]:
        return [
            Control(
                "corrupt-checkpoint",
                ControlKind.MUST_FIRE,
                make_ctx=lambda: {"corrupt": True},
                note="a defective input this gate will fail to notice",
            )
        ]


class _HonestGate(Gate):
    """A sound gate: blocks on its defective fixture, passes on its sound one."""

    id = "test.honest"
    description = "Fails exactly when the ctx carries the injected defect"
    events = (Lifecycle.SAVE,)

    def check(self, ctx: Any) -> GateResult:
        cov = Coverage(1, "checkpoints")
        if ctx["corrupt"]:
            return self.fail("injected defect present", cov)
        return self.ok("no defect found", cov)

    def controls(self) -> list[Control]:
        return [
            Control(
                "corrupt-checkpoint",
                ControlKind.MUST_FIRE,
                make_ctx=lambda: {"corrupt": True},
                note="the defect this gate exists to catch",
            ),
            Control(
                "sound-checkpoint",
                ControlKind.MUST_PASS,
                make_ctx=lambda: {"corrupt": False},
                note="known-good; proves this gate does not block unconditionally",
            ),
        ]


class TestControlsThatRaise:
    """A gate whose controls() raises is a finding, not an exception."""

    def test_raising_controls_reports_gate_exception_and_message(self) -> None:
        """If this did not hold, CI would believe the runner — not a gate — broke.

        The finding must carry the gate id, the exception type name and its
        message; anything less sends the reader to the wrong codebase.
        """
        reg = GateRegistry()
        reg.register(_ControlsExplodeGate())

        failures = verify_controls(reg)

        assert failures, "a gate whose controls() cannot run produced no finding at all"
        assert any(
            "test.controls_explode" in f
            and "RuntimeError" in f
            and "control fixture module failed to import" in f
            for f in failures
        ), f"finding must name the gate, the exception type and the message: {failures!r}"

    def test_raising_controls_loses_no_other_gate_findings(self) -> None:
        """If the loop aborted at the first gate, no one would learn that test.deaf
        reports success on a defective checkpoint.

        The exploding gate is registered *first*: recovering the deaf gate's
        MUST_FIRE finding afterwards proves the loop continued. A "no crash"
        test cannot distinguish a fixed verify_controls from one that returned
        early; this assertion can.
        """
        reg = GateRegistry()
        reg.register(_ControlsExplodeGate())
        reg.register(_DeafGate())

        failures = verify_controls(reg)

        assert any("test.controls_explode" in f and "RuntimeError" in f for f in failures)
        assert any("test.deaf/corrupt-checkpoint" in f and "MUST_FIRE" in f for f in failures)


class TestUnknownGateIds:
    """An unregistered id is a finding — dropping it would be a silent success."""

    def test_unknown_id_is_reported_and_the_list_is_not_empty(self) -> None:
        """If the id were dropped, verify_controls would return [] over zero
        gates — and [] is documented to mean "all controls held". That is the
        all([]) bug, committed by the very function written to catch it.
        """
        reg = GateRegistry()
        reg.register(_HonestGate())

        failures = verify_controls(reg, gate_ids=["test.no_such_gate"])

        assert failures != [], (
            "an empty return here would read as 'all controls held' while zero "
            "registered gates were actually verified — the exact silent-success "
            "failure this package exists to detect"
        )
        assert any("test.no_such_gate" in f for f in failures)

    def test_unknown_id_does_not_cost_known_ids_their_verification(self) -> None:
        """One typo in a five-gate list must not erase the other four.

        The unknown-id finding and the known gate's MUST_FIRE finding must both
        be present in the same returned list.
        """
        reg = GateRegistry()
        reg.register(_DeafGate())

        failures = verify_controls(reg, gate_ids=["test.no_such_gate", "test.deaf"])

        assert any("test.no_such_gate" in f for f in failures)
        assert any("test.deaf/corrupt-checkpoint" in f for f in failures)


class TestSoundConfiguration:
    """Positive control for this suite itself."""

    def test_sound_gate_with_holding_controls_returns_empty(self) -> None:
        """If this ever failed, every assertion above would be unfalsifiable: a
        verify_controls hard-coded to complain about everything would satisfy
        them all. Only a passing happy path proves the findings mean something.
        """
        reg = GateRegistry()
        reg.register(_HonestGate())

        failures = verify_controls(reg)

        assert failures == [], f"a sound gate with holding controls produced: {failures!r}"
