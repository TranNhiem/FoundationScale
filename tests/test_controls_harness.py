"""In-process tests for the controls runner, :mod:`foundationscale.gates.controls`.

Why this file exists
--------------------
The controls runner is the check that checks the checkers: it imports every gate
module so registration cannot be quietly skipped, refuses an empty registry, and
executes every gate's MUST_FIRE / MUST_PASS controls. It runs as its own CI job —
which is precisely how the framework's coverage report came to show it at 0%.
The detector built against "reported success over something never examined" was
itself never examined by the suite. Who checks the checker; here, this file.

Properties pinned:

* the run reports the gates and controls it actually saw, and zero of either is
  a failure, not banner decoration;
* a gate that PASSes a MUST_FIRE control reddens the whole run (the positive
  control for the positive controls — the assertion this file cannot skip);
* a gate that blocks a MUST_PASS control reddens the run;
* an unimportable gate module is a recorded finding, and the walk continues;
* the printed arithmetic cannot hide a control that was never executed.

Two defects in the shipped runner broke those promises and were pinned here as
strict xfails: a gate whose ``controls()`` cannot be built, and a gate
*subpackage* whose import raises a non-ImportError, both escaped :func:`main` as
tracebacks — documented as collected failures, shipped as crashes that never
printed the report at all. Both are fixed; the two tests now run as ordinary
assertions, and each carries a comment saying what it used to catch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import foundationscale.gates as gates_package
from foundationscale.gates import controls as controls_harness
from foundationscale.gates.core import (
    REGISTRY,
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
# Stub gates. Registered only into fresh registries; never touch the live
# REGISTRY — the suite's whole point is that the real one stays whole.
# ---------------------------------------------------------------------------


class _DeafGate(Gate):
    """Passes on a deliberately defective input — the gate the runner exists to expose."""

    id = "test.controls_harness.deaf"
    description = "Reports PASS on defective input; its MUST_FIRE control must convict it"
    events = (Lifecycle.SAVE,)

    def check(self, ctx: Any) -> GateResult:
        return self.ok("looks fine to me", Coverage(1, "exports", expected=1))

    def controls(self) -> list[Control]:
        return [
            Control(
                name="truncated-export",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: "bytes=5.94e9 expected=51.6e9",
                note="the defect is present; a working gate must block here",
            ),
        ]


class _HairTriggerGate(Gate):
    """Blocks on everything, including known-good input — the gate people disable."""

    id = "test.controls_harness.hair_trigger"
    description = "Fails every input; its MUST_PASS control must convict it"
    events = (Lifecycle.SAVE,)

    def check(self, ctx: Any) -> GateResult:
        return self.fail(f"refusing to trust {ctx!r}", Coverage(1, "exports", expected=1))

    def controls(self) -> list[Control]:
        return [
            Control(
                name="defective-export",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: "truncated",
            ),
            Control(
                name="healthy-export",
                kind=ControlKind.MUST_PASS,
                make_ctx=lambda: "intact",
            ),
        ]


class _SoundGate(Gate):
    """A gate whose verdict depends on the fixture: blocks the defective ctx only."""

    id = "test.controls_harness.sound"
    description = "Blocks defective inputs, passes healthy ones, honestly covered"
    events = (Lifecycle.SAVE,)

    def check(self, ctx: Any) -> GateResult:
        coverage = Coverage(1, "exports", expected=1)
        if ctx["defective"]:
            return self.fail("defect present in fixture", coverage)
        return self.ok("export intact", coverage)

    def controls(self) -> list[Control]:
        return [
            Control(
                name="defective-export",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: {"defective": True},
            ),
            Control(
                name="healthy-export",
                kind=ControlKind.MUST_PASS,
                make_ctx=lambda: {"defective": False},
            ),
        ]


class _ExplodingControlsGate(Gate):
    """Its control list cannot even be built — zero proven ability to fire."""

    id = "test.controls_harness.exploding_controls"
    description = "controls() raises; the runner must record this, not die of it"
    events = (Lifecycle.SAVE,)

    def check(self, ctx: Any) -> GateResult:
        return self.ok("unreachable in this suite", Coverage(1, "exports", expected=1))

    def controls(self) -> list[Control]:
        raise RuntimeError("this gate's control list cannot be constructed")


class TestLiveRegistryRun:
    """The runner against the real registry: its report must match ground truth
    readable off the public API, and the counts must be non-zero."""

    def test_run_result_agrees_with_verify_controls(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """If the console job reported a different verdict than the verifier it wraps,
        CI would trust a banner the underlying evidence contradicts."""
        expected_failures = verify_controls(REGISTRY)
        rc = controls_harness.main()
        out = capsys.readouterr().out
        assert rc == (0 if not expected_failures else 1)
        if expected_failures:
            assert f"{len(expected_failures)} control failure(s):" in out
            for failure in expected_failures:
                assert failure in out
            assert "result: FAILED" in out
        else:
            assert "result: OK" in out

    def test_reported_counts_match_the_registry_and_are_nonzero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A run that exercised no gates and no controls but printed OK is the vacuous
        pass this framework exists to prevent, one level up. Asserting the printed
        counts against registry ground truth — and against zero — is the guard."""
        controls_harness.main()
        out = capsys.readouterr().out
        gate_count = len(REGISTRY)
        control_count = sum(len(list(gate.controls())) for gate in REGISTRY)
        assert gate_count > 0, "no gates registered; this test must not pass over nothing"
        assert control_count > 0, "no controls declared; this test must not pass over nothing"
        assert f"gates registered: {gate_count}" in out
        assert f"controls declared: {control_count}" in out

    def test_imported_module_list_names_gate_modules_and_omits_the_runner(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A runner that omitted a gate module from the import list — or counted itself
        as one — would make the population it just audited un-auditable."""
        controls_harness.main()
        out = capsys.readouterr().out
        assert "foundationscale.gates.example" in out
        assert "foundationscale.gates.controls" not in out


class TestEmptyRegistryBlocks:
    def test_zero_registered_gates_is_a_failure_not_an_all_clear(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fresh_registry: GateRegistry,
    ) -> None:
        """Exit 0 over an empty registry is ``all([])`` transplanted one level up. The
        runner treats 'no gates ran' as its loudest failure; pinning that is pinning
        the reason the module exists."""
        monkeypatch.setattr(controls_harness, "REGISTRY", fresh_registry)
        rc = controls_harness.main()
        out = capsys.readouterr().out
        assert rc == 1
        assert "gates registered: 0" in out
        assert "controls declared: 0" in out
        assert "registry is empty" in out
        assert "vacuous" in out
        assert "result: FAILED" in out


class TestMustFireControlsAreEnforced:
    def test_gate_that_passes_defective_input_fails_the_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fresh_registry: GateRegistry,
    ) -> None:
        """The positive control for the positive controls. If the harness could not
        convict a deaf gate, every green controls job is theatre; this is the one
        assertion in the file that cannot be skipped."""
        fresh_registry.register(_DeafGate())
        monkeypatch.setattr(controls_harness, "REGISTRY", fresh_registry)
        rc = controls_harness.main()
        out = capsys.readouterr().out
        assert rc == 1
        assert "gates registered: 1" in out
        assert "controls declared: 1" in out
        assert "1 control failure(s):" in out
        assert "test.controls_harness.deaf/truncated-export" in out
        assert "MUST_FIRE control did not block" in out
        assert "result: FAILED" in out


class TestMustPassControlsAreEnforced:
    def test_gate_that_blocks_known_good_input_fails_the_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fresh_registry: GateRegistry,
    ) -> None:
        """A gate that blocks everything gets disabled instead of fixed; the harness
        must catch it via the MUST_PASS control, and must name that control while
        counting only it as failed."""
        fresh_registry.register(_HairTriggerGate())
        monkeypatch.setattr(controls_harness, "REGISTRY", fresh_registry)
        rc = controls_harness.main()
        out = capsys.readouterr().out
        assert rc == 1
        assert "controls declared: 2" in out
        assert "1 control failure(s):" in out
        assert "test.controls_harness.hair_trigger/healthy-export" in out
        assert "MUST_PASS control blocked" in out
        assert "test.controls_harness.hair_trigger/defective-export:" not in out


class TestSummaryArithmetic:
    def test_failures_plus_unreported_passes_equal_controls_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fresh_registry: GateRegistry,
    ) -> None:
        """The runner prints controls *declared* and controls *failed*; the passing
        remainder exists only by subtraction. Pin that arithmetic so a control that
        vanishes between counting and execution cannot drop out of the tally
        invisibly — the harness's own ``all([]）``."""
        fresh_registry.register(_SoundGate())
        fresh_registry.register(_DeafGate())
        monkeypatch.setattr(controls_harness, "REGISTRY", fresh_registry)
        rc = controls_harness.main()
        out = capsys.readouterr().out
        failures = verify_controls(fresh_registry)
        declared = 3  # 2 from _SoundGate + 1 from _DeafGate, known from the stubs
        passed = declared - len(failures)
        assert rc == 1
        assert (declared, len(failures), passed) == (3, 1, 2)
        assert f"controls declared: {declared}" in out
        assert f"{len(failures)} control failure(s):" in out
        # The passing controls belong to the sound gate; nothing it ran may be
        # reported as failed, and the failures that exist must be named. Scope
        # this to the failure section: the dispatch listing above names every
        # registered gate on purpose, so a whole-transcript check would read any
        # mention of a healthy gate as an accusation against it.
        failure_section = out.split(f"{len(failures)} control failure(s):", maxsplit=1)[1]
        assert "test.controls_harness.sound" not in failure_section
        for failure in failures:
            assert "/" in failure.split(":", maxsplit=1)[0]


class TestControlsListFailure:
    # Was a strict xfail: main() fed the full registry to verify_controls even after
    # _count_controls had already recorded that this gate's controls() raises, and
    # verify_controls calls list(gate.controls()) with no guard — so the exception
    # escaped main() as a traceback and the collected failure report was never printed.
    # A gate audit that dies instead of reporting is the failure it exists to catch.
    # Fixed: _count_controls now returns the ids it could not build, and main() hands
    # verify_controls only the gates just shown to have a buildable control list.
    def test_gate_whose_controls_cannot_be_built_is_reported_not_fatal(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fresh_registry: GateRegistry,
    ) -> None:
        """The module documents this case as a collected failure ('verify_controls
        never sees it'). Shipped behaviour should be: the finding is printed, the
        rest of the audit completes, and the exit code is 1 — not a traceback that
        buries every other finding."""
        fresh_registry.register(_SoundGate())
        fresh_registry.register(_ExplodingControlsGate())
        monkeypatch.setattr(controls_harness, "REGISTRY", fresh_registry)
        rc = controls_harness.main()
        out = capsys.readouterr().out
        assert rc == 1
        assert "test.controls_harness.exploding_controls" in out
        assert "controls() raised" in out
        assert "result: FAILED" in out


class TestImportBoundary:
    def test_gate_module_that_raises_at_import_is_recorded_and_walk_continues(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fresh_registry: GateRegistry,
    ) -> None:
        """A gate module that cannot import on this box never registered anything;
        recording it — and still auditing every neighbouring module — is the
        difference between a finding and a partial registry that looks whole."""
        (tmp_path / "wounded_probe_gate.py").write_text(
            'raise ValueError("probe import failure")\n'
        )
        (tmp_path / "clean_probe_gate.py").write_text(
            '"""Probe gate module that imports cleanly."""\n'
        )
        monkeypatch.setattr(gates_package, "__path__", [*gates_package.__path__, str(tmp_path)])
        fresh_registry.register(_SoundGate())
        monkeypatch.setattr(controls_harness, "REGISTRY", fresh_registry)
        rc = controls_harness.main()
        out = capsys.readouterr().out
        assert rc == 1
        assert "foundationscale.gates.wounded_probe_gate" in out
        assert "import raised ValueError: probe import failure" in out
        assert "foundationscale.gates.clean_probe_gate" in out

    # Was a strict xfail: _import_gate_modules called pkgutil.walk_packages with
    # onerror unset. The runner's own import of a raising gate *subpackage* was caught
    # and recorded, but advancing the generator makes pkgutil import that package
    # again internally, where a non-ImportError re-raised out of main() and the report
    # was never printed — so the modules most able to hide gates were the ones whose
    # findings vanished. Fixed: an onerror callback records the package name, deduped
    # against the loop's own record so a raising package is reported exactly once.
    def test_gate_subpackage_that_raises_at_import_is_recorded_not_fatal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fresh_registry: GateRegistry,
    ) -> None:
        """A raising gate *package* must be the same kind of finding as a raising
        gate *module*: recorded, reported, exit 1. If it escapes as a traceback,
        the promise 'the report shows all broken modules, not just the first' is
        broken by exactly the modules most able to hide gates."""
        package_dir = tmp_path / "wounded_probe_package"
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text(
            'raise RuntimeError("this gate package cannot import here")\n'
        )
        monkeypatch.setattr(gates_package, "__path__", [*gates_package.__path__, str(tmp_path)])
        fresh_registry.register(_SoundGate())
        monkeypatch.setattr(controls_harness, "REGISTRY", fresh_registry)
        rc = controls_harness.main()
        out = capsys.readouterr().out
        assert rc == 1
        assert "foundationscale.gates.wounded_probe_package" in out
        assert "result: FAILED" in out
