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
            # The healthy fixture keeps this double defective in exactly ONE
            # respect — deafness — so the harness's failure count measures
            # deafness alone. Without it, the no-MUST_PASS existence guard adds
            # a second finding whose subject is a missing declaration, which is
            # not what these tests assert. It honestly HOLDS: a gate that
            # passes everything also passes known-good input.
            Control(
                name="intact-export",
                kind=ControlKind.MUST_PASS,
                make_ctx=lambda: "bytes=51.6e9 expected=51.6e9",
                note="known-good input; this gate WOULD pass it — the ignored "
                "defective fixture above is the only defect under test",
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


def _hermetic_walk() -> tuple[list[str], list[str], list[str]]:
    """A walk reporting exactly one imported module: this one.

    The hermetic-registry tests below assemble fresh registries of test
    doubles and pin main()'s CONTROL accounting over them. Running the real
    package walk inside those tests does two things they never ask for: it
    imports every shipped gate into the process-wide REGISTRY for the rest of
    the session, and — since fix24's provenance reconciliation — it names each
    double as a gate from a module the walk never reached, because this module
    lives outside both walked roots. Both behaviours are CORRECT machinery
    aimed at the wrong population for these tests: import health over the real
    tree is owned by TestImportBoundary, the live-registry class, and the CI
    controls job. The one claim this walk makes — "exactly this module was
    imported before main() ran, and nothing failed" — is true (pytest did the
    importing), so the reconciliation is satisfied by a fact and still
    executes, never waived.
    """
    return [__name__], [], []


@pytest.fixture
def populated_registry() -> GateRegistry:
    """The live ``REGISTRY`` *after* the gate-module import walk, asserted non-empty.

    Why this fixture exists: gates register at IMPORT time, and the only walk
    that imports the gate modules is ``main()``'s. A live-registry test that
    reads ``verify_controls(REGISTRY)`` or raw registry counts BEFORE that walk
    reads an empty registry — where ``verify_controls`` honestly returns its
    one-item "0 gates targeted" refusal while ``main()`` goes on to populate
    and audit the real estate. The agreement assert then compares two honest
    numbers computed over two different populations: red in isolation, green in
    the full suite only because some earlier-collected module happened to
    import the gates package. Correctness by collection order is the vacuous
    shape wearing a scheduling costume, one level up — in the file whose whole
    job is guarding the vacuous shape.

    The walk driven here IS ``controls_harness._import_gate_modules`` — the
    private name touched deliberately, in an explicitly white-box test file:
    population is knowledge main() owns, and a re-implemented walk could drift
    from it, recreating the banner-vs-mechanism disagreement this suite exists
    to convict. Re-walking is cheap and safe: ``importlib.import_module`` is
    idempotent, so fixture walk plus ``main()`` walk costs one import total.

    All three asserts fail LOUDLY rather than admitting a hollow ground truth:
    unreadable gate modules (the audited estate itself is ill, and a ground
    truth read over the residue would carry an unstated denominator), zero gate
    modules found by the walk, and an empty registry after a clean walk. The
    third is the one that makes "I imported first and got 0 findings"
    distinguishable from "I imported nothing and got 0 findings" — those two
    statements must never compare equal in this suite.
    """
    imported, import_errors = controls_harness._import_gate_modules()
    assert not import_errors, (
        "gate modules failed to import; ground truth read over a partially "
        f"populated registry has an unstated denominator: {import_errors}"
    )
    assert imported, (
        "the gate-module walk found nothing to import — the live-registry tests have no subject"
    )
    assert len(REGISTRY) > 0, (
        "REGISTRY is empty after the gate-module walk — 'audited the live "
        "registry and got 0 findings' must never be indistinguishable from "
        "'audited an empty registry and got 0 findings'"
    )
    return REGISTRY


class TestLiveRegistryRun:
    """The runner against the real registry: its report must match ground truth
    readable off the public API, and the counts must be non-zero.

    Ground-truth reads in this class go through :func:`populated_registry`,
    which performs main()'s own import walk and ASSERTS the registry non-empty
    before any test may read from it — the population the ground truth
    describes is thereby guaranteed to be the population main() audited.
    """

    def test_run_result_agrees_with_verify_controls(
        self,
        populated_registry: GateRegistry,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """If the console job reported a different verdict than the verifier it wraps,
        CI would trust a banner the underlying evidence contradicts.

        The fixture is load-bearing, not decorative. ``verify_controls`` is
        ground truth only over the population main() audited, and gates
        register at import time. The previous form of this test read
        ``verify_controls(REGISTRY)`` BEFORE anything populated the registry:
        in isolation it compared main()'s verdict over the real estate against
        the verifier's empty-registry refusal ("0 gates targeted" — a one-item
        finding, not an empty list), failed as ``assert 0 == 1``, and in the
        full suite passed only by collection order. The agreement assertion
        below is unchanged; what changed is that its left- and right-hand sides
        are now provably views of the SAME population.
        """
        expected_failures = verify_controls(populated_registry)
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
        self,
        populated_registry: GateRegistry,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A run that exercised no gates and no controls but printed OK is the vacuous
        pass this framework exists to prevent, one level up. Asserting the printed
        counts against registry ground truth — and against zero — is the guard."""
        # main() still runs for its printed report — but the populated-registry
        # precondition the counts below depend on is now owned by the fixture,
        # not by this call happening to precede the reads. Statement order is
        # no longer load-bearing in this test, and the nonzero asserts below
        # stay as its in-body guard.
        controls_harness.main()
        out = capsys.readouterr().out
        gate_count = len(populated_registry)
        control_count = sum(len(list(gate.controls())) for gate in populated_registry)
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
        # The hermetic walk keeps the tally below exactly one in the fix24
        # world: run against the REAL walk, main() now reconciles the
        # (monkeypatched) registry against the walk's attempted set, this
        # test-module double is correctly named uncertified, and the header
        # reads "2 control failure(s):" — the machinery working as designed at
        # a population this test does not measure. That collision is measured
        # history in fix27; the injected walk states the one fact needed here.
        rc = controls_harness.main(walk=_hermetic_walk)
        out = capsys.readouterr().out
        assert rc == 1
        assert "gates registered: 1" in out
        # Two controls now: the defective fixture this gate must (and fails to)
        # block, and the healthy fixture it honestly passes. The single printed
        # failure must be the deafness itself — nothing else about the double
        # is broken, and the count below now proves exactly that. fix24's PASS
        # floor is quiet for this double by construction (its healthy fixture
        # affirms), so the one is the intended one, not a residue.
        assert "controls declared: 2" in out
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
        must catch it via the MUST_PASS control — and, since fix24, must ALSO say
        at gate level what one blocked healthy fixture out of one declared
        implies: zero affirmations, healthy-input behaviour verified zero times.
        "Counting only it as failed" was written before the per-gate PASS floor
        existed; today it would restate the staleness this edit removes."""
        fresh_registry.register(_HairTriggerGate())
        monkeypatch.setattr(controls_harness, "REGISTRY", fresh_registry)
        # Executable ground truth first, printout second: the header assert
        # below ties main()'s tally to verify_controls over the SAME registry,
        # so any finding entering main()'s list from a channel verify_controls
        # cannot see (a walk error, a provenance line) breaks the equality
        # here instead of hiding inside a hand-maintained literal count.
        failures = verify_controls(fresh_registry)
        rc = controls_harness.main(walk=_hermetic_walk)
        out = capsys.readouterr().out
        assert rc == 1
        assert "controls declared: 2" in out
        assert len(failures) == 2
        assert f"{len(failures)} control failure(s):" in out
        assert "test.controls_harness.hair_trigger/healthy-export" in out
        assert "MUST_PASS control blocked" in out
        # The second finding is fix24's per-gate PASS floor: its own claim,
        # carrying its own denominator, true independently of WHY no fixture
        # affirmed (this gate blocks one; an honestly all-abstaining gate
        # trips the same floor with no per-control conviction in sight).
        # Suppressing the floor when it overlaps a named conviction was
        # considered and rejected as a self-muting detector; the full argument
        # lives beside TestVerifyControls.test_must_pass_control_that_blocks_fails
        # in tests/test_gates_core.py, stated once, duplicated nowhere.
        assert "0 of 1 MUST_PASS control(s) reached PASS" in out
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
        rc = controls_harness.main(walk=_hermetic_walk)
        out = capsys.readouterr().out
        failures = verify_controls(fresh_registry)
        # Recomputed honestly from what the harness now counts: _DeafGate ships
        # its MUST_PASS half too, so declared = 2 (sound) + 2 (deaf) = 4. The
        # single failure is the deaf gate's ignored defective fixture; the three
        # passing controls exist only by subtraction. Before this repair,
        # (3, 1, 2) held only because verify_controls was blind to the deaf
        # gate's missing declaration — with the guard applied and the double
        # unrepaired the truth was (3, 2, 1), where the second "failure" was a
        # declaration confound, not an executed-control outcome.
        # The hermetic walk is load-bearing HERE of all places. This is the
        # one test that EQUATES main()'s printed tally with verify_controls'
        # returned tally over the same registry, and the equation holds only
        # while main()'s list contains nothing but control outcomes. Under the
        # real walk, fix24's provenance reconciliation — correctly, given the
        # attempted set it was fed — names both doubles as uncertified, main()
        # prints "3 control failure(s):", and TWO of the asserts below go red:
        # the header equality, and the failure_section guard itself, which
        # would find "test.controls_harness.sound" named inside it. Note what
        # that means: the trap this test built caught the off-target finding —
        # it reddened on its own tripwire, not by accident. The injected walk
        # keeps (4, 1, 3) a statement about controls alone. It deliberately
        # does NOT pin fix24's PASS floor: both doubles affirm one healthy
        # fixture each, so the floor is quiet here, and a floor-removal
        # regression is TestVerifyControls' beat, not this one's.
        declared = 4
        passed = declared - len(failures)
        assert rc == 1
        assert (declared, len(failures), passed) == (4, 1, 3)
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

    def test_excluded_gate_shrinks_the_certification_receipt_denominator(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fresh_registry: GateRegistry,
    ) -> None:
        """MUST_FIRE for mutation row
        ``controls/certification-receipt-fakes-its-denominator``.

        Red under the mutant: with one gate excluded from verification because
        its controls() cannot be built, the receipt prints "2 of 2" and the
        first assert below dies. Green on the shipped line ("1 of 2").

        The sister test in this class builds the divergent counters already;
        its intent is crash-vs-report ("reported, not fatal"), and it never
        reads the printed numeral — under both lines its assertions hold.
        This sibling pins the OTHER claim the same fixture makes, doctrine 2
        in print: the receipt a human eyeballs must carry a denominator the
        run did NOT compute from its own numerator. Fixture state that
        discriminates: len(verifiable_ids) != gate_count — with only healthy
        gates, anchor and mutant print identical strings and nothing is
        tested (the trap fix36 named). The hermetic walk keeps the tally to
        exactly the one intended finding (the controls() error): under the
        real walk the provenance reconciliation would add its own correct
        findings for these doubles, and rc==1 would become ambiguous evidence.
        """
        fresh_registry.register(_SoundGate())
        fresh_registry.register(_ExplodingControlsGate())
        monkeypatch.setattr(controls_harness, "REGISTRY", fresh_registry)
        rc = controls_harness.main(walk=_hermetic_walk)
        out = capsys.readouterr().out
        assert rc == 1
        assert "controls() raised" in out
        # Denominator stated as arithmetic: one gate verifiable of two
        # registered. The negative assert guards the opposite faking direction
        # (a line claiming completeness the run did not have).
        assert "controls executed for 1 of 2 registered gates" in out
        assert "controls executed for 2 of 2 registered gates" not in out

    def test_fully_verifiable_run_prints_n_of_n_honestly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fresh_registry: GateRegistry,
    ) -> None:
        """[GREEN-ON-BOTH-TREES — invariance fence, stated per the in-file
        convention] When nothing is excluded, anchor and mutant print the
        IDENTICAL string ("1 of 1"), so this fence cannot kill the receipt
        mutation and must never be cited as doing so. Its red-maker is the
        other direction of repair: a "fix" that touches the receipt line
        itself (deleting it, hardcoding it, printing 0 of 0) dies here, which
        is what keeps the MUST_FIRE sibling honest — the receipt exists,
        renders the true counters, and certifies exit 0 when certification is
        earned. Denominator: 1 verifiable of 1 registered, both stated.
        """
        fresh_registry.register(_SoundGate())
        monkeypatch.setattr(controls_harness, "REGISTRY", fresh_registry)
        rc = controls_harness.main(walk=_hermetic_walk)
        out = capsys.readouterr().out
        assert rc == 0
        assert "controls executed for 1 of 1 registered gates" in out


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
