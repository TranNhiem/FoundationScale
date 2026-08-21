"""Adversarial tests for the frozen gate contract in ``foundationscale.gates.core``.

Why this suite exists
---------------------
The gate contract is the only thing standing between a green check mark and a
silently wrong artifact. The forensic audit that motivated the framework found
checkpoint corruption, truncated exports, and zero-supervision training runs
that all reported success — and, sharpest of all, a verification tool that
reported ``all_identity: True`` on a corrupt artifact because the comparison
set was empty and ``all([])`` is ``True`` in Python.

These tests therefore attack the contract the way the estate actually failed:
they try to get PASS out of an empty comparison, out of a partial comparison,
out of an exception, out of a bare ``True`` return value, and out of a registry
that ran nothing. If any of those paths ever yields success again, this suite —
not production — must be where it is discovered.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from foundationscale.gates.core import (
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
    verify_controls,
)

# ---------------------------------------------------------------------------
# Fake gates used across the suite. Ids are prefixed ``test.`` and these are
# never registered into the global REGISTRY.
# ---------------------------------------------------------------------------


class _TensorIdentityGate(Gate):
    """Replays the forensic incident: compares tensor pairs and declares identity.

    With an empty ``pairs`` list, ``all(...)`` evaluates to ``True`` and a naive
    implementation reports full success. The framework must refuse.
    """

    id = "test.tensor_identity"
    description = "Every expert tensor is bitwise identical to its reference"
    events = (Lifecycle.PROMOTE,)

    def __init__(self, pairs: list[tuple[list[float], list[float]]]) -> None:
        self._pairs = pairs

    def check(self, ctx: Any) -> GateResult:
        # This is the exact shape of the audited bug: vacuous truth.
        identical = all(a == b for a, b in self._pairs)
        coverage = Coverage(checked=len(self._pairs), unit="tensor pairs")
        if identical:
            return self.ok("all_identity: True — every tensor matches", coverage)
        return self.fail("mismatched tensors found", coverage)

    def controls(self) -> list[Control]:
        return []


class _LayerCoverageGate(Gate):
    """Returns a caller-supplied coverage, to exercise the undercoverage rule."""

    id = "test.layer_coverage"
    description = "Checks decoder-layer activation statistics"
    events = (Lifecycle.BUILD,)

    def __init__(self, coverage: Coverage) -> None:
        self._coverage = coverage

    def check(self, ctx: Any) -> GateResult:
        return self.ok("layer statistics within bounds", self._coverage)

    def controls(self) -> list[Control]:
        return []


class _ExplodingGate(Gate):
    """Simulates the reward-module import failure: raises inside check()."""

    id = "test.exploding"
    description = "Raises unconditionally, like a reward module whose import failed"
    events = (Lifecycle.DATA,)

    def check(self, ctx: Any) -> GateResult:
        raise RuntimeError("reward module import fell over")

    def controls(self) -> list[Control]:
        return []


class _ReturnsTrueGate(Gate):
    """A gate authorial error the framework must catch: returns a bare bool."""

    id = "test.returns_true"
    description = "Returns True from check() instead of a GateResult"
    events = (Lifecycle.BUILD,)

    def check(self, ctx: Any) -> GateResult:
        return True  # type: ignore[return-value]

    def controls(self) -> list[Control]:
        return []


class _ReturnsNoneGate(Gate):
    """The other common authorial error: check() with no return at all."""

    id = "test.returns_none"
    description = "Falls off the end of check() and returns None"
    events = (Lifecycle.BUILD,)

    def check(self, ctx: Any) -> GateResult:
        pass  # type: ignore[return-value]

    def controls(self) -> list[Control]:
        return []


class _PassingBuildGate(Gate):
    id = "test.build_passes"
    description = "A well-behaved BUILD gate"
    events = (Lifecycle.BUILD,)

    def check(self, ctx: Any) -> GateResult:
        return self.ok("config resolved", Coverage(1, "configs", expected=1))

    def controls(self) -> list[Control]:
        return []


class _PassingSaveGate(Gate):
    id = "test.save_passes"
    description = "A well-behaved save gate registered for both save events"
    events = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE)

    def check(self, ctx: Any) -> GateResult:
        return self.ok("checkpoint sane", Coverage(4, "shards", expected=4))

    def controls(self) -> list[Control]:
        return []


class _RecordingGate(Gate):
    """Captures the ctx it receives, to prove the registry passes it unchanged."""

    id = "test.recording"
    description = "Records the context object handed to check()"
    events = (Lifecycle.DATA,)

    def __init__(self) -> None:
        self.seen: Any = None

    def check(self, ctx: Any) -> GateResult:
        self.seen = ctx
        return self.ok("noted", Coverage(1, "batches"))

    def controls(self) -> list[Control]:
        return []


class TestVacuousTruthRule:
    """The single most important behaviour: inspecting nothing is never PASS.

    This is the ``all([]) is True`` incident made into an executable test.
    """

    def test_empty_comparison_cannot_pass(self) -> None:
        gate = _TensorIdentityGate(pairs=[])
        result = gate.run(ctx=None)
        assert result.verdict is Verdict.VACUOUS
        assert result.verdict is not Verdict.PASS
        assert result.blocking
        # The author's claim must be preserved so the report shows what was
        # asserted versus what was actually examined.
        assert "0 tensor pairs" in result.detail
        assert "proves nothing" in result.detail
        assert "all_identity" in result.detail

    def test_vacuous_verdict_blocks_a_report(self, fresh_registry: GateRegistry) -> None:
        fresh_registry.register(_TensorIdentityGate(pairs=[]))
        report = fresh_registry.run(Lifecycle.PROMOTE, ctx=None, required=["test.tensor_identity"])
        assert not report.ok
        assert len(report.blocking) == 1
        with pytest.raises(GateBlocked):
            report.raise_if_blocking()

    def test_nonempty_identical_comparison_passes(self) -> None:
        gate = _TensorIdentityGate(pairs=[([1.0, 2.0], [1.0, 2.0])])
        result = gate.run(ctx=None)
        assert result.verdict is Verdict.PASS
        assert not result.blocking
        assert "all_identity" in result.detail

    def test_nonempty_mismatched_comparison_fails(self) -> None:
        gate = _TensorIdentityGate(pairs=[([1.0], [9.0])])
        result = gate.run(ctx=None)
        assert result.verdict is Verdict.FAIL
        assert result.blocking

    def test_coverage_none_helper_is_vacuous(self) -> None:
        cov = Coverage.none("experts")
        assert cov.is_vacuous
        assert cov.checked == 0

    def test_vacuous_fail_still_fails_not_vacuous(self) -> None:
        # A gate that *finds a defect* while examining nothing reports FAIL;
        # coverage is recorded but the downgrade machinery belongs to ok().
        class FailingOnEmpty(Gate):
            id = "test.fail_on_empty"
            description = "Fails even with zero coverage"
            events = (Lifecycle.SAVE,)

            def check(self, ctx: Any) -> GateResult:
                return self.fail("no experts on disk", Coverage.none("experts"))

            def controls(self) -> list[Control]:
                return []

        result = FailingOnEmpty().run(ctx=None)
        assert result.verdict is Verdict.FAIL
        assert result.blocking


class TestUndercoverageAndSampling:
    """ "3 of 205 checked" is not "checked" — unless the gate declares a sample."""

    def test_short_coverage_without_sample_declaration_blocks(self) -> None:
        cov = Coverage(checked=3, unit="layers", expected=205)
        result = _LayerCoverageGate(cov).run(ctx=None)
        assert result.verdict is Verdict.UNDERCOVERED
        assert result.blocking
        assert "3 of 205 layers" in result.detail
        assert "sample_reason" in result.detail

    def test_full_coverage_passes(self) -> None:
        cov = Coverage(checked=205, unit="layers", expected=205)
        result = _LayerCoverageGate(cov).run(ctx=None)
        assert result.verdict is Verdict.PASS

    def test_unknown_expected_with_full_honesty_passes(self) -> None:
        cov = Coverage(checked=19, unit="layers")
        result = _LayerCoverageGate(cov).run(ctx=None)
        assert result.verdict is Verdict.PASS
        assert result.coverage.fraction is None

    def test_declared_sample_passes_and_reason_survives_everywhere(self) -> None:
        reason = "deterministic stratified sample of every 64th layer"
        cov = Coverage(
            checked=3,
            unit="layers",
            expected=205,
            sampled=True,
            sample_reason=reason,
        )
        result = _LayerCoverageGate(cov).run(ctx=None)
        assert result.verdict is Verdict.PASS
        assert not result.blocking
        # The reason must survive into every rendering: a sample that reads as
        # full coverage in the report is the same lie told with extra steps.
        assert reason in str(result.coverage)
        assert reason in result.render()
        payload = result.to_dict()
        assert payload["sampled"] is True
        assert payload["sample_reason"] == reason
        assert payload["checked"] == 3
        assert payload["expected"] == 205

    def test_unsampled_result_serializes_sample_reason_as_none(self) -> None:
        result = _PassingBuildGate().run(ctx=None)
        assert result.to_dict()["sample_reason"] is None


class TestCoverageValidation:
    """Coverage refuses to encode impossible or unexplained claims."""

    def test_negative_checked_raises(self) -> None:
        with pytest.raises(ValueError, match="coverage cannot be negative"):
            Coverage(checked=-1, unit="tensors")

    def test_negative_expected_raises(self) -> None:
        with pytest.raises(ValueError, match="expected cannot be negative"):
            Coverage(checked=0, unit="tensors", expected=-5)

    def test_sampled_with_empty_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="requires sample_reason"):
            Coverage(checked=3, unit="layers", expected=205, sampled=True)

    def test_sampled_with_whitespace_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="requires sample_reason"):
            Coverage(
                checked=3,
                unit="layers",
                expected=205,
                sampled=True,
                sample_reason="   ",
            )

    def test_zero_expected_yields_no_fraction(self) -> None:
        assert Coverage(checked=0, unit="x", expected=0).fraction is None

    def test_is_short_semantics(self) -> None:
        assert Coverage(3, "layers", expected=205).is_short
        assert not Coverage(205, "layers", expected=205).is_short
        assert not Coverage(300, "layers", expected=205).is_short
        assert not Coverage(3, "layers").is_short

    def test_str_renders_counts_and_sample(self) -> None:
        assert str(Coverage(3, "layers", expected=205)) == "3/205 layers"
        assert str(Coverage(3, "layers")) == "3 layers"
        rendered = str(
            Coverage(3, "layers", expected=205, sampled=True, sample_reason="spot check")
        )
        assert "(sample: spot check)" in rendered


class TestFailClosed:
    """In the audited estate an exception counted as a pass. Here it is ERROR."""

    def test_exception_becomes_error_and_does_not_propagate(self) -> None:
        result = _ExplodingGate().run(ctx=None)  # must not raise
        assert result.verdict is Verdict.ERROR
        assert result.blocking
        assert "RuntimeError" in result.detail
        assert "reward module import fell over" in result.detail
        assert "traceback" in result.evidence
        assert "RuntimeError" in result.evidence["traceback"]
        assert result.coverage.is_vacuous

    def test_returning_true_is_error_not_pass(self) -> None:
        # A bare True must never be read as a pass — the audited verifier did
        # exactly that. The framework downgrades it to ERROR.
        result = _ReturnsTrueGate().run(ctx=None)
        assert result.verdict is Verdict.ERROR
        assert result.blocking
        assert "bool" in result.detail
        assert "GateResult" in result.detail

    def test_returning_none_is_error(self) -> None:
        result = _ReturnsNoneGate().run(ctx=None)
        assert result.verdict is Verdict.ERROR
        assert result.blocking
        assert "NoneType" in result.detail

    def test_error_result_blocks_a_report(self, fresh_registry: GateRegistry) -> None:
        fresh_registry.register(_ExplodingGate())
        report = fresh_registry.run(Lifecycle.DATA, ctx=None)
        assert not report.ok
        with pytest.raises(GateBlocked):
            report.raise_if_blocking()

    def test_successful_run_returns_authored_result_with_timing(self) -> None:
        # run() must hand back the gate's own verdict verbatim (plus duration),
        # not re-derive one.
        result = _PassingBuildGate().run(ctx=None)
        assert result.verdict is Verdict.PASS
        assert result.detail == "config resolved"


class TestSubclassEnforcement:
    """A gate missing its identifying attributes is rejected at class-definition
    time — before it can sit unregistered and unrun, the estate's usual bug."""

    def test_missing_id_raises(self) -> None:
        with pytest.raises(TypeError, match="must define a non-empty 'id'"):

            class MissingId(Gate):
                description = "no id"
                events = (Lifecycle.BUILD,)

                def check(self, ctx: Any) -> GateResult: ...

                def controls(self) -> list[Control]:
                    return []

    def test_missing_description_raises(self) -> None:
        with pytest.raises(TypeError, match="must define a non-empty 'description'"):

            class MissingDescription(Gate):
                id = "test.missing_description"
                events = (Lifecycle.BUILD,)

                def check(self, ctx: Any) -> GateResult: ...

                def controls(self) -> list[Control]:
                    return []

    def test_missing_events_raises(self) -> None:
        with pytest.raises(TypeError, match="must define a non-empty 'events'"):

            class MissingEvents(Gate):
                id = "test.missing_events"
                description = "no lifecycle events"

                def check(self, ctx: Any) -> GateResult: ...

                def controls(self) -> list[Control]:
                    return []

    def test_empty_string_id_raises(self) -> None:
        with pytest.raises(TypeError, match="must define a non-empty 'id'"):

            class EmptyId(Gate):
                id = ""
                description = "falsy id"
                events = (Lifecycle.BUILD,)

                def check(self, ctx: Any) -> GateResult: ...

                def controls(self) -> list[Control]:
                    return []

    def test_events_as_list_raises(self) -> None:
        with pytest.raises(TypeError, match="events must be a tuple"):

            class ListEvents(Gate):
                id = "test.list_events"
                description = "events as a list"
                events = [Lifecycle.BUILD]  # type: ignore[assignment]

                def check(self, ctx: Any) -> GateResult: ...

                def controls(self) -> list[Control]:
                    return []

    def test_events_containing_non_lifecycle_raises(self) -> None:
        with pytest.raises(TypeError, match="events must be a tuple"):

            class MixedEvents(Gate):
                id = "test.mixed_events"
                description = "events tuple with a string inside"
                events = (Lifecycle.BUILD, "save")  # type: ignore[assignment]

                def check(self, ctx: Any) -> GateResult: ...

                def controls(self) -> list[Control]:
                    return []

    def test_well_formed_subclass_defines_cleanly(self) -> None:
        gate = _PassingBuildGate()
        assert gate.id == "test.build_passes"
        assert gate.events == (Lifecycle.BUILD,)


class TestSkip:
    """skip() is the only sanctioned way to not-check: it must leave a reason."""

    def test_skip_does_not_block_and_keeps_reason(self) -> None:
        result = _PassingBuildGate().skip("single-GPU run; topology gate N/A")
        assert result.verdict is Verdict.SKIP
        assert not result.blocking
        assert result.detail == "single-GPU run; topology gate N/A"
        assert result.coverage.is_vacuous

    def test_skip_without_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a reason"):
            _PassingBuildGate().skip("")

    def test_skip_with_whitespace_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a reason"):
            _PassingBuildGate().skip("   ")


class TestRegistry:
    """The registry exists because in the estate the export byte check lived in
    one launcher script and was silently absent from the other."""

    def test_duplicate_id_rejected(self, fresh_registry: GateRegistry) -> None:
        fresh_registry.register(_PassingBuildGate())
        with pytest.raises(ValueError, match="duplicate gate id"):
            fresh_registry.register(_PassingBuildGate())

    def test_for_event_selects_by_lifecycle(self, fresh_registry: GateRegistry) -> None:
        fresh_registry.register(_PassingBuildGate())
        fresh_registry.register(_PassingSaveGate())
        assert [g.id for g in fresh_registry.for_event(Lifecycle.BUILD)] == ["test.build_passes"]
        assert [g.id for g in fresh_registry.for_event(Lifecycle.FIRST_SAVE)] == [
            "test.save_passes"
        ]
        assert [g.id for g in fresh_registry.for_event(Lifecycle.SAVE)] == ["test.save_passes"]
        assert fresh_registry.for_event(Lifecycle.LAUNCH) == []

    def test_run_passes_ctx_unchanged(self, fresh_registry: GateRegistry) -> None:
        recorder = _RecordingGate()
        fresh_registry.register(recorder)
        ctx = {"batch": object()}
        fresh_registry.run(Lifecycle.DATA, ctx=ctx)
        assert recorder.seen is ctx

    def test_run_only_executes_gates_for_that_event(self, fresh_registry: GateRegistry) -> None:
        fresh_registry.register(_PassingBuildGate())
        fresh_registry.register(_PassingSaveGate())
        report = fresh_registry.run(Lifecycle.BUILD, ctx=None)
        assert [r.gate_id for r in report.results] == ["test.build_passes"]
        assert report.ok

    def test_required_gate_that_did_not_run_blocks_even_with_all_passing(
        self, fresh_registry: GateRegistry
    ) -> None:
        # The exact estate pattern: everything that ran was green, but the gate
        # that mattered was never wired into this launcher. ok must be False.
        fresh_registry.register(_PassingBuildGate())
        report = fresh_registry.run(
            Lifecycle.BUILD,
            ctx=None,
            required={"test.build_passes", "checkpoint.expert_bytes"},
        )
        assert all(r.verdict is Verdict.PASS for r in report.results)
        assert report.missing == ("checkpoint.expert_bytes",)
        assert not report.ok
        with pytest.raises(GateBlocked):
            report.raise_if_blocking()

    def test_registry_that_silently_ran_zero_gates_blocks_when_required(
        self, fresh_registry: GateRegistry
    ) -> None:
        # Nothing registered at all: results is empty, and required= is what
        # turns that from a silent nothing into a loud block. One level up from
        # ``all([]) is True``.
        report = fresh_registry.run(Lifecycle.SAVE, ctx=None, required=["checkpoint.expert_bytes"])
        assert report.results == ()
        assert report.missing == ("checkpoint.expert_bytes",)
        assert not report.ok
        assert "MISSING" in report.render()

    def test_gate_registered_for_other_event_counts_as_missing(
        self, fresh_registry: GateRegistry
    ) -> None:
        # Being registered *somewhere* is not enough — a save gate does not run
        # at LAUNCH, and the caller asserting it must is told the truth.
        fresh_registry.register(_PassingSaveGate())
        report = fresh_registry.run(Lifecycle.LAUNCH, ctx=None, required=["test.save_passes"])
        assert report.missing == ("test.save_passes",)
        assert not report.ok

    def test_missing_list_is_sorted(self, fresh_registry: GateRegistry) -> None:
        report = fresh_registry.run(Lifecycle.DATA, ctx=None, required=["zeta.gate", "alpha.gate"])
        assert report.missing == ("alpha.gate", "zeta.gate")

    def test_run_without_required_over_empty_registry_is_vacuously_ok(
        self, fresh_registry: GateRegistry
    ) -> None:
        # Documents the asymmetry honestly: absent a required= assertion, the
        # framework cannot know a gate was supposed to exist.
        report = fresh_registry.run(Lifecycle.DATA, ctx=None)
        assert report.results == ()
        assert report.ok

    def test_get_contains_and_len(self, fresh_registry: GateRegistry) -> None:
        gate = _PassingBuildGate()
        fresh_registry.register(gate)
        assert "test.build_passes" in fresh_registry
        assert fresh_registry.get("test.build_passes") is gate
        assert len(fresh_registry) == 1
        with pytest.raises(KeyError):
            fresh_registry.get("nope")


class TestVerifyControls:
    """Every "X does not exist" claim must name the positive control that proves
    its detector could have fired. verify_controls is that rule, executable."""

    def test_gate_with_no_must_fire_control_fails(self, fresh_registry: GateRegistry) -> None:
        fresh_registry.register(_PassingBuildGate())  # controls() -> []
        failures = verify_controls(fresh_registry)
        assert len(failures) == 1
        assert "test.build_passes" in failures[0]
        assert "declares no MUST_FIRE control" in failures[0]

    def test_must_fire_control_that_gate_ignores_fails(self, fresh_registry: GateRegistry) -> None:
        class DeafGate(Gate):
            id = "test.deaf"
            description = "Passes on deliberately defective input"
            events = (Lifecycle.SAVE,)

            def check(self, ctx: Any) -> GateResult:
                return self.ok("looks fine", Coverage(1, "tensors"))

            def controls(self) -> list[Control]:
                return [
                    Control(
                        "corrupt-checkpoint",
                        ControlKind.MUST_FIRE,
                        lambda: {"bytes_written": 5.94e9, "expected": 51.6e9},
                        note="truncated export at rc=0",
                    )
                ]

        fresh_registry.register(DeafGate())
        failures = verify_controls(fresh_registry)
        assert len(failures) == 1
        assert "test.deaf/corrupt-checkpoint" in failures[0]
        assert "MUST_FIRE control did not block" in failures[0]
        assert "got PASS" in failures[0]
        assert "The defect was present and the gate reported success." in failures[0]

    def test_must_pass_control_that_blocks_fails(self, fresh_registry: GateRegistry) -> None:
        class BlocksEverything(Gate):
            id = "test.blocks_everything"
            description = "Blocks known-good input — the kind of gate people disable"
            events = (Lifecycle.SAVE,)

            def check(self, ctx: Any) -> GateResult:
                return self.fail("always unhappy", Coverage(1, "tensors"))

            def controls(self) -> list[Control]:
                return [
                    Control(
                        "broken-input",
                        ControlKind.MUST_FIRE,
                        lambda: "broken",
                    ),
                    Control(
                        "known-good",
                        ControlKind.MUST_PASS,
                        lambda: "healthy",
                    ),
                ]

        fresh_registry.register(BlocksEverything())
        failures = verify_controls(fresh_registry)
        # The MUST_FIRE control held; only the MUST_PASS one fails.
        assert len(failures) == 1
        assert "test.blocks_everything/known-good" in failures[0]
        assert "MUST_PASS control blocked" in failures[0]

    def test_raising_fixture_is_reported_not_raised(self, fresh_registry: GateRegistry) -> None:
        class BadFixtureGate(Gate):
            id = "test.bad_fixture"
            description = "Control fixture that cannot even be constructed"
            events = (Lifecycle.EXPORT,)

            def check(self, ctx: Any) -> GateResult:
                return self.ok("unreachable", Coverage(1, "dirs"))

            def controls(self) -> list[Control]:
                def boom() -> Any:
                    raise ValueError("kaboom in tmpdir setup")

                return [Control("bad-fixture", ControlKind.MUST_FIRE, boom)]

        fresh_registry.register(BadFixtureGate())
        failures = verify_controls(fresh_registry)  # must not raise
        assert len(failures) == 1
        assert "test.bad_fixture/bad-fixture" in failures[0]
        assert "fixture raised ValueError: kaboom in tmpdir setup" in failures[0]

    def test_correct_gate_passes_all_controls(self, fresh_registry: GateRegistry) -> None:
        class ExpertBytesGate(Gate):
            """The gate the estate needed: byte counts must match declaration."""

            id = "test.expert_bytes"
            description = "Expert parameter bytes match the declared shape"
            events = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE)

            def check(self, ctx: Any) -> GateResult:
                bad = [s for s in ctx["sizes"] if s != ctx["expected_bytes"]]
                cov = Coverage(len(ctx["sizes"]), "expert tensors", expected=ctx["declared"])
                if bad:
                    return self.fail(f"{len(bad)} experts with wrong byte count", cov)
                return self.ok("all expert byte counts match", cov)

            def controls(self) -> list[Control]:
                return [
                    Control(
                        "aliased-16-of-128",
                        ControlKind.MUST_FIRE,
                        lambda: {
                            "sizes": [16],
                            "expected_bytes": 64,
                            "declared": 1,
                        },
                        note="128 experts collapsed to 16 by local-name save",
                    ),
                    Control(
                        "healthy-checkpoint",
                        ControlKind.MUST_PASS,
                        lambda: {
                            "sizes": [64, 64],
                            "expected_bytes": 64,
                            "declared": 2,
                        },
                    ),
                ]

        fresh_registry.register(ExpertBytesGate())
        assert verify_controls(fresh_registry) == []

    def test_gate_ids_filter_limits_verification(self, fresh_registry: GateRegistry) -> None:
        fresh_registry.register(_PassingBuildGate())  # has no controls at all
        failures = verify_controls(fresh_registry, gate_ids=["test.build_passes"])
        assert len(failures) == 1
        # Asking for a gate that has controls but filtering it out yields nothing.
        assert (
            verify_controls(fresh_registry, gate_ids=[]) == []
            or True  # gate_ids empty list iterates no targets
        )
        assert verify_controls(fresh_registry, gate_ids=[]) == []


class TestGateReport:
    """The report is what operators and CI actually read; its truthfulness is
    the contract's user-visible surface."""

    def _mixed_report(self) -> GateReport:
        return GateReport(
            event=Lifecycle.SAVE,
            results=(
                GateResult("g.pass", Verdict.PASS, Coverage(4, "shards", 4)),
                GateResult("g.fail", Verdict.FAIL, Coverage(4, "shards", 4)),
                GateResult("g.vac", Verdict.VACUOUS, Coverage.none("experts")),
                GateResult("g.under", Verdict.UNDERCOVERED, Coverage(3, "layers", 205)),
                GateResult("g.skip", Verdict.SKIP, Coverage.none("units")),
                GateResult("g.err", Verdict.ERROR, Coverage.none("units")),
            ),
        )

    def test_blocking_selects_only_blocking_results(self) -> None:
        report = self._mixed_report()
        assert {r.gate_id for r in report.blocking} == {
            "g.fail",
            "g.vac",
            "g.under",
            "g.err",
        }
        assert not report.ok

    def test_ok_report(self) -> None:
        report = GateReport(
            event=Lifecycle.BUILD,
            results=(GateResult("g.pass", Verdict.PASS, Coverage(1, "configs", 1)),),
        )
        assert report.ok
        assert report.blocking == ()
        assert "all clear" in report.render()

    def test_render_shows_event_counts_symbols_and_missing(self) -> None:
        report = GateReport(
            event=Lifecycle.EXPORT,
            results=(GateResult("g.fail", Verdict.FAIL, Coverage(1, "dirs")),),
            missing=("export.byte_count",),
        )
        rendered = report.render()
        assert rendered.startswith("gates @ export: 1 run — 1 blocking, 1 MISSING")
        assert "  [   FAIL] g.fail: 1 dirs" in rendered
        assert "  [MISSING] export.byte_count: required but never ran" in rendered

    def test_to_json_round_trip(self) -> None:
        report = self._mixed_report()
        payload = json.loads(report.to_json())
        assert payload["event"] == "save"
        assert payload["ok"] is False
        assert payload["missing"] == []
        assert len(payload["results"]) == 6
        first = payload["results"][0]
        assert first["gate"] == "g.pass"
        assert first["verdict"] == "PASS"
        assert first["checked"] == 4
        assert first["expected"] == 4
        assert first["unit"] == "shards"
        # Stable under re-serialization: parsing and dumping twice matches.
        assert json.loads(report.to_json()) == payload

    def test_to_json_with_missing(self) -> None:
        report = GateReport(event=Lifecycle.LAUNCH, results=(), missing=("launch.manifest",))
        payload = json.loads(report.to_json())
        assert payload["ok"] is False
        assert payload["missing"] == ["launch.manifest"]

    def test_raise_if_blocking_raises_gateblocked_carrying_report(self) -> None:
        report = self._mixed_report()
        with pytest.raises(GateBlocked) as excinfo:
            report.raise_if_blocking()
        assert excinfo.value.report is report
        # The exception message IS the rendered report, so logs capture the
        # full context without extra plumbing.
        assert str(excinfo.value) == report.render()

    def test_raise_if_blocking_is_a_noop_when_ok(self) -> None:
        report = GateReport(
            event=Lifecycle.BUILD,
            results=(GateResult("g.pass", Verdict.PASS, Coverage(1, "configs", 1)),),
        )
        assert report.raise_if_blocking() is None


class TestTiming:
    """duration_s must be populated on every path; an unmeasured gate is one
    more silent gap in the evidence."""

    def test_success_path_duration_nonnegative(self) -> None:
        result = _PassingBuildGate().run(ctx=None)
        assert isinstance(result.duration_s, float)
        assert result.duration_s >= 0.0

    def test_success_path_duration_reflects_elapsed_time(self) -> None:
        class SlowGate(Gate):
            id = "test.slow"
            description = "Sleeps so its runtime is observable"
            events = (Lifecycle.DATA,)

            def check(self, ctx: Any) -> GateResult:
                time.sleep(0.02)
                return self.ok("done", Coverage(1, "batches"))

            def controls(self) -> list[Control]:
                return []

        result = SlowGate().run(ctx=None)
        assert result.duration_s >= 0.01

    def test_error_path_duration_populated(self) -> None:
        class SlowExplodingGate(Gate):
            id = "test.slow_exploding"
            description = "Sleeps, then raises"
            events = (Lifecycle.DATA,)

            def check(self, ctx: Any) -> GateResult:
                time.sleep(0.02)
                raise RuntimeError("late failure")

            def controls(self) -> list[Control]:
                return []

        result = SlowExplodingGate().run(ctx=None)
        assert result.verdict is Verdict.ERROR
        assert result.duration_s >= 0.01

    def test_duration_survives_serialization(self) -> None:
        result = _PassingBuildGate().run(ctx=None)
        assert "duration_s" in result.to_dict()
        payload = json.loads(GateReport(Lifecycle.BUILD, (result,)).to_json())
        assert payload["results"][0]["duration_s"] == round(result.duration_s, 4)
