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
that ran nothing. A registry sweep over zero gates is a blocking VACUOUS report
— a green "0 run — all clear" is ``all([])`` one level up — and the single
opt-out is constructing the registry with ``GateRegistry(event_allow_empty=...)``
for a deliberately gateless extension event. If any of those paths ever yields
success again, this suite — not production — must be where it is discovered.
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


def _registry_of(gate: Gate) -> GateRegistry:
    """One isolated registry holding exactly one gate.

    ``verify_controls`` reports per gate, so a test that needs three
    independent verdicts needs three registries; the ``fresh_registry``
    fixture supplies one per test, not one per assertion.
    """
    registry = GateRegistry()
    registry.register(gate)
    return registry


class _MustPassOnlyGate(Gate):
    """Whole and healthy in every respect but one: no MUST_FIRE control.

    Exists so the TestVerifyControls tests can assert ``len(failures) == 1`` and
    have that count mean "exactly one finding, and it is THE one under test"
    instead of "the framework was blind to the other missing kind". A
    control-free gate (``_PassingBuildGate``, ``controls() -> []``) earns TWO
    declaration findings since the MUST_PASS existence guard landed — one per
    missing kind, because each absent kind is behaviour verified against zero
    fixtures — so it cannot isolate the MUST_FIRE measurement anymore. The
    single MUST_PASS fixture below keeps this double's healthy-input behaviour
    proven; ``check()`` passes unconditionally, which is exactly what the
    fixture needs to hold, leaving the absent defective fixture as the only
    claim on trial.
    """

    id = "test.must_pass_only"
    description = "Ships a healthy-input fixture but no defective input to fire on"
    events = (Lifecycle.BUILD,)

    def check(self, ctx: Any) -> GateResult:
        return self.ok("config resolved", Coverage(1, "configs", expected=1))

    def controls(self) -> list[Control]:
        return [
            Control(
                "healthy-config",
                ControlKind.MUST_PASS,
                lambda: {"mode": "intact"},
                note="known-good input; the defect under test is the ABSENCE of "
                "any defective fixture alongside it",
            ),
        ]


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


class TestOvercoverage:
    """The denominator binds in both directions: ``checked > expected``.

    "500 of 256 reward samples examined" cannot be true as stated, so at least
    one of the numbers is wrong — double-counted units, a superset sweep, or a
    stale ``expected``. Before the fix this shape fell through the ``ok()``
    downgrade chain to PASS: the chain's final ``else`` was default-success,
    and overage was the last coverage classification with no explicit verdict.

    Doctrine-3 controls for the new detector leg:

    * MUST_FIRE: ``test_overcoverage_blocks_as_overcovered`` and
      ``test_overcoverage_by_one_blocks`` hand the gate a known-contradictory
      coverage; the detector MUST block, under its own name.
    * MUST_PASS: ``test_exact_denominator_passes`` and
      ``test_unknown_expected_cannot_be_over`` hand it known-healthy
      coverages; the detector MUST NOT fire on them. A rule that reddened
      every denominator — or tripped at equality instead of strictly above —
      would be caught here, not in production.
    """

    # -- MUST_FIRE: positive controls -----------------------------------------

    def test_overcoverage_blocks_as_overcovered(self) -> None:
        cov = Coverage(checked=500, unit="reward samples", expected=256)
        result = _LayerCoverageGate(cov).run(ctx=None)
        assert result.blocking
        assert result.verdict is Verdict.OVERCOVERED
        assert result.verdict is not Verdict.PASS
        # The contradiction is named with both numbers, the remediation is
        # stated, and the author's claim survives beside the verdict — the
        # same courtesy UNDERCOVERED extends, because a silenced claim is
        # evidence destroyed.
        assert "500 of 256 reward samples" in result.detail
        assert "exceeds" in result.detail
        assert "expected=None" in result.detail
        assert "layer statistics within bounds" in result.detail

    def test_overcoverage_by_one_blocks(self) -> None:
        # The rule must trip strictly above the denominator, not only on
        # dramatic overruns: 257/256 is as impossible as 500/256, and a gate
        # whose count is off by one is exactly the double-counted-batch shape.
        cov = Coverage(checked=257, unit="layers", expected=256)
        result = _LayerCoverageGate(cov).run(ctx=None)
        assert result.verdict is Verdict.OVERCOVERED
        assert result.blocking

    def test_sampled_declaration_does_not_pardon_overcoverage(self) -> None:
        # The pardon cannot leak. sampled= blesses a partial sweep; a count
        # above the denominator is not a partial anything, so an author must
        # not be able to buy their way out of the contradiction with a reason
        # string. If this ever passes, the declaration machinery has been
        # rewired to license a self-refuting claim.
        cov = Coverage(
            checked=500,
            unit="reward samples",
            expected=256,
            sampled=True,
            sample_reason="stratified sample of every 64th sample",
        )
        result = _LayerCoverageGate(cov).run(ctx=None)
        assert result.verdict is Verdict.OVERCOVERED
        assert result.blocking

    def test_overcoverage_string_names_the_defect(self) -> None:
        # "500/256 reward samples" printed bare reads as a typo. The string is
        # rendered where no verdict sits beside it, so it must carry the
        # contradiction itself.
        rendered = str(Coverage(500, "reward samples", expected=256))
        assert "500/256 reward samples" in rendered
        assert "over:" in rendered

    def test_overcovered_result_blocks_a_report(self, fresh_registry: GateRegistry) -> None:
        fresh_registry.register(_LayerCoverageGate(Coverage(500, "reward samples", expected=256)))
        report = fresh_registry.run(Lifecycle.BUILD, ctx=None)
        assert not report.ok
        assert len(report.blocking) == 1
        with pytest.raises(GateBlocked):
            report.raise_if_blocking()
        rendered = report.render()
        assert "OVER" in rendered
        assert "500/256 reward samples" in rendered

    def test_overcovered_verdict_survives_serialization(self) -> None:
        result = _LayerCoverageGate(Coverage(500, "reward samples", expected=256)).run(ctx=None)
        payload = json.loads(GateReport(Lifecycle.BUILD, (result,)).to_json())
        assert payload["ok"] is False
        assert payload["results"][0]["verdict"] == "OVERCOVERED"
        # Both halves of the contradiction travel on the wire; a consumer must
        # never have to re-derive that 500 > 256.
        assert payload["results"][0]["checked"] == 500
        assert payload["results"][0]["expected"] == 256

    # -- MUST_PASS: negative controls ------------------------------------------

    def test_exact_denominator_passes(self) -> None:
        # Boundary from below: a full, honest sweep must stay green, pinning
        # the rule to fire strictly ABOVE the denominator. A ">= " off-by-one
        # in is_over would turn every correct full-coverage gate red, and only
        # this test would notice.
        cov = Coverage(checked=256, unit="reward samples", expected=256)
        assert not cov.is_over
        result = _LayerCoverageGate(cov).run(ctx=None)
        assert result.verdict is Verdict.PASS
        assert not result.blocking
        assert "over:" not in str(cov)

    def test_unknown_expected_cannot_be_over(self) -> None:
        # expected=None is the sanctioned "denominator not knowable". With no
        # denominator there is nothing to contradict, and the framework must
        # not invent one to fail against — that would be a claim broader than
        # its evidence, run in reverse.
        cov = Coverage(checked=500, unit="reward samples")
        assert not cov.is_over
        result = _LayerCoverageGate(cov).run(ctx=None)
        assert result.verdict is Verdict.PASS

    def test_is_over_semantics(self) -> None:
        # Sibling of test_is_short_semantics, which pins that 300/205 is NOT
        # short: the other direction gets its own named property rather than
        # silently redefining "short".
        assert Coverage(300, "layers", expected=205).is_over
        assert not Coverage(205, "layers", expected=205).is_over
        assert not Coverage(3, "layers", expected=205).is_over
        assert not Coverage(300, "layers").is_over
        # expected=0 declares "none should exist": zero found is merely
        # vacuous (the first branch owns it), but THREE found contradicts the
        # denominator outright — 0 expected is a denominator, not an absence
        # of one.
        assert not Coverage(0, "layers", expected=0).is_over
        assert Coverage(3, "layers", expected=0).is_over


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
        # The zero-gate sweep is itself one blocking VACUOUS result now; the
        # required= id is reported missing alongside it.
        (marker,) = report.results
        assert marker.gate_id == "registry.empty_sweep.save"
        assert marker.verdict is Verdict.VACUOUS
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

    def test_run_without_required_over_empty_registry_blocks(
        self, fresh_registry: GateRegistry
    ) -> None:
        # Absent a required= assertion the framework still knows a sweep that
        # ran nothing proves nothing: the report blocks. The single sanctioned
        # opt-out is GateRegistry(event_allow_empty=...) for a deliberately
        # gateless extension event.
        report = fresh_registry.run(Lifecycle.DATA, ctx=None)
        assert not report.ok
        assert report.is_vacuous
        (only,) = report.results
        assert only.verdict is Verdict.VACUOUS
        assert only.coverage == Coverage.none("gates")

    def test_get_contains_and_len(self, fresh_registry: GateRegistry) -> None:
        gate = _PassingBuildGate()
        fresh_registry.register(gate)
        assert "test.build_passes" in fresh_registry
        assert fresh_registry.get("test.build_passes") is gate
        assert len(fresh_registry) == 1
        with pytest.raises(KeyError):
            fresh_registry.get("nope")


class TestEmptySweepBlocks:
    """A registry sweep over zero gates is the ``all([])`` bug one level up:
    it must report blocking VACUOUS, never a green "0 run — all clear"."""

    def test_empty_event_sweep_is_blocking_vacuous(self, fresh_registry: GateRegistry) -> None:
        report = fresh_registry.run(Lifecycle.SAVE, ctx=object())
        assert not report.ok
        assert report.is_vacuous
        (only,) = report.results
        assert only.verdict is Verdict.VACUOUS
        assert only.coverage == Coverage.none("gates")
        # The population the sweep drew from is a returned fact, not an inference.
        assert only.evidence["registered_gates"] == 0

    def test_populated_passing_event_still_ok(self, fresh_registry: GateRegistry) -> None:
        # Negative control: the hardening must not redden a real sweep.
        fresh_registry.register(_PassingSaveGate())
        report = fresh_registry.run(Lifecycle.SAVE, ctx=None)
        assert report.ok
        assert not report.is_vacuous
        assert [r.gate_id for r in report.results] == ["test.save_passes"]

    def test_empty_sweep_records_the_registered_population(
        self, fresh_registry: GateRegistry
    ) -> None:
        # Registered-but-not-for-this-event is the "not wired" shape: the
        # denominator belongs in the evidence, not in the operator's guesswork.
        fresh_registry.register(_PassingSaveGate())
        report = fresh_registry.run(Lifecycle.DATA, ctx=None)
        assert not report.ok
        (only,) = report.results
        assert only.verdict is Verdict.VACUOUS
        assert only.evidence["registered_gates"] == 1
        assert only.evidence["event"] == "data"

    def test_event_allow_empty_is_the_single_opt_out(self) -> None:
        registry = GateRegistry(event_allow_empty=(Lifecycle.DATA,))
        report = registry.run(Lifecycle.DATA, ctx=None)
        # The block is lifted; the report still states that zero gates ran.
        assert report.ok
        assert report.is_vacuous
        assert report.results == ()
        # The hatch does not leak to other events.
        blocked = registry.run(Lifecycle.SAVE, ctx=None)
        assert not blocked.ok

    def test_raise_if_blocking_raises_on_vacuous_report(self, fresh_registry: GateRegistry) -> None:
        report = fresh_registry.run(Lifecycle.EXPORT, ctx=None)
        with pytest.raises(GateBlocked):
            report.raise_if_blocking()

    def test_marker_prefix_ids_cannot_be_registered(self, fresh_registry: GateRegistry) -> None:
        # An author-named marker could sit in a vacuous report undetected and
        # spoof is_vacuous; the namespace belongs to the framework.
        class SpoofedMarker(Gate):
            id = "registry.empty_sweep.save"
            description = "attempts to occupy the framework's marker namespace"
            events = (Lifecycle.SAVE,)

            def check(self, ctx: Any) -> GateResult:
                return self.ok("saw things", Coverage(1, "exports", expected=1))

            def controls(self) -> list[Control]:
                return []

        with pytest.raises(ValueError, match="reserved"):
            fresh_registry.register(SpoofedMarker())


class TestVerifyControls:
    """Every "X does not exist" claim must name the positive control that proves
    its detector could have fired. verify_controls is that rule, executable."""

    def test_gate_with_no_must_fire_control_fails(self, fresh_registry: GateRegistry) -> None:
        # The double is healthy in every respect except the defect under test —
        # it declares a MUST_PASS fixture and none to fire on — so the count
        # below claims exactly one finding AND that it is the intended one.
        # Previously this reused _PassingBuildGate (controls() -> []): with the
        # MUST_PASS existence guard, a control-free gate earns two findings and
        # the == 1 below only ever held by the framework's blindness to the
        # second, off-target defect. _PassingBuildGate itself is left untouched:
        # its other 13 uses only run the gate — none invoke controls() — and its
        # "no controls at all" shape is a different defect from the one this
        # test measures.
        fresh_registry.register(_MustPassOnlyGate())
        failures = verify_controls(fresh_registry)
        assert len(failures) == 1
        assert "test.must_pass_only" in failures[0]
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
                    ),
                    # This gate passes on EVERYTHING, so its MUST_PASS control
                    # trivially holds — and that is what keeps the failure count
                    # at exactly one: without the declaration, the no-MUST_PASS
                    # existence guard appends a second finding whose subject (a
                    # missing fixture kind) is not what this test measures.
                    Control(
                        "intact-checkpoint",
                        ControlKind.MUST_PASS,
                        lambda: {"bytes_written": 51.6e9, "expected": 51.6e9},
                        note="healthy export; even a working gate passes this — "
                        "the ignored corruption fixture above is the sole "
                        "defect under test",
                    ),
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
        # Since fix24 there are exactly TWO findings here, and both are pinned
        # by content below — never a bare == 2 that a wording change would
        # slide past: the named fixture that blocked, and the per-gate PASS
        # floor naming the zero-of-one affirmation the block implies. Neither
        # finding is a duplicate of the other. The control finding reports
        # what ONE fixture observed; the floor reports a gate-level claim with
        # its own denominator, and it stays true with "blocked" swapped for
        # "abstained honestly" or "errored" — shapes in which NO per-control
        # conviction exists to carry the zero-affirmation news. The rejected
        # alternative — suppress the floor whenever a per-control failure
        # already convicts the same gate — is a detector that mutes itself
        # under overlap with another detector, the shape this project keeps
        # finding at the bottom of its incidents; and it would silently
        # undeclare the picture the day a second, PASSING healthy fixture
        # arrives, where the floor is satisfied yet the block still convicts
        # the gate, the healthy/hurtful split a maintainer most needs to see
        # whole. Two true statements about one gate cost one extra line of CI
        # output; one false silence is priced by the founding incident.
        assert len(failures) == 2
        assert any(
            "test.blocks_everything/known-good" in failure
            and "MUST_PASS control blocked" in failure
            for failure in failures
        )
        assert any(
            "test.blocks_everything: 0 of 1 MUST_PASS control(s) reached PASS" in failure
            for failure in failures
        )

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

                return [
                    Control("bad-fixture", ControlKind.MUST_FIRE, boom),
                    # The MUST_PASS fixture MUST build cleanly and pass: the
                    # raising fixture is this test's single injected defect, so
                    # everything else about the double has to be healthy — a
                    # second "fixture raised" line or a missing-MUST_PASS
                    # declaration would each break the exactly-one claim below
                    # for reasons unrelated to what the test exists to measure.
                    Control(
                        "healthy-fixture",
                        ControlKind.MUST_PASS,
                        lambda: {"dirs": 1},
                    ),
                ]

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

    def test_verify_controls_zero_targets_is_failure(self, fresh_registry: GateRegistry) -> None:
        """The verifier-layer ``all([])``: a call that targeted zero gates —
        empty registry or a selection that matched nothing — verified nothing
        and must say so; returning ``[]`` there is success over nothing."""
        empty_registry_failures = verify_controls(fresh_registry)
        assert empty_registry_failures
        assert any("0 gates targeted" in f for f in empty_registry_failures)

        fresh_registry.register(_PassingBuildGate())
        empty_selection_failures = verify_controls(fresh_registry, gate_ids=[])
        assert empty_selection_failures
        assert any("0 gates targeted" in f for f in empty_selection_failures)

    def test_gate_ids_filter_limits_verification(self, fresh_registry: GateRegistry) -> None:
        # Same repair as test_gate_with_no_must_fire_control_fails: the filtered
        # target must be clean apart from the one defect being counted, or
        # len(failures) == 1 measures a declaration confound alongside the
        # filter, not the filter alone. The content assertion (added, not
        # weakened) confirms the single finding is the intended one.
        fresh_registry.register(_MustPassOnlyGate())
        failures = verify_controls(fresh_registry, gate_ids=["test.must_pass_only"])
        assert len(failures) == 1
        assert "declares no MUST_FIRE control" in failures[0]
        # Filtering every gate out is a run over zero targets, and a run over
        # zero targets is a named failure — never an empty success list.
        zero_target = verify_controls(fresh_registry, gate_ids=[])
        assert zero_target
        assert any("0 gates targeted" in f for f in zero_target)

    def test_must_fire_only_gate_earns_the_missing_must_pass_finding(self) -> None:
        """A gate declaring only MUST_FIRE controls is a detector whose
        healthy-input behaviour was verified zero times, and that is a named
        failure — not a green run.

        The existence guard this pins was, until now, pinned only by accident:
        four TestVerifyControls tests and two harness tests reddened on it
        because their doubles happened to declare no MUST_PASS control, and
        repairing those confounds (so each ``== 1`` count measures the single
        defect it names) necessarily retired the accident. Deleting the guard
        from ``verify_controls`` would leave every one of those six green.
        This test exists so that deletion reddens something on purpose.

        The gate below is the adversary the guard was written against: a
        ``check()`` that blocks unconditionally BLOCKS its own MUST_FIRE
        fixture, so the per-control loop reports nothing, evaluates the
        MUST_PASS branch zero times, and ``all([])`` certifies it. Its verdict
        is the whole point — a detector that fires on everything detects
        nothing, and detectors that fire on everything are the ones operators
        switch off.
        """

        class _BlocksEverythingGate(Gate):
            id = "test.blocks_everything"
            description = "Fails unconditionally; ships no healthy-input fixture"
            events = (Lifecycle.SAVE,)

            def check(self, ctx: Any) -> GateResult:
                return self.fail("nope", Coverage(1, "tensors", expected=1))

            def controls(self) -> list[Control]:
                return [
                    Control(
                        "corrupt-checkpoint",
                        ControlKind.MUST_FIRE,
                        lambda: {"bytes_written": 5.94e9},
                        note="this gate blocks it — and blocks everything else too",
                    ),
                ]

        failures = verify_controls(_registry_of(_BlocksEverythingGate()))
        assert len(failures) == 1, failures  # the MUST_FIRE half is satisfied
        assert "test.blocks_everything" in failures[0]
        assert "declares no MUST_PASS control" in failures[0]
        assert "block on EVERYTHING" in failures[0]

        # An empty controls list is the same zero-trip shape twice over, so it
        # earns BOTH declaration findings — the core comment says so, and until
        # now nothing checked that it does.
        both = verify_controls(_registry_of(_PassingBuildGate()))
        assert len(both) == 2, both
        assert any("declares no MUST_FIRE control" in f for f in both)
        assert any("declares no MUST_PASS control" in f for f in both)

        # MUST_PASS control: a gate declaring both kinds, and honouring both,
        # still verifies clean. Without this the test above is satisfied by a
        # verify_controls that simply always reports a missing declaration.
        class _WholeGate(Gate):
            id = "test.whole"
            description = "Blocks the defect, passes the healthy input"
            events = (Lifecycle.SAVE,)

            def check(self, ctx: Any) -> GateResult:
                if ctx["bytes_written"] < ctx["expected"]:
                    return self.fail("truncated", Coverage(1, "tensors", expected=1))
                return self.ok("intact", Coverage(1, "tensors", expected=1))

            def controls(self) -> list[Control]:
                return [
                    Control(
                        "truncated",
                        ControlKind.MUST_FIRE,
                        lambda: {"bytes_written": 5.94e9, "expected": 51.6e9},
                    ),
                    Control(
                        "intact",
                        ControlKind.MUST_PASS,
                        lambda: {"bytes_written": 51.6e9, "expected": 51.6e9},
                    ),
                ]

        assert verify_controls(_registry_of(_WholeGate())) == []


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
