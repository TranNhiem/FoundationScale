"""Regression tests for the hunt-report repairs: fixture defect-guards (F3),
probe alias-control attribution (F2), and stated-abstention honesty (F6).

Fail-before / pass-after status per the repair contract is stated on every
test. The PASSES-BEFORE tests are not dead weight: they are the MUST_PASS
halves of the three repaired detectors, and each one would fail if a patch
over-corrected (refused legitimate fixtures, never credited a firing control,
or started blocking declared abstentions) — which is the doctrine-5 symmetric
failure of the defect being fixed.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

import foundationscale
from foundationscale.gates import fixtures
from foundationscale.gates.core import Coverage, Gate, GateResult, Lifecycle, Verdict


# The probe is a tool script, not an importable package module. Load it by path
# relative to the installed foundationscale package (src-layout checkout). If the
# file cannot be located this loader raises AT COLLECTION — loudly — because a
# suite that quietly skips the probe tests has verified nothing.
def _load_probe():
    pkg_file = Path(foundationscale.__file__).resolve()
    for candidate in pkg_file.parents:
        probe_path = candidate / "tools" / "real_checkpoint_probe.py"
        if probe_path.is_file():
            spec = importlib.util.spec_from_file_location("real_checkpoint_probe", probe_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules.setdefault(spec.name, module)
            spec.loader.exec_module(module)
            return module
    raise AssertionError(
        "tools/real_checkpoint_probe.py not found beside the installed "
        "foundationscale package — run these tests from a repo checkout; "
        "silently skipping them would be a vacuous green"
    )


probe = _load_probe()


# ---------------------------------------------------------------------------
# Finding 3 — defect-bearing fixture builders must refuse defect-free output
# ---------------------------------------------------------------------------


class TestAliasedFixtureGuards:
    def test_identity_period_is_refused(self):
        """[FAILS-BEFORE] The hunt acceptance test: (16, period=16) currently
        returns a fixture whose payloads are all byte-distinct (source map is
        the identity) with no error — a healthy set wearing a MUST_FIRE label.
        After the patch the builder refuses at the source."""
        with pytest.raises(ValueError, match="strictly smaller"):
            fixtures.make_aliased_experts(num_experts=16, period=16)

    def test_zero_experts_is_refused(self):
        """[FAILS-BEFORE] make_aliased_experts(0) currently returns ZERO tensors:
        the silent path in which a dependent MUST_FIRE control holds only via a
        gate's VACUOUS tripwire."""
        with pytest.raises(ValueError, match="ZERO tensors"):
            fixtures.make_aliased_experts(num_experts=0)

    def test_zero_layers_is_refused(self):
        """[FAILS-BEFORE] Same silent path, one loop level up: 16 % 8 == 0 today,
        then range(0) zero-trips the layer loop."""
        with pytest.raises(ValueError, match="zero layers"):
            fixtures.make_aliased_experts(num_experts=16, period=8, num_layers=0)

    def test_zero_period_is_refused_with_valueerror(self):
        """[FAILS-BEFORE] Currently dies as bare ZeroDivisionError from the
        divisibility guard — an unguarded crash, not a named refusal. The test
        demands the contract every other guard keeps: ValueError with a reason."""
        with pytest.raises(ValueError, match="genuinely distinct"):
            fixtures.make_aliased_experts(num_experts=16, period=0)

    def test_divisibility_guard_unchanged(self):
        """[PASSES-BEFORE — regression pin] The pre-existing guard the finding
        anchored to is preserved verbatim, message included."""
        with pytest.raises(ValueError, match="must be a multiple"):
            fixtures.make_aliased_experts(num_experts=16, period=6)


class TestLocalNameFixtureGuards:
    def test_zero_local_is_refused(self):
        """[FAILS-BEFORE] make_local_name_experts(num_local=0) currently returns
        zero tensors: the naming-signature defect cannot exist with no names."""
        with pytest.raises(ValueError, match="zero tensors means zero names"):
            fixtures.make_local_name_experts(num_local=0)

    def test_zero_layers_is_refused(self):
        """[FAILS-BEFORE] Same vacuity through the layer loop."""
        with pytest.raises(ValueError, match="zero tensors means zero names"):
            fixtures.make_local_name_experts(num_local=16, num_layers=0)


class TestEmptyFixtureGuard:
    def test_zero_declared_is_refused(self):
        """[FAILS-BEFORE] make_empty_experts(0) currently returns a dense-model
        artifact with no defect; the fixture's defect is absence WHILE DECLARED."""
        with pytest.raises(ValueError, match="ABSENT WHILE DECLARED"):
            fixtures.make_empty_experts(declared_expert_count=0)


class TestFixtureHealthyPaths:
    def test_aliased_default_fixture_materially_contains_aliasing(self):
        """[PASSES-BEFORE — MUST_PASS for the guard, and the incident pin]
        The guarded builder must still reproduce the audited geometry exactly:
        512 tensors (128 experts x 2 layers x 2 params) over exactly 64 distinct
        payloads (16 distinct experts x 2 layers x 2 params), with byte-identity
        between experts 16 apart in the same layer and parameter."""
        es = fixtures.make_aliased_experts()
        assert es.declared_expert_count == 128
        assert len(es.tensors) == 512
        assert len(set(es.tensors.values())) == 64
        for expert in (16, 100):
            assert (
                es.tensors[f"layers.0.experts.{expert}.linear_fc1.weight"]
                == es.tensors[f"layers.0.experts.{expert % 16}.linear_fc1.weight"]
            )
        assert sorted(set(es.expert_index.values())) == list(range(128))

    def test_aliased_half_period_builds_with_real_replication(self):
        """[PASSES-BEFORE] (16, period=8) is a legitimate replication and must
        NOT be swept away by the identity-period guard: 32 x 2 = 64 tensors over
        16 distinct payloads (8 sources x 1 layer x 2 params... default 2 layers
        -> 32 distinct)."""
        es = fixtures.make_aliased_experts(num_experts=16, period=8, num_layers=1)
        assert len(es.tensors) == 32
        assert len(set(es.tensors.values())) == 16

    def test_local_name_default_fixture_materially_contains_signature(self):
        """[PASSES-BEFORE — MUST_PASS] 32 keys, every one lacking a parseable
        global expert index: the defect signature is present in the artifact."""
        es = fixtures.make_local_name_experts()
        assert len(es.tensors) == 32
        assert es.expert_index == {}
        assert all(fixtures.parse_global_expert_index(name) is None for name in es.tensors)

    def test_dense_healthy_fixture_remains_buildable(self):
        """[PASSES-BEFORE — doctrine-5 pin] A zero-expert HEALTHY set is a
        legitimate dense-model MUST_PASS input. The guards refuse defect-free
        'broken' fixtures; they must not start refusing layouts, or the patch
        is minting defects where none exist."""
        es = fixtures.make_healthy_experts(num_experts=0)
        assert es.tensors == {}
        assert es.declared_expert_count == 0


# ---------------------------------------------------------------------------
# Finding 2 — the alias control credits only ATTRIBUTABLE detection
# ---------------------------------------------------------------------------


class TestAliasControlAttribution:
    def test_blocking_baseline_is_inconclusive_not_fired(self):
        """[FAILS-BEFORE] The hunt acceptance test: with a baseline that already
        blocks, the current tree returns status 'fired' alongside confounded
        True — and _deliver adds no blocking reason. After the patch the run is
        a stated abstention: INCONCLUSIVE, confounded, reason named. (The
        injected FAIL verdict itself being correct is what makes today's
        mislabel sting: detection happened AND attribution still failed.)"""
        ctx = fixtures.healthy_sharded_moe_ctx(num_experts=8, num_layers=2)
        baseline = GateResult(
            gate_id=probe.ExpertDistinctnessGate.id,
            verdict=Verdict.FAIL,
            coverage=Coverage(16, "expert tensors"),
            detail="pre-existing block on the unmodified artifact",
        )
        control = probe.run_alias_control(ctx, 2, baseline)
        assert control["status"] == "inconclusive"
        assert control["confounded"] is True
        assert control["verdict"] == "FAIL"
        assert "baseline" in control["inconclusive_reason"]

    def test_crash_verdict_is_not_credited(self, monkeypatch):
        """[FAILS-BEFORE] A gate that raises inside check() yields ERROR via
        Gate.run; today result.blocking credits that as 'fired'. After the
        patch only FAIL — a defect verdict, not a malfunction — can fire."""
        ctx = fixtures.healthy_sharded_moe_ctx(num_experts=8, num_layers=2)
        baseline = probe.ExpertDistinctnessGate().run(ctx)
        assert not baseline.blocking, (
            f"fixture drifted: the healthy sharded fixture must not block the "
            f"distinctness gate, got {baseline.verdict.value}: {baseline.detail}"
        )

        def crash_instead(self, ctx_arg):
            return GateResult(
                gate_id=self.id,
                verdict=Verdict.ERROR,
                coverage=Coverage.none("units"),
                detail="RuntimeError: synthesized crash on injected metadata",
            )

        monkeypatch.setattr(probe.ExpertDistinctnessGate, "run", crash_instead)
        control = probe.run_alias_control(ctx, 2, baseline)
        assert control["status"] == "inconclusive"
        assert control["verdict"] == "ERROR"
        assert control["confounded"] is False

    def test_clean_baseline_and_fail_verdict_fires(self):
        """[PASSES-BEFORE — MUST_PASS for the attribution rule] A genuinely
        attributable detection must still be credited: if the patch fixed the
        false positive by minting a never-firing control, this fails — the
        doctrine-5 symmetric defect."""
        ctx = fixtures.healthy_sharded_moe_ctx(num_experts=8, num_layers=2)
        baseline = probe.ExpertDistinctnessGate().run(ctx)
        assert not baseline.blocking, (
            f"fixture drifted: healthy sharded fixture must not block, got "
            f"{baseline.verdict.value}: {baseline.detail}"
        )
        control = probe.run_alias_control(ctx, 2, baseline)
        assert control["status"] == "fired"
        assert control["confounded"] is False
        # And the evidence the causal claim rests on is printed with it:
        assert control["baseline_verdict"] == baseline.verdict.value
        assert control["inconclusive_reason"] == ""

    def test_quiet_detector_reports_not_fired(self, monkeypatch):
        """[PASSES-BEFORE — control for the not_fired leg] A gate that accepts
        injected aliasing is a true negative and must say so (verdict is not
        FAIL, but non-blocking). Exercises the pre-existing branch label."""
        ctx = fixtures.healthy_sharded_moe_ctx(num_experts=8, num_layers=2)
        baseline = probe.ExpertDistinctnessGate().run(ctx)

        def accept_everything(self, ctx_arg):
            return GateResult(
                gate_id=self.id,
                verdict=Verdict.PASS,
                coverage=Coverage(16, "expert tensors"),
                detail="synthesized acceptance of aliased experts",
            )

        monkeypatch.setattr(probe.ExpertDistinctnessGate, "run", accept_everything)
        control = probe.run_alias_control(ctx, 2, baseline)
        assert control["status"] == "not_fired"

    def test_dense_artifact_still_reports_skipped(self):
        """[PASSES-BEFORE — regression pin for the untouched early return] The
        inapplicable-control path is pre-existing and unchanged; pinning it
        guards against the patch having moved or weakened the 'inapplicable
        controls must say so' behaviour."""
        ctx = fixtures.empty_expert_set_ctx()
        control = probe.run_alias_control(ctx, 2, None)
        assert control["status"] == "skipped"
        assert "zero expert-shaped tensors" in control["reason"]


# ---------------------------------------------------------------------------
# Findings 2+6 at the _deliver seam — exit-code honesty on real measurements
# ---------------------------------------------------------------------------


class _PassGate(Gate):
    """One unit examined, no defect: keeps non-control results out of the way."""

    id = "test.hunt.pass_gate"
    description = "PASS over one examined unit"
    events = (Lifecycle.FIRST_SAVE,)

    def check(self, ctx):
        return self.ok("one unit examined, no defect", Coverage(1, "units"))

    def controls(self):
        return []  # controls are not exercised through _deliver


class _BlankSkipGate(Gate):
    """Returns Verdict.SKIP with a blank reason BY HAND — bypassing Gate.skip(),
    exactly the move the gate-contract docstring warns against. If the probe
    prints '(reason stated)' over this, the claim is the probe's own all([])."""

    id = "test.hunt.blank_skip_gate"
    description = "SKIP whose reason string is blank"
    events = (Lifecycle.FIRST_SAVE,)

    def check(self, ctx):
        return self._result(Verdict.SKIP, Coverage.none("units"), detail="   ")

    def controls(self):
        return []


class _StatedSkipGate(Gate):
    id = "test.hunt.stated_skip_gate"
    description = "SKIP with a real stated reason, via the helper"
    events = (Lifecycle.FIRST_SAVE,)

    def check(self, ctx):
        return self.skip("not applicable in this test context")

    def controls(self):
        return []


def _minimal_inventory() -> dict:
    return {
        "origin": "test://fixture",
        "format": "test",
        "entries_total": 0,
        "real_tensors": 0,
        "extra_state_blobs": 0,
        "metadata_implied_bytes": 0,
        "uncounted_unknown_dtype_tensors": 0,
        "dtypes": {},
        "with_storage_id": 0,
        "without_storage_id": 0,
    }


def _minimal_declared() -> dict:
    return {
        "num_experts": 8,
        "num_moe_layers": 2,
        "declared_fqns": None,
        "expected_expert_bytes": None,
        "basis": {
            "num_experts": "test",
            "num_moe_layers": "test",
            "declared_fqns": "test",
            "expected_expert_bytes": "test",
        },
        "notes": [],
    }


def _run_deliver(monkeypatch, gate_classes, *, inject_alias):
    monkeypatch.setattr(probe, "_CHECKPOINT_GATES", tuple(gate_classes))
    args = argparse.Namespace(inject_alias=inject_alias, json_out=None)
    return probe._deliver(
        args=args,
        ckpt_path=Path("/test/ckpt"),
        config_path=Path("/test/config.json"),
        config={},
        inventory=_minimal_inventory(),
        declared=_minimal_declared(),
        ctx=fixtures.healthy_sharded_moe_ctx(num_experts=8, num_layers=2),
    )


class TestDeliverHonesty:
    def test_blank_reason_skip_blocks_clear(self, monkeypatch, capsys):
        """[FAILS-BEFORE — Finding 6] Today a blank-detail SKIP renders under
        'SKIP (reason stated)' and the run returns EXIT_CLEAR: an UNSTATED
        abstention accepted as a STATED one. After the patch it prints under
        'reason MISSING' and enters blocking_reasons."""
        code = _run_deliver(monkeypatch, [_BlankSkipGate], inject_alias=None)
        out = capsys.readouterr().out
        assert code == probe.EXIT_BLOCKED
        assert "reason MISSING" in out
        # An `assert <anything> or True` stood here. It could not fail — the
        # vacuous shape this repository exists to refuse, shipped inside a test
        # whose subject is an abstention that was accepted without stating a
        # reason. Deleting it removes no coverage: the assertion below is the
        # one that actually pins the rendered reason, and it can fail.
        # The blocking reason itself names the gate:
        assert any(
            "blank_skip" in line or "SKIP with no stated reason" in line
            for line in out.splitlines()
        )

    def test_stated_reason_skip_stays_clear(self, monkeypatch, capsys):
        """[PASSES-BEFORE — MUST_PASS for Finding 6] A legitimately DECLARED
        abstention must remain CLEAR-eligible and render under the stated
        header; if the patch started blocking all skips this fails — again the
        doctrine-5 symmetric defect."""
        code = _run_deliver(monkeypatch, [_StatedSkipGate], inject_alias=None)
        out = capsys.readouterr().out
        assert code == probe.EXIT_CLEAR
        assert "SKIP (reason stated)    : 1" in out
        assert "reason MISSING" not in out

    def test_control_with_no_baseline_is_inconclusive_and_blocks(self, monkeypatch, capsys):
        """[FAILS-BEFORE — Finding 2 at the exit-code seam] With the distinctness
        gate absent from the sweep, no baseline exists: today the injected FAIL
        is credited as 'fired' (confounded=False) and the run can exit CLEAR.
        After the patch attribution is unobservable -> INCONCLUSIVE -> blocks.

        Depends on the real ExpertDistinctnessGate firing on sharded-group
        aliasing, which the gate's own shipped MUST_FIRE controls require."""
        code = _run_deliver(monkeypatch, [_PassGate], inject_alias=2)
        out = capsys.readouterr().out
        assert code == probe.EXIT_BLOCKED
        assert "INCONCLUSIVE" in out
