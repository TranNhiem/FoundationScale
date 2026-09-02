"""fix24: the control layer certifying something other than what it claimed.

Three costumes of one defect, each pinned here:

* F1 — the CI controls harness certified 9 of 10 registered gates and printed
  all-clear, because its import walk stopped at the ``foundationscale.gates``
  package boundary while ``WeightParityGate`` registered from
  ``foundationscale.verify.parity``.
* F2 — ``verify_controls`` accepted a MUST_PASS control that ABSTAINED,
  because "not blocking" was the whole test of healthy-input behaviour.
* F3 — ``ExpertAliasGate`` had no verified behaviour in either direction on a
  positively declared dense model (declared 0 experts, 0 expert tensors).

Fail-before mechanics: each test below states, in its docstring, exactly what
was missing or wrong on the pre-fix tree that made it red. Several F2/F3 tests
are red on the old tree via ``AttributeError``/``TypeError`` — the vocabulary
they assert did not exist — which is stated per test rather than dressed up as
a behavioural red.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import foundationscale.gates.checkpoint_gates  # noqa: F401 -- registers the checkpoint gates
from foundationscale.gates import controls as controls_harness
from foundationscale.gates.core import (
    REGISTRY,
    AbstentionKind,
    Control,
    ControlKind,
    Coverage,
    Gate,
    GateRegistry,
    GateResult,
    Lifecycle,
    Verdict,
    verify_controls,
)
from foundationscale.gates.example import ExpertAliasGate, ExpertCheckContext


class _ProbeGate(Gate):
    """A gate whose verdict per context name is scripted by the test.

    MUST_FIRE input ``"defective"`` always blocks (so its MUST_FIRE control
    always holds and cannot pollute the MUST_PASS assertions); named healthy
    inputs either PASS or SKIP as the test declares. Registered only into
    per-test local registries — never into the process-global REGISTRY, which
    would leak across tests.
    """

    id = "test.probe"
    description = "scripted-verdict probe for the verify_controls certification rules"
    events = (Lifecycle.STEP_ZERO,)
    _MUST_PASS_CTXS = ("healthy", "stacked")

    def __init__(self, *, skip: frozenset[str], expect_skip: dict[str, str]) -> None:
        self._skip = skip
        self._expect_skip = expect_skip

    def check(self, ctx: str) -> GateResult:
        if ctx in self._skip:
            return self.skip(f"{ctx} is outside this probe's adjudicable scope")
        coverage = Coverage(1, "probe inputs", expected=1)
        if ctx == "defective":
            return self.fail("the injected defect fired", coverage)
        return self.ok(f"{ctx} adjudicated clean", coverage)

    def controls(self) -> list[Control]:
        out: list[Control] = [
            Control(
                "defective",
                ControlKind.MUST_FIRE,
                lambda: "defective",
                note="injected defect; must block",
            )
        ]
        for name in self._MUST_PASS_CTXS:
            kwargs: dict[str, str] = {}
            reason = self._expect_skip.get(name, "")
            if reason:
                # Omitting the keyword when there is no declaration keeps the
                # pre-fix tree's red honest for the unexpected-SKIP test: its
                # controls build fine on the old dataclass, so its redness
                # comes from SKIP being silently accepted, not from a TypeError.
                kwargs["expect_skip"] = reason
            out.append(
                Control(
                    name,
                    ControlKind.MUST_PASS,
                    lambda n=name: n,
                    note="known-healthy probe input",
                    **kwargs,
                )
            )
        return out


# ---------------------------------------------------------------------------
# F1 — the walk boundary
# ---------------------------------------------------------------------------


def test_gate_walk_reaches_sibling_gate_packages() -> None:
    """The walk must import foundationscale.verify.parity and register its gate.

    Red before: the walk covered ``foundationscale.gates`` alone, so
    ``imported`` held 5 modules, none of them parity; on a cold process
    ``REGISTRY`` then held 9 gates, not 10. Green after: the walk covers both
    declared roots. (If an earlier test already imported parity directly, the
    REGISTRY assertion could pass on the old tree — the ``imported``
    assertion cannot, because it reflects this call's own walk.)
    """
    imported, errors = controls_harness._import_gate_modules()
    assert errors == []
    assert "foundationscale.verify.parity" in imported
    assert "checkpoint.weight_parity" in REGISTRY


def test_provenance_reconciliation_names_gate_from_unwalked_module() -> None:
    """A registered gate from a module the walk never attempted is a finding.

    Red before: ``_uncertified_provenance_findings`` did not exist
    (AttributeError). Green after: defined, and names the gate id and module.
    """
    probe = _ProbeGate(skip=frozenset(), expect_skip={})
    registry = GateRegistry()
    registry.register(probe)
    findings = controls_harness._uncertified_provenance_findings(
        registry, {"foundationscale.gates.core"}
    )
    assert len(findings) == 1
    assert "test.probe" in findings[0]
    assert "never reached" in findings[0]
    assert (
        controls_harness._uncertified_provenance_findings(registry, {type(probe).__module__}) == []
    )


def test_package_census_classifies_known_first_party_packages() -> None:
    """The census knows gates/verify as walked and checkpoint/provenance as gate-free.

    Red before: ``_unclassified_package_findings`` did not exist. Green after:
    the four known packages produce no findings. Any OTHER first-party package
    on disk is intentionally not asserted away — the census's job is to name
    it loudly, and this test must not launder an unclassified package into a
    pass.
    """
    findings = controls_harness._unclassified_package_findings()
    known = (
        "foundationscale.gates",
        "foundationscale.verify",
        "foundationscale.checkpoint",
        "foundationscale.provenance",
    )
    for package in known:
        assert not any(f.startswith(f"{package}:") for f in findings)


def test_package_census_names_an_unclassified_first_party_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUST_FIRE for the census tripwire (mutation row
    ``controls/package-census-silenced``).

    Red under the mutant: ``if False and info.ispkg ...`` can never fire, the
    census returns [] over a genuinely unclassified package sitting on the
    walk's own root path, and the arithmetic assert below fails. Green on the
    shipped line. The pre-existing test
    (test_package_census_classifies_known_first_party_packages) ever asserted
    only ABSENCE over an already-classified tree — a census that fires on
    nothing satisfies it forever, which is exactly the hole the mutator found.

    Fixture state that discriminates (the both-readings-agree trap): the probe
    must be a REAL package on a path the census really walks — planted under
    ``foundationscale.__path__`` with an ``__init__.py`` — and on NEITHER
    list. Denominator (doctrine 2): exactly one finding, naming the probe.
    The baseline tree is classified-clean (867 green guarantees it, else
    TestLiveRegistryRun is red), so N=1 simultaneously pins the tripwire and
    re-pins that no OTHER unclassified sibling has landed. The co-planted
    plain module must never be named: the census counts package boundaries,
    not files, and naming a file would be the over-fire direction of the same
    defect (doctrine 5 is symmetric).
    """
    import foundationscale as root_pkg

    probe_pkg = tmp_path / "fscensus_probe_unclassified"
    probe_pkg.mkdir()
    (probe_pkg / "__init__.py").write_text(
        '"""Only a census probe: an unclassified first-party sibling."""\n'
    )
    (tmp_path / "fscensus_probe_module.py").write_text(
        '"""A plain module: NOT a package boundary; never census-named."""\n'
    )
    # One path element appended, monkeypatch-restored: this test does not lean
    # on the shipped tree's layout, it EXTENDS it with exactly one
    # unclassified package — and asks the shipped function what it sees.
    monkeypatch.setattr(root_pkg, "__path__", [*root_pkg.__path__, str(tmp_path)])
    findings = controls_harness._unclassified_package_findings()
    named = [f for f in findings if "foundationscale.fscensus_probe_unclassified" in f]
    assert len(findings) == 1 and named == findings, (
        "the census must find exactly the planted unclassified package and "
        f"nothing else on the shipped tree, got {findings!r} — an empty list "
        "here is the silenced tripwire, and anything more is a second "
        "unclassified package this suite did not plant"
    )
    assert "on nobody's list" in named[0]
    assert not any("fscensus_probe_module" in f for f in findings)


def test_package_census_stays_quiet_for_a_reviewed_gate_free_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[GREEN-ON-BOTH-TREES — over-fire fence, stated per the in-file
    convention rather than dressed as a kill]

    The identical probe construction, with the sibling monkeypatched ONTO the
    reviewed gate-free list: the census must stay silent. It also passes
    under the mutant (a census that NEVER fires is silent here as well) —
    which is precisely why this fence cannot kill the row and is paired with
    the MUST_FIRE twin above (doctrine 3: one kind of control is half a
    proof). Red-maker: a hair-trigger rewrite that drops the
    ``not in classified`` half of the condition and cries wolf over packages
    review already absolved. Denominator: zero findings permitted, and the
    probe is genuinely on disk, so the silence is an examined silence rather
    than an empty walk.
    """
    import foundationscale as root_pkg

    probe_pkg = tmp_path / "fscensus_probe_reviewed"
    probe_pkg.mkdir()
    (probe_pkg / "__init__.py").write_text('"""Reviewed as gate-free."""\n')
    monkeypatch.setattr(root_pkg, "__path__", [*root_pkg.__path__, str(tmp_path)])
    monkeypatch.setattr(
        controls_harness,
        "_KNOWN_GATELESS_PACKAGES",
        controls_harness._KNOWN_GATELESS_PACKAGES | {"foundationscale.fscensus_probe_reviewed"},
    )
    assert controls_harness._unclassified_package_findings() == []


def test_main_reaches_and_lists_weight_parity(capsys: pytest.CaptureFixture[str]) -> None:
    """End-to-end: main()'s output names checkpoint.weight_parity and the receipt.

    Red before: main() walked only the gates package, so
    ``checkpoint.weight_parity`` appeared nowhere in its output and the
    ``controls executed for`` receipt line did not exist — while exiting 0.
    Green after: the gate appears in the listing (with its declared
    context_type) and the certification denominator is printed. The exit code
    is deliberately NOT asserted: this test runs in a shared pytest process,
    and asserting rc==0 here would couple the test to global-registry hygiene
    of other test modules; the console-script run in CI is the certification,
    and it fails loudly on its own if anything in this test stays red.
    """
    controls_harness.main()
    out = capsys.readouterr().out
    assert "checkpoint.weight_parity — context_type: ParityGateContext" in out
    assert "controls executed for" in out


# ---------------------------------------------------------------------------
# F2 — declared abstentions, checked in both directions, plus the PASS floor
# ---------------------------------------------------------------------------


def test_unexpected_abstention_on_must_pass_fails() -> None:
    """A MUST_PASS control that SKIPs with no declaration is a failure.

    Red before: SKIP is non-blocking, so the old `not result.blocking` test
    accepted it and verify_controls returned []. Green after: the failure
    names the control and the missing declaration. The probe passes no
    expect_skip kwargs here, so the red mechanism really is acceptance, not
    a dataclass TypeError.
    """
    registry = GateRegistry()
    registry.register(_ProbeGate(skip=frozenset({"stacked"}), expect_skip={}))
    failures = verify_controls(registry)
    assert any("abstained" in f and "expect_skip" in f for f in failures)


def test_declared_abstention_plus_one_real_pass_certifies_clean() -> None:
    """A declared, reasoned abstention is accepted when another MUST_PASS affirms.

    Red before: the ``expect_skip`` keyword did not exist (TypeError inside
    controls(), surfaced as a "controls() raised" failure — red by absence of
    the vocabulary). Green after: failures == [].
    """
    registry = GateRegistry()
    registry.register(
        _ProbeGate(
            skip=frozenset({"stacked"}),
            expect_skip={
                "stacked": "per-expert identity inside a stacked tensor is "
                "metadata-invisible, as with ExpertDistinctnessGate"
            },
        )
    )
    assert verify_controls(registry) == []


def test_stale_expect_skip_fails_in_the_symmetric_direction() -> None:
    """Declaring expect_skip on a fixture the gate now PASSes is a failure.

    Red before: the keyword did not exist (TypeError), and the old code had no
    stale-declaration direction at all. Green after: named failure containing
    "stale"; and because "stacked" still genuinely passed, no zero-PASS guard
    failure accompanies it.
    """
    registry = GateRegistry()
    registry.register(
        _ProbeGate(skip=frozenset(), expect_skip={"healthy": "claimed unadjudicable"})
    )
    failures = verify_controls(registry)
    assert any("stale" in f for f in failures)
    assert not any("0 of" in f and "MUST_PASS" in f for f in failures)


def test_all_abstaining_must_pass_set_fails_even_fully_declared() -> None:
    """Every abstention honest and declared, zero affirmations: still a failure.

    This is the structural exposure measured in the finding — had
    healthy-sharded regressed to SKIP, ExpertDistinctnessGate would have
    certified green over zero healthy-input affirmations. Red before: the
    keyword did not exist AND no guard existed (old code returned []). Green
    after: the failure names the 0-of-2 numerator.
    """
    registry = GateRegistry()
    registry.register(
        _ProbeGate(
            skip=frozenset({"healthy", "stacked"}),
            expect_skip={
                "healthy": "declared unadjudicable for this test",
                "stacked": "declared unadjudicable for this test",
            },
        )
    )
    failures = verify_controls(registry)
    assert any("0 of 2 MUST_PASS" in f for f in failures)


def test_expect_skip_is_refused_on_must_fire_and_when_blank() -> None:
    """The declaration vocabulary cannot purchase an exemption on a positive control.

    Red before: the ``expect_skip`` parameter did not exist, so both
    constructions raised TypeError rather than the pinned ValueErrors. Green
    after: refused at construction, with the reasons in the messages.
    """
    with pytest.raises(ValueError, match="illegal on MUST_FIRE"):
        Control("x", ControlKind.MUST_FIRE, lambda: None, expect_skip="a reason")
    with pytest.raises(ValueError, match="must carry the reason"):
        Control("y", ControlKind.MUST_PASS, lambda: None, expect_skip="   ")


def test_distinctness_stacked_controls_are_labelled_not_exempted() -> None:
    """The two real SKIP-ing controls declare their abstentions and stay charged.

    Red before: ``Control.expect_skip`` did not exist (AttributeError). Green
    after: both controls declare a reason, both still abstain SKIP with kind
    NOT_ESTABLISHED — charged against every denominator, not exempted — and
    healthy-sharded still supplies the gate's affirmative PASS.
    """
    gate = REGISTRY.get("checkpoint.expert_distinctness")
    controls = {c.name: c for c in gate.controls()}
    for name in ("stacked-clean", "healthy-fused"):
        assert controls[name].expect_skip.strip()
        result = gate.run(controls[name].make_ctx())
        assert result.verdict is Verdict.SKIP
        assert result.abstention is AbstentionKind.NOT_ESTABLISHED
    healthy = gate.run(controls["healthy-sharded"].make_ctx())
    assert healthy.verdict is Verdict.PASS


# ---------------------------------------------------------------------------
# F3 — the dense door on ExpertAliasGate
# ---------------------------------------------------------------------------


def test_expert_alias_gate_abstains_declared_on_dense_model() -> None:
    """Declared 0 experts, 0 expert tensors: a declared NOT_APPLICABLE abstention.

    Red before: the `if not by_expert:` branch returned ok() over zero
    coverage, which the contract downgrades to blocking VACUOUS — a false
    failure for a legitimately dense artifact. Green after: SKIP with the
    machine-readable kind, non-blocking, and the word "dense" in the reason.
    """
    gate = ExpertAliasGate()
    result = gate.run(ExpertCheckContext(tensors={}, expert_index={}, declared_expert_count=0))
    assert result.verdict is Verdict.SKIP
    assert not result.blocking
    assert result.abstention is AbstentionKind.NOT_APPLICABLE
    assert "dense" in result.detail


def test_expert_alias_gate_dense_control_exists_and_declares() -> None:
    """The dense door ships controls; its MUST_PASS declares the abstention.

    Red before: no control named ``dense-model`` existed (KeyError). Green
    after: declared expect_skip, and the fixture round-trips to SKIP.
    """
    gate = ExpertAliasGate()
    controls = {c.name: c for c in gate.controls()}
    ctrl = controls["dense-model"]
    assert ctrl.kind is ControlKind.MUST_PASS
    assert ctrl.expect_skip.strip()
    assert gate.run(ctrl.make_ctx()).verdict is Verdict.SKIP


def test_expert_alias_gate_refuses_the_boolean_zero() -> None:
    """``False`` in the declared count is a malformed denominator, not a dense claim.

    Red before: no type guard existed; ``False`` fell into the `not by_expert`
    branch and the verdict blamed unresolvable indices, so the detail string
    never named the malformed denominator. Green after: VACUOUS (blocking,
    never the dense SKIP) and the detail names the offense.
    """
    gate = ExpertAliasGate()
    result = gate.run(ExpertCheckContext(tensors={}, expert_index={}, declared_expert_count=False))
    assert result.verdict is Verdict.VACUOUS
    assert result.blocking
    assert "non-negative integer" in result.detail
    controls = {c.name: c for c in gate.controls()}
    assert controls["malformed-dense-count-bool"].kind is ControlKind.MUST_FIRE
    assert gate.run(controls["malformed-dense-count-bool"].make_ctx()).blocking
