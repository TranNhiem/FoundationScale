"""Fail-before / pass-after tests for the three zero-work refusals added to
tools/mutate.py.

Finding 1: an all-control table exited 0 having fired zero MUST-FIRE mutants.
Finding 4: classify() scored a green run over ZERO executed tests as ALIVE.
Finding 5: the byte-for-byte restore verification was an `assert`, compiled
           out under python -O while the success line kept printing.

Polarity contract for this file, stated per the house rules: every MUST_FIRE
test FAILS on the current tree and PASSES after the patch, and says so in
its docstring. The MUST_PASS controls guard the new refusals against
over-firing; since the refusals do not exist on the current tree, a control
cannot over-fire there, so each control passes BEFORE and AFTER by
construction — its job is to redden the suite the day someone widens a
refusal (doctrine 3: the negative half of the detector). That polarity is
stated per test rather than hidden.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# tools/ is not a package; load the battery exactly as it lives on disk so the
# tests exercise the shipped file, never a copy. Registered in sys.modules so
# dataclass machinery resolves the module if anything introspects it.
_spec = importlib.util.spec_from_file_location("fs_mutate_under_test", ROOT / "tools" / "mutate.py")
mutate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mutate
_spec.loader.exec_module(mutate)

# Every scratch module below is this exact two-line file: the control anchor
# occurs once, the mutant anchor occurs once, and restore tests can compare
# against it byte-for-byte.
SCRATCH = "# touched by the battery\nMARKER = 1\n"

ATTRIBUTION = ("tests/test_marker.py::test_marker_rule",)


def _outcome(*, rc=0, passed=9, failed=(), errored=(), skipped=0, label="run"):
    """Build a SuiteOutcome directly; injected runners bypass run_suite itself."""
    return mutate.SuiteOutcome(
        rc=rc,
        passed=passed,
        failed=tuple(failed),
        errored=tuple(errored),
        skipped=skipped,
        duration_s=0.05,
        timed_out=False,
        junit_path=f"<fake>/{label}.xml",
    )


def _always_green(*, junit_dir, label, timeout_s=0, env=None):
    return _outcome(passed=9, label=label)


def _mutant_killer(mod_path):
    """Runner that is green except while the REAL mutant is on disk.

    The battery runs the suite with the mutant applied, so a runner reading
    the file observes the mutation state; two identical red runs earn a
    reproduced, attributed KILLED through the same confirm_kill path the
    shipped runner feeds.
    """

    def runner(*, junit_dir, label, timeout_s=0, env=None):
        if label == "baseline":
            return _outcome(passed=9, label=label)
        if "MARKER = 0" in mod_path.read_text("utf-8"):
            return _outcome(rc=1, passed=8, failed=ATTRIBUTION, label=label)
        return _outcome(passed=9, label=label)

    return runner


def _real_row():
    return {
        "name": "flip-marker",
        "what": "breaks the marker rule",
        "anchor": "MARKER = 1",
        "replacement": "MARKER = 0",
    }


def _control_row():
    return {
        "name": "inert-comment",
        "what": "comment-only edit; changes nothing",
        "anchor": "# touched by the battery",
        "replacement": "# touched by the battery (verified inert)",
        "must_survive": True,
    }


def _scratch(tmp_path):
    mod = tmp_path / "core.py"
    mod.write_text(SCRATCH, "utf-8")
    return mod


# --- Finding 1 ---------------------------------------------------------------


def test_all_control_table_exits_2_never_measured_not_0_over_0(tmp_path, capsys):
    """MUST_FIRE for the zero-MUST-FIRE guard.

    FAILS on the current tree: main() returns 0 (no SystemExit) after
    exercising only the control row. PASSES after the patch: die() raises
    SystemExit(2) at table validation, before the baseline run burns a suite.
    """
    mod = _scratch(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        mutate.main(
            [],
            table={"core": [_control_row()]},
            suite_runner=_always_green,
            module_paths={"core": mod},
        )
    assert excinfo.value.code == 2, (
        "an all-control run must be class 'never measured' (2), never class "
        "'suite has a gap' (1) and never the 0/0 success (0)"
    )
    assert "zero MUST-FIRE" in capsys.readouterr().err


def test_mixed_table_with_a_real_mutant_runs_kills_and_restores(tmp_path, capsys):
    """MUST_PASS control for the Finding 1 guard AND the Finding 5 rewrite.

    PASSES BEFORE and AFTER by construction (stated negative control): a
    table carrying one real mutant and one control is the healthy input;
    the new guard must not reject it, the kill must still be credited, the
    control must still be exercised, and the file must come back
    byte-for-byte.
    """
    mod = _scratch(tmp_path)
    rc = mutate.main(
        [],
        table={"core": [_real_row(), _control_row()]},
        suite_runner=_mutant_killer(mod),
        module_paths={"core": mod},
    )
    assert rc == 0
    assert mod.read_text("utf-8") == SCRATCH
    out = capsys.readouterr().out
    assert "MUST-PASS CONTROL: 1/1 exercised inert edit(s) survived" in out
    assert "1 module(s) restored byte-for-byte." in out


# --- Finding 4 ---------------------------------------------------------------


def test_zero_test_green_run_classifies_unscored_not_alive():
    """MUST_FIRE for the classify() zero-executed-tests guard.

    FAILS on the current tree: kind is ALIVE. PASSES after the patch: kind
    is UNSCORED and the reason names the zero denominator.
    """
    verdict = mutate.classify(_outcome(rc=0, passed=0))
    assert verdict.kind is mutate.TrialKind.UNSCORED
    assert "0 executed tests" in verdict.reason


def test_zero_test_green_trial_exits_2_not_1(tmp_path):
    """MUST_FIRE, integration: the ladder must price the zero-test green at
    exit 2, not exit 1.

    The table carries a MUST-PASS control row on purpose: without it the
    control-free-table refusal (already exit 2 on the current tree) would
    mask today's mispricing, and this test could not tell the two exits
    apart. FAILS on the current tree: the mutant row scores ALIVE over
    zero executed tests and main() returns 1. PASSES after the patch: the
    trial is UNSCORED and main() returns 2 while the control still passes.
    """
    mod = _scratch(tmp_path)

    def runner(*, junit_dir, label, timeout_s=0, env=None):
        if label == "baseline" or label.startswith("core-01"):
            return _outcome(passed=9, label=label)
        # The mutant row's trial: pytest "green" while the junit it
        # produced names zero executed tests.
        return _outcome(rc=0, passed=0, label=label)

    rc = mutate.main(
        [],
        table={"core": [_real_row(), _control_row()]},
        suite_runner=runner,
        module_paths={"core": mod},
    )
    assert rc == 2


def test_healthy_green_run_still_classifies_alive():
    """MUST_PASS control for the classify() guard.

    PASSES BEFORE and AFTER by construction: a green run that executed
    tests is the healthy input; the guard must not touch it.
    """
    assert mutate.classify(_outcome(rc=0, passed=7)).kind is mutate.TrialKind.ALIVE


def test_sibling_refusals_unchanged():
    """Guard against the edit disturbing neighbouring branches.

    PASSES BEFORE and AFTER: skip-contaminated greens and bare rc=1 runs
    keep their existing UNSCORED verdicts; the guard adds a floor, it does
    not move the walls.
    """
    assert mutate.classify(_outcome(rc=0, passed=7, skipped=1)).kind is mutate.TrialKind.UNSCORED
    assert mutate.classify(_outcome(rc=1, passed=7)).kind is mutate.TrialKind.UNSCORED


# --- Finding 5 ---------------------------------------------------------------


def test_restore_verification_is_real_code_not_an_assert(tmp_path, monkeypatch):
    """MUST_FIRE for the restore read-back being ordinary control flow.

    FAILS on the current tree two different ways, one per interpreter
    flag: under a normal interpreter the bare assert fires AssertionError
    (not RuntimeError), and under `python -O` the assert is compiled out
    and nothing raises at all while the success line prints. PASSES after
    the patch under BOTH flags: the if/raise is invisible to -O.
    """
    mod = _scratch(tmp_path)
    real_read_text = Path.read_text
    reads = {"n": 0}

    def lying_read_text(self, *args, **kwargs):
        if self == mod:
            reads["n"] += 1
            if reads["n"] >= 2:
                # The post-restore verification read: the filesystem
                # "returns" a tree that still carries the mutant.
                return "# mutant still on disk\n"
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", lying_read_text)
    with pytest.raises(RuntimeError) as excinfo:
        mutate.main(
            [],
            table={"core": [_real_row()]},
            suite_runner=_always_green,
            module_paths={"core": mod},
        )
    assert str(mod) in str(excinfo.value)
    assert "version control" in str(excinfo.value)
