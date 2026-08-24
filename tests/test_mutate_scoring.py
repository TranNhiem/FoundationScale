"""Unit pins for the tools/mutate.py scoring contract.

Every early test fails against the pre-fix battery: SuiteOutcome /
TrialVerdict / TrialKind / classify / confirm_kill / kill_candidate did not
exist, and the `if green ... else killed` branch mapped rc=2, bare rc=1 and
hangs to `[killed]`. The CLI-level pins cover: run_suite must arm
FS_FORBID_SKIPS itself (merged last so no caller can disarm it), bound every
run with a timeout, and main must let any UNSCORED outrank even survivors.

The final block pins the MUST-PASS control (doctrine 3 — a detector is two
claims) and the table seam the inert-control incident forced: a comment-only
mutant once scored [killed] because the meta-tests were coupled to the live
mutations.json. main() now takes table=/module_paths=; control rows report
on their own line, never count in caught=, and a control reported killed
voids the run with exit 2.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("fs_mutate", _REPO / "tools" / "mutate.py")
mutate = importlib.util.module_from_spec(_spec)
# Register before executing: @dataclass resolves string annotations through
# sys.modules[cls.__module__], which is None for a module that was created but
# never registered, and the decorator dies on the first frozen dataclass.
sys.modules[_spec.name] = mutate
_spec.loader.exec_module(mutate)


def _out(**kw):
    base = dict(
        rc=1,
        passed=200,
        failed=("tests/t.py::test_x",),
        errored=(),
        skipped=0,
        duration_s=1.0,
        timed_out=False,
        junit_path="j.xml",
    )
    return mutate.SuiteOutcome(**{**base, **kw})


def test_collection_error_is_unscored_never_killed():
    v = mutate.confirm_kill(
        _out(rc=2, failed=(), errored=("<collection>",)),
        _out(rc=2, failed=()),
    )
    assert v.kind is mutate.TrialKind.UNSCORED
    assert "rc=2" in v.reason  # pre-fix this same trace printed [killed]


def test_zero_attribution_rc1_is_unscored():
    v = mutate.classify(_out(failed=()))
    assert v.kind is mutate.TrialKind.UNSCORED
    assert v.reason.startswith("rc=1 with zero attributed failures")


def test_reproduced_kill_carries_attribution():
    v = mutate.confirm_kill(_out(), _out())
    assert v.kind is mutate.TrialKind.KILLED
    assert v.attribution == ("tests/t.py::test_x",)


def test_flaky_attribution_never_promotes():
    v = mutate.confirm_kill(_out(failed=("a::t",)), _out(failed=("b::t",)))
    assert v.kind is mutate.TrialKind.UNSCORED
    assert "not reproducible" in v.reason


def test_timeout_is_unscored():
    v = mutate.classify(_out(rc=-9, timed_out=True, failed=()))
    assert v.kind is mutate.TrialKind.UNSCORED


def test_green_with_skips_is_unscored():
    """A skip-poisoned green is never a survivor verdict."""
    v = mutate.classify(_out(rc=0, failed=(), passed=100, skipped=3))
    assert v.kind is mutate.TrialKind.UNSCORED
    assert "skip" in v.reason.lower()


def test_failure_mixed_with_error_is_unscored():
    """rc=1 with errors is contamination: the suite broke off-target."""
    v = mutate.classify(_out(errored=("tests/x.py::test_setup",)))
    assert v.kind is mutate.TrialKind.UNSCORED
    assert "error" in v.reason.lower()


def test_clean_green_is_alive():
    v = mutate.classify(_out(rc=0, failed=(), passed=211))
    assert v.kind is mutate.TrialKind.ALIVE


def test_classify_never_returns_killed():
    """Structural invariant: only confirm_kill can promote to KILLED."""
    outcomes = [
        _out(rc=0, failed=()),
        _out(rc=0, failed=(), skipped=1),
        _out(rc=1),
        _out(rc=1, failed=()),
        _out(rc=1, errored=("e",)),
        _out(rc=2, failed=()),
        _out(rc=-15, failed=()),
        _out(rc=-9, timed_out=True, failed=()),
    ]
    for out in outcomes:
        assert mutate.classify(out).kind is not mutate.TrialKind.KILLED, out


def test_error_contamination_blocks_candidate():
    """An attributed rc=1 with errors is not even a kill candidate."""
    out = _out(errored=("tests/x.py::test_setup",))
    assert not mutate.kill_candidate(out)
    v = mutate.confirm_kill(out, _out())
    assert v.kind is mutate.TrialKind.UNSCORED


def test_run_suite_exports_forbid_skips_and_bounds_time(monkeypatch, tmp_path):
    captured = {}

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured.update(kw)
        return _Proc()

    monkeypatch.setattr(mutate.subprocess, "run", fake_run)
    mutate.run_suite(junit_dir=tmp_path, label="pin", timeout_s=7)
    assert captured["timeout"] == 7
    assert captured["env"]["FS_FORBID_SKIPS"] == "1"
    assert any(
        str(a).startswith("--junitxml=") and str(tmp_path / "suite-pin.xml") in str(a)
        for a in captured["cmd"]
    )


def test_run_suite_arming_cannot_be_disabled_by_caller(monkeypatch, tmp_path):
    captured = {}

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kw):
        captured.update(kw)
        return _Proc()

    monkeypatch.setattr(mutate.subprocess, "run", fake_run)
    mutate.run_suite(junit_dir=tmp_path, label="pin", env={"FS_FORBID_SKIPS": "0", "EXTRA": "1"})
    assert captured["env"]["FS_FORBID_SKIPS"] == "1"  # arming merges last
    assert captured["env"]["EXTRA"] == "1"


def test_run_suite_timeout_marks_outcome(monkeypatch, tmp_path):
    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=kw.get("timeout", 0))

    monkeypatch.setattr(mutate.subprocess, "run", boom)
    out = mutate.run_suite(junit_dir=tmp_path, label="hang", timeout_s=1)
    assert out.timed_out
    assert out.rc == -9
    assert mutate.classify(out).kind is mutate.TrialKind.UNSCORED


# --- CLI-level pins: fixtures anchor at tmp_path, never the live tree ---------


_GUARD_ONE = {
    "name": "break-guard-one",
    "what": "the negative guard stops rejecting",
    "anchor": "if value < 0:",
    "replacement": "if False and value < 0:",
}

_GUARD_TWO = {
    "name": "break-guard-two",
    "what": "the upper guard stops rejecting",
    "anchor": "if value > 1:",
    "replacement": "if False and value > 1:",
}

_CONTROL = {
    "name": "inert-control",
    "what": "MUST SURVIVE: a comment-only edit; if this dies, attribution is unsound",
    "anchor": "if value < 0:",
    "replacement": "if value < 0:  # inert must-pass control",
    "must_survive": True,
}


def _fixture(tmp_path: Path, mutations: list[dict]):
    """A mutation table and module map anchored at a scratch file under tmp_path.

    WHY: before the table seam, main() always read tools/mutations.json
    against the live tree, so an outer battery mutant made these meta-tests
    fail deterministically and the battery banked that failure as a kill.
    """
    target = tmp_path / "scratch_guards.py"
    target.write_text(
        "def guarded(value):\n"
        "    if value < 0:  # scratch anchor one\n"
        "        raise ValueError(value)\n"
        "    if value > 1:  # scratch anchor two\n"
        "        raise ValueError(value)\n"
        "    return value\n",
        "utf-8",
    )
    return {"scratch": mutations}, {"scratch": target}


def test_unscored_outranks_survivor_exit(tmp_path, capsys):
    """Alive + unscored in one run must exit 2, never 1 and never 0."""
    table, paths = _fixture(tmp_path, [dict(_GUARD_ONE), dict(_GUARD_TWO)])
    state = {"trials": 0}

    def fake(**kw):
        if kw["label"] == "baseline":
            return _out(rc=0, failed=(), passed=400)
        state["trials"] += 1
        if state["trials"] % 2 == 1:
            return _out(rc=0, failed=(), passed=400)  # mutant survives
        return _out(rc=2, failed=())  # trial never measured

    rc = mutate.main((), suite_runner=fake, table=table, module_paths=paths)
    out = capsys.readouterr().out
    assert rc == 2
    assert "[ALIVE]" in out
    assert "[UNSCORED]" in out


def test_skip_poisoned_green_trial_is_unscored(tmp_path, capsys):
    """A green trial containing skips is not allowed to print [ALIVE]."""
    table, paths = _fixture(tmp_path, [dict(_GUARD_ONE)])

    def fake(**kw):
        if kw["label"] == "baseline":
            return _out(rc=0, failed=(), passed=400)
        return _out(rc=0, failed=(), passed=399, skipped=1)

    rc = mutate.main((), suite_runner=fake, table=table, module_paths=paths)
    out = capsys.readouterr().out
    assert rc == 2
    assert "[ALIVE]" not in out
    assert "[UNSCORED]" in out


def test_killed_must_survive_control_voids_the_run(tmp_path, capsys):
    """A dead inert control proves attribution unsound: exit 2, naming the control.

    WHY: this is the exact failure the battery shipped — an inert mutant
    banked as a kill. The control turns 'the battery fires no matter what'
    into a blocking failure instead of a green headline.

    The table pairs the control with one real MUST-FIRE mutant because a
    control-only table became a confounded double the day P3 landed:
    _validate_table() now refuses a selection of ZERO MUST-FIRE rows with
    exit 2 before baseline, so the old fixture died at validation and never
    reached the behaviour its name claims. The added mutant cannot
    manufacture the verdict under test: a KILLED mutant is the neutral,
    exit-0-compatible outcome and touches no rung of the exit ladder, so
    in this terminal state (0 unscored, 0 n/a, 1 control configured,
    0 alive) exactly one exit-2 branch of main() is reachable — `if
    ctrl_dead:` — and the assertions below name that branch's own text so
    the attribution is proven, not assumed.
    """
    table, paths = _fixture(tmp_path, [dict(_GUARD_ONE), dict(_CONTROL)])

    def runner(**kw):
        if kw["label"] == "baseline":
            return _out(rc=0, failed=(), passed=400)
        # One red for every trial, blind to which row is on disk. The
        # fixture deliberately does not learn which row is the control:
        # routing the same reproducible red to [killed] for one row and to
        # [CTRL DEAD] for the other is main()'s behaviour under test, not
        # a branch the runner may take on main()'s behalf.
        return _out()  # an attributed, reproducible red: a KILLED verdict

    rc = mutate.main((), suite_runner=runner, table=table, module_paths=paths)
    out = capsys.readouterr().out
    assert rc == 2
    assert "[CTRL DEAD]" in out
    assert "inert-control" in out
    assert "MUST-PASS" in out
    # The real mutant took the neutral path — killed mutants never void a
    # run — so the exit 2 above cannot have been bought by anything the
    # added row did.
    assert "[killed]" in out
    assert "break-guard-one" in out
    # Attribution proof: this sentence is printed by exactly one branch of
    # main() — the ctrl_dead branch — beside its return. If exit 2 ever
    # starts arriving by another route (unscored trial, zero controls), one
    # of these two strings goes silent and this test reddens.
    assert "1 MUST-PASS control(s) reported killed" in out
    assert "attribution is proven unsound" in out
    # The control was exercised and died: zero of one survived, one
    # configured — a returned fact, stated in words (doctrine 2)...
    assert "MUST-PASS CONTROL: 0/1" in out
    # ...and a dead control suppresses the module's percentage: caught=
    # over a module whose attribution just fired with no fault behind it
    # would be arithmetic, not evidence, so main() prints the refusal.
    assert "caught=--" in out


def test_surviving_must_survive_control_is_not_a_survivor(tmp_path, capsys):
    """The control passing must not put the battery into exit-1 (survivor) shape.

    The table pairs the control with one real MUST-FIRE mutant because a
    control-only table became a confounded double the day P3 landed:
    _validate_table() refuses a selection of ZERO MUST-FIRE rows, so the
    old fixture died at validation and never measured what its name claims.
    The added mutant must be KILLED, not left alive — a survivor would flip
    the run to exit 1 and print [ALIVE] for a reason that has nothing to do
    with the control, re-confounding the test in the opposite direction
    (doctrine 5's symmetry: a failure never caused by the thing named is as
    false as a pass never earned). The runner discriminates on `label`, the
    per-trial argument score_trial() hands every SuiteRunner. That is
    honest, not rigged: the shipped run_suite() behaves differently per
    label too, because main() swaps which mutant sits on disk before each
    call — behaviour keyed on "which mutation is applied" IS the contract,
    and the fake encodes ground truth about this table (a disabled guard
    has coverage; a comment does not). It still has to earn the KILLED:
    anything short of identical attribution on both confirmation runs
    demotes the trial to UNSCORED, and `rc == 0` below dies.
    """
    table, paths = _fixture(tmp_path, [dict(_GUARD_ONE), dict(_CONTROL)])

    def runner(**kw):
        if kw["label"] == "baseline":
            return _out(rc=0, failed=(), passed=400)
        if kw["label"].startswith("scratch-00"):
            # The real mutant: an attributed red, identical across both
            # confirmation runs, so confirm_kill promotes it to KILLED.
            return _out()
        return _out(rc=0, failed=(), passed=400)  # inert control: stays green

    rc = mutate.main((), suite_runner=runner, table=table, module_paths=paths)
    out = capsys.readouterr().out
    assert rc == 0
    assert "[ALIVE]" not in out
    # doctrine 2: the control's outcome is a returned fact, stated on its own
    # line in the summary even when everything passed — never by omission.
    assert "MUST-PASS CONTROL: 1/1" in out
    # The MUST-FIRE half of this table really fired and was really caught:
    # with the mutant anything but KILLED, the run cannot read rc == 0 with
    # no [ALIVE] line — so this pair pins WHICH verdict the green run
    # rests on, instead of letting any green run satisfy the name.
    assert "[killed]" in out
    assert "break-guard-one" in out


def test_control_rows_excluded_from_tally_and_denominator(tmp_path, capsys):
    """caught= and the killed/alive counts measure real mutants only.

    WHY: one real mutant killed plus one control passed is 1 of 1 scored,
    not 1 of 2 — otherwise a passing control deflates the denominator and a
    failed one could hide inside a percentage.
    """
    table, paths = _fixture(tmp_path, [dict(_GUARD_ONE), dict(_CONTROL)])

    def runner(**kw):
        if kw["label"] == "baseline":
            return _out(rc=0, failed=(), passed=400)
        if kw["label"].startswith("scratch-00"):
            return _out()  # real mutant: killed on two identical runs
        return _out(rc=0, failed=(), passed=400)  # control: suite stays green

    rc = mutate.main((), suite_runner=runner, table=table, module_paths=paths)
    out = capsys.readouterr().out
    assert rc == 0
    assert "caught=100%" in out  # denominator is 1 real mutant, not 2 rows
    tally_rows = [
        line.split()
        for line in out.splitlines()
        if "killed" in line and "alive" in line and "caught=" in line
    ]
    assert tally_rows, "per-module tally row missing"
    for parts in tally_rows:
        assert int(parts[parts.index("killed") - 1]) == 1  # control not counted
        assert int(parts[parts.index("alive") - 1]) == 0  # control pass is not a survivor


def test_injected_table_never_reads_live_table(tmp_path, capsys, monkeypatch):
    """main() must reach a verdict with tools/mutations.json absent on disk.

    Asserts something impossible today: before the seam, load_table() died
    at startup on a missing table, so no run could succeed with TABLE
    pointed at a nonexistent path. The injected table short-circuits the
    disk entirely.
    """
    monkeypatch.setattr(mutate, "TABLE", tmp_path / "no-such-table.json")
    # The injected table now carries a surviving control: a zero-control run
    # exits 2 by design (half a detector does not get to claim its catches),
    # and this test pins independence from the on-disk table, not the
    # zero-control verdict. The assertion below is unchanged.
    table, paths = _fixture(tmp_path, [dict(_GUARD_ONE), dict(_CONTROL)])

    def runner(**kw):
        if kw["label"] == "baseline":
            return _out(rc=0, failed=(), passed=400)
        if kw["label"].startswith("scratch-01"):
            return _out(rc=0, failed=(), passed=400)  # inert control: stays green
        return _out()

    rc = mutate.main((), suite_runner=runner, table=table, module_paths=paths)
    assert rc == 0


# --- zero-control pins: the control count is a returned fact, every run -------

_STALE_CONTROL = {
    "name": "inert-control-stale-anchor",
    "what": "MUST SURVIVE: comment-only edit whose anchor no longer matches the source",
    "anchor": "if value < -99:",
    "replacement": "if value < -99:  # inert must-pass control",
    "must_survive": True,
}


def test_all_killed_with_zero_controls_still_blocks(tmp_path, capsys):
    """Every real mutant killed, zero control rows: the run must not exit 0.

    WHY: delete this and the summary can again print the control result only
    `if ctrl_ok or ctrl_dead` — a zero-control run then reads exactly like a
    fully controlled one: killed / 0 alive / caught=100% resting on nothing.
    That is doctrine 2 inverted inside the tool built to reject it: without a
    MUST-PASS control the run cannot tell "the suite detected the fault" from
    "the suite reports detection whether or not there is a fault", and half a
    detector does not get to exit 0.
    """
    table, paths = _fixture(tmp_path, [dict(_GUARD_ONE), dict(_GUARD_TWO)])

    def runner(**kw):
        if kw["label"] == "baseline":
            return _out(rc=0, failed=(), passed=400)
        return _out()  # attributed, reproducible kill on both real mutants

    rc = mutate.main((), suite_runner=runner, table=table, module_paths=paths)
    out = capsys.readouterr().out
    assert rc == 2  # two kills do not buy a clean exit without a control
    assert "[killed]" in out  # the kills still report as returned facts...
    assert "caught=100%" in out  # ...the arithmetic is still printed...
    assert "MUST-PASS CONTROL: none configured" in out  # ...and so is its cost
    assert "unverified claim" in out


def test_stale_control_anchor_reports_configured_not_exercised(tmp_path, capsys):
    """A control row with a dead anchor must not read as "no controls".

    WHY: delete this and a control whose anchor drifted falls off the tally
    entirely: ctrl_ok and ctrl_dead are both empty, so the run prints what a
    zero-control run prints, and a battery that LOST its negative control to
    a refactored line is read as a battery that never wanted one. The three
    states — configured and exercised, configured but not exercised, never
    configured — must read as three states, the same way the battery already
    refuses to blur UNSCORED and ALIVE.
    """
    table, paths = _fixture(tmp_path, [dict(_GUARD_ONE), dict(_STALE_CONTROL)])

    def runner(**kw):
        if kw["label"] == "baseline":
            return _out(rc=0, failed=(), passed=400)
        return _out()  # the one real mutant: attributed, reproducible kill

    rc = mutate.main((), suite_runner=runner, table=table, module_paths=paths)
    out = capsys.readouterr().out
    assert rc == 2
    assert "MUST-PASS CONTROL: none configured" not in out  # one WAS configured
    assert "1 control row(s) configured" in out
    assert "configured-but-never-exercised: 1" in out
    assert "[ctrl ok]" not in out  # not exercised, so not a passing control either


def test_surviving_control_with_all_mutants_killed_exits_zero(tmp_path, capsys):
    """Positive control for the zero-control block: it must be satisfiable.

    WHY: delete this and nothing pins the block to fire only on an absent
    control — a guard that voided every run regardless would keep the two
    tests above green while making a clean controlled run impossible. A check
    that cannot pass is the same defect as a check that cannot fire, so this
    green-path pin is load-bearing: one surviving control plus every mutant
    killed must print the survived count and still exit 0.
    """
    table, paths = _fixture(tmp_path, [dict(_GUARD_ONE), dict(_CONTROL)])

    def runner(**kw):
        if kw["label"] == "baseline":
            return _out(rc=0, failed=(), passed=400)
        if kw["label"].startswith("scratch-01"):
            return _out(rc=0, failed=(), passed=400)  # inert control: stays green
        return _out()  # the one real mutant: killed on two identical runs

    rc = mutate.main((), suite_runner=runner, table=table, module_paths=paths)
    out = capsys.readouterr().out
    assert rc == 0
    assert "MUST-PASS CONTROL: 1/1 exercised inert edit(s) survived" in out
    assert "1 control row(s) configured" in out
    assert "configured-but-never-exercised" not in out
    assert "[ctrl ok]" in out
