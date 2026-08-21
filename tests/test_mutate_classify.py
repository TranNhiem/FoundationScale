"""End-to-end scoring tests for tools/mutate.py with two injected seams.

Seam one, suite_runner, came from the scoring redesign: before it, ANY
nonzero pytest exit was printed as `[killed]`, including collection errors
and zero-attribution runs.

Seam two, table/module_paths, exists because the battery was caught firing
with no fault behind it: these meta-tests used to drive main() against the
LIVE tools/mutations.json and the LIVE tree, so a mutant applied by the
outer battery made the inner main() refuse a now-stale live anchor and exit
2 — a deterministic failure the outer battery banked as a kill. A
comment-only INERT mutant scored [killed] exactly that way, attributed
solely to test_true_kill_still_scores. Every main() call in this file now
passes a scratch table anchored at files under tmp_path: mutating any
shipped module cannot change what these tests measure.

The suite runner is injected; the on-disk apply / byte-verified restore under
test are the real ones (against scratch files).
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

CANARY_XML = Path(__file__).resolve().with_name("test_mutate_junit_canary.xml")


def _out(**kw):
    base = dict(
        rc=1,
        passed=400,
        failed=("tests/t.py::test_x",),
        errored=(),
        skipped=0,
        duration_s=1.0,
        timed_out=False,
        junit_path="j.xml",
    )
    return mutate.SuiteOutcome(**{**base, **kw})


def _fake(trial):
    """Baseline always green; each trial label routed to `trial(label)`."""

    def fake(**kw):
        if kw["label"] == "baseline":
            return _out(rc=0, failed=(), errored=(), passed=400)
        return trial(kw["label"])

    return fake


_GUARD_MUT = {
    "name": "break-guard",
    "what": "the negative guard stops rejecting",
    "anchor": "if value < 0:",
    "replacement": "if False and value < 0:",
}


# The MUST-PASS half of the detector, in scratch form: a comment-only edit
# that changes no behaviour, so a suite that reports detecting it has proven
# its attribution unsound. Exit 0 is reserved for runs that carry one.
_INERT_CONTROL = {
    "name": "inert-control-must-pass",
    "what": "a comment is reworded and nothing else changes",
    "anchor": "# scratch anchor two",
    "replacement": "# scratch anchor 2",
    "must_survive": True,
}


def _fixture(tmp_path: Path, mutations: list[dict] | None = None):
    """A mutation table and module map anchored at a scratch file under tmp_path.

    WHY: before the table seam, these tests ran main() against the LIVE
    mutations.json and LIVE tree — the coupling that let an inert comment
    mutant bank a kill. Every test below reads scratch files only.
    """
    target = tmp_path / "scratch_module.py"
    target.write_text(
        "def guarded(value):\n"
        "    if value < 0:  # scratch anchor one\n"
        "        raise ValueError(value)\n"
        "    return value  # scratch anchor two\n",
        "utf-8",
    )
    if mutations is None:
        mutations = [dict(_GUARD_MUT)]
    return {"scratch": mutations}, {"scratch": target}


def test_nonzero_nonone_rc_is_unscored(tmp_path, capsys):
    """A collection error under a mutant is not a kill and blocks the battery.

    Pre-fix this exact trace printed `[killed] <name> 0 test(s):` and
    counted toward exit 0.
    """
    table, paths = _fixture(tmp_path)
    rc = mutate.main(
        (),
        suite_runner=_fake(
            lambda label: _out(rc=2, failed=(), errored=("<collection import error>",))
        ),
        table=table,
        module_paths=paths,
    )
    out = capsys.readouterr().out
    assert rc == 2
    assert "[UNSCORED]" in out
    assert "[killed]" not in out


def test_rc1_without_attribution_is_unscored(tmp_path, capsys):
    """rc=1 that names no failing test is not a kill."""
    table, paths = _fixture(tmp_path)
    rc = mutate.main(
        (),
        suite_runner=_fake(lambda label: _out(failed=(), errored=("<setup error>",))),
        table=table,
        module_paths=paths,
    )
    out = capsys.readouterr().out
    assert rc == 2
    assert "[killed]" not in out


def test_kill_requires_identical_rerun_attribution(tmp_path, capsys):
    """Divergent first/second run attribution is a flake, not detection."""

    def trial(label):
        if label.endswith("-run1"):
            return _out(failed=("tests/a.py::test_a",))
        return _out(failed=("tests/b.py::test_b",))

    table, paths = _fixture(tmp_path)
    rc = mutate.main((), suite_runner=_fake(trial), table=table, module_paths=paths)
    out = capsys.readouterr().out
    assert rc == 2
    assert "not reproducible" in out
    assert "[killed]" not in out


def test_timeout_is_unscored(tmp_path, capsys):
    """A runner that cannot finish is a failed measurement, in bounded time."""

    def trial(label):
        raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=1)

    table, paths = _fixture(tmp_path)
    rc = mutate.main((), suite_runner=_fake(trial), table=table, module_paths=paths)
    out = capsys.readouterr().out
    assert rc == 2
    assert "[UNSCORED]" in out
    assert "[killed]" not in out


def test_junit_parser_canary():
    """The parser must extract REAL pytest nodeids from real-shape junit XML.

    Real pytest writes dotted classnames (`tests.test_x`), never slashes, so
    the fixture does too — a canary that only sings in air it will never
    breathe is how the old slash-shaped fixture passed over a shape pytest
    does not produce. The last lines pin the property that matters: every
    attributed id must be one pytest could actually collect.
    """
    passed, failed, errored, skipped = mutate._parse_junit(CANARY_XML)
    assert failed == ("tests/test_x.py::test_y",)
    assert (passed, errored, skipped) == (2, (), 0)
    for nodeid in failed:
        path_part, _, _ = nodeid.partition("::")
        assert path_part.endswith(".py")
        assert "/" in path_part and "." not in path_part[:-3]


def test_true_kill_still_scores(tmp_path, capsys):
    """An attributed failure reproduced verbatim is a kill; the battery is green.

    The table carries a MUST-PASS control beside the real mutant because exit
    0 now means "controlled run, every mutant scored", not merely "every
    mutant scored": a table with no inert row blocks at 2. Wiring the control
    in keeps this test measuring the thing it was written to measure — that a
    reproduced, attributed failure scores as a kill — instead of quietly
    re-measuring the zero-control block. Delete the control here and the test
    stops distinguishing the two.
    """
    table, paths = _fixture(tmp_path, [dict(_GUARD_MUT), dict(_INERT_CONTROL)])

    def trial(label):
        # Index 01 is the control: its trial must come back green. An inert
        # edit reported killed voids the run, which is a different test's job.
        if "-01-" in label:
            return _out(rc=0, failed=(), errored=())
        return _out(failed=("tests/t.py::test_x",))

    rc = mutate.main((), suite_runner=_fake(trial), table=table, module_paths=paths)
    out = capsys.readouterr().out
    assert rc == 0
    assert "[killed]" in out
    assert "[UNSCORED]" not in out
    assert "1/1 exercised inert edit(s) survived" in out


def test_classify_routes_green_unreadable_evidence_to_unscored():
    """Finding 1: (rc=0, errored=<junit unreadable>) is 'never measured', not a gap.

    classify must honour the same sentinel on the green branch that it
    already honours on the rc=1 branch — otherwise exit 1 ("the suite has a
    gap") is printed over evidence nobody read, which is exit 2's meaning.
    """
    out = _out(rc=0, failed=(), errored=("<junit unreadable: [Errno 13] Permission denied>",))
    verdict = mutate.classify(out)
    assert verdict.kind is mutate.TrialKind.UNSCORED
    assert out.junit_path in verdict.reason


def test_green_run_with_unreadable_junit_exits_2(tmp_path, capsys):
    """Finding 1, end to end: a green-but-unreadable trial blocks the battery."""
    table, paths = _fixture(tmp_path)
    rc = mutate.main(
        (),
        suite_runner=_fake(lambda label: _out(rc=0, failed=(), errored=("<junit unreadable: x>",))),
        table=table,
        module_paths=paths,
    )
    out = capsys.readouterr().out
    assert rc == 2
    assert "[UNSCORED]" in out
    assert "[ALIVE]" not in out


def test_baseline_with_unreadable_junit_refuses_to_run(tmp_path, capsys):
    """Finding 1, one door down: the same sentinel must arm the baseline gate.

    A green baseline whose report could not be parsed says nothing about
    coverage; running the battery on top of it inherits the defect.
    """
    table, paths = _fixture(tmp_path)

    def runner(**kw):
        if kw["label"] == "baseline":
            return _out(rc=0, failed=(), passed=400, errored=("<junit unreadable: x>",))
        return _out(rc=0, failed=(), errored=())

    rc = mutate.main((), suite_runner=runner, table=table, module_paths=paths)
    out = capsys.readouterr().out
    assert rc == 2
    assert "EVIDENCE IS NOT WHOLE" in out


def test_nodeid_reconstructs_collectable_ids():
    """Finding 2: attribution must be pasteable back into `pytest <nodeid>`.

    Dotted module, class-nested method, and a parametrized name — the three
    shapes a naive conversion is most likely to get wrong.
    """
    assert mutate._nodeid("tests.test_run_event", "test_foo") == (
        "tests/test_run_event.py::test_foo"
    )
    assert mutate._nodeid("tests.test_parity.TestFoo", "test_bar") == (
        "tests/test_parity.py::TestFoo::test_bar"
    )
    assert mutate._nodeid("tests.test_parity", "test_cosine_guard[bf16]") == (
        "tests/test_parity.py::test_cosine_guard[bf16]"
    )


def test_caught_percentage_suppressed_when_any_trial_unmeasured(tmp_path, capsys):
    """Finding 3: no `caught=100%` beside a nonzero unscored count.

    WHY: `8 killed 0 alive 4 unscored caught=100%` claims a whole
    measurement over partial evidence. The fixture carries two mutations in
    one module; the first scores (killed), the second is unscored — the
    mixed case the old arithmetic misreported. The assertion no longer
    depends on the live table's shape.
    """
    mutations = [
        dict(_GUARD_MUT),
        {
            "name": "drop-fast-path",
            "what": "the tail return loses its marker",
            "anchor": "    return value  # scratch anchor two",
            "replacement": "    return value",
        },
    ]
    table, paths = _fixture(tmp_path, mutations)

    def trial(label):
        if "-00-" in label:
            return _out(failed=("tests/t.py::test_x",))
        return _out(rc=2, failed=(), errored=("<collection import error>",))

    rc = mutate.main((), suite_runner=_fake(trial), table=table, module_paths=paths)
    out = capsys.readouterr().out
    assert rc == 2
    seen_mixed = False
    for line in out.splitlines():
        if "unscored" not in line or "caught=" not in line:
            continue
        parts = line.split()
        if int(parts[parts.index("unscored") - 1]) > 0:
            seen_mixed = True
            assert "caught=100%" not in line
            assert "caught=--" in line
    assert seen_mixed, "no module exercised the mixed scored/unscored row"
