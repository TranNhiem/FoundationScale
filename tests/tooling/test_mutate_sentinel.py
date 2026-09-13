"""Control tests for the tools/mutate.py crash sentinel (finding #404).

WHAT IS UNDER TEST

A battery run strands a mutant in the tree if the process dies between
"applying the mutant" and "writing the original bytes back". The sentinel
mechanism answers that: before touching any file, mutate.py records every
measured file's original TEXT and SHA-256, keyed on the file's ABSOLUTE
path, in one sentinel JSON document. On startup, BEFORE any baseline or
mutation, a guard compares the recorded hashes against the bytes currently
on disk:

  * no sentinel           -> the guard is silent; the run proceeds (C1)
  * all hashes match      -> the previous run restored fine and died only
    before deleting its sentinel; the stale sentinel is deleted and the
    run continues (C2)
  * any file differs      -> refuse: exit 96, NAME the differing path (C3)
    and NAME the recovery command (C4)
  * ``--recover``         -> restore every recorded file byte-for-byte (C5),
    purge its __pycache__ entry (C7; finding #328: a same-second,
    size-preserving restore leaves a .pyc that still validates), remove
    the sentinel, and exit 0 WITHOUT ever invoking the battery (C9). With
    no sentinel, --recover is an idempotent success that says there is
    nothing to recover (C6).

C8 is the POSITIVE CONTROL: plant a sentinel whose recorded hash matches,
then change the file's bytes, and demand refusal. Without it, every other
leg here would also pass under a guard hardcoded to ``True``.

HARDENING LEGS AGAINST THE REAL IMPLEMENTATION

  * C10: the sentinel write is atomic — no ``.json.tmp`` sibling is left
    behind, and the file the writer returns parses as JSON.
  * C11: a sentinel truncated to half its bytes (the signature of a kill
    DURING the write) makes a normal run refuse 96 saying the sentinel
    cannot be parsed — never raise, never exit 1 — and makes ``--recover``
    refuse 96 saying the sentinel holds the only copy of the pre-battery
    bytes, so recovery is impossible.
  * C12: a backup keyed OUTSIDE ROOT neither crashes the writer nor
    mangles ``_display`` (``Path.relative_to`` RAISES on such paths).
  * C13: ``sentinel_path_for`` co-locates the sentinel with the tree it
    describes: IN_FLIGHT_SENTINEL for in-tree backups, beside the files
    for out-of-tree backups — never in ROOT for an injected table.
  * C14: the READ is scoped the same way the WRITE is, so a battery over
    an out-of-tree table does not read — or delete — the repository-root
    sentinel. This is the measured self-hit: the harness's own self-tests
    ARE such batteries and they run inside the suite a real battery drives.
  * C15: C14's mandatory pair — the same planted sentinel must STILL
    refuse an in-tree battery 96. Without C15, C14 would pass just as
    happily against a #404 mechanism that had been deleted outright.

INTERFACE PINNED BY THIS SUITE (read tools/mutate.py; every name is real)

  mutate.ROOT                       tree root; monkeypatched to the sandbox
  mutate.IN_FLIGHT_SENTINEL         the sentinel path, computed AT IMPORT
                                    TIME as ROOT / ".mutate_in_flight.json".
                                    Patching ROOT alone does NOT move it, so
                                    the sandbox patches BOTH and asserts both
                                    redirects before any test body runs
  mutate.write_in_flight_sentinel(backups, modules)
                                    backups: {absolute Path: original TEXT};
                                    writes atomically via a sibling
                                    ".json.tmp" plus rename; returns the
                                    path it wrote to
  mutate.sentinel_path_for(targets) where the sentinel for THOSE target
                                    files belongs. Takes any iterable of
                                    paths — not the backups mapping —
                                    because main() must resolve the same
                                    location at STARTUP, before a single
                                    backup has been taken
  mutate.sentinel_differences(sentinel)
                                    -> (present, [differing display paths]).
                                    The sentinel is a PARAMETER, and that is
                                    the whole point: reading the module-global
                                    unconditionally is finding C14
  mutate.recover_from_sentinel(sentinel=None)
                                    -> exit code. Defaults to the repository-
                                    root sentinel because --recover is an
                                    operator command with no table in hand;
                                    the parameter exists so this suite can
                                    drive recovery over its own scratch files
  mutate.SentinelUnreadable         raised by _load_sentinel() on a sentinel
                                    that exists but cannot be parsed
  mutate._display(path)             tree-relative inside ROOT, else the
                                    absolute path unchanged
  mutate._sha256_text(text)         hex digest of the UTF-8 encoding
  mutate.main(argv, *, table=..., module_paths=...)
                                    returns an exit code; understands
                                    --recover; table/module_paths are the
                                    keyword-only injection seams this suite
                                    uses instead of monkeypatching
                                    load_table
  mutate.check_baseline, mutate.score_trial, mutate.TrialKind,
  mutate.TrialVerdict               battery seams this suite replaces with
                                    doubles

SAFETY

Every test runs against a throwaway tree under tmp_path: mutate.ROOT AND
mutate.IN_FLIGHT_SENTINEL are monkeypatched to point there, the cwd is
moved with them, and the sandbox fixture actively asserts both redirects
took before any test body executes. No test may resolve, read, or write a
path inside the developer's real checkout: a test that mutates the
developer's actual src/ is a defect worse than the one being fixed,
because finding #404 IS "the tool left src/ mutated".

FAILURE DISCIPLINE

Each test can go red for exactly one reason, and that reason is written in
its assertion messages. CI exports FS_FORBID_SKIPS=1 and treats any skip
as a hard failure, so this file contains no pytest.skip: the tmp_path
sandbox and the battery doubles make every leg runnable everywhere.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _REPO_ROOT / "tools" / "mutate.py"

# The refusal code the guard leg contract pins: distinct from 2 (void run)
# and from 0/1 so CI can route a stranded-mutant refusal differently.
_REFUSAL = 96

UNIT_REL = "src/sandbox/unit.py"
UNIT_ORIGINAL = (
    '"""Tiny subject module for the mutate-sentinel control battery."""\n'
    "\n"
    "\n"
    "# battery control: nothing here is exercised by the suite\n"
    "\n"
    "\n"
    "def toggle():\n"
    '    return "safety-latched"  # ANCHOR: what the fake battery measures\n'
)

# One must-fire row (killed by the fake suite) and one first-class
# must_survive control row (a trailing, inert comment edit), because per the
# tool's own register a run voids without a configured control. Dict/list
# order matters: the control is index 1, so its trial label is "unit-01".
# Both anchors occur exactly once in UNIT_ORIGINAL, so the table survives
# _validate_table (and would survive check_anchor_freshness).
_TABLE = {
    "unit": [
        {
            "name": "unit-unlatch",
            "what": "return the unlatched value",
            "anchor": 'return "safety-latched"  # ANCHOR: what the fake battery measures',
            "replacement": 'return "safety-unlatched"  # ANCHOR: what the fake battery measures',
        },
        {
            "name": "unit-inert-trailing-comment",
            "what": "behaviour-preserving comment edit; the suite must let it pass",
            "anchor": "# battery control: nothing here is exercised by the suite",
            "replacement": "# battery control: nothing here is exercised by the suite (inert)",
            "must_survive": True,
        },
    ]
}


def _load_mutate():
    spec = importlib.util.spec_from_file_location("mutate_under_test", _TOOL)
    module = importlib.util.module_from_spec(spec)
    # Finding #100/D1, same idiom as tests/tooling/test_mutate_zero_work_
    # refusals.py: the module MUST be registered in sys.modules BEFORE
    # exec_module runs, because dataclasses inside tools/mutate.py resolve
    # their string annotations via sys.modules[cls.__module__] at class-body
    # execution time — without the early registration the exec itself fails,
    # before any test gets to say anything.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mutate():
    return _load_mutate()


def _sentinel_path(mutate):
    return Path(mutate.IN_FLIGHT_SENTINEL)


def _plant_sentinel(mutate, backups):
    """Write a sentinel exactly the way the tool does, via its own writer.

    ``backups`` is {absolute Path: original TEXT} — the real writer's
    contract, keyed on absolute paths with str values, not bytes.
    """
    target = mutate.write_in_flight_sentinel(dict(backups), ["unit"])
    assert target.is_file(), (
        f"test-setup failure: mutate.write_in_flight_sentinel did not create "
        f"{target} — the pinned interface (write_in_flight_sentinel / "
        "IN_FLIGHT_SENTINEL / ROOT) is not what the implementation provides"
    )
    return target


def _invoke(mutate, sandbox, argv):
    try:
        code = mutate.main(
            list(argv),
            table=copy.deepcopy(_TABLE),
            module_paths=dict(sandbox.module_paths),
        )
    except SystemExit as exc:  # tolerate an argparse-era main that sys.exit()s
        code = exc.code if isinstance(exc.code, int) else 1
    return code


@pytest.fixture()
def sandbox(tmp_path, monkeypatch, mutate):
    """A full fake tree under tmp_path; the real checkout is never in scope."""
    unit = tmp_path / UNIT_REL
    unit.parent.mkdir(parents=True)
    unit.write_text(UNIT_ORIGINAL, encoding="utf-8")

    monkeypatch.setattr(mutate, "ROOT", tmp_path)
    # IN_FLIGHT_SENTINEL was computed AT IMPORT TIME from the real ROOT, so
    # patching ROOT alone leaves the sentinel aimed at the developer's real
    # checkout. Patch it too, or every leg below reads and writes the real
    # repository root.
    monkeypatch.setattr(mutate, "IN_FLIGHT_SENTINEL", tmp_path / ".mutate_in_flight.json")
    monkeypatch.chdir(tmp_path)

    # Actively prove BOTH redirects before returning: a mis-patched ROOT or
    # IN_FLIGHT_SENTINEL here means later asserts would be measuring (and
    # recovering files inside) the developer's real tree.
    resolved_root = Path(mutate.ROOT).resolve()
    assert resolved_root == tmp_path.resolve() and resolved_root != _REPO_ROOT.resolve(), (
        f"sandbox redirect failed: mutate.ROOT is {resolved_root}; refusing "
        "to run rather than risk touching the real repository tree"
    )
    resolved_sentinel = Path(mutate.IN_FLIGHT_SENTINEL).resolve()
    assert resolved_sentinel == (
        tmp_path / ".mutate_in_flight.json"
    ).resolve() and not resolved_sentinel.is_relative_to(_REPO_ROOT.resolve()), (
        f"sandbox redirect failed: mutate.IN_FLIGHT_SENTINEL is {resolved_sentinel}; "
        "it was computed at import time from the real ROOT, so patching ROOT "
        "alone does not move it — refusing to run rather than read or write "
        "the real repository root"
    )
    assert unit.resolve().is_relative_to(tmp_path.resolve()) and not unit.resolve().is_relative_to(
        _REPO_ROOT.resolve()
    ), f"sandbox leak: subject file {unit} is not confined to {tmp_path}"
    return SimpleNamespace(
        root=tmp_path,
        unit=unit,
        rel=UNIT_REL,
        sentinel=tmp_path / ".mutate_in_flight.json",
        module_paths={"unit": unit},
    )


@pytest.fixture()
def battery(mutate, monkeypatch):
    """Deterministic battery doubles; the real pytest suite never runs."""
    calls = {"baseline": 0, "trials": []}

    def fake_check_baseline(suite_runner, *, junit_dir):
        calls["baseline"] += 1
        return True, "baseline green (test double — the real suite never ran)"

    def fake_score_trial(suite_runner, *, junit_dir, label):
        calls["trials"].append(label)
        if label.endswith("-01"):  # the must_survive control row
            return mutate.TrialVerdict(kind=mutate.TrialKind.ALIVE, attribution=(), reason="")
        return mutate.TrialVerdict(
            kind=mutate.TrialKind.KILLED,
            attribution=("tests/sandbox/test_unit.py::test_toggle",),
            reason="",
        )

    monkeypatch.setattr(mutate, "check_baseline", fake_check_baseline)
    monkeypatch.setattr(mutate, "score_trial", fake_score_trial)
    return calls


def test_c1_no_sentinel_present_guard_is_silent(mutate, sandbox, battery, capsys):
    assert not _sentinel_path(mutate).exists(), (
        "test-setup failure: fresh tmp_path carried a sentinel"
    )
    rc = _invoke(mutate, sandbox, [])
    out, err = capsys.readouterr()
    assert battery["baseline"] >= 1 and rc != _REFUSAL, (
        "C1: with NO sentinel on disk the guard must be silent and let the "
        "run start, but the battery was never reached "
        f"(baseline invocations={battery['baseline']}, rc={rc}); the guard "
        "is treating absence as corruption. Output was:\n" + out + err
    )


def test_c2_matching_sentinel_is_stale_and_cleared_and_the_run_continues(
    mutate, sandbox, battery, capsys
):
    sentinel = _plant_sentinel(mutate, {sandbox.unit: sandbox.unit.read_text("utf-8")})
    rc = _invoke(mutate, sandbox, [])
    out, err = capsys.readouterr()
    assert rc != _REFUSAL and battery["baseline"] >= 1, (
        "C2: every recorded hash MATCHES the live file — the benign "
        "post-restore race — yet the guard refused or blocked the run "
        f"(rc={rc}, baseline invocations={battery['baseline']}); it should "
        "have deleted the stale sentinel and continued. Output was:\n" + out + err
    )
    assert not sentinel.exists(), (
        "C2: the run continued but the stale (fully matching) sentinel was "
        "left on disk at "
        f"{sentinel}; the NEXT invocation will re-examine it, and any "
        "legitimate edit between now and then turns today's benign leftover "
        "into tomorrow's false 96 refusal."
    )


def test_c3_differing_file_exits_96_and_names_the_path(mutate, sandbox, battery, capsys):
    _plant_sentinel(mutate, {sandbox.unit: UNIT_ORIGINAL})
    sandbox.unit.write_text(UNIT_ORIGINAL + "\n# stranded mutant bytes\n", encoding="utf-8")
    rc = _invoke(mutate, sandbox, [])
    out, err = capsys.readouterr()
    combined = out + err
    assert rc == _REFUSAL, (
        "C3: a recorded file differs from its sentinel hash, so the run "
        f"must refuse with exit {_REFUSAL}; got rc={rc}. Output was:\n{combined}"
    )
    assert UNIT_REL in combined, (
        f"C3: exit {_REFUSAL} fired but the differing path was not NAMED — "
        f"expected the substring {UNIT_REL!r} in the output. An unnamed "
        "refusal sends the operator hunting through every backed-up file. "
        "Output was:\n" + combined
    )
    assert battery["baseline"] == 0, (
        "C3: the guard refused only AFTER spending a baseline suite run; the "
        "sentinel check must be the first thing a run does, before any "
        f"measurement is paid for (baseline invocations={battery['baseline']})."
    )


def test_c4_refusal_also_names_the_recovery_command(mutate, sandbox, capsys):
    _plant_sentinel(mutate, {sandbox.unit: UNIT_ORIGINAL})
    sandbox.unit.write_text(UNIT_ORIGINAL + "\n# stranded mutant bytes\n", encoding="utf-8")
    rc = _invoke(mutate, sandbox, [])
    out, err = capsys.readouterr()
    combined = out + err
    assert rc == _REFUSAL, (
        "C4 setup: expected the mismatch refusal "
        f"(exit {_REFUSAL}) so its message could be audited; got rc={rc}.\n{combined}"
    )
    assert "--recover" in combined, (
        "C4: the refusal message names no recovery command — expected the "
        "substring '--recover' so an operator staring at a red CI job at "
        "3am is told exactly how to restore the tree instead of guessing. "
        "Output was:\n" + combined
    )


def test_c5_recover_restores_byte_for_byte_and_removes_the_sentinel(mutate, sandbox, capsys):
    original = UNIT_ORIGINAL
    sentinel = _plant_sentinel(mutate, {sandbox.unit: original})
    sandbox.unit.write_text("# tree totalled by an interrupted battery run\n", encoding="utf-8")
    rc = _invoke(mutate, sandbox, ["--recover"])
    out, err = capsys.readouterr()
    assert rc == 0, (
        f"C5: --recover against a valid sentinel must exit 0; got rc={rc}. Output was:\n{out}{err}"
    )
    assert sandbox.unit.read_text("utf-8") == original, (
        "C5: --recover exited 0 but the restored bytes do NOT match the "
        "recorded original byte-for-byte — the tree is still not what the "
        "pre-crash run measured from, and 'recovered' is a lie. "
        f"First differing region visible in: {sandbox.unit}"
    )
    assert not sentinel.exists(), (
        f"C5: the restore succeeded but the sentinel at {sentinel} survived; "
        "the next run will refuse 96 against an already-clean tree and "
        "force a human to delete state the tool itself should have cleared."
    )


def test_c6_recover_with_no_sentinel_is_idempotent_success(mutate, sandbox, capsys):
    assert not _sentinel_path(mutate).exists(), (
        "test-setup failure: fresh tmp_path carried a sentinel"
    )
    rc = _invoke(mutate, sandbox, ["--recover"])
    out, err = capsys.readouterr()
    assert rc == 0, (
        "C6: --recover with NO sentinel means nothing is stranded; it must "
        f"be an idempotent success, not an error, but rc={rc}. A crash here "
        "breaks every 'mutate --recover && mutate' correctness script. "
        f"Output was:\n{out}{err}"
    )
    assert "nothing to recover" in (out + err).lower(), (
        "C6: the no-sentinel recovery succeeded silently; it must SAY there "
        "is nothing to recover so an operator reading automation output can "
        "distinguish 'tree was already clean' from 'restore happened'. "
        "Output was:\n" + out + err
    )


def test_c7_recover_purges_the_pycache_entry_of_each_restored_file(mutate, sandbox, capsys):
    _plant_sentinel(mutate, {sandbox.unit: UNIT_ORIGINAL})
    sandbox.unit.write_text("# stranded mutant bytes\n", encoding="utf-8")
    pyc = sandbox.unit.parent / "__pycache__" / f"unit.{sys.implementation.cache_tag}.pyc"
    pyc.parent.mkdir(parents=True)
    pyc.write_bytes(b"\x00\x00\x00\x00stale-pyc-that-still-validates\x00\x00\x00\x00")
    rc = _invoke(mutate, sandbox, ["--recover"])
    out, err = capsys.readouterr()
    assert rc == 0, (
        f"C7 setup: --recover did not succeed (rc={rc}), so the pycache "
        f"assertion below would be measuring a failed restore.\n{out}{err}"
    )
    assert not pyc.exists(), (
        "C7 (#328): --recover restored the file's bytes but left "
        f"{pyc} behind. A same-second, size-preserving restore produces a "
        ".pyc whose timestamp/size still validates against the restored "
        "source, so the next import silently executes the STRANDED MUTANT. "
        "Recovery must purge the __pycache__ entry of every file it touches."
    )


def test_c8_positive_control_the_hash_comparison_is_live(mutate, sandbox, battery, capsys):
    original = sandbox.unit.read_text("utf-8")
    _plant_sentinel(mutate, {sandbox.unit: original})  # matches BY CONSTRUCTION
    sandbox.unit.write_text(original + "\n# post-sentinel edit: the crash footprint\n", "utf-8")
    rc = _invoke(mutate, sandbox, [])
    out, err = capsys.readouterr()
    assert rc == _REFUSAL, (
        "C8 POSITIVE CONTROL FAILED: a sentinel whose recorded hash matched "
        "the file was planted, the file's bytes were then changed, and the "
        f"guard STILL let the run proceed (rc={rc}, brought to you by "
        f"output below). The hash comparison is hardcoded True, reads the "
        "wrong file, or never executes — the guard is VACUOUS, and every "
        "other leg in this file would also pass under a no-op guard, so "
        "they currently prove nothing. Output was:\n" + out + err
    )
    assert battery["baseline"] == 0, (
        "C8: refusal happened (exit 96) only after battery machinery ran "
        f"(baseline invocations={battery['baseline']}); a live check that "
        "fires late still lets a stranded mutant contaminate a baseline run."
    )


def test_c9_recover_never_runs_the_battery(mutate, sandbox, monkeypatch, capsys):
    _plant_sentinel(mutate, {sandbox.unit: UNIT_ORIGINAL})
    sandbox.unit.write_text("# stranded mutant bytes\n", encoding="utf-8")

    def _explode(*args, **kwargs):
        raise AssertionError(
            "C9: --recover invoked battery machinery (check_baseline / "
            "score_trial) — recovery must be pure restore + pycache purge + "
            "sentinel removal. Running the suite during recovery would "
            "measure the MID-RESTORE tree and bill the operator a baseline "
            "for an emergency repair path."
        )

    monkeypatch.setattr(mutate, "check_baseline", _explode)
    monkeypatch.setattr(mutate, "score_trial", _explode)
    rc = _invoke(mutate, sandbox, ["--recover"])  # any suite invocation raises out of here
    out, err = capsys.readouterr()
    assert rc == 0 and sandbox.unit.read_text("utf-8") == UNIT_ORIGINAL, (
        "C9: with the battery armed to explode, --recover itself failed — "
        f"rc={rc}, restored={sandbox.unit.read_text('utf-8') == UNIT_ORIGINAL}. "
        "Recovery must not depend on any part of the measurement path. "
        f"Output was:\n{out}{err}"
    )


def test_c10_sentinel_write_is_atomic_and_leaves_no_tmp_sibling(mutate, sandbox):
    target = _plant_sentinel(mutate, {sandbox.unit: UNIT_ORIGINAL})
    tmp_sibling = target.with_suffix(".json.tmp")
    assert not tmp_sibling.exists(), (
        "C10: the atomic-write temp file "
        f"{tmp_sibling} was left behind after write_in_flight_sentinel "
        f"returned. The sentinel's entire reason to exist is that the "
        "process can be killed at any instant; a leftover sibling means the "
        "write is not the single atomic rename the design promises, and the "
        "next reader cannot tell a finished write from an interrupted one."
    )
    assert target == _sentinel_path(mutate), (
        "C10: write_in_flight_sentinel returned "
        f"{target}, not the in-tree sentinel {_sentinel_path(mutate)} — the "
        "caller (and this suite) must be able to trust the returned path."
    )
    try:
        payload = json.loads(target.read_text("utf-8"))
    except ValueError as exc:
        raise AssertionError(
            f"C10: the sentinel at {target} does not parse as JSON ({exc}); "
            "a guard whose own record is unreadable cannot tell a stranded "
            "mutant from a clean tree."
        ) from exc
    key = str(sandbox.unit)
    assert key in payload["files"], (
        "C10: the sentinel must key its file records on the ABSOLUTE path "
        f"(expected {key!r} among {sorted(payload['files'])}); a relative "
        "key would break recording and recovery for injected tables whose "
        "files live outside the tree."
    )
    assert payload["files"][key]["text"] == UNIT_ORIGINAL and payload["files"][key][
        "sha256"
    ] == mutate._sha256_text(UNIT_ORIGINAL), (
        "C10: the recorded text/hash do not match the bytes that were backed "
        "up; recovery restores from THIS record, so a wrong record is a "
        "wrong restore."
    )


def test_c11_unreadable_sentinel_refuses_cleanly_and_never_crashes(
    mutate, sandbox, battery, capsys
):
    sentinel = _plant_sentinel(mutate, {sandbox.unit: UNIT_ORIGINAL})
    raw = sentinel.read_bytes()
    sentinel.write_bytes(raw[: len(raw) // 2])  # the signature of a kill DURING the write

    try:
        rc = _invoke(mutate, sandbox, [])
    except Exception as exc:
        raise AssertionError(
            "C11: a truncated sentinel made the normal run RAISE "
            f"{type(exc).__name__} ({exc}) instead of refusing with exit "
            f"{_REFUSAL}. An uncaught json.JSONDecodeError exits 1, which "
            "reads as an ordinary crash — or worse, as 'the suite has a "
            "gap' — rather than as a tree needing a look. That is exactly "
            "the defect this leg exists to catch."
        ) from exc
    out, err = capsys.readouterr()
    combined = out + err
    assert rc == _REFUSAL, (
        "C11: a sentinel that cannot be parsed must refuse with exit "
        f"{_REFUSAL} — NOT raise and NOT return 1 (a crash, or 'the suite "
        f"has a gap'); got rc={rc}. Output was:\n{combined}"
    )
    assert "cannot be parsed" in combined, (
        "C11: the refusal must SAY the sentinel cannot be parsed, so the "
        "operator knows the file list is unknown and the tree needs "
        "inspection — expected the substring 'cannot be parsed'. "
        "Output was:\n" + combined
    )
    assert battery["baseline"] == 0, (
        "C11: the unreadable-sentinel refusal happened only after battery "
        f"machinery ran (baseline invocations={battery['baseline']}); the "
        "guard must fire before any measurement is paid for."
    )

    try:
        rc = _invoke(mutate, sandbox, ["--recover"])
    except Exception as exc:
        raise AssertionError(
            "C11: --recover against a truncated sentinel RAISED "
            f"{type(exc).__name__} ({exc}) instead of refusing with exit "
            f"{_REFUSAL}; the emergency repair path must be the most "
            "robust path this tool has."
        ) from exc
    out, err = capsys.readouterr()
    combined = out + err
    assert rc == _REFUSAL, (
        "C11: --recover against an unparseable sentinel must refuse with "
        f"exit {_REFUSAL}, not crash and not pretend success; got rc={rc}. "
        f"Output was:\n{combined}"
    )
    assert "only copy of the pre-battery bytes" in combined and "recovery is impossible" in (
        combined
    ), (
        "C11: the --recover refusal must say WHY it cannot help: the "
        "sentinel holds the only copy of the pre-battery bytes, so recovery "
        "is impossible from it and the operator must restore from git. "
        "Output was:\n" + combined
    )
    assert sentinel.is_file(), (
        "C11: the unparseable sentinel was deleted; it must be LEFT in "
        "place for inspection, because it is the only evidence of what the "
        "interrupted battery touched."
    )


def test_c12_out_of_tree_backup_paths_neither_crash_nor_mangle_display(
    mutate, sandbox, tmp_path_factory
):
    outside_dir = tmp_path_factory.mktemp("outside-tree")
    outside_file = outside_dir / "scratch_module.py"
    outside_file.write_text("# scratch\n", encoding="utf-8")
    assert not outside_file.is_relative_to(Path(mutate.ROOT)), (
        f"test-setup failure: {outside_file} is inside the sandbox ROOT "
        f"{mutate.ROOT}; this leg needs a key OUTSIDE ROOT to exercise the "
        "Path.relative_to ValueError path."
    )
    try:
        target = mutate.write_in_flight_sentinel({outside_file: "# scratch\n"}, ["scratch"])
    except ValueError as exc:
        raise AssertionError(
            "C12: write_in_flight_sentinel crashed on an out-of-tree backup "
            f"key ({exc}). Path.relative_to RAISES on paths outside ROOT, "
            "so any _display/keying regression turns the meta-suite's own "
            "scratch fixtures into a crash."
        ) from exc
    assert target.is_file(), (
        f"C12: the out-of-tree write reported success but {target} is not "
        "on disk; the write must succeed exactly as an in-tree write does."
    )
    assert mutate._display(outside_file) == str(outside_file), (
        "C12: _display must return an out-of-tree path unchanged (the "
        f"absolute path); got {mutate._display(outside_file)!r} for "
        f"{outside_file}. A refusal or recovery message that mangles the "
        "path sends the operator looking at the wrong file."
    )
    assert mutate._display(sandbox.unit) == UNIT_REL, (
        "C12: _display regressed the IN-tree case while handling the "
        f"out-of-tree one: expected {UNIT_REL!r}, got "
        f"{mutate._display(sandbox.unit)!r}. In-tree paths stay "
        "tree-relative so refusal messages name files the operator "
        "recognises."
    )


def test_c13_sentinel_colocation_follows_the_tree_it_describes(mutate, sandbox, tmp_path_factory):
    in_tree = mutate.sentinel_path_for({sandbox.unit: UNIT_ORIGINAL})
    assert in_tree == _sentinel_path(mutate), (
        "C13: an all-in-tree backup set must place the sentinel at "
        f"IN_FLIGHT_SENTINEL ({_sentinel_path(mutate)}), where --recover "
        f"and the startup check look; sentinel_path_for returned {in_tree}."
    )
    outside_dir = tmp_path_factory.mktemp("outside-tree")
    outside_file = outside_dir / "scratch_module.py"
    outside_file.write_text("# scratch\n", encoding="utf-8")
    out_tree = mutate.sentinel_path_for({outside_file: "# scratch\n"})
    assert out_tree == outside_dir / _sentinel_path(mutate).name, (
        "C13: an out-of-tree backup set must place the sentinel BESIDE the "
        f"files it describes ({outside_dir / _sentinel_path(mutate).name}); "
        f"sentinel_path_for returned {out_tree}."
    )
    assert not out_tree.is_relative_to(Path(mutate.ROOT).resolve()) and not Path(
        out_tree
    ).is_relative_to(Path(mutate.ROOT)), (
        "C13: the out-of-tree sentinel must NOT live in ROOT "
        f"({mutate.ROOT}); got {out_tree}. A killed meta-suite run would "
        "otherwise strand a sentinel in the real repo root naming pytest "
        "tmpdir paths that pytest then deletes, and the next real battery "
        "would read the vanished files as hash mismatches and refuse 96 "
        "over nothing."
    )
    assert out_tree.is_relative_to(outside_dir), (
        f"C13: the out-of-tree sentinel {out_tree} is not beside the files "
        f"it describes ({outside_dir}); the sentinel must not outlive the "
        "tree it describes."
    )


# ---------------------------------------------------------------------------
# C14/C15 — the sentinel READ must be scoped exactly like the sentinel WRITE.
#
# Measured defect (this campaign): write_in_flight_sentinel placed the sentinel
# via sentinel_path_for(backups) — repository root for a repo battery, the
# targets' common parent for an out-of-tree one — while the STARTUP check read
# the module-global IN_FLIGHT_SENTINEL unconditionally. Those two are the same
# path for the operator and DIFFERENT paths for the harness's own self-tests,
# which drive mutate.main() over injected tables in pytest scratch dirs from
# INSIDE the suite a real battery runs. So during any real battery — sentinel on
# disk, mutant applied, therefore a recorded file differing — 20 of the 42
# self-tests refused 96, those 20 failures landed in every trial junit, the
# MUST-PASS inert control read as killed, and the battery declared its own
# attribution unsound and exited 2. A true verdict about a defect in the
# harness, not in the tree; the scanner-self-hit class.
#
# Both legs are driven against ONE planted state, differing only in WHERE the
# battery's targets live, because a fix that silences the false refusal must not
# silence the true one. C14 alone would pass just as well against a mechanism
# that had been deleted; C15 is what forbids that reading.
# ---------------------------------------------------------------------------


@pytest.fixture()
def stranded_in_tree(mutate, sandbox):
    """Plant the state a real battery holds mid-flight: sentinel + differing file.

    Written through the tool's own writer, so the sentinel lands wherever the
    implementation decides it belongs rather than wherever this test guesses —
    the leg would otherwise pass by planting at a path nothing reads.
    """
    planted = _plant_sentinel(mutate, {sandbox.unit: UNIT_ORIGINAL})
    assert planted == sandbox.sentinel, (
        f"test-setup failure: an IN-TREE backup put the sentinel at {planted}, "
        f"not at the sandbox root {sandbox.sentinel} — the premise of both legs "
        "below is that an in-tree battery writes to the root"
    )
    # A mutant in flight is exactly this: recorded bytes, then different bytes.
    sandbox.unit.write_text(
        UNIT_ORIGINAL.replace("safety-latched", "safety-unlatched"), encoding="utf-8"
    )
    present, differing = mutate.sentinel_differences(planted)
    assert present and differing, (
        "test-setup failure: the planted sentinel does not read as present-and-"
        f"differing (present={present}, differing={differing}); with no "
        "difference to see, neither leg below measures anything"
    )
    return planted


def test_c14_an_out_of_tree_battery_does_not_read_the_repository_sentinel(
    mutate, sandbox, battery, stranded_in_tree, tmp_path_factory, capsys
):
    outside = tmp_path_factory.mktemp("outside_the_tree")
    assert not outside.resolve().is_relative_to(Path(mutate.ROOT).resolve()), (
        f"test-setup failure: {outside} is inside the sandbox root "
        f"{mutate.ROOT}, so this leg would be the in-tree case in disguise"
    )
    unit = outside / "unit.py"
    unit.write_text(UNIT_ORIGINAL, encoding="utf-8")

    code = mutate.main([], table=copy.deepcopy(_TABLE), module_paths={"unit": unit})

    out = capsys.readouterr()
    assert code != _REFUSAL, (
        f"a battery whose every target lives under {outside} refused "
        f"{_REFUSAL} because of a sentinel at {stranded_in_tree}, which "
        "describes a tree it does not touch. This is the self-hit: the "
        "harness's own self-tests are such batteries, and they run inside "
        f"the suite a real battery drives.\nstdout:\n{out.out}\nstderr:\n{out.err}"
    )
    assert battery["baseline"] == 1, (
        f"the run neither refused nor reached check_baseline (rc={code}); it "
        "settled somewhere else, so this leg does not measure the sentinel "
        f"guard at all.\nstdout:\n{out.out}\nstderr:\n{out.err}"
    )
    assert str(stranded_in_tree) not in (out.out + out.err), (
        f"the out-of-tree battery named the in-tree sentinel {stranded_in_tree} "
        "in its output — it read a sentinel it does not own"
    )
    assert stranded_in_tree.is_file(), (
        f"the out-of-tree battery DELETED {stranded_in_tree} — it not only read "
        "another battery's sentinel, it took the stale-sentinel branch and "
        "removed the record a real recovery depends on"
    )


def test_c15_control_the_same_sentinel_still_refuses_an_in_tree_battery(
    mutate, sandbox, battery, stranded_in_tree, capsys
):
    """C14's mandatory pair: scoping the read must not disarm the mechanism.

    Same planted sentinel, same differing file; only the battery's targets move
    inside the tree. If this went green too, the read would be scoped to
    nothing and #404 would be silently repealed.
    """
    code = _invoke(mutate, sandbox, [])

    out = capsys.readouterr()
    assert code == _REFUSAL, (
        f"an IN-TREE battery returned {code} against a sentinel at "
        f"{stranded_in_tree} recording a file that now differs — the stranded-"
        "mutant refusal (#404) has been disarmed, which is the destructive way "
        f"to make C14 pass.\nstdout:\n{out.out}\nstderr:\n{out.err}"
    )
    assert battery["baseline"] == 0, (
        "the in-tree battery ran check_baseline before refusing — it measured a "
        "tree it had already decided it could not trust"
    )
    assert str(stranded_in_tree) in (out.out + out.err), (
        f"the refusal never named {stranded_in_tree}, so the operator cannot "
        "tell which sentinel to inspect"
    )
    assert sandbox.rel in (out.out + out.err), (
        f"the refusal never named the differing file {sandbox.rel}"
    )
