#!/usr/bin/env python3
"""Per-module coverage floor -- the gate that refuses a subsidised total.

WHY THIS GATE EXISTS
    CI enforces `--cov-fail-under=90`. That flag is a TOTAL: one number over
    the whole measured tree, and a total can be subsidised. Measured
    2026-09-03 the repository reports 92.79% -- comfortably green -- while
    FIVE of its 28 tracked modules sit below the 90 that number is compared
    against: train/__main__.py at 0.00%, train/__init__.py at 50.00%,
    train/loop.py at 78.43% (62.18% before this campaign's execution test),
    tools/emit_run_manifest.py at 80.92% and checkpoint/dcp_meta.py at
    81.36%. All five pass CI, because the other 23 average high enough to
    carry them across the line. The aggregate is being read as if it were a
    claim about every module, and it is not: a total is a claim about the
    sum of numerators over the sum of denominators, nothing more. Reading
    "92.79% overall" as "the training path is tested" is precisely the
    vacuous-truth defect this repository exists to eliminate -- a sweeping
    claim bought with evidence about something else. This gate adds the
    claim the total cannot make: EVERY tracked module, individually, inside
    a stated band around its own floor, or the run does not pass.

THE CLAIM
    The denominator is derived from the repository (`git ls-files` over
    src/foundationscale plus the modules named in TOOLS_IN_SCOPE), never
    from the report: a module no test imports is invisible to coverage, and
    invisible must not mean clean. A tracked module missing from the report
    is UNMEASURED.

    A report entry the tracked set does not carry is split on ONE axis --
    where the file lives -- because the two halves are different facts:
      IN-TREE, untracked   RED. The report measured a file under the repo
                           root that `git ls-files` does not carry, so the
                           denominator disagrees with the git index. That is
                           #244's defect: the census and the gate must agree
                           about what the repository IS.
      OUT-OF-TREE          EXCLUDED from the verdict, but COUNTED and NAMED
                           in the output. These are real and legitimate: the
                           gate tests write probe modules into a pytest
                           tmpdir and import them, so every honest run
                           reports rows like
                           /private/var/.../pytest-of-<user>/.../clean_probe_gate.py
                           (three measured 2026-09-03). Calling those drift
                           would make this gate RED on every clean tree, and
                           a gate that is always red is a gate that gets
                           disabled.
    This is a SPLIT, not an allowlist: no path is ever named as an
    exception, both arms carry controls, and an out-of-tree row cannot
    launder an in-tree one -- a report carrying both is still RED.

EXIT CODES
    0   CLEAR -- every tracked module was measured inside
        [floor, floor + SLACK]; the per-module claim holds with every unit
        of its denominator accounted for.
    5   RED -- a module measured below its floor, a floor went stale (see
        SLACK), or the report measured an IN-TREE file git does not track.
    95  UNMEASURED -- no report, an unreadable or non-JSON report, an empty
        "files" mapping (zero modules is NOT "all modules pass"), a
        denominator git could not supply, or tracked modules the report
        never saw. UNMEASURED outranks RED: "we do not know" must not be
        reported as "we know it is broken", nor laundered into clean.
    96  REFUSE -- a --self-test control missed its declared outcome. A gate
        whose controls did not fire has not been certified and must not be
        trusted to report on anyone.
    Exit 1 is never used: the top level catches every exception and fails
    closed at 5 rather than leak a traceback-shaped exit code.

FLOORS AND SLACK
    Each module's floor is an integer percentage from FLOORS, defaulting to
    DEFAULT_FLOOR (90). A floor is not a trophy case: measuring ABOVE
    floor + SLACK (5 points) is RED -- "floor is stale; raise it" -- because
    a floor far below reality is a floor nothing can touch, which is how the
    old total sat at 70 while the tree was at 93. A gate that cannot fail is
    not a gate. `--update` rewrites the FLOORS block of THIS FILE in place
    from a real measurement (measured value rounded down; entries that would
    sit inside the default band are omitted so the table stays small enough
    to audit by eye), prints every change, and touches nothing outside the
    two sentinel lines. Floors are never invented by hand -- the table is
    empty until --update fills it from evidence -- and --update writes it in
    two labelled halves. Entries BELOW the default are DEBT, not policy:
    each one names a module this repository does not adequately test, and
    the floor's only job there is to stop the number sliding while the tests
    get written. Entries ABOVE the default band are the ratchet.

CONTROLS (--self-test)
    Twelve, run in a temp dir against synthetic in-memory reports with an
    INJECTED tracked set and an INJECTED repo root, so the controls are
    hermetic and cannot pass or fail because of what happens to be
    committed or where the checkout lives:
      MUST_FIRE  a module one point below its floor            ->  5
      MUST_FIRE  a module far above floor + SLACK (stale)      ->  5
      MUST_FIRE  a tracked module absent from the report       -> 95
      MUST_FIRE  an empty "files" object                       -> 95
      MUST_FIRE  a missing report file                         -> 95
      MUST_FIRE  a report file that is not JSON                -> 95
      MUST_FIRE  an IN-TREE report row git does not track      ->  5
      MUST_FIRE  an out-of-tree row ALONGSIDE an in-tree one   ->  5
      MUST_PASS  every module exactly AT its floor             ->  0
      MUST_PASS  every module at floor + SLACK exactly         ->  0
      MUST_PASS  an out-of-tree row on an otherwise clean tree ->  0
      MUST_PASS  an absolute path UNDER the repo root, tracked ->  0
    The first two MUST_PASS controls pin both band boundaries as INCLUSIVE.
    The last two pin the exclusion axis: one proves the tmpdir shape is
    excluded rather than RED, the other proves an absolute in-tree path is
    normalised onto its tracked relative name instead of counting twice
    (once as drift, once as UNMEASURED). Control 8 is the drill that keeps
    the exclusion from becoming a hole. If any control does not produce its
    declared outcome, the gate prints which one and exits 96: an
    uncertified verdict is a vacuous one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

# Every tracked module was measured and sits inside [floor, floor + SLACK]: the per-module
# claim actually holds, over the full denominator.
EXIT_CLEAR = 0
# A module measured below its floor, a floor went stale (see SLACK), or the report measured an
# IN-TREE file the git index does not carry -- the denominator drifted. All three are real
# defects. Out-of-tree rows (a test's own tmpdir scratch) are excluded and named, not RED.
EXIT_RED = 5
# The gate could not measure: no report, an unreadable report, an empty "files", no git
# denominator, or tracked modules the report never saw. Outranks RED -- "we do not know" must
# not be reported as "we know it is broken", nor as clean.
EXIT_UNMEASURED = 95
# A --self-test control did not produce its declared outcome. An uncertified gate has not
# earned the right to report on anyone.
EXIT_REFUSE = 96

DEFAULT_FLOOR: int = 90

# SLACK is the width of a floor's honest band. Over-performing a floor by more than SLACK is a
# DEFECT, not a bonus: a floor far below reality is a floor nothing can touch, which is exactly
# how the previous total sat at 70 while the tree was at 93. A gate that cannot fail is not a
# gate; `--update` pulls a stale floor back into contact with reality in one command.
SLACK: int = 5

REPORT_DEFAULT = "coverage.json"

# Two of these three ADJUDICATE: a live-save gate and a run-manifest emitter decide what counts
# as a run, so their own execution paths must be covered by the claim. The rest of tools/ is
# deliberately OUT of the claim this gate makes, and is omitted in pyproject.toml
# [tool.coverage.run] WITH its measured percentages rather than dropped in silence.
#
# tools/__init__.py is here because `--cov=tools` resolves the tools PACKAGE, so coverage emits
# a row for the package marker on every run. It is git-tracked and it is in CI's denominator;
# leaving it out of this tuple would classify a legitimate row as in-tree drift and RED the
# gate on a clean tree. It must be in the claim or in nothing, and it is in CI's claim.
TOOLS_IN_SCOPE: tuple[str, ...] = (
    "tools/__init__.py",
    "tools/emit_run_manifest.py",
    "tools/live_save_gate.py",
)

GIT_PATHSPECS: tuple[str, ...] = (
    "src/foundationscale/**/*.py",
    "src/foundationscale/*.py",
)

FLOORS_BEGIN = "# --- FLOORS: generated by --update; do not hand-edit values ---"
FLOORS_END = "# --- end FLOORS ---"

# No number in this table may be invented by hand. Every entry is written by `--update` from a
# real coverage measurement and then committed; an entry without a measurement behind it would
# be the same vacuity this gate exists to remove. --update renders the table in two labelled
# halves -- DEBT below the default floor, RATCHET above the default band -- so that a module
# which meets no standard cannot sit unremarked among the ones that do. This comment states the
# RULE and no values: the values live in exactly one place, generated, and cannot drift from it.
# --- FLOORS: generated by --update; do not hand-edit values ---
FLOORS: dict[str, int] = {
    # DEBT -- 5 module(s) BELOW the default floor of 90.
    # Each of these is the aggregate subsidy this gate exists to expose, now
    # stated with its denominator instead of averaged away. The floor is not an
    # endorsement: it is a ratchet that stops the module rotting further while the
    # tests are written. Delete a line when its module reaches the default; do not
    # add a line by hand, and do not lower one to make a red run green.
    "src/foundationscale/checkpoint/dcp_meta.py": 81,  # set by --update; measured 81.4%
    "src/foundationscale/train/__init__.py": 50,  # set by --update; measured 50.0%
    "src/foundationscale/train/__main__.py": 0,  # set by --update; measured 0.0%
    "src/foundationscale/train/loop.py": 78,  # set by --update; measured 78.4%
    "tools/emit_run_manifest.py": 80,  # set by --update; measured 80.9%
    # RATCHET -- 18 module(s) ABOVE the default band. These record what
    # the tree already achieves, so a regression to a merely-passing 90% is RED
    # rather than invisible.
    "src/foundationscale/__init__.py": 100,  # set by --update; measured 100.0%
    "src/foundationscale/checkpoint/__init__.py": 100,  # set by --update; measured 100.0%
    "src/foundationscale/gates/__init__.py": 100,  # set by --update; measured 100.0%
    "src/foundationscale/gates/core.py": 98,  # set by --update; measured 99.0%
    "src/foundationscale/gates/example.py": 96,  # set by --update; measured 96.6%
    "src/foundationscale/gates/fixtures.py": 98,  # set by --update; measured 98.6%
    "src/foundationscale/gates/objective_gates.py": 99,  # set by --update; measured 99.6%
    "src/foundationscale/gates/probe.py": 97,  # set by --update; measured 97.1%
    "src/foundationscale/integrate.py": 100,  # set by --update; measured 100.0%
    "src/foundationscale/models/__init__.py": 100,  # set by --update; measured 100.0%
    "src/foundationscale/models/adapters.py": 97,  # set by --update; measured 97.8%
    "src/foundationscale/provenance/__init__.py": 100,  # set by --update; measured 100.0%
    "src/foundationscale/topology.py": 97,  # set by --update; measured 97.4%
    "src/foundationscale/train/cli.py": 97,  # set by --update; measured 97.1%
    "src/foundationscale/verify/__init__.py": 100,  # set by --update; measured 100.0%
    "src/foundationscale/verify/parity.py": 99,  # set by --update; measured 99.1%
    "tools/__init__.py": 100,  # set by --update; measured 100.0%
    "tools/live_save_gate.py": 98,  # set by --update; measured 98.5%
}
# --- end FLOORS ---


class GateArgumentParser(argparse.ArgumentParser):
    def exit(self, status: int = 0, message: str | None = None) -> None:
        # argparse exits 2 on bad arguments; without this remap an operator typo would escape
        # the four-code namespace this gate advertises to CI.
        if message:
            sys.stderr.write(message)
        raise SystemExit(EXIT_CLEAR if status == 0 else EXIT_REFUSE)


class ModuleOutcome(NamedTuple):
    # measured/missing are None when the report never saw the module: absent evidence is kept
    # as None so no code path can mistake "not measured" for 0% or for 100%.
    path: str
    status: str  # "clear" | "red-below" | "red-stale" | "unmeasured"
    measured: float | None
    floor: int
    missing: int | None


class GateResult(NamedTuple):
    # drift and excluded are the two halves of "the report measured something the tracked set
    # does not carry", split on WHERE the file lives. drift is RED (the git index disagrees with
    # the denominator); excluded is a test's own out-of-tree scratch, kept as a named list so a
    # dropped row is reported rather than vanishing.
    exit_code: int
    denominator: int
    src_count: int
    tools_count: int
    outcomes: list[ModuleOutcome]
    drift: list[str]
    excluded: list[str]
    load_error: str | None
    files_empty: bool


def percent_of(entry: dict[str, Any]) -> float | None:
    summary = entry.get("summary")
    if not isinstance(summary, dict):
        return None
    raw = summary.get("percent_covered")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def missing_count_of(entry: dict[str, Any]) -> int | None:
    # The file-level "missing_lines" list is preferred; the summary's integer count is the
    # fallback. Neither present means the count is UNKNOWN (None), not zero -- under-reporting
    # missing lines would soften every RED message that quotes it.
    file_level = entry.get("missing_lines")
    if isinstance(file_level, list):
        return len(file_level)
    summary = entry.get("summary")
    if isinstance(summary, dict):
        count = summary.get("missing_lines")
        if isinstance(count, int) and not isinstance(count, bool):
            return count
    return None


def locate_report_key(key: str, repo_root: Path) -> tuple[bool, str]:
    """Return (in_tree, normalised_path) for one report key.

    Purely lexical: no resolve(), no stat, nothing that reads the filesystem, so the verdict
    depends only on the report and the injected root. An absolute key UNDER the root is
    normalised onto its repository-relative name -- coverage can emit either form, and comparing
    the absolute one against a git-relative tracked set would mint a drift finding AND an
    UNMEASURED finding for the same file, out of a path notation.
    """
    normalised = os.path.normpath(key).replace("\\", "/")
    candidate = Path(normalised)
    if not candidate.is_absolute():
        return True, normalised
    if candidate.is_relative_to(repo_root):
        return True, candidate.relative_to(repo_root).as_posix()
    return False, normalised


def evaluate(
    files: dict[str, Any], tracked: Sequence[str], floors: Mapping[str, int], repo_root: Path
) -> tuple[list[ModuleOutcome], list[str], list[str]]:
    """Classify every tracked module; return (outcomes, drift, excluded).

    The tracked set is an ARGUMENT, never re-derived here: the verdict must claim about the
    repository's own file list in production, while --self-test feeds a synthetic one. A gate
    whose denominator can be silently swapped cannot be controlled, and an uncontrolled gate is
    an unmeasured axis wearing a verdict. repo_root is an argument for the same reason.

    Report keys the tracked set does not carry are split rather than lumped. An IN-TREE
    untracked row is drift and is RED: the report and the git index disagree about what this
    repository contains. An OUT-OF-TREE row is a test's own scratch -- the gate suite writes
    probe modules into a pytest tmpdir and imports them, which coverage duly measures -- and is
    excluded from the verdict but returned by name, because a silently dropped row reads exactly
    like a row that was never there.
    """
    tracked_set = set(tracked)
    drift: list[str] = []
    excluded: list[str] = []
    reported: dict[str, Any] = {}
    for key, entry in files.items():
        in_tree, normalised = locate_report_key(str(key), repo_root)
        if not in_tree:
            excluded.append(normalised)
            continue
        if normalised not in tracked_set:
            drift.append(normalised)
        reported[normalised] = entry
    drift.sort()
    excluded.sort()
    files = reported
    outcomes: list[ModuleOutcome] = []
    for path in tracked:
        floor = floors.get(path, DEFAULT_FLOOR)
        entry = files.get(path)
        measured: float | None = None
        missing: int | None = None
        if isinstance(entry, dict):
            measured = percent_of(entry)
            missing = missing_count_of(entry)
        if measured is None:
            # Absent from the report, or present but keyless: UNMEASURED, never assumed clean.
            # A new module no test imports must not pass by being invisible.
            outcomes.append(ModuleOutcome(path, "unmeasured", None, floor, missing))
        elif measured < floor:
            outcomes.append(ModuleOutcome(path, "red-below", measured, floor, missing))
        elif measured > floor + SLACK:
            # RED, not celebration: the floor has lost contact with the module it constrains.
            outcomes.append(ModuleOutcome(path, "red-stale", measured, floor, missing))
        else:
            outcomes.append(ModuleOutcome(path, "clear", measured, floor, missing))
    return outcomes, drift, excluded


def measure(
    report_path: str, tracked: Sequence[str], floors: Mapping[str, int], repo_root: Path
) -> GateResult:
    """Pure verdict: reads the report, classifies every tracked module, never prints.

    All printing lives in emit_human / emit_json so that --self-test can assert on verdicts
    over synthetic reports without scraping banner text.
    """
    src_count = sum(1 for p in tracked if p.startswith("src/"))
    tools_count = len(tracked) - src_count
    denominator = len(tracked)
    base = GateResult(
        exit_code=EXIT_UNMEASURED,
        denominator=denominator,
        src_count=src_count,
        tools_count=tools_count,
        outcomes=[],
        drift=[],
        excluded=[],
        load_error=None,
        files_empty=False,
    )
    if denominator == 0:
        # Same guard as the empty-files case, one layer up: a gate claiming about zero modules
        # is vacuous whatever the report says.
        return base._replace(
            load_error="the tracked set is empty; the gate would claim about nothing"
        )
    path = Path(report_path)
    if not path.is_file():
        return base._replace(
            load_error=f"report '{report_path}' does not exist; the gate could not measure"
        )
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return base._replace(
            load_error=f"report '{report_path}' is not usable JSON ({exc}); could not measure"
        )
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        return base._replace(
            load_error=f"report '{report_path}' carries no usable 'files' key; could not measure"
        )
    # Coverage on Windows can emit backslash paths; git never does. Comparing un-normalised keys
    # would mint drift findings out of path separators -- a defect of the comparator, not the tree.
    files: dict[str, Any] = {str(k).replace("\\", "/"): v for k, v in data["files"].items()}
    if not files:
        # THE anti-vacuity guard: all([]) is True, and an empty files mapping would sail through
        # every per-module loop unopposed. Zero modules measured is not "all modules pass".
        return base._replace(files_empty=True)
    outcomes, drift, excluded = evaluate(files, tracked, floors, repo_root)
    statuses = {o.status for o in outcomes}
    if "unmeasured" in statuses:
        # "We do not know" outranks RED: it must not be reported as broken, and never as clean.
        code = EXIT_UNMEASURED
    elif drift or "red-below" in statuses or "red-stale" in statuses:
        # `excluded` is deliberately absent from this condition and from the one above: an
        # out-of-tree row is not evidence about this repository either way. It cannot make the
        # verdict RED, and -- the half that matters -- it cannot soften one, because drift is
        # tested here on its own.
        code = EXIT_RED
    else:
        code = EXIT_CLEAR
    return GateResult(
        code, denominator, src_count, tools_count, outcomes, drift, excluded, None, False
    )


def discover_tracked() -> list[str] | None:
    """The denominator comes from the REPOSITORY, not the report: a module that no test ever
    imports is invisible to coverage, and invisible must never mean clean."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", *GIT_PATHSPECS],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        print(f"UNMEASURED: cannot run git to derive the denominator: {exc}")
        return None
    if proc.returncode != 0:
        print(
            "UNMEASURED: git ls-files refused "
            f"({proc.stderr.strip() or 'no stderr'}); the denominator is unknown"
        )
        return None
    paths = {line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()}
    return sorted(paths | set(TOOLS_IN_SCOPE))


def discover_repo_root() -> Path:
    """The boundary that separates drift (RED) from a test's own scratch (excluded).

    git decides it, not the process's cwd, so running the gate from a subdirectory cannot move
    the line between the two arms. cwd is the fallback only for the case where git is already
    unavailable -- and there discover_tracked() has already returned None, so the run is
    UNMEASURED before this value is used for anything.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return Path.cwd()
    top = proc.stdout.strip()
    if proc.returncode != 0 or not top:
        return Path.cwd()
    return Path(os.path.normpath(top))


def emit_human(result: GateResult) -> None:
    print(
        f"DENOMINATOR: {result.denominator} modules in scope "
        f"({result.src_count} from src/, {result.tools_count} declared tools/)"
    )
    if result.load_error is not None:
        print(f"UNMEASURED: {result.load_error}; this is NOT a pass")
        return
    if result.files_empty:
        print(
            'UNMEASURED: the report\'s "files" object is EMPTY -- zero modules were measured, '
            'and zero modules is not "all modules pass"; this is NOT a pass'
        )
        return
    for outcome in result.outcomes:
        missing_txt = (
            f"{outcome.missing} missing line(s)"
            if outcome.missing is not None
            else "missing-line count not recorded"
        )
        pct = outcome.measured if outcome.measured is not None else float("nan")
        if outcome.status == "red-below":
            print(
                f"RED below-floor {outcome.path}: measured {pct:.1f}% < floor {outcome.floor} "
                f"-- {missing_txt}"
            )
        elif outcome.status == "red-stale":
            print(
                f"RED stale-floor {outcome.path}: measured {pct:.1f}% > floor {outcome.floor} "
                f"+ slack {SLACK} -- the floor is stale; raise it (`--update` fixes this in one "
                "command). A floor nothing can touch is a gate that cannot fail"
            )
        elif outcome.status == "unmeasured":
            print(
                f"UNMEASURED {outcome.path}: tracked by git but absent from (or keyless in) the "
                "report -- a module no test imports must not pass by being invisible"
            )
    for path in result.drift:
        print(
            f"DRIFT {path}: measured by the report, under the repository root, and NOT carried "
            "by the git index -- the denominator and the tree disagree; refusing to ignore it"
        )
    # Named, never merely counted: an excluded row that no one can see is indistinguishable from
    # a row that was silently dropped, and the whole point of the split is that it is auditable.
    for path in result.excluded:
        print(
            f"EXCLUDED {path}: outside the repository root -- a test's own scratch, not evidence "
            "about this tree; excluded from the verdict and reported rather than dropped"
        )
    red_ones = [o for o in result.outcomes if o.status in ("red-below", "red-stale")]
    unseen = [o for o in result.outcomes if o.status == "unmeasured"]
    if unseen:
        print(
            f"COVERAGE-FLOOR UNMEASURED: {len(unseen)} of {result.denominator} tracked module(s) "
            "were never measured; 'we do not know' outranks RED and is reported neither as "
            "broken nor as clean -- this is NOT a pass"
        )
    elif red_ones or result.drift:
        print(
            f"COVERAGE-FLOOR RED: {len(red_ones)} floor finding(s), "
            f"{len(result.drift)} drifted report entrie(s)"
        )
    else:
        floors_used = [o.floor for o in result.outcomes]
        print(
            f"COVERAGE-FLOOR CLEAR: {result.denominator}/{result.denominator} modules measured "
            f"inside [floor, floor+{SLACK}]; floors span {min(floors_used)}..{max(floors_used)} "
            f"(default {DEFAULT_FLOOR}); {len(result.excluded)} out-of-tree report row(s) "
            "excluded and named above"
        )


def emit_json(result: GateResult) -> None:
    verdict = {0: "clear", 5: "red", 95: "unmeasured", 96: "refuse"}.get(
        result.exit_code, "unknown"
    )
    payload = {
        "exit_code": result.exit_code,
        "verdict": verdict,
        "denominator": {
            "total": result.denominator,
            "src": result.src_count,
            "tools": result.tools_count,
        },
        "load_error": result.load_error,
        "files_empty": result.files_empty,
        "modules": [
            {
                "path": o.path,
                "status": o.status,
                "measured": o.measured,
                "floor": o.floor,
                "missing_lines": o.missing,
            }
            for o in result.outcomes
        ],
        "drift": result.drift,
        "excluded_out_of_tree": result.excluded,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def compute_updated_floors(outcomes: Sequence[ModuleOutcome]) -> tuple[dict[str, int], list[str]]:
    """Build the new floor table from a real measurement.

    An entry is only worth writing when the measured value, rounded DOWN, lands outside
    [DEFAULT_FLOOR, DEFAULT_FLOOR + SLACK]; inside that band the default already tells the
    truth, and a table that restates the default for every module stops being auditable.
    Unmeasured modules are skipped (never floor-typed from thin air) and returned for reporting.
    """
    floors: dict[str, int] = {}
    skipped: list[str] = []
    for outcome in outcomes:
        if outcome.measured is None:
            skipped.append(outcome.path)
            continue
        candidate = math.floor(outcome.measured)
        if DEFAULT_FLOOR <= candidate <= DEFAULT_FLOOR + SLACK:
            continue
        floors[outcome.path] = candidate
    return floors, skipped


def render_floors_block(floors: Mapping[str, int], measured: Mapping[str, float]) -> list[str]:
    """Render the table in two LABELLED halves, both generated, neither hand-written.

    An entry below DEFAULT_FLOOR and an entry above it are not the same kind of fact, and a flat
    table hides that. `"train/__main__.py": 0` sitting unlabelled among the 100s reads as a
    standard this repository holds; it is the opposite -- a module that meets no standard, whose
    floor exists only to stop it rotting further. Splitting them puts the debt where a reader
    trips over it, with its own count, and every number still comes from --update.
    """
    if not floors:
        return ["FLOORS: dict[str, int] = {}\n"]
    debt = sorted(p for p in floors if floors[p] < DEFAULT_FLOOR)
    ratchet = sorted(p for p in floors if floors[p] >= DEFAULT_FLOOR)
    lines = ["FLOORS: dict[str, int] = {\n"]
    if debt:
        lines.append(
            f"    # DEBT -- {len(debt)} module(s) BELOW the default floor of {DEFAULT_FLOOR}.\n"
            "    # Each of these is the aggregate subsidy this gate exists to expose, now\n"
            "    # stated with its denominator instead of averaged away. The floor is not an\n"
            "    # endorsement: it is a ratchet that stops the module rotting further while the\n"
            "    # tests are written. Delete a line when its module reaches the default; do not\n"
            "    # add a line by hand, and do not lower one to make a red run green.\n"
        )
        for path in debt:
            lines.append(
                f"    {json.dumps(path)}: {floors[path]},  # set by --update; measured "
                f"{measured[path]:.1f}%\n"
            )
    if ratchet:
        if debt:
            lines.append("\n")
        lines.append(
            f"    # RATCHET -- {len(ratchet)} module(s) ABOVE the default band. These record "
            "what\n    # the tree already achieves, so a regression to a merely-passing "
            f"{DEFAULT_FLOOR}% is RED\n    # rather than invisible.\n"
        )
        for path in ratchet:
            lines.append(
                f"    {json.dumps(path)}: {floors[path]},  # set by --update; measured "
                f"{measured[path]:.1f}%\n"
            )
    lines.append("}\n")
    return lines


def apply_update(result: GateResult, self_path: Path) -> int:
    """Rewrite ONLY the FLOORS block of this file, between the two sentinel lines.

    Floors derive from measurement or they derive from nothing; an update that cannot see a
    real report refuses rather than guess.
    """
    if result.load_error is not None or result.files_empty:
        print(
            "UPDATE UNMEASURED: cannot write floors from a report that could not be measured; "
            "the table is untouched"
        )
        return EXIT_UNMEASURED
    floors, skipped = compute_updated_floors(result.outcomes)
    measured = {o.path: o.measured for o in result.outcomes if o.measured is not None}
    try:
        text_lines = self_path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as exc:
        print(f"UPDATE RED: cannot read {self_path}: {exc}")
        return EXIT_RED
    begin = [i for i, line in enumerate(text_lines) if line.rstrip("\n") == FLOORS_BEGIN]
    end = [i for i, line in enumerate(text_lines) if line.rstrip("\n") == FLOORS_END]
    if len(begin) != 1 or len(end) != 1 or begin[0] >= end[0]:
        print(
            f"UPDATE RED: the FLOORS sentinel block in {self_path} is damaged (need each "
            "sentinel exactly once, begin before end); refusing to rewrite blind"
        )
        return EXIT_RED
    new_lines = (
        text_lines[: begin[0] + 1] + render_floors_block(floors, measured) + text_lines[end[0] :]
    )
    try:
        self_path.write_text("".join(new_lines), encoding="utf-8")
    except OSError as exc:
        print(f"UPDATE RED: cannot write {self_path}: {exc}")
        return EXIT_RED
    for path in skipped:
        print(f"UPDATE skip {path}: unmeasured in this report; no floor written from no data")
    for path in sorted(set(FLOORS) | set(floors)):
        if path not in FLOORS:
            print(f"UPDATE added {path}: floor {floors[path]}")
        elif path not in floors:
            print(
                f"UPDATE removed {path}: measured value now inside "
                f"[{DEFAULT_FLOOR}, {DEFAULT_FLOOR + SLACK}]; the default covers it"
            )
        elif FLOORS[path] != floors[path]:
            print(f"UPDATE moved {path}: {FLOORS[path]} -> {floors[path]}")
    if floors == FLOORS:
        print("UPDATE: the report already sits inside every band; the table is unchanged")
    print(f"UPDATE ok: rewrote the FLOORS block in {self_path}; {len(floors)} entrie(s)")
    return EXIT_CLEAR


def run_self_test() -> int:
    """Drive twelve controls through measure() against synthetic reports in a temp dir.

    The tracked set and the repo root are both INJECTED, never discovered from the real git
    tree, so the controls are hermetic: they cannot pass or fail because of what happens to be
    committed or where this checkout lives. Each control is labelled and asserted against its
    declared exit code; a misfire means every verdict this gate returns is worthless until fixed.
    """
    tracked = ["src/foundationscale/alpha.py", "src/foundationscale/beta.py"]
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="covfloor-selftest-") as tmp:
        report_path = Path(tmp) / "coverage.json"
        # A synthetic root and a sibling that is NOT under it. Neither needs to exist: the
        # in-tree/out-of-tree split is lexical by construction, which is what makes it testable
        # without touching a filesystem that could differ between machines.
        root = Path(os.path.normpath(str(Path(tmp) / "repo")))
        outside = (Path(tmp) / "pytest-of-someone" / "probe_gate.py").as_posix()

        def entry(pct: float, missing: int) -> dict[str, Any]:
            return {
                "summary": {
                    "num_statements": 100,
                    "missing_lines": missing,
                    "percent_covered": pct,
                },
                "missing_lines": list(range(1, missing + 1)),
            }

        def write_files(files: dict[str, Any]) -> None:
            report_path.write_text(json.dumps({"files": files}), encoding="utf-8")

        def check(label: str, expected: int) -> None:
            got = measure(str(report_path), tracked, {}, root).exit_code
            behaved = got == expected
            print(
                f"control {label}: wanted exit {expected}, got {got} -- "
                + ("behaved" if behaved else "MISFIRED")
            )
            if not behaved:
                failures.append(label)

        alpha = "src/foundationscale/alpha.py"
        beta = "src/foundationscale/beta.py"

        write_files({alpha: entry(89.0, 11), beta: entry(92.0, 8)})
        check("MUST_FIRE a module one point below its floor -> EXIT_RED", EXIT_RED)

        write_files({alpha: entry(99.0, 1), beta: entry(92.0, 8)})
        check("MUST_FIRE a module far above floor+SLACK (stale floor) -> EXIT_RED", EXIT_RED)

        write_files({alpha: entry(92.0, 8)})
        check("MUST_FIRE a tracked module absent from the report -> EXIT_UNMEASURED", 95)

        write_files({})
        check('MUST_FIRE an empty "files" object -> EXIT_UNMEASURED', EXIT_UNMEASURED)

        if report_path.exists():
            report_path.unlink()
        check("MUST_FIRE a missing report file -> EXIT_UNMEASURED", EXIT_UNMEASURED)

        report_path.write_text("this is not json {", encoding="utf-8")
        check("MUST_FIRE a report file that is not JSON -> EXIT_UNMEASURED", EXIT_UNMEASURED)

        write_files({alpha: entry(92.0, 8), beta: entry(92.0, 8), "src/rogue.py": entry(100.0, 0)})
        check("MUST_FIRE an IN-TREE report row git does not track -> EXIT_RED", EXIT_RED)

        # THE DRILL for the exclusion. The out-of-tree arm exists so a pytest tmpdir row cannot
        # RED a clean tree; this asserts it cannot LAUNDER a dirty one either. Both shapes are
        # present at once, and the in-tree row must still decide the verdict. Without this
        # control the split would be an untested exception, which is the shape it replaced.
        write_files(
            {
                alpha: entry(92.0, 8),
                beta: entry(92.0, 8),
                outside: entry(100.0, 0),
                "src/rogue.py": entry(100.0, 0),
            }
        )
        check("MUST_FIRE an out-of-tree row ALONGSIDE an in-tree one -> EXIT_RED", EXIT_RED)

        write_files({alpha: entry(90.0, 10), beta: entry(90.0, 10)})
        check("MUST_PASS every module exactly AT its floor (inclusive) -> EXIT_CLEAR", EXIT_CLEAR)

        write_files({alpha: entry(95.0, 5), beta: entry(95.0, 5)})
        check("MUST_PASS every module at floor+SLACK exactly (inclusive) -> EXIT_CLEAR", 0)

        # The real report carries exactly this shape on every honest run: the gate suite's own
        # probe modules, written into a pytest tmpdir and imported, three of them on 2026-09-03.
        write_files({alpha: entry(92.0, 8), beta: entry(92.0, 8), outside: entry(100.0, 0)})
        check("MUST_PASS an out-of-tree row on an otherwise clean tree -> EXIT_CLEAR", EXIT_CLEAR)

        # An absolute path UNDER the root is the SAME file as its tracked relative name. Read
        # literally it would be drift AND leave alpha unmeasured -- two findings minted out of a
        # path notation, and UNMEASURED outranks RED, so the run would report 95 on a clean tree.
        write_files({(root / alpha).as_posix(): entry(92.0, 8), beta: entry(92.0, 8)})
        check("MUST_PASS an absolute path UNDER the repo root, tracked -> EXIT_CLEAR", EXIT_CLEAR)

    total = 12
    if failures:
        print(
            f"SELF-TEST DENOMINATOR: {total - len(failures)} of {total} controls behaved; "
            "detector verdicts are worthless until every control fires as declared"
        )
        for failure in failures:
            print("CONTROL DEFECT: " + failure)
        return EXIT_REFUSE
    print(
        f"SELF-TEST DENOMINATOR: {total} of {total} controls behaved; 8x MUST_FIRE produced the "
        "declared nonzero exits, 4x MUST_PASS pinned both band boundaries as inclusive and both "
        "arms of the in-tree/out-of-tree split"
    )
    return EXIT_CLEAR


def build_parser() -> argparse.ArgumentParser:
    parser = GateArgumentParser(
        prog="coverage_floor",
        description="Per-module coverage floor: --cov-fail-under is a total, and a total can "
        "be subsidised. This gate makes the per-module claim the total cannot.",
    )
    parser.add_argument(
        "--report",
        default=REPORT_DEFAULT,
        help="coverage JSON report to adjudicate (default: %(default)s)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite this file's FLOORS block in place from the current report",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the MUST_FIRE/MUST_PASS controls in a temp dir and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a machine-readable summary instead of the human report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    tracked = discover_tracked()
    if tracked is None:
        print(
            "COVERAGE-FLOOR UNMEASURED: the denominator could not be derived from git -- "
            "a gate that does not know what it claims about cannot pass"
        )
        return EXIT_UNMEASURED
    result = measure(args.report, tracked, FLOORS, discover_repo_root())
    if args.update:
        return apply_update(result, Path(__file__))
    if args.json:
        emit_json(result)
    else:
        emit_human(result)
    return result.exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 -- fail closed, never exit 1
        print(
            f"COVERAGE-FLOOR RED: unexpected exception escaped main(): "
            f"{type(exc).__name__}: {exc} -- this gate fails closed rather than risk a "
            "traceback exit code outside the 0/5/95/96 namespace"
        )
        sys.exit(EXIT_RED)
