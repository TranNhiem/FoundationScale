#!/usr/bin/env python3
"""Mutation battery for the FoundationScale verification framework.

A green test suite is not evidence. `243 passed` is exactly the shape of output
the audited estate produced for 472 steps at grad_norm 0.000 — a confident
report from a check that could not have said anything else. So before this
suite is allowed to count as verification, each load-bearing rule is
individually broken and the suite must notice. A mutation that survives is a
rule with no test behind it.

TWO PRECONDITIONS, both of which are this project's own failure mode turned
inward:

1. The suite must be GREEN before we start, or a survivor means nothing: a
   mutant "killed" by a test that was failing anyway says nothing about
   coverage.

2. The baseline must report ZERO SKIPPED TESTS. This is not fussiness. An
   earlier version of this file ran the suite under an interpreter with no
   torch installed, where every checkpoint test skipped. A mutation in
   `dcp.py` could not be killed there no matter how wrong it was, and the
   battery reported it as a surviving mutant: an alarm from a detector never
   connected to the thing it watches. A skipped test is the vacuous-success
   shape, and a battery whose baseline is full of them is measuring nothing
   while printing a reassuring column of `[killed]`. This is a hard failure —
   the battery exits nonzero (2), not a warning. Do not soften it.

   And say so plainly, because it is the most useful thing this file
   documents: this is the SAME defect that was found independently, at the
   same time, in this repository's own CI — 41 tests skipped, pipeline
   green. Two detectors disconnected in two different places, both reporting
   success. The failure this tool exists to catch is not hypothetical and
   not behind us; a suite whose skips nobody adds up drifts back into it by
   default.

ANCHOR DISCIPLINE

Each mutant is applied by replacing one anchor string in the real source. An
anchor that matches other than exactly once is REFUSED and reported as
`not applicable` — never silently skipped, never guessed at — because a
mutation landing in the wrong function tells you nothing about the rule you
meant to break. A refused anchor is not a passing one, and it must never be
able to look like one: ANY `n/a` row makes this file exit nonzero, whether it
is the whole table or one row of forty-two. The partial case is the one that
actually happened here — a run printed `40 killed, 0 alive, 2 n/a` and exited
0, claiming forty-two rules while measuring forty. A stale anchor means the
table no longer describes the module (usually because the module was fixed
underneath it), so re-derive it against the current source; deleting the row
would shrink the battery silently, which is the same trade. Anchors are
validated once when the table is assembled and re-checked here, row by row, at
run time.

BACKUP / RESTORE

Every module under test is backed up BEFORE any of them is touched, all are
restored in a `finally`, and each restore is verified byte-for-byte.
Restoring only the file being mutated when an exception fires would leave an
earlier module mutated on disk — a corrupted tree that looks perfectly fine.
A crash mid-run must not leave a mutant in the tree.

KILL CRITERIA

A mutant is never scored on pytest's exit code alone. rc=2 (collection
error), a bare rc=1 that names no failing test, a signal, and a hang are
all indistinguishable from detection if the return code is all you read —
and one flaky red run is indistinguishable from a real one. So a kill
requires two things: pytest exited 1 with the failing tests named in the
junit report, and a SECOND run with the mutant still applied reproduced
exactly the same set. Every other shape is UNSCORED: the battery claims
nothing about that rule and exits 2. Each run is bounded by
FS_MUTATE_SUITE_TIMEOUT (default 1800s) so a mutant that makes the suite
loopy costs one unscored trial instead of hanging until CI's SIGKILL
bypasses the finally-restore and strands the mutant in the tree.

MUST-PASS CONTROL

A detector is two claims (doctrine 3). The mutation table supplies the
MUST_FIRE half; it must also carry rows flagged "must_survive": inert,
behaviour-preserving edits — a trailing comment — that the suite MUST let
pass. Exactly such a control once scored KILLED in this battery, attributed
to a meta-test that failed only because the outer mutant had invalidated the
inner battery's live-anchor check: detection with no fault behind it. Since
then the control is first-class — reported on its own line EVERY run,
including runs whose table carries no control row at all: the line there
reads "none configured", stated in words with its cost, and the run voids
(exit 2). It is excluded from the caught= denominator and the killed/alive
tallies, and if it ever dies the whole run voids (exit 2), because
attribution that fires on a no-op cannot be believed about anything else
either. A control that was configured but never exercised — stale anchor,
unscored trial — reports as a third state, counted from the table and
listed alongside NOT APPLICABLE / UNSCORED: never a pass, never an absence.

EXIT STATUS

  0  every mutation in the table was applied and scored, and every one
     was killed on two identical attributed runs
  1  at least one mutant survived — the suite has a gap. A survivor
     outranks a refused anchor: a run with both exits 1, not 2, because a
     live survivor is direct evidence against the suite, whereas a stale
     anchor only says the table drifted underneath the module — the more
     informative state wins. Only an UNSCORED trial outranks a survivor:
     a battery that could not measure may claim nothing, not even "the
     suite has a gap".
  2  preconditions failed (red suite, skipped tests, missing/empty table,
     module unreadable), any mutation could not be applied (provided no
     mutant survived — a survivor takes exit 1; see above), any trial
     was UNSCORED (timeout, collection/usage error, red run naming no
     test, a green run whose junit records zero executed tests,
     attribution that did not reproduce) — for that part of the
     table, we never measured — a MUST-PASS control was reported KILLED,
     which proves the attribution machinery unsound and voids every other
     measurement the run printed, or the table carried ZERO MUST-PASS
     control rows: with the negative half of the detector never wired up,
     every caught= figure the run printed is an unverified claim, and
     claiming from silence is "we never measured" one level up; or the
     table carried ZERO MUST-FIRE rows — every row a control, so the
     positive half fired zero times and exit 0 there is 0 killed of
     0 fired: the 0/0 success in exit-code form

CI depends on the difference between 1 and 2: "the suite has a gap" and
"we never measured" are different states and must not share an exit code.

LAYOUT

This file lives at tools/mutate.py and finds the repository from its own
location — no absolute paths, no pinned interpreter, no usernames. Run it
with the Python you intend to verify under:

    python tools/mutate.py

`sys.executable` runs the suite, and precondition 2 is the tripwire that
proves that interpreter can actually see the optional dependencies the tests
need — the refusal, not a blessed path, is the guard. `mutations.json` is
committed beside this file. Run with `--help` for what a surviving mutant
means and what to do about it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NoReturn

# tools/mutate.py -> tools/ -> repository root. Nothing in this file is allowed
# to contain a machine-specific path: the earlier incarnation pinned one
# developer's absolute checkout and venv, which is exactly the "works on the box
# that wrote it" assumption the audit was about.
ROOT = Path(__file__).resolve().parent.parent
TOOLS = Path(__file__).resolve().parent
SRC = ROOT / "src" / "foundationscale"

# The mutation table is committed alongside this tool. A battery without its
# table is not a battery that found nothing — see load_table().
TABLE = TOOLS / "mutations.json"

# Run the suite with the interpreter running THIS FILE. There is deliberately
# no pinned venv here: the contract is "run tools/mutate.py with the Python you
# intend to verify under", and precondition 2 — zero skipped tests — is the
# tripwire that proves that interpreter can see the project's optional
# dependencies. Under a torch-less interpreter every checkpoint test skips and
# the battery refuses to run.
PY = sys.executable

# Values are TREE-root-relative, not SRC-relative: every src module restates
# its src/foundationscale prefix so that a file OUTSIDE the package can sit in
# the same table under one format. The previous SRC-relative shape made tools/
# literally unexpressible, which is how tools/live_save_gate.py — the tool that
# adjudicates the FIRST REAL SAVE of the real training run — spent its whole
# life with zero mutation coverage while the tally printed caught= over only
# what the table could name. SRC stays as the canonical package-root handle;
# nothing resolves against it anymore, and both resolution sites (load_table
# and main) join at ROOT with no per-module special-casing.
MODULE_PATHS = {
    "core": "src/foundationscale/gates/core.py",
    "dcp": "src/foundationscale/checkpoint/dcp.py",
    "manifest": "src/foundationscale/provenance/manifest.py",
    "topology": "src/foundationscale/topology.py",
    "parity": "src/foundationscale/verify/parity.py",
    "checkpoint_gates": "src/foundationscale/gates/checkpoint_gates.py",
    "controls": "src/foundationscale/gates/controls.py",
    "live_save_gate": "tools/live_save_gate.py",
    # #85: the emission adjudicator becomes nameable by mutation rows.
    "emit_run_manifest": "tools/emit_run_manifest.py",
}

_REQUIRED_KEYS = ("name", "what", "anchor", "replacement")

# tools/emit_run_manifest.py mutants (#85). Rows carry exactly _REQUIRED_KEYS;
# binding to the module is by anchor uniqueness inside the file, which is why
# MODULE_PATHS names the file first. The list carries 7 MUST-FIRE rows.
# KNOWN, re-observed on every test run: tests/test_emit_run_manifest.py pins
# shape, keys, and per-row anchor uniqueness, over however many rows the
# list carries. EXPECTED, not yet measured by the battery: the five rows
# whose behaviors that suite exercises directly should be killed; and the
# two main()-level rows (lora-count-fabricated, drill-never-arms) --
# survivors with no test behind them when this caveat was first written --
# should now be killed as well, because both main()-level legs have since
# been repaired: they had been dying on missing argparse flags before ever
# reaching main(), and a two-arm drill leg now pins EXIT_REFUSED and
# EXIT_UNMEASURED separately. EXPECTED is not MEASURED: the next battery
# run is what settles those two rows, one way or the other, and they stay
# published either way -- a row dropped to protect an exit code is the
# worst outcome available here.
# Publication caveat, updated: from the day this list was planted, its
# battery denominator was 0 -- mutations.json never carried the rows and
# nothing refused. This repair closes both flanks: load_table publishes
# the list by MERGE (EMBEDDED_TABLE: these 7 rows + 1 inert must-pass
# control = a module denominator of 8; the constant stays the single
# source and any copy appearing in the JSON is itself a refusal), and
# _validate_table(complete=True) makes ANY mapped-but-unpublished module
# a die() that names it with its expected row count. A denominator of 0
# for this module is no longer a silence; it is a refusal -- red, never
# green.
EMIT_RUN_MANIFEST_ROWS = [
    {
        "name": "emit_run_manifest.lora-zip-unstrict",
        "what": (
            "zip(..., strict=True) is the only length-pin between the "
            "producer and the on-disk verifier; strict=False lets a "
            "drifted keys tuple silently truncate the five-field record"
        ),
        "anchor": "    return list(zip(_LORA_ABSTENTION_RECORD_KEYS, values, strict=True))",
        "replacement": "    return list(zip(_LORA_ABSTENTION_RECORD_KEYS, values, strict=False))",
    },
    {
        "name": "emit_run_manifest.lora-status-not-abstained",
        "what": (
            "declared.status is the field the on-disk predicate reads "
            "to call this null a DECLARED state; swapping the literal "
            "makes an abstention serialize as a non-abstention"
        ),
        # Widened 2026-08-24: the #79 full-ft repair added a SECOND
        # five-field abstention record whose value list opens with the same
        # literal (emit_run_manifest.py:981), so the bare '"abstained",' line
        # stopped being unique and the row's own validity leg went red. The
        # following line is the LoRA record's WHO constant, which the full-ft
        # record does not carry -- that is what pins this row to the LoRA
        # producer. The full-ft twin at :981 is covered by the NEXT row, added
        # in the same edit that recorded the gap -- annexing it here would have
        # let one row claim two producers on one observation.
        "anchor": '        "abstained",\n        _LORA_ABSTENTION_WHO,',
        "replacement": '        "present",\n        _LORA_ABSTENTION_WHO,',
    },
    {
        "name": "emit_run_manifest.full-ft-status-not-abstained",
        "what": (
            "the full-ft expected_expert_bytes abstention has its own "
            "status literal (emit_run_manifest.py:981), a SECOND producer "
            "of the same declared state; swapping it makes the byte "
            "gate's None read as a populated denominator on disk"
        ),
        # Disjoint from the LoRA row above by construction: that anchor is
        # pinned by _LORA_ABSTENTION_WHO on the following line, this one by
        # the reason string the full-ft record opens with. Neither anchor can
        # match the other's site, so the two producers are two measurements.
        "anchor": (
            '        "abstained",\n'
            '        (\n'
            '            "pricing expert bytes requires knowing which FQN pattern the byte "'
        ),
        "replacement": (
            '        "present",\n'
            '        (\n'
            '            "pricing expert bytes requires knowing which FQN pattern the byte "'
        ),
    },
    {
        "name": "emit_run_manifest.lora-preexisting-key-rename",
        "what": (
            "producer and on-disk predicate share the keys tuple, so "
            "renaming the audited denominator key travels end-to-end "
            "with enforcement staying green; only the pinned literal "
            "key names in the tests catch it"
        ),
        "anchor": '    "declared.preexisting_iter_dirs",',
        "replacement": '    "declared.preexisting_save_dirs",',
    },
    {
        "name": "emit_run_manifest.lora-count-plus-one",
        "what": (
            "inflates the observed pre-existing save-dir count by one, "
            "so the serialized record disagrees with the measured "
            "estate the predicate compares against"
        ),
        "anchor": "        str(preexisting_saves),",
        "replacement": "        str(preexisting_saves + 1),",
    },
    {
        "name": "emit_run_manifest.lora-count-fabricated",
        "what": (
            "declared.preexisting_iter_dirs is the denominator an "
            "auditor checks against the estate; pinning it to 0 "
            "fabricates the count instead of measuring it"
        ),
        "anchor": "        lora_preexisting_saves = _count_save_dirs(ckpt_dir)",
        "replacement": "        lora_preexisting_saves = 0",
    },
    {
        "name": "emit_run_manifest.drill-never-arms",
        "what": (
            "the bare-null drill is the observed-firing leg for the "
            "on-disk abstention control; '== \"0\"' means the drill "
            "can never arm, a quiet control"
        ),
        "anchor": '    drill_bare_null = os.environ.get(_DRILL_BARE_NULL_ENV) == "1"',
        "replacement": '    drill_bare_null = os.environ.get(_DRILL_BARE_NULL_ENV) == "0"',
    },
]


SUITE_TIMEOUT_S = int(os.environ.get("FS_MUTATE_SUITE_TIMEOUT", "1800"))


class TrialKind(Enum):
    """What one mutant trial established — or failed to establish."""

    ALIVE = "ALIVE"  # suite green under mutation: the suite has a gap (exit 1)
    KILLED = "KILLED"  # attributed failure, reproduced on a second run (neutral)
    UNSCORED = "UNSCORED"  # the measurement itself failed: claim nothing (exit 2)


@dataclass(frozen=True)
class SuiteOutcome:
    """One suite run, structured.

    Attribution comes from the junit report, never from scraping stdout —
    human output is allowed to drift; the XML is pytest's documented output
    contract. `junit_path` is kept so every non-green claim still has its
    evidence on disk after the run.
    """

    rc: int  # pytest's exact returncode; -9 sentinel when the run timed out
    passed: int
    failed: tuple[str, ...]  # nodeids carrying <failure/> in the junit report
    errored: tuple[str, ...]  # nodeids / collection paths carrying <error/>
    skipped: int
    duration_s: float
    timed_out: bool
    junit_path: str
    stdout_tail: str = ""  # diagnostics only — never scored


@dataclass(frozen=True)
class TrialVerdict:
    """The battery's claim about one mutant.

    `attribution` is meaningful in exactly two states: a KILLED verdict (the
    tests that detected the break) and an UNSCORED kill candidate awaiting
    reproduction — a candidate that has not reproduced is not a kill.
    """

    kind: TrialKind
    attribution: tuple[str, ...]
    reason: str


SuiteRunner = Callable[..., SuiteOutcome]


def run_suite(
    *,
    junit_dir: Path,
    label: str,
    timeout_s: int = SUITE_TIMEOUT_S,
    env: dict[str, str] | None = None,
) -> SuiteOutcome:
    """Run the full suite once and return a structured outcome.

    No `-x`: two mutants killed by the same single test look identical to two
    mutants killed by genuinely different coverage unless the killing tests
    are named, and an early exit throws that away. The junit report is where
    the names come from.

    The skip guard is armed here, by the tool itself — never inherited from
    the caller's shell, and merged last so no `env` argument can switch it
    back off. The timeout kills the pytest process (pytest runs tests
    in-process, so expiring the process ends the run): a mutant that makes
    the suite loopy must cost one unscored trial, not the tree, because CI's
    SIGKILL would bypass the finally-restore in main() and leave the mutant
    on disk.
    """
    junit = junit_dir / f"suite-{label}.xml"
    # The arming line merges LAST: a caller may add to the environment but
    # cannot disarm the skip guard.
    run_env = {**os.environ, **(env or {}), "FS_FORBID_SKIPS": "1"}
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [
                PY,
                "-m",
                "pytest",
                "tests/",
                "--no-header",
                "-p",
                "no:cacheprovider",
                f"--junitxml={junit}",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=run_env,
        )
        rc, timed_out = proc.returncode, False
        tail = proc.stdout[-2000:]
        if proc.stderr.strip():
            # Under sys.executable portability the likeliest red is "this
            # interpreter has no pytest at all", which speaks only on stderr.
            tail += f"\n--- pytest stderr ---\n{proc.stderr[-1000:]}"
    except subprocess.TimeoutExpired:
        rc, timed_out, tail = -9, True, ""
    passed, failed, errored, skipped = _parse_junit(junit)
    return SuiteOutcome(
        rc=rc,
        passed=passed,
        failed=failed,
        errored=errored,
        skipped=skipped,
        duration_s=time.monotonic() - t0,
        timed_out=timed_out,
        junit_path=str(junit),
        stdout_tail=tail,
    )


def _nodeid(classname: str | None, name: str | None) -> str:
    """Rebuild a collectable pytest nodeid from a junit <testcase>.

    Real pytest writes junit classnames as DOTTED module paths —
    `tests.test_parity`, or `tests.test_parity.TestFoo` for a method —
    never the slash-and-`.py` shape humans type. Joining classname and name
    verbatim therefore names a test that does not exist, and the `[killed]`
    line is the human-facing record of what fired: it must be pasteable
    back into `pytest <nodeid>`. Module components precede the first
    Capitalized component (pytest's own class-naming convention); anything
    from there on is a class segment of the nodeid.
    """
    cls = classname or "?"
    test = name or "?"
    parts = cls.split(".")
    split_at = len(parts)
    for i, part in enumerate(parts):
        if part[:1].isupper():
            split_at = i
            break
    module, classes = parts[:split_at], parts[split_at:]
    path = "/".join(module) + ".py"
    return "::".join([path, *classes, test])


def _parse_junit(path: Path) -> tuple[int, tuple[str, ...], tuple[str, ...], int]:
    """Read counts and attribution out of pytest's junit report.

    An unreadable report is reported as an ERROR, never as emptiness:
    returning zeros here would let a truncated or drifted file masquerade as
    a suite that simply named nothing — exactly the silent degradation the
    junit rewrite exists to remove.
    """
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        return (0, (), (f"<junit unreadable: {exc}>",), 0)
    passed = 0
    skipped = 0
    failed: list[str] = []
    errored: list[str] = []
    for case in root.iter("testcase"):
        nodeid = _nodeid(case.get("classname"), case.get("name"))
        tags = {child.tag for child in case}
        if "failure" in tags:
            failed.append(nodeid)
        elif "error" in tags:
            errored.append(nodeid)
        elif "skipped" in tags:
            skipped += 1
        else:
            passed += 1
    # Collection errors never become <testcase> elements; pytest writes them
    # as <error> directly under <testsuite>. Dropping them would read a
    # collection failure as a clean run.
    for suite in root.iter("testsuite"):
        for err in suite.findall("error"):
            errored.append(f"<collection {err.get('message', '?')}>")
    return (passed, tuple(failed), tuple(errored), skipped)


def classify(out: SuiteOutcome) -> TrialVerdict:
    """Score one suite run. Pure — this is the entire per-run contract.

    Only a green run with zero skips is evidence the mutant survived. Only an
    rc=1 naming failing tests is even a *candidate* kill, and candidates are
    promoted solely by confirm_kill. Everything else — timeouts, collection
    and usage errors, red runs that name nothing, greens poisoned by skips —
    is UNSCORED, because none of it says anything about the rule the mutant
    broke. classify therefore never returns KILLED.
    """
    if out.timed_out:
        return TrialVerdict(
            TrialKind.UNSCORED,
            (),
            f"suite exceeded the {SUITE_TIMEOUT_S}s timeout under mutation "
            f"(rc={out.rc}); a hanging mutant measures nothing — junit: "
            f"{out.junit_path}",
        )
    if out.skipped:
        return TrialVerdict(
            TrialKind.UNSCORED,
            (),
            f"{out.skipped} skip(s) recorded despite the FS_FORBID_SKIPS=1 "
            "guard this tool exports — either the guard was ignored or the "
            "report is lying, and a skipped test cannot kill a mutant either way",
        )
    if out.rc == 0:
        if out.errored:
            # _parse_junit encodes an unreadable report as an error entry,
            # and the rc=1 branch already honours that sentinel; the green
            # branch must too. A green run over evidence that was never read
            # is "we never measured" (2), not "the suite has a gap" (1) —
            # die()'s contract says those two must never blur.
            return TrialVerdict(
                TrialKind.UNSCORED,
                (),
                f"rc=0 but the junit evidence is not whole ({out.errored[0]}): a "
                "green run over evidence that was never read is not a surviving "
                f"mutant — junit: {out.junit_path}",
            )
        if out.passed == 0:
            # A green return code over a junit report recording ZERO executed
            # tests detected nothing: "the suite stayed green" and "the suite
            # ran" have collapsed into the same non-event. Vanilla pytest
            # exits 5 on an empty collection, so reaching this branch means
            # the return code and the report disagree (an injected runner, a
            # plugin, a race on the junit file) — and contradictory evidence
            # blocks, it never scores. Above all it cannot be ALIVE: ALIVE's
            # reason asserts "no test detects the broken rule", a claim about
            # tests that never ran, priced at exit 1 ("the suite has a gap")
            # when the truth is exit 2 ("we never measured"). check_baseline
            # has always refused `not out.passed` for exactly this reason;
            # the trial side of the per-run contract needed the same floor.
            return TrialVerdict(
                TrialKind.UNSCORED,
                (),
                f"rc=0 but the junit report records 0 executed tests — a "
                f"green run over zero tests detected nothing (vanilla pytest "
                f"exits 5 on an empty collection, so the return code and the "
                f"report disagree; treat the pair as unreadable evidence) — "
                f"junit: {out.junit_path}",
            )
        return TrialVerdict(
            TrialKind.ALIVE,
            (),
            "suite green under mutation — no test detects the broken rule",
        )
    if out.rc == 1:
        if out.errored:
            return TrialVerdict(
                TrialKind.UNSCORED,
                (),
                f"rc=1 with {len(out.errored)} error(s) alongside any failures: "
                f"the suite broke in ways the mutant did not aim at — junit: "
                f"{out.junit_path}",
            )
        if not out.failed:
            # The sentence this redesign exists for:
            return TrialVerdict(
                TrialKind.UNSCORED,
                (),
                "rc=1 with zero attributed failures — a kill with no attribution is not a kill",
            )
        return TrialVerdict(
            TrialKind.UNSCORED,
            out.failed,
            f"candidate only: {len(out.failed)} attributed failure(s) awaiting reproduction",
        )
    return TrialVerdict(
        TrialKind.UNSCORED,
        (),
        f"pytest rc={out.rc}: a collection/usage/internal outcome says "
        f"nothing about this rule — junit: {out.junit_path}",
    )


def kill_candidate(out: SuiteOutcome) -> bool:
    """True iff the run is an attributed rc=1 with no contaminating signal."""
    return (
        out.rc == 1
        and not out.timed_out
        and out.skipped == 0
        and bool(out.failed)
        and not out.errored
    )


def confirm_kill(first: SuiteOutcome, second: SuiteOutcome) -> TrialVerdict:
    """Credit a kill only when the same tests fail twice. Pure.

    One red run is weather, not detection: rc=1 also covers escalated
    warnings, and a flaky test reds at will. The second run is the
    reproduction; divergent or contaminated attribution demotes the trial to
    UNSCORED, never to KILLED. A first run that was never a candidate returns
    its own classify() verdict unchanged, so a collection error keeps its
    rc=2 story instead of looking like a near-miss.
    """
    if not kill_candidate(first):
        return classify(first)
    if not kill_candidate(second):
        verdict = classify(second)
        return TrialVerdict(
            TrialKind.UNSCORED,
            (),
            f"candidate kill did not reproduce as a clean run: {verdict.reason}",
        )
    if set(second.failed) != set(first.failed):
        return TrialVerdict(
            TrialKind.UNSCORED,
            (),
            f"attribution not reproducible ({sorted(first.failed)} -> "
            f"{sorted(second.failed)}); suspect flake — the rule's coverage "
            "is unknown",
        )
    return TrialVerdict(
        TrialKind.KILLED,
        tuple(sorted(first.failed)),
        "attribution reproduced under a second application of the same mutant",
    )


def score_trial(runner: SuiteRunner, *, junit_dir: Path, label: str) -> TrialVerdict:
    """One mutant, up to two suite runs.

    Kills cost a confirmation rerun by design; FS_MUTATE_SUITE_TIMEOUT bounds
    each run, and a runner that itself raises TimeoutExpired is treated as a
    failed measurement (UNSCORED), never as a kill.
    """
    try:
        first = runner(junit_dir=junit_dir, label=f"{label}-run1", timeout_s=SUITE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return TrialVerdict(
            TrialKind.UNSCORED, (), "suite runner raised TimeoutExpired; nothing measured"
        )
    if not kill_candidate(first):
        return classify(first)
    try:
        second = runner(junit_dir=junit_dir, label=f"{label}-run2", timeout_s=SUITE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return TrialVerdict(
            TrialKind.UNSCORED,
            (),
            "confirmation run raised TimeoutExpired; candidate kill never reproduced",
        )
    return confirm_kill(first, second)


# --- legacy stdout scrapers (not used by the scorer) -------------------------
# tally()/failing_tests() parse pytest's human output. They scored every trial
# before junit attribution existed, which is how a bare rc=2 or a format drift
# could print `[killed]` with zero tests named. Kept importable for external
# callers; the battery never calls them.

_COUNT = re.compile(r"(\d+) (passed|failed|skipped|xfailed|xpassed|error)")


def tally(out: str) -> dict[str, int]:
    """Parse pytest's summary line into counts."""
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    return {kind: int(n) for n, kind in _COUNT.findall(tail)}


def failing_tests(out: str) -> list[str]:
    return [
        ln.split("::", 1)[-1].split(" ")[0] for ln in out.splitlines() if ln.startswith("FAILED")
    ]


def check_baseline(runner: SuiteRunner, *, junit_dir: Path) -> tuple[bool, str]:
    """Green AND complete, or we are not measuring anything."""
    try:
        out = runner(junit_dir=junit_dir, label="baseline", timeout_s=SUITE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, (
            f"BASELINE EXCEEDED the {SUITE_TIMEOUT_S}s timeout before the suite "
            "finished — fix the suite's runtime first; nothing else this run "
            "could report would mean anything."
        )
    if out.timed_out:
        return False, (
            f"BASELINE RAN PAST the {SUITE_TIMEOUT_S}s timeout — fix the suite's "
            "runtime first; nothing else this run could report would mean anything."
        )
    if out.rc != 0:
        detail = out.stdout_tail
        named = [f"  {n}" for n in (*out.failed[:10], *out.errored[:10])]
        if named:
            detail = "attributed:\n" + "\n".join(named) + (f"\n{detail}" if detail else "")
        return False, f"BASELINE IS RED (rc={out.rc}) — fix the suite first.\n{detail}"
    if out.errored:
        return False, (
            "BASELINE IS GREEN BUT ITS JUNIT EVIDENCE IS NOT WHOLE "
            f"({out.errored[0]}). A baseline whose report could not be read "
            "says nothing about what the suite covers, and running the battery "
            "on top of it is how 'we never measured' gets printed as a result. "
            "Fix the report first."
        )
    if out.skipped:
        return False, (
            f"BASELINE HAS {out.skipped} SKIPPED TEST(S) under {PY}.\n"
            "A skipped test cannot kill a mutant, so every mutation it would have "
            "caught gets reported as a rule with no test behind it — a false alarm "
            "from a detector that was never wired up. The junit tally above is "
            "the tripwire that caught this; FS_FORBID_SKIPS=1 is exported so a "
            "conftest can redden the run even earlier. Install the missing "
            "dependency or fix the skip condition; do not run the battery like this."
        )
    if not out.passed:
        return False, "BASELINE RAN ZERO TESTS — collection is broken."
    return True, f"baseline: {out.passed} passed, 0 skipped"


def die(msg: str) -> NoReturn:
    """Exit 2: this run measured nothing, and must not be mistaken for a result.

    Everything that prevents the battery from starting (no table, malformed
    table, unreadable module) is a precondition failure, not a surviving
    mutant — CI reads 1 as "the suite has a gap" and 2 as "we never measured",
    and the two must never blur.
    """
    print(msg, file=sys.stderr)
    sys.exit(2)


def _validate_table(
    data: dict[str, list[dict]],
    paths: dict[str, Path],
    *,
    complete: bool = False,
) -> None:
    """Shape-check a mutation table against its module map.

    complete=True adds the complement muster (the orphan tripwire): every
    mapped module must be published with at least one row, or die() names
    each unpublished module with the row count it owed. It stays opt-in so
    pre-existing injected-table callers keep their exact semantics; the
    disk path -- load_table -- always opts in, on the FULL merged view,
    before --module filtering. EMBEDDED_TABLE is referenced below its
    definition site and resolves at call time.

    Applied to the shipped table from disk and to injected tables alike: an
    in-memory table does not get a lower bar just because it never touched
    the filesystem. `must_survive` is the MUST-PASS half of the detector
    (doctrine 3); it is optional, but when present must be a real boolean
    or die() — a truthy string would silently arm the control path on a
    typo, and a control armed by accident is the vacuous-success shape in
    miniature. It also refuses a table carrying ZERO MUST-FIRE rows: every
    row a negative control means the positive half of the detector fires
    zero times, and that run's exit 0 would print 0 killed of 0 fired —
    the vacuous success one level up from the mutants. Half a proof does
    not get to run.
    """
    unknown = set(data) - set(paths)
    if unknown:
        die(f"table names modules with no path mapping: {sorted(unknown)}")
    if complete:
        # The complement of the historical check, and the orphan tripwire
        # (#85/#86/#89): registered-but-unpublished means zero published
        # rows, which is UNMEASURED, never covered -- a refusal, not a
        # silence. Runs on the FULL merged view only; load_table applies
        # the --module filter after it, so the check never sees modules a
        # filtered run legitimately set aside.
        unpublished = [mod for mod in sorted(paths) if not data.get(mod)]
        if unpublished:
            detail = []
            for mod in unpublished:
                rows = EMBEDDED_TABLE.get(mod)
                if rows:
                    detail.append(
                        f"  {mod}: 0 of {len(rows)} rows the tool itself embeds "
                        "(EMBEDDED_TABLE) -- the publish-by-merge step regressed"
                    )
                else:
                    detail.append(
                        f"  {mod}: 0 published; MODULE_PATHS maps it, so the "
                        "table owes it at least 1 row and carries 0"
                    )
            die(
                "mapped-but-unpublished module(s) in the mutation table: "
                f"{len(unpublished)} of {len(paths)} mapped modules carry 0 "
                "rows:\n" + "\n".join(detail)
            )
    if not any(data.values()):
        die(
            "the table selects zero mutations — a battery with no mutants proves "
            "nothing, so it does not get to exit 0."
        )
    for mod, muts in sorted(data.items()):
        for m in muts:
            missing = [k for k in _REQUIRED_KEYS if k not in m]
            if missing:
                die(f"{mod}: mutation entry missing key(s) {missing}: {str(m)[:120]!r}")
            if not isinstance(m["anchor"], str) or not isinstance(m["replacement"], str):
                die(f"{mod}/{m.get('name', '?')}: anchor and replacement must be strings")
            if "must_survive" in m and not isinstance(m["must_survive"], bool):
                die(f"{mod}/{m.get('name', '?')}: must_survive must be a boolean")
    must_fire_rows = sum(
        1 for muts in data.values() for m in muts if not m.get("must_survive", False)
    )
    if must_fire_rows == 0:
        # The 0/0 run in its purest form: every selected row is a negative
        # control, so the battery's positive half — deliberately break a
        # rule, demand the suite notice — is exercised ZERO times. A
        # surviving control can never redden such a run (surviving is what
        # a control MEANS), so the run would print a clean control line,
        # tally caught=n/a over zero scored trials, and fall off the end of
        # the exit ladder to `return 0`: the battery certifying detection
        # over a detector that fired never. That is the founding incident
        # in exit-code form. Reachable from the shipped CLI as well as from
        # injected tables, because load_table() validates AFTER the
        # --module filter: `--module X` on a module whose rows are all
        # controls selects exactly this table. die() beside the zero-rows
        # guard above — both guards say one sentence (a battery that fires
        # nothing proves nothing), once for an empty magazine and once for
        # a magazine of blanks. Placed AFTER the per-row loop so
        # must_survive has already been proven a real bool; read earlier,
        # `must_survive: "yes"` would count its row as a control on a typo
        # and trip this guard for the wrong reason.
        die(
            "the table selects zero MUST-FIRE mutations — every row is a "
            '"must_survive" control. Surviving controls cannot redden that '
            "run by construction, so its exit 0 would be 0 killed of 0 "
            "fired: the vacuous success this battery exists to refuse, in "
            "exit-code form. Controls are the detector's negative half "
            "(doctrine 3); run alone they are half a proof. Add a real "
            "mutant to the selection. This run exits 2: never measured."
        )


# R1+R2 (#85/#86/#89 orphan class). The emit_run_manifest rows are published
# by MERGE, never copied into tools/mutations.json: EMIT_RUN_MANIFEST_ROWS
# (above) is the single source for the fact, and load_table refuses a JSON
# that grows a copy -- two sources for one fact. The inert must-pass row is
# the module's MUST-PASS half (doctrine 3): it anchors on the same producer
# line as row lora-count-plus-one, whose uniqueness in
# tools/emit_run_manifest.py the shape test already enforces, and a trailing
# comment is stripped by the tokenizer -- no test, import, or bytecode
# comparison can observe it.
EMBEDDED_TABLE: dict[str, list[dict]] = {
    "emit_run_manifest": [
        *EMIT_RUN_MANIFEST_ROWS,
        {
            "name": "emit_run_manifest.inert-control-must-pass",
            "what": (
                "MUST SURVIVE: a trailing comment changes no behaviour, so "
                "the suite MUST stay green. If the battery reports this "
                "killed, its attribution machinery is proven unsound and "
                "the whole run is void -- exit 2."
            ),
            "anchor": "        str(preexisting_saves),\n",
            "replacement": (
                "        str(preexisting_saves),"
                "  # inert must-pass control\n"
            ),
            "must_survive": True,
        },
    ],
}


def load_table(only: str | None) -> dict[str, list[dict]]:
    """Read the committed table from disk, merge the embedded rows, muster.

    The complement muster (complete=True) runs exactly once per load, on
    the FULL merged view, BEFORE --module narrows it: the filter can
    therefore never trip the muster over modules it legitimately set
    aside, and the merged module names appear in the unknown-module
    'have:' list. main() may be handed a table instead.
    """
    if not TABLE.exists():
        die(
            f"no mutation table at {TABLE} — it is committed alongside this tool.\n"
            "Restore it rather than running without it: a battery with no table "
            "has not 'found nothing', it has not run, and it exits 2 to say so."
        )
    try:
        data = json.loads(TABLE.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        die(f"{TABLE} is not valid JSON: {exc}")
    dual = sorted(set(EMBEDDED_TABLE) & set(data))
    if dual:
        die(
            f"{TABLE} carries rows for {dual} but tools/mutate.py embeds them "
            "(EMBEDDED_TABLE) -- two sources for one fact. Delete the JSON "
            "key(s); never maintain a copy."
        )
    merged = {m: [dict(r) for r in rows] for m, rows in EMBEDDED_TABLE.items()}
    data = {**data, **merged}
    paths = {mod: ROOT / rel for mod, rel in MODULE_PATHS.items()}
    _validate_table(data, paths, complete=True)
    if only:
        if only not in data:
            die(f"unknown module {only!r}; have: {', '.join(sorted(data))}")
        data = {only: data[only]}
    return data


# --- anchor-freshness gate: is every published anchor still bindable? ---
#
# Ground truth here is RECONSTRUCTED, never read. The two obvious sources are
# both measurably wrong for this tree. A git HEAD snapshot lags any repair
# round that has not been committed yet, so it reports a stale row that is not
# stale -- a false alarm on a healthy corpus, and doctrine 5 prices that
# exactly like a false green. And the raw working tree IS one applied mutant
# while the battery's exec'd suite is running (tools/mutate.py writes the
# mutant, execs pytest, and restores only in `finally`), so a raw reader goes
# red for every mutation of that file regardless of behaviour, is named in
# every attribution, and reddens the inert must-pass control -- the text
# tripwire that already cost two repair rounds.
#
# The corpus is its own inverse map: every row carries anchor AND replacement.
# So the gate asks an EXISTENCE question instead of a recovery question:
#
#     is there a text P in which every one of this file's anchors occurs
#     exactly once, and from which the live bytes are reachable in zero or
#     one row applications?
#
# EXISTENCE, not uniqueness, and keyed on neither "which row is applied" nor
# "is the replacement present". Four legitimate corpus shapes are live today
# and each defeats a per-row reading:
#   * DELETION-STYLE -- the replacement is a strict substring of its own
#     anchor (2 rows), so `count(replacement) == 0` is unsatisfiable in a
#     pristine source and every such row would false-alarm forever;
#   * INSERTION-STYLE -- the anchor is a strict substring of its own
#     replacement (2 rows), so the applied state still binds that anchor
#     exactly once, which is correct and must not be refused;
#   * IDENTICAL SHARED ANCHOR -- `core` publishes 4 rows against one anchor,
#     so applying any one of them drives all four to 0x;
#   * NESTED ANCHORS -- in `parity` and `emit_run_manifest` one row's anchor
#     sits inside another's, so applying the inner row consumes the outer
#     row's anchor too.
# Demanding a unique applied row reddens 6 of the 70 legitimate applied
# states; demanding a unique pristine candidate is impossible, because a
# deletion-style row always generates a second self-consistent candidate.
#
# Placement: beside MODULE_PATHS and load_table, the two facts the gate
# joins. Safe because tools/mutate.py is not itself a MODULE_PATHS target --
# no battery leg writes a mutant into this file, so growing it here can evict
# no corpus anchor and the gate's own bytes are never under measurement.

MAX_FRESHNESS_CANDIDATES = 256


class FreshnessRefusedError(Exception):
    """The live bytes cannot be placed in the accepted state set."""


def _accepted_pristine_states(live: str, rows: list[dict]) -> list[str]:
    """Every text that could be the pristine source behind `live`.

    A candidate is `live` itself -- nothing applied, the normal case -- or
    `live` with ONE occurrence of one row's replacement rewritten back to
    that row's anchor. EVERY occurrence is enumerated, not only the
    single-occurrence case: a deletion-style row's replacement necessarily
    occurs inside its own intact anchor, so any count-based shortcut is
    reading the wrong thing.

    A candidate is ACCEPTED only when both hold:

      * every one of this file's anchors occurs EXACTLY once in it. Raw
        str.count, byte-exact, indentation included -- no strip, no
        normalisation, nothing fuzzy. The drift this gate exists to catch
        was eight leading spaces becoming four, and a tolerant matcher
        greens it.
      * `live` is reachable from it in zero or one row applications. This is
        the clause that stops a two-rows-applied or otherwise corrupt source
        from being laundered into "pristine" by a candidate that merely
        happens to bind every anchor.

    Returns every accepted candidate; the caller needs only whether the list
    is non-empty. Raises FreshnessRefusedError if enumeration exceeds
    MAX_FRESHNESS_CANDIDATES -- refusing rather than searching, since the
    shipped corpus peaks at 3 candidates across all 71 of its reachable
    states, so a blow-up means the input is not what this gate was built for.
    """
    candidates = [live]
    for row in rows:
        rep = row["replacement"]
        if not rep:
            continue
        at = live.find(rep)
        while at != -1:
            if len(candidates) >= MAX_FRESHNESS_CANDIDATES:
                raise FreshnessRefusedError(
                    "candidate enumeration exceeded "
                    f"{MAX_FRESHNESS_CANDIDATES} states over {len(rows)} "
                    "row(s): refusing rather than searching -- the shipped "
                    "corpus peaks at 3 candidates, so this input is not "
                    "what the freshness gate was built for"
                )
            candidates.append(live[:at] + row["anchor"] + live[at + len(rep):])
            at = live.find(rep, at + 1)
    accepted = []
    for cand in candidates:
        if not all(cand.count(row["anchor"]) == 1 for row in rows):
            continue
        reachable = {cand}
        for row in rows:
            reachable.add(cand.replace(row["anchor"], row["replacement"], 1))
        if live in reachable:
            accepted.append(cand)
    return accepted


def check_anchor_freshness(
    table: dict[str, list[dict]],
    paths: dict[str, Path],
) -> list[str]:
    """Resolve every published anchor exactly once, against RECOVERED bytes.

    The caller hands over load_table's FULL merged view -- the
    tools/mutations.json rows MERGED with EMBEDDED_TABLE -- because a gate
    over the JSON half alone repeats the orphan class where the
    emit_run_manifest rows sat outside the only table anything checked. A
    --module narrowing must never be passed in as if it were the corpus.

    Fail-closed enumerations, one problem string per finding, ALL findings
    accumulated and returned together so a single red names every offender
    at once: a zero-row corpus; a module publishing 0 rows beside measured
    ones; a row with an empty anchor or replacement; a row whose replacement
    EQUALS its anchor (that mutation edits nothing, so the battery would
    score a guaranteed survivor as a real mutant); a corpus module absent
    from `paths`; a source that will not read or will not decode; candidate
    enumeration blowing past its bound; and finally the substantive verdict,
    no pristine state accounting for the live bytes, which names each row
    whose anchor does not bind exactly once and its counts.

    An empty return is the only green.

    RESIDUAL, stated exactly: the verdict is that every published anchor
    BINDS, not that the tree is pristine. For an insertion-style row -- one
    whose anchor is a substring of its own replacement -- the applied state
    is accepted, correctly, because that anchor still resolves exactly once;
    this gate does not claim to detect that such a row is applied. Drift
    that displaces no anchor is likewise invisible to it. Whether the suite
    would actually KILL any mutant is the battery's verdict, never this
    gate's.
    """
    problems: list[str] = []
    if sum(len(rows) for rows in table.values()) == 0:
        problems.append(
            f"zero-row corpus: {len(table)} module(s) publish 0 rows between "
            "them -- a freshness gate over zero anchors has measured "
            "nothing, and unmeasured is never fresh"
        )
        return problems
    for mod in sorted(table):
        rows = table[mod]
        if not rows:
            problems.append(
                f"{mod}: 0 rows published beside measured modules -- "
                "unmeasured, never fresh"
            )
            continue
        blank = [
            r.get("name", "?") for r in rows
            if not r.get("anchor") or not r.get("replacement")
        ]
        if blank:
            problems.append(
                f"{mod}: {len(blank)} row(s) with an empty anchor or "
                f"replacement: {', '.join(blank)} -- an empty anchor binds "
                "everywhere and measures nothing"
            )
            continue
        noop = [r["name"] for r in rows if r["anchor"] == r["replacement"]]
        if noop:
            problems.append(
                f"{mod}: {len(noop)} row(s) whose replacement EQUALS their "
                f"anchor: {', '.join(noop)} -- such a row edits nothing, so "
                "the battery would score a guaranteed survivor as a mutant"
            )
        path = paths.get(mod)
        if path is None:
            problems.append(
                f"{mod}: no source mapping -- fail closed: an unmappable "
                "module's anchors are unresolved, never fresh"
            )
            continue
        try:
            live = path.read_text("utf-8")
        except (OSError, UnicodeError) as exc:
            problems.append(
                f"{mod}: cannot read source {path}: {exc} -- fail closed: "
                "unreadable is not empty, and a decode failure is not a "
                "missing file"
            )
            continue
        try:
            accepted = _accepted_pristine_states(live, rows)
        except FreshnessRefusedError as exc:
            problems.append(f"{mod}: {exc}")
            continue
        if not accepted:
            blamed = [r for r in rows if live.count(r["anchor"]) != 1] or rows
            counts = "; ".join(
                f"{r['name']} anchor {live.count(r['anchor'])}x, "
                f"replacement {live.count(r['replacement'])}x"
                for r in blamed
            )
            problems.append(
                f"{mod}: no pristine state accounts for these bytes over "
                f"{len(rows)} row(s) -- at least one anchor no longer binds "
                "exactly once, and reverting any single row does not repair "
                f"it; re-derive the offending row(s) against the current "
                f"source. Live counts: {counts}"
            )
    return problems


EPILOG = """\
exit status — CI relies on 1 and 2 being different:
  0  every applied mutant was KILLED on two identical attributed runs: each
     broken rule had a test behind it
  1  at least one mutant SURVIVED: the suite stayed green with a rule
     broken. A survivor outranks a refused anchor — with both present the
     exit is 1, because the survivor is actionable evidence against the
     suite while the anchor only means this table drifted. Only an
     UNSCORED trial beats a survivor: a battery that could not measure
     claims nothing, not even "the suite has a gap".
  2  never measured: suite red, skipped tests, no table, nothing applied
     (and no survivor to outrank the refusal), any trial UNSCORED
     (timeout, collection/usage error, rc=1 naming no failing test, a
     green run whose junit records zero executed tests, attribution that
     failed to reproduce), a MUST-PASS control reported
     killed — an inert edit dying proves attribution unsound, and an
     unsound battery claims nothing — or the table carrying ZERO MUST-PASS
     control rows at all: without the negative half of the detector, every
     caught= figure the run printed is an unverified claim, and claiming
     from silence is "we never measured" one level up; or the table
     carrying ZERO MUST-FIRE rows (every row a control) — the positive
     half fired zero times, and exit 0 there is 0 killed of 0 fired

MUST-PASS CONTROL
  One table row is flagged "must_survive": an inert, comment-only edit that
  changes no behaviour, so the suite MUST let it pass. It is the negative
  control half of the detector — a table of MUST_FIRE mutants without it is
  a detector that fires no matter what. A control reported ALIVE passes and
  is excluded from the caught= denominator and the killed/alive tallies; a
  control reported KILLED fails the whole battery (exit 2), because a kill
  with no fault behind it proves attribution unsound. The MUST-PASS line
  prints on every run: a run whose table carries no control row says "none
  configured" and voids (exit 2), and a control row whose anchor went stale
  reports as configured-but-never-exercised — never a pass, never an absence.

HOW A KILL IS SCORED
  A mutant is never scored on pytest's exit code alone: rc=2, a bare rc=1 and
  a timeout are indistinguishable from detection if the exit code is all you
  read, and one flaky red run is indistinguishable from a real one. A kill
  requires rc=1 with failing tests named in the junit report, reproduced
  EXACTLY by a second run with the mutant still applied. Kills therefore
  cost two suite runs; each run is bounded by FS_MUTATE_SUITE_TIMEOUT
  (default 1800s), and the runner exports FS_FORBID_SKIPS=1 itself so a skip
  anywhere is caught instead of silently disarming a detector.

WHAT A SURVIVING MUTANT MEANS
  The battery deliberately broke one documented rule — flipped a verdict,
  dropped a byte check, widened a tolerance — and the test suite stayed
  green. Whatever that rule protects is currently enforced by nothing, no
  matter how reassuring the pass count underneath it is. `42 passed` over a
  surviving mutant is the same shape as `243 passed` over a dead training
  run.

WHAT TO DO ABOUT A SURVIVOR
  Write the test that fails under the mutation and passes without it, then
  re-run the battery and watch the mutant die. Leave the mutation in
  mutations.json — it is now the specification for the test you owe. Do NOT
  delete or soften the mutation to quiet the battery: that converts "the
  suite has a hole" into "the battery cannot see the hole", which is
  strictly worse, because it also buys confidence.

WHY THE BASELINE MAY REFUSE TO RUN (exit 2)
  No mutant is tried until the full suite passes AND reports zero skipped
  tests. A skipped test cannot kill anything; a battery run on top of skips
  reads live detectors as dead ones. Run this file with the interpreter that
  has the project's optional dependencies installed (e.g. torch for the
  checkpoint modules):

      python tools/mutate.py                whole table
      python tools/mutate.py --module dcp   one module
      python tools/mutate.py --list         show the table, run nothing

mutations.json lives beside this file. Each entry names a module, the rule
being broken, an anchor string that must occur exactly once in that module's
source, and the anchor's replacement; an anchor that does not match exactly
once is refused, never guessed.
"""


def main(
    argv: list[str] | None = None,
    *,
    suite_runner: SuiteRunner = run_suite,
    table: dict[str, list[dict]] | None = None,
    module_paths: dict[str, Path] | None = None,
) -> int:
    ap = argparse.ArgumentParser(
        prog="tools/mutate.py",
        description=(
            "Mutation battery: break each load-bearing rule in the framework, one "
            "at a time, and require the test suite to notice. A mutant that "
            "survives is a rule with no test behind it."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--module", help="run one module's mutations only")
    ap.add_argument("--list", action="store_true", help="print the table and exit")
    args = ap.parse_args(argv)

    resolved_paths: dict[str, Path]
    if module_paths is None:
        resolved_paths = {mod: ROOT / rel for mod, rel in MODULE_PATHS.items()}
    else:
        # Injected tables resolve their own paths verbatim; this branch is
        # pinned by tests/test_mutate_zero_work_refusals.py and must not move.
        resolved_paths = {mod: Path(p) for mod, p in module_paths.items()}

    # The table/module_paths seam is the second half of the inert-control
    # fix. A behaviour-preserving mutant once scored KILLED because main()
    # always read the LIVE mutations.json against the LIVE tree: an outer
    # mutation invalidated one live anchor, the inner meta-suite refused it,
    # and its deterministic exit 2 was banked as detection. An injected
    # table runs against scratch files, so mutating any shipped module can
    # no longer change what the meta-suite measures.
    if table is None:
        table = load_table(args.module)
    else:
        _validate_table(table, resolved_paths)

    if args.list:
        for mod, muts in sorted(table.items()):
            print(f"\n{mod}  ({len(muts)} mutations)")
            for m in muts:
                print(f"  {m['name']:34s} {m['what'][:88]}")
        print(f"\n{sum(len(v) for v in table.values())} mutations across {len(table)} module(s)")
        return 0

    # A table entry that resolves to a file this tree does not have is an
    # unreadable artifact, and the rule for unreadable artifacts is fail
    # CLOSED with the entry NAMED — before the baseline, not after it:
    # check_baseline costs one full suite run, and the old shape spent that
    # run and then reported through the generic OSError catch below, which
    # can surface the path (via the exception) but never the table ENTRY.
    # --list is exempt by design (it mutates nothing and returns above);
    # a file vanishing between this check and the backup read below is
    # still caught by that catch — this guard exists so the common case
    # (a table that has outgrown the tree) refuses cheaply and by name.
    missing_files = [
        (mod, resolved_paths[mod])
        for mod in sorted(table)
        if not resolved_paths[mod].is_file()
    ]
    if missing_files:
        named = "\n".join(
            f"  {mod}: resolves to {path} — no such file"
            for mod, path in missing_files
        )
        print(
            f"{len(missing_files)} of {len(table)} table module(s) point at "
            f"files this tree does not have:\n{named}\n"
            "a battery that cannot read its measurement target measures "
            "nothing over it, and measuring nothing never exits 0. Fix "
            "MODULE_PATHS (tree-root-relative now) or the table entry; "
            "exiting 2.",
            file=sys.stderr,
        )
        return 2

    # One mkdtemp per battery run, deliberately never deleted: junit reports
    # are the evidence behind every non-green claim this tool prints.
    junit_dir = Path(tempfile.mkdtemp(prefix="foundationscale-mutate-"))

    ok, msg = check_baseline(suite_runner, junit_dir=junit_dir)
    print(msg)
    if not ok:
        return 2
    print()

    # Back every file up before touching any of them, and restore ALL of them
    # at the end. Restoring only the file being mutated when an exception
    # fires would leave an earlier module mutated on disk — a corrupted tree
    # that looks perfectly fine.
    try:
        backups = {resolved_paths[mod]: resolved_paths[mod].read_text("utf-8") for mod in table}
    except OSError as exc:
        print(
            f"cannot back up every module before mutating — {exc}\n"
            f"each table module must be readable at its mapped path (the shipped "
            f"defaults resolve under {ROOT}); for the committed table, is this "
            f"tools/ sitting at the repository root, next to src/ and tests/?",
            file=sys.stderr,
        )
        return 2

    results: dict[str, dict[str, list]] = {}
    try:
        for mod, muts in table.items():
            path = resolved_paths[mod]
            orig = backups[path]
            r = results.setdefault(
                mod,
                {
                    "killed": [],
                    "alive": [],
                    "na": [],
                    "unscored": [],
                    "control_passed": [],
                    "control_failed": [],
                },
            )
            print(f"--- {mod}  ({len(muts)} mutations)")
            for i, m in enumerate(muts):
                n = orig.count(m["anchor"])
                if n != 1:
                    r["na"].append((m["name"], f"anchor matched {n}x, expected 1"))
                    print(f"  [SKIP ] {m['name']:32s} anchor matched {n}x — not applied")
                    continue
                path.write_text(orig.replace(m["anchor"], m["replacement"], 1), "utf-8")
                try:
                    # One mutant, up to two suite runs: a kill is credited
                    # only when the same attributed failures reproduce.
                    # Anything else — timeout, collection error, red run
                    # naming nothing, drifting attribution — is UNSCORED.
                    verdict = score_trial(suite_runner, junit_dir=junit_dir, label=f"{mod}-{i:02d}")
                finally:
                    # Restore before the verdict is even recorded: an
                    # exception inside scoring must not strand this mutant.
                    path.write_text(orig, "utf-8")
                is_control = bool(m.get("must_survive", False))
                if is_control and verdict.kind is TrialKind.ALIVE:
                    r["control_passed"].append(m["name"])
                    print(
                        f"  [ctrl ok] {m['name']:32s} inert edit left the suite green, as it must"
                    )
                elif is_control and verdict.kind is TrialKind.KILLED:
                    r["control_failed"].append((m["name"], verdict.attribution))
                    print(
                        f"  [CTRL DEAD] {m['name']:31s} behaviour-preserving edit was "
                        "reported killed"
                    )
                elif verdict.kind is TrialKind.ALIVE:
                    r["alive"].append((m["name"], m["what"]))
                    print(f"  [ALIVE] {m['name']:32s} suite still green — NOT TESTED")
                elif verdict.kind is TrialKind.KILLED:
                    r["killed"].append((m["name"], verdict.attribution))
                    # The attribution IS the evidence for the kill, so the
                    # run record carries it in full, one nodeid per line.
                    # A 100-char truncation once made a control-dead run
                    # impossible to audit after the fact (R6).
                    print(
                        f"  [killed] {m['name']:31s} {len(verdict.attribution)} test(s), "
                        "reproduced in full:"
                    )
                    for nodeid in verdict.attribution:
                        print(f"    - {nodeid}")
                else:
                    r["unscored"].append((m["name"], verdict.reason))
                    print(f"  [UNSCORED] {m['name']:32s} trial failed, no verdict:")
                    print(f"    - {verdict.reason}")
    finally:
        for p, text in backups.items():
            p.write_text(text, "utf-8")
            # Not an assert. This read-back is the ONLY verification that
            # the live tree actually got its pre-battery bytes back, and
            # this tool's whole safety case rests on it: asserts are
            # compiled out under `python -O` / PYTHONOPTIMIZE=1 (a common
            # CI default, and this file tells operators to run it under
            # whatever interpreter they are verifying), after which the
            # "restored byte-for-byte" line below would keep printing with
            # nothing verified — the vacuous-success shape guarding the
            # one operation that can corrupt the tree. A guard that a
            # command-line flag can erase is not a guard; the check must
            # be ordinary control flow.
            if p.read_text("utf-8") != text:
                raise RuntimeError(
                    f"restore verification failed for {p}: the pre-battery "
                    f"bytes were written back and read again and do not "
                    f"match — the tree may still carry a mutant. Restore "
                    f"this file from version control before trusting ANY "
                    f"result this run printed."
                )
        print(f"\n{len(backups)} module(s) restored byte-for-byte.")

    print("\n=== per-module tally ===")
    total_alive = 0
    total_unscored = 0
    total_scored = 0
    for mod in sorted(results):
        r = results[mod]
        total_alive += len(r["alive"])
        total_unscored += len(r["unscored"])
        scored = len(r["killed"]) + len(r["alive"])
        total_scored += scored
        # A percentage is only honest when every trial in the module was
        # scored: `8 killed 0 alive 4 unscored caught=100%` claims a whole
        # measurement over a module a third of which measured nothing.
        # Suppress the number rather than print a precise-looking lie.
        # A dead control impugns every measurement this module reports: an
        # inert edit "killed" means attribution can fire with no fault
        # behind it, so caught= over this module would be arithmetic, not
        # evidence. Suppress the number rather than print a lie.
        if r["unscored"] or r["na"] or r["control_failed"]:
            pct = "--"
        elif scored:
            pct = f"{100 * len(r['killed']) // scored}%"
        else:
            pct = "n/a"
        print(
            f"  {mod:20s} {len(r['killed']):2d} killed  {len(r['alive']):2d} alive  "
            f"{len(r['na']):2d} n/a  {len(r['unscored']):2d} unscored   caught={pct}"
        )

    ctrl_ok = [(mod, n) for mod, r in results.items() for n in r["control_passed"]]
    ctrl_dead = [(mod, n, attr) for mod, r in results.items() for n, attr in r["control_failed"]]
    # R6: the attribution IS the evidence; persist ALL of it, untruncated,
    # before any summary rendering. The console lines above carry one
    # nodeid per line; this JSON artifact is the machine-readable twin of
    # the same claim, rendered from the same in-memory results in this
    # same run, so the two cannot drift apart within a run. Failing to
    # persist the evidence is a certification failure (fail closed),
    # never a footnote: a run that cannot keep its own evidence proves
    # nothing.
    report_path = ROOT / "artifacts" / "mutation-battery-report.json"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "modules": {
                        module: {
                            "killed": [
                                {"name": n, "attribution": list(attr)}
                                for n, attr in r["killed"]
                            ],
                            "alive": [
                                {"name": n, "what": w} for n, w in r["alive"]
                            ],
                            "unscored": [
                                {"name": n, "reason": why}
                                for n, why in r["unscored"]
                            ],
                            "na": [{"name": n, "what": w} for n, w in r["na"]],
                            "control_passed": list(r["control_passed"]),
                            "control_failed": [
                                {"name": n, "attribution": list(attr)}
                                for n, attr in r["control_failed"]
                            ],
                        }
                        for module, r in results.items()
                    }
                },
                fh,
                indent=2,
                sort_keys=True,
            )
            fh.write("\n")
    except OSError as exc:
        raise SystemExit(
            "refusing to certify: cannot write attribution artifact "
            f"{report_path}: {exc}; a run that cannot persist its own "
            "evidence proves nothing"
        ) from exc
    print(f"attribution artifact (complete, never truncated): {report_path}")
    # "Configured" is counted from the TABLE, never from verdicts: a control
    # row whose anchor went stale lands in r["na"], and one whose trial came
    # back UNSCORED lands in r["unscored"] — neither may collapse into "no
    # controls configured". Three states, three readings: exercised (and it
    # passed or died), configured but never exercised, never configured.
    ctrl_configured = sum(1 for muts in table.values() for m in muts if m.get("must_survive"))
    ctrl_exercised = len(ctrl_ok) + len(ctrl_dead)
    # The MUST-PASS half of the detector reports as a returned fact, on its
    # own line, EVERY run — never by omission (doctrine 2). A run whose table
    # carried no control row states the absence in words, with its cost:
    # without a negative control, nothing separates "the suite detected the
    # fault" from "the suite reports detection whether or not there is a
    # fault", so every caught= figure above is arithmetic, not evidence —
    # the same shape as the lie this battery was just fixed for.
    if ctrl_configured == 0:
        print(
            "\nMUST-PASS CONTROL: none configured. This run carried no negative "
            "control, so it cannot distinguish 'the suite detected the fault' "
            "from 'the suite reports detection whether or not there is a fault' "
            "— every caught= figure above is an unverified claim."
        )
    else:
        print(
            f"\nMUST-PASS CONTROL: {len(ctrl_ok)}/{ctrl_exercised} exercised inert "
            f"edit(s) survived ({ctrl_configured} control row(s) configured)."
        )
        if ctrl_exercised < ctrl_configured:
            print(
                f"  configured-but-never-exercised: {ctrl_configured - ctrl_exercised} "
                "control row(s) produced no verdict (stale anchor or unscored trial, "
                "listed below under NOT APPLICABLE / UNSCORED); a control that did "
                "not run is not a passing control."
            )
        for mod, n in ctrl_ok:
            print(f"  [ctrl ok] {mod}/{n}: no fault, no detection — attribution sound")
        for mod, n, attr in ctrl_dead:
            print(
                f"  [CTRL DEAD] {mod}/{n}: killed — an inert edit cannot be "
                "detected; the attribution machinery is unsound. Full "
                f"attribution ({len(attr)} test(s)):"
            )
            for nodeid in attr:
                print(f"    - {nodeid}")
            if not attr:
                print("    - <no attribution>")

    # Denominator reconciliation (doctrine 2, M6), stated EVERY run so the
    # scoreable count can never again sit next to the corpus row count as
    # an unexplained gap: scoreable rows + must-pass control rows = corpus
    # rows. Controls are unscoreable BY CONSTRUCTION (their evidence is
    # survival, not a kill), so corpus minus scoreable is exactly the
    # control count -- printed here, never left for a reader to infer.
    corpus_rows = sum(len(muts) for muts in table.values())
    print(
        f"\nCORPUS DENOMINATOR: {corpus_rows} corpus row(s) over "
        f"{len(table)} module(s) = {corpus_rows - ctrl_configured} "
        f"scoreable MUST-FIRE mutation(s) + {ctrl_configured} MUST-PASS "
        "control row(s); controls are excluded from the scoreable count "
        "by construction, so that difference is stated, not an omission."
    )

    un = [(mod, n, why) for mod, r in results.items() for n, why in r["unscored"]]
    if un:
        print("\nUNSCORED — the trial itself failed, so these rules were not measured:")
        for mod, n, why in un:
            print(f"  {mod}/{n}: {why}")

    na = [(mod, n, w) for mod, r in results.items() for n, w in r["na"]]
    if na:
        print("\nNOT APPLICABLE — these never ran, so they are not evidence either way:")
        for mod, n, w in na:
            print(f"  {mod}/{n}: {w}")

    if total_alive:
        print("\nSURVIVING MUTANTS — each is a rule the suite does not actually check:")
        for mod in sorted(results):
            for n, w in results[mod]["alive"]:
                print(f"  - {mod}/{n}: {w}")

    if un or na or total_alive or ctrl_dead or ctrl_configured == 0:
        print(f"\njunit reports for this run are kept at {junit_dir}")

    # Report every problem, then pick the exit code. UNSCORED outranks even
    # survivors: a battery that could not measure a rule may not claim
    # anything, not even "the suite has a gap". A survivor remains more
    # actionable than a refused anchor, so it keeps exit 1 when nothing was
    # unscored; everything was printed above and none of it is exit 0.
    if na:
        applied_note = (
            "no mutation was applied at all"
            if total_scored + total_unscored == 0
            else f"{total_scored + total_unscored} of "
            f"{total_scored + total_unscored + len(na)} mutations were applied"
        )
        print(
            f"\n{len(na)} mutation(s) never ran — {applied_note}. Those rules were "
            "not measured by this run, and an unapplied mutation is not a passing "
            "one. Re-derive the stale anchors against the current source, then "
            "run again."
        )
    # A dead MUST-PASS control is a proven lie in the attribution machinery:
    # the battery reported detection with no fault present. Every "kill"
    # this run printed could have been the same defect, so it voids the run
    # ahead of everything below — exit 2, claim nothing.
    if ctrl_dead:
        print(
            f"\n{len(ctrl_dead)} MUST-PASS control(s) reported killed — a "
            "behaviour-preserving edit cannot be detected, so this run's "
            "attribution is proven unsound and the battery claims nothing "
            "about any mutant. Fix the harness, not the suite. Exiting 2."
        )
        return 2
    if total_unscored:
        print(
            f"\n{total_unscored} trial(s) unscored — for those rules this run "
            "measured nothing (timeouts, collection errors and unreproduced "
            "attribution are listed above). Exiting 2."
        )
        return 2
    # A control-free run sits in the claim-nothing class beside UNSCORED: the
    # MUST_FIRE half of the detector ran alone (doctrine 3 — a detector is
    # two claims), so nothing this run printed distinguishes detection from
    # a battery that reports detection whether or not there is a fault. It
    # outranks a survivor for the same reason UNSCORED does: a battery whose
    # negative half was never wired up may claim nothing — not "the suite
    # has a gap", and certainly not caught=100% — even though every trial
    # scored. It trails UNSCORED only because "we never measured" is the
    # more fundamental failure; both exit 2.
    if ctrl_configured == 0:
        print(
            "\nThe table configured no MUST-PASS control row, so the MUST_FIRE "
            "half of the detector ran alone and every kill printed above is an "
            "unverified claim. It outranks survivors for the same reason "
            'UNSCORED does. Restore a "must_survive": true row to the table, '
            "then run again. Exiting 2."
        )
        return 2
    if total_alive:
        return 1
    if na:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
