#!/usr/bin/env python3
"""CI-SUITE-EXTRAS -- every job that EXECUTES the pytest suite must install the SAME extras.

Usage:
  ci_suite_extras.py [--workflow PATH]   gate the workflow (default .github/workflows/ci.yml)
  ci_suite_extras.py --self-test         run the MUST_FIRE / MUST_PASS controls over
                                         SYNTHETIC in-memory workflows; exits 0 only if
                                         every control lands in its required state

THE DEFECT (measured, real)
---------------------------
.github/workflows/ci.yml executes the pytest suite from more than one job: `check`
runs pytest directly, and `mutation` runs tools/mutate.py, which runs the WHOLE suite
per mutant. Before the fix, `check` installed .[checkpoint,train,dev] while `mutation`
installed .[checkpoint,dev]. When a new test module needed the `train` extra, all 9
mutation shards died at COLLECTION with "No module named 'tokenizers'" -- and the
battery reported that as `assert 96 == 5`: an UNMEASURED mutant reading as a WRONG
VERDICT. Nothing gated the divergence, because each job's install line was reviewed
as if it were local to that job. It is not: the suite is the unit, and every job that
executes the suite inherits the suite's dependency set, not its own.

THE CLAIM
---------
CLAIM: every CI job that EXECUTES the pytest suite installs the SAME set of extras.
Divergence is the defect. This gate has NO ORACLE for which extras set is correct --
it cannot know that the suite needs `train`, and it must not pretend to know. It
decides only that the suite-executing jobs AGREE. Which set is right is the workflow
author's question; that there is ONE set is this gate's.

THE DENOMINATOR
---------------
The set of jobs that execute the suite. A job executes the suite if any of its steps'
run text contains a real pytest invocation, OR invokes tools/mutate.py (which runs
the suite internally). Jobs that never execute the suite are outside the claim and
cannot turn it red no matter what they install. The denominator -- count and names --
is printed in EVERY verdict line, including the CLEAR one, because a verdict without
its denominator is the vacuous pass with better typography.

FOUR STATES (this repository's contract, non-negotiable)
--------------------------------------------------------
  0  EXIT_CLEAR       >=2 suite-executing jobs found AND their extras sets are equal.
  5  EXIT_RED         >=2 found and they differ (every job and set is named, plus the
                      symmetric difference); also one job carrying two DISTINCT extras
                      sets across its own install lines -- ambiguous is refused.
  95 EXIT_UNMEASURED  the workflow is unreadable/unparseable, OR ZERO suite-executing
                      jobs found, OR exactly ONE found. Zero and one are UNMEASURED,
                      never CLEAR: all([]) is True, and one job cannot disagree with
                      itself, so "agreement" over either denominator is vacuous.
  96 EXIT_REFUSE      bad CLI usage.

WHAT THIS GATE CANNOT SEE
-------------------------
  * WHICH extras set is correct. Agreement on the wrong set reads CLEAR by design.
  * YAML semantics. PyYAML is deliberately not required; this is an explicit
    indentation-aware structural scan (jobs: at column 0, job names at two spaces).
    It claims no full parse, ever.
  * Shell metaprogramming: command position is judged line-locally; a pytest run
    hidden inside an eval'd string, a function defined and never called, or a
    continuation line shaped to dodge the probe is outside the instrument.
  * tools/mutate.py lines carrying a metadata-only flag (--list-modules,
    --self-test, --help, --version) are NOT suite executions -- the shard
    enumerator is the measured false positive; the list is inline and stated.
  * What an extra CONTAINS. Two names for the same dependency set diverge; one name
    for two different sets across branches agrees. The gate gates names, not truths.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

EXIT_CLEAR = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96

DEFAULT_WORKFLOW = ".github/workflows/ci.yml"

# --- structural scanner needles -------------------------------------------------
_JOBS_KEY_RE = re.compile(r"^jobs:\s*$")
_JOB_HEADER_RE = re.compile(r"^  ([A-Za-z0-9_.-]+):\s*$")

# Extras: pip install -e ".[...]" / '.[...]' / bare .[...] -- the install context is
# part of the needle ON PURPOSE: comments mention .[dev] in prose, and prose must not
# parse as an install. Comment lines are skipped before this ever runs.
_EXTRAS_RE = re.compile(r"\bpip3?\s+install\s+-e\s+['\"]?\.\[([^\]'\"]*)\]['\"]?")

# A REAL pytest invocation: command position only (line start, after ; & | or $(, or
# python[3] -m pytest). "pytest-cov>=5" and "pip install pytest ..." are excluded
# below; both are install arguments, not invocations.
_PYTEST_INVOKE_RE = re.compile(r"(^|[;&|(]|\bpython3?\s+-m\s+)\s*pytest\b")
_PIP_INSTALL_PREFIX_RE = re.compile(r"\bpip3?\s+install\b[^;&|(]*$")
_RUN_KEY_RE = re.compile(r"^-?\s*run:\s*(?:[|>][-+]?)?\s*(.*)$")

# tools/mutate.py runs the whole suite per mutant -- EXCEPT its metadata-only modes,
# which measure nothing. mutation-shards calls --list-modules; that job must NOT enter
# the denominator. STATED LIMIT: the flag must sit on the same line as the invocation.
_MUTATE_PATH_RE = re.compile(r"\btools/mutate\.py\b")
_MUTATE_METADATA_FLAGS = ("--list-modules", "--self-test", "--help", "--version")


class WorkflowParseError(Exception):
    """ci.yml could not be read by the structural scanner; unparseable is not empty."""


@dataclass
class JobBlock:
    name: str
    lines: list[str] = field(default_factory=list)


@dataclass
class JobFacts:
    name: str
    extras_sets: set[frozenset[str]]
    executes_suite: bool

    @property
    def extras(self) -> frozenset[str]:
        # Well-defined only when len(extras_sets) <= 1; evaluate() refuses ambiguity
        # (a single job with two DISTINCT extras sets) before ever reading this.
        if not self.extras_sets:
            return frozenset()
        return next(iter(self.extras_sets))


def parse_jobs(text: str) -> list[JobBlock]:
    """Indentation-aware scan: jobs: at column 0, job names at exactly two spaces.

    NOT a YAML parse. Every line from a two-space header to the next header (or a
    dedent to column zero, which closes the jobs: mapping) belongs to that job.
    """
    lines = text.splitlines()
    start: int | None = None
    for i, line in enumerate(lines):
        if _JOBS_KEY_RE.match(line):
            start = i + 1
            break
    if start is None:
        raise WorkflowParseError(
            "no column-zero 'jobs:' key found -- not a workflow this gate can see"
        )
    jobs: list[JobBlock] = []
    current: JobBlock | None = None
    for line in lines[start:]:
        header = _JOB_HEADER_RE.match(line)
        if header is not None:
            current = JobBlock(name=header.group(1))
            jobs.append(current)
            continue
        if line.strip() == "":
            if current is not None:
                current.lines.append(line)
            continue
        if line.startswith("#"):
            continue  # column-zero comment between jobs belongs to no job block
        if not line.startswith(" "):
            break  # dedent to column zero: the jobs: mapping is over
        if current is not None:
            current.lines.append(line)
    if not jobs:
        raise WorkflowParseError(
            "'jobs:' maps to zero job headers -- an empty mapping is unparseable here"
        )
    return jobs


def _command_candidates(line: str) -> list[str]:
    """The line stripped, plus its payload if it is a (possibly bulleted) run: key."""
    stripped = line.strip()
    out = [stripped]
    m = _RUN_KEY_RE.match(stripped)
    if m is not None and m.group(1):
        out.append(m.group(1))
    return out


def _runs_pytest(line: str) -> bool:
    for cand in _command_candidates(line):
        for m in _PYTEST_INVOKE_RE.finditer(cand):
            if cand[m.end() :].startswith("-cov"):
                continue  # pytest-cov the installable plugin, not the runner
            if _PIP_INSTALL_PREFIX_RE.search(cand[: m.start()]):
                continue  # an argument to pip install, not an invocation
            return True
    return False


def _runs_mutate(line: str) -> bool:
    for cand in _command_candidates(line):
        if _MUTATE_PATH_RE.search(cand) is None:
            continue
        if any(flag in cand for flag in _MUTATE_METADATA_FLAGS):
            continue  # metadata mode only; the shard enumerator runs zero mutants
        return True
    return False


def job_facts(block: JobBlock) -> JobFacts:
    # STATED LIMITATION: the haystack for BOTH probes is the WHOLE job block, not an
    # individually attributed step. That is sound here because a job has one install
    # step; to keep it sound if that ever stops being true, EVERY install line's
    # extras are collected, and evaluate() goes RED if one job yields more than one
    # DISTINCT extras set. Comment lines and blank lines are skipped: prose mentions
    # of pytest or .[dev] are not commands.
    extras_sets: set[frozenset[str]] = set()
    executes = False
    for line in block.lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for group in _EXTRAS_RE.findall(line):
            extras_sets.add(frozenset(part.strip() for part in group.split(",") if part.strip()))
        if _runs_pytest(line) or _runs_mutate(line):
            executes = True
    return JobFacts(name=block.name, extras_sets=extras_sets, executes_suite=executes)


def _fmt_extras(extras: frozenset[str]) -> str:
    if not extras:
        return "{} (no extras installed)"
    return "{" + ", ".join(sorted(extras)) + "}"


def _denominator(suite: Sequence[JobFacts], total: int) -> str:
    base = f"denominator: {len(suite)}/{total} jobs execute the suite"
    if suite:
        base += ": " + ", ".join(f.name for f in suite)
    return base


def evaluate(text: str) -> tuple[int, list[str]]:
    """The gate proper: (exit code, verdict lines) over one workflow's text."""
    try:
        blocks = parse_jobs(text)
    except WorkflowParseError as exc:
        return EXIT_UNMEASURED, [
            f"UNMEASURED: ci.yml could not be parsed by the structural scanner: {exc}. "
            "Unparseable is not empty (denominator: 0 suite-executing jobs measurable)."
        ]
    facts = [job_facts(b) for b in blocks]
    suite = [f for f in facts if f.executes_suite]
    denom = _denominator(suite, len(facts))

    conflicts = [f for f in facts if len(f.extras_sets) > 1]
    if conflicts:
        lines = [
            f"RED: {len(conflicts)} job(s) carry more than one DISTINCT extras set across"
            f" their 'pip install -e' lines -- which set the suite runs under is ambiguous,"
            f" and ambiguous is refused ({denom})."
        ]
        for f in conflicts:
            rendered = "; ".join(
                _fmt_extras(s) for s in sorted(f.extras_sets, key=lambda s: sorted(s))
            )
            lines.append(f"RED:   job '{f.name}' installs conflicting extras sets: {rendered}")
        return EXIT_RED, lines

    if len(suite) == 0:
        return EXIT_UNMEASURED, [
            f"UNMEASURED: zero suite-executing jobs found across {len(facts)} job(s) "
            f"({denom}). Looked for real pytest invocations and tools/mutate.py runs in"
            " every job block and found none. all([]) is True: agreement over an empty"
            " denominator is the vacuous pass this repository refuses -- UNMEASURED,"
            " never CLEAR."
        ]
    if len(suite) == 1:
        return EXIT_UNMEASURED, [
            f"UNMEASURED: exactly one suite-executing job ('{suite[0].name}') found across"
            f" {len(facts)} job(s) ({denom}). One job cannot disagree with itself, so"
            " agreement over one job is vacuous -- UNMEASURED, never CLEAR."
        ]

    distinct = {f.extras for f in suite}
    if len(distinct) == 1:
        agreed = next(iter(distinct))
        return EXIT_CLEAR, [
            f"CLEAR: all {len(suite)} suite-executing jobs install the SAME extras"
            f" {_fmt_extras(agreed)} ({denom}). The gate decided only that they AGREE;"
            " whether that set is the right one is a question this gate has no oracle for."
        ]

    union: set[str] = set()
    common: set[str] | None = None
    for f in suite:
        union |= set(f.extras)
        common = set(f.extras) if common is None else common & set(f.extras)
    shared: set[str] = common if common is not None else set()
    lines = [
        f"RED: suite-executing jobs DISAGREE on installed extras ({denom}). Divergence"
        " is the defect: an extra absent from one of these jobs is a collection error"
        " the suite will report as a wrong verdict, wearing an exit code."
    ]
    for f in suite:
        lines.append(f"RED:   job '{f.name}' installs {_fmt_extras(f.extras)}")
    lines.append(
        "RED:   symmetric difference (extras installed by some but NOT EVERY"
        f" suite-executing job): {_fmt_extras(frozenset(union - shared))};"
        f" common to all: {_fmt_extras(frozenset(shared))}"
    )
    return EXIT_RED, lines


# --- controls --------------------------------------------------------------------
@dataclass
class Control:
    label: str
    kind: str  # "MUST_FIRE" | "MUST_PASS"
    text: str
    expected: int


def _wf(body: str) -> str:
    return "name: synthetic-ci\n\non: push\n\njobs:\n" + body


def self_test_controls() -> list[Control]:
    mf = "MUST_FIRE"
    mp = "MUST_PASS"
    return [
        Control(
            label="two-suite-jobs-different-extras",
            kind=mf,
            expected=EXIT_RED,
            text=_wf(
                '\n  alpha:\n    steps:\n      - run: pip install -e ".[checkpoint,dev]"'
                "\n      - run: pytest -q\n  beta:\n    steps:"
                '\n      - run: pip install -e ".[checkpoint,train,dev]"'
                "\n      - run: pytest -q\n"
            ),
        ),
        Control(
            label="pytest-job-with-no-install-line",
            kind=mf,
            expected=EXIT_RED,
            text=_wf(
                '\n  alpha:\n    steps:\n      - run: pip install -e ".[dev]"'
                "\n      - run: pytest -q\n  beta:\n    steps:\n      - run: pytest -q\n"
            ),
        ),
        Control(
            label="zero-suite-executing-jobs",
            kind=mf,
            expected=EXIT_UNMEASURED,
            text=_wf(
                '\n  alpha:\n    steps:\n      - run: pip install -e ".[dev]"'
                "\n      - run: python -m foundationscale.gates.controls"
                '\n  beta:\n    steps:\n      - run: pip install -e ".[dev]"'
                "\n      - run: ruff check src tests\n"
            ),
        ),
        Control(
            label="exactly-one-suite-executing-job",
            kind=mf,
            expected=EXIT_UNMEASURED,
            text=_wf(
                '\n  alpha:\n    steps:\n      - run: pip install -e ".[dev]"'
                "\n      - run: pytest -q\n  beta:\n    steps:"
                "\n      - run: ruff check src tests\n"
            ),
        ),
        Control(
            label="unparseable-yaml",
            kind=mf,
            expected=EXIT_UNMEASURED,
            text="name: broken\njobs: {{\n  - [unclosed\n::: not a workflow :::\n",
        ),
        Control(
            label="mutate-py-job-in-denominator-with-divergent-extras",
            kind=mf,
            expected=EXIT_RED,
            text=_wf(
                '\n  alpha:\n    steps:\n      - run: pip install -e ".[checkpoint,train,dev]"'
                "\n      - run: pytest -q\n  beta:\n    steps:"
                '\n      - run: pip install -e ".[checkpoint,dev]"'
                "\n      - run: python tools/mutate.py --module checkpoint_gates\n"
            ),
        ),
        Control(
            label="identical-extras-order-permuted",
            kind=mp,
            expected=EXIT_CLEAR,
            text=_wf(
                '\n  alpha:\n    steps:\n      - run: pip install -e ".[a,b]"'
                "\n      - run: pytest -q\n  beta:\n    steps:"
                '\n      - run: pip install -e ".[b,a]"\n      - run: pytest -q\n'
            ),
        ),
        Control(
            label="non-suite-job-with-different-extras-cannot-redden",
            kind=mp,
            expected=EXIT_CLEAR,
            text=_wf(
                '\n  alpha:\n    steps:\n      - run: pip install -e ".[dev]"'
                "\n      - run: pytest -q\n  beta:\n    steps:"
                '\n      - run: pip install -e ".[dev]"\n      - run: pytest -q'
                "\n  gamma:\n    steps:"
                '\n      - run: pip install -e ".[checkpoint,train,dev,docs]"'
                "\n      - run: python -m foundationscale.gates.controls\n"
            ),
        ),
        Control(
            label="installs-pytest-cov-and-mentions-pytest-never-runs-it",
            kind=mp,
            expected=EXIT_CLEAR,
            text=_wf(
                '\n  alpha:\n    steps:\n      - run: pip install -e ".[dev]"'
                "\n      - run: pytest -q\n  beta:\n    steps:"
                '\n      - run: pip install -e ".[dev]"\n      - run: pytest -q'
                "\n  gamma:\n    steps:"
                '\n      - run: pip install -e ".[docs]" "pytest-cov>=5"'
                "\n      # This job mentions pytest in prose but never invokes it."
                "\n      - run: python -m foundationscale.gates.controls\n"
            ),
        ),
    ]


def self_test() -> int:
    controls = self_test_controls()
    failures = 0
    for c in controls:
        got, _lines = evaluate(c.text)
        ok = got == c.expected
        if not ok:
            failures += 1
        outcome = "ok" if ok else "WRONG STATE -- a control that lies voids the licence"
        print(f"control {c.kind}/{c.label}: expected exit {c.expected}, got {got} -- {outcome}")
    n_ok = len(controls) - failures
    n_fire = sum(1 for c in controls if c.kind == "MUST_FIRE")
    n_pass = len(controls) - n_fire
    # House tally format, shared with checks/coverage_floor.py and
    # checks/packaging_reachability.py: "SELF-TEST DENOMINATOR: N of M controls
    # behaved; ...". It is not cosmetic. The launcher suite reads this line to
    # detect a self-test whose control set has SHRUNK -- a gate that quietly
    # drops controls keeps exiting 0 while measuring less, which is the failure
    # a bare rc check cannot see. The first draft of this gate printed its own
    # "controls: N passed (...)" wording and the suite leg, authored in parallel
    # and blind to it, parsed for the house format; the tally read as absent and
    # the MUST_PASS leg would have reported UNMEASURED against a working gate.
    # One format, one parser: a new gate conforms here rather than the suite
    # growing another speculative alternative for each gate's private phrasing.
    tally = (
        f"SELF-TEST DENOMINATOR: {n_ok} of {len(controls)} controls behaved; "
        f"{n_fire}x MUST_FIRE produced the declared nonzero exits, "
        f"{n_pass}x MUST_PASS stayed CLEAR over jobs this gate must not redden"
    )
    if failures:
        print(
            f"controls: {failures} control(s) missed their required state -- a gate"
            " whose controls misbehave has no licence to report a verdict."
        )
        print(tally)
        return EXIT_RED
    print(tally)
    return EXIT_CLEAR


class GateArgumentParser(argparse.ArgumentParser):
    """argparse that fails CLOSED: bad usage is REFUSE (96), never a guessed run."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(
            f"REFUSE: bad CLI usage: {message} -- this gate exits {EXIT_REFUSE} rather"
            " than guess what was meant.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_REFUSE)


def main(argv: Sequence[str] | None = None) -> int:
    parser = GateArgumentParser(
        description="Gate: every CI job that EXECUTES the pytest suite installs the "
        "SAME extras set. Divergence is the defect; agreement is all this gate claims."
    )
    parser.add_argument(
        "--workflow",
        default=DEFAULT_WORKFLOW,
        metavar="PATH",
        help=f"workflow file to gate (default: {DEFAULT_WORKFLOW})",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the MUST_FIRE/MUST_PASS controls over synthetic in-memory workflows",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        text = Path(args.workflow).read_text(encoding="utf-8")
    except OSError as exc:
        print(
            f"UNMEASURED: cannot read {args.workflow}: {exc} -- unreadable is not empty"
            " (denominator: 0 suite-executing jobs measurable)."
        )
        return EXIT_UNMEASURED
    code, lines = evaluate(text)
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
