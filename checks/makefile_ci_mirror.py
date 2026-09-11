#!/usr/bin/env python3
"""checks/makefile_ci_mirror.py -- the Makefile <-> CI mirror gate (finding #286).

The Makefile's `check` target and .github/workflows/ci.yml are supposed to
run the SAME checks. Today that mirror is a hand-synchronised convention,
and it has already drifted in BOTH directions (#231: CI's mypy was
narrower than the Makefile's; the `check` target's `-$(PY) -m mypy
checks` prefix was the reverse). This gate extracts the check commands
from BOTH files, normalises away the differences that are allowed
(interpreter spelling, env arming placement, make-vs-yaml quoting,
quoted command substitutions), and reports any check script that runs
on one side of the mirror and not on the other.

THE COMPARISON KEY IS THE SCRIPT, NOT THE ARGV. Each extracted command
is reduced to its first token -- the script path (tools/mutate.py,
checks/countables_drift.py), the tool name (pytest, ruff, mypy,
coverage), or the dotted module (foundationscale.gates.controls) --
and the two SETS of keys are compared. Arguments are not compared,
because the two sides legitimately differ in output paths ($(CURDIR)
against $RUNNER_TEMP), in sharding (one whole-corpus tools/mutate.py
run against a per-module job matrix), and in whether a --self-test
mode is invoked directly or reached through
checks/campaign_self_tests.py. The first version of this gate keyed on
the literal argument string and reported four findings on the real
tree; all four were false, because the claim is about WHICH checks
run and the detector's denominator was the argv. A gate that is RED
on a correct tree gets an exception added or gets deleted, so the
denominator is now the thing the claim is about.

THE BLIND SPOT, STATED: a check that runs on both sides with different
SEMANTIC arguments -- a narrower path set on one side, a mode disabled
on one side -- produces the same key on both sides and is invisible
here. That is the price of not reddening on environment syntax, and
this gate says so in its banner on every verdict, CLEAR or RED.
Narrowing a corpus or disabling a mode is real drift; it is simply
not THIS gate's claim, and it must be caught by review or by a gate
whose claim IS the arguments, not by quietly pretending the argv
comparison used to catch it.

Exit contract (never exit 1):
  0  CLEAR      the set of check-script keys extracted from the
                Makefile `check` tree equals the set extracted from
                the CI workflow.
  5  RED        at least one check-script key is Makefile-only or
                CI-only.
  95 UNMEASURED  a file is unreadable, or its format is unrecognised
                (no `check:` target, no `run:` entries, a `check`
                prerequisite with no target definition, an
                unresolvable $(VAR) in a recipe line that carries a
                check command, or zero check commands extractable
                from the Makefile side). "I could not parse it" is
                NOT "they disagree".
  96 REFUSE     CLI misuse, or --self-test found the gate's own
                controls broken (a gate that cannot fail its own
                fixtures must not be trusted to judge the repo).

EXACT INPUT THAT MAKES THIS GATE RED (also exercised by --self-test):
  In .github/workflows/ci.yml delete BOTH steps that invoke
  checks/coverage_floor.py while the Makefile's `coverage-floor`
  target still runs it. The gate prints `checks/coverage_floor.py`
  under "Makefile-only" and exits 5. The reverse direction is also
  RED: add `run: python3 tools/some_new_gate.py` to the workflow
  without adding it to any target in the Makefile's `check` tree and
  the gate prints it under "CI-only" and exits 5. Note that deleting
  only ONE of several steps that share a key (say, the ruff format
  step while the ruff check step survives) is NOT visible here --
  the key `ruff` still runs on both sides. That is the blind spot
  named above, pinned by MUST_NOT_FIRE leg 5 of the self-test so it
  stays measured rather than assumed. A gate with no failing input
  is worthless; this one has two, one per direction, because the
  drift has happened both ways.

WHAT COUNTS AS A CHECK COMMAND:
  A line whose normalised first token is one of the gated tool names
  (pytest, ruff, mypy, coverage -- the same tool family checks/
  makefile_tooling.py fences off per the Makefile note at line 260), a
  *.py script path (checks/*.py, tools/*.py), or a `python -m` module
  invocation (e.g. `foundationscale.gates.controls`). Everything else --
  heredoc bodies, fixture writers, echo/grep assertions -- is harness, and
  is allowed to differ syntactically between a make recipe and a yaml
  block scalar without being drift.

REJECTED FRAGMENTS:
  A segment whose first token fails the plausibility test for a key
  (it must match ^[A-Za-z0-9_./-]+$ after normalisation) but whose
  stem still looks like a check command is a slicing artefact -- a
  fragment the extractor cut out of a larger shell construct, such as
  the `pytest.skip("deliberate skip` half of the string that carries
  a semicolon inside the CI probe's heredoc. Fragments are never keys
  and never findings, but they are COUNTED, and the count prints on
  every verdict: small and non-zero is normal on the real tree, and
  a sudden jump means the extractor is losing sight of real commands.

WHAT IS DELIBERATELY NOT MIRRORED HERE:
  `pip install ...` lines. Provisioning is environment setup, not a check;
  which extras suite-executing jobs install is gated by
  checks/ci_suite_extras.py (finding #253). Widening this gate's scope to
  install lines would double-gate #253 and would false-RED on the shipped
  tree, where `install` is not a prerequisite of `check`.
"""

from __future__ import annotations

import re
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

CLEAR = 0
RED = 5
UNMEASURED = 95
REFUSE = 96

TOOLS = frozenset({"pytest", "ruff", "mypy", "coverage"})
PROVISIONING = frozenset({"pip"})

_DOLLAR = "\x00DOLLAR\x00"  # sentinel for make's `$$` (shell dollar)
_PY = "\x00PY\x00"  # sentinel for the make variable $(PY)

_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)[ \t]*[:+?]?=[ \t]*(.*)$")
_TARGET_RE = re.compile(r"^([^\s#:=][^:=]*?)[ \t]*:(?![:=])[ \t]*(.*)$")
_RUN_RE = re.compile(r"^( *)(-( ))?run:[ \t]*(.*)$")
_VAR_RE = re.compile(r"\$[\(\{]([A-Za-z_][A-Za-z0-9_]*)[\)\}]")
_ENV_RE = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=[^\s()]+ +)+")
# The unwrap tolerates quotes around the command substitution, because
# that is how the real tree writes it: ci.yml's shard enumeration is
# `modules="$(python tools/mutate.py --list-modules)"`. Without the
# optional quote arms the unwrap did not match, the interpreter strip
# below then matched the `python` buried mid-line, and the gate
# compared a fragment sliced out of the middle of the assignment.
_UNWRAP_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=[\"']?\$\((.*)\)[\"']? *;? *$")
_INTERP_RE = re.compile(r"^(?:" + _PY + r"|\S*python[0-9.]*)( -m)? ")
_MODULE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+")
_KEY_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
_ARTEFACT_RE = re.compile(r"[)\"'${}(]")
_WS_RE = re.compile(r"\s+")
_PREFIX_RE = re.compile(r"^[@\-+]+")
_SEGMENT_SPLIT_RE = re.compile(r";|&&|\|\|")


def _logical_lines(physical: Iterable[str]) -> Iterator[str]:
    """Join backslash continuations, then split each logical line into
    shell command segments (on `;`, `&&`, `||`)."""
    buf = ""
    for raw in physical:
        piece = raw.strip()
        if buf:
            piece = buf + piece
            buf = ""
        if piece.endswith("\\"):
            buf = piece[:-1] + " "
            continue
        for seg in _SEGMENT_SPLIT_RE.split(piece):
            seg = seg.strip()
            if seg:
                yield seg
    if buf.strip():
        for seg in _SEGMENT_SPLIT_RE.split(buf.strip()):
            seg = seg.strip()
            if seg:
                yield seg


def normalize(segment: str, variables: Mapping[str, str]) -> tuple[str | None, str | None]:
    """Reduce one shell command segment to its canonical comparison form.

    Returns (command_or_None, unresolved_var_or_None)."""
    s = _WS_RE.sub(" ", segment).strip()
    if not s or s.startswith("#"):
        return None, None
    s = _PREFIX_RE.sub("", s).strip()  # make recipe prefixes - @ +
    if not s:
        return None, None
    s = s.replace("$$", _DOLLAR)  # protect shell dollars
    s = s.replace("$(PY)", _PY).replace("${PY}", _PY)
    unresolved = None

    def _sub(match: re.Match[str]) -> str:
        nonlocal unresolved
        name = match.group(1)
        if name in variables:
            return variables[name]
        unresolved = name
        return match.group(0)

    for _ in range(10):  # expand defined make vars only
        expanded = _VAR_RE.sub(_sub, s)
        if expanded == s:
            break
        s = expanded
    s = s.replace(_DOLLAR, "$")

    while True:  # fixed point of the three strips
        m = _UNWRAP_RE.match(s)  # output=$(cmd) / output="$(cmd)" -> cmd
        if m:
            s = m.group(1).strip()
            continue
        m = _ENV_RE.match(s)  # FS_FORBID_SKIPS=1 cmd -> cmd
        if m:
            s = s[m.end() :]
            continue
        m = _INTERP_RE.match(s)  # $(PY) -m cmd / python3 cmd -> cmd
        if m:
            s = s[m.end() :]
            continue
        break
    s = s.strip()
    return (s or None), unresolved


def script_key(command: str) -> str | None:
    """Reduce a normalised command to its comparison key: the first
    whitespace-separated token, i.e. the script path, the tool name, or
    the dotted module. Arguments are deliberately NOT part of the key --
    the claim of this gate is about WHICH checks run, and the two sides
    legitimately differ in output paths, sharding and --self-test
    placement, all of which live in the argv.

    Returns None when the token is not a plausible script/tool/module
    token. A token holding a shell artefact (a parenthesis, a quote, a
    `$`, a brace) is not a command; it is a fragment the extractor
    sliced out of a larger construct, and it must never become a key."""
    token = command.split(" ", 1)[0]
    if _KEY_RE.fullmatch(token) is None:
        return None
    return token


def is_check_command(command: str) -> bool:
    token = command.split(" ", 1)[0]
    if token in PROVISIONING:
        return False
    if token in TOOLS or token.endswith(".py"):
        return True
    return _MODULE_RE.fullmatch(token) is not None


def _looks_like_check_stem(token: str) -> bool:
    """True when a token that FAILED the key plausibility test still has
    a check command's stem: cut it at the first shell artefact and what
    remains names a gated tool, a *.py path, or a dotted module. Such a
    token is the slicing artefact of a real check command -- the
    semicolon inside the CI probe's `pytest.skip("...")` string is the
    standing example -- and those are counted, so the narrowing stays
    visible rather than silent."""
    stem = _ARTEFACT_RE.split(token, 1)[0]
    if not stem:
        return False
    if stem in TOOLS or stem.endswith(".py"):
        return True
    return _MODULE_RE.fullmatch(stem) is not None


def parse_makefile(
    text: str,
) -> tuple[dict[str, str], dict[str, list[str]], list[str] | None]:
    """Return (variables, targets, check_prereqs_or_None)."""
    variables: dict[str, str] = {}
    targets: dict[str, list[str]] = {}
    check_prereqs = None
    current = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        if raw.startswith("\t"):
            if current is not None:
                targets.setdefault(current, []).append(raw.lstrip("\t"))
            continue
        while raw.rstrip().endswith("\\") and i < len(lines) and not lines[i].startswith("\t"):
            raw = raw.rstrip()[:-1] + " " + lines[i].strip()
            i += 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _ASSIGN_RE.match(raw)
        if m:
            variables[m.group(1)] = m.group(2).split("#", 1)[0].strip()
            current = None
            continue
        m = _TARGET_RE.match(raw)
        if m:
            names = [n for n in m.group(1).split() if not n.startswith(".")]
            prereq_text = m.group(2).split("#", 1)[0].split("|", 1)[0]
            prereqs = [t for t in prereq_text.split() if "$" not in t and "%" not in t]
            current = names[0] if names else None
            if current is not None:
                targets.setdefault(current, [])
            if current == "check":
                check_prereqs = prereqs
            continue
        current = None
    return variables, targets, check_prereqs


def parse_ci(text: str) -> list[tuple[int, list[str]]]:
    """Return [(start_lineno, [physical lines of that run step]), ...]."""
    entries = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        m = _RUN_RE.match(line)
        if not m:
            continue
        body = m.group(4)
        if body and body[0] not in "|>":
            entries.append((idx + 1, [body]))
            continue
        baseline = len(m.group(1)) + len(m.group(2) or "")
        block = []
        for follower in lines[idx + 1 :]:
            indent = len(follower) - len(follower.lstrip(" "))
            if follower.strip() and indent <= baseline:
                break
            block.append(follower)
        entries.append((idx + 1, block))
    return entries


def _collect(
    entries: Sequence[tuple[str, Sequence[str]]],
    variables: Mapping[str, str],
    origin_label: str,
) -> tuple[dict[str, dict[str, set[str]]], list[str], int]:
    """Normalise physical lines into a {key: {command: set(origins)}}
    map keyed by script_key, plus a list of parse problems (unresolved
    make-style variables in lines that would otherwise be check
    commands), plus a count of rejected fragments (segments that looked
    like check commands but whose first token failed the plausibility
    test). The full observed commands are kept per key so the report
    can show what each key was seen as; they just are not compared."""
    commands: dict[str, dict[str, set[str]]] = {}
    problems: list[str] = []
    fragments = 0
    for origin, lines in entries:
        for seg in _logical_lines(lines):
            command, unresolved = normalize(seg, variables)
            if command is None:
                continue
            key = script_key(command)
            if not is_check_command(command):
                if key is None and _looks_like_check_stem(command.split(" ", 1)[0]):
                    fragments += 1
                continue
            if key is None:
                # A check-shaped command whose own first token is a
                # slicing artefact: a fragment, never a key.
                fragments += 1
                continue
            if unresolved is not None:
                problems.append(
                    f"{origin_label} {origin}: cannot resolve $({unresolved}); "
                    f"without its value the command {command!r} cannot be "
                    f"compared across the mirror"
                )
                continue
            commands.setdefault(key, {}).setdefault(command, set()).add(str(origin))
    return commands, problems, fragments


def _origins(seen: Mapping[str, set[str]]) -> str:
    """Flatten a {command: set(origins)} map into one sorted origin list."""
    return ", ".join(sorted({o for origins in seen.values() for o in origins}))


def _banner(fragments: int) -> str:
    """The narrowing banner, printed on EVERY verdict, CLEAR and RED
    alike: what this gate compares, why it no longer compares argv, and
    what that choice cannot see. A narrowing that is only printed on
    the red runs is a narrowing nobody believes on the green ones."""
    return "\n".join(
        [
            "----------------------------------------------------------------------",
            "WHAT THIS GATE COMPARES, AND WHAT IT DOES NOT",
            "----------------------------------------------------------------------",
            "This gate compares WHICH check scripts run on each side of the",
            "mirror: the set of script paths, tool names and `python -m`",
            "modules extracted from the Makefile 'check' tree against the set",
            "extracted from the CI workflow's run steps. It does NOT compare",
            "their arguments.",
            "",
            "The narrowing is deliberate. The two sides legitimately differ in",
            'output paths ($(CURDIR)/... in the Makefile against "$RUNNER_TEMP/...',
            'in CI"), in sharding (one whole-corpus tools/mutate.py run in the',
            "Makefile against a per-module job matrix in CI), and in whether a",
            "--self-test mode is invoked directly or reached through",
            "checks/campaign_self_tests.py. Keying on the literal argument",
            "string made this gate RED on a correct tree -- all four of its",
            "findings that day were false -- and a gate that is RED on a",
            "correct tree gets an exception added or gets deleted.",
            "",
            "The cost is stated, not hidden: a check invoked on BOTH sides",
            "with different semantic arguments -- a narrower path set on one",
            "side, a mode disabled on one side -- yields the same key twice",
            "and is invisible here. That blind spot is real drift this gate",
            "cannot see. It is pinned by MUST_NOT_FIRE leg 5 of --self-test",
            "so it stays measured rather than assumed, and it must be caught",
            "by review or by a gate whose claim IS the arguments.",
            "",
            "SECOND NARROWING: shell suites are in NO denominator here. A",
            "check command is recognised only if it is a *.py path, one of",
            "the tool names, or a `python -m` module. Shell suites invoked",
            "as `bash <path>` are skipped on BOTH sides, so they are neither",
            "compared nor reported. That was measured, not assumed: the",
            "Makefile names launchers/test_launcher_contracts.sh and",
            "launchers/test_checks_gates.sh as literals in `bash <path>`",
            "recipes, but CI reaches those same two suites through a loop",
            'variable -- `for suite_pair in "<path>:$FLOOR" ...;',
            'suite=${suite_pair%:*}; output=$(bash "$suite" 2>&1)` -- so the',
            "literal path never appears in a CI command position at all.",
            "Widening the recogniser to *.sh would therefore emit two",
            "Makefile-only findings on a correct tree: exactly the false-RED",
            "failure this rewrite exists to remove. Other shell suites (for",
            "instance run_standing_gates.sh) DO appear as literals on both",
            "sides and would mirror fine -- it is the mixture that makes the",
            "widening unsafe, not shell as such. Consequence: a shell suite",
            "added to one side and not the other is invisible to this gate.",
            "It is pinned by MUST_NOT_FIRE leg 9 of --self-test. Closing it",
            "needs a CI-side extractor that resolves loop variables, which",
            "is a different claim and should be a different gate.",
            "",
            f"fragments rejected: {fragments} (segments that looked like check",
            "commands but whose first token was a slicing artefact, not a",
            "command; small and non-zero is normal on the real tree, and a",
            "sudden jump means the extractor is losing sight of real commands)",
        ]
    )


def evaluate(makefile_path: Path | str, ci_path: Path | str) -> tuple[int, str]:
    """Return (exit_code, human_readable_report)."""
    problems = []
    try:
        make_text = Path(makefile_path).read_text(encoding="utf-8")
    except OSError as exc:
        return UNMEASURED, f"UNMEASURED: Makefile unreadable at {makefile_path}: {exc}"
    try:
        ci_text = Path(ci_path).read_text(encoding="utf-8")
    except OSError as exc:
        return UNMEASURED, f"UNMEASURED: CI workflow unreadable at {ci_path}: {exc}"

    variables, targets, prereqs = parse_makefile(make_text)
    # $(CURDIR) is make's own absolute path to the directory it runs in, which
    # is the directory holding this Makefile. Seeding it is not a guess -- it is
    # the value make itself computes. Without it every recipe that mentions
    # $(CURDIR) is unresolvable, and since `countables` does, the gate abstained
    # on EVERY run of this repo. A gate that can only ever return UNMEASURED is
    # an inert gate (the #278/#293 shape), not a cautious one: it has no failing
    # input, so it certifies nothing. setdefault, not assignment, so an explicit
    # CURDIR := in the Makefile still wins.
    variables.setdefault("CURDIR", str(Path(makefile_path).resolve().parent))
    make_cmds: dict[str, dict[str, set[str]]] = {}
    make_fragments = 0
    if prereqs is None:
        problems.append(
            "no 'check:' target found in the Makefile; the mirror has no "
            "Makefile-side definition to compare against"
        )
    else:
        for name in prereqs:
            if name not in targets:
                problems.append(
                    f"check prerequisite {name!r} has no target definition "
                    f"in the Makefile; cannot know what CI should mirror"
                )
        if not problems:
            entries = [(f"target {name!r}", targets[name]) for name in prereqs]
            make_cmds, mk_problems, make_fragments = _collect(entries, variables, "Makefile")
            problems.extend(mk_problems)
            if not make_cmds and not mk_problems:
                problems.append(
                    "zero check commands extracted from the Makefile "
                    "'check' tree; the parser is blind, not the mirror clear"
                )

    ci_entries = parse_ci(ci_text)
    ci_cmds: dict[str, dict[str, set[str]]] = {}
    ci_fragments = 0
    if not ci_entries:
        problems.append("no 'run:' entries found in the CI workflow; unrecognised workflow format")
    else:
        ci_cmds, ci_problems, ci_fragments = _collect(
            [(f"line {n}", lines) for n, lines in ci_entries], {}, "CI"
        )
        problems.extend(ci_problems)

    if problems:
        report = [
            "UNMEASURED: could not parse one or both sides of the "
            "mirror; this is not a disagreement (exit 95, not 5):"
        ]
        report.extend(f"  - {p}" for p in problems)
        return UNMEASURED, "\n".join(report)

    fragments = make_fragments + ci_fragments
    only_make = sorted(set(make_cmds) - set(ci_cmds))
    only_ci = sorted(set(ci_cmds) - set(make_cmds))
    if only_make or only_ci:
        report = [
            "RED: the Makefile 'check' tree and the CI workflow run "
            "different check scripts (the hand-synchronised mirror has "
            "drifted again):"
        ]
        for key in only_make:
            report.append(f"  Makefile-only: {key}   [{_origins(make_cmds[key])}]")
            for cmd in sorted(make_cmds[key]):
                report.append(f"    observed as: {cmd}")
        for key in only_ci:
            report.append(f"  CI-only:       {key}   [{_origins(ci_cmds[key])}]")
            for cmd in sorted(ci_cmds[key]):
                report.append(f"    observed as: {cmd}")
        report.append(
            "Add the missing check to the other file, or delete it from "
            "both. Drift in either direction is RED."
        )
        report.append("")
        report.append(_banner(fragments))
        return RED, "\n".join(report)

    # #310: the count and the names come from ONE list, and the names are
    # printed. A bare "18 scripts mirrored" is unauditable -- it cannot be
    # told apart from 18 of the wrong scripts, or from an extractor that
    # found 18 things that are not checks at all. Enumerating the
    # denominator is what makes a CLEAR verdict checkable by a reader.
    both = sorted(make_cmds)
    report = [
        f"CLEAR: {len(both)} check scripts run on both sides of the "
        f"mirror (Makefile 'check' tree against {len(ci_entries)} CI run "
        f"steps); nothing is single-sided.",
        "",
        "The denominator, named rather than counted:",
    ]
    for key in both:
        report.append(
            f"  {key}   [Makefile: {_origins(make_cmds[key])}] [CI: {_origins(ci_cmds[key])}]"
        )
    report.append("")
    report.append(_banner(fragments))
    return CLEAR, "\n".join(report)


_MK_PROBE_RUFF = (
    "PY = python3\n"
    "\n"
    "check: probe lint\n"
    "\n"
    "probe:\n"
    "\t$(PY) checks/probe_gate.py --out $(CURDIR)/probe.json\n"
    "\n"
    "lint:\n"
    "\t$(PY) -m ruff check src tests\n"
)

_MK_PROBE = (
    "PY = python3\n"
    "\n"
    "check: probe\n"
    "\n"
    "probe:\n"
    "\t$(PY) checks/probe_gate.py --out $(CURDIR)/probe.json\n"
)

_MK_PROBE_SELFTEST = (
    "PY = python3\n\ncheck: probe\n\nprobe:\n\t$(PY) checks/probe_gate.py --self-test\n"
)

_MK_RUFF_ONLY = "PY = python3\n\ncheck: lint\n\nlint:\n\t$(PY) -m ruff check src tests\n"

_MK_MUTATE = (
    "PY = python3\n\ncheck: mutation\n\nmutation:\n\tFS_FORBID_SKIPS=1 $(PY) tools/mutate.py\n"
)

_MK_NO_CHECK = "lint:\n\t$(PY) -m ruff check .\n"

_CI_PROBE = (
    "name: CI\n"
    "jobs:\n"
    "  test:\n"
    "    steps:\n"
    "      - name: Probe\n"
    '        run: python3 checks/probe_gate.py --out "$RUNNER_TEMP/probe.json"\n'
)

_CI_PROBE_NO_SELFTEST = (
    "name: CI\n"
    "jobs:\n"
    "  test:\n"
    "    steps:\n"
    "      - name: Probe\n"
    "        run: python3 checks/probe_gate.py\n"
)

_CI_RUFF_ONLY = (
    "name: CI\njobs:\n  test:\n    steps:\n      - name: Ruff\n        run: ruff check src tests\n"
)

_CI_RUFF_PLUS = _CI_RUFF_ONLY + (
    "      - name: Extra gate\n        run: python3 tools/extra_gate.py --verify\n"
)

_CI_SHARDED = (
    "name: CI\n"
    "jobs:\n"
    "  mutation:\n"
    "    steps:\n"
    "      - name: Run one shard\n"
    "        run: |\n"
    '          python tools/mutate.py --module "${{ matrix.module }}"\n'
)

_CI_LIST_MODULES = (
    "name: CI\n"
    "jobs:\n"
    "  shards:\n"
    "    steps:\n"
    "      - name: Enumerate shards\n"
    "        run: |\n"
    '          modules="$(python tools/mutate.py --list-modules)"\n'
    '          echo "sharding across: $modules"\n'
)

_MK_SHELL_SUITE = (
    "PY = python3\n"
    "\n"
    "check: lint suites\n"
    "\n"
    "lint:\n"
    "\t$(PY) -m ruff check src tests\n"
    "\n"
    "suites:\n"
    "\tbash launchers/test_fixture_suite.sh\n"
)

# The shape MEASURED in .github/workflows/ci.yml: the two launcher suites
# are reached through a loop variable, so the suite path never occupies a
# command position anywhere in the workflow. This is why the recogniser
# is not widened to *.sh -- see the SECOND NARROWING paragraph of the
# banner.
_CI_SHELL_SUITE_VIA_LOOP = (
    "name: CI\n"
    "jobs:\n"
    "  test:\n"
    "    steps:\n"
    "      - name: Ruff\n"
    "        run: ruff check src tests\n"
    "      - name: Suites\n"
    "        run: |\n"
    '          for suite_pair in "launchers/test_fixture_suite.sh:31"; do\n'
    "            suite=${suite_pair%:*}\n"
    '            output=$(bash "$suite" 2>&1)\n'
    "          done\n"
)

_SELF_TEST_LEGS = [
    # (name, planted, demanded, makefile_text, ci_text, want_code, substrings)
    (
        "MUST_FIRE 1: a Makefile-only check script is RED",
        "planted: checks/probe_gate.py runs in the Makefile `check` tree and no CI step runs it",
        'demanded: exit 5, and the report names the script under "Makefile-only"',
        _MK_PROBE_RUFF,
        _CI_RUFF_ONLY,
        RED,
        ["RED", "Makefile-only", "checks/probe_gate.py"],
        [],
    ),
    (
        "MUST_FIRE 2: a CI-only check script is RED (the other direction)",
        "planted: tools/extra_gate.py runs in CI and no target in the "
        "Makefile `check` tree runs it",
        'demanded: exit 5, and the report names the script under "CI-only"',
        _MK_RUFF_ONLY,
        _CI_RUFF_PLUS,
        RED,
        ["RED", "CI-only", "tools/extra_gate.py"],
        [],
    ),
    (
        "MUST_NOT_FIRE 3: the same script with a different --out path is CLEAR",
        "planted: checks/probe_gate.py on both sides, writing "
        '$(CURDIR)/probe.json in the Makefile and "$RUNNER_TEMP/probe.json" in CI',
        "demanded: exit 0 -- the output path is environment syntax, not drift",
        _MK_PROBE,
        _CI_PROBE,
        CLEAR,
        ["CLEAR"],
        [],
    ),
    (
        "MUST_NOT_FIRE 4: a whole run against its own sharding is CLEAR",
        "planted: one whole-corpus tools/mutate.py run in the Makefile, "
        "tools/mutate.py --module X sharded across a matrix in CI",
        "demanded: exit 0 -- sharding is how CI runs the SAME check",
        _MK_MUTATE,
        _CI_SHARDED,
        CLEAR,
        ["CLEAR"],
        [],
    ),
    (
        "MUST_NOT_FIRE 5: --self-test on one side only is CLEAR",
        "planted: checks/probe_gate.py --self-test in the Makefile, "
        "checks/probe_gate.py bare in CI",
        "demanded: exit 0 -- this is the acknowledged blind spot, pinned "
        "here so it is measured rather than assumed",
        _MK_PROBE_SELFTEST,
        _CI_PROBE_NO_SELFTEST,
        CLEAR,
        ["CLEAR"],
        [],
    ),
    (
        "MUST_NOT_FIRE 6: the quoted command substitution is CLEAR",
        'planted: modules="$(python tools/mutate.py --list-modules)" in '
        "CI against a bare tools/mutate.py run in the Makefile",
        "demanded: exit 0, and fragments rejected is 0 -- the unwrap must "
        "yield the clean key, not a sliced fragment",
        _MK_MUTATE,
        _CI_LIST_MODULES,
        CLEAR,
        ["CLEAR", "fragments rejected: 0"],
        [],
    ),
    (
        "ABSTAIN 7: an unreadable CI workflow is UNMEASURED",
        "planted: a CI path that does not exist",
        "demanded: exit 95, not 5 -- 'I could not parse it' is not 'they disagree'",
        _MK_PROBE,
        None,
        UNMEASURED,
        ["UNMEASURED", "unreadable"],
        [],
    ),
    (
        "ABSTAIN 8: a Makefile with no check: target is UNMEASURED",
        "planted: a Makefile whose only target is `lint`",
        "demanded: exit 95, not 5 -- the mirror has no Makefile-side definition to compare against",
        _MK_NO_CHECK,
        _CI_PROBE,
        UNMEASURED,
        ["UNMEASURED", "check:"],
        [],
    ),
    (
        "MUST_NOT_FIRE 9: a shell suite is in NO denominator, silently",
        "planted: `bash launchers/test_fixture_suite.sh` as a literal in "
        "the Makefile `check` tree, and the SAME suite in CI reached only "
        "through a loop variable -- the measured ci.yml shape",
        "demanded: exit 0, AND the suite name appears nowhere in the "
        "report -- this is the second acknowledged blind spot, pinned "
        "here because widening the recogniser to *.sh would false-RED on "
        "exactly this correct tree",
        _MK_SHELL_SUITE,
        _CI_SHELL_SUITE_VIA_LOOP,
        CLEAR,
        ["CLEAR", "fragments rejected: 0"],
        ["launchers/test_fixture_suite.sh"],
    ),
    (
        "ABSTAIN 10: an unreadable Makefile is UNMEASURED (the other side)",
        "planted: a Makefile path that does not exist, against a readable CI "
        "workflow -- the mirror image of ABSTAIN 7",
        "demanded: exit 95, not 5 -- both sides of the mirror must abstain "
        "the same way, and without this leg the Makefile-unreadable branch "
        "had no failing input",
        None,
        _CI_PROBE,
        UNMEASURED,
        ["UNMEASURED", "Makefile unreadable"],
        [],
    ),
]


def _self_test() -> int:
    failures = []
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for idx, (
            name,
            planted,
            demanded,
            mk_text,
            ci_text,
            want,
            subs,
            forbidden,
        ) in enumerate(_SELF_TEST_LEGS):
            mk = base / f"leg{idx}.Makefile"
            ci = base / f"leg{idx}.ci.yml"
            try:
                if mk_text is None:
                    mk = base / f"leg{idx}.absent.Makefile"
                else:
                    mk.write_text(mk_text, encoding="utf-8")
                if ci_text is None:
                    ci = base / f"leg{idx}.absent.ci.yml"
                else:
                    ci.write_text(ci_text, encoding="utf-8")
                code, report = evaluate(mk, ci)
            except Exception as exc:  # the gate must never raise
                failures.append(f"{name}: evaluate raised {exc!r}")
                results.append((name, planted, demanded, False))
                continue
            leg_failures = []
            if code != want:
                leg_failures.append(f"want exit {want}, got {code}")
            for sub in subs:
                if sub not in report:
                    leg_failures.append(f"report missing {sub!r}")
            # A leg that only demands an exit code cannot tell "correctly
            # silent" from "silently absent". The forbidden list is how a
            # blind-spot leg states what must NOT appear: leg 9 is only
            # meaningful if the shell suite is missing from the report
            # entirely, not merely absent from the single-sided lists.
            for sub in forbidden:
                if sub in report:
                    leg_failures.append(f"report must not contain {sub!r}")
            if leg_failures:
                failures.append(f"{name}: {'; '.join(leg_failures)}\n{report}")
            results.append((name, planted, demanded, not leg_failures))
    for name, planted, demanded, ok in results:
        verdict = "PASS" if ok else "FAIL"
        print(f"{verdict}  {name}")
        print(f"      {planted}")
        print(f"      {demanded}")
    # The tally is printed in the "SELF-TEST DENOMINATOR: N of M controls
    # behaved" grammar six sibling gates already use (campaign_self_tests,
    # ci_suite_extras, coverage_floor, mutation_scope, packaging_reachability,
    # training_plane_probe). Inventing a seventh phrasing would give this gate
    # its own private parse, which is the drift class the campaign keeps
    # closing. It is printed on BOTH arms deliberately: a reader -- and the
    # shrink-floor leg in launchers/test_checks_gates.sh -- needs the
    # denominator most on the arm where the controls did NOT all behave, and a
    # failing self-test that prints no count cannot be told from one whose
    # control set was silently emptied.
    behaved = len(_SELF_TEST_LEGS) - len(failures)
    if failures:
        print(
            f"SELF-TEST DENOMINATOR: {behaved} of {len(_SELF_TEST_LEGS)} "
            "controls behaved; the rest are named below."
        )
        print(
            "REFUSE: makefile_ci_mirror.py failed its own controls; a "
            "gate that cannot fire on its own fixtures must not judge "
            "the repo:"
        )
        for failure in failures:
            print(f"  - {failure}")
        return REFUSE
    print(
        f"SELF-TEST DENOMINATOR: {len(_SELF_TEST_LEGS)} of "
        f"{len(_SELF_TEST_LEGS)} controls behaved; MUST_FIRE in both drift "
        "directions, MUST_NOT_FIRE on the five legitimate-divergence shapes "
        "including BOTH pinned blind spots -- same key different argv, and "
        "shell suites in no denominator -- ABSTAIN on unreadable and "
        "unparsable inputs."
    )
    return CLEAR


_USAGE = "usage: python3 checks/makefile_ci_mirror.py [--self-test] [--makefile PATH] [--ci PATH]"


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return _self_test()
    if "--self-test" in argv:
        print(_USAGE, file=sys.stderr)
        return REFUSE
    makefile = None
    ci = None
    args = list(argv)
    while args:
        flag = args.pop(0)
        if flag == "--makefile" and args:
            makefile = Path(args.pop(0))
        elif flag == "--ci" and args:
            ci = Path(args.pop(0))
        else:
            print(_USAGE, file=sys.stderr)
            return REFUSE
    root = Path(__file__).resolve().parent.parent
    if makefile is None:
        makefile = root / "Makefile"
    if ci is None:
        ci = root / ".github" / "workflows" / "ci.yml"
    try:
        code, report = evaluate(makefile, ci)
    except Exception as exc:  # never traceback-exit 1
        print(
            f"UNMEASURED: gate raised while parsing ({exc!r}); "
            f"'I could not parse it' is not 'they disagree'.",
            file=sys.stderr,
        )
        return UNMEASURED
    print(report)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
