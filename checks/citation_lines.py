#!/usr/bin/env python3
"""Citation-lines gate: every in-repo `path.ext:N` citation names a line that EXISTS.

Usage:
  citation_lines.py [ROOT]      audit every git-tracked file under ROOT (default .)
  citation_lines.py --self-test run the controls in a temporary tree and exit

The claim, stated narrowly on purpose. This repository cites its own source in
prose and comments as `path.ext:N` and `path.ext:N-M`. Nothing put those tokens
in a denominator, so a citation could rot silently when the file it names was
edited -- two were found stale by hand, and the class had no instrument. This
gate proves exactly one thing: every in-repo citation names a line that EXISTS.
It does NOT prove the line says what the surrounding prose claims it says. That
is a stronger claim and this gate does not make it: a citation pointing at the
wrong line of the right file passes here. A gate that implies the broader claim
while measuring only the narrower one is how a false GREEN gets manufactured.

Doctrine wiring:
  (1) The denominator is every citation-shaped token in every git-tracked file
      that decodes as UTF-8, and it is printed on every path. The file list
      comes from `git ls-files`, never a filesystem walk: a walk measures the
      machine (stale build trees, untracked scratch), not the repository. The
      token shape is a path ending in one of .py .sh .md .toml .yml .yaml .json
      .cfg .txt .ini, a colon, then N or N-M; the path must not be preceded by
      a path character and the number must not be followed by an identifier
      character, so `a.py:12` inside `x/a.py:1234abc` matches no truncated
      form. A zero-height denominator is a broken scanner, not a clean tree:
      REFUSE 96, never CLEAR. `all([])` is True and that is this repository's
      founding defect.
  (2) Resolution is three steps, in this order and only this order: the path
      exactly as written, if it is a tracked path; the path joined to the
      citing file's own directory, with `.` and `..` normalised lexically
      (never resolve(), which touches the filesystem and follows symlinks);
      and a UNIQUE suffix match among tracked files, accepted only when
      exactly one tracked path ends with `/` + the cited path (or equals it).
      More than one match is ambiguous and is NOT resolved: picking one would
      be a guess, and a guess inside a resolver is how a gate invents a
      verdict.
  (3) Three states. RESOLVED and in range (1 <= N <= M <= the target's line
      count) is a pass. RESOLVED and out of range -- or malformed, N > M -- is
      the RED, reported with the citing file and line, the token, the resolved
      target, and the target's true line count. UNRESOLVED is UNMEASURED,
      never RED: this repository cannot see files outside itself, so a
      citation of an external path is a claim the gate has no standing to
      judge.
  (4) There is no allowlist, and the third state is why. Two constructs in the
      tree have exactly the shape of a citation and are not citations: the
      `name:rc` arguments passed to `make_stub_campaign` in
      validation_campaigns/h100_validation/run_standing_gates.sh (eleven of
      them, spelled like `gate_alpha.py:5`), and the `site` string literals
      compared in validation_campaigns/h100_validation/fs_required_knobs.py
      (spelled like "launcher.sh:2"). The gate cannot tell them from a
      citation BY SHAPE and does not try. It does not need to: their paths
      name no tracked file, so step (2) puts them in UNMEASURED -- the same
      rule that handles a genuine out-of-repo citation. An allowlist would be
      the gate excepting its own residue, which is a worse instrument than
      one that reports honestly what it cannot see.
  (5) The bare `:NNN` and `:NNN-MMM` form -- a line citation whose file is
      implied by context rather than written (heaviest in tools/mutations.json
      prose and in the campaign's patch stages) -- is counted on its own
      DECLARED UNMEASURED axis: the count, the number of containing files, and
      the top few files by count. The gate does not infer which file each one
      means. Inferring the subject of a citation from proximity is the defect
      that produced hundreds of false verdicts in this repository's countables
      gate; the honest report is the number of claims the instrument cannot
      see. The bare form matches only where the colon is not preceded by a
      path, identifier, or colon character, so `a.py:12`, `http://x` and `::`
      do not feed it.
  (6) --self-test plants each failure mode in a temporary tree and asserts
      THIS scanner and THIS resolver produce the expected state, passing the
      file list in explicitly rather than shelling out to git. It then re-runs
      the must-fire controls with the range comparison neutered to always
      report "in range" and asserts they STOP firing. A harness that cannot
      fire proves nothing, and a gate whose harness cannot fire cannot be
      trusted: that run exits 96.

Exit-code namespace (exactly this and no other; this gate never exits 1):
    0  CLEAR      -- every resolved citation is in range
    5  RED        -- at least one resolved citation is out of range or malformed
    95 UNMEASURED -- a nonzero denominator in which nothing could be resolved
    96 REFUSE     -- git ls-files unavailable or empty, a zero-height
                     denominator, or a self-test harness that cannot fire

Unresolved citations and bare forms never change the exit code. They are
reported inside a passing run, and the verdict line prints their counts next
to the verified count, so the number of verified citations is never mistaken
for the number of citations.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final, NoReturn

EXIT_CLEAR: Final[int] = 0
EXIT_RED: Final[int] = 5
EXIT_UNMEASURED: Final[int] = 95
EXIT_REFUSE: Final[int] = 96

# A citation: a path in a known source suffix, a colon, then N or N-M. The
# lookbehind refuses a match that starts mid-path (`a.py` inside `x/a.py:12`);
# the lookahead refuses a number truncated out of a longer identifier (`1234`
# out of `1234abc`) and a range truncated out of a longer run (`12` out of
# `12-15x`). What does not match whole does not match at all.
CITATION: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w./\\-])"
    r"(?P<path>(?:[\w.-]+/)*[\w.-]+\.(?:py|sh|md|toml|ya?ml|json|cfg|txt|ini))"
    r":(?P<lo>[0-9]+)(?:-(?P<hi>[0-9]+))?"
    r"(?![\w-])"
)

# The implied-file form: a bare `:NNN` or `:NNN-MMM` whose file is carried by
# context. The lookbehind keeps `a.py:12` (identifier before the colon),
# `http://x` (identifier), and `::` (colon) out of this axis.
BARE: Final[re.Pattern[str]] = re.compile(r"(?<![\w./\\:-]):[0-9]+(?:-[0-9]+)?(?![\w-])")

# Distinct unresolved paths are capped in the report, and the cap is STATED
# when it bites: a silent cap reads as complete coverage.
UNRESOLVED_CAP: Final[int] = 20

_WORD: Final[dict[int, str]] = {
    EXIT_CLEAR: "CLEAR",
    EXIT_RED: "RED",
    EXIT_UNMEASURED: "UNMEASURED",
    EXIT_REFUSE: "REFUSE",
}

# The one comparison the gate makes, parameterised so the self-test can neuter
# it WITHOUT editing source text at runtime: the harness control injects a
# comparison that always reports "in range" and asserts the must-fire controls
# go silent.
RangeCheck = Callable[[int, int, int], bool]


def real_in_range(n: int, m: int, lines: int) -> bool:
    """The live comparison: 1 <= N <= M <= the target's line count."""
    return 1 <= n <= m <= lines


def always_in_range(_n: int, _m: int, _lines: int) -> bool:
    """The neutered comparison for the harness control: everything is in range."""
    return True


@dataclass(frozen=True)
class Citation:
    """One citation-shaped token found in a tracked file."""

    citing: str  # citing file, repo-relative
    line: int  # 1-based line number in the citing file
    token: str  # the matched text, e.g. docs/A.md:12-15
    cited: str  # the path as written
    lo: int
    hi: int


@dataclass(frozen=True)
class RedRow:
    """One measured finding: a resolved citation whose line is not there."""

    kind: str  # "out-of-range" | "malformed" | "unreadable"
    citing: str
    line: int
    token: str
    target: str  # the tracked path the citation resolved to
    target_lines: int  # the target's true line count (-1 if unreadable)


@dataclass
class Report:
    files_scanned: int = 0
    files_skipped: int = 0  # tracked but not UTF-8, or unreadable
    denominator: int = 0  # citation-shaped tokens seen
    verified: int = 0  # resolved and in range
    red: list[RedRow] = field(default_factory=list)
    unresolved: list[Citation] = field(default_factory=list)
    bare_count: int = 0  # implied-file tokens, a separate axis
    bare_by_file: dict[str, int] = field(default_factory=dict)


def _normalise(rel: str) -> str | None:
    """Collapse `.` and `..` inside a repo-relative path. None if it escapes the root.

    Lexical only -- never Path.resolve(), which touches the filesystem and
    follows symlinks (doctrine 2).
    """
    parts: list[str] = []
    for part in PurePosixPath(rel).parts:
        if part in (".", ""):
            continue
        if part == "..":
            if not parts:
                return None  # climbs out of the repository
            parts.pop()
            continue
        parts.append(part)
    return "/".join(parts) if parts else None


def resolve(tracked: set[str], citing_rel: str, cited: str) -> str | None:
    """The tracked path a citation names, or None. Three steps, only this order.

    1. the path exactly as written; 2. joined to the citing file's own
    directory and normalised lexically; 3. a UNIQUE suffix match -- exactly
    one tracked path ends with `/` + the cited path. An ambiguous suffix match
    returns None: picking one would be a guess, and a guess inside a resolver
    is how a gate invents a verdict (doctrine 2).
    """
    if cited in tracked:
        return cited
    base = str(PurePosixPath(citing_rel).parent)
    joined = _normalise(cited if base in (".", "") else f"{base}/{cited}")
    if joined is not None and joined in tracked:
        return joined
    matches = [p for p in tracked if p == cited or p.endswith("/" + cited)]
    if len(matches) == 1:
        return matches[0]
    return None


def _line_count(root: Path, rel: str, cache: dict[str, int]) -> int | None:
    """A target's true line count, from bytes so a non-UTF-8 target still counts."""
    if rel in cache:
        return cache[rel]
    try:
        data = (root / rel).read_bytes()
    except OSError:
        return None
    count = len(data.splitlines())
    cache[rel] = count
    return count


def scan(root: Path, files: Sequence[str], in_range: RangeCheck = real_in_range) -> Report:
    """Scan an explicit (root, files) corpus -- the self-test passes fixtures in.

    `files` is the tracked-path list; nothing here shells out to git, so the
    same function reads the real repository (list from `git ls-files`) and the
    temporary control trees (doctrine 6).
    """
    tracked = set(files)
    cache: dict[str, int] = {}
    report = Report()
    for rel in sorted(files):
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            report.files_skipped += 1
            continue
        report.files_scanned += 1
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in CITATION.finditer(line):
                report.denominator += 1
                cited = match.group("path")
                lo = int(match.group("lo"))
                hi_s = match.group("hi")
                hi = int(hi_s) if hi_s is not None else lo
                token = match.group(0)
                target = resolve(tracked, rel, cited)
                if target is None:
                    report.unresolved.append(Citation(rel, lineno, token, cited, lo, hi))
                    continue
                lines = _line_count(root, target, cache)
                if lines is None:
                    report.red.append(RedRow("unreadable", rel, lineno, token, target, -1))
                    continue
                if in_range(lo, hi, lines):
                    report.verified += 1
                    continue
                kind = "malformed" if lo > hi else "out-of-range"
                report.red.append(RedRow(kind, rel, lineno, token, target, lines))
            for _ in BARE.finditer(line):
                report.bare_count += 1
                report.bare_by_file[rel] = report.bare_by_file.get(rel, 0) + 1
    return report


def gate_exit_code(report: Report) -> int:
    if report.denominator == 0:
        return EXIT_REFUSE  # a zero-height denominator is a broken scanner
    if report.red:
        return EXIT_RED
    if report.verified == 0:
        return EXIT_UNMEASURED  # tokens existed and none resolved: nothing measured
    return EXIT_CLEAR


def format_red(row: RedRow) -> str:
    """One RED row: citing file AND line, the token, the target, the true count."""
    where = f"{row.citing}:{row.line}"
    if row.kind == "unreadable":
        return f"RED {where} [{row.token}] -> {row.target}: tracked but unreadable"
    if row.kind == "malformed":
        return (
            f"RED {where} [{row.token}] -> {row.target}: malformed range, N > M "
            f"(target has {row.target_lines} line(s))"
        )
    return (
        f"RED {where} [{row.token}] -> {row.target}: line out of range "
        f"(target has {row.target_lines} line(s))"
    )


def print_provenance() -> None:
    # A verdict is attributable only to the interpreter that produced it.
    print(f"interpreter: {sys.executable}")
    print(f"version_info: {sys.version_info}")


def print_report(report: Report) -> None:
    print(
        f"DENOMINATOR: {report.denominator} citation token(s) over "
        f"{report.files_scanned} git-tracked UTF-8 file(s) "
        f"({report.files_skipped} skipped: undecodable or unreadable)"
    )
    print(
        f"counts: {report.verified} verified in range / {len(report.red)} RED / "
        f"{len(report.unresolved)} unresolved / {report.bare_count} bare implied-file"
    )
    print(
        f"content axis (DECLARED UNMEASURED): {report.verified} resolved token(s) "
        "were checked for EXISTENCE only -- whether the cited line says what the "
        "surrounding prose claims is a stronger question this gate does not ask, "
        "so a citation aimed at the wrong line of the right file passes (#309)"
    )
    print(
        f"bare implied-file axis (DECLARED UNMEASURED): {report.bare_count} token(s) "
        f"in {len(report.bare_by_file)} file(s) -- which file each one means is "
        "declared, not inferred (doctrine 5)"
    )
    top = sorted(report.bare_by_file.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    if top:
        print("  top bare-form files: " + ", ".join(f"{name}={n}" for name, n in top))
    for row in report.red:
        print(format_red(row))
    distinct = sorted({c.cited for c in report.unresolved})
    if distinct:
        print(
            f"unresolved: {len(report.unresolved)} token(s), {len(distinct)} distinct "
            "path(s) -- UNMEASURED, never RED: this repository cannot see files "
            "outside itself, so an external citation is a claim the gate has no "
            "standing to judge (doctrine 3)"
        )
    shown = distinct[:UNRESOLVED_CAP]
    for path in shown:
        n = sum(1 for c in report.unresolved if c.cited == path)
        print(f"  UNRESOLVED {path} ({n} token(s))")
    if len(distinct) > len(shown):
        print(
            f"  ... and {len(distinct) - len(shown)} more distinct path(s) not shown "
            f"(cap {UNRESOLVED_CAP}) -- this list is truncated, not complete"
        )


def git_ls_files(root: Path) -> list[str]:
    """Git-tracked relative paths under root. Empty list if git cannot answer."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [p for p in out.decode("utf-8", errors="replace").split("\0") if p]


def audit(root: Path) -> int:
    files = git_ls_files(root)
    if not files:
        print(
            f"CITATION-LINES REFUSE: git ls-files gave nothing under {root} -- git "
            "unavailable, not a repository, or zero tracked files. A gate that "
            "cannot enumerate its corpus cannot trust itself."
        )
        return EXIT_REFUSE
    report = scan(root, files)
    print_report(report)
    rc = gate_exit_code(report)
    if rc == EXIT_REFUSE:
        print(
            "CITATION-LINES REFUSE: zero citation tokens in a nonzero corpus -- a "
            "zero-height denominator means the scanner is broken, not the "
            "repository clean (doctrine 1)."
        )
    elif rc == EXIT_UNMEASURED:
        print(
            "CITATION-LINES UNMEASURED: citation tokens were found but none "
            "resolved to a tracked file, so this run verified nothing."
        )
    print_provenance()
    print(
        f"CITATION-LINES {_WORD[rc]}: {report.verified} verified in range / "
        f"{len(report.red)} RED / {len(report.unresolved)} unresolved / "
        f"{report.bare_count} bare implied-file, over a denominator of "
        f"{report.denominator} token(s) -- verified is not the citation count"
    )
    return rc


# ---------------------------------------------------------------- self-test


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")


ControlFn = Callable[[RangeCheck], tuple[bool, str]]


def self_test() -> int:
    """Plant the exact defect, run the exact machinery (doctrine 6)."""
    print_provenance()
    print("self-test: controls run in a temporary directory, the file list passed in")

    def c_past_end(in_range: RangeCheck) -> tuple[bool, str]:
        # A citation past the end of a real target must RED, naming the line.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(
                root,
                {
                    "docs/a.md": "the loop is at target.py:5\n",
                    "target.py": "x = 1\ny = 2\n",
                },
            )
            rep = scan(root, ["docs/a.md", "target.py"], in_range)
        if not rep.red:
            return False, f"no RED row (verified={rep.verified})"
        row = rep.red[0]
        msg = format_red(row)
        ok = (
            len(rep.red) == 1
            and row.kind == "out-of-range"
            and row.target == "target.py"
            and row.target_lines == 2
            and "docs/a.md:1" in msg
        )
        return ok, msg

    def c_last_line(in_range: RangeCheck) -> tuple[bool, str]:
        # A citation exactly at the last line must PASS: the off-by-one
        # boundary in the safe direction.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(
                root,
                {
                    "docs/a.md": "the loop is at target.py:2\n",
                    "target.py": "x = 1\ny = 2\n",
                },
            )
            rep = scan(root, ["docs/a.md", "target.py"], in_range)
        ok = rep.verified == 1 and not rep.red and gate_exit_code(rep) == EXIT_CLEAR
        return ok, f"verified={rep.verified} red={len(rep.red)}"

    def c_range_past_end(in_range: RangeCheck) -> tuple[bool, str]:
        # A range N-M whose M is past the end must RED.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(
                root,
                {
                    "docs/a.md": "see target.py:1-5\n",
                    "target.py": "x = 1\ny = 2\n",
                },
            )
            rep = scan(root, ["docs/a.md", "target.py"], in_range)
        ok = (
            len(rep.red) == 1 and rep.red[0].kind == "out-of-range" and rep.red[0].target_lines == 2
        )
        return ok, f"red={len(rep.red)} kind={rep.red[0].kind if rep.red else '-'}"

    def c_malformed(in_range: RangeCheck) -> tuple[bool, str]:
        # A range whose N > M must RED as malformed.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(
                root,
                {
                    "docs/a.md": "see target.py:3-2\n",
                    "target.py": "x = 1\ny = 2\nz = 3\n",
                },
            )
            rep = scan(root, ["docs/a.md", "target.py"], in_range)
        ok = len(rep.red) == 1 and rep.red[0].kind == "malformed"
        return ok, f"red={len(rep.red)} kind={rep.red[0].kind if rep.red else '-'}"

    def c_ambiguous_basename(in_range: RangeCheck) -> tuple[bool, str]:
        # A path matching TWO tracked files by basename must be UNRESOLVED --
        # never resolved to either, never RED. Picking one would be a guess.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(
                root,
                {
                    "docs/a.md": "see shared.py:1 and ok.py:1\n",
                    "x/shared.py": "a = 1\n",
                    "y/shared.py": "b = 2\n",
                    "ok.py": "c = 3\n",
                },
            )
            files = ["docs/a.md", "x/shared.py", "y/shared.py", "ok.py"]
            rep = scan(root, files, in_range)
        paths = [c.cited for c in rep.unresolved]
        ok = (
            paths == ["shared.py"]
            and not rep.red
            and rep.verified == 1
            and gate_exit_code(rep) == EXIT_CLEAR
        )
        return ok, f"unresolved={paths} red={len(rep.red)} verified={rep.verified}"

    def c_external(in_range: RangeCheck) -> tuple[bool, str]:
        # A path naming no tracked file must be UNRESOLVED, and the run must
        # still exit 0: an out-of-repo citation is a claim the gate has no
        # standing to judge, not a finding.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(
                root,
                {
                    "docs/a.md": "mirrors upstream/lib/else.py:10; see ok.py:1\n",
                    "ok.py": "c = 3\n",
                },
            )
            rep = scan(root, ["docs/a.md", "ok.py"], in_range)
        paths = [c.cited for c in rep.unresolved]
        ok = (
            paths == ["upstream/lib/else.py"]
            and not rep.red
            and rep.verified == 1
            and gate_exit_code(rep) == EXIT_CLEAR
        )
        return ok, f"unresolved={paths} exit={gate_exit_code(rep)}"

    def c_bare_form(in_range: RangeCheck) -> tuple[bool, str]:
        # A bare `:123` must land on the implied-file axis and must NOT be
        # resolved; `http://x`, `::` and real citations must not feed the axis.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(
                root,
                {
                    "docs/a.md": (
                        "details at :123; not bare: http://x.invalid/y, a :: b, ok.py:1\n"
                    ),
                    "ok.py": "c = 3\n",
                },
            )
            rep = scan(root, ["docs/a.md", "ok.py"], in_range)
        ok = (
            rep.bare_count == 1
            and rep.bare_by_file == {"docs/a.md": 1}
            and rep.denominator == 1
            and rep.verified == 1
            and not rep.unresolved
        )
        return ok, f"bare={rep.bare_count} denominator={rep.denominator}"

    controls: list[tuple[str, str, ControlFn]] = [
        ("citation past the end of a real target", "MUST_FIRE", c_past_end),
        ("citation exactly at the last line", "MUST_PASS", c_last_line),
        ("range whose M is past the end", "MUST_FIRE", c_range_past_end),
        ("range whose N > M is malformed", "MUST_FIRE", c_malformed),
        ("two-way basename match unresolved, never guessed", "MUST_PASS", c_ambiguous_basename),
        ("out-of-repo path unresolved, run still clears", "MUST_PASS", c_external),
        ("bare :123 declared unmeasured, never resolved", "MUST_PASS", c_bare_form),
    ]

    behaved = 0
    misbehaved: list[str] = []
    for label, kind, fn in controls:
        try:
            ok, detail = fn(real_in_range)
        except (OSError, KeyError, ValueError, AssertionError) as exc:
            ok, detail = False, f"raised {exc!r}"
        print(f"  [{kind}] {label}: {'ok' if ok else 'MISBEHAVED'} ({detail})")
        if ok:
            behaved += 1
        else:
            misbehaved.append(label)

    # The positive control on the control harness itself: with the range
    # comparison neutered so it always reports "in range", every MUST_FIRE
    # control must now FAIL. If one still fires, the harness cannot fire, the
    # gate cannot be trusted, and neither can the build. The comparison is
    # parameterised -- no source text is edited at runtime.
    still_firing: list[str] = []
    for label, kind, fn in controls:
        if kind != "MUST_FIRE":
            continue
        ok, _ = fn(always_in_range)
        if ok:
            still_firing.append(label)
    harness_ok = not still_firing
    detail = "every MUST_FIRE control goes silent under a neutered comparison"
    if still_firing:
        detail = "still firing under a neutered comparison: " + "; ".join(still_firing)
    print(
        f"  [HARNESS] neutered comparison silences every MUST_FIRE: "
        f"{'ok' if harness_ok else 'MISBEHAVED'} ({detail})"
    )

    n_fire = sum(1 for _, k, _ in controls if k == "MUST_FIRE")
    n_pass = sum(1 for _, k, _ in controls if k == "MUST_PASS")
    have = behaved + (1 if harness_ok else 0)
    want = len(controls) + 1
    print(
        f"self-test denominator: {have} of {want} controls"
        f" ({n_fire} MUST_FIRE, {n_pass} MUST_PASS, 1 harness)"
    )
    if not harness_ok:
        print(
            "SELF-TEST REFUSE: the control harness cannot fire -- with the range "
            "comparison neutered it still reports RED, so a RED from this gate "
            "proves nothing. The gate cannot be trusted and neither can the build."
        )
        return EXIT_REFUSE
    if misbehaved:
        print("MISBEHAVED controls: " + "; ".join(misbehaved))
        return EXIT_RED
    return EXIT_CLEAR


class RefusingParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:  # argparse would exit 2; not in the namespace
        print(f"REFUSE (bad invocation): {message}", file=sys.stderr)
        raise SystemExit(EXIT_REFUSE)


def build_parser() -> argparse.ArgumentParser:
    p = RefusingParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("root", nargs="?", default=".", help="repository root (default: .)")
    p.add_argument(
        "--self-test",
        action="store_true",
        help="run the controls in a temporary directory and exit",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    root = Path(args.root)
    if not root.is_dir():
        print(f"REFUSE: not a directory: {root}", file=sys.stderr)
        return EXIT_REFUSE
    return audit(root)


if __name__ == "__main__":
    sys.exit(main())
