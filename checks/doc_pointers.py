#!/usr/bin/env python3
"""Doc-pointer gate: every pointer in shipped markdown must have a file behind it.

Usage:
  doc_pointers.py [ROOT]      audit every git-tracked *.md under ROOT (default .)
  doc_pointers.py --self-test run the controls in a temporary tree and exit

The class (#281). The README is a contents page: it closes most of its 27 sections
with a pointer to a chapter under `docs/`. Eighteen of those chapters did not exist.
Nothing was red, because a pointer is a DECLARATION and this repository had a gate
over declarations only for build stages (#190/#191, gate_stage_orphans). A dangling
pointer is the documentation instance of that same class -- a declaration with no
artifact behind it -- and it had no gate, so the count could only grow.

Doctrine wiring:
  (1) The denominator is printed on every path. A run that resolved zero pointers is
      UNMEASURED (95), never CLEAR -- `all([])` is True and that is this repository's
      founding defect.
  (2) TWO notations are extracted and each reports its OWN sub-denominator. This is
      #186's lesson stated as machinery: that gate read one of two citation notations
      and twelve pointers sat in no denominator while it printed CLEAR. If a notation
      ever goes inert here, its count goes to zero IN THE OUTPUT, where it is visible.
  (3) Resolution is permissive on purpose -- a target is satisfied if it resolves
      relative to the citing file (how a reader on GitHub follows it) OR relative to
      the repository root (this tree's prevailing convention for `docs/...`). A target
      that satisfies NEITHER is unambiguously dangling, so a RED here is not an
      argument about convention.
  (4) BOTH sides read git's index, not the filesystem. The corpus is tracked *.md; a
      target is satisfied only by a tracked path. This matters more than it sounds:
      a pointer to a file that exists on the author's disk but was never committed
      reads as fine locally and is dangling for every other reader on earth. A gate
      whose corpus is the index and whose resolver is the filesystem would call that
      clean. (#244's lesson: the census and the gate have to agree about what the
      repository IS.)
  (5) --self-test plants each notation's dangling form and asserts THIS extractor and
      THIS resolver produce the RED, then plants the resolvable form and asserts CLEAR.
      A control that does not self-match measures nothing.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

RC_CLEAR = 0
RC_RED = 5
RC_UNMEASURED = 95

# Notation 1: a standard markdown inline link, [text](target).
MD_LINK = re.compile(r"\[[^\]\n]*\]\(([^)\s]+)\)")
# Notation 2: this repository's own end-of-section citation, [-> path].
#
# The target runs to the first delimiter, and the closing bracket is NOT required
# to follow it. That is not laxity, it is the measured shape of the notation: two
# of the README's own pointers carry a prose tail --
#     [-> docs/ARCHITECTURE.md; the full L0-L6 design is
#     [-> docs/EXAMPLES.md - the catalogue that should exist]
# -- and an extractor anchored on `\s*\]` sees neither. That is #186 recurring
# inside the gate written to close #281: a notation variant in no denominator.
ARROW = re.compile(r"\[->\s*([^\]\s;,]+)")

NOTATIONS: dict[str, re.Pattern[str]] = {"md-link": MD_LINK, "arrow": ARROW}

SKIP_PREFIX = ("http://", "https://", "mailto:", "#", "<")

# Code formatting is the use/mention line. A path inside backticks is being SHOWN,
# not followed: `docs/CONTRACT.md` in SELF_AUDIT.md names a document that section
# argues should be written, and README's transcripts quote shell that mentions
# paths. Counting those as pointers manufactures dangles -- three of the twenty in
# this gate's first hand-rolled census were exactly that. The stripped span count
# is printed so this narrowing stays measured rather than silent.
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
INLINE_CODE = re.compile(r"(?P<t>`+)(?:(?!(?P=t))[^\n])*?(?P=t)")


def mask_code(text: str) -> tuple[str, int, int]:
    """Blank out fenced blocks and inline code, preserving every byte offset.

    Returns (masked text, fenced blocks masked, inline spans masked). Offsets are
    preserved by substituting spaces, so reported line numbers stay true.
    """
    out: list[str] = []
    fence_char: str | None = None
    blocks = 0
    for line in text.splitlines(keepends=True):
        m = FENCE.match(line)
        if fence_char is None and m:
            fence_char, blocks = m.group(1)[0], blocks + 1
            out.append(" " * len(line.rstrip("\n")) + line[len(line.rstrip("\n")) :])
            continue
        if fence_char is not None:
            if m and m.group(1)[0] == fence_char:
                fence_char = None
            out.append(" " * len(line.rstrip("\n")) + line[len(line.rstrip("\n")) :])
            continue
        out.append(line)
    joined = "".join(out)
    spans = 0

    def blank(m: re.Match[str]) -> str:
        nonlocal spans
        spans += 1
        return " " * len(m.group(0))

    return INLINE_CODE.sub(blank, joined), blocks, spans


def _is_local_doc(target: str) -> bool:
    """Only local repository paths are in the denominator.

    External URLs are somebody else's uptime, and a bare `#anchor` is a
    within-page jump with no file behind it by construction.
    """
    return bool(target) and not target.startswith(SKIP_PREFIX)


def _ls_files(root: Path, *pathspec: str) -> list[str]:
    """Git-tracked relative paths under root. Empty list if git cannot answer."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", *pathspec],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [p for p in out.decode("utf-8").split("\0") if p]


def tracked_markdown(root: Path) -> list[Path]:
    """The corpus: every git-tracked *.md under root (doctrine 4)."""
    return [root / p for p in _ls_files(root, "*.md")]


def tracked_paths(root: Path) -> set[str]:
    """The resolution universe: every tracked path, plus every directory holding one.

    Directories are synthesised because a pointer may legitimately target one
    (`docs/review/`), and git tracks files only -- a directory is not an entry in
    the index, so without this a real pointer would read as dangling.
    """
    universe: set[str] = set()
    for rel in _ls_files(root):
        universe.add(rel)
        parent = PurePosixPath(rel).parent
        while str(parent) not in (".", "/"):
            universe.add(str(parent))
            parent = parent.parent
    return universe


def extract(text: str) -> list[tuple[str, str, int]]:
    """Return (notation, target, line_number) for every local pointer in text."""
    found: list[tuple[str, str, int]] = []
    for name, pat in NOTATIONS.items():
        for m in pat.finditer(text):
            target = m.group(1).split("#", 1)[0].strip()
            if not _is_local_doc(target):
                continue
            found.append((name, target, text.count("\n", 0, m.start()) + 1))
    return found


def _normalise(rel: str) -> str | None:
    """Collapse `.` and `..` inside a repo-relative path. None if it escapes the root."""
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


def resolves(universe: set[str], citing_rel: str, target: str) -> bool:
    """True if target names a tracked path, read from the citing file OR from root.

    Permissive by design (doctrine 3): the citing-file base is how a reader on
    GitHub follows the link, the root base is this tree's prevailing convention
    for `docs/...` pointers written from inside `docs/`. A target that satisfies
    neither is dangling under every reading, so the RED is not a style argument.
    """
    bases = (str(PurePosixPath(citing_rel).parent), "")
    for base in bases:
        candidate = _normalise(f"{base}/{target}" if base not in (".", "") else target)
        if candidate is not None and candidate in universe:
            return True
    return False


def audit(root: Path) -> int:
    files = tracked_markdown(root)
    if not files:
        print(
            f"DOC-POINTERS UNMEASURED: git listed 0 tracked *.md under {root}"
            " -- no corpus, so no claim (doctrine 1)"
        )
        return RC_UNMEASURED

    universe = tracked_paths(root)
    per_notation: dict[str, int] = dict.fromkeys(NOTATIONS, 0)
    dangling: list[tuple[str, int, str, str]] = []
    total = 0
    unreadable = 0
    blocks = spans = 0

    for f in sorted(files):
        rel = f.relative_to(root).as_posix()
        try:
            raw = f.read_text(encoding="utf-8")
        except OSError as e:
            # Unreadable is RED, not empty -- missing is not zero (#190's doctrine 4).
            print(f"DOC-POINTERS RED-unreadable {rel}: {e}")
            unreadable += 1
            continue
        text, b, s = mask_code(raw)
        blocks, spans = blocks + b, spans + s
        for notation, target, line in extract(text):
            total += 1
            per_notation[notation] += 1
            if not resolves(universe, rel, target):
                dangling.append((rel, line, notation, target))

    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(per_notation.items()))
    print(
        f"DENOMINATOR: {total} local pointer(s) over {len(files)} git-tracked *.md,"
        f" resolved against {len(universe)} tracked path(s) [{breakdown}]"
    )
    print(
        f"NARROWED: {blocks} fenced block(s) and {spans} inline-code span(s) masked"
        " before extraction -- a path in code formatting is shown, not followed"
    )
    if total == 0:
        print(
            "DOC-POINTERS UNMEASURED: the corpus has files but zero local pointers"
            " -- either the tree really cites nothing, or both extractors have gone"
            " inert. Either way this run measured nothing (doctrine 1)."
        )
        return RC_UNMEASURED
    for name, count in sorted(per_notation.items()):
        if count == 0:
            print(
                f"NOTE: the '{name}' notation matched 0 pointers. That is not"
                " an error, but it is the shape a dead extractor makes (doctrine 2)."
            )

    if unreadable or dangling:
        for rel, line, notation, target in dangling:
            print(f"DANGLING {rel}:{line} [{notation}] -> {target}")
        print(
            f"DOC-POINTERS RED: {len(dangling)} dangling of {total} pointer(s),"
            f" {unreadable} unreadable file(s). A pointer with no file behind it is a"
            " declaration with no artifact -- the #190 class, in prose."
        )
        return RC_RED

    print(f"DOC-POINTERS CLEAR: {total}/{total} pointer(s) resolve to a file in the tree")
    return RC_CLEAR


# ---------------------------------------------------------------- self-test


def _git_init(d: Path) -> None:
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "selftest@example.invalid"],
        ["git", "config", "user.name", "selftest"],
    ):
        subprocess.run(cmd, cwd=d, check=True, capture_output=True)


def _stage(d: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=d, check=True, capture_output=True)


# (label, tracked files, files written but NEVER staged, expected rc, why)
Case = tuple[str, dict[str, str], dict[str, str], int, str]

CASES: list[Case] = [
    (
        "MUST_FIRE arrow notation dangles",
        {"README.md": "Section.\n\n[-> docs/GHOST.md]\n"},
        {},
        RC_RED,
        "the arrow extractor must catch its own notation",
    ),
    (
        "MUST_FIRE md-link notation dangles",
        {"README.md": "See [the chapter](docs/GHOST.md) for more.\n"},
        {},
        RC_RED,
        "the md-link extractor must catch its own notation",
    ),
    (
        "MUST_FIRE one of two resolves",
        {
            "README.md": "[-> docs/REAL.md] and [x](docs/GHOST.md)\n",
            "docs/REAL.md": "# Real\n",
        },
        {},
        RC_RED,
        "a partially-good corpus is still RED -- 1 of 2 is not CLEAR",
    ),
    (
        "MUST_FIRE an untracked file does not satisfy a pointer",
        {"README.md": "[-> docs/GHOST.md]\n"},
        # Written to disk, never staged. A filesystem-reading resolver would call
        # this CLEAR; every reader who clones the repo would get a 404. This is
        # the control for doctrine 4, and it fails if the resolver ever regresses
        # to os.path.exists.
        {"docs/GHOST.md": "# On my disk only\n"},
        RC_RED,
        "resolution reads the index, not the author's working tree (doctrine 4)",
    ),
    (
        "MUST_FIRE a target that climbs out of the repository",
        {"README.md": "[-> ../../etc/passwd]\n"},
        {},
        RC_RED,
        "a path escaping the root normalises to None and cannot resolve",
    ),
    (
        "MUST_PASS arrow resolves from root",
        {"README.md": "[-> docs/REAL.md]\n", "docs/REAL.md": "# Real\n"},
        {},
        RC_CLEAR,
        "the resolver must accept the tree's prevailing root-relative form",
    ),
    (
        "MUST_PASS sibling resolves relative to the citing file",
        {
            "README.md": "[-> docs/REAL.md]\n",
            "docs/REAL.md": "See [the other](SIBLING.md).\n",
            "docs/SIBLING.md": "# Sibling\n",
        },
        {},
        RC_CLEAR,
        "a reader on GitHub follows links relative to the file (doctrine 3)",
    ),
    (
        "MUST_PASS a pointer to a directory resolves",
        {"README.md": "[-> docs/review/]\n", "docs/review/D4.md": "# D4\n"},
        {},
        RC_CLEAR,
        "git tracks files only; directories are synthesised or every dir dangles",
    ),
    (
        "MUST_PASS a pointer to a non-markdown tracked file resolves",
        {"README.md": "[the loop](src/loop.py)\n", "src/loop.py": "x = 1\n"},
        {},
        RC_CLEAR,
        "the corpus is *.md but the resolution universe is the whole index",
    ),
    (
        "MUST_FIRE arrow pointer carrying a prose tail",
        # The measured shape of two of the README's own pointers. An extractor
        # anchored on `\s*\]` reads zero here and prints CLEAR -- #186's exact
        # failure, inside the gate written to close #281.
        {"README.md": "[-> docs/GHOST.md; the design lives here\n"},
        {},
        RC_RED,
        "the closing bracket is not part of the notation as it is actually written",
    ),
    (
        "MUST_FIRE arrow pointer with an em-dash gloss",
        {"README.md": "[-> docs/GHOST.md - the catalogue that should exist]\n"},
        {},
        RC_RED,
        "a glossed pointer is still a pointer",
    ),
    (
        "MUST_PASS a backticked path is a mention, not a pointer",
        # `docs/CONTRACT.md` in SELF_AUDIT.md names a document that section argues
        # SHOULD be written. Reading it as a live pointer manufactures a dangle --
        # three of twenty in the first hand-rolled census were this.
        {"README.md": "We should write `docs/GHOST.md` one day.\n"},
        {},
        RC_UNMEASURED,
        "masking inline code leaves zero pointers here, which is UNMEASURED not RED",
    ),
    (
        "MUST_PASS a fenced block does not contribute pointers",
        {
            "README.md": "```\n[-> docs/GHOST.md]\n```\n\n[-> docs/REAL.md]\n",
            "docs/REAL.md": "# Real\n",
        },
        {},
        RC_CLEAR,
        "a transcript quoting a pointer is showing it, not following it",
    ),
    (
        "MUST_FIRE a dangle AFTER masked regions is still found",
        {"README.md": "`x`\n\n```\ncode\n```\n\n[-> docs/GHOST.md]\n"},
        {},
        RC_RED,
        "masking must not consume the rest of the document",
    ),
    (
        "MUST_PASS external URLs and anchors are out of the denominator",
        {
            "README.md": "[up](https://example.invalid/x.md) [here](#section)\n",
            "docs/REAL.md": "[-> docs/REAL.md]\n",
        },
        {},
        RC_CLEAR,
        "an external URL is somebody else's uptime, not a missing file",
    ),
    (
        "UNMEASURED empty corpus",
        {},
        {},
        RC_UNMEASURED,
        "zero tracked markdown is no claim at all, not a pass (doctrine 1)",
    ),
    (
        "UNMEASURED corpus with files but no pointers",
        {"README.md": "Prose with no links at all.\n"},
        {},
        RC_UNMEASURED,
        "both extractors silent is indistinguishable from both extractors dead",
    ),
]


def _write(d: Path, files: dict[str, str]) -> None:
    for rel, body in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")


# Direct assertions on the extractor, where audit()'s exit code is too coarse to
# distinguish "found the right thing" from "found a different thing that is also RED".
UNIT_CONTROLS: list[tuple[str, str, list[tuple[str, str, int]]]] = [
    (
        "masking preserves byte offsets, so reported lines are true",
        "`x`\n\n```\ncode\n```\n\n[-> docs/GHOST.md]\n",
        [("arrow", "docs/GHOST.md", 7)],
    ),
    (
        "a prose tail is trimmed off the target, not swallowed into it",
        "[-> docs/A.md; the design lives here\n[-> docs/B.md - a gloss]\n",
        [("arrow", "docs/A.md", 1), ("arrow", "docs/B.md", 2)],
    ),
    (
        "both notations fire on one line and neither shadows the other",
        "[-> docs/A.md] and [text](docs/B.md)\n",
        [("md-link", "docs/B.md", 1), ("arrow", "docs/A.md", 1)],
    ),
    (
        "a fenced pointer and an inline-code path contribute nothing",
        "```\n[-> docs/A.md]\n```\nsee `docs/B.md` too\n",
        [],
    ),
]


def _unit_controls() -> int:
    """Assert WHAT the extractor found, not merely that something was found."""
    passed = 0
    for label, text, want in UNIT_CONTROLS:
        masked, _, _ = mask_code(text)
        got = sorted(extract(masked))
        ok = got == sorted(want)
        passed += ok
        print(f"UNIT-CONTROL {'PASS' if ok else 'FAIL'} {label}: got={got} want={sorted(want)}")
    print(f"UNIT-CONTROL DENOMINATOR: {passed} of {len(UNIT_CONTROLS)} extractor controls behaved")
    return passed


def self_test() -> int:
    """Plant the exact defect, run the exact machinery (doctrine 5)."""
    unit_ok = _unit_controls()
    passed = 0
    for label, tracked, untracked, want, why in CASES:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _git_init(d)
            _write(d, tracked)
            _stage(d)
            _write(d, untracked)  # after staging, so git never sees them
            got = audit(d)
        ok = got == want
        passed += ok
        print(f"SELF-TEST {'PASS' if ok else 'FAIL'} {label}: rc={got} want={want} -- {why}")

    fire = sum(1 for c in CASES if c[3] == RC_RED)
    clear = sum(1 for c in CASES if c[3] == RC_CLEAR)
    unmeasured = sum(1 for c in CASES if c[3] == RC_UNMEASURED)
    # The suite's canonical last line. launchers/test_checks_gates.sh parses it with
    #   sed -n 's/^self-test denominator: \([0-9]*\) of \([0-9]*\) controls.*/\1/p'
    # on `tail -n 1`, holds have==want, and holds `have` to a measured FLOOR -- so a
    # control set that silently SHRINKS is red even though every surviving control
    # still passes. rc=0 alone is not accepted, by that suite's explicit convention.
    # Both control families are summed into the one denominator on purpose: the
    # extractor controls are the ones guarding against #186 recurring here, and a
    # floor that only counted the end-to-end cases would let them be deleted.
    have = passed + unit_ok
    want = len(CASES) + len(UNIT_CONTROLS)
    print(
        f"self-test denominator: {have} of {want} controls"
        f" ({fire} MUST_FIRE, {clear} MUST_PASS, {unmeasured} MUST_BE_UNMEASURED,"
        f" {len(UNIT_CONTROLS)} extractor)"
    )
    return RC_CLEAR if have == want else RC_RED


def main(argv: Sequence[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    root = Path(argv[0]) if argv else Path.cwd()
    return audit(root.resolve())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
