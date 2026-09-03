#!/usr/bin/env python3
"""Countables drift gate: shipped markdown vs a measured census JSON.

WHY this gate is shaped the way it is
-------------------------------------
The first version of this gate scanned every digit group in every markdown
file and INFERRED, from proximity, which census key each number referred to.
Run over the real repository it returned

    VERDICT RED: 12 agreed / 865 drifted / 2158 unclassified
                 over 7259 digit groups scanned in 45 documents

and every one of the 865 was false: "2 dashes" was read as a shim-name
count, "## 3." as the DECISIONS.md line count, "74%" coverage as a gate
file count. It had also walked .venv. Its self-test passed because no
control exercised the association axis -- which key does this number claim?
An uncontrolled axis is an unmeasured axis wearing a verdict.

WHICH COUNTABLE A BARE NUMBER REFERS TO IS NOT DECIDABLE. This gate never
infers it. A number is in scope ONLY when a context-anchored pattern --
anchored on the surrounding WORDS -- claims it. The rules, quoted from the
corrector fix_countables.py and obeyed throughout:

1. EVERY pattern is context-anchored. A bare \\b982\\b match would flag
   "the 982-line DECISIONS.md" (which is CORRECT) as a test count. Numbers
   are only touched where the surrounding words identify what they count.
2. Historical masking is CLAUSE-level, never line-level. "...drafted
   against a census of 13,667 lines ... The package NOW measures 16,929
   lines across 19 files..." is ONE line carrying a dead claim and a live
   claim. Masking the whole line would shield a live stale number behind a
   historical marker.
3. A number and the file count beside it are rewritten TOGETHER or not at
   all -- one measurement stated in two tokens; updating the LOC and leaving
   "across 19 files" would state a pairing that never existed in any census.

There is NO "unclassified" category. A number no pattern claims is out of
scope and is never printed. The 2158-line noise column was the defect.

Exit-code namespace (exactly this and no other; this gate never exits 1):
    0  CLEAR      -- every matched site agrees, over a nonzero denominator
    5  RED        -- at least one anchored site drifted
    95 UNMEASURED -- nothing was measured (no files, or no pattern matched)
    96 REFUSE     -- bad invocation, unreadable input, unreadable census
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from re import Pattern
from typing import (
    Final,
    NoReturn,
)

EXIT_CLEAR: Final[int] = 0
EXIT_RED: Final[int] = 5
EXIT_UNMEASURED: Final[int] = 95
EXIT_REFUSE: Final[int] = 96

# Integer tokens, with or without thousands separators: 18,599 and 18599 are
# the same number stated twice.
NUMBER: Final[str] = r"[0-9][0-9,]*"

# The first gate walked .venv/lib/python3.14/site-packages and measured
# somebody else's repository. These path parts can never be evidence.
EXCLUDED_PARTS: Final[tuple[str, ...]] = (
    ".venv",
    "site-packages",
    "node_modules",
    ".git",
    "htmlcov",
    "build",
)


@dataclass(frozen=True)
class PatternSpec:
    """ONE context-anchored pattern.

    bindings pairs each census key with the named regex group carrying its
    number. A clause stating a LOC and a file count together binds BOTH here
    (rule 4): one regex, one match, one site -- they can never be measured
    or rewritten apart.
    """

    label: str
    regex: Pattern[str]
    bindings: tuple[tuple[str, str], ...]  # (census_key, group_name)


# Derived ONLY from the shipped clause evidence; each pattern cites the real
# clause that justifies its anchor. Clauses that merely collide on a value
# (94.2% coverage, "#181", "8xH100", "24 stages") are NOT countables of
# these keys and get no pattern -- that is the lesson of the 865.
PATTERNS: Final[tuple[PatternSpec, ...]] = (
    PatternSpec(
        label="N-line DECISIONS.md (decisions_loc)",
        # evidence: docs/review/D3_problems_weaknesses.md
        #   "**README routes every newcomer into the 982-line `DECISIONS.md`**"
        regex=re.compile(r"(?P<num>" + NUMBER + r")-line\s+`?DECISIONS\.md`?"),
        bindings=(("decisions_loc", "num"),),
    ),
    PatternSpec(
        label="h100_validation/ (N .py LOC, M files) pair (h100_loc+h100_files)",
        # evidence: docs/review/D4_feature_evaluation.md
        #   "| H100 validation harness | `h100_validation/` (31,313 .py LOC, 63 files) |"
        regex=re.compile(
            r"`?h100_validation/`?\s*\((?P<loc>" + NUMBER + r") \.py LOC,"
            r"\s*(?P<files>" + NUMBER + r") files\)"
        ),
        bindings=(("h100_loc", "loc"), ("h100_files", "files")),
    ),
    PatternSpec(
        label="h100_validation/ contributes N Python LOC (h100_loc)",
        # evidence: docs/review/D2_current_architecture.md
        #   "| Hardware validation plane | `h100_validation/` contributes
        #    31,313 Python LOC and 4,986 shell LOC |"
        regex=re.compile(r"`?h100_validation/`? contributes (?P<num>" + NUMBER + r") Python LOC"),
        bindings=(("h100_loc", "num"),),
    ),
    PatternSpec(
        label="launchers/ contains N shell LOC plus M Python LOC pair "
        "(launch_sh_loc+launch_py_loc)",
        # evidence: docs/review/D2_current_architecture.md
        #   "`launchers/` contains 9,274 shell LOC plus 1,615 Python LOC"
        regex=re.compile(
            r"`?launchers/`? contains (?P<sh>" + NUMBER + r") shell LOC plus"
            r" (?P<py>" + NUMBER + r") Python LOC"
        ),
        bindings=(("launch_sh_loc", "sh"), ("launch_py_loc", "py")),
    ),
    PatternSpec(
        label="launchers/*.sh (N LOC, ...) (launch_sh_loc)",
        # evidence: docs/review/D4_feature_evaluation.md
        #   "| Bash launchers | `launchers/*.sh` (9,274 LOC, 5 files); ... |"
        regex=re.compile(r"`?launchers/\*\.sh`?\s*\((?P<num>" + NUMBER + r") LOC"),
        bindings=(("launch_sh_loc", "num"),),
    ),
    PatternSpec(
        label="N shell LOC in launchers/ (launch_sh_loc)",
        # evidence: docs/review/D2_current_architecture.md
        #   "| Launch orchestration | 9,274 shell LOC in `launchers/`; ... |"
        regex=re.compile(r"(?P<num>" + NUMBER + r") shell LOC in `?launchers/`?"),
        bindings=(("launch_sh_loc", "num"),),
    ),
    # --- #240: the mermaid inventory and its prose twin -------------------
    #
    # These eight anchors exist because of a number that was never right, not
    # one that rotted. D2 stated `tools/` at 11,226 LOC; at a38781b, the commit
    # that WROTE the clause, tools/ was 8,838, and at HEAD it is 8,695 -- the
    # figure matches no commit in the history. The clause's own file count (8)
    # was correct that day, so the denominator really was tools/ and the LOC
    # was simply wrong from birth.
    #
    # It survived a full drift-gate rollout because tools_loc carried no
    # PatternSpec: the gate reported it among the 30 UNMEASURED keys and read
    # CLEAR, and UNMEASURED next to CLEAR reads as covered. A drift gate
    # catches numbers that ROT. Nothing catches a number that was wrong the day
    # it was written except putting it in a denominator, which is what these do.
    #
    # Mermaid node labels are anchored on their node text (`tools/<br/>`, not a
    # bare `N Python files`) for the #233 reason: an unanchored shape would
    # match every node in the diagram and blame whichever key was checked first.
    PatternSpec(
        label="tools/ contains N Python LOC (tools_loc)",
        # evidence: docs/review/D2_current_architecture.md
        #   "while `tools/` contains 8,695 Python LOC."
        regex=re.compile(r"`?tools/`? contains (?P<num>" + NUMBER + r") Python LOC"),
        bindings=(("tools_loc", "num"),),
    ),
    PatternSpec(
        label="mermaid tools/ node pair (tools_files+tools_loc)",
        # evidence: docs/review/D2_current_architecture.md
        #   'TOOLS["tools/<br/>9 Python files / 8,695 LOC"]'
        regex=re.compile(
            r"tools/<br/>(?P<files>" + NUMBER + r") Python files / (?P<loc>" + NUMBER + r") LOC"
        ),
        bindings=(("tools_files", "files"), ("tools_loc", "loc")),
    ),
    PatternSpec(
        label="mermaid tests/ node pair (tests_files+tests_loc)",
        # evidence: docs/review/D2_current_architecture.md
        #   'TESTS["tests/<br/>54 Python files / 28,288 LOC"]'
        regex=re.compile(
            r"tests/<br/>(?P<files>" + NUMBER + r") Python files / (?P<loc>" + NUMBER + r") LOC"
        ),
        bindings=(("tests_files", "files"), ("tests_loc", "loc")),
    ),
    PatternSpec(
        label="mermaid src/-as-importer node pair (src_files+src_loc)",
        # evidence: docs/review/D2_current_architecture.md
        #   'SRC["src/ as importer<br/>24 Python files / 18,706 LOC"]'
        regex=re.compile(
            r"src/ as importer<br/>(?P<files>"
            + NUMBER
            + r") Python files / (?P<loc>"
            + NUMBER
            + r") LOC"
        ),
        bindings=(("src_files", "files"), ("src_loc", "loc")),
    ),
    PatternSpec(
        label="mermaid src/foundationscale node pair (src_files+src_loc)",
        # evidence: docs/review/D2_current_architecture.md
        #   'FS["src/foundationscale<br/>24 Python files / 18,706 LOC<br/>..."]'
        # Same two keys as the node above and that is correct, not duplication:
        # src/ contains no .py outside the package (measured 0), so the two
        # nodes state one quantity twice and must never be allowed to disagree.
        regex=re.compile(
            r"src/foundationscale<br/>(?P<files>"
            + NUMBER
            + r") Python files / (?P<loc>"
            + NUMBER
            + r") LOC"
        ),
        bindings=(("src_files", "files"), ("src_loc", "loc")),
    ),
    PatternSpec(
        label="mermaid gates/ node file count (gates_files)",
        # evidence: docs/review/D2_current_architecture.md
        #   'GATES["gates/<br/>9 files / 10,059 LOC"]'
        # Only the file count is bound: the census has no gates_loc key, so the
        # LOC in this label sits in NO denominator and is not claimed measured.
        regex=re.compile(r"gates/<br/>(?P<num>" + NUMBER + r") files /"),
        bindings=(("gates_files", "num"),),
    ),
    PatternSpec(
        label="N private names cross the shim boundary (shim_private)",
        # evidence: docs/review/D2_current_architecture.md
        #   "and 60 private names still cross the boundary through the
        #    `tools/live_save_gate.py` compatibility shim."
        regex=re.compile(r"(?P<num>" + NUMBER + r") private names still cross the boundary"),
        bindings=(("shim_private", "num"),),
    ),
    PatternSpec(
        label="mermaid import edges, per source node (imports_tests+imports_tools+imports_src)",
        # evidence: docs/review/D2_current_architecture.md
        #   'TESTS -->|"94 Python import statements"| FS'
        # One regex, three keys, keyed on the SOURCE node -- the three edges are
        # textually identical apart from that node, so anchoring on the phrase
        # alone would bind all three to whichever key came first.
        regex=re.compile(
            r"TESTS -->\|\"(?P<tests>"
            + NUMBER
            + r") Python import statements\"\| FS\n"
            + r"\s*TOOLS -->\|\"(?P<tools>"
            + NUMBER
            + r") Python import statements\"\| FS\n"
            + r"\s*SRC -->\|\"(?P<src>"
            + NUMBER
            + r") Python import statements"
        ),
        bindings=(
            ("imports_tests", "tests"),
            ("imports_tools", "tools"),
            ("imports_src", "src"),
        ),
    ),
    # --- #241: the countable that lives in the build configuration ----------
    #
    # #240 was a false countable inside the scanned corpus. This is the same
    # defect one layer out, where the gate could not look at all: the scan set
    # is `docs README.md`, so every number stated in the Makefile and in
    # ci.yml -- the two files that DECIDE what CI measures -- sat in no
    # denominator. Both said "3 of 3 files under checks/ are unchecked" in the
    # commit that ADDED the fourth file, having copied the "3 files" out of
    # mypy's own "Found 10 errors in 3 files (checked 4 source files)": the
    # error-bearing count, printed as the denominator.
    #
    # The clause states the count ONCE, deliberately. An "N of N" phrasing
    # would bind one census key through two groups, and Site.tokens is keyed by
    # census key -- the second group would overwrite the first and only half
    # the phrase would rewrite, which is rule 4's failure mode rather than its
    # satisfaction. One quantity, one token.
    PatternSpec(
        label="all N files under checks/ are unchecked (checks_files)",
        # evidence: Makefile and .github/workflows/ci.yml, one clause each
        #   "all 4 files under checks/ are unchecked"
        regex=re.compile(r"all\s+(?P<num>" + NUMBER + r")\s+files under checks/"),
        bindings=(("checks_files", "num"),),
    ),
    # The mutation corpus, stated in the Makefile and re-derived by #242's CI
    # matrix. Two specs rather than one four-group spec: they are two clauses
    # on two lines, and the gate reads a clause. Each group binds a DIFFERENT
    # census key, which is what makes a multi-key spec legal at all -- binding
    # one key through two groups is rule 4's failure mode (Site.tokens is keyed
    # by census key, so the second group would overwrite the first and only
    # half the phrase would rewrite).
    PatternSpec(
        label="mutation corpus size pair (total_rows+mut_modules)",
        # evidence: Makefile
        #   "The mutation corpus is 78 rows over 9 modules."
        regex=re.compile(
            r"mutation corpus is\s+(?P<rows>" + NUMBER + r")\s+rows over\s+"
            r"(?P<mods>" + NUMBER + r")\s+modules"
        ),
        bindings=(("total_rows", "rows"), ("mut_modules", "mods")),
    ),
    PatternSpec(
        label="mutation corpus halves pair (must_fire+must_pass)",
        # evidence: Makefile
        #   "Of those, 69 are MUST_FIRE mutants and 9 are MUST_PASS controls."
        regex=re.compile(
            r"(?P<fire>" + NUMBER + r")\s+are MUST_FIRE mutants and\s+"
            r"(?P<passes>" + NUMBER + r")\s+are MUST_PASS controls"
        ),
        bindings=(("must_fire", "fire"), ("must_pass", "passes")),
    ),
)

# Historical masks, CLAUSE-level (rule 3). Each swallows the dead claim and
# stops at the sentence boundary so a LIVE claim on the same line stays in
# scope. Applied before matching, unmasked before writing: replacement spans
# are computed on the original text and the mask is length-preserving.
HISTORICAL_MASKS: Final[tuple[Pattern[str], ...]] = (
    # "...drafted against a census of 13,667 lines ..." is history; the lazy
    # clause runs to sentence end (a '.' followed by whitespace/EOL) so a
    # following "The package NOW measures ..." on the SAME line is untouched.
    re.compile(r"drafted against a census of " + NUMBER + r" lines.*?(?=\.(?:\s|$))"),
    # evidence: h100_validation/h100/EVIDENCE.md
    #   "The launcher exports **14** variables (an earlier count of 12 was
    #    wrong; ...)": the 12 is history, the 14 is the live claim.
    re.compile(r"earlier count of " + NUMBER + r" was wrong"),
)


def parse_number(token: str) -> int:
    return int(token.replace(",", ""))


def style_like(token: str, value: int) -> str:
    """Rewrite `value` in the thousands-separator style already used at this
    site -- a doc that wrote 18,599 keeps its commas; one that wrote 18599
    does not grow any."""
    return f"{value:,}" if "," in token else str(value)


def mask_historical(text: str) -> str:
    """Blank historical clauses with spaces (length-preserving, so every span
    found in the masked text is valid in the original). A number stated AS
    HISTORY is thereby neither reported nor rewritten."""
    chars = list(text)
    for rx in HISTORICAL_MASKS:
        for m in rx.finditer(text):
            for i in range(m.start(), m.end()):
                if chars[i] != "\n":
                    chars[i] = " "
    return "".join(chars)


@dataclass
class Site:
    """One anchored pattern match: the unit of the denominator."""

    path: Path
    label: str
    values: dict[str, int]  # census key -> number stated in the document
    expected: dict[str, int]  # census key -> measured census value
    tokens: dict[str, tuple[int, int, str]]  # census key -> (start, end, raw)

    @property
    def drifted_keys(self) -> list[str]:
        return [k for k, v in self.values.items() if v != self.expected[k]]

    @property
    def drifted(self) -> bool:
        # A pair site drifts as ONE site even if only one token moved.
        return bool(self.drifted_keys)


@dataclass
class Report:
    files_scanned: int = 0
    files_excluded: int = 0
    sites: list[Site] = field(default_factory=list)
    pattern_hits: dict[str, int] = field(default_factory=dict)

    @property
    def denominator(self) -> int:
        return len(self.sites)

    @property
    def agreeing(self) -> int:
        return sum(1 for s in self.sites if not s.drifted)

    @property
    def drifted(self) -> int:
        return sum(1 for s in self.sites if s.drifted)

    @property
    def zero_match_labels(self) -> list[str]:
        return [label for label, n in self.pattern_hits.items() if n == 0]


def collect_files(paths: Sequence[Path]) -> tuple[list[Path], int]:
    files: set[Path] = set()
    excluded = 0
    for p in paths:
        # #241: a DIRECTORY is walked for prose (*.md); a FILE named on the
        # command line is scanned whatever its suffix. Before this, the suffix
        # test below rejected explicitly-named non-markdown, so Makefile and
        # .github/workflows/ci.yml could not enter the scan set even when
        # passed by name -- and every countable stated in the build
        # configuration, the files that decide what CI measures, sat in no
        # denominator. That is where #241's "3 of 3 files under checks/" lived
        # while #240's gate reported CLEAR over the docs beside it.
        named_file = p.is_file()
        candidates = [p] if named_file else sorted(p.rglob("*.md"))
        for c in candidates:
            if not named_file and c.suffix != ".md":
                continue
            if any(part in EXCLUDED_PARTS for part in c.parts):
                excluded += 1
            else:
                files.add(c)
    return sorted(files), excluded


def read_all(files: Sequence[Path]) -> tuple[dict[Path, str], list[Path]]:
    """Read everything before anything is written: --fix fails CLOSED.
    If any document is unreadable, nothing is rewritten -- a gate that
    half-rewrites a corpus manufactures the very drift it reports."""
    texts: dict[Path, str] = {}
    unreadable: list[Path] = []
    for f in files:
        try:
            texts[f] = f.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            unreadable.append(f)
    return texts, unreadable


def scan(
    texts: dict[Path, str],
    census: dict[str, int],
    files_scanned: int,
    files_excluded: int,
) -> Report:
    report = Report(files_scanned=files_scanned, files_excluded=files_excluded)
    for spec in PATTERNS:
        report.pattern_hits[spec.label] = 0
        # Without the oracle value no comparison exists; fabricating one
        # would be inference, which is the failure this gate replaces.
        if any(key not in census for key, _ in spec.bindings):
            continue
        for path in sorted(texts):
            text = texts[path]
            masked = mask_historical(text)
            for m in spec.regex.finditer(masked):
                values: dict[str, int] = {}
                expected: dict[str, int] = {}
                tokens: dict[str, tuple[int, int, str]] = {}
                for key, group in spec.bindings:
                    token = m.group(group)
                    values[key] = parse_number(token)
                    expected[key] = census[key]
                    tokens[key] = (m.start(group), m.end(group), token)
                report.sites.append(
                    Site(
                        path=path, label=spec.label, values=values, expected=expected, tokens=tokens
                    )
                )
                report.pattern_hits[spec.label] += 1
    return report


RewritePlan = dict[Path, list[tuple[int, int, str]]]


def plan_rewrites(report: Report) -> RewritePlan:
    plans: RewritePlan = {}
    for site in report.sites:
        if not site.drifted:
            continue
        for key, (start, end, token) in site.tokens.items():
            # rule 4: a drifted pair rewrites BOTH tokens, the agreeing one
            # included -- they are one measurement stated in two tokens.
            new = style_like(token, site.expected[key])
            plans.setdefault(site.path, []).append((start, end, new))
    return plans


def apply_rewrites(plans: RewritePlan, texts: dict[Path, str]) -> int:
    rewrites = 0
    for path in sorted(plans):
        text = texts[path]
        for start, end, new in sorted(plans[path], key=lambda r: r[0], reverse=True):
            text = text[:start] + new + text[end:]
            rewrites += 1
        path.write_text(text, encoding="utf-8")
        texts[path] = text  # a second pass must see post-fix content
    return rewrites


def per_key(report: Report) -> dict[str, list[int]]:
    """census key -> [sites matched, agreeing, drifted]; drift is attributed
    to the key that actually disagrees, never to a neighbour."""
    stats: dict[str, list[int]] = {}
    for site in report.sites:
        for key in site.values:
            s = stats.setdefault(key, [0, 0, 0])
            s[0] += 1
            if key in site.drifted_keys:
                s[2] += 1
            else:
                s[1] += 1
    return stats


def key_label(report: Report, key: str) -> str:
    for site in report.sites:
        if key in site.values:
            return site.label
    return ""


def gate_exit_code(report: Report) -> int:
    if report.denominator == 0:
        return EXIT_UNMEASURED
    if report.drifted:
        return EXIT_RED
    return EXIT_CLEAR


def print_provenance() -> None:
    # A verdict is attributable only to the interpreter that produced it.
    print(f"interpreter: {sys.executable}")
    print(f"version_info: {sys.version_info}")


class RefusingParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:  # argparse would exit 2; not in the namespace
        print(f"REFUSE (bad invocation): {message}", file=sys.stderr)
        raise SystemExit(EXIT_REFUSE)


def build_parser() -> argparse.ArgumentParser:
    p = RefusingParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("paths", nargs="*", help="markdown roots (files or directories)")
    p.add_argument("--census", help="measured census JSON (the oracle)")
    p.add_argument(
        "--fix", action="store_true", help="rewrite drifted numbers in place (pairs together)"
    )
    p.add_argument(
        "--self-test", action="store_true", help="run controls in a temporary directory and exit"
    )
    return p


ControlFn = Callable[[Path], tuple[bool, str]]

# Fixture oracle, deliberately NOT the live census: these are the values the
# controls below assert against, so they must be stable literals. Every key any
# PatternSpec binds appears here -- a pattern whose key is absent is skipped by
# scan() and would sit in the self-test's blind spot rather than in its
# denominator, which is the #240 shape one layer down.
SELF_CENSUS: Final[dict[str, int]] = {
    "decisions_loc": 982,
    "h100_loc": 31313,
    "h100_files": 63,
    "launch_sh_loc": 9274,
    "launch_py_loc": 1615,
    "tools_files": 9,
    "tools_loc": 8695,
    "tests_files": 54,
    "tests_loc": 28288,
    "src_files": 24,
    "src_loc": 18706,
    "gates_files": 9,
    "shim_private": 60,
    "imports_tests": 94,
    "imports_tools": 12,
    "imports_src": 26,
    "checks_files": 4,
    "total_rows": 78,
    "mut_modules": 9,
    "must_fire": 69,
    "must_pass": 9,
}


def self_test() -> int:
    print_provenance()
    print("self-test: controls run in a temporary directory")

    def mk(root: Path, name: str, text: str) -> Path:
        doc = root / name
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_text(text, encoding="utf-8")
        return doc

    def scan_doc(doc: Path) -> Report:
        return scan({doc: doc.read_text(encoding="utf-8")}, SELF_CENSUS, 1, 0)

    def c_association(root: Path) -> tuple[bool, str]:
        # THE control whose absence caused the 865 false drifts: a document
        # dense with bare numbers and NO anchored clause must contribute
        # nothing -- the quoted strings are the false positives verbatim.
        doc = mk(
            root / "assoc",
            "noise.md",
            (
                "CLI options may be called with a single dash or 2 dashes.\n\n"
                "## 3. Decisions with teeth\n\n"
                "16 of 128 distinct experts, replicated 8x.\n\n"
                "`dcp_meta.py` at 74%, `checkpoint_gates.py` at 94% coverage.\n"
            ),
        )
        rep = scan_doc(doc)
        ok = rep.denominator == 0 and rep.drifted == 0 and rep.agreeing == 0
        return ok, f"denominator={rep.denominator} drifts={rep.drifted}"

    def c_wrong_anchor(root: Path) -> tuple[bool, str]:
        doc = mk(root / "wrong", "d.md", "Read the 900-line `DECISIONS.md` before anything else.\n")
        rep = scan_doc(doc)
        stats = per_key(rep)
        wrong_keys = [k for k, s in stats.items() if s[2]]
        ok = rep.drifted == 1 and wrong_keys == ["decisions_loc"]
        return ok, f"drifted={rep.drifted} blamed={wrong_keys}"

    def c_right_anchor(root: Path) -> tuple[bool, str]:
        doc = mk(root / "right", "d.md", "Everything starts at the 982-line `DECISIONS.md`.\n")
        rep = scan_doc(doc)
        # Must be an actual agreement -- a detector that matched nothing also
        # reports 0 drifts, and that detector is the failure mode being gated.
        ok = rep.drifted == 0 and rep.agreeing == 1
        return ok, f"agreeing={rep.agreeing} drifted={rep.drifted}"

    def c_pair_partial(root: Path) -> tuple[bool, str]:
        doc = mk(
            root / "pair", "d.md", "| `h100_validation/` (30,000 .py LOC, 63 files) | keep |\n"
        )
        rep = scan_doc(doc)
        stats = per_key(rep)
        ok = rep.drifted == 1 and stats["h100_loc"][2] == 1 and stats["h100_files"][1] == 1
        plans = plan_rewrites(rep)
        n = sum(len(v) for v in plans.values())
        texts = {doc: doc.read_text(encoding="utf-8")}
        apply_rewrites(plans, texts)
        after = doc.read_text(encoding="utf-8")
        # rule 4: BOTH tokens rewritten even though only the LOC drifted.
        ok = ok and n == 2 and "(31,313 .py LOC, 63 files)" in after
        return ok, f"drifted={rep.drifted} rewrites={n}"

    def c_historical(root: Path) -> tuple[bool, str]:
        # One line, a dead claim and a live one (rule 3). The historical half
        # cites numbers that WOULD drift on both keys if masking leaked.
        dead = (
            "drafted against a census of 25,000 lines for "
            "`h100_validation/` (25,000 .py LOC, 60 files), "
            "and was never re-measured then"
        )
        live = "`h100_validation/` (30,000 .py LOC, 63 files)"
        doc = mk(root / "hist", "d.md", f"The harness note was {dead}. Currently: {live}.\n")
        rep = scan_doc(doc)
        stats = per_key(rep)
        ok = rep.drifted == 1 and stats.get("h100_files", [0, 0, 0])[2] == 0  # 60 not seen
        plans = plan_rewrites(rep)
        texts = {doc: doc.read_text(encoding="utf-8")}
        apply_rewrites(plans, texts)
        after = doc.read_text(encoding="utf-8")
        ok = (
            ok
            and dead in after  # history byte-identical
            and "(30,000" not in after  # live stale number rewritten
            and "(31,313 .py LOC, 63 files)" in after
        )
        return ok, f"drifted={rep.drifted} mask_intact={dead in after}"

    def c_idempotent(root: Path) -> tuple[bool, str]:
        doc = mk(root / "idem", "d.md", "| `h100_validation/` (30,000 .py LOC, 61 files) |\n")
        texts = {doc: doc.read_text(encoding="utf-8")}
        first = apply_rewrites(plan_rewrites(scan_doc(doc)), texts)
        second = apply_rewrites(
            plan_rewrites(scan_doc(doc)), {doc: doc.read_text(encoding="utf-8")}
        )
        ok = first > 0 and second == 0
        return ok, f"first={first} second={second}"

    def c_zero_patterns(root: Path) -> tuple[bool, str]:
        # Only one clause present: every other anchored pattern must show up
        # in the instrument warnings, loudly -- silence is what hid the
        # first gate's broken association axis behind 0 drifts.
        doc = mk(root / "zero", "d.md", "The 982-line `DECISIONS.md`.\n")
        rep = scan_doc(doc)
        zeros = rep.zero_match_labels
        ok = bool(zeros) and any("launchers/ contains" in z for z in zeros)
        return ok, f"zero-match patterns={len(zeros)}"

    # _root, not root: this control needs no corpus on disk -- that is the
    # whole point of it -- but it keeps the uniform ControlFn signature the
    # other seven have, so the table below stays a table.
    def c_empty_corpus(_root: Path) -> tuple[bool, str]:
        rep = scan({}, SELF_CENSUS, 0, 0)
        ok = gate_exit_code(rep) == EXIT_UNMEASURED
        return ok, f"exit={gate_exit_code(rep)}"

    def c_mermaid_node(root: Path) -> tuple[bool, str]:
        # #240: the clause that was false at birth lived in a mermaid node
        # label, not in prose. Every pattern before #240 anchored on sentence
        # text, so a diagram could state any number it liked and the gate read
        # CLEAR over it. This asserts the node label is a SITE -- and that the
        # pair rewrites together, since a node carries files and LOC in one
        # label and half a corrected node is a worse artifact than none.
        doc = mk(
            root / "mermaid",
            "d.md",
            '  TOOLS["tools/<br/>9 Python files / 11,226 LOC"]\n',
        )
        rep = scan_doc(doc)
        texts = {doc: doc.read_text(encoding="utf-8")}
        rewritten = apply_rewrites(plan_rewrites(rep), texts)
        after = doc.read_text(encoding="utf-8")
        ok = rep.drifted == 1 and rewritten == 2 and "9 Python files / 8,695 LOC" in after
        return ok, f"drifted={rep.drifted} tokens={rewritten}"

    def c_import_edges(root: Path) -> tuple[bool, str]:
        # The three edges are textually identical apart from their source node,
        # so this pins that each number reaches its OWN key: the doc is built
        # with tools' and src' counts correct and only tests' wrong, and the
        # gate must blame imports_tests alone.
        doc = mk(
            root / "edges",
            "d.md",
            '  TESTS -->|"78 Python import statements"| FS\n'
            '  TOOLS -->|"12 Python import statements"| FS\n'
            '  SRC -->|"26 Python import statements, source not disaggregated"| FS\n',
        )
        rep = scan_doc(doc)
        stats = per_key(rep)
        wrong = sorted(k for k, s in stats.items() if s[2])
        ok = wrong == ["imports_tests"]
        return ok, f"blamed={wrong}"

    def c_config_files_scanned(root: Path) -> tuple[bool, str]:
        # #241: the anchor above is worthless unless the files carrying it can
        # ENTER the scan set. Before #241 they could not -- collect_files
        # rejected any candidate whose suffix was not .md, including one named
        # explicitly on the command line -- so a new PatternSpec pointed at the
        # Makefile would have matched nothing and reported itself as a
        # zero-match instrument warning, not as coverage. This control runs the
        # real collect_files over a suffixless `Makefile` and a `.yml`, which
        # is the pair the gate is now invoked with, and only then checks that
        # each is a drifted site. Asserting the drift without asserting the
        # membership would be the same vacuity one layer down.
        cfg = root / "cfg"
        cfg.mkdir(parents=True, exist_ok=True)
        mk(cfg, "Makefile", "# all 3 files under checks/ are unchecked by mypy.\n")
        mk(cfg, "ci.yml", "        # all 3 files under checks/ are unchecked.\n")
        named = [cfg / "Makefile", cfg / "ci.yml"]
        collected, _ = collect_files(named)
        if sorted(collected) != sorted(named):
            return False, f"scan set missing config files: collected={[p.name for p in collected]}"
        texts, unreadable = read_all(collected)
        rep = scan(texts, SELF_CENSUS, len(collected), 0)
        rewritten = apply_rewrites(plan_rewrites(rep), texts)
        fixed = sum(
            1 for p in named if "all 4 files under checks/" in p.read_text(encoding="utf-8")
        )
        ok = not unreadable and rep.drifted == 2 and rewritten == 2 and fixed == 2
        return ok, f"collected={len(collected)} drifted={rep.drifted} fixed={fixed}/2"

    controls: list[tuple[str, str, ControlFn]] = [
        ("association", "MUST_PASS", c_association),
        ("config files enter the scan set, and drift there", "MUST_FIRE", c_config_files_scanned),
        ("mermaid node pair is a site, and rewrites whole", "MUST_FIRE", c_mermaid_node),
        ("three identical edges, each number to its own key", "MUST_FIRE", c_import_edges),
        ("anchored clause, wrong number, blamed on right key", "MUST_FIRE", c_wrong_anchor),
        ("anchored clause, right number, counts as agreement", "MUST_PASS", c_right_anchor),
        ("pair, LOC-only drift, one site, both tokens rewritten", "MUST_FIRE", c_pair_partial),
        ("historical clause masked, live same-line number caught", "MUST_PASS", c_historical),
        ("--fix idempotent, second run rewrites 0", "MUST_PASS", c_idempotent),
        ("pattern matching nothing lands in instrument warnings", "MUST_FIRE", c_zero_patterns),
        ("empty corpus -> exit 95, never 0", "MUST_PASS", c_empty_corpus),
    ]

    behaved = 0
    misbehaved: list[str] = []
    n_fire = sum(1 for _, k, _ in controls if k == "MUST_FIRE")
    n_pass = sum(1 for _, k, _ in controls if k == "MUST_PASS")
    with tempfile.TemporaryDirectory(prefix="countables-gate-selftest-") as td:
        root = Path(td)
        for name, kind, fn in controls:
            try:
                ok, detail = fn(root / str(behaved + len(misbehaved)))
            except (OSError, KeyError, ValueError, AssertionError) as exc:
                ok, detail = False, f"raised {exc!r}"
            print(f"  [{kind}] {name}: {'ok' if ok else 'MISBEHAVED'} ({detail})")
            if ok:
                behaved += 1
            else:
                misbehaved.append(name)

    total = len(controls)
    print(
        f"self-test denominator: {behaved} of {total} controls "
        f"({n_fire} MUST_FIRE, {n_pass} MUST_PASS)"
    )
    if misbehaved:
        print("MISBEHAVED controls: " + "; ".join(misbehaved))
        return EXIT_RED
    return EXIT_CLEAR


def main(argv: Sequence[str] | None = None) -> int:
    print_provenance()
    args = build_parser().parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.paths:
        print("REFUSE: no paths given", file=sys.stderr)
        return EXIT_REFUSE
    if not args.census:
        print("REFUSE: --census is required outside --self-test", file=sys.stderr)
        return EXIT_REFUSE

    roots = [Path(p) for p in args.paths]
    missing = [p for p in roots if not p.exists()]
    if missing:
        print(
            "REFUSE: paths that do not exist: " + ", ".join(str(p) for p in missing),
            file=sys.stderr,
        )
        return EXIT_REFUSE

    census_path = Path(args.census)
    try:
        raw = json.loads(census_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"REFUSE: census unreadable: {exc}", file=sys.stderr)
        return EXIT_REFUSE
    if not isinstance(raw, dict):
        print("REFUSE: census is not a JSON object", file=sys.stderr)
        return EXIT_REFUSE
    # int-valued keys only; strings like repo_wide_loc_UNRECOVERABLE carry no
    # oracle and bools are not counts.
    census: dict[str, int] = {
        k: v for k, v in raw.items() if isinstance(v, int) and not isinstance(v, bool)
    }

    files, excluded = collect_files(roots)
    print(
        f"corpus: {len(files)} markdown files scanned, "
        f"{excluded} excluded (parts: {', '.join(EXCLUDED_PARTS)})"
    )

    texts, unreadable = read_all(files)
    if unreadable:
        print("REFUSE: unreadable documents (fail closed, nothing written):", file=sys.stderr)
        for f in unreadable:
            print(f"  {f}", file=sys.stderr)
        return EXIT_REFUSE

    report = scan(texts, census, len(files), excluded)
    stats = per_key(report)
    for key in sorted(stats):
        matched, agreed, drifted = stats[key]
        print(
            f"  {key:<16} [{key_label(report, key)}] "
            f"sites={matched} agreeing={agreed} drifted={drifted}"
        )

    zeros = report.zero_match_labels
    if zeros:
        print(
            "instrument warnings -- patterns that matched nothing "
            "(the clause moved or the pattern is wrong):"
        )
        for label in zeros:
            print(f"  - {label}")

    # SCOPE, on the wire. CLEAR is a verdict about the anchored sites below and
    # about nothing else -- a census key with no pattern is not "agreeing", it
    # is unmeasured. Leaving that unsaid is the #233 failure one level up: the
    # verdict would be read as covering the census it was only sampled from.
    bound = {key for spec in PATTERNS for key, _ in spec.bindings}
    unbound = sorted(set(census) - bound)
    print(
        f"scope: {len(bound)} of {len(census)} census keys carry an anchored "
        f"pattern; {len(unbound)} are UNMEASURED by this gate"
    )
    for i in range(0, len(unbound), 5):
        print("  unmeasured: " + ", ".join(unbound[i : i + 5]))

    # The denominator is anchored sites matched -- never digit groups, never
    # files. There is no unclassified column; out-of-scope numbers are silence.
    print(
        f"TOTAL: {report.denominator} anchored sites (the denominator), "
        f"{report.agreeing} agreeing, {report.drifted} drifted"
    )

    if report.denominator == 0:
        print("UNMEASURED: no files, or no pattern matched any site", file=sys.stderr)
        return EXIT_UNMEASURED

    for site in report.sites:
        if site.drifted:
            detail = "; ".join(
                f"{k}: doc={site.values[k]} census={site.expected[k]}" for k in site.values
            )
            print(f"DRIFT {site.path}: [{site.label}] {detail}")

    if report.drifted and args.fix:
        plans = plan_rewrites(report)
        try:
            n = apply_rewrites(plans, texts)
        except OSError as exc:
            print(f"REFUSE: write failed mid-fix: {exc}", file=sys.stderr)
            return EXIT_REFUSE
        print(f"--fix: rewrote {n} token(s) across {len(plans)} file(s)")

    verdict = "RED" if report.drifted else "CLEAR"
    print(
        f"VERDICT {verdict}: {report.agreeing} agreed / "
        f"{report.drifted} drifted over {report.denominator} anchored sites"
    )
    return gate_exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
