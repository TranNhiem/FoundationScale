#!/usr/bin/env python3
"""Verification-matrix gate: every row must be able to fail, and the doc must
match the ledger.

Usage:
  verification_matrix.py [--repo-root ROOT] [--matrix PATH] [--doc PATH]
  verification_matrix.py --self-test   plant each defect in a temp copy and exit

The class. validation_campaigns/verification_matrix/matrix.json is the
repository's ledger of claims, and docs/VERIFICATION_MATRIX.md is the prose
rendering of that ledger. Both are edited by hand, so they can drift in either
direction, and a row can rot in place: its control_arm can slide from a real
arm to a placeholder while every presence check stays green. Both shapes are
declarations that no longer measure anything, and neither had a gate.

Checks, each reported with its own verdict line:
  C1 CONTROL_ARM_PRESENT     every row's control_arm is non-empty after strip()
  C2 CONTROL_ARM_SUBSTANTIVE no control_arm is a placeholder
  C3 SCHEMA                  exact key set, unique ids, tier in {0,1}, the list
                             fields are lists, no dangling blocked_by id
  C4 COUNTABLES              the doc's "## The honest count" integers equal the
                             ones derived from the JSON, parsed positionally
  C5 DOC_JSON_AGREEMENT      every T0-/T1- table row in the doc equals the JSON

C2 is the check that actually bites. This document's own thesis is that a row
with no arm that must come out DIFFERENT is a row that cannot fail, and a row
that cannot fail is not a measurement. A presence-only check (C1) passes a
placeholder like "not recorded"; C2 exists because C1 cannot see that shape.

Doctrine wiring:
  (1) Exit codes are 0/5/95/96 and never 1. CLEAR means every check passed;
      RED means at least one failed; UNMEASURED means an input was missing or
      unparseable, or C4's paragraph could not be located -- an unmet
      precondition is not a failure and must not be reported as one; REFUSE
      is bad arguments.
  (2) C4 parses the honest-count paragraph POSITIONALLY and treats a missing
      or mis-shaped paragraph as UNMEASURED, never as agreement. A count the
      gate cannot find is not a count the gate has verified.
  (3) --self-test plants each defect in a tempfile.TemporaryDirectory() copy
      of the REAL files (the real files are never mutated) and asserts two
      things separately: REACHED-SITE (the check's own finding list names the
      planted row) and OUTCOME (the exit code). An exit-code-only assertion
      cannot distinguish "the right check caught the right row" from "the
      gate is RED for some other reason", and this repository has repeatedly
      shipped controls that cannot fail.
  (4) SC6 is the negative control: the unmodified files must come back CLEAR.
      Without it SC1-SC5 prove nothing, because a gate that always returns 5
      would pass every positive control. SC7 is the abstention control: a
      missing matrix is UNMEASURED (95), not RED. SC8 is the precedence
      control: with one check RED and another UNMEASURED at the same time the
      verdict must be RED, because 95 is the code this plane reads as "not a
      failure" and a fired check must not be reported as neutral. SC9 is the
      discrimination control, and it is the only one that does not plant into
      a file copy: the defect it guards lives in the COUNTING, not in the
      ledger. ``"MEASURED" in status`` also matches ``"UNMEASURED"`` (#199),
      so it would tally a row that explicitly refused for want of an arm as a
      row that measured something -- the one direction this countable must
      never fail in. Two synthetic rows differing only in the ``UN`` prefix
      are the whole instrument.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

# This repository's exit contract is 0/5/95/96 and never 1.
RC_CLEAR = 0
RC_RED = 5
RC_UNMEASURED = 95
RC_REFUSE = 96

CLEAR = "CLEAR"
RED = "RED"
UNMEASURED = "UNMEASURED"

DEFAULT_MATRIX = "validation_campaigns/verification_matrix/matrix.json"
DEFAULT_DOC = "docs/VERIFICATION_MATRIX.md"

JsonRow = dict[str, object]

EXPECTED_KEYS = {
    "id",
    "tier",
    "group",
    "claim",
    "run_arm",
    "control_arm",
    "cost",
    "status",
    "blocked_by",
    "findings",
}

# C2's exact-match placeholders, compared case-insensitively after strip().
PLACEHOLDER_EXACT = {"-", "n/a", "na", "none", "tbd", "todo", "unknown", "?"}
# C2's substring phrases, likewise case-insensitive.
PLACEHOLDER_PHRASES = (
    "not recorded",
    "not specified",
    "to be determined",
    "no control",
)

# The order C4 checks the honest-count integers in. The paragraph is parsed
# positionally, so this tuple IS the contract between the JSON and the prose.
COUNTABLE_ORDER = (
    "total",
    "tier0",
    "tier1",
    "no_key",
    "blocked",
    "unrunnable",
    "measured",
    "refusals",
)

HONEST_HEADING = re.compile(r"^##[ \t]+The honest count\b.*$", re.MULTILINE)
NEXT_HEADING = re.compile(r"^##[ \t]", re.MULTILINE)
INTEGER = re.compile(r"\d+")

# A doc table row adjudicated by C5: first cell is a row id like T0-1 / T1-23.
DOC_ROW_ID = re.compile(r"^T[01]-\d+$")
# Pipes inside a cell are escaped (\|) in markdown; split only on bare pipes.
UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
# The doc table's column order after the id cell.
DOC_FIELDS = ("claim", "run_arm", "control_arm", "cost", "status")


@dataclass
class CheckResult:
    """One check's verdict, with the findings the self-test asserts against."""

    name: str
    status: str  # CLEAR | RED | UNMEASURED
    summary: str
    findings: list[str] = field(default_factory=list)


def _row_label(row: JsonRow, index: int) -> str:
    rid = row.get("id")
    return rid if isinstance(rid, str) else f"<row {index}>"


def _status(row: JsonRow) -> str:
    value = row.get("status")
    return value if isinstance(value, str) else ""


# ------------------------------------------------------------------- checks


def check_control_arm_present(rows: list[JsonRow]) -> CheckResult:
    """C1: every row carries a control_arm that is non-empty after strip()."""
    name = "C1 CONTROL_ARM_PRESENT"
    findings = []
    for i, row in enumerate(rows):
        arm = row.get("control_arm")
        if not isinstance(arm, str):
            continue  # a missing or non-string arm is C3's finding, not C1's
        if not arm.strip():
            findings.append(f"{_row_label(row, i)}: control_arm is empty")
    if findings:
        return CheckResult(name, RED, f"{len(findings)} empty control_arm(s)", findings)
    return CheckResult(name, CLEAR, f"{len(rows)} row(s) carry a non-empty control_arm")


def check_control_arm_substantive(rows: list[JsonRow]) -> CheckResult:
    """C2: no control_arm is a placeholder.

    Exact placeholders are matched case-insensitively after strip(); the
    phrases are matched case-insensitively as substrings. This is the check
    that actually bites: C1 passes "not recorded", C2 does not.
    """
    name = "C2 CONTROL_ARM_SUBSTANTIVE"
    findings = []
    for i, row in enumerate(rows):
        arm = row.get("control_arm")
        if not isinstance(arm, str) or not arm.strip():
            continue  # empty is C1's finding; non-string is C3's
        low = arm.strip().lower()
        if low in PLACEHOLDER_EXACT:
            findings.append(f"{_row_label(row, i)}: control_arm is the placeholder {arm.strip()!r}")
            continue
        hit = next((p for p in PLACEHOLDER_PHRASES if p in low), None)
        if hit is not None:
            findings.append(
                f"{_row_label(row, i)}: control_arm contains placeholder phrase {hit!r}"
            )
    if findings:
        return CheckResult(name, RED, f"{len(findings)} placeholder control_arm(s)", findings)
    return CheckResult(name, CLEAR, "no control_arm is a placeholder")


def check_schema(rows: list[JsonRow]) -> CheckResult:
    """C3: exact key set, unique ids, tier in {0,1}, list fields, no dangles."""
    name = "C3 SCHEMA"
    findings: list[str] = []
    ids: list[str] = []
    for i, row in enumerate(rows):
        label = _row_label(row, i)
        keys = set(row)
        if keys != EXPECTED_KEYS:
            findings.append(
                f"{label}: key set drift, missing={sorted(EXPECTED_KEYS - keys)}"
                f" extra={sorted(keys - EXPECTED_KEYS)}"
            )
        rid = row.get("id")
        if isinstance(rid, str):
            ids.append(rid)
        else:
            findings.append(f"<row {i}>: id is missing or not a string")
        tier = row.get("tier")
        if isinstance(tier, bool) or tier not in (0, 1):
            findings.append(f"{label}: tier must be 0 or 1, got {tier!r}")
        for key in ("blocked_by", "findings"):
            if not isinstance(row.get(key), list):
                findings.append(f"{label}: {key} must be a list")
    seen: set[str] = set()
    for rid in ids:
        if rid in seen:
            findings.append(f"duplicate id {rid}")
        seen.add(rid)
    id_set = set(ids)
    for i, row in enumerate(rows):
        blocked_by = row.get("blocked_by")
        if not isinstance(blocked_by, list):
            continue
        for dep in blocked_by:
            if not isinstance(dep, str):
                findings.append(f"{_row_label(row, i)}: blocked_by entry {dep!r} is not a string")
            elif dep not in id_set:
                # A dangling dependency is RED, not ignored.
                findings.append(f"{_row_label(row, i)}: blocked_by names dangling id {dep}")
    if findings:
        return CheckResult(name, RED, f"{len(findings)} schema violation(s)", findings)
    return CheckResult(
        name,
        CLEAR,
        f"{len(rows)} row(s), {len(id_set)} unique id(s), no dangling blocked_by",
    )


def derive_countables(rows: list[JsonRow]) -> dict[str, int]:
    """The integers C4 requires the doc's honest-count paragraph to restate."""
    statuses = [_status(row) for row in rows]
    no_key = {i for i, s in enumerate(statuses) if s == "no key"}
    blocked: set[int] = set()
    for i, row in enumerate(rows):
        blocked_by = row.get("blocked_by")
        if isinstance(blocked_by, list) and blocked_by:
            blocked.add(i)
    # ANCHORED on purpose (the #199 class): a bare `"MEASURED" in s` also matches
    # "UNMEASURED", so a row that explicitly refuses for want of an arm would be
    # counted as a row that measured something -- the one direction this countable
    # must never fail in. The negation is checked first because it is the superstring.
    measured = sum(
        1 for s in statuses if (("MEASURED" in s and "UNMEASURED" not in s) or s.startswith("done"))
    )
    # CASE-INSENSITIVE on purpose: the matrix carries both "REFUSES by design"
    # (T1-3) and "refuses by design" (T1-22), so a case-sensitive match here
    # silently undercounts the refusals.
    refusals = sum(1 for s in statuses if "refuses by design" in s.lower())
    return {
        "total": len(rows),
        "tier0": sum(1 for row in rows if row.get("tier") == 0),
        "tier1": sum(1 for row in rows if row.get("tier") == 1),
        "no_key": len(no_key),
        "blocked": len(blocked),
        "unrunnable": len(no_key | blocked),
        "measured": measured,
        "refusals": refusals,
    }


def _honest_count_body(doc: str) -> str | None:
    """The text of the '## The honest count' section, or None if absent."""
    heading = HONEST_HEADING.search(doc)
    if heading is None:
        return None
    rest = doc[heading.end() :]
    following = NEXT_HEADING.search(rest)
    return rest[: following.start()] if following else rest


def check_countables(rows: list[JsonRow], doc: str) -> CheckResult:
    """C4: the doc's honest-count integers equal the derived ones, positionally."""
    name = "C4 COUNTABLES"
    derived = derive_countables(rows)
    want = [derived[key] for key in COUNTABLE_ORDER]
    body = _honest_count_body(doc)
    if body is None:
        # A count the gate cannot find is not a count the gate has verified.
        return CheckResult(
            name,
            UNMEASURED,
            "no '## The honest count' section in the doc -- cannot adjudicate",
        )
    stated = [int(m.group(0)) for m in INTEGER.finditer(body)]
    if len(stated) != len(want):
        return CheckResult(
            name,
            UNMEASURED,
            f"honest-count paragraph states {len(stated)} integer(s); the"
            f" expected shape is {len(want)} ({', '.join(COUNTABLE_ORDER)})",
        )
    findings = [
        f"{key}: doc states {got}, matrix derives {exp}"
        for key, got, exp in zip(COUNTABLE_ORDER, stated, want, strict=True)
        if got != exp
    ]
    if findings:
        return CheckResult(
            name,
            RED,
            f"{len(findings)} countable(s) disagree with the JSON",
            findings,
        )
    summary = ", ".join(f"{k}={v}" for k, v in zip(COUNTABLE_ORDER, want, strict=True))
    return CheckResult(name, CLEAR, f"doc restates the derived countables: {summary}")


def _norm_cell(cell: str) -> str:
    """Unescape the markdown pipe escape, drop inline-code ticks, collapse space.

    Backticks are MARKUP, not content: the doc is a markdown rendering and the
    JSON is the ledger, so requiring the ledger to carry the doc's inline-code
    ticks would put presentation in the record. Applied to BOTH sides, so the
    comparison stays symmetric. It removes only the tick characters -- two
    cells whose words differ still disagree, which is what C5 exists to catch.
    """
    return " ".join(cell.replace("\\|", "|").replace("`", "").split())


def _split_table_line(line: str) -> list[str]:
    """Split a markdown table line on bare pipes, dropping the outer empties."""
    cells = UNESCAPED_PIPE.split(line.strip())
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return cells


def parse_doc_rows(doc: str) -> tuple[dict[str, list[str]], list[str]]:
    """Map each T0-/T1- table row's id to its normalised cells, plus problems."""
    rows: dict[str, list[str]] = {}
    problems: list[str] = []
    for lineno, line in enumerate(doc.splitlines(), 1):
        if not line.strip().startswith("|"):
            continue
        cells = _split_table_line(line)
        if not cells or not DOC_ROW_ID.match(_norm_cell(cells[0])):
            continue
        rid = _norm_cell(cells[0])
        if len(cells) < 1 + len(DOC_FIELDS):
            problems.append(
                f"{rid}: doc row has {len(cells)} cell(s), expected"
                f" {1 + len(DOC_FIELDS)} (line {lineno})"
            )
            continue
        if rid in rows:
            problems.append(f"{rid}: appears twice in the doc tables (line {lineno})")
            continue
        rows[rid] = [_norm_cell(cell) for cell in cells[1 : 1 + len(DOC_FIELDS)]]
    return rows, problems


def check_doc_json_agreement(rows: list[JsonRow], doc: str) -> CheckResult:
    """C5: the doc tables and the JSON name the same ids with the same cells."""
    name = "C5 DOC_JSON_AGREEMENT"
    doc_rows, findings = parse_doc_rows(doc)
    json_rows: dict[str, JsonRow] = {}
    for row in rows:
        rid = row.get("id")
        if isinstance(rid, str):
            json_rows[rid] = row
    for rid in sorted(set(doc_rows) - set(json_rows)):
        findings.append(f"{rid}: present in the doc, absent from the JSON")
    for rid in sorted(set(json_rows) - set(doc_rows)):
        findings.append(f"{rid}: present in the JSON, absent from the doc")
    for rid in sorted(set(doc_rows) & set(json_rows)):
        json_row = json_rows[rid]
        # strict=True is safe: parse_doc_rows truncates every row it admits to
        # exactly len(DOC_FIELDS) cells, and refuses a shorter one above.
        for field_name, doc_value in zip(DOC_FIELDS, doc_rows[rid], strict=True):
            raw = json_row.get(field_name)
            # _norm_cell on BOTH sides, or the normalisation is one-sided and
            # a ledger cell carrying markdown ticks reads as drift against the
            # identical doc cell. Symmetric normalisation is what lets the
            # docstring's claim be true.
            json_value = _norm_cell(raw) if isinstance(raw, str) else ""
            if doc_value != json_value:
                findings.append(
                    f"{rid}: field {field_name!r} drifted, doc={doc_value!r} json={json_value!r}"
                )
    if findings:
        return CheckResult(
            name,
            RED,
            f"{len(findings)} disagreement(s) between doc and JSON",
            findings,
        )
    return CheckResult(
        name,
        CLEAR,
        f"{len(doc_rows)} doc row(s) agree with the JSON field-for-field",
    )


# -------------------------------------------------------------- adjudication


def load_matrix(path: Path) -> list[JsonRow] | None:
    """The matrix's rows, or None if the file is missing or unparseable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    rows = data.get("rows")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        return None
    return rows


def load_doc(path: Path) -> str | None:
    """The rendered markdown, or None if the file is missing or unreadable."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _report(result: CheckResult) -> None:
    print(f"{result.name} {result.status}: {result.summary}")
    for finding in result.findings:
        print(f"  {result.name} finding: {finding}")


def adjudicate(matrix_path: Path, doc_path: Path) -> tuple[int, dict[str, CheckResult]]:
    """Run all five checks; return (exit code, results keyed C1..C5)."""
    results: dict[str, CheckResult] = {}
    rows = load_matrix(matrix_path)
    if rows is None:
        print(
            f"VERIFICATION-MATRIX UNMEASURED: {matrix_path} is missing or is"
            " not a matrix JSON with a 'rows' list -- no artifact, so no claim"
        )
        return RC_UNMEASURED, results
    print(f"DENOMINATOR: {len(rows)} matrix row(s) under adjudication, 5 checks")
    results["C1"] = check_control_arm_present(rows)
    results["C2"] = check_control_arm_substantive(rows)
    results["C3"] = check_schema(rows)
    doc_text = load_doc(doc_path)
    if doc_text is None:
        results["C4"] = CheckResult(
            "C4 COUNTABLES", UNMEASURED, f"{doc_path} is missing or unreadable"
        )
        results["C5"] = CheckResult(
            "C5 DOC_JSON_AGREEMENT", UNMEASURED, f"{doc_path} is missing or unreadable"
        )
    else:
        results["C4"] = check_countables(rows, doc_text)
        results["C5"] = check_doc_json_agreement(rows, doc_text)
    for key in ("C1", "C2", "C3", "C4", "C5"):
        _report(results[key])
    statuses = {r.status for r in results.values()}
    # RED outranks UNMEASURED, and the reason is that these five checks are
    # INDEPENDENT. A whole-input failure already returned 95 above, before any
    # check ran. Past that point C4 abstaining says nothing about C5, which
    # read both artifacts and positively found disagreements -- and 95 is the
    # code this plane treats as "not a failure", so letting one check's
    # abstention set the verdict would report a fired RED as neutral. SC8
    # pins this: without it the precedence is a claim no scenario measures.
    if RED in statuses:
        rc, verdict = RC_RED, RED
    elif UNMEASURED in statuses:
        rc, verdict = RC_UNMEASURED, UNMEASURED
    else:
        rc, verdict = RC_CLEAR, CLEAR
    if verdict == CLEAR:
        detail = "all 5 checks passed"
    else:
        detail = "; ".join(f"{r.name}={r.status}" for r in results.values() if r.status != CLEAR)
    print(f"VERIFICATION-MATRIX {verdict}: {detail}")
    return rc, results


# ---------------------------------------------------------------- self-test
#
# Every control below asserts TWO things separately:
#   (a) REACHED-SITE -- the check under test actually ran and reported on the
#       planted row (the planted id appears in that check's own finding list);
#   (b) OUTCOME -- the returned exit code is the expected one.
# An assertion on the exit code alone is NOT acceptable: rc==5 cannot
# distinguish "the right check caught the planted row" from "the gate is RED
# for an unrelated reason (or always RED)", and this repository has repeatedly
# shipped controls that cannot fail.


def _fresh_copies(matrix_src: Path, doc_src: Path, dest: Path) -> tuple[Path, Path]:
    """Copy the REAL files into a temp dir; the real files are never mutated."""
    matrix = dest / "matrix.json"
    doc = dest / "VERIFICATION_MATRIX.md"
    shutil.copy2(matrix_src, matrix)
    shutil.copy2(doc_src, doc)
    return matrix, doc


def _plant_matrix_value(matrix: Path, row_index: int, key: str, value: object) -> str | None:
    """Set one field on one row of a matrix COPY; return that row's id."""
    data = json.loads(matrix.read_text(encoding="utf-8"))
    rid = data["rows"][row_index]["id"]
    data["rows"][row_index][key] = value
    matrix.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return rid if isinstance(rid, str) else None


def _corrupt_honest_count(doc: str) -> str | None:
    """Bump the FIRST integer of the honest-count section (the stated total)."""
    heading = HONEST_HEADING.search(doc)
    if heading is None:
        return None
    body = _honest_count_body(doc)
    if body is None:
        return None
    number = INTEGER.search(body)
    if number is None:
        return None
    start = heading.end() + number.start()
    end = heading.end() + number.end()
    return doc[:start] + str(int(number.group(0)) + 1) + doc[end:]


def _mutate_doc_status(doc: str, row_id: str, new_status: str) -> str | None:
    """Rewrite the LAST cell (the status) of row_id's doc table row."""
    lines = doc.splitlines(keepends=True)
    for i, line in enumerate(lines):
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        cells = UNESCAPED_PIPE.split(body)
        if len(cells) >= 3 and cells[1].strip() == row_id:
            cells[-2] = f" {new_status} "
            lines[i] = "|".join(cells) + ending
            return "".join(lines)
    return None


def _sc1(matrix_src: Path, doc_src: Path) -> bool:
    with tempfile.TemporaryDirectory() as td:
        matrix, doc = _fresh_copies(matrix_src, doc_src, Path(td))
        target = _plant_matrix_value(matrix, 0, "control_arm", "")
        if target is None:
            return False
        rc, results = adjudicate(matrix, doc)
    reached = any(target in f for f in results["C1"].findings)
    return reached and rc == RC_RED


def _sc2(matrix_src: Path, doc_src: Path) -> bool:
    with tempfile.TemporaryDirectory() as td:
        matrix, doc = _fresh_copies(matrix_src, doc_src, Path(td))
        target = _plant_matrix_value(matrix, 0, "control_arm", "not recorded in the design")
        if target is None:
            return False
        rc, results = adjudicate(matrix, doc)
    reached = any(target in f for f in results["C2"].findings)
    # C1 must NOT fire here: the planted arm is non-empty, so C1 cannot see it.
    # If C1 also fired, C2 would be redundant with it -- this separation is
    # what makes C2 the check that actually bites.
    c1_silent = results["C1"].status == CLEAR
    return reached and c1_silent and rc == RC_RED


def _sc3(matrix_src: Path, doc_src: Path) -> bool:
    with tempfile.TemporaryDirectory() as td:
        matrix, doc = _fresh_copies(matrix_src, doc_src, Path(td))
        if _plant_matrix_value(matrix, 0, "blocked_by", ["T9-99"]) is None:
            return False
        rc, results = adjudicate(matrix, doc)
    reached = any("T9-99" in f for f in results["C3"].findings)
    return reached and rc == RC_RED


def _sc4(matrix_src: Path, doc_src: Path) -> bool:
    with tempfile.TemporaryDirectory() as td:
        matrix, doc = _fresh_copies(matrix_src, doc_src, Path(td))
        corrupted = _corrupt_honest_count(doc.read_text(encoding="utf-8"))
        if corrupted is None:
            return False  # the control itself could not reach the plant site
        doc.write_text(corrupted, encoding="utf-8")
        rc, results = adjudicate(matrix, doc)
    reached = any(f.startswith("total:") for f in results["C4"].findings)
    return reached and rc == RC_RED


def _sc5(matrix_src: Path, doc_src: Path) -> bool:
    with tempfile.TemporaryDirectory() as td:
        matrix, doc = _fresh_copies(matrix_src, doc_src, Path(td))
        rows = load_matrix(matrix)
        target = rows[0].get("id") if rows else None
        if not isinstance(target, str):
            return False
        mutated = _mutate_doc_status(doc.read_text(encoding="utf-8"), target, "MUTATED-STATUS")
        if mutated is None:
            return False
        doc.write_text(mutated, encoding="utf-8")
        rc, results = adjudicate(matrix, doc)
    reached = any(target in f and "'status'" in f for f in results["C5"].findings)
    return reached and rc == RC_RED


def _sc6(matrix_src: Path, doc_src: Path) -> bool:
    with tempfile.TemporaryDirectory() as td:
        matrix, doc = _fresh_copies(matrix_src, doc_src, Path(td))
        rc, _results = adjudicate(matrix, doc)
    # NEGATIVE CONTROL: without an unmodified CLEAR, SC1-SC5 prove nothing,
    # because a gate that always returns 5 would pass every positive control.
    return rc == RC_CLEAR


def _sc7(matrix_src: Path, doc_src: Path) -> bool:
    with tempfile.TemporaryDirectory() as td:
        _matrix, doc = _fresh_copies(matrix_src, doc_src, Path(td))
        ghost = Path(td) / "no-such-matrix.json"
        rc, results = adjudicate(ghost, doc)
    # ABSTENTION CONTROL: an unmet precondition is UNMEASURED (95), not a
    # failure (5) -- and no check may report a verdict on data it never read.
    return rc == RC_UNMEASURED and not results


def _sc8(matrix_src: Path, doc_src: Path) -> bool:
    """PRECEDENCE CONTROL: one check RED and another UNMEASURED at once."""
    with tempfile.TemporaryDirectory() as td:
        matrix, doc = _fresh_copies(matrix_src, doc_src, Path(td))
        target = _plant_matrix_value(matrix, 0, "control_arm", "")
        if target is None:
            return False
        # Delete the honest-count heading so C4 can only abstain. C4 reads the
        # doc, C1 reads the JSON: two independent checks, one of each verdict.
        text = doc.read_text(encoding="utf-8")
        heading = HONEST_HEADING.search(text)
        if heading is None:
            return False
        doc.write_text(text[: heading.start()], encoding="utf-8")
        rc, results = adjudicate(matrix, doc)
    # REACHED-SITE: both states are genuinely present -- without this the
    # scenario could pass on a run where C4 never abstained at all.
    reached = (
        any(target in f for f in results["C1"].findings)
        and results["C1"].status == RED
        and results["C4"].status == UNMEASURED
    )
    # OUTCOME: RED wins. 95 here would report a fired check as "not measured".
    return reached and rc == RC_RED


def _sc9(_matrix_src: Path, _doc_src: Path) -> bool:
    """DISCRIMINATION CONTROL: "UNMEASURED" must not be counted as MEASURED.

    This one does not plant into a file copy, because the defect it guards is
    not a property of the ledger -- it is a property of the COUNTING. A bare
    ``"MEASURED" in status`` also matches ``"UNMEASURED"`` (the #199 class), so
    a row that explicitly refused for want of an arm would be added to the
    tally of rows that measured something, which is the one direction this
    countable must never fail in. The two synthetic rows below differ in
    exactly one character sequence -- the ``UN`` prefix -- so the assertion
    cannot pass for any reason other than the anchoring.

    It also asserts the OTHER direction (a real MEASURED row still counts), so
    a fix that simply stopped counting everything would not pass it.
    """

    def _row(status: str) -> JsonRow:
        return {"id": "T0-SC9", "tier": 0, "status": status, "blocked_by": []}

    measured = derive_countables([_row("MEASURED on 2xGB200, control arm differed")])["measured"]
    unmeasured = derive_countables([_row("UNMEASURED: no GPU arm was available")])["measured"]
    # Both halves are required. measured==1 alone passes a gate that counts
    # every row; unmeasured==0 alone passes a gate that counts none.
    return measured == 1 and unmeasured == 0


# The fourth element is the control's KIND, and the banner counts it rather
# than restating a breakdown by hand: a hand-written "(5 MUST_FIRE, 1 ..., 1
# ...)" summed to 7 the moment SC8 landed, while the total said 8. A banner
# whose names come from one list and whose count comes from another is the
# defect this campaign has already filed twice (#216, #310).
CONTROLS: list[tuple[str, Callable[[Path, Path], bool], str, str]] = [
    (
        "SC1 empty control_arm",
        _sc1,
        "C1 must fire and name the planted row",
        "MUST_FIRE",
    ),
    (
        "SC2 placeholder control_arm",
        _sc2,
        "C2 must fire while C1 stays silent -- C2 is not redundant",
        "MUST_FIRE",
    ),
    (
        "SC3 dangling blocked_by",
        _sc3,
        "C3 must fire and name the dangling id T9-99",
        "MUST_FIRE",
    ),
    (
        "SC4 corrupted honest count",
        _sc4,
        "C4 must report expected-vs-stated for 'total'",
        "MUST_FIRE",
    ),
    (
        "SC5 doc-only status drift",
        _sc5,
        "C5 must fire and name the drifted row",
        "MUST_FIRE",
    ),
    (
        "SC6 negative control (unmodified files)",
        _sc6,
        "the real files must be CLEAR, or SC1-SC5 prove nothing",
        "MUST_PASS_NEGATIVE",
    ),
    (
        "SC7 abstention control (missing matrix)",
        _sc7,
        "an unmet precondition is UNMEASURED (95), never RED",
        "MUST_ABSTAIN",
    ),
    (
        "SC8 precedence control (one RED + one UNMEASURED)",
        _sc8,
        "RED outranks UNMEASURED -- a fired check must not read as neutral",
        "MUST_OUTRANK",
    ),
    (
        "SC9 discrimination control (MEASURED vs UNMEASURED)",
        _sc9,
        "an UNMEASURED row must not be tallied as a row that measured something",
        "MUST_DISCRIMINATE",
    ),
]


def self_test(matrix: Path, doc: Path) -> int:
    """Plant the exact defect in a temp copy, run the exact machinery."""
    if not matrix.is_file() or not doc.is_file():
        print(
            "SELF-TEST UNMEASURED: the real matrix and doc must exist -- the"
            " controls plant defects in copies of them, and there is nothing"
            " to copy"
        )
        return RC_UNMEASURED
    passed = 0
    kinds: Counter[str] = Counter()
    for label, fn, why, kind in CONTROLS:
        ok = fn(matrix, doc)
        passed += ok
        kinds[kind] += 1
        print(f"SELF-TEST {'PASS' if ok else 'FAIL'} {label} -- {why}")
    breakdown = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
    print(f"self-test denominator: {passed} of {len(CONTROLS)} controls ({breakdown})")
    return RC_CLEAR if passed == len(CONTROLS) else RC_RED


# -------------------------------------------------------------------- entry


class _RefusingParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        # Bad arguments are a REFUSAL (96), never argparse's default 2, never 1.
        self.print_usage(sys.stderr)
        print(f"VERIFICATION-MATRIX REFUSE: {message}", file=sys.stderr)
        raise SystemExit(RC_REFUSE)


def build_parser() -> argparse.ArgumentParser:
    parser = _RefusingParser(
        prog="verification_matrix.py",
        description=(
            "Adjudicate the verification matrix: every row must carry a"
            " substantive control arm, and the JSON and its rendered markdown"
            " must agree."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repository root (default: the parent of this file's checks/ directory)",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help=f"matrix JSON, repo-relative (default: {DEFAULT_MATRIX})",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=None,
        help=f"rendered markdown, repo-relative (default: {DEFAULT_DOC})",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="plant each defect in a temp copy of the real files and exit",
    )
    return parser


def _resolve(repo_root: Path, path: Path) -> Path:
    """Repo-relative paths resolve against --repo-root; absolute ones stand."""
    return path if path.is_absolute() else repo_root / path


def main(argv: Sequence[str]) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root or Path(__file__).resolve().parent.parent
    matrix = _resolve(repo_root, args.matrix or Path(DEFAULT_MATRIX))
    doc = _resolve(repo_root, args.doc or Path(DEFAULT_DOC))
    if args.self_test:
        return self_test(matrix, doc)
    rc, _results = adjudicate(matrix, doc)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
