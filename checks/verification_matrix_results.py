# checks/verification_matrix_results.py
#!/usr/bin/env python3
"""Adjudicate the WHOLE verification matrix from its receipts.

matrix.json is the DECLARATION; per-pass receipts under
validation_campaigns/verification_matrix/receipts/<row_id>/<pass_id>.json
hold the ONLY results. This gate reads both and re-derives the aggregate on
every invocation -- no count of greens is ever stored anywhere, because a
stored count is a claim about the receipts that can drift away from them.
Absent receipts read UNMEASURED, never green and never red: "we did not
measure this" and "this failed" are different states and the exit contract
already has codes for both. A receipt whose claim_sha256 or
adjudicator_sha256 no longer matches the row as it reads today is STALE and
reads UNMEASURED with a reason naming the drift -- a stale receipt is
evidence for a claim nobody is making anymore.

Exit contract, and nothing else is permitted: 0 CLEAR, 5 RED, 95 UNMEASURED,
96 REFUSE. Never exit 1, never exit 2 -- argparse usage errors included,
because a usage error is a refusal to run, not a crash (#387).

Aggregate precedence: any RED row takes the whole gate to 5, because a
measured refutation is the most decision-relevant fact a campaign can
produce; next any REFUSAL (missing adjudicator, malformed receipt, broken
matrix) at 96, because the evidence tree as presented cannot be trusted;
next any UNMEASURED at 95. CLEAR only when every declared row stands green.
An empty rows list REFUSES 96: all([]) is True, and a matrix with no rows is
not a clean matrix, it is not a matrix.

--self-test builds every fixture in a tempfile.TemporaryDirectory and never
reads the real matrix.json; each control prints [PASS]/[FAIL] with the
observed value, and the self-test's own exit code follows the same contract
(0 all controls pass, 5 any control fails -- a self-test that cannot refute
its own gate proves nothing).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

CLEAR = 0
RED = 5
UNMEASURED = 95
REFUSE = 96

MATRIX_REL = Path("validation_campaigns") / "verification_matrix" / "matrix.json"
RECEIPTS_REL = Path("validation_campaigns") / "verification_matrix" / "receipts"
RECEIPTS_MODULE_REL = Path("validation_campaigns") / "verification_matrix" / "receipts.py"

_VERDICT_WORD = {"green": "GREEN", "red": "RED", "unmeasured": "UNMEASURED", "refused": "REFUSED"}
_VERDICT_EXIT = {"green": CLEAR, "red": RED, "unmeasured": UNMEASURED, "refused": REFUSE}


class _Refusal(Exception):
    """A gate-level refusal (96): the declaration or the receipts tree is in
    a state where no honest aggregate exists, so the gate refuses rather
    than inventing one."""


class _ContractParser(argparse.ArgumentParser):
    """argparse exits 2 on usage errors, which sits outside the four-state
    contract and is the one code a launcher's case statement does not handle
    (#387). A usage error is a refusal: exit 96."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(REFUSE, f"{self.prog}: error: {message}\n")


def _load_receipts_module(repo_root: Path) -> ModuleType:
    """Load the sibling receipts module BY PATH, so the one writer/reader
    shape is shared by construction. Refuse if it is absent: the gate cannot
    trust a verdict shape it cannot load."""
    path = repo_root / RECEIPTS_MODULE_REL
    if not path.is_file():
        raise _Refusal(
            f"the receipts module {RECEIPTS_MODULE_REL} does not exist under {repo_root}; "
            "without the one writer/reader shape there is nothing to adjudicate against"
        )
    spec = importlib.util.spec_from_file_location("verification_matrix_receipts", path)
    if spec is None or spec.loader is None:
        raise _Refusal(f"could not load the receipts module from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise _Refusal(f"the receipts module at {path} failed to load: {exc}") from exc
    return module


def _load_rows(matrix_path: Path) -> list[Any]:
    """Parse the declaration. Anything that is not a JSON object with a
    'rows' list is not a matrix to measure against; it is a refusal."""
    if not matrix_path.is_file():
        raise _Refusal(
            f"matrix file {matrix_path} does not exist; there is no declaration to "
            "measure receipts against"
        )
    try:
        data = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _Refusal(f"matrix file {matrix_path} is not readable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _Refusal(f"matrix file {matrix_path} is a {type(data).__name__}, not an object")
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise _Refusal(f"matrix file {matrix_path} carries no 'rows' list")
    return rows


def _aggregate_code(verdicts: list[str]) -> int:
    """Red dominates: a measured refutation outranks everything else. Then
    refusal (the evidence tree cannot be trusted as presented), then
    unmeasured. CLEAR requires every declared row green -- nothing less."""
    if "red" in verdicts:
        return RED
    if "refused" in verdicts:
        return REFUSE
    if "unmeasured" in verdicts:
        return UNMEASURED
    return CLEAR


def _adjudicate_row(
    row: Mapping[str, Any],
    *,
    receipts_root: Path,
    repo_root: Path,
    receipts_mod: ModuleType,
) -> tuple[str, str]:
    """One row's re-derived verdict, applying rule 3 (missing adjudicator
    file -> refusal, never red), rule 5 (no receipts -> unmeasured), rule 6
    (stale hashes -> unmeasured naming the drift), and the malformed-receipt
    refusal that keeps corrupt evidence from shrinking the denominator."""
    row_id = row["id"]
    adjudicator = row.get("adjudicator")
    adjudicator_bytes: bytes | None = None
    if adjudicator is not None:
        path = repo_root / adjudicator
        if path.is_file():
            try:
                adjudicator_bytes = path.read_bytes()
            except OSError as exc:
                return (
                    "refused",
                    f"adjudicator {adjudicator!r} exists but cannot be read ({exc}); "
                    "its bytes cannot be hashed against the receipts",
                )
    try:
        receipts = receipts_mod.read_receipts(receipts_root, row_id)
    except receipts_mod.ReceiptError as exc:
        return (
            "refused",
            f"malformed receipt: {exc} -- refusing (96): a receipt that cannot be read "
            "is not evidence, and it must not silently drop out of the denominator",
        )
    except OSError as exc:
        return ("refused", f"could not read the receipts for row {row_id}: {exc}")
    return receipts_mod.row_verdict(row=row, receipts=receipts, adjudicator_bytes=adjudicator_bytes)


def adjudicate_matrix(
    *,
    rows: list[Any],
    receipts_root: Path,
    repo_root: Path,
    receipts_mod: ModuleType,
    emit: Callable[[str], None],
) -> int:
    """Adjudicate every declared row and return the four-state gate code.

    WHY one function callable with any rows list and any emit target: the
    real run and the self-test must exercise the SAME derivation. A
    self-test that re-implements the gate proves nothing about the gate.
    The denominator is always len(rows) -- every row the matrix declares,
    whether or not it has receipts -- and it is stated in the summary so a
    reader can see what 'measured' was measured out of."""
    if not rows:
        emit(
            "REFUSED (96): the matrix declares 0 rows -- all([]) is True, but a matrix "
            "with no rows is not a clean matrix, it is not a matrix"
        )
        return REFUSE
    problems: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            problems.append(f"row {index} is a {type(row).__name__}, not an object")
            continue
        row_id = row.get("id")
        label = row_id if isinstance(row_id, str) else f"#{index}"
        if not isinstance(row_id, str) or not row_id.strip():
            problems.append(f"row #{index} carries no non-empty string 'id'")
        elif row_id in seen:
            problems.append(
                f"row id {row_id!r} is declared twice; receipts are addressed by row id, "
                "so a duplicate makes every receipt for it ambiguous"
            )
        else:
            seen.add(row_id)
        claim = row.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            problems.append(
                f"row {label}: no claim text; a row without a claim cannot be measured "
                "and must not count toward green"
            )
        adjudicator = row.get("adjudicator")
        if adjudicator is not None and (
            not isinstance(adjudicator, str) or not adjudicator.strip()
        ):
            problems.append(
                f"row {label}: 'adjudicator' must be a repo-relative path string or null"
            )
    if problems:
        for problem in problems:
            emit(f"REFUSED (96): the declaration is broken: {problem}")
        return REFUSE

    per_row: list[tuple[str, str, str]] = []
    for row in rows:
        verdict, reason = _adjudicate_row(
            row,
            receipts_root=receipts_root,
            repo_root=repo_root,
            receipts_mod=receipts_mod,
        )
        per_row.append((row["id"], verdict, reason))
        if verdict != "green":
            emit(f"{row['id']} {_VERDICT_WORD[verdict]} ({_VERDICT_EXIT[verdict]}): {reason}")

    counts = {word: 0 for word in _VERDICT_WORD}
    for _, verdict, _ in per_row:
        counts[verdict] += 1
    code = _aggregate_code([verdict for _, verdict, _ in per_row])
    total = len(rows)
    emit(
        f"MATRIX SUMMARY: {counts['green']}/{total} rows green "
        f"({counts['red']} red, {counts['unmeasured']} unmeasured, {counts['refused']} refused); "
        f"the denominator is all {total} declared rows, re-derived from the receipts tree "
        f"on every run and stored nowhere; matrix verdict {_word_for_code(code)} (exit {code})"
    )
    return code


def _word_for_code(code: int) -> str:
    return {
        CLEAR: "CLEAR",
        RED: "RED",
        UNMEASURED: "UNMEASURED",
        REFUSE: "REFUSED",
    }[code]


def _build_parser() -> argparse.ArgumentParser:
    p = _ContractParser(
        prog="verification_matrix_results",
        description=(
            "Adjudicate the whole verification matrix from its per-pass receipts. "
            "Exit codes: 0 CLEAR, 5 RED, 95 UNMEASURED, 96 REFUSE."
        ),
    )
    p.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="path to matrix.json (default: "
        "<repo>/validation_campaigns/verification_matrix/matrix.json)",
    )
    p.add_argument(
        "--receipts-root",
        type=Path,
        default=None,
        help="receipts tree root (default: "
        "<repo>/validation_campaigns/verification_matrix/receipts)",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repository root for resolving adjudicator paths (default: parent of checks/)",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="run the MUST-FIRE controls against temporary fixtures; never reads the real matrix",
    )
    return p


def _self_test(receipts_mod: ModuleType) -> int:
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    def control(name: str, fn: Callable[[], tuple[bool, str]]) -> None:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 -- a control that raises IS a failure
            ok, detail = False, f"control raised {type(exc).__name__}: {exc}"
        record(name, bool(ok), detail)

    with tempfile.TemporaryDirectory(prefix="vm_gate_selftest_") as tmp:
        base = Path(tmp)
        repo = base / "repo"
        (repo / "checks").mkdir(parents=True)
        receipts_root = base / "receipts"
        receipts_root.mkdir(parents=True)
        adj_rel = "checks/fake_adjudicator.py"
        adj_bytes = b"# fake adjudicator owned by the self-test\nDECISION = 'synthetic'\n"
        (repo / adj_rel).write_bytes(adj_bytes)

        def make_row(
            row_id: str, *, claim: str | None = None, adjudicator: str | None = adj_rel
        ) -> dict[str, Any]:
            return {
                "id": row_id,
                "tier": 0,
                "group": "self-test",
                "claim": claim if claim is not None else f"self-test claim for {row_id}",
                "run_arm": "synthetic run arm",
                "control_arm": "synthetic control arm",
                "cost": "0 GPU",
                "status": "fixture",
                "blocked_by": [],
                "findings": [],
                "adjudicator": adjudicator,
            }

        def run_gate(rows: list[Any]) -> tuple[int, list[str]]:
            lines: list[str] = []
            code = adjudicate_matrix(
                rows=rows,
                receipts_root=receipts_root,
                repo_root=repo,
                receipts_mod=receipts_mod,
                emit=lines.append,
            )
            return code, lines

        def line_for(lines: list[str], row_id: str) -> str:
            for line in lines:
                if line.startswith(row_id + " "):
                    return line
            return ""

        def write_pass(
            row: dict[str, Any],
            pass_id: str,
            exit_code: int,
            reason: str | None,
            *,
            path: str = adj_rel,
            data: bytes = adj_bytes,
            stamp: str = "2026-01-02T00:00:00+00:00",
        ) -> None:
            receipts_mod.write_receipt(
                receipts_root=receipts_root,
                row=row,
                pass_id=pass_id,
                exit_code=exit_code,
                reason=reason,
                arms={"run": {"loss": 2.1}, "control": {"refused": True}},
                adjudicator_path=path,
                adjudicator_bytes=data,
                written_at_utc=stamp,
            )

        def c1() -> tuple[bool, str]:
            row = make_row("S-1")
            code, lines = run_gate([row])
            line = line_for(lines, "S-1")
            ok = (
                code == UNMEASURED
                and line.startswith("S-1 UNMEASURED")
                and "never been measured" in line
                and not any(ln.startswith("S-1 GREEN") or ln.startswith("S-1 RED") for ln in lines)
            )
            return ok, f"exit={code} observed: {line[:90]}"

        def c2() -> tuple[bool, str]:
            row = make_row("S-2")
            write_pass(
                row,
                "gb200-p1",
                RED,
                "control rule refuted the claim: identical loss curves across arms",
            )
            code, lines = run_gate([row])
            line = line_for(lines, "S-2")
            ok = code == RED and line.startswith("S-2 RED")
            return ok, f"exit={code} observed: {line[:90]}"

        def c3() -> tuple[bool, str]:
            row = make_row("S-3")
            write_pass(row, "gb200-p1", CLEAR, None)
            edited = dict(row)
            edited["claim"] = row["claim"] + " -- edited after the measurement was taken"
            code, lines = run_gate([edited])
            line = line_for(lines, "S-3")
            ok = code == UNMEASURED and "UNMEASURED" in line and "claim drift" in line
            return ok, f"exit={code} drift named: {'claim drift' in line}; {line[:80]}"

        def c4() -> tuple[bool, str]:
            adj4 = "checks/fake_adjudicator_c4.py"
            before = b"# c4 adjudicator, first version\n"
            (repo / adj4).write_bytes(before)
            row = make_row("S-4", adjudicator=adj4)
            write_pass(row, "gb200-p1", CLEAR, None, path=adj4, data=before)
            (repo / adj4).write_bytes(b"# c4 adjudicator, edited after the run\n")
            code, lines = run_gate([row])
            line = line_for(lines, "S-4")
            ok = code == UNMEASURED and "UNMEASURED" in line and "adjudicator drift" in line
            return ok, f"exit={code} drift named: {'adjudicator drift' in line}; {line[:80]}"

        def c5() -> tuple[bool, str]:
            missing = "checks/the_adjudicator_that_was_never_written.py"
            row = make_row("S-5", adjudicator=missing)
            code, lines = run_gate([row])
            line = line_for(lines, "S-5")
            ok = (
                code == REFUSE
                and line.startswith("S-5 REFUSED")
                and missing in line
                and "not a red" in line
            )
            return ok, f"exit={code} observed: {line[:90]}"

        def c6() -> tuple[bool, str]:
            code, lines = run_gate([])
            first = lines[0] if lines else ""
            ok = code == REFUSE and "not a matrix" in first
            return ok, f"exit={code} observed: {first[:90]}"

        def c7() -> tuple[bool, str]:
            two = [make_row("S-7a"), make_row("S-7b")]
            _, lines2 = run_gate(two)
            _, lines3 = run_gate(two + [make_row("S-7c")])
            summary2 = lines2[-1] if lines2 else ""
            summary3 = lines3[-1] if lines3 else ""
            m2 = re.search(r"all (\d+) declared rows", summary2)
            m3 = re.search(r"all (\d+) declared rows", summary3)
            ok = (
                m2 is not None
                and m3 is not None
                and int(m2.group(1)) == 2
                and int(m3.group(1)) == 3
                and "/2" in summary2
                and "/3" in summary3
            )
            d2 = m2.group(1) if m2 else "?"
            d3 = m3.group(1) if m3 else "?"
            return ok, f"denominators observed: {d2} rows, then {d3} after one row added"

        def c8() -> tuple[bool, str]:
            row_bad = make_row("S-8")
            bad_dir = receipts_root / "S-8"
            bad_dir.mkdir(parents=True)
            (bad_dir / "broken.json").write_text("{ this is not json", encoding="utf-8")
            code1, lines1 = run_gate([row_bad])
            line1 = line_for(lines1, "S-8")
            ok1 = code1 == REFUSE and "broken.json" in line1 and line1.startswith("S-8 REFUSED")
            row_keys = make_row("S-8b")
            keys_dir = receipts_root / "S-8b"
            keys_dir.mkdir(parents=True)
            (keys_dir / "missing_keys.json").write_text(
                json.dumps({"row_id": "S-8b", "pass_id": "missing_keys", "verdict": "green"}),
                encoding="utf-8",
            )
            code2, lines2 = run_gate([row_keys])
            line2 = line_for(lines2, "S-8b")
            ok2 = (
                code2 == REFUSE
                and "missing_keys.json" in line2
                and line2.startswith("S-8b REFUSED")
            )
            return ok1 and ok2, (
                f"bad JSON -> exit={code1} names file: {'broken.json' in line1}; "
                f"missing keys -> exit={code2} names file: {'missing_keys.json' in line2}"
            )

        control("C1 a row with no receipts reads UNMEASURED, never green, never red", c1)
        control("C2 MUST-FIRE: a receipt recording exit 5 takes the whole gate to 5", c2)
        control(
            "C3 MUST-FIRE: editing the claim after the fact makes its receipt read "
            "UNMEASURED, and the reason NAMES the claim drift",
            c3,
        )
        control(
            "C4 MUST-FIRE: changing the adjudicator's bytes makes its receipt read "
            "UNMEASURED, naming the adjudicator drift",
            c4,
        )
        control("C5 a row whose 'adjudicator' path does not exist REFUSES 96, not a red", c5)
        control(
            "C6 an empty rows list REFUSES 96 -- all([]) is True, and zero rows is not a matrix",
            c6,
        )
        control("C7 the summary denominator equals len(rows) and moves when a row is added", c7)
        control(
            "C8 a malformed receipt REFUSES 96 naming the file -- it never silently "
            "drops out of the denominator",
            c8,
        )

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    failed = [name for name, ok, _ in checks if not ok]
    print(
        f"verification-matrix gate self-test: "
        f"{len(checks) - len(failed)}/{len(checks)} controls PASS"
    )
    if failed:
        for name in failed:
            print(f"FAIL: {name}")
        return RED
    return CLEAR


def main(argv: list[str]) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # error() raises SystemExit(96), --help SystemExit(0); both become
        # returned codes so nothing escapes main().
        return exc.code if isinstance(exc.code, int) else REFUSE

    repo_root = (
        args.repo_root if args.repo_root is not None else Path(__file__).resolve().parents[1]
    )
    matrix_path = args.matrix if args.matrix is not None else repo_root / MATRIX_REL
    receipts_root = (
        args.receipts_root if args.receipts_root is not None else repo_root / RECEIPTS_REL
    )

    try:
        receipts_mod = _load_receipts_module(repo_root)
        if args.self_test:
            return _self_test(receipts_mod)
        rows = _load_rows(matrix_path)
        return adjudicate_matrix(
            rows=rows,
            receipts_root=receipts_root,
            repo_root=repo_root,
            receipts_mod=receipts_mod,
            emit=print,
        )
    except _Refusal as exc:
        print(f"REFUSED (96): {exc}")
        return REFUSE
    except Exception as exc:  # noqa: BLE001 -- the contract has no code for "crashed"
        traceback.print_exc()
        print(
            f"REFUSED (96): an unexpected {type(exc).__name__} escaped the gate: {exc} -- "
            "the aggregate was NOT derived, so nothing is claimed"
        )
        return REFUSE


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
