"""Adjudicate a torchrun launcher exit code against the benchmark's own JSON verdict record.

torchrun does not propagate the child process's declared exit code faithfully: a child exiting
95 (UNMEASURED per the harness contract) surfaces at the shell as launcher rc 1, which belongs
to no declared namespace, leaving UNMEASURED indistinguishable from RED or a launcher crash.
This module restores the four-state contract at the process boundary by reading the JSON record
the benchmark wrote and letting its top-level ``verdict`` string pick the emitted code. The raw
``launcher_rc`` is recorded in the result and named in every message, but it is NEVER consulted
when choosing the code -- restoring that authority is the entire point of this file.

WHAT IS CLAIMED:
- The record's ``verdict`` string maps CLEAR -> 0, CLEAR_WITH_ABSTENTIONS -> 0, RED -> 5,
  UNMEASURED -> 95, REFUSE -> 96, and the emitted code always comes from that map alone.
- Absent, unreadable, non-object, or verdict-key-less records REFUSE (code 96, verdict None),
  with the message naming which of the four failure shapes occurred and the path involved.
- A verdict string outside the five known values REFUSES, naming the value seen and the
  accepted set; there is no silent default.
- A nonzero launcher_rc that disagrees with a 0-code verdict is surfaced verbatim in the
  message and never changes the code.
- ``--self-test`` plants every verdict plus negative controls inside a TemporaryDirectory and
  proves {0, 5, 95, 96} are all distinctly reachable; it returns 0 only if all controls pass.

WHAT IS NOT CLAIMED:
- Nothing beyond the presence and value of the top-level ``verdict`` key is validated; the
  rest of the record schema is out of scope for this adjudicator.
- No claim that the record is fresh, or that it was written by the same launch that produced
  launcher_rc. A stale record will adjudicate cleanly; provenance checks are out of scope.
- torchrun itself is not fixed. This module repairs the lost signal after the process
  boundary, it does not prevent torchrun from mangling the child's declared code.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

__all__ = ("BenchAdjudication", "adjudicate", "main")

CODE_CLEAR = 0
CODE_RED = 5
CODE_UNMEASURED = 95
CODE_REFUSE = 96

VERDICT_TO_CODE = {
    "CLEAR": CODE_CLEAR,
    "CLEAR_WITH_ABSTENTIONS": CODE_CLEAR,
    "RED": CODE_RED,
    "UNMEASURED": CODE_UNMEASURED,
    "REFUSE": CODE_REFUSE,
}

VERDICT_ORDER = ("CLEAR", "CLEAR_WITH_ABSTENTIONS", "RED", "UNMEASURED", "REFUSE")


@dataclass(frozen=True, slots=True)
class BenchAdjudication:
    """The verdict-governed decision for one torchrun launch.

    ``code`` is chosen solely from the record's verdict string. ``launcher_rc`` is the raw
    process-boundary return code, kept for the audit trail and echoed in ``message``; it is
    never an input to the decision. ``verdict`` is None exactly when the launch is REFUSED
    because there was no trustworthy verdict to read.
    """

    code: int
    verdict: str | None
    launcher_rc: int
    message: str


def _refuse(launcher_rc: int, message: str) -> BenchAdjudication:
    return BenchAdjudication(
        code=CODE_REFUSE, verdict=None, launcher_rc=launcher_rc, message=message
    )


def adjudicate(record_path: Path, launcher_rc: int) -> BenchAdjudication:
    """Pick the harness contract code from the record's verdict, never from ``launcher_rc``."""
    if not record_path.exists():
        return _refuse(
            launcher_rc,
            f"record absent at {record_path}; without the record there is no verdict to "
            "adjudicate, and that is a refusal, not a pass and not a RED",
        )
    try:
        raw = record_path.read_bytes()
    except OSError as exc:
        return _refuse(
            launcher_rc,
            f"record unreadable at {record_path}: I/O error ({exc}); no verdict to adjudicate",
        )
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _refuse(
            launcher_rc,
            f"record unreadable at {record_path}: not valid UTF-8 JSON ({exc}); "
            "no verdict to adjudicate",
        )
    if not isinstance(record, dict):
        return _refuse(
            launcher_rc,
            f"record at {record_path} is not a JSON object (got {type(record).__name__}); "
            "the field 'verdict' is required at the top level of an object",
        )
    if "verdict" not in record:
        return _refuse(
            launcher_rc,
            f"record at {record_path} is missing the required field 'verdict' "
            f"(expected 1 verdict key, found 0 among {len(record)} top-level keys: "
            f"{sorted(record)}); refusing rather than guessing",
        )
    verdict = record["verdict"]
    if not isinstance(verdict, str) or verdict not in VERDICT_TO_CODE:
        return _refuse(
            launcher_rc,
            f"field 'verdict' at {record_path} has unknown value {verdict!r}; accepted set is "
            f"{sorted(VERDICT_TO_CODE)} ({len(VERDICT_TO_CODE)} known values, 1 seen value "
            "outside the set); no default fallback is applied",
        )
    code = VERDICT_TO_CODE[verdict]
    if launcher_rc != 0 and code == CODE_CLEAR:
        message = (
            f"launcher rc={launcher_rc} disagrees with verdict {verdict}; the verdict governs "
            "because torchrun does not propagate the child's declared code -> code 0"
        )
    else:
        message = (
            f"verdict {verdict} governs -> code {code}; launcher rc={launcher_rc} recorded "
            "but never used to pick the code"
        )
    return BenchAdjudication(code=code, verdict=verdict, launcher_rc=launcher_rc, message=message)


def _run_self_test() -> int:
    controls: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        controls.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'} [{len(controls):02d}] {name}: {detail}")

    with tempfile.TemporaryDirectory(prefix="adjudicate-bench-exit-") as tmp:
        root = Path(tmp)
        observed: set[int] = set()
        for verdict in VERDICT_ORDER:
            path = root / f"record-{verdict}.json"
            path.write_text(json.dumps({"verdict": verdict}), encoding="utf-8")
            result = adjudicate(path, launcher_rc=1)
            observed.add(result.code)
            check(
                f"planted verdict {verdict}",
                result.code == VERDICT_TO_CODE[verdict] and result.verdict == verdict,
                f"expected code {VERDICT_TO_CODE[verdict]}, got {result.code} "
                "(launcher_rc=1 deliberately in play and ignored)",
            )
        check(
            "codes {0, 5, 95, 96} are all distinctly reachable",
            observed == {CODE_CLEAR, CODE_RED, CODE_UNMEASURED, CODE_REFUSE},
            f"observed code set {sorted(observed)} has {len(observed)} members, expected 4",
        )
        result = adjudicate(root / "missing.json", launcher_rc=0)
        check(
            "absent record refuses",
            result.code == CODE_REFUSE and result.verdict is None and "absent" in result.message,
            f"code={result.code} message={result.message}",
        )
        garbage = root / "garbage.json"
        garbage.write_bytes(b"\x00\xff not json at all")
        result = adjudicate(garbage, launcher_rc=0)
        check(
            "non-JSON bytes refuse",
            result.code == CODE_REFUSE
            and result.verdict is None
            and "unreadable" in result.message,
            f"code={result.code} message={result.message}",
        )
        array = root / "array.json"
        array.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        result = adjudicate(array, launcher_rc=0)
        check(
            "JSON array instead of object refuses",
            result.code == CODE_REFUSE
            and result.verdict is None
            and "not a JSON object" in result.message,
            f"code={result.code} message={result.message}",
        )
        no_verdict = root / "no-verdict.json"
        no_verdict.write_text(json.dumps({"note": "nothing here"}), encoding="utf-8")
        result = adjudicate(no_verdict, launcher_rc=0)
        check(
            "object with no verdict key refuses",
            result.code == CODE_REFUSE and result.verdict is None and "'verdict'" in result.message,
            f"code={result.code} message={result.message}",
        )
        unknown = root / "unknown.json"
        unknown.write_text(json.dumps({"verdict": "GREEN"}), encoding="utf-8")
        result = adjudicate(unknown, launcher_rc=0)
        check(
            "unknown verdict string refuses",
            result.code == CODE_REFUSE
            and result.verdict is None
            and "'GREEN'" in result.message
            and "UNMEASURED" in result.message,
            f"code={result.code} message={result.message}",
        )
        clear_rc1 = root / "clear-rc1.json"
        clear_rc1.write_text(json.dumps({"verdict": "CLEAR"}), encoding="utf-8")
        result = adjudicate(clear_rc1, launcher_rc=1)
        check(
            "nonzero launcher_rc with verdict CLEAR still yields 0",
            result.code == CODE_CLEAR
            and "disagrees" in result.message
            and "rc=1" in result.message,
            f"code={result.code} message={result.message}",
        )
    passed = sum(1 for _, ok, _ in controls if ok)
    print(f"{passed} of {len(controls)} controls passed")
    return CODE_CLEAR if passed == len(controls) else CODE_RED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adjudicate a torchrun launcher's exit code against the benchmark record."
    )
    parser.add_argument("--record", type=Path, default=None, help="path to result.json")
    parser.add_argument("--launcher-rc", type=int, default=None, help="torchrun exit status")
    parser.add_argument(
        "--self-test", action="store_true", help="run planted controls instead of adjudicating"
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return _run_self_test()
    if args.record is None or args.launcher_rc is None:
        print(
            "both --record and --launcher-rc are required "
            f"(record={args.record}, launcher_rc={args.launcher_rc}); "
            "refusing to adjudicate with an incomplete invocation"
        )
        return CODE_REFUSE
    result = adjudicate(args.record, args.launcher_rc)
    print(
        f"verdict={result.verdict} launcher_rc={result.launcher_rc} "
        f"code={result.code} :: {result.message}"
    )
    return result.code


if __name__ == "__main__":
    sys.exit(main())
