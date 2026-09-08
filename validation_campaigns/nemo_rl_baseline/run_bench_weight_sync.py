"""Committed bench caller that wires adjudicate_bench_exit to a real launch boundary.

The sibling adjudicator maps a record's top-level verdict to an exit code
(0 CLEAR / 5 RED / 95 UNMEASURED / 96 REFUSE) but is orphan code and openly
declares record freshness out of scope.  This wrapper fixes both: it spawns
the benchmark launcher, applies the staleness rule at the only place where
the launch boundary is known, and then lets a fresh record's verdict govern
the process exit code.  The launcher's own return code never picks the code.

WHAT IS CLAIMED:
- The record is adjudicated only when FRESH: it exists after the launcher
  returns AND (it did not exist before the launch OR its mtime_ns is strictly
  greater than the mtime_ns recorded before the launch).  Equal mtimes are
  not fresh; "not proven fresh" is the honest reading.
- A fresh record's verdict governs via the adjudicator, even if the launcher
  exited nonzero or hit the timeout (synthetic rc 124).
- A stale, absent, or spawn-failed (synthetic rc 127) launch REFUSES with
  code 96 and never reports a stale verdict.  Synthetic rc values are named
  synthetic because they did not come from the child.

WHAT IS NOT CLAIMED:
- The record's schema is validated only to the extent the adjudicator does.
- This does not prove the record describes THIS launch's configuration, only
  that something rewrote it after this launch began.
- mtime granularity bounds the freshness claim on filesystems with coarse
  timestamps.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    from adjudicate_bench_exit import CODE_REFUSE, BenchAdjudication, adjudicate
except ImportError:  # running as a package submodule
    from .adjudicate_bench_exit import CODE_REFUSE, BenchAdjudication, adjudicate

__all__ = ("BenchConfig", "BenchOutcome", "execute", "main")

TIMEOUT_RC = 124  # synthetic: the child was killed by us, never received from it
SPAWN_FAILURE_RC = 127  # synthetic: the child never ran

_OLD_NS = 1_000_000_000  # fixed ancient mtime for self-test staleness controls
_NEW_NS = 2_000_000_000  # strictly later; bumped with os.utime, never by sleeping

_SRC_NOP = "import sys; sys.exit(0)"
_SRC_WRITE_CLEAR = (
    "import json, sys; from pathlib import Path; "
    "Path(sys.argv[1]).write_text(json.dumps({'verdict': 'CLEAR'})); sys.exit(1)"
)
_SRC_WRITE_UNMEASURED = (
    "import json, sys; from pathlib import Path; "
    "Path(sys.argv[1]).write_text(json.dumps({'verdict': 'UNMEASURED'})); sys.exit(1)"
)
_SRC_REWRITE_RED = (
    "import json, os, sys; from pathlib import Path; "
    "p = Path(sys.argv[1]); p.write_text(json.dumps({'verdict': 'RED'})); "
    f"os.utime(p, ns=({_NEW_NS}, {_NEW_NS})); sys.exit(1)"
)


@dataclass(frozen=True, slots=True)
class BenchConfig:
    """Inputs to one guarded benchmark launch."""

    record: Path
    launcher_argv: tuple[str, ...]
    timeout: float | None = None


@dataclass(frozen=True, slots=True)
class BenchOutcome:
    """Result of one launch: the governing adjudication plus freshness provenance."""

    adjudication: BenchAdjudication
    launcher_rc: int
    timed_out: bool
    record_fresh: bool
    baseline_ns: int | None
    after_ns: int | None
    code: int


def _mtime_ns(path: Path) -> int | None:
    """Return st_mtime_ns, or None when the record cannot be statted."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _stale_message(record: Path, baseline_ns: int | None, after_ns: int | None) -> str:
    """Build the refusal message naming the path and both observed mtimes."""
    before = "absent before launch" if baseline_ns is None else f"mtime_ns before={baseline_ns}"
    after = "absent after launch" if after_ns is None else f"mtime_ns after={after_ns}"
    return (
        f"REFUSING: record {record} is not fresh for this launch "
        f"({before}; {after}); a stale record cannot speak for this launch."
    )


def _spawn_launcher(config: BenchConfig) -> tuple[int, bool, list[str]]:
    """Run the launcher; return (rc, timed_out, notes) with synthetic rc on failure."""
    try:
        proc = subprocess.run(list(config.launcher_argv), timeout=config.timeout, check=False)
    except subprocess.TimeoutExpired:
        note = f"launcher timed out after {config.timeout}s; rc {TIMEOUT_RC} is synthetic"
        return TIMEOUT_RC, True, [f"{note} (not from the child)"]
    except OSError as exc:
        note = f"spawn failed: {exc}; rc {SPAWN_FAILURE_RC} is synthetic"
        return SPAWN_FAILURE_RC, False, [f"{note} (not from the child)"]
    return proc.returncode, False, []


def execute(config: BenchConfig) -> BenchOutcome:
    """Spawn the launcher, apply the staleness rule, and adjudicate only fresh records."""
    baseline_ns = _mtime_ns(config.record)
    launcher_rc, timed_out, notes = _spawn_launcher(config)
    after_ns = _mtime_ns(config.record)
    fresh = after_ns is not None and (baseline_ns is None or after_ns > baseline_ns)
    if fresh:
        adjudication = adjudicate(config.record, launcher_rc)
        if notes:
            adjudication = BenchAdjudication(
                code=adjudication.code,
                verdict=adjudication.verdict,
                launcher_rc=adjudication.launcher_rc,
                message=f"{adjudication.message} Note: {'; '.join(notes)}.",
            )
    else:
        message = _stale_message(config.record, baseline_ns, after_ns)
        if notes:
            message += f" Note: {'; '.join(notes)}."
        adjudication = BenchAdjudication(
            code=CODE_REFUSE, verdict=None, launcher_rc=launcher_rc, message=message
        )
    return BenchOutcome(
        adjudication=adjudication,
        launcher_rc=launcher_rc,
        timed_out=timed_out,
        record_fresh=fresh,
        baseline_ns=baseline_ns,
        after_ns=after_ns,
        code=adjudication.code,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI args; everything after `--` is the launcher argv."""
    parser = argparse.ArgumentParser(description="Guarded bench launcher with staleness rule.")
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("launcher", nargs=argparse.REMAINDER)
    ns = parser.parse_args(argv)
    if ns.launcher and ns.launcher[0] == "--":
        ns.launcher = ns.launcher[1:]
    return ns


def _check(
    name: str,
    outcome: BenchOutcome,
    expect_code: int,
    expect_fresh: bool,
    expect_rc: int | None = None,
) -> bool:
    """Score one self-test control and print its PASS/FAIL line."""
    ok = outcome.code == expect_code and outcome.record_fresh == expect_fresh
    if expect_rc is not None:
        ok = ok and outcome.launcher_rc == expect_rc
    status = "PASS" if ok else "FAIL"
    print(
        f"{status} {name}: expected code={expect_code} fresh={expect_fresh}; "
        f"observed code={outcome.code} fresh={outcome.record_fresh} rc={outcome.launcher_rc}"
    )
    return ok


def _run_self_test() -> int:
    """Run the six controls in a TemporaryDirectory; 0 if all pass, else 5."""
    passed = 0
    total = 6
    with tempfile.TemporaryDirectory() as tmp:
        root, py = Path(tmp), sys.executable
        # 1: fresh CLEAR governs; launcher rc 1 does not.
        record = root / "c1.json"
        out = execute(BenchConfig(record, (py, "-c", _SRC_WRITE_CLEAR, str(record))))
        passed += _check("fresh CLEAR beats launcher rc=1", out, 0, True)
        # 2: fresh UNMEASURED survives the process boundary.
        record = root / "c2.json"
        out = execute(BenchConfig(record, (py, "-c", _SRC_WRITE_UNMEASURED, str(record))))
        passed += _check("fresh UNMEASURED yields 95 over rc=1", out, 95, True)
        # 3: pre-existing CLEAR untouched by the launcher is stale -> refuse.
        record = root / "c3.json"
        record.write_text('{"verdict": "CLEAR"}')
        os.utime(record, ns=(_OLD_NS, _OLD_NS))
        out = execute(BenchConfig(record, (py, "-c", _SRC_NOP)))
        passed += _check("stale record refused", out, 96, False, expect_rc=0)
        # 4: positive control: rewritten with strictly later mtime -> fresh RED.
        record = root / "c4.json"
        record.write_text('{"verdict": "CLEAR"}')
        os.utime(record, ns=(_OLD_NS, _OLD_NS))
        out = execute(BenchConfig(record, (py, "-c", _SRC_REWRITE_RED, str(record))))
        passed += _check("rewritten record adjudicated", out, 5, True)
        # 5: spawn failure -> synthetic rc 127, refusal.
        out = execute(BenchConfig(root / "c5.json", (str(root / "no-such-binary"),)))
        passed += _check("spawn failure refused", out, 96, False, expect_rc=SPAWN_FAILURE_RC)
        # 6: launcher exits 0 without writing; absent record -> refusal.
        out = execute(BenchConfig(root / "c6.json", (py, "-c", _SRC_NOP)))
        passed += _check("absent record after rc=0 refused", out, 96, False, expect_rc=0)
    print(f"{passed} of {total} controls passed")
    return 0 if passed == total else 5


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the governing exit code."""
    ns = _parse_args(argv)
    if ns.self_test:
        return _run_self_test()
    if ns.record is None:
        print("REFUSING (96): missing required field --record (or use --self-test)")
        return CODE_REFUSE
    if not ns.launcher:
        print("REFUSING (96): missing required launcher argv after '--'")
        return CODE_REFUSE
    config = BenchConfig(record=ns.record, launcher_argv=tuple(ns.launcher), timeout=ns.timeout)
    outcome = execute(config)
    print(outcome.adjudication.message)
    print(
        f"code={outcome.code} verdict={outcome.adjudication.verdict} "
        f"launcher_rc={outcome.launcher_rc} timed_out={outcome.timed_out} "
        f"record_fresh={outcome.record_fresh}"
    )
    return outcome.code


if __name__ == "__main__":
    sys.exit(main())
