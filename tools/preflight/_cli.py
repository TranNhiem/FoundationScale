"""CLI entry point."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Sequence
from typing import Any

from ._base import (
    EXIT_BLOCKED,
    EXIT_CLEAR,
    EXIT_TOOL_ERROR,
)
from ._config import (
    _load_config,
)
from ._core import (
    _REGISTRY_ORDER,
    REGISTRY,
    CheckResult,
    _execute,
)
from ._errors import (
    ConfigError,
    ToolError,
)
from ._report import (
    _render_report,
    _write_json_record,
)
from ._selftest import (
    run_self_test,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="foundationscale-preflight",
        description="Executable pre-flight blocklist for the E4B launch (design: risk-review §4).",
    )
    parser.add_argument(
        "--config", help="JSON config describing the run being gated (required unless --self-test)"
    )
    parser.add_argument("--json", dest="json_path", help="write the machine record to this path")
    parser.add_argument("--only", help="comma-separated check ids to run")
    parser.add_argument("--exclude", default="", help="comma-separated check ids to leave out")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="prove every check flips on synthesized corrupt artifacts and passes "
        "on a healthy world (design item 4's red-team drill, executable)",
    )
    parser.add_argument(
        "--list", action="store_true", help="list registered checks (diagnostic; NOT a clearance)"
    )
    args = parser.parse_args(argv)

    # Zero-check refusal, registry layer (doctrine 1 aimed at the tool itself).
    if not REGISTRY:
        print("preflight: BLOCKED — 0 checks are registered; a sweep over nothing proves nothing")
        return EXIT_BLOCKED

    if args.list:
        for chk in _REGISTRY_ORDER:
            print(f"  {chk.id} — {chk.title} [{len(chk.lanes)} MUST_FIRE lane(s)]")
        print("(diagnostic listing only — this output clears nothing)")
        return EXIT_CLEAR

    if args.self_test:
        try:
            code, _report = run_self_test()
        except Exception as exc:  # noqa: BLE001
            print(
                f"preflight: self-test could not run: {type(exc).__name__}: {exc}", file=sys.stderr
            )
            return EXIT_TOOL_ERROR
        return code

    if not args.config:
        print(
            "preflight: --config is required (fail closed: with no config there are no pins, "
            "and a pinless sweep is the vacuous case)",
            file=sys.stderr,
        )
        return EXIT_TOOL_ERROR

    try:
        cfg, cfg_sha = _load_config(args.config)
    except ToolError as exc:
        print(f"preflight: {exc}", file=sys.stderr)
        return EXIT_TOOL_ERROR
    except ConfigError as exc:
        print(
            f"preflight: BLOCKED — config names no launchable run "
            f"({exc.problems.__len__()} problem(s)):"
        )
        for p in exc.problems:
            print(f"  - {p}")
        print(
            "0 checks examined — the configuration layer refused before any artifact was touched."
        )
        return EXIT_BLOCKED

    # ---- selection, with the run_event discipline -----------------------------
    wanted = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None
    excluded = {x.strip() for x in args.exclude.split(",") if x.strip()}
    known = set(REGISTRY)
    unknown = ((wanted - known) if wanted else set()) | (excluded - known)
    if unknown:
        # A typo'd check id must read as a block, naming what was asked for and
        # what exists — silently dropping it would report "all selected checks
        # clear" over fewer checks than were asked for.
        print(f"preflight: BLOCKED — selection names unknown check id(s): {sorted(unknown)}")
        print(f"registered checks: {sorted(known)}")
        print("0 checks examined — selection refused.")
        return EXIT_BLOCKED
    selected = [
        c for c in _REGISTRY_ORDER if (wanted is None or c.id in wanted) and c.id not in excluded
    ]
    if not selected:
        print(
            f"preflight: BLOCKED — the selection ran 0 of {len(REGISTRY)} registered checks; "
            f"a sweep over nothing proves nothing. Broaden --only/--exclude."
        )
        return EXIT_BLOCKED

    shared: dict[str, Any] = {"_config_sha256": cfg_sha, "_run_name": cfg["run_name"]}
    results: list[CheckResult] = []
    try:
        for chk in selected:
            results.append(_execute(chk, cfg, dict(os.environ), shared))
    except Exception as exc:  # noqa: BLE001
        print(
            f"preflight: the tool itself failed mid-sweep: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc(limit=6)}",
            file=sys.stderr,
        )
        return EXIT_TOOL_ERROR

    manifest_sha = shared.get("manifest_sha256")
    clear = _render_report(cfg, results, manifest_sha, print)
    if args.json_path:
        try:
            _write_json_record(args.json_path, cfg, cfg_sha, results, manifest_sha, clear)
        except ToolError as exc:
            print(f"preflight: {exc}", file=sys.stderr)
            return EXIT_TOOL_ERROR
        print(f"machine record: {args.json_path}")
    return EXIT_CLEAR if clear else EXIT_BLOCKED
