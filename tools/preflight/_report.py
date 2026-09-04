"""Report rendering and banner."""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ._base import (
    TOOL_VERSION,
    Verdict,
)
from ._core import (
    CheckResult,
    _is_clear,
)
from ._errors import (
    ToolError,
)

# ---------------------------------------------------------------------------
# Report rendering + banner
# ---------------------------------------------------------------------------


def _render_report(
    cfg: dict[str, Any],
    results: Sequence[CheckResult],
    manifest_sha: str | None,
    out: Callable[[str], None],
) -> bool:
    n_pass = sum(1 for r in results if r.verdict is Verdict.PASS)
    clear = _is_clear(results)
    out(f"preflight @ {cfg.get('run_name', '?')}: {len(results)} checks run")
    for r in results:
        out("  " + r.render())
    # The denominator summary line: no reader tallies the column by hand.
    total_units = sum(r.coverage.checked for r in results)
    out(
        f"  — {n_pass}/{len(results)} checks PASS; {total_units} units examined across "
        f"{len(results)} checks"
    )
    if clear:
        smoke = bool(cfg.get("schedule", {}).get("smoke"))
        out("")
        out("=== FOUNDATIONSCALE PRE-FLIGHT BANNER ===")
        out(f"run: {cfg['run_name']}")
        out(f"manifest_sha256: {manifest_sha}")
        out(f"checks: {n_pass}/{len(results)} PASS")
        if smoke:
            # Design item 7, second sentence, made unrejectable: the ONLY
            # clearance this tool can emit for a smoke-tagged config carries
            # the qualifier inside the banner itself.
            out("clearance: CLEAR (SMOKE — this banner makes NO training-correctness claim)")
        else:
            out("clearance: CLEAR")
        out("# checkpoint writers: embed manifest_sha256 into every provenance record;")
        out("# resume must refuse any checkpoint naming a different hash (item 10).")
    else:
        blocking = [r for r in results if r.verdict is not Verdict.PASS]
        out(
            f"overall: BLOCKED — {len(blocking)} check(s) NOT-VERIFIED; a launch cleared "
            f"over them would be all([]) with extra steps"
        )
        if manifest_sha:
            out(f"(frozen manifest hash for reference: {manifest_sha})")
    return clear


def _write_json_record(
    path: str,
    cfg: dict,
    cfg_sha: str,
    results: Sequence[CheckResult],
    manifest_sha: str | None,
    clear: bool,
) -> None:
    record = {
        "tool": "foundationscale-preflight",
        "tool_version": TOOL_VERSION,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "run_name": cfg.get("run_name"),
        "config_sha256": cfg_sha,
        "overall": "CLEAR" if clear else "BLOCKED",
        "manifest_sha256": manifest_sha,
        "checks_run": len(results),
        "checks_passing": sum(1 for r in results if r.verdict is Verdict.PASS),
        "units_examined": sum(r.coverage.checked for r in results),
        "results": [r.to_dict() for r in results],
        "clearance_rule": "all checks PASS with checked > 0 and at least one check ran; "
        "SKIP/VACUOUS/INAPPLICABLE are NOT-VERIFIED and block (design item 4)",
    }
    try:
        Path(path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"could not write --json record to {path}: {exc}") from exc
