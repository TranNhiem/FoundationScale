"""Item 10 -- launch provenance."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from .._artifacts import (
    _parse_iso,
)
from .._base import (
    Coverage,
    Verdict,
)
from .._core import (
    CheckResult,
    _Check,
    _finalize,
    _shared_or_error,
    _stub_fn,
)

# ---------------------------------------------------------------------------
# Item 10 — launch provenance
# ---------------------------------------------------------------------------


def _check_launch_provenance(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 10: checkpoints carry THIS preflight's manifest hash; resume is
    statically shown to refuse mismatch; walltime has a floor; every declared
    artifact's mtime lies inside the declared job window."""
    err = _shared_or_error(
        _Check("launch_provenance", "", None, _stub_fn), shared, ["manifest_sha256"]
    )
    if err:
        return err
    manifest_sha = shared["manifest_sha256"]
    evidence: dict[str, Any] = {"manifest_sha256": manifest_sha}

    # (a) embedding: each probe checkpoint must name this exact manifest.
    ck_reports = []
    ck_bad = []
    for d in s["checkpoint_dirs"]:
        prov = Path(d) / "provenance.json"
        try:
            rec = json.loads(prov.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            ck_bad.append(
                f"{d}: provenance.json unreadable ({exc}) — "
                f"nothing ties this checkpoint to any preflight"
            )
            continue
        got = rec.get("manifest_hash") if isinstance(rec, dict) else None
        ck_reports.append({"dir": d, "embedded_manifest_hash": got, "matches": got == manifest_sha})
        if got != manifest_sha:
            ck_bad.append(
                f"{d}: embedded hash {str(got)[:12]}… != this clearance's {manifest_sha[:12]}…"
            )
    evidence["checkpoints"] = ck_reports

    # (b) resume guard, shown statically. Declared as what it is: a text
    # sweep proving the resume path NAMES the hash AND has a raise/exit —
    # existence proof of the guard, not of its every branch (human confirms;
    # see 'does NOT close').
    guard_hits = {}
    guard_bad = []
    for path in s["resume_guard_files"]:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            guard_bad.append(f"{path}: unreadable ({exc})")
            continue
        names_hash = "manifest_hash" in text
        blocks = ("raise" in text) or ("sys.exit" in text) or ("ManifestMismatch" in text)
        guard_hits[path] = {"names_manifest_hash": names_hash, "has_refusal": blocks}
        if not (names_hash and blocks):
            guard_bad.append(f"{path} (names_manifest_hash={names_hash}, has_refusal={blocks})")
    evidence["resume_guards"] = guard_hits

    # (c) walltime floor from cumulative elapsed_s.
    try:
        wt_rows = [
            json.loads(ln)
            for ln in Path(s["walltime_jsonl"]).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "launch_provenance",
            "Launch provenance",
            Verdict.ERROR,
            Coverage(0, "provenance artifacts", expected=len(s["artifacts"])),
            f"walltime metrics unreadable: {s['walltime_jsonl']}: {exc}",
            evidence,
        )
    elapsed = [
        r.get("elapsed_s")
        for r in wt_rows
        if isinstance(r, dict) and isinstance(r.get("elapsed_s"), (int, float))
    ]
    wall = max(elapsed) if elapsed else None
    evidence["walltime_s"] = wall
    evidence["min_walltime_s"] = s["min_walltime_s"]
    if wall is None:
        return _finalize(
            "launch_provenance",
            "Launch provenance",
            Verdict.FAIL,
            Coverage(0, "provenance artifacts", expected=len(s["artifacts"])),
            f"0 elapsed_s records in {len(wt_rows)} walltime rows — no wallclock evidence exists",
            evidence,
        )
    if wall < float(s["min_walltime_s"]):
        return _finalize(
            "launch_provenance",
            "Launch provenance",
            Verdict.FAIL,
            Coverage(0, "provenance artifacts", expected=len(s["artifacts"])),
            f"walltime {wall}s < floor {s['min_walltime_s']}s — evidence produced 'too fast' "
            f"is evidence that was fabricated, replayed, or misattributed",
            evidence,
        )

    # (d) artifact mtimes ⊂ job window (with declared clock skew).
    start = _parse_iso(s["job_window_utc"][0])
    end = _parse_iso(s["job_window_utc"][1])
    assert start is not None and end is not None  # guaranteed by _post_validate
    slack = int(s["mtime_slack_s"])
    checked = 0
    outside = []
    for path in s["artifacts"]:
        p = Path(path)
        if not p.exists():
            outside.append(f"{path} (absent)")
            continue
        mtime = _dt.datetime.fromtimestamp(p.stat().st_mtime, _dt.timezone.utc)
        checked += 1
        if not (
            start - _dt.timedelta(seconds=slack) <= mtime <= end + _dt.timedelta(seconds=slack)
        ):
            outside.append(f"{path} (mtime {mtime.isoformat(timespec='seconds')} outside window)")
    evidence["artifacts_examined"] = checked
    evidence["job_window_utc"] = s["job_window_utc"]
    if outside:
        return _finalize(
            "launch_provenance",
            "Launch provenance",
            Verdict.FAIL,
            Coverage(checked, "provenance artifacts", expected=len(s["artifacts"])),
            f"{len(outside)} artifacts outside the job window: " + "; ".join(outside[:4]),
            evidence,
        )
    if ck_bad or guard_bad:
        reasons = []
        if ck_bad:
            reasons.append("checkpoint/manifest tie failed: " + "; ".join(ck_bad[:2]))
        if guard_bad:
            reasons.append("resume guard not shown: " + "; ".join(guard_bad[:2]))
        return _finalize(
            "launch_provenance",
            "Launch provenance",
            Verdict.FAIL,
            Coverage(checked, "provenance artifacts", expected=len(s["artifacts"])),
            " | ".join(reasons),
            evidence,
        )

    return _finalize(
        "launch_provenance",
        "Launch provenance",
        Verdict.PASS,
        Coverage(checked, "provenance artifacts", expected=len(s["artifacts"])),
        f"{len(ck_reports)} checkpoints embed this manifest hash; "
        f"{len(guard_hits)} resume guards shown; walltime {wall:.0f}s ≥ floor; "
        f"{checked}/{len(s['artifacts'])} artifact mtimes inside the job window",
        evidence,
    )
