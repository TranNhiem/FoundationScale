"""Item 4 -- verdict schema, operationalized as the launch-time red team."""

from __future__ import annotations

from typing import Any

from .._base import (
    Coverage,
    Verdict,
)
from .._core import (
    _REGISTRY_ORDER,
    CheckResult,
    _finalize,
)
from .._fixtures import (
    _run_lane_against,
)

# ---------------------------------------------------------------------------
# Item 4 — verdict schema, operationalized as the launch-time red team
# ---------------------------------------------------------------------------


def _check_verdict_schema(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 4: 'corrupt-artifact red-team dry run must flip gates before any launch.'

    Re-runs every peer check's MUST_FIRE lanes against fresh corrupt worlds
    in this process. A lane that does not flip FAILS this check and the
    launch. A peer declaring zero lanes also FAILS by name (--self-test
    covers the MUST_PASS half offline; this enforces MUST_FIRE at every
    launch).

    'Flipped' means verdict.blocking and not ERROR: an ERROR means the
    detector died on the corrupt input — a verifier exception, not a
    demonstrated firing (doctrine 4 binds detectors too).
    """
    peers = [
        c
        for c in (registry if registry is not None else _REGISTRY_ORDER)
        if c.id != "verdict_schema"
    ]
    total_lanes = sum(len(c.lanes) for c in peers)
    evidence: dict[str, Any] = {
        "peers_examined": len(peers),
        "lanes_total": total_lanes,
        "lane_results": [],
    }
    cov = Coverage(0, "red-team lanes", expected=total_lanes if total_lanes else None)
    if not peers:
        return _finalize(
            "verdict_schema",
            "Verdict schema / red team",
            Verdict.ERROR,
            cov,
            "no peer checks to red-team — a registry of one meta-check verifies nothing",
            evidence,
        )

    unflipped: list[str] = []
    laneless: list[str] = []
    checked = 0
    for peer in peers:
        if not peer.lanes:
            laneless.append(peer.id)
            continue
        for lane in peer.lanes:
            checked += 1
            outcome_verdict = "ERROR"
            note = ""
            try:
                res = _run_lane_against(peer, lane)
                outcome_verdict = res.verdict.value
            except Exception as exc:  # noqa: BLE001 — a broken red-team lane is unproven, not green
                note = f"lane harness raised {type(exc).__name__}: {exc}"
            flipped = outcome_verdict not in (
                Verdict.PASS.value,
                Verdict.SKIP.value,
                Verdict.ERROR.value,
            )
            evidence["lane_results"].append(
                {
                    "check": peer.id,
                    "lane": lane.name,
                    "defect": lane.description,
                    "verdict": outcome_verdict,
                    "flipped": flipped,
                    **({"note": note} if note else {}),
                }
            )
            if not flipped:
                unflipped.append(f"{peer.id}/{lane.name} (got {outcome_verdict})")

    evidence["peers_with_lanes"] = len(peers) - len(laneless)
    cov = Coverage(checked, "red-team lanes", expected=total_lanes)
    if laneless:
        return _finalize(
            "verdict_schema",
            "Verdict schema / red team",
            Verdict.FAIL,
            cov,
            f"checks shipping NO MUST_FIRE lane: {', '.join(laneless)} — a check that has "
            f"never been shown to fire is not evidence of anything",
            evidence,
        )
    if unflipped:
        return _finalize(
            "verdict_schema",
            "Verdict schema / red team",
            Verdict.FAIL,
            cov,
            f"{len(unflipped)}/{total_lanes} red-team lanes did NOT flip: "
            + "; ".join(unflipped[:5])
            + (" …" if len(unflipped) > 5 else ""),
            evidence,
        )
    return _finalize(
        "verdict_schema",
        "Verdict schema / red team",
        Verdict.PASS,
        cov,
        f"all {checked}/{total_lanes} corrupt-artifact lanes flipped their gates "
        f"across {len(peers)} peer checks",
        evidence,
    )
