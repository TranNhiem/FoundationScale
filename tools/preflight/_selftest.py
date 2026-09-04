"""--self-test: prove every check can FAIL and can PASS."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._base import (
    Verdict,
)
from ._core import (
    _REGISTRY_ORDER,
    REGISTRY,
    _execute,
)
from ._fixtures import (
    _fresh_world,
    _run_lane_against,
)

# ---------------------------------------------------------------------------
# --self-test: prove every check can FAIL and can PASS, with denominators
# ---------------------------------------------------------------------------


def run_self_test(out: Callable[[str], None] = print) -> tuple[int, dict[str, Any]]:
    """Both halves, per check, on fresh synthesized worlds.

    MUST_PASS: healthy world -> the check must PASS (a check that blocks on
    everything satisfies every MUST_FIRE lane while verifying nothing — the
    exact hole verify_controls' MUST_PASS guard exists to close).
    MUST_FIRE: each declared lane -> the check must NOT return PASS; a lane
    that leaves the verdict PASS is a detector proven not to fire on the
    defect class it claims.

    Exitcode 0 iff every check proves both halves. The returned dict carries
    the denominators so callers (and tests) consume them structurally.
    """
    failures: list[str] = []
    checks_total = len(_REGISTRY_ORDER)
    fire_total = fire_proven = 0
    pass_total = pass_proven = 0
    per_check: list[dict[str, Any]] = []

    out(
        "preflight --self-test — proving each check can FAIL on a known defect "
        "and PASS on a healthy world"
    )
    for chk in _REGISTRY_ORDER:
        rec: dict[str, Any] = {
            "check": chk.id,
            "lanes": len(chk.lanes),
            "must_pass": None,
            "fire": [],
        }

        # ---- MUST_PASS half ------------------------------------------------
        pass_total += 1
        try:
            with _fresh_world() as world:
                shared: dict[str, Any] = {"_config_sha256": world._cfg_sha}  # type: ignore[attr-defined]
                baseline = _execute(REGISTRY["frozen_manifest"], world.cfg, world.env, shared)
                if chk.id == "frozen_manifest":
                    res = baseline
                elif baseline.verdict is not Verdict.PASS:
                    raise RuntimeError(f"healthy-world precondition failed: {baseline.detail}")
                else:
                    # The runtime check red-teams every peer; proving the
                    # mechanism on ONE real peer (with its real lanes) keeps
                    # the self-test sub-quadratic. Runtime runs all peers.
                    #
                    # The ternary carries the comment now because it is the only
                    # binding of `peers`: an earlier reading called it dead and
                    # proposed deleting it, but its `else None` arm is what binds
                    # `peers` for every non-verdict_schema check. The redundant
                    # half was the `if` block that recomputed the same list.
                    peers = [REGISTRY["frozen_manifest"]] if chk.id == "verdict_schema" else None
                    res = _execute(chk, world.cfg, world.env, shared, registry=peers)
            rec["must_pass"] = res.verdict.value
            if res.verdict is Verdict.PASS:
                pass_proven += 1
            else:
                failures.append(
                    f"{chk.id}: MUST_PASS half failed on a known-healthy world "
                    f"(got {res.verdict.value}: {res.detail})"
                )
        except Exception as exc:  # noqa: BLE001
            rec["must_pass"] = "HARNESS-ERROR"
            failures.append(f"{chk.id}: MUST_PASS harness raised {type(exc).__name__}: {exc}")

        # ---- MUST_FIRE half -------------------------------------------------
        if not chk.lanes:
            failures.append(
                f"{chk.id}: declares NO MUST_FIRE lane — a check that has never been "
                f"shown to fire may not certify a launch (design item 4)"
            )
        for lane in chk.lanes:
            fire_total += 1
            try:
                res = _run_lane_against(chk, lane)
                flipped = res.verdict is not Verdict.PASS and res.verdict is not Verdict.ERROR
                rec["fire"].append({"lane": lane.name, "verdict": res.verdict.value})
                if flipped:
                    fire_proven += 1
                else:
                    failures.append(
                        f"{chk.id}/{lane.name}: MUST_FIRE lane left verdict "
                        f"{res.verdict.value} ({res.detail[:120]}) — the defect "
                        f"'{lane.description}' did not flip the check"
                    )
            except Exception as exc:  # noqa: BLE001
                rec["fire"].append({"lane": lane.name, "verdict": "HARNESS-ERROR"})
                failures.append(
                    f"{chk.id}/{lane.name}: MUST_FIRE harness raised {type(exc).__name__}: {exc}"
                )
        per_check.append(rec)

    report = {
        "checks_total": checks_total,
        "must_fire_proven": fire_proven,
        "must_fire_total": fire_total,
        "must_pass_proven": pass_proven,
        "must_pass_total": pass_total,
        "failures": failures,
        "per_check": per_check,
    }
    out(f"checks: {checks_total}")
    out(f"MUST_FIRE proven: {fire_proven}/{fire_total} lanes")
    out(f"MUST_PASS proven: {pass_proven}/{pass_total} checks")
    if failures:
        out(f"\n{len(failures)} self-test failure(s):")
        for f in failures:
            out(f"  - {f}")
        out(
            "\nresult: FAILED — at least one check is not proven for this "
            "launch's artifact classes."
        )
        return 1, report
    out(
        "\nresult: OK — every check flipped on its deliberately corrupt "
        "artifacts and passed its healthy world."
    )
    return 0, report
