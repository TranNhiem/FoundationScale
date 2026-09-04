"""Item 7 -- schedule."""

from __future__ import annotations

from .._base import (
    Coverage,
    Verdict,
)
from .._core import (
    CheckResult,
    _finalize,
)

# ---------------------------------------------------------------------------
# Item 7 — schedule
# ---------------------------------------------------------------------------


def _check_schedule(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 7: the banner's schedule invariants, arithmetic only. 2 invariants:
    (a) train_iters == lr_decay_iters; (b) train_iters % save_interval == 0,
    pardoned ONLY by an explicit_final_save declaration — the pardon is
    surfaced in evidence, never silently folded into a pass."""
    ti, ldi, si = int(s["train_iters"]), int(s["lr_decay_iters"]), int(s["save_interval"])
    evidence = {
        "train_iters": ti,
        "lr_decay_iters": ldi,
        "save_interval": si,
        "explicit_final_save": s["explicit_final_save"],
        "smoke": s["smoke"],
    }
    checked = 0
    bad = []
    checked += 1  # invariant (a)
    if ti != ldi:
        bad.append(
            f"train_iters {ti} != lr_decay_iters {ldi} — "
            f"the LR schedule ends before/after training does"
        )
    checked += 1  # invariant (b)
    if si <= 0:
        bad.append(f"save_interval {si} <= 0")
    elif ti % si != 0:
        if s["explicit_final_save"]:
            evidence["final_save_pardon"] = (
                f"{ti} % {si} == {ti % si}; pardoned by explicit_final_save=true — "
                f"declared, not silent"
            )
        else:
            bad.append(
                f"train_iters {ti} %% save_interval {si} == {ti % si} "
                f"and explicit_final_save is false — "
                f"the final state is never written"
            )
    if bad:
        return _finalize(
            "schedule_consistency",
            "Schedule banner",
            Verdict.FAIL,
            Coverage(checked - len(bad), "schedule invariants", expected=2),
            " | ".join(bad),
            evidence,
        )
    return _finalize(
        "schedule_consistency",
        "Schedule banner",
        Verdict.PASS,
        Coverage(checked, "schedule invariants", expected=2),
        "train_iters == lr_decay_iters; save cadence lands on the final step"
        + (" (via declared explicit final save)" if evidence.get("final_save_pardon") else "")
        + ("; SMOKE run — banner will carry the qualifier" if s["smoke"] else ""),
        evidence,
    )
