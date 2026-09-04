"""Item 9 -- training dynamics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .._base import (
    Coverage,
    Verdict,
)
from .._core import (
    CheckResult,
    _finalize,
)

# ---------------------------------------------------------------------------
# Item 9 — training dynamics
# ---------------------------------------------------------------------------


def _check_training_dynamics(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 9: loss bands at pinned iterations, the <0.1 hard floor, LR on
    every evidence row, and sec/iter measured ONLY past iteration 100.

    Contract: dynamics.metrics_jsonl rows {"iteration": int, "loss": float,
    "lr": float, "iter_time_s": float}. Every record scanned is an evidence
    row, so EVERY record must carry a numeric lr — a row without it
    disqualifies the sweep (that is what 'lr on every evidence row' means
    operationally). Bands failing to find their iteration FAIL by name.
    """
    evidence: dict[str, Any] = {"bands": s["bands"], "hard_floor": s["hard_floor"]}
    try:
        rows = [
            json.loads(ln)
            for ln in Path(s["metrics_jsonl"]).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.ERROR,
            Coverage.none("banded iterations"),
            f"metrics unreadable: {s['metrics_jsonl']}: {exc}",
            evidence,
        )
    evidence["records_scanned"] = len(rows)
    if not rows:
        # Coverage will render this VACUOUS via the ladder only for PASS; an
        # empty metrics file is asserted directly here so the detail names it.
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.FAIL,
            Coverage.none("banded iterations"),
            "0 metric records — a dynamics claim over zero rows is doctrine 1 verbatim",
            evidence,
        )

    floor = float(s["hard_floor"])
    no_lr = []
    floor_hits = []
    by_iter: dict[int, dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("iteration"), int):
            return _finalize(
                "training_dynamics",
                "Training dynamics",
                Verdict.ERROR,
                Coverage(0, "banded iterations", expected=len(s["bands"])),
                "a metric record lacks an integer 'iteration' — the sweep cannot key its evidence",
                evidence,
            )
        it = r["iteration"]
        by_iter[it] = r
        lr = r.get("lr")
        if not isinstance(lr, (int, float)) or isinstance(lr, bool):
            no_lr.append(it)
        loss = r.get("loss")
        if isinstance(loss, (int, float)) and not isinstance(loss, bool) and loss < floor:
            floor_hits.append((it, loss))
    if no_lr:
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.FAIL,
            Coverage(0, "banded iterations", expected=len(s["bands"])),
            f"{len(no_lr)} evidence rows carry no numeric lr (e.g. iterations {no_lr[:5]}) — "
            f"item 9: lr on EVERY evidence row or the row is not evidence",
            evidence,
        )
    if floor_hits:
        it, loss = floor_hits[0]
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.FAIL,
            Coverage(0, "banded iterations", expected=len(s["bands"])),
            f"HARD FLOOR: loss {loss} < {floor} at iteration {it}"
            + (f" (+{len(floor_hits) - 1} more)" if len(floor_hits) > 1 else ""),
            evidence,
        )

    checked = 0
    band_misses = []
    band_out = []
    lrs = []
    for band in s["bands"]:
        rec = by_iter.get(int(band["iteration"]))
        if rec is None:
            band_misses.append(band["iteration"])
            continue
        checked += 1
        loss = rec.get("loss")
        lr = rec["lr"]
        lrs.append(float(lr))
        if not isinstance(loss, (int, float)) or isinstance(loss, bool):
            band_out.append(f"iter {band['iteration']}: loss not numeric ({loss!r})")
            continue
        if not (float(band["lo"]) <= float(loss) <= float(band["hi"])):
            band_out.append(
                f"iter {band['iteration']}: loss {loss} outside [{band['lo']}, {band['hi']}]"
            )
    if band_misses:
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.UNDERCOVERED,
            Coverage(checked, "banded iterations", expected=len(s["bands"])),
            f"no metric record exists at banded iterations {band_misses} — bands that were "
            f"never evaluated cannot be 'within band'",
            evidence,
        )
    if band_out:
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.FAIL,
            Coverage(checked, "banded iterations", expected=len(s["bands"])),
            " | ".join(band_out[:4]),
            evidence,
        )

    speed_rows = [r for r in by_iter.values() if r["iteration"] > 100]
    evidence["speed_window"] = {"iteration_gt": 100, "rows": len(speed_rows)}
    if not speed_rows:
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.FAIL,
            Coverage(checked, "banded iterations", expected=len(s["bands"])),
            f"0 records past iteration 100 (max seen: {max(by_iter)}) — item 9 forbids "
            f"reporting sec/iter from the warmup window, so none is reportable",
            evidence,
        )
    times = [r.get("iter_time_s") for r in speed_rows]
    if any(not isinstance(t, (int, float)) or isinstance(t, bool) for t in times):
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.FAIL,
            Coverage(checked, "banded iterations", expected=len(s["bands"])),
            "some post-100 records lack numeric iter_time_s — sec/iter would be partially measured",
            evidence,
        )
    mean_t = sum(float(t) for t in times) / len(times)
    evidence["sec_per_iter"] = {"mean": round(mean_t, 4), "samples": len(times)}
    evidence["lr_range_on_bands"] = [min(lrs), max(lrs)]

    return _finalize(
        "training_dynamics",
        "Training dynamics",
        Verdict.PASS,
        Coverage(checked, "banded iterations", expected=len(s["bands"])),
        f"{checked}/{len(s['bands'])} banded iterations in-band; 0 floor breaches over "
        f"{len(rows)} records; mean sec/iter {mean_t:.2f}s over {len(times)} post-100 rows; "
        f"lr present on all evidence rows",
        evidence,
    )
