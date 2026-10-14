"""P0.3 parity statistics: FS run log vs verl console log, per step window.

    python3 parity_stats.py <fs_run.log> <verl_console.log> [--band 0.05]

FS rollout lines: ``[agentic-rl:rollout] {json with "rollout_step", "reward_mean", ...}`` (0-based step).
verl console lines carry ``step:N`` and ``critic/score/mean:X`` (1-based step; verl step N trains on the
rollout FS would call step N-1, so verl steps are shifted by -1 to align).
Windows are over aligned steps; a window with no measured step on either side is reported as
UNMEASURED, never as a 0.0 or a pass. The relative difference is |verl - fs| / max(|fs|, 1e-9).
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable

WINDOWS = ((0, 19), (20, 49), (50, 99))
_FS = re.compile(r"\[agentic-rl:rollout\]\s*(\{.*\})")
_VERL_STEP = re.compile(r"\bstep:(\d+)\b")
_VERL_SCORE = re.compile(r"critic/score/mean:([-0-9.eE+]+)")


def fs_rewards(lines: Iterable[str]) -> dict[int, float]:
    out: dict[int, float] = {}
    for line in lines:
        m = _FS.search(line)
        if not m:
            continue
        rec = json.loads(m.group(1))
        step, mean = rec.get("rollout_step"), rec.get("reward_mean")
        if isinstance(step, int) and isinstance(mean, (int, float)) and not isinstance(mean, bool):
            out[step] = float(mean)
    return out


def verl_rewards(lines: Iterable[str]) -> dict[int, float]:
    out: dict[int, float] = {}
    for line in lines:
        s, v = _VERL_STEP.search(line), _VERL_SCORE.search(line)
        if s and v:
            out[int(s.group(1)) - 1] = float(v.group(1))  # align to FS's 0-based rollout step
    return out


def compare(fs: dict[int, float], verl: dict[int, float], band: float = 0.05) -> list[dict]:
    rows = []
    for lo, hi in WINDOWS:
        steps = [s for s in range(lo, hi + 1) if s in fs and s in verl]
        if not steps:
            rows.append({"window": f"{lo}-{hi}", "status": "UNMEASURED", "paired_steps": 0})
            continue
        f = sum(fs[s] for s in steps) / len(steps)
        v = sum(verl[s] for s in steps) / len(steps)
        rel = abs(v - f) / max(abs(f), 1e-9)
        rows.append(
            {
                "window": f"{lo}-{hi}",
                "paired_steps": len(steps),
                "fs_mean": round(f, 4),
                "verl_mean": round(v, 4),
                "rel_diff": round(rel, 4),
                "status": "WITHIN" if rel <= band else "OUTSIDE",
            }
        )
    return rows


def main(argv: list[str]) -> int:
    band = float(argv[argv.index("--band") + 1]) if "--band" in argv else 0.05
    with open(argv[1]) as fh:
        fs = fs_rewards(fh)
    with open(argv[2]) as fh:
        verl = verl_rewards(fh)
    rows = compare(fs, verl, band)
    print(json.dumps({"band": band, "fs_steps": len(fs), "verl_steps": len(verl), "windows": rows}, indent=1))
    measured = [r for r in rows if r["status"] != "UNMEASURED"]
    if not measured:
        return 95
    return 0 if all(r["status"] == "WITHIN" for r in measured) else 5


if __name__ == "__main__":
    sys.exit(main(sys.argv))
