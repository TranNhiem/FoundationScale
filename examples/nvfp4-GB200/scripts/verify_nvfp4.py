#!/usr/bin/env python3
"""Check a bf16 / NVFP4 control pair from train_nvfp4.sh, and report what NVFP4 cost.

usage: python scripts/verify_nvfp4.py RUN_DIR
       (RUN_DIR holds bf16/train.log and nvfp4/train.log, i.e. OUTPUT_DIR from the config)

NVFP4 can fail silently: a run that never enabled FP4 trains in bf16 and looks healthy.
Step time cannot tell the two apart (it is noisy, and NVFP4 is not always faster), so
this checks what the logs actually say:

  bf16-arm     the control resolved to bf16 and printed no FP4 setting
  nvfp4-arm    the treatment asked for NVFP4 -- the bf16_with_nvfp4_mixed recipe, or an
               explicit mixed_precision block with fp4 e2m1 and fp4_recipe nvfp4 -- AND
               the resolved model config says fp4 'e2m1'. (fp4_recipe alone proves
               nothing: the bf16 arm prints "fp4_recipe: nvfp4" too, as an inactive
               default.)
  completed    both arms logged every planned step
  finite       no NaN or skipped steps, every loss finite
  diverged     the two arms' losses are NOT bit-identical. Same data, same seed: if FP4
               were inactive the losses would match to the last digit, so agreement is
               the failure here, not the success.

Then it prints step time (median after warm-up), peak memory if logged, and the loss gap.

Exit codes: 0 all checks pass, 5 a check failed, 95 nothing to check (a log is missing
or has no iteration lines).
"""

from __future__ import annotations

import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

EXIT_PASS, EXIT_FAIL, EXIT_UNMEASURED = 0, 5, 95
WARMUP_STEPS = 3

_ITER = re.compile(
    r"iteration\s+(\d+)/\s*(\d+)\s*\|.*?elapsed time per iteration \(ms\):\s*([\d.]+)"
    r".*?lm loss:\s*(\S+)\s*\|.*?number of skipped iterations:\s*(\d+)"
    r"\s*\|\s*number of nan iterations:\s*(\d+)"
)
_MIXED = re.compile(r"^mixed_precision:[ \t]*(\S*)", re.M)
NVFP4_BLOCK = "block(fp4=e2m1,fp4_recipe=nvfp4)"
_FP4 = re.compile(r"^\s+fp4:\s*'?e2m1'?", re.M)
_MAX_ALLOC = re.compile(r"max allocated:\s*([\d.]+)")


@dataclass
class Arm:
    name: str
    text: str = ""
    steps: dict[int, tuple[float, str]] = field(default_factory=dict)  # step -> (ms, loss text)
    planned: int = 0
    skipped: int = 0
    nan: int = 0
    max_alloc_mb: float | None = None

    @classmethod
    def read(cls, run_dir: Path, name: str) -> Arm:
        arm = cls(name)
        log = run_dir / name / "train.log"
        if not log.is_file():
            return arm
        arm.text = log.read_text(errors="replace")
        for m in _ITER.finditer(arm.text):
            step, planned = int(m.group(1)), int(m.group(2))
            arm.steps[step] = (float(m.group(3)), m.group(4))
            arm.planned = planned
            arm.skipped = max(arm.skipped, int(m.group(5)))
            arm.nan = max(arm.nan, int(m.group(6)))
        allocs = [float(x) for x in _MAX_ALLOC.findall(arm.text)]
        arm.max_alloc_mb = max(allocs) if allocs else None
        return arm

    @property
    def mixed(self) -> str | None:
        """The recipe name, or a summary of an explicit ``mixed_precision:`` block."""
        m = _MIXED.search(self.text)
        if m is None:
            return None
        if m.group(1):
            return m.group(1).strip("'\"")
        block: list[str] = []
        for line in self.text[m.end() :].splitlines()[1:]:
            if line and not line.startswith(" "):
                break
            block.append(line.strip())
        fp4 = "fp4: e2m1" in block or "fp4: 'e2m1'" in block
        if fp4 and ("fp4_recipe: nvfp4" in block or "fp4_recipe: 'nvfp4'" in block):
            return NVFP4_BLOCK
        return "block(" + ("fp4=e2m1" if fp4 else "no fp4") + ")"

    def median_ms(self) -> float | None:
        times = [ms for step, (ms, _) in sorted(self.steps.items()) if step > WARMUP_STEPS]
        return statistics.median(times) if times else None


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return EXIT_UNMEASURED
    run_dir = Path(argv[1])
    bf16, fp4 = Arm.read(run_dir, "bf16"), Arm.read(run_dir, "nvfp4")
    missing = [a.name for a in (bf16, fp4) if not a.steps]
    if missing:
        logs = ", ".join(f"{m}/train.log" for m in missing)
        print(f"UNMEASURED: no iteration lines in {logs} under {run_dir}")
        return EXIT_UNMEASURED

    checks: list[tuple[str, bool, str]] = []
    bf16_fp4 = bool(_FP4.search(bf16.text))
    checks.append(
        (
            "bf16-arm",
            bf16.mixed == "bf16_mixed" and not bf16_fp4,
            f"mixed_precision={bf16.mixed}, fp4 e2m1 printed={bf16_fp4}",
        )
    )
    fp4_on = bool(_FP4.search(fp4.text))
    checks.append(
        (
            "nvfp4-arm",
            fp4.mixed in ("bf16_with_nvfp4_mixed", NVFP4_BLOCK) and fp4_on,
            f"mixed_precision={fp4.mixed}, fp4 e2m1 printed={fp4_on}",
        )
    )
    done = all(a.planned and max(a.steps) == a.planned for a in (bf16, fp4))
    checks.append(
        (
            "completed",
            done and bf16.planned == fp4.planned,
            f"bf16 {max(bf16.steps)}/{bf16.planned}, nvfp4 {max(fp4.steps)}/{fp4.planned}",
        )
    )
    nonfinite = [
        f"{a.name}@{s}"
        for a in (bf16, fp4)
        for s, (_, loss) in a.steps.items()
        if not math.isfinite(_float(loss))
    ]
    checks.append(
        (
            "finite",
            not nonfinite and not (bf16.skipped or bf16.nan or fp4.skipped or fp4.nan),
            f"skipped {bf16.skipped}/{fp4.skipped}, nan {bf16.nan}/{fp4.nan}, "
            f"non-finite losses {nonfinite or 'none'}",
        )
    )
    common = sorted(set(bf16.steps) & set(fp4.steps))
    identical = sum(bf16.steps[s][1] == fp4.steps[s][1] for s in common)
    checks.append(
        (
            "diverged",
            bool(common) and identical < len(common),
            f"{identical} of {len(common)} common steps print the same loss text",
        )
    )

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{name:<{width}}  {'PASS' if ok else 'FAIL'}  {detail}")

    b_ms, f_ms = bf16.median_ms(), fp4.median_ms()
    if b_ms and f_ms:
        print(
            f"\nstep time (median, steps > {WARMUP_STEPS}): bf16 {b_ms:.0f} ms, "
            f"nvfp4 {f_ms:.0f} ms  -> nvfp4 is {b_ms / f_ms:.2f}x the bf16 speed"
        )
    else:
        print(f"\nstep time: UNMEASURED (no steps after the first {WARMUP_STEPS})")
    if bf16.max_alloc_mb and fp4.max_alloc_mb:
        b_gib, f_gib = bf16.max_alloc_mb / 1024, fp4.max_alloc_mb / 1024
        print(f"peak allocated: bf16 {b_gib:.1f} GiB, nvfp4 {f_gib:.1f} GiB")
    last = max(common)
    b_loss, f_loss = _float(bf16.steps[last][1]), _float(fp4.steps[last][1])
    gaps = [abs(_float(fp4.steps[s][1]) - _float(bf16.steps[s][1])) for s in common]
    print(
        f"loss at step {last}: bf16 {b_loss:.4f}, nvfp4 {f_loss:.4f} "
        f"({(f_loss - b_loss) / b_loss:+.1%}); "
        f"mean |gap| over {len(common)} steps {statistics.mean(gaps):.4f}"
    )

    return EXIT_PASS if all(ok for _, ok, _ in checks) else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main(sys.argv))
