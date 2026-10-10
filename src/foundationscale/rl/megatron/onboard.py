"""Onboarding verdict for a model family on the Megatron RL lane.

FoundationScale takes model families from Megatron-Bridge (``AutoBridge``) rather than
writing them, so a new family -- or a Bridge upgrade -- needs proof that the family
trains correctly HERE, not just that Bridge can build it. Every family defect this lane
has met sat at that seam: a forward that disagreed with Hugging Face (missing shared-KV
layers), a logit soft-cap applied twice, an export that left tensors unwritten, a save
or resume that did not fit at scale.

This module turns the outputs of four short runs into one table:

``parity``
    Megatron token logprobs (the driver's ``--parity-only`` dump) against Hugging Face
    on the same rows: per-row mean and max absolute difference under a threshold.
``refit``
    Every step of an online run wrote at least one HF tensor and left none unwritten.
``step1``
    The first measured step is on-policy: ``ratio_mean`` ~ 1, ``clip_fraction`` ~ 0, a
    finite non-zero ``grad_norm``, and the optimizer step applied.
``save``
    Every save record is ``ok`` with no blocking gate verdict (a SKIP reads UNMEASURED).
``resume``
    A resumed run's ``RESUMED ... param_hash=`` equals the saved step's hash in the
    original run's metrics.

Each check is PASS, FAIL or UNMEASURED (its input was not supplied or held nothing);
an UNMEASURED check never reads as a pass. The CLI exits 0 when all requested checks
pass, 5 on any failure, 95 when nothing failed but something was unmeasured.

The comparison is torch-free; ``hf_token_logprobs`` imports torch and transformers lazily.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "Check",
    "check_parity",
    "check_refit",
    "check_resume",
    "check_save",
    "check_step1",
    "compare_row",
    "hf_token_logprobs",
    "main",
    "parity_from_dump",
    "verdict_exit",
]

PASS, FAIL, UNMEASURED = "PASS", "FAIL", "UNMEASURED"
_SKIP = "SKIP"
_RESUMED = re.compile(r"RESUMED from \S*?step_(\d+) at step \d+ param_hash=([0-9a-f]+)")


@dataclass(frozen=True)
class Check:
    name: str
    verdict: str
    detail: str


def compare_row(
    megatron: Sequence[float], hf: Sequence[float], ids: Sequence[int], pad_id: int | None
) -> dict[str, float]:
    """Absolute logprob differences over the real (non-pad) predicted tokens of one row.

    ``megatron`` is the driver's per-position dump (index 0 has no prediction and is
    skipped); ``hf`` holds the logprob of token ``i + 1`` at index ``i``.
    """
    diffs = [
        abs(float(m) - float(h))
        for m, h, tok in zip(megatron[1:], hf, ids[1:], strict=False)
        if pad_id is None or tok != pad_id
    ]
    if not diffs:
        return {"n": 0, "max_abs": math.nan, "mean_abs": math.nan}
    return {"n": len(diffs), "max_abs": max(diffs), "mean_abs": sum(diffs) / len(diffs)}


def hf_token_logprobs(model: Any, ids: Sequence[int]) -> list[float]:
    """Hugging Face logprob of each next token of ``ids`` (length ``len(ids) - 1``)."""
    import torch

    device = next(model.parameters()).device
    x = torch.tensor(list(ids), device=device)[None]
    with torch.no_grad():
        logits = model(x).logits[0, :-1].float()
    picked = torch.log_softmax(logits, -1).gather(-1, x[0, 1:, None])[:, 0]
    return [float(v) for v in picked.detach().cpu()]


def parity_from_dump(
    dump_path: str, hf_model: str, *, fp32: bool = False
) -> list[dict[str, float]]:
    """compare_row for every row of a driver ``--parity-only`` dump against ``hf_model``.

    ``fp32`` loads the HF model in float32 to match a ``--parity-only --fp32`` dump.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dump = json.loads(Path(dump_path).read_text())
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: Any = AutoModelForCausalLM.from_pretrained(
        hf_model, dtype=torch.float32 if fp32 else torch.bfloat16
    )
    model = model.to(device).eval()
    return [
        compare_row(mc, hf_token_logprobs(model, ids), ids, pad)
        for ids, mc in zip(dump["input_ids"], dump["logprobs"], strict=True)
    ]


def check_parity(
    rows: Iterable[Mapping[str, Any]], *, mean_tol: float = 0.05, max_tol: float = 1.0
) -> Check:
    """PASS when every compared row stays under both tolerances (nats per token)."""
    measured = [r for r in rows if r.get("n", 0)]
    if not measured:
        return Check("parity", UNMEASURED, "no compared rows")
    worst_mean = max(float(r["mean_abs"]) for r in measured)
    worst_max = max(float(r["max_abs"]) for r in measured)
    ok = worst_mean <= mean_tol and worst_max <= max_tol
    return Check(
        "parity",
        PASS if ok else FAIL,
        f"{len(measured)} rows: worst mean {worst_mean:.4f} (tol {mean_tol}), "
        f"worst max {worst_max:.4f} (tol {max_tol})",
    )


def _steps(metrics: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted((r for r in metrics if "step" in r and "save" not in r), key=lambda r: r["step"])


def check_refit(metrics: Iterable[Mapping[str, Any]]) -> Check:
    steps = [r for r in _steps(metrics) if "refit_unwritten" in r]
    if not steps:
        return Check("refit", UNMEASURED, "no step carries refit counts")
    bad = [r["step"] for r in steps if r["refit_unwritten"] or not r.get("refit_written")]
    written = min(int(r.get("refit_written") or 0) for r in steps)
    if bad:
        return Check("refit", FAIL, f"steps {bad} left tensors unwritten or wrote none")
    return Check("refit", PASS, f"{len(steps)} steps, 0 unwritten, >= {written} written each")


def check_step1(
    metrics: Iterable[Mapping[str, Any]], *, ratio_tol: float = 0.05, clip_tol: float = 0.05
) -> Check:
    steps = _steps(metrics)
    if not steps:
        return Check("step1", UNMEASURED, "no training step recorded")
    first = steps[0]
    ratio = float(first.get("ratio_mean", math.nan))
    clip = float(first.get("clip_fraction", math.nan))
    grad = float(first.get("grad_norm", math.nan))
    problems = []
    if not abs(ratio - 1.0) <= ratio_tol:
        problems.append(f"ratio_mean {ratio:.4f} not within {ratio_tol} of 1")
    if not clip <= clip_tol:
        problems.append(f"clip_fraction {clip:.4f} > {clip_tol}")
    if not (math.isfinite(grad) and grad > 0):
        problems.append(f"grad_norm {grad} not finite and positive")
    if first.get("update_ok") is False:
        problems.append("optimizer step not applied")
    detail = f"step {first['step']}: ratio {ratio:.4f}, clip {clip:.4f}, grad_norm {grad:.4g}"
    return Check("step1", FAIL if problems else PASS, "; ".join(problems) or detail)


def check_save(metrics: Iterable[Mapping[str, Any]]) -> Check:
    """FAIL on a save that is not ``ok`` or carries a blocking gate verdict.

    A gate that answered SKIP abstained: the save was not blocked, but nothing was
    measured, so the check reads UNMEASURED and names the abstaining gates.
    """
    saves = [r for r in metrics if "save" in r]
    if not saves:
        return Check("save", UNMEASURED, "no save record")
    bad, skipped = [], []
    for record in saves:
        gates = record.get("gates") or {}
        blocking = [g for g, v in gates.items() if v not in (PASS, _SKIP)]
        skipped += [f"{record['save']}:{g}" for g, v in gates.items() if v == _SKIP]
        if not record.get("ok") or blocking:
            bad.append(f"{record['save']}: ok={record.get('ok')} blocking={blocking}")
    if bad:
        return Check("save", FAIL, "; ".join(bad))
    n_gates = sum(len(r.get("gates") or {}) for r in saves)
    if skipped:
        return Check("save", UNMEASURED, f"{len(saves)} saves ok; gates abstained: {skipped}")
    return Check("save", PASS, f"{len(saves)} saves, {n_gates} gate verdicts all PASS")


def check_resume(metrics: Iterable[Mapping[str, Any]], resumed_log: str) -> Check:
    match = _RESUMED.search(resumed_log)
    if match is None:
        return Check("resume", UNMEASURED, "no RESUMED line in the resumed run's log")
    saved_step, restored = int(match.group(1)), match.group(2)
    by_step = {int(r["step"]): str(r.get("param_hash", "")) for r in _steps(metrics)}
    expected = by_step.get(saved_step)
    if not expected:
        return Check("resume", UNMEASURED, f"original run has no param_hash at step {saved_step}")
    same = restored.startswith(expected) or expected.startswith(restored)
    detail = f"step {saved_step}: restored {restored[:16]} vs saved {expected[:16]}"
    return Check("resume", PASS if same else FAIL, detail)


def verdict_exit(checks: Sequence[Check]) -> int:
    """0 all PASS, 5 any FAIL, 95 none failed but at least one UNMEASURED."""
    verdicts = {c.verdict for c in checks}
    if FAIL in verdicts:
        return 5
    return 95 if UNMEASURED in verdicts or not checks else 0


def _jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parity", help="JSON list of compare_row results")
    ap.add_argument("--parity-dump", help="driver --parity-only dump to compare against HF")
    ap.add_argument("--hf-model", help="HF checkpoint for --parity-dump (needs a GPU or CPU torch)")
    ap.add_argument("--fp32", action="store_true", help="load --hf-model in float32")
    ap.add_argument("--metrics", help="online run metrics JSONL (refit, step1, save, resume)")
    ap.add_argument("--resumed-log", help="stdout of the run resumed from --metrics' state")
    ap.add_argument("--mean-tol", type=float, default=0.05)
    ap.add_argument("--max-tol", type=float, default=1.0)
    args = ap.parse_args(argv)
    metrics = _jsonl(args.metrics) if args.metrics else []
    parity = json.loads(Path(args.parity).read_text()) if args.parity else []
    if args.parity_dump and args.hf_model:
        parity = parity_from_dump(args.parity_dump, args.hf_model, fp32=args.fp32)
    log = Path(args.resumed_log).read_text(errors="replace") if args.resumed_log else ""
    checks = [
        check_parity(parity, mean_tol=args.mean_tol, max_tol=args.max_tol),
        check_refit(metrics),
        check_step1(metrics),
        check_save(metrics),
        check_resume(metrics, log),
    ]
    width = max(len(c.name) for c in checks)
    for check in checks:
        print(f"{check.name:<{width}}  {check.verdict:<10}  {check.detail}")
    rc = verdict_exit(checks)
    print(f"ONBOARD_VERDICT rc={rc}", file=sys.stderr)
    return rc


if __name__ == "__main__":  # pragma: no cover -- module entry point, main() is tested
    raise SystemExit(main())
