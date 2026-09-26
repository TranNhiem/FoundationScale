
"""Feasibility verdicts for a training configuration on a given hardware target.

Memory thresholds: total per-GPU usage above 0.90 x mem is ``infeasible``;
above 0.80 is ``warn``. A budget (gpu-hours) or deadline (wall hours) that the
time estimate exceeds downgrades the verdict to ``infeasible`` — the run cannot
be afforded/finished as specified. A missing MFU makes time UNMEASURED
(finding TR-FEAS-004) but does not fabricate a verdict.

When the verdict is not ``ok``, alternatives are tried IN THIS ORDER and each
is re-estimated:

1. gradient checkpointing on (only when the base estimate ran without it);
2. micro_batch -> 1 (compensated with gradient accumulation);
3. FSDP sharding over all GPUs (when running DDP);
4. LoRA;
5. shorter seq_len (halve, floor 2048);
6. QLoRA — always ``executable: false``: QLoRA is not an FS feature today
   (``--adapter`` supports lora only);
7. tensor parallelism tp=2 / tp=4 — executable only when ``caps.check`` says
   tp executes; with a refusing caps the alternative is kept but marked
   non-executable, and without any caps it is non-executable ("unmeasured",
   never assumed);
8. a smaller variant in the same family (from the knowledge pack; absent or
   unknown family -> non-executable, never silently dropped).

Each alternative is ``{change, estimate, verdict, executable, missing}``.
``verdict`` uses ok/warn/infeasible, or ``unknown`` when nothing was estimable.
``caps.check`` needs a stage string; ``stage`` (default "sft") is used only for
those probes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.skills.training import knowledge as _knowledge
from foundationskills.skills.training.estimate import MemoryEstimate, TimeEstimate, estimate_memory, estimate_time
from foundationskills.skills.training.knowledge import Hardware, KnowledgeError, Variant

__all__ = ["Feasibility", "check_feasibility"]

WARN_FRACTION = 0.80
INFEASIBLE_FRACTION = 0.90
_VERDICT_RANK = {"ok": 0, "warn": 1, "infeasible": 2}


@dataclass(frozen=True)
class Feasibility:
    verdict: str  # "ok" | "warn" | "infeasible"
    findings: list[dict[str, Any]] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)


def _memory_verdict(total_gb: float, mem_gb: float) -> tuple[str, float]:
    frac = total_gb / mem_gb if mem_gb > 0 else float("inf")
    if frac > INFEASIBLE_FRACTION:
        return "infeasible", frac
    if frac > WARN_FRACTION:
        return "warn", frac
    return "ok", frac


def _downgrade(verdict: str, target: str) -> str:
    return target if _VERDICT_RANK[target] > _VERDICT_RANK[verdict] else verdict


def _alt_entry(change: str, est: MemoryEstimate, mem_gb: float, *, executable: bool, missing: str | None) -> dict[str, Any]:
    verdict, frac = _memory_verdict(est.total_per_gpu_gb, mem_gb)
    estimate = asdict(est)
    estimate["mem_fraction"] = frac
    estimate["hardware_mem_gb"] = mem_gb
    return {
        "change": change,
        "estimate": estimate,
        "verdict": verdict,
        "executable": executable,
        "missing": missing,
    }


def _alt_blocked(change: str, missing: str) -> dict[str, Any]:
    return {"change": change, "estimate": {}, "verdict": "unknown", "executable": False, "missing": missing}


def _smaller_variant(variant: Variant) -> tuple[_knowledge.Family | None, Variant | None]:
    for family in _knowledge.load_families().values():
        for candidate in family.variants:
            if candidate.id == variant.id and candidate.hidden == variant.hidden and candidate.size_b == variant.size_b:
                smaller = [v for v in family.variants if 0 < v.size_b < variant.size_b]
                return family, (max(smaller, key=lambda v: v.size_b) if smaller else None)
    return None, None


def _alternatives(
    variant: Variant,
    hardware: Hardware,
    *,
    gpus: int,
    method: str,
    seq_len: int,
    micro_batch: int,
    sharding: str,
    tp: int,
    grad_ckpt: bool,
    caps: FSCapabilities | None,
    stage: str,
) -> list[dict[str, Any]]:
    mem = hardware.mem_gb
    alts: list[dict[str, Any]] = []
    base_kw: dict[str, Any] = {"method": method, "sharding": sharding, "world": gpus, "tp": tp}

    if not grad_ckpt:
        est = estimate_memory(variant, seq_len=seq_len, micro_batch=micro_batch, grad_ckpt=True, **base_kw)
        alts.append(_alt_entry("enable gradient checkpointing", est, mem, executable=True, missing=None))

    if micro_batch > 1:
        est = estimate_memory(variant, seq_len=seq_len, micro_batch=1, grad_ckpt=grad_ckpt, **base_kw)
        alts.append(
            _alt_entry(
                f"micro_batch {micro_batch} -> 1 (compensate with gradient accumulation)",
                est, mem, executable=True, missing=None,
            )
        )

    if sharding != "fsdp":
        est = estimate_memory(
            variant, seq_len=seq_len, micro_batch=micro_batch, grad_ckpt=grad_ckpt,
            method=method, sharding="fsdp", world=gpus, tp=tp,
        )
        alts.append(
            _alt_entry(f"shard weights/grads/optimizer with FSDP over {gpus} GPUs", est, mem, executable=True, missing=None)
        )

    if method != "lora":
        est = estimate_memory(
            variant, seq_len=seq_len, micro_batch=micro_batch, grad_ckpt=grad_ckpt,
            method="lora", sharding=sharding, world=gpus, tp=tp,
        )
        alts.append(_alt_entry("switch to LoRA (train low-rank adapters only)", est, mem, executable=True, missing=None))

    if seq_len > 2048:
        new_seq = max(2048, seq_len // 2)
        est = estimate_memory(variant, seq_len=new_seq, micro_batch=micro_batch, grad_ckpt=grad_ckpt, **base_kw)
        alts.append(_alt_entry(f"shorten seq_len {seq_len} -> {new_seq}", est, mem, executable=True, missing=None))

    if method != "qlora":
        est = estimate_memory(
            variant, seq_len=seq_len, micro_batch=micro_batch, grad_ckpt=grad_ckpt,
            method="qlora", sharding=sharding, world=gpus, tp=tp,
        )
        alts.append(
            _alt_entry(
                "switch to QLoRA (4-bit base + LoRA)",
                est, mem,
                executable=False,
                missing="missing: qlora (FS --adapter supports lora only; verify)",
            )
        )

    if tp < 2:
        backend = "fsdp" if sharding == "fsdp" else "ddp"
        for new_tp in (2, 4):
            est = estimate_memory(
                variant, seq_len=seq_len, micro_batch=micro_batch, grad_ckpt=grad_ckpt,
                method=method, sharding=sharding, world=gpus, tp=new_tp,
            )
            if caps is None:
                alts.append(
                    _alt_entry(
                        f"tensor parallelism tp={new_tp}", est, mem,
                        executable=False,
                        missing="missing: FS capabilities probe (tp executability unmeasured)",
                    )
                )
            else:
                reason = caps.check(stage, tp=new_tp, backend=backend)
                alts.append(_alt_entry(f"tensor parallelism tp={new_tp}", est, mem, executable=reason is None, missing=reason))

    change = "switch to a smaller variant in the same family"
    try:
        family, smaller = _smaller_variant(variant)
    except KnowledgeError as exc:
        alts.append(_alt_blocked(change, f"knowledge pack unavailable: {exc}"))
    else:
        if family is None:
            alts.append(_alt_blocked(change, "variant not present in the training knowledge pack"))
        elif smaller is None:
            alts.append(_alt_blocked(change, f"no smaller variant in family {family.name}"))
        else:
            est = estimate_memory(
                smaller, seq_len=seq_len, micro_batch=micro_batch, grad_ckpt=grad_ckpt, **base_kw
            )
            alts.append(
                _alt_entry(
                    f"switch to smaller variant {family.name}/{smaller.id} ({smaller.size_b}B)",
                    est, mem, executable=True, missing=None,
                )
            )
    return alts


def check_feasibility(
    variant: Variant,
    hardware: Hardware,
    *,
    gpus: int,
    method: str,
    seq_len: int,
    micro_batch: int,
    tokens: int,
    budget_gpu_hours: float | None = None,
    deadline_hours: float | None = None,
    caps: FSCapabilities | None = None,
    sharding: str = "fsdp",
    tp: int = 1,
    grad_ckpt: bool = True,
    stage: str = "sft",
) -> Feasibility:
    """Verdict + cost findings + ranked alternatives for one configuration."""
    gpus = max(1, int(gpus))
    findings: list[dict[str, Any]] = []

    base = estimate_memory(
        variant, method=method, seq_len=seq_len, micro_batch=micro_batch,
        sharding=sharding, world=gpus, tp=tp, grad_ckpt=grad_ckpt,
    )
    verdict, frac = _memory_verdict(base.total_per_gpu_gb, hardware.mem_gb)
    findings.append(
        {
            "rule_id": "TR-FEAS-001",
            "severity": {"ok": "INFO", "warn": "WARN", "infeasible": "BLOCK"}[verdict],
            "detail": (
                f"memory {base.total_per_gpu_gb:.1f} GB/GPU = {frac:.1%} of {hardware.mem_gb:.0f} GB "
                f"({method}, {sharding} x{gpus}, tp={tp})"
            ),
            "passed": verdict != "infeasible",
        }
    )

    time_est: TimeEstimate | None = None
    try:
        time_est = estimate_time(variant, tokens=tokens, hardware=hardware, gpus=gpus, method=method,
                                 stage=stage, sharding=sharding, micro_batch=micro_batch, grad_ckpt=grad_ckpt)
    except Exception as exc:  # noqa: BLE001 - absence of evidence is UNMEASURED, never PASS
        findings.append(
            {
                "rule_id": "TR-FEAS-004",
                "severity": "WARN",
                "detail": f"time UNMEASURED: {exc}",
                "passed": None,
            }
        )
    if time_est is not None:
        if budget_gpu_hours is not None and time_est.gpu_hours > budget_gpu_hours:
            verdict = _downgrade(verdict, "infeasible")
            findings.append(
                {
                    "rule_id": "TR-FEAS-002",
                    "severity": "BLOCK",
                    "detail": f"needs {time_est.gpu_hours:.1f} gpu-hours > budget {budget_gpu_hours:.1f}",
                    "passed": False,
                }
            )
        if deadline_hours is not None and time_est.hours > deadline_hours:
            verdict = _downgrade(verdict, "infeasible")
            findings.append(
                {
                    "rule_id": "TR-FEAS-003",
                    "severity": "BLOCK",
                    "detail": f"needs {time_est.hours:.1f} h > deadline {deadline_hours:.1f} h on {gpus} GPUs",
                    "passed": False,
                }
            )

    alternatives: list[dict[str, Any]] = []
    if verdict != "ok":
        alternatives = _alternatives(
            variant, hardware, gpus=gpus, method=method, seq_len=seq_len,
            micro_batch=micro_batch, sharding=sharding, tp=tp,
            grad_ckpt=grad_ckpt, caps=caps, stage=stage,
        )
    return Feasibility(verdict=verdict, findings=findings, alternatives=alternatives)
