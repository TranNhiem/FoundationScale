
from __future__ import annotations

import pytest

from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.skills.training.feasibility import check_feasibility
from foundationskills.skills.training.knowledge import Hardware, Variant


def dense7b() -> Variant:
    return Variant(
        id="zz-not-in-knowledge", hf_id=None, local_path=None, size_b=7.0, active_b=None,
        arch="dense", hidden=4096, layers=32, heads=32, kv_heads=8, head_dim=128,
        vocab=32000, context_length=8192, num_experts=None, experts_per_token=None,
        tied_embeddings=False, instruct=False,
    )


def make_hw(mem: float = 80.0, tflops: float = 1000.0, mfu: float = 0.318, gpn: int = 8) -> Hardware:
    return Hardware(
        id="hw80", gpu_name="HW", mem_gb=mem, gpus_per_node=gpn,
        bf16_dense_tflops=tflops, peak_provenance="datasheet", interconnect="nvlink",
        mfu={
            "dense": {"value": mfu, "provenance": "literature", "evidence": "test"},
            "moe": {"value": mfu, "provenance": "literature", "evidence": "test"},
        },
        scheduler="slurm", cluster_rules=(), notes=(),
    )


def test_7b_full_on_1x80gb_is_infeasible_with_executable_lora_alternative() -> None:
    feas = check_feasibility(
        dense7b(), make_hw(), gpus=1, method="full", seq_len=4096, micro_batch=1, tokens=10**9
    )
    assert feas.verdict == "infeasible"
    assert feas.findings[0]["rule_id"] == "TR-FEAS-001" and feas.findings[0]["passed"] is False
    lora = [a for a in feas.alternatives if a["change"].startswith("switch to LoRA")]
    assert len(lora) == 1
    assert lora[0]["executable"] is True
    assert lora[0]["missing"] is None
    assert lora[0]["verdict"] == "ok"  # ~18 GB on an 80 GB card
    # changes are ordered per spec: lora before seq_len before qlora before tp
    changes = [a["change"] for a in feas.alternatives]
    assert changes.index("switch to LoRA (train low-rank adapters only)") < changes.index("shorten seq_len 4096 -> 2048")


def test_7b_full_on_8x80gb_fsdp_is_ok() -> None:
    feas = check_feasibility(
        dense7b(), make_hw(), gpus=8, method="full", seq_len=2048, micro_batch=1, tokens=10**9
    )
    assert feas.verdict in {"ok", "warn"}
    if feas.verdict == "ok":
        assert feas.alternatives == []


def test_qlora_alternative_is_marked_non_executable() -> None:
    feas = check_feasibility(
        dense7b(), make_hw(), gpus=1, method="full", seq_len=4096, micro_batch=1, tokens=10**9
    )
    qlora = [a for a in feas.alternatives if "QLoRA" in a["change"]]
    assert len(qlora) == 1
    assert qlora[0]["executable"] is False
    assert "qlora" in (qlora[0]["missing"] or "")


def test_tp_alternative_non_executable_when_caps_refuses_tp() -> None:
    caps = FSCapabilities(
        available=True, fs_version="0.0.0",
        train_objectives=("sft",), sharding_strategies=("ddp", "fsdp"),
        backends=("ddp", "fsdp"), executed_axes=("cp",), refused_axes=("pp", "ep", "tp"),
        axes_measured=True,
    )
    feas = check_feasibility(
        dense7b(), make_hw(), gpus=1, method="full", seq_len=4096, micro_batch=1, tokens=10**9, caps=caps
    )
    tps = [a for a in feas.alternatives if a["change"].startswith("tensor parallelism")]
    assert {a["change"] for a in tps} == {"tensor parallelism tp=2", "tensor parallelism tp=4"}
    assert all(a["executable"] is False for a in tps)
    assert all("tp" in (a["missing"] or "") for a in tps)


def test_tp_alternative_non_executable_without_caps() -> None:
    feas = check_feasibility(
        dense7b(), make_hw(), gpus=1, method="full", seq_len=4096, micro_batch=1, tokens=10**9, caps=None
    )
    tps = [a for a in feas.alternatives if a["change"].startswith("tensor parallelism")]
    assert tps and all(a["executable"] is False for a in tps)
    assert all("unmeasured" in (a["missing"] or "") for a in tps)


def test_tp_alternative_executable_when_caps_allows() -> None:
    caps = FSCapabilities(
        available=True, fs_version="0.0.0",
        train_objectives=("sft",), sharding_strategies=("ddp", "fsdp"),
        backends=("ddp", "fsdp"), executed_axes=("tp", "cp"), refused_axes=("pp", "ep"),
        axes_measured=True,
    )
    feas = check_feasibility(
        dense7b(), make_hw(), gpus=1, method="full", seq_len=4096, micro_batch=1, tokens=10**9, caps=caps
    )
    tps = [a for a in feas.alternatives if a["change"].startswith("tensor parallelism")]
    assert tps and all(a["executable"] is True for a in tps)


def test_unknown_variant_smaller_alternative_is_blocked_not_silent() -> None:
    feas = check_feasibility(
        dense7b(), make_hw(), gpus=1, method="full", seq_len=4096, micro_batch=1, tokens=10**9
    )
    smaller = [a for a in feas.alternatives if a["change"] == "switch to a smaller variant in the same family"]
    assert len(smaller) == 1
    assert smaller[0]["executable"] is False
    assert smaller[0]["missing"]


def test_budget_breach_is_infeasible() -> None:
    hw = make_hw(tflops=1000.0, mfu=0.318)
    feas = check_feasibility(
        dense7b(), hw, gpus=4, method="full", seq_len=2048, micro_batch=1,
        tokens=10**9, budget_gpu_hours=1.0,  # needs ~36.7 gpu-hours
    )
    assert feas.verdict == "infeasible"
    assert any(f["rule_id"] == "TR-FEAS-002" and f["passed"] is False for f in feas.findings)


def test_deadline_breach_is_infeasible() -> None:
    hw = make_hw(tflops=1000.0, mfu=0.318)
    feas = check_feasibility(
        dense7b(), hw, gpus=8, method="full", seq_len=2048, micro_batch=1,
        tokens=10**9, deadline_hours=1.0,  # needs ~4.6 h
    )
    assert feas.verdict == "infeasible"
    assert any(f["rule_id"] == "TR-FEAS-003" for f in feas.findings)


def test_grad_ckpt_alternative_listed_when_base_ran_without_it() -> None:
    feas = check_feasibility(
        dense7b(), make_hw(), gpus=1, method="full", seq_len=4096, micro_batch=1,
        tokens=10**9, grad_ckpt=False,
    )
    assert feas.alternatives[0]["change"] == "enable gradient checkpointing"
    assert feas.alternatives[0]["executable"] is True
    assert feas.alternatives[0]["verdict"] == "infeasible"  # ckpt alone cannot save 112 GB of states
