"""Mutation regressions for checkpoint-gate rules that were exercised but never pinned.

The ordinary controls establish only that a defective input produces a blocking verdict.
These tests pin the verdict, denominator and causal message that establish *why* the
input blocked, so an unrelated UNDERCOVERED or VACUOUS result cannot launder a mutated
rule back into the green suite.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from test_checkpoint_bridge import (
    EXPECTED_EXPERT_BYTES,
    _install_manifest,
    _ManifestStub,
    _write_healthy_moe_checkpoint,
)

from foundationscale.gates import fixtures as fx
from foundationscale.gates.checkpoint_gates import CheckpointGateContext
from foundationscale.gates.core import REGISTRY, Verdict

_INCIDENT_ACTUAL_BYTES = 5_710_000_000
_INCIDENT_DECLARED_BYTES = 45_700_000_000
_INCIDENT_BYTE_RATIO = _INCIDENT_ACTUAL_BYTES / _INCIDENT_DECLARED_BYTES
_SCALED_DECLARED_BYTES = round(EXPECTED_EXPERT_BYTES / _INCIDENT_BYTE_RATIO)

_HEALTHY_NUM_EXPERTS = 8
_HEALTHY_MOE_LAYERS = 2
_HEALTHY_EXPERT_TENSORS = 4

_FIRST_SAVE_GATE_ID = "checkpoint.first_save"
_FIRST_SAVE_SUBGATE_IDS = (
    "checkpoint.expert_distinctness",
    "checkpoint.expert_bytes",
    "checkpoint.save_complete",
)


@pytest.fixture
def torch_mod() -> Any:
    """Import torch only for the one test that needs bytes on disk."""
    return pytest.importorskip("torch")


def test_declared_byte_budget_blocks_checkpoint_at_incident_byte_ratio(
    tmp_path: Path,
    torch_mod: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An eighth of the declared expert bytes must FAIL, not merely block accidentally.

    The manifest used here declares 15,367 expert bytes for a real on-disk checkpoint
    holding 1,920. That is the incident's 5.71 GB / 45.70 GB ratio at fixture scale.
    If the deficit threshold is loosened to 99.9%, the gate reaches its success branch
    because the coverage denominator is deliberately complete: this test either proves
    a byte-volume FAIL, or proves nothing about byte volume at all.

    The positive control declares the same checkpoint's actual byte count. A pass there
    keeps the negative result from being explained by a gate that fires unconditionally.
    """
    declared_fqns = _write_healthy_moe_checkpoint(tmp_path, torch_mod)
    _install_manifest(
        monkeypatch,
        _ManifestStub(
            declared_fqns=declared_fqns,
            num_experts=_HEALTHY_NUM_EXPERTS,
            num_moe_layers=_HEALTHY_MOE_LAYERS,
            expected_expert_bytes=_SCALED_DECLARED_BYTES,
        ),
    )

    ctx = CheckpointGateContext.from_path(tmp_path)
    present_expert_bytes = sum(
        tensor.implied_nbytes
        for tensor in ctx.tensors
        if tensor.kind == "tensor" and ".linear_fc" in tensor.fqn
    )
    assert present_expert_bytes == EXPECTED_EXPERT_BYTES
    present_ratio = present_expert_bytes / _SCALED_DECLARED_BYTES
    assert math.isclose(present_ratio, _INCIDENT_BYTE_RATIO, rel_tol=1e-4)

    result = REGISTRY.get("checkpoint.expert_bytes").run(ctx)

    assert result.verdict is Verdict.FAIL, result.detail
    assert result.coverage.checked == _HEALTHY_EXPERT_TENSORS
    assert result.coverage.expected == _HEALTHY_EXPERT_TENSORS
    assert result.evidence["implied_expert_bytes"] == EXPECTED_EXPERT_BYTES
    assert result.evidence["expected_expert_bytes"] == _SCALED_DECLARED_BYTES
    assert result.evidence["ratio"] == pytest.approx(_INCIDENT_BYTE_RATIO, abs=0.001)
    assert f"{EXPECTED_EXPERT_BYTES:,}" in result.detail
    assert f"{_SCALED_DECLARED_BYTES:,}" in result.detail

    _install_manifest(
        monkeypatch,
        _ManifestStub(
            declared_fqns=declared_fqns,
            num_experts=_HEALTHY_NUM_EXPERTS,
            num_moe_layers=_HEALTHY_MOE_LAYERS,
            expected_expert_bytes=EXPECTED_EXPERT_BYTES,
        ),
    )
    full_volume_result = REGISTRY.get("checkpoint.expert_bytes").run(
        CheckpointGateContext.from_path(tmp_path)
    )

    assert full_volume_result.verdict is Verdict.PASS, full_volume_result.detail
    assert full_volume_result.coverage.checked == _HEALTHY_EXPERT_TENSORS
    assert full_volume_result.coverage.expected == _HEALTHY_EXPERT_TENSORS


def test_expert_coverage_denominator_counts_both_fused_projections() -> None:
    """Four MoE layers imply eight expert-weight tensors: fc1 and fc2 on every layer.

    If the denominator counted only one projection per layer, a checkpoint holding the
    four ``fc1`` weights would be reported as though the four absent ``fc2`` weights
    had also been inspected. That is precisely the condition core's vacuity machinery
    is supposed to reject, so this test pins both the returned ``Coverage.expected`` and
    the resulting UNDERCOVERED verdict.

    The full fixture is the positive control: all eight implied tensors are present
    and must pass with coverage 8/8.
    """
    full_ctx = fx.healthy_fused_moe_ctx(
        num_experts=_HEALTHY_NUM_EXPERTS,
        num_layers=4,
    )
    fc1_only = tuple(
        tensor for tensor in full_ctx.tensors if tensor.fqn.endswith("linear_fc1.weight")
    )
    assert len(fc1_only) == 4

    half_ctx = CheckpointGateContext(
        tensors=fc1_only,
        declared_fqns=full_ctx.declared_fqns,
        num_experts=full_ctx.num_experts,
        num_moe_layers=full_ctx.num_moe_layers,
        expected_expert_bytes=sum(tensor.implied_nbytes for tensor in fc1_only),
        origin=f"{full_ctx.origin}/fc1-only",
    )

    gate = REGISTRY.get("checkpoint.expert_bytes")
    full_result = gate.run(full_ctx)
    half_result = gate.run(half_ctx)

    assert full_result.verdict is Verdict.PASS, full_result.detail
    assert full_result.coverage.checked == 8
    assert full_result.coverage.expected == 8

    assert half_result.verdict is Verdict.UNDERCOVERED, half_result.detail
    assert half_result.blocking
    assert half_result.coverage.checked == 4
    assert half_result.coverage.expected == 8
    assert "4 of 8" in half_result.detail


def test_first_save_composite_preserves_all_subgate_failures() -> None:
    """Three positive sub-gate FAILs must remain three named failures in one report.

    A composite may not collapse distinctness, byte-volume and completeness failures
    into success, a sample report, or a generic VACUOUS result that no longer names
    which first-save checks found defects. In the incident configuration all three
    gates positively FAIL, and the composite's own message must carry all three ids
    so save #1 is stopped for visible reasons.

    The healthy fixture is the opposite control: all three sub-gates establish their
    facts and the composite must pass with coverage 3/3.
    """
    corrupt = fx.aliased_local_names_ctx()
    sub_results = {
        gate_id: REGISTRY.get(gate_id).run(corrupt) for gate_id in _FIRST_SAVE_SUBGATE_IDS
    }
    for gate_id, sub_result in sub_results.items():
        assert sub_result.verdict is Verdict.FAIL, f"{gate_id}: {sub_result.detail}"

    composite_result = REGISTRY.get(_FIRST_SAVE_GATE_ID).run(corrupt)

    assert composite_result.verdict is Verdict.FAIL, composite_result.detail
    assert composite_result.blocking
    for gate_id in _FIRST_SAVE_SUBGATE_IDS:
        assert f"{gate_id}={Verdict.FAIL.value}" in composite_result.detail

    # Per-expert layout: all three sub-gates can actually reach a verdict, so a
    # clean artifact promotes on a full 3/3 denominator.
    healthy_result = REGISTRY.get(_FIRST_SAVE_GATE_ID).run(fx.healthy_sharded_moe_ctx())

    assert healthy_result.verdict is Verdict.PASS, healthy_result.detail
    assert healthy_result.coverage.checked == len(_FIRST_SAVE_SUBGATE_IDS)
    assert healthy_result.coverage.expected == len(_FIRST_SAVE_SUBGATE_IDS)

    # Stacked/fused layout: distinctness cannot be established from metadata at
    # all, so the composite must report 2/3 and block. A composite that promoted
    # a stacked first save would be claiming a property nothing examined — the
    # same shape as the incident this gate exists to catch, one level up.
    stacked_result = REGISTRY.get(_FIRST_SAVE_GATE_ID).run(fx.healthy_fused_moe_ctx())

    assert stacked_result.verdict is Verdict.UNDERCOVERED, stacked_result.detail
    assert stacked_result.blocking
    assert stacked_result.coverage.checked == len(_FIRST_SAVE_SUBGATE_IDS) - 1
    assert stacked_result.coverage.expected == len(_FIRST_SAVE_SUBGATE_IDS)
