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
from foundationscale.gates.checkpoint_gates import CheckpointGateContext, TensorMeta
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


_EXPERT_GATES_FOR_MANIFESTLESS = (
    "checkpoint.expert_distinctness",
    "checkpoint.expert_bytes",
)


def _manifestless_ctx() -> CheckpointGateContext:
    """Exactly what from_path builds beside a manifestless checkpoint: all None."""
    tensors = tuple(
        TensorMeta(
            fqn=f"layers.{layer}.attention.self_attention.linear_proj.weight",
            shape=(256, 512),
            dtype="float32",
            storage_id=f"manifestless:L{layer}",
        )
        for layer in range(2)
    )
    return CheckpointGateContext(
        tensors=tensors,
        declared_fqns=None,
        num_experts=None,
        num_moe_layers=None,
        expected_expert_bytes=None,
        origin="test://manifestless",
    )


def test_manifestless_context_vacuous_blocks_expert_gates_never_dense_skips() -> None:
    """num_experts=None means UNKNOWN: zero experts over an unknown declaration
    is VACUOUS; only an explicit 0 is a dense model and may SKIP.

    A gutted MoE artifact (experts stripped or never written) without a manifest
    is selection-identical to a true dense model, and both expert gates answered
    it with a non-blocking SKIP asserting a declaration no context made
    ("context declares no experts"). The flipping assertions are the two
    `verdict is Verdict.VACUOUS` checks: on the current tree both gates return
    Verdict.SKIP from the `if not c.num_experts:` door. The negative controls
    pin the doors either side of the fix — explicit num_experts=0 still earns
    SKIP, and declared-MoE-with-empty-experts keeps its pre-existing VACUOUS
    wording — so a gate that merely started FAILING on everything cannot pass.
    """
    manifestless = _manifestless_ctx()

    distinctness = REGISTRY.get("checkpoint.expert_distinctness").run(manifestless)
    assert distinctness.verdict is Verdict.VACUOUS, distinctness.detail  # was SKIP
    assert distinctness.blocking
    assert distinctness.coverage.checked == 0  # doctrine 1: the zero is named
    assert "does not declare an expert count" in distinctness.detail
    assert "dense" not in distinctness.detail.lower()

    bytes_gate = REGISTRY.get("checkpoint.expert_bytes").run(manifestless)
    assert bytes_gate.verdict is Verdict.VACUOUS, bytes_gate.detail  # was SKIP
    assert bytes_gate.blocking
    assert bytes_gate.coverage.checked == 0
    assert "does not declare an expert count" in bytes_gate.detail

    # Negative control A: an EXPLICIT dense declaration keeps the original SKIP.
    declared_dense = CheckpointGateContext(
        tensors=manifestless.tensors,
        declared_fqns=tuple(t.fqn for t in manifestless.tensors),
        num_experts=0,
        num_moe_layers=0,
        expected_expert_bytes=0,
        origin="test://declared-dense",
    )
    for gate_id in _EXPERT_GATES_FOR_MANIFESTLESS:
        dense_result = REGISTRY.get(gate_id).run(declared_dense)
        assert dense_result.verdict is Verdict.SKIP, dense_result.detail
        assert not dense_result.blocking

    # Negative control B: declared MoE with an absent expert set keeps the
    # pre-existing enforced-VACUOUS path and its incident wording.
    empty = REGISTRY.get("checkpoint.expert_distinctness").run(fx.empty_expert_set_ctx())
    assert empty.verdict is Verdict.VACUOUS, empty.detail
    assert "model declares 128 experts" in empty.detail

    # The composite now names all three blocked sub-gates; today the two expert
    # gates SKIP past the failure report and only completeness is named.
    composite = REGISTRY.get(_FIRST_SAVE_GATE_ID).run(manifestless)
    assert composite.verdict is Verdict.FAIL, composite.detail
    for gate_id in (*_EXPERT_GATES_FOR_MANIFESTLESS, "checkpoint.save_complete"):
        assert f"{gate_id}={Verdict.VACUOUS.value}" in composite.detail

    # The shipped control fixtures must exhibit the same blocking pair; this
    # reference is why the controls job cannot lose the regression silently.
    shipped = fx.manifestless_moe_ctx()
    assert REGISTRY.get("checkpoint.expert_distinctness").run(shipped).verdict is Verdict.VACUOUS
    assert REGISTRY.get("checkpoint.expert_bytes").run(shipped).verdict is Verdict.VACUOUS


def test_unpriceable_dtype_blocks_byte_gate_and_qualifies_stacked_abstention() -> None:
    """An unrecognized dtype must NOT be priced at a silent 4 bytes/element.

    float8_e4m3fn is a dtype dcp_meta parses from real safetensors headers, but
    _DTYPE_BYTES does not price it. On the current tree implied_nbytes guesses
    4: this context's manifest declares the same (guessed) volume, so the byte
    gate PASSes an honest manifest against an inflated artifact, and the
    distinctness abstention claims "sibling projections price consistently
    across layers". Flipping assertions: `bytes_gate.verdict is VACUOUS` (PASS
    today) and `"price consistently across layers" not in distinctness.detail`
    (present today). The bfloat16 twin — identical names, shapes and storages —
    is the negative control: PASS and the full pricing claim survive intact.
    """
    legacy_default_width = 4  # the guess the old implied_nbytes made, spelled out
    f8_tensors = tuple(
        TensorMeta(
            fqn=f"model.language_model.layers.{layer}.experts.{projection}",
            shape=(8, *inner),
            dtype="float8_e4m3fn",
            storage_id=f"f8:L{layer}:{projection}",
        )
        for layer in range(2)
        for projection, inner in (("gate_up_proj", (16, 32)), ("down_proj", (32, 16)))
    )
    f8_ctx = CheckpointGateContext(
        tensors=f8_tensors,
        declared_fqns=tuple(t.fqn for t in f8_tensors),
        num_experts=8,
        num_moe_layers=2,
        expected_expert_bytes=sum(math.prod(t.shape) for t in f8_tensors) * legacy_default_width,
        origin="test://float8-stacked",
    )

    bytes_gate = REGISTRY.get("checkpoint.expert_bytes").run(f8_ctx)
    assert bytes_gate.verdict is Verdict.VACUOUS, bytes_gate.detail  # was PASS
    assert bytes_gate.blocking
    assert bytes_gate.coverage.checked == 0
    assert "float8_e4m3fn" in bytes_gate.detail
    assert bytes_gate.evidence.get("unpriceable_dtypes") == ["float8_e4m3fn"]
    assert f8_tensors[0].fqn in bytes_gate.evidence.get("unpriceable_fqns", [])

    distinctness = REGISTRY.get("checkpoint.expert_distinctness").run(f8_ctx)
    assert distinctness.verdict is Verdict.SKIP, distinctness.detail
    assert "price consistently across layers" not in distinctness.detail  # present today
    assert "float8_e4m3fn" in distinctness.detail
    assert "shapes could be compared" in distinctness.detail  # what WAS examined is named

    healthy = fx.stacked_hf_moe_ctx()
    healthy_bytes = REGISTRY.get("checkpoint.expert_bytes").run(healthy)
    assert healthy_bytes.verdict is Verdict.PASS, healthy_bytes.detail
    healthy_distinct = REGISTRY.get("checkpoint.expert_distinctness").run(healthy)
    assert healthy_distinct.verdict is Verdict.SKIP, healthy_distinct.detail
    assert "price consistently across layers" in healthy_distinct.detail

    shipped = fx.unpriceable_dtype_ctx()
    assert REGISTRY.get("checkpoint.expert_bytes").run(shipped).verdict is Verdict.VACUOUS
    shipped_distinct = REGISTRY.get("checkpoint.expert_distinctness").run(shipped)
    assert shipped_distinct.verdict is Verdict.SKIP
    assert "price consistently across layers" not in shipped_distinct.detail


def test_zero_declared_moe_layers_never_renders_an_expected_of_zero() -> None:
    """A manifest declaring ZERO MoE layers declares no expert population, so
    the expert-tensor denominator is absent (None), not 0.

    On the current tree _declared_tensor_count guards only `is None`, computes
    0 * 2 = 0, and this checkpoint renders coverage as 2/0 — an unqualified
    count in a denominator's clothes. The byte verdict itself is gated by
    expected_expert_bytes (guarded separately), so this pins the rendering
    exactly as the defect report sized it. The flipping assertion is
    `result.coverage.expected is None` (0 today); checked stays 2 and the
    volume verdict stays PASS so a blanket-failing gate cannot pass either.
    """
    tensors = tuple(
        TensorMeta(
            fqn=f"model.language_model.layers.{layer}.experts.down_proj",
            shape=(8, 32, 16),
            dtype="bfloat16",
            storage_id=f"zero-layer:L{layer}",
        )
        for layer in range(2)
    )
    ctx = CheckpointGateContext(
        tensors=tensors,
        declared_fqns=tuple(t.fqn for t in tensors),
        num_experts=8,
        num_moe_layers=0,
        expected_expert_bytes=sum(int(t.implied_nbytes or 0) for t in tensors),
        origin="test://zero-declared-layers",
    )

    result = REGISTRY.get("checkpoint.expert_bytes").run(ctx)

    assert result.verdict is Verdict.PASS, result.detail
    assert result.coverage.checked == 2
    assert result.coverage.expected is None  # was 0: the N/0 rendering
