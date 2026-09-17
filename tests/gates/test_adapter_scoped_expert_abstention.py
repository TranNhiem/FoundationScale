# SPDX-License-Identifier: Apache-2.0
"""Controls for the adapter-scoped door on the two expert gates (#471/#498).

#498 taught the declaration to read an expert count nested in a sub-config, so a
26B MoE VLM stopped being declared `dense`. That fix alone would have made
things worse for one real workflow: a LoRA-over-MoE checkpoint contains no
expert tensors BY DESIGN, so with an honest UNKNOWN count it fell through to the
`num_experts is None` door, which returns VACUOUS -- and VACUOUS blocks. Every
healthy adapter save over an MoE base would have stopped.

The honest verdict is neither the old vacuous PASS (reached by claiming a
128-expert model is dense) nor a block. It is a named abstention: the expert
weights are outside this run's declared scope, and the artifact itself says so
because every tensor it holds is adapter-namespaced.

The load-bearing test here is the gutted-checkpoint one. The UNKNOWN door exists
to catch an MoE checkpoint whose experts were stripped or never written, and a
new door that let that through would be a far worse defect than the one being
fixed. It is kept in the suite as the thing the door must never do.
"""

from __future__ import annotations

import pytest

from foundationscale.gates.checkpoint_gates import (
    CheckpointGateContext,
    ExpertByteVolumeGate,
    ExpertDistinctnessGate,
    TensorMeta,
    _is_adapter_only_artifact,
)
from foundationscale.gates.core import AbstentionKind, Verdict

GATES = (ExpertDistinctnessGate, ExpertByteVolumeGate)

PEFT_NAMES = [
    f"base_model.model.model.layers.{index}.self_attn.q_proj.lora_A.default.weight"
    for index in range(4)
]
BRIDGE_NAMES = [
    f"decoder.layers.{index}.self_attention.linear_qkv.adapter.linear_in.weight"
    for index in range(4)
]
GUTTED_NAMES = [
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.layers.0.self_attn.q_proj.weight",
]


def _context(names: list[str], num_experts: int | None) -> CheckpointGateContext:
    tensors = tuple(
        TensorMeta(fqn=name, shape=(8, 8), dtype="BF16", storage_id=f"s{index}")
        for index, name in enumerate(names)
    )
    return CheckpointGateContext(
        tensors=tensors,
        declared_fqns=tuple(names),
        num_experts=num_experts,
        num_moe_layers=None,
        expected_expert_bytes=None,
        origin="test",
    )


@pytest.mark.parametrize("gate_cls", GATES)
@pytest.mark.parametrize(
    ("label", "names"),
    [("hf-peft", PEFT_NAMES), ("megatron-bridge", BRIDGE_NAMES)],
)
def test_an_adapter_only_checkpoint_abstains_by_name(
    gate_cls: type, label: str, names: list[str]
) -> None:
    """Both adapter dialects, both gates. Not blocked, and not credited either."""
    result = gate_cls().run(_context(names, None))

    assert result.verdict is Verdict.SKIP, f"{label}: {result.detail}"
    assert result.abstention is AbstentionKind.NOT_APPLICABLE
    assert "adapter-only checkpoint" in str(result.detail)


@pytest.mark.parametrize("gate_cls", GATES)
def test_the_abstention_denies_the_dense_inference_in_words(gate_cls: type) -> None:
    """The reason must not be readable as "the model has no experts".

    That inference over an absence of evidence is this repository's founding
    incident, and the previous behaviour reached the same SKIP by making exactly
    it. Identical verdict, opposite epistemics -- so the distinction has to live
    in something a reader can check.
    """
    detail = str(gate_cls().run(_context(PEFT_NAMES, None)).detail)
    assert "NOT a claim that the model is dense" in detail
    assert "UNMEASURED" in detail


@pytest.mark.parametrize("gate_cls", GATES)
def test_a_gutted_checkpoint_still_blocks(gate_cls: type) -> None:
    """The property the new door must never weaken.

    An MoE checkpoint whose experts were stripped, or never written, reaches the
    same "no expert tensors" branch. It must still block: this is the case the
    UNKNOWN door was added for.
    """
    result = gate_cls().run(_context(GUTTED_NAMES, None))
    assert result.verdict is Verdict.VACUOUS, result.detail


@pytest.mark.parametrize("gate_cls", GATES)
def test_one_base_model_tensor_is_enough_to_close_the_door(gate_cls: type) -> None:
    """The `all` quantifier is the safety, so it gets its own control.

    A checkpoint that is mostly adapter but carries a base-model tensor is not
    adapter-SCOPED, and the expert question is answerable against it.
    """
    mixed = [*PEFT_NAMES, "model.embed_tokens.weight"]
    assert gate_cls().run(_context(mixed, None)).verdict is Verdict.VACUOUS


@pytest.mark.parametrize("gate_cls", GATES)
def test_a_declared_dense_model_keeps_its_original_reason(gate_cls: type) -> None:
    """The door is guarded on UNKNOWN, so the positive-dense path is untouched."""
    result = gate_cls().run(_context(PEFT_NAMES, 0))
    assert result.verdict is Verdict.SKIP
    assert "dense model" in str(result.detail)


def test_an_empty_artifact_is_not_adapter_only() -> None:
    """`all([])` is True; an empty artifact is NOTHING, not adapter-scoped."""
    assert _is_adapter_only_artifact(()) is False


def test_the_predicate_accepts_both_dialects_and_rejects_base_weights() -> None:
    def metas(names: list[str]) -> tuple[TensorMeta, ...]:
        return tuple(
            TensorMeta(fqn=n, shape=(8, 8), dtype="BF16", storage_id=f"s{i}")
            for i, n in enumerate(names)
        )

    assert _is_adapter_only_artifact(metas(PEFT_NAMES)) is True
    assert _is_adapter_only_artifact(metas(BRIDGE_NAMES)) is True
    assert _is_adapter_only_artifact(metas(GUTTED_NAMES)) is False
