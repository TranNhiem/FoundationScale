# SPDX-License-Identifier: Apache-2.0
"""Controls for #498: an expert count nested in a sub-config must still be read.

`_declare_checkpoint` decides dense-vs-MoE from two sources that must agree: the
config's expert-count key, and whether any parameter name mentions experts. The
config half was a flat `getattr` over the top level, and every multimodal config
is composite -- a VLM states its text tower's expert count on `text_config`, not
on itself.

Alone that is a missed key. Paired with an adapter-only checkpoint it is a false
declaration: no LoRA tensor sits in the expert namespace, so the tensor half is
legitimately silent too, both sources go quiet for unrelated reasons, and a
128-expert model is POSITIVELY declared dense with `num_experts=0`. The expert
gates are then not applicable and report nothing missing. MEASURED on GB200: a
26B MoE VLM with `text_config.num_experts=128` had its manifest record
`dense: config declares none of [...]`.

The last two tests are the pair that matters. One requires the adapter-only MoE
to reach UNKNOWN rather than dense; the other requires a genuinely dense
composite model to still be declared dense, so the fix cannot be "call
everything MoE".
"""

from __future__ import annotations

from typing import Any

from foundationscale.train.loop import _config_expert_counts, _declare_checkpoint


class _Tensor:
    def __init__(self, n: int = 16, width: int = 2) -> None:
        self._n = n
        self._w = width

    def numel(self) -> int:
        return self._n

    def element_size(self) -> int:
        return self._w


class _Config:
    """A config with no `to_dict()`, so the attribute-walk fallback is used."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _SerialisableConfig:
    """A config that serialises, which is the path a real transformers config takes.

    The stub above has no `to_dict()`, so every control written against it
    exercises only the fallback. Production does not use the fallback, so a
    suite that tested only the stub would be measuring a code path operators
    never reach.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return self._payload


class _Model:
    def __init__(self, names: list[str], *, config: Any) -> None:
        self._state = {n: _Tensor() for n in names}
        self.config = config

    def state_dict(self) -> dict[str, _Tensor]:
        return self._state


def _adapter_names(count: int = 110) -> list[str]:
    """Adapter-only tensor names: none carries an expert path segment.

    This is what a healthy LoRA checkpoint over an MoE base looks like, which is
    why the tensor half of the two-source contract cannot be the thing that
    catches a misread config here.
    """
    out: list[str] = []
    for layer in range(count // 2):
        base = f"base_model.model.model.layers.{layer}.self_attn.q_proj"
        out += [f"{base}.lora_A.weight", f"{base}.lora_B.weight"]
    return out


# --------------------------------------------------------------------------
# The reader itself.
# --------------------------------------------------------------------------


def test_a_nested_expert_count_is_found_and_the_flat_read_would_miss_it() -> None:
    """MUST-FIRE: the control must reproduce the defect before claiming the fix."""
    config = _Config(
        model_type="gemma4_fixture",
        text_config=_Config(enable_moe_block=True, num_experts=128, top_k_experts=8),
        vision_config=_Config(hidden_size=8),
    )
    # The read that shipped. If this ever finds the count, the fix below is
    # being credited for something the old code already did.
    assert getattr(config, "num_experts", None) is None

    assert _config_expert_counts(config) == {"text_config.num_experts": 128}


def test_a_serialising_config_is_read_through_to_dict() -> None:
    config = _SerialisableConfig(
        {"model_type": "x", "text_config": {"num_experts": 128, "enable_moe_block": True}}
    )
    assert _config_expert_counts(config) == {"text_config.num_experts": 128}


def test_a_top_level_expert_count_is_still_found() -> None:
    assert _config_expert_counts(_Config(num_local_experts=8)) == {"num_local_experts": 8}


def test_a_boolean_flag_is_not_read_as_a_count() -> None:
    """`enable_moe_block=True` is a flag; `isinstance(True, int)` is a trap.

    Were a bool accepted, a config naming one of the count keys as a flag would
    declare a one-expert layer and the byte gate would confirm a checkpoint many
    times short of its denominator.
    """
    config = _Config(text_config=_Config(enable_moe_block=True, num_experts=True))
    assert _config_expert_counts(config) == {}


def test_two_towers_declaring_different_counts_both_surface() -> None:
    """Disagreement must reach the caller, which refuses to adjudicate it.

    Hardcoding a `text_config`-first scope rule would return 128 and hide the
    vision tower entirely, turning a disagreement into a confident answer.
    """
    config = _Config(
        text_config=_Config(num_experts=128),
        vision_config=_Config(num_experts=16),
    )
    assert _config_expert_counts(config) == {
        "text_config.num_experts": 128,
        "vision_config.num_experts": 16,
    }


def test_the_walk_terminates_on_a_self_referential_config() -> None:
    config = _Config(num_experts=4)
    config.self_reference = config  # type: ignore[attr-defined]
    assert _config_expert_counts(config) == {"num_experts": 4}


# --------------------------------------------------------------------------
# The declaration the gates actually consume.
# --------------------------------------------------------------------------


def test_an_adapter_only_moe_checkpoint_is_not_declared_dense() -> None:
    """The GB200 observation, as a control.

    Both halves of the two-source contract are quiet, for unrelated reasons. The
    required outcome is UNKNOWN and fail-closed, not a positive dense
    declaration: `num_experts=0` tells the gates the expert properties are not
    applicable, and an un-run gate reports nothing missing.
    """
    config = _Config(
        model_type="gemma4_fixture",
        text_config=_Config(enable_moe_block=True, num_experts=128),
    )
    declared, notes = _declare_checkpoint(_Model(_adapter_names(), config=config))

    assert declared.num_experts is None, declared.moe_layer_basis
    assert "UNKNOWN" in (declared.moe_layer_basis or "")
    assert "128" in (declared.moe_layer_basis or "")
    assert notes["declaration.config_expert_keys"] == "text_config.num_experts=128"


def test_a_dense_composite_model_is_still_declared_dense() -> None:
    """The fix must not answer MoE for everything nested.

    Without this the suite would pass on a change that simply stopped minting
    dense, which would block every honest dense save instead.
    """
    config = _Config(model_type="gemma4_fixture", text_config=_Config(hidden_size=8))
    names = ["model.embed_tokens.weight", "model.layers.0.mlp.gate_proj.weight"]
    declared, notes = _declare_checkpoint(_Model(names, config=config))

    assert declared.num_experts == 0, declared.moe_layer_basis
    assert "dense" in (declared.moe_layer_basis or "")
    assert notes["declaration.config_expert_keys"] == "(none present)"
