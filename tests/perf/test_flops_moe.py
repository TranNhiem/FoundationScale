"""Hostile regression tests for MoE FLOPs charging in FlopsModel (#529).

The defect guarded here: flops_per_token charged 6 * non_embedding_parameters
for EVERY expert on a Mixture-of-Experts model when only top-k run per token,
a 6.51x overstatement measured on Qwen3.5-35B-A3B (GB200, 2026-09-20/21).
These tests pin the active-parameter formula, the UNMEASURED reason contract,
the bit-identical dense path, and the torch-free name-walk helpers, using the
measured estate shapes: gemma-4-26B-A4B, gemma-4-31B, Qwen3.5-35B-A3B,
Qwen3.5-122B-A10B and dense Qwen3.5-27B.
"""

from __future__ import annotations

from types import SimpleNamespace

from foundationscale.perf.telemetry import (
    FlopsModel,
    _is_routed_expert_parameter,
    _positive_int,
)


class _Parameter:
    """A torch-free parameter fake exposing only numel, as from_model requires."""

    def __init__(self, elements: int) -> None:
        self._elements = elements

    def numel(self) -> int:
        return self._elements


class Embedding:
    """Named this on purpose: from_model matches modules by class NAME."""

    def __init__(self, parameter: _Parameter) -> None:
        self._parameter = parameter

    def parameters(self, recurse: bool = False) -> list[_Parameter]:
        return [self._parameter]


class _FakeModel:
    """Minimal model-shaped object: config, named_parameters, named_modules."""

    def __init__(
        self,
        config: SimpleNamespace,
        parameters: list[tuple[str, _Parameter]],
        modules: list[tuple[str, object]],
    ) -> None:
        self.config = config
        self._parameters = parameters
        self._modules = modules

    def named_parameters(self) -> list[tuple[str, _Parameter]]:
        return list(self._parameters)

    def named_modules(self) -> list[tuple[str, object]]:
        return list(self._modules)


def test_dense_flops_are_bit_identical_to_pre_529_formula() -> None:
    """Deleting this lets float division or a refactored term creep into dense.

    The dense estimate must be EXACTLY 6N + 12 * L * h * s as one integer,
    because MFU numbers already published were computed from that integer and
    any rounding step moved into the path invalidates comparison to them.
    """
    non_embedding = 26_000_000_000
    layers, hidden, sequence = 34, 2560, 8192
    model = FlopsModel(
        parameters=27_000_000_000,
        non_embedding_parameters=non_embedding,
        layers=layers,
        hidden_size=hidden,
        sequence_length=sequence,
    )
    expected = 6 * non_embedding + 12 * layers * hidden * sequence
    assert model.experts_total is None
    assert model.unmeasured_reason is None
    assert type(model.flops_per_token) is int
    assert model.flops_per_token == expected


def test_null_num_experts_is_dense_not_unknown() -> None:
    """Deleting this lets ``num_experts: null`` flip dense gemma-4-31B to UNMEASURED.

    The real gemma-4-31B text_config states the key with a null value; keying
    on presence instead of value tags it MoE with an unknown count, destroys
    its MFU, and reproduces #529 from the consumer side.
    """
    config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=60,
            hidden_size=5120,
            num_experts=None,
        )
    )
    model = FlopsModel.from_hf_config(
        config,
        sequence_length=4096,
        parameters=31_000_000_000,
        non_embedding_parameters=30_000_000_000,
    )
    assert model is not None
    assert model.experts_total is None
    assert model.experts_active is None
    assert model.unmeasured_reason is None
    expected = 6 * 30_000_000_000 + 12 * 60 * 5120 * 4096
    assert model.flops_per_token == expected


def test_gemma_a4b_128_experts_without_top_k_is_unmeasured() -> None:
    """Deleting this reinstates the silent dense fallback that flattered MFU by ~6x.

    The measured gemma-4-26B-A4B config declares 128 experts and NO per-token
    key; the honest answer is UNMEASURED naming every key that was consulted,
    never a plausible-looking dense figure.
    """
    config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=34,
            hidden_size=2560,
            num_experts=128,
        )
    )
    model = FlopsModel.from_hf_config(
        config,
        sequence_length=8192,
        parameters=26_000_000_000,
        non_embedding_parameters=25_000_000_000,
        routed_expert_parameters=20_000_000_000,
    )
    assert model is not None
    assert model.experts_total == 128
    assert model.experts_active is None
    assert model.unmeasured_reason is not None
    assert model.unmeasured_reason.startswith("UNMEASURED:")
    assert "128 experts" in model.unmeasured_reason
    assert "num_experts_per_tok" in model.unmeasured_reason
    assert "num_experts_per_token" in model.unmeasured_reason
    assert "top_k" in model.unmeasured_reason
    dense = 6 * 25_000_000_000 + 12 * 34 * 2560 * 8192
    assert model.flops_per_token == dense


def test_qwen_35b_a3b_charges_active_experts_only() -> None:
    """Deleting this restores the 6.51x overstatement measured on Qwen3.5-35B-A3B.

    With 256 experts and 8 running per token, the 6N term must cover ACTIVE
    parameters: the result lands strictly below the naive dense figure, at a
    fraction consistent with a 1/32 routing of the routed parameter mass.
    """
    layers, hidden, sequence = 40, 2560, 4096
    non_embedding = 35_000_000_000
    routed = 30_000_000_000
    config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=layers,
            hidden_size=hidden,
            num_experts=256,
            num_experts_per_tok=8,
        )
    )
    model = FlopsModel.from_hf_config(
        config,
        sequence_length=sequence,
        parameters=35_000_000_001,
        non_embedding_parameters=non_embedding,
        routed_expert_parameters=routed,
    )
    assert model is not None
    assert model.experts_total == 256
    assert model.experts_active == 8
    assert model.unmeasured_reason is None
    naive = 6 * non_embedding + 12 * layers * hidden * sequence
    assert model.flops_per_token < naive
    ratio = model.flops_per_token / naive
    assert 0.15 < ratio < 0.25
    assert model.flops_per_token == 40_658_164_800


def test_full_routing_reduces_exactly_to_dense_integer() -> None:
    """Deleting this lets the MoE path diverge from dense when every expert runs.

    experts_active == experts_total must subtract exactly nothing, bit for
    bit; any float residue here shifts dense MFU by a rounding step (#529).
    """
    layers, hidden, sequence = 4, 8, 16
    non_embedding = 10_000
    config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=layers,
            hidden_size=hidden,
            num_experts=8,
            num_experts_per_tok=8,
        )
    )
    model = FlopsModel.from_hf_config(
        config,
        sequence_length=sequence,
        parameters=10_000,
        non_embedding_parameters=non_embedding,
        routed_expert_parameters=7_000,
    )
    assert model is not None
    assert model.unmeasured_reason is None
    expected = 6 * non_embedding + 12 * layers * hidden * sequence
    assert model.flops_per_token == expected
    assert model.flops_per_token == 66_144


def test_declared_experts_with_zero_routed_parameters_are_unmeasured() -> None:
    """Deleting this lets a MoE with no routed count bill the dense figure silently.

    A positive experts_total with routed_expert_parameters == 0 leaves the
    active fraction nothing to act on; the reason must name the missing value.
    """
    config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=12,
            hidden_size=1024,
            num_experts=64,
            num_experts_per_tok=8,
        )
    )
    model = FlopsModel.from_hf_config(
        config,
        sequence_length=2048,
        parameters=5_000_000_000,
        non_embedding_parameters=4_900_000_000,
        routed_expert_parameters=0,
    )
    assert model is not None
    assert model.experts_total == 64
    assert model.experts_active == 8
    assert model.unmeasured_reason is not None
    assert model.unmeasured_reason.startswith("UNMEASURED:")
    assert "routed_expert_parameters" in model.unmeasured_reason
    assert "from_model" in model.unmeasured_reason


def test_zero_or_negative_experts_total_never_divides() -> None:
    """Deleting this removes the divide-by-zero guard on degenerate expert counts.

    A zero experts_total is falsy and must take the dense path untouched; a
    negative one must still return a finite int rather than raise or nan out.
    """
    zero = FlopsModel(
        parameters=10_000,
        non_embedding_parameters=10_000,
        layers=2,
        hidden_size=4,
        sequence_length=8,
        routed_expert_parameters=1_000,
        experts_total=0,
        experts_active=4,
    )
    assert zero.flops_per_token == 60_768
    negative = FlopsModel(
        parameters=10_000,
        non_embedding_parameters=10_000,
        layers=2,
        hidden_size=4,
        sequence_length=8,
        routed_expert_parameters=1_000,
        experts_total=-4,
        experts_active=8,
    )
    result = negative.flops_per_token
    assert type(result) is int
    assert result == 42_768


def test_routed_expert_name_walk_accepts_both_storage_layouts() -> None:
    """Deleting this lets the FUSED expert layout go uncounted again.

    Measured on the real Qwen3.5-35B-A3B checkpoint index: the ``mtp`` head
    stores 768 INDEXED ``experts.<i>.`` tensors, while the language model
    stores every one of its 256 experts per layer FUSED into two unindexed
    stacked tensors, ``mlp.experts.gate_up_proj`` and ``mlp.experts.down_proj``.
    Requiring an index found 0 routed parameters in the language model of a
    model that is overwhelmingly experts by weight, so the whole #529 correction
    silently did nothing there. Both layouts must count.

    The shared expert (runs for EVERY token) and a plain ``mlp.`` path must stay
    always-active, or the dense work is understated.
    """
    prefix = "model.layers.3.mlp"
    # Indexed layout, as stored by the mtp head.
    assert _is_routed_expert_parameter(f"{prefix}.experts.7.gate_proj.weight") is True
    # FUSED layout, as stored by the language model -- the case that regressed.
    assert _is_routed_expert_parameter(f"{prefix}.experts.gate_up_proj") is True
    assert _is_routed_expert_parameter(f"{prefix}.experts.down_proj") is True
    # Always-active mass must never be scaled by the routing fraction.
    assert _is_routed_expert_parameter(f"{prefix}.shared_expert.gate_proj.weight") is False
    assert _is_routed_expert_parameter(f"{prefix}.shared_expert_gate.weight") is False
    assert _is_routed_expert_parameter(f"{prefix}.gate_proj.weight") is False
    # "experts" must be a whole path SEGMENT, not a substring of another name.
    assert _is_routed_expert_parameter(f"{prefix}.expertsy.gate_proj.weight") is False


def test_positive_int_rejects_flags_and_non_positive_counts() -> None:
    """Deleting this lets ``num_experts: true`` or ``0`` pose as an expert count.

    bool is an int subclass yet a flag is not a count -- the rule this module
    already applies to layer and hidden sizes -- and zero or negative values
    are not counts either; only a genuinely positive int qualifies.
    """
    assert _positive_int(SimpleNamespace(num_experts=True), ("num_experts",)) is None
    assert _positive_int(SimpleNamespace(num_experts=0), ("num_experts",)) is None
    assert _positive_int(SimpleNamespace(num_experts=-12), ("num_experts",)) is None
    assert _positive_int(SimpleNamespace(num_experts="8"), ("num_experts",)) is None
    flags = SimpleNamespace(num_experts=True, num_local_experts=16)
    assert _positive_int(flags, ("num_experts", "num_local_experts")) == 16
    missing = SimpleNamespace()
    assert _positive_int(missing, ("num_experts", "num_local_experts")) is None


def test_from_model_counts_only_routed_experts_not_shared() -> None:
    """Deleting this lets shared-expert or dense MLP mass leak into routed counts.

    from_model must judge routing by qualified name: the indexed experts.<i>
    parameters (300 + 300) are routed; the shared expert (400), the plain MLP
    block (500) and the embedding (1000) are always-active or non-embedding,
    or routed_expert_parameters inflates and flops_per_token is understated.
    """
    embedding_weight = _Parameter(1_000)
    model = _FakeModel(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                num_hidden_layers=1,
                hidden_size=4,
                num_experts=2,
                num_experts_per_tok=1,
            )
        ),
        parameters=[
            ("embed_tokens.weight", embedding_weight),
            ("layers.0.mlp.experts.0.gate_proj.weight", _Parameter(300)),
            ("layers.0.mlp.experts.1.gate_proj.weight", _Parameter(300)),
            ("layers.0.mlp.shared_expert.gate_proj.weight", _Parameter(400)),
            ("layers.0.mlp.gate_proj.weight", _Parameter(500)),
        ],
        modules=[("embed_tokens", Embedding(embedding_weight))],
    )
    built = FlopsModel.from_model(model, sequence_length=8)
    assert built is not None
    assert built.parameters == 2_500
    assert built.non_embedding_parameters == 1_500
    assert built.routed_expert_parameters == 600
    assert built.experts_total == 2
    assert built.experts_active == 1
    assert built.unmeasured_reason is None
    assert built.flops_per_token == 7_584
