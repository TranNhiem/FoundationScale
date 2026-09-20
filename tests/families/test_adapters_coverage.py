"""Hostile tests for layer-position coverage announcements.

The defect these pin down, measured 2026-09-20 on GB200: a conventional LoRA
declaration against Qwen3.5 -- a 3:1 hybrid in which only one layer in four
carries ``self_attn.{q,k,v,o}_proj`` -- selected exactly
``full_attention * 4`` modules on three model sizes, printed a plausible module
count, saved a checkpoint, and adapted ONE QUARTER of the network's depth with
nothing saying so. A module count cannot see that; a coverage line can. Every
test below exists to fail loudly if coverage goes back to being silent,
vacuous, or wrong about what a layer position is.

Skip-free and torch-free by construction, like the selection code it tests.
"""

from __future__ import annotations

from collections.abc import Iterable

from foundationscale.families.adapters import (
    _layer_key,
    select_adapter_modules,
)
from foundationscale.families.registry import REGISTRY, FamilySpec

ATTENTION_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")


class Linear:
    """Stands in for an adaptable projection."""


class LayerNorm:
    """Stands in for a module the predicate rejects."""


def _plain_linear(module: object) -> bool:
    return type(module) is Linear


def _spec(name: str) -> FamilySpec:
    for spec in REGISTRY:
        if spec.name == name:
            return spec
    raise AssertionError(f"registry has no family {name!r}")


def _coverage_lines(lines: Iterable[str], language: str) -> list[str]:
    quoted = repr(language)
    # "declared target leaves reached", not "indexed layer position": the
    # UNMEASURABLE line says "no integer-indexed layer positions", so the looser
    # marker matches the one line this helper exists to exclude and the
    # zero-position test passes vacuously.
    return [line for line in lines if "declared target leaves reached" in line and quoted in line]


def test_qwen35_hybrid_says_sixteen_of_sixtyfour() -> None:
    """Breaks: the measured Qwen3.5 defect -- 16-of-64 depth coverage reported
    (or not reported) as anything else -- ships again undetected."""
    graph: list[tuple[str, object]] = []
    for layer in range(64):
        if layer % 4 == 0:  # 16 full-attention positions: 0, 4, ..., 60
            for leaf in ATTENTION_LEAVES:
                graph.append((f"model.layers.{layer}.self_attn.{leaf}", Linear()))
        else:  # 48 linear-attention positions, adaptable but undeclared
            graph.append((f"model.layers.{layer}.linear_attn.in_proj_qkv", Linear()))
            graph.append((f"model.layers.{layer}.linear_attn.out_proj", Linear()))

    selected, lines = select_adapter_modules(graph, _spec("qwen3.5"), _plain_linear)

    # The module count that lied: 64 selected modules, exactly full_attention*4.
    assert len(selected) == 64
    coverage = _coverage_lines(lines, "model.layers")
    assert len(coverage) == 1
    assert "reached 16 of 64 indexed layer position(s) under 'model.layers'" in coverage[0]
    assert "48 position(s) hold adaptable modules" in coverage[0]
    assert "train no adapter" in coverage[0]


def test_gemma4_full_coverage_is_announced_not_silent() -> None:
    """Breaks: the full-coverage line disappears, making measured-and-complete
    indistinguishable from never-measured (PASS vs VACUOUS)."""
    graph: list[tuple[str, object]] = []
    for layer in range(8):
        for leaf in ATTENTION_LEAVES:
            graph.append((f"model.language_model.layers.{layer}.self_attn.{leaf}", Linear()))

    selected, lines = select_adapter_modules(graph, _spec("gemma4"), _plain_linear)

    assert len(selected) == 8 * len(ATTENTION_LEAVES)
    coverage = _coverage_lines(lines, "model.language_model")
    assert len(coverage) == 1
    expected = (
        "declared target leaves reached all 8 indexed layer position(s) "
        "under 'model.language_model'"
    )
    assert expected in coverage[0]
    assert "hold adaptable modules" not in coverage[0]


def test_first_integer_wins_for_moe_expert_paths() -> None:
    """Breaks: the expert index inside a layer is counted as depth, turning a
    one-layer model into an eight-layer one and corrupting every MoE coverage
    denominator."""
    spec = FamilySpec(
        name="moe",
        model_types=("moe_test",),
        language_prefixes=("model.layers",),
        tower_prefixes=(),
        adapter_leaf_modules=("gate_proj",),
        expert_count_path=(),
    )
    graph: list[tuple[str, object]] = [
        ("model.layers.3.mlp.experts.7.gate_proj", Linear()),
    ]
    assert _layer_key("model.layers.3.mlp.experts.7.gate_proj") == "model.layers.3"

    selected, lines = select_adapter_modules(graph, spec, _plain_linear)

    assert selected == ("model.layers.3.mlp.experts.7.gate_proj",)
    coverage = _coverage_lines(lines, "model.layers")
    assert len(coverage) == 1
    # Position count, not string luck: exactly ONE indexed position exists.
    assert "reached all 1 indexed layer position(s)" in coverage[0]
    assert "reached all 8" not in coverage[0]


def test_positions_with_nothing_adaptable_are_not_charged_to_the_declaration() -> None:
    """Breaks: pure norm stacks count against coverage, manufacturing false
    gaps that punish a declaration for layers it could never have adapted."""
    spec = FamilySpec(
        name="normyl",
        model_types=("norm_test",),
        language_prefixes=("model.layers",),
        tower_prefixes=(),
        adapter_leaf_modules=("q_proj",),
        expert_count_path=(),
    )
    graph: list[tuple[str, object]] = [
        ("model.layers.0.self_attn.q_proj", Linear()),
        ("model.layers.1.input_layernorm", LayerNorm()),
        ("model.layers.1.post_attention_layernorm", LayerNorm()),
    ]

    _, lines = select_adapter_modules(graph, spec, _plain_linear)

    coverage = _coverage_lines(lines, "model.layers")
    assert len(coverage) == 1
    assert "reached all 1 indexed layer position(s)" in coverage[0]
    assert "of 2" not in coverage[0]


def test_unindexed_language_prefix_is_unmeasurable_not_zero_of_zero() -> None:
    """Breaks: a model with no integer-indexed positions prints "0 of 0", a
    vacuous completeness claim dressed up as a coverage measurement."""
    spec = FamilySpec(
        name="flat",
        model_types=("flat_test",),
        language_prefixes=("model.blocks",),
        tower_prefixes=(),
        adapter_leaf_modules=("q_proj", "k_proj"),
        expert_count_path=(),
    )
    graph: list[tuple[str, object]] = [
        ("model.blocks.attention.q_proj", Linear()),
        ("model.blocks.attention.k_proj", Linear()),
    ]

    selected, lines = select_adapter_modules(graph, spec, _plain_linear)

    assert len(selected) == 2
    assert not _coverage_lines(lines, "model.blocks")
    unmeasurable = [line for line in lines if "UNMEASURABLE" in line]
    assert len(unmeasurable) == 1
    assert "'model.blocks'" in unmeasurable[0]
    assert "0 of 0" not in "\n".join(lines)


def test_a_generator_is_consumed_exactly_once() -> None:
    """Breaks: selection iterates the iterable twice; with a generator the
    second pass sees nothing and the denominator (or the selection) silently
    collapses."""
    spec = FamilySpec(
        name="gen",
        model_types=("gen_test",),
        language_prefixes=("model.layers",),
        tower_prefixes=(),
        adapter_leaf_modules=("q_proj",),
        expert_count_path=(),
    )

    def _modules() -> Iterable[tuple[str, object]]:
        for layer in range(4):
            yield (f"model.layers.{layer}.self_attn.q_proj", Linear())
            yield (f"model.layers.{layer}.mlp.expert_scale", Linear())

    selected, lines = select_adapter_modules(_modules(), spec, _plain_linear)

    assert len(selected) == 4
    coverage = _coverage_lines(lines, "model.layers")
    assert len(coverage) == 1
    assert "reached all 4 indexed layer position(s)" in coverage[0]


def test_a_vision_tower_dilutes_neither_numerator_nor_denominator() -> None:
    """Breaks: 32 indexed vision blocks leak into the language coverage count,
    turning full 4-of-4 language coverage into a fictional 4-of-36 gap."""
    spec = FamilySpec(
        name="vlm",
        model_types=("vlm_test",),
        language_prefixes=("model.language_model",),
        tower_prefixes=("model.vision_tower",),
        adapter_leaf_modules=("q_proj",),
        expert_count_path=(),
    )
    graph: list[tuple[str, object]] = []
    for layer in range(4):
        graph.append((f"model.language_model.layers.{layer}.self_attn.q_proj", Linear()))
    for block in range(32):
        graph.append((f"model.vision_tower.blocks.{block}.attn.q_proj", Linear()))

    selected, lines = select_adapter_modules(graph, spec, _plain_linear)

    assert len(selected) == 4
    coverage = _coverage_lines(lines, "model.language_model")
    assert len(coverage) == 1
    assert "reached all 4 indexed layer position(s)" in coverage[0]
    assert "of 36" not in coverage[0]
    # The tower is still announced EXCLUDED; it is only coverage it must not touch.
    assert any("model.vision_tower" in line and "EXCLUDED" in line for line in lines)
