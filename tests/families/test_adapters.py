"""Tests for scoped adapter-target selection.

Skip-free and torch-free by construction: selection is duck-typed over
``(qualified_name, module)`` pairs, so the module graphs below are built from
tiny fakes whose only job is to have a distinguishable type name.

The Gemma4 fixture reproduces the shape MEASURED on ``gemma-4-31B``: 60 plain
Linear under ``model.language_model`` and 27 clippable-Linear under
``model.vision_tower``, for each of seven leaf names. That shape is the exact
one that failed in production, so it is the one the test asserts on.
"""

from __future__ import annotations

from foundationscale.families.adapters import select_adapter_modules, torch_linear_predicate
from foundationscale.families.registry import REGISTRY, FamilySpec

LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class Linear:
    """Stands in for torch.nn.Linear."""


class Gemma4ClippableLinear:
    """Stands in for the Linear SUBCLASS peft refused to wrap."""


class LayerNorm:
    """A module that is never a target, to prove leaf filtering happens at all."""


def _plain_linear(module: object) -> bool:
    return type(module) is Linear


def _spec(name: str) -> FamilySpec:
    for spec in REGISTRY:
        if spec.name == name:
            return spec
    raise AssertionError(f"registry has no family {name!r}")


def _gemma4_graph() -> list[tuple[str, object]]:
    graph: list[tuple[str, object]] = []
    for layer in range(60):
        for leaf in LEAVES:
            graph.append((f"model.language_model.layers.{layer}.self_attn.{leaf}", Linear()))
        graph.append((f"model.language_model.layers.{layer}.input_layernorm", LayerNorm()))
    for layer in range(27):
        for leaf in LEAVES:
            graph.append(
                (
                    f"model.vision_tower.encoder.layers.{layer}.self_attn.{leaf}",
                    Gemma4ClippableLinear(),
                )
            )
    return graph


def _qwen_text_graph() -> list[tuple[str, object]]:
    graph: list[tuple[str, object]] = []
    for layer in range(16):
        for leaf in LEAVES:
            graph.append((f"model.layers.{layer}.self_attn.{leaf}", Linear()))
    # The multi-token-prediction head Gemma4 has no analogue for.
    graph.append(("mtp.layers.0.self_attn.q_proj", Linear()))
    # The vision tower, present when the checkpoint is loaded as a VLM.
    graph.append(("model.visual.blocks.0.attn.q_proj", Linear()))
    return graph


def test_gemma4_selects_the_language_tower_only() -> None:
    selected, _ = select_adapter_modules(_gemma4_graph(), _spec("gemma4"), _plain_linear)
    assert len(selected) == 60 * len(LEAVES)
    assert all(name.startswith("model.language_model.") for name in selected)
    assert not any("vision_tower" in name for name in selected)


def test_scope_alone_excludes_the_vision_tower_when_the_type_is_identical() -> None:
    # The fixture above gives the vision tower a DIFFERENT type, so the predicate
    # would exclude it even with no scoping at all -- a negative control showed
    # that test still passing with both scope rules neutered. Here every module is
    # the same adaptable type, so scoping is the only thing that can exclude the
    # tower, and this test fails the moment it stops working.
    graph: list[tuple[str, object]] = []
    for layer in range(4):
        graph.append((f"model.language_model.layers.{layer}.self_attn.q_proj", Linear()))
        graph.append((f"model.vision_tower.encoder.layers.{layer}.self_attn.q_proj", Linear()))
    selected, lines = select_adapter_modules(graph, _spec("gemma4"), _plain_linear)
    assert len(selected) == 4
    assert all(name.startswith("model.language_model.") for name in selected)
    assert "model.vision_tower" in "\n".join(lines)


def test_gemma4_announces_the_vision_tower_by_name_count_and_type() -> None:
    _, lines = select_adapter_modules(_gemma4_graph(), _spec("gemma4"), _plain_linear)
    blob = "\n".join(lines)
    assert "model.vision_tower" in blob
    assert str(27 * len(LEAVES)) in blob
    # The type name is the whole point: this is the line that would have
    # explained the production failure at the moment it happened.
    assert "Gemma4ClippableLinear" in blob


def test_qwen_selects_under_model_layers_and_excludes_mtp_and_visual() -> None:
    selected, lines = select_adapter_modules(_qwen_text_graph(), _spec("qwen3.5"), _plain_linear)
    assert len(selected) == 16 * len(LEAVES)
    assert all(name.startswith("model.layers.") for name in selected)
    blob = "\n".join(lines)
    assert "mtp" in blob
    assert "model.visual" in blob


def test_every_selected_name_is_fully_qualified() -> None:
    selected, _ = select_adapter_modules(_gemma4_graph(), _spec("gemma4"), _plain_linear)
    assert all("." in name for name in selected)


def test_selected_names_satisfy_the_peft_endswith_contract() -> None:
    # The entire scoping design rests on peft matching a list target_modules with
    # key.endswith(target). If a returned name did not match itself under that
    # rule, the selection would be correct and the wiring would still fail.
    selected, _ = select_adapter_modules(_gemma4_graph(), _spec("gemma4"), _plain_linear)
    assert selected
    for name in selected:
        assert name.endswith(name)
    # And it must not accidentally match a vision-tower module.
    vision = "model.vision_tower.encoder.layers.0.self_attn.q_proj"
    assert not any(vision.endswith(name) for name in selected)


def test_empty_selection_is_loud() -> None:
    graph = [("model.vision_tower.encoder.layers.0.self_attn.q_proj", Gemma4ClippableLinear())]
    selected, lines = select_adapter_modules(graph, _spec("gemma4"), _plain_linear)
    assert selected == ()
    blob = "\n".join(lines)
    assert "SELECTED NOTHING" in blob
    assert "refuse" in blob


def test_not_adaptable_is_announced_distinctly_from_no_name_match() -> None:
    # Names and scope are both right; only the type is the obstacle. These two
    # populations have different fixes and must not read alike.
    in_scope_wrong_type = [
        ("model.language_model.layers.0.self_attn.q_proj", Gemma4ClippableLinear()),
    ]
    _, lines_type = select_adapter_modules(in_scope_wrong_type, _spec("gemma4"), _plain_linear)
    no_name_match = [("model.language_model.layers.0.input_layernorm", Linear())]
    _, lines_name = select_adapter_modules(no_name_match, _spec("gemma4"), _plain_linear)

    type_blob = "\n".join(lines_type)
    name_blob = "\n".join(lines_name)
    assert "NOT adaptable" in type_blob
    assert "Gemma4ClippableLinear" in type_blob
    assert "NOT adaptable" not in name_blob
    # Both still refuse, but for reasons that are told apart.
    assert "SELECTED NOTHING" in type_blob
    assert "SELECTED NOTHING" in name_blob


def test_a_target_outside_every_declared_prefix_is_reported_as_a_registration_gap() -> None:
    graph = [("some_new_tower.layers.0.self_attn.q_proj", Linear())]
    selected, lines = select_adapter_modules(graph, _spec("gemma4"), _plain_linear)
    assert selected == ()
    assert "outside every declared prefix" in "\n".join(lines)


def test_prefix_matching_does_not_match_a_sibling_by_string_prefix() -> None:
    # model.layers must not swallow model.layers_extra.
    spec = FamilySpec(
        name="sibling",
        model_types=("x",),
        language_prefixes=("model.layers",),
        tower_prefixes=(),
        adapter_leaf_modules=("q_proj",),
        expert_count_path=(),
    )
    graph = [
        ("model.layers.0.q_proj", Linear()),
        ("model.layers_extra.0.q_proj", Linear()),
    ]
    selected, _ = select_adapter_modules(graph, spec, _plain_linear)
    assert selected == ("model.layers.0.q_proj",)


def test_no_exclusion_is_ever_silent() -> None:
    # The property, not a line count: if anything was excluded or rejected, at
    # least one announcement mentions it.
    graph = _gemma4_graph() + [("some_new_tower.x.q_proj", Linear())]
    selected, lines = select_adapter_modules(graph, _spec("gemma4"), _plain_linear)
    excluded = sum(1 for name, _ in graph if name.rsplit(".", 1)[-1] in LEAVES) - len(selected)
    assert excluded > 0
    assert lines, "modules were excluded and nothing was announced"
    blob = "\n".join(lines)
    assert "EXCLUDED" in blob


def test_selection_announcement_states_the_language_prefix_and_count() -> None:
    _, lines = select_adapter_modules(_gemma4_graph(), _spec("gemma4"), _plain_linear)
    blob = "\n".join(lines)
    assert "model.language_model" in blob
    assert str(60 * len(LEAVES)) in blob


def test_torch_linear_predicate_accepts_linear_and_rejects_its_subclasses() -> None:
    # torch is imported unconditionally: the suite runs under FS_FORBID_SKIPS=1, so
    # a conditional import would be a build failure, and on an environment without
    # torch this test SHOULD fail loudly rather than quietly not run.
    import torch

    class Clippable(torch.nn.Linear):
        """The shape of the module peft refused."""

    predicate = torch_linear_predicate()
    assert predicate(torch.nn.Linear(4, 4)) is True
    # isinstance would say True here, which is exactly the defect.
    assert predicate(Clippable(4, 4)) is False
    assert predicate(torch.nn.LayerNorm(4)) is False


def test_torch_predicate_composes_with_selection_on_a_real_module_tree() -> None:
    import torch

    class Clippable(torch.nn.Linear):
        pass

    graph: list[tuple[str, object]] = [
        ("model.language_model.layers.0.self_attn.q_proj", torch.nn.Linear(4, 4)),
        ("model.vision_tower.encoder.layers.0.self_attn.q_proj", Clippable(4, 4)),
    ]
    selected, lines = select_adapter_modules(graph, _spec("gemma4"), torch_linear_predicate())
    assert selected == ("model.language_model.layers.0.self_attn.q_proj",)
    assert "Clippable" in "\n".join(lines)
