"""Tests for the adapter target PLAN -- the policy the training loop calls.

``plan_adapter_targets`` is the whole of what ``loop.py`` knows about model
families, and it exists as a separate function precisely so these four branches
can be exercised without a GPU, a checkpoint, or peft. If a future family needs
a fifth branch, this file is where that shows up.

The configs below are the shapes MEASURED on the estate on 2026-09-20, not
invented ones; ``llama_9_ultra`` is the deliberate stand-in for a family nobody
has registered yet, which is the case the refusal exists for.
"""

from __future__ import annotations

from foundationscale.families import plan_adapter_targets
from foundationscale.families.adapters import AdapterPlan

GEMMA4 = {"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}}
QWEN35 = {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}}
UNKNOWN = {"model_type": "llama_9_ultra"}

LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class Linear:
    """Adaptable."""


class Clippable:
    """The Linear subclass peft refused, as a distinct fake type."""


def _plain_linear(module: object) -> bool:
    return type(module) is Linear


def _gemma4_modules() -> list[tuple[str, object]]:
    graph: list[tuple[str, object]] = []
    for layer in range(4):
        for leaf in LEAVES:
            graph.append((f"model.language_model.layers.{layer}.self_attn.{leaf}", Linear()))
            graph.append(
                (f"model.vision_tower.encoder.layers.{layer}.self_attn.{leaf}", Clippable())
            )
    return graph


def test_registered_family_with_no_declaration_scopes_to_the_language_tower() -> None:
    plan = plan_adapter_targets(GEMMA4, None, _gemma4_modules(), _plain_linear)
    assert not plan.refused
    assert plan.family == "gemma4"
    assert len(plan.targets) == 4 * len(LEAVES)
    assert all(name.startswith("model.language_model.") for name in plan.targets)
    assert not any("vision_tower" in name for name in plan.targets)


def test_registered_family_narrows_to_the_declared_leaves() -> None:
    # A declaration of two leaves must not silently become the family's seven.
    plan = plan_adapter_targets(GEMMA4, ["q_proj", "v_proj"], _gemma4_modules(), _plain_linear)
    assert not plan.refused
    assert len(plan.targets) == 4 * 2
    assert all(name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"} for name in plan.targets)


def test_unregistered_family_with_no_declaration_refuses_and_says_how_to_proceed() -> None:
    plan = plan_adapter_targets(UNKNOWN, None, [], _plain_linear)
    assert plan.refused
    assert plan.targets == ()
    assert plan.family is None
    assert plan.refusal is not None
    assert "llama_9_ultra" in plan.refusal
    # Both remedies, or the refusal is just a stop.
    assert "--adapter-target" in plan.refusal
    assert "FamilySpec" in plan.refusal


def test_unregistered_family_with_a_declaration_runs_but_says_it_could_not_scope() -> None:
    plan = plan_adapter_targets(UNKNOWN, ["q_proj"], _gemma4_modules(), _plain_linear)
    assert not plan.refused
    assert plan.targets == ("q_proj",)
    assert plan.family is None
    blob = "\n".join(plan.announcements)
    assert "UNSCOPED" in blob
    # The specific hazard has to be named, not gestured at.
    assert "vision" in blob and "audio" in blob


def test_a_qualified_declaration_is_passed_through_untouched() -> None:
    declared = ["model.language_model.layers.0.self_attn.q_proj"]
    plan = plan_adapter_targets(GEMMA4, declared, _gemma4_modules(), _plain_linear)
    assert not plan.refused
    assert plan.targets == tuple(declared)
    assert "verbatim" in "\n".join(plan.announcements)


def test_a_qualified_declaration_is_honoured_even_for_an_unregistered_family() -> None:
    # Qualified names need no registry: they already say what they mean. This is
    # the escape hatch that keeps a brand-new family trainable on day zero.
    declared = ["decoder.blocks.0.attn.q_proj"]
    plan = plan_adapter_targets(UNKNOWN, declared, [], _plain_linear)
    assert not plan.refused
    assert plan.targets == tuple(declared)


def test_a_mixed_declaration_counts_as_qualified_and_is_not_rewritten() -> None:
    # One dotted name in the list means the operator is naming modules. Scoping
    # half of it would produce a target set nobody wrote.
    declared = ["q_proj", "model.language_model.layers.3.self_attn.v_proj"]
    plan = plan_adapter_targets(GEMMA4, declared, _gemma4_modules(), _plain_linear)
    assert plan.targets == tuple(declared)


def test_registered_family_that_selects_nothing_refuses_rather_than_passing_an_empty_set() -> None:
    # Every candidate is the unwrappable subclass: names and scope are right and
    # the type is the obstacle. An empty target_modules would send peft back to
    # its own inference, which is the failure this whole path replaces.
    graph = [(f"model.language_model.layers.{i}.self_attn.q_proj", Clippable()) for i in range(4)]
    plan = plan_adapter_targets(GEMMA4, None, graph, _plain_linear)
    assert plan.refused
    assert plan.targets == ()
    assert plan.family == "gemma4"
    assert plan.refusal is not None
    assert "0 modules" in plan.refusal
    # The scope announcements survive the refusal -- they are the diagnosis.
    assert "NOT adaptable" in "\n".join(plan.announcements)


def test_a_refusal_never_carries_targets_and_a_plan_never_carries_both() -> None:
    plans = [
        plan_adapter_targets(UNKNOWN, None, [], _plain_linear),
        plan_adapter_targets(GEMMA4, None, [], _plain_linear),
        plan_adapter_targets(GEMMA4, None, _gemma4_modules(), _plain_linear),
        plan_adapter_targets(QWEN35, ["q_proj"], [], _plain_linear),
    ]
    for plan in plans:
        assert isinstance(plan, AdapterPlan)
        assert plan.refused == (plan.refusal is not None)
        if plan.refused:
            assert plan.targets == ()


def test_every_non_refusing_branch_announces_something() -> None:
    # Silence is the defect being fixed: a run that scoped, and a run that could
    # not scope, must not look the same in the log.
    for config, declared in (
        (GEMMA4, None),
        (GEMMA4, ["q_proj"]),
        (GEMMA4, ["model.language_model.layers.0.self_attn.q_proj"]),
        (UNKNOWN, ["q_proj"]),
    ):
        plan = plan_adapter_targets(config, declared, _gemma4_modules(), _plain_linear)
        assert not plan.refused
        assert plan.announcements, f"silent plan for config={config} declared={declared}"


def test_two_families_reach_different_targets_from_the_same_declaration() -> None:
    # The pluggability claim, as an assertion: one declaration, two families,
    # two different resolved scopes -- with no branch anywhere on family name.
    graph: list[tuple[str, object]] = [
        ("model.language_model.layers.0.self_attn.q_proj", Linear()),
        ("model.layers.0.self_attn.q_proj", Linear()),
        ("model.visual.blocks.0.attn.q_proj", Linear()),
        ("model.vision_tower.encoder.layers.0.self_attn.q_proj", Linear()),
    ]
    gemma = plan_adapter_targets(GEMMA4, ["q_proj"], graph, _plain_linear)
    qwen = plan_adapter_targets(QWEN35, ["q_proj"], graph, _plain_linear)
    assert gemma.targets == ("model.language_model.layers.0.self_attn.q_proj",)
    assert qwen.targets == (
        "model.language_model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.q_proj",
    )
