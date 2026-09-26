
from __future__ import annotations

import pytest

from foundationskills.skills.training.knowledge import Recipe
from foundationskills.skills.training.recipes import RecipeQuery, rank_recipes, select_recipe


def make_recipe(
    rid: str,
    *,
    family: str = "testfam",
    size: tuple[float, float] = (0.4, 0.6),
    arch: str = "dense",
    stage: str = "sft",
    goal: str = "general_chat",
    domain: str = "any",
    method: str = "lora",
    hardware: tuple[str, ...] = ("testgpu",),
) -> Recipe:
    return Recipe(
        id=rid,
        version="1.0.0",
        title=rid,
        index={
            "family": family,
            "size_b": [size[0], size[1]],
            "arch": arch,
            "stage": stage,
            "goal": goal,
            "domain": domain,
            "method": method,
            "hardware": list(hardware),
        },
        data={"format": stage},
        stages=(),
        hardware={},
        evaluation={},
        risks=(),
        provenance={},
        raw={},
    )


def perfect_query() -> RecipeQuery:
    return RecipeQuery(
        family="testfam", size_b=0.5, arch="dense", stage="sft",
        goal="general_chat", domain="any", method="lora", hardware="testgpu",
    )


def test_perfect_match_scores_all_points_and_is_not_derived() -> None:
    recipe = make_recipe("r-perfect")
    match = select_recipe(perfect_query(), recipes=[recipe])
    assert match is not None
    # family 3 + size/arch 2 + method 1 + goal 2 + domain exact 1.5 + hardware 1
    assert match.score == pytest.approx(10.5)
    assert match.derived is False
    assert match.adjustments == []
    assert any("family exact" in r for r in match.reasons)


def test_hard_keys_exclude() -> None:
    recipe = make_recipe("r1")
    stage_q = RecipeQuery(stage="pretrain")
    assert select_recipe(stage_q, recipes=[recipe]) is None
    arch_q = RecipeQuery(stage="sft", arch="moe")
    assert select_recipe(arch_q, recipes=[recipe]) is None
    method_q = RecipeQuery(stage="sft", method="full")
    assert select_recipe(method_q, recipes=[recipe]) is None


def test_family_mismatch_is_soft_and_derived() -> None:
    recipe = make_recipe("r-other-fam", family="otherfam")
    match = select_recipe(perfect_query(), recipes=[recipe])
    assert match is not None
    # 10.5 minus the family 3 points
    assert match.score == pytest.approx(7.5)
    assert match.derived is True
    assert any("otherfam" in adj for adj in match.adjustments)


def test_size_within_half_range_scores_one_point_and_derives() -> None:
    recipe = make_recipe("r-big", size=(10.0, 20.0))
    q = RecipeQuery(family="testfam", size_b=28.0, arch="dense", stage="sft", goal="general_chat", domain="any", method="lora", hardware="testgpu")
    match = select_recipe(q, recipes=[recipe])
    assert match is not None
    # family 3 + size ±50% 1 + method 1 + goal 2 + domain 1.5 + hardware 1
    assert match.score == pytest.approx(9.5)
    assert match.derived is True
    joined = " ".join(match.adjustments)
    assert "scaled" in joined and "LR" in joined


def test_domain_any_gets_half_point() -> None:
    recipe = make_recipe("r-any", domain="any")
    q = RecipeQuery(family="testfam", size_b=0.5, arch="dense", stage="sft", goal="general_chat", domain="medical", method="lora", hardware="testgpu")
    match = select_recipe(q, recipes=[recipe])
    assert match is not None
    # family 3 + size 2 + method 1 + goal 2 + domain-any 0.5 + hardware 1
    assert match.score == pytest.approx(9.5)
    assert match.derived is False  # family+size+goal still all exact


def test_rank_order_prefers_family_match() -> None:
    weak = make_recipe("r-weak", family="otherfam")
    strong = make_recipe("r-strong")
    ranked = rank_recipes(perfect_query(), recipes=[weak, strong])
    assert [m.recipe.id for m in ranked] == ["r-strong", "r-weak"]
    assert ranked[0].derived is False and ranked[1].derived is True


def test_rank_is_deterministic_on_ties() -> None:
    a = make_recipe("r-b")
    b = make_recipe("r-a")
    ranked = rank_recipes(perfect_query(), recipes=[a, b])
    assert [m.recipe.id for m in ranked] == ["r-a", "r-b"]  # id tiebreak


def test_no_candidates_returns_none_for_select() -> None:
    q = RecipeQuery(stage="rl")
    assert select_recipe(q, recipes=[]) is None
