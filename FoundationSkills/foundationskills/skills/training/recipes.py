
"""Recipe selection: score knowledge-pack recipes against a :class:`RecipeQuery`.

Hard keys: ``stage`` and, when given, ``arch`` and ``method`` — a recipe that
mismatches a hard key is excluded. Everything else is soft and additive:

- family exact: +3
- arch-compatible and size inside the recipe's ``size_b`` range: +2
- size within ±50% of the range: +1
- goal exact: +2
- domain exact: +1.5, recipe domain ``"any"``: +0.5
- hardware listed: +1
- (given) method matched: +1 (hard key survivors always get this point)

A match is ``derived=True`` unless family, size (in range) and goal all match
exactly; ``adjustments`` then describes the adaptation (parameter/token/LR
scaling, family porting notes) so a human can sanity-check the derivation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from foundationskills.skills.training.knowledge import Recipe, load_recipes

__all__ = ["RecipeQuery", "RecipeMatch", "select_recipe", "rank_recipes"]


@dataclass(frozen=True)
class RecipeQuery:
    """Constraints for recipe selection.

    ``stage`` is a hard key when given; ``stage=None`` means "no stage
    constraint" (not recommended — callers should pass it). ``arch`` and
    ``method`` are hard only when not None.
    """

    family: str | None = None
    size_b: float | None = None
    arch: str | None = None
    stage: str | None = None
    goal: str | None = None
    domain: str | None = None
    method: str | None = None
    hardware: str | None = None


@dataclass(frozen=True)
class RecipeMatch:
    recipe: Recipe
    score: float
    reasons: list[str] = field(default_factory=list)
    derived: bool = True
    adjustments: list[str] = field(default_factory=list)


def _size_range(recipe: Recipe) -> tuple[float | None, float | None]:
    raw = recipe.index.get("size_b")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    return None, None


def _score(recipe: Recipe, q: RecipeQuery) -> RecipeMatch | None:
    idx = recipe.index
    reasons: list[str] = []
    adjustments: list[str] = []
    score = 0.0

    r_stage = idx.get("stage")
    if q.stage is not None:
        if r_stage != q.stage:
            return None  # hard key
        reasons.append(f"stage exact (hard key): {q.stage}")
    r_arch = idx.get("arch")
    if q.arch is not None:
        if r_arch != q.arch:
            return None  # hard key
        reasons.append(f"arch exact (hard key): {q.arch}")
    r_method = idx.get("method")
    if q.method is not None:
        if r_method != q.method:
            return None  # hard key
        score += 1.0
        reasons.append(f"method exact: {q.method} (+1)")

    family_match = False
    r_family = idx.get("family")
    if q.family is not None:
        if r_family == q.family:
            family_match = True
            score += 3.0
            reasons.append(f"family exact: {q.family} (+3)")
        else:
            reasons.append(f"family mismatch: query={q.family} recipe={r_family} (+0)")
            adjustments.append(
                f"ported from family {r_family} to {q.family}; verify LoRA targets, chat template and tokenizer"
            )

    size_match = False
    lo, hi = _size_range(recipe)
    if q.size_b is not None and lo is not None and hi is not None:
        if lo <= q.size_b <= hi:
            size_match = True
            score += 2.0
            reasons.append(f"size {q.size_b}B in recipe range [{lo}, {hi}], arch compatible (+2)")
        elif lo * 0.5 <= q.size_b <= hi * 1.5:
            score += 1.0
            reasons.append(f"size {q.size_b}B within ±50% of recipe range [{lo}, {hi}] (+1)")
        else:
            reasons.append(f"size {q.size_b}B outside ±50% of recipe range [{lo}, {hi}] (+0)")
        if not size_match:
            mid = (lo + hi) / 2.0
            ratio = q.size_b / mid if mid > 0 else 1.0
            adjustments.append(f"tokens scaled by size ratio x{ratio:.2f} vs recipe midpoint {mid:.1f}B")
            adjustments.append("LR scaled for size (larger models take smaller peak LRs)")
    elif q.size_b is not None:
        reasons.append("recipe carries no size_b range; size unverifiable (+0)")

    goal_match = False
    r_goal = idx.get("goal")
    if q.goal is not None:
        if r_goal == q.goal:
            goal_match = True
            score += 2.0
            reasons.append(f"goal exact: {q.goal} (+2)")
        else:
            reasons.append(f"goal mismatch: query={q.goal} recipe={r_goal} (+0)")
            adjustments.append(f"recipe targets goal {r_goal}; re-check hparams and evaluation for {q.goal}")

    if q.domain is not None:
        r_domain = idx.get("domain")
        if r_domain == q.domain:
            score += 1.5
            reasons.append(f"domain exact: {q.domain} (+1.5)")
        elif r_domain == "any":
            score += 0.5
            reasons.append(f"recipe domain 'any' covers query domain {q.domain} (+0.5)")
        else:
            reasons.append(f"domain mismatch: query={q.domain} recipe={r_domain} (+0)")

    if q.hardware is not None:
        hw_list = idx.get("hardware") or []
        if q.hardware in hw_list:
            score += 1.0
            reasons.append(f"hardware listed: {q.hardware} (+1)")
        else:
            reasons.append(f"hardware {q.hardware} not in recipe hardware {list(hw_list)} (+0)")

    derived = not (family_match and size_match and goal_match)
    return RecipeMatch(recipe=recipe, score=score, reasons=reasons, derived=derived, adjustments=adjustments)


def rank_recipes(
    q: RecipeQuery,
    k: int = 3,
    *,
    recipes: Sequence[Recipe] | None = None,
    root: Path | None = None,
) -> list[RecipeMatch]:
    """Top-``k`` matches, descending score, ties broken by recipe id.

    ``recipes`` injects an explicit pool (tests); otherwise the knowledge pack
    at ``root`` is used.
    """
    pool = list(recipes) if recipes is not None else load_recipes(root)
    matches = [m for m in (_score(r, q) for r in pool) if m is not None]
    matches.sort(key=lambda m: (-m.score, m.recipe.id))
    return matches[: max(0, int(k))]


def select_recipe(
    q: RecipeQuery,
    *,
    recipes: Sequence[Recipe] | None = None,
    root: Path | None = None,
) -> RecipeMatch | None:
    """Best match or None when every recipe fails a hard key."""
    top = rank_recipes(q, k=1, recipes=recipes, root=root)
    return top[0] if top else None
