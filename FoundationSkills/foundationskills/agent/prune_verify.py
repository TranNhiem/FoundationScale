from __future__ import annotations
"""Stage-0 pruning gates for SKILL.md rewrite candidates (cheap, pre-eval).

Returns every blocking problem; an empty list means the candidate may go on to
spend evaluation budget later.  Checks are side-effect free and never raise.
"""

from typing import Any

from foundationskills.agent import frozen_surface, skillmd

MAX_GROWTH = 1.3  # a candidate may grow by at most 30% in characters

EMPTY_MSG = "candidate is empty"
GROWTH_MSG = "candidate grew more than 30%"

__all__ = ["MAX_GROWTH", "stage0"]


def _as_list(payload: Any) -> list[str]:
    """Normalise a checker payload (None | str | iterable | scalar) to strings."""
    if payload is None:
        return []
    if isinstance(payload, str):
        return [payload]
    try:
        return [str(item) for item in payload]
    except TypeError:
        return [str(payload)]


def _dedupe(problems: list[str]) -> list[str]:
    """Drop repeated messages while preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for problem in problems:
        if problem not in seen:
            seen.add(problem)
            out.append(problem)
    return out


def stage0(old_text: str, new_text: str, skill=None, parser=None) -> list[str]:
    """Return blocking problems for new_text; [] means it passes stage 0.

    Order: frozen-surface byte identity, bounded growth, non-blank candidate,
    and (only when skill and parser are given) a still-fresh generated usage block.
    """
    old = "" if old_text is None else str(old_text)
    new = "" if new_text is None else str(new_text)
    found: list[str] = []

    try:
        found.extend(_as_list(frozen_surface.assert_editable(old, new)))
    except Exception as exc:  # a broken guard must gate, not crash the loop
        found.append("frozen surface check failed: " + str(exc))

    if len(new) > MAX_GROWTH * len(old):
        found.append(GROWTH_MSG)
    if not new.strip():
        found.append(EMPTY_MSG)

    if skill is not None and parser is not None:
        try:
            found.extend(_as_list(skillmd.check(skill, parser, new)))
        except Exception as exc:  # pragma: no cover - defensive
            found.append("skillmd check failed: " + str(exc))

    return _dedupe(found)
