
"""Skill registry and producer/consumer chain planning."""
from __future__ import annotations

from collections import deque
from importlib import metadata
from typing import Any

from foundationskills.core.contract import BaseSkill


class SkillRegistry:
    """In-memory registry of skill instances."""

    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}

    @staticmethod
    def _required(skill: BaseSkill) -> None:
        missing: list[str] = []
        if not getattr(skill, "name", ""):
            missing.append("name")
        if not getattr(skill, "version", ""):
            missing.append("version")
        if not getattr(skill, "description", ""):
            missing.append("description")
        for attr in ("consumes", "produces", "rules"):
            if not isinstance(getattr(skill, attr, None), tuple):
                missing.append(attr)
        for attr in ("input_schema", "output_schema"):
            if not isinstance(getattr(skill, attr, None), dict):
                missing.append(attr)
        if getattr(skill, "scope", None) is None:
            missing.append("scope")
        if getattr(skill, "fs_interface", None) is None:
            missing.append("fs_interface")
        if missing:
            raise ValueError(f"skill is missing required class attributes: {', '.join(missing)}")

    def register(self, skill: BaseSkill) -> None:
        self._required(skill)
        if skill.name in self._skills:
            raise ValueError(f"duplicate skill name: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> BaseSkill:
        try:
            return self._skills[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(f"unknown skill {name!r}; available: {available}") from exc

    def names(self) -> list[str]:
        return sorted(self._skills)

    def producers_of(self, type: str) -> list[str]:
        return sorted(name for name, skill in self._skills.items() if type in skill.produces)

    def consumers_of(self, type: str) -> list[str]:
        return sorted(name for name, skill in self._skills.items() if type in skill.consumes)

    def plan_chain(self, have: set[str], want: str) -> list[str]:
        """BFS shortest chain whose cumulative produces can satisfy want.

        A skill is executable only when every consumed type is available.
        """
        if want in have:
            return []
        start = frozenset(have)
        queue: deque[tuple[frozenset[str], list[str]]] = deque([(start, [])])
        seen = {start}
        ordered = sorted(self._skills.values(), key=lambda s: s.name)
        while queue:
            available, path = queue.popleft()
            for skill in ordered:
                if not set(skill.consumes).issubset(available):
                    continue
                new_available = frozenset(set(available).union(skill.produces))
                if new_available == available or new_available in seen:
                    continue
                new_path = path + [skill.name]
                if want in new_available:
                    return new_path
                seen.add(new_available)
                queue.append((new_available, new_path))
        raise LookupError(f"no registered skill chain can produce {want!r} from {sorted(have)!r}")


REGISTRY = SkillRegistry()


def register_skill(skill: BaseSkill) -> None:
    REGISTRY.register(skill)


def get_skill(name: str) -> BaseSkill:
    return REGISTRY.get(name)


def list_skills() -> list[str]:
    return REGISTRY.names()


def load_entry_points(group: str = "foundationskills.skills") -> list[tuple[str, str]]:
    """Import/register entry points; tolerate failures by returning (name, error)."""
    failures: list[tuple[str, str]] = []
    try:
        eps = metadata.entry_points()
        selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
    except Exception as exc:
        return [(group, f"{type(exc).__name__}: {exc}")]
    for ep in selected:
        try:
            loaded = ep.load()
            obj = loaded() if callable(loaded) else loaded
            if isinstance(obj, BaseSkill):
                REGISTRY.register(obj)
            else:
                raise TypeError(f"entry point {ep.name!r} did not resolve to BaseSkill instance/callable")
        except Exception as exc:
            failures.append((ep.name, f"{type(exc).__name__}: {exc}"))
    return failures
