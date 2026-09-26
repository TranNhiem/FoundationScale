
"""Built-in skill registration.

``register_builtin_skills`` is idempotent (a skill whose name is already in
the registry is skipped) and lazy (skill modules import only on call). The
training-skill imports are tolerated with ``ImportError`` so this package
works during partial checkouts; the data_engine skill is mandatory and its
import errors propagate.
"""
from __future__ import annotations

from foundationskills.core.registry import REGISTRY, SkillRegistry


def register_builtin_skills(registry: SkillRegistry = REGISTRY) -> list[str]:
    """Register built-in skills into ``registry``; return the newly registered names."""
    before = set(registry.names())

    from foundationskills.skills.data_engine.skill import DataEngineSkill

    if DataEngineSkill.name not in before:
        registry.register(DataEngineSkill())

    try:
        from foundationskills.skills.training.skill import TrainingPlannerSkill

        if TrainingPlannerSkill.name not in registry.names():
            registry.register(TrainingPlannerSkill())
    except ImportError:
        pass
    try:
        from foundationskills.skills.training.emit_skill import TrainingEmitSkill

        if TrainingEmitSkill.name not in registry.names():
            registry.register(TrainingEmitSkill())
    except ImportError:
        pass

    return sorted(set(registry.names()) - before)


__all__ = ["register_builtin_skills"]
