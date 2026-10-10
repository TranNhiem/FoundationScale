"""Built-in skill registration.

Registered skill packages: the mandatory ``data_engine``; the tolerated
``training`` (planner + emit), ``evaluation`` and ``auto_research``; and the
four A3.1 discipline skills ``probe_first``, ``measured_or_unmeasured``,
``confirm_before_launch`` and ``refusal_surface``.

``register_builtin_skills`` is idempotent (a skill whose name is already in
the registry is skipped) and lazy (skill modules import only on call). Every
skill import except ``data_engine`` is tolerated with ``ImportError`` so this
package works during partial checkouts; the data_engine skill is mandatory
and its import errors propagate.
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

    try:
        from foundationskills.skills.evaluation.skill import EvalSkill

        if EvalSkill.name not in registry.names():
            registry.register(EvalSkill())
    except ImportError:
        pass

    try:
        from foundationskills.skills.auto_research.skill import AutoResearchSkill

        if AutoResearchSkill.name not in registry.names():
            registry.register(AutoResearchSkill())
    except ImportError:
        pass

    try:
        from foundationskills.skills.probe_first.skill import ProbeFirstSkill

        if ProbeFirstSkill.name not in registry.names():
            registry.register(ProbeFirstSkill())
    except ImportError:
        pass

    try:
        from foundationskills.skills.measured_or_unmeasured.skill import MeasuredOrUnmeasuredSkill

        if MeasuredOrUnmeasuredSkill.name not in registry.names():
            registry.register(MeasuredOrUnmeasuredSkill())
    except ImportError:
        pass

    try:
        from foundationskills.skills.confirm_before_launch.skill import ConfirmBeforeLaunchSkill

        if ConfirmBeforeLaunchSkill.name not in registry.names():
            registry.register(ConfirmBeforeLaunchSkill())
    except ImportError:
        pass

    try:
        from foundationskills.skills.refusal_surface.skill import RefusalSurfaceSkill

        if RefusalSurfaceSkill.name not in registry.names():
            registry.register(RefusalSurfaceSkill())
    except ImportError:
        pass

    return sorted(set(registry.names()) - before)


__all__ = ["register_builtin_skills"]
