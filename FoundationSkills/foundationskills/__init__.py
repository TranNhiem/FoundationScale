
"""FoundationSkills: agent-facing skills layer on top of FoundationScale."""
from __future__ import annotations

__version__ = "0.1.0"

from foundationskills.core.status import Status
from foundationskills.core.contract import (
    BaseSkill,
    Diagnosis,
    Finding,
    FSInterface,
    RuleSpec,
    Scope,
    Severity,
    SkillContext,
    SkillResult,
)
from foundationskills.core.artifacts import Artifact, ArtifactRef, ArtifactType
from foundationskills.core.registry import get_skill, list_skills, register_skill

__all__ = [
    "__version__",
    "Status",
    "Severity",
    "Finding",
    "Scope",
    "Diagnosis",
    "FSInterface",
    "SkillResult",
    "SkillContext",
    "BaseSkill",
    "RuleSpec",
    "Artifact",
    "ArtifactRef",
    "ArtifactType",
    "register_skill",
    "get_skill",
    "list_skills",
]
