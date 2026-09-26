
"""Core contracts and utilities for FoundationSkills."""
from __future__ import annotations

from foundationskills.core.status import EXIT_CODES, Status, worst
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
from foundationskills.core.artifacts import (
    Artifact,
    ArtifactRef,
    ArtifactType,
    read_artifact,
    register_artifact_type,
    verify_ref,
    write_artifact,
)
from foundationskills.core.provenance import Provenance, make_provenance
from foundationskills.core.registry import REGISTRY, SkillRegistry, get_skill, list_skills, register_skill
from foundationskills.core.schema import SchemaError, assert_valid, load_schema, validate

from foundationskills.core.provenance import sha256_file, sha256_json
from foundationskills.core.orchestrator import (
    ConfirmationRequired,
    Orchestrator,
    Step,
    plan_hash,
    require_confirmation,
)

__all__ = [
    "EXIT_CODES",
    "Status",
    "worst",
    "BaseSkill",
    "Diagnosis",
    "Finding",
    "FSInterface",
    "RuleSpec",
    "Scope",
    "Severity",
    "SkillContext",
    "SkillResult",
    "Artifact",
    "ArtifactRef",
    "ArtifactType",
    "read_artifact",
    "register_artifact_type",
    "verify_ref",
    "write_artifact",
    "Provenance",
    "make_provenance",
    "REGISTRY",
    "SkillRegistry",
    "get_skill",
    "list_skills",
    "register_skill",
    "SchemaError",
    "assert_valid",
    "load_schema",
    "validate",
    "sha256_file",
    "sha256_json",
    "ConfirmationRequired",
    "Orchestrator",
    "Step",
    "plan_hash",
    "require_confirmation",
]
