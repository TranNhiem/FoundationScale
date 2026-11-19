"""Harness identity: the model-harness pair attached to every launch.

A harness manifest pins skills, routing registry, host build, model pool, substrate and
reset semantics of a run; its hash names the harness exactly (atlas harness-engineering
§6 "version the harness").  Pure C2 increments 1-2: schema constants, skill pins, the
manifest builder and hasher, and non-raising validation and unmeasured-gap reporting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foundationskills.core.provenance import sha256_file, sha256_json

SCHEMA_ID = "fs.harness_manifest.v1"
REQUIRED = ("skills", "routing_registry_version", "host_version", "model_pool",
            "substrate_image", "seed", "reset_semantics", "workspace_snapshot")
UNMEASURED = "unmeasured"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SkillPin:
    """One pinned skill: name, version and the hash of its SKILL.md."""

    name: str
    version: str
    skill_md_sha256: str | None


@dataclass(frozen=True)
class HarnessManifest:
    """The full identity of a launch: the model-harness pair and everything it pins."""

    skills: tuple[SkillPin, ...]
    routing_registry_version: str
    foundationskills_version: str
    source_commit: str
    pool_registry_sha256: str
    pool_entries: tuple[str, ...]
    substrate_image: str
    seed: int | None
    reset_semantics: str
    workspace_snapshot: str
    interpreter_sha256: str | None = None


def manifest_payload(harness: HarnessManifest) -> dict[str, Any]:
    """Schema-shaped canonical dict (skills sorted by name, pool entries sorted)."""
    payload: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "skills": [
            {"name": pin.name, "version": pin.version, "skill_md_sha256": pin.skill_md_sha256}
            for pin in sorted(harness.skills, key=lambda pin: pin.name)
        ],
        "routing_registry_version": harness.routing_registry_version,
        "host_version": {"foundationskills": harness.foundationskills_version,
                         "source_commit": harness.source_commit},
        "model_pool": {"registry_sha256": harness.pool_registry_sha256,
                       "entries": sorted(harness.pool_entries)},
        "substrate_image": harness.substrate_image,
        "seed": harness.seed,
        "reset_semantics": harness.reset_semantics,
        "workspace_snapshot": harness.workspace_snapshot,
    }
    if harness.interpreter_sha256 is not None:
        payload["interpreter_sha256"] = harness.interpreter_sha256
    return payload


def manifest_hash(harness_or_payload: HarnessManifest | dict[str, Any]) -> str:
    """The 16-hex-char harness hash (same convention as orchestrator.plan_hash)."""
    payload = (harness_or_payload if isinstance(harness_or_payload, dict)
               else manifest_payload(harness_or_payload))
    return sha256_json(payload)[:16]


def validate_manifest(raw: Any) -> list[str]:
    """Human-readable problems with a decoded manifest; [] when valid. Never raises."""
    problems: list[str] = []
    if not isinstance(raw, dict):
        return ["manifest: not a mapping"]
    if raw.get("schema") != SCHEMA_ID:
        problems.append(f"schema: expected {SCHEMA_ID!r}, got {raw.get('schema')!r}")
    problems += [f"{key}: missing" for key in REQUIRED if key not in raw]
    allowed = set(REQUIRED) | {"schema", "interpreter_sha256"}
    problems += [f"{key}: unknown key" for key in raw if key not in allowed]
    skills = raw.get("skills")
    if "skills" in raw and (not isinstance(skills, list) or not skills):
        problems.append("skills: must be a non-empty array")
    elif isinstance(skills, list):
        for index, item in enumerate(skills):
            problems += _skill_problems(index, item)
    host = raw.get("host_version")
    if "host_version" in raw and not (isinstance(host, dict) and "foundationskills" in host and "source_commit" in host):
        problems.append("host_version: needs foundationskills and source_commit")
    pool = raw.get("model_pool")
    if "model_pool" in raw and not (isinstance(pool, dict) and "registry_sha256" in pool and "entries" in pool):
        problems.append("model_pool: needs registry_sha256 and entries")
    seed = raw.get("seed")
    if "seed" in raw and (isinstance(seed, bool) or (seed is not None and not isinstance(seed, int))):
        problems.append(f"seed: must be an integer or null, got {type(seed).__name__}")
    for key in ("reset_semantics", "workspace_snapshot"):
        if key in raw and (not isinstance(raw[key], str) or not raw[key]):
            problems.append(f"{key}: must be a non-empty string")
    return problems


def _skill_problems(index: int, item: Any) -> list[str]:
    """Problems with one skills[] entry (helper for validate_manifest)."""
    if not isinstance(item, dict):
        return [f"skills[{index}]: not a mapping"]
    problems = [
        f"skills[{index}].{key}: must be a non-empty string"
        for key in ("name", "version")
        if not isinstance(item.get(key), str) or not item[key]
    ]
    digest = item.get("skill_md_sha256")
    if digest is not None and not (isinstance(digest, str) and _HEX64.match(digest)):
        problems.append(f"skills[{index}].skill_md_sha256: must be 64 lowercase hex or null")
    return problems


def unmeasured_fields(raw: dict) -> list[str]:
    """Dotted names of fields whose value is the "unmeasured" marker: gaps, not errors."""
    gaps: list[str] = []
    if raw.get("routing_registry_version") == UNMEASURED:
        gaps.append("routing_registry_version")
    pool = raw.get("model_pool")
    if isinstance(pool, dict) and pool.get("registry_sha256") == UNMEASURED:
        gaps.append("model_pool.registry_sha256")
    return gaps


def skill_pin(name: str, version: str, skill_md: Path | None) -> SkillPin:
    """Hash SKILL.md into the pin; sha None when the path is absent or missing."""
    if skill_md is None or not Path(skill_md).is_file():
        return SkillPin(name=name, version=version, skill_md_sha256=None)
    try:
        return SkillPin(name=name, version=version, skill_md_sha256=sha256_file(skill_md))
    except OSError:
        return SkillPin(name=name, version=version, skill_md_sha256=None)
