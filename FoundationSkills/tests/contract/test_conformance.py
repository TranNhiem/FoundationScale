"""Standard skill contract: every registered skill -- current and future --
must pass these checks. A new skill that plugs in via register_skill or the
entry-point group is picked up automatically."""
from __future__ import annotations

import importlib.resources
import re
import sys

import pytest

from foundationskills.core import ArtifactType, BaseSkill, SkillRegistry
from foundationskills.core.artifacts import register_artifact_type  # noqa: F401  (future types extend the set)
from foundationskills.skills import register_builtin_skills

REQUIRED_SECTIONS = ("Purpose", "When to use", "Inputs", "Outputs", "Scope", "Validation rules",
                     "Failure handling", "FS interface", "Worked examples")
RULE_ID = re.compile(r"^[A-Z]{2,4}-[A-Z0-9_-]+$")
KNOWN_TYPES = {t.value for t in ArtifactType}


def _skills() -> list[BaseSkill]:
    registry = SkillRegistry()
    register_builtin_skills(registry)
    return [registry.get(n) for n in registry.names()]


SKILLS = _skills()


def test_builtin_skills_registered():
    names = {s.name for s in SKILLS}
    assert {"data_engine", "training.planner", "training.emit"} <= names


@pytest.mark.parametrize("skill", SKILLS, ids=lambda s: s.name)
def test_contract_attributes(skill):
    assert re.match(r"^[a-z][a-z0-9_.]*$", skill.name)
    assert re.match(r"^\d+\.\d+\.\d+$", skill.version)
    assert skill.description and isinstance(skill.description, str)
    assert set(skill.consumes) <= KNOWN_TYPES, f"unknown consumed types {set(skill.consumes) - KNOWN_TYPES}"
    assert skill.produces and set(skill.produces) <= KNOWN_TYPES
    for schema in (skill.input_schema, skill.output_schema):
        assert isinstance(schema, dict) and schema.get("type") == "object"
    assert skill.fs_interface is not None and isinstance(skill.fs_interface.emits, tuple)


@pytest.mark.parametrize("skill", SKILLS, ids=lambda s: s.name)
def test_rules_unique_and_must_fire_covered(skill):
    ids = [r.rule_id for r in skill.rules]
    assert ids and len(ids) == len(set(ids)), "rule ids must be unique"
    assert all(RULE_ID.match(i) for i in ids)
    fixtures = skill.must_fire_fixtures()
    missing = sorted(set(ids) - set(fixtures))
    assert not missing, f"every rule needs a MUST_FIRE fixture; missing: {missing}"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda s: s.name)
def test_skill_md_documents_the_contract(skill):
    package = sys.modules[type(skill).__module__].__package__
    text = importlib.resources.files(package).joinpath("SKILL.md").read_text(encoding="utf-8")
    headings = {h.strip() for h in re.findall(r"^##\s+(.+)$", text, re.M)}
    missing = [s for s in REQUIRED_SECTIONS if s not in headings]
    assert not missing, f"{package}/SKILL.md lacks sections {missing}"
    undocumented = [r.rule_id for r in skill.rules if r.rule_id not in text]
    assert not undocumented, f"{package}/SKILL.md does not list rules {undocumented}"


def test_skill_chain_reaches_launch_spec_from_raw_data():
    registry = SkillRegistry()
    register_builtin_skills(registry)
    chain = registry.plan_chain({"raw_data_ref", "goal_spec"}, "fs_launch_spec")
    assert chain[-1] == "training.emit" and "data_engine" in chain and "training.planner" in chain
