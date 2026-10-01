"""Standard skill contract: every registered skill -- current and future --
must pass these checks. A new skill that plugs in via register_skill or the
entry-point group is picked up automatically."""
from __future__ import annotations

import importlib.resources
import json
import re
import sys

import pytest
import yaml

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


# Agent Skills packaging (adopted from NeMo-RL skills/, see artifacts/research/nemo-rl-skills.md):
# each skill package's SKILL.md opens with routing frontmatter, and evals/evals.json carries
# positive and negative routing cases. One package may register several skills (training).
PACKAGES: dict[str, list[BaseSkill]] = {}
for _s in SKILLS:
    PACKAGES.setdefault(sys.modules[type(_s).__module__].__package__, []).append(_s)
SKILL_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")  # agentskills.io: lowercase + hyphens, <= 64


def _frontmatter(package: str) -> dict:
    text = importlib.resources.files(package).joinpath("SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert match, f"{package}/SKILL.md does not open with --- frontmatter ---"
    meta = yaml.safe_load(match.group(1))
    assert isinstance(meta, dict), f"{package}/SKILL.md frontmatter is not a mapping"
    return meta


FRONTMATTER = {p: _frontmatter(p) for p in PACKAGES}
PACKAGE_NAMES = {m.get("name") for m in FRONTMATTER.values()}


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_skill_md_frontmatter_routes_the_package(package):
    meta = FRONTMATTER[package]
    missing = [k for k in ("name", "license", "description", "when_to_use") if not meta.get(k)]
    assert not missing, f"{package}/SKILL.md frontmatter lacks {missing}"
    assert SKILL_NAME.match(meta["name"]) and len(meta["name"]) <= 64, meta["name"]
    assert isinstance(meta["description"], str) and len(meta["description"]) <= 1024
    assert "Do NOT use" in meta["description"], "description must say what the skill is not for"
    when = meta["when_to_use"]
    assert isinstance(when, list) and len(when) >= 3 and all(isinstance(w, str) and w for w in when)


def test_frontmatter_names_are_unique():
    names = [m["name"] for m in FRONTMATTER.values()]
    assert len(names) == len(set(names)), names


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_routing_evals_cover_positive_and_negative_cases(package):
    name = FRONTMATTER[package]["name"]
    path = importlib.resources.files(package).joinpath("evals", "evals.json")
    assert path.is_file(), f"{package} has no evals/evals.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(cases, list) and cases
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "eval ids must be unique"
    # a cited id must be a declared rule or one SKILL.md documents (e.g. DE-RDY-* readiness checks)
    doc = importlib.resources.files(package).joinpath("SKILL.md").read_text(encoding="utf-8")
    declared = {r.rule_id for s in SKILLS for r in s.rules} | set(re.findall(r"\b[A-Z]{2,4}-[A-Z]{2,5}-\d+\b", doc))
    for case in cases:
        assert case.get("question") and case.get("ground_truth") and case.get("expected_behavior"), case["id"]
        assert case["expected_skill"] in PACKAGE_NAMES | {None}, f"{case['id']}: unknown skill {case['expected_skill']!r}"
        assert case["should_trigger"] is (case["expected_skill"] == name), case["id"]
        prose = " ".join([case["ground_truth"], *case["expected_behavior"]])
        unknown = {r for r in re.findall(r"\b[A-Z]{2,4}-[A-Z]{2,5}-\d+\b", prose)} - declared
        assert not unknown, f"{case['id']} cites undeclared rules {sorted(unknown)}"
    positives = [c for c in cases if c["should_trigger"]]
    assert len(positives) >= 3, "need at least 3 cases that should trigger this skill"
    assert any(c["expected_skill"] is None for c in cases), "need a case no FoundationSkills skill should take"
    assert any(c["expected_skill"] not in (None, name) for c in cases), "need a case routed to a sibling skill"


def test_skill_chain_reaches_launch_spec_from_raw_data():
    registry = SkillRegistry()
    register_builtin_skills(registry)
    chain = registry.plan_chain({"raw_data_ref", "goal_spec"}, "fs_launch_spec")
    assert chain[-1] == "training.emit" and "data_engine" in chain and "training.planner" in chain
