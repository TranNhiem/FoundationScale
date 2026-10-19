"""A3.2: skill invocation classes and the call hierarchy (a user skill composes model skills, never a user skill)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from types import SimpleNamespace

from foundationskills.core.registry import SkillRegistry, call_hierarchy_problems
from foundationskills.skills import register_builtin_skills

from foundationskills.core.registry import INVOCATIONS


def _skills() -> dict:
    registry = SkillRegistry()
    register_builtin_skills(registry)
    return {name: registry.get(name) for name in registry.names()}


SKILLS = _skills()


def _frontmatter(skill) -> dict:
    package = sys.modules[type(skill).__module__].__package__
    path = Path(sys.modules[package].__file__).parent / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text.split("---", 2)[1])


@pytest.mark.parametrize("name", sorted(SKILLS))
def test_the_class_and_the_skill_md_declare_the_same_invocation(name: str) -> None:
    skill = SKILLS[name]
    assert skill.invocation in INVOCATIONS, f"{name}: invocation {skill.invocation!r} not in {INVOCATIONS}"
    declared = _frontmatter(skill)["metadata"]["invocation"]
    assert declared == skill.invocation, f"{name}: SKILL.md says {declared!r}, the class says {skill.invocation!r}"


def test_the_call_hierarchy_holds_over_every_registered_skill() -> None:
    assert call_hierarchy_problems(SKILLS) == []


def test_the_hierarchy_check_catches_each_violation() -> None:
    model = SimpleNamespace(invocation="model", invokes=())
    fake = {
        "m": model,
        "u_ok": SimpleNamespace(invocation="user", invokes=("m",)),
        "u_calls_u": SimpleNamespace(invocation="user", invokes=("u_ok",)),
        "u_missing": SimpleNamespace(invocation="user", invokes=("nope",)),
        "u_self": SimpleNamespace(invocation="user", invokes=("u_self",)),
        "m_composes": SimpleNamespace(invocation="model", invokes=("m",)),
        "bad_class": SimpleNamespace(invocation="agent", invokes=()),
        "bad_type": SimpleNamespace(invocation="user", invokes=["m"]),
    }
    problems = "\n".join(call_hierarchy_problems(fake))
    assert "u_calls_u invokes 'u_ok', a user-invoked skill" in problems
    assert "u_missing invokes 'nope', which is not a registered skill" in problems
    assert "u_self invokes itself" in problems
    assert "m_composes: a model-invoked skill declares invokes" in problems
    assert "bad_class: invocation 'agent'" in problems
    assert "bad_type: invokes must be a tuple" in problems
    assert "u_ok" not in problems.replace("'u_ok'", "")  # the legal composition raises nothing


def test_every_skill_runs_under_the_orchestrator(tmp_path) -> None:
    # measured 2026-10-18: the orchestrator's `inputs` key made seven strict-schema skills refuse every step
    from foundationskills.core.contract import SkillContext
    from foundationskills.core.orchestrator import Orchestrator, Step

    registry = SkillRegistry()
    register_builtin_skills(registry)
    for name in registry.names():
        fixture = next(iter(registry.get(name).must_fire_fixtures().values()))
        request = dict(fixture.get("request", fixture))
        orch = Orchestrator(registry, SkillContext(workdir=tmp_path / name.replace(".", "_")))
        result = orch.run([Step(name, request)])[0]
        assert "inputs: additional propert" not in str(result.refusal), f"{name}: {result.refusal}"


def test_a_user_skill_is_expected_only_when_the_question_invokes_it() -> None:
    # measured 2026-10-18: unmarked user skills drew 34 false routing triggers from natural-language requests
    from foundationskills.agent.routing_eval import load_index

    index = load_index()
    user_names = {entry["name"] for entry in index if entry["invocation"] == "user"}
    assert user_names, "no user-invoked package found"
    for entry in index:
        for case in entry["evals"]:
            expected = case.get("expected_skill")
            if expected in user_names:
                assert f"/{expected}" in case["question"], (
                    f"{case['id']}: expects user skill {expected!r} but the question never invokes /{expected}")


def test_a_user_skill_oracle_question_invokes_the_skill() -> None:
    import importlib

    from foundationskills.agent import oracle_cases

    for name, skill in SKILLS.items():
        if skill.invocation != "user":
            continue
        package = importlib.import_module(type(skill).__module__).__package__
        for case in oracle_cases.load_cases(package)[0]:
            slash = oracle_cases._slash_name(case)
            assert case["question"].startswith(f"/{slash} "), f"{case['id']}: question never invokes /{slash}"
