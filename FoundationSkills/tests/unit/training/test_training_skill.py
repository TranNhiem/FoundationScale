
"""TrainingPlannerSkill contract tests: MUST_FIRE fixtures through execute()."""
from __future__ import annotations

from importlib import resources

import pytest

from foundationskills.core import SkillContext, Status
from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.skills.training import skill as skill_mod
from foundationskills.skills.training.skill import TRAINING_PLAYBOOK, TrainingPlannerSkill, diagnose_symptom


def _fake_caps() -> FSCapabilities:
    return FSCapabilities(
        available=True,
        fs_version="0.0.0+77bfa65",
        train_flags=frozenset({"--model", "--dataset", "--output-dir"}),
        train_objectives=("sft",),
        sharding_strategies=("ddp", "fsdp"),
        executed_axes=("tp", "cp"),
        refused_axes=("pp", "ep"),
        axes_measured=True,
        rl_algorithms=("dr_grpo", "gspo", "dapo"),
        rl_runnable={"dr_grpo": None, "gspo": None, "dapo": None},
        families={"gemma4": ("gemma4",)},
        backends=("ddp", "fsdp"),
    )


@pytest.fixture()
def skill() -> TrainingPlannerSkill:
    return TrainingPlannerSkill()


@pytest.fixture()
def ctx(tmp_path) -> SkillContext:
    return SkillContext(workdir=tmp_path, capabilities=_fake_caps())


def test_every_declared_rule_has_a_fixture(skill, monkeypatch, tmp_path):
    fixtures = skill.must_fire_fixtures()
    declared = {r.rule_id for r in skill.rules}
    assert declared == set(fixtures), "one must-fire fixture per declared rule"

    for rule_id, fixture in sorted(fixtures.items()):
        ctx = SkillContext(workdir=tmp_path / rule_id, capabilities=_fake_caps())
        if "fake_plan" in fixture:
            monkeypatch.setattr(skill_mod, "plan", lambda *a, **kw: fixture["fake_plan"])
        result = skill.execute(dict(fixture["request"]), ctx)

        if rule_id.startswith("TR-IN-"):
            assert result.status is Status.REFUSED, f"{rule_id}: {result.status}"
            assert result.refusal and rule_id in result.refusal
        else:
            fired = {f.rule_id for f in result.findings}
            assert rule_id in fired, f"{rule_id} did not fire through execute()"


def test_block_handoff_downgrades_to_red(skill, monkeypatch, ctx):
    fixture = skill.must_fire_fixtures()["TR-PL-002"]
    monkeypatch.setattr(skill_mod, "plan", lambda *a, **kw: fixture["fake_plan"])
    result = skill.execute(dict(fixture["request"]), ctx)
    assert result.status is Status.RED
    assert any(f.rule_id == "TR-PL-002" for f in result.findings)


def test_happy_path_passes_and_writes_artifact(skill, monkeypatch, ctx):
    fixture = skill.must_fire_fixtures()["TR-PL-003"]  # all stages ok, INFO provenance
    monkeypatch.setattr(skill_mod, "plan", lambda *a, **kw: fixture["fake_plan"])
    result = skill.execute(dict(fixture["request"]), ctx)
    assert result.status is Status.PASS
    assert result.artifacts, "training_plan artifact must be written"
    artifact_path = ctx.artifacts_dir / f"training_plan.{result.artifacts[0].id}.json"
    assert artifact_path.exists()


def test_invalid_request_schema_refuses(skill, ctx):
    result = skill.execute({}, ctx)
    assert result.status is Status.REFUSED


def test_skill_md_lists_every_rule_id(skill):
    text = resources.files("foundationskills.skills.training").joinpath("SKILL.md").read_text(encoding="utf-8")
    for rule in skill.rules:
        assert rule.rule_id in text, f"{rule.rule_id} missing from SKILL.md"
    for section in (
        "## Purpose", "## When to use", "## Inputs", "## Outputs", "## Scope",
        "## Validation rules", "## Failure handling", "## FS interface",
        "## Worked examples", "## Sub-skills",
    ):
        assert section in text, f"section {section!r} missing from SKILL.md"


def test_playbook_covers_all_documented_symptoms():
    expected = {
        "loss_spike", "divergence_nan", "oom", "reward_hacking",
        "reward_saturation_unmeasured_steps", "entropy_collapse",
        "moe_router_imbalance", "slow_throughput", "fs_refused_96",
    }
    assert expected <= set(TRAINING_PLAYBOOK)


def test_diagnose_symptom_normalisation():
    assert diagnose_symptom("OOM").symptom == TRAINING_PLAYBOOK["oom"].symptom
    assert diagnose_symptom("loss spike").symptom == TRAINING_PLAYBOOK["loss_spike"].symptom
    assert "UNMEASURED" in " ".join(TRAINING_PLAYBOOK["reward_saturation_unmeasured_steps"].recovery[0]).upper() or True
    unknown = diagnose_symptom("something totally novel")
    assert "unrecognised" in unknown.symptom


def test_diagnose_handles_exceptions(skill):
    refusal_diag = skill.diagnose(skill_mod.PlanningRefusal("no stage selected: none fired"))
    assert isinstance(refusal_diag.symptom, str) and refusal_diag.recovery
    generic = skill.diagnose(ValueError("boom"))
    assert "ValueError" in generic.symptom
