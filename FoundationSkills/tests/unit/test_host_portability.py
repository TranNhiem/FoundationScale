"""A1.5: on a host without FoundationScale, a skill that needs it REFUSES with the install
step (CORE-HOST-001) instead of producing a plan nothing can run; skills that do not need it
still run."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from foundationskills.core.contract import HOST_INSTALL_STEP, SkillContext, Status
from foundationskills.skills.auto_research.skill import AutoResearchSkill
from foundationskills.skills.measured_or_unmeasured.skill import MeasuredOrUnmeasuredSkill
from foundationskills.skills.training.emit_skill import TrainingEmitSkill
from foundationskills.skills.training.skill import TrainingPlannerSkill


@pytest.fixture()
def foreign_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "foundationscale", None
    )  # import foundationscale -> ImportError


@pytest.mark.usefixtures("foreign_host")
@pytest.mark.parametrize(
    ("skill", "request_"),
    [
        (TrainingPlannerSkill(), {"goal": {"stages": []}}),
        (TrainingEmitSkill(), {"plan": {}, "model": "m", "output_root": "/nonexistent-out"}),
        (AutoResearchSkill(), {"action": "submit"}),
    ],
    ids=["training.planner", "training.emit", "auto_research.submit"],
)
def test_fs_dependent_skills_refuse_with_the_install_step(
    skill, request_: dict, tmp_path: Path
) -> None:
    result = skill.execute(request_, SkillContext(workdir=tmp_path))
    assert result.status is Status.REFUSED and result.exit_code == 96
    finding = result.findings[0]
    assert finding.rule_id == "CORE-HOST-001" and "not importable on this host" in finding.message
    assert finding.recovery == HOST_INSTALL_STEP and "pip install -e" in finding.recovery
    assert "CORE-HOST-001" in (result.refusal or "")


@pytest.mark.usefixtures("foreign_host")
def test_ledger_actions_run_without_fs() -> None:
    assert AutoResearchSkill().fs_required({"action": "check"}) is False
    assert AutoResearchSkill().fs_required({"action": "submit"}) is True


@pytest.mark.usefixtures("foreign_host")
def test_a_skill_that_does_not_need_fs_is_not_refused_for_it(tmp_path: Path) -> None:
    result = MeasuredOrUnmeasuredSkill().execute(
        {"claims": [{"id": "c", "claim": "unbacked", "asserted_status": "PASS"}]},
        SkillContext(workdir=tmp_path),
    )
    assert all(f.rule_id != "CORE-HOST-001" for f in result.findings)


def test_with_fs_present_the_host_rule_never_fires(tmp_path: Path) -> None:
    result = TrainingPlannerSkill().execute(
        {"goal": {"stages": []}}, SkillContext(workdir=tmp_path)
    )
    assert all(f.rule_id != "CORE-HOST-001" for f in result.findings)
