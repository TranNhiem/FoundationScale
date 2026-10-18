"""A3.2 composition helper: inner outcomes map to classes; inner findings become evidence rows; undeclared refused."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from foundationskills.core.contract import SkillContext
from foundationskills.core.status import Status
from foundationskills.skills import _workflow


def test_running_an_undeclared_skill_is_refused(tmp_path) -> None:
    caller = SimpleNamespace(name="fs_x", invokes=("probe_first",))
    with pytest.raises(ValueError, match="does not declare"):
        _workflow.run_step(caller, "evaluation", {}, SkillContext(workdir=tmp_path))


def test_a_refused_inner_run_is_classed_refused_and_its_findings_become_rows(tmp_path) -> None:
    caller = SimpleNamespace(name="fs_x", invokes=("probe_first",))
    fixture = next(iter(_workflow.registry().get("probe_first").must_fire_fixtures().values()))
    result, step = _workflow.run_step(caller, "probe_first", dict(fixture["request"]), SkillContext(workdir=tmp_path))
    assert result.status is Status.REFUSED and _workflow.outcome_class(result) == "refused"
    assert step.as_row()["exit_code"] == 96 and step.findings and all("rule_id" in f for f in step.findings)


@pytest.mark.parametrize(("status", "klass"), [(Status.PASS, "ok"), (Status.RED, "red"),
                                               (Status.UNMEASURED, "unmeasured"), (Status.REFUSED, "refused")])
def test_every_status_has_one_outcome_class(status, klass) -> None:
    assert _workflow.outcome_class(SimpleNamespace(status=status)) == klass
