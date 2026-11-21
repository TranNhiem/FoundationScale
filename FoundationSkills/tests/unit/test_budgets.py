"""C4.2: a declared budget breached by a measured counter is never PASS."""

import time

from foundationskills.core.contract import SkillContext
from foundationskills.core.orchestrator import plan_hash
from foundationskills.interfaces.fs.launch import launch
from foundationskills.skills.compression.skill import CompressionSkill


class SlowCheck(CompressionSkill):
    budgets = {"wall_clock_ms": 1}

    def _execute_core(self, request, ctx):
        time.sleep(0.01)
        return super()._execute_core(request, ctx)


def test_a_refused_call_keeps_refused_and_gains_core_budget(tmp_path):
    res = SlowCheck().execute({"action": "quantize"}, SkillContext(workdir=tmp_path))
    assert res.status.value == "REFUSED" and any(f.rule_id == "CORE-BUDGET" for f in res.findings)


def test_a_pass_over_budget_becomes_red(tmp_path):
    from foundationskills.core.contract import SkillResult
    from foundationskills.core.status import Status

    class SlowPass(SlowCheck):
        def _execute_core(self, request, ctx):
            time.sleep(0.01)
            return SkillResult(Status.PASS, {})

    res = SlowPass().execute({"action": "check"}, SkillContext(workdir=tmp_path))
    assert res.status.value == "RED" and res.findings[-1].rule_id == "CORE-BUDGET"


def test_no_budget_declared_changes_nothing(tmp_path):
    res = CompressionSkill().execute({"action": "quantize"}, SkillContext(workdir=tmp_path))
    assert not any(f.rule_id == "CORE-BUDGET" for f in res.findings)


def _spec(tmp_path, budget):
    return {"stage_name": "probe", "entry": "fskills-merge", "argv": ["/bin/sleep", "0.3"], "env": {}, "sbatch": None,
            "dry_run_argv": None, "expected_outputs": [], "executable": True, "missing": None,
            "cwd": str(tmp_path), "output_dir": str(tmp_path / "out"), "budget": budget}


def test_launch_overrun_turns_a_success_red(tmp_path):
    spec = _spec(tmp_path, {"wall_seconds": 0.05})
    out = launch(spec, confirm=plan_hash(spec), submit=False)
    assert out["returncode"] == 5 and out["budget_breach"] is True


def test_launch_within_budget_stays_green(tmp_path):
    spec = _spec(tmp_path, {"wall_seconds": 30})
    out = launch(spec, confirm=plan_hash(spec), submit=False)
    assert out["returncode"] == 0 and out["budget_breach"] is False
