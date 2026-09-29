"""Exit-code matrix for EvalSkill with an injected fake harness (no GPU, no lm_eval import)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from foundationskills.core.contract import SkillContext
from foundationskills.core.status import Status
from foundationskills.skills.evaluation.runner import HarnessOutcome, HarnessRequest
from foundationskills.skills.evaluation.skill import EvalSkill

POLICY = """
policy_version: 1
default: {k: 2.0, abs_epsilon: 0.01}
tasks:
  mmlu: {metric: "acc,none", num_fewshot: 5, dataset: cais/mmlu}
  gsm8k: {metric: "exact_match,strict-match", num_fewshot: 5}
  judged: {metric: "score", judge: true}
"""


class FakeHarnessRunner:
    """Scores per role; ``None`` = harness ran but reported no metric; an Exception string = harness error."""

    name = "fake-lm-eval"

    def __init__(self, checkpoint=0.50, baseline=0.50, stderr=0.004):
        self.scores = {"checkpoint": checkpoint, "baseline": baseline}
        self.stderr = stderr
        self.calls: list[HarnessRequest] = []

    def run(self, request: HarnessRequest) -> HarnessOutcome:
        self.calls.append(request)
        score = self.scores[request.role]
        argv = ("lm-eval", "run", "--tasks", request.task)
        if isinstance(score, str):
            return HarnessOutcome(1, None, argv, score)
        per_task = {} if score is None else {"acc,none": score, "acc_stderr,none": self.stderr}
        return HarnessOutcome(0, {"results": {request.task: per_task},
                                  "n-samples": {request.task: {"effective": 100}},
                                  "task_hashes": {request.task: "abc"}}, argv)


def _model(path: Path, chat=False) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    if chat:
        (path / "chat_template.jinja").write_text("x")
    return path


@pytest.fixture
def env(tmp_path):
    cache = tmp_path / "cache"
    (cache / "hub" / "datasets--cais--mmlu").mkdir(parents=True)
    policy = tmp_path / "eval_policy.yaml"
    policy.write_text(POLICY)
    req = {
        "checkpoint": str(_model(tmp_path / "ck")),
        "base": str(_model(tmp_path / "base")),
        "benchmarks": ["mmlu"],
        "policy": str(policy),
        "eval_cache": str(cache),
        "out": str(tmp_path / "eval" / "eval_report.json"),
        "baseline_cache": str(tmp_path / "baselines"),
    }
    return tmp_path, req


def _run(tmp_path, req, runner=None, version="0.4.12"):
    skill = EvalSkill(runner=runner or FakeHarnessRunner(), version_probe=lambda: version)
    return skill, skill.execute(req, SkillContext(workdir=tmp_path))


def _report(req):
    return json.loads(Path(req["out"]).read_text())


def test_fixtures_cover_every_rule():
    skill = EvalSkill(runner=FakeHarnessRunner())
    assert set(skill.must_fire_fixtures()) == {r.rule_id for r in skill.rules}


def test_pass_writes_a_schema_valid_report(env):
    tmp_path, req = env
    _, result = _run(tmp_path, req)
    assert result.status is Status.PASS and result.exit_code == 0
    rep = _report(req)
    assert rep["verdict"] == "PASS" and rep["harness"]["version"] == "0.4.12"
    row = rep["benchmarks"][0]
    assert row["score"] == row["baseline"] == 0.5 and row["baseline_provenance"] == "measured"
    assert row["n"] == 100 and row["task_hash"] == "abc" and row["num_fewshot"] == 5
    assert result.artifacts[0].type == "eval_report"


def test_three_point_drop_is_red(env):
    tmp_path, req = env
    _, result = _run(tmp_path, req, FakeHarnessRunner(checkpoint=0.47, baseline=0.50))
    assert result.status is Status.RED and result.exit_code == 5
    assert any(f.rule_id == "EV-HO-001" for f in result.findings)
    assert _report(req)["verdict"] == "RED"


def test_three_point_improvement_passes(env):
    tmp_path, req = env
    _, result = _run(tmp_path, req, FakeHarnessRunner(checkpoint=0.53, baseline=0.50))
    assert result.status is Status.PASS


def test_skipped_benchmark_is_red(env):
    tmp_path, req = env
    _, result = _run(tmp_path, req, FakeHarnessRunner(checkpoint=None))
    assert result.status is Status.RED
    assert any(f.rule_id == "EV-HO-002" for f in result.findings)
    assert _report(req)["benchmarks"][0]["status"] == "skipped"


def test_harness_error_is_unmeasured_with_null_score(env):
    tmp_path, req = env
    _, result = _run(tmp_path, req, FakeHarnessRunner(checkpoint="CUDA out of memory"))
    assert result.status is Status.UNMEASURED and result.exit_code == 95
    row = _report(req)["benchmarks"][0]
    assert row["score"] is None and "CUDA out of memory" in row["error"]


def test_unstaged_dataset_is_unmeasured_named_and_never_run(env):
    tmp_path, req = env
    (Path(req["eval_cache"]) / "hub" / "datasets--cais--mmlu").rmdir()
    runner = FakeHarnessRunner()
    _, result = _run(tmp_path, req, runner)
    assert result.status is Status.UNMEASURED
    assert "missing input: dataset cais/mmlu not staged" in _report(req)["benchmarks"][0]["error"]
    assert runner.calls == []


def test_breach_beats_unmeasured(env):
    tmp_path, req = env
    (Path(req["eval_cache"]) / "hub" / "datasets--cais--mmlu").rmdir()
    req["benchmarks"] = ["mmlu", "gsm8k"]
    runner = FakeHarnessRunner(checkpoint=0.40, baseline=0.50)
    runner.run = _gsm8k_metric(runner.run)
    _, result = _run(tmp_path, req, runner)
    assert result.status is Status.RED
    assert [b["status"] for b in _report(req)["benchmarks"]] == ["unmeasured", "red"]


def _gsm8k_metric(inner):
    def run(request):
        out = inner(request)
        per = out.results["results"][request.task]
        out.results["results"][request.task] = {"exact_match,strict-match": per["acc,none"],
                                                "exact_match_stderr,strict-match": per["acc_stderr,none"]}
        return out
    return run


def test_limited_run_can_never_pass_and_does_not_seed_the_cache(env):
    tmp_path, req = env
    req["limit"] = 8
    _, result = _run(tmp_path, req)
    assert result.status is Status.UNMEASURED
    assert any(f.rule_id == "EV-HO-003" for f in result.findings)
    assert _report(req)["limited"] is True
    assert not Path(req["baseline_cache"]).exists()


def test_second_run_reuses_the_cached_baseline(env):
    tmp_path, req = env
    _run(tmp_path, req)
    runner = FakeHarnessRunner()
    _, result = _run(tmp_path, req, runner)
    assert result.status is Status.PASS
    assert [c.role for c in runner.calls] == ["checkpoint"]
    assert _report(req)["benchmarks"][0]["baseline_provenance"] == "cached"


def test_changed_seed_misses_the_cache(env):
    tmp_path, req = env
    _run(tmp_path, req)
    runner = FakeHarnessRunner()
    _run(tmp_path, {**req, "seed": 7}, runner)
    assert [c.role for c in runner.calls] == ["checkpoint", "baseline"]


def test_adapter_is_run_on_its_base_with_peft_and_identical_flags(env):
    tmp_path, req = env
    ad = tmp_path / "adapter"
    ad.mkdir()
    (ad / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": req["base"]}))
    (ad / "tokenizer_config.json").write_text("{}")
    runner = FakeHarnessRunner()
    _, result = _run(tmp_path, {**{k: v for k, v in req.items() if k != "base"}, "checkpoint": str(ad)}, runner)
    assert result.status is Status.PASS
    ck, bl = runner.calls
    assert ck.model_dir == bl.model_dir == req["base"] and ck.adapter_dir == str(ad) and bl.adapter_dir is None
    assert ck.tokenizer_dir == str(ad) and bl.tokenizer_dir is None
    for field in ("num_fewshot", "seed", "dtype", "chat_template", "gen_kwargs", "limit"):
        assert getattr(ck, field) == getattr(bl, field)


def test_chat_template_only_when_both_ship_one(env):
    tmp_path, req = env
    (Path(req["checkpoint"]) / "chat_template.jinja").write_text("x")
    runner = FakeHarnessRunner()
    _run(tmp_path, req, runner)
    assert runner.calls[0].chat_template is False
    (Path(req["base"]) / "chat_template.jinja").write_text("x")
    runner = FakeHarnessRunner()
    _run(tmp_path, {**req, "baseline_cache": str(tmp_path / "bl2")}, runner)
    assert runner.calls[0].chat_template is True


def test_missing_stderr_without_explicit_epsilon_refuses_without_a_report(env):
    tmp_path, req = env
    _, result = _run(tmp_path, req, FakeHarnessRunner(stderr="N/A"))
    assert result.status is Status.REFUSED and result.exit_code == 96
    assert "no measured stderr" in result.refusal
    assert not Path(req["out"]).exists()


# -- must-fire refusals: exit 96, no report file --------------------------------------------------

@pytest.mark.parametrize("rule,mutate,version", [
    ("EV-IN-001", lambda r, t: r.update(checkpoint=str(t / "missing")), "0.4.12"),
    ("EV-IN-002", lambda r, t: r.pop("base"), "0.4.12"),
    ("EV-IN-003", lambda r, t: r.update(eval_cache=str(t / "missing")), "0.4.12"),
    ("EV-IN-004", lambda r, t: None, "0.4.11"),
    ("EV-IN-004", lambda r, t: None, None),
    ("EV-IN-005", lambda r, t: r.update(benchmarks=["not_in_policy"]), "0.4.12"),
    ("EV-IN-005", lambda r, t: r.update(benchmarks=["judged"]), "0.4.12"),
    ("EV-IN-005", lambda r, t: r.update(policy=str(t / "missing.yaml")), "0.4.12"),
])
def test_must_fire_refusals_write_no_report(env, rule, mutate, version):
    tmp_path, req = env
    mutate(req, tmp_path)
    runner = FakeHarnessRunner()
    _, result = _run(tmp_path, req, runner, version=version)
    assert result.status is Status.REFUSED and result.exit_code == 96
    assert rule in result.refusal
    assert "missing input:" in result.refusal or "precondition failed:" in result.refusal
    assert not Path(req["out"]).exists() and runner.calls == []


def test_must_fire_ev_ho_004_report_gone_before_handoff(env):
    tmp_path, req = env

    class Deleting(EvalSkill):
        def check_handoff(self, result, ctx):
            os.unlink(result.artifacts[0].path)
            return super().check_handoff(result, ctx)

    skill = Deleting(runner=FakeHarnessRunner(), version_probe=lambda: "0.4.12")
    result = skill.execute(req, SkillContext(workdir=tmp_path))
    assert result.status is Status.RED
    assert any(f.rule_id == "EV-HO-004" and "missing on disk" in f.message for f in result.findings)


def test_unknown_request_field_is_refused(env):
    tmp_path, req = env
    _, result = _run(tmp_path, {**req, "download": True})
    assert result.status is Status.REFUSED
