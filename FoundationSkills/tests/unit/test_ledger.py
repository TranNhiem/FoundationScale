"""C4.1: append-only ledger rows, best-effort writes, summaries, and the execute()/launch() write sites."""

import json

from foundationskills.core import ledger
from foundationskills.core.contract import SkillContext
from foundationskills.skills.compression.skill import CompressionSkill


def test_record_marks_unmeasured_counters_with_reasons():
    row = ledger.record("skill", "s", "1", verdict="PASS", exit_code=0, wall_clock_ms=1.5, request={"a": 1})
    assert row["tokens_in"] == {"value": None, "null_reason": "no_llm_call"}
    assert row["gpu_seconds"]["null_reason"] == "not_launched"
    assert row["cost_estimate_usd"]["null_reason"] == "no_price_table" and len(row["input_hash"]) == 16


def test_append_is_best_effort_and_read_skips_torn_lines(tmp_path):
    assert ledger.append(None, {}) is False
    assert ledger.append(tmp_path / "no" / "perm", ledger.record("skill", "s", "1", verdict="RED", exit_code=5,
                                                                 wall_clock_ms=2.0)) is True
    path = tmp_path / "no" / "perm" / ledger.FILENAME
    with path.open("a") as handle:
        handle.write("{torn\n")
    assert len(ledger.read(path)) == 1


def test_summarize_counts_verdicts_and_sums_only_measured():
    rows = [ledger.record("skill", "s", "1", verdict="PASS", exit_code=0, wall_clock_ms=10, tokens_in=5),
            ledger.record("skill", "s", "1", verdict="RED", exit_code=5, wall_clock_ms=30)]
    out = ledger.summarize(rows)["s"]
    assert out["calls"] == 2 and out["verdicts"] == {"PASS": 1, "RED": 1}
    assert out["tokens_in"] == 5 and out["wall_clock_ms_p50"] == 20 and out["nulls"]["tokens_in:no_llm_call"] == 1


def test_refused_call_writes_nothing_in_the_workdir_but_reaches_the_shared_ledger(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    monkeypatch.setenv("FSKILLS_LEDGER_DIR", str(shared))
    work = tmp_path / "work"
    res = CompressionSkill().execute({"action": "quantize"}, SkillContext(workdir=work))
    assert not (work / ledger.FILENAME).exists()  # REFUSED writes nothing into the workdir
    rows = ledger.read(shared / ledger.FILENAME)
    assert res.status.value == "REFUSED" and len(rows) == 1
    assert rows[0]["skill_name"] == "compression" and rows[0]["exit_code"] == 96 and "CP-IN-007" in rows[0]["rules_fired"]


def test_launch_writes_a_row_beside_the_launch_log(tmp_path):
    from foundationskills.core.orchestrator import plan_hash
    from foundationskills.interfaces.fs.launch import launch

    spec = {"stage_name": "probe", "entry": "fskills-merge", "argv": ["/bin/true"], "env": {}, "sbatch": None,
            "dry_run_argv": None, "expected_outputs": [], "executable": True, "missing": None,
            "cwd": str(tmp_path), "output_dir": str(tmp_path / "out")}
    launch(spec, confirm=plan_hash(spec), submit=False)
    rows = ledger.read(tmp_path / "out" / ledger.FILENAME)
    assert len(rows) == 1 and rows[0]["kind"] == "launch" and rows[0]["exit_code"] == 0
    assert rows[0]["gpu_seconds"]["null_reason"] == "gpu_count_unmeasured"  # it ran; the GPU count is the gap
    assert rows[0]["harness_hash"] and json.dumps(rows[0])


def test_eval_ledger_rows_carry_measured_tokens_and_skip_reused_calls():
    from foundationskills.agent.behaviour_eval import ledger_rows

    report = {"cases": "oracle", "agent_model": "a", "judge_model": "j", "calls": [
        {"package": "p", "arm": "with_skill", "case_id": "c1", "rep": 0, "measured": True,
         "agent_prompt_tokens": 100, "judge_prompt_tokens": 50, "agent_tokens": 20, "judge_tokens": 5},
        {"package": "p", "arm": "with_skill", "case_id": "c2", "rep": 0, "measured": True, "reused": True}]}
    rows = ledger_rows(report)
    assert len(rows) == 1 and rows[0]["kind"] == "eval"
    assert rows[0]["tokens_in"]["value"] == 150 and rows[0]["tokens_out"]["value"] == 25
