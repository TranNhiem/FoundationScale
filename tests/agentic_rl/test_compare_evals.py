"""compare_evals: task-clustered paired comparison of two eval dumps."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "compare_evals", Path(__file__).resolve().parents[2] / "examples/agentic-rl-GB200/scripts/compare_evals.py")
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


def _dump(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("".join(json.dumps({"session": s, "reward": r}) + "\n" for s, r in rows), encoding="utf-8")
    return p


def test_clear_improvement_is_better_and_tasks_are_the_unit(tmp_path):
    base = _dump(tmp_path, "a", [(f"t{i}#{k}", 0.0) for i in range(20) for k in range(4)])
    cand = _dump(tmp_path, "b", [(f"t{i}#{k}", 1.0 if k < 2 else 0.0) for i in range(20) for k in range(4)])
    res = ce.compare(ce.per_task(base), ce.per_task(cand), 2000, 0)
    assert res["paired_tasks"] == 20 and res["reward"]["diff"] == 0.5 and res["reward"]["verdict"] == "better"
    assert res["resolve"]["cand"] == 0.5


def test_abstentions_are_dropped_not_zeroed_and_unpaired_tasks_counted(tmp_path):
    base = _dump(tmp_path, "a", [("t1#0", 0.5), ("t1#1", None), ("t2#0", 0.2)])
    cand = _dump(tmp_path, "b", [("t1#0", 0.5), ("t3#0", 1.0)])
    res = ce.compare(ce.per_task(base), ce.per_task(cand), 500, 0)
    assert res["paired_tasks"] == 1 and res["base_only"] == 1 and res["cand_only"] == 1
    assert res["reward"]["base"] == 0.5 and res["reward"]["verdict"] == "no detectable difference"


def test_no_paired_tasks_is_unmeasured(tmp_path):
    res = ce.compare(ce.per_task(_dump(tmp_path, "a", [("x#0", 1.0)])), ce.per_task(_dump(tmp_path, "b", [("y#0", 1.0)])), 100, 0)
    assert res["reward"]["verdict"] == "unmeasured"
