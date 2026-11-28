"""Gate verdicts are reused only on the exact task bytes they were measured on."""

import importlib.util
import json
from pathlib import Path

_P = (
    Path(__file__).resolve().parents[2]
    / "examples/agentic-rl-GB200/scripts/validate_harbor_tasks.py"
)
_spec = importlib.util.spec_from_file_location("validate_harbor_tasks", _P)
vht = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vht)


def _task(root: Path, uid: str, test_body: str) -> Path:
    d = root / uid / "tests"
    d.mkdir(parents=True, exist_ok=True)
    (d / "test_x.py").write_text(test_body)
    return root / uid


ROW = {"uid": "t1", "messages": [{"role": "user", "content": "fix"}], "metadata": {"workdir": "/w"}}


def test_fingerprint_tracks_task_bytes_and_row(tmp_path):
    d = _task(tmp_path, "t1", "assert True\n")
    fp = vht.row_fingerprint(ROW, d)
    assert vht.reusable("t1", {"fingerprint": fp}, tmp_path, ROW)
    changed_row = {**ROW, "metadata": {"workdir": "/other"}}
    assert not vht.reusable("t1", {"fingerprint": fp}, tmp_path, changed_row)  # row is evidence too
    (d / "tests" / "test_x.py").write_text("assert False\n")
    assert not vht.reusable("t1", {"fingerprint": fp}, tmp_path, ROW)
    assert not vht.reusable(
        "t1", {}, tmp_path, ROW
    )  # legacy rows without a fingerprint never reused


def _rows(tmp_path, uids):
    rows = {u: {**ROW, "uid": u} for u in uids}
    p = tmp_path / "tasks.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    return p, rows


def _main(tmp_path, tasks, gate):
    gp = tmp_path / "gate.json"
    gp.write_text(json.dumps(gate))
    return vht.main(
        [
            "--tasks",
            str(tasks),
            "--tasks-root",
            str(tmp_path),
            "--out",
            str(tmp_path / "out.json"),
            "--from-gate",
            str(gp),
        ]
    )


def test_from_gate_pass_plus_stale_is_unmeasured_not_red(tmp_path):
    tasks, rows = _rows(tmp_path, ["fresh", "stale"])
    fresh = _task(tmp_path, "fresh", "a\n")
    _task(tmp_path, "stale", "b\n")
    gate = {
        "fresh": {"status": "PASS", "fingerprint": vht.row_fingerprint(rows["fresh"], fresh)},
        "stale": {"status": "PASS", "fingerprint": "0" * 64},
    }
    assert _main(tmp_path, tasks, gate) == 95


def test_from_gate_judges_only_the_current_tasks(tmp_path):
    tasks, rows = _rows(tmp_path, ["fresh", "missing"])
    fresh = _task(tmp_path, "fresh", "a\n")
    gate = {
        "fresh": {"status": "PASS", "fingerprint": vht.row_fingerprint(rows["fresh"], fresh)},
        "other": {"status": "RED"},
    }  # a verdict for a task not in --tasks is not evidence
    assert _main(tmp_path, tasks, gate) == 95  # "missing" has no stored verdict: UNMEASURED


def test_from_gate_red_is_red(tmp_path):
    tasks, rows = _rows(tmp_path, ["fresh", "bad"])
    fresh = _task(tmp_path, "fresh", "a\n")
    gate = {
        "fresh": {"status": "PASS", "fingerprint": vht.row_fingerprint(rows["fresh"], fresh)},
        "bad": {"status": "RED"},
    }
    assert _main(tmp_path, tasks, gate) == 5
