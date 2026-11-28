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


def test_fingerprint_tracks_task_bytes(tmp_path):
    d = _task(tmp_path, "t1", "assert True\n")
    fp = vht.task_fingerprint(d)
    assert vht.reusable("t1", {"fingerprint": fp}, tmp_path)
    (d / "tests" / "test_x.py").write_text("assert False\n")
    assert not vht.reusable("t1", {"fingerprint": fp}, tmp_path)
    assert not vht.reusable(
        "t1", {}, tmp_path
    )  # legacy rows without a fingerprint are never reused


def test_from_gate_demotes_stale_pass(tmp_path):
    fresh = _task(tmp_path, "fresh", "a\n")
    _task(tmp_path, "stale", "b\n")
    gate = {
        "fresh": {"status": "PASS", "fingerprint": vht.task_fingerprint(fresh)},
        "stale": {"status": "PASS", "fingerprint": "0" * 64},
    }
    gp = tmp_path / "gate.json"
    gp.write_text(json.dumps(gate))
    rc = vht.main(
        [
            "--tasks",
            str(tmp_path / "none.jsonl"),
            "--tasks-root",
            str(tmp_path),
            "--out",
            str(tmp_path / "out.json"),
            "--from-gate",
            str(gp),
        ]
    )
    assert rc == 5  # one PASS, one UNMEASURED: never a clean exit
