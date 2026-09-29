from __future__ import annotations

import json

import pytest

from foundationskills.core.schema import SchemaError
from foundationskills.skills.evaluation.report import validate_report, verdict, write_report


@pytest.mark.parametrize("statuses,limited,expected", [
    (["pass", "pass"], False, "PASS"),
    (["pass", "red"], False, "RED"),
    (["pass", "skipped"], False, "RED"),
    (["unmeasured", "red"], False, "RED"),
    (["pass", "unmeasured"], False, "UNMEASURED"),
    (["pass"], True, "UNMEASURED"),
    ([], False, "UNMEASURED"),
])
def test_verdict_matrix(statuses, limited, expected):
    assert verdict(statuses, limited) == expected


def _report(**kw):
    r = {"checkpoint": "/c", "verdict": "PASS",
         "benchmarks": [{"name": "mmlu", "score": 0.5, "baseline": 0.5, "stderr": 0.01, "status": "pass"}],
         "harness": {"name": "lm-eval", "version": "0.4.12"}}
    r.update(kw)
    return r


def test_extra_provenance_fields_validate():
    assert validate_report(_report()) == []


def test_write_is_atomic_and_leaves_no_temp(tmp_path):
    sha = write_report(tmp_path / "eval" / "eval_report.json", _report())
    assert len(sha) == 64
    assert json.loads((tmp_path / "eval" / "eval_report.json").read_text())["verdict"] == "PASS"
    assert [p.name for p in (tmp_path / "eval").iterdir()] == ["eval_report.json"]


def test_invalid_report_is_not_written(tmp_path):
    with pytest.raises(SchemaError):
        write_report(tmp_path / "r.json", _report(verdict="REFUSED"))
    assert not (tmp_path / "r.json").exists()
