"""M6 evidence-bound records: ``derive_result`` is wired into the record check and the record store.

The record entry points need no connector: every test drives them unbound on a tiny attributor
(``_Attributor``) with an m1-style request payload that carries the campaign spec (metric names and
the rl floor). Legacy asserted/crash rows keep their pre-M6 behaviour and are classified, never bound.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import foundationskills.skills.auto_research.skill as _skill_module
from foundationskills.skills.auto_research.evidence import sha256_file
from foundationskills.skills.auto_research.ledger import Ledger
from foundationskills.skills.auto_research.skill import _campaign, _spec, result_problems

FINGERPRINT = "sha256:" + "1" * 64


class _Attributor:
    """Minimal attributor: the record entry points need only ``finding``."""

    def finding(self, rule_id: str, message: str, detail: dict[str, Any], recovery: str) -> SimpleNamespace:
        return SimpleNamespace(rule_id=rule_id, message=message, detail=detail, recovery=recovery)


def _call(name: str, *args: Any) -> Any:
    """The skill's ``name`` entry point, invoked unbound (neither entry point uses instance state)."""
    owner = next(obj for obj in vars(_skill_module).values() if isinstance(obj, type) and name in vars(obj))
    return getattr(owner, name)(_Attributor(), *args)


def _base(tmp_path: Path) -> str:
    return str((tmp_path / "models" / "base").resolve())


def _rl_spec(tmp_path: Path) -> dict[str, Any]:
    """The rl-stage spec (AR-IN-013 shape): objective ``arc_challenge_acc``, rl floor 0.5."""
    default = _spec()
    return _spec(
        objective={"metric": "arc_challenge_acc", "direction": "max", "benchmarks": ["arc_challenge"]},
        eval_policy={
            "fingerprint": default["eval_policy"]["fingerprint"],
            "metrics": ["arc_challenge_acc", "rl_measured_fraction"],
        },
        base={"model": _base(tmp_path), "fingerprint": default["base"]["fingerprint"]},
        stage="rl",
        rl_min_measured_fraction=0.5,
        confirm={
            **default["confirm"],
            "guardrails": ["rl_measured_fraction"],
            "guardrail_directions": {"rl_measured_fraction": "max"},
        },
    )


def _rl_manifest(tmp_path: Path, status: str, checkpoint: str) -> str:
    Path(checkpoint).mkdir(parents=True, exist_ok=True)  # a bound rl row needs its saved checkpoint on disk
    payload = {
        "status": status,
        "steps_measured": 12,
        "checkpoint": f"saved: {checkpoint}",
        "measured_fraction": 0.75,
        "min_measured_fraction": 0.5,
    }
    path = tmp_path / "fskills_rl_manifest.json"
    path.write_text(json.dumps(payload))
    return str(path)


def _eval_report(tmp_path: Path, checkpoint: str, base: str) -> str:
    payload = {
        "verdict": "PASS",
        "limited": False,
        "policy": {"fingerprint": FINGERPRINT},
        "checkpoint": checkpoint,
        "base": base,
        # every benchmark score carries both spellings so the score is readable either way
        "benchmarks": [{"name": "arc_challenge", "score": 0.62, "stderr": 0.01, "value": 0.62, "se": 0.01}],
    }
    path = tmp_path / "eval_report.json"
    path.write_text(json.dumps(payload))
    return str(path)


def _identity(evidence: dict[str, Any], trial: str = "t1", **asserts: Any) -> dict[str, Any]:
    return {"trial": trial, "role": "candidate", "seed": 101, "evidence": evidence, **asserts}


def _ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl")


def _flow(result: dict[str, Any], spec: dict[str, Any], ledger: Ledger) -> tuple[list[Any], Any]:
    """Check then record one ``record`` request (a refused check stops before the store)."""
    request = {"action": "record", "spec": spec, "result": result}
    findings = _call("_check_record_request", request, spec, ledger)
    return findings, None if findings else _call("_record", request, ledger, _campaign(spec))


def _body(row: Any) -> dict[str, Any]:
    inner = row.get("body") if isinstance(row, dict) and isinstance(row.get("body"), dict) else row
    return dict(inner or {})


def _stored(ledger: Ledger, spec: dict[str, Any]) -> dict[str, Any]:
    rows = [_body(row) for row in ledger.results(_campaign(spec))]
    assert rows
    return rows[-1]


def _payload(outcome: Any) -> dict[str, Any]:
    """The ``SkillResult`` payload dict (core field name; only the carried fields are read)."""
    blobs = [getattr(outcome, name, None) for name in ("payload", "data", "body", "result")]
    blobs += list(vars(outcome).values()) if hasattr(outcome, "__dict__") else []
    for blob in blobs:
        if isinstance(blob, dict) and "recorded" in blob:
            return dict(blob)
    raise AssertionError(f"no record payload carried by {outcome!r}")


def test_record_bound_rl_pass_is_ok_with_rehashed_evidence(tmp_path: Path) -> None:
    spec, ledger = _rl_spec(tmp_path), _ledger(tmp_path)
    checkpoint = str((tmp_path / "checkpoints" / "t1" / "final").resolve())
    manifest = _rl_manifest(tmp_path, "PASS", checkpoint)
    report = _eval_report(tmp_path, checkpoint, base=_base(tmp_path))

    findings, outcome = _flow(_identity({"run_manifests": [manifest], "eval_report": report}), spec, ledger)

    assert findings == []
    stored = _stored(ledger, spec)
    assert stored["status"] == "ok"
    assert stored["evidence_class"] == "bound"
    assert stored["evidence"]["run_manifests"][0]["sha256"] == sha256_file(manifest)
    assert stored["evidence"]["eval_report"]["sha256"] == sha256_file(report)
    assert stored["metrics"]["arc_challenge_acc"]["value"] == pytest.approx(0.62)
    assert stored["metric_sources"]["rl_measured_fraction"] == "run_manifest"
    assert _payload(outcome)["status"] == "ok"
    assert result_problems(stored) == []


def test_record_bound_unmeasured_manifest_stays_unmeasured(tmp_path: Path) -> None:
    spec, ledger = _rl_spec(tmp_path), _ledger(tmp_path)
    checkpoint = str((tmp_path / "checkpoints" / "t1" / "final").resolve())
    manifest = _rl_manifest(tmp_path, "UNMEASURED: steps never measured", checkpoint)
    report = _eval_report(tmp_path, checkpoint, base=_base(tmp_path))

    findings, outcome = _flow(_identity({"run_manifests": [manifest], "eval_report": report}), spec, ledger)

    assert findings == []
    stored = _stored(ledger, spec)
    assert stored["status"] == "unmeasured"
    assert stored["metrics"]["arc_challenge_acc"]["value"] == pytest.approx(0.62)  # unmeasured keeps its metrics (D2)
    assert result_problems(stored) == []
    assert _payload(outcome)["status"] == "unmeasured"


def test_check_refuses_asserted_ok_over_unmeasured_evidence(tmp_path: Path) -> None:
    spec, ledger = _rl_spec(tmp_path), _ledger(tmp_path)
    checkpoint = str((tmp_path / "checkpoints" / "t1" / "final").resolve())
    manifest = _rl_manifest(tmp_path, "UNMEASURED: steps never measured", checkpoint)
    report = _eval_report(tmp_path, checkpoint, base=_base(tmp_path))
    result = _identity({"run_manifests": [manifest], "eval_report": report}, status="ok")

    findings, outcome = _flow(result, spec, ledger)

    assert [finding.rule_id for finding in findings] == ["AR-IN-012"]
    assert "assertion_contradicts_evidence" in findings[0].message
    assert outcome is None
    assert list(ledger.results(_campaign(spec))) == []  # refused before anything is appended


def test_check_refuses_missing_evidence_file(tmp_path: Path) -> None:
    spec, ledger = _rl_spec(tmp_path), _ledger(tmp_path)
    missing = str(tmp_path / "fskills_rl_manifest.json")  # never written
    report = _eval_report(tmp_path, _base(tmp_path), base=_base(tmp_path))

    findings, outcome = _flow(_identity({"run_manifests": [missing], "eval_report": report}), spec, ledger)

    assert [finding.rule_id for finding in findings] == ["AR-IN-011"]
    assert outcome is None
    assert list(ledger.results(_campaign(spec))) == []


def test_record_legacy_row_is_classified_asserted(tmp_path: Path) -> None:
    spec, ledger = _rl_spec(tmp_path), _ledger(tmp_path)
    result = {
        "trial": "t2", "role": "candidate", "seed": 102, "status": "ok", "limited": False, "steps": 0,
        "eval_policy_fingerprint": FINGERPRINT, "metrics": {"arc_challenge_acc": {"value": 0.51, "se": 0.01}},
    }

    findings, outcome = _flow(result, spec, ledger)

    assert findings == []
    stored = _stored(ledger, spec)
    assert stored["evidence_class"] == "asserted"
    assert "evidence" not in stored
    assert _payload(outcome)["evidence_class"] == "asserted"


def test_record_crash_row_is_classified_crash(tmp_path: Path) -> None:
    spec, ledger = _rl_spec(tmp_path), _ledger(tmp_path)
    result = {
        "trial": "t3", "role": "candidate", "seed": 103, "status": "crash", "limited": False, "steps": 0,
        "eval_policy_fingerprint": FINGERPRINT, "metrics": {},
    }

    findings, outcome = _flow(result, spec, ledger)

    assert findings == []
    stored = _stored(ledger, spec)
    assert stored["evidence_class"] == "crash"
    assert _payload(outcome)["status"] == "crash"


def test_record_baseline_eval_is_bound_with_measured_fraction_absent(tmp_path: Path) -> None:
    spec, ledger = _rl_spec(tmp_path), _ledger(tmp_path)
    report = _eval_report(tmp_path, _base(tmp_path), base=_base(tmp_path))  # eval of the untrained base

    findings, outcome = _flow(_identity({"run_manifests": [], "eval_report": report}, trial="base"), spec, ledger)

    assert findings == []
    stored = _stored(ledger, spec)
    assert stored["status"] == "ok"
    assert stored["evidence_class"] == "bound"
    assert stored["metric_sources"]["rl_measured_fraction"] == "absent"
    point = stored["metrics"]["rl_measured_fraction"]
    assert point.get("value") is None and point.get("se") is None
    assert result_problems(stored) == []
    assert _payload(outcome)["status"] == "ok"


def test_result_problems_admits_unmeasured_rows_and_absent_metrics_only() -> None:
    row = {
        "trial": "t1", "role": "candidate", "seed": 101, "status": "unmeasured", "limited": False, "steps": 0,
        "eval_policy_fingerprint": FINGERPRINT,
        "metrics": {"arc_challenge_acc": {"value": None, "se": None}},
        "metric_sources": {"arc_challenge_acc": "absent"},
    }

    assert result_problems(row) == []
    assert any("finite number" in problem for problem in result_problems({**row, "metric_sources": {}}))
    assert any("'crash' or 'unmeasured'" in problem for problem in result_problems({**row, "status": "failed"}))
