"""A4.6 experiment index: auto_research action "index" over journals, FS run manifests and campaign ledgers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from foundationskills.core.status import Status
from foundationskills.skills.auto_research.experiment_index import IndexRefused, build_index
from foundationskills.skills.auto_research.ledger import Ledger
from foundationskills.skills.auto_research.skill import AutoResearchSkill

try:  # the same relative-import approach the other auto_research tests use
    from .test_auto_research_skill import _ctx, _materialize
except ImportError:  # pragma: no cover - run as a top-level module
    from tests.unit.auto_research.test_auto_research_skill import _ctx, _materialize  # type: ignore[no-redef]

CONTRACT = '{"PASS": 0, "RED": 5, "UNMEASURED": 95, "REFUSE": 96}'


def _cfg(**values: Any) -> dict[str, Any]:
    return {k: {"key": k, "value": v, "source": "cli", "env_value": None, "findings": []} for k, v in values.items()}


def _manifest(run_id: str, exit_code: str | None, telemetry: dict[str, Any], frames: str | None = None,
              **over: Any) -> dict[str, Any]:
    config = _cfg(model="/m", dataset="/d.jsonl", exit_contract=CONTRACT)
    if exit_code is not None:
        config.update(_cfg(**{"extra.exit": exit_code}))
    return {"schema_version": 1, "run_id": run_id, "attempt": 1, "job_id": None, "created_at": "2026-10-19T00:00:00+00:00",
            "fingerprint": "f" * 64, "code": {"status": "not_captured", "commit": "abc", "probe_failed": False},
            "config": config, "environment": {"values": {"FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES": frames} if frames else {}},
            "telemetry": telemetry, "artifact_paths": {"output_dir": f"/runs/{run_id}"}, **over}


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "estate"
    steps = [{"step_index": 0, "skill": "data_engine", "status": "PASS", "exit_code": 0, "artifact_ids": ["ds"],
              "refusal": None, "timestamp": "t0"},
             {"step_index": 1, "skill": "training.planner", "status": "REFUSED", "exit_code": 96, "artifact_ids": [],
              "refusal": "TR-IN-006: rows carry video", "timestamp": "t1"}]
    _write(root / "work" / "journal.jsonl", "\n".join(json.dumps(s) for s in steps) + "\n{broken\n")
    measured = {"key": "train_loss", "value": 1.3, "source": "measured", "unit": None}
    _write(root / "runs" / "a" / "run_manifest.json",
           _manifest("a", "0", {"train_loss": measured,
                                "perf_mfu": {"key": "perf_mfu", "value": None, "source": "unmeasured", "unit": None}},
                     frames="16"))
    _write(root / "runs" / "b" / "run_manifest.json", _manifest("b", "96", {}))
    _write(root / "runs" / "c" / "run_manifest.json", _manifest("c", None, {}))
    _write(root / "runs" / "d" / "run_manifest.json", _manifest("d", "0", {}, schema_version=2))
    ledger = Ledger(root / "ledger")
    ledger.append("campaign_approved", "c1", "-", {"spec_hash": "h"})
    ledger.append("campaign_approved", "c2", "-", {"spec_hash": "h"})
    ledger.append("campaign_closed", "c2", "-", {"reason": "budget"})
    return root


def _rows(index: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [r for r in index["rows"] if r["kind"] == kind]


def test_index_rows_are_copied_from_what_each_record_measured(tmp_path: Path) -> None:
    index = build_index([_tree(tmp_path)])

    steps = _rows(index, "skill_step")
    assert [(s["skill"], s["status"], s["exit_code"]) for s in steps] == [
        ("data_engine", "PASS", 0), ("training.planner", "REFUSED", 96)]
    assert steps[1]["refusal"] == "TR-IN-006: rows carry video"
    runs = {r["run_id"]: r for r in _rows(index, "fs_run")}
    assert set(runs) == {"a", "b", "c"}  # d (schema_version 2) is unreadable, never silently indexed
    assert (runs["a"]["outcome"], runs["a"]["exit_code"], runs["a"]["video_frames"]) == ("PASS", 0, "16")
    assert runs["a"]["telemetry"] == {"train_loss": 1.3} and runs["a"]["unmeasured_telemetry"] == ["perf_mfu"]
    assert (runs["b"]["outcome"], runs["b"]["exit_code"]) == ("REFUSE", 96)
    assert runs["c"]["outcome"] is None and "no exit code" in runs["c"]["outcome_reason"]
    campaigns = {r["campaign"]: r for r in _rows(index, "campaign")}
    assert campaigns["c2"]["closed"] is True and campaigns["c1"]["closed"] is False
    reasons = sorted(u["reason"] for u in index["unreadable"])
    assert len(reasons) == 2 and any("not JSON" in r for r in reasons) and any("schema_version 2" in r for r in reasons)
    assert index["counts"] == {"skill_step": 2, "fs_run": 3, "campaign": 2, "unreadable": 2}


def test_index_is_deterministic_and_dedups_overlapping_roots(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    once = build_index([root])
    assert build_index([root]) == once
    assert build_index([root, root / "runs"])["counts"] == {**once["counts"]}


def test_a_tampered_ledger_is_unreadable_and_yields_no_rows(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    chain = root / "ledger" / "chain.jsonl"
    lines = chain.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["campaign"] = "forged"
    chain.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n", encoding="utf-8")

    index = build_index([root])

    assert _rows(index, "campaign") == []
    assert any("does not verify" in u["reason"] for u in index["unreadable"])


def test_a_symlink_loop_under_a_root_does_not_hang(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    os.symlink(root, root / "runs" / "loop")
    assert build_index([root])["counts"]["fs_run"] == 3


@pytest.mark.parametrize("roots", [[], ["missing"], ["empty"]])
def test_unusable_roots_are_refused(tmp_path: Path, roots: list[str]) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(IndexRefused):
        build_index([tmp_path / r for r in roots])


def test_the_skill_writes_an_experiment_index_and_names_unreadable_sources(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    result = AutoResearchSkill().execute({"action": "index", "scan_roots": [str(root)]}, _ctx(tmp_path / "w"))

    assert result.status is Status.UNMEASURED  # two unreadable sources: never PASS
    assert [f.rule_id for f in result.findings] == ["AR-HO-011"]
    assert result.payload["verdict"] == "UNMEASURED"
    assert result.artifacts[0].type == "experiment_index"
    assert Path(result.artifacts[0].path).is_file()


def test_the_skill_passes_a_clean_index(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write(root / "a" / "run_manifest.json", _manifest("a", "0", {}))
    result = AutoResearchSkill().execute({"action": "index", "scan_roots": [str(root)]}, _ctx(tmp_path / "w"))

    assert result.status is Status.PASS and result.payload["verdict"] == "PASS"
    assert result.payload["counts"]["fs_run"] == 1 and result.artifacts


@pytest.mark.parametrize("rule_id, status", [("AR-IN-014", Status.REFUSED), ("AR-HO-011", Status.UNMEASURED)])
def test_the_index_fixtures_fire(tmp_path: Path, rule_id: str, status: Status) -> None:
    fixture = AutoResearchSkill().must_fire_fixtures()[rule_id]
    assert set(fixture) <= {"files", "request"}
    result = AutoResearchSkill().execute(_materialize(fixture, tmp_path), _ctx(tmp_path))

    assert result.status is status
    fired = {f.rule_id for f in result.findings}
    assert rule_id in fired or rule_id in (result.refusal or "")
