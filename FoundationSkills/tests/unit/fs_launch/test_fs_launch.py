"""/fs-launch (A3.2): the launch gate. The launcher is patched; nothing is ever submitted or run from a test."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from foundationskills.core.contract import SkillContext
from foundationskills.core.orchestrator import plan_hash
from foundationskills.core.status import Status
from foundationskills.interfaces.fs.launch import LaunchRefused
from foundationskills.skills.fs_launch import skill as fl
from foundationskills.skills.fs_launch.skill import FsLaunchSkill

SPEC_PAYLOAD: dict[str, Any] = {"entry": "fskills-eval", "executable": True, "run_name": "q27"}
SPEC_REL = "launch/fs_launch_spec.q27.json"
HASH = plan_hash(SPEC_PAYLOAD)


@pytest.fixture()
def ctx(tmp_path: Path) -> SkillContext:
    target = tmp_path / SPEC_REL
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(SPEC_PAYLOAD, sort_keys=True), encoding="utf-8")
    return SkillContext(workdir=tmp_path)


def _never_launch(*_a: Any, **_k: Any) -> dict:
    raise AssertionError("a refused input must never reach the launcher")


def test_a_missing_hash_names_the_command_and_never_echoes_the_hash(ctx, monkeypatch) -> None:
    monkeypatch.setattr(fl, "_LAUNCH", _never_launch)
    result = FsLaunchSkill().execute({"launch_spec": SPEC_REL}, ctx)
    assert result.status is Status.REFUSED and [f.rule_id for f in result.findings] == ["FL-IN-001"]
    assert "fskills hash --spec" in result.findings[0].message
    assert HASH not in json.dumps([f.__dict__ for f in result.findings], default=str) + str(result.refusal)


def test_a_wrong_hash_is_refused_with_the_inner_findings_as_evidence(ctx, monkeypatch) -> None:
    monkeypatch.setattr(fl, "_LAUNCH", _never_launch)
    result = FsLaunchSkill().execute({"launch_spec": SPEC_REL, "user_plan_hash": "0" * 16}, ctx)
    assert result.status is Status.REFUSED and [f.rule_id for f in result.findings] == ["FL-IN-002"]
    inner = [row["rule_id"] for row in result.findings[0].evidence["confirm_before_launch"]]
    assert "CB-IN-002" in inner  # composed skill's rule travels as evidence, never as an undeclared finding


def test_a_launcher_refusal_is_fl_in_003(ctx, monkeypatch) -> None:
    def refuse(*_a: Any, **_k: Any) -> dict:
        raise LaunchRefused("launch refused: executable is not true")

    monkeypatch.setattr(fl, "_LAUNCH", refuse)
    result = FsLaunchSkill().execute({"launch_spec": SPEC_REL, "user_plan_hash": HASH, "submit": False}, ctx)
    assert result.status is Status.REFUSED and "FL-IN-003" in [f.rule_id for f in result.findings]
    assert "executable is not true" in json.dumps([f.message for f in result.findings])


def test_a_failed_local_run_is_red_and_still_recorded(ctx, monkeypatch) -> None:
    monkeypatch.setattr(fl, "_LAUNCH", lambda *_a, **_k: {"job_id": None, "pid": 1, "command": "x", "dry_run_rc": 0, "returncode": 5})
    result = FsLaunchSkill().execute({"launch_spec": SPEC_REL, "user_plan_hash": HASH, "submit": False}, ctx)
    assert result.status is Status.RED and "FL-HO-001" in [f.rule_id for f in result.findings]
    assert any(ref.type == "launch_record" for ref in result.artifacts)


def test_a_confirmed_submission_passes_with_a_launch_record(ctx, monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def submit(spec: dict, *, confirm: str, submit: bool) -> dict:
        seen.update(confirm=confirm, submit=submit)
        return {"job_id": "4242", "command": "sbatch x", "dry_run_rc": 0, "returncode": None}

    monkeypatch.setattr(fl, "_LAUNCH", submit)
    result = FsLaunchSkill().execute({"launch_spec": SPEC_REL, "user_plan_hash": HASH}, ctx)
    assert result.status is Status.PASS, result.refusal
    assert seen == {"confirm": HASH, "submit": True}  # the user's hash is the one the launcher gets
    record = [ref for ref in result.artifacts if ref.type == "launch_record"]
    assert record and result.payload["plan_hash"] == HASH and result.payload["submitted"] is True


def test_it_composes_only_confirm_before_launch() -> None:
    assert FsLaunchSkill.invocation == "user" and FsLaunchSkill.invokes == ("confirm_before_launch",)


# Launch authority: fs_launch for planned training runs; auto_research launches campaign trials only under its
# own user-approved launch envelope. Any other skill importing the launcher would be a second, ungated path.
LAUNCHERS = {"foundationskills.skills.fs_launch.skill", "foundationskills.skills.auto_research.skill"}


def test_no_other_skill_module_imports_the_launcher() -> None:
    root = Path(fl.__file__).resolve().parents[1]  # foundationskills/skills
    offenders = []
    for path in root.rglob("*.py"):
        module = "foundationskills." + ".".join(path.relative_to(root.parent).with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "foundationskills.interfaces.fs.launch":
                if module not in LAUNCHERS and any(alias.name == "launch" for alias in node.names):
                    offenders.append(module)
    assert offenders == [], f"skill modules that can launch outside /fs-launch: {offenders}"
