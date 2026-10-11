"""Tests for the FoundationSkills vetting gate (A6.1): ``run_gate`` and ``write_report``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from foundationskills.vetting.lock import file_sha256, read_tree, subset_digest


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------

_SKILL_MD = (
    "---\n"
    "name: test-skill\n"
    "description: A test skill for vetting.\n"
    "license: MIT\n"
    "---\n"
    "\n"
    "# Test Skill\n"
    "\n"
    "This is a test skill body for the vetting gate.\n"
).encode()

_NOTES_MD = b"Some reference notes.\n"


def _build_lock(skill_dir: Path, *, entry_overrides: dict | None = None) -> Path:
    """Write a ``skills.lock`` whose per-file shas match *skill_dir* and return its path."""
    files = read_tree(skill_dir)
    file_shas = {p: file_sha256(d) for p, d in sorted(files.items())}
    entry: dict = {
        "source": "github.com/mattpocock/skills",
        "publisher": "mattpocock",
        "commit": "a" * 40,
        "path": "skills/test-skill",
        "license": "MIT",
        "files": file_shas,
        "reviews": [],
    }
    if entry_overrides:
        entry.update(entry_overrides)
    lock: dict = {
        "schema": "fskills.skills_lock/1",
        "allowlist": {
            "sources": ["github.com/mattpocock/skills"],
            "publishers": ["mattpocock"],
            "spdx": ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "CC0-1.0"],
        },
        "skills": {"test-skill": entry},
    }
    lp = skill_dir.parent / "skills.lock"
    lp.write_text(json.dumps(lock, indent=2, sort_keys=True))
    return lp


def _make_clean_skill(tmp_path: Path) -> tuple[Path, Path, str]:
    """Create a clean skill tree + matching lock; return ``(skill_dir, lock_path, name)``."""
    skill_dir = tmp_path / "skill"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_bytes(_SKILL_MD)
    (skill_dir / "references" / "notes.md").write_bytes(_NOTES_MD)
    lock_path = _build_lock(skill_dir)
    return skill_dir, lock_path, "test-skill"


def _update_lock_shas(skill_dir: Path, lock_path: Path) -> None:
    """Recompute every file sha in the lock so it matches the tree on disk."""
    lock = json.loads(lock_path.read_text())
    files = read_tree(skill_dir)
    lock["skills"]["test-skill"]["files"] = {p: file_sha256(d) for p, d in sorted(files.items())}
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True))


def _edit_lock_entry(lock_path: Path, **update: object) -> None:
    lock = json.loads(lock_path.read_text())
    lock["skills"]["test-skill"].update(update)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True))


def _advisory_ok(description: str, body: str) -> dict:
    return {"consistent": True, "reason": "placement is fine"}


def _advisory_bad(description: str, body: str) -> dict:
    return {"consistent": False, "reason": "body diverges from description"}


def _find_row(report: dict, row_id: str) -> dict:
    return next(c for c in report["checks"] if c["id"] == row_id)


# ---------------------------------------------------------------------------
# clean skill
# ---------------------------------------------------------------------------


def test_clean_skill_with_advisory_pass(tmp_path: Path) -> None:
    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    from foundationskills.vetting.gate import run_gate

    report = run_gate(skill_dir, lock_path, name, advisory=_advisory_ok)
    assert report["status"] == "PASS", report
    assert report["exit_code"] == 0
    assert len(report["checks"]) == 15
    for check in report["checks"]:
        assert check["status"] == "PASS", f"{check['id']}: {check}"

    # no timestamp keys anywhere
    from foundationskills.vetting.gate import write_report

    s = json.dumps(report, indent=2, sort_keys=True)
    timestamp_keys = {"timestamp", "created_at", "created", "updated_at", "updated", "time", "date"}

    def _walk(obj: object) -> None:
        if isinstance(obj, dict):
            for k in obj:
                assert k not in timestamp_keys, f"found timestamp key {k!r}"
                _walk(obj[k])
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(report)

    # two runs dump byte-identical JSON
    report2 = run_gate(skill_dir, lock_path, name, advisory=_advisory_ok)
    j1 = json.dumps(report, indent=2, sort_keys=True) + "\n"
    j2 = json.dumps(report2, indent=2, sort_keys=True) + "\n"
    assert j1 == j2, "two runs must dump byte-identical JSON"


def test_clean_skill_without_advisory_unmeasured(tmp_path: Path) -> None:
    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    from foundationskills.vetting.gate import run_gate

    report = run_gate(skill_dir, lock_path, name)
    assert report["status"] == "UNMEASURED", report
    assert report["exit_code"] == 95
    not_pass = [c for c in report["checks"] if c["status"] != "PASS"]
    assert len(not_pass) == 1
    assert not_pass[0]["id"] == "VG-11"


# ---------------------------------------------------------------------------
# poisoned fixtures — one test per check
# ---------------------------------------------------------------------------


def test_poison_network(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    md = (skill_dir / "SKILL.md").read_bytes()
    (skill_dir / "SKILL.md").write_bytes(md + b"\nRun: curl https://example.com\n")
    _update_lock_shas(skill_dir, lock_path)
    report = run_gate(skill_dir, lock_path, name)
    assert report["status"] == "RED"
    row = _find_row(report, "VG-04")
    assert row["status"] == "RED"


def test_poison_credential(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    md = (skill_dir / "SKILL.md").read_bytes()
    (skill_dir / "SKILL.md").write_bytes(md + b"\nThe key lives at ~/.ssh/id_rsa.\n")
    _update_lock_shas(skill_dir, lock_path)
    report = run_gate(skill_dir, lock_path, name)
    assert report["status"] == "RED"
    row = _find_row(report, "VG-05")
    assert row["status"] == "RED"


def test_poison_dynamic_exec(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    md = (skill_dir / "SKILL.md").read_bytes()
    (skill_dir / "SKILL.md").write_bytes(md + b"\nUse eval(code) below.\n")
    _update_lock_shas(skill_dir, lock_path)
    report = run_gate(skill_dir, lock_path, name)
    assert report["status"] == "RED"
    row = _find_row(report, "VG-06")
    assert row["status"] == "RED"


def test_poison_html_comment(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    md = (skill_dir / "SKILL.md").read_bytes()
    (skill_dir / "SKILL.md").write_bytes(md + b"\n<!-- hidden payload -->\n")
    _update_lock_shas(skill_dir, lock_path)
    report = run_gate(skill_dir, lock_path, name)
    assert report["status"] == "RED"
    row = _find_row(report, "VG-07")
    assert row["status"] == "RED"


def test_poison_invisible_unicode(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    md = (skill_dir / "SKILL.md").read_bytes()
    (skill_dir / "SKILL.md").write_bytes(md + ("\U000E0041".encode("utf-8")))
    _update_lock_shas(skill_dir, lock_path)
    report = run_gate(skill_dir, lock_path, name)
    assert report["status"] == "RED"
    row = _find_row(report, "VG-08")
    assert row["status"] == "RED"


def test_poison_allowed_tools(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    poisoned = _SKILL_MD.replace(b"license: MIT\n", b"license: MIT\nallowed-tools: Bash\n")
    (skill_dir / "SKILL.md").write_bytes(poisoned)
    _update_lock_shas(skill_dir, lock_path)
    report = run_gate(skill_dir, lock_path, name)
    assert report["status"] == "RED"
    row = _find_row(report, "VG-09")
    assert row["status"] == "RED"


def test_poison_scripts_no_review(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.sh").write_bytes(b"#!/bin/bash\necho hello\n")
    _update_lock_shas(skill_dir, lock_path)
    report = run_gate(skill_dir, lock_path, name)
    assert report["status"] == "REFUSED"
    row = _find_row(report, "VG-10")
    assert row["status"] == "REFUSED"


def test_poison_github_no_review(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    (skill_dir / ".github" / "workflows").mkdir(parents=True)
    (skill_dir / ".github" / "workflows" / "x.yml").write_bytes(b"name: CI\non: push\n")
    _update_lock_shas(skill_dir, lock_path)
    report = run_gate(skill_dir, lock_path, name)
    assert report["status"] == "REFUSED"
    row = _find_row(report, "VG-14")
    assert row["status"] == "REFUSED"


def test_poison_tampered_file(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    # tamper WITHOUT recomputing lock shas
    md = (skill_dir / "SKILL.md").read_bytes()
    (skill_dir / "SKILL.md").write_bytes(md.replace(b"test", b"TEST"))
    report = run_gate(skill_dir, lock_path, name)
    assert report["status"] == "REFUSED"
    row = _find_row(report, "VG-02")
    assert row["status"] == "REFUSED"


def test_poison_unlisted_source(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    _edit_lock_entry(lock_path, source="github.com/evil/skills")
    report = run_gate(skill_dir, lock_path, name)
    assert report["status"] == "REFUSED"
    row = _find_row(report, "VG-01")
    assert row["status"] == "REFUSED"


def test_poison_epl_license(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    poisoned = _SKILL_MD.replace(b"license: MIT\n", b"license: EPL-2.0\n")
    (skill_dir / "SKILL.md").write_bytes(poisoned)
    _edit_lock_entry(lock_path, license="EPL-2.0")
    _update_lock_shas(skill_dir, lock_path)
    report = run_gate(skill_dir, lock_path, name)
    assert report["status"] == "REFUSED"
    row = _find_row(report, "VG-03")
    assert row["status"] == "REFUSED"


# ---------------------------------------------------------------------------
# scanner crash
# ---------------------------------------------------------------------------


def test_scanner_crash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    skill_dir, lock_path, name = _make_clean_skill(tmp_path)

    def _raise(*_a: object, **_kw: object) -> object:
        raise RuntimeError("simulated crash")

    monkeypatch.setattr("foundationskills.vetting.scan.check_network", _raise)

    from foundationskills.vetting.gate import run_gate

    report = run_gate(skill_dir, lock_path, name, advisory=_advisory_ok)
    assert report["status"] == "REFUSED"
    assert report["exit_code"] == 96
    vg04 = _find_row(report, "VG-04")
    assert vg04["status"] == "REFUSED"
    assert "scanner failure" in vg04["note"]
    assert "RuntimeError" in vg04["note"]
    vg15 = _find_row(report, "VG-15")
    assert vg15["status"] == "REFUSED"
    assert "scanner failure" in vg15["note"]


# ---------------------------------------------------------------------------
# advisory divergence review
# ---------------------------------------------------------------------------


def test_advisory_divergence(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    files = read_tree(skill_dir)
    div_digest = subset_digest(files, ["SKILL.md"])

    # case: no divergence review → UNMEASURED
    report = run_gate(skill_dir, lock_path, name, advisory=_advisory_bad)
    row = _find_row(report, "VG-11")
    assert row["status"] == "UNMEASURED"
    assert "held for human review" in row["note"]
    assert report["status"] == "UNMEASURED"

    # case: divergence review reject → RED
    _edit_lock_entry(
        lock_path,
        reviews=[
            {
                "kind": "divergence",
                "files_sha256": div_digest,
                "reviewer": "human-reviewer",
                "date": "2024-01-01",
                "decision": "reject",
            }
        ],
    )
    report = run_gate(skill_dir, lock_path, name, advisory=_advisory_bad)
    row = _find_row(report, "VG-11")
    assert row["status"] == "RED"
    assert report["status"] == "RED"

    # case: divergence review approve → PASS
    _edit_lock_entry(
        lock_path,
        reviews=[
            {
                "kind": "divergence",
                "files_sha256": div_digest,
                "reviewer": "human-reviewer",
                "date": "2024-01-01",
                "decision": "approve",
            }
        ],
    )
    report = run_gate(skill_dir, lock_path, name, advisory=_advisory_bad)
    row = _find_row(report, "VG-11")
    assert row["status"] == "PASS"
    assert report["status"] == "PASS"


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def test_install_success(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    project_root = tmp_path
    install_to = project_root / ".agents" / "skills" / name
    report = run_gate(
        skill_dir,
        lock_path,
        name,
        advisory=_advisory_ok,
        install_to=install_to,
        project_root=project_root,
    )
    assert report["status"] == "PASS"
    assert report["exit_code"] == 0
    assert report["installed"] is not None
    orig = read_tree(skill_dir)
    installed = read_tree(install_to)
    assert orig == installed


def test_install_home_scope_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda _cls: tmp_path))

    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    install_to = tmp_path / ".claude" / "skills" / name
    report = run_gate(
        skill_dir,
        lock_path,
        name,
        advisory=_advisory_ok,
        install_to=install_to,
        project_root=tmp_path,
    )
    assert report["status"] == "REFUSED"
    assert report["exit_code"] == 96
    vg12 = _find_row(report, "VG-12")
    assert vg12["status"] == "REFUSED"
    assert report["installed"] is None
    assert not install_to.exists()


def test_install_red_gate_not_copied(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    md = (skill_dir / "SKILL.md").read_bytes()
    (skill_dir / "SKILL.md").write_bytes(md + b"\ncurl https://example.com\n")
    _update_lock_shas(skill_dir, lock_path)
    install_to = tmp_path / ".agents" / "skills" / name
    report = run_gate(
        skill_dir,
        lock_path,
        name,
        install_to=install_to,
        project_root=tmp_path,
    )
    assert report["status"] == "RED"
    assert report["installed"] is None
    assert not install_to.exists()


def test_install_existing_different_content(tmp_path: Path) -> None:
    from foundationskills.vetting.gate import run_gate

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    install_to = tmp_path / ".agents" / "skills" / name
    install_to.mkdir(parents=True)
    (install_to / "dummy.md").write_text("different content entirely")
    report = run_gate(
        skill_dir,
        lock_path,
        name,
        advisory=_advisory_ok,
        install_to=install_to,
        project_root=tmp_path,
    )
    assert report["status"] == "REFUSED"
    assert report["exit_code"] == 96
    vg12 = _find_row(report, "VG-12")
    assert vg12["status"] == "REFUSED"
    assert "different content" in vg12["note"]
    assert report["installed"] is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_cmd_vet(tmp_path: Path) -> None:
    from foundationskills.vetting.cli import cmd_vet

    skill_dir, lock_path, name = _make_clean_skill(tmp_path)
    out_path = tmp_path / "report.json"
    ns = argparse.Namespace(
        skill_dir=skill_dir,
        lock=lock_path,
        name=name,
        out=out_path,
        install_to=None,
        project_root=None,
        advisory_base_url=None,
        advisory_model=None,
        advisory_api_key_env=None,
        timeout=120.0,
    )
    exit_code = cmd_vet(ns)
    assert exit_code == 95
    assert out_path.exists()
    report = json.loads(out_path.read_text())
    assert report["status"] == "UNMEASURED"
    assert report["exit_code"] == 95


def test_a_reject_review_wins_over_an_approval_of_the_same_bytes(tmp_path: Path) -> None:
    """Fail closed: an approve and a reject over the same digest decide reject (network, github and divergence)."""
    from foundationskills.vetting.lock import review_decision

    base = {"files_sha256": "d" * 64, "reviewer": "r", "decision": "approve"}
    entry = {"reviews": [dict(base, kind="network", date="2026-01-02"),
                         dict(base, kind="network", date="2026-01-01", decision="reject")]}
    assert review_decision(entry, "network", "d" * 64)["decision"] == "reject"
    assert review_decision(entry, "network", "e" * 64) is None


def test_requests_prose_does_not_fire_but_library_use_does() -> None:
    """Real-sweep false positive (mattpocock triage): 'feature requests.' is prose, not the requests library."""
    from foundationskills.vetting import scan

    prose = {"OUT-OF-SCOPE.md": b"Stores records of rejected feature requests.\nOne file per concept.\n"}
    assert scan.check_network({}, prose).status == "PASS"
    for code in (b"import requests\nrequests.get(url)\n", b"x = 1\nfrom requests import Session\n"):
        assert scan.check_network({}, {"tool.py": code}).status == "RED"


def test_cli_advisory_calls_backend_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real-sweep bug: the CLI advisory called a non-existent backend.chat(); it must use complete()."""
    from types import SimpleNamespace

    from foundationskills.vetting import cli as vet_cli

    seen = {}

    class Backend:
        def complete(self, messages, **kwargs):
            seen["messages"] = messages
            return SimpleNamespace(content='```json\n{"consistent": true, "reason": "ok"}\n```', error=None)

    monkeypatch.setattr("foundationskills.skills.data_engine.llm_backend.make_backend", lambda cfg, op: Backend())
    advisory = vet_cli._build_advisory("http://x", "m", None, 10.0)
    assert advisory("desc", "body") == {"consistent": True, "reason": "ok"}
    assert seen["messages"][1]["content"].startswith("Description:")
