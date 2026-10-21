"""A6.2 re-vet: every vendored third-party skill PASSes the gate offline at its pin, and nothing unpinned is vendored."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundationskills.vetting.intake import revet

THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"
LOCK = THIRD_PARTY / "skills.lock"
VENDORED = sorted(p for p in (THIRD_PARTY / "skills").iterdir() if p.is_dir()) if (THIRD_PARTY / "skills").is_dir() else []


def _pinned() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))["skills"]


def test_vendored_and_pinned_sets_match():
    assert VENDORED, "third_party/skills is empty: the intake vendored nothing"
    assert sorted(p.name for p in VENDORED) == sorted(_pinned())


@pytest.mark.parametrize("skill_dir", VENDORED, ids=[p.name for p in VENDORED])
def test_vendored_skill_revets_pass(skill_dir: Path):
    report = revet(skill_dir, LOCK, skill_dir.name)
    failing = [f"{c['id']} {c['status']}" for c in report["checks"] if c["status"] != "PASS"]
    assert (report.get("verdict") or report.get("status")) == "PASS", failing


def test_every_pin_has_licence_evidence_and_a_notice():
    notices = (THIRD_PARTY / "THIRD_PARTY_NOTICES").read_text(encoding="utf-8")
    for name, entry in _pinned().items():
        assert entry["license"] and entry.get("license_evidence"), name
        assert name in notices, name
