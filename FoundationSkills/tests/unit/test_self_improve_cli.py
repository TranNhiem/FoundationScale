"""C9.7: `fskills self-improve hash|admit` end to end on a copy of a real skill package (no model calls)."""

import json
import shutil
from pathlib import Path

import pytest

import foundationskills.skills.compression.skill as comp
from foundationskills import cli


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    src = Path(comp.__file__).with_name("SKILL.md")
    fake = tmp_path / "skill.py"
    fake.write_text("")
    shutil.copy(src, tmp_path / "SKILL.md")
    monkeypatch.setattr("inspect.getfile", lambda obj: str(fake))  # point the CLI at the copied package
    return tmp_path


def _write(tmp_path, old, measured=True, gain=0.1):
    new = old.replace("# ", "# ", 1) + "\nA clarified sentence of prose.\n"
    cand = tmp_path / "cand.md"
    cand.write_text(new)
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps({"measured": measured, "target": "behaviour_case_pass",
                              "incumbent": {"behaviour_case_pass": 0.8, "oracle_item_pass": 0.9, "routing_recall": 0.95},
                              "candidate": {"behaviour_case_pass": 0.8 + gain, "oracle_item_pass": 0.9,
                                            "routing_recall": 0.95},
                              # admit requires a fired activation beacon and significant paired evidence (atlas v2)
                              "beacon": {"condition": "clarified", "fired": 3},
                              "paired": {"target": "behaviour_case_pass", "pairs": [[0.0, 1.0]] * 8}}))
    return cand, ev, new


def test_admit_writes_only_with_the_typed_back_hash(pkg, capsys):
    old = (pkg / "SKILL.md").read_text()
    cand, ev, new = _write(pkg, old)
    base = ["self-improve", "admit", "--skill", "compression", "--candidate", str(cand), "--evidence", str(ev)]
    assert cli.main(base) == 96 and (pkg / "SKILL.md").read_text() == old  # no hash: refused, nothing written
    capsys.readouterr()
    assert cli.main(["self-improve", "hash", "--skill", "compression", "--candidate", str(cand),
                     "--evidence", str(ev)]) == 0
    token = capsys.readouterr().out.strip()
    assert cli.main(base + ["--confirm", token]) == 0 and (pkg / "SKILL.md").read_text() == new


def test_admit_red_without_gain_writes_nothing(pkg, capsys):
    old = (pkg / "SKILL.md").read_text()
    cand, ev, _ = _write(pkg, old, gain=0.0)
    cli.main(["self-improve", "hash", "--skill", "compression", "--candidate", str(cand), "--evidence", str(ev)])
    token = capsys.readouterr().out.strip()
    rc = cli.main(["self-improve", "admit", "--skill", "compression", "--candidate", str(cand), "--evidence", str(ev),
                   "--confirm", token])
    assert rc == 5 and (pkg / "SKILL.md").read_text() == old
