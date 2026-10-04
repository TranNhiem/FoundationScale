"""Ledger tests: chain lines that are valid JSON yet not objects (``null``, ``5``, ``[]``)."""
from __future__ import annotations

import pytest

from foundationskills.skills.auto_research.ledger import Ledger

NON_OBJECTS = ["null", "5", "[]", '"text"']


def _payload(value: float, trial: str = "t1") -> dict:
    return {"trial": trial, "seed": 1, "metrics": {"val_accuracy": {"value": value, "se": 0.001}}}


def _ledger_with_junk(tmp_path, lines: list[str]) -> Ledger:
    """A one-entry ledger whose chain carries extra non-object JSON lines at the end."""
    ledger = Ledger(tmp_path / "ledger")
    ledger.append("trial_result", "c1", "t1", _payload(0.5))
    with open(ledger.chain, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")
    return ledger


class TestVerifyNonObjectLines:
    @pytest.mark.parametrize("line", NON_OBJECTS)
    def test_reports_named_problem_for_the_line(self, tmp_path, line):
        ledger = _ledger_with_junk(tmp_path, [line])
        problems = ledger.verify()
        assert problems and "seq 1" in problems[0] and "line 2" in problems[0]

    @pytest.mark.parametrize("line", NON_OBJECTS)
    def test_continues_after_a_non_object_line(self, tmp_path, line):
        ledger = _ledger_with_junk(tmp_path, [line, "not json"])
        problems = ledger.verify()
        assert any("seq 1" in p and "line 2" in p for p in problems)
        assert any("line 3" in p for p in problems)


class TestReadableViewsSkipNonObjectLines:
    @pytest.mark.parametrize("line", NON_OBJECTS)
    def test_entries_skips_non_object_lines(self, tmp_path, line):
        ledger = _ledger_with_junk(tmp_path, [line])
        assert [e["seq"] for e in ledger.entries()] == [0]

    def test_head_results_and_launches_do_not_crash(self, tmp_path):
        ledger = _ledger_with_junk(tmp_path, ["null", "5", "[]"])
        entry = ledger.entries()[0]
        assert ledger.head() == {"count": 1, "head_hash": entry["hash"]}
        assert [p["trial"] for p in ledger.results("c1")] == ["t1"]
        assert ledger.launches("c1") == []
