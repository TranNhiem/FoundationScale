"""Unit tests for Tier-A deterministic scoring (agent/deterministic_scoring.py)."""

from __future__ import annotations

import importlib

import pytest

import foundationskills.agent.deterministic_scoring as ds
from foundationskills.core.contract import RuleSpec, Severity

CASE = {"case_id": "unit", "phase": "input"}


def _expected_items(rule: RuleSpec) -> list[str]:
    for name in ("foundationskills.oracle_cases", "foundationskills.oracle_cases.expectations",
                 "foundationskills.agent.oracle_cases", "foundationskills.core.oracle_cases",
                 "oracle_cases"):
        try:
            fn = getattr(importlib.import_module(name), "expected_items", None)
        except ImportError:
            continue
        if fn is not None:
            return fn(rule)
    raise AssertionError("oracle_cases.expected_items is not importable")


def _rule(sev: Severity, phase: str, rid: str = "CP-IN-004", desc: str = "Mishandles input") -> RuleSpec:
    return RuleSpec(rid, desc, sev, phase)


def test_outcome_constants_match_expected_items():
    exp_in = _expected_items(_rule(Severity.BLOCK, "input"))
    exp_ho = _expected_items(_rule(Severity.BLOCK, "handoff", rid="CP-HO-1"))
    consts = {v for k, v in vars(ds).items() if isinstance(v, str) and not k.startswith("__")}
    assert ds.SCHEMA == "fskills.det_scoring/1"
    assert exp_in[1] in consts and exp_ho[1] in consts and exp_in[1] != exp_ho[1]
    assert [ds.det_check(i, "REFUSED", CASE) for i in exp_in] == [None, True, False]
    assert [ds.det_check(i, "RED", CASE) for i in exp_ho] == [None, True, False]


def test_cites_rule_exact_token_matching():
    item = "Cites rule CP-IN-004"
    assert ds.det_check(item, "We cite rule CP-IN-004 here.", CASE) is True
    assert ds.det_check(item, "(CP-IN-004).", CASE) is True
    assert ds.det_check(item, "CP-IN-0045", CASE) is False          # longer token: no match
    assert ds.det_check(item, "cp-in-004", CASE) is False            # case-sensitive
    assert ds.det_check(item, "XCP-IN-004", CASE) is False
    assert ds.det_check(item, "no rule mentioned", CASE) is False
    other = _expected_items(_rule(Severity.BLOCK, "input", rid="CP-OUT-7"))[2]
    assert ds.det_check(other, "see CP-OUT-7", CASE) is True
    assert ds.det_check(other, "see CP-OUT-7b", CASE) is False


def test_refused_and_exit_96_detection():
    item = _expected_items(_rule(Severity.BLOCK, "input"))[1]
    for ans in ("REFUSED", "refused", "We Refused to run", "exits with 96", "exit code 96", "96"):
        assert ds.det_check(item, ans, CASE) is True, ans
    for ans in ("", "ran it anyway", "1960", "9600", "096"):
        assert ds.det_check(item, ans, CASE) is False, ans


def test_red_is_case_sensitive_plus_exit_5():
    item = _expected_items(_rule(Severity.BLOCK, "handoff", rid="CP-HO-1"))[1]
    assert ds.det_check(item, "the result is RED", CASE) is True
    assert ds.det_check(item, "RED.", CASE) is True
    assert ds.det_check(item, "exit 5", CASE) is True
    assert ds.det_check(item, "exit code 5", CASE) is True
    assert ds.det_check(item, "two red birds flew by", CASE) is False   # lowercase prose
    assert ds.det_check(item, "red", CASE) is False
    assert ds.det_check(item, "REDDIT link", CASE) is False
    assert ds.det_check(item, "", CASE) is False


def test_undecidable_items_return_none():
    warn = _expected_items(_rule(Severity.WARN, "input"))
    blk = _expected_items(_rule(Severity.BLOCK, "input"))
    assert warn[0].startswith("Identifies the problem:") and blk[0].startswith("Identifies the problem:")
    for item in (blk[0], warn[0], warn[1],
                 "Records it as information only; it neither blocks nor fails the run",
                 "free-text behaviour item", ""):
        for ans in ("REFUSED", "RED", "CP-IN-004", "everything failed", None):
            assert ds.det_check(item, ans, CASE) is None, (item, ans)


def test_empty_answer_is_false_and_never_raises():
    blk = _expected_items(_rule(Severity.BLOCK, "input"))
    for ans in (None, "", "   "):
        assert ds.det_check(blk[1], ans, CASE) is False
        assert ds.det_check(blk[2], ans, CASE) is False
    for item in (None, "", "Cites rule", "Cites rule ", blk[2], "junk item", 42):
        assert ds.det_check(item, None, CASE) in (True, False, None)


def test_det_scores_walks_every_item():
    blk = _expected_items(_rule(Severity.BLOCK, "input"))
    assert ds.det_scores(blk, "REFUSED", CASE) == [None, True, False]
    assert ds.det_scores(blk, "REFUSED rule CP-IN-004", CASE) == [None, True, True]
    assert ds.det_scores([], "REFUSED", CASE) == []


@pytest.mark.parametrize(
    "det, judged, expected",
    [(None, None, None), (None, True, True), (None, False, False), (True, None, True),
     (False, None, False), (True, True, True), (False, False, False),
     (True, False, None), (False, True, None)],
)
def test_reconcile_truth_table(det, judged, expected):
    assert ds.reconcile(det, judged) is expected


def test_case_decidable_needs_one_checkable_item():
    assert ds.case_decidable([]) is False
    assert ds.case_decidable([None, None, None]) is False
    warn = _expected_items(_rule(Severity.WARN, "input"))
    assert ds.case_decidable(ds.det_scores([warn[0], warn[1]], "whatever", CASE)) is False
    assert ds.case_decidable([None, False]) is True
    assert ds.case_decidable([False]) is True
    blk = _expected_items(_rule(Severity.BLOCK, "input"))
    assert ds.case_decidable(ds.det_scores(blk, "REFUSED", CASE)) is True
