"""Unit tests for MODULE 1: foundationskills/core/change_contract.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from foundationskills.core.change_contract import (  # noqa: E402
    FIELDS,
    REQUIRED_INVARIANTS,
    check,
    needs_contract,
    parse_contract,
)

CORE = "FoundationSkills/foundationskills/core/verifier.py"
SKILL = "FoundationSkills/foundationskills/skills/verifier.py"
FS_MOD = "FoundationSkills/foundationskills/interfaces/fs/gate.py"
VET = "FoundationSkills/foundationskills/vetting/audit.py"
ROLLBACK = "deadbeef"


def make_msg(*, drop=(), empty=(), preserves=None, rollback=ROLLBACK, header="Change-Contract:"):
    body = {
        "Component": "verifier",
        "Failure-Mode": "false PASS on must-fire fixtures",
        "Predicted-Gain": "+2pp gate recall",
        "Preserves": "; ".join(REQUIRED_INVARIANTS) if preserves is None else preserves,
        "Falsifier": "beacon misses 3 must-fire fixtures",
        "Rollback": rollback,
    }
    for key in drop:
        body.pop(key)
    for key in empty:
        body[key] = ""
    lines = ([header] if header else []) + [f"{key}: {body[key]}" for key in body]
    return "\n".join(lines) + "\n\nthe rest of the commit message"


# ---------------------------- needs_contract -------------------------------
def test_needs_contract_protected_paths():
    for path in (CORE, SKILL, FS_MOD, VET):
        assert needs_contract([path]) is True
    assert needs_contract(["docs/index.rst", CORE]) is True


def test_needs_contract_ignores_tests_markdown_unprotected():
    assert needs_contract(["FoundationSkills/foundationskills/core/tests/test_verifier.py"]) is False
    assert needs_contract(["FoundationSkills/foundationskills/skills/x/tests/y.py"]) is False
    assert needs_contract(["FoundationSkills/foundationskills/vetting/NOTE.md"]) is False
    assert needs_contract(["README.md", "scripts/tool.py", "docs/x.rst"]) is False
    assert needs_contract([]) is False


# ---------------------------- parse_contract -------------------------------
def test_parse_contract_full_message():
    contract = parse_contract(make_msg())
    assert contract is not None
    assert contract.component == "verifier"
    assert contract.preserves == tuple(REQUIRED_INVARIANTS)
    assert contract.rollback == ROLLBACK


def test_parse_contract_header_and_field_rules():
    assert parse_contract("plain commit message") is None
    assert parse_contract(make_msg(header="Change-Contract")) is None  # exact line required
    for field in FIELDS:
        assert parse_contract(make_msg(drop=(field,))) is None, field
        assert parse_contract(make_msg(empty=(field,))) is None, field


# --------------------------------- check -----------------------------------
def test_check_full_valid_message_is_clean():
    assert check(make_msg(), [CORE]) == []


def test_check_missing_header_is_called_out():
    problems = check("just a commit message", [CORE])
    assert len(problems) == 1
    assert "Change-Contract" in problems[0]


def test_check_names_a_missing_field():
    problems = check(make_msg(drop=("Component",)), [CORE])
    assert len(problems) == 1 and "Component" in problems[0]
    problems = check(make_msg(empty=("Predicted-Gain",)), [CORE])
    assert len(problems) == 1 and "Predicted-Gain" in problems[0]


def test_check_names_missing_invariants():
    for invariant in REQUIRED_INVARIANTS:
        keep = "; ".join(i for i in REQUIRED_INVARIANTS if i != invariant)
        assert check(make_msg(preserves=keep), [CORE]) == ["Preserves must list: %s" % invariant]
    problems = check(make_msg(preserves="TYPED-BACK HASH; real-gpu budget"), [CORE])
    assert problems == ["Preserves must list: 0/5/95/96 exits"]


def test_check_bad_rollback():
    for bad in ("abc", "not-a-commit", "g" * 8, ROLLBACK * 6):
        assert check(make_msg(rollback=bad), [CORE]) == ["Rollback must be a 7-40 char hex commit"]
    assert check(make_msg(rollback="a" * 7), [CORE]) == []
    assert check(make_msg(rollback="f" * 40), [CORE]) == []


def test_check_unprotected_diff_without_block_is_clean():
    assert check("random commit message", ["docs/readme.md", "scripts/t.py"]) == []
    assert check("", ["unprotected/file.py"]) == []
    assert check("", []) == []


def test_check_never_raises_on_empty_strings():
    assert isinstance(check("", [CORE]), list) and check("", [CORE])
    assert parse_contract("") is None
    assert parse_contract("Change-Contract:\n\n") is None
    assert needs_contract([""]) is False
    assert check("Change-Contract:\n\ntrailing", [""]) == []
