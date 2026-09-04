"""Regression tests for the reference gate.

These exist because of a specific miss. The generated `ExpertAliasGate` shipped with
`_expert_fingerprint` hashing each tensor's *raw name*, which contains the expert
index — so expert 0 and expert 16 hashed differently no matter what bytes they held,
`alias_map` was always empty, and the gate returned PASS on the exact 128-saved-as-16
artifact it was written to catch.

The full 61-test core suite passed against that broken gate. Only the MUST_FIRE
control caught it. That is the right outcome for the controls mechanism and the wrong
outcome for the test suite, because a rule with no test behind it is a rule that
survives the next refactor. So the control's finding is pinned here as a test.
"""

from __future__ import annotations

import pytest

from foundationscale.gates.core import Verdict
from foundationscale.gates.example import ExpertAliasGate, ExpertCheckContext
from foundationscale.gates.fixtures import (
    make_aliased_experts,
    make_empty_experts,
    make_healthy_experts,
    make_local_name_experts,
)

to_context = ExpertCheckContext.from_expert_set


@pytest.fixture
def gate() -> ExpertAliasGate:
    return ExpertAliasGate()


def test_aliased_experts_are_blocked(gate: ExpertAliasGate) -> None:
    """128 experts stored as 16 replicated ones must not pass. The original defect."""
    result = gate.run(to_context(make_aliased_experts(num_experts=128, period=16)))
    assert result.verdict is Verdict.FAIL, (
        f"the 128-saved-as-16 artifact returned {result.verdict.name}: {result.detail}"
    )
    assert result.verdict.blocking
    assert result.coverage.checked == 128


def test_alias_period_is_diagnosed_not_just_alarmed(gate: ExpertAliasGate) -> None:
    """The failure must identify *how* it aliases, not merely that it does."""
    result = gate.run(to_context(make_aliased_experts(num_experts=128, period=16)))
    assert result.evidence["alias_period"] == 16
    assert result.evidence["distinct_experts"] == 16
    assert len(result.evidence["offenders"]) == 112


def test_fingerprint_ignores_the_expert_index_it_compares(gate: ExpertAliasGate) -> None:
    """The direct unit-level statement of the bug.

    Two experts holding byte-identical content must fingerprint identically. If the
    expert index leaks into the hash they never will, and no content comparison in
    this gate can ever fire.
    """
    experts = make_aliased_experts(num_experts=32, period=16)
    grouped: dict[int, dict[str, bytes]] = {}
    for name, blob in experts.tensors.items():
        grouped.setdefault(experts.expert_index[name], {})[name] = blob

    assert grouped[0] != grouped[16], "control: the raw keys really do differ"
    assert list(grouped[0].values()) == list(grouped[16].values()), (
        "control: the fixture must produce byte-identical content for 0 and 16"
    )
    assert gate._expert_fingerprint(grouped[0]) == gate._expert_fingerprint(grouped[16])


def test_healthy_experts_pass(gate: ExpertAliasGate) -> None:
    """A gate that blocks everything gets switched off, which is a gate that blocks nothing."""
    result = gate.run(to_context(make_healthy_experts(num_experts=8)))
    assert result.verdict is Verdict.PASS, result.detail
    assert not result.verdict.blocking


def test_local_names_are_blocked_before_content_is_read(gate: ExpertAliasGate) -> None:
    result = gate.run(to_context(make_local_name_experts()))
    assert result.verdict is Verdict.FAIL
    assert "LOCAL" in result.detail or "local" in result.detail


def test_empty_expert_set_is_vacuous_not_passing(gate: ExpertAliasGate) -> None:
    """The `all([])` case, at the gate level rather than the contract level."""
    result = gate.run(to_context(make_empty_experts()))
    assert result.verdict is Verdict.VACUOUS
    assert result.verdict.blocking
    assert result.coverage.checked == 0
