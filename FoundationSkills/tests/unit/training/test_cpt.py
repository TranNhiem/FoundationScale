
"""Unit tests for the CPT policy (pure heuristics; no knowledge base needed)."""
from __future__ import annotations

import sys
from types import SimpleNamespace

from foundationskills.skills.training.cpt import EPOCHS_CAP, cpt_policy


def _variant(size_b: float) -> SimpleNamespace:
    return SimpleNamespace(id=f"fake-{size_b}b", size_b=size_b, arch="dense", active_b=size_b, instruct=False)


def test_lr_size_bands() -> None:
    assert cpt_policy(_variant(1.0), 1_000_000, "reasoning", True)["lr"] == 3e-5
    assert cpt_policy(_variant(3.0), 1_000_000, "reasoning", True)["lr"] == 3e-5
    assert cpt_policy(_variant(8.0), 1_000_000, "reasoning", True)["lr"] == 2e-5
    assert cpt_policy(_variant(12.0), 1_000_000, "reasoning", True)["lr"] == 1.5e-5
    assert cpt_policy(_variant(100.0), 1_000_000, "reasoning", True)["lr"] == 1e-5


def test_token_budget_epochs_capped() -> None:
    policy = cpt_policy(_variant(8.0), 2_000_000_000, "domain_expert", True)
    # min(domain x 4, min(20x domain, 200B cap)) = 8B
    assert policy["token_budget"] == 8_000_000_000
    assert policy["epochs_cap"] == EPOCHS_CAP == 4
    assert policy["epochs_planned"] == 4.0


def test_token_budget_size_cap_binds_for_huge_corpus() -> None:
    policy = cpt_policy(_variant(8.0), 1_000_000_000_000, "domain_expert", True)
    # recommended = min(20T, 200B) = 200B; epochs-limited = 4T -> 200B wins
    assert policy["token_budget"] == 200_000_000_000


def test_replay_ratio_within_literature_band() -> None:
    policy = cpt_policy(_variant(8.0), 2_000_000_000, "domain_expert", True)
    assert 0.05 <= policy["replay_ratio"] <= 0.30


def test_preserve_general_false_disables_replay() -> None:
    policy = cpt_policy(_variant(8.0), 2_000_000_000, "domain_expert", False)
    assert policy["replay_ratio"] == 0.0


def test_replay_fallback_when_mixing_rules_unimportable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "foundationskills.skills.data_engine.mix", None)
    policy = cpt_policy(_variant(8.0), 2_000_000_000, "domain_expert", True)
    assert policy["replay_ratio"] == 0.25
    assert any("default" in line for line in policy["because"])


def test_schedule_discloses_ibrahim_rewarm_note() -> None:
    schedule = cpt_policy(_variant(8.0), 2_000_000_000, "domain_expert", True)["schedule"]
    assert schedule["warmup_ratio"] == 0.01
    assert schedule["min_lr_ratio"] == 0.10
    assert "Ibrahim et al. 2024" in schedule["note"]
    assert "rewarm" in schedule["note"]


def test_because_explains_every_choice() -> None:
    policy = cpt_policy(_variant(8.0), 2_000_000_000, "domain_expert", True)
    joined = " ".join(policy["because"])
    assert policy["because"] and all(isinstance(line, str) and line for line in policy["because"])
    assert "lr" in joined and "token budget" in joined and "schedule" in joined
