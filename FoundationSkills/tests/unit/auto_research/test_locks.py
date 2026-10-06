"""AR-LG-002 close lock: every mutating action after ``campaign_closed`` is REFUSED (nothing appended)."""
from __future__ import annotations

import copy
import json

from foundationskills.skills.auto_research.ledger import ledger_files
from foundationskills.skills.auto_research.locks import MUTATING_ACTIONS, closed_campaign, closing_check

CAMPAIGN = "c1"
FINDING = [("AR-LG-002", "campaign_closed")]
READ_ONLY = ("check", "close")  # ALLOWLIST: everything else is refused on a closed campaign (AR-LG-002)


def _entries(events: list[tuple[str, str, str, dict]]) -> list[dict]:
    """Real ``Ledger.entries()`` dicts (hash-chain text parsed back in; no I/O)."""
    chain = ledger_files(events)["ledger/chain.jsonl"]
    return [json.loads(line) for line in chain.splitlines() if line.strip()]


def _approved() -> list[dict]:
    return _entries(
        [
            ("campaign_approved", CAMPAIGN, "", {"campaign_hash": "sha256:" + "0" * 64}),
            ("job_submitted", CAMPAIGN, "t1", {"seed": 101, "phase": "confirm", "job_id": 111}),
        ]
    )


def _closed() -> list[dict]:
    return _entries(
        [
            ("campaign_approved", CAMPAIGN, "", {"campaign_hash": "sha256:" + "0" * 64}),
            ("campaign_closed", CAMPAIGN, "", {"reason": "budget exhausted"}),
        ]
    )


def test_mutating_actions_are_the_m2_actions_plus_m3_propose():
    assert MUTATING_ACTIONS == ("envelope", "launch", "submit", "cancel", "record", "claim", "propose")


def test_not_closed_before_the_campaign_closed_entry():
    entries = _approved()
    assert closed_campaign(entries, CAMPAIGN) is False
    for action in MUTATING_ACTIONS:
        assert closing_check(entries, CAMPAIGN, action) == [], action


def test_closed_after_the_campaign_closed_entry_and_scoped_to_its_campaign():
    entries = _closed()
    assert closed_campaign(entries, CAMPAIGN) is True
    assert closed_campaign(entries, "c2") is False
    # bare op/campaign dicts (the documented minimum keys) work too
    assert closed_campaign([{"op": "campaign_closed", "campaign": CAMPAIGN}], CAMPAIGN) is True


def test_every_mutating_action_after_close_is_ar_lg_002_campaign_closed():
    entries = _closed()
    assert [closing_check(entries, CAMPAIGN, action) for action in MUTATING_ACTIONS] == [FINDING] * 7


def test_read_only_actions_after_close_stay_open():
    entries = _closed()
    for action in READ_ONLY:
        assert closing_check(entries, CAMPAIGN, action) == [], action


def test_closing_check_is_pure_and_leaves_the_chain_untouched():
    entries = _closed()
    before = copy.deepcopy(entries)
    for _ in range(3):
        assert closing_check(entries, CAMPAIGN, "claim") == FINDING
    assert entries == before
    assert len(entries) == 2  # chain count preserved (nothing appended / rewritten)


def test_approval_without_close_keeps_launch_actions_open():
    assert closing_check(_approved(), CAMPAIGN, "envelope") == []
    assert closing_check(_approved(), CAMPAIGN, "submit") == []
