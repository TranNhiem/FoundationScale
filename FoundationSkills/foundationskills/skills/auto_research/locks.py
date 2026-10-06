"""Close lock: a closed campaign admits no further mutating action (AR-LG-002).

Pure over ``Ledger.entries()`` dicts (keys ``op``/``campaign``): the check only decides whether a
mutating action may continue - it appends and rewrites nothing (M2 spec sections 1 and 4).
"""
from __future__ import annotations

from typing import Any

# ALLOWLIST of actions that stay open on a closed campaign (inspection / a close re-report).
# Every other action - the seven MUTATING_ACTIONS AND unknown names AND junk like None / "" / "claim "
# - is refused with AR-LG-002 (M2 spec section 1: close is final). MUTATING_ACTIONS is kept exported
# for documentation only (the lock does not use it to decide).
READ_ONLY_ACTIONS = ("check", "close")
MUTATING_ACTIONS = ("envelope", "launch", "submit", "cancel", "record", "claim", "propose")


def closed_campaign(entries: list[dict[str, Any]] | None, campaign: str) -> bool:
    """True once ``campaign`` carries a ``campaign_closed`` entry (close is final: no re-open).

    Non-dict rows in ``entries`` are skipped along the way (never raised over): only ``dict``
    rows are inspected. ``entries=None`` or a non-list is treated as "no entries".
    """
    if not isinstance(entries, list):
        return False
    return any(
        isinstance(entry, dict) and entry.get("op") == "campaign_closed" and entry.get("campaign") == campaign
        for entry in entries
    )


def closing_check(entries: list[dict[str, Any]] | None, campaign: str, action: str) -> list[tuple[str, str]]:
    """``[("AR-LG-002", "campaign_closed")]`` when ``str(action)`` is NOT in the READ_ONLY_ACTIONS allowlist on a closed campaign.

    ALLOWLIST semantics are mandatory: only ``READ_ONLY_ACTIONS`` (``check``, ``close``) stay open on a
    closed campaign (inspection of the final report / a close re-report). Every other action - the seven
    mutation ops in ``MUTATING_ACTIONS``, an unknown name, or junk like ``None``/``""``/``"claim "``
    - is refused with ``AR-LG-002`` (M2 spec section 1: close is final).
    """
    if str(action) in READ_ONLY_ACTIONS or not closed_campaign(entries, campaign):
        return []
    return [("AR-LG-002", "campaign_closed")]
