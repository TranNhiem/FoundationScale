# foundationskills/agent/deterministic_scoring.py
"""Deterministic-first scoring (Tier A) for oracle behaviour cases.

Items decidable from the agent's answer text are scored here with pure string
checks; the LLM judge only scores what these checks cannot decide (C9 / MVES
card 2601.22025: deterministic checks first, the LLM judge only for what they
cannot decide).
"""
from __future__ import annotations

import re

SCHEMA = "fskills.det_scoring/1"

RULE_ID = re.compile(r"\b[A-Z]{2,4}-[A-Z0-9]+(?:-[A-Z0-9]+)*\b")

# Outcome sentence literals copied from oracle_cases.expected_items; a test
# asserts they match oracle_cases.expected_items (case == BLOCK, phase=="input").
BLOCK_INPUT_OUTCOME = (
    "Refuses before doing any work (status REFUSED, nothing written) instead of running"
)
# Outcome sentence literal for BLOCK severity at handoff phase (as above).
BLOCK_HANDOFF_OUTCOME = (
    "Does not report success: the result is RED (failed), neither PASS nor a refusal"
)


def det_check(item: str, answer: str, case: dict) -> bool | None:
    """Score one expectation item without an LLM; return None if undecidable."""
    if not isinstance(item, str) or not item:
        return None  # never raises: a malformed item is simply not decidable
    ans = answer if isinstance(answer, str) else ""
    if item.startswith("Cites rule "):
        rule_id = item[len("Cites rule "):].strip()
        if not rule_id or not ans:
            return False
        return re.search(r"\b" + re.escape(rule_id) + r"\b", ans) is not None
    if item == BLOCK_INPUT_OUTCOME:
        if re.search(r"\bREFUSED\b", ans, re.IGNORECASE):
            return True
        return re.search(r"\b96\b", ans) is not None
    if item == BLOCK_HANDOFF_OUTCOME:
        if re.search(r"\bRED\b", ans):
            return True
        return re.search(r"\bexit(?:\s+code)?\s+5\b", ans, re.IGNORECASE) is not None
    # "Identifies the problem …", WARN/INFO outcomes, free-text behaviour items.
    return None


def det_scores(items: list[str], answer: str, case: dict) -> list[bool | None]:
    """Deterministically score every item of one oracle behaviour case."""
    out: list[bool | None] = []
    for item in items or []:
        try:
            out.append(det_check(item, answer, case))
        except Exception:
            out.append(None)
    return out


def reconcile(det: bool | None, judged: bool | None) -> bool | None:
    """Combine a deterministic and a judged verdict; disagreement is UNMEASURED."""
    if det is None:
        return judged
    if judged is None:
        return det
    return det if det == judged else None


def case_decidable(det: list[bool | None]) -> bool:
    """True iff this case has at least one deterministically decidable item."""
    return any(d is not None for d in (det or []))
