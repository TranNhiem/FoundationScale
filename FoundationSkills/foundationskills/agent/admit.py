"""C9 increment 5: the admission gate for self-improvement candidates.
Admission is conjunctive per MVES card 2601.22025; the operator types back the hash per candidate.
Activation-beacon and paired-significance gates reject inert candidates and apparent gains from noise.
This module never writes files: it returns a verdict dict and the operator decides what happens.
"""
from __future__ import annotations

import hashlib
import random

try:  # package layout or plain-path execution
    from foundationskills.core.orchestrator import plan_hash
except ImportError:  # pragma: no cover
    from core.orchestrator import plan_hash

try:  # package layout or plain-path execution
    from foundationskills.agent.frozen_surface import assert_editable
except ImportError:  # pragma: no cover
    from agent.frozen_surface import assert_editable

PROTECTED = ("oracle_item_pass", "routing_recall")
MIN_PAIRS = 3

CONFIRM_REASON = (
    "confirmation required: type back the admission hash printed by "
    "`fskills self-improve hash`"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def admission_payload(skill: str, old_text: str, new_text: str, evidence: dict) -> dict:
    """Build the admission document (schema fskills.self_improve.admission/1) per candidate.

    It carries the skill name, the sha256 of the incumbent and the candidate texts and the
    evidence exactly as supplied, so the operator can re-derive identical bytes and hash.
    """
    return {
        "schema": "fskills.self_improve.admission/1",
        "skill": skill,
        "old_sha256": _sha256(old_text),
        "new_sha256": _sha256(new_text),
        "evidence": evidence,
    }


def admission_hash(payload: dict) -> str:
    """Return the 16 hex chars the operator types back; refusals must never echo them."""
    return plan_hash(payload)


def _table(evidence: dict, key: str) -> dict:
    value = evidence.get(key)
    return value if isinstance(value, dict) else {}


def paired_signflip_p(pairs) -> tuple[float, float]:
    """Return mean candidate-minus-incumbent gain and its one-sided sign-flip p-value."""
    differences = [candidate - incumbent for incumbent, candidate in pairs]
    mean_gain = sum(differences) / len(differences) if differences else 0.0
    nonzero = [difference for difference in differences if difference != 0]
    if not nonzero:
        return mean_gain, 1.0

    observed = sum(nonzero)
    count = len(nonzero)
    if count <= 16:
        assignments = 1 << count
        hits = 0
        for mask in range(assignments):
            total = sum(
                value if mask & (1 << index) else -value
                for index, value in enumerate(nonzero)
            )
            if total >= observed:
                hits += 1
        return mean_gain, hits / assignments

    draws = 20000
    hits = 0
    rng = random.Random(0)
    for _ in range(draws):
        total = sum(value * rng.choice((-1, 1)) for value in nonzero)
        if total >= observed:
            hits += 1
    return mean_gain, (hits + 1) / (draws + 1)


def _valid_rate_pair(pair) -> bool:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return False
    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= value <= 1.0
        for value in pair
    )


def admit(skill: str, old_text: str, new_text: str, evidence: dict,
          confirm: str | None) -> dict:
    """Return the admission verdict: REFUSED, UNMEASURED, RED or PASS with the measured gain.

    Every result carries "verdict" and a "reasons" list; a PASS also carries the "gain",
    paired p-value and pair count. Nothing is written to disk: the operator keeps or discards
    the candidate by hand.
    """
    reasons = list(assert_editable(old_text, new_text))
    expected = admission_hash(admission_payload(skill, old_text, new_text, evidence))
    if (confirm or "").strip() != expected:
        reasons.append(CONFIRM_REASON)
    if reasons:
        return {"verdict": "REFUSED", "reasons": reasons}

    target = evidence.get("target")
    incumbent = _table(evidence, "incumbent")
    candidate = _table(evidence, "candidate")
    gaps: list[str] = []
    if evidence.get("measured") is not True:
        gaps.append("evidence not measured: run the harness before admitting a candidate")
    if target is None:
        gaps.append("target metric not declared in evidence")
    metrics = tuple(dict.fromkeys(([target] if target is not None else []) + list(PROTECTED)))
    for side, table in (("incumbent", incumbent), ("candidate", candidate)):
        for metric in metrics:
            if metric not in table:
                gaps.append(f"metric missing in {side}: {metric}")

    beacon = _table(evidence, "beacon")
    condition = beacon.get("condition")
    fired = beacon.get("fired")
    if (
        not isinstance(condition, str)
        or not condition.strip()
        or not isinstance(fired, int)
        or isinstance(fired, bool)
        or fired < 0
    ):
        gaps.append("candidate declares no activation beacon")

    paired = _table(evidence, "paired")
    pairs = paired.get("pairs")
    paired_valid = (
        paired.get("target") == "behaviour_case_pass"
        and isinstance(pairs, (list, tuple))
        and len(pairs) >= MIN_PAIRS
        and all(_valid_rate_pair(pair) for pair in pairs)
    )
    if not paired_valid:
        gaps.append("paired evidence missing: need >= 3 aligned (case, rep) pairs")

    if gaps:
        return {"verdict": "UNMEASURED", "reasons": gaps}

    red: list[str] = []
    if candidate[target] <= incumbent[target]:
        red.append(f"no gain on target metric: {target}")
    for metric in PROTECTED:
        if candidate[metric] < incumbent[metric]:
            red.append(f"protected metric went down: {metric}")
    if fired == 0:
        red.append("inert candidate: activation beacon never fired")

    mean_gain, p = paired_signflip_p(pairs)
    if not (mean_gain > 0 and p < 0.05):
        red.append(f"gain not significant: paired p={p:.4f} (need < 0.05, mean gain > 0)")
    if red:
        return {"verdict": "RED", "reasons": red}
    return {
        "verdict": "PASS",
        "gain": candidate[target] - incumbent[target],
        "paired_p": p,
        "n_pairs": len(pairs),
        "reasons": [],
    }
