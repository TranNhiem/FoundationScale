"""Pure claim chain, downgrades and champion derivation for auto_research (no I/O).

A claim body is the hash-identified statement that one confirm gain was measured over the
confirm seed set against a reference (``"baseline"`` for the chain root). ``prev`` links bodies
into a chain: at close it is re-derived from ledger evidence and a claim whose seeds are no
longer measured for BOTH sides is downgraded in the report (the ledger entry is never rewritten).
"""
from __future__ import annotations

from typing import Any, Iterable

try:  # package import first, plain module path on fixture copies
    from .ledger import canonical, sha256_hex
except ImportError:  # pragma: no cover
    from foundationskills.skills.auto_research.ledger import canonical, sha256_hex

ROOT = "baseline"
PHASE = "confirm"
STAT_KEYS = ("verdict", "mean_delta", "tau", "n_pairs")


def _body(payload: dict[str, Any]) -> dict[str, Any]:
    """The hashed body behind a payload (payloads may carry their derived ``claim_id``)."""
    return {key: value for key, value in payload.items() if key != "claim_id"}


def _measured(result: Any) -> bool:
    """Evidence only: an ok run that is not budget-limited (missing/crash/limited never is)."""
    return isinstance(result, dict) and result.get("status") == "ok" and not result.get("limited")


def claim_body(
    trial: str,
    reference: str,
    prev: str | None,
    seeds: Iterable[Any],
    decision: dict[str, Any] | None,
    fingerprint: str | None,
) -> dict[str, Any]:
    """The claim body (never carries ``claim_id``): one byte change moves the id; missing stats are None."""
    return {
        "trial": trial,
        "phase": PHASE,
        "reference": reference,
        "prev": prev,
        "seeds": list(seeds or []),
        "stats": {key: (decision or {}).get(key) for key in STAT_KEYS},  # missing keys -> None, never 0
        "eval_policy_fingerprint": str(fingerprint or ""),
    }


def claim_id(body: dict[str, Any]) -> str:
    """``sha256:<hex>`` over the canonical body bytes; every body byte is covered."""
    return f"sha256:{sha256_hex(canonical(body))}"


def build_claim(
    spec: dict[str, Any],
    trial: str,
    decision: dict[str, Any] | None,
    set_status: dict[str, Any] | None,
    prev: str | None,
) -> dict[str, Any] | None:
    """The ``claim`` ledger payload for one earned gain; None without ``accepted_gain`` or a complete set."""
    spec = dict(spec or {})
    decision = dict(decision or {})
    set_status = dict(set_status or {})
    if decision.get("verdict") != "accepted_gain" or not set_status.get("complete"):
        return None  # never claim what is not an accepted gain over a complete measured set
    reference = str(decision.get("reference") or set_status.get("reference") or ROOT)
    fingerprint = str(
        decision.get("eval_policy_fingerprint")
        or dict(spec.get("eval_policy") or {}).get("fingerprint")
        or ""
    )
    body = claim_body(
        trial, reference, prev or ROOT, set_status.get("paired") or [], decision, fingerprint
    )
    return {**body, "claim_id": claim_id(body)}


def claim_entries(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ledger claim payloads in chain order, servicing the ``prev`` links (root ``"baseline"`` first).

    A payload without ``claim_id``/``prev`` is a body being linked: its ``prev`` is serviced
    FIRST and its id is then derived over the FINAL body (the first claim roots at
    ``"baseline"``); entries not reachable from the root keep first-seen order after the rooted
    chain.
    """
    entries: list[dict[str, Any]] = []
    tail = ROOT
    for payload in list(payloads):
        if not isinstance(payload, dict):
            continue  # a non-dict claim row is skipped, never linked
        entry = dict(payload)
        linked = not entry.get("prev")
        if linked:  # service the ``prev`` link FIRST ...
            entry["prev"] = tail
        if linked or not entry.get("claim_id"):
            entry["claim_id"] = claim_id(_body(entry))  # ... so the id covers the FINAL body
        tail = entry["claim_id"]
        entries.append(entry)
    buckets: dict[str, list[int]] = {}
    for idx, entry in enumerate(entries):
        buckets.setdefault(str(entry.get("prev")), []).append(idx)
    order: list[int] = []
    seen: set[int] = set()
    queue = list(buckets.get(ROOT, []))
    while queue:
        idx = queue.pop(0)
        if idx in seen:  # self-loop / cycle guard
            continue
        seen.add(idx)
        order.append(idx)
        queue.extend(buckets.get(str(entries[idx].get("claim_id")), []))
    order.extend(idx for idx in range(len(entries)) if idx not in seen)
    return [entries[idx] for idx in order]


def derive_chain(
    claims: list[dict[str, Any]], results_by_key: dict[tuple[Any, Any], dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """(chain, drops) over ``{(trial, seed) -> result}``; a lost seed downgrades its claim.

    Every seed needs a measured (ok, non-limited) result for BOTH the claim trial and its
    reference (``"baseline"`` rows are keyed ``('baseline', seed)`` by the caller). Downgrades never
    cascade: a claim whose ``prev`` is downgraded keeps its own evidence-derived status.
    """
    chain: list[dict[str, Any]] = []
    drops: list[str] = []
    for entry in claim_entries(list(claims)):
        trial, reference = entry.get("trial"), entry.get("reference")
        seeds = list(entry.get("seeds") or [])
        lost = [
            seed
            for seed in seeds
            if not _measured(results_by_key.get((trial, seed)))
            or not _measured(results_by_key.get((reference, seed)))
        ]
        status = "downgraded" if lost else "accepted"
        if lost:
            drops.append(f"claim_downgraded:{entry.get('claim_id')}")
        chain.append(
            {
                "claim_id": entry.get("claim_id"),
                "trial": trial,
                "reference": reference,
                "prev": entry.get("prev"),
                "seeds": seeds,
                "stats": dict(entry.get("stats") or {}),
                "status": status,
            }
        )
    return chain, drops


def champion(chain: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Last surviving accepted claim (walks back over downgraded tails) else the ``"baseline"`` root."""
    for entry in reversed(list(chain or [])):
        if not isinstance(entry, dict):
            continue  # a non-dict row is skipped
        if entry.get("status") == "accepted":
            return {"trial": entry.get("trial"), "claim_id": entry.get("claim_id")}
    return {"trial": None, "claim_id": ROOT}


def reference_rows(
    chain: list[dict[str, Any]] | None, results: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """The champion trial's rows (baseline-role rows while the root holds); later deltas compare against it."""
    rows = [row for row in list(results or []) if isinstance(row, dict)]
    champ = champion(chain)
    if champ.get("claim_id") == ROOT:
        return [row for row in rows if row.get("role") == "baseline"]
    return [row for row in rows if row.get("trial") == champ.get("trial")]
