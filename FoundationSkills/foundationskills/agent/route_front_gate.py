"""Stage-1 front gate for deterministic two-stage trigger routing.

Scores requests against precompiled registry entries using alias hits,
keyword overlap and negative penalties – pure stdlib, zero model calls.
"""

from __future__ import annotations

import re

from foundationskills.agent.trigger_registry import tokens

__all__ = ["score", "front_gate"]


def _as_phrase(phrase: str, text: str) -> bool:
    """Return True when *phrase* occurs in *text* at word boundaries."""
    if not phrase:
        return False
    if phrase[0].isalnum():
        return bool(re.search(r"\b" + re.escape(phrase) + r"\b", text))
    return phrase in text


def score(request: str, skill: dict) -> float:
    """Return a deterministic routing score for *request* against *skill*.

    + 10 for an alias whole-phrase hit (or "/<name>" for user skills);
    + 1  per distinct request token found in *skill*["keywords"];
    - 5  per negative whose phrase tokens match the request at ≥ 60 %.
    """
    lower = request.lower()
    req_toks = set(tokens(request))
    s = 0.0

    # Alias hit → + 10.0
    aliases = skill.get("aliases") or []
    hit = any(_as_phrase(a.lower(), lower) for a in aliases)
    if not hit and skill.get("invocation") == "user":
        slash = "/" + skill["name"].lower()
        hit = slash in lower or lower.startswith(slash)
    if hit:
        s += 10.0

    # Keyword-token overlap → + 1.0 each
    kws = set(skill.get("keywords") or [])
    s += float(len(req_toks & kws))

    # Negative-clause penalty → − 5.0 each (≥ 60 % token coverage)
    for neg in skill.get("negatives") or []:
        pt = set(tokens(neg.get("phrase", "")))
        if pt and len(pt & req_toks) / len(pt) >= 0.6:
            s -= 5.0

    return s


def _slash_forms(name: str) -> tuple[str, tuple[str, ...]]:
    """Return *name* lowered and its accepted ``/…`` trigger forms."""
    low = name.lower()
    for pfx in ("fsskills-", "fs-"):
        if low.startswith(pfx):
            return low, ("/" + low, "/" + low[len(pfx):])
    return low, ("/" + low,)


def _slash_match(req_lower: str, forms: tuple[str, ...]) -> bool:
    """Return True when *req_lower* opens with one of *forms* (word edge)."""
    for f in forms:
        if req_lower.startswith(f):
            rest = req_lower[len(f):]
            if not rest or rest[0] in " \t\n,":
                return True
    return False


def _routable(name: str) -> bool:
    """False when FSKILLS_LIFECYCLE_FILE marks the skill quarantined or retired (approved stages only)."""
    import os

    path = os.environ.get("FSKILLS_LIFECYCLE_FILE")
    if not path:
        return True
    from foundationskills.core import lifecycle

    return lifecycle.routable(lifecycle.load_state(path), name)


def front_gate(
    request: str,
    registry: dict,
    *,
    k: int = 3,
    include_user_skills: bool = False,
) -> dict:
    """Return top-*k* candidates, an ``abstain`` flag and ``redirects``.

    Skills with ``invocation == "user"`` are included only when
    *include_user_skills* is True or *request* opens with ``/<name>``.
    """
    lower = request.lower().strip()
    req_toks = set(tokens(request))
    candidates: list[dict] = []
    redirects: set[str] = set()

    for skill in registry.get("skills") or []:
        # -- user-skill gate (A3.2) --
        own = True
        if skill.get("invocation") == "user" and not include_user_skills:
            _, forms = _slash_forms(skill["name"])
            own = _slash_match(lower, forms)

        # -- collect redirect targets from every matched negative --
        for neg in skill.get("negatives") or []:
            pt = set(tokens(neg.get("phrase", "")))
            if pt and len(pt & req_toks) / len(pt) >= 0.6:
                redirects.add(neg["route_to"])

        if not own:
            continue
        if not _routable(skill["name"]):  # C10.2: an operator-approved quarantine/retirement leaves automatic routing
            continue

        sc = score(request, skill)
        if sc > 0:
            candidates.append({"name": skill["name"], "score": sc})

    candidates.sort(key=lambda c: (-c["score"], c["name"]))
    candidates = candidates[:k]

    return {
        "candidates": candidates,
        "abstain": not candidates,
        "redirects": sorted(redirects),
    }
