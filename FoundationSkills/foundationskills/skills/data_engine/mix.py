
"""Mixture design for the Data Engine, driven by ``mixing_rules.yaml``.

Rule selection: the most specific matching rule wins, where specificity is
the number of ``when`` keys (``stage_in``, ``goal_in``, ``preserve_general``).
Components of a winning rule that are not listed in ``available`` are dropped
and the survivors renormalized; if none survive, the next rule is tried. If no
rule matches at all, a single ``all`` component is returned (no mixing).

Silent-spec choices documented here:

* Rule component entries may carry ``ratio`` (a weight), ``min`` (granted
  first) and ``max`` (a clamp). Allocation: grant mins, split the residual by
  weight, clamp at max, then renormalize to exactly 1.0 (±1e-9 by adjusting
  the largest component).
* Scarcity: for CPT, when ``domain_tokens < 500M`` the domain share is capped
  at 0.6, the excess moves to the general replay component, and the domain
  component carries ``max_epochs: 4`` (to be honored by the ``mix`` op).
* Tool-calling mixtures are flagged: their synthetic component is phase-2
  data and the output rationale says so.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any

import yaml

DOMAIN_SCARCITY_TOKENS = 500_000_000
_PHASE2_COMPONENTS = {"tool_traces"}


class MixtureError(ValueError):
    """Raised when mixing rules cannot be loaded or applied."""


@lru_cache(maxsize=1)
def load_mixing_rules() -> list[dict[str, Any]]:
    """Load and lightly validate foundationskills/skills/data_engine/mixing_rules.yaml."""
    rel = resources.files("foundationskills.skills.data_engine").joinpath("mixing_rules.yaml")
    path = str(rel)
    try:
        with rel.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise MixtureError(f"mixing rules file not found: {path}") from exc
    rules = (raw or {}).get("rules") if isinstance(raw, dict) else None
    if not isinstance(rules, list) or not rules:
        raise MixtureError(f"mixing rules file {path} must define a non-empty 'rules' list")
    for rule in rules:
        if not isinstance(rule, dict) or not rule.get("id") or not isinstance(rule.get("components"), dict):
            raise MixtureError(f"mixing rules file {path}: every rule needs 'id' and 'components'")
    return [dict(r) for r in rules]


def _matches(rule: dict[str, Any], *, goal: str | None, stage: str, preserve_general: bool) -> bool:
    when = rule.get("when") or {}
    if "stage_in" in when and stage not in (when["stage_in"] or []):
        return False
    if "goal_in" in when and goal not in (when["goal_in"] or []):
        return False
    if "preserve_general" in when and bool(when["preserve_general"]) != bool(preserve_general):
        return False
    return True


def _allocate(entries: dict[str, dict[str, float]]) -> dict[str, float]:
    """Grant mins, then water-fill the residual by ratio weight: a component that
    would exceed its max is pinned there and its excess re-split among the rest.
    Renormalizing after clamping would push a capped component back over its cap,
    so caps are honoured exactly; if every component is capped below 1.0 in total,
    the shortfall is reported by the caller through the realised ratios."""
    names = list(entries)
    lo = {n: max(float(entries[n].get("min", 0.0)), 0.0) for n in names}
    hi = {n: float(entries[n].get("max", 1.0)) for n in names}
    weight = {n: max(float(entries[n].get("ratio", 1.0)), 0.0) for n in names}
    alloc = dict(lo)
    free = [n for n in names if alloc[n] < hi[n]]
    residual = max(1.0 - sum(alloc.values()), 0.0)
    while residual > 1e-12 and free:
        wsum = sum(weight[n] for n in free) or float(len(free))
        share = {n: residual * ((weight[n] / wsum) if wsum else 1.0 / len(free)) for n in free}
        pinned = [n for n in free if alloc[n] + share[n] >= hi[n] - 1e-12]
        if not pinned:
            for n in free:
                alloc[n] += share[n]
            residual = 0.0
            break
        for n in pinned:
            residual -= hi[n] - alloc[n]
            alloc[n] = hi[n]
        free = [n for n in free if n not in pinned]
    total = sum(alloc.values())
    if free and abs(1.0 - total) > 0:
        # exact-sum correction on the largest component that still has headroom
        largest = max(free, key=lambda n: alloc[n])
        alloc[largest] += 1.0 - total
    return alloc


def design_mixture(
    *,
    goal: str | None,
    stage: str,
    domain_tokens: int | float | None,
    total_tokens: int | None = None,
    preserve_general: bool,
    available: list[str],
) -> dict[str, Any]:
    """Design a token mixture for one stage.

    Returns ``{components: [{name, ratio, tokens, source_hint, because}],
    rationale: [str], citations: [str]}`` where ratios sum to 1.0 (±1e-9).
    A scarce-domain CPT mixture also sets ``max_epochs`` on the domain
    component and explains the replay growth in ``rationale``.
    """
    rules = sorted(
        load_mixing_rules(), key=lambda r: -len((r.get("when") or {}))
    )
    avail = list(available or [])
    rationale: list[str] = []
    chosen: tuple[dict[str, Any], dict[str, dict[str, float]]] | None = None

    for rule in rules:
        if not _matches(rule, goal=goal, stage=stage, preserve_general=preserve_general):
            continue
        comps: dict[str, dict[str, float]] = {}
        dropped: list[str] = []
        for name, entry in (rule.get("components") or {}).items():
            if avail and name not in avail:
                dropped.append(str(name))
                continue
            comps[str(name)] = dict(entry or {})
        if not comps:
            continue
        for name in dropped:
            rationale.append(
                f"rule {rule['id']}: component {name!r} is not in `available`; dropped and renormalized."
            )
        chosen = (rule, comps)
        break

    if chosen is None:
        rationale.append(
            f"no mixing rule matched (stage={stage!r}, goal={goal!r}, preserve_general={preserve_general}); "
            "falling back to a single undifferentiated component."
        )
        return {
            "components": [
                {
                    "name": "all",
                    "ratio": 1.0,
                    "tokens": None if total_tokens is None else int(total_tokens),
                    "source_hint": "pool all available sources",
                    "because": "fallback: no mixing rule matched",
                }
            ],
            "rationale": rationale,
            "citations": [],
        }

    rule, comps = chosen
    rationale.append(f"rule {rule['id']}: {rule.get('because', '')}")

    stage_l = stage.lower()
    if stage_l in {"cpt", "pretrain"} and preserve_general and domain_tokens is not None:
        if DOMAIN_SCARCITY_TOKENS > 0 and float(domain_tokens) < DOMAIN_SCARCITY_TOKENS:
            domain_name = "domain" if "domain" in comps else None
            if domain_name is not None:
                rationale.append(
                    f"small domain corpus ({int(float(domain_tokens)):,} tokens < 500M): cap domain epochs "
                    "at 4 in the mix op and grow the replay share (Ibrahim et al. 2024)."
                )
                comps[domain_name] = dict(comps[domain_name], max=0.6)
                comps[domain_name]["__scarce__"] = 1.0  # marker consumed below

    alloc = _allocate(comps)
    components: list[dict[str, Any]] = []
    for name, entry in comps.items():
        ratio = alloc[name]
        tokens = None if total_tokens is None else int(round(ratio * int(total_tokens)))
        source_hint = {
            "general_replay": "broad general/web corpus (see catalog discover())",
            "general_instruction_replay": "general instruction SFT set (e.g. smoltalk slice)",
            "general_chat": "general chat SFT data",
            "verifiable_math_replay": "math prompts with verifiable answers (e.g. GSM8K/MATH train)",
        }.get(name, f"data matching component {name!r} (see catalog discover())")
        comp: dict[str, Any] = {
            "name": name,
            "ratio": ratio,
            "tokens": tokens,
            "source_hint": source_hint,
            "because": f"{rule['id']}: {rule.get('because', '')}",
        }
        if entry.pop("__scarce__", None):
            comp["max_epochs"] = 4
        components.append(comp)

    # fix token rounding so token totals match total_tokens exactly
    if total_tokens is not None and components:
        drift = int(total_tokens) - sum(int(c["tokens"]) for c in components)
        largest = max(components, key=lambda c: c["ratio"])
        largest["tokens"] = int(largest["tokens"]) + drift

    if any(c["name"] in _PHASE2_COMPONENTS for c in components):
        rationale.append(
            "PHASE-2: synthetic tool-call traces require the 'synthesize' / 'toolcall_format' ops, "
            "which are not implemented yet (see data_engine.phase2.PHASE2); "
            "sourcing them hits Phase2NotImplemented."
        )
        for c in components:
            if c["name"] in _PHASE2_COMPONENTS:
                c["source_hint"] = "PHASE-2 (not executable yet): " + c["source_hint"]

    drift = 1.0 - sum(c["ratio"] for c in components)
    if abs(drift) > 1e-9:  # defensive: _allocate already exacts the sum
        maxc = max(components, key=lambda c: c["ratio"])
        maxc["ratio"] += drift

    citations = []
    if rule.get("citation"):
        citations.append(str(rule["citation"]))
    return {"components": components, "rationale": rationale, "citations": citations}
