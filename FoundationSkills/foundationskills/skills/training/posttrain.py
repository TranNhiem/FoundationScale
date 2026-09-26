
"""Post-training stage and algorithm selection against measured FS capabilities.

Algorithm facts (which RL algorithms actually run) always come from
``caps.rl_runnable`` / ``caps.check`` — never hard-coded. New algorithms only
need an algorithm card YAML: when the knowledge loaders are importable, card
names for a stage extend the candidate list beyond FS's registry names.
"""
from __future__ import annotations

from typing import Any

# FS-measured runnable RL order of preference (reference-free group RL).
RL_PREFERENCE_ORDER: tuple[str, ...] = ("dr_grpo", "gspo", "dapo")

_SUPERVISED_ALGORITHMS = {"pretrain": "causal_lm", "cpt": "causal_lm", "sft": "sft"}


def _algorithm_cards_for_stage(stage: str) -> list[str]:
    """Card names for ``stage``; empty when the knowledge base is unavailable."""
    try:
        from foundationskills.skills.training.knowledge import load_algorithm_cards

        cards = load_algorithm_cards()
    except Exception:
        return []
    names: list[str] = []
    for name, card in cards.items():
        card_stage = getattr(card, "stage", None)
        if card_stage is None and hasattr(card, "raw") and isinstance(card.raw, dict):
            card_stage = card.raw.get("stage")
        if card_stage == stage:
            names.append(str(name))
    return names


def _requested_rl_algorithm(goal: Any, data_facts: dict) -> str:
    if isinstance(data_facts, dict) and data_facts.get("requested_rl_algorithm"):
        return str(data_facts["requested_rl_algorithm"])
    if isinstance(goal, dict):
        for key in ("rl_algorithm", "preferred_rl_algorithm", "algorithm_preference"):
            if goal.get(key):
                return str(goal[key])
    return "grpo"


def _first_runnable_rl(caps: Any) -> str | None:
    for name, *_ in _rl_candidates(caps):
        if caps.check("rl", algorithm=name) is None:
            return name
    return None


def _rl_candidates(caps: Any, requested: str | None = None) -> list[tuple[str, str | None]]:
    """Ordered (name, refusal|None) RL candidates: requested, FS preference
    order, algorithm-card names, remaining registry names."""
    order: list[str] = []
    if requested:
        order.append(requested)
    order.extend(a for a in RL_PREFERENCE_ORDER if a not in order)
    for name in _algorithm_cards_for_stage("rl"):
        if name not in order:
            order.append(name)
    for name in getattr(caps, "rl_algorithms", ()) or ():
        if name not in order:
            order.append(name)
    checked: list[tuple[str, str | None]] = []
    for name in order:
        try:
            checked.append((name, caps.check("rl", algorithm=name)))
        except Exception as exc:  # noqa: BLE001 - treat as missing, keep checking others
            checked.append((name, f"missing: caps.check failed for {name}: {exc}"))
    return checked


def select_posttrain_stages(goal: Any, data_facts: dict, caps: Any) -> list[dict[str, Any]]:
    """Which post-train stages the goal+data justify, in execution order.

    SFT is the default floor (it is also the RL prerequisite in practice);
    preference is only proposed for pair data WITHOUT verifiable answers
    (answers make RL the better stage); RL requires verifiable answers.
    """
    facts = dict(data_facts or {})
    has_pairs = bool(facts.get("has_pairs"))
    has_answers = bool(facts.get("has_verifiable_answers"))
    has_instructions = bool(facts.get("has_instructions"))

    stages: list[dict[str, Any]] = []
    if has_instructions or has_answers or not has_pairs:
        stages.append(
            {
                "stage": "sft",
                "name": "sft",
                "because": "instruction/chat data present (or no stronger signal); SFT is the default post-training stage",
                "requested_algorithm": None,
            }
        )
    if has_pairs and not has_answers:
        stages.append(
            {
                "stage": "preference",
                "name": "preference",
                "because": "preference pairs without verifiable answers: the preference family is the ask (likely non-executable on FS today)",
                "requested_algorithm": str(facts.get("requested_preference_algorithm") or "dpo"),
            }
        )
    if has_answers:
        stages.append(
            {
                "stage": "rl",
                "name": "rl",
                "because": "verifiable gold answers enable RL with verifiable rewards (GVR-style group RL)",
                "requested_algorithm": _requested_rl_algorithm(goal, facts),
            }
        )
    return stages


def select_algorithm(stage: str, goal: Any, data_facts: dict, caps: Any) -> dict[str, Any]:
    """Pick the algorithm for one stage.

    Returns ``{algorithm, because, runnable, fallback, missing}``; ``missing``
    is FS's own refusal text when the stage cannot run on the installed FS.
    """
    facts = dict(data_facts or {})
    stage = str(stage)

    if stage in _SUPERVISED_ALGORITHMS:
        algorithm = _SUPERVISED_ALGORITHMS[stage]
        missing = caps.check(stage, backend="fsdp")
        if algorithm == "causal_lm":
            because = "FS has no dedicated pretrain objective; CPT/pretraining run through foundationscale-train --objective sft as causal LM over the corpus `text` column"
        else:
            because = "SFT runs through foundationscale-train --objective sft (the only measured objective)"
        return {"algorithm": algorithm, "because": because, "runnable": missing is None, "fallback": None, "missing": missing}

    if stage == "rl":
        requested = _requested_rl_algorithm(goal, facts)
        checked = _rl_candidates(caps, requested)
        runnable = [(name, miss) for name, miss in checked if miss is None]
        if runnable:
            chosen = runnable[0][0]
            fallback = runnable[1][0] if len(runnable) > 1 else None
            if chosen == requested:
                because = f"{chosen} requested and measured runnable by FS RLTrainer"
            else:
                refusal = next((miss for name, miss in checked if name == requested), None) or "not runnable"
                because = (
                    f"{requested.upper()} requested; FS runs {chosen} (reference-free) today — "
                    f"FS refuses {requested} ({str(refusal)[:120]})"
                )
            return {"algorithm": chosen, "because": because, "runnable": True, "fallback": fallback, "missing": None}
        first_missing = checked[0][1] if checked else "missing: RL algorithm registry unmeasured"
        return {
            "algorithm": requested,
            "because": "no measured-runnable RL algorithm on the installed FS (registered != runnable)",
            "runnable": False,
            "fallback": None,
            "missing": first_missing,
        }

    if stage == "preference":
        requested = str(facts.get("requested_preference_algorithm") or "dpo")
        missing = caps.check("preference", algorithm=requested)
        has_answers = bool(facts.get("has_verifiable_answers"))
        rl_fallback = _first_runnable_rl(caps) if has_answers else None
        because = (
            f"{requested} requested; FS's preference/online family is 'not wired to this loop' on the installed FS"
        )
        if rl_fallback is not None:
            because += f"; proposed (a): run an RL stage with verifiable rewards instead ({rl_fallback})"
        else:
            because += "; proposed (b): keep a non-executable DPO stage with `missing` set until FS wires the preference loop"
        return {
            "algorithm": requested,
            "because": because,
            "runnable": missing is None,
            "fallback": rl_fallback,
            "missing": missing if missing is not None else None if rl_fallback is None else f"preference family not wired; RL fallback {rl_fallback} preferred",
        }

    return {
        "algorithm": None,
        "because": f"unknown stage {stage!r}; cannot select an algorithm",
        "runnable": False,
        "fallback": None,
        "missing": f"missing: algorithm support for stage {stage!r}",
    }
