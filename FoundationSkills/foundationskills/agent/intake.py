
"""Structured intake: the load-bearing facts each skill needs in a goal_spec.

Spec-silent choices documented here:

- Facts are keyed by a *dotted path* into the goal_spec payload.
- ``missing_facts`` returns every missing fact (flagged ``load_bearing``);
  callers that only want the planning-critical ones filter on that flag.
- A fact is "present" only when the path resolves to something other than
  None, "", or an empty list.
- ``goal_from_answers`` raises ValueError (naming the missing answer) rather
  than producing an invalid goal_spec silently.
"""
from __future__ import annotations

from typing import Any

REQUIRED_FACTS: dict[str, list[dict[str, Any]]] = {
    "training.planner": [
        {
            "key": "objective",
            "question": "What should the model be able to do after training?",
            "why": "drives stage selection and the plan's goal decision",
            "load_bearing": True,
        },
        {
            "key": "goal",
            "question": "Which goal bucket fits best: general_chat|reasoning|math|code|domain_expert|tool_calling?",
            "why": "stage rules and recipe indexing key on the goal bucket",
            "load_bearing": False,
        },
        {
            "key": "base_model.name_or_path",
            "question": "Which base model (hf_id, local path, or knowledge-base variant id)?",
            "why": "model analysis: family, size, architecture, instruct status",
            "load_bearing": True,
        },
        {
            "key": "base_model.size_b",
            "question": "How many parameters (in billions) does the base model have?",
            "why": "needed for memory/time estimates when the model is not in the knowledge base",
            "load_bearing": False,
        },
        {
            "key": "base_model.arch",
            "question": "Is the base model dense or MoE?",
            "why": "MoE changes the FLOPs model (active params) and method choice",
            "load_bearing": False,
        },
        {
            "key": "data.sources",
            "question": "Where is the training data (uri + kind), roughly how many tokens/examples?",
            "why": "data analysis: domain token estimate and stage selection",
            "load_bearing": True,
        },
        {
            "key": "hardware.gpu",
            "question": "Which GPU will this run on?",
            "why": "hardware analysis: memory envelope and MFU estimate",
            "load_bearing": True,
        },
        {
            "key": "hardware.nodes",
            "question": "How many nodes?",
            "why": "GPU count drives sharding and time estimates",
            "load_bearing": True,
        },
        {
            "key": "hardware.gpus_per_node",
            "question": "How many GPUs per node?",
            "why": "GPU count drives sharding and time estimates",
            "load_bearing": True,
        },
        {
            "key": "data_facts.has_verifiable_answers",
            "question": "Do any examples have verifiable gold answers (enables RL)?",
            "why": "without verifiable answers an RL stage is a plan error",
            "load_bearing": True,
        },
        {
            "key": "data_facts.has_pairs",
            "question": "Do you have chosen/rejected preference pairs?",
            "why": "pairs without verifiable answers point at the preference family (non-executable on FS today)",
            "load_bearing": False,
        },
        {
            "key": "data_facts.has_instructions",
            "question": "Do you have instruction/chat data for SFT?",
            "why": "SFT stage selection",
            "load_bearing": False,
        },
        {
            "key": "preserve_general",
            "question": "Must the model keep its general capabilities (enables replay)?",
            "why": "CPT replay ratio and mixture design",
            "load_bearing": False,
        },
        {
            "key": "rl_algorithm",
            "question": "Is a specific RL algorithm requested?",
            "why": "FS only runs dr_grpo/gspo/dapo; requests are rerouted with an explanation",
            "load_bearing": False,
        },
    ],
    "data_engine": [
        {
            "key": "data.sources",
            "question": "Where is the raw data (uri + kind)?",
            "why": "the pipeline cannot be recommended without sources",
            "load_bearing": True,
        },
        {
            "key": "domain",
            "question": "What domain is the data?",
            "why": "mixture and catalog matching",
            "load_bearing": False,
        },
    ],
}


def _resolve(payload: dict, dotted_key: str) -> tuple[bool, Any]:
    node: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    if node is None or node == "" or node == []:
        return False, node
    return True, node


def missing_facts(goal: dict, skills: list[str]) -> list[dict[str, Any]]:
    """All facts from ``skills``' REQUIRED_FACTS not present in ``goal``."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for skill in skills:
        for fact in REQUIRED_FACTS.get(skill, []):
            key = fact["key"]
            if (skill, key) in seen:
                continue
            seen.add((skill, key))
            present, _ = _resolve(goal or {}, key)
            if not present:
                out.append({**fact, "skill": skill})
    return out


def goal_from_answers(answers: dict) -> dict:
    """Build a goal_spec payload from structured intake answers.

    Expected answer keys: objective, capabilities (list), goal, domain,
    preserve_general, base_model {name_or_path, family?, size_b?, arch?,
    model_type?}, sources [{uri, kind, approx_tokens?, approx_examples?,
    domain?, license?}], hardware {gpu, gpus_per_node, nodes}, budget?,
    data_facts?, rl_algorithm?.
    """
    answers = dict(answers or {})
    objective = str(answers.get("objective") or "").strip()
    if not objective:
        raise ValueError("goal_from_answers: 'objective' answer is required")

    base = dict(answers.get("base_model") or {})
    if not str(base.get("name_or_path") or "").strip():
        raise ValueError("goal_from_answers: 'base_model.name_or_path' answer is required")
    base_model: dict[str, Any] = {"name_or_path": str(base["name_or_path"])}
    for key in ("family", "arch", "model_type"):
        if base.get(key):
            base_model[key] = base[key]
    if base.get("size_b") is not None:
        base_model["size_b"] = float(base["size_b"])

    sources: list[dict[str, Any]] = []
    for src in answers.get("sources") or []:
        if not isinstance(src, dict) or not src.get("uri") or not src.get("kind"):
            raise ValueError("goal_from_answers: every source needs 'uri' and 'kind'")
        entry: dict[str, Any] = {"uri": str(src["uri"]), "kind": str(src["kind"])}
        for key in ("approx_tokens", "approx_examples"):
            if src.get(key) is not None:
                entry[key] = int(src[key])
        for key in ("domain", "license"):
            if src.get(key):
                entry[key] = str(src[key])
        sources.append(entry)
    if not sources:
        raise ValueError("goal_from_answers: at least one data source is required")

    hw = dict(answers.get("hardware") or {})
    missing_hw = [k for k in ("gpu", "gpus_per_node", "nodes") if hw.get(k) in (None, "")]
    if missing_hw:
        raise ValueError(f"goal_from_answers: hardware answer missing {', '.join(missing_hw)}")

    capabilities = [str(c) for c in (answers.get("capabilities") or [])] or [objective]
    payload: dict[str, Any] = {
        "objective": objective,
        "target_capabilities": capabilities,
        "domain": answers.get("domain"),
        "preserve_general": bool(answers.get("preserve_general", True)),
        "base_model": base_model,
        "data": {"sources": sources},
        "hardware": {
            "gpu": str(hw["gpu"]),
            "gpus_per_node": int(hw["gpus_per_node"]),
            "nodes": int(hw["nodes"]),
        },
    }
    if answers.get("budget"):
        bonus = dict(answers["budget"])
        payload["budget"] = {k: v for k, v in bonus.items() if v is not None}
    for extra in ("goal", "rl_algorithm", "data_facts"):
        if answers.get(extra) not in (None, "", {}):
            payload[extra] = answers[extra]
    return payload
