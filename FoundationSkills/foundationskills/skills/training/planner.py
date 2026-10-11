
"""Goal -> training_plan planner.

Pipeline (each step appends to ``decisions``): goal -> model analysis -> data
analysis -> hardware analysis -> stage selection (stage_rules.yaml) ->
algorithm selection -> recipe match -> method -> estimate -> feasibility.

Documented simplifications where the spec is silent:

- Goal "kind" (the recipe/stage-rule vocabulary) is inferred from an explicit
  ``goal["goal"]``/``goal["goal_kind"]`` plus keyword matches over the
  objective and target capabilities; ``goal_in`` matches on ANY inferred kind.
- Unknown stage-rule condition keys fail closed (rule does not fire).
- ``order`` rules are interpreted as ordering hints in file order: hinted
  stages sort first, in hint order; the rest keep their relative add order.
- Stage token heuristics (sft ~ domain/100 clamped to [5M, 200M], rl 2M prompt
  tokens, preference ~ domain/200 clamped to [5M, 50M]) are stated in the
  stage estimate assumptions; RL is planned on 1 GPU because FS RLTrainer is
  single-device.
- Data readiness is only "ready" when a readiness_report with verdict PASS is
  supplied; otherwise each stage carries a data_engine handoff.
"""
from __future__ import annotations

import dataclasses

from types import SimpleNamespace
from typing import Any

from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.skills.training.cpt import cpt_policy
from foundationskills.skills.training.estimate import estimate_memory, estimate_time
from foundationskills.skills.training.feasibility import check_feasibility
from foundationskills.skills.training.knowledge import find_variant, load_algorithm_cards, load_hardware, load_stage_rules
from foundationskills.skills.training.method import select_method
from foundationskills.skills.training.posttrain import select_algorithm
from foundationskills.skills.training.recipes import RecipeQuery, rank_recipes, select_recipe

PLANNING_STEPS: tuple[str, ...] = (
    "goal",
    "model_analysis",
    "data_analysis",
    "hardware_analysis",
    "stage_selection",
    "algorithm_selection",
    "recipe_match",
    "method",
    "estimate",
    "feasibility",
)

_METHODS = ("full", "lora", "qlora")
# Adapters train best at ~10x the full-FT peak LR (Biderman et al. 2024,
# arXiv:2405.09673; Schulman et al. 2025, "LoRA Without Regret"). The CPT policy's
# LR table is full-FT; applied unscaled, a LoRA CPT barely moves.
LORA_LR_MULTIPLIER = 10.0
_KNOWN_GOALS = ("general_chat", "reasoning", "math", "code", "domain_expert", "tool_calling")
_FORMAT_BY_STAGE = {"pretrain": "pretrain", "cpt": "cpt", "sft": "sft", "preference": "preference", "rl": "rl"}
_VERDICT_ORDER = {"ok": 0, "warn": 1, "infeasible": 2}


def _gate_alternatives(alternatives: list, missing: str | None) -> list:
    """An alternative changes a memory knob, not a capability: a stage gap blocks it too."""
    if missing is None:
        return alternatives
    return [
        {**a, "executable": False,
         "missing": "; ".join(m for m in (a.get("missing"), missing) if m)} if isinstance(a, dict) else a
        for a in alternatives
    ]


class PlanningRefusal(Exception):
    """The planner refuses: a load-bearing input is missing or no stage applies."""


# ---------------------------------------------------------------------------
# inference helpers


def _goal_kinds(goal: dict) -> list[str]:
    kinds: list[str] = []
    explicit = goal.get("goal") or goal.get("goal_kind")
    if explicit:
        kinds.append(str(explicit))
    text = " ".join(
        [str(goal.get("objective") or ""), *[str(c) for c in (goal.get("target_capabilities") or [])]]
    ).lower()
    keyword_map = {
        "reasoning": "reasoning",
        "math": "math",
        "code": "code",
        "coding": "code",
        "tool": "tool_calling",
        "chat": "general_chat",
        "assistant": "general_chat",
        "domain": "domain_expert",
        "expert": "domain_expert",
        "speciali": "domain_expert",
    }
    for needle, kind in keyword_map.items():
        if needle in text and kind not in kinds:
            kinds.append(kind)
    for kind in kinds:
        if kind not in _KNOWN_GOALS:
            kinds.remove(kind)
    return kinds or ["general_chat"]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _resolve_hardware(hw_map: dict, hardware_id: str | None, gpu_name: str | None) -> tuple[Any | None, str | None]:
    """Map an exact id, or a loose GPU name ("h100", "DGX H100", "NVIDIA GB200"),
    to a knowledge hardware entry. The GPU family token (the id before its first
    "-", e.g. "h100" of "h100-80gb") must appear in the request, or the request
    must be a prefix of the id. Ambiguity is a miss, never a guess."""
    def norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    if hardware_id and hardware_id in hw_map:
        return hw_map[hardware_id], hardware_id
    wanted = norm(f"{hardware_id or ''} {gpu_name or ''}")
    if not wanted:
        return None, None
    by_id = [k for k in hw_map if (norm(k.split("-", 1)[0]) and norm(k.split("-", 1)[0]) in wanted)
             or norm(k).startswith(wanted)]
    by_name = [k for k, hw in hw_map.items()
               if norm(str(_get(hw, "gpu_name", ""))) and norm(str(_get(hw, "gpu_name", ""))) in wanted]
    for hits in (by_id, by_name):  # id family first; the GPU name only breaks a miss
        if len(hits) == 1:
            return hw_map[hits[0]], hits[0]
        if len(hits) > 1:
            return None, None
    return None, None


def _when_matches(when: dict, ctx: dict) -> bool:
    for key, val in (when or {}).items():
        if key == "goal_in":
            if not ({str(v) for v in (val or [])} & set(ctx["goal_kinds"])):
                return False
        elif key == "domain_tokens_gte":
            if not ctx["domain_tokens"] >= float(val):
                return False
        elif key == "domain_tokens_lt":
            if not ctx["domain_tokens"] < float(val):
                return False
        elif key in ("has_pairs", "has_verifiable_answers", "has_instructions", "base_is_instruct", "preserve_general"):
            if bool(ctx.get(key, False)) != bool(val):
                return False
        else:
            return False  # unknown condition: fail closed, never guess
    return True


_HP_ALIASES = {
    "max_sequence_length": ("seq_len", "max_seq_len"),
    "per_device_batch_size": ("micro_batch", "micro_batch_size"),
    "gradient_checkpointing": ("grad_ckpt",),
    "sharding_strategy": ("sharding",),
}
_LIFECYCLE_RANK = {"pretrain": 0, "cpt": 1, "sft": 2, "preference": 3, "rl": 4}


def _evaluate_stage_rules(rules: list[dict], ctx: dict) -> tuple[list[dict], list[str]]:
    added: list[dict] = []
    skipped: set[str] = set()
    order_hints: list[str] = []
    fired: list[str] = []
    for rule in rules:
        when = rule.get("when") or {}
        if not _when_matches(when, ctx):
            continue
        action = rule.get("action")
        stage = str(rule.get("stage") or "")
        if not stage:
            continue
        if action == "add_stage":
            skipped.discard(stage) if False else None
            if not any(a["stage"] == stage for a in added):
                added.append(
                    {
                        "stage": stage,
                        "name": stage,
                        "rule_id": rule.get("id", "?"),
                        "algorithm_hint": rule.get("algorithm_hint"),
                        "because": str(rule.get("because") or ""),
                    }
                )
                fired.append(f"{rule.get('id', '?')}:+{stage}")
        elif action == "skip_stage":
            skipped.add(stage)
            added = [a for a in added if a["stage"] != stage]
            fired.append(f"{rule.get('id', '?')}:-{stage}")
        elif action == "order":
            if stage not in order_hints:
                order_hints.append(stage)
                fired.append(f"{rule.get('id', '?')}:~{stage}")
    remaining = [a for a in added if a["stage"] not in skipped]
    # Order is the canonical lifecycle unless the goal states its own. "order"
    # rules explain the ordering; they do not compute it, because applying hints
    # in file order once put SFT ahead of CPT.
    explicit = [str(s) for s in (ctx.get("stage_order") or [])]
    rank = {name: i for i, name in enumerate(explicit)} if explicit else _LIFECYCLE_RANK
    remaining = [a for _, a in sorted(enumerate(remaining),
                                      key=lambda pair: (rank.get(pair[1]["stage"], len(rank)), pair[0]))]
    return remaining, fired


def _apply_operator_stages(requested: Any, selected: list[dict], rules: list[dict],
                           fired: list[str], decide: Any, stage_order: Any = None) -> list[dict]:
    """``goal.stages``: the operator's explicit stage set replaces the rule
    selection. Every disagreement with the rules is logged as an
    ``operator_override`` decision quoting the rule's reason, never hidden."""
    if not isinstance(requested, list) or not requested or not all(isinstance(s, str) for s in requested):
        raise PlanningRefusal(f"missing input: goal.stages must be a non-empty list of stage names, got {requested!r}")
    unknown = [s for s in requested if s not in _LIFECYCLE_RANK]
    if unknown:
        raise PlanningRefusal(f"missing input: goal.stages has unknown stage(s) {unknown}; known: {list(_LIFECYCLE_RANK)}")
    by_stage = {a["stage"]: a for a in selected}
    rule_by_id = {str(r.get("id")): r for r in rules}
    out: list[dict] = []
    explicit = [str(s) for s in (stage_order or [])]
    rank = {name: i for i, name in enumerate(explicit)} if explicit else _LIFECYCLE_RANK
    for stage in sorted(dict.fromkeys(requested), key=lambda s: (rank.get(s, len(rank)), _LIFECYCLE_RANK[s])):
        if stage in by_stage:
            out.append(by_stage[stage])
            continue
        skip_ids = [f.split(":")[0] for f in fired if f.endswith(f":-{stage}")]
        why = "; ".join(str(rule_by_id[i].get("because", "")) for i in skip_ids if i in rule_by_id)
        decide("operator_override", f"+{stage}",
               f"operator requested {stage} against the stage rules ({', '.join(skip_ids) or 'no rule added it'})"
               + (f": {why}" if why else ""))
        out.append({"stage": stage, "name": stage, "rule_id": "operator", "algorithm_hint": None,
                    "because": f"operator requested {stage} via goal.stages"})
    for stage, sel in by_stage.items():
        if stage not in requested:
            decide("operator_override", f"-{stage}",
                   f"operator omitted {stage} that rule {sel.get('rule_id', '?')} added: {sel.get('because', '')}")
    return out


def _synthesize_variant(base_model: dict) -> tuple[Any, Any, str | None]:
    """Fallback model description from goal facts. Returns (family, variant, note)."""
    size_b = base_model.get("size_b")
    if size_b is None:
        raise PlanningRefusal(
            "missing input: base_model.size_b (model not found in the knowledge base; "
            "state the parameter count in billions so memory/time can be estimated)"
        )
    size_b = float(size_b)
    arch = str(base_model.get("arch") or "dense")
    variant = SimpleNamespace(
        id=str(base_model.get("name_or_path")),
        hf_id=None,
        local_path=None,
        size_b=size_b,
        active_b=size_b if arch == "dense" else None,
        arch=arch,
        hidden=None,
        layers=None,
        heads=None,
        kv_heads=None,
        head_dim=None,
        vocab=None,
        context_length=None,
        num_experts=None,
        experts_per_token=None,
        tied_embeddings=False,
        instruct=False,
    )
    family = SimpleNamespace(
        name=str(base_model.get("family") or "unknown"),
        display_name=str(base_model.get("name_or_path")),
        fs_family=None,
        model_types=(),
        raw={"name": str(base_model.get("family") or "unknown")},
    )
    note = (
        f"model {base_model.get('name_or_path')!r} not in the knowledge base; "
        "using goal.base_model.size_b/arch as stated facts (unvalidated)"
    )
    return family, variant, note


def _sft_processed_tokens(tokens: int, hparams: dict, readiness_stats: dict, micro_batch: int,
                          assumptions: list[str]) -> int:
    """Tokens FS actually pushes through the model for an SFT stage: epochs x data, padded.

    MEASURED 2026-10-07 (E4B LoRA, AIEC 9.08M tokens, 2 epochs, b4): the estimate priced one
    unpadded pass (9.08M) while FS ran 2 epochs and pads every row to the longest in its
    micro-batch (no packing): 30.6M tokens processed, 1.69x padding, 3.3x the planned hours.
    Padding is estimated from the readiness length histogram as E[max of micro_batch rows] at
    0.75 x each power-of-two bin's upper edge (1.85x predicted on that run); absent histogram ->
    no padding factor, stated as an assumption."""
    processed = float(tokens)
    epochs = hparams.get("epochs")
    if epochs and hparams.get("max_steps") is None and hparams.get("tokens") is None:
        processed *= float(epochs)
        assumptions.append(f"sft processed tokens: {epochs} epoch(s) x {tokens:,}")
    hist = readiness_stats.get("length_hist") if isinstance(readiness_stats, dict) else None
    records, data_tokens = readiness_stats.get("num_records"), readiness_stats.get("num_tokens")
    if isinstance(hist, dict) and hist and records and data_tokens and micro_batch >= 1:
        try:
            bins = sorted((int(k), int(v)) for k, v in hist.items())
        except (TypeError, ValueError):
            bins = []
        total = sum(v for _, v in bins)
        if total > 0:
            cdf = last = expected_max = 0.0
            for upper, count in bins:
                cdf += count / total
                expected_max += (cdf ** micro_batch - last ** micro_batch) * 0.75 * upper
                last = cdf
            factor = max(1.0, expected_max / (float(data_tokens) / float(records)))
            processed *= factor
            assumptions.append(f"sft padding: FS pads each row to the longest in its micro-batch (no packing); "
                               f"x{factor:.2f} from the readiness length histogram at micro_batch {micro_batch}")
    else:
        assumptions.append("sft padding: no readiness length histogram; padding overhead NOT priced")
    return int(processed)


def _stage_tokens(stage: str, data_facts: dict, domain_tokens: int, assumptions: list[str]) -> int:
    if stage == "pretrain":
        tokens = max(domain_tokens * 5, 1_000_000_000)
        assumptions.append(f"pretrain tokens heuristic: max(5x domain, 1B) = {tokens:,}")
        return int(tokens)
    if stage == "sft":
        if data_facts.get("sft_tokens"):
            tokens = int(data_facts["sft_tokens"])
            assumptions.append(f"sft tokens: {tokens:,} (from sources tagged use_for sft, or data_facts.sft_tokens)")
        else:
            tokens = int(min(max(domain_tokens // 100, 5_000_000), 200_000_000))
            assumptions.append(f"sft tokens heuristic: clamp(domain/100, 5M, 200M) = {tokens:,} (state data_facts.sft_tokens to override)")
        return tokens
    if stage == "preference":
        tokens = int(min(max(domain_tokens // 200, 5_000_000), 50_000_000))
        assumptions.append(f"preference tokens heuristic: clamp(domain/200, 5M, 50M) = {tokens:,}")
        return tokens
    if stage == "rl":
        if data_facts.get("rl_tokens"):
            tokens = int(data_facts["rl_tokens"])
            assumptions.append(f"rl tokens: {tokens:,} (stated in data_facts.rl_tokens)")
            return tokens
        # An RL step trains on every sampled completion, not on the prompt set:
        # tokens = prompts x group_size x (prompt_len + completion_len).
        prompts = int(data_facts.get("rl_examples") or 20_000)
        group = int(data_facts.get("rl_group_size") or 8)
        prompt_len = int(data_facts.get("rl_prompt_tokens") or 256)
        gen_len = int(data_facts.get("rl_max_new_tokens") or 1024)
        tokens = prompts * group * (prompt_len + gen_len)
        assumptions.append(
            f"rl tokens heuristic: {prompts:,} prompts x group {group} x ({prompt_len} prompt + {gen_len} generated) "
            f"= {tokens:,} trained tokens for one pass; generation cost is added by the estimator's rl factor"
        )
        return tokens
    return int(domain_tokens)


# Recipe fs.args flag -> canonical hparam key. Emitter-owned flags (objective,
# adapter) and structural ones are not hparams.
_FS_ARG_TO_HPARAM = {"adapter-rank": "lora_rank", "adapter-alpha": "lora_alpha", "adapter-dropout": "lora_dropout",
                     "adapter-target": "lora_targets"}
_FS_ARG_SKIP = frozenset({"objective", "adapter", "model", "dataset", "output-dir", "profile-name", "profile-path",
                          "nodes", "gpus-per-node", "dry-run", "dp"})


def _recipe_stage_hparams(recipe_raw: dict, stage: str) -> dict:
    """The recipe stage's hparams AND its fs.args, folded into one canonical
    dict. fs.args used to ride only on the recipe and never reached the plan,
    so a real E2E command lost the recipe's precision/optimizer/warmup/batch
    settings; folding them in before estimation also keeps estimate == command."""
    for stage_entry in recipe_raw.get("stages", []) or []:
        if not isinstance(stage_entry, dict):
            continue
        if stage_entry.get("stage") == stage or stage_entry.get("name") == stage:
            out: dict[str, Any] = {}
            for raw_flag, value in dict(((stage_entry.get("fs") or {}).get("args")) or {}).items():
                flag = str(raw_flag).lstrip("-").replace("_", "-")
                if flag in _FS_ARG_SKIP:
                    continue
                out[_FS_ARG_TO_HPARAM.get(flag, flag.replace("-", "_"))] = value
            hparams = stage_entry.get("hparams")
            if isinstance(hparams, dict):
                out.update(hparams)  # explicit recipe hparams win over its fs.args spelling
            return out
    return {}


def _card_requires(algorithm: str | None) -> tuple[str, ...]:
    """The algorithm card's declared requirements (e.g. reference_model)."""
    try:
        card = load_algorithm_cards().get(str(algorithm)) if algorithm else None
    except Exception:  # noqa: BLE001 - a missing card declares nothing
        return ()
    raw = _get(card, "raw", {}) or {} if card is not None else {}
    return tuple(str(r) for r in (raw.get("requires") or _get(card, "requires", ()) or ()))


def _card_default_hparams(algorithm: str | None) -> dict:
    if not algorithm:
        return {}
    try:
        cards = load_algorithm_cards()
    except Exception:
        return {}
    card = cards.get(algorithm)
    if card is None:
        return {}
    raw = _get(card, "raw", {}) or {}
    out: dict[str, Any] = {}
    for key, spec in (raw.get("key_hparams") or {}).items():
        if isinstance(spec, dict) and "default" in spec:
            out[key] = spec["default"]
    return out

# ---------------------------------------------------------------------------
# VLM (A4.1): goal_spec declarations -> stage fields, and the gate rules
#
# Evidence base: examples/vlm-2xH200/ (main 270efd5; five VLMs, LoRA r16/a32,
# 16K/32K, image+video+text on 2xH200). Facts the gates defend:
#   - FS auto-targets LoRA only for FS-registered families; every other family
#     needs explicit lora_targets / --adapter-target (emit_train.py:331-348
#     records "family not FS-registered and no lora_targets" as a missing item).
#   - video rows need a video column and a frame count
#     (FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN / FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES).
#   - FS refuses --tp (exit 96) wherever it cannot also save (docs/STATUS.md
#     --tp row); --cp is measured end to end only on text (Qwen2.5-7B, #542).
#   - 32K training is measured with --fused-loss liger (FS's fused Liger loss);
#     128K context is a planning claim, not a measurement.

_FUSED_LOSS_LIGER = "liger"
_VIDEO_COLUMN_DEFAULT = "video"  # data.fs_columns.video_column default when video is declared
_OVERLONG_DEFAULT = "drop"  # data.fs_columns.overlong default
_SEQ_LEN_LIGER_FLOOR = 32768  # >= 32K needs --fused-loss liger (TR-PL-008)
_SEQ_LEN_GEMMA4_31B_CEILING = 32768  # gemma-4-31B-it planned above 32K is RED (TR-PL-007)
_SEQ_LEN_CLAIM_CEILING = 131072  # >= 128K is an unmeasured planning claim (TR-PL-006)
_GEMMA4_31B_ID_TOKENS = ("gemma431b",)  # normalised id fragments ("gemma-4-31B-it" -> "gemma431bit")


def _norm_declaration(raw: Any) -> list[str]:
    """Lower-cased, de-duplicated declaration tokens (order-preserving)."""
    out: list[str] = []
    for item in raw or ():
        tok = str(item).strip().lower()
        if tok and tok not in out:
            out.append(tok)
    return out


def _norm_targets(raw: Any) -> list[str] | None:
    """Declared lora_targets: ``None`` when nothing is declared (FS auto), else the
    normalised list. A declaration of only blank entries normalises to the empty
    list — a vacuous declaration, refused by TR-IN-007."""
    if raw is None:
        return None
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    return [str(t).strip() for t in items if str(t).strip()]


def _int_or_none(value: Any) -> int | None:
    """A declared integer fact; a bool or an unparseable value is not a fact."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def vlm_goal_fields(goal: dict) -> dict[str, Any]:
    """The A4.1 goal_spec declarations (shared contract), normalised.

    Goal keys (all optional): ``modalities`` (list of 'image'|'video'|'text', else
    ``data_facts.modalities``), ``video_frames`` (int, else
    ``data.fs_columns.video_frames``), ``max_sequence_length`` (int), ``lora_targets``
    (list[str]), ``fused_loss`` ('liger'), ``tp``/``cp`` (int).
    ``goal.data.fs_columns`` is the dataset's field -> column mapping and may name
    ``video_column``, ``video_frames`` and ``overlong``. ``goal.module_inventory`` (or
    ``goal.base_model.module_inventory``) is optional: without one TR-IN-007's
    "a target matches no module" check cannot run and the FS post-wrap refusal stands.

    Returns the normalised facts: ``modalities`` (sorted run modalities — empty when
    nothing declares one, never inferred), ``row_modalities`` (sorted modalities the
    rows carry: ``data_facts.modalities``, else the run's), ``fs_columns`` (the
    declared mapping passed through plus the normalised ``video_column`` /
    ``video_frames`` / ``overlong`` keys), ``fused_loss``, ``lora_targets``
    (``None`` = not declared, else normalised), ``max_sequence_length``, ``tp``,
    ``cp``, ``module_inventory``. Unknown modality tokens are carried through to the
    stage and refused by the emit gate FS-IN-004 (no fs_columns feed them).
    """
    goal = goal if isinstance(goal, dict) else {}
    data = goal.get("data") or {}
    facts = goal.get("data_facts") or {}
    cols: dict[str, Any] = {}
    for src in (data, facts):
        declared_cols = src.get("fs_columns") if isinstance(src, dict) else None
        if isinstance(declared_cols, dict):
            cols.update({str(k): v for k, v in declared_cols.items()})
    modalities = sorted(_norm_declaration(goal.get("modalities")) or _norm_declaration(facts.get("modalities")))
    row_modalities = sorted(_norm_declaration(facts.get("modalities")) or list(modalities))
    frames = goal.get("video_frames")
    if frames is None:
        frames = cols.get("video_frames")
    named_column = cols.get("video_column")
    if not isinstance(named_column, str) or not named_column.strip():
        named_column = cols.get("video")
    video_column = str(named_column).strip() if isinstance(named_column, str) and named_column.strip() else None
    overlong = cols.get("overlong")
    fs_columns = dict(cols)
    fs_columns["video_column"] = video_column or (_VIDEO_COLUMN_DEFAULT if "video" in modalities else None)
    fs_columns["video_frames"] = _int_or_none(frames)
    fs_columns["overlong"] = str(overlong).strip() if isinstance(overlong, str) and overlong.strip() else _OVERLONG_DEFAULT
    inventory_raw = goal.get("module_inventory") or (goal.get("base_model") or {}).get("module_inventory")
    module_inventory = [str(m) for m in inventory_raw] if isinstance(inventory_raw, (list, tuple)) and inventory_raw else None
    fused = goal.get("fused_loss")
    return {
        "modalities": modalities,
        "row_modalities": row_modalities,
        "fs_columns": fs_columns,
        "fused_loss": str(fused).strip().lower() if isinstance(fused, str) and fused.strip() else None,
        "lora_targets": _norm_targets(goal.get("lora_targets")),
        "max_sequence_length": _int_or_none(goal.get("max_sequence_length")),
        "tp": _int_or_none(goal.get("tp")),
        "cp": _int_or_none(goal.get("cp")),
        "module_inventory": module_inventory,
    }


def vlm_stage_fields(goal: dict) -> dict[str, Any]:
    """The A4.1 fields the planner copies into every training stage (shared contract).

    Returns a stage-shaped fragment: ``modalities`` (sorted list), ``hparams`` with
    ``fused_loss`` ('liger' | None), ``lora_targets`` (explicit list | None for FS
    auto) and ``max_sequence_length`` (the declared value), and ``data.fs_columns``
    with ``video_column`` ('video' by default when video is in modalities),
    ``video_frames`` and ``overlong`` ('drop' by default). ``plan()`` merges this into
    each stage: an operator declaration wins over recipe/card hparams and lands in
    hparams before the estimate resolves max_sequence_length, so
    estimate == command.
    """
    fields = vlm_goal_fields(goal)
    return {
        "modalities": list(fields["modalities"]),
        "hparams": {
            "fused_loss": fields["fused_loss"],
            "lora_targets": list(fields["lora_targets"]) if fields["lora_targets"] is not None else None,
            "max_sequence_length": fields["max_sequence_length"],
        },
        "data": {"fs_columns": dict(fields["fs_columns"])},
    }


def vlm_gates(stage: dict, goal: dict, caps: FSCapabilities | None) -> list[tuple[str, str]]:
    """The A4.1 gate rules (TR-IN-005..008, TR-PL-006..008) over one training stage.

    Returns ``(rule_id, message)`` pairs; the training skill turns each into a
    Finding (``self.finding(rule_id, message, evidence, recovery)``). Phase (shared
    contract): ``check_inputs`` surfaces TR-IN-005..008 and TR-PL-008 (BLOCK input ->
    REFUSED), ``check_handoff`` surfaces TR-PL-006/007 (BLOCK handoff -> RED).
    TR-PL-008 is a handoff subject that must fire at input: the handoff template can
    only emit findings, never a REFUSED result.

    Conditions (the spec's, and no others):

    - TR-IN-005: a VLM run (image/video rows, a "vlm" model_type, or a vlm family)
      whose family is not FS-registered, training anything but full fine-tuning, with
      no explicit lora_targets -> names "register the family in FS" and "pass explicit
      lora_targets / --adapter-target" (FS auto-targets LoRA only for FS-registered
      families).
    - TR-IN-006: rows carry video but the feed has no video_column or video_frames ->
      names FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN / FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES.
    - TR-IN-007: a declared lora_targets list empty after normalisation, or a target
      matching no module in the plan's module inventory (only checkable when an
      inventory is carried; otherwise the FS post-wrap refusal stands).
    - TR-IN-008: tp > 1 (FS refuses --tp, exit 96, wherever it cannot also save) or
      cp > 1 with image/video rows (--cp is measured end to end only on text).
    - TR-PL-006: planned max_sequence_length >= 131072, or a goal claim that it works.
    - TR-PL-007: gemma-4-31B-it planned above 32768.
    - TR-PL-008: max_sequence_length >= 32768 without fused_loss 'liger'.

    ``stage`` is the planned stage (its hparams carry the facts the estimate priced);
    ``goal`` is the goal_spec payload (the declaration/claim channel). ``caps``
    (FSCapabilities) is the shared contract's capability object: the gate facts are
    knowledge-loop facts (FS-registered families, measured sequence floors) and the
    installed FS's own refusals stand behind them — the --tp refusal is taken from
    ``caps.check`` when the capability object answers.
    """
    stage = stage if isinstance(stage, dict) else {}
    goal = goal if isinstance(goal, dict) else {}
    declared = vlm_goal_fields(goal)
    hparams = stage.get("hparams") if isinstance(stage.get("hparams"), dict) else {}
    data = stage.get("data") if isinstance(stage.get("data"), dict) else {}
    fs_columns = data.get("fs_columns") if isinstance(data.get("fs_columns"), dict) else declared["fs_columns"]

    modalities = set(_norm_declaration(stage.get("modalities")) or declared["modalities"]) | set(declared["row_modalities"])
    multimodal_rows = bool({"image", "video"} & modalities)

    # model identity: the goal's ref first, the knowledge base's ids/name as backup
    model_ref = str((goal.get("base_model") or {}).get("name_or_path") or "")
    family = variant = None
    if model_ref:
        try:
            found = find_variant(model_ref)
        except Exception:  # noqa: BLE001 - a broken knowledge base must not crash a gate
            found = None
        if found is not None:
            family, variant = found
    identity = " ".join(
        "".join(ch for ch in str(token).lower() if ch.isalnum())
        for token in (model_ref, str(_get(variant, "id", "") or ""), str(_get(family, "name", "") or ""))
    )
    is_vlm_family = (
        multimodal_rows
        or "vlm" in {str(t).lower() for t in (_get(family, "model_types", ()) or ())}
        or str(goal.get("model_type") or "").lower() == "vlm"
    )
    method = str(stage.get("method") or goal.get("method") or "").lower()
    named = stage.get("name") or stage.get("stage")
    where = f"stage {str(named)!r} " if named else "the run "

    # the facts the gates read: the planned stage wins where it carries a value
    # (that is what the estimate priced), the goal is the declaration/claim channel.
    raw_targets = hparams.get("lora_targets")
    if raw_targets is None:
        raw_targets = declared["lora_targets"]
    targets = _norm_targets(raw_targets)
    video_column = fs_columns.get("video_column") or declared["fs_columns"].get("video_column")
    frames = fs_columns.get("video_frames")
    frames = frames if frames is not None else declared["fs_columns"].get("video_frames")
    frames = _int_or_none(frames)
    lengths = [
        n
        for n in (_int_or_none(hparams.get("max_sequence_length")), declared["max_sequence_length"])
        if n is not None
    ]
    longest = max(lengths) if lengths else None
    fused_loss = hparams.get("fused_loss") or declared["fused_loss"]
    tp, cp = declared["tp"], declared["cp"]

    out: list[tuple[str, str]] = []

    # TR-IN-005: FS auto-targets LoRA only for FS-registered families
    if is_vlm_family and method != "full" and not targets and _get(family, "fs_family") is None:
        out.append((
            "TR-IN-005",
            f"the VLM family behind {(model_ref or 'unknown')!r} is not FS-registered and no explicit lora_targets "
            "are declared: register the family in FS, or pass explicit lora_targets / --adapter-target "
            "(FS auto-targets LoRA only for FS-registered families)",
        ))

    # TR-IN-006: rows carry video only when a video column and a frame count feed them
    if "video" in modalities:
        missing = [name for name, value in (("video_column", video_column), ("video_frames", frames)) if not value]
        if missing:
            out.append((
                "TR-IN-006",
                "rows carry video but the feed has no "
                + " and ".join(missing)
                + ": declare goal.data.fs_columns.video_column and goal.video_frames "
                "(FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN / FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES)",
            ))

    # TR-IN-007: an explicit lora_targets declaration must give usable names
    if targets is not None:
        if not targets:
            out.append((
                "TR-IN-007",
                "explicit lora_targets is empty after normalisation: pass a non-empty --adapter-target list "
                "(or drop the goal's lora_targets key to let FS auto-target an FS-registered family)",
            ))
        else:
            inventory = stage.get("module_inventory") or declared["module_inventory"]
            unmatched = [t for t in targets if inventory and not any(t == m or t in m for m in inventory)]
            if unmatched:
                out.append((
                    "TR-IN-007",
                    f"explicit lora_targets {unmatched!r} match no module in the module inventory "
                    f"({len(inventory or [])} module(s)): pass --adapter-target names the wrapped model has "
                    "(without an inventory the FS post-wrap refusal stands)",
                ))

    # TR-IN-008: transport axes the measured FS products do not run or have not measured
    if tp is not None and tp > 1:
        message = (
            f"tp={tp} is REFUSED: FS refuses --tp (exit 96) wherever it cannot also save "
            "(docs/STATUS.md --tp row); plan tp=1 and shard with ddp/fsdp"
        )
        if caps is not None and callable(getattr(caps, "check", None)):
            try:
                reported = caps.check(str(stage.get("stage") or "sft"), tp=tp)
            except Exception:  # noqa: BLE001 - probing never decides a gate
                reported = None
            if reported:
                message = f"{message} (the installed FS reports: {reported})"
        out.append(("TR-IN-008", message))
    if cp is not None and cp > 1 and multimodal_rows:
        out.append((
            "TR-IN-008",
            f"cp={cp} is planned with image/video rows: --cp is measured end to end only on text "
            "(Qwen2.5-7B, #542), so image/video rows with cp>1 are unmeasured; plan cp=1 for image/video "
            "rows or keep the rows to text",
        ))

    # TR-PL-006/007/008: the sequence-length claims (planned value and goal claim)
    if longest is not None and longest >= _SEQ_LEN_CLAIM_CEILING:
        out.append((
            "TR-PL-006",
            f"{where}plans max_sequence_length {longest} >= 131072: a 128K claim is a planning fact, not a "
            "measurement (the VLM evidence base measures 16K/32K); the claim marks the handoff RED",
        ))
    if (
        longest is not None
        and longest > _SEQ_LEN_GEMMA4_31B_CEILING
        and any(tok in identity for tok in _GEMMA4_31B_ID_TOKENS)
    ):
        out.append((
            "TR-PL-007",
            f"{where}plans max_sequence_length {longest} > 32768 for gemma-4-31B-it (the measured evidence "
            "covers 32K at most); the stage marks the handoff RED",
        ))
    if (
        longest is not None
        and longest >= _SEQ_LEN_LIGER_FLOOR
        and str(fused_loss or "").lower() != _FUSED_LOSS_LIGER
    ):
        out.append((
            "TR-PL-008",
            f"{where}plans max_sequence_length {longest} >= 32768 without fused_loss 'liger': run with "
            "--fused-loss liger (FS's fused Liger loss is the measured 32K loss path) or plan below 32768",
        ))
    return out


def _open_questions(goal: dict) -> list[str]:
    """Missing load-bearing facts; prefers the intake module, else a builtin check."""
    try:
        from foundationskills.agent.intake import missing_facts

        facts = missing_facts(goal, ["training.planner"])
        return [f"{f['key']} — {f['question']}" for f in facts if f.get("load_bearing")]
    except Exception:
        questions: list[str] = []
        if not goal.get("goal") and not goal.get("goal_kind"):
            questions.append("goal — which training goal best matches (general_chat|reasoning|math|code|domain_expert|tool_calling)?")
        if not (goal.get("data_facts") or {}).get("has_verifiable_answers"):
            questions.append("data_facts.has_verifiable_answers — do examples have verifiable gold answers (enables RL)?")
        return questions


# ---------------------------------------------------------------------------
# the planner


def plan(
    goal: dict,
    *,
    caps: FSCapabilities,
    readiness: dict | None = None,
    hardware_id: str | None = None,
) -> dict:
    """Build a training_plan payload from a goal_spec payload.

    Raises PlanningRefusal (naming the missing input) when a load-bearing
    input is absent or stage selection yields nothing.
    """
    if not isinstance(goal, dict):
        raise PlanningRefusal("missing input: goal_spec payload")
    base_model = goal.get("base_model") or {}
    if not str(base_model.get("name_or_path") or "").strip():
        raise PlanningRefusal("missing input: goal.base_model.name_or_path")
    if caps is None:
        caps = FSCapabilities(available=False, fs_version=None)

    decisions: list[dict[str, str]] = []
    provenance_notes: list[str] = []

    def decide(step: str, choice: str, because: str) -> None:
        decisions.append({"step": step, "choice": str(choice), "because": str(because)})

    # 1. goal ---------------------------------------------------------------
    objective = str(goal.get("objective") or "")
    preserve_general = bool(goal.get("preserve_general", True))
    goal_kinds = _goal_kinds(goal)
    decide("goal", objective, f"goal kinds inferred: {', '.join(goal_kinds)}; preserve_general={preserve_general}")

    # 2. model analysis ------------------------------------------------------
    model_ref = str(base_model["name_or_path"])
    found = find_variant(model_ref)
    if found is not None:
        family, variant = found
        decide(
            "model_analysis",
            f"{_get(family, 'name', '?')}/{_get(variant, 'id', model_ref)}",
            f"matched {_get(variant, 'size_b', '?')}B {_get(variant, 'arch', '?')} variant by id/hf_id/local path/config",
        )
    else:
        family, variant, note = _synthesize_variant(base_model)
        provenance_notes.append(note)
        decide("model_analysis", model_ref, note)
    fs_family = _get(family, "fs_family")
    if fs_family is None:
        decide(
            "model_analysis",
            "explicit --adapter-target required for LoRA",
            "FS won't auto-target LoRA for this family; pass --adapter-target (family is not registered in FS)",
        )

    # 3. data analysis -------------------------------------------------------
    declared_facts = dict(goal.get("data_facts") or {})
    sources = [s for s in ((goal.get("data") or {}).get("sources") or []) if isinstance(s, dict)]
    for src in sources:  # rv13: a non-number is a named refusal, not a ValueError
        for key in ("approx_tokens", "approx_examples"):
            if src.get(key) is not None and (isinstance(src[key], bool) or not isinstance(src[key], (int, float))):
                raise PlanningRefusal(f"missing input: goal.data.sources[].{key} must be a number, got {src[key]!r}")
    tagged = any(src.get("use_for") for src in sources)

    def _sum_for(stage_name: str | None, key: str) -> int:
        """F8: sum only the sources tagged for a stage; untagged goals count all."""
        return sum(int(src.get(key) or 0) for src in sources
                   if stage_name is None or not tagged or stage_name in (src.get("use_for") or []))

    approx_tokens = _sum_for("cpt", "approx_tokens") if tagged else _sum_for(None, "approx_tokens")
    readiness_stats = (readiness or {}).get("stats") or {}
    domain_tokens = readiness_stats.get("num_tokens")
    domain_tokens = int(domain_tokens) if isinstance(domain_tokens, (int, float)) else int(approx_tokens)
    data_facts = {
        "domain_tokens": domain_tokens,
        "has_pairs": bool(declared_facts.get("has_pairs", False)),
        "has_verifiable_answers": bool(declared_facts.get("has_verifiable_answers", False)),
        "has_instructions": bool(
            declared_facts.get(
                "has_instructions",
                any(int(s.get("approx_examples") or 0) > 0 for s in sources if isinstance(s, dict))
                or any(k in {"general_chat", "domain_expert", "tool_calling"} for k in goal_kinds),
            )
        ),
        "requested_rl_algorithm": declared_facts.get("requested_rl_algorithm") or goal.get("rl_algorithm"),
        "requested_preference_algorithm": declared_facts.get("requested_preference_algorithm")
        or goal.get("preference_algorithm"),  # rv40: was dropped, silently replaced by "dpo"
        "rl_examples": declared_facts.get("rl_examples") or (_sum_for("rl", "approx_examples") or None),
        # the gold the RL data carries; FS verifies only single-letter MCQ gold
        "answer_kind": declared_facts.get("answer_kind")
        or next((src.get("answer_kind") for src in sources
                 if src.get("answer_kind") and (not tagged or "rl" in (src.get("use_for") or []))), None),
    }
    if not tagged and len(sources) > 1:
        decide("data_analysis", "sources not tagged with use_for",
               "all sources' examples/tokens counted for every stage; tag sources with use_for for per-stage sizing")
    if tagged and "sft_tokens" not in declared_facts and _sum_for("sft", "approx_tokens"):
        # F8: a source tagged for SFT states its own size; the domain/100 heuristic is for untagged goals
        data_facts["sft_tokens"] = _sum_for("sft", "approx_tokens")
    for key in ("sft_tokens", "rl_tokens", "rl_group_size", "rl_prompt_tokens", "rl_max_new_tokens"):
        if key in declared_facts:
            data_facts[key] = declared_facts[key]
    decide(
        "data_analysis",
        f"domain_tokens={domain_tokens:,}",
        "from the readiness report" if readiness_stats.get("num_tokens") else "from goal.data.sources approx_tokens (approximate)",
    )

    # 4. hardware analysis ---------------------------------------------------
    hw_map = load_hardware()
    goal_hw = goal.get("hardware") or {}
    hw, hw_resolved_id = _resolve_hardware(hw_map, hardware_id, goal_hw.get("gpu"))
    if hw is None:
        known = ", ".join(sorted(hw_map)) or "<none>"
        wanted = hardware_id or goal_hw.get("gpu") or "<unspecified>"
        raise PlanningRefusal(f"missing input: a hardware knowledge entry matching {wanted!r} (known ids: {known})")
    nodes = max(1, int(goal_hw.get("nodes") or 1))
    gpus_per_node = max(1, int(goal_hw.get("gpus_per_node") or int(_get(hw, "gpus_per_node", 1) or 1)))
    gpus = nodes * gpus_per_node
    decide(
        "hardware_analysis",
        f"{hw_resolved_id} x {gpus} GPU(s) ({nodes} node x {gpus_per_node})",
        f"goal.hardware.gpu {goal_hw.get('gpu')!r} mapped to knowledge id {hw_resolved_id!r} (fuzzy)",
    )

    # 5. stage selection ------------------------------------------------------
    rules = load_stage_rules()
    rule_ctx = {
        "stage_order": goal.get("stage_order"),
        "goal_kinds": goal_kinds,
        "domain_tokens": domain_tokens,
        "has_pairs": data_facts["has_pairs"],
        "has_verifiable_answers": data_facts["has_verifiable_answers"],
        "has_instructions": data_facts["has_instructions"],
        "base_is_instruct": bool(_get(variant, "instruct", False)),
        "preserve_general": preserve_general,
    }
    selected, fired = _evaluate_stage_rules(rules, rule_ctx)
    if goal.get("stages") is not None:
        selected = _apply_operator_stages(goal.get("stages"), selected, rules, fired, decide,
                                          stage_order=goal.get("stage_order"))
    if not selected:
        raise PlanningRefusal(
            "no stage selected: "
            f"{len(rules)} stage rules evaluated and none added a stage for "
            f"goal_kinds={goal_kinds}, domain_tokens={domain_tokens:,}, facts={data_facts}. "
            "Add a stage_rules.yaml entry or restate the goal/data facts."
        )
    decide(
        "stage_selection",
        " -> ".join(a["stage"] for a in selected),
        "; ".join(fired) if fired else "stage_rules.yaml",
    )
    if any(a["stage"] == "rl" for a in selected):
        decide(
            "hardware_analysis",
            "rl planned on 1 GPU",
            "FS RLTrainer is single-device; the RL stage gets 1 GPU regardless of cluster size",
        )

    # 6-10. per-stage: algorithm -> recipe -> method -> estimate -> feasibility
    # A4.1 (shared contract): the goal_spec's VLM declarations are copied into every
    # stage below (vh fields land in hparams.data/modalities).
    vlm_fields = vlm_stage_fields(goal)
    if (
        vlm_fields["hparams"]["fused_loss"] is not None
        or vlm_fields["hparams"]["lora_targets"] is not None
        or vlm_fields["hparams"]["max_sequence_length"] is not None
        or set(vlm_fields["modalities"]) - {"text"}
    ):
        decide(
            "hparams",
            "vlm fields: " + ", ".join(vlm_fields["modalities"]) or "none declared",
            "copied from the goal_spec (operator declaration): fused_loss="
            f"{vlm_fields['hparams']['fused_loss']!r}, lora_targets={vlm_fields['hparams']['lora_targets']!r}, "
            f"max_sequence_length={vlm_fields['hparams']['max_sequence_length']!r}, "
            f"fs_columns={vlm_fields['data']['fs_columns']!r}",
        )
    budget = goal.get("budget") or {}
    budget_gpu_hours = budget.get("gpu_hours")
    stages_out: list[dict] = []
    feas_verdicts: list[str] = []
    feas_findings: list[dict] = []
    feas_alternatives: list[dict] = []

    for sel in selected:
        stage = sel["stage"]
        gpus_for_stage = 1 if stage in ("rl", "preference") else gpus  # FS RL/Preference trainers are single-device
        assumptions: list[str] = []
        cpt: dict | None = None
        if stage == "cpt":
            try:
                cpt = cpt_policy(variant, domain_tokens, goal_kinds[0], preserve_general)
            except ValueError as exc:  # e.g. unknown model size: a named refusal, not a guess
                raise PlanningRefusal(str(exc)) from exc
            tokens = int(cpt["token_budget"])
        else:
            tokens = _stage_tokens(stage, data_facts, domain_tokens, assumptions)

        # algorithm
        algo_sel = select_algorithm(stage, goal, data_facts, caps)
        algorithm = algo_sel.get("algorithm")
        decide("algorithm_selection", f"{stage}: {algorithm}", algo_sel.get("because", ""))

        # recipe match
        # Method first, from goal/data/hardware rules; it then selects the recipe,
        # because recipe hparams are method-specific (a LoRA LR on a full FT is wrong).
        rule_choice = select_method(goal=goal_kinds[0], stage=stage, data_tokens=tokens, variant=variant,
                                    hardware=hw, gpus=gpus_for_stage, prefer=None)
        rule_method = str(_get(rule_choice, "method", "full"))
        if goal.get("method") is not None:
            # Operator override: the rules still run so the log says what they wanted and why.
            requested = goal.get("method")
            if requested not in ("full", "lora", "qlora"):
                raise PlanningRefusal(f"missing input: goal.method must be one of full|lora|qlora (got {requested!r})")
            forced = select_method(goal=goal_kinds[0], stage=stage, data_tokens=tokens, variant=variant,
                                   hardware=hw, gpus=gpus_for_stage, prefer=requested)
            forced_method = str(_get(forced, "method", requested))
            if forced_method != rule_method:
                decide("operator_override", f"{stage}: method {forced_method}",
                       f"goal.method={requested!r} overrides the rules, which chose {rule_method}: "
                       f"{'; '.join(_get(rule_choice, 'because', []) or [])}")
            rule_choice, rule_method = forced, forced_method
        adapter_ok = (caps.rl_adapter_support if stage == "rl" else
                      caps.pref_adapter_support if stage == "preference" else True)
        if stage in ("rl", "preference") and rule_method in ("lora", "qlora") and adapter_ok is not True:
            # FS's RL/Preference trainers have no adapter fields: they train the full
            # model (masters/optimizer on host). Planning LoRA there would be fiction.
            decide("method", f"{stage}: full (FS trainer has no adapter support)",
                   f"rules preferred {rule_method}, but FS {'RLTrainer' if stage == 'rl' else 'PreferenceTrainer'} "
                   f"config declares no adapter/LoRA field (measured): the full model trains, fp32 masters and "
                   f"optimizer state on the host")
            rule_choice = {"method": "full", "because": [f"FS {stage} trainer supports full fine-tuning only"],
                           "alternatives": []}
            rule_method = "full"
        query = RecipeQuery(
            family=_get(family, "name"),
            size_b=float(_get(variant, "size_b", 0.0) or 0.0),
            arch=_get(variant, "arch"),
            stage=stage,
            goal=goal_kinds[0],
            domain=goal.get("domain"),
            method=rule_method if rule_method in _METHODS else None,
            hardware=hw_resolved_id,
        )
        match = select_recipe(query)
        recipe_method_note = ""
        if match is None and query.method is not None:
            fallback = select_recipe(dataclasses.replace(query, method=None))
            if fallback is not None:
                fb_method = (_get(_get(fallback, "recipe"), "raw", {}).get("index") or {}).get("method")
                recipe_method_note = (f"no {rule_method} recipe; nearest recipe "
                                      f"{_get(_get(fallback, 'recipe'), 'id')} is {fb_method}, so its hparams are "
                                      f"NOT applied (algorithm-card defaults used)")
        ranked = rank_recipes(query, k=3)
        del ranked  # ranked is for diagnostics; the best match is authoritative
        recipe = _get(match, "recipe") if match is not None else None
        recipe_raw = _get(recipe, "raw", {}) or {}
        recipe_id = _get(recipe, "id") if recipe is not None else None
        recipe_derived = bool(_get(match, "derived", False)) if match is not None else True
        recipe_provenance = _get((recipe_raw.get("provenance") or {}), "status") if recipe_raw else None
        if recipe is None:
            recipe_reasons = "no recipe matched; algorithm-card defaults and planner heuristics (derived)"
            provenance_notes.append(f"stage {stage}: no recipe matched; using derived heuristics (not validated)")
        else:
            recipe_reasons = "; ".join(_get(match, "reasons", []) or []) or f"recipe {recipe_id}"
            status_txt = recipe_provenance or "unvalidated"
            provenance_notes.append(
                f"stage {stage}: recipe {recipe_id} provenance {status_txt}"
                + (" — not yet validated on FoundationScale" if status_txt != "validated" else "")
            )
            if recipe_derived:
                adjustments = "; ".join(_get(match, "adjustments", []) or [])
                provenance_notes.append(f"stage {stage}: recipe match is derived ({adjustments or 'soft keys differ'})")
        decide("recipe_match", f"{stage}: {recipe_id or 'none'}", recipe_reasons)

        # method (decided above, before the recipe)
        method_choice = rule_choice
        if recipe_method_note:
            provenance_notes.append(f"stage {stage}: {recipe_method_note}")
        method = str(_get(method_choice, "method", "full"))
        if method not in _METHODS:
            provenance_notes.append(f"stage {stage}: select_method returned {method!r}; clamped to 'full'")
            method = "full"
        method_because = "; ".join(str(b) for b in (_get(method_choice, "because", []) or [])) or method
        decide(
            "method",
            f"{stage}: {method}" + (" (1 GPU)" if stage == "rl" else ""),
            method_because,
        )

        # hparams: card defaults <- recipe stage hparams <- cpt policy overlay
        hparams = _card_default_hparams(algorithm)
        recipe_stage_algo = next((st.get("algorithm") for st in (recipe_raw.get("stages") or [])
                                  if isinstance(st, dict) and st.get("stage") == stage), None)
        if recipe_raw and recipe_stage_algo not in (None, algorithm):
            # hparams like beta/lr are algorithm-specific (DPO beta 0.1 vs SimPO ~2.0):
            # a recipe written for another algorithm contributes none of them
            provenance_notes.append(f"stage {stage}: recipe {recipe_id} is for {recipe_stage_algo}, not {algorithm}; "
                                    f"its hparams are NOT applied (algorithm-card defaults used)")
        else:
            hparams.update(_recipe_stage_hparams(recipe_raw, stage))
        if cpt is not None:
            cpt_lr = cpt["lr"]
            recipe_lr = _recipe_stage_hparams(recipe_raw, stage).get("learning_rate") if recipe_raw else None
            if method in ("lora", "qlora"):
                if recipe_lr is not None and (recipe_raw.get("index") or {}).get("method") == method:
                    cpt_lr = float(recipe_lr)
                    lr_because = f"{method} recipe {recipe_id} sets its own LR {cpt_lr:g}; the full-FT CPT table does not apply"
                else:
                    cpt_lr = float(f"{cpt['lr'] * LORA_LR_MULTIPLIER:.6g}")
                    lr_because = (f"CPT table LR {cpt['lr']:g} is full-FT; {method} trains best at "
                                  f"~{LORA_LR_MULTIPLIER:g}x (Biderman et al. 2024; Schulman et al. 2025)")
                decide("hparams", f"{stage}: learning_rate {cpt_lr:g}", lr_because)
            hparams.update(
                {
                    "learning_rate": cpt_lr,
                    "lr_scheduler_type": "cosine",
                    "warmup_ratio": cpt["schedule"]["warmup_ratio"],
                    "min_lr_ratio": cpt["schedule"]["min_lr_ratio"],
                    "schedule_note": cpt["schedule"]["note"],
                    "epochs_cap": cpt["epochs_cap"],
                    "replay_ratio": cpt["replay_ratio"],
                    "token_budget": cpt["token_budget"],
                }
            )
            # the CPT token budget IS the total; a recipe "epochs" alongside it would contradict it
            hparams.pop("epochs", None)
        # A4.1 (shared contract): the goal_spec's VLM declarations are copied into
        # every stage, and an operator declaration wins over recipe/card hparams. It
        # must land in hparams before max_sequence_length resolves below: the
        # estimate has to price exactly what is emitted (estimate == command).
        if vlm_fields["hparams"]["fused_loss"] is not None:
            hparams["fused_loss"] = vlm_fields["hparams"]["fused_loss"]
        else:
            hparams.setdefault("fused_loss", None)
        if vlm_fields["hparams"]["lora_targets"] is not None:
            hparams["lora_targets"] = list(vlm_fields["hparams"]["lora_targets"])
        else:
            hparams.setdefault("lora_targets", None)
        if vlm_fields["hparams"]["max_sequence_length"] is not None:
            hparams["max_sequence_length"] = int(vlm_fields["hparams"]["max_sequence_length"])
            for alias in _HP_ALIASES.get("max_sequence_length", ()):  # drop the recipe's spelling: one value
                hparams.pop(alias, None)
        else:
            hparams.setdefault("max_sequence_length", None)
        def _first(*keys: str, default: Any = None) -> Any:
            return next((hparams[k] for k in keys if hparams.get(k) is not None), default)

        # Multimodal rows at long context train at one row per device: that is the measured
        # VLM LoRA recipe (examples/vlm-2xH200/scripts/train_sft.sh:9,24 -- batch 1, grad-accum 8).
        # A text recipe's micro-batch would price (and OOM) several 16K image/video rows at once.
        if {"image", "video"} & set(vlm_fields["modalities"]) and int(_first(
                "max_sequence_length", "seq_len", "max_seq_len", default=0) or 0) >= 8192:
            for alias in ("micro_batch", "micro_batch_size"):
                hparams.pop(alias, None)
            hparams["per_device_batch_size"] = 1
            hparams.setdefault("gradient_accumulation_steps", 8)
            assumptions.append("vlm: per-device batch 1 (grad-accum "
                               f"{hparams['gradient_accumulation_steps']}) for image/video rows at long "
                               "context, as measured in examples/vlm-2xH200")

        seq_len = int(_first("max_sequence_length", "seq_len", "max_seq_len", default=4096))
        micro_batch = int(_first("per_device_batch_size", "micro_batch", "micro_batch_size", default=2))
        grad_ckpt = bool(_first("gradient_checkpointing", "grad_ckpt", default=True))
        sharding = "ddp" if stage == "rl" else str(_first("sharding_strategy", "sharding", default="fsdp"))
        variant_raw = dict(_get(variant, "raw", {}) or {})
        if grad_ckpt and variant_raw.get("fs_grad_ckpt") is False:
            # measured per-variant refusal (Gemma-4 E4B: KV-shared layers need the cache that
            # checkpointing disables); the estimate below then prices full activations honestly
            grad_ckpt = False
            hparams["gradient_checkpointing"] = False
            hparams.pop("grad_ckpt", None)
            decide("method", f"{stage}: gradient_checkpointing off",
                   f"FS refuses gradient checkpointing for {_get(variant, 'id')}: "
                   f"{variant_raw.get('fs_grad_ckpt_evidence', 'measured refusal')}")
        if sharding == "fsdp" and variant_raw.get("fs_fsdp") is False:
            # measured per-variant gap (e.g. Gemma-4 26B-A4B MoE: FS's FSDP auto-wrap
            # cannot find the layer class); DDP replicates, and feasibility below
            # prices that honestly (a full FT that does not fit becomes infeasible)
            sharding = "ddp"
            hparams["sharding_strategy"] = "ddp"
            hparams.pop("sharding", None)
            decide("method", f"{stage}: sharding ddp",
                   f"FS FSDP does not work for {_get(variant, 'id')}: "
                   f"{variant_raw.get('fs_fsdp_evidence', 'measured failure')}")
        if stage != "rl":
            # The emitted command must BE the configuration that was estimated:
            # FS defaults are seq 128, batch 1, no grad ckpt (measured on GB200: an
            # unstated grad_ckpt made a 32 GB plan run at 63 GB).
            for key, value in (("max_sequence_length", seq_len), ("per_device_batch_size", micro_batch),
                               ("gradient_checkpointing", grad_ckpt), ("sharding_strategy", sharding)):
                if not any(hparams.get(k) is not None for k in (key, *_HP_ALIASES.get(key, ()))):
                    hparams[key] = value
        if (stage == "rl" and goal.get("rl_min_measured_fraction") is not None
                and hparams.get("min_measured_fraction") is None):  # an operator stage value wins
            # An operator-declared verdict floor, never a planner guess: the RL driver
            # exits 95 when fewer steps measure (zero-variance groups are UNMEASURED in FS).
            hparams["min_measured_fraction"] = float(goal["rl_min_measured_fraction"])
            decide("hparams", f"{stage}: min_measured_fraction {hparams['min_measured_fraction']:g}",
                   "declared by goal.rl_min_measured_fraction")

        # estimate
        mem = estimate_memory(
            variant,
            method=method,
            seq_len=seq_len,
            micro_batch=micro_batch,
            sharding=sharding if sharding in ("ddp", "fsdp") else "fsdp",
            grad_ckpt=grad_ckpt,
            world=gpus_for_stage,
            lora_rank=int(_first("lora_rank", "lora_r", default=16)),
            reference_copy=(stage == "preference" and "reference_model" in _card_requires(algorithm)),
            optimizer="host_adamw" if stage in ("rl", "preference") else "adamw",
        )
        processed = tokens
        if stage == "sft":
            processed = _sft_processed_tokens(tokens, hparams, readiness_stats, micro_batch, assumptions)
        time_kwargs: dict[str, Any] = {"tokens": processed, "hardware": hw, "gpus": gpus_for_stage, "method": method,
                                       "stage": stage, "sharding": sharding, "micro_batch": micro_batch,
                                       "grad_ckpt": grad_ckpt}
        te = estimate_time(variant, **time_kwargs)
        estimate = {
            "tokens": tokens,
            "tokens_processed": processed,
            "gpus": gpus_for_stage,
            "total_flops": _get(te, "total_flops"),
            "tokens_per_s_per_gpu": _get(te, "tokens_per_s_per_gpu"),
            "hours": _get(te, "hours"),
            "gpu_hours": _get(te, "gpu_hours"),
            "mfu": _get(te, "mfu"),
            "mfu_provenance": _get(te, "mfu_provenance"),
            "memory_gb_per_gpu": _get(mem, "total_per_gpu_gb"),
            "assumptions": [*(list(_get(mem, "assumptions", []) or [])), *(list(_get(te, "assumptions", []) or [])), *assumptions],
        }
        gpu_hours = float(_get(te, "gpu_hours", 0.0) or 0.0)
        decide("estimate", f"{stage}: {gpu_hours:.1f} GPU-h over {tokens:,} tokens", "; ".join(assumptions) or "6*N*D flops model; hardware MFU")

        # feasibility
        feas = check_feasibility(
            variant,
            hw,
            gpus=gpus_for_stage,
            method=method,
            seq_len=seq_len,
            micro_batch=micro_batch,
            tokens=processed,
            budget_gpu_hours=budget_gpu_hours,
            deadline_hours=None,
            caps=caps,
            # price the configuration that is emitted (found 2026-10-07: a hard-coded fsdp and the
            # default grad_ckpt=True judged a no-checkpointing E4B plan at 77.7 GB instead of 138 GB,
            # and would price a measured-ddp variant as if FSDP sharded it)
            sharding=sharding if sharding in ("ddp", "fsdp") else "fsdp",
            grad_ckpt=grad_ckpt,
            tp=1,
            stage=stage,
            mem_extra={
                "lora_rank": int(_first("lora_rank", "lora_r", default=16)),
                "reference_copy": stage == "preference" and "reference_model" in _card_requires(algorithm),
                "optimizer": "host_adamw" if stage in ("rl", "preference") else "adamw",
            },
        )
        stage_feas = {
            "verdict": _get(feas, "verdict", "warn"),
            "findings": list(_get(feas, "findings", []) or []),
            "alternatives": list(_get(feas, "alternatives", []) or []),
        }
        feas_verdicts.append(str(stage_feas["verdict"]))
        for f in stage_feas["findings"]:
            if isinstance(f, dict):
                feas_findings.append({"stage": stage, **f})
        # executable from measured capabilities only
        if stage in ("preference", "rl"):
            answer_kind = data_facts.get("answer_kind") if stage == "rl" else None
            missing = caps.check(stage, algorithm=algorithm, multi_gpu_rl=(stage == "rl" and gpus_for_stage > 1),
                                 answer_kind=answer_kind)
            if missing and "checkpoint" in missing and caps.check(
                    stage, algorithm=algorithm, answer_kind=answer_kind, require_checkpoint=False) is None:
                decide("executability", f"{stage}: measurement-only",
                       f"{algorithm} runs on the installed FS, but FS RLTrainer persists no trained policy, so the "
                       f"stage can be run to measure rewards/throughput and cannot hand weights to a later stage "
                       f"(core gap: RL checkpoint persistence)")
        else:
            missing = caps.check(stage, backend="fsdp", tp=1)
        if missing is None and algo_sel.get("missing"):
            missing = algo_sel["missing"]
        executable = missing is None
        stage_feas["alternatives"] = _gate_alternatives(stage_feas["alternatives"], missing)
        for a in stage_feas["alternatives"]:
            if isinstance(a, dict):
                feas_alternatives.append({"stage": stage, **a})

        # data handoff
        data_format = _FORMAT_BY_STAGE.get(stage, stage)
        ready = bool(
            readiness
            and readiness.get("verdict") == "PASS"
            and (readiness.get("format") in (None, data_format))
        )
        data_entry: dict[str, Any] = {
            "format": data_format,
            "ready": ready,
            "tokens": tokens,
            # A4.1 (shared contract): video_column / video_frames / overlong travel
            # with the dataset's declared fs_columns so training.emit can set the
            # FOUNDATIONSCALE_TRAIN_* environment from one place.
            "fs_columns": dict(vlm_fields["data"]["fs_columns"]),
        }
        if not ready:
            data_entry["handoff"] = "data_engine"
            data_entry["target_format"] = data_format

        because_parts = [
            sel.get("because") or f"stage rule {sel.get('rule_id', '?')}",
            str(algo_sel.get("because", "")),
            f"recipe: {recipe_reasons}",
            f"method: {method_because}",
        ]
        if not executable:
            because_parts.append(f"not executable on the installed FS: {missing}")

        stages_out.append(
            {
                "name": str(sel.get("name") or stage),
                "stage": stage,
                "algorithm": algorithm,
                "method": method,
                "modalities": list(vlm_fields["modalities"]),
                "recipe_id": recipe_id,
                "because": " | ".join(p for p in because_parts if p),
                "hparams": hparams,
                "data": data_entry,
                "executable": bool(executable),
                "missing": missing,
                "estimate": estimate,
                "recipe_provenance": recipe_provenance,
                "recipe_derived": recipe_derived,
                "algorithm_runnable": bool(algo_sel.get("runnable", False)),
                "algorithm_fallback": algo_sel.get("fallback"),
                "feasibility": stage_feas,
            }
        )
        decide(
            "feasibility",
            f"{stage}: {stage_feas['verdict']} ({'executable' if executable else 'missing: ' + str(missing)[:80]})",
            f"memory {estimate['memory_gb_per_gpu']}GB/GPU estimate; {len(stage_feas['alternatives'])} alternative(s) considered",
        )

    overall = max(feas_verdicts, key=lambda v: _VERDICT_ORDER.get(v, 1)) if feas_verdicts else "warn"
    decide(
        "feasibility",
        f"overall: {overall}",
        f"worst of {len(feas_verdicts)} stage verdict(s); alternatives aggregated per stage",
    )

    return {
        "goal": dict(goal),
        "stages": stages_out,
        "decisions": decisions,
        "feasibility": {"verdict": overall, "findings": feas_findings, "alternatives": feas_alternatives},
        "provenance_notes": provenance_notes,
        "open_questions": _open_questions(goal),
        "data_facts": data_facts,
        "goal_kinds": goal_kinds,
        "model": {
            "ref": model_ref,
            "family": _get(family, "name"),
            "fs_family": fs_family,
            "variant_id": _get(variant, "id"),
            "size_b": _get(variant, "size_b"),
            "arch": _get(variant, "arch"),
            "instruct": bool(_get(variant, "instruct", False)),
            "in_knowledge_base": found is not None,
        },
        "hardware": {"id": hw_resolved_id, "nodes": nodes, "gpus_per_node": gpus_per_node, "gpus": gpus},
    }


__all__ = [
    "plan",
    "PlanningRefusal",
    "PLANNING_STEPS",
    "vlm_gates",
    "vlm_goal_fields",
    "vlm_stage_fields",
]
