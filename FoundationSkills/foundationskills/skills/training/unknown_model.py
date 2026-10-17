"""training.planner fallback for a model the knowledge base does not list.

The architecture is MEASURED from the checkpoint's config.json (``text_config`` first, for VLMs)
or STATED in ``goal.base_model.architecture`` (flagged unvalidated); the size always comes from
``base_model.size_b``.
Nothing is guessed or borrowed from a neighbouring variant: with neither source the planner refuses.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from foundationskills.skills.training.knowledge import Family, Variant, load_families


def config_facts(path: str) -> dict[str, Any] | None:
    """Architecture facts read from ``<path>/config.json`` (``text_config`` first, for VLMs);
    None when unreadable
    or when a fact every estimate needs is absent."""
    try:
        cfg = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    text = cfg["text_config"] if isinstance(cfg.get("text_config"), dict) else cfg

    def pick(*keys: str) -> Any:
        for key in keys:
            for source in (text, cfg):
                if source.get(key) is not None:
                    return source[key]
        return None

    facts = {
        "hidden": pick("hidden_size", "d_model"),
        "layers": pick("num_hidden_layers", "n_layer"),
        "heads": pick("num_attention_heads", "n_head"),
        "kv_heads": pick("num_key_value_heads"),
        "head_dim": pick("head_dim"),
        "vocab": pick("vocab_size"),
        "context_length": pick("max_position_embeddings", "max_sequence_length"),
        "num_experts": pick("num_experts", "num_local_experts"),
        "experts_per_token": pick("num_experts_per_tok"),
        "tied": bool(pick("tie_word_embeddings")),
        "model_type": cfg.get("model_type"),
        "vision": isinstance(cfg.get("vision_config"), dict),
    }
    required = ("hidden", "layers", "heads", "vocab", "context_length")
    if any(not isinstance(facts[k], int) or isinstance(facts[k], bool) for k in required):
        return None
    if facts["kv_heads"] is None:
        facts["kv_heads"] = facts["heads"]
    return facts


def stated_architecture(stated: Any) -> dict[str, Any] | None:
    """``goal.base_model.architecture`` as stated facts (the fields config.json must provide)."""
    if not isinstance(stated, dict):
        return None
    facts = {
        key: stated.get(key)
        for key in (
            "hidden",
            "layers",
            "heads",
            "kv_heads",
            "head_dim",
            "vocab",
            "context_length",
            "num_experts",
            "experts_per_token",
        )
    }
    if any(
        not isinstance(facts[k], int) or isinstance(facts[k], bool)
        for k in ("hidden", "layers", "heads", "vocab", "context_length")
    ):
        return None
    facts["kv_heads"] = facts["kv_heads"] or facts["heads"]
    facts.update(
        tied=bool(stated.get("tied_embeddings")), model_type=stated.get("model_type"), vision=False
    )
    return facts


def synthesize_variant(base_model: dict) -> tuple[Any, Any, str | None]:
    """A model the knowledge base does not list: a real Variant from the goal's ``size_b`` plus
    the architecture MEASURED from the checkpoint's config.json, inside the KB family when the
    family (or the config's
    model_type) is known. Refuses rather than plan on guessed dimensions."""
    from foundationskills.skills.training.planner import (
        PlanningRefusal,  # noqa: PLC0415 - planner imports us
    )

    size_b = base_model.get("size_b")
    if size_b is None:
        raise PlanningRefusal(
            "missing input: base_model.size_b (model not found in the knowledge base; "
            "state the parameter count in billions so memory/time can be estimated)"
        )
    size_b = float(size_b)
    ref = str(base_model.get("name_or_path"))
    facts = config_facts(ref)
    source = "config.json"
    if facts is None:
        facts = stated_architecture(base_model.get("architecture"))
        source = "goal.base_model.architecture (stated, unvalidated)"
    if facts is None:
        raise PlanningRefusal(
            f"missing input: base_model architecture ({ref!r} is not in the knowledge base and "
            "has no readable config.json): point name_or_path at the local checkpoint directory, "
            "or state base_model.architecture "
            "{hidden, layers, heads, kv_heads, vocab, context_length}"
        )
    arch = str(base_model.get("arch") or ("moe" if facts["num_experts"] else "dense"))
    active_b = base_model.get("active_b")
    variant = Variant(
        id=ref,
        hf_id=None,
        local_path=ref,
        size_b=size_b,
        active_b=float(active_b) if active_b is not None else (size_b if arch == "dense" else None),
        arch=arch,
        hidden=facts["hidden"],
        layers=facts["layers"],
        heads=facts["heads"],
        kv_heads=facts["kv_heads"],
        head_dim=facts["head_dim"],
        vocab=facts["vocab"],
        context_length=facts["context_length"],
        num_experts=facts["num_experts"],
        experts_per_token=facts["experts_per_token"],
        tied_embeddings=facts["tied"],
        instruct=False,
        raw={"source": f"goal.base_model.size_b + {source}", "config": f"{ref}/config.json"},
    )
    families = load_families()
    wanted = str(base_model.get("family") or "")
    known = families.get(wanted) or next(
        (
            f
            for f in families.values()
            if facts["model_type"] and facts["model_type"] in f.model_types
        ),
        None,
    )
    if known is not None:
        family = dataclasses.replace(known, variants=(*known.variants, variant))
        where = f"KB family {known.name!r}"
    else:
        family = Family(
            name=wanted or "unknown",
            display_name=ref,
            fs_family=None,
            model_types=(str(facts["model_type"]),) if facts["model_type"] else (),
            modalities=("text", "image") if facts["vision"] else ("text",),
            variants=(variant,),
            tokenizer={},
            chat_template={},
            special_tokens={},
            lora_targets=(),
            quirks=(),
            provenance={"source": "synthesized from goal.base_model + config.json (unvalidated)"},
            raw={"name": wanted or "unknown"},
        )
        where = "no KB family (FS will need explicit --adapter-target for LoRA)"
    note = (
        f"model {ref!r} not in the knowledge base: size_b {size_b} from the goal, architecture "
        f"(hidden {facts['hidden']}, layers {facts['layers']}, "
        f"heads {facts['heads']}/{facts['kv_heads']}, "
        f"vocab {facts['vocab']}, context {facts['context_length']}) from {source}; {where}"
    )
    return family, variant, note


def linear_attention_note(model_ref: str, variant: Any, caps: Any) -> str | None:
    """A disclosure when the model declares linear-attention layers and the probe measured the
    fast-path kernels missing: FS then trains on the torch fallback, several times slower than the
    time estimate."""
    path = getattr(variant, "local_path", None) or model_ref
    try:
        cfg = json.loads((Path(str(path)) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    if "linear_attention" not in (text.get("layer_types") or cfg.get("layer_types") or []):
        return None
    key = (getattr(caps, "probe_keys", None) or {}).get("linear_attention_kernels") or {}
    if key.get("status") != "measured" or all((key.get("value") or {}).values()):
        return None
    return (
        "model declares linear-attention layers and the probe measured "
        + str(key.get("reason"))
        + ": FS trains on the torch fallback (measured 2026-10-17: Qwen3.5-4B SFT about 29 s/step "
        "at 11% GPU utilisation); time estimates assume the fast path, so expect a several-fold "
        "longer run or install "
        "flash-linear-attention and causal-conv1d"
    )
