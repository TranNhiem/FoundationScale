
"""Recommend a ``data_pipeline_spec`` from formats, source kinds, goal and domain.

Silent-spec choices (documented because ops from other tasks interpret these
configs): op configs use self-describing keys — ``clean.pii_mode``,
``dedup.level``/``dedup.near_dedup``, ``quality.ruleset``,
``tokenize.pack``/``tokenize.tokenizer``, ``format.target_format`` (+ format
-specific keys). Ops are expected to tolerate keys they don't use. For
multi-source pretrain/CPT a single ``inline`` mix component pools the upstream
records deterministically; token-weighted ratios require measurements and
belong to ``design_mixture`` with explicit components.
"""
from __future__ import annotations

from typing import Any

from foundationskills.core.schema import assert_valid, load_schema

FORMATS = ("pretrain", "cpt", "sft", "preference", "rl", "mm_sft")


def recommend_pipeline(
    *,
    target_format: str,
    sources: list[dict],
    goal: str | None,
    algorithm: str | None,
    tokenizer: str | None,
    chat_template_family: str | None,
    domain: str | None,
    benchmarks: list[str],
    seq_len: int = 4096,
) -> dict[str, Any]:
    """Build a data_pipeline_spec payload (with a rationale) for the target format."""
    if target_format not in FORMATS:
        raise ValueError(f"unknown target_format {target_format!r}; known: {sorted(FORMATS)}")

    ops: list[dict[str, Any]] = []
    rationale: list[str] = []

    def add(name: str, config: dict[str, Any], why: str) -> None:
        ops.append({"op": name, "config": config})
        rationale.append(f"{name}: {why}")

    # 1. ingest
    add(
        "ingest",
        {"sources": list(sources)},
        f"read {len(sources)} source(s) into canonical records (kind-driven readers)",
    )

    pretrain_like = target_format in {"pretrain", "cpt"}
    sft_like = target_format in {"sft", "mm_sft"}
    preference_like = target_format == "preference"
    rl_like = target_format == "rl"

    # 2. clean
    if pretrain_like:
        add(
            "clean",
            {"pii_mode": "strict", "doc_cleaning": True},
            "strict PII redaction plus boilerplate/document cleaning for web-derived pretraining text",
        )
    else:
        add(
            "clean",
            {"pii_mode": "strict"},
            "strict PII redaction on human/model conversations before anything trains on them",
        )

    # 3. dedup
    if pretrain_like:
        add(
            "dedup",
            {"level": "document", "near_dedup": True},
            "document-level exact + near dedup; repeated boilerplate wastes tokens",
        )
    else:
        add(
            "dedup",
            {"level": "sample", "near_dedup": False},
            "sample-level exact dedup so no example is over-weighted",
        )

    # 4. quality
    if pretrain_like:
        add(
            "quality",
            {"ruleset": "gopher_fineweb"},
            "gopher/fineweb-style web-quality heuristics and cutoffs",
        )
    elif sft_like:
        add(
            "quality",
            {"ruleset": "sft_checks"},
            "SFT structural checks (roles, alternation, empty answers) and length sanity",
        )
    elif preference_like:
        add(
            "quality",
            {"ruleset": "preference_checks"},
            "preference pair sanity: identical chosen/rejected and empty responses are dropped",
        )
    else:
        add(
            "quality",
            {"ruleset": "rl_checks"},
            "RL prompt checks: a parseable gold answer (gold key) must exist per record",
        )

    # 5. decontam
    if benchmarks:
        add(
            "decontam",
            {"benchmarks": list(benchmarks)},
            f"remove near-duplicates of the evaluation benchmarks ({', '.join(benchmarks)}) so eval stays honest",
        )
    else:
        rationale.append(
            "decontam: skipped (no benchmarks declared); add `benchmarks` to get decontamination"
        )

    # 6. mix (multi-source pretraining-like only)
    if pretrain_like and len(sources) > 1:
        add(
            "mix",
            {
                "components": [{"name": "pooled", "ratio": 1.0, "inline": True}],
                "seed": 0,
                "max_epochs": 4,
            },
            "pool all sources deterministically (uniform); use data_engine.mix.design_mixture "
            "with explicit path components once per-source token measurements justify ratios",
        )

    # 7. format
    format_cfg: dict[str, Any] = {"target_format": target_format}
    if target_format in {"sft", "mm_sft"}:
        format_cfg["chat_template_family"] = chat_template_family
        if tokenizer:
            # the model's own chat_template beats the builtin approximation
            format_cfg["tokenizer"] = tokenizer
    if rl_like:
        format_cfg["gold_key"] = "answer"
    add("format", format_cfg, f"render records into the FS-consumed {target_format!r} rows")

    # 8. tokenize
    pack = pretrain_like
    add(
        "tokenize",
        {"pack": pack, "chunk": pretrain_like, "tokenizer": tokenizer, "seq_len": int(seq_len)},
        (
            f"measure token counts and seq-length stats (tokenizer={tokenizer!r}, seq_len={int(seq_len)}); "
            + ("chunk documents longer than seq_len at paragraph boundaries (FS truncates at max_sequence_length, "
               "so an unchunked long document loses its tail) and pack for throughput"
               if pack else "no packing: example boundaries matter post-pretraining")
        ),
    )

    spec: dict[str, Any] = {
        "target_format": target_format,
        "ops": ops,
        "seed": 0,
        "tokenizer": tokenizer,
        "rationale": rationale,
        "chat_template_family": chat_template_family,
        "goal": goal,
        "algorithm": algorithm,
        "domain": domain,
        "recommended": True,
    }
    assert_valid(spec, load_schema("artifacts/data_pipeline_spec"), "recommended data_pipeline_spec")
    return spec
