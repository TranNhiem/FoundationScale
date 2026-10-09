"""``difficulty_filter`` op: offline pass-rate band filter for GRPO-style RL (Polaris / Skywork-OR1 practice).

WHY (measured): GRPO on gemma-4-E4B-it / ARC-Challenge measured only 5 of 32 steps -- the base policy already
solves most prompts, so most groups come out all-correct: reward variance 0, advantage 0, and FS marks the
step UNMEASURED. A group can only teach where the policy is not yet deterministic, so this op keeps only
prompts whose k-sample BASE-policy pass rate is strictly inside a declared band (default ``0 < p < 1``).

- The prompt is built by FS's own parser -- ``foundationscale.rl.corpus._parse_record(record, index,
  gold_key=...)`` -- so the sample is byte-identical to what ``RLTrainer`` trains on; an offline filter that
  re-wrote the prompt would measure a different distribution from the one RL fills. A record that will not
  parse (``BatchRefusal`` or anything else out of ``_parse_record``) drops as ``difficulty_unparseable_record``;
  a Sample whose gold is ``None`` drops as ``difficulty_no_gold`` (nothing verifiable to score against).
- k completions per prompt come from ``_ROLLOUT_FACTORY(cfg)`` (default backend ``hf-generate``: FS's
  ``resolve_prompt_surface`` + ``AutoModelForCausalLM`` with the ``AutoModelForImageTextToText`` fallback
  RLTrainer uses, left padding, and the RL sampling surface -- ``do_sample``/``temperature``/``top_p``/
  ``top_k=0`` -- so the measured pass rate is the pass rate RL will train on). The factory is called ONCE per
  invocation, and only when at least one prompt is measurable -- never per record.
- Scoring reuses ``foundationscale.rl.rewards.MCQLetterReward``: 1.0 / 0.0 / ``None`` (abstention). Per prompt
  ``correct`` counts the 1.0 rows and ``scored`` the non-``None`` rows -- an unparseable completion is
  UNMEASURED, not wrong, so it leaves the denominator instead of biasing it. ``scored < 2`` drops the prompt as
  ``difficulty_unscorable``: with fewer than two scored completions the reward variance this op exists to
  measure cannot be measured at all. ``pass_rate = correct / scored``, kept iff
  ``keep_above < pass_rate < keep_below``, else ``difficulty_too_easy`` (``pass_rate >= keep_below``) or
  ``difficulty_too_hard`` (``pass_rate <= keep_above``).
- Kept records are copies, in input order, gaining ``meta["pass_rate"]`` (rounded 4), ``meta["pass_scored"]``
  and ``meta["pass_k"]``; every other meta key survives untouched and the record is otherwise unchanged.
  A record whose ``meta`` is present but not an object is dropped (``difficulty_meta_not_object``) before rollouts.

Drops: ``difficulty_unparseable_record``, ``difficulty_no_gold``, ``difficulty_unscorable``,
``difficulty_too_easy``, ``difficulty_too_hard``. Everything else is a measurement, never a removal.

Refusals (``OpUnavailable``, skill DE-IN-008): an empty band (``keep_above >= keep_below``, naming both
values), ``foundationscale`` not importable, torch/transformers not importable, no CUDA under ``device: auto``
(k-sample generation on CPU is not a supported runtime), and both auto model classes failing to load. FS
refusals (``_refuse_exit_96`` -> ``SystemExit(96)``) are re-raised as ``OpUnavailable`` so the skill reports
DE-IN-008 instead of exiting the process.

``stats.backend``: ``hf-generate:<model>@<revision>`` (revision from :func:`_resolve_revision`; "unknown" when
it cannot be read offline -- never guessed).

``stats.extra``: ``model``, ``model_revision``, ``k``, ``temperature``, ``top_p``, ``max_new_tokens``,
``keep_above``, ``keep_below``, ``seed``, ``rollouts`` (completions generated), ``abstentions`` (``None``
scores), ``prompts_scored`` (prompts with ``scored >= 2``), ``pass_rate_hist`` (count of scored prompts per
``f"{pass_rate:.3f}"`` key, sorted by key; sums to ``prompts_scored``), ``kept_fraction`` (kept /
prompts_scored, or ``None`` when nothing scored) and ``wall_s``.
"""
from __future__ import annotations

import os
import time
from collections import Counter
from typing import Any, Callable, Iterable, Iterator

from foundationskills.skills.data_engine.ops.base import (
    FunctionOp,
    OpStats,
    OpUnavailable,
    counted,
    register_op,
)
from foundationskills.skills.data_engine.ops.semantic_dedup import _resolve_revision, _spec_problem


CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,  # unknown keys refused: a silently ignored key changes the measurement
    "required": ["model"],
    "properties": {
        "model": {"type": "string", "minLength": 1},
        "k": {"type": "integer", "minimum": 2, "default": 8},
        "temperature": {"type": "number", "exclusiveMinimum": 0.0, "default": 1.0},
        "top_p": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0, "default": 1.0},
        "max_new_tokens": {"type": "integer", "minimum": 1, "default": 64},  # FS RLTrainConfig default
        "keep_above": {"type": "number", "minimum": 0.0, "exclusiveMaximum": 1.0, "default": 0.0},
        "keep_below": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0, "default": 1.0},
        "batch_size": {"type": "integer", "minimum": 1, "default": 8},  # prompts per generate call
        "seed": {"type": "integer", "default": 0},
        "device": {"type": "string", "default": "auto"},
        "dtype": {"enum": ["bfloat16", "float16", "float32"], "default": "bfloat16"},
        "gold_key": {"type": "string", "default": "answer"},
        "answer_pattern": {"type": ["string", "null"], "default": None},
    },
}


def _matches_kind(value: Any, kind: str) -> bool:
    """One naive JSON-schema type check (bool is not an integer/number here)."""
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list)
    return isinstance(value, dict)


def _value_problems(value: Any, spec: dict[str, Any]) -> list[str]:
    """Why one cfg value fails one property spec (the JSON-schema subset this op declares)."""
    declared = spec.get("type")
    kinds = (declared,) if isinstance(declared, str) else tuple(declared or ())
    if kinds and not any(_matches_kind(value, kind) for kind in kinds):
        return [f"{value!r} is not {' or '.join(kinds)}"]
    problems: list[str] = []
    if "enum" in spec and value not in spec["enum"]:
        problems.append(f"{value!r} is not one of {spec['enum']}")
    for bound, ok, what in (
        ("minimum", lambda got, want: got >= want, "below the minimum"),
        ("exclusiveMinimum", lambda got, want: got > want, "not above the exclusive minimum"),
        ("maximum", lambda got, want: got <= want, "above the maximum"),
        ("exclusiveMaximum", lambda got, want: got < want, "not below the exclusive maximum"),
    ):
        if bound in spec and isinstance(value, (int, float)) and not isinstance(value, bool):
            if not ok(value, spec[bound]):
                problems.append(f"{value!r} is {what} {spec[bound]}")
    if "minLength" in spec and isinstance(value, str) and len(value) < int(spec["minLength"]):
        problems.append(f"{value!r} is shorter than minLength {spec['minLength']}")
    return problems


def _schema_problems(cfg: dict, schema: dict[str, Any] = CONFIG_SCHEMA) -> list[str]:
    """Cfg values that do not satisfy ``schema`` (unknown keys included), as printable problems."""
    problems: list[str] = []
    properties = dict(schema.get("properties", {}))
    for key in cfg:
        if key not in properties:
            problems.append(f"unknown key {key!r}")
    for key in schema.get("required", []):
        if key not in cfg:
            problems.append(f"missing required key {key!r}")
    for key, spec in properties.items():
        if key in cfg:
            problems.extend(f"{key!r}: {item}" for item in _value_problems(cfg[key], spec))
    return problems


def _load_rollout(cfg: dict) -> Callable[[list], list[list[str]]]:
    """Real backend (overridden by tests through :data:`_ROLLOUT_FACTORY`): k sampled HF ``generate`` rows."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    except ImportError as exc:
        raise OpUnavailable(f"difficulty_filter: torch/transformers are not importable ({exc})") from exc

    model_id = str(cfg.get("model", ""))
    k = int(cfg.get("k", 8))
    batch_size = int(cfg.get("batch_size", 8))
    max_new_tokens = int(cfg.get("max_new_tokens", 64))
    temperature = float(cfg.get("temperature", 1.0))
    top_p = float(cfg.get("top_p", 1.0))
    device = str(cfg.get("device", "auto"))
    if device == "auto":
        if not torch.cuda.is_available():
            raise OpUnavailable(
                "difficulty_filter: no CUDA device; k-sample generation on CPU is not a supported runtime "
                "(set device: cpu to force it)"
            )
        device = "cuda"

    try:
        from foundationscale.rl.prompt_surface import encode_prompts, resolve_prompt_surface
    except Exception as exc:
        raise OpUnavailable(f"difficulty_filter: foundationscale.rl.prompt_surface is not importable ({exc})") from exc
    try:
        surface = resolve_prompt_surface(model_id, needs_images=False)
    except SystemExit as exc:  # _refuse_exit_96 must report DE-IN-008 here, never end the process
        raise OpUnavailable(f"difficulty_filter: FS refused: {exc}") from exc
    except Exception as exc:
        raise OpUnavailable(f"difficulty_filter: prompt surface for {model_id!r} is unavailable ({exc})") from exc

    tokenizer = surface.surface if surface.kind == "tokenizer" else surface.surface.tokenizer
    tokenizer.padding_side = "left"  # RLTrainer pads left before generate: conditioning must match training
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)

    dtype = getattr(torch, str(cfg.get("dtype", "bfloat16")), torch.bfloat16)
    loaded = None
    try:
        loaded = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except Exception:
        try:
            loaded = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
        except Exception as exc:  # noqa: BLE001 - the two-class fallback IS the contract
            raise OpUnavailable(
                f"difficulty_filter: model load failed for {model_id!r} under both auto classes ({exc})"
            ) from exc
    model = loaded.to(device)
    model.eval()
    torch.manual_seed(int(cfg.get("seed", 0)))  # once at load: every sampled draw follows it

    def rollout(samples: list) -> list[list[str]]:
        groups: list[list[str]] = [[] for _ in samples]
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            try:
                with torch.no_grad():
                    ids = encode_prompts(surface, chunk, device)
                    out = model.generate(
                        **ids,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=k,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=0,  # sentinel: never inherit the checkpoint's cutoff
                        pad_token_id=tokenizer.pad_token_id,
                    )
            except SystemExit as exc:  # an FS refusal inside encode/generate becomes DE-IN-008
                raise OpUnavailable(f"difficulty_filter: FS refused: {exc}") from exc
            prompt_width = ids["input_ids"].shape[1]
            texts = tokenizer.batch_decode(out[:, prompt_width:], skip_special_tokens=True)
            for offset in range(len(chunk)):  # rows are prompt-major: k completions per prompt, in draw order
                row = offset * k
                groups[start + offset] = list(texts[row:row + k])
        return groups

    return rollout


_ROLLOUT_FACTORY: Callable[[dict], Callable[[list], list[list[str]]]] = _load_rollout


def _difficulty_filter_op(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    model = str(cfg.get("model", ""))
    k = int(cfg.get("k", 8))
    temperature = float(cfg.get("temperature", 1.0))
    top_p = float(cfg.get("top_p", 1.0))
    max_new_tokens = int(cfg.get("max_new_tokens", 64))
    keep_above = float(cfg.get("keep_above", 0.0))
    keep_below = float(cfg.get("keep_below", 1.0))
    seed = int(cfg.get("seed", 0))
    gold_key = str(cfg.get("gold_key", "answer"))
    if keep_above >= keep_below:
        raise OpUnavailable(
            f"difficulty_filter: keep_above ({keep_above}) >= keep_below ({keep_below}): the band is empty and "
            "every measured prompt would drop"
        )

    # FS owns both the prompt and the score; imported lazily so this module reads without it
    try:
        from foundationscale.rl.corpus import _parse_record
        from foundationscale.rl.rewards import MCQLetterReward
    except SystemExit as exc:  # _refuse_exit_96 must report DE-IN-008 here, never end the process
        raise OpUnavailable(f"difficulty_filter: FS refused: {exc}") from exc
    except Exception as exc:
        raise OpUnavailable(f"difficulty_filter: foundationscale is not importable ({exc})") from exc

    started = time.monotonic()
    pending: list[tuple[dict, Any]] = []
    for index, rec in enumerate(counted(records, stats)):
        try:
            sample = _parse_record(rec, index, gold_key=gold_key)
        except Exception:  # BatchRefusal or anything else: this record cannot become a Sample
            stats.drop("difficulty_unparseable_record")
            continue
        if sample.gold is None:
            stats.drop("difficulty_no_gold")  # nothing verifiable to score against
            continue
        if rec.get("meta") is not None and not isinstance(rec.get("meta"), dict):
            stats.drop("difficulty_meta_not_object")  # pass_rate cannot be added without losing the field
            continue
        pending.append((rec, sample))

    groups: list[list[str]] = []
    if pending:  # no measurable prompt -> no backend work (semantic_dedup's empty-entry idiom)
        groups = _ROLLOUT_FACTORY(cfg)([sample for _rec, sample in pending])
        if len(groups) != len(pending):  # a short backend answer must not pass as unscorable prompts
            raise OpUnavailable(
                f"difficulty_filter: rollout backend returned {len(groups)} completion group(s) for "
                f"{len(pending)} prompt(s)"
            )
    scorer = MCQLetterReward(answer_pattern=(cfg.get("answer_pattern") or None))

    rollouts = 0
    abstentions = 0
    prompts_scored = 0
    histogram: Counter[str] = Counter()
    kept_records: list[tuple[dict, float, int]] = []
    for index, (rec, sample) in enumerate(pending):
        completions = groups[index]
        rollouts += len(completions)
        correct = 0
        scored = 0
        for response in completions:
            value = scorer.score(response=response, gold=sample.gold)
            if value is None:
                abstentions += 1  # UNMEASURED: leaves the denominator, never counted as wrong
                continue
            scored += 1
            if value == 1.0:
                correct += 1
        if scored < 2:
            stats.drop("difficulty_unscorable")  # reward variance is unmeasurable on <2 scored completions
            continue
        pass_rate = correct / scored
        prompts_scored += 1
        histogram[f"{pass_rate:.3f}"] += 1  # the input distribution is a measurement, kept or dropped
        if pass_rate >= keep_below:
            stats.drop("difficulty_too_easy")
            continue
        if pass_rate <= keep_above:
            stats.drop("difficulty_too_hard")
            continue
        kept_records.append((rec, pass_rate, scored))

    kept = len(kept_records)
    revision = _resolve_revision(model)  # offline snapshot read; "unknown" when none can be read
    stats.backend = f"hf-generate:{model}@{revision}"
    stats.extra.update(
        {
            "model": model,
            "model_revision": revision,
            "k": k,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "keep_above": keep_above,
            "keep_below": keep_below,
            "seed": seed,
            "rollouts": rollouts,
            "abstentions": abstentions,
            "prompts_scored": prompts_scored,
            "pass_rate_hist": {key: histogram[key] for key in sorted(histogram)},
            "kept_fraction": (kept / prompts_scored) if prompts_scored else None,
            "wall_s": round(time.monotonic() - started, 2),
        }
    )

    for rec, pass_rate, scored in kept_records:  # input order, copies only
        out_rec = dict(rec)
        meta = dict(rec.get("meta")) if isinstance(rec.get("meta"), dict) else {}
        meta.update({"pass_rate": round(pass_rate, 4), "pass_scored": scored, "pass_k": k})
        out_rec["meta"] = meta
        stats.records_out += 1
        yield out_rec


def preflight(cfg: dict) -> list[str]:
    """Cheap, offline, deterministic checks for this cfg: schema, band, imports and local model resolution."""
    problems = _schema_problems(cfg)
    above, below = cfg.get("keep_above", 0.0), cfg.get("keep_below", 1.0)
    if isinstance(above, (int, float)) and isinstance(below, (int, float)) and float(above) >= float(below):
        problems.append(f"keep_above ({above}) >= keep_below ({below}): the band is empty")
    for name in ("foundationscale", "torch", "transformers"):
        missing = _spec_problem(name)  # the import probe semantic_dedup's preflight uses (find_spec, offline)
        if missing:
            problems.append(missing)
    model = str(cfg.get("model", ""))
    if not os.path.isdir(model):
        resolved = False
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(model, local_files_only=True)  # cache lookup only: preflight never downloads
            resolved = True
        except Exception:
            resolved = False
        if not resolved:
            problems.append(
                f"{model}: not a directory and absent from the local HF cache (preflight never downloads)"
            )
    return problems


register_op(FunctionOp("difficulty_filter", _difficulty_filter_op, CONFIG_SCHEMA))
