
"""Format op: convert canonical records into the exact FS-ready output shapes.

Output record shapes (this is precisely what downstream FS consumes):

  pretrain/cpt : {"id", "text", "meta"}
  sft          : {"id", "messages", "text", "meta"}   # text = rendered chat
  mm_sft       : sft + {"image": <images[0]>}
  preference   : {"id", "prompt", "chosen", "rejected", "meta"}
  rl           : ShareGPT, i.e. {"id", "conversations": [{"from": "human"|"gpt",
                 "value": str}, ...], "meta", <gold_key>: <answer str>} plus an
                 optional "system" string. This matches
                 ``foundationscale.rl.corpus._parse_record`` exactly: the final
                 turn is the "gpt" reply the policy trains toward, and the gold
                 is DECLARED under ``gold_key`` (default "answer").

Deliberate choices where the spec is silent:

- The RL gold is written as a STRING verbatim. FS's declared-gold reader
  (``_declared_gold``) refuses a non-string value and keeps only a single A-Z
  letter, so free-text answers yield ``Sample.gold is None`` -- FS's normal,
  auditable abstention state -- and never an exception.
- ``cfg["tokenizer"]`` may be a path/name string (loaded lazily via
  transformers.AutoTokenizer) or an already-constructed tokenizer-like object
  (used directly; this keeps the op testable without transformers installed).
  Any tokenizer load/apply failure falls back to builtin templates and is
  recorded in ``stats.extra["tokenizer_error"]``; a failure is never silently
  treated as success.
- A record that cannot convert is dropped with reason
  ``unconvertible:<target>`` in strict mode (default), or passed through
  UNCHANGED in non-strict mode, counted in
  ``stats.modified["unconvertible_passthrough"]``.
- For sft/mm_sft the op always sets ``stats.extra["sft_loss_scope"] =
  "full_sequence"``: FS's sft objective trains on the full rendered text with
  NO assistant-only loss masking (measured, FS 77bfa65). This must be
  disclosed downstream (see report.DE-RDY-010).
- The op records ``stats.extra["gold_key"]`` for rl and
  ``stats.extra["fs_columns"]`` hints for every format so the pipeline layer
  can fill the dataset payload's fs_columns.
"""
from __future__ import annotations

from typing import Callable, Any, Iterable, Iterator

from foundationskills.skills.data_engine.ops.base import FunctionOp, OpStats, counted, register_op

__all__ = ["format_op", "render_chat", "TARGET_FORMATS"]

TARGET_FORMATS = ("pretrain", "cpt", "sft", "mm_sft", "preference", "rl")


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------


def _render_gemma4(messages: list[dict[str, Any]], *, thinking: bool = False) -> str:
    """Gemma-4 builtin template, copied from the real gemma-4-E4B-it
    chat_template output (2026-09-26): ``<|turn>{role}\n...<turn|>\n`` with the
    assistant role spelled ``model``. (The earlier builtin emitted Gemma-2/3
    ``<start_of_turn>`` markers -- wrong for this family.) No ``<bos>``: FS's
    tokenizer call adds it. ``thinking`` mirrors enable_thinking=True, which the
    real template renders as a ``<|think|>`` system prefix."""
    system = "\n\n".join(str(m["content"]) for m in messages if m["role"] == "system")
    parts: list[str] = []
    if thinking or system:
        parts.append(f"<|turn>system\n{'<|think|>' + chr(10) if thinking else ''}{system}<turn|>\n")
    body = [m for m in messages if m["role"] != "system"]
    for m in body:
        role = "model" if m["role"] == "assistant" else "user"
        parts.append(f"<|turn>{role}\n{m['content']}<turn|>\n")
    if not body or body[-1]["role"] != "assistant":
        parts.append("<|turn>model\n")
    return "".join(parts)


def _render_chatml(messages: list[dict[str, str]]) -> str:
    """ChatML template (FS family qwen3.5 and the generic 'chatml' tag)."""
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    if not messages or messages[-1]["role"] != "assistant":
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _render_generic(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n"


def _normalise_family(family: str | None) -> str | None:
    if not family:
        return None
    tag = str(family).lower().replace("_", ".").replace("-", ".")
    if "gemma4" in tag or "gemma" in tag:
        return "gemma4"
    if "qwen3.5" in tag or "qwen3" in tag:
        return "qwen3.5"
    if "chatml" in tag:
        return "chatml"
    return None


def _render_internal(
    messages: list[dict[str, str]],
    *,
    family: str | None,
    tokenizer: Any = None,
) -> tuple[str, str, str | None]:
    """Render and report (text, source, error). source in {"tokenizer",
    "gemma4", "qwen3.5", "chatml", "generic"}; error is set when a tokenizer
    was given but applying its chat template failed (builtin was used instead).
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            if isinstance(text, str):
                return text, "tokenizer", None
        except Exception as exc:  # noqa: BLE001 - recorded, builtin used
            error = f"apply_chat_template failed: {type(exc).__name__}: {exc}"
        else:
            error = "apply_chat_template returned a non-string"
    else:
        error = None
    builtin = _normalise_family(family)
    if builtin == "gemma4":
        return _render_gemma4(messages), "gemma4", error
    if builtin == "qwen3.5":
        return _render_chatml(messages), "qwen3.5", error
    if builtin == "chatml":
        return _render_chatml(messages), "chatml", error
    return _render_generic(messages), "generic", error


def _strip_bos(text: str, tokenizer: Any) -> str:
    """FS tokenizes the text column with special tokens on, which prepends BOS;
    a template that already wrote ``<bos>`` would give every example two."""
    bos = getattr(tokenizer, "bos_token", None) if tokenizer is not None else None
    for token in filter(None, (bos, "<bos>")):
        if isinstance(token, str) and text.startswith(token):
            return text[len(token):]
    return text


def _inject_gemma4(messages: list[dict[str, Any]], tokenizer: Any) -> str:
    """Gemma-4 renders a trace as ``<|channel>thought\n...\n<channel|>`` only on
    tool-call turns and strips it from content, so render content-only in
    thinking mode and insert the trace after the LAST ``<|turn>model\n``."""
    trace = next((m.get("reasoning_content") for m in reversed(messages)
                  if m.get("role") == "assistant" and m.get("reasoning_content")), None)
    plain = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(plain, tokenize=False, enable_thinking=True)
    else:
        text = _render_gemma4(plain, thinking=True)
    marker = "<|turn>model\n"
    at = text.rfind(marker)
    if trace is None or at < 0:
        return text
    at += len(marker)
    return text[:at] + f"<|channel>thought\n{trace}\n<channel|>" + text[at:]


REASONING_INJECTORS: dict[str, Callable[[list[dict[str, Any]], Any], str]] = {"gemma4": _inject_gemma4}


def render_with_reasoning(messages: list[dict[str, Any]], *, family: str | None,
                          tokenizer: Any = None) -> tuple[str, str]:
    """Render, then VERIFY every reasoning trace survived. Returns (text, mode),
    mode in {"none", "native", "injected", "inline_fallback", "lost"}.

    Templates disagree: measured, Qwen3.5 renders reasoning_content natively as
    ``<think>``; Gemma-4 silently drops it. "lost" means a tokenizer template
    dropped the trace and no injector exists -- the op drops such records."""
    traces = [m["reasoning_content"] for m in messages
              if m.get("role") == "assistant" and isinstance(m.get("reasoning_content"), str)]
    if not traces:
        return render_chat(messages, family=family, tokenizer=tokenizer), "none"
    has_template = tokenizer is not None and getattr(tokenizer, "chat_template", None)
    if has_template:
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            if isinstance(text, str) and all(t in text for t in traces):
                return text, "native"
        except Exception:  # noqa: BLE001 - fall through to the injector
            pass
    injector = REASONING_INJECTORS.get(_normalise_family(family) or "")
    if injector is not None:
        text = injector(messages, tokenizer if has_template else None)
        if all(t in text for t in traces):
            return text, "injected"
    if has_template:
        return render_chat(messages, family=family, tokenizer=tokenizer), "lost"
    inline = [dict(m) for m in messages]
    for m in inline:
        if m.get("role") == "assistant" and m.get("reasoning_content"):
            m["content"] = f"<think>\n{m.pop('reasoning_content')}\n</think>\n\n{m['content']}"
    return render_chat(inline, family=family, tokenizer=None), "inline_fallback"


def render_chat(messages: list[dict[str, str]], *, family: str | None, tokenizer: Any = None) -> str:
    """Render a conversation. Uses ``tokenizer.apply_chat_template`` when a
    tokenizer object with a chat_template is given; otherwise the builtin
    template for family "gemma4"/"qwen3.5"/"chatml"; otherwise a generic
    ``ROLE: text`` fallback (the op records template_fallback for that case).
    """
    text, _source, _error = _render_internal(messages, family=family, tokenizer=tokenizer)
    return text


# ---------------------------------------------------------------------------
# Record conversion helpers
# ---------------------------------------------------------------------------

_ROLES = ("system", "user", "assistant")


def _valid_messages(rec: dict) -> list[dict[str, str]] | None:
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    out: list[dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role, content = m.get("role"), m.get("content")
        if role not in _ROLES or not isinstance(content, str):
            return None
        turn = {"role": role, "content": content}
        if role == "assistant" and isinstance(m.get("reasoning_content"), str) and m["reasoning_content"].strip():
            turn["reasoning_content"] = m["reasoning_content"]  # rendered + verified downstream
        out.append(turn)
    return out


def _first_str(rec: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = rec.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _messages_for_sft(rec: dict) -> list[dict[str, str]] | None:
    """messages, Alpaca (instruction/input/output), or prompt+answer -> messages."""
    msgs = _valid_messages(rec)
    if msgs is not None:
        return msgs
    instruction = rec.get("instruction")
    if isinstance(instruction, str) and instruction:
        extra_input = rec.get("input")
        content = instruction
        if isinstance(extra_input, str) and extra_input.strip():
            content = f"{instruction}\n\n{extra_input}"
        response = _first_str(rec, ("output", "answer", "response", "chosen"))
        if response is None:
            return None
        msgs = [{"role": "user", "content": content}, {"role": "assistant", "content": response}]
        system = rec.get("system")
        if isinstance(system, str) and system:
            msgs.insert(0, {"role": "system", "content": system})
        return msgs
    prompt = _first_str(rec, ("prompt", "question"))
    answer = _first_str(rec, ("answer", "output", "response"))
    if prompt is not None and answer is not None:
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    return None


def _as_sharegpt(rec: dict) -> tuple[list[dict[str, str]], str | None, str] | None:
    """Build (conversations, system|None, final_answer) accepted by FS's parser."""
    msgs = _valid_messages(rec)
    if msgs is not None:
        system: str | None = None
        turns: list[dict[str, str]] = []
        for m in msgs:
            if m["role"] == "system":
                system = m["content"] if system is None else f"{system}\n\n{m['content']}"
            else:
                turns.append({"from": "human" if m["role"] == "user" else "gpt", "value": m["content"]})
        if not turns or turns[-1]["from"] != "gpt":
            return None
        return turns, system, turns[-1]["value"]
    prompt = _first_str(rec, ("prompt", "question"))
    answer = _first_str(rec, ("answer", "output", "response"))
    if prompt is None or answer is None:
        return None
    system = rec.get("system")
    turns = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
    return turns, system if isinstance(system, str) else None, answer


def _load_tokenizer(path: str) -> tuple[Any | None, str | None]:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        return None, f"transformers unavailable: {type(exc).__name__}: {exc}"
    try:
        return AutoTokenizer.from_pretrained(path), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _convert(
    rec: dict,
    target: str,
    *,
    family: str | None,
    tokenizer: Any,
    system_prompt: str | None,
    gold_key: str,
    image_column: str,
    index: int,
) -> tuple[dict, str | None, str | None] | None:
    """Return (out_record, template_source, template_error) or None if unconvertible."""
    meta = dict(rec.get("meta") or {})
    raw_id = rec.get("id")
    rid = str(raw_id) if raw_id is not None else f"rec-{index}"

    if target in ("pretrain", "cpt"):
        text = rec.get("text")
        if not isinstance(text, str):
            return None
        return {"id": rid, "text": text, "meta": meta}, None, None

    if target in ("sft", "mm_sft"):
        messages = _messages_for_sft(rec)
        if messages is None:
            return None
        if system_prompt and not (messages and messages[0]["role"] == "system"):
            messages = [{"role": "system", "content": system_prompt}, *messages]
        text, source, error = _render_internal(messages, family=family, tokenizer=tokenizer)
        mode = "none"
        if any(m.get("reasoning_content") for m in messages):
            text, mode = render_with_reasoning(messages, family=family, tokenizer=tokenizer)
        text = _strip_bos(text, tokenizer)
        meta["reasoning_render"] = mode
        out: dict[str, Any] = {"id": rid, "messages": messages, "text": text, "meta": meta}
        if target == "mm_sft":
            image: str | None = None
            images = rec.get("images")
            if isinstance(images, list) and images and isinstance(images[0], str):
                image = images[0]
            elif isinstance(rec.get("image"), str):
                image = rec["image"]
            if image is None:
                return None
            out[image_column] = image
        return out, source, error

    if target == "preference":
        prompt, chosen, rejected = rec.get("prompt"), rec.get("chosen"), rec.get("rejected")
        if not all(isinstance(v, str) and v for v in (prompt, chosen, rejected)):
            return None
        return {"id": rid, "prompt": prompt, "chosen": chosen, "rejected": rejected, "meta": meta}, None, None

    if target == "rl" and isinstance(rec.get("choices"), list) and rec.get("choices"):
        # FS RL verifies ONLY a single-letter gold on a question that ends with
        # "Answer with a single letter." (foundationscale.rl.corpus).
        stem = _first_str(rec, ("prompt", "question"))
        letter = str(rec.get("answer") or "").strip()
        if stem is None or not (len(letter) == 1 and "A" <= letter <= "Z"):
            return None
        lines = [f"{c.get('label')}. {c.get('text')}" for c in rec["choices"] if isinstance(c, dict)]
        question = f"{stem}\n" + "\n".join(lines) + "\nAnswer with a single letter."
        out = {"id": rid, "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": letter}],
               "meta": meta, gold_key: letter}
        if system_prompt:
            out["system"] = system_prompt
        return out, None, None

    if target == "rl":
        built = _as_sharegpt(rec)
        if built is None:
            return None
        turns, system, answer = built
        if system_prompt and system is None:
            system = system_prompt
        out = {"id": rid, "conversations": turns, "meta": meta}
        if system:
            out["system"] = system
        gold = (answer if isinstance(answer, str) else str(answer)).strip()
        if not (len(gold) == 1 and "A" <= gold.upper() <= "Z"):
            return None  # free-form gold: FS RL cannot verify it (dropped with a named reason)
        out[gold_key] = gold.upper()
        return out, None, None

    return None


_FS_COLUMN_HINTS = {
    "pretrain": {"text_column": "text", "image_column": None, "gold_key": None},
    "cpt": {"text_column": "text", "image_column": None, "gold_key": None},
    "sft": {"text_column": "text", "image_column": None, "gold_key": None},
    "preference": {"text_column": None, "image_column": None, "gold_key": None},
}

_SFT_LOSS_DISCLOSURE = (
    "FS trains the sft objective on the FULL rendered text with no "
    "assistant-only loss masking (measured, FS commit 77bfa65)."
)


def _format_records(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    target = cfg.get("target_format")
    if target not in TARGET_FORMATS:
        raise ValueError(f"format op: unknown target_format {target!r}; expected one of {TARGET_FORMATS}")
    family = cfg.get("chat_template_family")
    system_prompt = cfg.get("system_prompt")
    gold_key = cfg.get("gold_key") or "answer"
    image_column = cfg.get("image_column") or "image"
    strict = bool(cfg.get("strict", True))

    tokenizer: Any = None
    tok_spec = cfg.get("tokenizer")
    if target in ("sft", "mm_sft") and tok_spec:
        if isinstance(tok_spec, str):
            tokenizer, error = _load_tokenizer(tok_spec)
            if error:
                stats.extra["tokenizer_error"] = error
        else:
            tokenizer = tok_spec  # an already-constructed tokenizer-like object

    if target in ("sft", "mm_sft"):
        stats.extra["sft_loss_scope"] = "full_sequence"
        stats.extra["sft_loss_disclosure"] = _SFT_LOSS_DISCLOSURE
    if target == "rl":
        stats.extra["gold_key"] = gold_key
        stats.extra["rl_reward_kind"] = "mcq_letter"
    hints = dict(_FS_COLUMN_HINTS.get(target, {"text_column": None, "image_column": None, "gold_key": None}))
    if target == "mm_sft":
        hints["text_column"], hints["image_column"] = "text", image_column
    if target == "rl":
        hints["gold_key"] = gold_key
    stats.extra["fs_columns"] = hints

    template_sources: dict[str, int] = {}
    for rec in counted(records, stats):
        index = stats.records_in - 1
        if not isinstance(rec, dict):
            if strict:
                stats.drop(f"unconvertible:{target}")
                continue
            stats.modified["unconvertible_passthrough"] += 1
            stats.records_out += 1
            yield rec
            continue
        converted = _convert(
            rec,
            target,
            family=family,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            gold_key=gold_key,
            image_column=image_column,
            index=index,
        )
        if converted is None:
            if strict:
                if target == "rl" and (_first_str(rec, ("answer", "output", "response")) is not None
                                       or rec.get("choices")):
                    stats.drop("rl_answer_not_mcq_letter")  # FS RL rewards only single-letter MCQ gold
                else:
                    stats.drop(f"unconvertible:{target}")
                continue
            stats.modified["unconvertible_passthrough"] += 1
            stats.records_out += 1
            yield rec
            continue
        out, source, error = converted
        mode = (out.get("meta") or {}).get("reasoning_render") if isinstance(out, dict) else None
        if mode and mode != "none":
            modes = stats.extra.setdefault("reasoning_render", {})
            modes[mode] = modes.get(mode, 0) + 1
            if mode == "lost":
                stats.drop("reasoning_lost_in_template")  # never keep a reasoning record without its trace
                continue
        if error:
            stats.extra.setdefault("tokenizer_error", error)
        if source:
            template_sources[source] = template_sources.get(source, 0) + 1
            if source == "generic":
                stats.extra["template_fallback"] = True
        stats.records_out += 1
        yield out
    if template_sources:
        stats.extra["template_sources"] = template_sources
        stats.extra["template_source"] = max(template_sources, key=template_sources.get)


_CONFIG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,  # unknown keys refused: a silently ignored key measures nothing
    "required": ["target_format"],
    "additionalProperties": True,
    "properties": {
        "target_format": {"enum": list(TARGET_FORMATS)},
        "chat_template_family": {"type": ["string", "null"]},
        "tokenizer": {"type": ["string", "null"]},
        "system_prompt": {"type": ["string", "null"]},
        "gold_key": {"type": "string", "minLength": 1},
        "image_column": {"type": "string", "minLength": 1},
        "strict": {"type": "boolean"},
    },
}

format_op = register_op(FunctionOp("format", _format_records, _CONFIG_SCHEMA))
