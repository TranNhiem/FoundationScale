"""Op ``llm_enhance``: LLM-based data enhancement (WRAP/Nemotron-CC rephrasing,
document-grounded QA synthesis, verifiable reasoning traces, and LLM-as-judge
filtering) via a registered backend.

Config::

    {"backend": {...},          # REQUIRED; passed to make_backend (endpoint or kind)
     "mode": str,               # REQUIRED: rephrase | qa_synth | reasoning_trace | judge
     "field": str,              # default "text": source field of the document modes
     "max_chars": int,          # default 8000; longer inputs truncated (input_truncated)
     "workers": int,            # default 4; concurrent backend calls, order preserved
     "max_calls": int | null,   # record budget; later records pass through unenhanced
     "temperature": float,      # default 0.0 (0.7 for rephrase when unset)
     "max_tokens": int,         # default 2048
     "seed": int,               # default 0
     "min_overlap": float,      # default 0.5; grounding guard (measured, see below)
     # rephrase only:
     "style": str,              # wikipedia | textbook | qa | plain (default wikipedia)
     "emit": str,               # append | replace (default append)
     # qa_synth only:
     "n_pairs": int,            # default 3
     # reasoning_trace only:
     "answer_kind": str,        # exact | mcq_letter (default exact)
     "keep_unverified": bool,   # default false
     # judge only:
     "rubric": str,             # default helpfulness/correctness/harmlessness rubric
     "min_score": int,          # default 3 (1..5)
     "drop_inconsistent": bool} # default false (pairwise judge)

Grounding guard: ``_overlap(candidate, source)`` is the fraction of the
candidate's lowercase word tokens (``[a-z0-9]+``, length >= 3, minus a small
English stopword set) that also occur in the source's token set; an empty
candidate token set scores 0.0. Every synthetic output carries its measured
overlap in meta, and the mean over all measured candidates (kept and dropped)
lands in ``stats.extra["overlap_mean"]``. Nothing is ever guessed: scores,
verdicts and labels that could not be obtained are None and counted as
unmeasured.

Simplifications (documented per the silent-spec rule):

* The call budget is charged once per input record, not per HTTP call; a
  pairwise judge record spends one slot although it issues two backend calls.
* ``max_chars`` cuts at a raw character boundary; no tokenizer is involved.
* All OpStats mutation happens in the caller thread. Worker threads only build
  ``_Outcome`` values; the main loop applies drops/counters in input order, so
  accounting stays deterministic with ``workers`` > 1.
* A judge verdict that cannot be obtained (backend error or unparseable JSON)
  is unmeasured: pointwise records are kept with ``judge_score`` None; pairwise
  records are kept with ``judge_agrees`` None, counted in ``extra["unmeasured"]>``,
  and are never dropped as ``judge_inconsistent`` -- a verdict is never guessed.
* Document modes with a missing/blank source field drop as ``no_text``; judge on
  a record with neither messages nor text also drops as ``no_text``.
* ``qa_synth`` converts documents and never re-emits the source record; backend
  failures drop as ``llm_error`` and unparseable payloads as ``llm_invalid_json``.
* Records the LLM read are emitted as shallow copies with the provenance entry
  appended to ``meta["llm"]`` (existing entries are kept, never overwritten).
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from foundationskills.skills.data_engine.llm_backend import (
    LLMBackend,
    LLMOpError,
    LLMResponse,
    budget,
    make_backend,
    map_ordered,
    parse_json_object,
    prompt_hash,
)
from foundationskills.skills.data_engine.ops.base import (
    FunctionOp,
    OpStats,
    counted,
    register_op,
)

_NAME = "llm_enhance"
_MODES = ("rephrase", "qa_synth", "reasoning_trace", "judge")
_STYLES = ("wikipedia", "textbook", "qa", "plain")
_WORD_RE = re.compile(r"[a-z0-9]+")
_ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*:\s*(.+?)\s*$")
_MCQ_LETTER_RE = re.compile(r"\b([A-Za-z])\b")

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "are", "was",
        "were", "have", "has", "had", "not", "but", "its", "his", "her",
        "she", "him", "they", "them", "their", "you", "your", "our",
        "ours", "can", "could", "will", "would", "shall", "should", "may",
        "might", "must", "about", "into", "over", "under", "between",
        "through", "during", "after", "before", "because", "while", "where",
        "when", "which", "what", "who", "whom", "how", "why", "than",
        "then", "there", "here", "these", "those", "such", "each", "any",
        "all", "both", "more", "most", "other", "some", "only", "also",
        "very", "just", "like", "being", "been",
    }
)

_DEFAULT_RUBRIC = (
    "Score the response for helpfulness, correctness and harmlessness. "
    "Penalise wrong facts, evasiveness, and unsafe or offensive content."
)

_STYLE_PHRASES = {
    "wikipedia": "a Wikipedia encyclopedia article",
    "textbook": "a clear textbook chapter",
    "qa": "a question-and-answer explanation",
    "plain": "plain, simple language",
}

_REPHRASE_TEMPLATE = (
    "Rewrite the following document in the style of {style}. Preserve every "
    "fact from the original and add nothing new. Output only the rewritten "
    "document.\n\nDocument:\n{text}\n\nRewrite:"
)
_QA_TEMPLATE = (
    "You are creating instruction-tuning data grounded ONLY in the document "
    "below. Write up to {n_pairs} question/answer pairs that can be answered "
    "solely from the document. Reply with exactly one JSON object of the form "
    '{{"pairs": [{{"question": "...", "answer": "..."}}]}}.'
    "\n\nDocument:\n{text}"
)
_TRACE_TEMPLATE = (
    "Solve the following problem. Reason step by step, then conclude with a "
    "final line of the form 'Answer: <answer>'.\n\nProblem:\n{problem}"
)
_JUDGE_POINT_TEMPLATE = (
    "You are grading a piece of content with this rubric:\n{rubric}\n\n"
    "Content:\n{content}\n\nReply with exactly one JSON object: "
    '{{"score": <integer 1-5>, "reason": "short reason"}}.'
)
_JUDGE_PAIR_TEMPLATE = (
    "Two candidate responses follow. Choose the better one according to this "
    "rubric:\n{rubric}\n\nPrompt:\n{prompt}\n\nResponse A:\n{a}\n\n"
    "Response B:\n{b}\n\nReply with exactly one JSON object: "
    '{{"winner": "A"}} or {{"winner": "B"}}.'
)


@dataclass(frozen=True)
class _Ctx:
    backend: LLMBackend
    mode: str
    field: str
    max_chars: int
    temperature: float
    max_tokens: int
    seed: int
    min_overlap: float
    ph: str


@dataclass
class _Outcome:
    """Per-record result built on a worker thread, applied on the main loop."""

    outputs: list[dict] = field(default_factory=list)
    drops: list[str] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    modified: Counter = field(default_factory=Counter)
    overlaps: list[float] = field(default_factory=list)
    budget_skip: bool = False


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _WORD_RE.findall(text.lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _overlap(candidate: str, source: str) -> float:
    """Measured grounding: share of candidate content tokens present in source."""
    cand = _content_tokens(candidate)
    if not cand:
        return 0.0
    return len(cand & _content_tokens(source)) / len(cand)


def _prov(ctx: _Ctx, cache_hit: bool) -> dict:
    return {
        "op": _NAME,
        "mode": ctx.mode,
        "model": ctx.backend.model,
        "backend": ctx.backend.kind,
        "prompt_hash": ctx.ph,
        "cache_hit": bool(cache_hit),
    }


def _meta_with(rec: dict, additions: dict, provs: list[dict]) -> dict:
    meta = dict(rec.get("meta") or {})
    seen = list(meta.get("llm") or [])
    seen.extend(provs)
    meta.update(additions)
    meta["llm"] = seen
    return meta


def _call(
    ctx: _Ctx, messages: list[dict], out: _Outcome, *, json_mode: bool = False
) -> LLMResponse:
    try:
        resp = ctx.backend.complete(
            messages,
            temperature=ctx.temperature,
            max_tokens=ctx.max_tokens,
            seed=ctx.seed,
            json_mode=json_mode,
        )
    except Exception as exc:  # a raising backend is an error, not a crash
        resp = LLMResponse(
            content=None,
            finish_reason=None,
            prompt_tokens=0,
            completion_tokens=0,
            cache_hit=False,
            error=f"backend raised: {exc}",
        )
    out.counts["calls"] += 1
    out.counts["prompt_tokens"] += max(int(resp.prompt_tokens), 0)
    out.counts["completion_tokens"] += max(int(resp.completion_tokens), 0)
    if resp.cache_hit:
        out.counts["cache_hits"] += 1
    if resp.error is not None:
        out.counts["errors"] += 1
    return resp


def _truncate(text: str, max_chars: int, out: _Outcome) -> str:
    if len(text) > max_chars:
        out.counts["input_truncated"] += 1
        return text[:max_chars]
    return text


def _doc_text(rec: dict, ctx: _Ctx, out: _Outcome) -> str:
    raw = rec.get(ctx.field)
    if not isinstance(raw, str) or not raw.strip():
        return ""
    return _truncate(raw, ctx.max_chars, out)


def _rephrase_one(rec: dict, cfg: dict, ctx: _Ctx) -> _Outcome:
    out = _Outcome()
    style = str(cfg.get("style") or "wikipedia")
    emit = str(cfg.get("emit") or "append")
    src = _doc_text(rec, ctx, out)
    if not src:
        out.drops.append("no_text")
        return out
    prompt = _REPHRASE_TEMPLATE.format(
        style=_STYLE_PHRASES.get(style, style), text=src
    )
    resp = _call(ctx, [{"role": "user", "content": prompt}], out)
    provs = [_prov(ctx, resp.cache_hit)]
    source_id = str(rec.get("id", ""))
    original = dict(rec)  # the document was read by the LLM -> provenance
    original["meta"] = _meta_with(rec, {}, provs)
    content = (resp.content or "").strip()

    def _fallback() -> None:  # append: keep original; replace: fall back to it
        out.outputs.append(original)
        if emit == "replace":
            out.counts["fallback_original"] += 1

    if resp.error is not None or not content:
        out.drops.append("llm_error")
        _fallback()
        return out
    ov = _overlap(content, src)
    out.overlaps.append(ov)
    if ov < ctx.min_overlap:
        out.drops.append("ungrounded")
        _fallback()
        return out
    synth = {
        "id": f"{source_id}#rephrase-{style}",
        "text": content,
        "meta": _meta_with(
            rec,
            {"synthetic": True, "source_id": source_id, "overlap": ov},
            provs,
        ),
    }
    if emit == "append":
        out.outputs.extend([original, synth])
    else:
        out.outputs.append(synth)
    out.modified["rephrased"] += 1
    out.modified["llm_provenance"] += len(out.outputs)
    return out


def _qa_one(rec: dict, cfg: dict, ctx: _Ctx) -> _Outcome:
    out = _Outcome()
    src = _doc_text(rec, ctx, out)
    if not src:
        out.drops.append("no_text")
        return out
    n_pairs = int(cfg.get("n_pairs", 3))
    prompt = _QA_TEMPLATE.format(n_pairs=n_pairs, text=src)
    resp = _call(ctx, [{"role": "user", "content": prompt}], out, json_mode=True)
    provs = [_prov(ctx, resp.cache_hit)]
    source_id = str(rec.get("id", ""))
    if resp.error is not None or not (resp.content or "").strip():
        out.drops.append("llm_error")
        return out
    data = parse_json_object(resp.content)
    pairs = data.get("pairs") if isinstance(data, dict) else None
    if not isinstance(pairs, list):
        out.drops.append("llm_invalid_json")
        return out
    src_meta = rec.get("meta") or {}
    kept = 0
    for pair in pairs:
        out.counts["pairs_generated"] += 1
        question = str(pair.get("question") or "").strip() if isinstance(pair, dict) else ""
        answer = str(pair.get("answer") or "").strip() if isinstance(pair, dict) else ""
        if not question or not answer:
            out.drops.append("llm_invalid_pair")
            continue
        ov = _overlap(answer, src)
        out.overlaps.append(ov)
        if ov < ctx.min_overlap:
            out.drops.append("ungrounded")
            continue
        kept += 1
        out.counts["pairs_kept"] += 1
        additions = {"synthetic": True, "source_id": source_id, "overlap": ov}
        for key in ("domain", "license"):
            if src_meta.get(key) is not None:
                additions[key] = src_meta[key]
        out.outputs.append(
            {
                "id": f"{source_id}#qa{kept}",
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "meta": _meta_with(rec, additions, provs),
            }
        )
    if out.outputs:
        out.modified["qa_pairs"] += len(out.outputs)
        out.modified["llm_provenance"] += len(out.outputs)
    return out


def _last_user_prompt(rec: dict) -> str:
    prompt = rec.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    messages = rec.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return ""


def _final_answer(trace: str) -> str | None:
    matches = _ANSWER_LINE_RE.findall(trace)
    return matches[-1] if matches else None


def _normalise(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _answers_match(extracted: str, gold: str, kind: str) -> bool:
    if kind == "mcq_letter":
        got = _MCQ_LETTER_RE.search(extracted)
        want = _MCQ_LETTER_RE.search(gold)
        if not got or not want:
            return False
        return got.group(1).upper() == want.group(1).upper()
    return _normalise(extracted) == _normalise(gold)


def _trace_one(rec: dict, cfg: dict, ctx: _Ctx) -> _Outcome:
    out = _Outcome()
    question = _last_user_prompt(rec)
    if not question:
        out.drops.append("no_prompt")
        return out
    kind = str(cfg.get("answer_kind") or "exact")
    prompt = _TRACE_TEMPLATE.format(problem=question)
    resp = _call(ctx, [{"role": "user", "content": prompt}], out)
    provs = [_prov(ctx, resp.cache_hit)]
    if resp.error is not None or not (resp.content or "").strip():
        out.drops.append("llm_error")
        return out
    trace = resp.content or ""
    extracted = _final_answer(trace)
    if extracted is None:
        out.drops.append("trace_no_answer")
        return out
    gold_raw = rec.get("answer")
    gold = "" if gold_raw is None else str(gold_raw).strip()
    record = dict(rec)  # keep "prompt" and "answer"; messages become the SFT pair
    record["messages"] = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": trace},
    ]
    if not gold:
        if not cfg.get("keep_unverified"):
            out.drops.append("unverifiable")
            return out
        record["meta"] = _meta_with(rec, {"verified": None}, provs)  # not guessed
        out.modified["unverified_kept"] += 1
    else:
        out.counts["traces_attempted"] += 1
        if not _answers_match(extracted, gold, kind):
            out.drops.append("trace_wrong_answer")
            return out
        out.counts["traces_verified"] += 1
        record["meta"] = _meta_with(rec, {"verified": True}, provs)
        out.modified["verified"] += 1
    out.modified["llm_provenance"] += 1
    out.outputs.append(record)
    return out


def _render_chat(messages: list) -> str:
    lines = []
    for msg in messages:
        if isinstance(msg, dict):
            lines.append(f"{msg.get('role', '?')}: {msg.get('content', '')}")
    return "\n".join(lines)


def _coerce_score(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        score = value
    elif isinstance(value, float) and float(value).is_integer():
        score = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        score = int(value.strip())
    else:
        return None
    return score if 1 <= score <= 5 else None


def _winner(data: dict | None) -> str | None:
    if isinstance(data, dict) and data.get("winner") in ("A", "B"):
        return str(data["winner"])
    return None


def _judge_content(rec: dict, ctx: _Ctx, out: _Outcome) -> str:
    messages = rec.get("messages")
    if isinstance(messages, list) and messages:
        rendered = _render_chat(messages)
        if rendered.strip():
            return _truncate(rendered, ctx.max_chars, out)
    return _doc_text(rec, ctx, out)


def _judge_point(rec: dict, cfg: dict, ctx: _Ctx) -> _Outcome:
    out = _Outcome()
    content = _judge_content(rec, ctx, out)
    if not content:
        out.drops.append("no_text")
        return out
    rubric = str(cfg.get("rubric") or _DEFAULT_RUBRIC)
    min_score = int(cfg.get("min_score", 3))
    prompt = _JUDGE_POINT_TEMPLATE.format(rubric=rubric, content=content)
    resp = _call(ctx, [{"role": "user", "content": prompt}], out, json_mode=True)
    provs = [_prov(ctx, resp.cache_hit)]
    score: int | None = None
    if resp.error is None and (resp.content or "").strip():
        score = _coerce_score((parse_json_object(resp.content) or {}).get("score"))
    record = dict(rec)
    record["meta"] = _meta_with(rec, {"judge_score": score}, provs)
    out.modified["llm_provenance"] += 1
    out.modified["judged"] += 1
    if score is None:
        out.counts["unmeasured"] += 1  # kept with score None -- never guessed
        out.outputs.append(record)
        return out
    if score < min_score:
        out.drops.append("judge_low_score")
        return out
    out.outputs.append(record)
    return out


def _judge_pair(rec: dict, cfg: dict, ctx: _Ctx) -> _Outcome:
    out = _Outcome()
    chosen = str(rec.get("chosen") or "")
    rejected = str(rec.get("rejected") or "")
    problem = str(rec.get("prompt") or "")
    rubric = str(cfg.get("rubric") or _DEFAULT_RUBRIC)
    provs: list[dict] = []
    verdicts: list[str | None] = []  # order swap: (chosen=A,rejected=B) then reverse
    for a_text, b_text in ((chosen, rejected), (rejected, chosen)):
        prompt = _JUDGE_PAIR_TEMPLATE.format(
            rubric=rubric, prompt=problem, a=a_text, b=b_text
        )
        resp = _call(ctx, [{"role": "user", "content": prompt}], out, json_mode=True)
        provs.append(_prov(ctx, resp.cache_hit))
        if resp.error is not None or not (resp.content or "").strip():
            verdicts.append(None)
            continue
        verdicts.append(_winner(parse_json_object(resp.content)))
    first, second = verdicts
    record = dict(rec)
    out.counts["pairs_judged"] += 1
    if first is None or second is None:
        agree: bool | None = None  # no usable verdict: kept, counted unmeasured
        out.counts["unmeasured"] += 1
    else:
        chosen_wins = first == "A" and second == "B"
        rejected_wins = first == "B" and second == "A"
        if chosen_wins or rejected_wins:
            out.counts["pairs_consistent"] += 1
        if rejected_wins:
            record["meta"] = _meta_with(rec, {"judge_agrees": False}, provs)
            out.drops.append("judge_prefers_rejected")
            out.modified["judged"] += 1
            out.modified["llm_provenance"] += 1
            return out
        agree = True if chosen_wins else None
    if agree is None and first is not None and second is not None:
        out.counts["pairs_inconsistent"] += 1
        if cfg.get("drop_inconsistent"):
            out.drops.append("judge_inconsistent")
            out.modified["judged"] += 1
            out.modified["llm_provenance"] += 1
            out.modified["assigned_judge_agrees"] += 1
            return out
    record["meta"] = _meta_with(rec, {"judge_agrees": agree}, provs)
    out.modified["judged"] += 1
    out.modified["llm_provenance"] += 1
    out.outputs.append(record)
    return out


def _judge_one(rec: dict, cfg: dict, ctx: _Ctx) -> _Outcome:
    chosen = rec.get("chosen")
    rejected = rec.get("rejected")
    if (
        isinstance(chosen, str)
        and chosen.strip()
        and isinstance(rejected, str)
        and rejected.strip()
    ):
        return _judge_pair(rec, cfg, ctx)
    return _judge_point(rec, cfg, ctx)


_HANDLERS = {
    "rephrase": _rephrase_one,
    "qa_synth": _qa_one,
    "reasoning_trace": _trace_one,
    "judge": _judge_one,
}


def _pick_temperature(cfg: dict, mode: str) -> float:
    raw = cfg.get("temperature")
    if raw is None:
        return 0.7 if mode == "rephrase" else 0.0
    return float(raw)


def _template_params(cfg: dict, mode: str) -> tuple[str, str]:
    if mode == "rephrase":
        return _REPHRASE_TEMPLATE, str(cfg.get("style") or "wikipedia")
    if mode == "qa_synth":
        return _QA_TEMPLATE, str(cfg.get("n_pairs", 3))
    if mode == "reasoning_trace":
        return _TRACE_TEMPLATE, str(cfg.get("answer_kind") or "exact")
    joined = _JUDGE_POINT_TEMPLATE + _JUDGE_PAIR_TEMPLATE
    return joined, str(cfg.get("rubric") or _DEFAULT_RUBRIC)


def llm_enhance(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    mode = cfg.get("mode")
    if mode not in _MODES:
        known = ", ".join(_MODES)
        raise LLMOpError(f"llm_enhance: unknown mode {mode!r}; known modes: {known}")
    if "backend" not in cfg:
        raise LLMOpError(
            "llm_enhance: missing input: config.backend "
            "(an openai_compatible endpoint or a registered backend kind)"
        )
    backend = make_backend(cfg.get("backend"), _NAME)
    stats.backend = f"llm:{backend.kind}:{backend.model}"
    template, params = _template_params(cfg, mode)
    ph = prompt_hash(_NAME, mode, template, params)
    stats.extra["model"] = backend.model
    stats.extra["mode"] = mode
    stats.extra["prompt_hash"] = ph
    ctx = _Ctx(
        backend=backend,
        mode=mode,
        field=str(cfg.get("field") or "text"),
        max_chars=int(cfg.get("max_chars", 8000)),
        temperature=_pick_temperature(cfg, mode),
        max_tokens=int(cfg.get("max_tokens", 2048)),
        seed=int(cfg.get("seed", 0)),
        min_overlap=float(cfg.get("min_overlap", 0.5)),
        ph=ph,
    )
    handler = _HANDLERS[mode]
    limit = cfg.get("max_calls")
    call_budget = budget(int(limit)) if limit is not None else budget(None)
    workers = max(int(cfg.get("workers", 4)), 1)

    def _inputs() -> Iterator[tuple[bool, dict]]:
        for rec in counted(records, stats):
            yield (call_budget.take(), rec)  # ordered, so the budget is deterministic

    def _work(item: tuple[bool, dict]) -> _Outcome:
        allowed, rec = item
        if not allowed:
            return _Outcome(outputs=[rec], budget_skip=True)  # untouched, no provenance
        return handler(rec, cfg, ctx)

    totals: Counter = Counter()
    modified: Counter = Counter()
    overlaps: list[float] = []
    for outcome in map_ordered(_work, _inputs(), workers):
        for reason in outcome.drops:
            stats.drop(reason)
        totals.update(outcome.counts)
        modified.update(outcome.modified)
        overlaps.extend(outcome.overlaps)
        if outcome.budget_skip:
            stats.extra["budget_exhausted"] = True
        for record in outcome.outputs:
            stats.records_out += 1
            yield record

    stats.modified.update(modified)
    for key in (
        "calls", "cache_hits", "prompt_tokens", "completion_tokens",
        "errors", "input_truncated",
    ):
        stats.extra[key] = totals.get(key, 0)
    for key in (
        "pairs_generated", "pairs_kept", "fallback_original",
        "unmeasured", "pairs_inconsistent",
    ):
        if totals.get(key):
            stats.extra[key] = totals[key]
    stats.extra["overlap_mean"] = sum(overlaps) / len(overlaps) if overlaps else None
    if mode == "reasoning_trace":
        attempted = totals.get("traces_attempted", 0)
        verified = totals.get("traces_verified", 0)
        stats.extra["verified_rate"] = verified / attempted if attempted else None
    if mode == "judge":
        judged_pairs = totals.get("pairs_judged", 0)
        consistent = totals.get("pairs_consistent", 0)
        stats.extra["position_consistency"] = (
            consistent / judged_pairs if judged_pairs else None
        )


llm_enhance_op = FunctionOp(_NAME, llm_enhance, config_schema={
    "type": "object",
    "required": ["backend", "mode"],
    "additionalProperties": False,
    "properties": {
        "backend": {"type": "object"},
        "mode": {"type": "string", "enum": list(_MODES)},
        "field": {"type": "string", "minLength": 1},
        "max_chars": {"type": "integer", "minimum": 1},
        "workers": {"type": "integer", "minimum": 1},
        "max_calls": {"type": ["integer", "null"], "minimum": 1},
        "temperature": {"type": ["number", "null"], "minimum": 0.0, "maximum": 2.0},
        "max_tokens": {"type": "integer", "minimum": 1},
        "seed": {"type": "integer"},
        "min_overlap": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "style": {"type": "string", "enum": list(_STYLES)},
        "emit": {"type": "string", "enum": ["append", "replace"]},
        "n_pairs": {"type": "integer", "minimum": 1, "maximum": 20},
        "answer_kind": {"type": "string", "enum": ["mcq_letter", "exact"]},
        "keep_unverified": {"type": "boolean"},
        "rubric": {"type": "string", "minLength": 1},
        "min_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "drop_inconsistent": {"type": "boolean"},
    },
})
register_op(llm_enhance_op)
