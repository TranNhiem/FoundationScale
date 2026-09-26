
"""``quality`` op: heuristic quality filters (Gopher / C4 / FineWeb style) and SFT checks.

Decisions where the spec is silent:

- Every evaluated rule contributes to the score:
  ``score = 1 - failed/evaluated``; None (unmeasured, never filtered) when
  nothing evaluated. Rules are
  *flagged* into ``meta.quality = {"score", "failed": [...]}``; records are
  only dropped (``low_quality``) when ``cfg.min_quality_score`` is set.
- Word-based rules (mean word length, symbol ratio, alpha-word ratio,
  stopwords) are skipped when the text is mostly CJK, with
  ``stats.extra["cjk_skipped_word_rules"] = True``.
- The stopword rule only applies to Latin-script text ("for en").
- "Truncated" SFT answers: no terminal punctuation and length >= 400 chars
  (a proxy for "near a max sequence limit").
- "Short answer": question >= 100 chars and answer < max(10, 5% of question).
- ``scorer: {kind: "callable", path: "module:function"}`` multiplies the
  heuristic score by an external 0..1 score; a load failure is recorded in
  ``stats.extra["scorer_error"]`` and the external score is left UNMEASURED
  (the record is not silently credited).

Exports :func:`coverage_report`, the domain/diversity analysis used by the
readiness/planning layer.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import string
import unicodedata
from collections import Counter
from typing import Any, Callable, Iterable, Iterator

from foundationskills.skills.data_engine.ops.base import FunctionOp, OpStats, counted, register_op
from foundationskills.skills.data_engine.ops.clean import mostly_cjk, primary_text


CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    # Unknown keys are refused: a recommender once sent {"ruleset": ...}, which this
    # op ignored, so quality measured NOTHING on 23k real records.
    "additionalProperties": False,
    "properties": {
        "heuristics": {"anyOf": [{"enum": ["gopher", "c4", "fineweb", "none"]},
                                 {"type": "array", "items": {"enum": ["gopher", "c4", "fineweb"]}}]},
        "sft_checks": {"type": "boolean"},
        "min_quality_score": {"type": ["number", "null"]},
        "scorer": {
            "type": ["object", "null"],
            "properties": {"kind": {"enum": ["callable"]}, "path": {"type": "string"}},
        },
    },
}

_EN_STOPWORDS = frozenset({
    "the", "a", "an", "and", "of", "to", "in", "is", "that", "for", "on",
    "with", "as", "by", "it", "from", "or", "be", "are", "was", "were",
    "this", "i", "you", "he", "she", "they", "we",
})
_BULLET_CHARS = ("•", "-", "*", "·", "◦")
_TERMINAL_CHARS = tuple(".!?。！？…") + ('"', "'", "”", "’", ")", "】", "」", "』")
_CODE_DOMAINS = {"code", "programming", "software", "dev", "engineering"}
_REFUSAL_MARKERS = ("as an ai language model", "i cannot help", "i can't help")
_TRUNCATION_MIN_CHARS = 400


def _latin_share(text: str) -> float:
    letters = sum(1 for ch in text if ch.isalpha())
    if not letters:
        return 0.0
    latin = sum(1 for ch in text if ch.isalpha() and ord(ch) < 0x250)
    return latin / letters


def _gopher_eval(text: str, *, word_rules: bool) -> list[tuple[str, bool]]:
    """Gopher-style rules as (tag, failed) pairs."""
    out: list[tuple[str, bool]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if word_rules:
        words = text.split()
        if not words:
            out.append(("gopher:mean_word_length", True))
        else:
            mean_len = sum(len(w) for w in words) / len(words)
            out.append(("gopher:mean_word_length", mean_len < 3.0 or mean_len > 10.0))
            sym = sum(1 for w in words if not any(c.isalnum() for c in w)) / len(words)
            out.append(("gopher:symbol_ratio", sym >= 0.1))
            alpha = sum(1 for w in words if any(c.isalpha() for c in w)) / len(words)
            out.append(("gopher:alpha_words", alpha < 0.8))
            if _latin_share(text) >= 0.8:
                lowered = {w.strip(string.punctuation).lower() for w in words}
                out.append(("gopher:stopwords", len(lowered & _EN_STOPWORDS) < 2))
    if lines:
        bullets = sum(1 for line in lines if line[:1] in _BULLET_CHARS) / len(lines)
        out.append(("gopher:bullet_ratio", bullets >= 0.9))
        ellipses = sum(1 for line in lines if "..." in line) / len(lines)
        out.append(("gopher:ellipsis_ratio", ellipses >= 0.3))
    return out


def looks_like_code(text: str, domain: str | None = None) -> bool:
    """Code-like records must not be judged by prose rules (braces, terminal
    punctuation and punctuation density are normal in code)."""
    if (domain or "").lower() in _CODE_DOMAINS or "```" in text:
        return True
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    indented = sum(1 for line in lines if line.startswith(("    ", "\t"))) / len(lines)
    symbols = sum(text.count(ch) for ch in "{};=()") / max(len(text), 1)
    # density needs enough text to mean anything ("{value}" in one sentence is prose)
    return indented >= 0.3 or (len(text) >= 80 and symbols >= 0.05)


def _c4_eval(text: str, *, domain: str | None) -> list[tuple[str, bool]]:
    """Rules whose precondition is unmet (fewer than 3 lines) are NOT evaluated:
    counting them as passes would inflate the score with checks that never ran."""
    out: list[tuple[str, bool]] = []
    low = text.lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    code = looks_like_code(text, domain)
    if len(lines) >= 3 and not code:
        terminal = sum(1 for line in lines if line.endswith(_TERMINAL_CHARS)) / len(lines)
        out.append(("c4:terminal_punct", terminal < 0.5))
    out.append(("c4:lorem_ipsum", "lorem ipsum" in low))
    js_boiler = "enable javascript" in low or "javascript is disabled" in low or "javascript is turned off" in low
    out.append(("c4:javascript_boilerplate", js_boiler))
    if not code:
        out.append(("c4:curly_braces", "{" in text or "}" in text))
    return out


def _fineweb_eval(text: str, domain: str | None = None) -> list[tuple[str, bool]]:
    out: list[tuple[str, bool]] = []
    code = looks_like_code(text, domain)
    if not code and text:
        punct = sum(1 for ch in text if unicodedata.category(ch).startswith("P"))
        out.append(("fineweb:punct_ratio", punct / len(text) > 0.25))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_failed = dup_failed = False
    if len(lines) >= 3:
        short = sum(1 for line in lines if len(line) < 30) / len(lines)
        short_failed = short > 0.67 and not code
        dup_failed = (1.0 - len(set(lines)) / len(lines)) > 0.3
        out.append(("fineweb:short_line_ratio", short_failed))
        out.append(("fineweb:dup_lines", dup_failed))
    return out


def _message_roles(rec: dict) -> list[str]:
    messages = rec.get("messages")
    if not isinstance(messages, list):
        return []
    return [str(m.get("role", "")) for m in messages if isinstance(m, dict)]


def _sft_eval(rec: dict) -> list[tuple[str, bool]]:
    """SFT structural/content checks as (tag, failed) pairs."""
    out: list[tuple[str, bool]] = []
    messages = rec.get("messages")
    answer: str | None = None
    question: str | None = None

    if isinstance(messages, list) and messages:
        roles = _message_roles(rec)
        out.append(("sft:missing_assistant", "assistant" not in roles))
        assistant_contents = [
            str(m.get("content", "")) for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        if assistant_contents:
            answer = assistant_contents[-1]
        user_contents = [
            str(m.get("content", "")) for m in messages if isinstance(m, dict) and m.get("role") == "user"
        ]
        if user_contents:
            question = user_contents[-1]
        seq = roles[1:] if roles and roles[0] == "system" else roles
        bad_order = bool(seq) and seq[0] != "user"
        if not bad_order:
            bad_order = any(seq[i] == seq[i + 1] for i in range(len(seq) - 1))
        out.append(("sft:bad_role_order", bad_order))

    if answer is None:
        if isinstance(rec.get("answer"), str):
            answer = rec["answer"]
        elif isinstance(rec.get("chosen"), str):
            answer = rec["chosen"]
    if question is None and isinstance(rec.get("prompt"), str):
        question = rec["prompt"]
    if answer is None and question is None:
        return out  # not an SFT-shaped record; nothing else applies
    if answer is None:
        out.append(("sft:empty_answer", True))
        return out

    stripped = answer.strip()
    out.append(("sft:empty_answer", not stripped))
    if not stripped:
        return out  # further checks would be vacuous
    out.append(("sft:echo", bool(question) and question is not None and stripped == question.strip()))
    if not looks_like_code(stripped):  # code answers legitimately end in braces/fences
        out.append((
            "sft:truncated",
            len(stripped) >= _TRUNCATION_MIN_CHARS and not stripped.endswith(_TERMINAL_CHARS),
        ))
    low = stripped.lower()
    out.append(("sft:refusal_boilerplate", any(marker in low for marker in _REFUSAL_MARKERS)))
    short = bool(question) and question is not None and len(question) >= 100 and len(stripped) < max(10.0, 0.05 * len(question))
    out.append(("sft:short_answer", short))
    return out


def _load_scorer(cfg: dict, stats: OpStats) -> Callable[[str], float] | None:
    scorer_cfg = cfg.get("scorer")
    if not scorer_cfg:
        return None
    if scorer_cfg.get("kind") != "callable" or not scorer_cfg.get("path"):
        stats.extra["scorer_error"] = f"unsupported scorer config: {scorer_cfg!r}"
        return None
    path = str(scorer_cfg["path"])
    module_name, sep, func_name = path.partition(":")
    if not sep:
        stats.extra["scorer_error"] = f"scorer path must be 'module:function', got {path!r}"
        return None
    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, func_name)
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        stats.extra["scorer_error"] = f"scorer load failed: {type(exc).__name__}: {exc}"
        return None
    if not callable(fn):
        stats.extra["scorer_error"] = f"scorer {path!r} is not callable"
        return None
    return fn


def _is_sft_shape(rec: dict) -> bool:
    return any(isinstance(rec.get(k), (list, str)) and bool(rec.get(k)) for k in ("messages", "prompt", "answer", "chosen"))


def _quality_op(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    raw_heuristics = cfg.get("heuristics", "none")
    heuristic_sets = [raw_heuristics] if isinstance(raw_heuristics, str) else list(raw_heuristics)
    heuristic_sets = [h for h in heuristic_sets if h != "none"]
    sft_checks = bool(cfg.get("sft_checks", False))
    min_score = cfg.get("min_quality_score")
    scorer_fn = _load_scorer(cfg, stats)

    for rec in counted(records, stats):
        if not isinstance(rec, dict):
            stats.drop("non_dict_record")
            continue
        text = primary_text(rec)
        cjk = mostly_cjk(text)
        if cjk:
            stats.extra["cjk_skipped_word_rules"] = True
        domain = None
        meta = rec.get("meta")
        if isinstance(meta, dict):
            domain = meta.get("domain")

        evaluated: list[tuple[str, bool]] = []
        if "gopher" in heuristic_sets:
            evaluated.extend(_gopher_eval(text, word_rules=not cjk))
        if "c4" in heuristic_sets:
            evaluated.extend(_c4_eval(text, domain=None if domain is None else str(domain)))
        if "fineweb" in heuristic_sets:
            evaluated.extend(_fineweb_eval(text, None if domain is None else str(domain)))
        if sft_checks and _is_sft_shape(rec):
            evaluated.extend(_sft_eval(rec))

        failed = [tag for tag, is_failed in evaluated if is_failed]
        # No rule evaluated is absence of evidence: score None, never 1.0.
        score: float | None = (1.0 - len(failed) / len(evaluated)) if evaluated else None
        quality: dict[str, Any] = {"score": None if score is None else round(score, 4), "failed": failed}
        if score is None:
            quality["unmeasured"] = True
            stats.modified["quality_unmeasured"] += 1
        if looks_like_code(text, None if domain is None else str(domain)):
            stats.modified["code_aware_records"] += 1
        if scorer_fn is not None:
            try:
                external = float(scorer_fn(text))
            except Exception as exc:  # noqa: BLE001 - recorded, record not credited
                stats.extra["scorer_error"] = f"scorer call failed: {type(exc).__name__}: {exc}"
            else:
                quality["external_score"] = round(external, 4)
                score = (1.0 if score is None else score) * max(0.0, min(1.0, external))
                quality["score"] = round(score, 4)
                quality.pop("unmeasured", None)

        rec_meta = rec.setdefault("meta", {})
        if isinstance(rec_meta, dict):
            rec_meta["quality"] = quality

        if min_score is not None and score is not None and score < float(min_score):
            stats.drop("low_quality")
            continue
        stats.records_out += 1
        yield rec


register_op(FunctionOp("quality", _quality_op, CONFIG_SCHEMA))


def coverage_report(records: Iterable[dict]) -> dict[str, Any]:
    """Domain/diversity analysis over records: domain/lang counts, top 3-grams,
    distinct-1/2 ratios, and a simple lexical clustering.

    Clustering: each record is bucketed by the hash of its top-3 sorted content
    words; the report gives the cluster count and the Shannon entropy of
    cluster sizes (0 means all records in one bucket).
    """
    recs = [rec for rec in records if isinstance(rec, dict)]
    domains: Counter[str] = Counter()
    langs: Counter[str] = Counter()
    trigrams: Counter[str] = Counter()
    all_tokens: list[str] = []
    clusters: Counter[str] = Counter()

    for rec in recs:
        meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        domains[str(meta.get("domain") or "unknown")] += 1
        langs[str(meta.get("lang") or "unknown")] += 1
        tokens = [w.lower() for w in primary_text(rec).split() if w.strip()]
        all_tokens.extend(tokens)
        for i in range(len(tokens) - 2):
            trigrams[" ".join(tokens[i:i + 3])] += 1
        content = [t for t in tokens if t not in _EN_STOPWORDS and any(c.isalpha() for c in t)]
        top_words = sorted(word for word, _ in Counter(content).most_common(3))
        bucket = hashlib.sha1(" ".join(top_words).encode("utf-8")).hexdigest()[:10]
        clusters[bucket] += 1

    distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
    bigrams = list(zip(all_tokens, all_tokens[1:]))
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0

    total_records = len(recs)
    entropy = 0.0
    if total_records:
        for size in clusters.values():
            p = size / total_records
            entropy -= p * math.log2(p)

    return {
        "records": total_records,
        "domains": dict(domains),
        "langs": dict(langs),
        "top_3grams": [[gram, count] for gram, count in trigrams.most_common(50)],
        "distinct_1": round(distinct_1, 6),
        "distinct_2": round(distinct_2, 6),
        "clusters": {
            "count": len(clusters),
            "size_entropy": round(entropy, 6),
            "buckets": dict(clusters),
        },
    }
