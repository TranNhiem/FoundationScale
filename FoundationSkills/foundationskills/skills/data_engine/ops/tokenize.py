
"""Tokenize op: measure per-record token lengths, truncation, and packing.

With transformers installed and ``cfg["tokenizer"]`` naming a loadable
tokenizer, token counts are real (``tokenizer.encode(add_special_tokens=False)``).
Without it the op falls back to an APPROXIMATE count (ceil(chars/4) for
non-CJK, chars/1.5 for CJK), records ``stats.backend = "approx"`` and
``stats.extra["approximate"] = True``; the readiness report must therefore
downgrade the min-tokens check to UNMEASURED. A broken tokenizer mid-stream
also flips the approximate flag per-record and records
``stats.extra["tokenizer_error"]``.

Deliberate spec-silent choices:
- ``field`` defaults to "auto" (not "text") so preference/rl records are
  counted correctly out of the box: preference counts prompt+chosen, rl
  counts the human turns plus the declared gold answer.
- All lengths are kept in memory; the histogram/percentiles are computed on a
  deterministic subsample (seed 0) of at most ``sample_for_hist`` lengths,
  while totals and truncation_rate always use the full stream.
- One EOS token per record is added when ``add_eos`` is true.
- ``stats.extra["domain_counts"]`` tallies ``meta.domain`` when present, which
  the readiness report surfaces as ``domain_breakdown``.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Iterable, Iterator

from foundationskills.skills.data_engine.ops.base import FunctionOp, OpStats, counted, register_op

__all__ = ["tokenize_op", "pack_lengths"]

_CJK_RANGES = (
    (0x4E00, 0x9FFF),   # CJK unified ideographs
    (0x3400, 0x4DBF),   # CJK extension A
    (0xF900, 0xFAFF),   # CJK compatibility ideographs
    (0x3040, 0x30FF),   # hiragana + katakana
    (0xAC00, 0xD7AF),   # hangul syllables
)


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def _approx_tokens(text: str) -> int:
    """Approximation: ceil(other/4 + cjk/1.5); never 0 for non-empty text."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return max(1, math.ceil(other / 4 + cjk / 1.5))


def _load_tokenizer(path: str) -> tuple[Any | None, str | None]:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        return None, f"transformers unavailable: {type(exc).__name__}: {exc}"
    try:
        return AutoTokenizer.from_pretrained(path), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _count_text(tokenizer: Any, text: str, stats: OpStats) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception as exc:  # noqa: BLE001 - recorded, approximation used
            stats.extra.setdefault("tokenizer_error", f"encode failed: {type(exc).__name__}: {exc}")
            stats.extra["approximate"] = True
    return _approx_tokens(text)


def _texts_for(rec: dict, field: str) -> list[str]:
    """Texts to count for one record (see module docstring for 'auto' rules)."""
    if field == "text":
        text = rec.get("text")
        return [text] if isinstance(text, str) else []
    # auto
    text = rec.get("text")
    if isinstance(text, str):
        return [text]
    prompt, chosen = rec.get("prompt"), rec.get("chosen")
    if isinstance(prompt, str) and isinstance(chosen, str):
        return [prompt, chosen]  # preference: prompt + chosen
    conv = rec.get("conversations")
    if isinstance(conv, list):
        values = [
            t.get("value")
            for t in conv
            if isinstance(t, dict) and isinstance(t.get("value"), str)
        ]
        human = [t.get("value") for t in conv if isinstance(t, dict) and t.get("from") == "human" and isinstance(t.get("value"), str)]
        gpt = [t.get("value") for t in conv if isinstance(t, dict) and t.get("from") == "gpt" and isinstance(t.get("value"), str)]
        gold = rec.get("answer")
        if isinstance(gold, str) and gold:
            return [*human, gold]  # rl: prompt + declared answer
        if gpt:
            return [*human, gpt[-1]]
        return values[-2:] if len(values) >= 2 else values
    return []


def _bucket_upper(length: int) -> int:
    bucket = 1
    while bucket < length:
        bucket <<= 1
    return bucket


def _percentile(sorted_values: list[int], p: float) -> int:
    if not sorted_values:
        return 0
    rank = math.ceil(p / 100 * len(sorted_values)) - 1
    return sorted_values[max(0, min(rank, len(sorted_values) - 1))]


def pack_lengths(lengths: list[int], seq_len: int) -> dict[str, Any]:
    """First-fit-decreasing packing of sequences of the given token lengths.

    Each length is capped at seq_len (a longer record occupies its own bin and
    contributes seq_len to the numerator). Returns efficiency =
    sum(min(l, seq_len)) / (num_bins * seq_len) plus the bin count.
    """
    fitted = sorted((min(int(l), seq_len) for l in lengths), reverse=True)
    remaining: list[int] = []  # free space per bin, indexed in creation order
    for length in fitted:
        placed = False
        for idx in range(len(remaining)):
            if remaining[idx] >= length:
                remaining[idx] -= length
                placed = True
                break
        if not placed:
            remaining.append(seq_len - length)
    total = sum(fitted)
    num_bins = len(remaining)
    efficiency = (total / (num_bins * seq_len)) if num_bins else 0.0
    return {
        "efficiency": efficiency,
        "num_sequences": num_bins,
        "seq_len": seq_len,
        "total_tokens": total,
    }



# FS tokenizes with truncation=True at max_sequence_length (train/loop.py), so a
# corpus record longer than seq_len silently loses its tail. For raw-text stages
# the Data Engine splits such records instead. Two tokens of headroom cover the
# BOS the FS tokenizer call adds and the EOS counted here.
_CHUNK_HEADROOM = 2


def _token_windows(tokenizer: Any, text: str, budget: int, stats: OpStats) -> list[str]:
    """Split one over-long span into pieces of <= budget tokens."""
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            return [tokenizer.decode(ids[i:i + budget]) for i in range(0, len(ids), budget)]
        except Exception as exc:  # noqa: BLE001 - recorded; char windows used
            stats.extra.setdefault("tokenizer_error", f"window split failed: {type(exc).__name__}: {exc}")
    n = max(_approx_tokens(text), 1)
    width = max(int(len(text) * budget / n * 0.95), 1)
    return [text[i:i + width] for i in range(0, len(text), width)]


def _chunk_text(tokenizer: Any, text: str, budget: int, stats: OpStats) -> list[str]:
    """Greedy paragraph packing into chunks of <= budget tokens; a paragraph
    longer than the budget is itself split into token windows."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for para in text.split("\n\n"):
        if not para.strip():
            continue
        n = _count_text(tokenizer, para, stats)
        if n > budget:
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            chunks.extend(_token_windows(tokenizer, para, budget, stats))
            continue
        # +2 for the paragraph separator's tokens
        if current and current_tokens + n + 2 > budget:
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(para)
        current_tokens += n + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _tokenize_records(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    seq_len = int(cfg.get("seq_len", 4096))
    field = cfg.get("field") or "auto"
    pack = bool(cfg.get("pack", False))
    add_eos = bool(cfg.get("add_eos", True))
    sample_for_hist = int(cfg.get("sample_for_hist", 100000))

    tokenizer: Any = None
    tok_spec = cfg.get("tokenizer")
    if tok_spec:
        if isinstance(tok_spec, str):
            tokenizer, error = _load_tokenizer(tok_spec)
            if error:
                stats.extra["tokenizer_error"] = error
        else:
            tokenizer = tok_spec  # tokenizer-like object handed in directly
    stats.backend = "transformers" if tokenizer is not None else "approx"
    if tokenizer is None:
        stats.extra["approximate"] = True

    lengths: list[int] = []
    domain_counts: Counter = Counter()
    total_tokens = 0
    truncated = 0
    chunk = bool(cfg.get("chunk", False))
    # Example-level formats cannot be chunked (it would cut an answer from its
    # question); drop_overlong removes examples FS would truncate, counted.
    drop_overlong = bool(cfg.get("drop_overlong", False))
    budget = max(seq_len - _CHUNK_HEADROOM, 1)

    def expanded(stream: Iterable[dict]) -> Iterator[dict]:
        for rec in stream:
            text = rec.get("text") if isinstance(rec, dict) else None
            if not (chunk and isinstance(text, str)) or _count_text(tokenizer, text, stats) <= budget:
                yield rec
                continue
            pieces = _chunk_text(tokenizer, text, budget, stats)
            stats.modified["chunked_records"] += 1
            stats.modified["chunks_emitted"] += len(pieces)
            for k, piece in enumerate(pieces):
                out = dict(rec)
                out["id"] = f"{rec.get('id')}#{k}"
                out["text"] = piece
                out["meta"] = {**(rec.get("meta") or {}), "chunk_of": rec.get("id"), "chunk_index": k}
                yield out

    for rec in expanded(counted(records, stats)):
        texts = _texts_for(rec, field)
        num = sum(_count_text(tokenizer, t, stats) for t in texts)
        if add_eos and texts:
            num += 1
        if drop_overlong and num > seq_len:
            stats.drop("overlong")
            continue
        total_tokens += num
        lengths.append(num)
        if num > seq_len:
            truncated += 1
        meta = rec.get("meta") if isinstance(rec, dict) else None
        domain = (meta or {}).get("domain") if isinstance(meta, dict) else None
        if isinstance(domain, str) and domain:
            domain_counts[domain] += 1
        out = dict(rec)
        out_meta = dict(meta) if isinstance(meta, dict) else {}
        out_meta["num_tokens"] = num
        out["meta"] = out_meta
        stats.records_out += 1
        yield out

    sample = lengths
    if len(sample) > sample_for_hist:
        sample = random.Random(0).sample(sample, sample_for_hist)
    ordered = sorted(sample)
    hist: dict[str, int] = {}
    for length in sample:
        key = str(_bucket_upper(length))
        hist[key] = hist.get(key, 0) + 1
    stats.extra.update(
        {
            "num_tokens": total_tokens,
            "records": len(lengths),
            "length_hist": hist,
            "p50": _percentile(ordered, 50),
            "p90": _percentile(ordered, 90),
            "p99": _percentile(ordered, 99),
            "max": ordered[-1] if ordered else 0,
            "truncation_rate": (truncated / len(lengths)) if lengths else 0.0,
            "domain_counts": dict(domain_counts),
        }
    )
    if len(lengths) > len(sample):
        stats.extra["hist_sampled"] = True
    if pack:
        stats.extra["packing"] = pack_lengths(lengths, seq_len)


_CONFIG_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "tokenizer": {"type": ["string", "null"]},
        "seq_len": {"type": "integer", "minimum": 1},
        "field": {"enum": ["text", "auto"]},
        "pack": {"type": "boolean"},
        "chunk": {"type": "boolean"},
        "drop_overlong": {"type": "boolean"},
        "add_eos": {"type": "boolean"},
        "sample_for_hist": {"type": "integer", "minimum": 1},
    },
}

tokenize_op = register_op(FunctionOp("tokenize", _tokenize_records, _CONFIG_SCHEMA))
