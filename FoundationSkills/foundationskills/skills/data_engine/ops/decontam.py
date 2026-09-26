
"""``decontam`` op: n-gram overlap decontamination against evaluation benchmarks.

- Benchmark n-grams: word-level on lowercased text for Latin scripts,
  char-level for text containing CJK characters. Default n = 13.
- Sources: ``cfg.sources[name]`` points to a ``.jsonl`` (known text keys:
  text/question/prompt/problem/answer/solution) or ``.txt`` (documents split
  on blank lines) file.
- Registry: builtin name -> HF id map (gsm8k, math, mmlu, humaneval, mbpp,
  arc, hellaswag, truthfulqa, ifeval). HF ids are loaded with ``datasets``
  *only* when the library is importable AND ``cfg.allow_download`` is true.
- A benchmark that cannot be materialized is recorded in
  ``stats.extra["unmeasured_benchmarks"]`` with its reason — records are NOT
  credited as decontaminated against it (absence of evidence is UNMEASURED).
- ``action: "drop"`` removes hit records (reason ``decontam``);
  ``action: "flag"`` keeps them and writes ``meta.decontam = {"hits": [...]}``.

``stats.extra``: ``hits_by_benchmark``, ``decontam_hits``,
``benchmarks_checked``, ``unmeasured_benchmarks`` (+``unmeasured_reasons``).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from foundationskills.skills.data_engine.ops.base import FunctionOp, OpStats, counted, register_op
from foundationskills.skills.data_engine.ops.clean import primary_text


CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,  # unknown keys refused: a silently ignored key measures nothing
    "properties": {
        "benchmarks": {"type": "array", "items": {"type": "string"}},
        "ngram": {"type": "integer", "minimum": 1},
        "sources": {"type": "object", "additionalProperties": {"type": "string"}},
        "action": {"enum": ["drop", "flag"]},
        "allow_download": {"type": "boolean"},
    },
}

_BLANK_LINE_RE = re.compile(r"\n\s*\n+")
_TEXT_KEYS = ("text", "question", "prompt", "problem", "answer", "solution", "query", "input")

# name -> HF dataset id; loading is gated on allow_download + datasets import
_BENCHMARK_HF: dict[str, str] = {
    "gsm8k": "openai/gsm8k",
    "math": "EleutherAI/hendrycks_math",
    "mmlu": "cais/mmlu",
    "humaneval": "openai/openai_humaneval",
    "mbpp": "google-research-datasets/mbpp",
    "arc": "allenai/ai2_arc",
    "hellaswag": "Rowan/hellaswag",
    "truthfulqa": "truthfulqa/truthful_qa",
    "ifeval": "google/IFEval",
}


def _contains_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF:
            return True
    return False


def _ngrams(text: str, n: int) -> set[str]:
    low = " ".join(text.lower().split())
    if not low:
        return set()
    if _contains_cjk(low):
        compact = "".join(low.split())
        return {compact[i:i + n] for i in range(len(compact) - n + 1)}
    tokens = low.split()
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _norm_whole(text: str) -> bytes:
    """Whole-text key for items shorter than n: an n-gram index cannot see them."""
    return hashlib.blake2b(" ".join(text.lower().split()).encode("utf-8"), digest_size=16).digest()


def _hashed_ngrams(text: str, n: int) -> set[bytes]:
    return {hashlib.blake2b(gram.encode("utf-8"), digest_size=16).digest() for gram in _ngrams(text, n)}


def _collect_texts(obj: Any, out: list[str]) -> None:
    if isinstance(obj, str):
        if obj.strip():
            out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_texts(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _collect_texts(value, out)


def _load_local_benchmark(path: Path) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        docs: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    pieces = [str(obj[k]) for k in _TEXT_KEYS if isinstance(obj.get(k), str)]
                    if not pieces:  # unfamiliar column names: every string field is benchmark text
                        _collect_texts(obj, pieces)
                    if pieces:
                        docs.append("\n".join(pieces))
                elif isinstance(obj, str):
                    docs.append(obj)
        return docs
    content = path.read_text(encoding="utf-8", errors="replace")
    return [part for part in _BLANK_LINE_RE.split(content) if part.strip()]


def _load_hf_benchmark(name: str, hf_id: str) -> list[str]:
    from datasets import load_dataset  # type: ignore

    docs: list[str] = []
    dataset = None
    last_exc: Exception | None = None
    for split in ("test", "validation", "train"):
        try:
            dataset = load_dataset(hf_id, split=split)
            break
        except Exception as exc:  # noqa: BLE001 - try the next split
            last_exc = exc
    if dataset is None:
        raise IngestErrorLike(f"could not load benchmark {name!r} from {hf_id!r}: {last_exc}")
    for row in dataset:
        if isinstance(row, dict):
            _collect_texts({k: v for k, v in row.items() if k in _TEXT_KEYS}, docs)
    return docs


class IngestErrorLike(ValueError):
    """Internal: benchmark download/parse failure -> benchmark is UNMEASURED."""


def _load_benchmark_docs(name: str, cfg: dict) -> tuple[list[str] | None, str | None]:
    """Return (docs, None) or (None, reason). None means UNMEASURED, never PASS."""
    sources = cfg.get("sources") or {}
    if name in sources:
        path = Path(str(sources[name]))
        if not path.is_file():
            return None, f"benchmark source file missing: {path}"
        try:
            docs = _load_local_benchmark(path)
        except OSError as exc:
            return None, f"benchmark source unreadable: {path}: {exc}"
        if not docs:
            return None, f"benchmark source produced no documents: {path}"
        return docs, None

    hf_id = _BENCHMARK_HF.get(name)
    if hf_id is None:
        return None, "unknown benchmark (not in sources, not in the builtin registry)"
    if not cfg.get("allow_download", False):
        return None, f"allow_download is false and no local source given for {name!r} (hf id {hf_id!r})"
    try:
        import datasets  # noqa: F401  type: ignore
    except ImportError:
        return None, f"optional dependency 'datasets' unavailable; cannot fetch benchmark {name!r}"
    try:
        docs = _load_hf_benchmark(name, hf_id)
    except IngestErrorLike as exc:
        return None, str(exc)
    if not docs:
        return None, f"benchmark {name!r} loaded but produced no documents"
    return docs, None


def _decontam_op(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    names = [str(n) for n in (cfg.get("benchmarks") or [])]
    n = int(cfg.get("ngram", 13))
    action = str(cfg.get("action", "drop"))

    benchmark_sets: dict[str, set[bytes]] = {}
    short_items: dict[str, set[bytes]] = {}
    unmeasured: dict[str, str] = {}
    for name in names:
        docs, reason = _load_benchmark_docs(name, cfg)
        if reason is not None:
            unmeasured[name] = reason
            continue
        grams: set[bytes] = set()
        shorts: set[bytes] = set()
        for doc in docs or []:
            doc_grams = _hashed_ngrams(doc, n)
            if doc_grams:
                grams |= doc_grams
            elif doc.strip():
                shorts.add(_norm_whole(doc))  # shorter than n: index the whole item
        if not grams and not shorts:
            # nothing indexable: claiming it was "checked" would be a vacuous pass
            unmeasured[name] = "benchmark produced zero n-grams and zero items"
            continue
        benchmark_sets[name] = grams
        short_items[name] = shorts

    hits_by_benchmark: dict[str, int] = {name: 0 for name in benchmark_sets}
    hit_records = 0

    for rec in counted(records, stats):
        if not isinstance(rec, dict):
            stats.drop("non_dict_record")
            continue
        text = primary_text(rec)
        grams = _hashed_ngrams(text, n)
        whole = _norm_whole(text)
        # A record shorter than n has no n-grams; compare it whole instead of
        # letting a leaked short item pass as structurally clean.
        hits = [name for name, grams_set in benchmark_sets.items()
                if (grams and grams & grams_set) or whole in short_items.get(name, set())]
        if hits:
            hit_records += 1
            for name in hits:
                hits_by_benchmark[name] += 1
            if action == "drop":
                stats.drop("decontam")
                continue
            meta = rec.setdefault("meta", {})
            if isinstance(meta, dict):
                meta["decontam"] = {"hits": sorted(hits)}
            stats.modified["decontam_flagged"] += 1
        stats.records_out += 1
        yield rec

    stats.extra["hits_by_benchmark"] = hits_by_benchmark
    stats.extra["decontam_hits"] = hit_records
    stats.extra["benchmarks_checked"] = sorted(benchmark_sets)
    stats.extra["unmeasured_benchmarks"] = sorted(unmeasured)
    if unmeasured:
        stats.extra["unmeasured_reasons"] = unmeasured


register_op(FunctionOp("decontam", _decontam_op, CONFIG_SCHEMA))
