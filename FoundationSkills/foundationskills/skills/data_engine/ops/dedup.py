
"""``dedup`` op: exact and near-duplicate removal.

- Key (``key: "auto"``): ``text``, else the concatenated message contents,
  else ``prompt`` (+``chosen``). Records without any key text are passed
  through and counted in ``stats.extra["records_without_key"]`` (they cannot
  be deduplicated, so there is no evidence either way).
- Exact: sha256 of a normalized key (lowercase, punctuation stripped,
  whitespace collapsed).
- Near: builtin MinHash (word n-gram shingles, hashed with blake2b; a
  deterministic universal-hash permutation family modulo the Mersenne prime
  2**61-1) with LSH banding. When ``near.backend == "datasketch"`` the
  datasketch implementation is used instead; if it is not importable we fall
  back to the builtin and set ``stats.extra["datasketch_fallback"]`` (a missing
  optional dep is never silently treated as success).
- ``scope: "sample"`` additionally dedups paragraphs across the corpus
  (first occurrence wins; later duplicates are removed from the record text
  and counted in ``stats.modified["dup_paragraphs_removed"]``).
- Streaming with LSH state; memory is O(#records x bands).

``stats.extra``: ``exact_dupes``, ``near_dupes``, ``dedup_rate``.
Exports :func:`minhash_signature` and :func:`jaccard_estimate` for reuse and
testing.
"""
from __future__ import annotations

import hashlib
import re
import string
from typing import Any, Iterable, Iterator, Sequence

from foundationskills.skills.data_engine.ops.base import FunctionOp, OpStats, counted, register_op


CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "exact": {"type": "boolean"},
        "near": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "num_perm": {"type": "integer", "minimum": 8},
                "bands": {"type": "integer", "minimum": 1},
                "ngram": {"type": "integer", "minimum": 1},
                "threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "backend": {"enum": ["builtin", "datasketch"]},
            },
        },
        "scope": {"enum": ["document", "sample"]},
        "key": {"enum": ["auto", "text"]},
    },
}

_MHASH_PRIME = (1 << 61) - 1
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


def _normalize_key(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCT_TABLE).split())


def _record_key(rec: dict, key_cfg: str) -> str:
    text = rec.get("text")
    if isinstance(text, str) and text.strip():
        return text
    if key_cfg == "text":
        return ""
    messages = rec.get("messages")
    if isinstance(messages, list):
        joined = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        if joined.strip():
            return joined
    parts = [rec.get("prompt"), rec.get("chosen")]
    return "\n".join(p for p in parts if isinstance(p, str))


def _shingles(text: str, ngram: int) -> list[str]:
    """Word n-gram shingles over the normalized text; char n-grams when the
    text has too few tokens (e.g. unsegmented CJK)."""
    tokens = _normalize_key(text).split()
    if not tokens:
        return []
    if len(tokens) >= max(2, ngram):
        return [" ".join(tokens[i:i + ngram]) for i in range(len(tokens) - ngram + 1)]
    if len(tokens) > 1:
        return [" ".join(tokens)]
    compact = tokens[0]
    n = max(1, min(ngram, len(compact)))
    return [compact[i:i + n] for i in range(len(compact) - n + 1)]


def _shingle_hashes(text: str, ngram: int) -> set[int]:
    return {
        int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big") % _MHASH_PRIME
        for gram in _shingles(text, ngram)
    }


def _permutation_coeffs(index: int) -> tuple[int, int]:
    """Deterministic permutation family h -> (a_i*h + b_i) mod prime."""
    a = (0x9E3779B97F4A7C15 * (index + 1) + 0x2545F4914F6CDD1D) % _MHASH_PRIME
    b = (0xC2B2AE3D27D4EB4F * (index + 1) + index) % _MHASH_PRIME
    return (a or 1), b


def minhash_signature(text: str, num_perm: int, ngram: int) -> tuple[int, ...]:
    """MinHash signature of the text's shingles (builtin permutation family)."""
    hashes = _shingle_hashes(text, ngram)
    if not hashes:
        return tuple(_MHASH_PRIME for _ in range(num_perm))
    signature: list[int] = []
    for i in range(num_perm):
        a, b = _permutation_coeffs(i)
        best = _MHASH_PRIME
        for h in hashes:
            value = (a * h + b) % _MHASH_PRIME
            if value < best:
                best = value
        signature.append(best)
    return tuple(signature)


def jaccard_estimate(a: Sequence[int], b: Sequence[int]) -> float:
    """Estimated Jaccard similarity from two MinHash signatures."""
    if not a or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def _band_key(part: Sequence[int]) -> bytes:
    return hashlib.blake2b(repr(tuple(part)).encode("utf-8"), digest_size=12).digest()


def _dedup_paragraphs(rec: dict, seen_paragraphs: set[str], stats: OpStats) -> None:
    text = rec.get("text")
    if not isinstance(text, str):
        return
    kept: list[str] = []
    removed = 0
    for para in _PARA_SPLIT_RE.split(text):
        norm = _normalize_key(para)
        if not norm:
            if para.strip():
                kept.append(para)
            continue
        if norm in seen_paragraphs:
            removed += 1
            continue
        seen_paragraphs.add(norm)
        kept.append(para)
    if removed:
        stats.modified["dup_paragraphs_removed"] += removed
        rec["text"] = "\n\n".join(kept)


def _dedup_op(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    exact_on = bool(cfg.get("exact", True))
    near_cfg = cfg.get("near") or {}
    near_on = bool(near_cfg.get("enabled", True))
    num_perm = int(near_cfg.get("num_perm", 128))
    bands = max(1, int(near_cfg.get("bands", 16)))
    ngram = int(near_cfg.get("ngram", 5))
    threshold = float(near_cfg.get("threshold", 0.8))
    scope = str(cfg.get("scope", "document"))
    key_cfg = str(cfg.get("key", "auto"))

    rows = max(1, num_perm // bands)
    seen_exact: set[str] = set()
    band_buckets: dict[tuple[int, bytes], list[tuple[int, ...]]] = {}
    seen_paragraphs: set[str] = set()
    stats.extra["records_without_key"] = 0

    use_datasketch = near_on and near_cfg.get("backend") == "datasketch"
    ds_minhash_cls = ds_lsh_cls = None
    ds_lsh = None
    ds_store: dict[str, Any] = {}
    if use_datasketch:
        try:
            from datasketch import MinHash as _DSMinHash  # type: ignore
            from datasketch import MinHashLSH as _DSLSH  # type: ignore

            ds_minhash_cls, ds_lsh_cls = _DSMinHash, _DSLSH
            ds_lsh = ds_lsh_cls(threshold=threshold, num_perm=num_perm)
            stats.backend = "datasketch"
        except ImportError:
            use_datasketch = False
            stats.extra["datasketch_fallback"] = (
                "datasketch not installed; builtin MinHash used (permutation family approximation)"
            )

    for rec in counted(records, stats):
        if not isinstance(rec, dict):
            stats.drop("non_dict_record")
            continue
        key_text = _record_key(rec, key_cfg)
        norm = _normalize_key(key_text)
        if not norm:
            stats.extra["records_without_key"] += 1
            stats.records_out += 1
            yield rec
            continue

        if exact_on:
            digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            if digest in seen_exact:
                stats.drop("exact_dup")
                continue
            seen_exact.add(digest)

        if near_on:
            if use_datasketch and ds_minhash_cls is not None and ds_lsh is not None:
                m = ds_minhash_cls(num_perm=num_perm)
                for gram in set(_shingles(norm, ngram)):
                    m.update(gram.encode("utf-8"))
                dup = False
                for cand_key in ds_lsh.query(m):
                    other = ds_store[cand_key]
                    va = list(getattr(m, "hashvalues", []))
                    vb = list(getattr(other, "hashvalues", []))
                    n = min(len(va), len(vb))
                    est = (sum(1 for i in range(n) if va[i] == vb[i]) / n) if n else 0.0
                    if est >= threshold:
                        dup = True
                        break
                if dup:
                    stats.drop("near_dup")
                    continue
                ds_key = f"rec-{stats.records_in}"
                ds_store[ds_key] = m
                ds_lsh.insert(ds_key, m)
            else:
                signature = minhash_signature(norm, num_perm, ngram)
                dup = False
                for band in range(bands):
                    part = signature[band * rows:(band + 1) * rows]
                    for other in band_buckets.get((band, _band_key(part)), ()):
                        if jaccard_estimate(signature, other) >= threshold:
                            dup = True
                            break
                    if dup:
                        break
                if dup:
                    stats.drop("near_dup")
                    continue
                for band in range(bands):
                    part = signature[band * rows:(band + 1) * rows]
                    band_buckets.setdefault((band, _band_key(part)), []).append(signature)

        if scope == "sample":
            _dedup_paragraphs(rec, seen_paragraphs, stats)

        stats.records_out += 1
        yield rec

    exact_dupes = stats.dropped.get("exact_dup", 0)
    near_dupes = stats.dropped.get("near_dup", 0)
    total_dupes = exact_dupes + near_dupes
    stats.extra["exact_dupes"] = exact_dupes
    stats.extra["near_dupes"] = near_dupes
    stats.extra["dedup_rate"] = (total_dupes / stats.records_in) if stats.records_in else 0.0


register_op(FunctionOp("dedup", _dedup_op, CONFIG_SCHEMA))
