from __future__ import annotations
"""Atlas P3 routing / C13: a machine-readable trigger registry.
Compiled from frontmatter, so routing does not hinge on prose alone.
Pure stdlib only, no model calls (C5 increments I1+I2).
"""

import hashlib
import json
import re
from collections import Counter

SCHEMA = "fskills.trigger_registry/1"

STOPWORDS = frozenset((
    "the a an to of for and or with on in is it this that my me i use using run "
    "do not be by from as at any all can will so then into per via its your you we"
).split())

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._+-]*")
_DO_NOT_RE = re.compile(r"Do not use", re.IGNORECASE)
_SKILL_RE = re.compile(r"\(([a-z0-9][a-z0-9-]+)\)")
_CLAUSE_RE = re.compile(r",|\bor\b|\bfor\b", re.IGNORECASE)


def stem(word: str) -> str:
    """A light suffix normalizer so serve/serving/served and quantize/quantized meet (no external stemmer)."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3 and not word.endswith("ss"):
            word = word[: -len(suffix)]
            break
    return word[:-1] if word.endswith("e") and len(word) > 3 else word


def tokens(text: str) -> list[str]:
    """Lowercase, stemmed word tokens in order, minus stopwords and pure numbers."""
    out = []
    for tok in _TOKEN_RE.findall((text or "").lower()):
        if tok not in STOPWORDS and not tok.isdigit():
            out.append(stem(tok))
    return out


def negatives(description: str) -> list[dict]:
    """Redirect phrases parsed from the 'Do NOT use ...' clause.

    Every '(<skill-name>)' parenthetical yields
    {"phrase": words since the previous comma/"or"/"for"/start,
     "route_to": skill-name}; [] when no such clause exists.
    """
    m = _DO_NOT_RE.search(description)
    if m is None:
        return []
    region = description[m.end():]
    out = []
    for sk in _SKILL_RE.finditer(region):
        start = 0
        for cm in _CLAUSE_RE.finditer(region[: sk.start()]):
            start = cm.end()
        out.append({"phrase": region[start:sk.start()].strip(), "route_to": sk.group(1)})
    return out


def _body(description: str) -> str:
    """Description text before the first 'Do NOT use' marker."""
    m = _DO_NOT_RE.search(description)
    return description[: m.start()] if m else description


def _aliases(name: str) -> list[str]:
    """Sorted alias set: full name, stripped name, spaced stripped name."""
    stripped = name
    for prefix in ("fskills-", "fs-"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    return sorted({name, stripped, stripped.replace("-", " ")})


def skill_entry(entry: dict) -> dict:
    """One registry record: name, invocation, keywords, aliases, negatives."""
    name = entry["name"]
    description = entry.get("description") or ""
    counts = Counter(tokens(_body(description)))
    for example in entry.get("when_to_use") or []:
        counts.update(tokens(example))
    keywords = [tok for tok, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:40]]
    return {
        "name": name,
        "invocation": entry.get("invocation") or "model",
        "keywords": keywords,
        "aliases": _aliases(name),
        "negatives": negatives(description),
    }


def _index_fallback(index: list[dict]) -> str:
    blob = json.dumps(sorted(e.get("name", "") for e in index))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compile_registry(index: list[dict]) -> dict:
    """Compile the routing index into a deterministic machine-readable registry."""
    try:
        from foundationskills.agent import routing_eval as _rv
        digest = _rv._index_sha256(index)
    except Exception:
        digest = _index_fallback(index)
    return {
        "schema": SCHEMA,
        "index_sha256": digest,
        "skills": [skill_entry(e) for e in sorted(index, key=lambda e: e.get("name", ""))],
    }
