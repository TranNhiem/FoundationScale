
"""Public dataset catalog discovery.

The catalog is a static, best-effort knowledge file; licenses are as known at
authoring time and every entry marked ``verify`` must be confirmed before
redistribution. Discovery is deterministic: filter, score, stable sort.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any

import yaml


class CatalogError(ValueError):
    """Raised when the catalog file is missing or malformed."""


@lru_cache(maxsize=1)
def load_catalog() -> list[dict[str, Any]]:
    """Load foundationskills/skills/data_engine/catalog/public_datasets.yaml."""
    rel = resources.files("foundationskills.skills.data_engine").joinpath(
        "catalog", "public_datasets.yaml"
    )
    path = str(rel)
    try:
        with rel.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise CatalogError(f"public dataset catalog not found: {path}") from exc
    if not isinstance(raw, list) or not raw:
        raise CatalogError(f"catalog {path} must be a non-empty YAML list")
    entries: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("id"):
            raise CatalogError(f"catalog {path}: every entry needs an 'id'")
        entries.append(dict(item))
    return entries


def discover(
    *,
    goal: str | None,
    stage: str | None,
    formats: list[str] | None,
    domain: str | None,
    languages: list[str] | None,
    license_ok: list[str] | None = None,
    k: int = 8,
) -> list[dict[str, Any]]:
    """Rank catalog entries for a need.

    Hard filters: ``formats`` (entry must share at least one format), and
    ``license_ok`` when given (an entry survives iff its license is in the
    allowlist or is ``verify``, the latter kept but flagged). ``languages``
    filters to entries sharing a language or marked multilingual. Scoring is
    additive and every point is reported in the returned ``reasons``.
    """
    want_formats = {str(f) for f in formats or []}
    want_langs = {str(l) for l in languages or []}
    allow = {str(l) for l in license_ok} if license_ok is not None else None

    scored: list[tuple[int, dict[str, Any], list[str]]] = []
    for entry in load_catalog():
        entry_formats = {str(f) for f in entry.get("formats") or []}
        if want_formats and not (want_formats & entry_formats):
            continue
        license_value = str(entry.get("license") or "verify")
        if allow is not None and license_value not in allow and license_value != "verify":
            continue
        entry_langs = {str(l) for l in entry.get("languages") or []}
        if want_langs and "multilingual" not in entry_langs and not (want_langs & entry_langs):
            continue

        score = 0
        reasons: list[str] = []
        if want_formats & entry_formats:
            score += 2
            reasons.append(f"format match: {sorted(want_formats & entry_formats)}")
        if stage and stage in entry_formats:
            score += 3
            reasons.append(f"supports stage {stage!r}")
        if goal and goal in {str(g) for g in entry.get("goals") or []}:
            score += 4
            reasons.append(f"goal match: {goal!r}")
        if domain:
            entry_domains = {str(d) for d in entry.get("domains") or []}
            if domain in entry_domains:
                score += 3
                reasons.append(f"domain match: {domain!r}")
            elif "general" in entry_domains or "web" in entry_domains:
                score += 1
                reasons.append("general-domain data usable as replay/mixture filler")
        if license_value == "verify":
            reasons.append("license 'verify': confirm on the source page before use")
        if allow is not None and license_value in allow:
            reasons.append(f"license {license_value!r} in allowlist")
        if not entry.get("hf_id"):
            score -= 2
            reasons.append("no HF id: placeholder entry describing a data GAP, not a corpus")
        scored.append((score, entry, reasons))

    scored.sort(key=lambda triple: (-triple[0], str(triple[1]["id"])))
    out: list[dict[str, Any]] = []
    for score, entry, reasons in scored[: max(int(k), 0)]:
        row = dict(entry)
        row["score"] = score
        row["reasons"] = reasons
        out.append(row)
    return out
