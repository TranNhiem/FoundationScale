
"""Op ``mix``: sample several components to a shared-ratio mixture.

Config::

    {"components": [
        {"name": str,
         "ratio": float,              # relative target share, renormalized
         "path": str | null,          # a .jsonl file or a directory of *.jsonl shards
         "inline": bool               # default false; true = "the upstream records"
        }, ...],
     "total_records": int | null,     # mixture size in records
     "total_tokens": int | null,      # mixture size in tokens (approximate; see below)
     "seed": int,                     # default 0
     "max_epochs": int                # default 4; upsampling cap per component
    }

Simplifications (documented per the silent-spec rule):

* Components are materialized in memory. ``path`` components read JSONL only;
  a directory means its sorted ``*.jsonl`` shards. Raw kinds (pdf/html/parquet)
  belong to ``ingest``, so pipeline specs that want mixing across raw sources
  should instead consume these components' ingested shards.
* At most ONE component may be ``inline``. It substitutes the upstream iterator
  and is consumed lazily from the op input. With no inline component the input
  iterator is ignored (upstream ops are not pulled at all).
* ``total_tokens`` converts to records via a whitespace-word heuristic
  (~1.3 tokens/word over up to 64 sampled records per component). This is an
  approximation: ``stats.extra["approximate"]`` is set true.
* When ``total_records``/``total_tokens`` are both absent, the mixture total is
  the sum of available component sizes, so oversized components downsample and
  undersized ones upsample to hit the requested ratios.

Everything is deterministic given ``seed`` (``random.Random(f"{seed}:{name}")``;
``random.seed`` on a ``str`` is stable across processes).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Iterator

from foundationskills.skills.data_engine.ops.base import FunctionOp, OpStats, counted, register_op

_NAME = "mix"
_TOKENS_PER_WORD = 1.3  # coarse whitespace heuristic for total_tokens conversion
_SAMPLE_FOR_TOKENS = 64


def _read_component_file(path: Path, stats: OpStats, name: str) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats.drop(f"mix_bad_line:{name}")
                continue
            if isinstance(rec, dict):
                records.append(rec)
            else:
                stats.drop(f"mix_bad_line:{name}")
    return records


def _load_component(component: dict, records: Iterable[dict], stats: OpStats) -> list[dict] | None:
    """Load one component. ``None`` means the iterator is consumed by the caller."""
    name = str(component.get("name", "component"))
    if component.get("inline"):
        return None
    raw = component.get("path")
    if not raw:
        stats.drop(f"mix_no_source:{name}")
        return []
    path = Path(str(raw))
    if path.is_file():
        return _read_component_file(path, stats, name)
    if path.is_dir():
        out: list[dict] = []
        for shard in sorted(path.glob("*.jsonl")):
            out.extend(_read_component_file(shard, stats, name))
        return out
    stats.drop(f"mix_bad_line:{name}") if False else stats.drop(f"mix_missing_path:{name}")
    return []


def _avg_tokens(records: list[dict]) -> float:
    sample = records[:_SAMPLE_FOR_TOKENS] or [{}]
    words = 0
    for rec in sample:
        text = str(rec.get("text") or rec.get("prompt") or "")
        words += max(len(text.split()), 1)
    return max((words / len(sample)) * _TOKENS_PER_WORD, 1.0)


def _targets(sizes: dict[str, int], avgtok: dict[str, float], cfg: dict, stats: OpStats) -> tuple[dict[str, int], int]:
    names = list(sizes)
    if cfg.get("total_records") is not None:
        total = int(cfg["total_records"])
        return {}, total
    if cfg.get("total_tokens") is not None:
        stats.extra["approximate"] = True  # token->record conversion is a word heuristic
        wanted = {n: 0 for n in names}
        return wanted, int(cfg["total_tokens"])
    return {}, sum(sizes.values())


def _mix(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    components = list(cfg.get("components") or [])
    if not components:
        raise ValueError("mix requires a non-empty 'components' list")
    inline = [c for c in components if c.get("inline")]
    if len(inline) > 1:
        raise ValueError("mix supports at most one component with inline: true (the upstream records)")

    seed = int(cfg.get("seed", 0))
    max_epochs = int(cfg.get("max_epochs", 4))
    if max_epochs < 1:
        raise ValueError("mix max_epochs must be >= 1")

    names = [str(c.get("name", "component")) for c in components]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:  # rv35: pools are keyed by name; a collision would sample the wrong data
        raise ValueError(f"mix components need unique names; duplicated: {duplicates}")
    pools: dict[str, list[dict]] = {}
    for component in components:
        name = str(component.get("name", "component"))
        loaded = _load_component(component, records, stats)
        if loaded is None:  # inline: drain the upstream iterator once
            loaded = list(counted(records, stats))
        pools[name] = loaded
        stats.extra.setdefault("components", {})[name] = {"available": len(loaded)}

    total_ratio = sum(float(c.get("ratio", 1.0)) for c in components)
    if total_ratio <= 0:
        raise ValueError("mix component ratios must sum to a positive number")

    avgtok = {n: _avg_tokens(r) for n, r in pools.items()}
    token_targets, total = _targets({n: len(r) for n, r in pools.items()}, avgtok, cfg, stats)
    use_token_targets = bool(token_targets)

    capped: dict[str, dict] = {}
    kept_lists: list[tuple[str, list[dict]]] = []
    for component in components:
        name = str(component.get("name", "component"))
        pool = pools[name]
        if not pool:
            stats.drop(f"mix_empty_component:{name}")
            continue
        share = float(component.get("ratio", 1.0)) / total_ratio
        if use_token_targets:
            target = max(int(round((share * total) / avgtok[name])), 0)
        else:
            target = max(int(round(share * total)), 0)
        cap = len(pool) * max_epochs
        kept_n = min(target, cap)
        if kept_n < target:
            capped[name] = {
                "requested": int(target),
                "kept": int(kept_n),
                "max_epochs": max_epochs,
                "available": len(pool),
            }
        rng = random.Random(f"{seed}:{name}")
        indices: list[int] = []
        while len(indices) < kept_n:
            epoch = list(range(len(pool)))
            rng.shuffle(epoch)
            indices.extend(epoch)
        chosen = []
        for idx in indices[:kept_n]:
            rec = dict(pool[idx])
            meta = dict(rec.get("meta") or {})
            meta["mix_component"] = name
            rec["meta"] = meta
            chosen.append(rec)
        kept_lists.append((name, chosen))
        stats.extra["components"][name].update({"target": int(target), "kept": len(chosen)})

    combined: list[dict] = []
    for _name, chosen in kept_lists:
        combined.extend(chosen)
    random.Random(f"{seed}:__order__").shuffle(combined)

    if capped:
        stats.extra["capped"] = capped
    total_kept = len(combined)
    stats.extra["realized"] = {
        name: (len(chosen) / total_kept if total_kept else 0.0) for name, chosen in kept_lists
    }
    stats.extra["seed"] = seed
    stats.extra["max_epochs"] = max_epochs
    for rec in combined:
        stats.records_out += 1
        yield rec


mix_op = FunctionOp(_NAME, _mix, config_schema={"type": "object", "required": ["components"]})
register_op(mix_op)
