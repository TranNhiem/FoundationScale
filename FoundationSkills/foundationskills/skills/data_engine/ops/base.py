"""Operator contract for the Data Engine pipeline.

A pipeline is an ordered list of ``{"op": name, "config": {...}}`` steps. The
first step is always ``ingest`` (a source: it ignores its input iterator); every
later step is a streaming transformer ``records -> records``. Each op counts
what it saw, kept and dropped (by reason) into its own ``OpStats`` so the
readiness report can say where data went instead of only how much is left.

Canonical record (keys absent when not applicable)::

    {"id": str,
     "text": str,                         # raw corpus text / rendered SFT text
     "messages": [{"role", "content"}],   # conversational data
     "prompt": str, "chosen": str, "rejected": str,   # preference
     "answer": str,                       # verifiable gold answer (RL)
     "images": [str],                     # image paths/URIs (multimodal)
     "meta": {"source": str, "lang": str|None, "domain": str|None,
              "license": str|None, ...}}
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Protocol


@dataclass
class OpStats:
    """Per-op accounting. ``dropped`` is keyed by a short reason slug."""

    name: str
    records_in: int = 0
    records_out: int = 0
    dropped: Counter = field(default_factory=Counter)
    modified: Counter = field(default_factory=Counter)
    extra: dict[str, Any] = field(default_factory=dict)
    backend: str = "builtin"   # which implementation ran (e.g. "datatrove", "builtin")

    def drop(self, reason: str) -> None:
        self.dropped[reason] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "records_in": self.records_in,
            "records_out": self.records_out,
            "dropped": dict(self.dropped),
            "modified": dict(self.modified),
            "extra": self.extra,
            "backend": self.backend,
        }


class Op(Protocol):
    name: str
    config_schema: dict[str, Any]

    def __call__(self, records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]: ...


OPS: dict[str, Op] = {}


def register_op(op: Op) -> Op:
    if op.name in OPS:
        raise ValueError(f"duplicate data op {op.name!r}")
    OPS[op.name] = op
    return op


class FunctionOp:
    """Adapter: wrap a generator function ``fn(records, cfg, stats)`` as an Op."""

    def __init__(self, name: str, fn: Callable[[Iterable[dict], dict, OpStats], Iterator[dict]],
                 config_schema: dict[str, Any] | None = None) -> None:
        self.name = name
        self._fn = fn
        self.config_schema = config_schema or {"type": "object"}

    def __call__(self, records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
        return self._fn(records, cfg, stats)


def counted(records: Iterable[dict], stats: OpStats) -> Iterator[dict]:
    """Wrap an input iterator so ``records_in`` is counted as it is consumed."""
    for rec in records:
        stats.records_in += 1
        yield rec
