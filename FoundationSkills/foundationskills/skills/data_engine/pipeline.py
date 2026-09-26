
"""Streaming pipeline runner for the Data Engine.

Chains ``OPS`` generators lazily (records flow one at a time through the op
graph; only ``mix`` materializes, which it documents), writes shards
atomically (tmp file + ``os.replace``), and records per-op ``stats.json``.

Silent-spec choices:

* ``num_tokens`` comes from the ``tokenize`` op's stats extra (key searched in
  order: ``num_tokens``, ``total_tokens``, ``tokens_out``, ``tokens``). If that
  op marks its stats ``extra["approximate"]`` (a builtin fallback counting
  words), the dataset's ``num_tokens`` is null and the estimate moves to the
  extra field ``num_tokens_approx`` — an approximation must not masquerade as
  a measurement.
* ``schema.columns`` is the union of keys over the first 100 written records.
* Domain counts (``meta.domain``) and shard counts go into a synthetic
  ``pipeline`` stats entry appended to stats.json.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from foundationskills.core.provenance import sha256_file
from foundationskills.core.schema import assert_valid, load_schema, validate
from foundationskills.skills.data_engine.ops import base as _ops_base
from foundationskills.skills.data_engine.ops.base import OpStats, counted
from foundationskills.skills.data_engine.phase2 import PHASE2, Phase2NotImplemented


class PipelineError(ValueError):
    """The pipeline spec is inconsistent or references unknown operators."""


@dataclass
class PipelineResult:
    dataset: dict[str, Any]
    stats: list[dict[str, Any]]
    shards: list[Path]


_SCHEMA_COLUMNS_SAMPLE = 100
_TOKEN_EXTRA_KEYS = ("num_tokens", "total_tokens", "tokens_out", "tokens")
_FS_COLUMNS: dict[str, tuple[str | None, str | None, str | None]] = {
    "pretrain": ("text", None, None),
    "cpt": ("text", None, None),
    "sft": ("text", None, None),
    "mm_sft": ("text", "image", None),
    "preference": (None, None, None),
    "rl": (None, None, "answer"),
}


def _atomic_write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _chain(spec_ops: list[dict[str, Any]]) -> tuple[Any, list[OpStats]]:
    names = []
    for step in spec_ops:
        name = str(step.get("op", ""))
        if name in PHASE2:
            raise Phase2NotImplemented(f"phase-2: {name}")
        names.append(name)
    OPS = _ops_base.OPS  # late-bound: the registry is resolved per run, not frozen at import
    unknown = [n for n in names if n not in OPS]
    if unknown:
        raise PipelineError(
            f"unknown data op(s): {', '.join(unknown)}; known ops: {', '.join(sorted(OPS))}"
        )
    if not names or names[0] != "ingest":
        raise PipelineError("the first pipeline op must be 'ingest' (the source); every later op is a transformer")

    stats: list[OpStats] = []
    records: Iterable[dict] = iter(())
    for step in spec_ops:
        op = _ops_base.OPS[names[len(stats)]]
        op_stats = OpStats(name=op.name)
        stats.append(op_stats)
        records = op(records, dict(step.get("config") or {}), op_stats)
    return records, stats


def _num_tokens(stats: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    """Return (measured, approximate) token counts from the last tokenize op."""
    measured: int | None = None
    approximate: int | None = None
    for entry in stats:
        if entry.get("name") != "tokenize":
            continue
        extra = entry.get("extra") or {}
        value = next((extra[k] for k in _TOKEN_EXTRA_KEYS if isinstance(extra.get(k), (int, float))), None)
        if value is None:
            continue
        if extra.get("approximate"):
            approximate = int(value)
        else:
            measured = int(value)
    return measured, approximate


def run_pipeline(spec: dict[str, Any], out_dir: Path, *, shard_records: int = 50000) -> PipelineResult:
    """Execute a data_pipeline_spec into ``out_dir`` and build the dataset payload.

    Raises ``PipelineError`` for spec problems (unknown op, first op not
    ingest) and ``Phase2NotImplemented`` for phase-2 op names.
    """
    assert_valid(spec, load_schema("artifacts/data_pipeline_spec"), "data_pipeline_spec")
    if shard_records < 1:
        raise PipelineError("shard_records must be >= 1")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Shards get a directory of their own: FS loaders read EVERY .json/.jsonl in
    # a dataset directory, so a stats.json beside the shards is parsed as corpus
    # (measured: RLTrainer refused with "holds a top-level dict").
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    records, op_stats = _chain(list(spec["ops"]))

    shards: list[Path] = []
    shard_rows: list[dict[str, Any]] = []
    columns: set[str] = set()
    domain_counts: dict[str, int] = {}
    num_records = 0
    buffer: list[str] = []
    buffer_count = 0

    def flush() -> None:
        nonlocal buffer, buffer_count
        if not buffer:
            return
        shard_path = shard_dir / f"shard-{len(shards):05d}.jsonl"
        _atomic_write_lines(shard_path, buffer)
        shard_rows.append(
            {"path": str(shard_path), "sha256": sha256_file(shard_path), "records": buffer_count}
        )
        shards.append(shard_path)
        buffer = []
        buffer_count = 0

    pipeline_stats = OpStats(name="pipeline")
    for rec in records:
        if not isinstance(rec, dict):  # a non-strict op passed a non-record through
            pipeline_stats.drop("non_dict_record")
            continue
        if num_records < _SCHEMA_COLUMNS_SAMPLE:
            columns.update(str(k) for k in rec.keys())
        dom = (rec.get("meta") or {}).get("domain") if isinstance(rec.get("meta"), dict) else None
        if dom is not None:
            key = str(dom)
            domain_counts[key] = domain_counts.get(key, 0) + 1
        buffer.append(json.dumps(rec, ensure_ascii=False, sort_keys=True))
        buffer_count += 1
        num_records += 1
        if buffer_count >= shard_records:
            flush()
    flush()

    pipeline_stats.records_out = num_records
    pipeline_stats.extra = {
        "domain_counts": domain_counts,
        "num_shards": len(shards),
        "shard_records": shard_records,
        "out_dir": str(out_dir),
    }
    stats_dicts = [s.to_dict() for s in op_stats] + [pipeline_stats.to_dict()]

    stats_path = out_dir / "stats.json"
    _atomic_write_lines(stats_path, [json.dumps({"stats": stats_dicts}, ensure_ascii=False, sort_keys=True)])

    measured_tokens, approx_tokens = _num_tokens(stats_dicts)
    text_column, image_column, gold_key = _FS_COLUMNS[spec["target_format"]]
    # the format op writes the columns it actually emitted; prefer that measured hint
    fmt_hint = next(((st.get("extra") or {}).get("fs_columns") for st in reversed(stats_dicts)
                     if st.get("name") == "format" and (st.get("extra") or {}).get("fs_columns")), None)
    if isinstance(fmt_hint, dict):
        text_column = fmt_hint.get("text_column", text_column)
        image_column = fmt_hint.get("image_column", image_column)
        gold_key = fmt_hint.get("gold_key", gold_key)
    dataset: dict[str, Any] = {
        "format": spec["target_format"],
        "shards": shard_rows,
        "schema": {"columns": sorted(columns)},
        "num_records": num_records,
        "num_tokens": measured_tokens,
        "tokenizer": spec.get("tokenizer"),
        "chat_template_family": spec.get("chat_template_family"),
        "fs_columns": {
            "text_column": text_column,
            "image_column": image_column,
            "gold_key": gold_key,
        },
        "stats_path": str(stats_path),
        "domain_counts": domain_counts,
        "seed": spec.get("seed"),
    }
    if measured_tokens is None and approx_tokens is not None:
        dataset["num_tokens_approx"] = approx_tokens

    if num_records > 0:
        errors = validate(dataset, load_schema("artifacts/dataset"))
        if errors:
            raise PipelineError("internal error: built dataset payload invalid: " + "; ".join(errors))
    # num_records == 0 yields an intentionally invalid dataset payload (empty
    # shards); the owning skill reports DE-HO-003 instead of writing it.

    return PipelineResult(dataset=dataset, stats=stats_dicts, shards=shards)
