
"""Readiness report: measured verdict on whether a dataset is FS-ready.

``build_readiness`` consumes the dataset payload, the per-op stats dicts
(:meth:`OpStats.to_dict`), and a requirements dict, and emits a payload valid
against ``schemas/artifacts/readiness_report.json``. Doctrine: a check that was
not run is ``passed=None`` (UNMEASURED), never a silent PASS; absence of
evidence is UNMEASURED. Verdict: RED if any check is false, UNMEASURED if any
is null, else PASS.

Spec-silent choices, documented here:
- Requirements defaults: ``max_truncation_rate`` 0.02, ``require_dedup``
  True, ``require_decontam`` False, ``max_pii_remaining`` 0.
- DE-RDY-005 recognises the decontam op's ``extra["hits"]`` (falling back to
  the sum of its dropped counts), ``extra["action"]`` ("remove"|"flag"), and
  ``extra["unmeasured_benchmarks"]`` (a non-empty list downgrades to null).
- DE-RDY-006 reads ``extra["pii_remaining"]`` from the clean op; no clean op
  (or no recorded figure) means PII is unmeasured, i.e. null.
- DE-RDY-008 is null for non-conversational targets or when no template
  evidence exists; a template_fallback finding for sft/mm_sft is a failure.
- DE-RDY-009 reads the first record of every shard on disk; unreadable or
  recordless shards make the check null, missing columns make it false.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["build_readiness", "render_markdown"]

_CONVERSATIONAL = ("sft", "mm_sft")


def _ops_by_name(stats: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for s in stats or []:
        name = s.get("name")
        if isinstance(name, str):
            out.setdefault(name, []).append(s)
    return out


def _first(by: dict[str, list[dict]], name: str) -> dict | None:
    found = by.get(name)
    return found[0] if found else None


def _extra(op: dict | None) -> dict:
    if not op:
        return {}
    extra = op.get("extra")
    return extra if isinstance(extra, dict) else {}


def _domain_breakdown(by: dict[str, list[dict]]) -> dict[str, int]:
    for name in ("tokenize", "mix", "ingest", "clean", "dedup", "decontam", "format"):
        counts = _extra(_first(by, name)).get("domain_counts")
        if isinstance(counts, dict) and counts:
            return dict(counts)
    for ops in by.values():
        for op in ops:
            counts = _extra(op).get("domain_counts")
            if isinstance(counts, dict) and counts:
                return dict(counts)
    return {}


def _expected_columns(fmt: str | None, fs_columns: dict) -> set[str]:
    fs = fs_columns if isinstance(fs_columns, dict) else {}
    text_col = fs.get("text_column") or "text"
    base = {"id", "meta"}
    if fmt in ("pretrain", "cpt"):
        return base | {text_col}
    if fmt == "sft":
        return base | {"messages", text_col}
    if fmt == "mm_sft":
        return base | {"messages", text_col, fs.get("image_column") or "image"}
    if fmt == "preference":
        return base | {"prompt", "chosen", "rejected"}
    if fmt == "rl":
        cols = base | {"conversations"}
        gold_key = fs.get("gold_key")
        if gold_key:
            cols.add(str(gold_key))
        return cols
    return base | {text_col}


def _check(rule_id: str, passed: bool | None, detail: str) -> dict:
    return {"rule_id": rule_id, "passed": passed, "detail": detail}


def _first_shard_record(path: str) -> dict | None:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                value = json.loads(text)
                return value if isinstance(value, dict) else None
    return None


def _readiness_009(dataset: dict) -> dict:
    fmt = dataset.get("format")
    expected = _expected_columns(fmt if isinstance(fmt, str) else None, dataset.get("fs_columns") or {})
    problems: list[str] = []
    unreadable: list[str] = []
    saw_record = False
    for shard in dataset.get("shards") or []:
        if not isinstance(shard, dict):
            unreadable.append("<malformed shard entry>")
            continue
        try:
            if int(shard.get("records", 1)) == 0:
                continue
        except Exception:
            pass
        path = shard.get("path")
        record: dict | None = None
        try:
            if isinstance(path, str):
                record = _first_shard_record(path)
        except Exception as exc:  # noqa: BLE001 - unreadable shard is unmeasured, not pass
            unreadable.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        if record is None:
            unreadable.append(f"{path}: no JSON record found")
            continue
        saw_record = True
        missing = sorted(expected - set(record.keys()))
        if missing:
            problems.append(f"{path}: first record missing columns {missing}")
    if problems:
        return _check("DE-RDY-009", False, "; ".join(problems))
    if unreadable or not saw_record:
        detail = "; ".join(unreadable) if unreadable else "no shard records to inspect"
        return _check("DE-RDY-009", None, f"schema unmeasured: {detail}")
    return _check("DE-RDY-009", True, f"first record of every shard carries the fs columns {sorted(expected)}")


def fmt_op_has_tokenizer(fmt_op: dict | None) -> bool:
    return bool(fmt_op) and bool(((fmt_op or {}).get("extra") or {}).get("template_sources", {}).get("tokenizer"))


def build_readiness(dataset: dict, stats: list[dict], requirements: dict) -> dict:
    """Build a readiness_report payload for ``dataset`` from per-op ``stats``.

    requirements keys: target_format, min_tokens, min_records,
    max_truncation_rate (0.02), require_dedup (True), require_decontam (False),
    max_pii_remaining (0), chat_template_family (str|null).
    """
    reqs = dict(requirements or {})
    by = _ops_by_name(list(stats or []))
    fmt = dataset.get("format") if isinstance(dataset, dict) else None
    conv = fmt in _CONVERSATIONAL

    tok = _first(by, "tokenize")
    dedup = _first(by, "dedup")
    decontam = _first(by, "decontam")
    clean = _first(by, "clean")
    fmt_op = _first(by, "format")

    tok_extra = _extra(tok)
    num_tokens = dataset.get("num_tokens")
    if num_tokens is None:
        num_tokens = tok_extra.get("num_tokens")
    approximate = bool(tok) and (tok.get("backend") == "approx" or bool(tok_extra.get("approximate")))

    truncation_rate = tok_extra.get("truncation_rate")
    dedup_rate: float | None = None
    if dedup is not None:
        records_in = dedup.get("records_in") or 0
        dropped_total = sum((dedup.get("dropped") or {}).values())
        dedup_rate = (dropped_total / records_in) if records_in else None
    decontam_hits: int | None = None
    if decontam is not None:
        dec_extra = _extra(decontam)
        hits = dec_extra.get("hits", dec_extra.get("decontam_hits"))
        decontam_hits = int(hits) if hits is not None else int(sum((decontam.get("dropped") or {}).values()))
    pii_remaining: Any = _extra(clean).get("pii_remaining") if clean is not None else None

    stats_section = {
        "num_records": int(dataset.get("num_records") or 0),
        "num_tokens": int(num_tokens) if num_tokens is not None else None,
        "length_hist": tok_extra.get("length_hist"),
        "domain_breakdown": _domain_breakdown(by),
        "dedup_rate": dedup_rate,
        "decontam_hits": decontam_hits,
        "pii_remaining": pii_remaining,
        "truncation_rate": truncation_rate,
    }
    if approximate:
        stats_section["token_counts_approximate"] = True

    checks: list[dict] = []

    # DE-RDY-001: format matches target_format
    target = reqs.get("target_format")
    if target is None:
        checks.append(_check("DE-RDY-001", None, "target_format not declared in requirements"))
    else:
        ok = fmt == target
        checks.append(_check("DE-RDY-001", ok, f"dataset format {fmt!r} vs required {target!r}"))

    # DE-RDY-002: num_tokens >= min_tokens (approximate counts are unmeasured)
    min_tokens = reqs.get("min_tokens")
    if num_tokens is None:
        checks.append(_check("DE-RDY-002", None, "num_tokens unknown (no tokenize op ran)"))
    elif approximate:
        checks.append(
            _check("DE-RDY-002", None, "token counts are approximate (backend=approx); min-tokens check unmeasured")
        )
    elif min_tokens is None:
        checks.append(_check("DE-RDY-002", True, f"num_tokens={num_tokens}; no minimum declared"))
    else:
        ok = num_tokens >= min_tokens
        checks.append(_check("DE-RDY-002", ok, f"num_tokens={num_tokens} vs min_tokens={min_tokens}"))

    # DE-RDY-003: num_records >= min_records
    min_records = reqs.get("min_records")
    num_records = stats_section["num_records"]
    if min_records is None:
        checks.append(_check("DE-RDY-003", True, f"num_records={num_records}; no minimum declared"))
    else:
        ok = num_records >= min_records
        checks.append(_check("DE-RDY-003", ok, f"num_records={num_records} vs min_records={min_records}"))

    # DE-RDY-004: dedup ran
    require_dedup = bool(reqs.get("require_dedup", True))
    if dedup is not None:
        rate_txt = f"{dedup_rate:.4f}" if dedup_rate is not None else "unmeasured (0 input records)"
        checks.append(_check("DE-RDY-004", True, f"dedup ran; drop rate {rate_txt}"))
    elif require_dedup:
        checks.append(_check("DE-RDY-004", False, "require_dedup is set but no dedup op ran"))
    else:
        checks.append(_check("DE-RDY-004", True, "dedup not required by requirements"))

    # DE-RDY-005: decontam ran, no unmeasured benchmarks, no hits left flagged
    require_decontam = bool(reqs.get("require_decontam", False))
    if decontam is None:
        if require_decontam:
            checks.append(_check("DE-RDY-005", False, "require_decontam is set but no decontam op ran"))
        else:
            checks.append(_check("DE-RDY-005", True, "decontam not required by requirements"))
    else:
        dec_extra = _extra(decontam)
        unmeasured_benchmarks = dec_extra.get("unmeasured_benchmarks") or []
        action = dec_extra.get("action", "remove")
        if unmeasured_benchmarks:
            checks.append(
                _check("DE-RDY-005", None, f"decontam left benchmarks unmeasured: {list(unmeasured_benchmarks)}")
            )
        elif decontam_hits and action == "flag":
            checks.append(
                _check("DE-RDY-005", False, f"{decontam_hits} decontamination hit(s) remain with action=flag")
            )
        else:
            tail = f"{decontam_hits} hit(s) removed" if decontam_hits else "0 hits"
            checks.append(_check("DE-RDY-005", True, f"decontam ran: {tail}"))

    # DE-RDY-006: pii_remaining <= max (unmeasured when no clean op ran)
    max_pii = reqs.get("max_pii_remaining", 0)
    if clean is None:
        checks.append(_check("DE-RDY-006", None, "no clean op ran; PII unmeasured"))
    elif pii_remaining is None:
        checks.append(_check("DE-RDY-006", None, "clean op ran but reported no pii_remaining figure"))
    else:
        ok = pii_remaining <= max_pii
        checks.append(_check("DE-RDY-006", ok, f"pii_remaining={pii_remaining} vs max_pii_remaining={max_pii}"))

    # DE-RDY-007: truncation_rate <= max
    max_trunc = reqs.get("max_truncation_rate", 0.02)
    if truncation_rate is None:
        checks.append(_check("DE-RDY-007", None, "truncation_rate unknown (no tokenize op ran)"))
    else:
        ok = truncation_rate <= max_trunc
        checks.append(_check("DE-RDY-007", ok, f"truncation_rate={truncation_rate} vs max={max_trunc}"))

    # DE-RDY-008: the chat template ACTUALLY applied (format op's measured
    # template_source), never a default: an absent source is unmeasured (rv10).
    req_family = reqs.get("chat_template_family")
    fmt_extra = _extra(fmt_op)
    if not conv:
        checks.append(_check("DE-RDY-008", True, f"not applicable: format {fmt!r} carries no chat template"))
    elif fmt_op is None or not fmt_extra.get("template_source"):
        checks.append(_check("DE-RDY-008", None, "format op recorded no template_source; template unmeasured"))
    else:
        source = str(fmt_extra["template_source"])
        modes = dict(fmt_extra.get("reasoning_render") or {})
        problems: list[str] = []
        if fmt_extra.get("template_fallback") or source == "generic":
            problems.append("generic ROLE: fallback used; the family's chat template was NOT applied")
        if modes.get("inline_fallback") and fmt_extra.get("tokenizer_error") is None and fmt_op_has_tokenizer(fmt_op):
            problems.append(f"{modes['inline_fallback']} reasoning trace(s) inlined as <think> despite a tokenizer")
        if req_family is not None and source not in ("tokenizer", str(req_family)):
            problems.append(f"template {source!r} does not match required family {req_family!r}")
        declared = dataset.get("chat_template_family")
        if req_family is not None and declared is not None and declared != req_family:
            problems.append(f"dataset declares chat_template_family {declared!r} but {req_family!r} is required")
        detail = f"template_source={source!r}" + (f"; reasoning_render={modes}" if modes else "")
        if modes.get("lost"):
            detail += f"; {modes['lost']} record(s) dropped because the template lost their reasoning trace"
        checks.append(_check("DE-RDY-008", not problems, "; ".join(problems) or detail))

    # DE-RDY-009: shard schemas carry the fs columns
    checks.append(_readiness_009(dataset))

    # DE-RDY-010: full-sequence SFT loss disclosure (sft/mm_sft only, informational)
    if conv:
        scope = fmt_extra.get("sft_loss_scope")
        checks.append(
            _check(
                "DE-RDY-010",
                True,
                "DISCLOSURE: FS's sft objective trains on the full rendered text with NO assistant-only "
                f"loss masking (measured, FS commit 77bfa65; format op sft_loss_scope={scope or 'full_sequence'}).",
            )
        )

    if any(c["passed"] is False for c in checks):
        verdict = "RED"
    elif any(c["passed"] is None for c in checks):
        verdict = "UNMEASURED"
    else:
        verdict = "PASS"

    dataset_id = dataset.get("dataset_id") or dataset.get("name") or "unknown"
    return {
        "verdict": verdict,
        "dataset_id": str(dataset_id),
        "stats": stats_section,
        "checks": checks,
        "requirements": reqs,
    }


def render_markdown(report: dict) -> str:
    """Render a readiness report as a human-readable Markdown table."""
    lines = [
        f"# Data readiness: {report.get('dataset_id', 'unknown')}",
        "",
        f"**Verdict:** {report.get('verdict', 'UNMEASURED')}",
        "",
        "## Stats",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for key, value in (report.get("stats") or {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Checks", "", "| rule | result | detail |", "|---|---|---|"])
    marks = {True: "PASS", False: "FAIL", None: "UNMEASURED"}
    for check in report.get("checks") or []:
        mark = marks[check.get("passed")]
        detail = str(check.get("detail", "")).replace("|", "\\|")
        lines.append(f"| {check.get('rule_id', '?')} | {mark} | {detail} |")
    lines.append("")
    return "\n".join(lines)
