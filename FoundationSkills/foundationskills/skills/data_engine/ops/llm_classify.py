"""Op ``llm_classify``: LLM-based classification of records over named axes.

The "annotate" half of annotate-then-distill (the FineWeb-Edu pattern): each
record is rated on a configurable set of categorical and ordinal axes (domain,
topic, quality, educational value, safety, ...) in ONE chat call that covers
all axes, and the labels land in ``rec["meta"][output_key]``. Optional
``filters`` then drop records whose measured labels fail a predicate.

Config::

    {"backend": {...},            # required; make_backend(cfg["backend"], "llm_classify")
     "axes": [                    # required, >= 1; exactly one of labels/scale per axis
        {"name": str,
         "description": str,
         "labels": [str, ...],    # categorical axis (>= 2 allowed labels)
         # OR
         "scale": [min, max]      # ordinal axis, inclusive integer range
        }, ...],
     "field": str,                # default "text"; see fallbacks below
     "max_chars": int,            # default 6000; input truncated before prompting
     "sample": {"rate": float, "seed": int},   # optional deterministic subset, rate in (0, 1]
     "filters": {axis: {"in": [...]} | {"gte": n} | {"lte": n}},   # optional
     "drop_unmeasured": bool,     # default false; drop records with any None axis
     "output_key": str,           # default "llm_labels"
     "workers": int,              # default 4; bounded-window pool, input order preserved
     "max_calls": int | null,     # call budget; null = unlimited
     "temperature": float,        # default 0.0
     "max_tokens": int,           # default 512
     "seed": int,                 # default 0
     "cache_dir": str | null      # optional on-disk response cache around the backend
    }

Simplifications and doctrines (documented per the silent-spec rule):

* Field fallbacks: with no (or empty) ``field`` value the op classifies the
  concatenated ``"role: content"`` message lines; for preference records it
  uses the prompt plus newline plus chosen. A record with no extractable text
  at all is dropped with reason ``empty_text`` -- there is no document to
  measure, and staying silent would fake a label.
* Never faked: values failing validation (unknown label, non-integral or
  out-of-range ordinal, unparseable reply) become None for that axis and are
  counted under ``stats.extra["invalid"][axis]``; backend failures make every
  axis None and bump ``stats.extra["errors"]``. No heuristic substitutes.
* ``sample.rate`` buckets records on ``sha256(f"{seed}:{record_id}") / 2**64
  < rate``; records without an ``id`` share a single bucket (``""``). Sampled-
  out records pass through completely untouched: no labels key (not even
  None), no provenance; they are counted in ``stats.extra["sampled_out"]``.
* Once ``max_calls`` is exhausted, remaining sampled records get all-None
  labels (counted unmeasured, ``budget_exhausted: true`` set) and NO call is
  made, so they get no ``meta["llm"]`` provenance entry -- nothing touched
  the LLM for them. They are only dropped under ``drop_unmeasured``.
* ``labeled``/``unmeasured``/``distribution`` describe every record the LLM
  rated, INCLUDING records later dropped by filters (they were measured even
  if not emitted). ``sampled_out`` records fall in none of those buckets.
* Case-insensitive categorical matches are normalised to the canonical
  label; ordinals accept ints, integral floats and numeric strings ("4",
  "4.0"). Anything else is invalid, not guessed.
* Deterministic given ``seed`` (temperature defaults to 0.0); calls go over
  ``map_ordered`` so output order always equals input order, and all stats
  mutation happens on the consuming thread.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from foundationskills.skills.data_engine.llm_backend import (
    CachedBackend,
    LLMOpError,
    budget,
    make_backend,
    map_ordered,
    parse_json_object,
    prompt_hash,
)
from foundationskills.skills.data_engine.ops.base import FunctionOp, OpStats, counted, register_op

_NAME = "llm_classify"
_DEFAULT_FIELD = "text"
_DEFAULT_MAX_CHARS = 6000
_DEFAULT_OUTPUT_KEY = "llm_labels"
_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class _Axis:
    """One classification axis: categorical (``labels``) XOR ordinal (``scale``)."""

    name: str
    description: str
    labels: tuple[str, ...] | None = None
    scale: tuple[int, int] | None = None


@dataclass(frozen=True)
class _Filter:
    """A parsed filter: categorical keep-list XOR numeric bounds."""

    axis: str
    keep_in: frozenset[str] | None = None
    gte: float | None = None
    lte: float | None = None


@dataclass
class _Outcome:
    """Worker result; ALL stats mutation happens on the consuming thread."""

    record: dict
    kind: str  # "empty_text" | "unsampled" | "classified"
    labels: dict[str, object] = field(default_factory=dict)
    called: bool = False
    cache_hit: bool = False
    error: bool = False
    truncated: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    invalid: tuple[str, ...] = ()


def _parse_axes(raw: object) -> list[_Axis]:
    """Validate ``config.axes`` loudly; names must be unique."""
    if not isinstance(raw, list) or not raw:
        raise LLMOpError(
            f"{_NAME}: missing input: config.axes (a non-empty list of axis specs)")
    axes: list[_Axis] = []
    seen: set[str] = set()
    for index, spec in enumerate(raw):
        where = f"config.axes[{index}]"
        if not isinstance(spec, dict):
            raise LLMOpError(f"{_NAME}: {where} must be an object with name/description")
        name = str(spec.get("name") or "").strip()
        if not name:
            raise LLMOpError(f"{_NAME}: missing input: {where}.name")
        if name in seen:
            raise LLMOpError(f"{_NAME}: duplicate axis name {name!r} in config.axes")
        seen.add(name)
        has_labels = spec.get("labels") is not None
        has_scale = spec.get("scale") is not None
        if has_labels == has_scale:
            raise LLMOpError(
                f"{_NAME}: {where} needs exactly one of 'labels' or 'scale'")
        description = str(spec.get("description") or "")
        if has_labels:
            labels = spec["labels"]
            ok = (isinstance(labels, list) and len(labels) >= 2
                  and all(isinstance(lab, str) and lab for lab in labels))
            if not ok:
                raise LLMOpError(f"{_NAME}: {where}.labels must be a list of >= 2 strings")
            axes.append(_Axis(name, description, labels=tuple(labels)))
        else:
            scale = spec["scale"]
            ok = (isinstance(scale, list) and len(scale) == 2
                  and all(isinstance(v, int) and not isinstance(v, bool) for v in scale))
            if not ok:
                raise LLMOpError(f"{_NAME}: {where}.scale must be [int_min, int_max]")
            if scale[0] >= scale[1]:
                raise LLMOpError(f"{_NAME}: {where}.scale needs int_min < int_max")
            axes.append(_Axis(name, description, scale=(int(scale[0]), int(scale[1]))))
    return axes


def _parse_filters(raw: object, axes: list[_Axis]) -> list[_Filter]:
    """Validate ``config.filters``; references to unknown axes refuse loudly."""
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise LLMOpError(f"{_NAME}: config.filters must map axis names to filter objects")
    by_name = {axis.name: axis for axis in axes}
    filters: list[_Filter] = []
    for name, spec in raw.items():
        if name not in by_name:
            raise LLMOpError(
                f"{_NAME}: missing input: filters reference unknown axis {name!r}; "
                f"known axes: {sorted(by_name)}")
        axis = by_name[name]
        if not isinstance(spec, dict):
            raise LLMOpError(f"{_NAME}: config.filters[{name!r}] must be an object")
        if axis.labels is not None:
            keep = spec.get("in")
            if (not isinstance(keep, list) or not keep
                    or any(not isinstance(v, str) for v in keep)):
                raise LLMOpError(
                    f"{_NAME}: filters[{name!r}] on a categorical axis needs an "
                    f"'in' list of allowed labels")
            filters.append(_Filter(axis=name, keep_in=frozenset(keep)))
            continue
        bounds: dict[str, float] = {}
        for key in ("gte", "lte"):
            bound = spec.get(key)
            if bound is None:
                continue
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise LLMOpError(f"{_NAME}: filters[{name!r}].{key} must be a number")
            bounds[key] = float(bound)
        if not bounds:
            raise LLMOpError(
                f"{_NAME}: filters[{name!r}] on an ordinal axis needs 'gte' and/or 'lte'")
        filters.append(_Filter(axis=name, gte=bounds.get("gte"), lte=bounds.get("lte")))
    return filters


def _extract_text(rec: dict, field: str) -> str:
    """Return the document text: field, else message lines, else prompt+chosen."""
    value = rec.get(field)
    if isinstance(value, str) and value.strip():
        return value
    messages = rec.get("messages")
    if isinstance(messages, list) and messages:
        lines = []
        for message in messages:
            if isinstance(message, dict):
                lines.append(f"{message.get('role', '?')}: {message.get('content', '')}")
        joined = "\n".join(lines).strip()
        if joined:
            return joined
    prompt = rec.get("prompt") if isinstance(rec.get("prompt"), str) else ""
    chosen = rec.get("chosen") if isinstance(rec.get("chosen"), str) else ""
    if prompt or chosen:
        return f"{prompt}\n{chosen}".strip()
    return ""


def _system_prompt(axes: list[_Axis]) -> str:
    """Render the rubric: every axis, its description and its allowed values."""
    lines = [
        "You are labeling documents for an LLM training-data pipeline.",
        "Rate EVERY document on each axis below, using exactly the allowed values:",
    ]
    for axis in axes:
        description = axis.description or "No description."
        if axis.labels is not None:
            allowed = "one of " + json.dumps(list(axis.labels), ensure_ascii=False)
        else:
            low, high = axis.scale or (0, 0)
            allowed = f"an integer in [{low}, {high}]"
        lines.append(f'- "{axis.name}": {description} Value: {allowed}.')
    keys = ", ".join(json.dumps(axis.name) for axis in axes)
    lines.append(
        f"Reply with ONLY a JSON object {{{keys}}} mapping each axis name to its value. "
        "No prose, no markdown.")
    return "\n".join(lines)


def _in_sample(rec_id: object, seed: int, rate: float) -> bool:
    """Deterministic per-record bucket on sha256(f"{seed}:{id}"); no id shares ''."""
    key = f"{seed}:{rec_id if rec_id is not None else ''}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64) < rate


def _coerce_categorical(value: object, labels: tuple[str, ...]) -> str | None:
    """Exact or case-insensitive label match; anything else is None, never guessed."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if cleaned in labels:
        return cleaned
    folded = cleaned.casefold()
    for label in labels:
        if label.casefold() == folded:
            return label
    return None


def _coerce_ordinal(value: object, low: int, high: int) -> int | None:
    """Ints, integral floats and numeric strings within range; else None."""
    number: float | None = None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = float(value)
    elif isinstance(value, float):
        number = value
    elif isinstance(value, str) and _NUMERIC_RE.match(value.strip()):
        number = float(value.strip())
    if number is None or not number.is_integer():
        return None
    as_int = int(number)
    return as_int if low <= as_int <= high else None


def _drop_reason(labels: dict[str, object], filters: list[_Filter],
                 drop_unmeasured: bool, axes: list[_Axis]) -> str | None:
    """None labels are kept unless drop_unmeasured; failing a filter drops."""
    if drop_unmeasured and any(labels.get(axis.name) is None for axis in axes):
        return "llm_unmeasured"
    for flt in filters:
        value = labels.get(flt.axis)
        if value is None:  # unmeasured is never filtered out, only counted
            continue
        if flt.keep_in is not None and value not in flt.keep_in:
            return f"llm_filter:{flt.axis}"
        if flt.gte is not None and float(value) < flt.gte:
            return f"llm_filter:{flt.axis}"
        if flt.lte is not None and float(value) > flt.lte:
            return f"llm_filter:{flt.axis}"
    return None


def _make_worker(backend, axes: list[_Axis], system: str, call_budget, field: str,
                 max_chars: int, rate: float, sample_seed: int, temperature: float,
                 max_tokens: int, seed: int) -> Callable[[dict], _Outcome]:
    """Build the per-record classify closure; never raises, never fakes labels."""

    def _work(rec: dict) -> _Outcome:
        text = _extract_text(rec, field)
        if not text.strip():
            return _Outcome(record=rec, kind="empty_text")
        if rate < 1.0 and not _in_sample(rec.get("id"), sample_seed, rate):
            return _Outcome(record=rec, kind="unsampled")
        truncated = len(text) > max_chars
        labels: dict[str, object] = {axis.name: None for axis in axes}
        if not call_budget.take():  # exhausted: all-None, NO call, NO provenance
            return _Outcome(record=rec, kind="classified", labels=labels,
                            truncated=truncated)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": text[:max_chars]}]
        try:
            resp = backend.complete(messages, temperature=temperature,
                                    max_tokens=max_tokens, seed=seed, json_mode=True)
        except Exception:  # a failed backend call must never fake labels
            return _Outcome(record=rec, kind="classified", labels=labels, called=True,
                            error=True, truncated=truncated)
        base = dict(record=rec, kind="classified", labels=labels, called=True,
                    cache_hit=resp.cache_hit, truncated=truncated,
                    prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens)
        if resp.error is not None:
            return _Outcome(error=True, **base)
        parsed = parse_json_object(resp.content)
        invalid: list[str] = []
        for axis in axes:
            value: object = None
            if parsed is not None:
                raw = parsed.get(axis.name)
                if axis.labels is not None:
                    value = _coerce_categorical(raw, axis.labels)
                else:
                    low, high = axis.scale or (0, 0)
                    value = _coerce_ordinal(raw, low, high)
            if value is None:
                invalid.append(axis.name)
            labels[axis.name] = value
        return _Outcome(invalid=tuple(invalid), **base)

    return _work


def _classify(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    axes = _parse_axes(cfg.get("axes"))
    backend = make_backend(cfg.get("backend"), _NAME)
    kind, model = backend.kind, backend.model
    cache_dir = cfg.get("cache_dir")
    if cache_dir:
        backend = CachedBackend(backend, str(cache_dir))
    stats.backend = f"llm:{kind}:{model}"

    field_name = str(cfg.get("field", _DEFAULT_FIELD))
    max_chars = int(cfg.get("max_chars", _DEFAULT_MAX_CHARS))
    if max_chars < 1:
        raise LLMOpError(f"{_NAME}: max_chars must be >= 1, got {max_chars}")
    output_key = str(cfg.get("output_key", _DEFAULT_OUTPUT_KEY)).strip() or _DEFAULT_OUTPUT_KEY
    workers = int(cfg.get("workers", 4))
    if workers < 1:
        raise LLMOpError(f"{_NAME}: workers must be >= 1, got {workers}")
    temperature = float(cfg.get("temperature", 0.0))
    max_tokens = int(cfg.get("max_tokens", 512))
    seed = int(cfg.get("seed", 0))
    drop_unmeasured = bool(cfg.get("drop_unmeasured", False))
    call_budget = budget(cfg.get("max_calls"))

    sample_cfg = cfg.get("sample") or {}
    if not isinstance(sample_cfg, dict):
        raise LLMOpError(f"{_NAME}: config.sample must be an object with rate/seed")
    rate = float(sample_cfg.get("rate", 1.0))
    sample_seed = int(sample_cfg.get("seed", 0))
    if not 0.0 < rate <= 1.0:
        raise LLMOpError(f"{_NAME}: sample.rate must be in (0, 1], got {rate}")

    filters = _parse_filters(cfg.get("filters"), axes)
    system = _system_prompt(axes)
    phash = prompt_hash(system)
    extra = stats.extra
    extra.update({
        "calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "errors": 0, "invalid": {axis.name: 0 for axis in axes},
        "unmeasured": 0, "labeled": 0, "sampled_out": 0,
        "distribution": {axis.name: {} for axis in axes},
        "model": model, "prompt_hash": phash, "axes": [axis.name for axis in axes],
    })

    work = _make_worker(backend, axes, system, call_budget, field_name, max_chars,
                        rate, sample_seed, temperature, max_tokens, seed)
    for outcome in map_ordered(work, counted(records, stats), workers):
        rec = outcome.record
        if outcome.kind == "empty_text":
            stats.drop("empty_text")
            continue
        if outcome.kind == "unsampled":
            extra["sampled_out"] += 1
            stats.records_out += 1
            yield rec  # untouched: no labels key, no provenance
            continue
        if outcome.truncated:
            stats.modified["input_truncated"] += 1
        if outcome.called:
            extra["calls"] += 1
            extra["prompt_tokens"] += outcome.prompt_tokens
            extra["completion_tokens"] += outcome.completion_tokens
            if outcome.cache_hit:
                extra["cache_hits"] += 1
        else:
            extra["budget_exhausted"] = True
        if outcome.error:
            extra["errors"] += 1
        for axis_name in outcome.invalid:
            extra["invalid"][axis_name] += 1
        if any(outcome.labels.get(axis.name) is None for axis in axes):
            extra["unmeasured"] += 1
        else:
            extra["labeled"] += 1
        for axis in axes:
            value = outcome.labels.get(axis.name)
            if value is not None:
                dist = extra["distribution"][axis.name]
                dist[str(value)] = dist.get(str(value), 0) + 1
        reason = _drop_reason(outcome.labels, filters, drop_unmeasured, axes)
        if reason is not None:
            stats.drop(reason)
            continue
        rec = dict(rec)
        meta = dict(rec.get("meta") or {})
        meta[output_key] = {axis.name: outcome.labels.get(axis.name) for axis in axes}
        if outcome.called:  # append, never overwrite; no call => no claim
            history = meta.get("llm")
            history = list(history) if isinstance(history, list) else []
            history.append({"op": _NAME, "model": model, "backend": kind,
                            "prompt_hash": phash, "cache_hit": outcome.cache_hit})
            meta["llm"] = history
        rec["meta"] = meta
        stats.modified["labels_added"] += 1
        stats.records_out += 1
        yield rec


llm_classify_op = FunctionOp(_NAME, _classify, config_schema={
    "type": "object", "required": ["backend", "axes"], "additionalProperties": False,
    "properties": {
        "backend": {"type": "object"},
        "axes": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["name"], "additionalProperties": False,
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
                "labels": {"type": "array", "minItems": 2, "items": {"type": "string"}},
                "scale": {"type": "array", "minItems": 2, "maxItems": 2,
                          "items": {"type": "integer"}},
            },
        }},
        "field": {"type": "string", "minLength": 1},
        "max_chars": {"type": "integer", "minimum": 1},
        "sample": {"type": "object", "additionalProperties": False, "properties": {
            "rate": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
            "seed": {"type": "integer"},
        }},
        "filters": {"type": "object"},
        "drop_unmeasured": {"type": "boolean"},
        "output_key": {"type": "string", "minLength": 1},
        "workers": {"type": "integer", "minimum": 1},
        "max_calls": {"type": ["integer", "null"], "minimum": 1},
        "temperature": {"type": "number", "minimum": 0},
        "max_tokens": {"type": "integer", "minimum": 1},
        "seed": {"type": "integer"},
        "cache_dir": {"type": ["string", "null"]},
    },
})
register_op(llm_classify_op)
