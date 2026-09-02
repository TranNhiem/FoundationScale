from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from foundationscale.provenance.manifest import (
    _EXPERT_COUNT_KEYS,
    _NESTED_LM_SCOPE_KEY,
)


class AdapterRefusal(ValueError):
    # Raised when a stated config fact has the wrong type. Fail closed; never coerce.
    pass


class Architecture(str, Enum):
    DENSE = "dense"
    MOE = "moe"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class Classification:
    architecture: Architecture
    num_routed_experts: int | None
    evidence: str
    adapter: str


class ModelAdapter(Protocol):
    name: str

    def matches(self, config: dict[str, Any]) -> bool: ...

    def classify(self, config: dict[str, Any]) -> Classification: ...

    def enable_moe_block_flag(self, config: dict[str, Any]) -> tuple[bool | None, str]:
        # Legacy emitter seam: Gemma semantics live behind the adapter, not in core.
        ...


# ONE auditable dialect table. Do not add key names in branches.
# kind: 'flag' means affirmative MoE declaration; 'count' means routed-expert count.
# scopes are checked in the listed order; nested scopes are checked before top level.
#
# NESTED_SCOPES is PINNED to the manifest's single definition rather than
# restated. This adapter's `enable_moe_block_flag` is one half of a two-sided
# comparison in tools/emit_run_manifest.py: the emitter reads the affirmative
# flag HERE and the routed count from `declared_from_hf_config`, then refuses
# when the two disagree. Widening only this side is therefore not a safe
# superset -- a config nesting under `llm_config` would read flag=True while
# the producer, which traverses text_config and top level only, found no count,
# and the emitter would refuse with "no routed-expert count was found under the
# keys this tool understands". That refusal fails closed but states a reason
# that is false: the count exists, in a scope the producer never looked at. A
# claim mismatched to its evidence is a defect even when the verdict is safe.
# To support another nesting, widen `_NESTED_LM_SCOPE_KEY`'s consumers in the
# SAME commit; test_model_adapters.py pins the two sides together.
NESTED_SCOPES: tuple[str, ...] = (_NESTED_LM_SCOPE_KEY,)

# Count keys carrying MEASURED provenance (see the manifest's docstring: a
# production Gemma-4 26B-A4B declared `text_config.num_experts`, and a
# DeepSeek-family config `n_routed_experts`). Imported, never restated -- a
# second narrower copy in the probe is what once made one config MoE to the
# library and dense to the probe.
_MEASURED_COUNT_KEYS: tuple[str, ...] = _EXPERT_COUNT_KEYS

# Additional published HF dialects for the same quantity. These extend
# `classify_config`, which is a standalone classifier with no second side to
# disagree with; they are deliberately NOT reachable from the emitter seam,
# which reads only the affirmative flag. Kept separate from the measured set so
# a reader can tell which names this repo has actually observed in the wild.
_EXTENSION_COUNT_KEYS: tuple[str, ...] = ("num_routed_experts", "moe_num_experts")

DIALECT_TABLE: tuple[dict[str, str], ...] = (
    {"kind": "flag", "key": "enable_moe_block"},
    *({"kind": "count", "key": k} for k in _MEASURED_COUNT_KEYS),
    *({"kind": "count", "key": k} for k in _EXTENSION_COUNT_KEYS),
)

# The adapter may be WIDER than the producer's vocabulary (extension keys only
# reach the standalone classifier) but must never be NARROWER: a measured key
# the producer honours and this table omits would classify a real MoE as
# "no dialect keys present". Checked with a raise, not an assert -- `python -O`
# strips asserts, and a guard that vanishes under a flag is not a guard.
_missing = tuple(
    k for k in _MEASURED_COUNT_KEYS if not any(row["key"] == k for row in DIALECT_TABLE)
)
if _missing:  # pragma: no cover -- import-time structural guard
    raise ImportError(
        f"model dialect table is narrower than the manifest's measured expert-count "
        f"vocabulary; missing {_missing}. Adding a key to _EXPERT_COUNT_KEYS without "
        f"adding it here would silently classify a declaring config as undetermined"
    )


@dataclass(frozen=True)
class _Signal:
    kind: str
    dotted: str
    value: bool | int


def _scopes(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for scope in NESTED_SCOPES:
        nested = config.get(scope)
        if isinstance(nested, dict):
            out.append((scope, nested))
    out.append(("top level", config))
    return out


def _refuse_unless_bool(dotted: str, raw: object) -> bool:
    """Return the flag, or refuse. ONE definition, called from both readers.

    ``classify`` and ``enable_moe_block_flag`` both read the affirmative key
    and must refuse identically; the message was duplicated verbatim between
    them, which is two places for one rule to drift.
    """
    if not isinstance(raw, bool):
        raise AdapterRefusal(
            f"{dotted} is present but not a JSON boolean ({raw!r}); a stated "
            f"declaration of the wrong type is refused, not coerced - coerce "
            f"it and a quoted 'false' would classify as MoE while looking "
            f"like a config fact"
        )
    return raw


def _collect(config: dict[str, Any], kinds: set[str]) -> list[_Signal]:
    signals: list[_Signal] = []
    for scope_name, scope in _scopes(config):
        for row in DIALECT_TABLE:
            key = row["key"]
            kind = row["kind"]
            if kind not in kinds or key not in scope:
                continue
            dotted = key if scope_name == "top level" else f"{scope_name}.{key}"
            raw = scope[key]
            if kind == "flag":
                signals.append(_Signal(kind, dotted, _refuse_unless_bool(dotted, raw)))
            else:
                # `isinstance(True, int)` is True in Python, so the bool check
                # must come FIRST: without it `num_experts: true` would be
                # read as the count 1 and declare a one-expert MoE.
                if isinstance(raw, bool) or not isinstance(raw, int):
                    raise AdapterRefusal(
                        f"{dotted} is present but not a JSON integer routed-expert "
                        f"count ({raw!r}); a stated count of the wrong type is "
                        f"refused, not coerced"
                    )
                if raw < 0:
                    raise AdapterRefusal(
                        f"{dotted} is negative ({raw}); expert counts are unsigned"
                    )
                signals.append(_Signal(kind, dotted, raw))
    return signals


def _verdict(signals: list[_Signal], adapter: str) -> Classification:
    flags_t = [s for s in signals if s.kind == "flag" and s.value is True]
    flags_f = [s for s in signals if s.kind == "flag" and s.value is False]
    counts_p = [s for s in signals if s.kind == "count" and int(s.value) > 0]
    counts_z = [s for s in signals if s.kind == "count" and int(s.value) == 0]

    def named(items: list[_Signal]) -> str:
        return "; ".join(f"{s.dotted}={s.value!r}" for s in items)

    conflicts: list[str] = []
    if flags_f and counts_p:
        conflicts.append(
            f"dense flag ({named(flags_f)}) vs positive expert count ({named(counts_p)})"
        )
    if flags_t and counts_z:
        conflicts.append(f"MoE flag ({named(flags_t)}) vs zero expert count ({named(counts_z)})")
    positives = {int(s.value) for s in counts_p}
    if len(positives) > 1:
        conflicts.append(f"divergent positive expert counts ({named(counts_p)})")
    if flags_t and flags_f:
        conflicts.append(f"contradictory MoE flags ({named(flags_t + flags_f)})")
    if conflicts:
        return Classification(
            Architecture.UNDETERMINED, None, "conflict: " + "; ".join(conflicts), adapter
        )

    if counts_p:
        only = counts_p[0]
        extra = f"; corroborated by {named(flags_t)}" if flags_t else ""
        return Classification(
            Architecture.MOE, int(only.value), f"{named(counts_p)}{extra}", adapter
        )
    if flags_t:
        return Classification(Architecture.MOE, None, named(flags_t), adapter)
    if flags_f or counts_z:
        num = 0 if counts_z else None
        return Classification(Architecture.DENSE, num, named(flags_f + counts_z), adapter)
    # The evidence string NAMES the scopes actually searched rather than
    # restating a fixed list: when NESTED_SCOPES changes, a hardcoded sentence
    # here becomes a false account of what was looked at, which is the same
    # drift the table above is pinned against.
    searched = ", ".join((*NESTED_SCOPES, "top level"))
    return Classification(
        Architecture.UNDETERMINED,
        None,
        f"no MoE dialect keys present in {searched}; absence is unmeasured, not dense",
        adapter,
    )


class _Base:
    name = "base"
    _flag_keys: tuple[str, ...] = ()

    def matches(self, config: dict[str, Any]) -> bool:  # noqa: ARG002 -- Protocol shape
        # The base never claims a config. Subclasses that DO claim one read the
        # argument; the parameter stays named so the Protocol matches by name.
        return False

    def classify(self, config: dict[str, Any]) -> Classification:
        return _verdict(_collect(config, {"flag", "count"}), self.name)

    def enable_moe_block_flag(self, config: dict[str, Any]) -> tuple[bool | None, str]:
        for scope_name, scope in _scopes(config):
            for key in self._flag_keys:
                if key not in scope:
                    continue
                dotted = key if scope_name == "top level" else f"{scope_name}.{key}"
                return _refuse_unless_bool(dotted, scope[key]), scope_name
        return None, ""


class GenericHFAdapter(_Base):
    # No `classify` override: it would be byte-identical to _Base's. A
    # duplicated method that cannot differ is a second place to fix a bug in.
    name = "generic"
    _flag_keys = tuple(row["key"] for row in DIALECT_TABLE if row["kind"] == "flag")


class GemmaAdapter(_Base):
    name = "gemma"
    _flag_keys = ("enable_moe_block",)

    def matches(self, config: dict[str, Any]) -> bool:
        model_type = config.get("model_type")
        return isinstance(model_type, str) and model_type.lower().startswith("gemma")

    def classify(self, config: dict[str, Any]) -> Classification:
        # Gemma owns enable_moe_block affirmative semantics; counts still corroborate so
        # a false flag beside a live routed count is UNDETERMINED, never a silent win.
        return _verdict(_collect(config, {"flag", "count"}), self.name)


_GENERIC = GenericHFAdapter()
_BUILTINS: tuple[ModelAdapter, ...] = (GemmaAdapter(),)
_REGISTERED: list[ModelAdapter] = list(_BUILTINS)


def registry_snapshot() -> tuple[ModelAdapter, ...]:
    return tuple(_REGISTERED)


def register_adapter(adapter: ModelAdapter) -> None:
    # Deterministic precedence is call order of register_adapter after built-ins.
    # The generic heuristic is NEVER registered; it is the declared explicit fallback.
    # Reject duplicate names and the reserved generic fallback so attribution cannot drift.
    if adapter.name == _GENERIC.name or any(a.name == adapter.name for a in _REGISTERED):
        raise AdapterRefusal(f"duplicate or reserved model adapter name: {adapter.name}")
    _REGISTERED.append(adapter)


def select_adapter(
    config: dict[str, Any], adapters: Sequence[ModelAdapter] | None = None
) -> ModelAdapter:
    ordered = tuple(adapters) if adapters is not None else registry_snapshot()
    for adapter in ordered:
        if adapter.name == _GENERIC.name:
            raise AdapterRefusal(
                "generic fallback must not be registered; it is selected explicitly"
            )
        if adapter.matches(config):
            return adapter
    return _GENERIC


def classify_config(
    config: dict[str, Any], adapters: Sequence[ModelAdapter] | None = None
) -> Classification:
    if not isinstance(config, dict):
        raise AdapterRefusal(f"model config is not a JSON object: {type(config).__name__}")
    return select_adapter(config, adapters).classify(config)
