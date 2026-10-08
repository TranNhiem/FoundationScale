"""``AgenticRLConfig``: declared, provenance-tracked configuration for one
FoundationScale agentic RL run, parsed from a JSON file plus repeatable
``--set dotted.key=value`` command-line overrides.

Every leaf value resolves from exactly one of three sources, recorded in
``AgenticRLConfig.provenance`` keyed by its dotted path: ``"cli"`` (a ``--set``
override), ``"config"`` (the JSON file), or ``"default"`` (neither supplied --
the field's own declared default). This is the doctrine ``train/cli.py``
applies to its own declared axes, carried into this plane: a run's manifest
must be able to say WHERE every knob's value came from, never merely what it
is. An unknown key, a value of the wrong type, or a missing REQUIRED key is
refused by :class:`AgenticConfigRefusal`, naming the exact dotted key -- this
module imports no torch and no transformers (``RLTrainConfig`` itself has no
such import at module scope; see its own module for where the lazy imports
live), so building a declared, provenance-tracked config is possible, and
required of ``--dry-run``, before touching a GPU.

Every ``--set`` override is also required to land somewhere: after every
section is built, :func:`build_config` refuses (naming the exact dotted
key(s)) any override that no field resolver above ever read -- an unknown
section, a field name typo'd within a known section, or a key this loader
declares not ``--set``-able at all (``engine.env``/``env.env``'s mapping
values, ``engine.command``'s full argv). Before this check existed, such an
override was silently dropped: the run proceeded as if ``--set`` had never
been given, with provenance still claiming the config file's (or default's)
value.

``trainer{}`` is the one section this module does NOT hand-declare a field
list for: its keys are READ from ``rl.trainer.RLTrainConfig``'s own
dataclass fields (minus ``model``/``algorithm``/``rollout_source``/``dataset``,
which are owned elsewhere or refused outright -- see
:data:`_TRAINER_FORBIDDEN_KEYS`: ``dataset`` is refused by name because the
rollout source replaces the corpus draw entirely, so a dataset declared here
would never be read), so a field RLTrainer adds or removes is reflected here
automatically rather than drifting behind a hand-maintained duplicate.

A JSON object anywhere in the file may carry a ``"_doc"`` string key, used by
the example config to comment itself (JSON has no comments); this loader
strips it before any key is validated and never treats it as an unknown key.
"""

from __future__ import annotations

import json
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Literal

from foundationscale.rl.trainer import RLTrainConfig

__all__ = (
    "AgenticConfigRefusal",
    "AgenticRLConfig",
    "BudgetConfig",
    "EngineConfig",
    "EnvConfig",
    "HarnessConfig",
    "PolicyConfig",
    "Provenance",
    "RewardConfig",
    "RolloutConfig",
    "SamplingConfig",
    "load_config",
    "parse_set_overrides",
)

Provenance = Literal["config", "cli", "default"]

_DOC_KEY = "_doc"
# Same literal as weight_sync.DiskWeightSync's own private constant (duplicated
# rather than imported: this module declares zero dependency on weight_sync.py
# or anything under engines/, keeping the "no torch, no transformers" claim
# above free of extra import surface to audit). Only used to refuse, at config
# time, a weight_sync_mode="restart" declaration whose command never names it.
_MODEL_PATH_PLACEHOLDER = "{model_path}"
# Owned elsewhere, or refused outright: model/algorithm come from the
# `policy` section (one source of truth for what is trained and how);
# rollout_source is always the internally-built RolloutHost -- never a value
# a config file could state; dataset is refused by name -- the rollout
# source replaces the corpus draw entirely, so a declared dataset would
# never be read (see RLTrainConfig.dataset / RLTrainer.run()'s own refusal).
_TRAINER_FORBIDDEN_KEYS = frozenset({"model", "algorithm", "rollout_source", "dataset"})
_TRAINER_FORBIDDEN_REASONS: dict[str, str] = {
    "model": "model is declared under 'policy' instead",
    "algorithm": "algorithm is declared under 'policy' instead",
    "rollout_source": "rollout_source is always the internally-built RolloutHost",
    "dataset": (
        "the rollout source replaces the corpus, so a dataset here would be declared and never read"
    ),
}


class AgenticConfigRefusal(ValueError):
    """A malformed config file, an unknown/mistyped/missing key, or a bad
    ``--set`` override. Every message names the full dotted key.
    """


def _described(value: object) -> str:
    return f"{type(value).__name__} {value!r}"


def _strip_doc(raw: object, *, dotted_prefix: str) -> dict[str, Any]:
    """Return ``raw`` as a plain ``dict``, with any ``"_doc"`` key removed.

    Refuses a non-mapping section outright: every section of this config is a
    JSON object, and a list/scalar in its place names no field to read.
    """
    if not isinstance(raw, Mapping):
        raise AgenticConfigRefusal(f"{dotted_prefix}: is {_described(raw)}, expected a JSON object")
    return {key: value for key, value in raw.items() if key != _DOC_KEY}


def _optional(py_type: Any) -> tuple[bool, Any]:
    """``(True, T)`` for a declared ``T | None`` union, else ``(False, py_type)``."""
    if typing.get_origin(py_type) is types.UnionType:
        args = typing.get_args(py_type)
        non_none = tuple(a for a in args if a is not type(None))
        if len(args) == 2 and len(non_none) == 1:
            return True, non_none[0]
    return False, py_type


def _validate_json_scalar(value: object, py_type: Any, *, dotted_key: str) -> Any:
    """Check a JSON-decoded ``value`` against a declared scalar ``py_type``
    (``str``/``int``/``float``/``bool``, optionally ``| None``), returning it
    unchanged (floats accept an int value, widened).
    """
    optional, base_type = _optional(py_type)
    if value is None:
        if optional:
            return None
        raise AgenticConfigRefusal(f"{dotted_key}: is null, and this key does not accept null")
    if base_type is bool:
        if type(value) is not bool:
            raise AgenticConfigRefusal(f"{dotted_key}: is {_described(value)}, expected a bool")
        return value
    if base_type is int:
        if type(value) is not int:
            raise AgenticConfigRefusal(f"{dotted_key}: is {_described(value)}, expected an int")
        return value
    if base_type is float:
        if type(value) is bool or not isinstance(value, (int, float)):
            raise AgenticConfigRefusal(f"{dotted_key}: is {_described(value)}, expected a float")
        return float(value)
    if base_type is str:
        if not isinstance(value, str):
            raise AgenticConfigRefusal(f"{dotted_key}: is {_described(value)}, expected a str")
        return value
    raise AgenticConfigRefusal(
        f"{dotted_key}: this loader has no scalar rule for declared type {base_type!r} "
        f"(internal: every trainer field must be str/int/float/bool, optionally | None)"
    )


def _coerce_cli_scalar(raw: str, py_type: Any, *, dotted_key: str) -> Any:
    """Parse a ``--set`` override's raw string against a declared scalar ``py_type``."""
    optional, base_type = _optional(py_type)
    if optional and raw.strip().lower() == "null":
        return None
    if base_type is bool:
        lowered = raw.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise AgenticConfigRefusal(
            f"--set {dotted_key}={raw!r}: expected 'true' or 'false' for a bool key"
        )
    if base_type is int:
        try:
            return int(raw)
        except ValueError as exc:
            raise AgenticConfigRefusal(f"--set {dotted_key}={raw!r}: not a valid int") from exc
    if base_type is float:
        try:
            return float(raw)
        except ValueError as exc:
            raise AgenticConfigRefusal(f"--set {dotted_key}={raw!r}: not a valid float") from exc
    if base_type is str:
        return raw
    raise AgenticConfigRefusal(
        f"--set {dotted_key}={raw!r}: this loader has no scalar rule for declared type "
        f"{base_type!r} (internal)"
    )


def _resolve_scalar(
    *,
    dotted_key: str,
    field_name: str,
    section_raw: Mapping[str, Any],
    overrides: Mapping[str, str],
    py_type: Any,
    default: Any,
    required: bool,
    provenance: dict[str, Provenance],
) -> Any:
    """One leaf's three-way resolution: ``--set`` wins over the config file,
    which wins over the declared default; refuses a still-missing REQUIRED leaf.
    """
    if dotted_key in overrides:
        value = _coerce_cli_scalar(overrides[dotted_key], py_type, dotted_key=dotted_key)
        provenance[dotted_key] = "cli"
        return value
    if field_name in section_raw:
        value = _validate_json_scalar(section_raw[field_name], py_type, dotted_key=dotted_key)
        provenance[dotted_key] = "config"
        return value
    if required:
        raise AgenticConfigRefusal(f"{dotted_key}: missing required key")
    provenance[dotted_key] = "default"
    return default


def _refuse_unknown_keys(
    section_raw: Mapping[str, Any], allowed: frozenset[str], *, section: str
) -> None:
    unknown = sorted(set(section_raw) - allowed)
    if unknown:
        dotted = ", ".join(f"{section}.{key}" for key in unknown)
        raise AgenticConfigRefusal(f"unknown key(s): {dotted}")


def _refuse_unconsumed_overrides(
    overrides: Mapping[str, str], provenance: Mapping[str, Provenance]
) -> None:
    """Refuse any ``--set`` override whose dotted key was never read while building
    every section above.

    A key that WAS read sets ``provenance[key] == "cli"`` -- either via
    :func:`_resolve_scalar`, or one of this module's few hand-written cross-field
    resolutions (``engine.kind``, ``engine.weight_sync_mode``, ``engine.reload_mode``,
    ``harness.parser``) that set it the same way. Anything left in ``overrides``
    with no matching ``"cli"`` entry named no real field this loader declares: an
    unknown section (``nosuch.section=1``), a field name typo'd within a known
    section (``policy.model_paht=...``), or a key explicitly refused earlier by its
    own dedicated message (``engine.env``, ``env.env``, ``engine.command`` -- none
    of which reaches this check, since each raises before ``build_config`` gets
    here). Before this check existed, every one of these was silently dropped: the
    run proceeded as if ``--set`` had never been given, with provenance still
    claiming the config file's (or default's) value.
    """
    consumed = {key for key, source in provenance.items() if source == "cli"}
    unconsumed = sorted(set(overrides) - consumed)
    if unconsumed:
        raise AgenticConfigRefusal(
            f"--set: key(s) never read by any declared field: {', '.join(unconsumed)}"
        )


@dataclass(frozen=True)
class PolicyConfig:
    model_path: str
    algorithm: str = "agentic_grpo"


@dataclass(frozen=True)
class EngineConfig:
    kind: Literal["sglang", "vllm"]
    command: tuple[str, ...]
    port: int
    served_model_name: str | None = None
    startup_timeout_s: float = 120.0
    # Per-HTTP-request ceiling for generation and control calls. Long agentic
    # generations (thousands of tokens) outlast short defaults -- measured: 12k-token
    # music responses timed out at 60 s and every episode abstained as infra.
    request_timeout_s: float = 600.0
    weight_sync_mode: Literal["reload_endpoint", "restart"] = "reload_endpoint"
    host: str = "127.0.0.1"
    log_dir: str = "."
    # False (default): the CLI refuses a config that would run more than one
    # step with no weight sync wired, naming the gap -- running many steps
    # while the fleet keeps serving the INITIAL weights makes every rollout
    # after the first silently off-policy, without the run ever saying so.
    # True is a declared, explicit opt-in to that staleness (e.g. a smoke run
    # that does not care), recorded in provenance like every other key.
    allow_stale_policy: bool = False
    # The declared lag bound the WEIGHT_SYNC gates
    # (foundationscale.gates.agentic_gates.StalenessGate) read as
    # RolloutHost.declared_max_lag: how many steps a rollout session's policy
    # version may trail the just-published one. 1 is the S0 default -- a
    # synchronous weight sync should never leave a session more than one
    # publish behind.
    max_policy_lag: int = 1
    # Forwarded verbatim to EngineServerSpec.reload_mode (-> a kind="vllm"
    # server's VLLMClient.reload_mode); None means the /collective_rpc
    # "reload_weights" RPC is unproven for this deployment (see
    # engines/vllm.py's UNPROBED ASSUMPTIONS). REQUIRED (refused if absent) for
    # kind="vllm" + weight_sync_mode="reload_endpoint": without it,
    # VLLMClient.reload_weights refuses at the first publish(), after a GPU is
    # already resident -- refusing at config time instead costs nothing.
    reload_mode: Literal["collective_rpc"] | None = None
    # The engine subprocess's COMPLETE environment -- an explicit allow-list,
    # never merged with this process's own os.environ (same "nothing is ever
    # inherited" rule EngineServerSpec.env and EnvSpec.env both already keep).
    # A real vLLM/SGLang launch typically needs at least PATH, HOME,
    # CUDA_VISIBLE_DEVICES, and (for vLLM's dev-RPC reload) VLLM_SERVER_DEV_MODE.
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvConfig:
    backend: str
    exec_timeout_s: float = 120.0
    episode_timeout_s: float = 3600.0
    max_output_bytes: int = 65536
    env: Mapping[str, str] = field(default_factory=dict)
    allow_unsandboxed: bool = False


@dataclass(frozen=True)
class HarnessConfig:
    parser: Literal["qwen_xml", "hermes_json"]
    name: str = "native_tool_loop"
    tools: Literal["default"] = "default"


@dataclass(frozen=True)
class BudgetConfig:
    step_limit: int
    max_response_tokens: int
    max_observation_chars: int


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    max_new_tokens: int


@dataclass(frozen=True)
class RolloutConfig:
    tasks_path: str
    tasks_per_step: int
    group_size: int
    max_concurrency: int
    group_by_harness: bool = False
    seed: int = 0
    # The declared bound the ROLLOUT gates
    # (foundationscale.gates.agentic_gates.RolloutAbstentionGate) read as
    # RolloutHost.declared_max_infra_rate. None (the default) is a declared
    # absence, not a declared 0.0 or 1.0 -- the gate abstains honestly
    # (AbstentionKind.NOT_ESTABLISHED) rather than adjudicating a bound nobody set.
    max_infra_rate: float | None = None


@dataclass(frozen=True)
class RewardConfig:
    abc2midi_bin: str
    kind: Literal["music"] = "music"
    timeout_s: float = 60.0


@dataclass(frozen=True)
class AgenticRLConfig:
    """The fully resolved, provenance-tracked configuration for one run.

    ``trainer`` is a plain ``dict`` of the ``RLTrainConfig`` keyword arguments
    this config declared or defaulted (everything EXCEPT ``model``/
    ``algorithm``/``rollout_source``, supplied by the caller at the point it
    builds the real ``RLTrainConfig`` -- see ``cli.py`` -- and ``dataset``,
    refused outright since the rollout source replaces the corpus), not a
    dataclass of its own: its field set is read from ``RLTrainConfig`` dynamically (see the
    module docstring), so there is no second, hand-maintained schema to drift.
    """

    policy: PolicyConfig
    trainer: Mapping[str, Any]
    engine: EngineConfig
    env: EnvConfig
    harness: HarnessConfig
    budget: BudgetConfig
    sampling: SamplingConfig
    rollout: RolloutConfig
    reward: RewardConfig
    publish_root: str
    provenance: Mapping[str, Provenance]

    def as_json(self) -> dict[str, Any]:
        """The resolved config as a plain JSON-able dict, with ``provenance``
        alongside it -- exactly what ``cli.py --dry-run`` prints.
        """

        def _section(obj: Any) -> Any:
            if isinstance(obj, Mapping):
                return dict(obj)
            return {f.name: getattr(obj, f.name) for f in dataclass_fields(obj)}

        return {
            "policy": _section(self.policy),
            "trainer": _section(self.trainer),
            "engine": _section(self.engine),
            "env": _section(self.env),
            "harness": _section(self.harness),
            "budget": _section(self.budget),
            "sampling": _section(self.sampling),
            "rollout": _section(self.rollout),
            "reward": _section(self.reward),
            "publish_root": self.publish_root,
            "provenance": dict(self.provenance),
        }


def parse_set_overrides(raw_overrides: Sequence[str]) -> dict[str, str]:
    """Parse repeated ``--set dotted.key=value`` strings into a dict, refusing a
    malformed entry (no ``=``) or a dotted key repeated across two ``--set``s.
    """
    overrides: dict[str, str] = {}
    for entry in raw_overrides:
        if "=" not in entry:
            raise AgenticConfigRefusal(
                f"--set {entry!r}: expected 'dotted.key=value' (no '=' found)"
            )
        dotted_key, _, value = entry.partition("=")
        dotted_key = dotted_key.strip()
        if not dotted_key:
            raise AgenticConfigRefusal(f"--set {entry!r}: the dotted key is empty")
        if dotted_key in overrides:
            raise AgenticConfigRefusal(
                f"--set {dotted_key}: given twice ({overrides[dotted_key]!r} and {value!r})"
            )
        overrides[dotted_key] = value
    return overrides


def _section_overrides(overrides: Mapping[str, str], *, section: str) -> dict[str, str]:
    """Filter ``overrides`` to this section's keys -- KEEPING the full dotted key
    (never stripped): every downstream check (``_resolve_scalar`` and this
    module's few hand-written cross-field checks) looks up by the full
    ``section.field`` key, so the filtered dict must still be keyed that way.
    """
    prefix = f"{section}."
    return {dotted: value for dotted, value in overrides.items() if dotted.startswith(prefix)}


def _build_policy(
    raw: Mapping[str, Any], overrides: dict[str, str], provenance: dict[str, Provenance]
) -> PolicyConfig:
    section = _strip_doc(raw, dotted_prefix="policy")
    local_overrides = _section_overrides(overrides, section="policy")
    _refuse_unknown_keys(section, frozenset({"model_path", "algorithm"}), section="policy")
    model_path = _resolve_scalar(
        dotted_key="policy.model_path",
        field_name="model_path",
        section_raw=section,
        overrides=local_overrides,
        py_type=str,
        default=None,
        required=True,
        provenance=provenance,
    )
    algorithm = _resolve_scalar(
        dotted_key="policy.algorithm",
        field_name="algorithm",
        section_raw=section,
        overrides=local_overrides,
        py_type=str,
        default="agentic_grpo",
        required=False,
        provenance=provenance,
    )
    return PolicyConfig(model_path=model_path, algorithm=algorithm)


def _trainer_field_table() -> dict[str, tuple[Any, Any, bool]]:
    """``{field_name: (py_type, default, required)}`` for every ``RLTrainConfig``
    field this config may set -- every declared field MINUS
    :data:`_TRAINER_FORBIDDEN_KEYS`, read dynamically so a future RLTrainConfig
    field change needs no edit here.
    """
    hints = typing.get_type_hints(RLTrainConfig)
    table: dict[str, tuple[Any, Any, bool]] = {}
    for declared_field in dataclass_fields(RLTrainConfig):
        if declared_field.name in _TRAINER_FORBIDDEN_KEYS:
            continue
        required = declared_field.default is MISSING and declared_field.default_factory is MISSING
        default = None if required else declared_field.default
        table[declared_field.name] = (hints[declared_field.name], default, required)
    return table


def _build_trainer(
    raw: Mapping[str, Any], overrides: dict[str, str], provenance: dict[str, Provenance]
) -> dict[str, Any]:
    section = _strip_doc(raw, dotted_prefix="trainer")
    local_overrides = _section_overrides(overrides, section="trainer")
    table = _trainer_field_table()
    forbidden_present = sorted(set(section) & _TRAINER_FORBIDDEN_KEYS)
    if forbidden_present:
        details = "; ".join(
            f"trainer.{key} ({_TRAINER_FORBIDDEN_REASONS[key]})" for key in forbidden_present
        )
        raise AgenticConfigRefusal(f"unknown key(s): {details}")
    _refuse_unknown_keys(section, frozenset(table), section="trainer")
    resolved: dict[str, Any] = {}
    for field_name, (py_type, default, required) in table.items():
        resolved[field_name] = _resolve_scalar(
            dotted_key=f"trainer.{field_name}",
            field_name=field_name,
            section_raw=section,
            overrides=local_overrides,
            py_type=py_type,
            default=default,
            required=required,
            provenance=provenance,
        )
    return resolved


def _build_engine(
    raw: Mapping[str, Any], overrides: dict[str, str], provenance: dict[str, Provenance]
) -> EngineConfig:
    section = _strip_doc(raw, dotted_prefix="engine")
    local_overrides = _section_overrides(overrides, section="engine")
    allowed = frozenset(
        {
            "kind",
            "command",
            "port",
            "served_model_name",
            "startup_timeout_s",
            "request_timeout_s",
            "weight_sync_mode",
            "host",
            "log_dir",
            "allow_stale_policy",
            "max_policy_lag",
            "reload_mode",
            "env",
        }
    )
    _refuse_unknown_keys(section, allowed, section="engine")

    kind_raw = section.get("kind")
    if "engine.kind" in local_overrides:
        kind_raw = local_overrides["engine.kind"]
        provenance["engine.kind"] = "cli"
    elif "kind" in section:
        provenance["engine.kind"] = "config"
    else:
        raise AgenticConfigRefusal("engine.kind: missing required key")
    if kind_raw not in ("sglang", "vllm"):
        raise AgenticConfigRefusal(
            f"engine.kind: is {_described(kind_raw)}, expected 'sglang' or 'vllm'"
        )
    kind: Literal["sglang", "vllm"] = kind_raw

    if "engine.command" in local_overrides:
        raise AgenticConfigRefusal(
            "--set engine.command=...: the engine subprocess's full argv is a JSON "
            "array, not settable via --set; edit the config file"
        )
    if "command" not in section:
        raise AgenticConfigRefusal("engine.command: missing required key")
    command_raw = section["command"]
    if (
        isinstance(command_raw, (str, bytes))
        or not isinstance(command_raw, Sequence)
        or not command_raw
        or not all(isinstance(part, str) and part for part in command_raw)
    ):
        raise AgenticConfigRefusal(
            f"engine.command: is {_described(command_raw)}, expected a non-empty JSON array "
            f"of non-empty strings (the full argv)"
        )
    provenance["engine.command"] = "config"
    command = tuple(command_raw)

    port = _resolve_scalar(
        dotted_key="engine.port",
        field_name="port",
        section_raw=section,
        overrides=local_overrides,
        py_type=int,
        default=None,
        required=True,
        provenance=provenance,
    )
    served_model_name = _resolve_scalar(
        dotted_key="engine.served_model_name",
        field_name="served_model_name",
        section_raw=section,
        overrides=local_overrides,
        py_type=str | None,
        default=None,
        required=False,
        provenance=provenance,
    )
    startup_timeout_s = _resolve_scalar(
        dotted_key="engine.startup_timeout_s",
        field_name="startup_timeout_s",
        section_raw=section,
        overrides=local_overrides,
        py_type=float,
        default=120.0,
        required=False,
        provenance=provenance,
    )
    request_timeout_s = _resolve_scalar(
        dotted_key="engine.request_timeout_s",
        field_name="request_timeout_s",
        section_raw=section,
        overrides=local_overrides,
        py_type=float,
        default=600.0,
        required=False,
        provenance=provenance,
    )
    weight_sync_mode_raw = section.get("weight_sync_mode", "reload_endpoint")
    if "engine.weight_sync_mode" in local_overrides:
        weight_sync_mode_raw = local_overrides["engine.weight_sync_mode"]
        provenance["engine.weight_sync_mode"] = "cli"
    elif "weight_sync_mode" in section:
        provenance["engine.weight_sync_mode"] = "config"
    else:
        provenance["engine.weight_sync_mode"] = "default"
    if weight_sync_mode_raw not in ("reload_endpoint", "restart"):
        raise AgenticConfigRefusal(
            f"engine.weight_sync_mode: is {_described(weight_sync_mode_raw)}, expected "
            f"'reload_endpoint' or 'restart'"
        )
    weight_sync_mode: Literal["reload_endpoint", "restart"] = weight_sync_mode_raw
    if weight_sync_mode == "restart" and not any(
        _MODEL_PATH_PLACEHOLDER in part for part in command
    ):
        raise AgenticConfigRefusal(
            f"engine.command: is {_described(command)}, but engine.weight_sync_mode is "
            f"'restart', which requires the command to declare the "
            f"{_MODEL_PATH_PLACEHOLDER!r} placeholder at least once -- substituted with "
            f"policy.model_path on the FIRST launch, and with each published checkpoint "
            f"thereafter by weight_sync.DiskWeightSync -- or a config-time crash is "
            f"deferred to a cluster launch that sends the literal placeholder to the "
            f"engine as an argv entry"
        )
    host = _resolve_scalar(
        dotted_key="engine.host",
        field_name="host",
        section_raw=section,
        overrides=local_overrides,
        py_type=str,
        default="127.0.0.1",
        required=False,
        provenance=provenance,
    )
    log_dir = _resolve_scalar(
        dotted_key="engine.log_dir",
        field_name="log_dir",
        section_raw=section,
        overrides=local_overrides,
        py_type=str,
        default=".",
        required=False,
        provenance=provenance,
    )
    allow_stale_policy = _resolve_scalar(
        dotted_key="engine.allow_stale_policy",
        field_name="allow_stale_policy",
        section_raw=section,
        overrides=local_overrides,
        py_type=bool,
        default=False,
        required=False,
        provenance=provenance,
    )
    max_policy_lag = _resolve_scalar(
        dotted_key="engine.max_policy_lag",
        field_name="max_policy_lag",
        section_raw=section,
        overrides=local_overrides,
        py_type=int,
        default=1,
        required=False,
        provenance=provenance,
    )
    reload_mode_raw = section.get("reload_mode")
    if "engine.reload_mode" in local_overrides:
        # Routed through the SAME cli-scalar coercion every other optional field
        # uses (via _resolve_scalar) rather than taking the raw string verbatim:
        # without it, "--set engine.reload_mode=null" produced the literal str
        # "null" here, which then failed the "None or 'collective_rpc'" check
        # below -- the loader's own null -> None convention bypassed for this
        # one hand-written field.
        reload_mode_raw = _coerce_cli_scalar(
            local_overrides["engine.reload_mode"], str | None, dotted_key="engine.reload_mode"
        )
        provenance["engine.reload_mode"] = "cli"
    elif "reload_mode" in section:
        provenance["engine.reload_mode"] = "config"
    else:
        provenance["engine.reload_mode"] = "default"
    if reload_mode_raw is not None and reload_mode_raw != "collective_rpc":
        raise AgenticConfigRefusal(
            f"engine.reload_mode: is {_described(reload_mode_raw)}, expected None or "
            f"'collective_rpc'"
        )
    reload_mode: Literal["collective_rpc"] | None = reload_mode_raw
    env_mapping = section.get("env", {})
    if "engine.env" in local_overrides:
        raise AgenticConfigRefusal(
            "--set engine.env=...: the engine subprocess's env allow-list is a JSON "
            "object, not settable via --set; edit the config file"
        )
    if not isinstance(env_mapping, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env_mapping.items()
    ):
        raise AgenticConfigRefusal(
            f"engine.env: is {_described(env_mapping)}, expected a JSON object of str to str"
        )
    provenance["engine.env"] = "config" if "env" in section else "default"
    if kind == "vllm" and not served_model_name:
        raise AgenticConfigRefusal(
            "engine.served_model_name: missing required key (required when engine.kind "
            "is 'vllm': every /v1/completions request must carry it as 'model')"
        )
    if kind == "vllm" and weight_sync_mode == "reload_endpoint" and reload_mode is None:
        raise AgenticConfigRefusal(
            "engine.reload_mode: missing required key (required when engine.kind is "
            "'vllm' and engine.weight_sync_mode is 'reload_endpoint': without it, "
            "VLLMClient.reload_weights refuses at the first publish() -- declare "
            "engine.reload_mode='collective_rpc' once it is proven on the cluster, or "
            "use engine.weight_sync_mode='restart' instead)"
        )
    return EngineConfig(
        kind=kind,
        command=command,
        port=port,
        served_model_name=served_model_name,
        startup_timeout_s=startup_timeout_s,
        request_timeout_s=request_timeout_s,
        weight_sync_mode=weight_sync_mode,
        host=host,
        max_policy_lag=max_policy_lag,
        log_dir=log_dir,
        allow_stale_policy=allow_stale_policy,
        reload_mode=reload_mode,
        env=dict(env_mapping),
    )


def _build_env(
    raw: Mapping[str, Any], overrides: dict[str, str], provenance: dict[str, Provenance]
) -> EnvConfig:
    section = _strip_doc(raw, dotted_prefix="env")
    local_overrides = _section_overrides(overrides, section="env")
    allowed = frozenset(
        {
            "backend",
            "exec_timeout_s",
            "episode_timeout_s",
            "max_output_bytes",
            "env",
            "allow_unsandboxed",
        }
    )
    _refuse_unknown_keys(section, allowed, section="env")
    backend = _resolve_scalar(
        dotted_key="env.backend",
        field_name="backend",
        section_raw=section,
        overrides=local_overrides,
        py_type=str,
        default=None,
        required=True,
        provenance=provenance,
    )
    exec_timeout_s = _resolve_scalar(
        dotted_key="env.exec_timeout_s",
        field_name="exec_timeout_s",
        section_raw=section,
        overrides=local_overrides,
        py_type=float,
        default=120.0,
        required=False,
        provenance=provenance,
    )
    episode_timeout_s = _resolve_scalar(
        dotted_key="env.episode_timeout_s",
        field_name="episode_timeout_s",
        section_raw=section,
        overrides=local_overrides,
        py_type=float,
        default=3600.0,
        required=False,
        provenance=provenance,
    )
    max_output_bytes = _resolve_scalar(
        dotted_key="env.max_output_bytes",
        field_name="max_output_bytes",
        section_raw=section,
        overrides=local_overrides,
        py_type=int,
        default=65536,
        required=False,
        provenance=provenance,
    )
    env_mapping = section.get("env", {})
    if "env.env" in local_overrides:
        raise AgenticConfigRefusal(
            "--set env.env=...: the env allow-list is a JSON object, not settable via --set; "
            "edit the config file"
        )
    if not isinstance(env_mapping, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env_mapping.items()
    ):
        raise AgenticConfigRefusal(
            f"env.env: is {_described(env_mapping)}, expected a JSON object of str to str"
        )
    provenance["env.env"] = "config" if "env" in section else "default"
    allow_unsandboxed = _resolve_scalar(
        dotted_key="env.allow_unsandboxed",
        field_name="allow_unsandboxed",
        section_raw=section,
        overrides=local_overrides,
        py_type=bool,
        default=False,
        required=False,
        provenance=provenance,
    )
    return EnvConfig(
        backend=backend,
        exec_timeout_s=exec_timeout_s,
        episode_timeout_s=episode_timeout_s,
        max_output_bytes=max_output_bytes,
        env=dict(env_mapping),
        allow_unsandboxed=allow_unsandboxed,
    )


def _build_harness(
    raw: Mapping[str, Any], overrides: dict[str, str], provenance: dict[str, Provenance]
) -> HarnessConfig:
    section = _strip_doc(raw, dotted_prefix="harness")
    local_overrides = _section_overrides(overrides, section="harness")
    _refuse_unknown_keys(section, frozenset({"name", "parser", "tools"}), section="harness")
    name = _resolve_scalar(
        dotted_key="harness.name",
        field_name="name",
        section_raw=section,
        overrides=local_overrides,
        py_type=str,
        default="native_tool_loop",
        required=False,
        provenance=provenance,
    )
    if name != "native_tool_loop":
        raise AgenticConfigRefusal(
            f"harness.name: is {_described(name)}, but 'native_tool_loop' is the only "
            f"harness this plane builds"
        )
    parser_raw = section.get("parser")
    if "harness.parser" in local_overrides:
        parser_raw = local_overrides["harness.parser"]
        provenance["harness.parser"] = "cli"
    elif "parser" in section:
        provenance["harness.parser"] = "config"
    else:
        raise AgenticConfigRefusal("harness.parser: missing required key")
    if parser_raw not in ("qwen_xml", "hermes_json"):
        raise AgenticConfigRefusal(
            f"harness.parser: is {_described(parser_raw)}, expected 'qwen_xml' or 'hermes_json'"
        )
    parser: Literal["qwen_xml", "hermes_json"] = parser_raw
    tools = _resolve_scalar(
        dotted_key="harness.tools",
        field_name="tools",
        section_raw=section,
        overrides=local_overrides,
        py_type=str,
        default="default",
        required=False,
        provenance=provenance,
    )
    if tools != "default":
        raise AgenticConfigRefusal(
            f"harness.tools: is {_described(tools)}, but 'default' is the only tool "
            f"registry this plane builds"
        )
    return HarnessConfig(name=name, parser=parser, tools="default")


def _build_budget(
    raw: Mapping[str, Any], overrides: dict[str, str], provenance: dict[str, Provenance]
) -> BudgetConfig:
    section = _strip_doc(raw, dotted_prefix="budget")
    local_overrides = _section_overrides(overrides, section="budget")
    allowed = frozenset({"step_limit", "max_response_tokens", "max_observation_chars"})
    _refuse_unknown_keys(section, allowed, section="budget")
    values = {
        name: _resolve_scalar(
            dotted_key=f"budget.{name}",
            field_name=name,
            section_raw=section,
            overrides=local_overrides,
            py_type=int,
            default=None,
            required=True,
            provenance=provenance,
        )
        for name in sorted(allowed)
    }
    return BudgetConfig(**values)


def _build_sampling(
    raw: Mapping[str, Any], overrides: dict[str, str], provenance: dict[str, Provenance]
) -> SamplingConfig:
    section = _strip_doc(raw, dotted_prefix="sampling")
    local_overrides = _section_overrides(overrides, section="sampling")
    int_fields = frozenset({"top_k", "max_new_tokens"})
    float_fields = frozenset(
        {"temperature", "top_p", "min_p", "presence_penalty", "repetition_penalty"}
    )
    allowed = int_fields | float_fields
    _refuse_unknown_keys(section, allowed, section="sampling")
    values: dict[str, Any] = {}
    for name in sorted(allowed):
        values[name] = _resolve_scalar(
            dotted_key=f"sampling.{name}",
            field_name=name,
            section_raw=section,
            overrides=local_overrides,
            py_type=int if name in int_fields else float,
            default=None,
            required=True,
            provenance=provenance,
        )
    return SamplingConfig(**values)


def _build_rollout(
    raw: Mapping[str, Any], overrides: dict[str, str], provenance: dict[str, Provenance]
) -> RolloutConfig:
    section = _strip_doc(raw, dotted_prefix="rollout")
    local_overrides = _section_overrides(overrides, section="rollout")
    allowed = frozenset(
        {
            "tasks_path",
            "tasks_per_step",
            "group_size",
            "max_concurrency",
            "group_by_harness",
            "seed",
            "max_infra_rate",
        }
    )
    _refuse_unknown_keys(section, allowed, section="rollout")
    tasks_path = _resolve_scalar(
        dotted_key="rollout.tasks_path",
        field_name="tasks_path",
        section_raw=section,
        overrides=local_overrides,
        py_type=str,
        default=None,
        required=True,
        provenance=provenance,
    )
    tasks_per_step = _resolve_scalar(
        dotted_key="rollout.tasks_per_step",
        field_name="tasks_per_step",
        section_raw=section,
        overrides=local_overrides,
        py_type=int,
        default=None,
        required=True,
        provenance=provenance,
    )
    group_size = _resolve_scalar(
        dotted_key="rollout.group_size",
        field_name="group_size",
        section_raw=section,
        overrides=local_overrides,
        py_type=int,
        default=None,
        required=True,
        provenance=provenance,
    )
    max_concurrency = _resolve_scalar(
        dotted_key="rollout.max_concurrency",
        field_name="max_concurrency",
        section_raw=section,
        overrides=local_overrides,
        py_type=int,
        default=None,
        required=True,
        provenance=provenance,
    )
    group_by_harness = _resolve_scalar(
        dotted_key="rollout.group_by_harness",
        field_name="group_by_harness",
        section_raw=section,
        overrides=local_overrides,
        py_type=bool,
        default=False,
        required=False,
        provenance=provenance,
    )
    seed = _resolve_scalar(
        dotted_key="rollout.seed",
        field_name="seed",
        section_raw=section,
        overrides=local_overrides,
        py_type=int,
        default=0,
        required=False,
        provenance=provenance,
    )
    max_infra_rate = _resolve_scalar(
        dotted_key="rollout.max_infra_rate",
        field_name="max_infra_rate",
        section_raw=section,
        overrides=local_overrides,
        py_type=float | None,
        default=None,
        required=False,
        provenance=provenance,
    )
    return RolloutConfig(
        tasks_path=tasks_path,
        tasks_per_step=tasks_per_step,
        group_size=group_size,
        max_concurrency=max_concurrency,
        group_by_harness=group_by_harness,
        seed=seed,
        max_infra_rate=max_infra_rate,
    )


def _build_reward(
    raw: Mapping[str, Any], overrides: dict[str, str], provenance: dict[str, Provenance]
) -> RewardConfig:
    section = _strip_doc(raw, dotted_prefix="reward")
    local_overrides = _section_overrides(overrides, section="reward")
    allowed = frozenset({"kind", "abc2midi_bin", "timeout_s"})
    _refuse_unknown_keys(section, allowed, section="reward")
    kind = _resolve_scalar(
        dotted_key="reward.kind",
        field_name="kind",
        section_raw=section,
        overrides=local_overrides,
        py_type=str,
        default="music",
        required=False,
        provenance=provenance,
    )
    if kind != "music":
        raise AgenticConfigRefusal(
            f"reward.kind: is {_described(kind)}, but 'music' is the only reward this plane builds"
        )
    abc2midi_bin = _resolve_scalar(
        dotted_key="reward.abc2midi_bin",
        field_name="abc2midi_bin",
        section_raw=section,
        overrides=local_overrides,
        py_type=str,
        default=None,
        required=True,
        provenance=provenance,
    )
    timeout_s = _resolve_scalar(
        dotted_key="reward.timeout_s",
        field_name="timeout_s",
        section_raw=section,
        overrides=local_overrides,
        py_type=float,
        default=60.0,
        required=False,
        provenance=provenance,
    )
    return RewardConfig(kind="music", abc2midi_bin=abc2midi_bin, timeout_s=timeout_s)


_SECTION_NAMES = (
    "policy",
    "trainer",
    "engine",
    "env",
    "harness",
    "budget",
    "sampling",
    "rollout",
    "reward",
)


def build_config(raw_config: Mapping[str, Any], overrides: Mapping[str, str]) -> AgenticRLConfig:
    """Build and validate an :class:`AgenticRLConfig` from an already-JSON-decoded
    mapping plus parsed ``--set`` overrides (see :func:`parse_set_overrides`).
    """
    top = _strip_doc(raw_config, dotted_prefix="<config root>")
    allowed_top = frozenset({*_SECTION_NAMES, "publish_root"})
    unknown_top = sorted(set(top) - allowed_top)
    if unknown_top:
        raise AgenticConfigRefusal(f"unknown key(s): {', '.join(unknown_top)}")
    provenance: dict[str, Provenance] = {}
    sections: dict[str, Any] = {}
    for name in _SECTION_NAMES:
        section_raw = top.get(name, {})
        builder = {
            "policy": _build_policy,
            "trainer": _build_trainer,
            "engine": _build_engine,
            "env": _build_env,
            "harness": _build_harness,
            "budget": _build_budget,
            "sampling": _build_sampling,
            "rollout": _build_rollout,
            "reward": _build_reward,
        }[name]
        sections[name] = builder(section_raw, dict(overrides), provenance)
    publish_root = _resolve_scalar(
        dotted_key="publish_root",
        field_name="publish_root",
        section_raw=top,
        overrides=dict(overrides),
        py_type=str,
        default=None,
        required=True,
        provenance=provenance,
    )
    _refuse_unconsumed_overrides(overrides, provenance)
    return AgenticRLConfig(
        policy=sections["policy"],
        trainer=sections["trainer"],
        engine=sections["engine"],
        env=sections["env"],
        harness=sections["harness"],
        budget=sections["budget"],
        sampling=sections["sampling"],
        rollout=sections["rollout"],
        reward=sections["reward"],
        publish_root=publish_root,
        provenance=provenance,
    )


def load_config(path: str, raw_overrides: Sequence[str]) -> AgenticRLConfig:
    """Read ``path`` as JSON, apply ``--set`` overrides, and build a validated,
    provenance-tracked :class:`AgenticRLConfig`.

    Refuses (naming ``path``) a missing/unreadable file or invalid JSON, before
    any key is validated.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise AgenticConfigRefusal(f"{path}: could not be read ({exc})") from exc
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgenticConfigRefusal(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(decoded, Mapping):
        raise AgenticConfigRefusal(f"{path}: is {_described(decoded)}, expected a JSON object")
    overrides = parse_set_overrides(raw_overrides)
    return build_config(decoded, overrides)
