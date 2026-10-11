"""Agentic-RL skill layer over ``foundationscale.agentic_rl``: plan, gate, render config.

FoundationScale owns the whole agentic plane (foundationscale.agentic_rl: config schema,
CLI, contracts, rewards, envs, gateway, and the enroot sandboxes / Harbor env backend), so
this module is deliberately a THIN skill layer on top of it. It does exactly three things
and nothing more:

1. ``plan_agentic`` -- turn ``goal["agentic"]`` into one ``agentic_rl`` stage record;
2. ``agentic_gates`` -- adjudicate the TR-* input/handoff gates against the runtime's
   agentic capability probe (``FSCapabilities.agentic``);
3. ``build_agentic_config`` -- render the ``AgenticRLConfig`` JSON that ``emit_agentic``
   writes and FS's own ``--dry-run`` then validates (FS-HO-005: FS owns the schema, so a
   key this module wrote that ``foundationscale.agentic_rl.config`` does not declare
   would make the whole line worthless).

Measurement is never assumed. The ONLY defaults this module may insert are verbatim
values of ``examples/agentic_rl/music_s0.json`` (the measured S0 declaration) for keys
the goal schema does not name -- see ``_MUSIC_S0_FILLS``, every entry cited. Values that
example does not declare (engine request_timeout_s / max_policy_lag / allow_stale_policy,
engine.log_dir, engine.env, trainer.save_dir, rollout.max_infra_rate, the policy
retention knobs, everything under ``stability``) are never defaulted: they stay an
explicit declaration, or an FS dataclass default with its own provenance, or they are
absent and the downstream gates / FS ``--dry-run`` adjudicate that absence honestly.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Final, Mapping

from foundationskills.interfaces.fs.agentic_probe import Runtime, runtime_from_dict

__all__ = [
    "BLOCK_RULES",
    "GATE_SEVERITY",
    "agentic_gates",
    "blocking_findings",
    "build_agentic_config",
    "plan_agentic",
]

_STAGE_NAME: Final[str] = "agentic_rl"
_METHOD: Final[str] = "full"

# Phase/severity per the consolidated A4.2 rule list. ``agentic_gates`` returns
# (rule_id, message) pairs; the severity lives here so emit/verification can ask
# "is this finding blocking?" without re-encoding the specification table.
GATE_SEVERITY: Final[dict[str, str]] = {
    "TR-IN-010": "BLOCK",  # input: algorithm/engine/env backend/reward kind/parser not probed
    "TR-IN-011": "BLOCK",  # input: multi-GPU claimed while agentic multi-GPU is unmeasured
    "TR-IN-012": "BLOCK",  # input: reward handle / tasks file / env declaration missing
    "TR-IN-013": "BLOCK",  # input: >1 step with neither weight sync nor stale-policy opt-in
    "TR-IN-014": "BLOCK",  # input: engine env cannot start the server (FS never merges os.environ)
    "TR-PL-009": "WARN",   # handoff: the measured control chosen as the default arm
}

BLOCK_RULES: Final[frozenset[str]] = frozenset(
    rule for rule, severity in GATE_SEVERITY.items() if severity == "BLOCK"
)

_PROBE_REMEDY: Final[str] = "run the agentic probe in the declared runtime"
_MUSIC_S0: Final[str] = "examples/agentic_rl/music_s0.json"

# The backends whose isolation level is "none" (envs/base.py check_isolation refuses them
# unless EnvSpec.allow_unsandboxed is True): "local" is the only registered backend and it
# runs model-chosen shell commands directly on the host (envs/base.py:478-502 registers
# LocalSubprocessBackend at :512). A future isolated backend simply is not in this set.
_UNSANDBOXED_BACKENDS: Final[frozenset[str]] = frozenset({"local"})

# Field names of each ``foundationscale.agentic_rl.config`` section (config.py:256-413),
# ordered to mirror examples/agentic_rl/music_s0.json section by section with the extra
# config.py fields (servable_publish/keep_last/archive_every, allow_stale_policy,
# max_policy_lag, request_timeout_s, max_infra_rate) appended after the example order.
# build_agentic_config writes ONLY these names -- nothing else is ever written, and the
# loader records provenance for whatever it filled itself.
_SECTION_ORDER: Final[dict[str, tuple[str, ...]]] = {
    "policy": ("model_path", "algorithm", "servable_publish", "keep_last", "archive_every"),
    "trainer": ("max_steps", "learning_rate", "logprob_micro_batch", "sharding", "save_dir", "seed"),
    "engine": (
        "kind", "command", "port", "served_model_name", "startup_timeout_s",
        "weight_sync_mode", "reload_mode", "host", "log_dir", "env",
        "allow_stale_policy", "max_policy_lag", "request_timeout_s",
    ),
    "env": ("backend", "exec_timeout_s", "episode_timeout_s", "max_output_bytes", "env", "allow_unsandboxed"),
    "harness": ("name", "parser", "tools"),
    "budget": ("step_limit", "max_response_tokens", "max_observation_chars"),
    "sampling": (
        "temperature", "top_p", "top_k", "min_p",
        "presence_penalty", "repetition_penalty", "max_new_tokens",
    ),
    "rollout": (
        "tasks_path", "tasks_per_step", "group_size", "max_concurrency",
        "group_by_harness", "seed", "max_infra_rate",
    ),
    "reward": ("kind", "abc2midi_bin", "timeout_s"),
    "stability": ("ema_alpha", "drop_from_peak", "min_steps", "best_k", "action"),
}

# These four trainer key names are REFUSED by the FS loader by name (the agentic trainer
# dict is RLTrainConfig kwargs with model/algorithm/rollout_source supplied at the point
# the CLI builds the real RLTrainConfig, and dataset replaced by the rollout source --
# config.py:387-399). build_agentic_config therefore never writes them.
_TRAINER_FORBIDDEN: Final[frozenset[str]] = frozenset({"model", "algorithm", "rollout_source", "dataset"})

# The goal schema spells learning_rate as "lr" (same alias the RL emit uses).
_TRAINER_ALIASES: Final[dict[str, str]] = {"learning_rate": "lr"}

# Normalisation defaults: the keys examples/agentic_rl/music_s0.json declares that the
# goal schema does not name. Every value is taken VERBATIM from that file -- nothing here
# is invented -- and it is applied only where the goal leaves the key ABSENT (a declared
# value always wins). Instance-specific values the example also carries (policy.model_path
# :5, rollout.tasks_path :75, trainer.save_dir :14, engine.command :20-26, engine.log_dir
# :33, engine.env :34-40, publish_root :88) are never defaulted: the goal declares them.
_MUSIC_S0_FILLS: Final[dict[str, dict[str, Any]]] = {
    "trainer": {
        "logprob_micro_batch": 0,  # music_s0.json:12
        "seed": 0,                 # music_s0.json:15
    },
    "engine": {
        "startup_timeout_s": 300.0,  # music_s0.json:29
        "host": "127.0.0.1",         # music_s0.json:32
    },
    "env": {
        "exec_timeout_s": 60.0,      # music_s0.json:45
        "episode_timeout_s": 600.0,  # music_s0.json:46
        "max_output_bytes": 65536,   # music_s0.json:47
    },
    "harness": {
        "name": "native_tool_loop",  # music_s0.json:53
        "tools": "default",          # music_s0.json:55
    },
    "rollout": {
        "group_by_harness": False,   # music_s0.json:79
        "seed": 0,                   # music_s0.json:80
    },
    "reward": {
        "timeout_s": 60.0,           # music_s0.json:86
    },
}


# --------------------------------------------------------------------------- planning


def plan_agentic(goal: Mapping[str, Any]) -> dict[str, Any]:
    """Plan one ``agentic_rl`` stage from ``goal["agentic"]`` (or the block itself).

    The stage record is what the planner/estimator and ``emit_agentic`` later consume::

        {"name": "agentic_rl", "stage": "agentic_rl", "algorithm": ..., "method": "full",
         "agentic": <normalised block>, "runtime": <Runtime dict>, "gpus": int,
         "executable": None}

    ``executable`` is None here: executability is measured later (probe + gates + FS
    ``--dry-run``), never decided at planning time. Raises ``ValueError`` only for a
    caller error (no agentic block at all, a section that is not a mapping, a malformed
    ``gpus``/``runtime`` declaration); every missing VALUE stays missing and the gates
    adjudicate it.
    """
    raw = _goal_block(goal)
    block = _normalise_block(raw)
    return {
        "name": _STAGE_NAME,
        "stage": _STAGE_NAME,
        "algorithm": block.get("algorithm"),
        "method": _METHOD,
        "agentic": block,
        "runtime": dict(block["runtime"]),
        "gpus": block["gpus"],
        "executable": None,
    }


def _goal_block(goal: Mapping[str, Any]) -> Mapping[str, Any]:
    """The agentic goal block: ``goal["agentic"]``, or the block passed directly."""
    if "agentic" in goal:
        block = goal.get("agentic")
        if not isinstance(block, Mapping):
            raise ValueError(
                "goal['agentic'] must be the agentic goal block (a mapping of algorithm/"
                f"model/tasks_path/engine/...), got {type(block).__name__}"
            )
        return block
    if any(key in goal for key in ("engine", "reward", "model", "algorithm")):
        # convenience: the caller passed the block itself rather than a goal wrapper
        return goal
    raise ValueError(
        "goal has no 'agentic' block and does not look like one (no engine/reward/model/"
        "algorithm keys): there is nothing to plan an agentic stage from"
    )


def _normalise_block(raw: Mapping[str, Any]) -> dict[str, Any]:
    """A normalised copy of the agentic block: explicit keys first, then sections.

    Section keys absent from the goal gain ONLY ``_MUSIC_S0_FILLS`` values (verbatim
    ``examples/agentic_rl/music_s0.json``), plus the one pairing the FS loader itself
    refuses without (engine.reload_mode for the vLLM reload_endpoint path). Declared
    values are never overwritten.
    """
    block: dict[str, Any] = {
        key: raw[key]
        for key in ("algorithm", "model", "tasks_path", "publish_root", "default_arm")
        if key in raw
    }
    block["gpus"] = _declared_int(raw.get("gpus", 1), "agentic.gpus")
    block["runtime"] = _normalise_runtime(raw.get("runtime"))
    stability = raw.get("stability")
    if stability is not None and not isinstance(stability, Mapping):
        raise ValueError(
            f"agentic.stability: expected a mapping or None (the optional collapse guard), "
            f"got {type(stability).__name__}"
        )
    block["stability"] = _copy_section(stability) if isinstance(stability, Mapping) else None

    for name in ("policy", "trainer", "engine", "env", "harness", "budget", "sampling",
                 "rollout", "reward"):
        value = raw.get(name)
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(
                f"agentic.{name}: expected a mapping of declared keys, got {type(value).__name__}"
            )
        section = _copy_section(value)
        for key, fill in _MUSIC_S0_FILLS.get(name, {}).items():
            section.setdefault(key, fill)
        if (
            name == "engine"
            and section.get("kind") == "vllm"
            and section.get("weight_sync_mode") == "reload_endpoint"
        ):
            # music_s0.json:31 pairs weight_sync_mode='reload_endpoint' with
            # reload_mode='collective_rpc'; the EngineConfig loader REFUSES the pairing
            # without it (config.py:303-310) after a GPU is already resident. Fill the
            # example's value -- never some other mode -- and only if silent.
            section.setdefault("reload_mode", "collective_rpc")
        block[name] = section
    return block


def _copy_section(value: Mapping[str, Any] | None) -> dict[str, Any]:
    section = dict(value or {})
    # music_s0.json carries a '_doc' string in every section (JSON has no comments): it is
    # documentation, never a declared key, and the loader strips it before validating.
    section.pop("_doc", None)
    return section


def _declared_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{where}: expected an integer GPU count, got {value!r}")
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: expected an integer GPU count, got {value!r}") from None
    if count < 1:
        raise ValueError(
            f"{where}: got {count} -- at least one tray is declared (the measured agentic "
            f"control runs the whole stack on 1 GB200)"
        )
    return count


def _normalise_runtime(value: Any) -> dict[str, Any]:
    """``goal.agentic.runtime`` as the canonical Runtime dict (absent means ``host``)."""
    if value is not None and not isinstance(value, Mapping):
        raise ValueError(
            f"agentic.runtime: expected a Runtime mapping (kind/mounts/env/...) or None, "
            f"got {type(value).__name__}"
        )
    rt: Runtime = runtime_from_dict(value if isinstance(value, Mapping) else None)
    return rt.to_dict()


# ----------------------------------------------------------------------------- gates


def agentic_gates(stage: Mapping[str, Any], agentic_caps: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """Run the agentic input/handoff gates over one planned stage.

    ``agentic_caps`` is the ``FSCapabilities.agentic`` block (the ``probe_agentic``
    output). Every finding is ``(rule_id, message)``; severity per ``GATE_SEVERITY``.
    A capability the probe did not measure is reported as UNMEASURED (TR-IN-010) and
    never assumed -- a missing measurement is not a passing one.
    """
    block = _stage_block(stage)
    findings: list[tuple[str, str]] = []
    findings.extend(_gates_probe(block, agentic_caps))
    findings.extend(_gates_multi_gpu(stage, block, agentic_caps))
    findings.extend(_gates_handles(block))
    findings.extend(_gates_staleness(block))
    findings.extend(_gates_engine_env(block))
    findings.extend(_gates_default_arm(block))
    return findings


def blocking_findings(findings: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The subset of ``agentic_gates`` findings that refuse the plan (BLOCK severity)."""
    return [finding for finding in findings if finding[0] in BLOCK_RULES]


def _stage_block(stage: Mapping[str, Any]) -> Mapping[str, Any]:
    if "agentic" in stage:
        block = stage.get("agentic")
        return block if isinstance(block, Mapping) else {}
    return stage  # the caller passed the agentic block directly


def _section(block: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = block.get(name)
    return value if isinstance(value, Mapping) else {}


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _gates_probe(block: Mapping[str, Any], caps: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """TR-IN-010: every named component must be one the runtime probe measured."""
    findings: list[tuple[str, str]] = []
    probed: Mapping[str, Any] = caps if isinstance(caps, Mapping) else {}
    measured = bool(probed) and probed.get("measured") is not False

    def member(cap_key: str, value: Any, subject: str, noun: str) -> None:
        listed = probed.get(cap_key) if measured else None
        if not isinstance(listed, (list, tuple, set, frozenset)):
            findings.append((
                "TR-IN-010",
                f"{subject} {value!r} is UNMEASURED: the agentic probe recorded no "
                f"{cap_key} for the declared runtime -- {_PROBE_REMEDY}",
            ))
            return
        if value in listed:
            return
        findings.append((
            "TR-IN-010",
            f"{subject} {value!r} is not among the {noun} the runtime probe measured in "
            f"{cap_key} ({sorted(str(name) for name in listed)}) -- nothing registered "
            f"there executes it; {_PROBE_REMEDY} (or declare a probed one)",
        ))

    member("algorithms", block.get("algorithm"), "algorithm", "registered agentic objectives")

    engine = _section(block, "engine")
    kind = engine.get("kind")
    engines_value = probed.get("engines_importable") if measured else None
    engines: Mapping[str, Any] = engines_value if isinstance(engines_value, Mapping) else {}
    if engines.get(kind) is not True:
        findings.append((
            "TR-IN-010",
            f"engine.kind {kind!r} is not importable inside the declared runtime (probe "
            f"engines_importable measured {dict(engines)!r}) -- the serving engine cannot "
            f"start there; {_PROBE_REMEDY}",
        ))

    member("env_backends", _section(block, "env").get("backend"), "env.backend",
           "registered env backends")
    member("reward_kinds", _section(block, "reward").get("kind"), "reward.kind",
           "declared reward kinds")
    member("parsers", _section(block, "harness").get("parser"), "harness.parser",
           "harness parser names")
    return findings


def _gates_multi_gpu(
    stage: Mapping[str, Any], block: Mapping[str, Any], caps: Mapping[str, Any] | None
) -> list[tuple[str, str]]:
    """TR-IN-011: refuse an unmeasured multi-GPU claim."""
    probed: Mapping[str, Any] = caps if isinstance(caps, Mapping) else {}
    if probed.get("multi_gpu") is True:
        return []
    gpus = _as_number(block.get("gpus", stage.get("gpus", 1)))
    sharding = _section(block, "trainer").get("sharding")
    reasons: list[str] = []
    if gpus is not None and gpus > 1:
        reasons.append(f"gpus={int(gpus)}")
    if sharding != "none":
        # The rule compares trainer.sharding to the measured single-device "none"
        # literally: an UNDECLARED sharding is not measured to be "none" either, so it is
        # refused here instead of assumed (absence of a declaration is UNMEASURED).
        detail = " (undeclared -- never assumed to be the measured single-device 'none')"
        reasons.append(f"trainer.sharding={sharding!r}" + ("" if sharding is not None else detail))
    if not reasons:
        return []
    return [(
        "TR-IN-011",
        "agentic multi-GPU is UNMEASURED (agentic_caps.multi_gpu is not True) but the "
        f"plan declares {' and '.join(reasons)} -- one GPU tray with "
        "trainer.sharding='none' is the only measured agentic control; drop to it or "
        "measure a multi-GPU probe slice first",
    )]


def _gates_handles(block: Mapping[str, Any]) -> list[tuple[str, str]]:
    """TR-IN-012: every handle a scorer/runner reads must be present and readable."""
    findings: list[tuple[str, str]] = []
    reward = _section(block, "reward")
    rollout = _section(block, "rollout")
    env = _section(block, "env")

    if reward.get("kind") == "music":
        handle = reward.get("abc2midi_bin")
        if not isinstance(handle, str) or not handle.strip():
            findings.append((
                "TR-IN-012",
                "reward handle missing: kind 'music' scores submitted ABC through "
                "abc2midi but reward.abc2midi_bin is not declared -- the scorer has "
                "nothing to run, and no score may be invented",
            ))
        elif not _executable_on_host(handle):
            findings.append((
                "TR-IN-012",
                f"reward handle missing: reward.abc2midi_bin {handle!r} is not an "
                "executable file on the host (os.access X_OK failed) -- checked on the "
                "host path as given (the runtime mounts $HOME at the same place), and no "
                "score may be invented",
            ))

    tasks = rollout.get("tasks_path", block.get("tasks_path"))
    if not _readable_file(tasks):
        findings.append((
            "TR-IN-012",
            f"rollout task pool missing: rollout.tasks_path {tasks!r} is not a readable "
            "file -- a step cannot draw tasks it cannot read, and none may be invented",
        ))

    backend = env.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        findings.append((
            "TR-IN-012",
            f"env.backend {backend!r} is not set -- an episode has nowhere to run and no "
            "backend may be assumed",
        ))
    elif backend in _UNSANDBOXED_BACKENDS and env.get("allow_unsandboxed") is not True:
        findings.append((
            "TR-IN-012",
            f"env backend {backend!r} has isolation 'none' (model-chosen shell commands "
            "run directly on the host) but allow_unsandboxed is not declared True -- the "
            "explicit opt-in EnvSpec/check_isolation require is missing",
        ))
    return findings


def _executable_on_host(value: str) -> bool:
    """True when value names an executable: the path as given (the runtime mounts $HOME at
    the same place), or a PATH-resolved bare name the way music_s0.json:85 declares
    ``abc2midi``."""
    if os.path.isfile(value) and os.access(value, os.X_OK):
        return True
    if os.sep not in value:
        resolved = shutil.which(value)
        return bool(resolved and os.path.isfile(resolved) and os.access(resolved, os.X_OK))
    return False


def _readable_file(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and os.path.isfile(value)
        and os.access(value, os.R_OK)
    )


def _gates_engine_env(block: Mapping[str, Any]) -> list[tuple[str, str]]:
    """TR-IN-014: a declared engine.env is the server's COMPLETE environment (FS never merges os.environ,
    engines/fleet.py:86,277). Without PATH the server cannot find its binaries/libraries (measured A4.2:
    libcudart.so.12 not found); a dev-RPC reload needs VLLM_SERVER_DEV_MODE=1 (config.py:315). An absent
    engine.env is filled from the measured recipe by build_agentic_config, so only a declared one is checked."""
    engine = _section(block, "engine")
    env = engine.get("env")
    if engine.get("kind") != "vllm" or not isinstance(env, Mapping):
        return []
    out: list[tuple[str, str]] = []
    if not env.get("PATH"):
        out.append(("TR-IN-014", "engine.env is declared without PATH: FS starts the vLLM server with exactly this "
                                 "environment (never merged with os.environ), so it cannot load its CUDA libraries; "
                                 "declare PATH (and HOME, CUDA_VISIBLE_DEVICES) or omit engine.env for the measured recipe"))
    if (engine.get("reload_mode") == "collective_rpc" or engine.get("weight_sync_mode") == "reload_endpoint") \
            and str(env.get("VLLM_SERVER_DEV_MODE", "")) != "1":
        out.append(("TR-IN-014", "engine.weight_sync_mode reload_endpoint / reload_mode collective_rpc needs "
                                 "VLLM_SERVER_DEV_MODE=1 in engine.env (vLLM's dev RPC); without it every weight "
                                 "sync fails"))
    return out


def _gates_staleness(block: Mapping[str, Any]) -> list[tuple[str, str]]:
    """TR-IN-013: more than one step needs a weight sync or an explicit staleness opt-in."""
    trainer = _section(block, "trainer")
    engine = _section(block, "engine")
    max_steps = _as_number(trainer.get("max_steps"))
    if max_steps is None or max_steps <= 1:
        return []
    if engine.get("weight_sync_mode") not in (None, "none"):
        return []
    if engine.get("allow_stale_policy") is True:
        return []
    return [(
        "TR-IN-013",
        f"trainer.max_steps={int(max_steps)} runs more than one step while "
        f"engine.weight_sync_mode is {engine.get('weight_sync_mode')!r} (no weight sync "
        "wired) and engine.allow_stale_policy is not declared True -- every rollout after "
        "the first would silently serve the INITIAL weights and be off-policy without the "
        "run ever saying so; declare a weight sync (reload_endpoint/restart) or the "
        "explicit staleness opt-in",
    )]


def _gates_default_arm(block: Mapping[str, Any]) -> list[tuple[str, str]]:
    """TR-PL-009 (WARN): the measured control named as the default arm, on the record."""
    if block.get("algorithm") != "agentic_grpo_token_mean":
        return []
    if block.get("default_arm") is not True:
        return []
    return [(
        "TR-PL-009",
        "algorithm 'agentic_grpo_token_mean' (the measured control) is declared as the "
        "default arm -- keep the arm id and its comparison partner on the handoff record "
        "so the result stays reproducible",
    )]


# ------------------------------------------------------------------ FS config rendering


def build_agentic_config(stage: Mapping[str, Any]) -> dict[str, Any]:
    """Render the ``AgenticRLConfig`` JSON for one planned stage.

    Sections and their key order mirror ``examples/agentic_rl/music_s0.json`` (policy,
    trainer, engine, env, harness, budget, sampling, rollout, reward, publish_root, and
    ``stability`` only when the goal declared a guard). Only field names
    ``foundationscale.agentic_rl.config`` declares are written (``_SECTION_ORDER``), so a
    stray goal key can never break the loader. ``provenance`` is never written here (the
    loader records it), and the forbidden trainer names ``model``/``algorithm``/
    ``rollout_source``/``dataset`` (config.py:387-399) are never written either.
    """
    block = _stage_block(stage)
    model = block.get("model")

    policy_src = _copy_section(block.get("policy"))
    policy_src.setdefault("model_path", model)
    policy_src.setdefault("algorithm", block.get("algorithm"))

    rollout_src = _copy_section(block.get("rollout"))
    # the goal schema names tasks_path at block level; RolloutConfig reads it in rollout.
    if "tasks_path" not in rollout_src and block.get("tasks_path"):
        rollout_src["tasks_path"] = block["tasks_path"]

    engine_src = _copy_section(block.get("engine"))
    engine_cfg = _build_section(engine_src, _SECTION_ORDER["engine"])
    if "command" not in engine_cfg and engine_src.get("kind") == "vllm":
        synth = _gb200_vllm_command(model, engine_src)
        if synth is not None:
            engine_cfg["command"] = synth
    if "env" not in engine_cfg and engine_src.get("kind") == "vllm":
        engine_cfg["env"] = _gb200_engine_env(block, engine_src)

    config: dict[str, Any] = {
        "policy": _build_section(policy_src, _SECTION_ORDER["policy"]),
        "trainer": _build_section(
            _copy_section(block.get("trainer")), _SECTION_ORDER["trainer"], _TRAINER_ALIASES
        ),
        "engine": engine_cfg,
        "env": _build_section(_copy_section(block.get("env")), _SECTION_ORDER["env"]),
        "harness": _build_section(_copy_section(block.get("harness")), _SECTION_ORDER["harness"]),
        "budget": _build_section(_copy_section(block.get("budget")), _SECTION_ORDER["budget"]),
        "sampling": _build_section(_copy_section(block.get("sampling")), _SECTION_ORDER["sampling"]),
        "rollout": _build_section(rollout_src, _SECTION_ORDER["rollout"]),
        "reward": _build_section(_copy_section(block.get("reward")), _SECTION_ORDER["reward"]),
    }
    if block.get("publish_root") is not None:
        config["publish_root"] = _jsonable(block.get("publish_root"))
    stability = block.get("stability")
    if isinstance(stability, Mapping):
        config["stability"] = _build_section(stability, _SECTION_ORDER["stability"])
    return config


def _build_section(
    source: Mapping[str, Any],
    fields: tuple[str, ...],
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Declared values for the fields config.py declares, in music_s0's order.

    ``None`` is written nowhere ("no declaration"), while ``False``/``0`` are declarations
    and pass through. ``aliases`` maps a config field to its goal spelling (``lr`` is the
    goal spelling of ``learning_rate``). The forbidden trainer names are never in a
    ``fields`` tuple, so they can never reach the JSON.
    """
    src = dict(source)
    for field, alias in (aliases or {}).items():
        if field not in src and alias in src:
            src[field] = src[alias]
    out: dict[str, Any] = {}
    for field in fields:
        if field in src and src[field] is not None:
            out[field] = _jsonable(src[field])
    return out


def _gb200_vllm_command(model: Any, engine: Mapping[str, Any]) -> list[str] | None:
    """The vLLM serve argv of the MEASURED GB200 single-GPU agentic run (ablate_tm_s3: ``vllm serve <model>
    --served-model-name <n> --port <p> --gpu-memory-utilization 0.5 --max-model-len 16384``), with this
    stage's identity substituted. 0.5, not music_s0's 0.85: on one GPU the trainer shares the device (the
    S0 example targets a separate engine GPU). ``max_model_len`` is a goal-side knob, never an FS config key.
    Nothing is synthesised when model/served name/port are absent -- FS ``--dry-run`` then names it."""
    served = engine.get("served_model_name")
    port = engine.get("port")
    if not isinstance(model, str) or not model:
        return None
    if not isinstance(served, str) or not served or isinstance(port, bool) or port is None:
        return None
    try:
        port_text = str(int(port))
        max_len = str(int(engine.get("max_model_len", 16384)))
        util = float(engine.get("gpu_memory_utilization", 0.5))
    except (TypeError, ValueError):
        return None
    if not 0.0 < util < 1.0:
        return None
    # util is a goal-side knob (not an FS config key): the measured 0.5 left a 9B full fine-tune ~91 GiB on
    # main's trainer, which OOMed (A4.2 GPU run); a declared lower share hands the trainer the difference
    return ["vllm", "serve", model, "--served-model-name", served, "--port", port_text,
            "--gpu-memory-utilization", f"{util:g}", "--max-model-len", max_len]


def _gb200_engine_env(block: Mapping[str, Any], engine: Mapping[str, Any]) -> dict[str, str]:
    """FS hands the engine server a COMPLETE environment, never merged with os.environ
    (engines/fleet.py:86,277): with no engine.env the server starts with nothing and vLLM fails to load
    libcudart (measured, A4.2 GPU run). This is the measured ablate_tm_s3 engine env; VLLM_SERVER_DEV_MODE
    is only set for the dev-RPC reload that needs it (config.py:315)."""
    runtime = block.get("runtime") if isinstance(block.get("runtime"), Mapping) else {}
    runtime_env = dict((str(k), str(v)) for k, v in (runtime.get("env") or ()))
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin",
           "HOME": runtime_env.get("HOME") or os.environ.get("HOME", ""),
           "CUDA_VISIBLE_DEVICES": str(engine.get("cuda_visible_devices", runtime_env.get("CUDA_VISIBLE_DEVICES", "0"))),
           "PYTHONNOUSERSITE": "1"}
    if engine.get("reload_mode") == "collective_rpc" or engine.get("weight_sync_mode") == "reload_endpoint":
        env["VLLM_SERVER_DEV_MODE"] = "1"
    return env


def _jsonable(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
