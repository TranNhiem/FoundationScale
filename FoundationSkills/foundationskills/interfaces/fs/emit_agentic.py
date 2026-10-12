"""Emit an ``fs_launch_spec`` payload for a FoundationScale agentic RL stage.

FS owns the agentic schema end to end (``foundationscale.agentic_rl.config``,
``cli.py``), so this emitter invents nothing: it (1) runs the skill's agentic
RULE gates over the normalised stage and the measured runtime capabilities,
(2) writes the FS ``AgenticRLConfig`` the skill's ``build_agentic_config``
produced, and (3) proves that config with FS's own ``--dry-run`` INSIDE the
declared runtime (FS-HO-005). The emitted ``argv`` wraps the FS module
invocation (``-m foundationscale.agentic_rl --config <path>``) in exactly the
``enroot start`` envelope the measured launcher uses -- the agentic stack has
no host vLLM, so a host launch would be a different (unmeasured) experiment.

The companion handoff rule FS-HO-006 is claim-time, not emit-time: a record
that maps TIMEOUT/STEP_LIMIT to abstention or claims PASS over zero measured
steps is REFUSED where the claim is made (``rl_driver``/CLI), which is why
``emitted spec`` carries no score and no verdict here.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from foundationskills.interfaces.fs.agentic_probe import (
    runtime_argv,
    runtime_env,
    runtime_from_dict,
)

try:  # the agentic skill layer supplies the FS config and the rule gates
    from foundationskills.skills.training.agentic import (  # type: ignore[import-not-found]
        agentic_gates,
        build_agentic_config,
    )
except Exception:  # noqa: BLE001 - absent skill layer: emit reports it, never substitutes one
    agentic_gates = None  # type: ignore[assignment]
    build_agentic_config = None  # type: ignore[assignment]

#: FS's --dry-run is a config check, bound like the probe (it may resolve a
#: tokenizer); a run that cannot finish inside this is reported, not waited on.
_DRY_RUN_TIMEOUT_S: float = 300.0

#: Rule-id phases that refuse execution (-PL- notes are handoff WARNs).
_BLOCKER_PHASES: tuple[str, ...] = ("IN", "HO")


def _is_blocking(rule_id: str) -> bool:
    """An ``*-IN-*``/``*-HO-*`` rule id blocks; an unclassifiable id also blocks.

    An id this module cannot classify is treated as a REFUSAL -- an unknown
    check failing open is exactly the "absence of measurement read as
    evidence" failure the doctrine forbids.
    """

    parts = str(rule_id or "").split("-")
    if len(parts) >= 3 and parts[0] in ("TR", "FS"):
        return parts[1] in _BLOCKER_PHASES
    return True


def _proc_tail(proc: Any) -> str:
    """Last lines of the dry-run's stderr (else stdout) -- the refusal body."""

    stderr = (getattr(proc, "stderr", "") or "").strip()
    stdout = (getattr(proc, "stdout", "") or "").strip()
    kept = [line for line in (stderr or stdout).splitlines() if line.strip()]
    return "\n".join(kept[-5:])


def emit_agentic(
    stage: Mapping[str, Any],
    *,
    output_dir: str,
    run_name: str,
    fs_src: str,
    agentic_caps: Mapping[str, Any] | None,
    dry_run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Build one ``fs_launch_spec`` payload for an agentic RL stage.

    Executable only when ALL hold: the agentic capabilities were measured in
    the declared runtime and make the engine importable there (FS-IN-005),
    every BLOCK gate from ``agentic_gates`` passes, the FS config is written,
    and FS's own ``--dry-run`` of that config exits 0 (FS-HO-005). Otherwise
    ``missing`` NAMES why (dry-run rc and its output tail included) -- no
    score, no verdict, no invented schema.
    """

    notes: list[str] = []
    missing: list[str] = []
    agentic = dict(stage.get("agentic") or {})
    rt = runtime_from_dict(stage.get("runtime") or agentic.get("runtime"))

    caps_measured = bool(agentic_caps) and (agentic_caps or {}).get("measured") is True
    engines_importable = (agentic_caps or {}).get("engines_importable") or {}
    engine_kind = str((agentic.get("engine") or {}).get("kind") or "vllm")

    # FS-IN-005: an agentic stage without a runtime the probe measured is
    # REFUSED -- naming the container, because that is what must be probed.
    if not caps_measured or engines_importable.get(engine_kind) is not True:
        missing.append(
            "FS-IN-005: engine "
            f"{engine_kind!r} is not measured importable in the declared runtime "
            f"(kind={rt.kind!r}, container={rt.container!r}); run the agentic probe in that "
            f"runtime before this stage is executable"
        )

    config_path = f"{output_dir}/agentic_rl_config.json"
    argv = runtime_argv(
        rt,
        [rt.python, "-m", "foundationscale.agentic_rl", "--config", config_path],
        extra_env={"PYTHONPATH": fs_src},
    )
    dry_run_argv = [*argv, "--dry-run"]

    gate_hits = list(agentic_gates(stage, agentic_caps) if agentic_gates is not None else [])
    blocking_rules: list[str] = []
    for rule_id, message in gate_hits:
        if _is_blocking(rule_id):
            blocking_rules.append(str(rule_id))
            missing.append(f"{rule_id}: {message}")
        else:
            notes.append(f"{rule_id} (WARN): {message}")

    config_built = False
    if build_agentic_config is None:
        missing.append(
            "build_agentic_config is unavailable (foundationskills.skills.training.agentic not "
            "importable): no FS AgenticRLConfig was written"
        )
    else:
        config = build_agentic_config(stage)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(config_path).write_text(json.dumps(config, indent=2), encoding="utf-8")
        config_built = True
        notes.append(f"wrote FS AgenticRLConfig to {config_path}")

    # FS-HO-005: FS owns the schema -- only FS's own --dry-run may clear a config.
    rc: int | None = None
    tail = ""
    if config_built:
        run_env = {**os.environ, **runtime_env(rt)}
        try:
            proc = dry_run(
                dry_run_argv,
                env=run_env,
                capture_output=True,
                text=True,
                timeout=_DRY_RUN_TIMEOUT_S,
            )
            rc = int(getattr(proc, "returncode", 0))
            tail = _proc_tail(proc)
        except Exception as exc:  # noqa: BLE001 - a handoff never raises at an operator
            tail = f"{type(exc).__name__}: {exc}"
            missing.append(f"FS-HO-005: FS --dry-run of {config_path} could not run: {tail}")
        else:
            if rc != 0:
                missing.append(
                    f"FS-HO-005: FS --dry-run of {config_path} did not exit 0 (rc={rc}); tail: {tail}"
                )

    executable = bool(config_built and caps_measured and not blocking_rules and rc == 0)
    if not executable and not missing:
        missing.append(
            "executable conditions unmet (measured agentic caps, no BLOCK gate, FS --dry-run rc == 0)"
        )

    return {
        "kind": "agentic_rl",
        "run_name": run_name,
        "config_path": config_path,
        "argv": argv,
        "dry_run_argv": dry_run_argv,
        "env": runtime_env(rt),
        "executable": executable,
        "missing": None if executable else "; ".join(missing),
        "notes": notes,
        "dry_run": {"rc": rc, "tail": tail},
    }
