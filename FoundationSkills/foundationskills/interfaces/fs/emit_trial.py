"""Render ONE auto_research trial over the FS training/eval emitters and gate its submission.

``emit_trial`` builds one trial fact and never submits. ``kind="eval_only"`` (the
three noise-floor baseline repeats, operator decision 3) is emitted by
``emit_eval`` with the trial's ``nodes``/``gpus_per_node``, ``hardware_id`` and
``probe`` (forwarded as ``version_probe``: its facts are advisory -- they land in
``notes`` and gate nothing at emit). ``kind="train"`` calls
``emit_train_fn(**train_request)`` verbatim (the request is opaque to
auto_research) and, when the repo ships no ``interfaces.fs.emit_train``, returns
the REFUSED fact ``{"state": "REFUSED", "reason": "emit_train_missing"}``: a
missing train emit is never silently degraded into an eval.

``submit_trial`` gates in front of ``launch`` (order is deliberate, do not
reorder): AR-LN-006 token equality first, then AR-LN-003 fabric state -- an
IMEX fabric that is not ``ready`` (refused OR unmeasured) never launches,
because an unknown fabric is not positive evidence -- then ``launch_fn`` with
``confirm`` forwarded untouched (``launch`` re-checks it via
``require_confirmation`` over ``plan_hash`` of the full fs_launch_spec). On any
refusal neither the sbatch file nor the runner is touched.

The rendered sbatch is asserted to carry ``--time=10-00:00:00``: the estate rule
must hold for the render this trial would ship, so a render bug raises
``ValueError`` loudly instead of earning a job on a wrong wall clock.
"""
from __future__ import annotations

import importlib
import subprocess
from typing import Any, Callable

from foundationskills.core.orchestrator import plan_hash
from foundationskills.interfaces.fs.emit_eval import emit_eval
from foundationskills.interfaces.fs.launch import LaunchRefused, launch

__all__ = ("emit_trial", "submit_trial")

try:  # interfaces.fs.fabric is its own unit; without it this gate stays conservative
    from foundationskills.interfaces.fs.fabric import launch_blocking
except ImportError:  # pragma: no cover - only when that module is absent

    def launch_blocking(state: str) -> bool:
        """Only ``ready`` is evidence to launch: refused AND unmeasured both block."""
        return state != "ready"


_KINDS = {"eval_only": "eval_request", "train": "train_request"}
_SBOTIME = "--time=10-00:00:00"


def _load_emit_train() -> Callable[..., dict[str, Any]] | None:
    """``interfaces.fs.emit_train.emit_train`` when the repo ships one; None (REFUSED) when it does not."""
    try:
        module = importlib.import_module("foundationskills.interfaces.fs.emit_train")
    except ImportError:
        return None
    fn = getattr(module, "emit_train", None)
    return fn if callable(fn) else None


def _refused(reason: str, trial_spec: dict[str, Any]) -> dict[str, Any]:
    """A REFUSED fact: nothing rendered, nothing submitted; the reason is named and counted as a drop."""
    fact = {"state": "REFUSED", "reason": reason}
    return {
        "fs_launch_spec": fact,
        "trial_spec": trial_spec,
        "confirm": plan_hash(fact),
        "notes": [],
        "drops": [reason],
        "executable": False,
        "missing": [reason],
    }


def emit_trial(
    request: dict[str, Any],
    *,
    hardware_id: str = "gb200",
    emit_train_fn: Callable[..., dict[str, Any]] | None = None,
    emit_eval_fn: Callable[..., dict[str, Any]] = emit_eval,
    probe: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Emit one trial fact: {fs_launch_spec, trial_spec, confirm, notes, drops, executable, missing}.

    ``confirm`` is ``plan_hash`` over the emitted fs_launch_spec -- the same hash
    ``launch`` requires -- and must never be recomputed or filled in downstream.
    """
    payload = request or {}
    trial_spec = payload.get("trial_spec") or {}
    kind = str(trial_spec.get("kind") or "")
    key = _KINDS.get(kind)
    if key is None:
        return _refused(f"trial_kind_unknown:{kind}", trial_spec)
    trial_request = trial_spec.get(key)
    if not isinstance(trial_request, dict):
        return _refused(f"emit_request_missing:{key}", trial_spec)

    overrides: list[str] = []
    if kind == "train":
        fn = emit_train_fn if emit_train_fn is not None else _load_emit_train()
        if fn is None:
            return _refused("emit_train_missing", trial_spec)
        train_args = dict(trial_request)
        # G2 (gate what actually runs): the RENDERED node shape is never taken from the opaque
        # train_request - nodes/gpus_per_node are forced to the trial_spec values and the override is named.
        for shape_key in ("nodes", "gpus_per_node"):
            forced = int(trial_spec.get(shape_key) or 1)
            carried = train_args.get(shape_key)
            if shape_key in train_args and carried is not None and carried != forced:
                overrides.append(f"overrode train_request.{shape_key}")
            train_args[shape_key] = forced
        spec = fn(**train_args)  # train_request is opaque to auto_research beyond the forced shape keys
    else:
        spec = emit_eval_fn(
            trial_request,
            nodes=int(trial_spec.get("nodes") or 1),
            gpus_per_node=int(trial_spec.get("gpus_per_node") or 1),
            hardware_id=hardware_id,
            version_probe=probe,
        )

    sbatch = spec.get("sbatch")
    if isinstance(sbatch, str) and _SBOTIME not in sbatch:
        raise ValueError(f"render bug: sbatch for trial {trial_spec.get('trial')!r} lacks {_SBOTIME}")

    notes = [*overrides, *(str(n) for n in (spec.get("notes") or []))]
    if overrides:  # the named overrides ride with the render as well as the fact notes
        spec = {**spec, "notes": notes}
    return {
        "fs_launch_spec": spec,
        "trial_spec": trial_spec,
        "confirm": plan_hash(spec),
        "notes": notes,
        "drops": [str(d) for d in (spec.get("drops") or [])],
        "executable": spec.get("executable") is True,
        "missing": [str(m) for m in (spec.get("missing") or [])],
    }


def submit_trial(
    fs_launch_spec: dict[str, Any],
    *,
    expected_token: str,
    supplied_token: str,
    confirm: str,
    fabric: dict[str, Any],
    launch_fn: Callable[..., dict[str, Any]] = launch,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Submit one emitted trial behind AR-LN-006 (token) and AR-LN-003 (fabric); returns the ``launch()`` dict.

    A refusal is a ``LaunchRefused`` carrying the rule id and the named reason,
    raised before any sbatch write or runner call -- tests assert the recorded
    runner argv stays empty.
    """
    if not expected_token or supplied_token != expected_token:
        raise LaunchRefused("AR-LN-006 token_mismatch")
    state = str((fabric or {}).get("state") or "unmeasured")
    if launch_blocking(state):
        reason = str((fabric or {}).get("reason") or f"fabric_{state}")
        raise LaunchRefused(f"AR-LN-003 {reason}")
    return launch_fn(fs_launch_spec, confirm=confirm, runner=runner)
