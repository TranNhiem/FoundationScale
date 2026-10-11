"""Agentic-RL capability probe, measured INSIDE the runtime a stage declares.

FoundationScale (``foundationscale.agentic_rl``) owns the whole agentic
plane: config schema, CLI, contracts, rewards, envs and the engine gateway.
This module reimplements none of it and adds no dependency -- it only
MEASURES what that plane exposes, so a planner can gate on measurement
instead of assumption (doctrine: a missing/failed field is UNMEASURED and is
never assumed).

The probe runs inside a declared ``Runtime`` for a measured reason (real
ablation run ``~/fsar-data/ablate_tm_s3``, 2026-10-11): the agentic stack is
launched through ``enroot start ... vllmsrv3 python3 -m
foundationscale.agentic_rl``, and the host has no vLLM, so a host probe would
truthfully report every engine absent and every agentic plan would read
UNMEASURED forever. ``runtime_argv`` wraps both the probe and (in
``emit_agentic``) the FS module invocation in exactly the ``enroot start``
envelope the measured launcher uses.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Python source executed INSIDE the declared runtime. It prints exactly ONE
#: JSON line (``fs_version``, ``algorithms``, ``engines_importable``,
#: ``env_backends``, ``reward_kinds``, ``parsers``, ``errors``) and NEVER
#: raises: every sub-probe is wrapped in try/except and appends to ``errors``
#: instead of failing the whole measurement.
AGENTIC_PROBE_SNIPPET: str = '''\
"""Measure FS's agentic RL plane; print exactly one JSON line; never raise."""
from __future__ import annotations

import json
import typing


def _push(errors, label, exc):
    errors.append("%s: %s: %s" % (label, type(exc).__name__, exc))


def _literal_args(owner, field_name, errors, label):
    """``typing.get_args`` of one dataclass field annotation (``[]`` when unreadable)."""
    try:
        import dataclasses

        annotation = None
        for f in dataclasses.fields(owner):
            if f.name == field_name:
                annotation = f.type
                break
        args = typing.get_args(annotation) if annotation is not None else ()
        if not args:
            # config.py may defer annotations (PEP 563): resolve before get_args.
            hints = typing.get_type_hints(owner)
            args = typing.get_args(hints.get(field_name, ()))
        return sorted({str(a) for a in args})
    except Exception as exc:
        _push(errors, label, exc)
        return []


def _probe_fs_version(errors):
    try:
        import importlib.metadata

        version = importlib.metadata.version("foundationscale")
        if version:
            return str(version)
    except Exception:
        pass
    try:
        import foundationscale

        return str(getattr(foundationscale, "__version__", ""))
    except Exception as exc:
        _push(errors, "fs_version", exc)
        return ""


def _probe_algorithms(errors):
    """Registered agentic objective names: the foundationscale.agentic_rl
    registry when this build exposes one, plus the foundationscale.rl
    registry filtered to names starting 'agentic_'."""
    agentic_hits, rl_hits = {}, {}
    targets = (
        ("foundationscale.agentic_rl",
         ("registry", "registered_algorithms", "algorithms", "available_algorithms",
          "OBJECTIVES", "_REGISTRY"), agentic_hits),
        ("foundationscale.rl",
         ("registry", "registered_algorithms", "algorithms", "available_algorithms",
          "_REGISTRY"), rl_hits),
        ("foundationscale.rl.registry",
         ("registered_algorithms", "algorithms", "available_algorithms", "_REGISTRY"), rl_hits),
    )
    for mod_name, attrs, sink in targets:
        try:
            mod = __import__(mod_name, fromlist=["*"])
        except Exception as exc:
            _push(errors, "algorithms.import " + mod_name, exc)
            continue
        for attr in attrs:
            try:
                obj = getattr(mod, attr, None)
                value = obj() if callable(obj) else obj
                if isinstance(value, dict):
                    sink.update({str(k): True for k in value})
                elif isinstance(value, (list, tuple, set, frozenset)):
                    sink.update({str(k): True for k in value})
            except Exception as exc:
                _push(errors, "algorithms.%s.%s" % (mod_name, attr), exc)
    merged = dict(agentic_hits)
    merged.update({k: True for k in rl_hits if k.startswith("agentic_")})
    return sorted(merged)


def _probe_engines(errors):
    out = {}
    try:
        import importlib.util
    except Exception as exc:
        _push(errors, "engines_importable", exc)
        return out
    for name in ("vllm", "sglang"):
        try:
            out[name] = importlib.util.find_spec(name) is not None
        except Exception as exc:
            _push(errors, "engines_importable." + name, exc)
            out[name] = False
    return out


def _probe_env_backends(errors):
    try:
        from foundationscale.agentic_rl.envs.base import available_env_backends

        return sorted({str(n) for n in available_env_backends()})
    except Exception as exc:
        _push(errors, "env_backends", exc)
        return []


def _probe_reward_kinds(errors):
    try:
        from foundationscale.agentic_rl.config import RewardConfig
    except Exception as exc:
        _push(errors, "reward_kinds", exc)
        return []
    return _literal_args(RewardConfig, "kind", errors, "reward_kinds")


def _probe_parsers(errors):
    try:
        from foundationscale.agentic_rl.config import HarnessConfig
    except Exception as exc:
        _push(errors, "parsers", exc)
        return []
    return _literal_args(HarnessConfig, "parser", errors, "parsers")


def main():
    errors = []
    payload = {
        "fs_version": _probe_fs_version(errors),
        "algorithms": _probe_algorithms(errors),
        "engines_importable": _probe_engines(errors),
        "env_backends": _probe_env_backends(errors),
        "reward_kinds": _probe_reward_kinds(errors),
        "parsers": _probe_parsers(errors),
        "errors": errors,
    }
    print(json.dumps(payload, default=str))


if __name__ == "__main__":
    main()
'''

#: Keys the probe snippet is contracted to report; a payload missing one of
#: them is padded (never filtered down), so gates can read measurement
#: honestly instead of guessing.
_PROBE_KEYS: tuple[str, ...] = (
    "fs_version",
    "algorithms",
    "engines_importable",
    "env_backends",
    "reward_kinds",
    "parsers",
    "errors",
)


@dataclass(frozen=True)
class Runtime:
    """The wrapper an agentic stage's argv and probe run inside.

    ``kind="host"`` runs the given command verbatim on this machine;
    ``kind="enroot"`` wraps it in the ``enroot start`` envelope of the
    measured launcher (``enroot start --rw --mount src:dst --env k=v
    <container> <cmd...>``), so the emitted argv is EXACTLY the one that was
    measured, never a re-assembled lookalike.
    """

    kind: str = "host"
    container: str | None = None
    mounts: tuple[tuple[str, str], ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    enroot_data_path: str | None = None
    python: str = "python3"
    rw: bool = False
    # entrypoint override (`enroot start --rc SCRIPT`): an image whose entrypoint is a server (vllmsrv3's is
    # `vllm serve`) swallows the command as server arguments; the measured launcher passes a script that
    # `exec "$@"` (~/fsar-data/rc_exec.sh) so the command runs as given
    rc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-able mapping ``runtime_from_dict`` reads back one for one."""

        return {
            "kind": self.kind,
            "container": self.container,
            "mounts": [[src, dst] for src, dst in self.mounts],
            "env": [[key, value] for key, value in self.env],
            "enroot_data_path": self.enroot_data_path,
            "python": self.python,
            "rw": self.rw,
            "rc": self.rc,
        }


def _pairs(raw: Any, where: str) -> tuple[tuple[str, str], ...]:
    """Normalise ``{src: dst}``/``[[src, dst], ...]`` into a pair tuple."""

    if not raw:
        return ()
    if isinstance(raw, Mapping):
        return tuple((str(key), str(value)) for key, value in raw.items())
    out: list[tuple[str, str]] = []
    for index, item in enumerate(raw):
        if isinstance(item, Mapping):
            if len(item) != 1:
                raise ValueError(
                    f"{where}[{index}]: a mapping entry must hold exactly one src->dst pair "
                    f"(got {len(item)}) -- a mount/env pair is two strings, never a batch"
                )
            key, value = next(iter(item.items()))
            out.append((str(key), str(value)))
            continue
        try:
            seq = tuple(item)
        except TypeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"{where}[{index}]: expected a [src, dst] pair, got {_desc(item)}") from exc
        if len(seq) != 2:
            raise ValueError(f"{where}[{index}]: expected a [src, dst] pair (2 strings), got {len(seq)}")
        out.append((str(seq[0]), str(seq[1])))
    return tuple(out)


def _desc(value: Any) -> str:
    """``type(value)`` spelled out -- used in refusals, never a score."""

    return f"{type(value).__name__} {value!r}"


def runtime_from_dict(d: Mapping[str, Any] | None) -> Runtime:
    """Build a :class:`Runtime` from a declared mapping (``None``/``{}`` -> host)."""

    if not d:
        return Runtime()
    container = d.get("container")
    enroot_data_path = d.get("enroot_data_path")
    return Runtime(
        kind=str(d.get("kind") or "host"),
        container=None if container is None else str(container),
        mounts=_pairs(d.get("mounts"), "runtime.mounts"),
        env=_pairs(d.get("env"), "runtime.env"),
        enroot_data_path=None if enroot_data_path is None else str(enroot_data_path),
        python=str(d.get("python") or "python3"),
        rw=bool(d.get("rw", False)),
        rc=None if d.get("rc") in (None, "") else str(d.get("rc")),
    )


def runtime_argv(
    rt: Runtime,
    cmd: Sequence[str],
    extra_env: Mapping[str, str] | None = None,
) -> list[str]:
    """``cmd`` as it really runs: verbatim on host, inside the ``enroot start``
    envelope for ``kind="enroot"``.

    Order is the launcher's order and is never re-sorted: ``enroot start``,
    ``--rw``, ``--mount`` pairs, ``--env`` pairs (``rt.env`` then
    ``extra_env``), the container name, then ``cmd``.

    Refuses (``ValueError``) anything that would make the envelope lie: an
    empty container for ``kind="enroot"``, a container name or an env value
    containing whitespace/newline (``enroot`` would carry it verbatim).
    """

    if rt.kind != "enroot":
        return list(cmd)
    if not rt.container:
        raise ValueError(
            f"runtime_argv: kind is 'enroot' but container is {_desc(rt.container)}: the "
            f"agentic stack runs inside a container (e.g. 'vllmsrv3') -- declaring none would "
            f"launch the engine on the host, where no engine was measured"
        )
    if any(char.isspace() for char in rt.container):
        raise ValueError(
            f"runtime_argv: container name {_desc(rt.container)} contains whitespace/newline: "
            f"the argv would carry it as one element and 'enroot start' would read a different "
            f"container than the one measured"
        )
    env_pairs: tuple[tuple[str, str], ...] = tuple(rt.env) + tuple(
        (str(key), str(value)) for key, value in (extra_env or {}).items()
    )
    for key, value in env_pairs:
        if any(char.isspace() for char in value):
            raise ValueError(
                f"runtime_argv: env value for {_desc(key)} is {_desc(value)} and contains "
                f"whitespace/newline: '--env' carries it verbatim into the container and would "
                f"not set the variable this wrapper reports as set"
            )
    argv: list[str] = ["enroot", "start"]
    if rt.rw:
        argv.append("--rw")
    for src, dst in rt.mounts:
        argv += ["--mount", f"{src}:{dst}"]
    for key, value in env_pairs:
        argv += ["--env", f"{key}={value}"]
    if rt.rc:
        if any(char.isspace() for char in rt.rc):
            raise ValueError(f"runtime_argv: rc script path {_desc(rt.rc)} contains whitespace/newline")
        argv += ["--rc", rt.rc]
    argv.append(rt.container)
    argv += list(cmd)
    return argv


def runtime_env(rt: Runtime) -> dict[str, str]:
    """Host-side process env this runtime declares (nothing else is inherited)."""

    if rt.kind == "enroot" and rt.enroot_data_path:
        return {"ENROOT_DATA_PATH": rt.enroot_data_path}
    return {}


def _tail(text: str, *, max_lines: int = 5) -> str:
    """Last non-empty lines of ``text`` -- the refusal body, never a verdict."""

    kept = [line for line in (text or "").strip().splitlines() if line.strip()]
    return "\n".join(kept[-max_lines:])


def _parse_probe_json(stdout: str) -> dict[str, Any] | None:
    """The probe's one JSON line (later lines win), or None when unparsable."""

    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def probe_agentic(
    rt: Runtime,
    *,
    fs_src: str | None = None,
    run: Callable[..., Any] = subprocess.run,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Run ``AGENTIC_PROBE_SNIPPET`` inside ``rt`` and read its one JSON line.

    Success returns the parsed payload plus ``{"runtime": rt.to_dict(),
    "measured": True}`` (missing probe keys are padded, never dropped). A
    timeout, a non-zero exit, unparsable stdout or a probe that cannot start
    is REPORTED as ``{"runtime": ..., "measured": False, "errors": [reason]}``
    -- this function never raises and never converts a missing measurement
    into a positive one.
    """

    argv = runtime_argv(
        rt,
        [rt.python, "-c", AGENTIC_PROBE_SNIPPET],
        extra_env={"PYTHONPATH": fs_src} if fs_src else None,
    )
    env = {**os.environ, **runtime_env(rt)}
    unmeasured = {"runtime": rt.to_dict(), "measured": False}
    try:
        proc = run(argv, env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {**unmeasured, "errors": [f"agentic probe timed out after {timeout_s} s"]}
    except Exception as exc:  # noqa: BLE001 - a probe that cannot start is UNMEASURED, never a crash
        return {**unmeasured, "errors": [f"agentic probe could not start: {type(exc).__name__}: {exc}"]}
    stdout = getattr(proc, "stdout", "") or ""
    stderr = getattr(proc, "stderr", "") or ""
    rc = getattr(proc, "returncode", 0)
    if rc != 0:
        return {**unmeasured, "errors": [f"agentic probe exited {rc}: {_tail(stderr or stdout)}"]}
    payload = _parse_probe_json(stdout)
    if payload is None:
        return {**unmeasured, "errors": [f"agentic probe printed no JSON line: {_tail(stdout or stderr)}"]}
    report: dict[str, Any] = dict(payload)
    for key in _PROBE_KEYS:
        if key in report:
            continue
        report[key] = "" if key == "fs_version" else ({} if key == "engines_importable" else [])
    report["runtime"] = rt.to_dict()
    report["measured"] = True
    return report
