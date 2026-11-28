"""Per-episode Enroot containers for Harbor-format tasks (the ``harbor_task`` backend).

A Harbor task (Terminal-Bench / Skill2Env layout: ``environment/Dockerfile``, ``tests/test.sh``,
``instruction.md``, ``task.toml``) is graded by ``tests/test.sh`` writing
``/logs/verifier/reward.json`` (a mapping of named rewards) or ``/logs/verifier/reward.txt``
(one number) -- not by pytest output. This backend reuses the ``enroot_task`` container, vault
and lifecycle unchanged and replaces only the verifier read-out: ``verify()`` deletes any reward
file the agent could have planted, restores the hidden tests, runs ``test_cmd`` and reads the
reward file from the container's root filesystem.

Every numeric reward leaf becomes one synthetic test ``harbor::<dotted.key>`` ("passed" only at
exactly 1, as Harbor's oracle gate requires), plus ``harbor::all`` -- passed only when every leaf
is 1. A task therefore carries ``fail_to_pass='["harbor::all"]'`` and ``rewards/swe.py`` scores it
unchanged: resolved iff the Harbor verifier would award full reward. A missing, unreadable or
leaf-less reward file is ``swe_status="error"`` (the reward abstains), never a scored zero.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import shutil
from collections.abc import Callable, Mapping
from typing import Any

from foundationscale.agentic_rl.envs.base import EnvInfraError, EnvSpec
from foundationscale.agentic_rl.envs.sandbox_task import (
    EnrootTaskBackend,
    TaskSandboxEnvironment,
    _described,
)
from foundationscale.agentic_rl.integrity import IntegrityRefusal
from foundationscale.agentic_rl.sandbox.enroot import SandboxError

__all__ = (
    "ALL_TEST",
    "HarborTaskBackend",
    "HarborTaskEnvironment",
    "reward_tests",
    "short_instance_id",
)

REWARD_DIR = "/logs/verifier"
REWARD_FILES = ("reward.json", "reward.txt")
ALL_TEST = "harbor::all"
_MAX_REWARD_BYTES = 1 << 20
# enroot names are <= 63 chars and enroot_task names a sandbox "fsar-<instance id>-<6 hex>": 51
# chars of id at most
_MAX_INSTANCE_ID = 48


def short_instance_id(instance_id: str) -> str:
    """``instance_id`` unchanged when short enough, else a 39-char prefix plus 8 hex of its
    sha256 (unique, stable)."""
    if len(instance_id) <= _MAX_INSTANCE_ID:
        return instance_id
    digest = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:8]
    return f"{instance_id[:39]}-{digest}"


def _leaves(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    """Every numeric (non-bool) leaf of a JSON value as ``(key, value)``; non-finite -> NaN."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        # A non-finite or overflowing leaf is kept as NaN, never dropped: dropping it would let the
        # remaining leaves pass harbor::all with a reward component unmeasured. reward_tests then
        # abstains on the whole file.
        try:
            number = float(value)
        except OverflowError:
            return [(prefix or "reward", math.nan)]
        return [(prefix or "reward", number if math.isfinite(number) else math.nan)]
    if isinstance(value, Mapping):
        out: list[tuple[str, float]] = []
        for key in sorted(value, key=str):
            out.extend(_leaves(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list):
        out = []
        for index, item in enumerate(value):
            out.extend(_leaves(item, f"{prefix}.{index}" if prefix else str(index)))
        return out
    return []


def reward_tests(name: str, text: str) -> dict[str, str] | None:
    """Map one reward file's text to ``test id -> "passed"|"failed"``; None when it has no
    leaves."""
    try:
        value: Any = json.loads(text) if name.endswith(".json") else float(text.strip())
    except ValueError:
        return None
    leaves = _leaves(value)
    if not leaves or any(not math.isfinite(score) for _, score in leaves):
        return None  # no leaves, or a component that is not a finite number: unmeasured, abstain
    tests = {f"harbor::{key}": ("passed" if score == 1.0 else "failed") for key, score in leaves}
    tests[ALL_TEST] = "passed" if all(score == 1.0 for _, score in leaves) else "failed"
    return tests


class HarborTaskEnvironment(TaskSandboxEnvironment):
    """``TaskSandboxEnvironment`` whose verifier reads Harbor's reward file instead of pytest.

    With ``fresh_sandbox`` (the default for the ``harbor_task`` backend) the verifier runs in a
    NEW container from the
    pristine image, into which only the agent's workdir is copied: the agent is root in its own
    container and could
    otherwise shadow an interpreter the verifier calls (measured 2026-11-24: replacing ``node``
    with a script that writes
    reward.json scored 1.0 on a held-out task without doing it). Task metadata
    ``verify_mode="inplace"`` opts out.
    """

    def __init__(self, *, fresh_sandbox: Callable[[], Any] | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._fresh_sandbox = fresh_sandbox

    def _verify_sync(self) -> dict[str, str]:
        if self._fresh_sandbox is None:
            return self._verify_inplace()
        return self._verify_fresh()

    def _verify_inplace(self) -> dict[str, str]:
        try:
            for name in REWARD_FILES:  # a reward the agent planted must never be read back
                path = self._sandbox.host_path(f"{REWARD_DIR}/{name}")
                if path.is_file() or path.is_symlink():
                    path.unlink()
            self._vault.unseal()
            result = self._sandbox.exec(
                self._config.test_cmd,
                cwd=self._config.workdir,
                env=dict(self._spec.env) or None,
                timeout_s=self._config.test_timeout_s,
            )
        except (SandboxError, IntegrityRefusal, OSError) as exc:
            raise EnvInfraError(
                "verify_failed",
                f"HarborTaskEnvironment.verify: clearing old rewards, restoring the hidden tests "
                f"and running "
                f"{_described(self._config.test_cmd)} raised {type(exc).__name__} {exc}: "
                f"the verifier phase could "
                f"not run at all -- an infra failure, never a scored zero",
            ) from exc
        return _read_rewards(self._sandbox, result, verify_mode="inplace")

    def _verify_fresh(self) -> dict[str, str]:
        if self._config.workdir.rstrip("/") == "":
            # copying "/" would replace the whole pristine rootfs with the agent's (tampered
            # binaries included)
            return {
                "swe_status": "error",
                "swe_error": "fresh verification needs a non-root workdir",
                "harbor_verify_mode": "fresh",
            }
        workdir = self._config.workdir.rstrip("/")
        if any(p == workdir or p.startswith(workdir + "/") for p in self._config.hidden_paths):
            # the agent's copy of the workdir would REPLACE the pristine hidden tests it contains
            return {
                "swe_status": "error",
                "swe_error": "fresh verification refuses a workdir that contains hidden test paths",
                "harbor_verify_mode": "fresh",
            }
        factory = self._fresh_sandbox
        assert factory is not None  # _verify_sync routes here only when a factory is set
        fresh: Any = None
        try:
            fresh = factory()
            fresh.start()
            src = self._sandbox.host_path(self._config.workdir)
            dst = fresh.host_path(self._config.workdir)
            if dst.exists() or dst.is_symlink():
                shutil.rmtree(dst) if dst.is_dir() and not dst.is_symlink() else dst.unlink()
            if src.is_dir():
                shutil.copytree(src, dst, symlinks=True, ignore=_special_files)
            for (
                name
            ) in REWARD_FILES:  # a reward planted inside the copied workdir must never be read back
                planted = fresh.host_path(f"{REWARD_DIR}/{name}")
                if planted.is_file() or planted.is_symlink():
                    planted.unlink()
            result = fresh.exec(
                self._config.test_cmd,
                cwd=self._config.workdir,
                env=dict(self._spec.env) or None,
                timeout_s=self._config.test_timeout_s,
            )
            return _read_rewards(fresh, result, verify_mode="fresh")
        except (SandboxError, IntegrityRefusal, OSError, shutil.Error) as exc:
            raise EnvInfraError(
                "verify_failed",
                f"HarborTaskEnvironment.verify: starting the pristine verifier container, copying "
                f"{self._config.workdir} into it and running "
                f"{_described(self._config.test_cmd)} raised "
                f"{type(exc).__name__} {exc}: the verifier phase could not run at all -- "
                f"an infra failure, never a "
                f"scored zero",
            ) from exc
        finally:
            if fresh is not None:
                with contextlib.suppress(SandboxError, OSError):
                    fresh.stop(delete=True)


def _special_files(directory: str, names: list[str]) -> set[str]:
    """copytree ignore hook: skip FIFOs, sockets and device nodes (a FIFO would block the copy)."""
    import os
    import stat
    from pathlib import Path

    skipped = set()
    for name in names:
        try:
            mode = os.lstat(Path(directory) / name).st_mode
        except OSError:
            continue
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
            skipped.add(name)
    return skipped


def _read_rewards(sandbox: Any, result: Any, *, verify_mode: str) -> dict[str, str]:
    """The ``swe_*`` mapping from the reward file ``test_cmd`` left in ``sandbox`` (see the
    module docstring)."""
    if result.timed_out:
        return {
            "swe_status": "error",
            "swe_error": "test_cmd timed out",
            "harbor_verify_mode": verify_mode,
        }
    for name in REWARD_FILES:
        try:
            path = sandbox.host_path(
                f"{REWARD_DIR}/{name}"
            )  # refuses paths resolving outside the rootfs
            if not path.is_file() or path.stat().st_size > _MAX_REWARD_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except (SandboxError, OSError):
            continue
        tests = reward_tests(name, text)
        if tests is not None:
            return {
                "swe_status": "ok",
                "swe_tests": json.dumps(tests, sort_keys=True),
                "swe_exit_code": str(result.return_code),
                "harbor_reward_file": name,
                "harbor_verify_mode": verify_mode,
            }
        return {
            "swe_status": "error",
            "swe_error": f"{name} has no numeric reward",
            "harbor_verify_mode": verify_mode,
        }
    return {
        "swe_status": "error",
        "swe_error": f"no reward file under {REWARD_DIR} (exit {result.return_code})",
        "harbor_verify_mode": verify_mode,
    }


class HarborTaskBackend(EnrootTaskBackend):
    """``enroot_task`` with a Harbor reward read-out (``name`` is ``"harbor_task"``)."""

    name: str = "harbor_task"

    def create_for_task(
        self, spec: EnvSpec, task: Any, *, instance_id: str
    ) -> HarborTaskEnvironment:
        captured: dict[str, Any] = {}
        factory = self._sandbox_factory

        def capture(sandbox_spec: Any) -> Any:
            captured["spec"] = sandbox_spec
            return factory(sandbox_spec)

        self._sandbox_factory = capture
        try:
            # Harbor task names (Skill2Env uids) run long; the enroot name limit would refuse every
            # episode
            env = super().create_for_task(spec, task, instance_id=short_instance_id(instance_id))
        finally:
            self._sandbox_factory = factory
        fresh = None
        if str((getattr(task, "metadata", None) or {}).get("verify_mode", "fresh")) != "inplace":
            base = captured["spec"]
            fresh = lambda: factory(dataclasses.replace(base, name=base.name + "-v"))  # noqa: E731
        return HarborTaskEnvironment(
            spec=env._spec,
            config=env._config,
            sandbox=env._sandbox,
            vault=env._vault,
            fresh_sandbox=fresh,
        )
