"""mini-SWE-agent environment backed by a FoundationScale EnrootSandbox.

The agent process (and its LLM calls) stays on the host; only the shell
commands the agent emits are executed inside an enroot container.
"""

from __future__ import annotations

import contextlib
import dataclasses
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from foundationscale.agentic_rl.sandbox.enroot import (
    EnrootSandbox,
    Runner,
    SandboxError,
    SandboxRefusal,
    SandboxSpec,
)

_NETWORKS = ("no-network", "public")


@dataclass(frozen=True)
class EnrootMiniSweConfig:
    """Configuration for :class:`EnrootMiniSweEnvironment`."""

    image: str
    data_root: str
    image_store: str
    scratch_root: str
    name: str | None = None
    network: str = "no-network"
    cwd: str = "/"
    env: Mapping[str, str] = field(default_factory=dict)
    timeout: int = 30
    memory_mb: int | None = None
    cpu_set: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", f"mini-{secrets.token_hex(4)}")
        if self.network not in _NETWORKS:
            raise SandboxRefusal(f"network: expected one of {_NETWORKS}, got {self.network!r}")


class EnrootMiniSweEnvironment:
    """Execute mini-SWE-agent shell commands inside an EnrootSandbox."""

    def __init__(
        self,
        *,
        config_class: type = EnrootMiniSweConfig,
        runner: Runner | None = None,
        **kwargs: Any,
    ) -> None:
        known = {f.name for f in dataclasses.fields(EnrootMiniSweConfig)}
        unknown = sorted(set(kwargs) - known)
        if unknown:
            raise SandboxRefusal(
                f"config: expected only keys {sorted(known)}, got unknown keys {unknown}"
            )
        self.config = config_class(**kwargs)
        assert isinstance(self.config, EnrootMiniSweConfig)
        spec = SandboxSpec(
            name=self.config.name or f"mini-{secrets.token_hex(4)}",
            image=self.config.image,
            data_root=self.config.data_root,
            image_store=self.config.image_store,
            scratch_root=self.config.scratch_root,
            network=self.config.network,
            cpu_set=self.config.cpu_set,
            memory_mb=self.config.memory_mb,
            env=self.config.env,
            workdir=self.config.cwd,
        )
        self.sandbox = EnrootSandbox(spec, runner=runner)
        self.sandbox.start()
        self._uname = self._probe_uname()

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the sandbox and return the result as a dict."""
        command = action.get("command", "")
        try:
            result = self.sandbox.exec(
                command,
                cwd=cwd or self.config.cwd,
                env=self.config.env,
                timeout_s=timeout or self.config.timeout,
            )
        except SandboxError as e:
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
            self._check_finished(output)
            return output
        output_text = result.stdout + (result.stderr if result.stderr else "")
        if result.timed_out:
            output = {
                "output": output_text,
                "returncode": -1,
                "exception_info": (
                    f"An error occurred while executing the command: timed out after "
                    f"{timeout or self.config.timeout} seconds"
                ),
            }
        else:
            output = {
                "output": output_text,
                "returncode": result.return_code,
                "exception_info": "",
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict) -> None:
        """Raises Submitted if the output indicates task completion."""

        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            # Imported only on completion: mini-SWE-agent is the caller's runtime, not FS's.
            from minisweagent.exceptions import Submitted

            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    _UNAME_KEYS = ("system", "node", "release", "version", "machine")

    def _probe_uname(self) -> dict[str, str]:
        """The sandbox's own platform.uname()-style fields (mini-SWE-agent's templates use them).

        Read once from inside the container, never from the host, so templates describe the
        environment the agent's commands actually run in. Falls back to fixed values.
        """
        fallback = {
            "system": "Linux",
            "node": self.sandbox.spec.name,
            "release": "",
            "version": "",
            "machine": "aarch64",
        }
        try:
            result = self.sandbox.exec(
                "uname -s; uname -n; uname -r; uname -v; uname -m", timeout_s=30
            )
        except SandboxError:
            return fallback
        lines = result.stdout.splitlines()
        if result.return_code != 0 or len(lines) != len(self._UNAME_KEYS):
            return fallback
        return dict(zip(self._UNAME_KEYS, (line.strip() for line in lines), strict=True))

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        merged: dict[str, Any] = dataclasses.asdict(self.config)
        merged.update(self._uname)
        merged.update(kwargs)
        return merged

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": dataclasses.asdict(self.config),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def cleanup(self) -> None:
        self.sandbox.stop()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.cleanup()
