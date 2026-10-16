# Line 79 (parse_dockerfile's `continue` for an empty logical step) is unreachable: `parts` can only
# be empty when `joined` is empty/whitespace, which is filtered out earlier by the `if not logical`
# check; skip it.  383-384/386 share one test (all three lines are one `try/except/raise` block).

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from foundationscale.agentic_rl.sandbox.build import (
    build_image,
)
from foundationscale.agentic_rl.sandbox.enroot import (
    RunResult,
)


class FakeRunner:
    """Record argv and emulate enroot side effects for import/create/start/remove."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, str] | None, float | None]] = []
        self.fail_commands: set[str] = set()
        self.fail_stderr: str = "boom"
        self.fail_returncode: int = 1
        self.exec_results: dict[str, tuple[int, str, str]] = {}
        self.data_root: str | None = None
        self.exported_root: Path | None = None
        self.image_environment: str | None = None
        self.export_returncode: int = 0
        self.export_stderr: str = "export exploded"

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> RunResult:
        args = tuple(argv)
        self.calls.append((args, dict(env) if env is not None else None, timeout_s))
        if env and "ENROOT_DATA_PATH" in env:
            self.data_root = env["ENROOT_DATA_PATH"]
        if args and args[0] == "enroot":
            return self._enroot(args)
        return RunResult(returncode=0, stdout="", stderr="")

    def _enroot(self, args: tuple[str, ...]) -> RunResult:
        sub = args[1]
        if sub == "import":
            out = Path(args[args.index("-o") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"sqsh")
            return RunResult(returncode=0, stdout="", stderr="")
        if sub == "create":
            name = args[args.index("-n") + 1]
            root = Path(self.data_root or ".") / name
            (root / "etc").mkdir(parents=True, exist_ok=True)
            if self.image_environment is not None:  # enroot writes the image config env here
                (root / "etc" / "environment").write_text(self.image_environment)
            return RunResult(returncode=0, stdout="", stderr="")
        if sub == "start":
            name = args[-4]
            command = args[-1]
            # the sandbox runs `cd <dir> && <cmd>`: match on the command's tail
            if any(command.endswith(fail) for fail in self.fail_commands):
                return RunResult(
                    returncode=self.fail_returncode, stdout="", stderr=self.fail_stderr
                )
            key = next((k for k in self.exec_results if command.endswith(k)), None)
            code, out, err = self.exec_results.get(key, (0, "", "")) if key else (0, "", "")
            return RunResult(returncode=code, stdout=out, stderr=err)
        if sub == "remove":
            name = args[-1]
            root = Path(self.data_root or ".") / name
            if root.exists():
                import shutil

                shutil.rmtree(root)
            return RunResult(returncode=0, stdout="", stderr="")
        if sub == "export":
            out = Path(args[args.index("-o") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"exported")
            # Snapshot what was exported: the build removes its container afterwards.
            import shutil

            self.exported_root = Path(str(out) + ".rootfs")
            shutil.copytree(Path(self.data_root or ".") / args[-1], self.exported_root)
            return RunResult(
                returncode=self.export_returncode,
                stdout="",
                stderr=self.export_stderr,
            )
        return RunResult(returncode=0, stdout="", stderr="")

    def argvs(self) -> list[tuple[str, ...]]:
        return [call[0] for call in self.calls]


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _build(tmp_path: Path, dockerfile: str, runner: FakeRunner, **kwargs):
    df = _write(tmp_path, "Dockerfile", dockerfile)
    context = tmp_path / "ctx"
    context.mkdir(exist_ok=True)
    out = tmp_path / "out.sqsh"
    return build_image(
        df,
        context,
        out,
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
        **kwargs,
    )


@pytest.mark.parametrize("argument", ['["/bin/app", ', '["/bin/app", 3]', '{"cmd": "x"}'])
def test_entrypoint_exec_form_that_is_not_a_json_string_array_is_refused(argument: str) -> None:
    from foundationscale.agentic_rl.sandbox.build import DockerfileStep, _parse_entrypoint
    from foundationscale.agentic_rl.sandbox.enroot import SandboxRefusal

    with pytest.raises(SandboxRefusal, match="ENTRYPOINT: expected a JSON string array"):
        _parse_entrypoint(
            DockerfileStep(lineno=4, instruction="ENTRYPOINT", argument=argument), argument
        )
