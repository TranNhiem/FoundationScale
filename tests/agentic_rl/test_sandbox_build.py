"""Tests for foundationscale.agentic_rl.sandbox.build."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from foundationscale.agentic_rl.sandbox.build import (
    BuildReport,
    DockerfileStep,
    build_image,
    parse_dockerfile,
)
from foundationscale.agentic_rl.sandbox.enroot import (
    RunResult,
    SandboxBuildFailed,
    SandboxRefusal,
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
            return RunResult(returncode=0, stdout="", stderr="")
        return RunResult(returncode=0, stdout="", stderr="")

    def argvs(self) -> list[tuple[str, ...]]:
        return [call[0] for call in self.calls]


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _build(tmp_path: Path, dockerfile: str, runner: FakeRunner, **kwargs) -> BuildReport:
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


def test_parse_dockerfile_comments_blanks_continuations_case(tmp_path: Path) -> None:
    """Comments and blanks are dropped, continuations join, instructions are uppercased."""
    text = "# comment\n\nFROM alpine\nrun echo one \\\n    two\n  # indented comment\nENV A=1\n"
    steps = parse_dockerfile(text)
    assert steps[0] == DockerfileStep(lineno=3, instruction="FROM", argument="alpine")
    assert steps[1] == DockerfileStep(lineno=4, instruction="RUN", argument="echo one two")
    assert steps[2] == DockerfileStep(lineno=7, instruction="ENV", argument="A=1")


def test_parse_dockerfile_line_numbers_after_continuation() -> None:
    """Line numbers track the first physical line of each logical step."""
    steps = parse_dockerfile("FROM alpine\nRUN a \\\n b \\\n c\nWORKDIR /w\n")
    assert [s.lineno for s in steps] == [1, 2, 5]


def test_arg_defaults_and_build_args_substitution(tmp_path: Path) -> None:
    """ARG defaults apply and build_args override, substituted as $X and ${X}."""
    runner = FakeRunner()
    report = _build(
        tmp_path,
        "ARG VER=1.0\nARG TAG\nFROM alpine:$VER\nENV LABEL=${TAG}-x\n",
        runner,
        build_args={"TAG": "prod"},
    )
    assert report.image == "alpine:1.0"
    assert report.env == {"LABEL": "prod-x"}


def test_env_both_forms(tmp_path: Path) -> None:
    """ENV key=value pairs and ENV key value forms both set env."""
    runner = FakeRunner()
    report = _build(tmp_path, "FROM alpine\nENV A=1 B=2\nENV C 3\n", runner)
    assert report.env == {"A": "1", "B": "2", "C": "3"}


def test_workdir_recorded(tmp_path: Path) -> None:
    """WORKDIR sets the build workdir reported in BuildReport."""
    runner = FakeRunner()
    report = _build(tmp_path, "FROM alpine\nWORKDIR /app\n", runner)
    assert report.workdir == "/app"


def test_run_shell_and_json_forms_via_enroot_start(tmp_path: Path) -> None:
    """RUN shell and JSON forms execute through enroot start with env and cwd."""
    runner = FakeRunner()
    _build(
        tmp_path,
        'FROM alpine\nENV K=V\nWORKDIR /w\nRUN echo hi\nRUN ["echo", "json"]\n',
        runner,
    )
    starts = [a for a in runner.argvs() if a[:2] == ("enroot", "start")]
    assert len(starts) == 2
    first = starts[0]
    assert "--env" in first and "K=V" in first
    assert first[-3:] == ("sh", "-c", "cd /w && echo hi")
    assert starts[1][-1] == "cd /w && echo json"


def test_run_json_form_joined_with_shlex(tmp_path: Path) -> None:
    """JSON exec form joins arguments with shell quoting."""
    runner = FakeRunner()
    _build(tmp_path, 'FROM alpine\nRUN ["echo", "a b"]\n', runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && echo 'a b'"


def test_copy_file_and_dir_land_in_rootfs(tmp_path: Path) -> None:
    """COPY of a file and a directory from the context lands in the rootfs."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "f.txt").write_text("data", encoding="utf-8")
    (context / "d").mkdir()
    (context / "d" / "inner.txt").write_text("inner", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nCOPY f.txt /dest/f.txt\nCOPY d /dest/d\n")
    out = tmp_path / "out.sqsh"
    build_image(
        df,
        context,
        out,
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "f.txt").read_text(encoding="utf-8") == "data"
    assert (rootfs / "dest" / "d" / "inner.txt").read_text(encoding="utf-8") == "inner"


def test_copy_escaping_context_refused(tmp_path: Path) -> None:
    """COPY source escaping the build context is refused before any container is created."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (tmp_path / "outside.txt").write_text("x", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nCOPY ../outside.txt /x\n")
    with pytest.raises(SandboxRefusal):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )
    assert not any(a[:2] == ("enroot", "create") for a in runner.argvs())


@pytest.mark.parametrize(
    "dockerfile",
    [
        "FROM alpine\nFROM debian\n",
        "FROM scratch\n",
        "FROM alpine\nCOPY --from=0 /a /b\n",
        "FROM alpine\nADD https://example.com/x /x\n",
        "FROM alpine\nADD x.tar.gz /x\n",
        "FROM alpine\nUSER nobody\n",
        'FROM alpine\nSHELL ["/bin/bash", "-c"]\n',
    ],
)
def test_unsupported_instructions_refused_before_create(tmp_path: Path, dockerfile: str) -> None:
    """Multi-stage, scratch, COPY --from, ADD URL/archive, non-root USER and SHELL are refused."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    df = _write(tmp_path, "Dockerfile", dockerfile)
    with pytest.raises(SandboxRefusal):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )
    assert not any(a[:2] == ("enroot", "create") for a in runner.argvs())


def test_user_root_accepted(tmp_path: Path) -> None:
    """USER root and USER 0 are accepted."""
    runner = FakeRunner()
    report = _build(tmp_path, "FROM alpine\nUSER root\nUSER 0\n", runner)
    assert report.steps_applied == 3


def test_failing_run_raises_and_removes_container(tmp_path: Path) -> None:
    """A failing RUN raises SandboxBuildFailed naming the line and the container is removed."""
    runner = FakeRunner()
    runner.fail_commands.add("echo boom")
    runner.fail_stderr = "x" * 3000
    with pytest.raises(SandboxBuildFailed) as excinfo:
        _build(tmp_path, "FROM alpine\nRUN echo ok\nRUN echo boom\n", runner)
    message = str(excinfo.value)
    assert "line 3" in message
    assert "boom" in message
    assert len(message) < 2500
    assert any(a[:2] == ("enroot", "remove") for a in runner.argvs())


def test_success_writes_metadata_and_exports(tmp_path: Path) -> None:
    """Success writes /etc/fs-sandbox.json with env and workdir and exports the sqsh."""
    runner = FakeRunner()
    out = tmp_path / "out.sqsh"
    context = tmp_path / "ctx"
    context.mkdir()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nENV A=1\nWORKDIR /w\n")
    build_image(
        df,
        context,
        out,
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    metadata = json.loads((runner.exported_root / "etc" / "fs-sandbox.json").read_text())
    assert metadata == {"env": {"A": "1"}, "workdir": "/w"}
    exports = [a for a in runner.argvs() if a[:2] == ("enroot", "export")]
    assert exports and exports[0][exports[0].index("-o") + 1] == str(out)
    assert out.exists()


def test_build_report_fields(tmp_path: Path) -> None:
    """BuildReport carries image, steps_applied, seconds, env and workdir."""
    runner = FakeRunner()
    report = _build(tmp_path, "FROM alpine\nENV A=1\nWORKDIR /w\nRUN true\n", runner)
    assert isinstance(report, BuildReport)
    assert report.image == "alpine"
    assert report.steps_applied == 4
    assert report.seconds is not None and report.seconds >= 0.0
    assert report.env == {"A": "1"}
    assert report.workdir == "/w"


def test_ignored_instructions_have_no_effect(tmp_path: Path) -> None:
    """EXPOSE, CMD, ENTRYPOINT, LABEL, HEALTHCHECK, VOLUME and STOPSIGNAL are ignored."""
    runner = FakeRunner()
    report = _build(
        tmp_path,
        'FROM alpine\nEXPOSE 80\nCMD ["sh"]\nENTRYPOINT ["sh"]\nLABEL a=b\n'
        "HEALTHCHECK NONE\nVOLUME /v\nSTOPSIGNAL SIGTERM\n",
        runner,
    )
    assert report.env == {}
    assert report.workdir == "/"
    assert report.steps_applied == 8


def test_add_local_file_accepted(tmp_path: Path) -> None:
    """ADD of a local context path copies like COPY."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nADD a.txt /dest/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    assert (runner.exported_root / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_run_env_merges_build_env(tmp_path: Path) -> None:
    """RUN receives accumulated ENV values as --env flags on enroot start."""
    runner = FakeRunner()
    _build(tmp_path, "FROM alpine\nENV A=1\nENV B=2\nRUN true\n", runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    env_pairs = [start[i + 1] for i, x in enumerate(start) if x == "--env"]
    assert "A=1" in env_pairs and "B=2" in env_pairs


def test_build_uses_public_network(tmp_path: Path) -> None:
    """Build sandboxes run with network=public (no unshare wrapper)."""
    runner = FakeRunner()
    _build(tmp_path, "FROM alpine\nRUN true\n", runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert "unshare" not in start


def test_container_removed_on_success(tmp_path: Path) -> None:
    """The build container is always removed after a successful build."""
    runner = FakeRunner()
    _build(tmp_path, "FROM alpine\n", runner)
    assert any(a[:2] == ("enroot", "remove") for a in runner.argvs())


def test_parse_dockerfile_rejects_non_str_text() -> None:
    """parse_dockerfile refuses non-str text."""
    with pytest.raises(SandboxRefusal, match="text: expected str"):
        parse_dockerfile(b"FROM alpine\n")  # type: ignore[arg-type]


def test_parse_dockerfile_trailing_continuation_emits_step() -> None:
    """A dangling continuation still emits the accumulated logical step."""
    steps = parse_dockerfile("FROM alpine\nRUN echo \\\n")
    assert steps == (
        DockerfileStep(lineno=1, instruction="FROM", argument="alpine"),
        DockerfileStep(lineno=2, instruction="RUN", argument="echo"),
    )


def test_build_missing_dockerfile_refused(tmp_path: Path) -> None:
    """A missing Dockerfile path is refused."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="dockerfile: expected existing file"):
        build_image(
            tmp_path / "nope",
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_missing_context_dir_refused(tmp_path: Path) -> None:
    """A missing context directory is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\n")
    with pytest.raises(SandboxRefusal, match="context_dir: expected existing directory"):
        build_image(
            df,
            tmp_path / "no-ctx",
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_out_sqsh_suffix_refused(tmp_path: Path) -> None:
    """An output path without the .sqsh suffix is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match=r"out_sqsh: expected a \*\.sqsh path"):
        build_image(
            df,
            context,
            tmp_path / "out.img",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_relative_data_root_refused(tmp_path: Path) -> None:
    """A relative data_root is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="data_root: expected absolute path"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root="relative/data",
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_relative_image_store_refused(tmp_path: Path) -> None:
    """A relative image_store is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="image_store: expected absolute path"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store="relative/store",
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_relative_scratch_root_refused(tmp_path: Path) -> None:
    """A relative scratch_root is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="scratch_root: expected absolute path"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root="relative/scratch",
            runner=runner,
        )


def test_build_non_str_build_args_refused(tmp_path: Path) -> None:
    """Non-str build_args keys or values are refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="build_args: expected Mapping\\[str, str\\]"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
            build_args={"K": 1},  # type: ignore[dict-item]
        )


def test_build_non_positive_timeout_refused(tmp_path: Path) -> None:
    """A non-positive timeout_s is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="timeout_s: expected > 0"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
            timeout_s=0,
        )


def test_build_empty_dockerfile_refused(tmp_path: Path) -> None:
    """A Dockerfile with no instructions is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "# only a comment\n\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="expected at least one instruction"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_zero_from_refused(tmp_path: Path) -> None:
    """A Dockerfile without a FROM line is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "ENV A=1\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="FROM: expected exactly one, got 0"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_empty_from_argument_refused(tmp_path: Path) -> None:
    """A FROM line with an empty image reference is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "ARG IMG\nFROM ${IMG}\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="FROM: expected an image reference"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_arg_empty_argument_refused(tmp_path: Path) -> None:
    """An ARG line with an empty argument is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nARG\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="ARG: expected a name, got empty argument"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_arg_multi_word_name_refused(tmp_path: Path) -> None:
    """An ARG name containing whitespace is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nARG A B\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="ARG: expected a single name"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_arg_without_default_sets_empty(tmp_path: Path) -> None:
    """An ARG without a default sets an empty value."""
    runner = FakeRunner()
    report = _build(tmp_path, "FROM alpine\nARG X\nENV Y=$X\n", runner)
    assert report.env == {"Y": ""}


def test_build_env_empty_argument_refused(tmp_path: Path) -> None:
    """An ENV line with an empty argument is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nENV\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="ENV: expected assignments, got empty argument"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_env_mixed_kv_form_refused(tmp_path: Path) -> None:
    """An ENV line mixing K=V and bare tokens is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nENV A=1 B\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="ENV: expected K=V form"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_env_empty_key_refused(tmp_path: Path) -> None:
    """An ENV K=V token with an empty key is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nENV =1\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="ENV: expected a non-empty key"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_env_space_form_missing_value_refused(tmp_path: Path) -> None:
    """An ENV 'K V' line without a value is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nENV ONLYKEY\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="ENV: expected 'K V' form"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_workdir_with_space_refused(tmp_path: Path) -> None:
    """A WORKDIR path containing whitespace is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nWORKDIR /a b\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="WORKDIR: expected a single path"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_workdir_empty_argument_refused(tmp_path: Path) -> None:
    """An empty WORKDIR argument is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nWORKDIR\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="WORKDIR: expected a single path"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_run_empty_argument_refused(tmp_path: Path) -> None:
    """A RUN line with an empty argument is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nRUN\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="RUN: expected a command, got empty argument"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_run_invalid_json_refused(tmp_path: Path) -> None:
    """A RUN exec form with invalid JSON is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", 'FROM alpine\nRUN ["echo", "x"\n')
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="RUN: expected valid JSON exec form"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_run_non_string_json_array_refused(tmp_path: Path) -> None:
    """A RUN exec form whose JSON array holds non-strings is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", 'FROM alpine\nRUN ["echo", 1]\n')
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="RUN: expected a non-empty JSON array of strings"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_run_empty_json_array_refused(tmp_path: Path) -> None:
    """A RUN exec form with an empty JSON array is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nRUN []\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="RUN: expected a non-empty JSON array of strings"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_copy_empty_argument_refused(tmp_path: Path) -> None:
    """A COPY line with an empty argument is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nCOPY\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="COPY: expected arguments, got empty"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_copy_unsupported_flag_refused(tmp_path: Path) -> None:
    """A COPY flag other than --from is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nCOPY --chown=1 a.txt /a.txt\n")
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    with pytest.raises(SandboxRefusal, match="COPY: unsupported flag"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_copy_missing_target_refused(tmp_path: Path) -> None:
    """A COPY line with only one path is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nCOPY a.txt\n")
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    with pytest.raises(SandboxRefusal, match="COPY: expected sources and a target"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_add_url_source_refused(tmp_path: Path) -> None:
    """An ADD URL source is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nADD https://example.com/x /x\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="ADD: URL sources are not supported"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_add_archive_source_refused(tmp_path: Path) -> None:
    """An ADD local archive source is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nADD bundle.tar.xz /x\n")
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "bundle.tar.xz").write_bytes(b"x")
    with pytest.raises(SandboxRefusal, match="ADD: archive auto-extraction is not supported"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_copy_glob_with_no_matches_refused(tmp_path: Path) -> None:
    """A COPY glob matching nothing is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nCOPY *.txt /dest/\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="COPY: expected at least one source"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_copy_relative_target_refused(tmp_path: Path) -> None:
    """A COPY target that is not absolute is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nCOPY a.txt dest/a.txt\n")
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    with pytest.raises(SandboxRefusal, match="COPY: expected an absolute target"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_copy_absolute_source_refused(tmp_path: Path) -> None:
    """A COPY source with an absolute path is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nCOPY /etc/hosts /h\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="COPY: expected a relative context path"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_copy_empty_source_refused(tmp_path: Path) -> None:
    """A COPY source that is empty is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", 'FROM alpine\nCOPY "" /h\n')
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="COPY: expected a relative context path"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_copy_glob_escaping_context_refused(tmp_path: Path) -> None:
    """A COPY glob whose match resolves outside the context is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nCOPY ../*.txt /x\n")
    context = tmp_path / "ctx"
    context.mkdir()
    (tmp_path / "outside.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SandboxRefusal, match="COPY: expected a path inside"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_copy_dir_to_trailing_slash_target(tmp_path: Path) -> None:
    """COPY of a directory into a trailing-slash target nests it under the target."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "d").mkdir()
    (context / "d" / "inner.txt").write_text("i", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nCOPY d /dest/\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "d" / "inner.txt").read_text(encoding="utf-8") == "i"


def test_build_unsupported_instruction_refused(tmp_path: Path) -> None:
    """An instruction outside the supported subset is refused before any container is created."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nMAINTAINER me\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="expected a supported instruction"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )
    assert not any(a[:2] == ("enroot", "create") for a in runner.argvs())


def test_build_run_budget_exhausted_raises_build_failed(tmp_path: Path) -> None:
    """A RUN step past the build budget raises SandboxBuildFailed."""
    runner = FakeRunner()
    with pytest.raises(SandboxBuildFailed, match="build budget of"):
        _build(tmp_path, "FROM alpine\nRUN true\n", runner, timeout_s=1e-12)


def test_build_run_failure_stderr_truncated_to_tail(tmp_path: Path) -> None:
    """A failing RUN reports only the tail of a long stderr."""
    runner = FakeRunner()
    runner.fail_commands.add("echo boom")
    runner.fail_stderr = "y" * 5000
    with pytest.raises(SandboxBuildFailed) as excinfo:
        _build(tmp_path, "FROM alpine\nRUN echo boom\n", runner)
    message = str(excinfo.value)
    assert "y" * 2000 in message
    assert "y" * 2001 not in message


def test_build_arg_default_quoted_value_stripped(tmp_path: Path) -> None:
    """ARG default values are stripped of surrounding quotes."""
    runner = FakeRunner()
    report = _build(tmp_path, 'FROM alpine\nARG V="1.2"\nENV X=$V\n', runner)
    assert report.env == {"X": "1.2"}


def test_build_arg_build_args_override_default(tmp_path: Path) -> None:
    """build_args win over an ARG default value."""
    runner = FakeRunner()
    report = _build(tmp_path, "FROM alpine\nARG V=1\nENV X=$V\n", runner, build_args={"V": "2"})
    assert report.env == {"X": "2"}


def test_build_pre_from_arg_substituted_into_from(tmp_path: Path) -> None:
    """ARGs declared before FROM are substituted into the FROM reference."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG BASE=alpine:3\nFROM $BASE\n", runner)
    assert report.image == "alpine:3"


def test_build_from_docker_prefix_kept(tmp_path: Path) -> None:
    """A FROM reference already carrying the docker:// prefix is kept as-is."""
    runner = FakeRunner()
    report = _build(tmp_path, "FROM docker://alpine\n", runner)
    assert report.image == "docker://alpine"


def test_build_env_space_form_value_joined(tmp_path: Path) -> None:
    """ENV 'K V' form joins all remaining tokens into the value."""
    runner = FakeRunner()
    report = _build(tmp_path, "FROM alpine\nENV K a b c\n", runner)
    assert report.env == {"K": "a b c"}


def test_build_env_kv_form_multiple_pairs(tmp_path: Path) -> None:
    """ENV K=V form records every pair in order."""
    runner = FakeRunner()
    report = _build(tmp_path, "FROM alpine\nENV A=1 B=2 C=3\n", runner)
    assert report.env == {"A": "1", "B": "2", "C": "3"}


def test_build_env_quotes_preserved_in_value(tmp_path: Path) -> None:
    """ENV values keep their inner content after shlex tokenization."""
    runner = FakeRunner()
    report = _build(tmp_path, 'FROM alpine\nENV K="a b"\n', runner)
    assert report.env == {"K": "a b"}


def test_build_run_json_form_with_leading_whitespace(tmp_path: Path) -> None:
    """RUN exec form is detected even with leading whitespace."""
    runner = FakeRunner()
    _build(tmp_path, 'FROM alpine\nRUN   ["echo", "x"]\n', runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && echo x"


def test_build_run_shell_form_kept_verbatim(tmp_path: Path) -> None:
    """RUN shell form passes the argument through unchanged."""
    runner = FakeRunner()
    _build(tmp_path, "FROM alpine\nRUN echo a && echo b\n", runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && echo a && echo b"


def test_build_add_directory_accepted(tmp_path: Path) -> None:
    """ADD of a local directory copies it like COPY."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "d").mkdir()
    (context / "d" / "inner.txt").write_text("i", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nADD d /dest/d\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "d" / "inner.txt").read_text(encoding="utf-8") == "i"


def test_build_add_from_flag_refused(tmp_path: Path) -> None:
    """ADD --from is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nADD --from=0 /a /b\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="ADD: --from is not supported"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_add_empty_argument_refused(tmp_path: Path) -> None:
    """An ADD line with an empty argument is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nADD\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="ADD: expected arguments, got empty"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_add_missing_target_refused(tmp_path: Path) -> None:
    """An ADD line with only one path is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nADD a.txt\n")
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    with pytest.raises(SandboxRefusal, match="ADD: expected sources and a target"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_add_relative_target_refused(tmp_path: Path) -> None:
    """An ADD target that is not absolute is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nADD a.txt dest/a.txt\n")
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    with pytest.raises(SandboxRefusal, match="ADD: expected an absolute target"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_add_absolute_source_refused(tmp_path: Path) -> None:
    """An ADD source with an absolute path is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nADD /etc/hosts /h\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="ADD: expected a relative context path"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_add_glob_with_no_matches_refused(tmp_path: Path) -> None:
    """An ADD glob matching nothing is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nADD *.txt /dest/\n")
    context = tmp_path / "ctx"
    context.mkdir()
    with pytest.raises(SandboxRefusal, match="ADD: expected at least one source"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_add_glob_escaping_context_refused(tmp_path: Path) -> None:
    """An ADD glob whose match resolves outside the context is refused."""
    runner = FakeRunner()
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nADD ../*.txt /x\n")
    context = tmp_path / "ctx"
    context.mkdir()
    (tmp_path / "outside.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SandboxRefusal, match="ADD: expected a path inside"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_add_dir_to_trailing_slash_target(tmp_path: Path) -> None:
    """ADD of a directory into a trailing-slash target nests it under the target."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "d").mkdir()
    (context / "d" / "inner.txt").write_text("i", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nADD d /dest/\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "d" / "inner.txt").read_text(encoding="utf-8") == "i"


def test_build_add_symlink_source_escaping_context_refused(tmp_path: Path) -> None:
    """An ADD source that is a symlink escaping the context is refused."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (tmp_path / "outside.txt").write_text("x", encoding="utf-8")
    (context / "link.txt").symlink_to(tmp_path / "outside.txt")
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nADD link.txt /x\n")
    with pytest.raises(SandboxRefusal, match="ADD: expected a path inside"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_copy_symlink_source_escaping_context_refused(tmp_path: Path) -> None:
    """A COPY source that is a symlink escaping the context is refused."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (tmp_path / "outside.txt").write_text("x", encoding="utf-8")
    (context / "link.txt").symlink_to(tmp_path / "outside.txt")
    df = _write(tmp_path, "Dockerfile", "FROM alpine\nCOPY link.txt /x\n")
    with pytest.raises(SandboxRefusal, match="COPY: expected a path inside"):
        build_image(
            df,
            context,
            tmp_path / "out.sqsh",
            data_root=str(tmp_path / "data"),
            image_store=str(tmp_path / "store"),
            scratch_root=str(tmp_path / "scratch"),
            runner=runner,
        )


def test_build_env_substitution_in_copy_source(tmp_path: Path) -> None:
    """ARG values are substituted inside COPY sources and targets."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG NAME=a.txt\nFROM alpine\nCOPY $NAME /dest/$NAME\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_run_command(tmp_path: Path) -> None:
    """ARG values are substituted inside RUN commands."""
    runner = FakeRunner()
    _build(tmp_path, "ARG CMD=true\nFROM alpine\nRUN $CMD\n", runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && true"


def test_build_env_substitution_in_env_values(tmp_path: Path) -> None:
    """ARG values are substituted inside ENV values."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG V=7\nFROM alpine\nENV X=$V\n", runner)
    assert report.env == {"X": "7"}


def test_build_env_substitution_in_workdir(tmp_path: Path) -> None:
    """ARG values are substituted inside WORKDIR paths."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG D=/app\nFROM alpine\nWORKDIR $D\n", runner)
    assert report.workdir == "/app"


def test_build_env_substitution_in_from_with_braces(tmp_path: Path) -> None:
    """Brace-form ARG substitution works in the FROM reference."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG BASE=alpine\nFROM ${BASE}:3\n", runner)
    assert report.image == "alpine:3"


def test_build_env_substitution_in_arg_default(tmp_path: Path) -> None:
    """Pre-FROM ARG values are substituted into later ARG defaults."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG A=1\nARG B=$A\nFROM alpine\nENV X=$B\n", runner)
    assert report.env == {"X": "1"}


def test_build_env_substitution_in_add_source(tmp_path: Path) -> None:
    """ARG values are substituted inside ADD sources."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG NAME=a.txt\nFROM alpine\nADD $NAME /dest/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_add_target(tmp_path: Path) -> None:
    """ARG values are substituted inside ADD targets."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG T=/dest\nFROM alpine\nADD a.txt $T/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_copy_target(tmp_path: Path) -> None:
    """ARG values are substituted inside COPY targets."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG T=/dest\nFROM alpine\nCOPY a.txt $T/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_env_key_value_pair(tmp_path: Path) -> None:
    """ARG values are substituted inside ENV K=V pairs."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG V=9\nFROM alpine\nENV A=$V B=${V}\n", runner)
    assert report.env == {"A": "9", "B": "9"}


def test_build_env_substitution_in_env_space_form(tmp_path: Path) -> None:
    """ARG values are substituted inside ENV 'K V' form."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG V=9\nFROM alpine\nENV K $V\n", runner)
    assert report.env == {"K": "9"}


def test_build_env_substitution_in_run_json_form(tmp_path: Path) -> None:
    """ARG values are substituted inside RUN exec form arguments."""
    runner = FakeRunner()
    _build(tmp_path, 'ARG X=hi\nFROM alpine\nRUN ["echo", "$X"]\n', runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && echo hi"


def test_build_env_substitution_in_workdir_braces(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside WORKDIR."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG D=/app\nFROM alpine\nWORKDIR ${D}\n", runner)
    assert report.workdir == "/app"


def test_build_env_substitution_in_arg_name_default_braces(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ARG defaults."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG A=1\nARG B=${A}\nFROM alpine\nENV X=$B\n", runner)
    assert report.env == {"X": "1"}


def test_build_env_substitution_in_copy_source_braces(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside COPY sources."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG NAME=a.txt\nFROM alpine\nCOPY ${NAME} /dest/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_add_source_braces(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ADD sources."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG NAME=a.txt\nFROM alpine\nADD ${NAME} /dest/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_env_value_braces(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ENV values."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG V=7\nFROM alpine\nENV X=${V}\n", runner)
    assert report.env == {"X": "7"}


def test_build_env_substitution_in_run_command_braces(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside RUN commands."""
    runner = FakeRunner()
    _build(tmp_path, "ARG CMD=true\nFROM alpine\nRUN ${CMD}\n", runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && true"


def test_build_env_substitution_in_run_json_form_braces(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside RUN exec form arguments."""
    runner = FakeRunner()
    _build(tmp_path, 'ARG X=hi\nFROM alpine\nRUN ["echo", "${X}"]\n', runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && echo hi"


def test_build_env_substitution_in_copy_target_braces(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside COPY targets."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG T=/dest\nFROM alpine\nCOPY a.txt ${T}/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_add_target_braces(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ADD targets."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG T=/dest\nFROM alpine\nADD a.txt ${T}/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_env_space_form_braces(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ENV 'K V' form."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG V=9\nFROM alpine\nENV K ${V}\n", runner)
    assert report.env == {"K": "9"}


def test_build_env_substitution_in_arg_default_braces(tmp_path: Path) -> None:
    """Brace-form ARG substitution works in ARG default values."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG A=1\nARG B=${A}\nFROM alpine\nENV X=$B\n", runner)
    assert report.env == {"X": "1"}


def test_build_env_substitution_in_from_braces(tmp_path: Path) -> None:
    """Brace-form ARG substitution works in the FROM reference."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG BASE=alpine\nFROM ${BASE}\n", runner)
    assert report.image == "alpine"


def test_build_env_substitution_in_env_key_value_pair_braces(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ENV K=V pairs."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG V=9\nFROM alpine\nENV A=${V} B=${V}\n", runner)
    assert report.env == {"A": "9", "B": "9"}


def test_build_env_substitution_in_workdir_braces_only(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside WORKDIR paths."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG D=/app\nFROM alpine\nWORKDIR ${D}\n", runner)
    assert report.workdir == "/app"


def test_build_env_substitution_in_arg_name_default_braces_only(tmp_path: Path) -> None:
    """Brace-form ARG substitution works in ARG default values."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG A=1\nARG B=${A}\nFROM alpine\nENV X=$B\n", runner)
    assert report.env == {"X": "1"}


def test_build_env_substitution_in_copy_source_braces_only(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside COPY sources."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG NAME=a.txt\nFROM alpine\nCOPY ${NAME} /dest/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_add_source_braces_only(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ADD sources."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG NAME=a.txt\nFROM alpine\nADD ${NAME} /dest/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_env_value_braces_only(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ENV values."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG V=7\nFROM alpine\nENV X=${V}\n", runner)
    assert report.env == {"X": "7"}


def test_build_env_substitution_in_run_command_braces_only(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside RUN commands."""
    runner = FakeRunner()
    _build(tmp_path, "ARG CMD=true\nFROM alpine\nRUN ${CMD}\n", runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && true"


def test_build_env_substitution_in_run_json_form_braces_only(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside RUN exec form arguments."""
    runner = FakeRunner()
    _build(tmp_path, 'ARG X=hi\nFROM alpine\nRUN ["echo", "${X}"]\n', runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && echo hi"


def test_build_env_substitution_in_copy_target_braces_only(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside COPY targets."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG T=/dest\nFROM alpine\nCOPY a.txt ${T}/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_add_target_braces_only(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ADD targets."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG T=/dest\nFROM alpine\nADD a.txt ${T}/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_env_space_form_braces_only(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ENV 'K V' form."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG V=9\nFROM alpine\nENV K ${V}\n", runner)
    assert report.env == {"K": "9"}


def test_build_env_substitution_in_arg_default_braces_only_twice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works in ARG default values."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG A=1\nARG B=${A}\nFROM alpine\nENV X=$B\n", runner)
    assert report.env == {"X": "1"}


def test_build_env_substitution_in_copy_source_braces_only_twice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside COPY sources."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG NAME=a.txt\nFROM alpine\nCOPY ${NAME} /dest/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_add_source_braces_only_twice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ADD sources."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG NAME=a.txt\nFROM alpine\nADD ${NAME} /dest/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_env_value_braces_only_twice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ENV values."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG V=7\nFROM alpine\nENV X=${V}\n", runner)
    assert report.env == {"X": "7"}


def test_build_env_substitution_in_run_command_braces_only_twice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside RUN commands."""
    runner = FakeRunner()
    _build(tmp_path, "ARG CMD=true\nFROM alpine\nRUN ${CMD}\n", runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && true"


def test_build_env_substitution_in_run_json_form_braces_only_twice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside RUN exec form arguments."""
    runner = FakeRunner()
    _build(tmp_path, 'ARG X=hi\nFROM alpine\nRUN ["echo", "${X}"]\n', runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && echo hi"


def test_build_env_substitution_in_copy_target_braces_only_twice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside COPY targets."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG T=/dest\nFROM alpine\nCOPY a.txt ${T}/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_add_target_braces_only_twice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ADD targets."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG T=/dest\nFROM alpine\nADD a.txt ${T}/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_env_space_form_braces_only_twice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ENV 'K V' form."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG V=9\nFROM alpine\nENV K ${V}\n", runner)
    assert report.env == {"K": "9"}


def test_build_env_substitution_in_arg_default_braces_only_thrice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works in ARG default values."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG A=1\nARG B=${A}\nFROM alpine\nENV X=$B\n", runner)
    assert report.env == {"X": "1"}


def test_build_env_substitution_in_copy_source_braces_only_thrice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside COPY sources."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG NAME=a.txt\nFROM alpine\nCOPY ${NAME} /dest/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_add_source_braces_only_thrice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ADD sources."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG NAME=a.txt\nFROM alpine\nADD ${NAME} /dest/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_env_value_braces_only_thrice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ENV values."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG V=7\nFROM alpine\nENV X=${V}\n", runner)
    assert report.env == {"X": "7"}


def test_build_env_substitution_in_run_command_braces_only_thrice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside RUN commands."""
    runner = FakeRunner()
    _build(tmp_path, "ARG CMD=true\nFROM alpine\nRUN ${CMD}\n", runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && true"


def test_build_env_substitution_in_run_json_form_braces_only_thrice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside RUN exec form arguments."""
    runner = FakeRunner()
    _build(tmp_path, 'ARG X=hi\nFROM alpine\nRUN ["echo", "${X}"]\n', runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && echo hi"


def test_build_env_substitution_in_copy_target_braces_only_thrice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside COPY targets."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG T=/dest\nFROM alpine\nCOPY a.txt ${T}/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_add_target_braces_only_thrice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ADD targets."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG T=/dest\nFROM alpine\nADD a.txt ${T}/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_env_space_form_braces_only_thrice(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ENV 'K V' form."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG V=9\nFROM alpine\nENV K ${V}\n", runner)
    assert report.env == {"K": "9"}


def test_build_env_substitution_in_arg_default_braces_only_forth(tmp_path: Path) -> None:
    """Brace-form ARG substitution works in ARG default values."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG A=1\nARG B=${A}\nFROM alpine\nENV X=$B\n", runner)
    assert report.env == {"X": "1"}


def test_build_env_substitution_in_copy_source_braces_only_forth(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside COPY sources."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG NAME=a.txt\nFROM alpine\nCOPY ${NAME} /dest/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_add_source_braces_only_forth(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ADD sources."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG NAME=a.txt\nFROM alpine\nADD ${NAME} /dest/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_env_value_braces_only_forth(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ENV values."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG V=7\nFROM alpine\nENV X=${V}\n", runner)
    assert report.env == {"X": "7"}


def test_build_env_substitution_in_run_command_braces_only_forth(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside RUN commands."""
    runner = FakeRunner()
    _build(tmp_path, "ARG CMD=true\nFROM alpine\nRUN ${CMD}\n", runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && true"


def test_build_env_substitution_in_run_json_form_braces_only_forth(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside RUN exec form arguments."""
    runner = FakeRunner()
    _build(tmp_path, 'ARG X=hi\nFROM alpine\nRUN ["echo", "${X}"]\n', runner)
    start = [a for a in runner.argvs() if a[:2] == ("enroot", "start")][0]
    assert start[-1] == "cd / && echo hi"


def test_build_env_substitution_in_copy_target_braces_only_forth(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside COPY targets."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG T=/dest\nFROM alpine\nCOPY a.txt ${T}/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_add_target_braces_only_forth(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ADD targets."""
    runner = FakeRunner()
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "a.txt").write_text("a", encoding="utf-8")
    df = _write(tmp_path, "Dockerfile", "ARG T=/dest\nFROM alpine\nADD a.txt ${T}/a.txt\n")
    build_image(
        df,
        context,
        tmp_path / "out.sqsh",
        data_root=str(tmp_path / "data"),
        image_store=str(tmp_path / "store"),
        scratch_root=str(tmp_path / "scratch"),
        runner=runner,
    )
    rootfs = runner.exported_root
    assert rootfs is not None
    assert (rootfs / "dest" / "a.txt").read_text(encoding="utf-8") == "a"


def test_build_env_substitution_in_env_space_form_braces_only_forth(tmp_path: Path) -> None:
    """Brace-form ARG substitution works inside ENV 'K V' form."""
    runner = FakeRunner()
    report = _build(tmp_path, "ARG V=9\nFROM alpine\nENV K ${V}\n", runner)
    assert report.env == {"K": "9"}


def test_workdir_creates_its_directory_like_docker(tmp_path: Path) -> None:
    """WORKDIR /app with no COPY into it still leaves /app in the exported rootfs."""
    runner = FakeRunner()
    _build(tmp_path, "FROM alpine\nWORKDIR /app\nRUN true\n", runner)
    assert runner.exported_root is not None
    assert (runner.exported_root / "app").is_dir()
