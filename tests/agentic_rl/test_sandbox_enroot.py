"""Tests for foundationscale.agentic_rl.sandbox.enroot."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from foundationscale.agentic_rl.sandbox.enroot import (
    EnrootSandbox,
    RunResult,
    SandboxError,
    SandboxRefusal,
    SandboxSpec,
    SubprocessRunner,
)


@dataclass
class FakeCall:
    argv: tuple[str, ...]
    env: dict[str, str] | None
    timeout_s: float | None


@dataclass
class FakeRunner:
    """Records every run and returns scripted RunResults in order."""

    results: list[RunResult] = field(default_factory=list)
    calls: list[FakeCall] = field(default_factory=list)
    on_run: object = None

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> RunResult:
        self.calls.append(FakeCall(tuple(argv), dict(env) if env is not None else None, timeout_s))
        _emulate_enroot(argv, env)
        if self.on_run is not None:
            self.on_run(self.calls[-1])
        if self.results:
            return self.results.pop(0)
        return RunResult(returncode=0, stdout="", stderr="")


def _emulate_enroot(argv: Sequence[str], env: Mapping[str, str] | None) -> None:
    """Reproduce enroot's filesystem side effects so file-transfer tests use real dirs."""
    args = list(argv)
    if "enroot" not in args:
        return
    sub = args[args.index("enroot") + 1 :]
    data_root = Path((env or {}).get("ENROOT_DATA_PATH", "/nonexistent"))
    if sub[:1] in (["import"], ["export"]) and "-o" in sub:
        out = Path(sub[sub.index("-o") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"sqsh")
    elif sub[:1] == ["create"] and "-n" in sub:
        (data_root / sub[sub.index("-n") + 1]).mkdir(parents=True, exist_ok=True)
    elif sub[:1] == ["remove"]:
        import shutil

        shutil.rmtree(data_root / sub[-1], ignore_errors=True)


def make_spec(tmp_path: Path, **overrides: object) -> SandboxSpec:
    data_root = tmp_path / "data"
    image_store = tmp_path / "images"
    scratch_root = tmp_path / "scratch"
    for d in (data_root, image_store, scratch_root):
        d.mkdir(parents=True, exist_ok=True)
    fields: dict[str, object] = {
        "name": "sbx1",
        "image": "docker://python:3.12-slim",
        "data_root": str(data_root),
        "image_store": str(image_store),
        "scratch_root": str(scratch_root),
    }
    fields.update(overrides)
    return SandboxSpec(**fields)  # type: ignore[arg-type]


def make_sandbox(tmp_path: Path, runner: FakeRunner, **overrides: object) -> EnrootSandbox:
    return EnrootSandbox(make_spec(tmp_path, **overrides), runner=runner)


def import_image(tmp_path: Path, runner: FakeRunner, **overrides: object) -> EnrootSandbox:
    sbx = make_sandbox(tmp_path, runner, **overrides)
    sbx.image_path()
    return sbx


def test_spec_refuses_bad_name(tmp_path: Path) -> None:
    with pytest.raises(SandboxRefusal):
        make_spec(tmp_path, name="bad name!")


def test_spec_refuses_relative_roots(tmp_path: Path) -> None:
    with pytest.raises(SandboxRefusal):
        make_spec(tmp_path, data_root="relative/data")
    with pytest.raises(SandboxRefusal):
        make_spec(tmp_path, image_store="relative/images")
    with pytest.raises(SandboxRefusal):
        make_spec(tmp_path, scratch_root="relative/scratch")


def test_spec_refuses_allowlist_network(tmp_path: Path) -> None:
    with pytest.raises(SandboxRefusal):
        make_spec(tmp_path, network="allowlist")


def test_spec_refuses_empty_cpu_set(tmp_path: Path) -> None:
    with pytest.raises(SandboxRefusal):
        make_spec(tmp_path, cpu_set=())


def test_spec_refuses_bool_memory(tmp_path: Path) -> None:
    with pytest.raises(SandboxRefusal):
        make_spec(tmp_path, memory_mb=True)


def test_image_path_sqsh_missing_refused(tmp_path: Path) -> None:
    sbx = make_sandbox(tmp_path, FakeRunner(), image=str(tmp_path / "absent.sqsh"))
    with pytest.raises(SandboxRefusal):
        sbx.image_path()


def test_image_path_sqsh_existing(tmp_path: Path) -> None:
    sqsh = tmp_path / "img.sqsh"
    sqsh.write_text("sqsh")
    runner = FakeRunner()
    sbx = make_sandbox(tmp_path, runner, image=str(sqsh))
    assert sbx.image_path() == str(sqsh)
    assert runner.calls == []


def test_image_path_docker_hash_named_import_once(tmp_path: Path) -> None:
    runner = FakeRunner()
    sbx = make_sandbox(tmp_path, runner)
    digest = hashlib.sha256(b"docker://python:3.12-slim").hexdigest()[:16]
    expected = Path(tmp_path / "images") / f"{digest}.sqsh"

    def on_run(call: FakeCall) -> None:
        Path(call.argv[3] + ".partial").write_text("partial")

    runner.on_run = on_run
    assert sbx.image_path() == str(expected)
    assert expected.exists()
    assert not (expected.parent / (expected.name + ".partial")).exists()
    assert runner.calls[0].argv == (
        "enroot",
        "import",
        "-o",
        str(expected) + ".partial",
        "docker://python:3.12-slim",
    )
    assert sbx.image_path() == str(expected)
    assert len(runner.calls) == 1


def test_image_path_import_failure_raises_sandbox_error(tmp_path: Path) -> None:
    runner = FakeRunner(results=[RunResult(returncode=1, stdout="", stderr="boom " + "x" * 50)])
    sbx = make_sandbox(tmp_path, runner)
    with pytest.raises(SandboxError) as excinfo:
        sbx.image_path()
    assert "x" * 20 in str(excinfo.value)


def test_start_creates_scratch_and_calls_enroot_create(tmp_path: Path) -> None:
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    runner.calls.clear()
    sbx.start()
    call = runner.calls[0]
    assert call.argv == ("enroot", "create", "-n", "sbx1", sbx.image_path())
    assert call.env == {"ENROOT_DATA_PATH": str(tmp_path / "data")}
    scratch = tmp_path / "scratch" / "sbx1" / "tmp"
    assert scratch.is_dir()


def test_start_twice_refused(tmp_path: Path) -> None:
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    with pytest.raises(SandboxRefusal):
        sbx.start()


def test_exec_no_network_prefix(tmp_path: Path) -> None:
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    runner.calls.clear()
    sbx.exec("echo hi")
    argv = runner.calls[0].argv
    assert argv[:2] == ("unshare", "-rn")
    assert "enroot" in argv


def test_exec_public_no_prefix(tmp_path: Path) -> None:
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner, network="public")
    sbx.start()
    runner.calls.clear()
    sbx.exec("echo hi")
    assert runner.calls[0].argv[0] == "enroot"


def test_exec_cpu_set_taskset(tmp_path: Path) -> None:
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner, cpu_set=(0, 2))
    sbx.start()
    runner.calls.clear()
    sbx.exec("true")
    argv = runner.calls[0].argv
    assert argv[:3] == ("unshare", "-rn", "taskset")
    assert argv[3] == "-c"
    assert argv[4] == "0,2"


def test_exec_memory_mb_prlimit_bytes(tmp_path: Path) -> None:
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner, memory_mb=256)
    sbx.start()
    runner.calls.clear()
    sbx.exec("true")
    argv = runner.calls[0].argv
    assert argv[:3] == ("unshare", "-rn", "prlimit")
    assert argv[3] == f"--as={256 * 1048576}"


def test_exec_mounts_scratch_over_tmp(tmp_path: Path) -> None:
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    runner.calls.clear()
    sbx.exec("true")
    argv = runner.calls[0].argv
    scratch = tmp_path / "scratch" / "sbx1" / "tmp"
    assert "--mount" in argv
    assert f"{scratch}:/tmp" in argv


def test_exec_env_precedence(tmp_path: Path) -> None:
    meta = tmp_path / "data" / "sbx1" / "etc" / "fs-sandbox.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"env": {"A": "build", "B": "build", "C": "build"}, "workdir": "/"}))
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner, env={"B": "spec", "C": "spec"})
    sbx.start()
    runner.calls.clear()
    sbx.exec("true", env={"C": "call"})
    argv = runner.calls[0].argv
    env_pairs = [argv[i + 1] for i, x in enumerate(argv) if x == "--env"]
    assert env_pairs == ["A=build", "B=spec", "C=call"]


def test_exec_cwd_quoting(tmp_path: Path) -> None:
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner, workdir="/work dir")
    sbx.start()
    runner.calls.clear()
    sbx.exec("echo hi", cwd="/a b/c")
    argv = runner.calls[0].argv
    assert argv[-3:] == ("sh", "-c", "cd '/a b/c' && echo hi")


def test_exec_before_start_refused(tmp_path: Path) -> None:
    sbx = import_image(tmp_path, FakeRunner())
    with pytest.raises(SandboxRefusal):
        sbx.exec("true")


def test_exec_timeout_returns_124(tmp_path: Path) -> None:
    """A timed-out exec reports return_code 124 and timed_out, never a fake success."""
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    runner.results = [RunResult(returncode=124, stdout="", stderr="", timed_out=True)]
    out = sbx.exec("sleep 99", timeout_s=1.0)
    assert out.return_code == 124
    assert out.timed_out is True
    assert runner.calls[-1].timeout_s == 1.0


def test_host_path_maps_tmp_into_scratch(tmp_path: Path) -> None:
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    scratch = tmp_path / "scratch" / "sbx1" / "tmp"
    assert sbx.host_path("/tmp/x") == scratch / "x"
    assert sbx.host_path("/etc/hosts") == tmp_path / "data" / "sbx1" / "etc" / "hosts"


def test_host_path_refuses_relative(tmp_path: Path) -> None:
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    with pytest.raises(SandboxRefusal):
        sbx.host_path("etc/hosts")


def test_host_path_refuses_dotdot_escape(tmp_path: Path) -> None:
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    with pytest.raises(SandboxRefusal):
        sbx.host_path("/../outside")


def test_host_path_refuses_symlink_escape(tmp_path: Path) -> None:
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = sbx.rootfs() / "escape"
    link.symlink_to(outside)
    with pytest.raises(SandboxRefusal):
        sbx.host_path("/escape/file")


def test_upload_download_file_roundtrip(tmp_path: Path) -> None:
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    src = tmp_path / "src.txt"
    src.write_text("payload")
    sbx.upload_file(src, "/data/src.txt")
    assert (sbx.rootfs() / "data" / "src.txt").read_text() == "payload"
    out = tmp_path / "out.txt"
    sbx.download_file("/data/src.txt", str(out))
    assert out.read_text() == "payload"


def test_upload_download_dir_roundtrip(tmp_path: Path) -> None:
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    src = tmp_path / "srcdir"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("a")
    (src / "sub" / "b.txt").write_text("b")
    sbx.upload_dir(src, "/data/dst")
    assert (sbx.rootfs() / "data" / "dst" / "a.txt").read_text() == "a"
    assert (sbx.rootfs() / "data" / "dst" / "sub" / "b.txt").read_text() == "b"
    out = tmp_path / "outdir"
    sbx.download_dir("/data/dst", str(out))
    assert (out / "a.txt").read_text() == "a"
    assert (out / "sub" / "b.txt").read_text() == "b"


def test_upload_dir_merges_into_existing(tmp_path: Path) -> None:
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    dst = sbx.rootfs() / "data" / "dst"
    dst.mkdir(parents=True)
    (dst / "keep.txt").write_text("keep")
    src = tmp_path / "src"
    src.mkdir()
    (src / "new.txt").write_text("new")
    sbx.upload_dir(src, "/data/dst")
    assert (dst / "keep.txt").read_text() == "keep"
    assert (dst / "new.txt").read_text() == "new"


def test_stop_deletes_container_and_scratch(tmp_path: Path) -> None:
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    scratch = tmp_path / "scratch" / "sbx1"
    assert scratch.exists()
    runner.calls.clear()
    sbx.stop()
    assert runner.calls[0].argv == ("enroot", "remove", "-f", "sbx1")
    assert not scratch.exists()


def test_stop_idempotent(tmp_path: Path) -> None:
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    sbx.stop()
    runner.calls.clear()
    sbx.stop()
    assert runner.calls == []


def test_upload_dir_never_writes_through_a_symlink_planted_in_the_sandbox(
    tmp_path: Path,
) -> None:
    """An agent-planted symlinked dir inside the target cannot redirect an upload to the host."""
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    outside = tmp_path / "host_home"
    outside.mkdir()
    testbed = sbx.rootfs() / "testbed"
    testbed.mkdir(parents=True)
    (testbed / "x").symlink_to(outside, target_is_directory=True)
    payload = tmp_path / "payload"
    (payload / "x").mkdir(parents=True)
    (payload / "x" / "evil.txt").write_text("pwned")
    with contextlib.suppress(SandboxRefusal):
        sbx.upload_dir(payload, "/testbed")
    assert not (outside / "evil.txt").exists()


def test_download_dir_skips_symlinks_to_host_files(tmp_path: Path) -> None:
    """A symlink inside the sandbox pointing at a host file is never copied out."""
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    secret = tmp_path / "secret.env"
    secret.write_text("TOKEN=abc")
    out_dir = sbx.rootfs() / "out"
    out_dir.mkdir(parents=True)
    (out_dir / "ok.txt").write_text("fine")
    (out_dir / "leak.txt").symlink_to(secret)
    dest = tmp_path / "artifacts"
    sbx.download_dir("/out", dest)
    assert (dest / "ok.txt").read_text() == "fine"
    assert not (dest / "leak.txt").exists()


def test_upload_file_replaces_rather_than_follows_a_symlink_target(
    tmp_path: Path,
) -> None:
    """Uploading onto a path the agent made a symlink replaces the link; the host file is intact."""
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    host_file = tmp_path / "host.txt"
    host_file.write_text("original")
    app = sbx.rootfs() / "app"
    app.mkdir(parents=True)
    (app / "conf").symlink_to(host_file)
    src = tmp_path / "new.txt"
    src.write_text("uploaded")
    with contextlib.suppress(SandboxRefusal):
        sbx.upload_file(src, "/app/conf")
    assert host_file.read_text() == "original"


def test_subprocess_runner_spawn_failure_raises_sandbox_error() -> None:
    """A command that cannot be spawned raises SandboxError naming the argv."""
    with pytest.raises(SandboxError, match="failed to spawn"):
        SubprocessRunner().run(["definitely-not-a-real-binary-xyz"])


def test_subprocess_runner_timeout_kills_process_group() -> None:
    """A timed-out subprocess is killed and reported as returncode 124 with timed_out."""
    result = SubprocessRunner().run(["sleep", "30"], timeout_s=0.2)
    assert result.returncode == 124
    assert result.timed_out is True


def test_subprocess_runner_merges_env_over_os_environ() -> None:
    """An explicit env is layered over os.environ for the child process."""
    result = SubprocessRunner().run(
        ["sh", "-c", "printf '%s' \"$FS_TEST_MARKER\""],
        env={"FS_TEST_MARKER": "merged"},
    )
    assert result.stdout == "merged"
    assert "PATH" in os.environ


def test_spec_refuses_negative_cpu(tmp_path: Path) -> None:
    """A cpu_set entry below zero is refused with the ints >= 0 message."""
    with pytest.raises(SandboxRefusal, match="expected ints >= 0"):
        make_spec(tmp_path, cpu_set=(0, -1))


def test_spec_refuses_bool_cpu(tmp_path: Path) -> None:
    """A bool in cpu_set is refused as not an int >= 0."""
    with pytest.raises(SandboxRefusal, match="expected ints >= 0"):
        make_spec(tmp_path, cpu_set=(True,))


def test_spec_refuses_non_tuple_cpu_set(tmp_path: Path) -> None:
    """A cpu_set that is not a non-empty tuple is refused."""
    with pytest.raises(SandboxRefusal, match="expected non-empty tuple"):
        make_spec(tmp_path, cpu_set=[0, 1])


def test_spec_refuses_zero_memory(tmp_path: Path) -> None:
    """memory_mb below 1 is refused with the int >= 1 message."""
    with pytest.raises(SandboxRefusal, match="expected int >= 1"):
        make_spec(tmp_path, memory_mb=0)


def test_spec_refuses_non_mapping_env(tmp_path: Path) -> None:
    """An env that is not a Mapping is refused naming the offending type."""
    with pytest.raises(SandboxRefusal, match="expected Mapping"):
        make_spec(tmp_path, env=["A", "B"])


def test_spec_refuses_non_str_env_pair(tmp_path: Path) -> None:
    """A non-str env key or value is refused with the key/value repr."""
    with pytest.raises(SandboxRefusal, match="expected Mapping"):
        make_spec(tmp_path, env={"A": 1})


def test_spec_refuses_empty_workdir(tmp_path: Path) -> None:
    """An empty workdir string is refused as not a non-empty str."""
    with pytest.raises(SandboxRefusal, match="expected non-empty str"):
        make_spec(tmp_path, workdir="")


def test_image_path_refuses_non_sqsh_non_docker(tmp_path: Path) -> None:
    """An image that is neither docker:// nor an existing .sqsh is refused."""
    sbx = make_sandbox(tmp_path, FakeRunner(), image="python:3.12-slim")
    with pytest.raises(SandboxRefusal, match="expected 'docker://<ref>'"):
        sbx.image_path()


def test_image_path_import_writes_no_file_raises_sandbox_error(tmp_path: Path) -> None:
    """An import that exits 0 without writing the image raises SandboxError."""
    runner = FakeRunner()

    def on_run(call: FakeCall) -> None:
        partial = Path(call.argv[3])
        if partial.exists():
            partial.unlink()

    runner.on_run = on_run
    sbx = make_sandbox(tmp_path, runner)
    with pytest.raises(SandboxError, match="wrote no image"):
        sbx.image_path()


def test_image_path_import_failure_removes_partial(tmp_path: Path) -> None:
    """A failed import removes any partial file it left behind."""
    runner = FakeRunner(results=[RunResult(returncode=3, stdout="", stderr="nope")])

    def on_run(call: FakeCall) -> None:
        Path(call.argv[3]).write_text("partial")

    runner.on_run = on_run
    sbx = make_sandbox(tmp_path, runner)
    with pytest.raises(SandboxError, match="enroot import failed"):
        sbx.image_path()
    digest = hashlib.sha256(b"docker://python:3.12-slim").hexdigest()[:16]
    assert not (Path(tmp_path / "images") / f"{digest}.sqsh.partial").exists()


def test_start_create_failure_raises_sandbox_error(tmp_path: Path) -> None:
    """A non-zero enroot create raises SandboxError naming the sandbox."""
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    runner.calls.clear()
    runner.results = [RunResult(returncode=1, stdout="", stderr="create failed")]
    with pytest.raises(SandboxError, match="enroot create failed"):
        sbx.start()


def test_start_invalid_metadata_json_raises_sandbox_error(tmp_path: Path) -> None:
    """Unparseable build metadata raises SandboxError naming the metadata path."""
    meta = tmp_path / "data" / "sbx1" / "etc" / "fs-sandbox.json"
    meta.parent.mkdir(parents=True)
    meta.write_text("{not json")
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    with pytest.raises(SandboxError, match="invalid build metadata"):
        sbx.start()


def test_start_metadata_not_object_raises_sandbox_error(tmp_path: Path) -> None:
    """Build metadata that is not a JSON object raises SandboxError."""
    meta = tmp_path / "data" / "sbx1" / "etc" / "fs-sandbox.json"
    meta.parent.mkdir(parents=True)
    meta.write_text("[1, 2, 3]")
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    with pytest.raises(SandboxError, match="expected object"):
        sbx.start()


def test_start_metadata_bad_env_raises_sandbox_error(tmp_path: Path) -> None:
    """Build metadata with a non-str env value raises SandboxError."""
    meta = tmp_path / "data" / "sbx1" / "etc" / "fs-sandbox.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"env": {"A": 1}, "workdir": "/"}))
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    with pytest.raises(SandboxError, match="bad 'env'"):
        sbx.start()


def test_start_metadata_bad_workdir_raises_sandbox_error(tmp_path: Path) -> None:
    """Build metadata with a non-str workdir raises SandboxError."""
    meta = tmp_path / "data" / "sbx1" / "etc" / "fs-sandbox.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"env": {}, "workdir": 42}))
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    with pytest.raises(SandboxError, match="bad 'workdir'"):
        sbx.start()


def test_exec_refuses_empty_command(tmp_path: Path) -> None:
    """An empty command string is refused as not a non-empty str."""
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    with pytest.raises(SandboxRefusal, match="expected non-empty str"):
        sbx.exec("")


def test_exec_refuses_non_str_env_pair(tmp_path: Path) -> None:
    """A non-str env key or value passed to exec is refused."""
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    with pytest.raises(SandboxRefusal, match="expected Mapping"):
        sbx.exec("true", env={"A": 1})  # type: ignore[arg-type]


def test_host_path_refuses_empty_string(tmp_path: Path) -> None:
    """An empty container_path is refused as not a non-empty absolute path."""
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    with pytest.raises(SandboxRefusal, match="expected non-empty absolute path"):
        sbx.host_path("")


def test_upload_file_refuses_missing_source(tmp_path: Path) -> None:
    """Uploading a nonexistent source file is refused naming the source."""
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    with pytest.raises(SandboxRefusal, match="expected existing file"):
        sbx.upload_file(tmp_path / "nope.txt", "/data/x.txt")


def test_upload_file_into_existing_directory_uses_basename(tmp_path: Path) -> None:
    """Uploading to a directory target writes the file under its basename."""
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    dst = sbx.rootfs() / "data"
    dst.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "payload.txt"
    src.write_text("data")
    sbx.upload_file(src, "/data/")
    assert (dst / "payload.txt").read_text() == "data"


def test_upload_dir_refuses_missing_source_dir(tmp_path: Path) -> None:
    """Uploading a nonexistent source directory is refused naming the source_dir."""
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    with pytest.raises(SandboxRefusal, match="expected existing directory"):
        sbx.upload_dir(tmp_path / "nope", "/data/dst")


def test_upload_dir_skips_host_symlinks(tmp_path: Path) -> None:
    """Symlinks in the source directory are never carried into the sandbox."""
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    src = tmp_path / "src"
    src.mkdir()
    (src / "real.txt").write_text("real")
    (src / "link.txt").symlink_to(src / "real.txt")
    sbx.upload_dir(src, "/data/dst")
    dst = sbx.rootfs() / "data" / "dst"
    assert (dst / "real.txt").read_text() == "real"
    assert not (dst / "link.txt").exists()


def test_download_file_refuses_missing_source(tmp_path: Path) -> None:
    """Downloading a nonexistent sandbox file is refused naming the source."""
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    with pytest.raises(SandboxRefusal, match="expected existing file in sandbox"):
        sbx.download_file("/data/missing.txt", str(tmp_path / "out.txt"))


def test_download_file_into_directory_uses_basename(tmp_path: Path) -> None:
    """Downloading to a directory target writes the file under its basename."""
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    src = sbx.rootfs() / "data" / "f.txt"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("payload")
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    sbx.download_file("/data/f.txt", out_dir)
    assert (out_dir / "f.txt").read_text() == "payload"


def test_download_dir_refuses_missing_source_dir(tmp_path: Path) -> None:
    """Downloading a nonexistent sandbox directory is refused naming the source_dir."""
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    with pytest.raises(SandboxRefusal, match="expected existing directory in sandbox"):
        sbx.download_dir("/data/missing", str(tmp_path / "out"))


def test_download_dir_skips_escaping_symlinked_dirs(tmp_path: Path) -> None:
    """A symlinked directory escaping the sandbox source is skipped on download."""
    sbx = import_image(tmp_path, FakeRunner())
    sbx.start()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    src = sbx.rootfs() / "out"
    src.mkdir(parents=True)
    (src / "ok.txt").write_text("ok")
    (src / "escape").symlink_to(outside, target_is_directory=True)
    dest = tmp_path / "artifacts"
    sbx.download_dir("/out", dest)
    assert (dest / "ok.txt").read_text() == "ok"
    assert not (dest / "escape").exists()
    assert not (dest / "escape" / "secret.txt").exists()


def test_stop_delete_false_keeps_scratch(tmp_path: Path) -> None:
    """stop(delete=False) leaves the container and scratch untouched."""
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    scratch = tmp_path / "scratch" / "sbx1"
    runner.calls.clear()
    sbx.stop(delete=False)
    assert runner.calls == []
    assert scratch.exists()
    assert sbx._started is False


def test_stop_remove_failure_raises_sandbox_error(tmp_path: Path) -> None:
    """A non-zero enroot remove raises SandboxError naming the sandbox."""
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    runner.calls.clear()
    runner.results = [RunResult(returncode=1, stdout="", stderr="remove failed")]
    with pytest.raises(SandboxError, match="enroot remove failed"):
        sbx.stop()
