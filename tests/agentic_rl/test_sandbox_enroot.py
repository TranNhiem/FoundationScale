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
    # spawn()/terminate(): the no-network holder (recorded lazily in self.spawned / self.terminated)
    def spawn(self, argv) -> int:
        self.__dict__.setdefault("spawned", []).append(list(argv))
        return 4242

    def terminate(self, pid: int) -> None:
        self.__dict__.setdefault("terminated", []).append(pid)


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
        Path(call.argv[3]).write_text("sqsh")  # the -o output path

    runner.on_run = on_run
    assert sbx.image_path() == str(expected)
    assert expected.exists()
    assert not list(expected.parent.glob("*.partial"))
    argv = runner.calls[0].argv
    assert argv[:3] == ("enroot", "import", "-o")
    assert argv[3].startswith(str(expected) + ".") and argv[3].endswith(".partial")
    assert argv[4] == "docker://python:3.12-slim"
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
    # every exec joins ONE namespace held for the sandbox's lifetime (as a Docker container's), so a service
    # started by one exec is reachable from the next; the holder brings loopback up, with no route off the box
    assert argv[:6] == ("nsenter", "--target=4242", "--user", "--net", "--preserve-credentials", "--")
    (holder,) = runner.spawned
    assert holder[:2] == ["unshare", "-rn"] and "ip link set lo up" in holder[-1]
    assert "enroot" in argv


@pytest.mark.parametrize("network", ["no-network", "public"])
def test_exec_runs_as_container_root(tmp_path: Path, network: str) -> None:
    # Docker semantics: RUN and task commands are root. Without --root a public-network exec (every build
    # step) kept the caller's uid and dpkg refused "requires superuser privilege".
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner, network=network)
    sbx.start()
    runner.calls.clear()
    sbx.exec("echo hi")
    argv = runner.calls[0].argv
    i = argv.index("start")
    assert argv[i - 1] == "enroot"
    assert "--root" in argv[i + 1 : argv.index(sbx.spec.name)]


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
    assert argv[0] == "nsenter"
    i = argv.index("taskset")
    assert argv[i + 1] == "-c"
    assert argv[i + 2] == "0,2"


def test_exec_memory_mb_cgroup_scope(tmp_path: Path) -> None:
    # A cgroup v2 limit (Docker --memory semantics), not an rlimit: RLIMIT_AS broke V8/Chromium and
    # RLIMIT_DATA broke ASan's MAP_NORESERVE shadow (both measured on real Skill2Env tasks).
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner, memory_mb=256)
    sbx.start()
    runner.calls.clear()
    sbx.exec("true")
    argv = runner.calls[0].argv
    assert argv[:4] == ("systemd-run", "--user", "--scope", "--quiet")
    assert "MemoryMax=256M" in argv and "MemorySwapMax=0" in argv and "OOMPolicy=continue" in argv
    rest = argv[argv.index("--") + 1 :]
    assert rest[0] == "nsenter"
    assert "prlimit" not in argv


def test_start_refuses_when_memory_scope_unavailable(tmp_path: Path) -> None:
    """Fail closed: a memory limit that cannot be enforced must not run unbounded."""
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner, memory_mb=256)
    runner.results = [RunResult(returncode=1, stdout="", stderr="Failed to connect to bus")]
    with pytest.raises(SandboxError, match="cgroup v2 memory scope"):
        sbx.start()


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
    assert env_pairs == ["HOME=/root", "A=build", "B=spec", "C=call"]  # HOME: Docker default, lowest precedence


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
    assert not list(Path(tmp_path / "images").glob(f"{digest}.sqsh*.partial"))


def test_image_path_concurrent_importer_partial_untouched(tmp_path: Path) -> None:
    """Another importer's in-flight temp file is neither reused nor deleted (R0 race: shared .partial)."""
    runner = FakeRunner()
    sbx = make_sandbox(tmp_path, runner)
    digest = hashlib.sha256(b"docker://python:3.12-slim").hexdigest()[:16]
    images = Path(tmp_path / "images")
    images.mkdir(parents=True, exist_ok=True)
    other = images / f"{digest}.sqsh.partial"
    other.write_text("another worker, mid-import")

    def on_run(call: FakeCall) -> None:
        assert call.argv[3] != str(other)
        Path(call.argv[3]).write_text("sqsh")

    runner.on_run = on_run
    assert sbx.image_path() == str(images / f"{digest}.sqsh")
    assert other.read_text() == "another worker, mid-import"


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


def test_subprocess_runner_kill_all_ends_an_inflight_run() -> None:
    """kill_all reaches a run blocked on a process that never exits, child processes included."""
    import threading
    import time

    runner = SubprocessRunner()
    out: list[RunResult] = []
    worker = threading.Thread(target=lambda: out.append(runner.run(["sh", "-c", "sleep 60 & sleep 60"])))
    worker.start()
    deadline = time.monotonic() + 10
    while not runner._live and time.monotonic() < deadline:
        time.sleep(0.05)
    runner.kill_all()
    worker.join(10)
    assert not worker.is_alive()
    assert out[0].returncode == -9
    assert not runner._live


def test_stop_kills_running_execs_first(tmp_path: Path) -> None:
    runner = FakeRunner()
    killed: list[int] = []
    runner.kill_all = lambda: killed.append(len(runner.calls))  # type: ignore[attr-defined]
    sbx = import_image(tmp_path, runner)
    sbx.start()
    sbx.stop(delete=True)
    assert killed, "stop() must kill in-flight execs before removing the container"
    assert runner.calls[killed[0]].argv[:3] == ("enroot", "remove", "-f")


def test_exec_disables_enroot_login_shell(tmp_path: Path) -> None:
    """A login shell sources the image's /etc/profile, which resets PATH and drops the image ENV PATH
    (golang: "go: not found"). Docker never uses one; enroot defaults to yes and reads "" as default."""
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    runner.calls.clear()
    sbx.exec("go version")
    assert runner.calls[0].env is not None
    assert runner.calls[0].env.get("ENROOT_LOGIN_SHELL") == "no"


def test_exec_defaults_home_like_docker(tmp_path: Path) -> None:
    """HOME=/root unless the image's environment sets it (Docker semantics); caller env still wins."""
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    runner.calls.clear()
    sbx.exec("go env GOCACHE")
    assert "HOME=/root" in runner.calls[0].argv
    environment = Path(sbx.rootfs()) / "etc" / "environment"
    environment.parent.mkdir(parents=True, exist_ok=True)
    environment.write_text("PATH=/usr/bin\nHOME=/home/app\n")
    runner.calls.clear()
    sbx.exec("true")
    assert not any(a.startswith("HOME=") for a in runner.calls[0].argv)
    runner.calls.clear()
    sbx.exec("true", env={"HOME": "/work"})
    assert "HOME=/work" in runner.calls[0].argv


def test_stop_kills_container_processes_and_the_netns_holder(tmp_path: Path, monkeypatch) -> None:
    """Docker's stop kills every container process; daemons used to outlive their sandbox (73 leaked)."""
    runner = FakeRunner()
    sbx = import_image(tmp_path, runner)
    sbx.start()
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(sbx, "_container_pids", lambda: [111, 222])
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    sbx.stop(delete=True)
    assert sorted(p for p, _ in killed) == [111, 222]
    assert runner.terminated == [4242]


def test_container_pids_ignores_host_processes(tmp_path: Path) -> None:
    """Host processes (this one, a plain child) have no root mount at the sandbox rootfs, so they never match."""
    import subprocess

    sbx = make_sandbox(tmp_path, FakeRunner())
    child = subprocess.Popen(["sleep", "30"])
    try:
        pids = sbx._container_pids()
        assert os.getpid() not in pids and child.pid not in pids
    finally:
        child.kill()
        child.wait()


def test_subprocess_runner_spawn_waits_for_ready_and_refuses_silence() -> None:
    """spawn() returns only after the helper says ready (its namespaces exist); a silent helper is refused."""
    runner = SubprocessRunner()
    pid = runner.spawn(["sh", "-c", "sleep 0.3; echo ready; exec sleep 30"])
    try:
        assert os.path.exists(f"/proc/{pid}")
    finally:
        runner.terminate(pid)
    with pytest.raises(SandboxError, match="expected 'ready'"):
        runner.spawn(["sleep", "30"], ready_timeout_s=0.5)
