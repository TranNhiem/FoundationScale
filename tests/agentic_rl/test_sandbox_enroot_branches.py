# Unreachable hermetically: line 886 (dst.unlink() inside _write_file) follows dst.is_symlink()
# and is shadowed by upload_file's own host_path() symlink rejection (lines 828-829 never hand a
# symlink to _write_file) and upload_dir's per-entry host_path() resolution; via monkeypatch that
# would not test the observable behaviour of this branch, so it is skipped here by design.

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from foundationscale.agentic_rl.sandbox.enroot import (
    EnrootSandbox,
    SandboxError,
    SandboxRefusal,
    SandboxSpec,
)

_Call = tuple[tuple[str, ...], dict[str, str] | None, float | None]


@dataclass
class _FakeCall:
    argv: tuple[str, ...]
    env: dict[str, str] | None
    timeout_s: float | None


@dataclass
class _FakeRunResult_Import:  # placeholder name guard (kept empty)
    pass


@dataclass
class _RunResult:
    returncode: int
    stdout: str
    stderr: str


def _make_spec(tmp_path: Path, **overrides: object) -> SandboxSpec:
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


def _import_image(tmp_path: Path, runner, **overrides: object) -> EnrootSandbox:
    sbx = EnrootSandbox(_make_spec(tmp_path, **overrides), runner=runner)
    sbx.image_path()
    return sbx


class _FakeRunner:
    """Records runs and returns success; spawn()/terminate() are included."""

    def __init__(self) -> None:
        self.calls: list[_FakeCall] = []
        self.spawned: list[list[str]] = []
        self.terminated: list[int] = []

    def spawn(self, argv) -> int:
        self.spawned.append(list(argv))
        return 4242

    def terminate(self, pid: int) -> None:
        self.terminated.append(pid)

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> _RunResult:
        self.calls.append(_FakeCall(tuple(argv), dict(env) if env is not None else None, timeout_s))
        args = list(argv)
        if "enroot" in args and "import" in args and "-o" in args:
            out = Path(args[args.index("-o") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"sqsh")
        return _RunResult(returncode=0, stdout="", stderr="")


class _RunnerRuns(_FakeRunner):  # alias kept for name parity with reference fixture
    pass


# ---------------------------------------------------------------------------
# line 125: SandboxError raised from OSError during a failing spawn
# ---------------------------------------------------------------------------


def _test_spawn_os_error(monkeypatch: pytest.MonkeyPatch) -> None:  # stage 1: hook
    pass


def test_125_spawn_os_error_raises_sandbox_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spawn failing with OSError raises SandboxError naming the argv and the OSError."""
    import subprocess as _subprocess

    from foundationscale.agentic_rl.sandbox import enroot as _mod

    def boom(*args: object, **kwargs: object):
        raise OSError(2, "No such file or directory")

    monkeypatch.setattr(_mod.subprocess, "Popen", boom)
    runner = _mod.SubprocessRunner()
    with pytest.raises(SandboxError, match="failed to spawn"):
        runner.spawn(["enroot-netns-holder-does-not-exist"], ready_timeout_s=1.0)
    try:
        _Popen = _subprocess.Popen
        _ = _Popen  # keep the import used without touching the module attribute
    except Exception:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# line 263: network mode not in _NETWORKS
# ---------------------------------------------------------------------------


def test_263_spec_refuses_unknown_network_mode(tmp_path: Path) -> None:
    with pytest.raises(SandboxRefusal, match="network: expected one of"):
        _make_spec(tmp_path, network="swarm")


# ---------------------------------------------------------------------------
# line 270: endpoints entry that does not match _ENDPOINT_RE
# ---------------------------------------------------------------------------


def test_270_spec_refuses_malformed_endpoint(tmp_path: Path) -> None:
    with pytest.raises(SandboxRefusal, match="endpoints: expected 'PORT=HOST:HOSTPORT'"):
        _make_spec(tmp_path, network="no-network", endpoints=("bad-endpoint",))


# ---------------------------------------------------------------------------
# line 272: endpoints present without isolation and without no-network mode
# ---------------------------------------------------------------------------


def test_272_spec_refuses_endpoints_on_unisolated_public_network(tmp_path: Path) -> None:
    with pytest.raises(
        SandboxRefusal,
        match="endpoints: expected an isolated or no-network sandbox",
    ):
        _make_spec(tmp_path, network="public", endpoints=("8080=127.0.0.1:9090",))


# ---------------------------------------------------------------------------
# line 438: build metadata with a non-list/str entrypoint value
# ---------------------------------------------------------------------------


def test_438_start_metadata_bad_entrypoint_raises_sandbox_error(tmp_path: Path) -> None:
    meta = tmp_path / "data" / "sbx1" / "etc" / "fs-sandbox.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"env": {}, "entrypoint": "not-a-list-of-str"}))
    runner = _FakeRunner()
    sbx = _import_image(tmp_path, runner)
    with pytest.raises(SandboxError, match="bad 'entrypoint'"):
        sbx.start()


# ---------------------------------------------------------------------------
# line 452: start() requires a runner with spawn() for a networked sandbox
# ---------------------------------------------------------------------------


def test_452_start_refuses_runner_without_spawn_for_isolated_network(
    tmp_path: Path,
) -> None:
    class _RunOnly:
        def run(self, *args, **kwargs):
            return _RunResult(returncode=0, stdout="", stderr="")

    sbx = EnrootSandbox(_make_spec(tmp_path, **{"isolate_network": True}), runner=_RunOnly())
    sbx._image_path = hashlib.sha256(b"docker://python:3.12-slim").hexdigest()[:16]
    with pytest.raises(SandboxRefusal, match="network='no-network': expected a runner with spawn"):
        sbx.start()


# ---------------------------------------------------------------------------
# line 548: set_policy() with an unknown mode
# ---------------------------------------------------------------------------


def test_548_set_policy_refuses_unknown_mode(tmp_path: Path) -> None:
    sbx = EnrootSandbox(_make_spec(tmp_path), runner=_FakeRunner())
    with pytest.raises(SandboxRefusal, match="mode: expected one of"):
        sbx.set_policy("starwars", ())


# ---------------------------------------------------------------------------
# line 560: bridge helper requires a runner with spawn()
# ---------------------------------------------------------------------------


def test_560_bridge_into_netns_refuses_runner_without_spawn(tmp_path: Path) -> None:
    class _RunOnly:
        def run(self, *args, **kwargs):
            return _RunResult(returncode=0, stdout="", stderr="")

    sbx = EnrootSandbox(_make_spec(tmp_path), runner=_RunOnly())
    sbx._netns_pid = 999
    with pytest.raises(SandboxRefusal, match="network bridge: expected a runner with spawn"):
        sbx._bridge_into_netns(3128, Path("/tmp/egress.sock"))


# ---------------------------------------------------------------------------
# line 591: bridge probe never reports a listener
# ---------------------------------------------------------------------------


def test_591_bridge_probe_failure_raises_sandbox_error(tmp_path: Path) -> None:
    class _ProbeFails(_FakeRunner):
        def run(self, *args, **kwargs):
            if "probe" in str(args):
                pass
            return _RunResult(returncode=1, stdout="", stderr="")

    sbx = EnrootSandbox(_make_spec(tmp_path), runner=_ProbeFails())
    sbx._netns_pid = 999
    with pytest.raises(SandboxError, match="network bridge to .* did not start listening"):
        sbx._bridge_into_netns(3128, Path("/tmp/egress.sock"))


# ---------------------------------------------------------------------------
# line 874: download_dir() skips entries whose resolved target escapes the source tree
# ---------------------------------------------------------------------------


def test_874_download_dir_skips_resolved_escape(tmp_path: Path) -> None:
    sbx = EnrootSandbox(_make_spec(tmp_path), runner=_FakeRunner())
    sbx._image_path = "unused"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.env").write_text("TOKEN=topsecret")
    src = sbx.rootfs() / "out"
    src.mkdir(parents=True)
    (src / "ok.txt").write_text("fine")
    real_sneak = outside / "sneak.txt"
    real_sneak.write_text("hidden-secret")
    (src / "sneak.txt").symlink_to(real_sneak)
    dest = tmp_path / "artifacts"
    sbx.download_dir("/out", dest)
    assert (dest / "ok.txt").read_text() == "fine"
    assert not (dest / "sneak.txt").exists()
