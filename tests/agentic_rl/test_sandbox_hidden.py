"""Tests for foundationscale.agentic_rl.sandbox.hidden (BINDING API section B)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from foundationscale.agentic_rl.integrity import IntegrityRefusal
from foundationscale.agentic_rl.sandbox.enroot import ExecResult
from foundationscale.agentic_rl.sandbox.hidden import (
    HiddenSpec,
    HiddenVault,
    probe_egress,
    require_no_egress,
)


class FakeSandbox:
    """Minimal EnrootSandbox surface: host_path maps '/x' to tmp_path/'rootfs'/x."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.rootfs_dir = root / "rootfs"
        self.rootfs_dir.mkdir(parents=True, exist_ok=True)
        self.scripted: list[ExecResult] = []
        self.commands: list[str] = []

    def host_path(self, container_path: str) -> Path:
        if not container_path.startswith("/"):
            raise ValueError(f"expected absolute path, got {container_path!r}")
        return self.rootfs_dir / container_path.lstrip("/")

    def exec(self, command: str, *, timeout_s: float | None = None) -> ExecResult:
        self.commands.append(command)
        return self.scripted.pop(0) if self.scripted else ExecResult("", "", 1, False)


def _make(tmp_path: Path) -> tuple[FakeSandbox, Path]:
    sandbox = FakeSandbox(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    return sandbox, vault


def _seed(sandbox: FakeSandbox) -> None:
    tests = sandbox.host_path("/tests")
    tests.mkdir(parents=True)
    (tests / "t.py").write_text("x")
    sol = sandbox.host_path("/solution")
    sol.mkdir(parents=True)
    (sol / "s.py").write_text("y")
    git = sandbox.host_path("/testbed/.git")
    git.mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main")
    (sandbox.host_path("/testbed")).mkdir(parents=True, exist_ok=True)
    (sandbox.host_path("/testbed/app.py")).write_text("z")


def test_seal_unseal_round_trip(tmp_path: Path) -> None:
    """Seal hides paths and unseal restores them byte-for-byte."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(git_worktrees=("/testbed",)), vault)
    moved = vault_obj.seal()
    assert set(moved) == {"/tests", "/solution", "/testbed/.git"}
    assert not sandbox.host_path("/tests").exists()
    assert not sandbox.host_path("/solution").exists()
    assert not sandbox.host_path("/testbed/.git").exists()
    assert sandbox.host_path("/testbed/app.py").exists()
    restored = vault_obj.unseal()
    assert set(restored) == {"/tests", "/solution", "/testbed/.git"}
    assert (sandbox.host_path("/tests") / "t.py").read_text() == "x"
    assert (sandbox.host_path("/solution") / "s.py").read_text() == "y"
    assert (sandbox.host_path("/testbed/.git") / "HEAD").read_text() == "ref: refs/heads/main"


def test_seal_hides_git_worktree_dot_git(tmp_path: Path) -> None:
    """A git worktree's .git directory is hidden alongside the listed paths."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=(), git_worktrees=("/testbed",)), vault)
    moved = vault_obj.seal()
    assert moved == ("/testbed/.git",)
    assert not sandbox.host_path("/testbed/.git").exists()
    assert sandbox.host_path("/testbed/app.py").exists()


def test_seal_skips_missing_paths(tmp_path: Path) -> None:
    """Missing hidden paths are skipped, not an error, and only existing ones are moved."""
    sandbox, vault = _make(tmp_path)
    tests = sandbox.host_path("/tests")
    tests.mkdir(parents=True)
    (tests / "t.py").write_text("x")
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=("/tests", "/solution")), vault)
    moved = vault_obj.seal()
    assert moved == ("/tests",)
    assert not tests.exists()
    assert not sandbox.host_path("/solution").exists()


def test_seal_skips_missing_worktree_git(tmp_path: Path) -> None:
    """A worktree without a .git entry is skipped silently during seal."""
    sandbox, vault = _make(tmp_path)
    sandbox.host_path("/testbed").mkdir(parents=True)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=(), git_worktrees=("/testbed",)), vault)
    assert vault_obj.seal() == ()
    assert sandbox.host_path("/testbed").exists()


def test_double_seal_refused(tmp_path: Path) -> None:
    """Sealing twice is refused even when the first seal moved nothing."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(), vault)
    vault_obj.seal()
    with pytest.raises(IntegrityRefusal):
        vault_obj.seal()


def test_seal_refuses_vault_inside_rootfs(tmp_path: Path) -> None:
    """A vault directory inside the sandbox rootfs is refused at construction."""
    sandbox, _ = _make(tmp_path)
    inside = sandbox.host_path("/vault")
    inside.mkdir(parents=True)
    with pytest.raises(IntegrityRefusal):
        HiddenVault(sandbox, HiddenSpec(), inside)


def test_seal_refuses_vault_nested_inside_rootfs(tmp_path: Path) -> None:
    """A vault nested deeper under the rootfs is refused too."""
    sandbox, _ = _make(tmp_path)
    inside = sandbox.host_path("/var/vault/deep")
    inside.mkdir(parents=True)
    with pytest.raises(IntegrityRefusal):
        HiddenVault(sandbox, HiddenSpec(), inside)


def test_assert_sealed_passes_when_hidden(tmp_path: Path) -> None:
    """assert_sealed is silent while every hidden path stays hidden."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(git_worktrees=("/testbed",)), vault)
    vault_obj.seal()
    vault_obj.assert_sealed()


def test_assert_sealed_fires_when_path_reappears(tmp_path: Path) -> None:
    """MUST_FIRE: a hidden path reappearing in the sandbox is named in the refusal."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(), vault)
    vault_obj.seal()
    sandbox.host_path("/tests").mkdir(parents=True)
    with pytest.raises(IntegrityRefusal, match="/tests"):
        vault_obj.assert_sealed()


def test_assert_sealed_fires_when_git_reappears(tmp_path: Path) -> None:
    """MUST_FIRE: a worktree .git reappearing is named in the refusal."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=(), git_worktrees=("/testbed",)), vault)
    vault_obj.seal()
    sandbox.host_path("/testbed/.git").mkdir(parents=True)
    with pytest.raises(IntegrityRefusal, match="/testbed/.git"):
        vault_obj.assert_sealed()


def test_assert_sealed_fires_when_not_sealed(tmp_path: Path) -> None:
    """MUST_FIRE: assert_sealed before any seal names the still-present hidden paths."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(), vault)
    with pytest.raises(IntegrityRefusal, match="/tests"):
        vault_obj.assert_sealed()


def test_unseal_refuses_when_not_sealed(tmp_path: Path) -> None:
    """Unseal without a prior seal is refused."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(), vault)
    with pytest.raises(IntegrityRefusal):
        vault_obj.unseal()


def test_unseal_refuses_double_unseal(tmp_path: Path) -> None:
    """Unsealing twice is refused after the first unseal restored everything."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(), vault)
    vault_obj.seal()
    vault_obj.unseal()
    with pytest.raises(IntegrityRefusal):
        vault_obj.unseal()


def test_unseal_refuses_agent_planted_file(tmp_path: Path) -> None:
    """MUST_FIRE: an agent-planted file at a hidden path blocks unseal and is named."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(), vault)
    vault_obj.seal()
    planted = sandbox.host_path("/tests")
    planted.mkdir(parents=True)
    (planted / "planted.py").write_text("cheat")
    with pytest.raises(IntegrityRefusal, match="agent wrote to hidden path"):
        vault_obj.unseal()
    assert (planted / "planted.py").exists()


def test_unseal_refuses_agent_planted_git(tmp_path: Path) -> None:
    """MUST_FIRE: an agent-planted worktree .git blocks unseal and is named."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=(), git_worktrees=("/testbed",)), vault)
    vault_obj.seal()
    planted = sandbox.host_path("/testbed/.git")
    planted.mkdir(parents=True)
    with pytest.raises(IntegrityRefusal, match="agent wrote to hidden path"):
        vault_obj.unseal()


def test_probe_egress_true(tmp_path: Path) -> None:
    """probe_egress returns True when the scripted command exits 0."""
    sandbox, _ = _make(tmp_path)
    sandbox.scripted.append(ExecResult("", "", 0, False))
    assert probe_egress(sandbox) is True
    assert len(sandbox.commands) == 1


def test_probe_egress_false(tmp_path: Path) -> None:
    """probe_egress returns False when the scripted command exits non-zero."""
    sandbox, _ = _make(tmp_path)
    sandbox.scripted.append(ExecResult("", "", 1, False))
    assert probe_egress(sandbox) is False


def test_probe_egress_false_on_timeout(tmp_path: Path) -> None:
    """A timed-out probe counts as no egress (non-zero return code)."""
    sandbox, _ = _make(tmp_path)
    sandbox.scripted.append(ExecResult("", "", 124, True))
    assert probe_egress(sandbox) is False


def test_require_no_egress_passes_when_false(tmp_path: Path) -> None:
    """require_no_egress is silent when the probe reports no egress."""
    sandbox, _ = _make(tmp_path)
    sandbox.scripted.append(ExecResult("", "", 1, False))
    require_no_egress(sandbox)


def test_require_no_egress_fires_when_true(tmp_path: Path) -> None:
    """MUST_FIRE: require_no_egress raises when the probe reports egress."""
    sandbox, _ = _make(tmp_path)
    sandbox.scripted.append(ExecResult("", "", 0, False))
    with pytest.raises(IntegrityRefusal):
        require_no_egress(sandbox)


def test_hidden_spec_rejects_non_tuple_paths(tmp_path: Path) -> None:
    """HiddenSpec refuses a paths value that is not a tuple."""
    with pytest.raises(IntegrityRefusal, match="paths: expected tuple"):
        HiddenSpec(paths=["/tests"])  # type: ignore[arg-type]


def test_hidden_spec_rejects_non_tuple_worktrees(tmp_path: Path) -> None:
    """HiddenSpec refuses a git_worktrees value that is not a tuple."""
    with pytest.raises(IntegrityRefusal, match="git_worktrees: expected tuple"):
        HiddenSpec(git_worktrees=["/testbed"])  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["tests", "", 3, None])
def test_hidden_spec_rejects_bad_path_entries(value: object) -> None:
    """HiddenSpec refuses non-absolute or non-string path entries."""
    with pytest.raises(IntegrityRefusal, match="expected absolute container paths"):
        HiddenSpec(paths=(value,))  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["/", "/tests/", "/testbed//"])
def test_hidden_spec_rejects_root_or_trailing_slash(value: str) -> None:
    """HiddenSpec refuses '/' and paths with a trailing slash."""
    with pytest.raises(IntegrityRefusal):
        HiddenSpec(paths=(value,))


def test_hidden_spec_rejects_bad_worktree_entries() -> None:
    """HiddenSpec applies the same entry rules to git_worktrees."""
    with pytest.raises(IntegrityRefusal, match="git_worktrees: expected absolute"):
        HiddenSpec(git_worktrees=("testbed",))


def test_hidden_spec_rejects_duplicate_paths() -> None:
    """HiddenSpec refuses duplicate entries in paths."""
    with pytest.raises(IntegrityRefusal, match="paths: expected unique entries"):
        HiddenSpec(paths=("/tests", "/tests"))


def test_hidden_spec_rejects_duplicate_worktrees() -> None:
    """HiddenSpec refuses duplicate entries in git_worktrees."""
    with pytest.raises(IntegrityRefusal, match="git_worktrees: expected unique entries"):
        HiddenSpec(git_worktrees=("/testbed", "/testbed"))


def test_hidden_spec_rejects_overlapping_hidden_paths() -> None:
    """HiddenSpec refuses when a worktree .git collides with an explicit path."""
    with pytest.raises(IntegrityRefusal, match="hidden paths: expected unique entries"):
        HiddenSpec(paths=("/testbed/.git",), git_worktrees=("/testbed",))


def test_hidden_vault_rejects_non_spec(tmp_path: Path) -> None:
    """HiddenVault refuses a spec that is not a HiddenSpec instance."""
    sandbox, vault = _make(tmp_path)
    with pytest.raises(IntegrityRefusal, match="spec: expected HiddenSpec"):
        HiddenVault(sandbox, {}, vault)  # type: ignore[arg-type]


def test_hidden_vault_rejects_vault_equal_to_rootfs(tmp_path: Path) -> None:
    """A vault directory equal to the rootfs is refused at construction."""
    sandbox, _ = _make(tmp_path)
    with pytest.raises(IntegrityRefusal, match="vault_dir: expected a host path outside"):
        HiddenVault(sandbox, HiddenSpec(), sandbox.rootfs_dir)


def test_seal_refuses_vault_target_already_present(tmp_path: Path) -> None:
    """Seal refuses when the vault already holds an entry for a hidden path."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    (vault / "tests").mkdir(parents=True)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=("/tests",)), vault)
    with pytest.raises(IntegrityRefusal, match="vault target already exists"):
        vault_obj.seal()


def test_assert_sealed_fires_when_unsealed_and_absent(tmp_path: Path) -> None:
    """MUST_FIRE: assert_sealed on an unsealed vault with nothing present still refuses."""
    sandbox, vault = _make(tmp_path)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=("/tests",)), vault)
    with pytest.raises(IntegrityRefusal, match="expected sealed vault"):
        vault_obj.assert_sealed()


def test_assert_sealed_detects_symlink(tmp_path: Path) -> None:
    """A symlink planted at a hidden path counts as present."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=("/tests",)), vault)
    vault_obj.seal()
    (sandbox.rootfs_dir / "tests").symlink_to(sandbox.rootfs_dir / "solution")
    with pytest.raises(IntegrityRefusal, match="/tests"):
        vault_obj.assert_sealed()


def test_unseal_refuses_agent_planted_symlink(tmp_path: Path) -> None:
    """MUST_FIRE: a symlink planted at a hidden path blocks unseal."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=("/tests",)), vault)
    vault_obj.seal()
    (sandbox.rootfs_dir / "tests").symlink_to(sandbox.rootfs_dir / "solution")
    with pytest.raises(IntegrityRefusal, match="agent wrote to hidden path"):
        vault_obj.unseal()


def test_unseal_refuses_missing_vault_entry(tmp_path: Path) -> None:
    """MUST_FIRE: a vaulted entry vanishing before unseal is refused."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=("/tests",)), vault)
    vault_obj.seal()
    shutil.rmtree(vault / "tests")
    with pytest.raises(IntegrityRefusal, match="vault entry missing"):
        vault_obj.unseal()


def test_unseal_skips_paths_never_moved(tmp_path: Path) -> None:
    """Unseal skips hidden paths that were missing at seal time."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=("/tests", "/solution", "/absent")), vault)
    vault_obj.seal()
    assert vault_obj.unseal() == ("/tests", "/solution")


def test_unseal_restores_worktree_git(tmp_path: Path) -> None:
    """Unseal restores a worktree .git entry and leaves the worktree alone."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=(), git_worktrees=("/testbed",)), vault)
    vault_obj.seal()
    assert vault_obj.unseal() == ("/testbed/.git",)
    assert (sandbox.host_path("/testbed/.git") / "HEAD").exists()


@pytest.mark.parametrize("host", ["", None, 5])
def test_probe_egress_rejects_bad_host(host: object, tmp_path: Path) -> None:
    """probe_egress refuses a host that is not a non-empty string."""
    sandbox, _ = _make(tmp_path)
    with pytest.raises(IntegrityRefusal, match="host: expected non-empty str"):
        probe_egress(sandbox, host=host)  # type: ignore[arg-type]


@pytest.mark.parametrize("timeout", [True, "5", None, 0, -1, float("inf"), float("nan")])
def test_probe_egress_rejects_bad_timeout(timeout: object, tmp_path: Path) -> None:
    """probe_egress refuses non-finite or non-positive timeouts and non-numbers."""
    sandbox, _ = _make(tmp_path)
    with pytest.raises(IntegrityRefusal, match="timeout_s: expected finite number > 0"):
        probe_egress(sandbox, timeout_s=timeout)  # type: ignore[arg-type]


def test_probe_egress_quotes_host(tmp_path: Path) -> None:
    """probe_egress shell-quotes the host in the getent half of the command."""
    sandbox, _ = _make(tmp_path)
    sandbox.scripted.append(ExecResult("", "", 1, False))
    probe_egress(sandbox, host="bad host; rm -rf /")
    assert "getent hosts 'bad host; rm -rf /'" in sandbox.commands[0]


def test_probe_egress_passes_timeout_to_exec(tmp_path: Path) -> None:
    """probe_egress forwards the validated timeout to sandbox.exec."""
    sandbox, _ = _make(tmp_path)
    sandbox.scripted.append(ExecResult("", "", 1, False))
    probe_egress(sandbox, timeout_s=2)
    assert "socket.create_connection(('example.com',443),5)" in sandbox.commands[0]


def test_require_no_egress_rejects_bad_host(tmp_path: Path) -> None:
    """require_no_egress propagates host validation refusals."""
    sandbox, _ = _make(tmp_path)
    with pytest.raises(IntegrityRefusal, match="host: expected non-empty str"):
        require_no_egress(sandbox, host="")


def test_require_no_egress_rejects_bad_timeout(tmp_path: Path) -> None:
    """require_no_egress propagates timeout validation refusals."""
    sandbox, _ = _make(tmp_path)
    with pytest.raises(IntegrityRefusal, match="timeout_s: expected finite number > 0"):
        require_no_egress(sandbox, timeout_s=0.0)


def test_assert_sealed_refuses_dangling_symlink(tmp_path: Path) -> None:
    """MUST_FIRE: a dangling symlink planted at a hidden path makes assert_sealed refuse."""

    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=("/tests",)), vault)
    vault_obj.seal()
    (sandbox.rootfs_dir / "tests").symlink_to(tmp_path / "nonexistent-target")
    with pytest.raises(IntegrityRefusal):
        vault_obj.assert_sealed()


def test_unseal_refuses_dangling_symlink(tmp_path: Path) -> None:
    """MUST_FIRE: a dangling symlink planted at a hidden path makes unseal refuse."""

    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=("/tests",)), vault)
    vault_obj.seal()
    (sandbox.rootfs_dir / "tests").symlink_to(tmp_path / "nonexistent-target")
    with pytest.raises(IntegrityRefusal):
        vault_obj.unseal()


def test_seal_refuses_symlinked_parent_dir(tmp_path: Path) -> None:
    """MUST_FIRE: a symlinked parent dir makes seal of a nested hidden path refuse."""

    sandbox, vault = _make(tmp_path)
    real = tmp_path / "real-testbed"
    real.mkdir()
    (real / ".git").mkdir()
    (real / ".git" / "HEAD").write_text("ref: refs/heads/main")
    (sandbox.rootfs_dir / "testbed").symlink_to(real)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=(), git_worktrees=("/testbed",)), vault)
    with pytest.raises(IntegrityRefusal):
        vault_obj.seal()


def test_unseal_refuses_symlinked_parent_dir(tmp_path: Path) -> None:
    """MUST_FIRE: a symlinked parent dir makes unseal of a nested hidden path refuse."""

    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=(), git_worktrees=("/testbed",)), vault)
    vault_obj.seal()
    real = tmp_path / "real-testbed"
    real.mkdir()
    shutil.rmtree(sandbox.rootfs_dir / "testbed")
    (sandbox.rootfs_dir / "testbed").symlink_to(real)
    with pytest.raises(IntegrityRefusal):
        vault_obj.unseal()


def test_seal_refuses_hidden_path_symlink(tmp_path: Path) -> None:
    """MUST_FIRE: a hidden path that is itself a symlink at seal time is refused."""

    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    shutil.rmtree(sandbox.rootfs_dir / "tests")
    (sandbox.rootfs_dir / "tests").symlink_to(sandbox.rootfs_dir / "solution")
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=("/tests",)), vault)
    with pytest.raises(IntegrityRefusal):
        vault_obj.seal()


def test_seal_refuses_container_path_with_dotdot(tmp_path: Path) -> None:
    """MUST_FIRE: a container path containing '..' parts is refused."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=("/tests/../solution",)), vault)
    with pytest.raises(IntegrityRefusal):
        vault_obj.seal()


def test_unseal_refuses_container_path_with_dotdot(tmp_path: Path) -> None:
    """MUST_FIRE: a container path containing '..' parts is refused at unseal."""
    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=("/tests/../solution",)), vault)
    with pytest.raises(IntegrityRefusal):
        vault_obj.unseal()


def test_unseal_refuses_when_dest_appears_during_rename(tmp_path: Path) -> None:
    """MUST_FIRE: a dest appearing between the check and the rename refuses."""
    import pathlib

    sandbox, vault = _make(tmp_path)
    _seed(sandbox)
    vault_obj = HiddenVault(sandbox, HiddenSpec(paths=("/tests",)), vault)
    vault_obj.seal()
    original_rename = pathlib.Path.rename

    def boom(self, target):
        raise OSError("dest appeared")

    pathlib.Path.rename = boom
    try:
        with pytest.raises(IntegrityRefusal):
            vault_obj.unseal()
    finally:
        pathlib.Path.rename = original_rename
