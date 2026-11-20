"""ARL-G14 DiffGuard: guarded-file fingerprints and the SweReward tamper policy."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from foundationscale.agentic_rl.contracts import Termination
from foundationscale.agentic_rl.envs.diff_guard import changed_paths, fingerprint
from foundationscale.agentic_rl.harness.base import EpisodeOutcome, EpisodeTask
from foundationscale.agentic_rl.rewards.swe import SweReward


def _rootfs(tmp_path: Path) -> Path:
    root = tmp_path / "rootfs"
    (root / "repo" / "pkg").mkdir(parents=True)
    (root / "repo" / "pkg" / "mod.py").write_text("x = 1\n")
    (root / "repo" / "setup.cfg").write_text("[tool:pytest]\n")
    site = root / "usr" / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True)
    (site / "dist.pth").write_text("import os\n")
    return root


def test_fingerprint_covers_guarded_names_and_site_pth_but_not_source(tmp_path: Path) -> None:
    root = _rootfs(tmp_path)
    (root / "repo" / ".git").mkdir()
    (root / "repo" / ".git" / "conftest.py").write_text("ignored\n")  # VCS dirs are skipped
    (root / "repo" / "notes.pth").write_text("not site\n")  # *.pth only counts in site dirs
    prints = fingerprint(root, "/repo")
    assert sorted(prints) == [
        "/repo/setup.cfg",
        "/usr/lib/python3.11/site-packages/dist.pth",
    ]


def test_changed_paths_names_added_removed_and_edited_guarded_files(tmp_path: Path) -> None:
    root = _rootfs(tmp_path)
    before = fingerprint(root, "/repo")
    (root / "repo" / "pkg" / "mod.py").write_text("x = 2\n")  # the fix itself: not guarded
    (root / "repo" / "conftest.py").write_text("def pytest_runtest_makereport(): ...\n")
    (root / "repo" / "setup.cfg").write_text("[tool:pytest]\naddopts = -p no:x\n")
    (root / "usr" / "lib" / "python3.11" / "site-packages" / "dist.pth").unlink()
    after = fingerprint(root, "/repo")
    assert changed_paths(before, after) == [
        "/repo/conftest.py",
        "/repo/setup.cfg",
        "/usr/lib/python3.11/site-packages/dist.pth",
    ]
    assert changed_paths(after, after) == []


def test_symlinks_are_recorded_by_target_and_never_followed(tmp_path: Path) -> None:
    root = _rootfs(tmp_path)
    outside = tmp_path / "outside_secret.py"
    outside.write_text("secret\n")
    (root / "repo" / "conftest.py").symlink_to(outside)
    prints = fingerprint(root, "/repo")
    assert prints["/repo/conftest.py"] == f"symlink:{outside}"


_TASK = EpisodeTask(
    uid="t",
    session_id="t#0",
    messages=({"role": "user", "content": "fix"},),
    metadata={"fail_to_pass": '["a"]', "pass_to_pass": "[]"},
)


def _outcome(**meta: str) -> EpisodeOutcome:
    base = {"swe_status": "ok", "swe_tests": json.dumps({"a": "passed"})}
    return EpisodeOutcome(
        trajectory_turns=(),
        prompt_turns=(),
        termination=Termination.STOP,
        submitted_answer=None,
        abstention_reason=None,
        metadata={**base, **meta},
    )


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({"swe_guard": "measured", "swe_tampered": "[]"}, (1.0, "resolved")),
        (
            {"swe_guard": "measured", "swe_tampered": '["/repo/conftest.py"]'},
            (0.0, "test_harness_tampering"),
        ),
        ({"swe_guard": "unmeasured"}, (None, "tamper_guard_unmeasured")),
        ({}, (None, "tamper_guard_unmeasured")),
    ],
)
def test_tamper_zero_policy(meta: dict[str, str], expected: tuple[float | None, str]) -> None:
    verdict = asyncio.run(SweReward(tamper="zero").score(_TASK, _outcome(**meta)))
    assert (verdict.value, verdict.reason) == expected


def test_tamper_report_policy_keeps_scoring_unchanged() -> None:
    tampered = _outcome(swe_guard="measured", swe_tampered='["/repo/conftest.py"]')
    verdict = asyncio.run(SweReward().score(_TASK, tampered))
    assert (verdict.value, verdict.reason) == (1.0, "resolved")


def test_unknown_tamper_policy_is_refused() -> None:
    with pytest.raises(ValueError, match="tamper"):
        SweReward(tamper="warn")
