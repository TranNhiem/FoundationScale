"""#487: a run that cannot be attributed to a commit must SAY SO on the console.

T0-9 was measured, not assumed, and the measurement split the row. The manifest
already recorded an unversioned tree honestly -- `status=not_a_repository`,
`commit=null` -- so nothing was ever misattributed. What failed was loudness:
nothing outside `provenance/manifest.py` read `code.status`, so the run exited 0
over eleven lines without one of them mentioning that it was unattributable. The
truth was in the file and only in the file.

These two tests are a PAIR and neither is worth anything alone:

  the arm      an unversioned tree must produce the warning. On its own this
               passes for a warning that fires unconditionally, which would be
               the same defect wearing the opposite costume -- a line printed on
               every run is a line nobody reads.
  the control  a real checkout with a real commit must NOT produce it, while
               still producing the ordinary manifest line, so the silence is
               shown to be a decision rather than an absence of output.

The repository for the control is built here with ``git init`` rather than
borrowed from the checkout the suite is running inside: a CI runner's clone can
be shallow, detached, or (in a packaging job) not a repository at all, and a
control whose premise depends on the harness's own working directory measures
the harness.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from foundationscale.train import loop

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

WARNING = "UNATTRIBUTABLE RUN"


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60, check=True
    )


def _cfg(output_dir: Path) -> loop.TrainConfig:
    # The five genuinely required fields plus the cluster profile this dataclass
    # fails closed without. Nothing here loads a model: _emit_manifest is called
    # directly, which is the whole point -- the warning must not depend on having
    # got far enough to train.
    return loop.TrainConfig(
        model="a-model",
        dataset="a-dataset",
        output_dir=str(output_dir),
        nodes=1,
        gpus_per_node=1,
        profile_name="local-single-node",
    )


@requires_git
def test_unversioned_tree_warns_on_the_console(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No git metadata anywhere: the manifest is honest AND the run says so."""
    tree = tmp_path / "unpacked_tarball"
    tree.mkdir()
    # capture_code_provenance interrogates the CURRENT WORKING DIRECTORY, and
    # `git rev-parse` walks UPWARD, so a tree created inside an enclosing
    # repository is not unversioned at all. pytest's tmp_path is outside this
    # checkout, which is what makes the chdir meaningful rather than decorative.
    monkeypatch.chdir(tree)

    path = loop._emit_manifest(_cfg(tmp_path / "out"), stage="dry_run")

    assert path.exists(), "the manifest must still be written; the warning is in addition to it"
    out = capsys.readouterr().out
    assert WARNING in out, (
        "an unversioned tree exited without one line saying the run is unattributable; "
        f"this is exactly the #487 defect. Got:\n{out}"
    )
    assert "no commit was captured" in out
    assert "not_a_repository" in out, "the warning must name the status it is reporting"


@requires_git
def test_real_checkout_stays_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A checkout with a real commit must NOT warn -- the control for the above.

    Without this, a warning hard-coded to print on every run would satisfy the
    arm above and reintroduce the defect as noise instead of silence.
    """
    repo = tmp_path / "checkout"
    repo.mkdir()
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "probes@foundationscale.test")
    _run_git(repo, "config", "user.name", "Provenance Probe")
    (repo / "README.md").write_text("a tracked file, so HEAD is born\n", encoding="utf-8")
    _run_git(repo, "add", "README.md")
    _run_git(repo, "commit", "-q", "-m", "initial")
    monkeypatch.chdir(repo)

    loop._emit_manifest(_cfg(tmp_path / "out"), stage="dry_run")

    out = capsys.readouterr().out
    assert "run manifest" in out, (
        "the control produced no manifest line at all, so its silence about "
        "attribution is an absence of output rather than a decision"
    )
    assert WARNING not in out, (
        "a run with a real commit warned that it could not be attributed; a warning "
        f"that fires unconditionally is not read. Got:\n{out}"
    )
