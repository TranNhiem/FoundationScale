"""Probe-accounting tests for capture_code_provenance.

Controls for a detector-shaped code path, per doctrine 3: for every defect the
capture must now expose (MUST_FIRE) there is a paired healthy input whose verdict
must not have moved (MUST_PASS). All fixtures are real git repositories built in
tmp_path; no probe output is ever mocked, because the defects under test live
precisely in the gap between what git emitted and what the caller pretended it
emitted.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from foundationscale.provenance import manifest as m

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git binary required")


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [
            GIT,
            "-c",
            "user.email=audit@example.invalid",
            "-c",
            "user.name=Audit",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(repo),
            *args,
        ],
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {args[0]} failed: {proc.stderr.strip()}")
    return proc


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    return r


def _tracked_file(repo: Path, rel: str, text: str = "x = 1\n") -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    _git(repo, "add", "--", rel)
    return p


def _commit(repo: Path, msg: str = "initial") -> None:
    _git(repo, "commit", "-q", "-m", msg)


def _findings_for(prov: m.CodeProvenance) -> tuple[str, ...]:
    """Wrap the capture in a minimal RunManifest so findings can be asserted.

    allowlist is non-empty so the unrelated 'no allowlist declared' finding does
    not pollute substring assertions.
    """
    man = m.RunManifest(
        run_id="capture-test",
        attempt=1,
        code=prov,
        config={},
        environment=m.CapturedEnvironment(
            allowlist=("FOUNDATIONSCALE_",), values={}, source_var_count=12
        ),
        topology=m.Topology(
            nodes=1,
            gpus_per_node=1,
            tensor_parallel=1,
            pipeline_parallel=1,
            data_parallel=1,
        ),
    )
    return man.findings


# ---------------------------------------------------------------------------
# Finding 1 — failed probes must never read as CLEAN
# ---------------------------------------------------------------------------


def test_finding1_must_fire_outside_repo_pathspec_cannot_read_clean(
    repo: Path, tmp_path: Path
) -> None:
    """MUST_FIRE, [fails-before: VERDICT].

    The hunt report's exact trace: a capture path that drifts outside the
    repository (an absolute $VAR path). Pre-patch every probe exits rc=128, the
    four optimistic zeros agree with each other, and the record says CLEAN.
    """
    _tracked_file(repo, "src/train.py")
    _commit(repo)
    outside = tmp_path / "home" / "u" / "code"
    outside.mkdir(parents=True)
    (outside / "runner.py").write_text("print('this executes')\n")

    prov = m.capture_code_provenance(repo, [str(outside)])

    assert prov.status is m.CaptureStatus.NOT_CAPTURED  # pre-patch: CLEAN
    assert prov.paths[0].status is m.PathStatus.NOT_CAPTURED  # pre-patch: CLEAN
    assert prov.paths[0].exists is True  # the path WAS there; git could not address it
    assert {"status", "ls-files", "diff", "diff --name-only"} == set(prov.paths[0].failed_probes)
    assert any("NOT CAPTURED" in f and "of 4 probes" in f for f in _findings_for(prov))


def test_finding1_must_fire_corrupt_index_not_clean(repo: Path) -> None:
    """MUST_FIRE, [fails-before: VERDICT].

    rev-parse reads refs (succeeds); status/ls-files/diff need the index (fail).
    Pre-patch this yields CLEAN over a possibly-dirty tree.
    """
    _tracked_file(repo, "src/train.py")
    _commit(repo)
    (repo / ".git" / "index").write_bytes(b"this is not a git index")

    prov = m.capture_code_provenance(repo)

    assert prov.status is m.CaptureStatus.NOT_CAPTURED  # pre-patch: CLEAN
    assert prov.probe_failed is True
    assert prov.paths[0].failed_probes  # per-path altitude also failed
    findings = _findings_for(prov)
    assert any("repo-wide git status probe" in f for f in findings)
    assert any("of 4 probes" in f for f in findings)


def test_finding1_must_pass_clean_tree_stays_clean(repo: Path) -> None:
    """MUST_PASS, [fails-before: FIELD].

    Verdict-level assertions hold on BOTH trees by design — the guard that the
    repair never touches the healthy path. The probe-accounting fields are new
    surface, so this goes red pre-patch via AttributeError and green after.
    """
    _tracked_file(repo, "src/train.py")
    _commit(repo)

    prov = m.capture_code_provenance(repo)  # default scope "."

    assert prov.status is m.CaptureStatus.CLEAN
    assert prov.commit is not None
    assert prov.probe_failed is False  # new field — AttributeError pre-patch
    assert prov.paths[0].status is m.PathStatus.CLEAN
    assert prov.paths[0].failed_probes == ()
    assert prov.paths[0].ignored_files == 0
    assert _findings_for(prov) == ()


# ---------------------------------------------------------------------------
# Finding 3 — gitignored code inside the capture scope is not invisible
# ---------------------------------------------------------------------------


def test_finding3_must_fire_gitignored_code_in_scope(repo: Path) -> None:
    """MUST_FIRE, [fails-before: VERDICT].

    Every probe exits 0 here — this is Finding 3's distinction from Finding 1:
    no failure propagates, the blind spot is in the argv. Pre-patch the path
    reads CLEAN with exists=True over bytes that execute.
    """
    _tracked_file(repo, ".gitignore", "gen/\n")
    _tracked_file(repo, "src/train.py")
    _commit(repo)
    (repo / "gen").mkdir()
    (repo / "gen" / "runner.py").write_text("print('generated, executes')\n")

    prov = m.capture_code_provenance(repo, ["gen"])

    assert prov.paths[0].status is m.PathStatus.NOT_CAPTURED  # pre-patch: CLEAN
    assert prov.paths[0].ignored_files >= 1  # porcelain may collapse dirs; >=1 is the floor
    assert prov.paths[0].failed_probes == ()  # probes SUCCEEDED; policy veto, not error path
    assert prov.status is m.CaptureStatus.NOT_CAPTURED
    assert any("ignored" in f and "NOT CAPTURED" in f for f in _findings_for(prov))


def test_finding3_must_fire_gitignored_code_default_scope(repo: Path) -> None:
    """MUST_FIRE, [fails-before: VERDICT]. Same defect through the default scope."""
    _tracked_file(repo, ".gitignore", "gen/\n")
    _tracked_file(repo, "src/train.py")
    _commit(repo)
    (repo / "gen").mkdir()
    (repo / "gen" / "runner.py").write_text("x = 1\n")

    prov = m.capture_code_provenance(repo)  # default "."

    assert prov.status is m.CaptureStatus.NOT_CAPTURED  # pre-patch: CLEAN
    assert prov.paths[0].ignored_files >= 1


def test_finding3_must_pass_gitignored_build_artifacts_outside_scope(repo: Path) -> None:
    """MUST_PASS, [fails-before: FIELD].

    Doctrine 5's symmetric guard, pinned as executable code: an ignored build/
    tree OUTSIDE the declared capture scope must NOT mint a false NOT_CAPTURED.
    The scope declaration is the integrator's stated contract; this patch fixes
    a false pass, it does not invent a false failure. Verdict assertions hold on
    both trees; the field assertions are the new surface.
    """
    _tracked_file(repo, ".gitignore", "build/\n")
    _tracked_file(repo, "src/train.py")
    _commit(repo)
    (repo / "build").mkdir()
    (repo / "build" / "out.bin").write_bytes(b"\x00" * 64)

    prov = m.capture_code_provenance(repo, ["src"])

    assert prov.status is m.CaptureStatus.CLEAN  # held before AND after — the point
    assert prov.paths[0].status is m.PathStatus.CLEAN
    assert prov.paths[0].ignored_files == 0  # new field — AttributeError pre-patch
    assert prov.probe_failed is False


# ---------------------------------------------------------------------------
# Finding 6 — every blocking verdict must name its reason
# ---------------------------------------------------------------------------


def test_finding6_must_fire_dirty_outside_scope_names_reason(repo: Path) -> None:
    """MUST_FIRE, [fails-before: FINDINGS].

    The verdict was already NOT_CAPTURED on the old tree (fail-closed, which is
    why this finding is MINOR); the defect is the EMPTY explanation. The
    findings assertion is the pin; the verdict assertion is the guard that the
    patch changed the narration, not the polarity.
    """
    _tracked_file(repo, "src/train.py")
    _tracked_file(repo, "other/notes.txt")
    _commit(repo)
    (repo / "other" / "notes.txt").write_text("edited outside the scope\n")

    prov = m.capture_code_provenance(repo, ["src"])

    assert prov.status is m.CaptureStatus.NOT_CAPTURED  # identical on both trees
    assert prov.diff_bytes == 0
    findings = _findings_for(prov)
    assert findings != ()  # pre-patch: empty — the defect
    assert any("OUTSIDE every captured path" in f for f in findings)


def test_finding6_must_fire_unborn_head_names_reason(repo: Path) -> None:
    """MUST_FIRE, [fails-before: FINDINGS].

    git init, no commits, empty tree: per-path probes succeed with zeros, every
    row lands CLEAN, the rollup forces NOT_CAPTURED via commit is None, and
    _derive_findings says nothing at all on the old tree.
    """
    prov = m.capture_code_provenance(repo)

    assert prov.status is m.CaptureStatus.NOT_CAPTURED  # identical on both trees
    assert prov.commit is None
    assert prov.probe_failed is False  # no probe failed; the commit is absent
    assert any("no resolvable HEAD commit" in f for f in _findings_for(prov))


def test_finding6_must_pass_captured_diff_still_captured_and_silent(repo: Path) -> None:
    """MUST_PASS, [fails-before: FIELD].

    A healthy CAPTURED: tracked file modified inside the declared scope. Verdict
    and silence both hold on both trees — the new narration must not fire
    spuriously on a record that is honestly captured.
    """
    f = _tracked_file(repo, "src/train.py")
    _commit(repo)
    f.write_text("x = 2  # edited in scope\n")

    prov = m.capture_code_provenance(repo, ["src"])

    assert prov.status is m.CaptureStatus.CAPTURED  # both trees
    assert prov.paths[0].status is m.PathStatus.CAPTURED
    assert prov.diff_bytes > 0
    assert prov.probe_failed is False  # new field — AttributeError pre-patch
    assert prov.paths[0].failed_probes == ()
    assert _findings_for(prov) == ()


# ---------------------------------------------------------------------------
# Wire accounting — the new fields must round-trip without self-flagging
# ---------------------------------------------------------------------------


def test_new_fields_round_trip_through_store(repo: Path, tmp_path: Path) -> None:
    """MUST_FIRE/MUST_PASS pair, [fails-before: VERDICT].

    Saves and reloads one probe-failure manifest and one ignored-in-scope
    manifest. Asserts (a) the fields survive the wire, and (b) this reader does
    not treat its OWN new keys as unknown — Edit B exists precisely so no
    'loader ignored unknown keys' finding appears on a self-written record.
    """
    _tracked_file(repo, ".gitignore", "gen/\n")
    _tracked_file(repo, "src/train.py")
    _commit(repo)
    (repo / "gen").mkdir()
    (repo / "gen" / "runner.py").write_text("x = 1\n")
    prov = m.capture_code_provenance(repo, ["gen"])  # pre-patch: CLEAN → red
    assert prov.status is m.CaptureStatus.NOT_CAPTURED

    def _manifest(p: m.CodeProvenance, attempt: int) -> m.RunManifest:
        return m.RunManifest(
            run_id="wire-test",
            attempt=attempt,
            code=p,
            config={},
            environment=m.CapturedEnvironment(
                allowlist=("FOUNDATIONSCALE_",), values={}, source_var_count=4
            ),
            topology=m.Topology(
                nodes=1,
                gpus_per_node=1,
                tensor_parallel=1,
                pipeline_parallel=1,
                data_parallel=1,
            ),
        )

    store = m.ManifestStore(tmp_path / "store")
    store.save(_manifest(prov, 1))
    loaded = store.latest("wire-test")
    assert loaded.code.paths[0].ignored_files == prov.paths[0].ignored_files
    assert loaded.code.paths[0].failed_probes == prov.paths[0].failed_probes
    assert loaded.code.probe_failed is False
    assert not any("loader ignored unknown" in f for f in loaded.findings)

    # Second attempt: corrupt the index, capture again, round-trip the
    # probe_failed leg.
    (repo / ".git" / "index").write_bytes(b"corrupt")
    prov2 = m.capture_code_provenance(repo)
    assert prov2.probe_failed is True  # pre-patch: AttributeError / CLEAN
    store.save(_manifest(prov2, 2))
    loaded2 = m.load(store._run_dir("wire-test") / "attempt-0002.json")
    assert loaded2.code.probe_failed is True
    assert any("repo-wide git status probe" in f for f in loaded2.findings)
