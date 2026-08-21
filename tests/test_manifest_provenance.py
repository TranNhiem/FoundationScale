"""Provenance-capture tests for ``foundationscale.provenance.manifest``.

Why this file exists
--------------------
Three audited incidents set the bar for these tests, and each test names the one it
prevents from recurring:

* The predecessor captured provenance as ``[ -r capsule ] && source capsule || true``
  and reached 0% provenance coverage against 77 result dirs without anyone knowing —
  the failure was designed to be invisible. Capture outcome is therefore asserted as
  an explicit status enum, never inferred from "no exception was raised".
* ``uncommitted.patch`` was 0 bytes over a directory containing *zero git-tracked
  files*, and that empty capture was reported as "clean". Both shapes of that lie are
  reproduced on a synthetic repo in ``tmp_path``: untracked living code, and
  modified-tracked files with a 0-byte diff (unborn ``HEAD``). Each must assert
  :class:`~foundationscale.provenance.manifest.PathStatus`.NOT_CAPTURED and roll up to
  ``CaptureStatus.NOT_CAPTURED`` — never CLEAN.
* 24 runs with byte-identical argv split 12/12 between two objectives because the
  deciding environment variable was recorded nowhere. ``capture_environment`` and
  ``ConfigResolver`` are the mechanisms that would have answered it.

This file tests the component whose own thesis is "a green report is the default
output of a broken check", so every negative assertion is paired with a positive
control proving the detector can fire, and every collection that can be empty —
the allowlist, the repository, the capture scope — is exercised as an empty input.
``all([]) is True`` shipped in the audit's own verifier; it does not ship here.

No network, no cluster, no scheduler. Tests that need a real git repository are
skipped when no ``git`` binary is on PATH; every ``NOT_A_REPOSITORY`` path is tested
with plain directories so it runs regardless.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from foundationscale.provenance import manifest

requires_git = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="no git binary on PATH: this test captures provenance from a real repository",
)

_SUBMIT_ENV: dict[str, str] = {
    "NCCL_ALGO": "ring",
    "SLURM_JOB_ID": "117",
    "FOUNDATIONSCALE_OBJECTIVE": "moe",  # the switch that split 24 runs 12/12
    "HOME": "/home/submitter",  # submit-machine noise that srun --export=ALL would leak
    "PATH": "/usr/bin",
}


# ---------------------------------------------------------------------------
# Synthetic git infrastructure
# ---------------------------------------------------------------------------


def _run_git(repo: Path, *args: str) -> None:
    """Run a git command against ``repo`` and fail the test loudly on any error."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr.strip()}"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real git repository in ``tmp_path`` with an identity configured for commits."""
    root = tmp_path / "repo"
    root.mkdir()
    _run_git(root, "init", "-q")
    _run_git(root, "config", "user.email", "probes@foundationscale.test")
    _run_git(root, "config", "user.name", "Provenance Probe")
    return root


def _commit_file(repo: Path, rel: str, content: str) -> Path:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _run_git(repo, "add", rel)
    _run_git(repo, "commit", "-q", "-m", f"add {rel}")
    return path


# ---------------------------------------------------------------------------
# Manifest fixture helpers (manual constructors — no git required)
# ---------------------------------------------------------------------------

_TOPOLOGY = manifest.Topology(
    nodes=1,
    gpus_per_node=8,
    tensor_parallel=2,
    pipeline_parallel=1,
    data_parallel=4,
)

_DEFAULT_ENV = manifest.CapturedEnvironment(
    allowlist=("FOUNDATIONSCALE_",),
    values={},
    source_var_count=0,
)


def _clean_code() -> manifest.CodeProvenance:
    return manifest.CodeProvenance(
        status=manifest.CaptureStatus.CLEAN,
        root="/virtual/repo",
        commit="0" * 40,
        dirty_files=0,
        untracked_files=0,
        diff_sha256=None,
        diff_bytes=0,
        paths=(),
        entrypoint=None,
        entrypoint_captured=None,
    )


def _manifest(
    *,
    code: manifest.CodeProvenance | None = None,
    config: Mapping[str, manifest.EffectiveValue] | None = None,
    environment: manifest.CapturedEnvironment | None = None,
) -> manifest.RunManifest:
    return manifest.RunManifest(
        run_id="probe",
        attempt=1,
        code=code if code is not None else _clean_code(),
        config=dict(config) if config is not None else {},
        environment=environment if environment is not None else _DEFAULT_ENV,
        topology=_TOPOLOGY,
    )


# ---------------------------------------------------------------------------
# NOT_A_REPOSITORY — deliberately without any git repository (or git binary)
# ---------------------------------------------------------------------------


def test_nonexistent_root_is_not_a_repository(tmp_path: Path) -> None:
    prov = manifest.capture_code_provenance(tmp_path / "was-never-created")

    assert prov.status is manifest.CaptureStatus.NOT_A_REPOSITORY
    assert prov.commit is None
    assert prov.diff_sha256 is None
    assert prov.diff_bytes == 0
    assert prov.paths == ()
    assert prov.root is not None


def test_plain_directory_is_not_a_repository(tmp_path: Path) -> None:
    """Code on NFS/local disk with no .git must not look like a clean checkout."""
    workdir = tmp_path / "nfs_code"
    (workdir / "ops").mkdir(parents=True)
    launcher = workdir / "ops" / "launch.sh"
    launcher.write_text("#!/bin/sh\nexec python train.py\n", encoding="utf-8")

    prov = manifest.capture_code_provenance(workdir, entrypoint=launcher)

    assert prov.status is manifest.CaptureStatus.NOT_A_REPOSITORY
    assert prov.commit is None
    assert prov.paths == ()
    # Containment is geometry, not capture: True here must never be read as "snapshotted".
    assert prov.entrypoint_captured is True

    record = _manifest(code=prov)
    assert any("not a git repository" in finding for finding in record.findings)


# ---------------------------------------------------------------------------
# Entrypoint containment — the launcher that actually ran escaped every snapshot
# ---------------------------------------------------------------------------


def test_entrypoint_outside_root_is_named_as_uncaptured(tmp_path: Path) -> None:
    """Incident shape: the real launcher lived under a different root entirely."""
    code_root = tmp_path / "code"
    code_root.mkdir()
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    launcher = ops_dir / "repro_capsule.sh"
    launcher.write_text("echo launching the real job\n", encoding="utf-8")

    prov = manifest.capture_code_provenance(code_root, entrypoint=launcher)

    assert prov.entrypoint == str(launcher)
    assert prov.entrypoint_captured is False

    record = _manifest(code=prov)
    assert any("lies outside the provenance root" in finding for finding in record.findings)


def test_entrypoint_inside_root_is_the_containment_control(tmp_path: Path) -> None:
    """Positive control: the False claim above means something only if True is reachable."""
    code_root = tmp_path / "code"
    (code_root / "bin").mkdir(parents=True)
    launcher = code_root / "bin" / "launch.sh"
    launcher.write_text("echo launching\n", encoding="utf-8")

    prov = manifest.capture_code_provenance(code_root, entrypoint=launcher)

    assert prov.entrypoint_captured is True
    assert not any(
        "lies outside the provenance root" in finding for finding in _manifest(code=prov).findings
    )


# ---------------------------------------------------------------------------
# capture_code_provenance on a real repository (skipped without git)
# ---------------------------------------------------------------------------


@requires_git
def test_fully_committed_tree_reports_clean_with_a_real_commit(repo: Path) -> None:
    """Positive control: an honest CLEAN exists, so NOT_CAPTURED asserts something."""
    _commit_file(repo, "src/train.py", "print('v1')\n")

    prov = manifest.capture_code_provenance(repo)

    assert prov.status is manifest.CaptureStatus.CLEAN
    assert prov.commit is not None
    assert len(prov.commit) == 40
    assert prov.dirty_files == 0
    assert [p.path for p in prov.paths] == ["."]
    assert prov.paths[0].status is manifest.PathStatus.CLEAN
    assert prov.paths[0].tracked_files == 1
    # An honest zero: nothing differs, so the diff is empty. The NOT_CAPTURED tests
    # below also carry diff_bytes == 0 — the byte count alone cannot tell these apart,
    # which is exactly why the incident's verifier was fooled and this gate is not.
    assert prov.diff_bytes == 0


@requires_git
def test_modified_tracked_file_reports_captured_with_nonzero_diff(repo: Path) -> None:
    """The second positive control: the CAPTURED pathway demonstrably fires."""
    tracked = _commit_file(repo, "src/train.py", "print('v1')\n")
    tracked.write_text("print('v2: trust region actually engaged')\n", encoding="utf-8")

    prov = manifest.capture_code_provenance(repo, diff_paths=("src",))

    (cover,) = prov.paths
    assert cover.status is manifest.PathStatus.CAPTURED
    assert cover.modified_tracked_files == 1
    assert cover.untracked_files == 0
    assert cover.captured_files == 1
    assert cover.captured_bytes > 0

    assert prov.status is manifest.CaptureStatus.CAPTURED
    assert prov.dirty_files == 1
    assert prov.diff_bytes == cover.captured_bytes
    assert prov.diff_sha256 is not None


@requires_git
def test_living_untracked_tree_is_not_captured_and_never_clean(repo: Path) -> None:
    """Reproduces the audited 0-byte ``uncommitted.patch`` defect.

    The code that will execute sits in a directory with zero git-tracked files.
    ``git diff HEAD -- <path>`` *succeeds* and returns zero bytes — a capture that
    "worked" and stored nothing. Byte count alone looks identical to CLEAN; the
    untracked-file count is the only honest signal, and it must flip the status.
    """
    _commit_file(repo, "README.md", "seed\n")
    living = repo / "sdpo_impl"
    living.mkdir()
    (living / "train.py").write_text("print('the code that actually ran')\n", encoding="utf-8")

    prov = manifest.capture_code_provenance(repo, diff_paths=("sdpo_impl",))

    (cover,) = prov.paths
    assert cover.tracked_files == 0
    assert cover.untracked_files == 1
    assert cover.captured_bytes == 0
    assert cover.status is manifest.PathStatus.NOT_CAPTURED
    assert prov.status is manifest.CaptureStatus.NOT_CAPTURED


@requires_git
def test_staged_changes_with_unborn_head_produce_zero_bytes_and_block(repo: Path) -> None:
    """Modified-tracked with zero commits: ``git diff HEAD`` fails, yielding 0 bytes.

    Porcelain reports the staged file as modified, but there is no HEAD to diff
    against, so the diff is empty. Zero bytes over modified tracked files is the
    second half of the incident pair and must also never read as clean.
    """
    staged = repo / "train.py"
    staged.write_text("print('never committed')\n", encoding="utf-8")
    _run_git(repo, "add", "train.py")

    prov = manifest.capture_code_provenance(repo, diff_paths=("train.py",))

    (cover,) = prov.paths
    assert cover.modified_tracked_files == 1
    assert cover.untracked_files == 0
    assert cover.captured_bytes == 0
    assert cover.status is manifest.PathStatus.NOT_CAPTURED
    assert prov.status is manifest.CaptureStatus.NOT_CAPTURED


@requires_git
def test_one_bad_path_taints_the_rollup_despite_a_captured_sibling(repo: Path) -> None:
    """A clean-looking sibling must not launder a path the diff cannot see."""
    tracked = _commit_file(repo, "src/train.py", "print('v1')\n")
    tracked.write_text("print('v2')\n", encoding="utf-8")
    living = repo / "fresh_impl"
    living.mkdir()
    (living / "objective.py").write_text("LOSS = 'the other one'\n", encoding="utf-8")

    prov = manifest.capture_code_provenance(repo, diff_paths=("src", "fresh_impl"))

    statuses = {p.path: p.status for p in prov.paths}
    assert statuses["src"] is manifest.PathStatus.CAPTURED  # capture is working here
    assert statuses["fresh_impl"] is manifest.PathStatus.NOT_CAPTURED
    # Bytes WERE captured overall — and the record must still not claim CAPTURED.
    assert prov.diff_bytes > 0
    assert prov.status is manifest.CaptureStatus.NOT_CAPTURED


@requires_git
def test_capture_scope_naming_a_path_that_never_existed_is_flagged(repo: Path) -> None:
    """The audited capture variable expanded to ``$SDPO``: a subtree nothing edited."""
    _commit_file(repo, "src/train.py", "print('real code')\n")

    prov = manifest.capture_code_provenance(repo, diff_paths=("SDPO",))

    (cover,) = prov.paths
    assert cover.exists is False
    assert cover.status is manifest.PathStatus.NO_SUCH_PATH
    # A scope entry that cannot exist is path drift, not a clean capture of nothing.
    assert prov.status is manifest.CaptureStatus.NOT_CAPTURED


@requires_git
def test_diff_hash_tracks_the_actual_bytes_being_captured(repo: Path) -> None:
    """The commit-hash-plus-diff claim must bind to the bytes, not impersonate them."""
    tracked = _commit_file(repo, "src/train.py", "print('v1')\n")
    tracked.write_text("print('v2')\n", encoding="utf-8")
    first = manifest.capture_code_provenance(repo, diff_paths=("src",))
    tracked.write_text("print('v3 — different bytes entirely')\n", encoding="utf-8")
    second = manifest.capture_code_provenance(repo, diff_paths=("src",))

    assert first.diff_sha256 is not None
    assert second.diff_sha256 is not None
    assert first.diff_sha256 != second.diff_sha256
    assert first.diff_bytes != second.diff_bytes


# ---------------------------------------------------------------------------
# capture_environment — the unrecorded variable that split 24 runs 12/12
# ---------------------------------------------------------------------------


def test_default_allowlist_captures_incident_variables_with_qualified_counts() -> None:
    env = manifest.capture_environment(environ=_SUBMIT_ENV)

    assert env.values == {
        "NCCL_ALGO": "ring",
        "SLURM_JOB_ID": "117",
        "FOUNDATIONSCALE_OBJECTIVE": "moe",
    }
    assert env.allowlist == manifest.DEFAULT_ENV_PREFIXES
    # N of M, stated: 3 captured out of 5 present, 2 deliberately not examined.
    assert env.captured == 3
    assert env.source_var_count == 5
    assert env.omitted == 2


def test_custom_allowlist_records_what_was_excluded() -> None:
    """The allowlist is part of the record: a reader must see what was never looked at."""
    env = manifest.capture_environment(prefixes=("SLURM_",), environ=_SUBMIT_ENV)

    assert env.values == {"SLURM_JOB_ID": "117"}
    assert env.allowlist == ("SLURM_",)

    payload = env.to_dict()
    assert payload["allowlist"] == ["SLURM_"]
    assert payload["source_var_count"] == 5
    assert manifest.CapturedEnvironment.from_dict(payload) == env


def test_empty_allowlist_surfaces_as_a_manifest_finding_not_a_clean_zero() -> None:
    """``prefixes=()`` filters everything out: ``any([])`` is False, capture is empty.

    That empty capture is honest arithmetic ("0 of 5"), but it must not launder into
    a manifest that reads as clean — the scope itself is the defect, so the manifest
    must carry a finding naming it.
    """
    env = manifest.capture_environment(prefixes=(), environ=_SUBMIT_ENV)

    assert env.values == {}
    assert env.captured == 0
    assert env.source_var_count == 5  # 0 of 5 is a qualified count; it proves nothing by itself

    record = _manifest(environment=env)
    assert any("no allowlist" in finding for finding in record.findings), record.findings

    # Positive control: the same pool under a declared scope carries no such finding,
    # so the finding names the empty scope rather than distrusting env capture itself.
    scoped = manifest.capture_environment(prefixes=("NCCL_",), environ=_SUBMIT_ENV)
    assert not any("no allowlist" in finding for finding in _manifest(environment=scoped).findings)


# ---------------------------------------------------------------------------
# ConfigResolver — effective values and the provenance of their resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["cli", "default", "config:sweep/base.yaml#optimizer.lr"])
def test_record_effective_accepts_cli_default_and_config_sources(source: str) -> None:
    resolver = manifest.ConfigResolver(environ={})

    recorded = resolver.record_effective("optimizer.lr", "3e-05", source)

    assert recorded.value == "3e-05"
    assert recorded.source == source
    assert recorded.findings == ()


def test_record_effective_accepts_env_source_and_stores_the_live_value() -> None:
    resolver = manifest.ConfigResolver(environ={"FOUNDATIONSCALE_OBJECTIVE": "moe"})

    recorded = resolver.record_effective("objective", "moe", "env:FOUNDATIONSCALE_OBJECTIVE")

    assert recorded.env_value == "moe"
    assert recorded.findings == ()


@pytest.mark.parametrize(
    "source",
    [
        "env",
        "ENV:X",
        "env:1LEADING_DIGIT",
        "env:HAS-DASH",
        "config:no_anchor",
        "config:#no_key",
        "default-ish",
        "cli ",
        "",
        "magic",
    ],
)
def test_record_effective_rejects_sources_outside_the_grammar(source: str) -> None:
    """A free-text source is the 18-key string bag coming back; reject, half-record nothing."""
    resolver = manifest.ConfigResolver(environ={})

    with pytest.raises(ValueError, match="invalid source"):
        resolver.record_effective("optimizer.lr", "0.1", source)

    assert resolver.freeze() == {}  # the acceptance tests prove this dict CAN hold entries


def test_env_shadowed_default_is_a_finding_not_a_footnote() -> None:
    """The incident exactly: variable set in env, but the run resolved the default."""
    resolver = manifest.ConfigResolver(environ={"OBJECTIVE_SWITCH": "moe"})

    recorded = resolver.record_effective("OBJECTIVE_SWITCH", "dense", "default")

    assert recorded.value == "dense"
    assert recorded.env_value == "moe"
    assert any("env-shadowed default" in finding for finding in recorded.findings)

    # ...and the finding must surface at manifest level, where the 24-run diff needed it.
    record = _manifest(config=resolver.freeze())
    assert any("env-shadowed default" in finding for finding in record.findings)


def test_unshadowed_default_carries_no_finding() -> None:
    """Positive control: the shadow detector can be silent, so its firing means something."""
    resolver = manifest.ConfigResolver(environ={})

    recorded = resolver.record_effective("OBJECTIVE_SWITCH", "dense", "default")

    assert recorded.env_value is None
    assert recorded.findings == ()


def test_env_source_naming_a_variable_that_is_not_set_cannot_be_corroborated() -> None:
    resolver = manifest.ConfigResolver(environ={"SOME_OTHER_VAR": "x"})

    recorded = resolver.record_effective("objective", "moe", "env:GHOST_SWITCH")

    assert recorded.value == "moe"
    assert recorded.env_value is None
    assert any("unverifiable env source" in finding for finding in recorded.findings)


def test_env_source_value_that_moved_since_resolution_is_flagged_as_drift() -> None:
    resolver = manifest.ConfigResolver(environ={"SWITCH": "moe"})

    recorded = resolver.record_effective("objective", "dense", "env:SWITCH")

    assert recorded.env_value == "moe"
    assert any("env-source drift" in finding for finding in recorded.findings)


def test_env_source_is_quiet_when_value_and_variable_agree() -> None:
    """Positive control: drift requires a mismatch; agreement records cleanly."""
    resolver = manifest.ConfigResolver(environ={"SWITCH": "moe"})

    recorded = resolver.record_effective("objective", "moe", "env:SWITCH")

    assert recorded.env_value == "moe"
    assert recorded.findings == ()


def test_resolver_snapshots_environ_at_construction() -> None:
    """Mutating the caller's mapping afterwards must not retro-edit the record."""
    live: dict[str, str] = {"SWITCH": "dense"}
    resolver = manifest.ConfigResolver(environ=live)
    live["SWITCH"] = "moe"

    recorded = resolver.record_effective("objective", "dense", "env:SWITCH")

    assert recorded.env_value == "dense"
    assert recorded.findings == ()


def test_recording_a_key_twice_raises_instead_of_overwriting_the_first_resolution() -> None:
    """Overwrite-in-place destroyed 27 manifests on disk; it does not survive in memory either."""
    resolver = manifest.ConfigResolver(environ={})
    first = resolver.record_effective("lr", "0.1", "cli")

    with pytest.raises(ValueError, match="already recorded"):
        resolver.record_effective("lr", "0.2", "default")

    frozen = resolver.freeze()
    assert frozen["lr"] is first  # the earlier resolution survived the rejected overwrite
    assert frozen["lr"].source == "cli"


def test_freeze_returns_a_deterministically_sorted_mapping() -> None:
    resolver = manifest.ConfigResolver(environ={})
    resolver.record_effective("zeta.lr", "1", "cli")
    resolver.record_effective("alpha.lr", "2", "cli")
    resolver.record_effective("mu.lr", "3", "cli")

    assert list(resolver.freeze()) == ["alpha.lr", "mu.lr", "zeta.lr"]


# ---------------------------------------------------------------------------
# xfail: defects found in the frozen module — see === NOTES ===
# ---------------------------------------------------------------------------


@requires_git
def test_repo_without_any_commit_is_never_reported_clean(repo: Path) -> None:
    """`git init` and nothing else: there is no recorded commit to be 'clean' against.

    CaptureStatus.CLEAN's own contract is "working tree matches the recorded commit" —
    with commit=None that claim has no referent. This is the ``all([]) is True`` shape
    inside the module that audits everyone else for it.
    """
    prov = manifest.capture_code_provenance(repo)

    assert prov.commit is None
    assert prov.status is not manifest.CaptureStatus.CLEAN


def test_retrying_an_identical_save_resolves_to_the_same_file(tmp_path: Path) -> None:
    """Docstring contract: 'a retried write of identical bytes is simply the same file'."""
    store = manifest.ManifestStore(tmp_path / "store")
    record = _manifest()

    first = store.save(record)
    again = store.save(record)

    assert again == first


def test_bare_string_prefixes_are_rejected_not_split_into_characters() -> None:
    """``capture_environment("NCCL_")`` is the natural call-site mistake.

    ``str`` is a ``Sequence[str]``, so it type-checks and then captures variables
    starting with 'N', 'C', 'C', 'L' or '_' — storing that char-list as the stated
    allowlist, laundering a typo into a false claim of scope.
    """
    with pytest.raises(TypeError, match="sequence of strings"):
        # Deliberately wrong-shaped argument; the signature permits it today.
        manifest.capture_environment("NCCL_", environ=_SUBMIT_ENV)
