"""Tests for manifest storage, semantic fingerprints and field-level diffing.

These tests exist because the module under test replaces mechanisms that failed
silently in production: overwrite-in-place manifest writes that destroyed 27 of 62
launches' records, and a provenance format that could not answer why 24 runs with
byte-identical argv split 12/12 between two objectives (incident 4). Every test here
pairs a "nothing bad happened" assertion with a positive control proving the
assertion could have failed — a green report is the default output of a broken
check, so the check itself gets checked.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from foundationscale.provenance.manifest import (
    SCHEMA_VERSION,
    CapturedEnvironment,
    CaptureStatus,
    CodeProvenance,
    DiffPathCoverage,
    EffectiveValue,
    ManifestError,
    ManifestExistsError,
    ManifestMissing,
    ManifestStore,
    ManifestVersionError,
    PathStatus,
    RunManifest,
    Topology,
    load,
    require_manifest,
)

_BASE_CODE = CodeProvenance(
    status=CaptureStatus.CLEAN,
    root="/repo",
    commit="a" * 40,
    dirty_files=0,
    untracked_files=0,
    diff_sha256="0" * 64,
    diff_bytes=0,
    paths=(),
    entrypoint="/repo/train.py",
    entrypoint_captured=True,
)

_BASE_CONFIG: dict[str, EffectiveValue] = {
    "batch_size": EffectiveValue(
        key="batch_size", value="512", source="config:train.yaml#batch_size"
    ),
    "lr": EffectiveValue(key="lr", value="0.001", source="cli"),
}

_BASE_ENV = CapturedEnvironment(
    allowlist=("NCCL_", "OBJECTIVE_"),
    values={"NCCL_DEBUG": "WARN", "OBJECTIVE_SWITCH": "dense"},
    source_var_count=50,
)

_BASE_TOPOLOGY = Topology(
    nodes=2,
    gpus_per_node=8,
    tensor_parallel=2,
    pipeline_parallel=2,
    data_parallel=4,
    expert_parallel=1,
)

_CLEAN_PATH_COVERAGE = DiffPathCoverage(
    path=".",
    exists=True,
    tracked_files=42,
    modified_tracked_files=0,
    untracked_files=0,
    captured_files=0,
    captured_bytes=0,
    status=PathStatus.CLEAN,
)

_CREATED_AT = "2024-05-01T00:00:00+00:00"


def make_manifest(
    *,
    run_id: str = "run-001",
    attempt: int = 1,
    code: CodeProvenance | None = None,
    config: Mapping[str, EffectiveValue] | None = None,
    environment: CapturedEnvironment | None = None,
    topology: Topology | None = None,
    job_id: str | None = "12345",
    created_at: str = _CREATED_AT,
    artifact_paths: Mapping[str, str] | None = None,
) -> RunManifest:
    """Build a fully deterministic manifest.

    ``created_at`` is pinned so two calls produce byte-identical records unless a
    caller deliberately perturbs a field; every fingerprint test depends on that.
    """
    return RunManifest(
        run_id=run_id,
        attempt=attempt,
        code=code if code is not None else _BASE_CODE,
        config=config if config is not None else _BASE_CONFIG,
        environment=environment if environment is not None else _BASE_ENV,
        topology=topology if topology is not None else _BASE_TOPOLOGY,
        job_id=job_id,
        created_at=created_at,
        artifact_paths=dict(artifact_paths or {"ckpt": "/out/ckpt"}),
    )


def make_env_switch(value: str) -> CapturedEnvironment:
    """The single-variable difference that split 24 runs 12/12 (incident 4)."""
    return dataclasses.replace(_BASE_ENV, values={**_BASE_ENV.values, "OBJECTIVE_SWITCH": value})


def make_manifest_with_entrypoint_hole() -> RunManifest:
    """The audited estate's real hole: the launcher lived outside every snapshot."""
    return make_manifest(
        code=dataclasses.replace(
            _BASE_CODE,
            entrypoint="/somewhere-else/launch.sh",
            entrypoint_captured=False,
        )
    )


def _with_code(**changes: object) -> RunManifest:
    return make_manifest(code=dataclasses.replace(_BASE_CODE, **changes))


def _with_topology(**changes: object) -> RunManifest:
    return make_manifest(topology=dataclasses.replace(_BASE_TOPOLOGY, **changes))


def _temp_files(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.iterdir() if p.name.startswith("."))


# ---------------------------------------------------------------------------
# Storage: atomic, never-overwrite writes
# ---------------------------------------------------------------------------


def test_save_writes_exactly_one_loadable_file(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    manifest = make_manifest(run_id="run-7", attempt=1)
    written = store.save(manifest)

    assert written.parent == tmp_path / "run-7"
    # Attempt-only name. The fingerprint used to be in here, which is precisely what
    # inverted the store's two guarantees: identical retries collided and raised, while
    # genuinely conflicting records got distinct names and quietly coexisted.
    assert re.fullmatch(r"attempt-0001\.json", written.name)
    # Everything in the run dir is the manifest: no hidden state, no temp files.
    assert sorted((tmp_path / "run-7").iterdir()) == [written]
    assert load(written) == manifest


def test_save_never_overwrites_an_existing_record(tmp_path: Path) -> None:
    """The defect being prevented is `os.replace`: the silent destroyer of records."""
    store = ManifestStore(tmp_path)
    manifest = make_manifest(run_id="victim", attempt=1)
    run_dir = tmp_path / "victim"
    run_dir.mkdir(parents=True)
    original = run_dir / "attempt-0001.json"
    original.write_text("previous attempt record\n", encoding="utf-8")

    with pytest.raises(ManifestExistsError):
        store.save(manifest)

    # The prior record is byte-for-byte intact — the failure communicated instead
    # of silently destroying it, which is the entire point of the store.
    assert original.read_text(encoding="utf-8") == "previous attempt record\n"
    assert sorted(run_dir.iterdir()) == [original]


def test_successful_save_leaves_no_temp_files(tmp_path: Path) -> None:
    # Positive control for the detector itself: it must be able to see a leftover.
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / ".attempt-0001-deadbeef.tmp.1").write_text("x", encoding="utf-8")
    assert _temp_files(probe), "detector cannot see a temp file; the test would be vacuous"

    store = ManifestStore(tmp_path / "store")
    store.save(make_manifest(run_id="tidy", attempt=1))
    assert _temp_files(tmp_path / "store" / "tidy") == []


def test_failed_save_leaves_no_temp_files(tmp_path: Path) -> None:
    """A refused write must clean up after itself; residue would corrupt later scans."""
    store = ManifestStore(tmp_path)
    manifest = make_manifest(run_id="untidy", attempt=1)
    run_dir = tmp_path / "untidy"
    run_dir.mkdir(parents=True)
    blocker = run_dir / "attempt-0001.json"
    blocker.write_text("occupant\n", encoding="utf-8")

    with pytest.raises(ManifestExistsError):
        store.save(manifest)

    assert _temp_files(run_dir) == []
    assert blocker.read_text(encoding="utf-8") == "occupant\n"


def test_identical_retry_converges_to_one_file(tmp_path: Path) -> None:
    """A crash-safe launcher re-runs its manifest write; that retry must not crash."""
    store = ManifestStore(tmp_path)
    manifest = make_manifest(run_id="retry", attempt=1)
    first = store.save(manifest)
    second = store.save(manifest)

    assert second == first
    assert sorted((tmp_path / "retry").iterdir()) == [first]
    assert len(store.attempts("retry")) == 1


def test_conflicting_manifest_for_same_attempt_is_rejected(tmp_path: Path) -> None:
    """Two different computations must never share a (run_id, attempt) slot."""
    store = ManifestStore(tmp_path)
    original = make_manifest(run_id="clash", attempt=1)
    store.save(original)

    intruder = make_manifest(
        run_id="clash",
        attempt=1,
        topology=dataclasses.replace(_BASE_TOPOLOGY, data_parallel=8),
    )
    # Prove the slot conflict is real before asking the store to reject it. A
    # test whose two manifests were identical would pass vacuously.
    assert intruder.fingerprint() != original.fingerprint()

    with pytest.raises(ManifestExistsError):
        store.save(intruder)

    survivors = store.attempts("clash")
    assert len(survivors) == 1
    assert survivors[0] == original


# ---------------------------------------------------------------------------
# Attempt allocation and enumeration
# ---------------------------------------------------------------------------


def test_allocate_attempt_starts_at_one_and_never_reuses(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)

    assert store.allocate_attempt("roll") == 1
    store.save(make_manifest(run_id="roll", attempt=1))
    assert store.allocate_attempt("roll") == 2
    store.save(make_manifest(run_id="roll", attempt=2))
    assert store.allocate_attempt("roll") == 3


def test_attempts_returns_every_record_oldest_first(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    # Deliberately write out of order (2, 3, 1): with three records, correct
    # ordering cannot be luck the way two records ordered by creation could be.
    for attempt in (2, 3, 1):
        store.save(make_manifest(run_id="rel", attempt=attempt))

    attempts = store.attempts("rel")
    assert len(attempts) == 3
    assert [m.attempt for m in attempts] == [1, 2, 3]
    assert all(m.run_id == "rel" for m in attempts)


def test_corrupt_record_in_a_canonical_slot_is_not_silently_skipped(tmp_path: Path) -> None:
    """The coverage the lookalike test used to provide, in its post-fix form.

    Before the filename fix, `attempt-0001.json` was a near-miss name that the scanner
    filtered out. It is now the canonical slot name, so an unreadable file there is a
    CORRUPT RECORD, not a lookalike — and the two must not be conflated.

    Skipping it would let a run directory whose every record is corrupt report zero
    attempts, which reads as "nothing ever ran": the vacuous-success shape this whole
    package exists to refuse. Failing closed is the only honest answer, because the
    scanner genuinely cannot say what happened in that slot.
    """
    store = ManifestStore(tmp_path)
    run_dir = tmp_path / "corrupt"
    run_dir.mkdir(parents=True)

    # Positive control: the scanner reads a good record in this very directory, so the
    # raise below is about the corrupt file and not about a scanner that never works.
    store.save(make_manifest(run_id="corrupt", attempt=1))
    assert len(store.attempts("corrupt")) == 1

    (run_dir / "attempt-0002.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ManifestError):
        store.attempts("corrupt")


def test_only_well_formed_records_count_as_attempts(tmp_path: Path) -> None:
    """A directory of lookalike files is *zero* attempts, not partial credit.

    This is the vacuous case for the store's scanners: near-miss filenames (the
    shell-capsule equivalents) must not satisfy ``allocate_attempt`` or
    ``attempts``.
    """
    store = ManifestStore(tmp_path)
    run_dir = tmp_path / "mixed"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "attempt-0001-NOTHEXDIGITX.json").write_text("{}", encoding="utf-8")
    (run_dir / "attempt-abcd.json").write_text("{}", encoding="utf-8")
    (run_dir / "attempt-.json").write_text("{}", encoding="utf-8")
    (run_dir / "attempt-0001.json.bak").write_text("{}", encoding="utf-8")

    assert store.attempts("mixed") == ()
    assert store.allocate_attempt("mixed") == 1, "lookalikes must not reserve attempt 1"

    # Positive control: a real record in the same directory *is* seen, so the
    # empty results above reflect the filters, not a broken scanner.
    store.save(make_manifest(run_id="mixed", attempt=1))
    real = store.attempts("mixed")
    assert len(real) == 1
    assert real[0].attempt == 1


def test_reads_against_missing_root_create_nothing_and_report_empty(tmp_path: Path) -> None:
    """Reads must be side-effect free: a peek must not conjure a store into being."""
    store = ManifestStore(tmp_path / "no-such-store")
    assert store.attempts("ghost") == ()
    with pytest.raises(ManifestMissing):
        store.latest("ghost")
    with pytest.raises(ManifestMissing):
        require_manifest(store, "ghost")
    assert not (tmp_path / "no-such-store").exists()


# ---------------------------------------------------------------------------
# require_manifest: the negation of `|| true`
# ---------------------------------------------------------------------------


def test_require_manifest_fails_closed_until_a_record_exists(tmp_path: Path) -> None:
    """The negation of `[ -r capsule ] && source capsule || true`."""
    store = ManifestStore(tmp_path)

    with pytest.raises(ManifestMissing):
        require_manifest(store, "cold-run")

    # A present directory with no valid record must still fail closed; the
    # predecessor's capsule "existed" in exactly this sense.
    run_dir = tmp_path / "cold-run"
    run_dir.mkdir(parents=True)
    (run_dir / "capsule.sh").write_text("export X=1\n", encoding="utf-8")
    with pytest.raises(ManifestMissing):
        require_manifest(store, "cold-run")

    # Positive control: once a manifest is written, the very same call must find
    # it — otherwise the "missing" results above prove nothing about detection.
    attempt = store.allocate_attempt("cold-run")
    store.save(make_manifest(run_id="cold-run", attempt=attempt))
    assert require_manifest(store, "cold-run").run_id == "cold-run"


def test_require_manifest_returns_the_latest_attempt(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    store.save(make_manifest(run_id="multi", attempt=1, job_id="first"))
    store.save(make_manifest(run_id="multi", attempt=2, job_id="second"))

    latest = require_manifest(tmp_path, "multi")  # path form of the store argument
    assert latest.attempt == 2
    assert latest.job_id == "second"


# ---------------------------------------------------------------------------
# Fingerprint: what the computation is, and only that
# ---------------------------------------------------------------------------


def test_fingerprint_is_a_stable_sha256_hex() -> None:
    fingerprint = make_manifest().fingerprint()
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)
    assert fingerprint == make_manifest().fingerprint()


_EXCLUDED_LAUNCH_METADATA: list[tuple[str, Callable[[], RunManifest]]] = [
    ("run_id", lambda: make_manifest(run_id="completely-different-run")),
    ("attempt", lambda: make_manifest(attempt=7)),
    ("job_id", lambda: make_manifest(job_id="99999999")),
    ("created_at", lambda: make_manifest(created_at="2030-12-31T23:59:59+00:00")),
    ("artifact_paths", lambda: make_manifest(artifact_paths={"ckpt": "/elsewhere/ckpt"})),
]


@pytest.mark.parametrize(
    ("field_name", "mutate"),
    _EXCLUDED_LAUNCH_METADATA,
    ids=[name for name, _ in _EXCLUDED_LAUNCH_METADATA],
)
def test_fingerprint_ignores_launch_metadata(
    field_name: str, mutate: Callable[[], RunManifest]
) -> None:
    """Excluded fields identify the launch, not the computation; they must not move."""
    base = make_manifest()
    # Detector control, run first: the fingerprint must be *capable* of moving,
    # otherwise "unchanged" is vacuous — a hash of nothing never changes either.
    moved = make_manifest(environment=make_env_switch("moe"))
    assert moved.fingerprint() != base.fingerprint()

    assert mutate().fingerprint() == base.fingerprint(), (
        f"{field_name} is launch metadata; it must not move the fingerprint"
    )


_INCLUDED_SEMANTIC_FIELDS: list[tuple[str, Callable[[], RunManifest]]] = [
    ("code.status", lambda: _with_code(status=CaptureStatus.CAPTURED)),
    ("code.commit", lambda: _with_code(commit="b" * 40)),
    ("code.dirty_files", lambda: _with_code(dirty_files=892)),
    ("code.diff_sha256", lambda: _with_code(diff_sha256="f" * 64)),
    ("code.diff_bytes", lambda: _with_code(diff_bytes=4096)),
    ("code.entrypoint_captured", lambda: _with_code(entrypoint_captured=False)),
    ("code.paths", lambda: _with_code(paths=(_CLEAN_PATH_COVERAGE,))),
    (
        "config.value",
        lambda: make_manifest(
            config={**_BASE_CONFIG, "lr": EffectiveValue(key="lr", value="0.01", source="cli")}
        ),
    ),
    (
        "config.source",
        lambda: make_manifest(
            config={
                **_BASE_CONFIG,
                "lr": EffectiveValue(key="lr", value="0.001", source="default"),
            }
        ),
    ),
    ("environment.values", lambda: make_manifest(environment=make_env_switch("moe"))),
    (
        "environment.allowlist",
        lambda: make_manifest(environment=dataclasses.replace(_BASE_ENV, allowlist=("NCCL_",))),
    ),
    ("topology.tensor_parallel", lambda: _with_topology(tensor_parallel=4)),
    ("topology.data_parallel", lambda: _with_topology(data_parallel=16)),
    ("topology.expert_parallel", lambda: _with_topology(expert_parallel=8)),
]


@pytest.mark.parametrize(
    ("field_name", "mutate"),
    _INCLUDED_SEMANTIC_FIELDS,
    ids=[name for name, _ in _INCLUDED_SEMANTIC_FIELDS],
)
def test_fingerprint_tracks_semantic_fields(
    field_name: str, mutate: Callable[[], RunManifest]
) -> None:
    """These fields define the computation; a fingerprint that ignores one is a lie.

    This half is the one that catches a fingerprint over an empty payload: such a
    fingerprint passes every exclusion test above and detects nothing at all.
    """
    base = make_manifest()
    mutated = mutate()
    assert mutated.fingerprint() != base.fingerprint(), (
        f"{field_name} is semantically significant; changing it must move the "
        f"fingerprint — a fingerprint that cannot move authenticates nothing"
    )


# ---------------------------------------------------------------------------
# differs_from: the query the 24-run audit needed
# ---------------------------------------------------------------------------


def test_differs_from_isolates_the_objective_switch() -> None:
    """Incident 4 reproduction: argv identical, one env var splits the runs 12/12.

    The audited estate's manifests could not answer *where* 24 runs diverged. The
    answer must be exactly one field — nothing under config, code or topology.
    """
    arm_dense = make_manifest()
    arm_moe = make_manifest(environment=make_env_switch("moe"))

    diffs = arm_dense.differs_from(arm_moe)
    assert len(diffs) == 1
    diff = diffs[0]
    assert diff.field == "environment.values.OBJECTIVE_SWITCH"
    assert diff.self_value == "dense"
    assert diff.other_value == "moe"
    assert str(diff) == "environment.values.OBJECTIVE_SWITCH: 'dense' -> 'moe'"


def test_differs_from_marks_absent_fields_in_both_directions() -> None:
    """`<absent>` must render; a missing key is not the same as an equal value."""
    reduced = dataclasses.replace(
        _BASE_ENV,
        values={k: v for k, v in _BASE_ENV.values.items() if k != "OBJECTIVE_SWITCH"},
    )
    full = make_manifest()
    partial = make_manifest(environment=reduced)

    forward = full.differs_from(partial)
    assert len(forward) == 1
    assert str(forward[0]) == "environment.values.OBJECTIVE_SWITCH: 'dense' -> <absent>"

    backward = partial.differs_from(full)
    assert len(backward) == 1
    assert str(backward[0]) == "environment.values.OBJECTIVE_SWITCH: <absent> -> 'dense'"


def test_identical_manifests_report_no_differences() -> None:
    first, second = make_manifest(), make_manifest()
    assert first.fingerprint() == second.fingerprint()
    # The empty list is the honest "no defect found" — but it is only trustworthy
    # because the next test proves differences are actually detected.
    assert first.differs_from(second) == []
    assert second.differs_from(first) == []


def test_differing_manifests_never_report_an_empty_diff() -> None:
    """Positive control: changed input must produce non-empty output.

    An empty list here is `all([]) is True` one level up — the bug the audit's own
    verifier shipped while looking for that bug in someone else's code.
    """
    base = make_manifest()
    other = make_manifest(topology=dataclasses.replace(_BASE_TOPOLOGY, data_parallel=16))
    assert base.fingerprint() != other.fingerprint()

    diffs = base.differs_from(other)
    assert diffs, "fingerprints differ but differs_from produced an empty list"
    assert any(d.field == "topology.data_parallel" for d in diffs)


# ---------------------------------------------------------------------------
# load(): schema discipline and recomputed findings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", [0, 2, -1, "1"])
def test_load_rejects_any_schema_version_but_the_current_one(
    tmp_path: Path, version: object
) -> None:
    store = ManifestStore(tmp_path)
    written = store.save(make_manifest(run_id="ver", attempt=1))

    # Positive control: the same loader accepts the file before tampering, so a
    # rejection below is the version check, not a loader that fails everything.
    assert load(written).run_id == "ver"

    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
    data["schema_version"] = version
    written.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ManifestVersionError):
        load(written)


def test_load_rejects_a_manifest_with_no_schema_version(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    written = store.save(make_manifest(run_id="noversion", attempt=1))
    data = json.loads(written.read_text(encoding="utf-8"))
    del data["schema_version"]
    assert "schema_version" not in data  # the tamper must actually land
    written.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ManifestVersionError):
        load(written)


def test_baseline_manifest_reports_no_findings() -> None:
    """A clean manifest is a strong claim, not an empty one — so verify the detector."""
    assert make_manifest_with_entrypoint_hole().findings, (
        "fixture is broken: an entrypoint outside the root must produce a finding"
    )
    assert make_manifest().findings == ()


def test_entrypoint_outside_provenance_root_is_a_finding() -> None:
    manifest = make_manifest_with_entrypoint_hole()
    assert any("entrypoint" in finding for finding in manifest.findings)


def test_load_recomputes_findings_and_ignores_stored_ones(tmp_path: Path) -> None:
    """A hand-edited manifest must not be able to erase its own blemishes."""
    manifest = make_manifest_with_entrypoint_hole()
    assert manifest.findings, "this test needs a real finding to prove recomputation"

    store = ManifestStore(tmp_path)
    written = store.save(manifest)
    data = json.loads(written.read_text(encoding="utf-8"))
    data["findings"] = ["everything is fine"]
    written.write_text(json.dumps(data), encoding="utf-8")
    # Prove the tamper landed on disk before asserting anything about load().
    assert json.loads(written.read_text(encoding="utf-8"))["findings"] == ["everything is fine"]

    reloaded = load(written)
    # The real findings return...
    assert reloaded.findings == manifest.findings
    assert any("entrypoint" in finding for finding in reloaded.findings)
    # ...and the fabricated serenity does not survive.
    assert not any("everything is fine" in finding for finding in reloaded.findings)


def test_load_strips_fabricated_findings_from_a_clean_manifest(tmp_path: Path) -> None:
    """Recomputation is symmetric: invented blemishes are as untrusted as erased ones."""
    manifest = make_manifest()
    assert manifest.findings == ()

    store = ManifestStore(tmp_path)
    written = store.save(manifest)
    data = json.loads(written.read_text(encoding="utf-8"))
    data["findings"] = ["scary-looking but invented"]
    written.write_text(json.dumps(data), encoding="utf-8")
    assert json.loads(written.read_text(encoding="utf-8"))["findings"] == [
        "scary-looking but invented"
    ]

    assert load(written).findings == ()
