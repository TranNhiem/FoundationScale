"""Tests for the telemetry section of ``RunManifest`` and the ``TelemetryEntry`` type.

Telemetry is what the run MEASURED; config is what the run DECLARED, and the two are
separate sections for that reason. These tests exist because the predecessor shape
folded them together: outcome data landed in config under ``telemetry.<key>`` keys,
and a manifest this writer emitted raised a spurious "unknown key" finding on its
own reload. Every test pairs the claim with the control that proves the assertion
could have failed — a green report is the default output of a broken check.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from foundationscale.provenance.manifest import (
    CapturedEnvironment,
    CaptureStatus,
    CodeProvenance,
    EffectiveValue,
    RunManifest,
    TelemetryEntry,
    Topology,
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

_BASE_TELEMETRY: dict[str, TelemetryEntry] = {
    "train_runtime_s": TelemetryEntry(
        key="train_runtime_s", value=1234.56, source="measured", unit="s"
    ),
    "total_flos": TelemetryEntry(key="total_flos", value=987654321, source="derived", unit="flops"),
}

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
    telemetry: Mapping[str, TelemetryEntry] | None = None,
) -> RunManifest:
    """Build a fully deterministic manifest carrying telemetry.

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
        telemetry=dict(telemetry if telemetry is not None else _BASE_TELEMETRY),
    )


# ---------------------------------------------------------------------------
# The public path
# ---------------------------------------------------------------------------


def test_telemetry_entry_is_reachable_from_the_package() -> None:
    """Catches the class landing in ``manifest.__all__`` but not in the package
    re-export list, which train/loop.py imports from — and which it imports inside
    a try/except that degrades to the flat fallback manifest, so the ImportError
    is swallowed and surfaces as a config entry that "is not a record"."""
    import foundationscale.provenance as provenance

    assert provenance.TelemetryEntry is TelemetryEntry
    assert "TelemetryEntry" in provenance.__all__


# ---------------------------------------------------------------------------
# The section: top-level, typed, and quiet on reload
# ---------------------------------------------------------------------------


def test_telemetry_serialises_as_a_top_level_section() -> None:
    """Catches telemetry folded into config, which would report an outcome in the
    section reserved for what the run declared."""
    document = make_manifest().to_dict()

    assert "telemetry" in document
    assert set(document["telemetry"]) == set(_BASE_TELEMETRY)
    # The defect being closed is the `config["telemetry.<key>"]` flattening: no
    # config key may carry outcome data under a telemetry prefix.
    assert not any(str(key).startswith("telemetry.") for key in document["config"])


def test_telemetry_round_trip_preserves_python_types() -> None:
    """Catches a from_dict that stringifies value: ``1.0 == 1`` in Python, so only
    an isinstance assertion detects a float that came back as an int or a string."""
    manifest = make_manifest()
    loaded = RunManifest.from_dict(manifest.to_dict())

    assert loaded.telemetry == manifest.telemetry
    runtime = loaded.telemetry["train_runtime_s"].value
    flos = loaded.telemetry["total_flos"].value
    assert isinstance(runtime, float)
    assert isinstance(flos, int)
    assert not isinstance(flos, bool)


def test_telemetry_round_trip_raises_no_findings() -> None:
    """Catches ``"telemetry"`` missing from the loader's known-key set, which makes
    every manifest this writer emits raise a spurious unknown-key finding on reload."""
    manifest = make_manifest()
    # Fixture control: the baseline is clean, so any finding below came from the
    # round-trip itself and not from the construction.
    assert manifest.findings == ()

    loaded = RunManifest.from_dict(manifest.to_dict())
    assert not any("telemetry" in finding or "unknown" in finding for finding in loaded.findings)


def test_manifest_without_a_telemetry_key_loads_cleanly() -> None:
    """Catches a loader that requires the key: manifests written before telemetry
    existed must stay readable, which the append-only store contract promises."""
    document = make_manifest().to_dict()
    del document["telemetry"]
    assert "telemetry" not in document  # the tamper must actually land

    loaded = RunManifest.from_dict(document)
    assert loaded.telemetry == {}
    assert not any("telemetry" in finding or "unknown" in finding for finding in loaded.findings)


# ---------------------------------------------------------------------------
# The fingerprint: blind to outcomes, alive to inputs
# ---------------------------------------------------------------------------


def test_fingerprint_is_blind_to_telemetry() -> None:
    """Catches telemetry in the semantic payload, which would give every run a
    unique fingerprint and destroy the redundancy contract it exists to provide."""
    base = make_manifest()
    other = make_manifest(
        telemetry={
            "train_runtime_s": TelemetryEntry(
                key="train_runtime_s", value=9999.99, source="measured", unit="s"
            ),
        }
    )
    # The perturbation must be real, otherwise the equality below is vacuous.
    assert base.telemetry != other.telemetry

    assert base.fingerprint() == other.fingerprint()
    assert base.differs_from(other) == []

    # Positive control: a semantic change MUST move the fingerprint, otherwise the
    # equality above could come from a fingerprint that hashes nothing at all.
    moved = make_manifest(
        config={
            **_BASE_CONFIG,
            "lr": EffectiveValue(key="lr", value="0.01", source="cli"),
        }
    )
    assert moved.fingerprint() != base.fingerprint()


# ---------------------------------------------------------------------------
# TelemetryEntry validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["cli", "", "MEASURED"])
def test_telemetry_source_is_validated(source: str) -> None:
    """Catches a free-text source — the 18-key string bag coming back one level
    down — and a refusal message that names neither the value nor the legal set."""
    with pytest.raises(ValueError) as excinfo:
        TelemetryEntry(key="train_runtime_s", value=1.0, source=source)
    message = str(excinfo.value)
    assert repr(source) in message
    for legal in ("measured", "derived", "unmeasured"):
        assert legal in message


def test_the_three_legal_sources_do_not_raise() -> None:
    """Control for the refusal above: a validator that rejects everything would
    pass the parametrized test while accepting nothing."""
    TelemetryEntry(key="a", value=1.0, source="measured")
    TelemetryEntry(key="b", value=2, source="derived")
    TelemetryEntry(key="c", value="no NVML access", source="unmeasured")


@pytest.mark.parametrize("value", [0, None, ""])
def test_unmeasured_entry_cannot_be_a_number(value: object) -> None:
    """Catches absence reported as a measurement: an unmeasured entry whose value
    is 0, None or "" reads downstream as a measured zero."""
    # Absence reported as 0 reads as a measurement; this class exists to refuse it.
    with pytest.raises(ValueError):
        TelemetryEntry(key="peak_memory_allocated_bytes", value=value, source="unmeasured")


def test_unmeasured_entry_with_a_reason_round_trips() -> None:
    """Control for the refusal above: the legal shape of absence — present,
    sourced, and explained — must be accepted and survive the wire."""
    entry = TelemetryEntry(
        key="peak_memory_allocated_bytes",
        value="torch.cuda.max_memory_allocated unavailable on this platform",
        source="unmeasured",
        unit="bytes",
    )
    manifest = make_manifest(telemetry={entry.key: entry})
    loaded = RunManifest.from_dict(manifest.to_dict())
    assert loaded.telemetry[entry.key] == entry


@pytest.mark.parametrize("key", ["", "   "])
def test_empty_key_is_refused(key: str) -> None:
    """Catches a metric with no name: an empty key would serialize into a section
    no reader can address."""
    with pytest.raises(ValueError):
        TelemetryEntry(key=key, value=1.0, source="measured")
