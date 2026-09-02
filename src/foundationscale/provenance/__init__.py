"""Run provenance: launch manifests that cannot lie, cannot be overwritten, and
cannot be skipped.

This package exists because the system it replaced — a fail-open shell fragment
sourced as ``|| true`` before every ``srun`` — was measurably absent: 0% provenance
coverage in one audited estate, 29% in the other, with 27 launches' records
silently overwritten, a git commit field that was false in 100% of bundles, and a
0-byte "uncommitted patch" capturing a subtree nothing edited. Meanwhile 24 runs
with byte-identical argv trained under two different objectives because the switch
was an environment variable no manifest ever recorded.

The public surface re-exported here is the replacement contract:

* :class:`~foundationscale.provenance.manifest.RunManifest` — structured,
  typed provenance (identity, code, effective config, scoped environment,
  topology, artifact paths) with :meth:`RunManifest.fingerprint` and
  :meth:`RunManifest.differs_from` for cross-run comparison.
* :func:`~foundationscale.provenance.manifest.capture_code_provenance` — records
  commit *and* dirty state *and* diff hash as separate claims, and reports a
  0-byte patch over untracked code as ``NOT_CAPTURED``, never "clean".
* :class:`~foundationscale.provenance.manifest.ManifestStore` — append-only,
  attempt-keyed, atomic, no-clobber writes.
* :func:`~foundationscale.provenance.manifest.require_manifest` — the fail-closed
  entry point that replaces ``|| true``.

Standard library only; importing this package never requires torch, git
availability, or a repository — every absence is represented as data
(:class:`CaptureStatus`), not as an import error.
"""

from __future__ import annotations

from foundationscale.provenance.manifest import (
    DEFAULT_ENV_PREFIXES,
    SCHEMA_VERSION,
    CapturedEnvironment,
    CaptureStatus,
    CodeProvenance,
    ConfigResolver,
    Difference,
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
    TopologyConsistency,
    capture_code_provenance,
    capture_environment,
    load,
    require_manifest,
)

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_ENV_PREFIXES",
    "CaptureStatus",
    "PathStatus",
    "DiffPathCoverage",
    "CodeProvenance",
    "Topology",
    "TopologyConsistency",
    "CapturedEnvironment",
    "EffectiveValue",
    "ConfigResolver",
    "Difference",
    "RunManifest",
    "ManifestStore",
    "ManifestError",
    "ManifestMissing",
    "ManifestVersionError",
    "ManifestExistsError",
    "capture_environment",
    "capture_code_provenance",
    "load",
    "require_manifest",
]
