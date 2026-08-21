"""Run provenance manifests: what *actually* launched a run, recorded so it cannot lie.

Why this module exists
----------------------
The predecessor to this module was a 31-line shell fragment, ``_repro_capsule.sh``,
sourced immediately before ``srun`` as ``[ -r ... ] && source ... || true`` — fail-open
by construction, skippable by accident. A ground-truth probe of the two audited
estates measured exactly how that "capsule" failed, and every class in this module
answers one of those measurements:

* **Coverage was never known.** One repo had 35 provenance bundles against 119 result
  dirs (29% coverage); the other had 77 result dirs and **zero** bundles (0%).
  Nothing counted, so nobody could have said this. This module's
  :func:`require_manifest` inverts the default: if provenance was not written, the
  job raises. The ``|| true`` is the reason 0% coverage went unnoticed, and it does
  not survive contact with an exception raised before ``srun``.

* **Overwrite-in-place destroyed history.** The ledger records 62 launches against
  35 bundles; 7 run-ids were launched up to 5 times each, rewriting the same path, and
  27 launches' manifests are simply gone. :class:`ManifestStore` writes are keyed by
  run-id *and* attempt, linked no-clobber, so a relaunch is a new record, not a
  quieter deletion.

* **The git commit was a lie in all 35 bundles.** Every manifest recorded the same
  ``git_commit``, whose HEAD was dated three weeks before the runs it supposedly
  described, against a working tree with 892 dirty files. A commit id *for a dirty
  tree* is a false statement. :func:`capture_code_provenance` therefore records the
  commit *and* the dirty-file count *and* a hash of the actual diff as one atomic
  claim you can check, instead of one string you cannot.

* **The diff captured the wrong subtree.** ``uncommitted.patch`` was
  ``git diff HEAD -- $SDPO``, but the directories actually being edited contained
  *zero git-tracked files*, so every patch was 0 bytes — a successful-looking capture
  of nothing. Worse, the real entrypoint lived under a different root entirely and
  appears in no snapshot. This module records, per captured path, how many files are
  tracked, how many are untracked, and how many bytes the diff actually held; a
  0-byte patch over an untracked directory is reported as
  :attr:`CaptureStatus.NOT_CAPTURED`, never as "clean". That distinction is the whole
  defect.

* **Environment was never captured**, while ``srun --export=ALL`` propagated the
  entire interactive submit environment. This is how 24 runs with byte-identical
  argv split 12/12 between two objectives on one unrecorded variable (incident 4 of
  the audit). :func:`capture_environment` captures an explicitly prefixed slice and
  records the allowlist *and* the total variable count it drew from — an env capture
  that does not state its scope is an unqualified count, and an unqualified count is
  not a fact. :meth:`RunManifest.differs_from` is the field-level query those 24 runs
  needed and could not answer.

This module is standard-library only and performs all git interaction through short,
timeout-bounded subprocess calls, treating any subprocess or repository failure as
"not captured" rather than as an exception that provenance code would swallow.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "DEFAULT_ENV_PREFIXES",
    "CaptureStatus",
    "PathStatus",
    "DiffPathCoverage",
    "CodeProvenance",
    "Topology",
    "CapturedEnvironment",
    "EffectiveValue",
    "ConfigResolver",
    "Difference",
    "DeclaredCheckpoint",
    "RunManifest",
    "ManifestStore",
    "ManifestError",
    "ManifestMissing",
    "ManifestVersionError",
    "ManifestExistsError",
    "capture_environment",
    "capture_code_provenance",
    "capture_state_dict_keys",
    "declared_from_hf_config",
    "declared_from_megatron_args",
    "require_manifest",
    "load",
]

SCHEMA_VERSION = 1
"""The only manifest schema version this module will load.

:func:`load` refuses anything else, including future versions: a silent best-effort
parse of a schema the reader does not understand is precisely how the predecessor
schema (a flat 18-key all-string dict, 5 of whose keys were empty in 33 of 35
bundles) metastasized while looking fine.
"""

SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (SCHEMA_VERSION,)
"""Every manifest schema version :func:`load` accepts, upgraded in memory on the way in.

Refusal is honest only if acceptance is stated: this tuple is the acceptance half
of the version contract. A version outside it raises :class:`ManifestVersionError`
naming both what was found and what is supported; a version inside it passes
through :func:`_upgrade_to_current` — today the identity, and the single seam
where any future vN → vN+1 lift must live, so that an honest bump upgrades stored
manifests in memory instead of orphaning every append-only store that holds them.
"""

DEFAULT_ENV_PREFIXES: tuple[str, ...] = (
    "FOUNDATIONSCALE_",
    "NCCL_",
    "TORCH_",
    "CUDA_",
    "OMP_",
    "SLURM_",
    "NVTE_",
    "PYTORCH_",
)
"""Default environment capture allowlist.

These prefixes were chosen from the incident record: ``SLURM_*`` because submission
flags decide topology, ``NCCL_*``/``NVTE_*``/``PYTORCH_*`` because they have each
changed numerics or collectives silently in the audited estate, and
``FOUNDATIONSCALE_`` because the framework's own switches were the variable that
split 24 runs 12/12. The allowlist is stored inside every manifest so a reader can
always see what was *not* captured.
"""

_GIT_TIMEOUT_S = 60
_PATH_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_SOURCE_RE = re.compile(r"^(cli|default|env:[A-Za-z_][A-Za-z0-9_]*|config:[^#\s]+#[^#\s]+)$")
_ATTEMPT_RE = re.compile(r"^attempt-(\d+)\.json$")


class CaptureStatus(str, Enum):
    """Whether the code that will execute is actually recoverable from the manifest.

    The ordering of severity is deliberate: ``CLEAN`` and ``CAPTURED`` are honest
    statements about the code; ``NOT_CAPTURED`` means bytes that will run exist
    nowhere in the record, and ``NOT_A_REPOSITORY`` means there was never a record
    to make. Both of the latter block downstream tooling that fails closed.
    """

    CLEAN = "clean"
    """Working tree (within the captured scope) matches the recorded commit."""

    CAPTURED = "captured"
    """Tree differs from HEAD, and the diff bytes + hash are stored in the manifest."""

    NOT_CAPTURED = "not_captured"
    """Executing code exists that the diff cannot see (untracked files, or edits
    outside every captured path, or an entrypoint outside the repo root). This is
    the exact shape of the 0-byte-``uncommitted.patch`` defect."""

    NOT_A_REPOSITORY = "not_a_repository"
    """No git metadata at the capture root. A commit hash is impossible here."""


class PathStatus(str, Enum):
    """Per-path capture outcome, one per entry of ``CodeProvenance.paths``."""

    CLEAN = "clean"
    CAPTURED = "captured"
    NOT_CAPTURED = "not_captured"
    NO_SUCH_PATH = "no_such_path"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ManifestError(RuntimeError):
    """Base class for provenance failures.

    All provenance failure modes raise rather than returning sentinel values. The
    predecessor system communicated every one of these conditions with the exit
    status of ``|| true``, which is to say it communicated nothing at all.
    """


class ManifestMissing(ManifestError):
    """Raised by :func:`require_manifest` when no manifest exists for a run id."""


class ManifestVersionError(ManifestError):
    """Raised when a manifest declares a schema version this reader cannot interpret.

    Attributes:
        found: The ``schema_version`` value as loaded — deliberately typed
            ``object``, because an absent, boolean, float or string version is
            refused through this same path rather than through a second taxonomy.
        supported: The versions this reader can interpret
            (:data:`SUPPORTED_SCHEMA_VERSIONS`). A caller routing the failure must
            not have to parse either fact back out of the message text.
    """

    def __init__(self, message: str, *, found: object, supported: tuple[int, ...]) -> None:
        super().__init__(message)
        self.found = found
        self.supported = supported


class ManifestExistsError(ManifestError):
    """Raised when a write would overwrite an existing manifest.

    Overwrite-in-place destroyed 27 of 62 launches' manifests in the audited estate;
    this exception is that defect expressed as a type.
    """


# ---------------------------------------------------------------------------
# JSON-boundary validation
# ---------------------------------------------------------------------------


def _expect_mapping(value: object, field: str) -> Mapping[str, object]:
    """Narrow an untrusted loaded field to a mapping, refusing anything else.

    Every value in a manifest loaded from disk is ``object`` until checked.
    Claiming a shape without checking it (``cast``) would trust a record this
    module did not just write — exactly the failure mode the schema-version
    check in :func:`load` exists to prevent — so a wrong shape raises here, at
    the boundary, with the field name attached.
    """
    if not isinstance(value, Mapping):
        raise ManifestError(
            f"corrupt manifest: field {field!r} must be a mapping, got {type(value).__name__}"
        )
    return value


def _expect_list(value: object, field: str) -> list[object]:
    """List variant of :func:`_expect_mapping`."""
    if not isinstance(value, list):
        raise ManifestError(
            f"corrupt manifest: field {field!r} must be a list, got {type(value).__name__}"
        )
    return value


def _expect_int(value: object, field: str) -> int:
    """Int variant of :func:`_expect_mapping`; JSON booleans are refused."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(
            f"corrupt manifest: field {field!r} must be an int, got {type(value).__name__}"
        )
    return value


# ---------------------------------------------------------------------------
# Wire-format key registries
# ---------------------------------------------------------------------------
# Known keys per level of the v1 document. Loaded keys outside these sets are not
# refused — an append-only store must stay readable beside a slightly newer
# writer — but they are never dropped in silence either: each is stashed on
# RunManifest._loaded_extra_keys and resurfaced as a finding, because a field
# that vanishes on load (the checkpoint gates' denominators are the standing
# example) is a denominator nobody can prove was ever written.

_MANIFEST_KNOWN_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "attempt",
        "job_id",
        "created_at",
        "fingerprint",
        "code",
        "config",
        "environment",
        "topology",
        "artifact_paths",
        "findings",
        "declared",
    }
)
_CODE_KNOWN_KEYS = frozenset(
    {
        "status",
        "root",
        "commit",
        "dirty_files",
        "untracked_files",
        "diff_sha256",
        "diff_bytes",
        "paths",
        "entrypoint",
        "entrypoint_captured",
    }
)
_PATH_KNOWN_KEYS = frozenset(
    {
        "path",
        "exists",
        "tracked_files",
        "modified_tracked_files",
        "untracked_files",
        "captured_files",
        "captured_bytes",
        "status",
    }
)
_ENVIRONMENT_KNOWN_KEYS = frozenset({"allowlist", "values", "source_var_count"})
_TOPOLOGY_KNOWN_KEYS = frozenset(
    {
        "nodes",
        "gpus_per_node",
        "tensor_parallel",
        "pipeline_parallel",
        "data_parallel",
        "expert_parallel",
    }
)
_EFFECTIVE_VALUE_KNOWN_KEYS = frozenset({"key", "value", "source", "env_value", "findings"})


def _unknown_keys(data: Mapping[str, object], known: frozenset[str]) -> tuple[str, ...]:
    """Keys present in a loaded mapping that this reader does not understand."""
    return tuple(sorted(str(key) for key in data if str(key) not in known))


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapturedEnvironment:
    """The captured slice of the process environment, with its scope attached.

    Attributes:
        allowlist: The prefixes that were captured, in the order they were given.
            Stored next to the values so any reader knows what was excluded.
        values: The captured variables, sorted by name.
        source_var_count: Total number of variables in the environment the capture
            drew from. With ``len(values)`` this makes the capture a qualified
            count: N of M variables, under allowlist A.
    """

    allowlist: tuple[str, ...]
    values: Mapping[str, str]
    source_var_count: int

    @property
    def captured(self) -> int:
        return len(self.values)

    @property
    def omitted(self) -> int:
        return self.source_var_count - len(self.values)

    def to_dict(self) -> dict[str, object]:
        return {
            "allowlist": list(self.allowlist),
            "values": dict(self.values),
            "source_var_count": self.source_var_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CapturedEnvironment:
        return cls(
            allowlist=tuple(
                str(p) for p in _expect_list(data["allowlist"], "environment.allowlist")
            ),
            values={
                str(k): str(v)
                for k, v in _expect_mapping(data["values"], "environment.values").items()
            },
            source_var_count=_expect_int(data["source_var_count"], "environment.source_var_count"),
        )


def capture_environment(
    prefixes: Sequence[str] = DEFAULT_ENV_PREFIXES,
    *,
    environ: Mapping[str, str] | None = None,
) -> CapturedEnvironment:
    """Capture the environment variables matching ``prefixes``, recording the scope.

    Args:
        prefixes: Variable-name prefixes to capture. Defaults to
            :data:`DEFAULT_ENV_PREFIXES`.
        environ: The environment to read. Defaults to :data:`os.environ`; pass a
            mapping explicitly in tests or when capturing a launcher script's
            computed environment rather than this process's.

    Returns:
        A :class:`CapturedEnvironment`. Even when nothing matched, the result states
        its allowlist and the size of the pool it drew from, so "nothing captured"
        is a derived claim rather than an assumption.
    """
    if isinstance(prefixes, str):
        raise TypeError(
            "prefixes must be a sequence of strings, not one string: "
            f"{prefixes!r} would be iterated character by character, silently "
            "widening the allowlist; wrap it, e.g. (prefix,) or [prefix]"
        )
    src = dict(os.environ if environ is None else environ)
    pfx = tuple(prefixes)
    values = {key: src[key] for key in sorted(src) if any(key.startswith(p) for p in pfx)}
    return CapturedEnvironment(allowlist=pfx, values=values, source_var_count=len(src))


# ---------------------------------------------------------------------------
# Effective config capture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectiveValue:
    """One resolved configuration value and the provenance of its resolution.

    An effective value answers "what did the run actually use?", which the audited
    estate could not do: manifests recorded declared values, while resolution layers
    (CLI over env over config file over library default) silently produced something
    else. The 24-run split happened because the decisive value came from an env var
    nothing recorded.

    Attributes:
        key: The configuration key.
        value: The resolved value the run will use, stringified.
        source: One of ``"cli"``, ``"default"``, ``"env:NAME"`` or
            ``"config:path#key"``.
        env_value: The live value of the implicated environment variable at record
            time, when one is implicated; ``None`` otherwise. For ``default``-sourced
            values this is the shadowing variable's value; for ``env:``-sourced
            values it is what the build actually saw.
        findings: Anomalies detected at record time (see :class:`ConfigResolver`).
    """

    key: str
    value: str
    source: str
    env_value: str | None = None
    findings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "env_value": self.env_value,
            "findings": list(self.findings),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EffectiveValue:
        return cls(
            key=str(data["key"]),
            value=str(data["value"]),
            source=str(data["source"]),
            env_value=None if data.get("env_value") is None else str(data["env_value"]),
            findings=tuple(str(f) for f in _expect_list(data.get("findings", []), "findings")),
        )


class ConfigResolver:
    """Collects effective configuration values with their resolution provenance.

    Build one of these during argument/config resolution — at the single point in
    launcher code where the final value is decided — then hand ``freeze()`` to the
    :class:`RunManifest`. Recording anywhere else means recording a value that later
    layers may still overwrite, which is how declared/effective divergence happens.

    Args:
        environ: Environment consulted for shadow/drift checks. Defaults to
            :data:`os.environ` at construction time (a snapshot copy, so that a
            mutated ``os.environ`` during startup cannot retro-edit the findings).
    """

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ: dict[str, str] = dict(os.environ if environ is None else environ)
        self._values: dict[str, EffectiveValue] = {}

    def record_effective(self, key: str, value: object, source: str) -> EffectiveValue:
        """Record the value a run will actually use, and where it came from.

        Args:
            key: Configuration key, e.g. ``"learning_rate"``. For the env-shadow
                check to be meaningful, key naming should match the corresponding
                environment variable exactly (``key`` is looked up verbatim).
            value: The resolved value; stringified for storage.
            source: ``"cli"``, ``"default"``, ``"env:NAME"`` or
                ``"config:path#key"``. Anything else raises ``ValueError`` — a
                free-text source is the 18-key string bag coming back.

        Returns:
            The recorded :class:`EffectiveValue`, including any findings.

        Raises:
            ValueError: On a malformed ``source``, or if ``key`` was already
                recorded. Double-recording a key would overwrite the earlier
                resolution — the same overwrite failure this module exists to
                prevent, one level down.
        """
        if not _SOURCE_RE.match(source):
            raise ValueError(
                f"invalid source {source!r} for key {key!r}: expected 'cli', "
                f"'default', 'env:NAME' or 'config:path#key'"
            )
        if key in self._values:
            raise ValueError(
                f"effective value for {key!r} already recorded "
                f"(source={self._values[key].source!r}); recording twice would "
                f"silently overwrite the first resolution"
            )

        str_value = str(value)
        env_value: str | None = None
        findings: list[str] = []

        if source == "default":
            # A value that resolved all the way to the library default while an
            # environment variable of the same name is set means the env var never
            # reached the resolver. In the audit this exact shape flipped the
            # objective for 12 of 24 byte-identical-argv runs; it is a finding, not
            # a footnote.
            live = self._environ.get(key)
            if live is not None:
                env_value = live
                findings.append(
                    f"env-shadowed default: {key!r} is set in the environment to "
                    f"{live!r}, but the effective value resolved to the library "
                    f"default {str_value!r} — the variable never reached the resolver"
                )
        elif source.startswith("env:"):
            name = source[len("env:") :]
            live = self._environ.get(name)
            if live is None:
                findings.append(
                    f"unverifiable env source: {source!r} names a variable that is "
                    f"not set at record time; the resolution cannot be corroborated"
                )
            else:
                env_value = live
                if live != str_value:
                    findings.append(
                        f"env-source drift: {name!r} is {live!r} at record time but "
                        f"the recorded effective value is {str_value!r} — the value "
                        f"changed between resolution and manifest write"
                    )

        ev = EffectiveValue(
            key=key,
            value=str_value,
            source=source,
            env_value=env_value,
            findings=tuple(findings),
        )
        self._values[key] = ev
        return ev

    def freeze(self) -> dict[str, EffectiveValue]:
        """Return a sorted-by-key copy of everything recorded so far."""
        return dict(sorted(self._values.items()))


# ---------------------------------------------------------------------------
# Code provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffPathCoverage:
    """What git could and could not see under one captured path.

    Attributes:
        path: The path, relative to the repo root (``.`` for the whole repo).
        exists: Whether the path existed at capture time. A capture-list entry for a
            path that does not exist is itself evidence of path drift (the audited
            ``$SDPO`` capture pointed at a subtree nothing edited).
        tracked_files: Files git tracks under this path. Zero means ``git diff``
            structurally cannot capture edits here.
        modified_tracked_files: Tracked files with staged or unstaged changes.
        untracked_files: Untracked files under this path. Every one of these is a
            file that will execute and that ``git diff HEAD`` will never contain.
        captured_files: Files the captured diff actually touched under this path.
        captured_bytes: Size in bytes of the diff restricted to this path. The
            measured defect is ``captured_bytes == 0`` with ``untracked_files > 0``.
        status: Rolled-up :class:`PathStatus`.
    """

    path: str
    exists: bool
    tracked_files: int
    modified_tracked_files: int
    untracked_files: int
    captured_files: int
    captured_bytes: int
    status: PathStatus

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "exists": self.exists,
            "tracked_files": self.tracked_files,
            "modified_tracked_files": self.modified_tracked_files,
            "untracked_files": self.untracked_files,
            "captured_files": self.captured_files,
            "captured_bytes": self.captured_bytes,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DiffPathCoverage:
        return cls(
            path=str(data["path"]),
            exists=bool(data["exists"]),
            tracked_files=_expect_int(data["tracked_files"], "tracked_files"),
            modified_tracked_files=_expect_int(
                data["modified_tracked_files"], "modified_tracked_files"
            ),
            untracked_files=_expect_int(data["untracked_files"], "untracked_files"),
            captured_files=_expect_int(data["captured_files"], "captured_files"),
            captured_bytes=_expect_int(data["captured_bytes"], "captured_bytes"),
            status=PathStatus(str(data["status"])),
        )


@dataclass(frozen=True)
class CodeProvenance:
    """The code a run executed, stated as a checkable set of claims.

    The predecessor recorded one claim — a commit hash — and it was false in 35 of
    35 bundles (same commit, three weeks older than the runs, 892 dirty files). This
    record makes the three independent claims — *which commit*, *how dirty*, *what
    differed* — separately, so no single one can impersonate the others.
    """

    status: CaptureStatus
    root: str | None
    commit: str | None
    """HEAD's full hash, or ``None`` when there is no usable repository. A commit
    without a clean/diffed tree is a false statement; always read together with
    :attr:`dirty_files`, :attr:`diff_sha256` and :attr:`status`."""

    dirty_files: int
    """Total files (staged + unstaged, tracked) plus untracked files repo-wide."""

    untracked_files: int
    """Repo-wide untracked files. Untracked files are invisible to ``git diff``;
    any nonzero value within the captured scope forces NOT_CAPTURED."""

    diff_sha256: str | None
    """SHA-256 of the exact diff bytes captured for the scoped paths."""

    diff_bytes: int
    """Byte length of that diff. ``0`` alongside a dirty tree is never "clean"."""

    paths: tuple[DiffPathCoverage, ...]
    """Per-path attribution. ``()`` when there was no repository at all."""

    entrypoint: str | None = None
    """The script actually launched, as recorded by the caller."""

    entrypoint_captured: bool | None = None
    """Whether the entrypoint lies under the capture root. The audited estate's real
    entrypoint lived under a different root and appeared in no snapshot; a
    ``False`` here is that hole, made legible."""

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "root": self.root,
            "commit": self.commit,
            "dirty_files": self.dirty_files,
            "untracked_files": self.untracked_files,
            "diff_sha256": self.diff_sha256,
            "diff_bytes": self.diff_bytes,
            "paths": [p.to_dict() for p in self.paths],
            "entrypoint": self.entrypoint,
            "entrypoint_captured": self.entrypoint_captured,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CodeProvenance:
        return cls(
            status=CaptureStatus(str(data["status"])),
            root=None if data.get("root") is None else str(data["root"]),
            commit=None if data.get("commit") is None else str(data["commit"]),
            dirty_files=_expect_int(data["dirty_files"], "code.dirty_files"),
            untracked_files=_expect_int(data["untracked_files"], "code.untracked_files"),
            diff_sha256=None if data.get("diff_sha256") is None else str(data["diff_sha256"]),
            diff_bytes=_expect_int(data["diff_bytes"], "code.diff_bytes"),
            paths=tuple(
                DiffPathCoverage.from_dict(_expect_mapping(p, "code.paths[]"))
                for p in _expect_list(data["paths"], "code.paths")
            ),
            entrypoint=None if data.get("entrypoint") is None else str(data["entrypoint"]),
            entrypoint_captured=(
                None
                if data.get("entrypoint_captured") is None
                else bool(data["entrypoint_captured"])
            ),
        )


_GIT_ENV_STRIP_PREFIX = "GIT_"
_GIT_CONFIG_ARGS: tuple[str, ...] = ("-c", "core.quotepath=false")
_GIT_DIFF_ARGS: tuple[str, ...] = ("--no-ext-diff", "--no-textconv")


def _hermetic_git_env() -> dict[str, str]:
    """The environment every git probe runs under, with every ambient say removed.

    Two ambient channels would otherwise let the launcher's context reshape the
    bytes being hashed and attested, and both are closed here:

    * **Environment.** The whole ``GIT_*`` namespace outranks ``-C`` in git's
      resolution order: ``GIT_DIR``/``GIT_WORK_TREE`` re-point every probe at a
      repository the record never names, and
      ``GIT_CONFIG_COUNT``/``GIT_CONFIG_PARAMETERS`` inject wholesale config.
      The caller's namespace is dropped in full; only the probes' own pins are
      re-added below.
    * **System/global file config.** ``/etc/gitconfig`` and ``~/.gitconfig``
      ride in on ``HOME`` and carry exactly the knobs that rewrite diff output
      (external diff drivers, ``core.autocrlf``, attribute defaults). They are
      neutralized with ``GIT_CONFIG_NOSYSTEM=1`` plus ``GIT_CONFIG_SYSTEM`` /
      ``GIT_CONFIG_GLOBAL`` pointed at the null device. Env-var pins are used
      because gits too old to know a name treat it as inert, and
      ``GIT_CONFIG_NOSYSTEM`` predates ``GIT_CONFIG_SYSTEM`` (git 2.32) by
      years, so the system file is covered on both sides of that boundary.

    **Residual channel, stated rather than hidden:** repository-local
    ``.git/config`` cannot be switched off from the environment without
    breaking the read of the very repository under observation. Its two
    highest-impact levers over diff bytes — external diff drivers and
    textconv — are disabled per invocation on the argv (see
    :func:`_git_argv`), and path quoting is pinned there; whatever else a local
    config can do to the diff lives inside the recorded root, i.e. inside the
    scope the manifest attests about. In-tree ``.gitattributes`` is likewise
    content of the attested tree, not ambient influence.
    ``GIT_OPTIONAL_LOCKS=0`` is pinned so a probe never writes into the
    repository it observes, and ``LC_ALL=C`` keeps diagnostics locale-stable.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(_GIT_ENV_STRIP_PREFIX)}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["LC_ALL"] = "C"
    return env


def _git_argv(root: Path, args: Sequence[str]) -> list[str]:
    """The one argv every probe runs — the single enforcement point for hermeticity.

    ``-c core.quotepath=false`` pins path quoting so otherwise-inherited config
    cannot vary the bytes that feed ``diff_sha256``. For ``diff`` invocations,
    ``--no-ext-diff`` and ``--no-textconv`` disable external diff drivers and
    textconv filters, both of which rewrite diff output by default once
    configured. The flags are spliced in directly after the subcommand: appended
    at the end they would land after the ``--`` pathspec separator, where git
    would read them as paths rather than options.
    """
    cmd = list(args)
    if cmd[:1] == ["diff"]:
        cmd[1:1] = _GIT_DIFF_ARGS
    return ["git", *_GIT_CONFIG_ARGS, "-C", str(root), *cmd]


def _git(root: Path, args: Sequence[str]) -> tuple[int, str] | None:
    """Run a git command, returning ``(rc, stdout)``; ``None`` on any failure.

    Returns ``None`` rather than raising for every failure mode (git absent,
    timeout, non-repo directory) so that callers degrade to ``NOT_A_REPOSITORY``
    or ``NOT_CAPTURED`` instead of crashing the launcher — and so that "the code is
    gone" is represented in the record, not lost in a traceback.
    """
    try:
        proc = subprocess.run(
            _git_argv(root, args),
            capture_output=True,
            text=True,
            encoding="utf-8",
            # Raw non-UTF-8 filenames from porcelain output must not crash capture.
            errors="surrogateescape",
            env=_hermetic_git_env(),
            timeout=_GIT_TIMEOUT_S,
        )
        return proc.returncode, proc.stdout
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _git_bytes(root: Path, args: Sequence[str]) -> bytes | None:
    """Binary variant of :func:`_git` for diff payloads, which need exact bytes."""
    try:
        proc = subprocess.run(
            _git_argv(root, args),
            capture_output=True,
            env=_hermetic_git_env(),
            timeout=_GIT_TIMEOUT_S,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _porcelain_counts(root: Path, rel_path: str) -> tuple[int, int]:
    """Return ``(modified_tracked, untracked)`` counts under ``rel_path``.

    Counts come from ``--porcelain=v1``; rename entries count as one changed file,
    which is the correct denominator for *files that differ*, not for diff hunks.
    """
    out = _git(root, ["status", "--porcelain=v1", "--untracked-files=all", "--", rel_path])
    if out is None or out[0] != 0:
        return 0, 0
    modified = untracked = 0
    for line in out[1].splitlines():
        if not line:
            continue
        if line.startswith("??"):
            untracked += 1
        else:
            modified += 1
    return modified, untracked


def _tracked_count(root: Path, rel_path: str) -> int:
    out = _git(root, ["ls-files", "-z", "--", rel_path])
    if out is None or out[0] != 0:
        return 0
    return sum(1 for name in out[1].split("\0") if name)


def _diff_bytes(root: Path, rel_path: str) -> bytes | None:
    """Diff of ``rel_path`` against HEAD, exactly as it would be stored.

    Returns ``None`` (treated as zero bytes) when the repo has no HEAD — every file
    is then untracked from git's perspective, and the correct status is
    NOT_CAPTURED, which the caller derives.
    """
    return _git_bytes(root, ["diff", "--binary", "HEAD", "--", rel_path])


def _diff_file_count(root: Path, rel_path: str) -> int:
    out = _git(root, ["diff", "--name-only", "-z", "HEAD", "--", rel_path])
    if out is None or out[0] != 0:
        return 0
    return sum(1 for name in out[1].split("\0") if name)


def capture_code_provenance(
    root: str | os.PathLike[str],
    diff_paths: Sequence[str] = (),
    *,
    entrypoint: str | os.PathLike[str] | None = None,
) -> CodeProvenance:
    """Capture git provenance for the code about to execute, without lying about it.

    Args:
        root: Repository root that is *supposed* to contain the executing code.
        diff_paths: Repo-relative paths whose changes must be captured. Empty means
            the whole repo (scope ``.``). Paths should cover everything the run
            imports or invokes: this is where the predecessor failed, diffing
            ``$SDPO`` while the living code sat in a zero-tracked-file tree.
        entrypoint: The script actually being launched. When supplied, the result
            records whether it lies under ``root`` at all — the "entrypoint outside
            every snapshot" hole.

    Returns:
        A :class:`CodeProvenance`. Never raises for git/repo problems: absence of
        git metadata is itself the provenance fact (:attr:`CaptureStatus.NOT_A_REPOSITORY`
        or :attr:`CaptureStatus.NOT_CAPTURED`).
    """
    root_path = Path(root).expanduser()
    entry_str = None if entrypoint is None else str(entrypoint)

    # Resolve the entrypoint's containment up front; it is recorded regardless of
    # whether the root turns out to be a repository.
    entry_in_root: bool | None = None
    if entrypoint is not None:
        try:
            resolved_root = root_path.resolve()
            entry_in_root = Path(entrypoint).resolve().is_relative_to(resolved_root)
        except OSError:
            entry_in_root = False

    def _not_repo() -> CodeProvenance:
        return CodeProvenance(
            status=CaptureStatus.NOT_A_REPOSITORY,
            root=str(root_path),
            commit=None,
            dirty_files=0,
            untracked_files=0,
            diff_sha256=None,
            diff_bytes=0,
            paths=(),
            entrypoint=entry_str,
            entrypoint_captured=entry_in_root,
        )

    if not root_path.is_dir():
        return _not_repo()
    probe = _git(root_path, ["rev-parse", "--is-inside-work-tree"])
    if probe is None or probe[0] != 0 or probe[1].strip() != "true":
        return _not_repo()

    head = _git(root_path, ["rev-parse", "--verify", "HEAD"])
    commit = head[1].strip() if head is not None and head[0] == 0 else None

    scope: tuple[str, ...] = tuple(diff_paths) if diff_paths else (".",)
    dirty_modified, dirty_untracked = _porcelain_counts(root_path, ".")

    per_path: list[DiffPathCoverage] = []
    combined = bytearray()
    for rel in scope:
        exists = (root_path / rel).exists()
        tracked = _tracked_count(root_path, rel)
        modified, untracked = _porcelain_counts(root_path, rel)
        blob = _diff_bytes(root_path, rel) or b""
        captured_files = _diff_file_count(root_path, rel) if commit else 0
        combined += blob

        if not exists:
            status = PathStatus.NO_SUCH_PATH
        elif untracked > 0:
            # Files that will execute but that `git diff HEAD` structurally cannot
            # contain. This is the 0-byte-patch-over-a-living-tree defect: the
            # capture "succeeded" and stored nothing.
            status = PathStatus.NOT_CAPTURED
        elif modified > 0 and not blob:
            # Tracked changes but nothing in the diff (e.g. unborn HEAD). Either
            # way the bytes that will run are not in the record.
            status = PathStatus.NOT_CAPTURED
        elif modified > 0:
            status = PathStatus.CAPTURED
        else:
            status = PathStatus.CLEAN
        per_path.append(
            DiffPathCoverage(
                path=rel,
                exists=exists,
                tracked_files=tracked,
                modified_tracked_files=modified,
                untracked_files=untracked,
                captured_files=captured_files,
                captured_bytes=len(blob),
                status=status,
            )
        )

    dirty_total = dirty_modified + dirty_untracked
    if commit is None:
        # Unborn HEAD: the least reproducible state a repository can be in — there
        # is no commit to anchor the record against, and `git diff HEAD` is empty
        # by construction. It must never read as CLEAN.
        overall = CaptureStatus.NOT_CAPTURED
    elif any(p.status in (PathStatus.NOT_CAPTURED, PathStatus.NO_SUCH_PATH) for p in per_path):
        overall = CaptureStatus.NOT_CAPTURED
    elif dirty_total == 0:
        overall = CaptureStatus.CLEAN
    elif combined:
        overall = CaptureStatus.CAPTURED
    else:
        overall = CaptureStatus.NOT_CAPTURED

    diff_sha = hashlib.sha256(bytes(combined)).hexdigest()
    return CodeProvenance(
        status=overall,
        root=str(root_path),
        commit=commit,
        dirty_files=dirty_total,
        untracked_files=dirty_untracked,
        diff_sha256=diff_sha,
        diff_bytes=len(combined),
        paths=tuple(per_path),
        entrypoint=entry_str,
        entrypoint_captured=entry_in_root,
    )


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Topology:
    """The parallel layout the run used, as *effective* integers.

    Topology decides numerics (a TP change is a different reduction order and, as
    the audit's vocab-clamp incident showed, sometimes a different *semantics*), so
    it belongs in the fingerprint.
    """

    nodes: int
    gpus_per_node: int
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    expert_parallel: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "tensor_parallel": self.tensor_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "data_parallel": self.data_parallel,
            "expert_parallel": self.expert_parallel,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Topology:
        return cls(
            nodes=_expect_int(data["nodes"], "topology.nodes"),
            gpus_per_node=_expect_int(data["gpus_per_node"], "topology.gpus_per_node"),
            tensor_parallel=_expect_int(data["tensor_parallel"], "topology.tensor_parallel"),
            pipeline_parallel=_expect_int(data["pipeline_parallel"], "topology.pipeline_parallel"),
            data_parallel=_expect_int(data["data_parallel"], "topology.data_parallel"),
            expert_parallel=(
                None
                if data.get("expert_parallel") is None
                else _expect_int(data["expert_parallel"], "topology.expert_parallel")
            ),
        )


# ---------------------------------------------------------------------------
# Declared checkpoint: the denominators the checkpoint gates adjudicate
# ---------------------------------------------------------------------------

_DECLARED_KNOWN_KEYS = frozenset(
    {
        "num_experts",
        "num_moe_layers",
        "expected_expert_bytes",
        "declared_fqns",
        "naming_convention",
        "expert_weight_pattern",
        "tensors_per_expert_layer",
        "dtype_widths",
        "moe_layer_basis",
    }
)

_KNOWN_DTYPE_WIDTHS: dict[str, int] = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}
"""Bytes per element for dtypes a config file or argument parser can name.

Deliberately mirrors ``gates.checkpoint_gates._DTYPE_BYTES``: this module is
standard-library-only, and two small tables that must each stay honest beats an
import edge that drags the gates package (and its registry side effects) into
launch-time provenance.
"""


@dataclass(frozen=True)
class DeclaredCheckpoint:
    """What this run's checkpoint was supposed to contain — the gates' denominators.

    The checkpoint gates can only say "128 experts shrank to 16" against a
    *declared* 128. Every denominator they adjudicate must therefore be produced
    at launch, from the same resolved config the trainer consumed, and recorded
    here — this block is where "declared" stops being whatever string an
    integrator typed. A gate handed no denominator does not fail open; it
    reports SKIP or VACUOUS. The block is an optional field of the v1 schema:
    manifests written before it existed load with ``declared=None`` and keep
    exactly the honest abstentions they had.

    Attributes:
        num_experts: Declared routed experts per MoE layer; ``None`` for a dense
            model.
        num_moe_layers: Layers carrying routed experts.
        expected_expert_bytes: Total expert tensor bytes the checkpoint should
            hold — the 45.70 GB the incident checkpoint never had. Computed by
            the caller from resolved shapes and ``dtype_widths``.
        declared_fqns: The full declared tensor list. Canonical wire form is
            sorted and unique; ``__post_init__`` canonicalizes so two writers
            producing the same declaration serialize identically. Empty means
            "no list was captured" — the gates read that as *no denominator*,
            never as "all zero present".
        naming_convention: One of ``"megatron-core"``, ``"hf-moe"``,
            ``"deepspeed-moe"`` or ``"custom"``.
        expert_weight_pattern: Regex naming expert tensors; required iff
            ``naming_convention == "custom"``.
        tensors_per_expert_layer: Expert weight tensors per MoE layer
            (fc1+fc2 → 2; w1/w2/w3 → 3).
        dtype_widths: Byte width per dtype name used in the expected-bytes
            arithmetic.
        moe_layer_basis: How ``num_moe_layers`` was established, in words a
            reader can audit against the raw config — explicit key, derived
            arithmetic, the all-routed convention, or a stated refusal;
            ``None`` only when this record never addressed the question. An
            unexplained denominator is an unaccountable one: the gates consume
            the integer, but a human replaying the record needs to know whether
            the number was read, derived, assumed or declined.
    """

    num_experts: int | None = None
    num_moe_layers: int | None = None
    expected_expert_bytes: int | None = None
    declared_fqns: tuple[str, ...] = ()
    naming_convention: str = "megatron-core"
    expert_weight_pattern: str | None = None
    tensors_per_expert_layer: int = 2
    dtype_widths: Mapping[str, int] = field(default_factory=dict)
    moe_layer_basis: str | None = None

    def __post_init__(self) -> None:
        for name in ("num_experts", "num_moe_layers", "expected_expert_bytes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"declared.{name} must be a positive int or None, got {value!r}")
        object.__setattr__(self, "declared_fqns", tuple(sorted(set(self.declared_fqns))))
        for dtype, width in self.dtype_widths.items():
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                raise ValueError(
                    f"declared.dtype_widths[{dtype!r}] must be a positive byte width, got {width!r}"
                )
        if isinstance(self.tensors_per_expert_layer, bool) or self.tensors_per_expert_layer < 1:
            raise ValueError(
                f"declared.tensors_per_expert_layer must be >= 1, got "
                f"{self.tensors_per_expert_layer!r}"
            )
        if self.naming_convention == "custom" and not self.expert_weight_pattern:
            raise ValueError(
                "naming_convention='custom' requires expert_weight_pattern — an "
                "uncheckable convention is a SKIP by another name"
            )
        object.__setattr__(self, "dtype_widths", MappingProxyType(dict(self.dtype_widths)))

    def to_dict(self) -> dict[str, object]:
        return {
            "num_experts": self.num_experts,
            "num_moe_layers": self.num_moe_layers,
            "expected_expert_bytes": self.expected_expert_bytes,
            "declared_fqns": list(self.declared_fqns),
            "naming_convention": self.naming_convention,
            "expert_weight_pattern": self.expert_weight_pattern,
            "tensors_per_expert_layer": self.tensors_per_expert_layer,
            "dtype_widths": dict(self.dtype_widths),
            "moe_layer_basis": self.moe_layer_basis,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DeclaredCheckpoint:
        """Rebuild the block, refusing keys this reader does not understand.

        Denominator fields are held to a stricter standard than the rest of the
        document (where unknown keys are tolerated and surfaced as findings): a
        silently ignored ``"num_expert"`` (sic) would load as
        ``num_experts=None`` and turn every expert gate into a SKIP that looks
        like a configuration gap. Unknown keys here raise at the boundary.
        So do denominators that fail validation: through :func:`load`, a zero
        expert count or a negative byte width is corrupt stored content and
        surfaces as :class:`ManifestError` (chaining the ``ValueError``),
        never as that bare ``ValueError`` — callers route failures by catching
        this module's own error type, and a contractually routed failure must
        not require a second, undocumented catch.
        """
        unknown = sorted(str(key) for key in data if str(key) not in _DECLARED_KNOWN_KEYS)
        if unknown:
            raise ManifestError(
                f"corrupt manifest: unknown declared keys {unknown!r} — a "
                f"denominator this reader cannot name is one it cannot vouch for"
            )

        def _optional_int(field_name: str) -> int | None:
            raw = data.get(field_name)
            return None if raw is None else _expect_int(raw, f"declared.{field_name}")

        try:
            return cls(
                num_experts=_optional_int("num_experts"),
                num_moe_layers=_optional_int("num_moe_layers"),
                expected_expert_bytes=_optional_int("expected_expert_bytes"),
                declared_fqns=tuple(
                    str(fqn)
                    for fqn in _expect_list(data.get("declared_fqns", []), "declared.declared_fqns")
                ),
                naming_convention=str(data.get("naming_convention", "megatron-core")),
                expert_weight_pattern=(
                    None
                    if data.get("expert_weight_pattern") is None
                    else str(data["expert_weight_pattern"])
                ),
                tensors_per_expert_layer=_expect_int(
                    data.get("tensors_per_expert_layer", 2),
                    "declared.tensors_per_expert_layer",
                ),
                dtype_widths={
                    str(dtype): _expect_int(width, f"declared.dtype_widths[{dtype!r}]")
                    for dtype, width in _expect_mapping(
                        data.get("dtype_widths", {}), "declared.dtype_widths"
                    ).items()
                },
                moe_layer_basis=(
                    None if data.get("moe_layer_basis") is None else str(data["moe_layer_basis"])
                ),
            )
        except ValueError as exc:
            # __post_init__ raising ValueError is correct at *build* time, where
            # the caller still holds resolver context to fix. Through from_dict
            # the same condition is corrupt stored content, which the load()
            # contract promises as ManifestError; re-raising under this module's
            # own type keeps one failure class in one taxonomy.
            raise ManifestError(f"corrupt manifest: declared is invalid: {exc}") from exc


_NESTED_LM_SCOPE_KEY = "text_config"
"""Where multimodal HF configs nest their language model.

Measured on a production Gemma-4 26B-A4B ``config.json`` (the 48 GiB
checkpoint): ``text_config.num_experts`` was 128 while the top level carried
no expert key at all. A producer that reads only the top level declares that
MoE *dense* — not a smaller claim but a false one, and one that silently
disarms every expert gate.
"""

_EXPERT_COUNT_KEYS: tuple[str, ...] = ("num_local_experts", "n_routed_experts", "num_experts")
"""Routed-expert count keys this producer understands, in precedence order."""

_UNMODELED_SPARSITY_KEYS: tuple[str, ...] = (
    "moe_layer_freq",
    "decoder_sparse_step",
    "mlp_only_layers",
)
"""Keys announcing a sparse/dense layer interleave this producer does not model.

Any of them present forces ``num_moe_layers=None``: a missing denominator
makes the gates abstain loudly (SKIP), where a fabricated one — "else every
hidden layer" — makes them adjudicate against fiction and lie quietly.
"""


def _resolve_dtype_widths(
    dtype_names: Sequence[str],
    dtype_widths: Mapping[str, int] | None,
    owner: str,
) -> Mapping[str, int]:
    """Price each named dtype in bytes per element, refusing any without a width.

    An unpriced dtype would let ``expected_expert_bytes`` arithmetic proceed on
    a guessed element size — a false number wearing a computed look.
    """
    table = dict(_KNOWN_DTYPE_WIDTHS)
    if dtype_widths:
        table.update(dtype_widths)
    widths: dict[str, int] = {}
    for name in dtype_names:
        if name not in table:
            raise ValueError(
                f"{owner} uses dtype {name!r}, whose byte width is unknown; pass "
                f"dtype_widths={{{name!r}: <bytes-per-element>}} — expert-byte "
                "arithmetic over an unpriced dtype is a false number"
            )
        widths[name] = table[name]
    return widths


def _resolve_hf_moe_depth(
    scope: Mapping[str, object],
    num_hidden_layers: int,
    scope_prefix: str,
    owner: str,
) -> tuple[int | None, str]:
    """Routed-layer depth for a config scope known to declare routed experts.

    Returns ``(depth, basis)``. ``depth`` is ``None`` — a first-class
    abstention, which the gates read as SKIP — whenever the scope carries a
    sparsity key this producer cannot honour arithmetically. The doctrine
    applies directly: a wrong positive integer makes a gate lie quietly, while
    an absent denominator makes it abstain loudly. ``basis`` always records, in
    words auditable against the raw config, which of the three outcomes fired:
    arithmetic over ``first_k_dense_replace``, refusal over unmodeled
    interleave keys, or the all-routed convention.
    """
    dense_prefix = scope.get("first_k_dense_replace")
    if dense_prefix is not None:
        if isinstance(dense_prefix, bool) or not isinstance(dense_prefix, int) or dense_prefix < 0:
            raise ValueError(
                f"{owner}: {scope_prefix}first_k_dense_replace must be a "
                f"non-negative int, got {dense_prefix!r}"
            )
        if dense_prefix >= num_hidden_layers:
            raise ValueError(
                f"{owner}: {scope_prefix}first_k_dense_replace ({dense_prefix}) "
                f"consumes all {num_hidden_layers} hidden layers, yet routed "
                f"experts are declared — a self-contradicting config is refused "
                f"here, not averaged into a denominator"
            )
        return (
            num_hidden_layers - dense_prefix,
            f"{scope_prefix}num_hidden_layers({num_hidden_layers}) - "
            f"{scope_prefix}first_k_dense_replace({dense_prefix})",
        )
    unmodeled = sorted(key for key in _UNMODELED_SPARSITY_KEYS if scope.get(key) is not None)
    if unmodeled:
        where = scope_prefix.rstrip(".") or "top level"
        return None, (
            f"abstained: {where} declares {', '.join(unmodeled)}; the interleave "
            f"is real but unmodeled, so the depth is refused rather than invented"
        )
    return num_hidden_layers, f"{scope_prefix}num_hidden_layers (all layers routed)"


def declared_from_hf_config(
    config: str | os.PathLike[str] | Mapping[str, object],
    *,
    declared_fqns: Sequence[str] = (),
    tensors_per_expert_layer: int = 3,
    dtype_widths: Mapping[str, int] | None = None,
    expected_expert_bytes: int | None = None,
) -> DeclaredCheckpoint:
    """Build the declared block from a Hugging Face ``config.json``.

    The routed-expert count comes from ``num_local_experts`` /
    ``n_routed_experts`` / ``num_experts`` (whichever the architecture names).
    Multimodal configs nest the language model under ``text_config`` — measured
    on a production Gemma-4 26B-A4B ``config.json``: the model's 128 experts
    lived exclusively at ``text_config.num_experts`` while the top level named
    no expert at all, and a top-level-only search returned a *dense*
    declaration for a 128-expert MoE. The nested scope is therefore searched
    before the top level, and every depth fact is resolved inside whichever
    single scope yielded the expert count — never stitched across scopes.

    Depth is where this helper refuses to manufacture evidence. An explicit
    ``num_moe_layers`` is honoured verbatim. ``first_k_dense_replace``
    (DeepSeek-V2/V3: the first N layers are dense) is honoured arithmetically
    as ``num_hidden_layers - N``. ``moe_layer_freq`` / ``decoder_sparse_step`` /
    ``mlp_only_layers`` announce a sparse/dense interleave this producer does
    not model; their presence forces ``num_moe_layers=None``, an abstention
    every gate reads as SKIP — the previous "else every ``num_hidden_layers``
    layer" rule was no HF default but a fabricated denominator, the founding
    incident's overstated expected volume handed back to the gates wearing a
    computed look. Only with none of those keys in scope does depth mean "every
    hidden layer" (the Mixtral/Qwen-style all-routed layout), and whatever was
    resolved — or declined — is recorded in words under
    :attr:`DeclaredCheckpoint.moe_layer_basis`, so a reader can audit where the
    number came from instead of merely trusting that it exists.

    A config with no routed-expert key in either scope — an explicit ``null``
    counts as *no* key — yields a dense declaration (``num_experts=None``),
    not a guess.

    Args:
        config: Path to ``config.json``, or an already-parsed mapping.
        declared_fqns: Declared tensor list, when captured (see
            :func:`capture_state_dict_keys`); empty means "not captured".
        tensors_per_expert_layer: Expert weight tensors per MoE layer. Defaults
            to 3 (``gate_proj``/``up_proj``/``down_proj``, the HF layout).
        dtype_widths: Extra or overriding byte widths per dtype name.
        expected_expert_bytes: Precomputed declared expert volume, when the
            caller has resolved shapes; ``None`` leaves that denominator for the
            trainer to supply later.

    Raises:
        ValueError: The config names a dtype whose byte width is unknown, or an
            expert/layer count that is not a positive integer.
    """
    if isinstance(config, (str, os.PathLike)):
        owner = f"HF config {config}"
        parsed = json.loads(Path(config).read_text(encoding="utf-8"))
        if not isinstance(parsed, Mapping):
            raise ValueError(f"{owner} does not parse to a JSON object")
        data: Mapping[str, object] = parsed
    else:
        owner = "HF config"
        data = config

    # Multimodal configs nest the language model under ``text_config``. Measured
    # on a production Gemma-4 26B-A4B config.json: the top level names no expert
    # key at all, so a top-level-only search returned a *dense* declaration for
    # a 128-expert MoE — a declaration that disarms every expert gate silently.
    # The nested scope is searched first; depth is then resolved strictly inside
    # whichever scope yielded the expert count.
    scopes: list[tuple[str, Mapping[str, object]]] = []
    nested = data.get(_NESTED_LM_SCOPE_KEY)
    if isinstance(nested, Mapping):
        scopes.append((_NESTED_LM_SCOPE_KEY, nested))
    scopes.append(("", data))

    num_experts: int | None = None
    scope_name = ""
    scope: Mapping[str, object] = data
    for candidate_name, candidate in scopes:
        found_key = next(
            (key for key in _EXPERT_COUNT_KEYS if candidate.get(key) is not None),
            None,
        )
        if found_key is None:
            continue
        raw_experts = candidate[found_key]
        if isinstance(raw_experts, bool) or not isinstance(raw_experts, int) or raw_experts <= 0:
            prefix = f"{candidate_name}." if candidate_name else ""
            raise ValueError(
                f"{owner}.{prefix}{found_key} must be a positive int, got {raw_experts!r}"
            )
        num_experts = raw_experts
        scope_name = candidate_name
        scope = candidate
        break

    num_moe_layers: int | None = None
    moe_layer_basis: str | None = None
    if num_experts is not None:
        scope_prefix = f"{scope_name}." if scope_name else ""
        raw_layers = scope.get("num_moe_layers")
        raw_hidden = scope.get("num_hidden_layers")
        if raw_layers is not None:
            if isinstance(raw_layers, bool) or not isinstance(raw_layers, int) or raw_layers <= 0:
                raise ValueError(
                    f"{owner} MoE layer count must be a positive int, got {raw_layers!r}"
                )
            num_moe_layers = raw_layers
            moe_layer_basis = f"{scope_prefix}num_moe_layers"
        elif raw_hidden is not None:
            if isinstance(raw_hidden, bool) or not isinstance(raw_hidden, int) or raw_hidden <= 0:
                raise ValueError(
                    f"{owner} MoE layer count must be a positive int, got {raw_hidden!r}"
                )
            num_moe_layers, moe_layer_basis = _resolve_hf_moe_depth(
                scope, raw_hidden, scope_prefix, owner
            )

    dtype_names: tuple[str, ...] = ()
    for _, candidate in scopes:
        raw_dtype = candidate.get("torch_dtype")
        if raw_dtype is None:
            raw_dtype = candidate.get("dtype")
        if raw_dtype is not None:
            dtype_names = (str(raw_dtype),)
            break

    return DeclaredCheckpoint(
        num_experts=num_experts,
        num_moe_layers=num_moe_layers,
        expected_expert_bytes=expected_expert_bytes,
        declared_fqns=tuple(declared_fqns),
        naming_convention="hf-moe",
        tensors_per_expert_layer=tensors_per_expert_layer,
        dtype_widths=_resolve_dtype_widths(dtype_names, dtype_widths, owner),
        moe_layer_basis=moe_layer_basis,
    )


def declared_from_megatron_args(
    args: object,
    *,
    declared_fqns: Sequence[str] = (),
    dtype_widths: Mapping[str, int] | None = None,
    expected_expert_bytes: int | None = None,
) -> DeclaredCheckpoint:
    """Build the declared block from resolved Megatron-style arguments.

    Duck-typed against any attribute object (an ``argparse.Namespace`` works —
    no torch, no Megatron import). Reads ``num_experts`` (absent or ``None``
    means dense), the MoE depth from ``num_moe_layers`` else ``num_layers``, and
    the precision from the ``bf16`` / ``fp16`` flags (default ``float32``).
    Mirroring the *resolved* args, not the recipe file, is what keeps declared
    from diverging from effective: the resolver already did the work, and this
    records its answer.
    """
    owner = "Megatron args"
    num_experts = getattr(args, "num_experts", None)
    if num_experts is not None and (
        isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts <= 0
    ):
        raise ValueError(f"{owner}.num_experts must be a positive int, got {num_experts!r}")
    num_moe_layers: int | None = None
    moe_layer_basis: str | None = None
    if num_experts is not None:
        num_moe_layers = getattr(args, "num_moe_layers", None) or getattr(args, "num_layers", None)
        if num_moe_layers is not None:
            moe_layer_basis = (
                "num_moe_layers" if getattr(args, "num_moe_layers", None) else "num_layers"
            )
    if getattr(args, "bf16", False):
        dtype_names = ("bfloat16",)
    elif getattr(args, "fp16", False):
        dtype_names = ("float16",)
    else:
        dtype_names = ("float32",)
    return DeclaredCheckpoint(
        num_experts=num_experts,
        num_moe_layers=num_moe_layers,
        expected_expert_bytes=expected_expert_bytes,
        declared_fqns=tuple(declared_fqns),
        naming_convention="megatron-core",
        tensors_per_expert_layer=2,
        dtype_widths=_resolve_dtype_widths(dtype_names, dtype_widths, owner),
        moe_layer_basis=moe_layer_basis,
    )


def capture_state_dict_keys(state_dict: object) -> tuple[str, ...]:
    """Declared tensor names from a live state dict, minus metadata blobs.

    The 26B estate's checkpoint metadata held ~8,970 keys of which ~8,042 were
    ``_extra_state`` bookkeeping blobs; a declared list that counts keys lets a
    completeness check pass on a gutted checkpoint. Duck-typed so launch-time
    capture needs no torch import here: pass the mapping returned by
    ``model.state_dict()``, or the model itself (its ``state_dict()`` is
    called). The result is the canonical wire form: sorted and unique.
    """
    accessor = getattr(state_dict, "state_dict", None)
    if callable(accessor):
        state_dict = accessor()
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "capture_state_dict_keys needs a state-dict mapping or a model with "
            f"state_dict(), got {type(state_dict).__name__}"
        )
    return tuple(sorted({str(key) for key in state_dict if "_extra_state" not in str(key)}))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Difference:
    """One field-level semantic disagreement between two manifests.

    This is the answer shape the 24-run audit needed: not "the manifests differ"
    but ``environment.values.OBJECTIVE_SWITCH: 'dense' -> 'moe'``.
    """

    field: str
    self_value: object
    other_value: object

    def __str__(self) -> str:
        return (
            f"{self.field}: {self.value_repr(self.self_value)} -> "
            f"{self.value_repr(self.other_value)}"
        )

    @staticmethod
    def value_repr(v: object) -> str:
        return "<absent>" if v is _ABSENT else repr(v)


_ABSENT = object()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _semantic_diff(
    prefix: str,
    a: object,
    b: object,
    out: list[Difference],
) -> None:
    """Recursive, absent-aware comparison over JSON-shaped structures."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for key in sorted(set(a) | set(b)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in a:
                out.append(Difference(child, _ABSENT, b[key]))
            elif key not in b:
                out.append(Difference(child, a[key], _ABSENT))
            else:
                _semantic_diff(child, a[key], b[key], out)
    elif a != b:
        out.append(Difference(prefix, a, b))


@dataclass(frozen=True)
class RunManifest:
    """The complete, structured provenance record of one launch attempt.

    Replaces the predecessor's flat 18-key all-string dict (5 of whose keys were
    empty in 33 of 35 bundles) with typed, nested provenance that is either present
    and meaningful or absent in a way that surfaces as a finding.

    Construct this *after* config resolution, code capture and environment capture,
    then hand it to :meth:`ManifestStore.save`. ``attempt`` should come from
    :meth:`ManifestStore.allocate_attempt` so relaunches never collide.

    Attributes:
        run_id: Caller's run identifier. Shared across relaunch attempts.
        attempt: 1-based attempt number within ``run_id``.
        code: What code will execute — see :class:`CodeProvenance`.
        config: Effective configuration values (key → :class:`EffectiveValue`),
            from :meth:`ConfigResolver.freeze`.
        environment: The captured environment slice, with its scope.
        topology: The parallel layout.
        job_id: Scheduler job id when known (e.g. ``$SLURM_JOB_ID``); excluded from
            the fingerprint — it identifies the launch, not the computation.
        created_at: ISO-8601 UTC timestamp of manifest creation; excluded from the
            fingerprint for the same reason.
        artifact_paths: Output locations (checkpoint dir, log dir, ...); excluded
            from the fingerprint: two runs writing to different paths may still be
            the same computation.
        declared: The checkpoint denominators this run declared it would be
            judgeable against; ``None`` for manifests predating the block, whose
            gates keep their honest SKIP/VACUOUS abstentions. Deliberately
            excluded from the fingerprint: it names *verification targets*, and
            the computation those verify is already captured in config — keeping
            it out holds fingerprints stable across stores that never wrote the
            block.
        schema_version: Must equal :data:`SCHEMA_VERSION`.
        findings: Derived at construction (``__post_init__``); see
            :meth:`_derive_findings`.
    """

    run_id: str
    attempt: int
    code: CodeProvenance
    config: Mapping[str, EffectiveValue]
    environment: CapturedEnvironment
    topology: Topology
    job_id: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    declared: DeclaredCheckpoint | None = None
    schema_version: int = SCHEMA_VERSION
    findings: tuple[str, ...] = field(init=False, default=())
    _loaded_extra_keys: tuple[tuple[str, tuple[str, ...]], ...] = field(
        init=False, default=(), repr=False, compare=False
    )
    """Loader bookkeeping, not content: ``(scope, keys)`` pairs naming fields a
    newer writer emitted and this reader could not interpret. Excluded from
    equality, from :meth:`to_dict`, and from :meth:`fingerprint`; its only outward
    expression is one finding per scope in :attr:`findings`."""

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if self.attempt < 1:
            raise ValueError(f"attempt is 1-based, got {self.attempt}")
        if self.schema_version != SCHEMA_VERSION:
            # Construction-time rather than load-time, so a caller running mixed
            # module versions fails at the write site, where the context still
            # exists, instead of at some later reader.
            raise ManifestVersionError(
                f"cannot build manifest with schema_version {self.schema_version}; "
                f"this module supports {SCHEMA_VERSION}",
                found=self.schema_version,
                supported=SUPPORTED_SCHEMA_VERSIONS,
            )
        object.__setattr__(self, "findings", tuple(self._derive_findings()))

    # -- findings ---------------------------------------------------------------

    def _derive_findings(self) -> list[str]:
        """Collect every anomaly this manifest knows about its own provenance.

        Findings are the "negative control" of a manifest: each one names something
        the record *cannot* vouch for, so a clean manifest is a strong claim rather
        than an empty one.
        """
        findings: list[str] = []

        code = self.code
        if code.status is CaptureStatus.NOT_A_REPOSITORY:
            findings.append(
                f"code provenance unavailable: {code.root!r} is not a git repository; "
                f"no commit, diff or dirty state could be recorded"
            )
        elif code.status is CaptureStatus.NOT_CAPTURED:
            for p in code.paths:
                if p.status is PathStatus.NO_SUCH_PATH:
                    findings.append(
                        f"capture path {p.path!r} does not exist under {code.root!r}; "
                        f"it cannot have contributed to the snapshot"
                    )
                elif p.status is PathStatus.NOT_CAPTURED:
                    findings.append(
                        f"code under {p.path!r} is NOT CAPTURED: {p.untracked_files} "
                        f"untracked / {p.modified_tracked_files} modified tracked / "
                        f"{p.captured_bytes} diff bytes — a 0-byte patch over "
                        f"untracked files is not a clean tree"
                    )
        if code.entrypoint_captured is False:
            findings.append(
                f"entrypoint {code.entrypoint!r} lies outside the provenance root "
                f"{code.root!r}; it appears in no snapshot — the exact hole through "
                f"which the audited estate's real launcher escaped capture"
            )

        for ev in self.config.values():
            findings.extend(f"{ev.key}: {f}" for f in ev.findings)

        if not self.environment.allowlist:
            findings.append(
                "environment capture declared no allowlist at all: the capture has "
                "no stated scope and proves nothing about the submit environment"
            )
        for scope, keys in self._loaded_extra_keys:
            findings.append(
                f"loader ignored unknown {scope} keys: {list(keys)!r} — the field "
                f"was written by a schema this reader does not fully know, and its "
                f"content formed no part of any claim in this record"
            )
        return findings

    # -- semantic identity --------------------------------------------------------

    def _semantic_payload(self) -> dict[str, object]:
        """Everything that determines what the run computes, and nothing else.

        Deliberately excluded: ``run_id``, ``attempt``, ``job_id``,
        ``created_at``, ``artifact_paths`` — none of these change the training
        mathematics. Two runs whose payloads are equal may be redundant; two whose
        payloads differ *would train differently*, which is the contract
        :meth:`fingerprint` is for.
        """
        return {
            "code": {
                "status": self.code.status.value,
                "commit": self.code.commit,
                "dirty_files": self.code.dirty_files,
                "diff_sha256": self.code.diff_sha256,
                "diff_bytes": self.code.diff_bytes,
                "entrypoint_captured": self.code.entrypoint_captured,
                "paths": [
                    {
                        "path": p.path,
                        "status": p.status.value,
                        "tracked_files": p.tracked_files,
                        "untracked_files": p.untracked_files,
                        "modified_tracked_files": p.modified_tracked_files,
                        "captured_files": p.captured_files,
                    }
                    for p in self.code.paths
                ],
            },
            "config": {
                key: {"value": ev.value, "source": ev.source}
                for key, ev in sorted(self.config.items())
            },
            "environment": {
                "allowlist": list(self.environment.allowlist),
                "values": dict(sorted(self.environment.values.items())),
            },
            "topology": self.topology.to_dict(),
        }

    def fingerprint(self) -> str:
        """Content hash over the semantically significant fields only.

        Returns:
            Lowercase hex SHA-256. Two manifests with equal fingerprints are the
            same computation (possibly relaunches or output-path variants); two
            that would train differently — different objective switch, different
            diff, different topology — cannot collide without a hash collision in
            SHA-256 itself.
        """
        canonical = json.dumps(
            self._semantic_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def differs_from(self, other: RunManifest) -> list[Difference]:
        """Field-level semantic comparison against another manifest.

        Args:
            other: The manifest to compare against.

        Returns:
            One :class:`Difference` per differing semantic field, sorted by field
            path; ``[]`` iff the fingerprints match. This is the query the 24
            byte-identical-argv runs required: not "are these the same" (they were
            not) but *where* they diverged (one unrecorded env var).
        """
        diffs: list[Difference] = []
        _semantic_diff("", self._semantic_payload(), other._semantic_payload(), diffs)
        return diffs

    # -- serialization -------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "fingerprint": self.fingerprint(),
            "code": self.code.to_dict(),
            "config": {k: v.to_dict() for k, v in sorted(self.config.items())},
            "environment": self.environment.to_dict(),
            "topology": self.topology.to_dict(),
            "artifact_paths": dict(self.artifact_paths),
            "declared": None if self.declared is None else self.declared.to_dict(),
            "findings": list(self.findings),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RunManifest:
        """Rebuild a manifest from parsed JSON, recomputing findings from scratch.

        Serialized findings are ignored on load: they are derivable, and trusting a
        stored copy would let a hand-edited manifest erase its own blemishes.

        Keys this reader does not know — at any level of the document — are
        tolerated rather than refused, because an append-only store must stay
        readable beside a slightly newer writer; but they are never dropped
        invisibly. Each unknown key is stashed on :attr:`_loaded_extra_keys` (off
        the fingerprint, off the reserialization) and resurfaced as a finding, so
        an un-interpreted field is a stated limitation of the read rather than a
        silent hole in it — the difference between "the denominators were never
        written" and "this reader could not see them".
        """
        try:
            code_data = _expect_mapping(data["code"], "code")
            environment_data = _expect_mapping(data["environment"], "environment")
            topology_data = _expect_mapping(data["topology"], "topology")
            config_data = _expect_mapping(data["config"], "config")

            extras: list[tuple[str, tuple[str, ...]]] = []
            for scope, mapping, known in (
                ("top-level", data, _MANIFEST_KNOWN_KEYS),
                ("code", code_data, _CODE_KNOWN_KEYS),
                ("environment", environment_data, _ENVIRONMENT_KNOWN_KEYS),
                ("topology", topology_data, _TOPOLOGY_KNOWN_KEYS),
            ):
                unknown = _unknown_keys(mapping, known)
                if unknown:
                    extras.append((scope, unknown))
            for index, raw_path in enumerate(_expect_list(code_data["paths"], "code.paths")):
                path_data = _expect_mapping(raw_path, "code.paths[]")
                unknown = _unknown_keys(path_data, _PATH_KNOWN_KEYS)
                if unknown:
                    extras.append((f"code path {path_data.get('path', index)!r}", unknown))
            for key, raw_value in config_data.items():
                value_data = _expect_mapping(raw_value, f"config[{key!r}]")
                unknown = _unknown_keys(value_data, _EFFECTIVE_VALUE_KNOWN_KEYS)
                if unknown:
                    extras.append((f"config entry {key!r}", unknown))

            record = cls(
                run_id=str(data["run_id"]),
                attempt=_expect_int(data["attempt"], "attempt"),
                code=CodeProvenance.from_dict(code_data),
                config={
                    str(k): EffectiveValue.from_dict(_expect_mapping(v, f"config[{k!r}]"))
                    for k, v in config_data.items()
                },
                environment=CapturedEnvironment.from_dict(environment_data),
                topology=Topology.from_dict(topology_data),
                job_id=None if data.get("job_id") is None else str(data["job_id"]),
                created_at=str(data.get("created_at", "")),
                artifact_paths={
                    str(k): str(v)
                    for k, v in _expect_mapping(
                        data.get("artifact_paths", {}), "artifact_paths"
                    ).items()
                },
                declared=(
                    None
                    if data.get("declared") is None
                    else DeclaredCheckpoint.from_dict(_expect_mapping(data["declared"], "declared"))
                ),
                schema_version=_expect_int(data["schema_version"], "schema_version"),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ManifestError(f"corrupt manifest: missing or malformed field ({exc!r})") from exc
        if extras:
            # Findings were derived before the extras existed; re-derive so the
            # loaded record states what it could not honour.
            object.__setattr__(record, "_loaded_extra_keys", tuple(extras))
            object.__setattr__(record, "findings", tuple(record._derive_findings()))
        return record


# ---------------------------------------------------------------------------
# Store: never overwrite, always atomic
# ---------------------------------------------------------------------------


def _safe_component(name: str) -> str:
    """Map a run id to a safe single path component.

    Run ids arrive from user flags and job names; sanitizing them here is what lets
    ``run_id`` stay free-form without letting ``../`` wander out of the store.
    """
    safe = _PATH_SAFE.sub("_", name).strip("._")
    if not safe:
        raise ValueError(f"run id {name!r} has no filesystem-safe characters")
    return safe


class ManifestStore:
    """Append-only, attempt-keyed manifest storage.

    Layout: ``<root>/<run_id>/attempt-NNNN.json`` — one slot per attempt number.
    The identity check happens at write time: a byte-identical retry converges onto
    the existing record, while different content claiming the same slot raises
    :class:`ManifestExistsError` naming both fingerprints.

    Args:
        root: Store root directory. Created lazily on first write; reads against a
            missing root behave as "no manifests", which :func:`require_manifest`
            then rejects.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _run_dir(self, run_id: str) -> Path:
        return self.root / _safe_component(run_id)

    def allocate_attempt(self, run_id: str) -> int:
        """Return the next attempt number for ``run_id`` (1 if never launched).

        Call this *before* building the manifest so the attempt lands inside the
        record, then :meth:`save`. A racing retry with identical content converges
        onto the existing record; conflicting content in the same slot raises.
        """
        highest = 0
        run_dir = self._run_dir(run_id)
        if run_dir.is_dir():
            for entry in run_dir.iterdir():
                match = _ATTEMPT_RE.match(entry.name)
                if match:
                    highest = max(highest, int(match.group(1)))
        return highest + 1

    def save(self, manifest: RunManifest) -> Path:
        """Write ``manifest`` atomically, refusing to overwrite anything.

        The write is temp-file + ``os.link`` rather than temp + ``os.replace``:
        ``replace`` overwrites, and overwrite-in-place is precisely how 27 of 62
        launches' manifests disappeared in the audited estate. ``os.link`` fails
        rather than clobbering an existing record.

        Args:
            manifest: The manifest to persist. Use
                :meth:`allocate_attempt` first so re-launches get fresh slots.

        Returns:
            The path written — or, for a byte-identical retry, the existing path.
            Identical retries converge onto the record already on disk rather than
            failing or clobbering it.

        Raises:
            ManifestExistsError: A record for this attempt with different content
                already exists — i.e. two computations claim the same slot. Never
                silently resolved.
        """
        run_dir = self._run_dir(manifest.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        final = run_dir / f"attempt-{manifest.attempt:04d}.json"
        payload = manifest.to_json().encode("utf-8")
        tmp = run_dir / f".{final.name}.tmp.{os.getpid()}"
        with tmp.open("wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, final)
        except FileExistsError as exc:
            if final.read_bytes() == payload:
                # Byte-identical retry: the record on disk *is* this manifest.
                # Converge onto it — no error, no clobber.
                return final
            try:
                existing_fp = load(final).fingerprint()[:12]
            except ManifestError:
                existing_fp = "<unparseable>"
            raise ManifestExistsError(
                f"refusing to overwrite {final}: attempt {manifest.attempt} of run "
                f"{manifest.run_id!r} already has a *different* manifest on disk "
                f"(existing fingerprint {existing_fp}, new fingerprint "
                f"{manifest.fingerprint()[:12]}). Use allocate_attempt() so "
                f"relaunches get their own record."
            ) from exc
        finally:
            tmp.unlink(missing_ok=True)
        return final

    def attempts(self, run_id: str) -> tuple[RunManifest, ...]:
        """Load every recorded launch attempt of ``run_id``, oldest first.

        Returns:
            All attempts, sorted by attempt number; empty tuple when the run id has
            no records. This is the list of which "35 bundles for 62 launches" was
            one possible, lossy answer: here it is exhaustive by construction.
        """
        run_dir = self._run_dir(run_id)
        if not run_dir.is_dir():
            return ()
        records: list[tuple[int, RunManifest]] = []
        for entry in run_dir.iterdir():
            match = _ATTEMPT_RE.match(entry.name)
            if match:
                records.append((int(match.group(1)), load(entry)))
        return tuple(m for _, m in sorted(records, key=lambda r: r[0]))

    def latest(self, run_id: str) -> RunManifest:
        """The most recent attempt, or raise :class:`ManifestMissing`."""
        all_attempts = self.attempts(run_id)
        if not all_attempts:
            raise ManifestMissing(f"no provenance manifests recorded for run {run_id!r}")
        return all_attempts[-1]


def _upgrade_to_current(data: Mapping[str, object], *, from_version: int) -> Mapping[str, object]:
    """Lift an accepted manifest mapping to the current schema, in memory.

    Version 1 is the only accepted version today, so this is the identity. It is
    nonetheless the one seam where any future vN → vN+1 lift belongs: if
    compatibility shims accrete field by field instead, the version marker stops
    describing the file — the same decay by which the predecessor's flat string
    bag kept "loading" while half its keys meant nothing. Stored manifests stay
    readable by construction, not by accumulated special cases.

    ``from_version`` is checked rather than merely accepted. A lift that ignores
    which version it is lifting *from* is the identity dressed as a migration:
    the day a v2 is accepted at the boundary, an absent branch here must be a
    refusal, not a silent pass-through of foreign fields.

    Raises:
        ManifestVersionError: No lift is defined from ``from_version``.
    """
    if from_version != SCHEMA_VERSION:
        raise ManifestVersionError(
            f"no upgrade path from schema version {from_version} to "
            f"{SCHEMA_VERSION}; the version was accepted at the load boundary "
            "but no lift was written for it",
            found=from_version,
            supported=SUPPORTED_SCHEMA_VERSIONS,
        )
    return data


def load(path: str | os.PathLike[str]) -> RunManifest:
    """Load a manifest from disk, refusing versions this reader cannot interpret.

    The acceptance check types *and* ranges the version at the boundary. A bool,
    float, string or absent ``schema_version`` is a version problem and raises
    :class:`ManifestVersionError` here — letting JSON ``true`` or ``1.0`` slip
    past the boundary (both equal ``1`` in Python) to die later as a field-shape
    :class:`ManifestError` would give one failure class two taxonomies, and
    callers cannot route what they cannot name.

    Args:
        path: Manifest file written by :meth:`ManifestStore.save`.

    Returns:
        The parsed :class:`RunManifest`.

    Raises:
        ManifestVersionError: The file declares a schema version outside
            :data:`SUPPORTED_SCHEMA_VERSIONS`, or one not even interpretable as a
            version. Carries machine-readable ``found``/``supported`` attributes.
        ManifestError: Unparseable or incomplete content.
    """
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest at {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ManifestError(f"manifest at {path} is not a JSON object")
    version = data.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ManifestVersionError(
            f"refusing to load manifest at {path}: schema_version must be an "
            f"integer, got {version!r} — a version the reader cannot even type "
            f"is one it cannot honestly parse",
            found=version,
            supported=SUPPORTED_SCHEMA_VERSIONS,
        )
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ManifestVersionError(
            f"refusing to load manifest at {path}: schema_version={version!r}, "
            f"supported={SUPPORTED_SCHEMA_VERSIONS}",
            found=version,
            supported=SUPPORTED_SCHEMA_VERSIONS,
        )
    return RunManifest.from_dict(_upgrade_to_current(data, from_version=version))


def require_manifest(
    store: ManifestStore | str | os.PathLike[str],
    run_id: str,
) -> RunManifest:
    """Fail closed: return the run's manifest, or raise if none was written.

    Call this at launch time — before ``srun``, before any gate that consumes
    provenance — anywhere the predecessor reached for
    ``[ -r capsule ] && source capsule || true``. The ``|| true`` is the exact
    mechanism by which 77 result dirs accumulated zero provenance bundles and
    nobody was ever paged: the failure was designed to be invisible. This function
    is its negation.

    Args:
        store: A :class:`ManifestStore`, or a path to a store root.
        run_id: The run whose provenance must exist.

    Returns:
        The most recent attempt's :class:`RunManifest`.

    Raises:
        ManifestMissing: No manifest exists for ``run_id``.
    """
    if not isinstance(store, ManifestStore):
        store = ManifestStore(store)
    return store.latest(run_id)
