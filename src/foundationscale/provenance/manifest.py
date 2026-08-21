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

__all__ = [
    "SCHEMA_VERSION",
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
    "RunManifest",
    "ManifestStore",
    "ManifestError",
    "ManifestMissing",
    "ManifestVersionError",
    "ManifestExistsError",
    "capture_environment",
    "capture_code_provenance",
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
    """Raised when a manifest on disk declares an unknown schema version."""


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


def _git(root: Path, args: Sequence[str]) -> tuple[int, str] | None:
    """Run a git command, returning ``(rc, stdout)``; ``None`` on any failure.

    Returns ``None`` rather than raising for every failure mode (git absent,
    timeout, non-repo directory) so that callers degrade to ``NOT_A_REPOSITORY``
    or ``NOT_CAPTURED`` instead of crashing the launcher — and so that "the code is
    gone" is represented in the record, not lost in a traceback.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
        return proc.returncode, proc.stdout
    except (OSError, subprocess.TimeoutExpired):
        return None


def _git_bytes(root: Path, args: Sequence[str]) -> bytes | None:
    """Binary variant of :func:`_git` for diff payloads, which need exact bytes."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout
    except (OSError, subprocess.TimeoutExpired):
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
    schema_version: int = SCHEMA_VERSION
    findings: tuple[str, ...] = field(init=False, default=())

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
                f"this module supports {SCHEMA_VERSION}"
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
            "findings": list(self.findings),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RunManifest:
        """Rebuild a manifest from parsed JSON, recomputing findings from scratch.

        Serialized findings are ignored on load: they are derivable, and trusting a
        stored copy would let a hand-edited manifest erase its own blemishes.
        """
        try:
            config_data = _expect_mapping(data["config"], "config")
            return cls(
                run_id=str(data["run_id"]),
                attempt=_expect_int(data["attempt"], "attempt"),
                code=CodeProvenance.from_dict(_expect_mapping(data["code"], "code")),
                config={
                    str(k): EffectiveValue.from_dict(_expect_mapping(v, f"config[{k!r}]"))
                    for k, v in config_data.items()
                },
                environment=CapturedEnvironment.from_dict(
                    _expect_mapping(data["environment"], "environment")
                ),
                topology=Topology.from_dict(_expect_mapping(data["topology"], "topology")),
                job_id=None if data.get("job_id") is None else str(data["job_id"]),
                created_at=str(data.get("created_at", "")),
                artifact_paths={
                    str(k): str(v)
                    for k, v in _expect_mapping(
                        data.get("artifact_paths", {}), "artifact_paths"
                    ).items()
                },
                schema_version=_expect_int(data["schema_version"], "schema_version"),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ManifestError(f"corrupt manifest: missing or malformed field ({exc!r})") from exc


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


def load(path: str | os.PathLike[str]) -> RunManifest:
    """Load a manifest from disk, refusing unknown schema versions.

    Args:
        path: Manifest file written by :meth:`ManifestStore.save`.

    Returns:
        The parsed :class:`RunManifest`.

    Raises:
        ManifestVersionError: The file declares any schema version this module does
            not implement. Best-effort parsing of an unknown schema is how the
            predecessor's flat string bag stayed "valid" while half its keys meant
            nothing.
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
    if version != SCHEMA_VERSION:
        raise ManifestVersionError(
            f"refusing to load manifest at {path}: schema_version={version!r}, "
            f"supported={SCHEMA_VERSION}"
        )
    return RunManifest.from_dict(data)


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
