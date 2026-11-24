"""Atomic, quota-bounded storage for environment snapshot artifacts.

A snapshot becomes visible only when its JSON descriptor is atomically committed.
Archive size and SHA-256 are checked during commit and on every restore path lookup,
disk use is bounded by reservations and store-wide quotas, and leased snapshots are
never reaped.

This module does not take snapshots itself, does not make archives deterministic,
and never deduplicates archives: equal environment signatures do not imply equal
archives.  It is also not a security boundary for archives, descriptors, or callers.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import cast

SNAPSHOT_SCHEMA = "fs.env_snapshot/v1"

_ID_RE = re.compile(r"^snap_[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_LAUNCHER = (
    "import os,resource,signal,sys\n"
    "n=int(sys.argv[1])\n"
    # Python starts with SIGXFSZ ignored and exec keeps ignored signals ignored; restore the
    # default so the exporter is killed at the limit instead of seeing a silent short write.
    "signal.signal(signal.SIGXFSZ,signal.SIG_DFL)\n"
    "resource.setrlimit(resource.RLIMIT_FSIZE,(n,n))\n"
    "os.execvp(sys.argv[2],sys.argv[2:])\n"
)


class SnapshotRefusal(ValueError):
    """A snapshot configuration, argument, or identifier was refused."""


class SnapshotIntegrityError(RuntimeError):
    """A snapshot is uncommitted, missing, malformed, or fails integrity checks."""


class SnapshotQuotaError(RuntimeError):
    """A reservation would exceed the configured whole-store byte quota."""


@dataclass(frozen=True)
class SnapshotLimits:
    """Resource limits for archives and the snapshot store."""

    max_bytes: int = 8 * 1024**3
    total_bytes: int = 64 * 1024**3
    timeout_s: float = 300.0
    ttl_s: float = 86400.0

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_bytes", self.max_bytes),
            ("total_bytes", self.total_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise SnapshotRefusal(
                    f"field {field_name!r} has value {value!r}; "
                    "legal values are positive integers (not bool)"
                )

        for field_name, real in (("timeout_s", self.timeout_s), ("ttl_s", self.ttl_s)):
            if (
                type(real) is bool
                or not isinstance(real, (int, float))
                or not math.isfinite(real)
                or real <= 0
            ):
                raise SnapshotRefusal(
                    f"field {field_name!r} has value {real!r}; "
                    "legal values are finite, positive real numbers"
                )

        if self.max_bytes > self.total_bytes:
            raise SnapshotRefusal(
                f"field 'max_bytes' has value {self.max_bytes!r}; legal values are "
                f"integers in 1..{self.total_bytes}, because total_bytes="
                f"{self.total_bytes!r}"
            )


def _integrity_error(
    where: str,
    field: str,
    value: object,
    reason: str,
) -> SnapshotIntegrityError:
    return SnapshotIntegrityError(f"{where}: field {field!r} has value {value!r}; {reason}")


def _required_field(
    raw: Mapping[object, object],
    field: str,
    *,
    where: str,
) -> object:
    if field not in raw:
        raise _integrity_error(where, field, "<missing>", "field is required")
    return raw[field]


@dataclass(frozen=True)
class SnapshotRecord:
    """The committed descriptor for one snapshot archive."""

    snapshot_id: str
    archive_bytes: int
    archive_sha256: str
    env_signature: str
    created_at: float
    leased: bool
    meta: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "meta", dict(self.meta))

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": SNAPSHOT_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "archive_bytes": self.archive_bytes,
            "archive_sha256": self.archive_sha256,
            "env_signature": self.env_signature,
            "created_at": self.created_at,
            "leased": self.leased,
            "meta": dict(self.meta),
        }
        return payload

    @classmethod
    def from_json(cls, raw: object, *, where: str) -> SnapshotRecord:
        if not isinstance(raw, dict):
            raise _integrity_error(
                where,
                "descriptor",
                raw,
                "legal value is a JSON object",
            )
        data = cast(dict[object, object], raw)

        schema = _required_field(data, "schema", where=where)
        if schema != SNAPSHOT_SCHEMA:
            raise _integrity_error(
                where,
                "schema",
                schema,
                f"legal value is {SNAPSHOT_SCHEMA!r}",
            )

        snapshot_id = _required_field(data, "snapshot_id", where=where)
        if not isinstance(snapshot_id, str) or _ID_RE.fullmatch(snapshot_id) is None:
            raise _integrity_error(
                where,
                "snapshot_id",
                snapshot_id,
                f"legal values match {_ID_RE.pattern!r}",
            )

        archive_bytes = _required_field(data, "archive_bytes", where=where)
        if (
            isinstance(archive_bytes, bool)
            or not isinstance(archive_bytes, int)
            or archive_bytes <= 0
        ):
            raise _integrity_error(
                where,
                "archive_bytes",
                archive_bytes,
                "legal values are positive integers (not bool)",
            )

        archive_sha256 = _required_field(data, "archive_sha256", where=where)
        if not isinstance(archive_sha256, str) or _SHA256_RE.fullmatch(archive_sha256) is None:
            raise _integrity_error(
                where,
                "archive_sha256",
                archive_sha256,
                "legal values are 64 lowercase hexadecimal characters",
            )

        env_signature = _required_field(data, "env_signature", where=where)
        if not isinstance(env_signature, str) or not env_signature:
            raise _integrity_error(
                where,
                "env_signature",
                env_signature,
                "legal values are non-empty strings",
            )

        created_at = _required_field(data, "created_at", where=where)
        if isinstance(created_at, bool) or not isinstance(created_at, Real):
            raise _integrity_error(
                where,
                "created_at",
                created_at,
                "legal values are finite real numbers",
            )
        try:
            created_at_value = float(created_at)
        except (OverflowError, ValueError) as exc:
            raise _integrity_error(
                where,
                "created_at",
                created_at,
                "legal values are finite real numbers",
            ) from exc
        if not math.isfinite(created_at_value):
            raise _integrity_error(
                where,
                "created_at",
                created_at,
                "legal values are finite real numbers",
            )

        leased = _required_field(data, "leased", where=where)
        if not isinstance(leased, bool):
            raise _integrity_error(
                where,
                "leased",
                leased,
                "legal values are true and false",
            )

        raw_meta = _required_field(data, "meta", where=where)
        if not isinstance(raw_meta, dict):
            raise _integrity_error(
                where,
                "meta",
                raw_meta,
                "legal value is an object mapping strings to strings",
            )
        meta: dict[str, str] = {}
        for key, value in cast(dict[object, object], raw_meta).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise _integrity_error(
                    where,
                    "meta",
                    raw_meta,
                    "every key and value must be a string",
                )
            meta[key] = value

        return cls(
            snapshot_id=snapshot_id,
            archive_bytes=archive_bytes,
            archive_sha256=archive_sha256,
            env_signature=env_signature,
            created_at=created_at_value,
            leased=leased,
            meta=meta,
        )


def _raise_walk_error(error: OSError) -> None:
    raise error


def rootless_lossy_entries(
    root: Path,
    *,
    exclude: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return sorted paths whose setuid/setgid bits rootless extraction can lose.

    This is MEASURED on GB200 aarch64 with rootless enroot 4.0.1: squashfs retains
    a setgid directory bit, while rootless extraction clears it.  Such a rootfs can
    never restore to an equal signature, so callers must refuse it up front.

    Exclude items are relative POSIX path prefixes by path component.  Walk errors
    are deliberately propagated as OSError instances.
    """
    root_path = Path(root)
    exclude_prefixes = tuple(PurePosixPath(item).parts for item in exclude)

    def is_excluded(parts: tuple[str, ...]) -> bool:
        return any(parts[: len(prefix)] == prefix for prefix in exclude_prefixes)

    lossy: list[str] = []
    for dirpath, dirnames, filenames in os.walk(
        root_path,
        topdown=True,
        followlinks=False,
        onerror=_raise_walk_error,
    ):
        parent = Path(dirpath)
        parent_parts = parent.relative_to(root_path).parts
        dirnames.sort()
        dirnames[:] = [name for name in dirnames if not is_excluded((*parent_parts, name))]

        for name in (*dirnames, *filenames):
            parts = (*parent_parts, name)
            if is_excluded(parts):
                continue
            mode = (parent / name).lstat().st_mode
            if mode & (stat.S_ISGID | stat.S_ISUID):
                lossy.append("/".join(parts))

    return tuple(sorted(lossy))


def bounded_argv(argv: Sequence[str], max_bytes: int) -> list[str]:
    """Wrap an export command with an RLIMIT_FSIZE enforced during writing."""
    if not argv:
        raise SnapshotRefusal(
            f"field 'argv' has value {list(argv)!r}; legal values are non-empty command sequences"
        )
    if type(max_bytes) is not int or max_bytes <= 0:
        raise SnapshotRefusal(
            f"field 'max_bytes' has value {max_bytes!r}; "
            "legal values are positive integers (not bool)"
        )
    return [sys.executable, "-c", _LAUNCHER, str(max_bytes), *argv]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class SnapshotStore:
    """Filesystem-backed atomic snapshot artifact store."""

    def __init__(
        self,
        root: str | Path,
        limits: SnapshotLimits | None = None,
        *,
        clock: Callable[[], float] = time.time,
        on_remove: Callable[[SnapshotRecord], None] | None = None,
    ) -> None:
        """``on_remove`` runs after a committed snapshot is uncommitted by ``delete()`` or
        ``reap()`` (descriptor gone, under the store lock) to release what the snapshot
        references outside the store -- e.g. its retained hidden-test vault. A failing hook
        never resurrects the snapshot; the error propagates to the caller."""
        root_path = Path(root)
        if not root_path.is_absolute():
            raise SnapshotRefusal(
                f"field 'root' has value {str(root_path)!r}; legal values are absolute paths"
            )
        root_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root = root_path
        self.limits = limits if limits is not None else SnapshotLimits()
        self._clock = clock
        self._on_remove = on_remove

    def _archive_path(self, snapshot_id: str) -> Path:
        return self.root / f"{snapshot_id}.sqsh"

    def _descriptor_path(self, snapshot_id: str) -> Path:
        return self.root / f"{snapshot_id}.json"

    def _temporary_archive_path(self, snapshot_id: str) -> Path:
        return self.root / f".tmp-{snapshot_id}.sqsh"

    def _reservation_path(self, snapshot_id: str) -> Path:
        return self.root / f".resv-{snapshot_id}"

    def _temporary_descriptor_path(self, snapshot_id: str) -> Path:
        return self.root / f".{snapshot_id}.json.tmp"

    @staticmethod
    def _check_id(snapshot_id: str) -> None:
        if not isinstance(snapshot_id, str) or _ID_RE.fullmatch(snapshot_id) is None:
            raise SnapshotRefusal(
                f"field 'snapshot_id' has value {snapshot_id!r}; "
                f"legal values match {_ID_RE.pattern!r}"
            )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        lock_fd = os.open(
            self.root / ".lock",
            os.O_CREAT | os.O_RDWR,
            0o600,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    @staticmethod
    def _unlink_quietly(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)

    def _write_descriptor(self, record: SnapshotRecord) -> None:
        temporary = self._temporary_descriptor_path(record.snapshot_id)
        descriptor = self._descriptor_path(record.snapshot_id)
        serialized = json.dumps(
            record.to_json(),
            sort_keys=True,
            separators=(",", ":"),
        )
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(descriptor)

    def _fsync_directory(self) -> None:
        directory_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def usage(self) -> int:
        """Return committed archive bytes plus all live reservation amounts."""
        total = 0
        for path in self.root.iterdir():
            name = path.name
            if name.endswith(".sqsh"):
                snapshot_id = name[: -len(".sqsh")]
                if (
                    _ID_RE.fullmatch(snapshot_id) is not None
                    and self._descriptor_path(snapshot_id).exists()
                ):
                    try:
                        total += self.get(snapshot_id).archive_bytes
                    except SnapshotIntegrityError:
                        # A malformed descriptor must not wedge every later commit; count the
                        # bytes actually on disk instead (reap leaves such files for an operator).
                        total += path.lstat().st_size

            if name.startswith(".resv-"):
                try:
                    with path.open("r", encoding="utf-8") as stream:
                        amount = int(stream.read().strip())
                    if amount <= 0:
                        raise ValueError("reservation amount must be positive")
                except (OSError, ValueError):
                    amount = self.limits.max_bytes
                total += amount
        return total

    def commit(
        self,
        *,
        export: Callable[[Path, float], object],
        env_signature: str,
        meta: Mapping[str, str] | None = None,
    ) -> SnapshotRecord:
        if not isinstance(env_signature, str) or not env_signature:
            raise SnapshotRefusal(
                f"field 'env_signature' has value {env_signature!r}; "
                "legal values are non-empty strings"
            )
        if meta is not None and not isinstance(meta, Mapping):
            raise SnapshotRefusal(
                f"field 'meta' has value {meta!r}; legal values map strings to strings"
            )
        meta_copy: dict[str, str] = {}
        if meta is not None:
            for key, value in meta.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise SnapshotRefusal(
                        f"field 'meta' has value {meta!r}; every key and value must be a string"
                    )
                meta_copy[key] = value

        snapshot_id = "snap_" + secrets.token_hex(16)
        temporary = self._temporary_archive_path(snapshot_id)
        archive = self._archive_path(snapshot_id)
        reservation = self._reservation_path(snapshot_id)
        descriptor_temporary = self._temporary_descriptor_path(snapshot_id)

        with self._locked():
            current_usage = self.usage()
            if current_usage + self.limits.max_bytes > self.limits.total_bytes:
                raise SnapshotQuotaError(
                    "reservation refused: "
                    f"usage={current_usage}, max_bytes={self.limits.max_bytes}, "
                    f"total_bytes={self.limits.total_bytes}"
                )
            try:
                with reservation.open("w", encoding="utf-8") as stream:
                    stream.write(str(self.limits.max_bytes))
            except BaseException:
                self._unlink_quietly(reservation)
                raise

        descriptor_committed = False
        try:
            export(temporary, self.limits.timeout_s)

            try:
                archive_stat = temporary.lstat()
            except FileNotFoundError as exc:
                raise SnapshotIntegrityError(
                    f"{temporary}: field 'archive' has value {str(temporary)!r}; "
                    "export must leave a regular file"
                ) from exc
            if not stat.S_ISREG(archive_stat.st_mode):
                raise SnapshotIntegrityError(
                    f"{temporary}: field 'archive' has value {str(temporary)!r}; "
                    f"mode {oct(archive_stat.st_mode)} is not a regular, "
                    "non-symlink file"
                )

            archive_size = archive_stat.st_size
            if archive_size > self.limits.max_bytes:
                raise SnapshotIntegrityError(
                    f"{temporary}: field 'archive_bytes' has value {archive_size}; "
                    f"size exceeds max_bytes={self.limits.max_bytes}"
                )
            if archive_size <= 0:
                raise SnapshotIntegrityError(
                    f"{temporary}: field 'archive_bytes' has value {archive_size}; "
                    f"legal sizes are 1..{self.limits.max_bytes}"
                )

            archive_digest = _sha256_file(temporary)
            archive_fd = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(archive_fd)
            finally:
                os.close(archive_fd)

            record = SnapshotRecord(
                snapshot_id=snapshot_id,
                archive_bytes=archive_size,
                archive_sha256=archive_digest,
                env_signature=env_signature,
                created_at=self._clock(),
                leased=False,
                meta=meta_copy,
            )

            with self._locked():
                temporary.replace(archive)
                self._write_descriptor(record)
                descriptor_committed = True
                self._fsync_directory()
                reservation.unlink(missing_ok=True)
            return record
        except BaseException:
            self._unlink_quietly(temporary)
            if not descriptor_committed:
                self._unlink_quietly(archive)
            self._unlink_quietly(descriptor_temporary)
            self._unlink_quietly(reservation)
            raise

    def get(self, snapshot_id: str) -> SnapshotRecord:
        self._check_id(snapshot_id)
        descriptor = self._descriptor_path(snapshot_id)
        where = str(descriptor)
        try:
            with descriptor.open("r", encoding="utf-8") as stream:
                raw: object = json.load(stream)
        except FileNotFoundError as exc:
            raise SnapshotIntegrityError(
                f"{where}: field 'descriptor' has value {str(descriptor)!r}; "
                f"snapshot {snapshot_id!r} is not committed"
            ) from exc
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise SnapshotIntegrityError(
                f"{where}: field 'descriptor' has value {str(exc)!r}; descriptor is malformed JSON"
            ) from exc

        record = SnapshotRecord.from_json(raw, where=where)
        if record.snapshot_id != snapshot_id:
            raise _integrity_error(
                where,
                "snapshot_id",
                record.snapshot_id,
                f"value must equal requested id {snapshot_id!r}",
            )
        return record

    def archive_path(self, snapshot_id: str) -> Path:
        record = self.get(snapshot_id)
        archive = self._archive_path(snapshot_id)
        try:
            archive_stat = archive.lstat()
        except FileNotFoundError as exc:
            raise SnapshotIntegrityError(
                f"{archive}: field 'archive' has value {str(archive)!r}; "
                f"archive for {snapshot_id!r} is missing"
            ) from exc

        if not stat.S_ISREG(archive_stat.st_mode):
            raise SnapshotIntegrityError(
                f"{archive}: field 'archive' has value {str(archive)!r}; "
                f"mode {oct(archive_stat.st_mode)} is not a regular, "
                "non-symlink file"
            )
        if archive_stat.st_size != record.archive_bytes:
            raise SnapshotIntegrityError(
                f"{archive}: field 'archive_bytes' has value "
                f"{archive_stat.st_size}; committed size is {record.archive_bytes}"
            )

        actual_digest = _sha256_file(archive)
        if actual_digest != record.archive_sha256:
            raise SnapshotIntegrityError(
                f"{archive}: field 'archive_sha256' has value {actual_digest!r}; "
                f"committed digest is {record.archive_sha256!r}"
            )
        return archive

    def _set_leased(self, snapshot_id: str, *, leased: bool) -> SnapshotRecord:
        self._check_id(snapshot_id)
        with self._locked():
            record = self.get(snapshot_id)
            updated = replace(record, leased=leased)
            self._write_descriptor(updated)
            return updated

    def lease(self, snapshot_id: str) -> SnapshotRecord:
        return self._set_leased(snapshot_id, leased=True)

    def release(self, snapshot_id: str) -> SnapshotRecord:
        return self._set_leased(snapshot_id, leased=False)

    def delete(self, snapshot_id: str) -> None:
        self._check_id(snapshot_id)
        with self._locked():
            record = self.get(snapshot_id)
            if record.leased:
                raise SnapshotRefusal(
                    f"field 'leased' has value {record.leased!r} for "
                    f"snapshot {snapshot_id!r}; legal value for deletion is False"
                )
            self._descriptor_path(snapshot_id).unlink()
            self._archive_path(snapshot_id).unlink(missing_ok=True)
            if self._on_remove is not None:
                self._on_remove(record)

    @staticmethod
    def _is_reapable_temporary_name(name: str) -> bool:
        return (
            (name.startswith(".tmp-") and name.endswith(".sqsh"))
            or name.startswith(".resv-")
            or (name.startswith(".") and name.endswith(".json.tmp"))
        )

    def reap(self) -> tuple[str, ...]:
        """Reap stale temporary artifacts, expired snapshots, and stale orphans."""
        removed: list[str] = []
        with self._locked():
            now = self._clock()
            entries = list(self.root.iterdir())

            for path in entries:
                if not self._is_reapable_temporary_name(path.name):
                    continue
                try:
                    modified_at = path.lstat().st_mtime
                except FileNotFoundError:
                    continue
                if now - modified_at > self.limits.ttl_s:
                    path.unlink()
                    removed.append(path.name)

            for path in entries:
                name = path.name
                if not name.endswith(".json") or name.startswith("."):
                    continue
                snapshot_id = name[: -len(".json")]
                if _ID_RE.fullmatch(snapshot_id) is None:
                    continue
                try:
                    record = self.get(snapshot_id)
                except (SnapshotIntegrityError, OSError):
                    # A malformed descriptor is evidence for an operator, not garbage.
                    continue
                if record.leased:
                    continue
                if now - record.created_at <= self.limits.ttl_s:
                    continue
                self._descriptor_path(snapshot_id).unlink(missing_ok=True)
                self._archive_path(snapshot_id).unlink(missing_ok=True)
                removed.append(snapshot_id)
                if self._on_remove is not None:
                    self._on_remove(record)

            for path in entries:
                name = path.name
                if not name.endswith(".sqsh"):
                    continue
                snapshot_id = name[: -len(".sqsh")]
                if _ID_RE.fullmatch(snapshot_id) is None:
                    continue
                if self._descriptor_path(snapshot_id).exists():
                    continue
                try:
                    modified_at = path.lstat().st_mtime
                except FileNotFoundError:
                    continue
                if now - modified_at > self.limits.ttl_s:
                    path.unlink(missing_ok=True)
                    removed.append(path.name)

        return tuple(sorted(removed))
