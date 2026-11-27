from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from foundationscale.agentic_rl.sandbox.snapshot_store import (
    SNAPSHOT_SCHEMA,
    SnapshotIntegrityError,
    SnapshotLimits,
    SnapshotQuotaError,
    SnapshotRecord,
    SnapshotRefusal,
    SnapshotStore,
    bounded_argv,
    rootless_lossy_entries,
)


def make_id(number: int) -> str:
    return f"snap_{number:032x}"


def make_export(payload: bytes) -> Callable[[Path, float], None]:
    def export(path: Path, timeout_s: float) -> None:
        del timeout_s
        path.write_bytes(payload)

    return export


def valid_record_json(snapshot_id: str) -> dict[str, object]:
    return {
        "schema": SNAPSHOT_SCHEMA,
        "snapshot_id": snapshot_id,
        "archive_bytes": 8,
        "archive_sha256": "a" * 64,
        "env_signature": "test-signature",
        "created_at": 123.5,
        "leased": False,
        "meta": {"team": "rl"},
    }


class FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"max_bytes": True}, "max_bytes"),
        ({"total_bytes": False}, "total_bytes"),
        ({"max_bytes": 0}, "max_bytes"),
        ({"total_bytes": 0}, "total_bytes"),
        ({"max_bytes": -1}, "max_bytes"),
        ({"total_bytes": -1}, "total_bytes"),
        ({"max_bytes": 1.5}, "max_bytes"),
        ({"max_bytes": 11, "total_bytes": 10}, "max_bytes"),
        ({"timeout_s": 0.0}, "timeout_s"),
        ({"timeout_s": float("nan")}, "timeout_s"),
        ({"timeout_s": float("inf")}, "timeout_s"),
        ({"ttl_s": 0.0}, "ttl_s"),
        ({"ttl_s": -1.0}, "ttl_s"),
        ({"ttl_s": float("nan")}, "ttl_s"),
    ],
)
def test_limits_refuse_bad_values(kwargs: dict[str, object], field: str) -> None:
    with pytest.raises(SnapshotRefusal) as exc_info:
        SnapshotLimits(**kwargs)  # type: ignore[arg-type]

    assert field in str(exc_info.value)


def test_limits_allow_equal_archive_and_total_limits() -> None:
    limits = SnapshotLimits(max_bytes=128, total_bytes=128)

    assert limits.max_bytes == 128
    assert limits.total_bytes == 128


def test_snapshot_record_json_round_trip_and_meta_copy() -> None:
    metadata = {"team": "rl"}
    record = SnapshotRecord(
        snapshot_id=make_id(1),
        archive_bytes=8,
        archive_sha256="a" * 64,
        env_signature="test-signature",
        created_at=123.5,
        leased=False,
        meta=metadata,
    )
    metadata["team"] = "changed"

    assert record.meta == {"team": "rl"}
    raw = record.to_json()
    assert raw["schema"] == SNAPSHOT_SCHEMA
    assert SnapshotRecord.from_json(raw, where="record-test") == record


@pytest.mark.parametrize("raw", [None, [], "not a mapping", 42])
def test_snapshot_record_from_json_rejects_non_mappings(raw: object) -> None:
    with pytest.raises(SnapshotIntegrityError) as exc_info:
        SnapshotRecord.from_json(raw, where="record-test")

    assert "record-test" in str(exc_info.value)


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "snapshot_id",
        "archive_bytes",
        "archive_sha256",
        "env_signature",
        "created_at",
        "leased",
        "meta",
    ],
)
def test_snapshot_record_from_json_requires_every_field(field: str) -> None:
    raw = valid_record_json(make_id(2))
    raw.pop(field)

    with pytest.raises(SnapshotIntegrityError) as exc_info:
        SnapshotRecord.from_json(raw, where="missing-field-test")

    message = str(exc_info.value)
    assert "missing-field-test" in message
    assert field in message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "fs.env_snapshot/v2"),
        ("snapshot_id", "../escape"),
        ("snapshot_id", "snap_not_hex"),
        ("archive_bytes", True),
        ("archive_bytes", 0),
        ("archive_bytes", -1),
        ("archive_bytes", 1.5),
        ("archive_bytes", "8"),
        ("archive_sha256", "A" * 64),
        ("archive_sha256", "g" * 64),
        ("archive_sha256", "a" * 63),
        ("env_signature", ""),
        ("env_signature", 7),
        ("created_at", float("nan")),
        ("created_at", float("inf")),
        ("created_at", "yesterday"),
        ("leased", 0),
        ("leased", "false"),
        ("meta", []),
        ("meta", {"team": 7}),
        ("meta", {7: "rl"}),
    ],
)
def test_snapshot_record_from_json_rejects_bad_fields(field: str, value: object) -> None:
    raw = valid_record_json(make_id(3))
    raw[field] = value

    with pytest.raises(SnapshotIntegrityError) as exc_info:
        SnapshotRecord.from_json(raw, where="bad-field-test")

    message = str(exc_info.value)
    assert "bad-field-test" in message
    assert field in message


def test_store_root_must_be_absolute(tmp_path: Path) -> None:
    with pytest.raises(SnapshotRefusal, match="root"):
        SnapshotStore("relative-snapshot-store")
    with pytest.raises(SnapshotRefusal, match="root"):
        SnapshotStore(Path("relative-snapshot-store"))

    root = tmp_path / "nested" / "store"
    SnapshotStore(str(root))
    assert root.is_dir()


def test_commit_is_atomic_and_round_trips(tmp_path: Path) -> None:
    root = tmp_path / "store"
    clock = FakeClock(1234.5)
    limits = SnapshotLimits(max_bytes=1024, total_bytes=4096, timeout_s=12.5)
    store = SnapshotStore(root, limits, clock=clock)
    payload = b"squashfs-like archive bytes"
    export_calls: list[tuple[Path, float]] = []

    def export(path: Path, timeout_s: float) -> None:
        export_calls.append((path, timeout_s))
        assert path.parent == root
        path.write_bytes(payload)

    record = store.commit(
        export=export,
        env_signature="env-signature",
        meta={"image": "demo"},
    )

    assert len(export_calls) == 1
    temporary_path, timeout_s = export_calls[0]
    assert temporary_path == root / f".tmp-{record.snapshot_id}.sqsh"
    assert timeout_s == 12.5
    assert record.archive_bytes == len(payload)
    assert record.archive_sha256 == hashlib.sha256(payload).hexdigest()
    assert record.env_signature == "env-signature"
    assert record.created_at == 1234.5
    assert record.leased is False
    assert record.meta == {"image": "demo"}

    assert store.get(record.snapshot_id) == record
    archive = store.archive_path(record.snapshot_id)
    assert archive == root / f"{record.snapshot_id}.sqsh"
    assert archive.read_bytes() == payload

    descriptor_path = root / f"{record.snapshot_id}.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    assert descriptor["schema"] == SNAPSHOT_SCHEMA
    assert descriptor == record.to_json()

    assert sorted(path.name for path in root.iterdir()) == sorted(
        [".lock", descriptor_path.name, archive.name]
    )


def test_commit_refuses_bad_signature_and_meta(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "store")

    with pytest.raises(SnapshotRefusal, match="env_signature"):
        store.commit(export=make_export(b"data"), env_signature="")
    with pytest.raises(SnapshotRefusal, match="env_signature"):
        store.commit(
            export=make_export(b"data"),
            env_signature=7,  # type: ignore[arg-type]
        )
    with pytest.raises(SnapshotRefusal, match="meta"):
        store.commit(
            export=make_export(b"data"),
            env_signature="sig",
            meta={"bad": 1},  # type: ignore[dict-item]
        )
    with pytest.raises(SnapshotRefusal, match="meta"):
        store.commit(
            export=make_export(b"data"),
            env_signature="sig",
            meta={1: "bad"},  # type: ignore[dict-item]
        )


def test_commit_failure_leaves_nothing(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = SnapshotStore(root, SnapshotLimits(max_bytes=32, total_bytes=64))
    failure = RuntimeError("export failed")

    def failing_export(path: Path, timeout_s: float) -> None:
        del timeout_s
        path.write_bytes(b"partial archive")
        raise failure

    with pytest.raises(RuntimeError) as exc_info:
        store.commit(export=failing_export, env_signature="sig")
    assert exc_info.value is failure
    assert [path.name for path in root.iterdir()] == [".lock"]

    with pytest.raises(SnapshotIntegrityError, match="size|max_bytes|exceed"):
        store.commit(export=make_export(b"x" * 33), env_signature="sig")
    assert [path.name for path in root.iterdir()] == [".lock"]

    with pytest.raises(SnapshotIntegrityError, match="size|empty"):
        store.commit(export=make_export(b""), env_signature="sig")
    assert [path.name for path in root.iterdir()] == [".lock"]


def test_export_writing_a_symlink_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "store"
    target = tmp_path / "outside-archive"
    target.write_bytes(b"archive")
    store = SnapshotStore(root, SnapshotLimits(max_bytes=32, total_bytes=64))

    def symlink_export(path: Path, timeout_s: float) -> None:
        del timeout_s
        path.symlink_to(target)

    with pytest.raises(SnapshotIntegrityError, match="regular|symlink"):
        store.commit(export=symlink_export, env_signature="sig")

    assert [path.name for path in root.iterdir()] == [".lock"]
    assert target.read_bytes() == b"archive"


def test_quota_refuses_when_reservation_would_exceed_total(tmp_path: Path) -> None:
    root = tmp_path / "store"
    limits = SnapshotLimits(max_bytes=64, total_bytes=64)
    store = SnapshotStore(root, limits)
    first = store.commit(export=make_export(b"12345678"), env_signature="first")

    def unexpected_export(path: Path, timeout_s: float) -> None:
        del path, timeout_s
        raise AssertionError("export must not run after a quota refusal")

    usage = store.usage()
    with pytest.raises(SnapshotQuotaError) as exc_info:
        store.commit(export=unexpected_export, env_signature="second")

    message = str(exc_info.value)
    assert str(usage) in message
    assert str(limits.max_bytes) in message
    assert str(limits.total_bytes) in message
    assert list(root.glob(".resv-*")) == []
    assert store.archive_path(first.snapshot_id).read_bytes() == b"12345678"


def test_archive_path_rejects_digest_and_size_mismatch(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "store")
    record = store.commit(export=make_export(b"abcdefgh"), env_signature="sig")
    archive = tmp_path / "store" / f"{record.snapshot_id}.sqsh"

    archive.write_bytes(b"Xbcdefgh")
    with pytest.raises(SnapshotIntegrityError, match="sha256|digest"):
        store.archive_path(record.snapshot_id)

    archive.write_bytes(b"abcdefgh-extra")
    with pytest.raises(SnapshotIntegrityError, match="size"):
        store.archive_path(record.snapshot_id)


def test_archive_path_rejects_a_symlink(tmp_path: Path) -> None:
    root = tmp_path / "store"
    target = tmp_path / "outside"
    target.write_bytes(b"archive")
    store = SnapshotStore(root)
    record = store.commit(export=make_export(b"archive"), env_signature="sig")
    archive = root / f"{record.snapshot_id}.sqsh"

    archive.unlink()
    archive.symlink_to(target)

    with pytest.raises(SnapshotIntegrityError, match="regular|symlink"):
        store.archive_path(record.snapshot_id)


def test_get_refuses_bad_id_and_uncommitted_and_malformed(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = SnapshotStore(root)

    with pytest.raises(SnapshotRefusal, match="snapshot_id"):
        store.get("../x")
    with pytest.raises(SnapshotIntegrityError, match="not committed"):
        store.get(make_id(100))

    snapshot_id = make_id(101)
    descriptor = root / f"{snapshot_id}.json"
    raw = valid_record_json(snapshot_id)
    raw["schema"] = "wrong-schema"
    descriptor.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SnapshotIntegrityError, match="schema"):
        store.get(snapshot_id)

    raw = valid_record_json(snapshot_id)
    raw["archive_bytes"] = "8"
    descriptor.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SnapshotIntegrityError, match="archive_bytes"):
        store.get(snapshot_id)

    raw = valid_record_json(make_id(102))
    descriptor.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SnapshotIntegrityError, match="snapshot_id"):
        store.get(snapshot_id)

    descriptor.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SnapshotIntegrityError):
        store.get(snapshot_id)


def test_lease_blocks_delete_and_reap(tmp_path: Path) -> None:
    root = tmp_path / "store"
    clock = FakeClock(100.0)
    store = SnapshotStore(
        root,
        SnapshotLimits(max_bytes=128, total_bytes=1024, ttl_s=10.0, timeout_s=1.0),
        clock=clock,
    )
    record = store.commit(export=make_export(b"archive"), env_signature="sig")

    leased = store.lease(record.snapshot_id)
    assert leased.leased is True
    assert leased.created_at == record.created_at
    assert store.get(record.snapshot_id).leased is True

    with pytest.raises(SnapshotRefusal, match="leased"):
        store.delete(record.snapshot_id)

    clock.advance(11.0)
    assert store.reap() == ()
    assert store.get(record.snapshot_id).leased is True
    assert store.archive_path(record.snapshot_id).read_bytes() == b"archive"


def test_release_allows_reap(tmp_path: Path) -> None:
    root = tmp_path / "store"
    clock = FakeClock(100.0)
    store = SnapshotStore(
        root,
        SnapshotLimits(max_bytes=128, total_bytes=1024, ttl_s=10.0, timeout_s=1.0),
        clock=clock,
    )
    record = store.commit(export=make_export(b"archive"), env_signature="sig")
    store.lease(record.snapshot_id)

    released = store.release(record.snapshot_id)
    assert released.leased is False
    assert store.get(record.snapshot_id).leased is False

    clock.advance(11.0)
    removed = store.reap()
    assert removed
    assert removed == tuple(sorted(removed))
    assert not (root / f"{record.snapshot_id}.json").exists()
    assert not (root / f"{record.snapshot_id}.sqsh").exists()
    with pytest.raises(SnapshotIntegrityError, match="not committed"):
        store.get(record.snapshot_id)


def test_delete_removes_descriptor_and_archive(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = SnapshotStore(root)
    record = store.commit(export=make_export(b"archive"), env_signature="sig")

    store.delete(record.snapshot_id)

    assert not (root / f"{record.snapshot_id}.json").exists()
    assert not (root / f"{record.snapshot_id}.sqsh").exists()
    with pytest.raises(SnapshotIntegrityError, match="not committed"):
        store.get(record.snapshot_id)


def test_reap_removes_stale_temp_and_orphans_but_keeps_fresh_and_leased(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    clock = FakeClock(1000.0)
    store = SnapshotStore(
        root,
        SnapshotLimits(max_bytes=128, total_bytes=1024, ttl_s=50.0, timeout_s=1.0),
        clock=clock,
    )
    expired = store.commit(export=make_export(b"expired"), env_signature="old")
    leased = store.commit(export=make_export(b"leased"), env_signature="leased")
    store.lease(leased.snapshot_id)

    clock.advance(60.0)
    fresh = store.commit(export=make_export(b"fresh"), env_signature="fresh")
    now = clock()

    stale_paths = [
        root / f".tmp-{make_id(41)}.sqsh",
        root / f".resv-{make_id(42)}",
        root / f".{make_id(43)}.json.tmp",
    ]
    fresh_paths = [
        root / f".tmp-{make_id(44)}.sqsh",
        root / f".resv-{make_id(45)}",
        root / f".{make_id(46)}.json.tmp",
    ]
    stale_orphan = root / f"{make_id(47)}.sqsh"
    fresh_orphan = root / f"{make_id(48)}.sqsh"

    for path in stale_paths:
        path.write_bytes(b"stale")
        os.utime(path, (1000.0, 1000.0))
    for path in fresh_paths:
        path.write_bytes(b"fresh")
        os.utime(path, (now, now))

    stale_orphan.write_bytes(b"orphan")
    os.utime(stale_orphan, (1000.0, 1000.0))
    fresh_orphan.write_bytes(b"orphan")
    os.utime(fresh_orphan, (now, now))

    malformed_id = make_id(49)
    malformed_descriptor = root / f"{malformed_id}.json"
    malformed_archive = root / f"{malformed_id}.sqsh"
    malformed_descriptor.write_text("{broken", encoding="utf-8")
    malformed_archive.write_bytes(b"do not delete")
    os.utime(malformed_descriptor, (1000.0, 1000.0))
    os.utime(malformed_archive, (1000.0, 1000.0))

    removed = store.reap()

    assert removed == tuple(sorted(removed))
    for path in stale_paths:
        assert not path.exists()
        assert path.name in removed
    assert not stale_orphan.exists()
    assert stale_orphan.name in removed

    for path in fresh_paths:
        assert path.exists()
    assert fresh_orphan.exists()

    assert not (root / f"{expired.snapshot_id}.json").exists()
    assert not (root / f"{expired.snapshot_id}.sqsh").exists()
    assert store.get(leased.snapshot_id).leased is True
    assert store.archive_path(leased.snapshot_id).read_bytes() == b"leased"
    assert store.archive_path(fresh.snapshot_id).read_bytes() == b"fresh"

    assert malformed_descriptor.exists()
    assert malformed_archive.read_bytes() == b"do not delete"


def test_usage_counts_reservations_and_malformed_reservation_as_max(tmp_path: Path) -> None:
    root = tmp_path / "store"
    limits = SnapshotLimits(max_bytes=100, total_bytes=1000)
    store = SnapshotStore(root, limits)
    usage_during_export: list[int] = []

    def export(path: Path, timeout_s: float) -> None:
        del timeout_s
        usage_during_export.append(store.usage())
        path.write_bytes(b"12345")

    assert store.usage() == 0
    record = store.commit(export=export, env_signature="sig")
    assert usage_during_export == [100]
    assert store.usage() == record.archive_bytes == 5

    valid_reservation = root / f".resv-{make_id(201)}"
    valid_reservation.write_text("25", encoding="utf-8")
    assert store.usage() == 30

    malformed_reservation = root / f".resv-{make_id(202)}"
    malformed_reservation.write_text("not-an-integer", encoding="utf-8")
    assert store.usage() == 30 + limits.max_bytes

    orphan = root / f"{make_id(203)}.sqsh"
    orphan.write_bytes(b"uncommitted bytes do not count")
    assert store.usage() == 130


def test_rootless_lossy_entries_finds_setgid_dir_and_respects_exclude(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rootfs"
    root.mkdir()

    setgid_dir = root / "gid-dir"
    setgid_dir.mkdir()
    setgid_dir.chmod(0o2755)

    setuid_file = root / "setuid.bin"
    setuid_file.write_bytes(b"binary")
    setuid_file.chmod(0o4755)

    plain = root / "plain"
    plain.mkdir()
    (plain / "file").write_text("ordinary", encoding="utf-8")

    nested_gid = root / "nested" / "gid"
    nested_gid.mkdir(parents=True)
    nested_gid.chmod(stat.S_ISGID | 0o755)

    skipped_gid = root / "skip" / "gid"
    skipped_gid.mkdir(parents=True)
    skipped_gid.chmod(stat.S_ISGID | 0o755)

    nested_skipped_gid = root / "nested" / "skip" / "gid"
    nested_skipped_gid.mkdir(parents=True)
    nested_skipped_gid.chmod(stat.S_ISGID | 0o755)

    expected = tuple(
        sorted(
            [
                "gid-dir",
                "nested/gid",
                "nested/skip/gid",
                "setuid.bin",
                "skip/gid",
            ]
        )
    )
    assert rootless_lossy_entries(root) == expected
    assert rootless_lossy_entries(
        root,
        exclude=("skip", "nested/skip"),
    ) == ("gid-dir", "nested/gid", "setuid.bin")


def test_bounded_argv_limits_file_size(tmp_path: Path) -> None:
    """A native writer (dd, like mksquashfs) is killed by SIGXFSZ at the limit."""
    large_file = tmp_path / "large-output"
    argv = ["dd", "if=/dev/zero", f"of={large_file}", "bs=4096", "count=1"]
    command = bounded_argv(argv, 1024)

    assert command[0] == sys.executable
    assert command[1] == "-c"
    assert command[3] == "1024"
    assert command[4:] == argv

    result = subprocess.run(command, check=False, capture_output=True, timeout=30)
    assert result.returncode == -signal.SIGXFSZ
    assert large_file.stat().st_size <= 1024

    small_file = tmp_path / "small-output"
    small_script = "import sys; open(sys.argv[1], 'wb').write(b'ok')"
    small_result = subprocess.run(
        bounded_argv([sys.executable, "-c", small_script, str(small_file)], 1024),
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert small_result.returncode == 0, small_result.stderr.decode(errors="replace")
    assert small_file.read_bytes() == b"ok"


def test_bounded_argv_refuses_empty_and_bad_limit() -> None:
    with pytest.raises(SnapshotRefusal, match="argv"):
        bounded_argv([], 1024)

    for limit in (0, -1, True, 1.5):
        with pytest.raises(SnapshotRefusal, match="max_bytes"):
            bounded_argv([sys.executable], limit)  # type: ignore[arg-type]


def test_usage_counts_a_malformed_descriptor_by_archive_size(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "store", SnapshotLimits(max_bytes=64, total_bytes=1024))
    record = store.commit(
        export=lambda path, timeout: path.write_bytes(b"a" * 10), env_signature="envsig_x"
    )
    (tmp_path / "store" / f"{record.snapshot_id}.json").write_text("{}")
    assert store.usage() == 10


def test_export_that_writes_nothing_is_refused(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "store", SnapshotLimits(max_bytes=64, total_bytes=1024))
    with pytest.raises(SnapshotIntegrityError, match="regular file"):
        store.commit(export=lambda path, timeout: None, env_signature="envsig_x")
    assert sorted(p.name for p in (tmp_path / "store").iterdir()) == [".lock"]


def test_archive_path_refuses_a_missing_archive(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "store", SnapshotLimits(max_bytes=64, total_bytes=1024))
    record = store.commit(
        export=lambda path, timeout: path.write_bytes(b"abc"), env_signature="envsig_x"
    )
    (tmp_path / "store" / f"{record.snapshot_id}.sqsh").unlink()
    with pytest.raises(SnapshotIntegrityError, match="missing"):
        store.archive_path(record.snapshot_id)


# --- fixes from the 2026-11-27 review ---------------------------------------------------


def test_limits_refuse_a_ttl_not_exceeding_the_export_timeout() -> None:
    with pytest.raises(SnapshotRefusal, match="ttl_s"):
        SnapshotLimits(ttl_s=300.0, timeout_s=300.0)


def test_rootless_lossy_entries_reports_a_setgid_root(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    root.mkdir()
    root.chmod(0o2755)
    assert rootless_lossy_entries(root) == (".",)


def test_delete_releases_references_even_if_the_archive_unlink_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    removed: list[str] = []
    store = SnapshotStore(
        tmp_path / "store",
        SnapshotLimits(max_bytes=64, total_bytes=1024),
        on_remove=lambda record: removed.append(record.snapshot_id),
    )
    record = store.commit(export=lambda path, timeout: path.write_bytes(b"abc"), env_signature="e")
    real_unlink = Path.unlink

    def failing_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.suffix == ".sqsh":
            raise OSError("archive unlink failed")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    with pytest.raises(OSError, match="archive unlink failed"):
        store.delete(record.snapshot_id)
    assert removed == [record.snapshot_id]
