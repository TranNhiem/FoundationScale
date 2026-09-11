#!/usr/bin/env python3
"""T1-6: a checkpoint whose shard is a symlink to the base is CAUGHT (#343).

The claim under test is not "the detector raises". It is the narrower and more
useful one: *no link-following stat is issued for a linked shard at all*. A guard
placed after ``Path.exists()`` still refuses, but ``exists()`` has already followed
the link and observed the target's size -- the byte credit is read before the
refusal, so the ORDER of the two questions is load-bearing rather than stylistic.
This harness therefore measures three verdicts and one counter:

  ARM R (real)      a real checkpoint directory must read cleanly. This is the
                    control arm the matrix requires: it must be GREEN whether or
                    not the fix is present, so a RED here is a false alarm, not a
                    detection.
  ARM L (link)      a directory whose shards are symlinks to ARM R's shards --
                    byte-identical content, different inode shape -- must be
                    REFUSED, and the refusal must name the link's target.
  ARM D (dangling)  a symlink whose target does not exist must be reported as a
                    LINK, not as a missing shard. Both facts are true; "is a link"
                    is the more specific one and the one that explains the byte
                    credit.
  BYTE CREDIT       os.stat / os.path.getsize / Path.stat are wrapped for the
                    duration of ARM L and every link-following call on a shard
                    path is counted. The arm requires zero.

Two ways to run it:

  --self-test   builds its own fixtures under TemporaryDirectory and proves each
                instrument fires BOTH ways (see CONTROLS below). Pure stdlib, no
                GPU, no cluster, no estate access -- a laptop and the estate see
                the same eight controls.

  <ckpt-dir>    runs the three arms against a REAL safetensors checkpoint. Only
                shard HEADERS are read (8-byte length prefix + JSON), never tensor
                data, so a 50 GB checkpoint costs the same as a 50 KB one. The
                scratch directories hold symlinks and one copy of the index file;
                nothing under the checkpoint directory is written or removed.

Exit codes follow the repository's four-state contract: 0 every arm as expected,
5 an arm disagreed, 95 the inputs needed to measure were unavailable, 96 refused.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

GREEN = 0
RED = 5
UNMEASURED = 95
REFUSE = 96

INDEX_NAME = "model.safetensors.index.json"


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def _shard_names(index_path: Path) -> list[str]:
    weight_map = json.loads(index_path.read_text())["weight_map"]
    return sorted(set(weight_map.values()))


def _link_dir(dest: Path, index_path: Path, targets: dict[str, str]) -> Path:
    """A checkpoint directory whose shards are symlinks to ``targets``."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(index_path, dest / INDEX_NAME)
    for name, target in targets.items():
        os.symlink(target, os.fspath(dest / name))
    return dest


def _run_arms(real: Path, out: list[str]) -> list[str]:
    """Run ARM R, ARM L (with the byte-credit probe) and ARM D. Returns failures."""
    from foundationscale.checkpoint.dcp import CheckpointFormatError
    from foundationscale.checkpoint.dcp_meta import read_metadata

    index_path = real / INDEX_NAME
    names = _shard_names(index_path)
    sizes = {n: os.lstat(real / n).st_size for n in names}
    out.append(f"real checkpoint: {len(names)} shard(s), {sum(sizes.values())} bytes total")

    failures: list[str] = []

    # ARM R -- the control arm. A real checkpoint must stay readable.
    try:
        meta = read_metadata(os.fspath(real))
        out.append(f"ARM R (real):     GREEN -- {len(meta.tensors)} entries, format={meta.format}")
    except Exception as exc:  # noqa: BLE001 -- any raise at all is the failure
        failures.append(f"ARM R raised on a real checkpoint: {type(exc).__name__}: {exc}")
        out.append(f"ARM R (real):     RED   -- {type(exc).__name__}: {exc}")

    scratch = Path(tempfile.mkdtemp(prefix="t1_6_"))
    try:
        # ARM L -- identical bytes, link inode, plus the byte-credit counter.
        linked = _link_dir(scratch / "linked", index_path, {n: os.fspath(real / n) for n in names})
        shard_paths = {os.fspath(linked / n) for n in names}
        follows: list[str] = []
        with _count_link_follows(shard_paths, follows):
            try:
                read_metadata(os.fspath(linked))
            except CheckpointFormatError as exc:
                if "symbolic link" in str(exc):
                    out.append(f"ARM L (link):     RED as required -- {str(exc)[:120]}")
                else:
                    failures.append(f"ARM L refused for another reason: {exc}")
                    out.append(f"ARM L (link):     WRONG REASON -- {str(exc)[:120]}")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"ARM L raised {type(exc).__name__}, not CheckpointFormatError")
                out.append(f"ARM L (link):     WRONG TYPE -- {type(exc).__name__}: {exc}")
            else:
                failures.append("ARM L did NOT raise on a directory of symlinked shards")
                out.append("ARM L (link):     GREEN -- DEFECT: the link was accepted")

        if follows:
            failures.append(
                f"{len(follows)} link-following stat(s) on shard paths before the refusal: "
                f"the target's bytes were read even though the shard was refused"
            )
            out.append(f"BYTE CREDIT:      RED   -- {len(follows)} link-following stat(s)")
        else:
            out.append("BYTE CREDIT:      GREEN -- 0 link-following stats on shard paths")

        # ARM D -- dangling link is a LINK, not a missing shard.
        dangling = _link_dir(
            scratch / "dangling",
            index_path,
            {n: os.fspath(scratch / f"__absent__{n}") for n in names},
        )
        try:
            read_metadata(os.fspath(dangling))
        except CheckpointFormatError as exc:
            msg = str(exc)
            if "symbolic link" in msg:
                out.append(f"ARM D (dangling): RED as a LINK -- {msg[:120]}")
            elif "missing" in msg:
                failures.append("ARM D reported a dangling link as 'missing', not as a link")
                out.append(f"ARM D (dangling): reported MISSING -- {msg[:120]}")
            else:
                failures.append(f"ARM D refused for a third reason: {msg}")
                out.append(f"ARM D (dangling): OTHER -- {msg[:120]}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"ARM D raised {type(exc).__name__}: {exc}")
            out.append(f"ARM D (dangling): WRONG TYPE -- {type(exc).__name__}: {exc}")
        else:
            failures.append("ARM D did NOT raise on dangling links")
            out.append("ARM D (dangling): GREEN -- DEFECT")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return failures


class _count_link_follows:  # noqa: N801 -- a context manager used as a verb
    """Count link-following stats issued on any path in ``watched``.

    ``os.lstat`` is deliberately NOT wrapped: it is the correct call and counting it
    would make the instrument fire on the fix as well as on the defect.
    """

    def __init__(self, watched: set[str], sink: list[str]) -> None:
        self._watched = watched
        self._sink = sink

    def __enter__(self) -> _count_link_follows:
        self._stat, self._getsize, self._pathstat = os.stat, os.path.getsize, Path.stat
        watched, sink = self._watched, self._sink
        real_stat, real_getsize, real_pathstat = self._stat, self._getsize, self._pathstat

        def counted_stat(path, *a, **kw):  # type: ignore[no-untyped-def]
            if os.fspath(path) in watched and kw.get("follow_symlinks", True):
                sink.append(os.fspath(path))
            return real_stat(path, *a, **kw)

        def counted_getsize(path):  # type: ignore[no-untyped-def]
            if os.fspath(path) in watched:
                sink.append(os.fspath(path))
            return real_getsize(path)

        def counted_pathstat(self_, *a, **kw):  # type: ignore[no-untyped-def]
            if os.fspath(self_) in watched and kw.get("follow_symlinks", True):
                sink.append(os.fspath(self_))
            return real_pathstat(self_, *a, **kw)

        os.stat = counted_stat  # type: ignore[assignment]
        os.path.getsize = counted_getsize  # type: ignore[assignment]
        Path.stat = counted_pathstat  # type: ignore[assignment,method-assign]
        return self

    def __exit__(self, *exc: object) -> None:
        os.stat = self._stat  # type: ignore[assignment]
        os.path.getsize = self._getsize  # type: ignore[assignment]
        Path.stat = self._pathstat  # type: ignore[assignment,method-assign]


# ---------------------------------------------------------------------------
# Self-test -- eight controls, each planted so it fires BOTH ways
# ---------------------------------------------------------------------------


def _synthetic_checkpoint(root: Path) -> Path:
    """A minimal but genuine two-shard safetensors checkpoint."""
    root.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for i, (key, count) in enumerate((("a.weight", 4), ("b.weight", 6)), start=1):
        name = f"model-{i:05d}-of-00002.safetensors"
        nbytes = count * 2
        header = json.dumps(
            {key: {"dtype": "BF16", "shape": [count], "data_offsets": [0, nbytes]}}
        ).encode()
        with (root / name).open("wb") as fh:
            fh.write(struct.pack("<Q", len(header)))
            fh.write(header)
            fh.write(b"\0" * nbytes)
        weight_map[key] = name
    (root / INDEX_NAME).write_text(json.dumps({"weight_map": weight_map}))
    return root


def _self_test() -> int:
    from foundationscale.checkpoint.dcp import CheckpointFormatError
    from foundationscale.checkpoint.dcp_meta import read_metadata

    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    with tempfile.TemporaryDirectory(prefix="t1_6_selftest_") as tmp:
        base = Path(tmp)
        real = _synthetic_checkpoint(base / "real")
        index_path = real / INDEX_NAME
        names = _shard_names(index_path)

        # C1/C2: the link detector must fire on links and NOT on regular files.
        linked = _link_dir(base / "linked", index_path, {n: os.fspath(real / n) for n in names})
        try:
            read_metadata(os.fspath(linked))
        except CheckpointFormatError as exc:
            record("C1 link is refused", "symbolic link" in str(exc), str(exc)[:90])
        else:
            record("C1 link is refused", False, "no raise")
        copied = _synthetic_checkpoint(base / "copied")
        try:
            read_metadata(os.fspath(copied))
        except Exception as exc:  # noqa: BLE001
            record("C2 regular shard is NOT refused", False, f"{type(exc).__name__}: {exc}")
        else:
            record("C2 regular shard is NOT refused", True, "read cleanly")

        # C3/C4: the byte-credit counter must count a real follow and only that.
        watched = {os.fspath(linked / names[0])}
        planted: list[str] = []
        with _count_link_follows(watched, planted):
            os.stat(os.fspath(linked / names[0]))
        record("C3 counter fires on a link-following stat", len(planted) == 1, f"{planted!r}")
        quiet: list[str] = []
        with _count_link_follows(watched, quiet):
            os.lstat(os.fspath(linked / names[0]))
            os.stat(os.fspath(real / names[0]))
        record("C4 counter silent on lstat and on unwatched paths", not quiet, f"{quiet!r}")

        # C5: the counter is restored on exit -- a leaked patch would silently poison
        # every later arm, and the poisoned arm would still print a verdict.
        before = (os.stat, os.path.getsize, Path.stat)
        with _count_link_follows(watched, []) as probe:
            patched = (os.stat, os.path.getsize, Path.stat) != before
        restored = (os.stat, os.path.getsize, Path.stat) == before
        record(
            "C5 the probe patches on entry and restores on exit",
            patched and restored and probe is not None,
            f"patched={patched} restored={restored}",
        )

        # C6/C7: dangling link reads as a LINK; a genuinely absent shard reads as MISSING.
        dangling = _link_dir(
            base / "dangling", index_path, {n: os.fspath(base / f"__absent__{n}") for n in names}
        )
        try:
            read_metadata(os.fspath(dangling))
        except CheckpointFormatError as exc:
            record("C6 dangling link reads as a link", "symbolic link" in str(exc), str(exc)[:90])
        else:
            record("C6 dangling link reads as a link", False, "no raise")
        absent = _synthetic_checkpoint(base / "absent")
        for n in names:
            (absent / n).unlink()
        try:
            read_metadata(os.fspath(absent))
        except CheckpointFormatError as exc:
            record("C7 absent shard reads as missing", "missing" in str(exc), str(exc)[:90])
        else:
            record("C7 absent shard reads as missing", False, "no raise")

        # C8: the harness itself must REFUSE an input it cannot measure. The detail
        # reports the code that was OBSERVED, not the one expected -- a hardcoded
        # detail prints the right answer next to a failing control.
        empty = base / "empty"
        empty.mkdir()
        rc_empty = main([str(empty)])
        record("C8 no index yields UNMEASURED", rc_empty == UNMEASURED, f"rc={rc_empty}")

        # C9: end-to-end, the three arms agree on a synthetic checkpoint.
        rc_real = main([str(real)])
        record("C9 all arms agree on a fixture", rc_real == GREEN, f"rc={rc_real}")

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    failed = [name for name, ok, _ in checks if not ok]
    print(f"T1-6 self-test: {len(checks) - len(failed)}/{len(checks)} controls PASS")
    if failed:
        for name in failed:
            print(f"FAIL: {name}")
        return RED
    return GREEN


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return REFUSE
    if argv[0] == "--self-test":
        return _self_test()

    real = Path(argv[0])
    if not (real / INDEX_NAME).is_file():
        print(f"UNMEASURED: no {INDEX_NAME} under {real}")
        return UNMEASURED

    out: list[str] = []
    failures = _run_arms(real, out)
    for line in out:
        print(line)
    print("=" * 72)
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        print(f"T1-6 VERDICT RED: {len(failures)} arm(s) disagreed")
        return RED
    print("T1-6 VERDICT GREEN: real reads, link refused, dangling refused as a link, 0 byte credit")
    return GREEN


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
