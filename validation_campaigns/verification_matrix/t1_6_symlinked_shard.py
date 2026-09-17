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
                instrument fires BOTH ways (see CONTROLS below). Stdlib fixtures,
                no GPU, no cluster, no estate access -- a laptop and the estate
                see the same seventeen controls, 14 of which also pass against a
                pre-fix tree. ``foundationscale`` itself is required and is
                bootstrapped from the checkout's own ``src/`` when no
                distribution is installed, because the suite that runs this file
                supplies whatever interpreter it is running under.

  --out-dir DIR <ckpt-dir>        (--ckpt-dir DIR is the same input as a flag,
                                   because a runner can only forward flags)
                runs the three arms against a REAL safetensors checkpoint. Only
                shard HEADERS are read (8-byte length prefix + JSON), never tensor
                data, so a 50 GB checkpoint costs the same as a 50 KB one. The
                scratch directories hold symlinks and one copy of the index file;
                nothing under the checkpoint directory is written or removed.
                ``--out-dir`` receives one JSON payload per instrument, including
                an instrument that measured RED: an absent payload cannot be told
                apart from an arm that never ran, so in this mode the flag is
                required rather than optional.

Exit codes follow the repository's four-state contract: 0 every arm as expected,
5 an arm disagreed, 95 the inputs needed to measure were unavailable, 96 refused.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# The row directory is a sibling import root, exactly as `python3 t1_6_...py`
# gives it. This row had NO boundary handler at all, so an escaping exception
# left main() and CPython exited 1 -- outside the four-state contract the
# docstring declares.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_row import write_arm_payload  # noqa: E402
from t1_interpreter_floor import classify_boundary_exception  # noqa: E402

GREEN = 0
RED = 5
UNMEASURED = 95
REFUSE = 96

ROW_ID = Path(__file__).stem
# BYTE_CREDIT is an arm even though its verdict is a counter: it is one of the
# observations this row promises to leave behind.
ARM_NAMES = ("arm_r", "arm_l", "byte_credit", "arm_d")


@dataclass(frozen=True)
class _ArmReport:
    """The observed state of one in-process arm, before it reaches disk."""

    arm: str
    status: str
    reason: str
    summary: str


def _complete_arm_reports(
    reports: list[_ArmReport], *, status: str, reason: str, summary: str
) -> None:
    """Give every arm not reached by a boundary the boundary's observed status.

    A shared precondition is not silence about the arms it prevented. It is an
    UNMEASURED or CANNOT-MEASURE fact for each one, and the reporting seam must
    be able to distinguish that from a file whose writer was never invoked.
    """

    already_reported = {report.arm for report in reports}
    for arm in ARM_NAMES:
        if arm not in already_reported:
            reports.append(_ArmReport(arm, status, reason, summary))


def _write_arm_payloads(out_dir: Path, reports: list[_ArmReport]) -> list[Path]:
    """Write exactly one runner payload for every named arm and return the paths.

    These arms are observed in this process and are not subprocess launches, so
    ``launcher_exit_code`` is deliberately absent rather than synthesised as 0
    or 5. Absent and present-but-empty carry different meanings in the runner.
    """

    observed = [report.arm for report in reports]
    if len(observed) != len(ARM_NAMES) or set(observed) != set(ARM_NAMES):
        raise RuntimeError(
            f"T1-6 arm payload set is {observed!r}, expected exactly {list(ARM_NAMES)!r}; "
            "leaving an instrument silent is itself a reporting defect (#493)"
        )
    paths: list[Path] = []
    for report in reports:
        paths.append(
            write_arm_payload(
                out_dir,
                ROW_ID,
                report.arm,
                status=report.status,
                reason=report.reason,
                excerpts=[report.summary],
            )
        )
    return paths


INDEX_NAME = "model.safetensors.index.json"


# ---------------------------------------------------------------------------
# Import bootstrap
# ---------------------------------------------------------------------------


def _repo_src_root(start: Path) -> Path | None:
    """Walk upward from ``start`` for the ``src/`` of a src-layout checkout, or None.

    The marker is ``src/foundationscale/__init__.py`` -- the package's own module file,
    not a directory that merely happens to be called ``src`` -- so an unrelated tree with
    the same directory name does not answer for this repository.
    """
    for parent in [start, *start.parents]:
        candidate = parent / "src"
        if (candidate / "foundationscale" / "__init__.py").is_file():
            return candidate
    return None


def _ensure_package_importable() -> str | None:
    """Make ``foundationscale`` importable; return the reason it is not, or None.

    ``checks/campaign_self_tests.py`` executes this file with the interpreter that is
    running the suite, and the launcher-contracts CI job runs that suite under a bare
    ``python3`` which has never had ``pip install -e .`` applied to it. An installed
    distribution therefore cannot be assumed, so this falls back to the checkout's own
    ``src/``. An environment where NEITHER works is UNMEASURED (95), never RED (5): a
    measurement that could not be taken is a different fact from one that disagreed, and
    collapsing the two is how an environment gap gets read as a defect in the tree.
    """
    try:
        import foundationscale  # noqa: F401  -- probing importability, not using it
    except ImportError:
        pass
    else:
        return None
    here = Path(__file__).resolve().parent
    src = _repo_src_root(here)
    if src is None:
        return f"no src/foundationscale/__init__.py in any ancestor of {here}"
    sys.path.insert(0, os.fspath(src))
    try:
        import foundationscale  # noqa: F401  -- probing importability, not using it
    except ImportError as exc:
        return f"{src} is on sys.path and `import foundationscale` still failed: {exc}"
    return None


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


class _CliRefusal(Exception):
    """A parsed usage problem that must stay inside the four-state contract."""


class _T16ArgumentParser(argparse.ArgumentParser):
    """argparse with usage failures lifted from its exit 2 to this row's 96."""

    def error(self, message):  # type: ignore[no-untyped-def]
        raise _CliRefusal(f"{self.prog}: error: {message}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = _T16ArgumentParser(
        allow_abbrev=False,
        description="Measure symlinked safetensors shards without following them.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="build fixtures under TemporaryDirectory and run all controls",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        metavar="DIR",
        help="directory receiving one JSON payload per arm; required without --self-test",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        metavar="DIR",
        help="real checkpoint directory, as a flag; the runner can only forward flags",
    )
    parser.add_argument(
        "checkpoint_dir",
        nargs="?",
        type=Path,
        metavar="ckpt-dir",
        help="real checkpoint directory; kept positional for existing runners",
    )
    args = parser.parse_args(argv)

    # #494: run_row.py forwards only the flags this file declares in --help, so a
    # checkpoint reachable ONLY as a positional made T1-6 unreproducible through the
    # documented runner path even though the matrix reports it MEASURED. The
    # positional stays for existing callers; --ckpt-dir is the spelling a runner can
    # actually produce. Two DIFFERENT values refuse rather than resolve: picking one
    # would guess which checkpoint the operator meant, which is the silent fallback
    # this row refuses everywhere else.
    if args.ckpt_dir is not None:
        if args.checkpoint_dir is not None and args.checkpoint_dir != args.ckpt_dir:
            raise _CliRefusal(
                f"--ckpt-dir {args.ckpt_dir} and positional ckpt-dir "
                f"{args.checkpoint_dir} name different directories"
            )
        args.checkpoint_dir = args.ckpt_dir
    return args


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


def _run_arms(real: Path, out: list[str], reports: list[_ArmReport]) -> list[str]:
    """Run the four instruments, appending both console evidence and reports.

    The measurement branches below are unchanged; every append to ``reports``
    records the status that branch had already selected rather than inferring
    status later by parsing prose.
    """
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
        summary = f"ARM R (real):     GREEN -- {len(meta.tensors)} entries, format={meta.format}"
        out.append(summary)
        reports.append(_ArmReport("arm_r", "GREEN", "real checkpoint read cleanly", summary))
    except Exception as exc:  # noqa: BLE001 -- any raise at all is the failure
        reason = f"ARM R raised on a real checkpoint: {type(exc).__name__}: {exc}"
        failures.append(reason)
        summary = f"ARM R (real):     RED   -- {type(exc).__name__}: {exc}"
        out.append(summary)
        reports.append(_ArmReport("arm_r", "RED", reason, summary))

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
                    summary = f"ARM L (link):     RED as required -- {str(exc)[:120]}"
                    out.append(summary)
                    reports.append(
                        _ArmReport(
                            "arm_l",
                            "GREEN",
                            "the refusal named a symbolic-link target",
                            summary,
                        )
                    )
                else:
                    reason = f"ARM L refused for another reason: {exc}"
                    failures.append(reason)
                    summary = f"ARM L (link):     WRONG REASON -- {str(exc)[:120]}"
                    out.append(summary)
                    reports.append(_ArmReport("arm_l", "RED", reason, summary))
            except Exception as exc:  # noqa: BLE001
                reason = f"ARM L raised {type(exc).__name__}, not CheckpointFormatError"
                failures.append(reason)
                summary = f"ARM L (link):     WRONG TYPE -- {type(exc).__name__}: {exc}"
                out.append(summary)
                reports.append(_ArmReport("arm_l", "RED", reason, summary))
            else:
                reason = "ARM L did NOT raise on a directory of symlinked shards"
                failures.append(reason)
                summary = "ARM L (link):     GREEN -- DEFECT: the link was accepted"
                out.append(summary)
                reports.append(_ArmReport("arm_l", "RED", reason, summary))

        if follows:
            reason = (
                f"{len(follows)} link-following stat(s) on shard paths before the refusal: "
                f"the target's bytes were read even though the shard was refused"
            )
            failures.append(reason)
            summary = f"BYTE CREDIT:      RED   -- {len(follows)} link-following stat(s)"
            out.append(summary)
            reports.append(_ArmReport("byte_credit", "RED", reason, summary))
        else:
            summary = "BYTE CREDIT:      GREEN -- 0 link-following stats on shard paths"
            out.append(summary)
            reports.append(
                _ArmReport(
                    "byte_credit",
                    "GREEN",
                    "zero link-following stats were observed on shard paths",
                    summary,
                )
            )

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
                summary = f"ARM D (dangling): RED as a LINK -- {msg[:120]}"
                out.append(summary)
                reports.append(
                    _ArmReport(
                        "arm_d",
                        "GREEN",
                        "the dangling target was still reported as a symbolic link",
                        summary,
                    )
                )
            elif "missing" in msg:
                reason = "ARM D reported a dangling link as 'missing', not as a link"
                failures.append(reason)
                summary = f"ARM D (dangling): reported MISSING -- {msg[:120]}"
                out.append(summary)
                reports.append(_ArmReport("arm_d", "RED", reason, summary))
            else:
                reason = f"ARM D refused for a third reason: {msg}"
                failures.append(reason)
                summary = f"ARM D (dangling): OTHER -- {msg[:120]}"
                out.append(summary)
                reports.append(_ArmReport("arm_d", "RED", reason, summary))
        except Exception as exc:  # noqa: BLE001
            reason = f"ARM D raised {type(exc).__name__}: {exc}"
            failures.append(reason)
            summary = f"ARM D (dangling): WRONG TYPE -- {type(exc).__name__}: {exc}"
            out.append(summary)
            reports.append(_ArmReport("arm_d", "RED", reason, summary))
        else:
            reason = "ARM D did NOT raise on dangling links"
            failures.append(reason)
            summary = "ARM D (dangling): GREEN -- DEFECT"
            out.append(summary)
            reports.append(_ArmReport("arm_d", "RED", reason, summary))
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
# Self-test -- fifteen controls, each planted so it fires BOTH ways
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

    def call_main(args: list[str]) -> tuple[int | None, str]:
        """Let payload-writing failures become control failures, not boundary noise."""

        try:
            return main(args), ""
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

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
        empty_report_dir = base / "arm_payloads_empty"
        empty_report_dir.mkdir()
        rc_empty, empty_call_error = call_main(["--out-dir", str(empty_report_dir), str(empty)])
        empty_detail = f"rc={rc_empty}"
        if empty_call_error:
            empty_detail += f", call error={empty_call_error}"
        record("C8 no index yields UNMEASURED", rc_empty == UNMEASURED, empty_detail)

        # C9: end-to-end, the three arms agree on a synthetic checkpoint.
        real_report_dir = base / "arm_payloads_real"
        real_report_dir.mkdir()
        rc_real, real_call_error = call_main(["--out-dir", str(real_report_dir), str(real)])
        real_detail = f"rc={rc_real}"
        if real_call_error:
            real_detail += f", call error={real_call_error}"
        record("C9 all arms agree on a fixture", rc_real == GREEN, real_detail)

        # C10/C11: the import bootstrap is itself an instrument, and an instrument that
        # can only answer one way proves nothing. C10 is the arm this file depends on --
        # the walk finds THIS checkout from THIS file's directory. C11 is the negative
        # control at the same shape: started somewhere with no package above it, the walk
        # must return None rather than volunteering an unrelated ``src``. Without C11 a
        # walk that returned the first directory it saw would look correct in C10.
        found = _repo_src_root(Path(__file__).resolve().parent)
        record(
            "C10 src-layout walk finds this checkout",
            found is not None and (found / "foundationscale" / "__init__.py").is_file(),
            f"src={found}",
        )
        outside = base / "no_package_here" / "nested"
        outside.mkdir(parents=True)
        record(
            "C11 walk declines a tree with no package",
            _repo_src_root(outside) is None,
            f"walk({outside}) -> {_repo_src_root(outside)}",
        )

        # C12-C15 exercise the reporting seam by name and by content, never by a
        # caller-spelled path. Counting files alone would miss a duplicated arm,
        # and reading a checked-in name would repeat the brittleness fixed above.
        try:
            real_payload_paths = tuple(real_report_dir.glob("*.json"))
            real_payloads = [json.loads(path.read_text()) for path in real_payload_paths]
            real_payload_arms = {payload.get("arm") for payload in real_payloads}
            real_payload_error = ""
        except Exception as exc:  # noqa: BLE001
            real_payload_paths = ()
            real_payloads = []
            real_payload_arms = set()
            real_payload_error = f"{type(exc).__name__}: {exc}"
        real_payload_detail = (
            f"{len(real_payload_paths)} file(s), "
            f"arms={sorted(str(arm) for arm in real_payload_arms)}"
        )
        if real_payload_error:
            real_payload_detail += f", read error={real_payload_error}"
        record(
            "C12 every arm writes exactly one payload",
            len(real_payload_paths) == len(ARM_NAMES) and real_payload_arms == set(ARM_NAMES),
            real_payload_detail,
        )

        failing_report_dir = base / "arm_payloads_failing"
        failing_report_dir.mkdir()
        rc_failed, failed_call_error = call_main(
            ["--out-dir", str(failing_report_dir), str(absent)]
        )
        try:
            failed_payload_paths = tuple(failing_report_dir.glob("*.json"))
            failed_payloads = [json.loads(path.read_text()) for path in failed_payload_paths]
            failed_payload_arms = {payload.get("arm") for payload in failed_payloads}
            failed_payload_error = ""
        except Exception as exc:  # noqa: BLE001
            failed_payload_paths = ()
            failed_payloads = []
            failed_payload_arms = set()
            failed_payload_error = f"{type(exc).__name__}: {exc}"
        failed_payload_detail = (
            f"rc={rc_failed}, {len(failed_payload_paths)} file(s), "
            f"arms={sorted(str(arm) for arm in failed_payload_arms)}"
        )
        if failed_call_error:
            failed_payload_detail += f", call error={failed_call_error}"
        if failed_payload_error:
            failed_payload_detail += f", read error={failed_payload_error}"
        # The fixture is an ABSENT shard, which this adjudicator answers 96 and
        # says why: the arms did not complete, and an incomplete run refutes
        # nothing. Asserting RED here would be asserting against the project's
        # own rule that a harness fault is 96, never 5 -- C7 already covers the
        # absent shard through the path that classifies it. What is worth
        # pinning is the #493 property: a run that does NOT come back GREEN
        # still leaves one payload per arm on disk, so the evidence survives
        # the failure that makes it interesting.
        record(
            "C13 a non-GREEN run still leaves one payload per arm",
            rc_failed != GREEN
            and len(failed_payload_paths) == len(ARM_NAMES)
            and failed_payload_arms == set(ARM_NAMES),
            failed_payload_detail,
        )

        rc_without_out_dir, without_out_error = call_main([str(real)])
        without_out_detail = f"rc={rc_without_out_dir}"
        if without_out_error:
            without_out_detail += f", call error={without_out_error}"
        record(
            "C14 --out-dir absent without --self-test refuses",
            rc_without_out_dir == REFUSE,
            without_out_detail,
        )

        def _arm_status(payloads: list, name: str) -> str:
            """The status the writer put on disk for one arm, or "" if absent."""
            for payload in payloads:
                if isinstance(payload, dict) and payload.get("arm") == name:
                    return str(payload.get("status", ""))
            return ""

        # The MUST-FIRE that actually closes #493. Naming one expected literal
        # would pass against a writer that hardcodes that literal, which is the
        # exact class of bug #493 is about: evidence that looks present and is
        # not the observation. Comparing the SAME arm across a clean run and a
        # faulting run cannot be satisfied by any constant -- if the status were
        # hardcoded, or dropped and defaulted, the two would be equal.
        clean_status = _arm_status(real_payloads, "arm_r")
        failed_status = _arm_status(failed_payloads, "arm_r")
        record(
            "C15 MUST-FIRE: the payload records the OBSERVED status, not a constant",
            bool(clean_status) and bool(failed_status) and clean_status != failed_status,
            f"arm_r clean={clean_status!r} vs faulting={failed_status!r}",
        )

        # C16/C17 close #494. C16 is an EQUIVALENCE control: the same checkpoint
        # supplied as --ckpt-dir must reach the same code path as the positional.
        # A flag that were declared but ignored would refuse for a missing
        # ckpt-dir instead of matching the positional run, so this cannot pass
        # vacuously -- and being declared is exactly what makes run_row forward it.
        flag_report_dir = base / "arm_payloads_flag"
        flag_report_dir.mkdir()
        rc_flag, flag_call_error = call_main(
            ["--out-dir", str(flag_report_dir), "--ckpt-dir", str(real)]
        )
        try:
            flag_arms = {
                json.loads(path.read_text()).get("arm") for path in flag_report_dir.glob("*.json")
            }
            flag_read_error = ""
        except Exception as exc:  # noqa: BLE001
            flag_arms = set()
            flag_read_error = f"{type(exc).__name__}: {exc}"
        flag_detail = (
            f"flag rc={rc_flag} vs positional rc={rc_real}, "
            f"arms={sorted(str(arm) for arm in flag_arms)}"
        )
        if flag_call_error:
            flag_detail += f", call error={flag_call_error}"
        if flag_read_error:
            flag_detail += f", read error={flag_read_error}"
        record(
            "C16 --ckpt-dir reaches the same path as the positional",
            rc_flag == rc_real and flag_arms == real_payload_arms == set(ARM_NAMES),
            flag_detail,
        )

        # C17 MUST-FIRE: two DIFFERENT checkpoints, one per spelling, is an
        # ambiguous declaration. Resolving it by preferring either spelling would
        # measure a checkpoint the operator did not ask for and then report the
        # answer as that operator's result.
        conflict_dir = base / "arm_payloads_conflict"
        conflict_dir.mkdir()
        rc_conflict, conflict_call_error = call_main(
            ["--out-dir", str(conflict_dir), "--ckpt-dir", str(real), str(absent)]
        )
        conflict_detail = f"rc={rc_conflict}"
        if conflict_call_error:
            conflict_detail += f", call error={conflict_call_error}"
        record(
            "C17 MUST-FIRE: conflicting --ckpt-dir and positional refuse",
            rc_conflict == REFUSE,
            conflict_detail,
        )

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
        print("CANNOT-MEASURE: --out-dir is required without --self-test")
        return REFUSE
    try:
        args = _parse_args(argv)
    except _CliRefusal as exc:
        print(f"CANNOT-MEASURE: {exc} -- usage errors exit 96, never argparse's 2")
        return REFUSE

    if args.self_test:
        if args.checkpoint_dir is not None:
            print("CANNOT-MEASURE: --self-test builds its own fixtures and takes no ckpt-dir")
            return REFUSE
        reason = _ensure_package_importable()
        if reason is not None:
            print(f"UNMEASURED: foundationscale is not importable -- {reason}")
            return UNMEASURED
        return _self_test()

    if args.out_dir is None:
        print("CANNOT-MEASURE: --out-dir is required without --self-test")
        return REFUSE
    if args.checkpoint_dir is None:
        print("CANNOT-MEASURE: ckpt-dir is required without --self-test")
        return REFUSE

    out_dir: Path = args.out_dir
    real: Path = args.checkpoint_dir
    reports: list[_ArmReport] = []

    reason = _ensure_package_importable()
    if reason is not None:
        detail = f"foundationscale is not importable -- {reason}"
        _complete_arm_reports(reports, status="UNMEASURED", reason=detail, summary=detail)
        _write_arm_payloads(out_dir, reports)
        print(f"UNMEASURED: {detail}")
        return UNMEASURED

    if not (real / INDEX_NAME).is_file():
        detail = f"no {INDEX_NAME} under {real}"
        _complete_arm_reports(reports, status="UNMEASURED", reason=detail, summary=detail)
        _write_arm_payloads(out_dir, reports)
        print(f"UNMEASURED: {detail}")
        return UNMEASURED

    out: list[str] = []
    try:
        failures = _run_arms(real, out, reports)
    except Exception as exc:  # noqa: BLE001 - classified, never adjudicated (#417)
        import traceback

        traceback.print_exc()
        code, why = classify_boundary_exception(exc)
        name = "UNMEASURED" if code == UNMEASURED else "CANNOT_MEASURE"
        boundary_detail = (
            f"unexpected {type(exc).__name__} escaped the arms: {exc}; "
            f"classified {name} ({code}): {why}"
        )
        _complete_arm_reports(reports, status=name, reason=boundary_detail, summary=boundary_detail)
        _write_arm_payloads(out_dir, reports)
        for line in out:
            print(line)
        print(
            f"T1-6 VERDICT {name}: unexpected {type(exc).__name__} escaped the arms: "
            f"{exc}; classified {name} ({code}): {why}"
        )
        return code

    _write_arm_payloads(out_dir, reports)
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
    # #417: an escaped exception is classified, never adjudicated -- and never
    # allowed to become CPython's exit 1, which is outside this row's declared
    # {0, 5, 95, 96}.
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        code, reason = classify_boundary_exception(exc)
        name = "UNMEASURED" if code == UNMEASURED else "CANNOT_MEASURE"
        print(
            f"T1-6 VERDICT {name}: unexpected {type(exc).__name__} escaped the run "
            f"body: {exc}; classified {name} ({code}): {reason}",
            file=sys.stderr,
        )
        sys.exit(code)
