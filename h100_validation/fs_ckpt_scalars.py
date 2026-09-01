#!/usr/bin/env python3
"""Read the plain scalars a rank-local checkpoint recorded, without torch and without GPUs.

WHY THIS EXISTS
---------------
A rank-local sharded checkpoint stores per-rank bookkeeping (the fixed-batch loss taken
before the save, the global step, the optimizer-state count) alongside the tensors. Those
scalars are the evidence a resume proof reasons about, and until this tool existed there
was no way to look at them:

  * `torch.load` on such a payload raises
    "Need to initialize default process group using init_process_group before loading
    ShardedTensor" -- the tensors cannot be reconstructed outside a live world.
  * so inspecting a scalar appeared to require an allocation, a container and 8 GPUs,
    which is a preposterous cost for reading eight floats.

Neither is actually necessary. The payload is a zip; its `data.pkl` describes an ordinary
Python dict whose scalar values are plain floats and ints. A stub unpickler that refuses
to construct anything but builtins recovers those scalars and never touches storage. That
makes checkpoint auditing a login-node operation with the stdlib alone, which matters on
estates whose host interpreter predates the training stack entirely.

WHAT IT MEASURED THE FIRST TIME IT RAN
--------------------------------------
It read the per-rank `fixed_loss_before_save` out of the checkpoint whose resume this
framework had just refused, and found the eight ranks holding two distinct values with a
spread of 0.1757088900 -- identical, to every printed digit, to the 0.17570888996124268
that the run had reported as a *restore* error. The number was fully determined before the
checkpoint was written. A control model's checkpoints, read the same way, showed a spread
of exactly 0.0 across 8 of 8 ranks, which is why its proof had been passing.

DENOMINATORS
------------
Absence is reported, never assumed. A payload that does not record a key yields None for
that rank and is counted as absent, so "0 of 8 ranks recorded this" can never be misread as
"8 of 8 ranks agreed". The final save in this framework is called without a fixed loss, so
absent is a legitimate and expected state, not a defect.

Exit codes follow the four-state contract: 0 measured, 95 unmeasured (nothing to read),
96 refuse (a malformed or unreadable payload).
"""

import argparse
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path
# Type comments rather than annotations, and typing rather than PEP 585 generics: the host
# interpreter on a login node can predate the training stack by years (measured: 3.6.8), and
# this tool's whole value is that it runs THERE, with no torch and no allocation. `from
# __future__ import annotations` is a hard SyntaxError on 3.6, so it is deliberately absent.
from typing import Any, Dict, List  # noqa: F401  (referenced from type comments)

EXIT_OK = 0
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96

# Modules whose classes may be constructed for real. Everything else -- every torch,
# distributed and sharded-tensor symbol -- is stubbed, so no tensor storage is touched
# and no process group is required.
_REAL_MODULES = frozenset({"builtins", "__builtin__", "collections", "copyreg", "_codecs"})


class _StubMeta(type):
    """Answer arbitrary attribute access.

    The pickle names nested classes (`<module>.ProcessGroupState`, and friends). A plain
    class raises AttributeError on those and the load dies, so the metaclass returns the
    stub itself for any name.
    """

    def __getattr__(cls, name):
        return cls


class _Stub(metaclass=_StubMeta):
    """Stand in for any non-builtin the payload references."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _Stub

    def __setstate__(self, state):
        pass

    def __call__(self, *args, **kwargs):
        return _Stub()


class _ScalarUnpickler(pickle.Unpickler):
    """Reconstruct the payload's shape, but nothing that owns memory."""

    def find_class(self, module, name):
        if module in _REAL_MODULES:
            try:
                return super().find_class(module, name)
            except Exception:
                return _Stub
        return _Stub

    def persistent_load(self, pid):
        # Persistent ids address tensor storage. Returning a stub is what keeps this
        # reader from paging a multi-gigabyte shard in to read one float.
        return _Stub()


def read_payload_scalars(path, keys):
    """Return the requested top-level scalars, or None per key when not recorded."""
    with zipfile.ZipFile(str(path)) as archive:
        members = [n for n in archive.namelist() if n.endswith("data.pkl")]
        if not members:
            raise ValueError(f"{path.name} is not a torch zip payload (no data.pkl)")
        obj = _ScalarUnpickler(io.BytesIO(archive.read(members[0]))).load()
    if not isinstance(obj, dict):
        raise ValueError(f"{path.name} top level is {type(obj).__name__}, not a dict")
    out = {}  # type: Dict[str, Any]
    for key in keys:
        value = obj.get(key)
        out[key] = value if isinstance(value, (int, float, str, bool)) else None
    return out


def survey(checkpoint_dir, world_size, keys):
    """Read every rank payload and state, per key, how many of them recorded it."""
    per_rank = {key: [] for key in keys}  # type: Dict[str, List[Any]]
    missing_files = []  # type: List[str]
    for rank in range(world_size):
        path = checkpoint_dir / f"rank-{rank:05d}.pt"
        if not path.is_file():
            missing_files.append(path.name)
            for key in keys:
                per_rank[key].append(None)
            continue
        scalars = read_payload_scalars(path, keys)
        for key in keys:
            per_rank[key].append(scalars[key])

    manifest = {}  # type: Dict[str, Any]
    manifest_path = checkpoint_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())

    # A stated world size that is not met means the caller asked about a population this
    # tool did not see. Reporting "8 of 9" and then asserting a bare `identical_across_ranks`
    # would hand the reader the exact confusion this file exists to prevent, so an incomplete
    # denominator is a declared state: the evidence is still printed, the agreement claim is
    # renamed to say what it actually covers, and the exit is UNMEASURED rather than PASS.
    complete = not missing_files
    report = {  # type: Dict[str, Any]
        "checkpoint": str(checkpoint_dir),
        "ranks": f"{world_size - len(missing_files)} of {world_size} rank payloads present",
        "missing_payloads": missing_files,
        "complete": complete,
        "keys": {},
    }
    for key in keys:
        values = per_rank[key]
        numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        entry = {
            "per_rank": values,
            "recorded": f"{len(numeric)} of {world_size} ranks recorded this key",
            "manifest_value": manifest.get(key),
        }
        if numeric:
            entry["min"] = min(numeric)
            entry["max"] = max(numeric)
            entry["spread"] = max(numeric) - min(numeric)
            agree = max(numeric) == min(numeric)
            if complete:
                entry["identical_across_ranks"] = agree
            else:
                # Deliberately a different key name: a consumer looking for the unqualified
                # claim finds nothing rather than finding a claim wider than its evidence.
                entry["identical_across_present_ranks"] = agree
            first = values[0]
            if isinstance(first, (int, float)) and not isinstance(first, bool):
                entry["max_abs_deviation_from_rank0"] = max(abs(v - first) for v in numeric)
            entry["status"] = "MEASURED" if complete else "PARTIAL"
        else:
            # Absent is a state, not agreement. The final save records no fixed loss.
            entry["status"] = "UNMEASURED"
            entry["why"] = "no rank payload recorded this key"
        report["keys"][key] = entry
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--world-size", type=int, required=True,
                        help="number of rank payloads to expect; stated, never inferred")
    parser.add_argument("--key", action="append", default=None,
                        help="repeatable; defaults to the scalars a resume proof reads")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    keys = tuple(args.key) if args.key else (
        "fixed_loss_before_save", "global_step", "world_size", "optimizer_state_count",
    )
    if not args.checkpoint_dir.is_dir():
        print(f"REFUSE: {args.checkpoint_dir} is not a directory", file=sys.stderr)
        return EXIT_REFUSE
    if args.world_size <= 0:
        print("REFUSE: --world-size must be positive", file=sys.stderr)
        return EXIT_REFUSE

    try:
        report = survey(args.checkpoint_dir, args.world_size, keys)
    except Exception as exc:
        print(f"REFUSE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_REFUSE

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(report["ranks"])
        for key, entry in report["keys"].items():
            print(f"  {key}: {entry['recorded']}")
            print(f"    per_rank = {entry['per_rank']}")
            if entry["status"] == "MEASURED":
                print(f"    spread = {entry['spread']!r}  "
                      f"identical_across_ranks = {entry['identical_across_ranks']}")
            elif entry["status"] == "PARTIAL":
                print(f"    spread = {entry['spread']!r}  "
                      f"identical_across_present_ranks = "
                      f"{entry['identical_across_present_ranks']}")
                print(f"    PARTIAL -- {len(report['missing_payloads'])} of {args.world_size} "
                      f"declared payloads absent: {report['missing_payloads']}")
            else:
                print(f"    UNMEASURED -- {entry['why']}")

    # Fail closed on both counts: nothing read, and read over a short denominator.
    any_measured = any(e["status"] == "MEASURED" for e in report["keys"].values())
    return EXIT_OK if (any_measured and report["complete"]) else EXIT_UNMEASURED


if __name__ == "__main__":
    raise SystemExit(main())
