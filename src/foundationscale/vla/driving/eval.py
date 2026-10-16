"""``python -m foundationscale.vla.driving.eval``: open-loop trajectory evaluation of a driving VLA.

FS owns the clip budget, the metric (:mod:`foundationscale.vla.driving.metrics`, Alpamayo's minADE
semantics) and the report; the backend serves one clip's sampled trajectories and ground truth per
request (:mod:`foundationscale.vla.adapters.alpamayo.server`, started by ``--serve-cmd`` in the
backend's own environment). The report holds every clip, failures included: a clip that raised
counts in the denominator and makes the run UNMEASURED (exit 95), never a better mean. Exit 0 is a
measurement, 95 unmeasured (no server, or any clip failed), 96 refused (bad arguments).
"""

from __future__ import annotations

import argparse
import json
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from foundationscale.vla.adapters.alpamayo.client import AlpamayoTrajectoryClient
from foundationscale.vla.driving.metrics import min_ade

EXIT_OK, EXIT_UNMEASURED, EXIT_REFUSE = 0, 95, 96
__all__ = ["EXIT_OK", "EXIT_REFUSE", "EXIT_UNMEASURED", "evaluate", "main"]


def evaluate(client: Any, clips: int) -> dict[str, Any]:
    """Score ``clips`` clips from ``client`` in its order; every clip is recorded, failed or not."""
    records: list[dict[str, Any]] = []
    for index in range(clips):
        try:
            clip_id, pred, gt = client.next()
            scores = min_ade(pred, gt)
            records.append({"index": index, "clip_id": clip_id, **scores, "error": None})
        except Exception as exc:  # noqa: BLE001  -- recorded; the run becomes UNMEASURED
            records.append(
                {"index": index, "clip_id": None, "error": f"{type(exc).__name__}: {exc}"}
            )
    scored = [r for r in records if r["error"] is None]
    keys = sorted({k for r in scored for k in r if k.startswith("min_ade")})
    means = {k: sum(r[k] for r in scored) / len(scored) for k in keys} if scored else {}
    return {
        "clips": clips,
        "scored": len(scored),
        "clip_errors": clips - len(scored),
        "measured": len(scored) == clips,
        **{f"mean_{k}": v for k, v in means.items()},
        "per_clip": records,
    }


def _wait(host: str, port: int, timeout_s: float, proc: Any) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(5)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m foundationscale.vla.driving.eval", description=__doc__
    )
    parser.add_argument("--backend", required=True, choices=["alpamayo"])
    parser.add_argument("--model", required=True, help="checkpoint the server serves (recorded)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--clips", type=int, default=None, help="default: every clip the server holds"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5650)
    parser.add_argument("--serve-cmd", default=None, help="start the server: {model}, {port}")
    parser.add_argument("--serve-timeout-s", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if args.out.is_dir():
        print(
            f"REFUSAL (exit 96): --out {args.out} is a directory, not a report file",
            file=sys.stderr,
        )
        return EXIT_REFUSE
    if args.clips is not None and args.clips < 1:
        print(f"REFUSAL (exit 96): --clips {args.clips} must be >= 1", file=sys.stderr)
        return EXIT_REFUSE
    command = None
    if args.serve_cmd:
        try:
            command = args.serve_cmd.format(model=args.model, port=args.port)
        except (KeyError, IndexError, ValueError) as exc:
            print(f"REFUSAL (exit 96): --serve-cmd does not render: {exc!r}", file=sys.stderr)
            return EXIT_REFUSE
    proc = subprocess.Popen(shlex.split(command), start_new_session=True) if command else None  # noqa: S603
    try:
        if not _wait(args.host, args.port, args.serve_timeout_s, proc):
            print(f"UNMEASURED (exit 95): no server at {args.host}:{args.port}", file=sys.stderr)
            return EXIT_UNMEASURED
        client = AlpamayoTrajectoryClient(args.host, args.port)
        clips = args.clips if args.clips is not None else client.total()
        report = evaluate(client, clips)
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=120)
    report = {"backend": args.backend, "model": args.model, **report}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    mean = report.get("mean_min_ade")
    print(
        f"{args.backend}: minADE {mean} m over {report['scored']}/{report['clips']} clips"
        f" -> {args.out}"
    )
    if not report["measured"]:
        first = next(r["error"] for r in report["per_clip"] if r["error"])
        print(
            f"UNMEASURED (exit 95): {report['clip_errors']} clip(s) failed (first: {first})",
            file=sys.stderr,
        )
        return EXIT_UNMEASURED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
