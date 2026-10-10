"""``python -m foundationscale.vla.eval``: the FS LIBERO harness against one backend's server.

The Level 3 entry point for VLA registry models (:mod:`foundationscale.upstream.levels`). Each
backend's PUBLISHED evaluation protocol is one named row of :data:`PROTOCOLS` -- init mode, settle
steps, replan cadence, trials per task, seed -- so a Level 3 rerun rolls out exactly what was
reproduced. The policy server is the upstream's own (openpi ``serve_policy.py``, GR00T
``run_gr00t_server.py``): either already running at ``--host/--port``, or started by
``--serve-cmd`` (a deployment-specific command line with ``{model}`` and ``{port}`` placeholders,
kept out of the repo) and stopped when the evaluation ends. The report JSON carries
``success_rate_pct`` -- the number :func:`foundationscale.upstream.levels.judge` compares.
Exit 0 when a report was written; 96 on a refused declaration; 95 when the server never came up.
"""

from __future__ import annotations

import argparse
import json
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foundationscale.vla.eval.libero_runner import run_libero
from foundationscale.vla.eval.plan import EpisodePlan, EvalPlanError

EXIT_OK = 0
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96


@dataclass(frozen=True)
class Protocol:
    """One upstream's published LIBERO evaluation protocol."""

    init: str
    num_steps_wait: int
    replan_steps: int
    trials_per_task: int
    seed: int
    default_port: int
    source: str


PROTOCOLS: dict[str, Protocol] = {
    # openpi examples/libero/main.py: fixed suite init states, 10 settle steps, replan every 5,
    # 50 trials per task, seed 7.
    "openpi": Protocol("suite_init_states", 10, 5, 50, 7, 8000, "openpi examples/libero/main.py"),
    # GR00T examples/LIBERO/README.md + gr00t/eval/rollout_policy.py: seeded resets, no settle
    # steps, 8 action steps per inference, 20 episodes per task.
    "gr00t": Protocol("seeded_reset", 0, 8, 20, 0, 5555, "Isaac-GR00T examples/LIBERO/README.md"),
}

SUITE_TASKS = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_10": 10}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m foundationscale.vla.eval", description=__doc__)
    parser.add_argument("--backend", required=True, choices=sorted(PROTOCOLS))
    parser.add_argument("--model", required=True, help="checkpoint the server serves (recorded)")
    parser.add_argument("--suite", required=True, choices=sorted(SUITE_TASKS))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--trials-per-task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--serve-cmd", default=None, help="start the server: {model}, {port}")
    parser.add_argument("--serve-timeout-s", type=float, default=900.0)
    return parser


def _wait_for_port(host: str, port: int, timeout_s: float, proc: Any) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        with socket.socket() as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(2.0)
    return False


def _policy(backend: str, host: str, port: int, protocol: Protocol) -> Any:
    if backend == "openpi":
        from foundationscale.vla.adapters.openpi.libero import OpenPiLiberoPolicy  # noqa: PLC0415

        return OpenPiLiberoPolicy(host, port, replan_steps=protocol.replan_steps)
    from foundationscale.vla.adapters.gr00t.libero import Gr00tLiberoPolicy  # noqa: PLC0415

    return Gr00tLiberoPolicy(host, port, n_action_steps=protocol.replan_steps)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = PROTOCOLS[args.backend]
    port = args.port if args.port is not None else protocol.default_port
    try:
        plan = EpisodePlan(
            suite=args.suite,
            task_ids=tuple(range(SUITE_TASKS[args.suite])),
            trials_per_task=args.trials_per_task or protocol.trials_per_task,
            seed=protocol.seed if args.seed is None else args.seed,
            init=protocol.init,  # type: ignore[arg-type]
            num_steps_wait=protocol.num_steps_wait,
        )
    except EvalPlanError as exc:
        print(f"REFUSAL (exit 96): {exc}", file=sys.stderr)
        return EXIT_REFUSE
    proc = None
    if args.serve_cmd:
        command = args.serve_cmd.format(model=args.model, port=port)
        proc = subprocess.Popen(shlex.split(command), start_new_session=True)  # noqa: S603
    try:
        if not _wait_for_port(args.host, port, args.serve_timeout_s, proc):
            print(
                f"UNMEASURED (exit 95): no policy server at {args.host}:{port} within "
                f"{args.serve_timeout_s:.0f}s",
                file=sys.stderr,
            )
            return EXIT_UNMEASURED
        report = run_libero(plan, _policy(args.backend, args.host, port, protocol))
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=60)
    low, high = report.wilson_ci()
    payload = report.to_json() | {
        "backend": args.backend,
        "model": args.model,
        "protocol_source": protocol.source,
        "success_rate_pct": 100.0 * report.rate(),
        "wilson95_pct": [100.0 * low, 100.0 * high],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(
        f"{args.backend} {args.suite}: {report.successes()}/{report.trials()} = "
        f"{100.0 * report.rate():.2f}% (Wilson 95% {100.0 * low:.1f}-{100.0 * high:.1f}) "
        f"-> {args.out}"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
