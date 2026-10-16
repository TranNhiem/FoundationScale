"""The Long-WAM policy server for the FS LIBERO harness; run it in Long-WAM's own environment.

    python -m foundationscale.vla.adapters.longwam.server --config EVAL_YAML --port P [k=v ...]

It builds the policy the way Long-WAM's own LIBERO evaluation does -- ``longwam.evaluation.
read_settings`` over Long-WAM's eval config plus the same ``key=value`` overrides, the global seed
through ``set_global_seed``, then ``longwam.runtime.create_policy`` -- and answers ``reset`` with
``policy.reset()`` and ``act`` with ``policy.select_action(policy_batch(obs, task))``, the call
Long-WAM's own loop makes on every step. Upstream public functions only; nothing is copied.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

from foundationscale.vla.adapters.longwam.libero import DEFAULT_AUTHKEY, decode_array, encode_array

__all__ = ["handle", "main", "serve"]


def handle(
    policy: Any, batch_fn: Callable[[dict[str, Any], str], Any], message: Any
) -> dict[str, Any]:
    """One request -> one reply; a bad request or a policy error is a refusal, never a crash."""
    if not isinstance(message, dict) or message.get("op") not in ("reset", "act"):
        return {"ok": False, "error": f"unknown request {message!r}; expected op reset or act"}
    try:
        if message["op"] == "reset":
            policy.reset()
            return {"ok": True}
        obs = {key: decode_array(value) for key, value in message["obs"].items()}
        action = policy.select_action(batch_fn(obs, str(message["task"])))[0]
        action = action.detach().cpu().numpy() if hasattr(action, "detach") else action
        return {"ok": True, "action": encode_array(action)}
    except Exception as exc:  # noqa: BLE001  -- reported to the client, which records the episode as failed
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def serve(
    policy: Any, batch_fn: Callable[..., Any], listener: Any, *, max_connections: int | None = None
) -> int:
    """Answer requests one connection at a time (the runner opens one); returns requests served."""
    served = connections = 0
    from multiprocessing import AuthenticationError  # noqa: PLC0415

    while max_connections is None or connections < max_connections:
        try:
            conn = listener.accept()
        except (EOFError, ConnectionError, AuthenticationError):
            # the FS runner first probes the port with a bare TCP connect that never authenticates;
            # that is a readiness check, not a client, and must not take the server down
            continue
        connections += 1
        try:
            while True:
                try:
                    message = conn.recv()
                except EOFError:
                    break
                conn.send(handle(policy, batch_fn, message))
                served += 1
        finally:
            conn.close()
    return served


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m foundationscale.vla.adapters.longwam.server")
    parser.add_argument(
        "--config", required=True, help="Long-WAM eval YAML (e.g. configs/eval/libero.yaml)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "overrides", nargs="*", help="Long-WAM key=value overrides (checkpoint=... etc.)"
    )
    args = parser.parse_args(argv)

    from multiprocessing.connection import Listener  # noqa: PLC0415

    from longwam.benchmarks.libero.adapter import policy_batch  # noqa: PLC0415
    from longwam.benchmarks.libero.eval_libero_single import set_global_seed  # noqa: PLC0415
    from longwam.evaluation import read_settings  # noqa: PLC0415
    from longwam.runtime import create_policy  # noqa: PLC0415

    settings = read_settings(args.config, args.overrides)
    set_global_seed(int(settings.seed), get_worker_init_fn=False)
    policy = create_policy(settings)
    with Listener((args.host, args.port), authkey=DEFAULT_AUTHKEY) as listener:
        print(f"Long-WAM policy server on {args.host}:{args.port}", flush=True)
        serve(policy, policy_batch, listener)
    return 0


if __name__ == "__main__":
    sys.exit(main())
