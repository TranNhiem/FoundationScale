"""The Alpamayo trajectory server for the FS driving harness; run it in Alpamayo's own environment.

    python -m foundationscale.vla.adapters.alpamayo.server --port P \
        --config-module alpamayo2_sft.configs --config-name sft_eval_trajectory [overrides ...]

It builds everything the way alpamayo-recipes' own evaluator (``alpamayo2_sft.evaluate_hf``) does:
the same Hydra config, ``set_seed(config.seed)`` once, ``build_trainer(config)``, the model prepared
for evaluation and the eval dataloader walked in order. Each ``next`` request runs the evaluator's
chunked sampling on the next clip (``sample_trajectories_from_data`` in chunks of
``num_samples_per_forward`` until ``num_traj_samples``) and returns the predictions ``[N, K, T, 3]``
and the ground truth ``[T, 3]`` -- so an FS run that walks the same clips in the same order sees the
same samples. FS owns the clip budget, the metric and the report. Upstream public API only.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from typing import Any

from foundationscale.vla.wire import DEFAULT_AUTHKEY, encode_array

__all__ = ["TrajectoryStream", "handle", "main", "serve"]


class TrajectoryStream:
    """The evaluator's per-clip loop, one clip per :meth:`next`."""

    def __init__(self, batches: Iterator[Any], predict: Any, total: int) -> None:
        self._batches = batches
        self._predict = predict
        self.total = total
        self.index = 0

    def next(self) -> dict[str, Any]:
        batch = next(self._batches)  # StopIteration -> refused by handle()
        pred, gt, clip_id = self._predict(batch)
        reply = {
            "ok": True,
            "index": self.index,
            "clip_id": clip_id,
            "pred": encode_array(pred),
            "gt": encode_array(gt),
        }
        self.index += 1
        return reply


def handle(stream: TrajectoryStream, message: Any) -> dict[str, Any]:
    """One request -> one reply; a bad request, exhausted stream or model error is a refusal."""
    if not isinstance(message, dict) or message.get("op") not in ("info", "next"):
        return {"ok": False, "error": f"unknown request {message!r}; expected op info or next"}
    if message["op"] == "info":
        return {"ok": True, "total": stream.total, "index": stream.index}
    try:
        return stream.next()
    except StopIteration:
        return {"ok": False, "error": f"no clip left after {stream.index} of {stream.total}"}
    except Exception as exc:  # noqa: BLE001  -- reported to the client, which records the clip as failed
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def serve(stream: TrajectoryStream, listener: Any, *, max_connections: int | None = None) -> int:
    """Answer one connection at a time; a port probe that never authenticates is skipped."""
    from multiprocessing import AuthenticationError  # noqa: PLC0415

    served = connections = 0
    while max_connections is None or connections < max_connections:
        try:
            conn = listener.accept()
        except (EOFError, ConnectionError, AuthenticationError):
            continue
        connections += 1
        try:
            while True:
                try:
                    message = conn.recv()
                except EOFError:
                    break
                conn.send(handle(stream, message))
                served += 1
        finally:
            conn.close()
    return served


def build_stream(
    config: Any,
) -> TrajectoryStream:  # pragma: no cover - needs Alpamayo, a GPU and PAI data
    """The evaluator's setup and per-clip prediction, via alpamayo-recipes' public functions."""
    from itertools import islice  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from alpamayo2_sft.evaluate_hf import build_trainer  # noqa: PLC0415
    from transformers import set_seed  # noqa: PLC0415

    set_seed(config.seed)
    trainer = build_trainer(config)
    model = trainer.accelerator.prepare_model(trainer.model, evaluation_mode=True)
    gen = trainer.accelerator.unwrap_model(model)
    gen.eval()
    loader = trainer.get_eval_dataloader()
    steps = config.evaluate.max_eval_steps
    total = steps if steps is not None and steps >= 0 else len(loader)
    batches = iter(islice(loader, total))
    dtype = getattr(torch, config.evaluate.autocast_dtype)
    ev = config.evaluate

    def predict(batch: Any) -> tuple[Any, Any, str]:
        data = {"tokenized_data": batch["tokenized_data"], **batch["traj_data"]}
        chunks, remaining = [], ev.num_traj_samples
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            while remaining > 0:
                size = min(remaining, ev.num_samples_per_forward)
                pred, _, _ = gen.sample_trajectories_from_data(
                    data=data,
                    num_traj_samples=size,
                    num_traj_sets=ev.num_traj_sets,
                    top_p=ev.top_p,
                    temperature=ev.temperature,
                    max_generation_length=ev.max_generation_length,
                )
                chunks.append(pred)
                remaining -= size
        pred = torch.cat(chunks, dim=2)[0].float().cpu().numpy()  # [N, K, T, 3]
        gt = batch["traj_data"]["ego_future_xyz"][:, -1][0].float().cpu().numpy()  # [T, 3]
        clip = batch.get("clip_id", [None])
        return pred, gt, str(clip[0] if isinstance(clip, list | tuple) else clip)

    return TrajectoryStream(batches, predict, total)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m foundationscale.vla.adapters.alpamayo.server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config-module", default="alpamayo2_sft.configs")
    parser.add_argument("--config-name", default="sft_eval_trajectory")
    parser.add_argument(
        "overrides", nargs="*", help="Hydra overrides, as for alpamayo2_sft.evaluate_hf"
    )
    args = parser.parse_args(argv)

    from multiprocessing.connection import Listener  # noqa: PLC0415

    from hydra import compose, initialize_config_module  # noqa: PLC0415

    with initialize_config_module(config_module=args.config_module, version_base=None):
        config = compose(config_name=args.config_name, overrides=list(args.overrides))
    stream = build_stream(config)
    with Listener((args.host, args.port), authkey=DEFAULT_AUTHKEY) as listener:
        print(
            f"Alpamayo trajectory server on {args.host}:{args.port} ({stream.total} clips)",
            flush=True,
        )
        serve(stream, listener)
    return 0


if __name__ == "__main__":
    sys.exit(main())
