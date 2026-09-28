"""MegatronRLTrainer: the rung-0/1 Megatron-Core RL driver for FoundationScale.

Runnable inside the Megatron-Bridge container under torchrun::

    python -m foundationscale.rl.megatron.driver --hf-model DIR --tp 2 --pp 1 \\
        --steps 10 --rollout-jsonl rows.jsonl --metrics-out metrics.jsonl

Rungs 0-1 use MOCK rollout: rows are fixed ``{"prompt", "completion",
"reward"}`` JSONL records, tokenized once, so the loop exercises old-logprob
capture, group-normalized advantages, forward_backward, and the distributed
optimizer without a generation engine. ``--parity-only`` loads the model,
computes mcore token logprobs for the rows, and dumps them for an external
HF comparison script (rung 0's parity gate).

Model build follows the known-working Bridge call pattern from
``run_gspo.py``: recipe/AutoBridge config -> parallelism fields ->
``initialize_megatron`` -> ``ProcessGroupCollection.use_mpu_process_groups``
-> ``get_model(...)`` -> ``setup_optimizer(...)`` with the distributed
optimizer's fp32 masters.

Every heavy import (megatron, bridge, transformers) is lazy, so this module
imports on a CPU-only box; calling anything that needs megatron without it
installed raises a named ImportError. One JSON line per step goes to
``--metrics-out`` with step, loss, ratio_mean, clip_fraction, grad_norm,
tokens, and param_hash (rank-local sha256 of a fixed parameter slice, for
DP-equality checks; it is a SHARED-id equality probe, not a content hash of
the full model).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from foundationscale.rl.megatron.lane_config import MegatronLaneConfig

__all__ = (
    "MockRolloutRow",
    "MegatronRLTrainer",
    "build_arg_parser",
    "collate_mock_batch",
    "group_relative_advantages",
    "load_mock_rollouts",
    "main",
    "param_hash",
    "require_megatron",
)


@dataclass(frozen=True, slots=True)
class MockRolloutRow:
    """One mock-rollout record: prompt/completion strings plus reward."""

    prompt: str
    completion: str
    reward: float


def require_megatron(caller: str) -> None:
    """Named refusal for the lazy megatron dependency."""
    try:
        import megatron.core  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"{caller} requires megatron-core (and megatron-bridge), which "
            f"are only present in the Megatron-Bridge container; the megatron "
            f"lane imports them lazily so CPU tooling can import this module, "
            f"but training cannot proceed without them"
        ) from exc


def load_mock_rollouts(path: str | os.PathLike[str]) -> list[MockRolloutRow]:
    """Load ``{"prompt", "completion", "reward"}`` JSONL; every line must carry
    all three fields, and a malformed line is refused with its line number."""
    rows: list[MockRolloutRow] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = tuple(k for k in ("prompt", "completion", "reward") if k not in record)
            if missing:
                raise ValueError(
                    f"{path}:{lineno}: mock rollout row is missing {missing}; "
                    f"each record must declare prompt, completion and reward -- "
                    f"a fabricated default would train on a row that was never "
                    f"scored"
                )
            rows.append(
                MockRolloutRow(
                    prompt=str(record["prompt"]),
                    completion=str(record["completion"]),
                    reward=float(record["reward"]),
                )
            )
    if not rows:
        raise ValueError(f"{path}: 0 mock rollout rows; a step needs at least 1")
    return rows


def group_relative_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[int],
    eps: float = 1e-4,
) -> list[float]:
    """GRPO-style group normalization: per group, ``(r - mean) / (std + eps)``
    with population std. Groups are identified by ``group_ids`` (prompts hold
    equal-id completions). Every reward must belong to a group of size >= 1;
    size-1 groups collapse to advantage 0.0 (no signal, honestly reported).
    """
    if len(rewards) != len(group_ids):
        raise ValueError(
            f"group_relative_advantages: {len(rewards)} rewards but "
            f"{len(group_ids)} group ids; 1 id per reward is required"
        )
    groups: dict[int, list[int]] = {}
    for idx, gid in enumerate(group_ids):
        groups.setdefault(int(gid), []).append(idx)
    out = [0.0] * len(rewards)
    for members in groups.values():
        values = [float(rewards[i]) for i in members]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)
        for i, v in zip(members, values, strict=True):
            # A zero-variance group carries no signal; with eps=0 it would divide by 0.
            out[i] = 0.0 if std == 0.0 else (v - mean) / (std + eps)
    return out


def collate_mock_batch(
    rows: Sequence[MockRolloutRow],
    advantages: Sequence[float],
    tokenizer: Any,
    seq_len: int,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Tokenize and pad mock rows into the carrier dict pp_step consumes.

    ``input_ids [B,S]`` right-padded; ``loss_mask`` is 1 over completion
    tokens only (prompt positions are context, not supervision -- design
    section 2's carrier contract) and future-shifted downstream by the loss.
    Rows longer than ``seq_len`` are right-truncated; the completion mask is
    truncated with them so masks can never exceed the id plane.
    """
    if len(rows) != len(advantages):
        raise ValueError(
            f"collate_mock_batch: {len(rows)} rows but {len(advantages)} "
            f"advantages; every rollout row needs exactly one advantage scalar"
        )
    batch_ids: list[list[int]] = []
    batch_mask: list[list[float]] = []
    for row in rows:
        p_ids = tokenizer(row.prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(row.completion, add_special_tokens=False)["input_ids"]
        ids = list(p_ids) + list(c_ids)
        mask = [0.0] * len(p_ids) + [1.0] * len(c_ids)
        ids = ids[:seq_len]
        mask = mask[:seq_len]
        batch_ids.append(ids)
        batch_mask.append(mask)
    width = max((len(r) for r in batch_ids), default=1)
    for ids, mask in zip(batch_ids, batch_mask, strict=True):
        pad = width - len(ids)
        ids.extend([pad_id] * pad)
        mask.extend([0.0] * pad)
    return {
        "input_ids": torch.tensor(batch_ids, dtype=torch.long),
        "loss_mask": torch.tensor(batch_mask, dtype=torch.float32),
        "sample_mask": torch.tensor(
            [1.0 if any(m > 0 for m in row) else 0.0 for row in batch_mask],
            dtype=torch.float32,
        ),
        "advantages": torch.tensor([float(a) for a in advantages], dtype=torch.float32),
        "attention_mask": (torch.tensor(batch_ids, dtype=torch.long) != pad_id).to(torch.float32),
    }


def param_hash(model: Any, numel: int = 4096) -> str:
    """Rank-local sha256 over the first ``numel`` entries of the first
    parameter, fp32-cast. Used for DP-equality smoke checks ("dpcheck"): it is
    an equality probe across ranks of the same shard, NOT a full-model hash,
    and is named as such in the metrics row."""
    params = (
        list(model.parameters())
        if not isinstance(model, (list, tuple))
        else [p for chunk in model for p in chunk.parameters()]
    )
    if not params:
        raise ValueError("param_hash: model exposes 0 parameters")
    flat = params[0].detach().to(torch.float32).reshape(-1)[:numel].cpu()
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()


def _iter_microbatches(
    batch: dict[str, torch.Tensor], mbs: int
) -> Iterator[dict[str, torch.Tensor]]:
    total = batch["input_ids"].shape[0]
    for start in range(0, total, mbs):
        yield {k: v[start : start + mbs] for k, v in batch.items()}


class MegatronRLTrainer:
    """Minimal rung-0/1 RL loop over a Bridge-built Megatron model.

    Owns: model build (Bridge), old-logprob capture, mock rollout collation,
    the normalized forward_backward pass, optimizer step, and the per-step
    JSON metrics row. Checkpoints, refit, and vLLM rollout are out of scope
    for rungs 0-1.
    """

    def __init__(self, cfg: MegatronLaneConfig, hf_model: str, *, lr: float, seed: int) -> None:
        require_megatron("MegatronRLTrainer.__init__")
        self.cfg = cfg
        self.hf_model = hf_model
        self.lr = float(lr)
        self.seed = int(seed)
        self.state: Any = None
        self.model: Any = None
        self.optimizer: Any = None
        self.scheduler: Any = None
        self.pg: Any = None
        self._built = False

    # -- build -------------------------------------------------------------

    def build(self) -> None:
        """Initialize model parallelism and build model+optimizer via Megatron-Bridge.

        Uses the provider call sequence measured to work in the Bridge container
        (rung-0 parity probe): AutoBridge.from_hf_pretrained -> to_megatron_provider
        -> parallelism fields -> finalize -> initialize_model_parallel ->
        provide_distributed_model (DDP-wrapped) -> get_megatron_optimizer with a
        distributed optimizer and fp32 masters. The learning rate is constant; the
        RL loop owns step counting, so no scheduler is built.
        """
        require_megatron("MegatronRLTrainer.build")
        import torch.distributed as dist
        from megatron.bridge import AutoBridge
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
        from megatron.core.process_groups_config import ProcessGroupCollection

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        world = dist.get_world_size()
        self.cfg.validate(world)

        torch.manual_seed(self.seed)
        bridge = AutoBridge.from_hf_pretrained(self.hf_model)
        provider = bridge.to_megatron_provider(load_weights=True)
        provider.tensor_model_parallel_size = self.cfg.tp
        provider.pipeline_model_parallel_size = self.cfg.pp
        provider.context_parallel_size = self.cfg.cp
        provider.sequence_parallel = bool(self.cfg.sp and self.cfg.tp > 1)
        provider.expert_model_parallel_size = self.cfg.ep
        provider.expert_tensor_parallel_size = self.cfg.etp if self.cfg.etp > 0 else None
        provider.params_dtype = torch.bfloat16
        provider.bf16 = True
        provider.finalize()
        provider.initialize_model_parallel(seed=self.seed)
        pg = ProcessGroupCollection.use_mpu_process_groups()

        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            grad_reduce_in_fp32=True,
            average_in_collective=False,
            check_for_nan_in_grad=True,
        )
        model_list = provider.provide_distributed_model(
            ddp_config=ddp_config, wrap_with_ddp=True, bf16=True, pg_collection=pg
        )
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=self.lr,
                min_lr=self.lr,
                weight_decay=0.0,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                clip_grad=1.0,
            ),
            model_list,
            pg_collection=pg,
        )
        self.state, self.model, self.optimizer, self.scheduler, self.pg = (
            provider,
            model_list[0] if len(model_list) == 1 else model_list,
            optimizer,
            None,
            pg,
        )
        self._built = True

    # -- logprobs ----------------------------------------------------------

    def capture_logprobs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Old-logprob capture: same forward path as training, forward_only."""
        if not self._built:
            raise RuntimeError("capture_logprobs before build()")
        from megatron.core.pipeline_parallel import get_forward_backward_func

        from foundationscale.rl.megatron.pp_step import make_logprob_forward_step

        device = next(self.model.parameters()).device
        mb = [
            {k: v.to(device) for k, v in micro.items()}
            for micro in _iter_microbatches(batch, self.cfg.mbs)
        ]
        out = get_forward_backward_func()(
            forward_step_func=make_logprob_forward_step(self.cfg),
            data_iterator=iter(mb),
            model=self.model,
            num_microbatches=len(mb),
            seq_length=batch["input_ids"].shape[1],
            micro_batch_size=self.cfg.mbs,
            forward_only=True,
        )
        chunks = [o["logprobs"] for o in out if isinstance(o, dict) and "logprobs" in o]
        if not chunks:
            raise RuntimeError(
                "logprob capture returned 0 microbatch payloads on the last "
                "stage; a dropped metrics/collection channel is a refusal, "
                "not a zero"
            )
        return torch.cat(chunks, dim=0)

    # -- train -------------------------------------------------------------

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        objective_loss_fn: Any,
        den: Any,
    ) -> dict[str, float | str]:
        """One normalized forward_backward + optimizer step."""
        if not self._built:
            raise RuntimeError("train_step before build()")
        import torch.distributed as dist
        from megatron.core.pipeline_parallel import get_forward_backward_func

        from foundationscale.rl.megatron.pp_step import make_forward_step

        self.model.zero_grad_buffer()
        self.optimizer.zero_grad()
        device = next(self.model.parameters()).device
        mb = [
            {k: v.to(device) for k, v in micro.items()}
            for micro in _iter_microbatches(batch, self.cfg.mbs)
        ]
        losses_reduced = get_forward_backward_func()(
            forward_step_func=make_forward_step(objective_loss_fn, den, self.cfg),
            data_iterator=iter(mb),
            model=self.model,
            num_microbatches=len(mb),
            seq_length=batch["input_ids"].shape[1],
            micro_batch_size=self.cfg.mbs,
            forward_only=False,
            do_not_average_loss=True,
        )
        from megatron.core.distributed import finalize_model_grads

        model_list = self.model if isinstance(self.model, list) else [self.model]
        finalize_model_grads(model_list, None, pg_collection=self.pg)
        update_ok, grad_norm, _ = self.optimizer.step()
        if update_ok and self.scheduler is not None:
            self.scheduler.step(increment=self.cfg.gbs)
        for chunk in model_list:
            if hasattr(chunk, "start_param_sync"):
                chunk.start_param_sync(force_sync=True)

        token_scale: dict[str, float] = {
            "loss": 0.0,
            "ratio_mean": 0.0,
            "clip_fraction": 0.0,
            "tokens": 0.0,
        }
        for entry in losses_reduced or []:
            if not isinstance(entry, dict):
                continue
            weight = float(entry.get("tokens", 0.0))
            token_scale["tokens"] += weight
            token_scale["loss"] += float(entry.get("loss", 0.0))
            if weight > 0.0:
                token_scale["ratio_mean"] += float(entry.get("ratio_mean", 0.0)) * weight
                token_scale["clip_fraction"] += float(entry.get("clip_fraction", 0.0)) * weight
        total = token_scale["tokens"]
        if total > 0.0:
            token_scale["ratio_mean"] /= total
            token_scale["clip_fraction"] /= total
        if dist.is_initialized():
            reduced = torch.tensor(
                [
                    token_scale["loss"],
                    token_scale["ratio_mean"],
                    token_scale["clip_fraction"],
                    total,
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(reduced)
            token_scale["loss"] = float(reduced[0])
            gtotal = float(reduced[3])
            token_scale["tokens"] = gtotal
            if gtotal > 0.0:
                token_scale["ratio_mean"] = float(reduced[1]) / gtotal
                token_scale["clip_fraction"] = float(reduced[2]) / gtotal
        token_scale["grad_norm"] = float(grad_norm or 0.0)
        return {**token_scale, "param_hash": param_hash(self.model)}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="foundationscale.rl.megatron.driver")
    ap.add_argument("--hf-model", required=True, help="HF model dir inside the Bridge container")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--cp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--etp", type=int, default=1)
    ap.add_argument("--sp", action="store_true")
    ap.add_argument("--mbs", type=int, default=1)
    ap.add_argument("--gbs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--num-experts", type=int, default=0)
    ap.add_argument("--softcap", type=float, default=None)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--group-size", type=int, default=1, help="completions per prompt in mock rows")
    ap.add_argument("--rollout-jsonl", required=True)
    ap.add_argument("--metrics-out", required=True)
    ap.add_argument("--parity-only", action="store_true")
    ap.add_argument("--parity-rows", type=int, default=8)
    ap.add_argument("--out", default="", help="parity logprob dump path")
    ap.add_argument(
        "--algorithm",
        default="dapo",
        help="registered token-level objective whose declared axes the loss reads",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    require_megatron("driver.main")
    from transformers import AutoTokenizer

    from foundationscale.rl.megatron.normalization import compute_denominators
    from foundationscale.rl.registry import lookup_algorithm
    from foundationscale.rl.torch_backend import TensorPolicyLoss

    cfg = MegatronLaneConfig(
        tp=args.tp,
        pp=args.pp,
        cp=args.cp,
        ep=args.ep,
        etp=args.etp,
        sp=bool(args.sp),
        mbs=args.mbs,
        gbs=args.gbs,
        seq_len=args.seq_len,
        softcap=args.softcap,
        head_dim=args.head_dim,
        num_experts=args.num_experts,
    )
    trainer = MegatronRLTrainer(cfg, args.hf_model, lr=args.lr, seed=args.seed)
    trainer.build()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    pad_id = int(
        getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0
    )

    rows = load_mock_rollouts(args.rollout_jsonl)

    if args.parity_only:
        if not args.out:
            raise ValueError("--parity-only requires --out for the logprob dump")
        batch = collate_mock_batch(
            rows[: args.parity_rows],
            [0.0] * min(len(rows), args.parity_rows),
            tokenizer,
            args.seq_len,
            pad_id,
        )
        logprobs = trainer.capture_logprobs(batch)
        Path(args.out).write_text(
            json.dumps({"input_ids": batch["input_ids"].tolist(), "logprobs": logprobs.tolist()})
            + "\n",
            encoding="utf-8",
        )
        return 0

    objective = getattr(lookup_algorithm(args.algorithm), "_objective", None)
    if objective is None or getattr(objective, "reduction", None) != "token_mean":
        raise SystemExit(
            f"--algorithm {args.algorithm!r} does not carry a token_mean objective; "
            "rung 1 normalizes with token-level global denominators only"
        )
    objective_loss_fn = TensorPolicyLoss(objective=objective)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(metrics_path).open("a", encoding="utf-8") as fh:
        for step in range(args.steps):
            group_ids = [i // max(1, args.group_size) for i in range(len(rows))]
            advantages = group_relative_advantages([r.reward for r in rows], group_ids)
            batch = collate_mock_batch(rows, advantages, tokenizer, args.seq_len, pad_id)
            old_logprobs = trainer.capture_logprobs(batch)[:, 1:]
            batch["old_logprobs"] = old_logprobs.cpu()
            den = compute_denominators(
                "token", batch["loss_mask"][:, 1:], batch["sample_mask"], group=None
            )
            metrics = trainer.train_step(batch, objective_loss_fn, den)
            fh.write(json.dumps({"step": step, **metrics}) + "\n")
            fh.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
