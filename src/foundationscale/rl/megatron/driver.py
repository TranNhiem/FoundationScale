"""MegatronRLTrainer: the Megatron-Core RL driver for FoundationScale (rungs 0-2).

Runnable inside the Megatron-Bridge container under torchrun::

    python -m foundationscale.rl.megatron.driver --hf-model DIR --tp 2 --pp 1 \\
        --steps 10 --rollout-jsonl rows.jsonl --metrics-out metrics.jsonl

Rungs 0-1 use MOCK rollout: rows are fixed ``{"prompt", "completion",
"reward"}`` JSONL records, tokenized once, so the loop exercises old-logprob
capture, group-normalized advantages, forward_backward, and the distributed
optimizer without a generation engine. ``--parity-only`` loads the model,
computes mcore token logprobs for the rows, and dumps them for an external
HF comparison script (rung 0's parity gate).

Rung 2 (``--online``) is real on-policy RL: each step the CURRENT policy
generates (an HF copy on global rank 0, refit from the mcore weights through
``export_hf_weights``), a verifiable reward scores the completions, and the
same train step consumes them under any TP/PP/CP/EP/ETP/DP layout. The rollout
helpers live in :mod:`foundationscale.rl.megatron.online`. On MoE models the
router's load-balancing loss still produces gradient on a step whose
advantages are all zero; that is Megatron's configured ``moe_aux_loss_coeff``,
not a policy update.

Model build follows the known-working Bridge call pattern from
``run_gspo.py``: recipe/AutoBridge config -> parallelism fields ->
``initialize_megatron`` -> ``ProcessGroupCollection.use_mpu_process_groups``
-> ``get_model(...)`` -> ``setup_optimizer(...)`` with the distributed
optimizer's fp32 masters.

Every heavy import (megatron, bridge, transformers) is lazy, so this module
imports on a CPU-only box; calling anything that needs megatron without it
installed raises a named ImportError. One JSON line per step goes to
``--metrics-out`` with step, loss, ratio_mean, clip_fraction, grad_norm,
tokens, update_ok, and param_hash (rank-local sha256 of a strided sample of
every parameter, for DP-equality and movement checks; a sample, not a content
hash of the full model).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foundationscale.rl.megatron.lane_config import MegatronLaneConfig
from foundationscale.rl.megatron.logprobs import check_softcap_owner

if TYPE_CHECKING:
    import torch


__all__ = (
    "MockRolloutRow",
    "MegatronRLTrainer",
    "build_arg_parser",
    "collate_mock_batch",
    "group_relative_advantages",
    "load_mock_rollouts",
    "main",
    "needs_reference",
    "param_hash",
    "reduce_step_metrics",
    "reference_gap",
    "require_megatron",
    "swapped_parameters",
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


@contextlib.contextmanager
def swapped_parameters(
    params: Sequence[torch.Tensor], replacement: Sequence[torch.Tensor]
) -> Iterator[None]:
    """Swap ``replacement`` values into ``params`` for the duration of the block.

    The reference run must see the policy's forward with frozen weights: every
    original is stashed on host first and restored in ``finally``, bit-exactly,
    so the live policy survives even a body that raises. Count and shape checks
    run before any parameter is touched -- a partial swap would silently mix
    frozen and updated weights inside one forward.
    """
    import torch

    if len(params) != len(replacement):
        raise ValueError(
            f"swapped_parameters: {len(params)} parameters but {len(replacement)} "
            f"replacements; 1 replacement per parameter is required"
        )
    for i, (p, r) in enumerate(zip(params, replacement, strict=True)):
        if tuple(p.shape) != tuple(r.shape):
            raise ValueError(
                f"swapped_parameters: parameter {i} has shape {tuple(p.shape)} but "
                f"its replacement has shape {tuple(r.shape)}; the frozen reference "
                f"must match the policy snapshot it replaces"
            )
    stashes = [p.detach().to("cpu", copy=True) for p in params]
    try:
        with torch.no_grad():
            for p, r in zip(params, replacement, strict=True):
                p.data.copy_(r)
        yield
    finally:
        with torch.no_grad():
            for p, stash in zip(params, stashes, strict=True):
                p.data.copy_(stash)


def needs_reference(objective: Any) -> bool:
    """True when ``objective`` declares a nonzero KL weight (GRPO's k3 term and
    friends); only then does the batch need ``reference_logprobs``."""
    return float(getattr(objective, "kl_weight", 0.0) or 0.0) != 0.0


def reference_gap(
    old_logprobs: torch.Tensor, reference_logprobs: torch.Tensor, mask: torch.Tensor
) -> float | None:
    """Masked mean of |old_logprobs - reference_logprobs| over ``mask > 0``.

    Positive control for a KL objective: exactly 0.0 at step 0 (the policy has
    not moved off the reference), > 0 as soon as it does. An empty mask measures
    nothing and returns None (logged as null) -- never 0.0, the passing value.
    Accumulated in float64.
    """
    import torch

    kept = mask > 0
    count = int(kept.sum().item())
    if count == 0:
        return None
    diff = (old_logprobs.to(torch.float64) - reference_logprobs.to(torch.float64)).abs()
    return float(diff[kept].sum().item() / count)


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
    import torch

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


def param_hash(model: Any, per_param: int = 64) -> str:
    """Rank-local sha256 over a strided sample of EVERY parameter, fp32-cast.

    Used for DP-equality and movement smoke checks. A slice of one parameter
    is blind: the first parameter is the embedding, and its leading rows are
    token ids no batch touches, so they get zero gradient and never move.
    Sampling ``per_param`` evenly-strided entries of each parameter sees any
    update that reaches any parameter. It is still a sample, NOT a full-model
    hash, and is named as such in the metrics row."""
    import torch

    params = (
        list(model.parameters())
        if not isinstance(model, (list, tuple))
        else [p for chunk in model for p in chunk.parameters()]
    )
    if not params:
        raise ValueError("param_hash: model exposes 0 parameters")
    digest = hashlib.sha256()
    for p in params:
        flat = p.detach().reshape(-1)
        stride = max(1, flat.numel() // per_param)
        digest.update(flat[::stride][:per_param].to(torch.float32).cpu().numpy().tobytes())
    return digest.hexdigest()


def reduce_step_metrics(entries: Sequence[Any], all_reduce: Any = None) -> dict[str, float]:
    """Token-weighted step metrics from mcore's per-microbatch ``losses_reduced``.

    ``ratio_mean`` and ``clip_fraction`` arrive as per-microbatch means; they
    are summed token-weighted, all-reduced as SUMS, and divided once by the
    global token count. ``loss`` is a sum of per-microbatch shares of the
    global objective. ``all_reduce`` sums a float64 tensor in place (None on
    a single process)."""
    import torch

    sums = [0.0, 0.0, 0.0, 0.0]  # loss, ratio*tokens, clip*tokens, tokens
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        weight = float(entry.get("tokens", 0.0))
        sums[0] += float(entry.get("loss", 0.0))
        sums[1] += float(entry.get("ratio_mean", 0.0)) * weight
        sums[2] += float(entry.get("clip_fraction", 0.0)) * weight
        sums[3] += weight
    if all_reduce is not None:
        reduced = torch.tensor(sums, dtype=torch.float64)
        all_reduce(reduced)
        sums = [float(v) for v in reduced]
    total = sums[3]
    return {
        "loss": sums[0],
        "ratio_mean": sums[1] / total if total > 0.0 else 0.0,
        "clip_fraction": sums[2] / total if total > 0.0 else 0.0,
        "tokens": total,
    }


def _maybe_dump_grads(model_list: list[Any]) -> None:
    """Debug probe: ``MEG_GRAD_DUMP=<prefix>`` writes each rank's per-parameter
    grad sum-of-squares once, so parallel layouts can be diffed by name."""
    prefix = os.environ.pop("MEG_GRAD_DUMP", "")
    if not prefix:
        return
    import torch.distributed as dist

    rows = {}
    for chunk in model_list:
        for name, p in chunk.named_parameters():
            g = getattr(p, "main_grad", None)
            if g is None:
                g = p.grad
            rows[name] = {
                "sumsq": float(g.double().pow(2).sum()) if g is not None else None,
                "shape": list(p.shape),
                "tp_sharded": bool(getattr(p, "tensor_model_parallel", False)),
                "allreduce": bool(getattr(p, "allreduce", True)),
            }
    rank = dist.get_rank() if dist.is_initialized() else 0
    Path(f"{prefix}.rank{rank}.json").write_text(json.dumps(rows))


def restore_tp_attributes(model_list: list[Any]) -> int:
    """Mark mcore-native TP-sharded weights as ``tensor_model_parallel``.

    mcore sets the attribute only inside its ``perform_initialization`` branch,
    and Bridge disables that when loading HF weights. The vocab-parallel
    embedding and output layer then read as TP duplicates, so the grad norm
    (and clipping) counts only tp rank 0's shard: 4.84 vs 5.06 on the tiny MoE
    at TP2. TE layers set their own attributes and are untouched. Returns the
    number of weights marked.
    """
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
        VocabParallelEmbedding,
    )

    marked = 0
    for chunk in model_list:
        for module in chunk.modules():
            if isinstance(module, VocabParallelEmbedding | ColumnParallelLinear):
                dim = 0
            elif isinstance(module, RowParallelLinear):
                dim = 1
            else:
                continue
            weight = getattr(module, "weight", None)
            if weight is None or getattr(weight, "tensor_model_parallel", False):
                continue
            weight.tensor_model_parallel = True
            weight.partition_dim = dim
            if not hasattr(weight, "partition_stride"):
                weight.partition_stride = 1
            marked += 1
    return marked


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

    def __init__(
        self,
        cfg: MegatronLaneConfig,
        hf_model: str,
        *,
        lr: float,
        seed: int,
        fp32: bool = False,
        attention_backend: str | None = None,
    ) -> None:
        require_megatron("MegatronRLTrainer.__init__")
        self.cfg = cfg
        # fp32 weights exist for the parity gate: they separate implementation
        # error from bf16 rounding. Training runs bf16 with fp32 masters.
        self.fp32 = fp32
        # TE's fused-attention backward has aborted (SIGABRT) late in long RL runs;
        # Megatron refuses NVTE_* env overrides, so the backend is chosen here.
        self.attention_backend = attention_backend
        self.hf_model = hf_model
        self.lr = float(lr)
        self.seed = int(seed)
        self.state: Any = None
        self.model: Any = None
        self.optimizer: Any = None
        self.scheduler: Any = None
        self.pg: Any = None
        # Kept after build: rung 2 refits its generation copy through export_hf_weights.
        self.bridge: Any = None
        self._built = False
        self._reference: list[torch.Tensor] | None = None

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
        import torch

        require_megatron("MegatronRLTrainer.build")
        import torch.distributed as dist
        from megatron.bridge import AutoBridge
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
        from megatron.core.process_groups_config import ProcessGroupCollection

        if not dist.is_initialized():
            # Bind before the first collective: unbound ranks all share cuda:0, which NCCL
            # rejects as "invalid usage" at the optimizer's all_gather_object.
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            # Rank 0 alone generates and runs the held-out eval while every other rank
            # waits in broadcast_rows; the 10-minute NCCL default killed a 2-GPU run
            # whose 800-prompt PRE eval outlasted it, so the wait must be bounded by
            # work, not by the watchdog.
            dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
        world = dist.get_world_size()
        self.cfg.validate(world)

        torch.manual_seed(self.seed)
        bridge = AutoBridge.from_hf_pretrained(self.hf_model)
        self.bridge = bridge
        provider = bridge.to_megatron_provider(load_weights=True)
        check_softcap_owner(self.cfg.softcap, getattr(provider, "final_logit_softcapping", None))
        provider.tensor_model_parallel_size = self.cfg.tp
        provider.pipeline_model_parallel_size = self.cfg.pp
        provider.context_parallel_size = self.cfg.cp
        provider.sequence_parallel = bool(self.cfg.sp and self.cfg.tp > 1)
        provider.expert_model_parallel_size = self.cfg.ep
        provider.expert_tensor_parallel_size = self.cfg.etp if self.cfg.etp > 0 else None
        provider.params_dtype = torch.float32 if self.fp32 else torch.bfloat16
        provider.bf16 = not self.fp32
        if self.attention_backend is not None:
            from megatron.core.transformer.enums import AttnBackend

            provider.attention_backend = AttnBackend[self.attention_backend]
        if dist.get_rank() == 0:
            print(f"attention_backend={getattr(provider, 'attention_backend', None)}", flush=True)
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
            ddp_config=ddp_config, wrap_with_ddp=True, bf16=not self.fp32, pg_collection=pg
        )
        if self.cfg.tp > 1:
            marked = restore_tp_attributes(model_list)
            if dist.get_rank() == 0:
                print(f"[meg] restored tensor_model_parallel on {marked} weights", flush=True)
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=self.lr,
                min_lr=self.lr,
                weight_decay=0.0,
                bf16=not self.fp32,
                params_dtype=torch.float32 if self.fp32 else torch.bfloat16,
                use_distributed_optimizer=True,
                clip_grad=1.0,
            ),
            model_list,
            # mcore refuses gloo groups alongside an explicit pg_collection
            use_gloo_process_groups=False,
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
        import torch

        if not self._built:
            raise RuntimeError("capture_logprobs before build()")
        from megatron.core.pipeline_parallel import get_forward_backward_func

        from foundationscale.rl.megatron.pp_step import (
            make_logprob_forward_step,
            padded_seq_len,
            sp_tp,
        )

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
            seq_length=padded_seq_len(batch["input_ids"].shape[1], self.cfg.cp, sp_tp(self.cfg)),
            micro_batch_size=self.cfg.mbs,
            forward_only=True,
        )
        pp_group = self.pg.pp
        pp_size = pp_group.size() if pp_group is not None else 1
        is_last_stage = pp_size == 1 or pp_group.rank() == pp_size - 1
        if is_last_stage:
            chunks = [o["logprobs"] for o in out if isinstance(o, dict) and "logprobs" in o]
            if not chunks:
                raise RuntimeError(
                    "logprob capture returned 0 microbatch payloads on the last "
                    "stage; a dropped metrics/collection channel is a refusal, "
                    "not a zero"
                )
            logprobs = torch.cat(chunks, dim=0).float()
        else:
            logprobs = torch.empty(batch["input_ids"].shape, dtype=torch.float32, device=device)
        if pp_size > 1:
            import torch.distributed as dist

            # Only the last stage computes logits; every stage needs old logprobs in its batch.
            last = dist.get_global_rank(pp_group, pp_size - 1)
            dist.broadcast(logprobs, src=last, group=pp_group)
        return logprobs

    # -- reference policy --------------------------------------------------

    def _local_parameters(self) -> list[torch.Tensor]:
        """Every parameter of this rank's model chunks in a deterministic order:
        chunk order (``self.model`` may be one module or a chunk list), then
        ``named_parameters`` order."""
        chunks = self.model if isinstance(self.model, (list, tuple)) else [self.model]
        return [p for chunk in chunks for _, p in chunk.named_parameters()]

    def snapshot_reference(self) -> dict[str, int]:
        """Freeze the initial policy as the reference: this rank's shard only
        (TP/PP/EP/ETP-local), copied to host. Held on host so it costs no device
        memory, and frozen -- it scores ``reference_logprobs`` for KL objectives
        while the policy trains away from it."""
        if not self._built:
            raise RuntimeError("snapshot_reference before build()")
        if self._reference is not None:
            raise RuntimeError("snapshot_reference: reference policy already frozen")
        import torch

        frozen: list[torch.Tensor] = []
        numel = 0
        for p in self._local_parameters():
            snap = p.detach().to("cpu", copy=True)
            if torch.cuda.is_available():
                snap = snap.pin_memory()
            frozen.append(snap)
            numel += snap.numel()
        self._reference = frozen
        return {"reference_tensors": len(frozen), "reference_numel": numel}

    def capture_reference_logprobs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Logprobs of ``batch`` under the frozen reference weights.

        Collective (it calls ``capture_logprobs``): EVERY rank must enter this
        call on every step it is used. The policy weights are swapped to the
        reference and restored bit-exactly around the forward."""
        if self._reference is None:
            raise RuntimeError("capture_reference_logprobs before snapshot_reference()")
        import torch

        # Read-only scoring: no autograd graph for a forward whose output is detached.
        with torch.no_grad(), swapped_parameters(self._local_parameters(), self._reference):
            return self.capture_logprobs(batch)

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

        from foundationscale.rl.megatron.pp_step import (
            make_forward_step,
            padded_seq_len,
            sp_tp,
        )

        self.model.zero_grad_buffer()
        self.optimizer.zero_grad()
        device = next(self.model.parameters()).device
        mb = [
            {k: v.to(device) for k, v in micro.items()}
            for micro in _iter_microbatches(batch, self.cfg.mbs)
        ]
        losses_reduced = get_forward_backward_func()(
            forward_step_func=make_forward_step(
                objective_loss_fn, den, self.cfg, num_microbatches=len(mb)
            ),
            data_iterator=iter(mb),
            model=self.model,
            num_microbatches=len(mb),
            seq_length=padded_seq_len(batch["input_ids"].shape[1], self.cfg.cp, sp_tp(self.cfg)),
            micro_batch_size=self.cfg.mbs,
            forward_only=False,
        )
        from megatron.core.distributed import finalize_model_grads

        model_list = self.model if isinstance(self.model, list) else [self.model]
        finalize_model_grads(model_list, None, pg_collection=self.pg)
        _maybe_dump_grads(model_list)
        update_ok, grad_norm, _ = self.optimizer.step()
        if update_ok and self.scheduler is not None:
            self.scheduler.step(increment=self.cfg.gbs)
        for chunk in model_list:
            if hasattr(chunk, "start_param_sync"):
                chunk.start_param_sync(force_sync=True)

        def _sum_across_ranks(t: torch.Tensor) -> None:
            on_dev = t.to(device)
            # dp only: TP peers hold the same tokens, and CP ranks all see the full
            # gathered sequence, so any wider sum overcounts.
            dist.all_reduce(on_dev, group=self.pg.dp)
            # Only the last PP stage holds losses_reduced; the others contribute zeros.
            if self.pg.pp is not None and self.pg.pp.size() > 1:
                dist.all_reduce(on_dev, group=self.pg.pp)
            t.copy_(on_dev.cpu())

        token_scale: dict[str, Any] = reduce_step_metrics(
            losses_reduced or [], _sum_across_ranks if dist.is_initialized() else None
        )
        token_scale["update_ok"] = bool(update_ok)
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
    ap.add_argument(
        "--rollout-jsonl", default="", help="mock rollout rows (rungs 0-1); not read by --online"
    )
    ap.add_argument(
        "--online",
        action="store_true",
        help="rung 2: the current policy generates and a verifiable reward scores every step",
    )
    ap.add_argument("--dataset", default="", help="--online prompt corpus (ShareGPT JSON/JSONL)")
    ap.add_argument("--gold-key", default=None, help="--online record key holding the gold letter")
    ap.add_argument(
        "--answer-pattern", default=None, help="--online regex whose one group is the answer letter"
    )
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--overlong-cache",
        type=int,
        default=0,
        help="rung 2: soft overlong penalty window in tokens (DAPO); 0 disables shaping",
    )
    ap.add_argument(
        "--overlong-factor",
        type=float,
        default=1.0,
        help="rung 2: penalty at the --max-new-tokens cap when --overlong-cache > 0",
    )
    ap.add_argument(
        "--unparsed-reward",
        type=float,
        default=None,
        help="score a scorer abstention on a prompt with gold as this reward "
        "(format failure); default keeps it masked as unmeasured",
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="sampler nucleus; <1.0 biases the gradient, training logprobs are untruncated",
    )
    ap.add_argument("--heldout", default="", help="--online greedy held-out corpus")
    ap.add_argument("--heldout-n", type=int, default=0, help="first N held-out rows; 0 = all")
    ap.add_argument("--heldout-max-new", type=int, default=0, help="0 = --max-new-tokens")
    ap.add_argument("--metrics-out", required=True)
    ap.add_argument("--parity-only", action="store_true")
    ap.add_argument("--parity-rows", type=int, default=8)
    ap.add_argument("--fp32", action="store_true", help="fp32 weights (parity gate only)")
    ap.add_argument(
        "--attention-backend",
        choices=["flash", "fused", "unfused", "local", "auto"],
        default=None,
        help="Megatron AttnBackend; default leaves the provider's choice",
    )
    ap.add_argument("--out", default="", help="parity logprob dump path")
    ap.add_argument(
        "--algorithm",
        default="dapo",
        help="registered token-level objective whose declared axes the loss reads",
    )
    return ap


def _run_online(
    args: argparse.Namespace, trainer: MegatronRLTrainer, tokenizer: Any, pad_id: int
) -> int:
    """Rung 2: every step the CURRENT policy generates, a verifiable reward scores, mcore trains.

    Global rank 0 holds an HF copy of the policy, refit from the Megatron weights
    before each rollout (``export_hf_weights`` is collective, so every rank joins).
    Rows are broadcast; advantages are computed over the whole batch and each DP rank
    trains its contiguous slice, exactly as on the mock path. A step whose rows are
    all abstained or empty is skipped on every rank (the decision is a function of
    the broadcast rows, so no rank can diverge into a collective the others skip).
    """
    import time

    import torch
    import torch.distributed as dist

    from foundationscale.rl.corpus import load_sharegpt
    from foundationscale.rl.megatron import online
    from foundationscale.rl.megatron.normalization import compute_denominators
    from foundationscale.rl.megatron.pp_step import loss_unit
    from foundationscale.rl.registry import lookup_algorithm
    from foundationscale.rl.rewards import MCQLetterReward
    from foundationscale.rl.torch_backend import TensorPolicyLoss

    if not args.dataset:
        raise SystemExit("--online requires --dataset")
    if args.steps < 1:
        raise SystemExit(f"--steps must be >= 1, got {args.steps}")
    objective = getattr(lookup_algorithm(args.algorithm), "_objective", None)
    if objective is None:
        raise SystemExit(f"--algorithm {args.algorithm!r} carries no tensor objective")
    objective_loss_fn = TensorPolicyLoss(objective=objective)
    unit = loss_unit(objective_loss_fn)
    group_size = max(1, args.group_size)
    n_rows = args.prompts_per_step * group_size
    samples = load_sharegpt(args.dataset, gold_key=args.gold_key)
    heldout = (
        load_sharegpt(args.heldout, gold_key=args.gold_key, limit=args.heldout_n or None)
        if args.heldout
        else ()
    )
    reward = MCQLetterReward(answer_pattern=args.answer_pattern)
    dp_group = trainer.pg.dp
    dp_size, dp_rank = dp_group.size(), dp_group.rank()
    writer = not dist.is_initialized() or dist.get_rank() == 0
    if needs_reference(objective):
        stats = trainer.snapshot_reference()
        if writer:
            print(
                f"[meg] frozen reference: {stats['reference_tensors']} tensors, "
                f"{stats['reference_numel']} params held on host",
                flush=True,
            )
    device = torch.device("cuda", torch.cuda.current_device())
    hf_model: Any = None
    if writer:
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=torch.bfloat16)
        hf_model.to(device)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    heldout_max_new = args.heldout_max_new or args.max_new_tokens

    def _refit() -> tuple[dict[str, int], float]:
        t0 = time.perf_counter()
        stats = online.refit_hf_policy(trainer.bridge, trainer.model, hf_model, is_writer=writer)
        return stats, time.perf_counter() - t0

    with contextlib.ExitStack() as stack:
        fh = stack.enter_context(metrics_path.open("a", encoding="utf-8")) if writer else None

        def _heldout(tag: str) -> None:
            if not heldout or fh is None:
                return
            result = online.greedy_heldout(
                hf_model,
                tokenizer,
                heldout,
                max_new_tokens=heldout_max_new,
                reward=reward,
                device=device,
            )
            fh.write(json.dumps({"heldout": tag, **result}) + "\n")
            fh.flush()
            print(f"HELDOUT_{tag.upper()} {json.dumps(result)}", flush=True)

        # Refit before the PRE eval too: it proves the export path round-trips the
        # unchanged weights before any training depends on it.
        stats, refit_s = _refit()
        _heldout("pre")
        for step in range(args.steps):
            if step:
                stats, refit_s = _refit()
            t0 = time.perf_counter()
            rows: list[Any] | None = None
            if writer:
                indices = online.sample_prompt_indices(
                    len(samples), args.prompts_per_step, seed=args.seed, step=step
                )
                rows = online.generate_group_rows(
                    hf_model,
                    tokenizer,
                    samples,
                    indices,
                    group_size=group_size,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    reward=reward,
                    device=device,
                )
            all_rows = online.broadcast_rows(rows)
            gen_s = time.perf_counter() - t0
            if len(all_rows) != n_rows:
                raise RuntimeError(f"step {step}: {len(all_rows)} rollout rows, expected {n_rows}")
            # Shape only the advantages: logged reward stays the scorer's verdict.
            scored_rows = all_rows
            format_failures = None
            if args.unparsed_reward is not None:
                scored_rows, format_failures = online.format_failure_rows(
                    scored_rows, reward=args.unparsed_reward
                )
            overlong_penalty = None
            if args.overlong_cache > 0:
                scored_rows, overlong_penalty = online.overlong_shaped_rows(
                    scored_rows,
                    max_new_tokens=args.max_new_tokens,
                    cache_tokens=args.overlong_cache,
                    factor=args.overlong_factor,
                )
            advantages, sample_mask = online.scored_group_advantages(scored_rows)
            record: dict[str, Any] = {
                "step": step,
                # The scorer's view, before any reshaping: "unparsed" must stay comparable
                # between runs with and without --unparsed-reward. Trained rows are
                # scored + format_failures.
                **online.step_rollout_metrics(all_rows),
                "refit_s": round(refit_s, 3),
                "gen_s": round(gen_s, 3),
                "refit_written": stats["written"],
                "refit_unwritten": stats["unwritten"],
            }
            if overlong_penalty is not None:
                record["overlong_penalty_mean"] = round(overlong_penalty, 5)
            if format_failures is not None:
                record["format_failures"] = format_failures
            if not any(m > 0 for m in sample_mask):
                record["skipped"] = "no scored, non-empty row in the batch"
            else:
                lo, hi = online.shard_rows(all_rows, dp_size, dp_rank, group_size)
                batch = online.collate_token_batch(
                    all_rows[lo:hi], advantages[lo:hi], sample_mask[lo:hi], args.seq_len, pad_id
                )
                batch["old_logprobs"] = trainer.capture_logprobs(batch)[:, 1:].cpu()
                if needs_reference(objective):
                    batch["reference_logprobs"] = trainer.capture_reference_logprobs(batch)[
                        :, 1:
                    ].cpu()
                    record["ref_gap"] = reference_gap(
                        batch["old_logprobs"],
                        batch["reference_logprobs"],
                        batch["loss_mask"][:, 1:],
                    )
                declared = (float(n_rows), float(args.seq_len - 1)) if unit == "dr_grpo" else None
                den = compute_denominators(
                    unit,
                    batch["loss_mask"][:, 1:],
                    batch["sample_mask"],
                    group=dp_group,
                    declared=declared,
                )
                record.update(trainer.train_step(batch, objective_loss_fn, den))
            if fh is not None:
                fh.write(json.dumps(record) + "\n")
                fh.flush()
        _refit()
        _heldout("post")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    import torch

    args = build_arg_parser().parse_args(argv)
    if args.fp32:
        # The torch TF32 flags do not reach TE's grouped expert GEMM; only cuBLAS's own
        # override does, and cuBLAS reads it once, at handle creation. Measured on a
        # 6-layer Gemma-4 MoE: 0.55 nats max vs HF with the flags off, 1.0e-3 with this.
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    require_megatron("driver.main")
    from transformers import AutoTokenizer

    from foundationscale.rl.megatron.normalization import compute_denominators
    from foundationscale.rl.megatron.pp_step import loss_unit
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
    trainer = MegatronRLTrainer(
        cfg,
        args.hf_model,
        lr=args.lr,
        seed=args.seed,
        fp32=args.fp32,
        attention_backend=args.attention_backend,
    )
    trainer.build()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=False)
    pad_id = int(
        getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0
    )

    if args.online:
        return _run_online(args, trainer, tokenizer, pad_id)
    if not args.rollout_jsonl:
        raise SystemExit("--rollout-jsonl is required unless --online")
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
        if args.fp32:
            # An fp32 gate that silently runs TF32 matmuls measures TF32, not the
            # implementation: record what the stack had enabled, then force full fp32.
            print(
                f"parity fp32: allow_tf32 matmul={torch.backends.cuda.matmul.allow_tf32}"
                f" cudnn={torch.backends.cudnn.allow_tf32}"
                f" precision={torch.get_float32_matmul_precision()}",
                flush=True,
            )
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        logprobs = trainer.capture_logprobs(batch)
        import torch.distributed as dist

        # Every rank holds the gathered logprobs; concurrent writers would interleave.
        if dist.is_initialized() and dist.get_rank() != 0:
            return 0
        Path(args.out).write_text(
            json.dumps({"input_ids": batch["input_ids"].tolist(), "logprobs": logprobs.tolist()})
            + "\n",
            encoding="utf-8",
        )
        return 0

    if args.steps < 1:
        raise SystemExit(f"--steps must be >= 1, got {args.steps}")
    objective = getattr(lookup_algorithm(args.algorithm), "_objective", None)
    if objective is None:
        raise SystemExit(f"--algorithm {args.algorithm!r} carries no tensor objective")
    objective_loss_fn = TensorPolicyLoss(objective=objective)
    unit = loss_unit(objective_loss_fn)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    dp_group = trainer.pg.dp
    dp_size, dp_rank = dp_group.size(), dp_group.rank()
    if len(rows) % dp_size or (len(rows) // dp_size) % max(1, args.group_size):
        raise SystemExit(
            f"{len(rows)} rows do not split into {dp_size} DP shards of whole groups "
            f"(group_size={args.group_size})"
        )
    per_rank = len(rows) // dp_size
    import torch.distributed as dist

    # Metrics are already reduced over dp and pp, so one writer suffices.
    writer = not dist.is_initialized() or dist.get_rank() == 0
    if needs_reference(objective):
        stats = trainer.snapshot_reference()
        if writer:
            print(
                f"[meg] frozen reference: {stats['reference_tensors']} tensors, "
                f"{stats['reference_numel']} params held on host",
                flush=True,
            )
    with contextlib.ExitStack() as stack:
        fh = stack.enter_context(metrics_path.open("a", encoding="utf-8")) if writer else None
        for step in range(args.steps):
            group_ids = [i // max(1, args.group_size) for i in range(len(rows))]
            advantages = group_relative_advantages([r.reward for r in rows], group_ids)
            # Advantages see whole groups; each DP rank then trains its contiguous slice.
            lo, hi = dp_rank * per_rank, (dp_rank + 1) * per_rank
            batch = collate_mock_batch(
                rows[lo:hi], advantages[lo:hi], tokenizer, args.seq_len, pad_id
            )
            old_logprobs = trainer.capture_logprobs(batch)[:, 1:]
            batch["old_logprobs"] = old_logprobs.cpu()
            step_extra: dict[str, Any] = {}
            if needs_reference(objective):
                batch["reference_logprobs"] = trainer.capture_reference_logprobs(batch)[:, 1:].cpu()
                step_extra["ref_gap"] = reference_gap(
                    batch["old_logprobs"],
                    batch["reference_logprobs"],
                    batch["loss_mask"][:, 1:],
                )
            # dr_grpo divides by declared constants (B_g, L_cap), not a measured count.
            declared = (float(len(rows)), float(args.seq_len - 1)) if unit == "dr_grpo" else None
            den = compute_denominators(
                unit,
                batch["loss_mask"][:, 1:],
                batch["sample_mask"],
                group=dp_group,
                declared=declared,
            )
            metrics = trainer.train_step(batch, objective_loss_fn, den)
            if fh is not None:
                fh.write(json.dumps({"step": step, **step_extra, **metrics}) + "\n")
                fh.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
