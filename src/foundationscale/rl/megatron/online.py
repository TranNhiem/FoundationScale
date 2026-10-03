"""On-policy rollout for the Megatron RL lane: refit, generate, verify, collate (RUNG 2).

The driver's rungs 0-1 train on MOCK rollouts: prompt/completion/reward rows read
from a JSONL file written before the first step. The loop is structurally real --
old logprob capture, group-normalized advantages, forward_backward, distributed
optimizer -- but the data never moves with the policy, so an improvement is
invisible and a collapse cannot be observed. RUNG 2 closes that gap: each step the
CURRENT policy generates its own completions, a verifiable reward scores them, and
the rows reach the same ``collate`` carrier dict that ``MegatronRLTrainer.train_step``
already consumes.

Generation runs on an HF copy of the policy living ONLY on global rank 0. A
dedicated engine (vLLM/sglang) is out of scope for this rung: one HF ``generate``
is slow but auditable, and the only thing that must be exactly right is the refit.
Weights cross Megatron -> HF through Megatron-Bridge's
``bridge.export_hf_weights(model_list, cpu=False, show_progress=False)``, which is a
COLLECTIVE gather whose generator EVERY rank must drain, or the ranks desynchronize
inside the gather. Rank 0 copies each (hf_name, tensor) into ``hf_model.state_dict()``
with ``.copy_(...)`` under ``torch.no_grad()``; the other ranks consume and drop. A
yielded name the HF state dict does not hold is a refit LEAK -- the HF copy would
silently keep stale weights -- so it RAISES. HF entries the export never yields are
REPORTED in the returned counts rather than passed over in silence; with tied
embeddings the export has no ``lm_head.weight`` to give at all, exactly as the
checkpoint is designed, so that one case is excluded and nothing else is.

Rows are sampled and scored on rank 0 and broadcast once with
``torch.distributed.broadcast_object_list``, so every rank collates its own DP slice
from identical rows. Nothing here re-tokenizes: ids travel with the row, because ids
that round-trip twice through a tokenizer are the ids of a DIFFERENT text.

Abstention is first class. A scorer that cannot read a completion returns ``None``;
such a row is UNMEASURED, not wrong, and must not move the policy -- it is masked to
0 and kept out of the advantage statistics. A masked row is not a zero reward: a zero
reward drags its group's mean down and teaches the policy something false.

Every heavy import (torch, transformers) is lazy inside functions, so this module
imports on a CPU-only box and the pure-pipeline helpers test without a GPU, a Megatron
install, or a model.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from foundationscale.rl.corpus import Sample

__all__ = (
    "OnlineRow",
    "broadcast_rows",
    "collate_token_batch",
    "generate_group_rows",
    "greedy_heldout",
    "refit_hf_policy",
    "render_prompt_ids",
    "sample_prompt_indices",
    "scored_group_advantages",
    "overlong_shaped_rows",
    "shard_rows",
    "step_rollout_metrics",
)


@dataclass(frozen=True, slots=True)
class OnlineRow:
    """One on-policy rollout row: ids, verdict, and its group.

    ``prompt_ids`` and ``completion_ids`` are token ids AS GENERATED, never
    re-derived from text. ``reward`` is ``None`` when the scorer ABSTAINED --
    the row is unmeasured, not wrong, and is masked before it can touch the
    policy. ``group`` is the prompt's position in ``prompt_indices``; the
    ``group_size`` completions of one prompt share a value and are normalized
    against each other. ``finished`` records that an EOS was emitted before the
    budget ran out; a budget-truncated completion is the policy talking past
    its own stop token and is reported, not repaired.
    """

    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    reward: float | None
    group: int
    finished: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_ids", tuple(int(t) for t in self.prompt_ids))
        object.__setattr__(self, "completion_ids", tuple(int(t) for t in self.completion_ids))


def sample_prompt_indices(n_items: int, k: int, *, seed: int, step: int) -> list[int]:
    """Pick ``k`` distinct prompt indices for one rollout step.

    Deterministic in ``(seed, step)`` and WITHOUT replacement inside a step: a
    prompt drawn twice in one step would put two advantage estimates of the same
    question into one batch, which the group normalization reads as evidence the
    model is inconsistent with itself. A different step may re-draw the same
    prompt -- that is a fresh sample from the current policy and is exactly the
    point of rung 2.
    """
    if k < 1:
        raise ValueError(
            f"sample_prompt_indices: k {k} is below 1 of at least 1 required prompt(s); "
            f"a run with 0 prompts generates nothing to train on"
        )
    if k > n_items:
        raise ValueError(
            f"sample_prompt_indices: k {k} exceeds n_items {n_items} of at least {k} "
            f"required prompt(s); sampling without replacement cannot draw {k} "
            f"distinct prompts from {n_items}"
        )
    rng = random.Random(seed * 1_000_003 + step)
    return list(rng.sample(range(n_items), k))


def scored_group_advantages(
    rows: Sequence[OnlineRow], eps: float = 1e-4
) -> tuple[list[float], list[float]]:
    """Group-normalized advantages plus the mask that decides what trains.

    Returns ``(advantages, sample_mask)``, one pair per row. Statistics are
    computed per group over the SCORED members only (``reward is not None``):
    ``(r - mean) / (std + eps)`` with population std. A zero-variance group --
    including a single scored member -- carries no signal and gets advantage
    ``0.0``, but its scored rows keep ``sample_mask == 1.0``: measured-but-flat
    is honest information (the group agreed), unlike an abstention.

    ``sample_mask`` is ``0.0`` where nothing can be learned: ``reward is None``
    (the scorer abstained; it never checked the completion, so neither a zero
    nor a group mean may be invented for it) and completions that are EMPTY
    (context-only rows carry no token to give a gradient). Those rows' group
    statistics still include their reward when they have one -- the mask is
    what stops them moving the policy, and dropping their measured reward from
    the mean would bias the very group it belongs to.
    """
    advantages = [0.0] * len(rows)
    sample_mask = [0.0] * len(rows)
    groups: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        if row.reward is None:
            continue  # an abstention is UNMEASURED: excluded from stats, masked out
        groups.setdefault(int(row.group), []).append(index)
        sample_mask[index] = 1.0 if row.completion_ids else 0.0
    for members in groups.values():
        values: list[float] = []
        for index in members:
            reward = rows[index].reward
            if reward is not None:
                values.append(float(reward))
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)
        for index, value in zip(members, values, strict=True):
            # Zero variance (incl. a single scored member) is a flat group, not a
            # division-by-zero; eps=0 would make it one.
            advantages[index] = 0.0 if std == 0.0 else (value - mean) / (std + eps)
    return advantages, sample_mask


def overlong_shaped_rows(
    rows: Sequence[OnlineRow],
    *,
    max_new_tokens: int,
    cache_tokens: int,
    factor: float = 1.0,
) -> tuple[list[OnlineRow], float]:
    """Grade the over-long failure on a RAMP instead of a cliff (DAPO soft punishment).

    WHY: a completion that spends the whole ``max_new_tokens`` budget is the policy
    talking past its own stop token, and a single penalty dropped at the limit is a
    cliff the policy can only discover by falling off it. The soft shape grades the
    damage instead -- nothing at or below ``max_new_tokens - cache_tokens``, a LINEAR
    ramp across the ``cache_tokens`` window, and the flat ``-factor`` once the budget
    is spent -- so the signal starts before the wall and reaching the wall is never
    cheap.

    A row with ``finished is False`` pays the same flat ``-factor`` at EVERY length: a
    budget truncation is the same failure described by a token count instead of a
    length, so it is paid in full, not by the piece. Shaping lands only where a reward
    was MEASURED -- a scorer abstention stays ``None`` and is kept out of the returned
    mean, because an unmeasured row must not be handed an invented score nor borrow
    weight from rows that were actually scored.

    Every row comes back a NEW object (``dataclasses.replace``); the input is never
    touched. ``cache_tokens`` must be a real window: 0 would mean "never penalize",
    which is the CALLER's decision to skip this call, not a value this helper accepts.
    """

    if max_new_tokens < 1:
        raise ValueError(
            f"overlong_shaped_rows: max_new_tokens must be >= 1 (got {max_new_tokens})"
        )
    if cache_tokens < 1:
        raise ValueError(f"overlong_shaped_rows: cache_tokens must be >= 1 (got {cache_tokens})")
    if cache_tokens > max_new_tokens:
        raise ValueError(
            f"overlong_shaped_rows: cache_tokens ({cache_tokens}) must be <= "
            f"max_new_tokens ({max_new_tokens})"
        )
    if factor < 0:
        raise ValueError(f"overlong_shaped_rows: factor must be >= 0 (got {factor})")

    start = max_new_tokens - cache_tokens
    shaped: list[OnlineRow] = []
    penalty_sum = 0.0
    scored = 0
    for row in rows:
        reward = row.reward
        n_tokens = len(row.completion_ids)
        if not row.finished or n_tokens >= max_new_tokens:
            penalty = -factor
        elif n_tokens <= start:
            penalty = 0.0
        else:
            penalty = -(n_tokens - start) / cache_tokens * factor
        if reward is None:
            # Unmeasured stays unmeasured: no shaping here may invent a verdict.
            shaped.append(replace(row, reward=None))
            continue
        penalty_sum += penalty
        scored += 1
        shaped.append(replace(row, reward=reward + penalty))
    return shaped, penalty_sum / scored if scored else 0.0


def collate_token_batch(
    rows: Sequence[OnlineRow],
    advantages: Sequence[float],
    sample_mask: Sequence[float],
    seq_len: int,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Build the carrier dict pp_step consumes, straight from row ids.

    Same contract as the driver's ``collate_mock_batch``: ``input_ids [B,S]``
    long right-padded, ``loss_mask`` float and 1 over completion tokens only
    (prompt positions are context, not supervision, and the loss shifts it
    downstream), ``sample_mask`` and ``advantages`` float [B], and
    ``attention_mask`` float [B,S] with 1 on real tokens. NO tokenizer appears
    here on purpose: ``prompt_ids + completion_ids`` are the ids the policy
    emitted, and re-encoding them would score a different sequence than the one
    sampled.

    Rows run longer than ``seq_len`` are right-truncated; when truncation eats
    the COMPLETE completion, that row yields no completion token at all and its
    ``sample_mask`` is forced to 0.0 -- the row is unmeasured, not a zero. The
    attention mask is built from per-row LENGTHS, never from ``ids != pad_id``:
    ``pad_id`` is routinely the EOS id (the driver falls back to exactly that),
    and EOS is a real completion token, so a pad-scan would mask the stop token
    of every row that stops.
    """
    import torch

    if len(advantages) != len(rows) or len(sample_mask) != len(rows):
        raise ValueError(
            f"collate_token_batch: {len(rows)} rows but {len(advantages)} advantages "
            f"and {len(sample_mask)} sample masks; every row needs exactly one of each"
        )
    if seq_len < 1:
        raise ValueError(
            f"collate_token_batch: seq_len {seq_len} is below 1 of at least 1 required "
            f"token(s); a 0-length window cannot hold even one completion token"
        )
    id_rows: list[list[int]] = []
    loss_rows: list[list[float]] = []
    lengths: list[int] = []
    kept_mask: list[float] = []
    for row, mask_value in zip(rows, sample_mask, strict=True):
        prompt = tuple(row.prompt_ids)
        ids = (prompt + tuple(row.completion_ids))[:seq_len]
        prompt_kept = min(len(prompt), len(ids))
        completion_kept = len(ids) - prompt_kept
        id_rows.append(list(ids))
        loss_rows.append([0.0] * prompt_kept + [1.0] * completion_kept)
        lengths.append(len(ids))
        kept_mask.append(float(mask_value) if completion_kept else 0.0)
    width = max((len(row_ids) for row_ids in id_rows), default=1)
    for row_ids, loss in zip(id_rows, loss_rows, strict=True):
        pad = width - len(row_ids)
        row_ids.extend([int(pad_id)] * pad)
        loss.extend([0.0] * pad)
    return {
        "input_ids": torch.tensor(id_rows, dtype=torch.long),
        "loss_mask": torch.tensor(loss_rows, dtype=torch.float32),
        "sample_mask": torch.tensor(kept_mask, dtype=torch.float32),
        "advantages": torch.tensor([float(a) for a in advantages], dtype=torch.float32),
        "attention_mask": torch.tensor(
            [[1.0] * kept + [0.0] * (width - kept) for kept in lengths], dtype=torch.float32
        ),
    }


def shard_rows(
    rows: Sequence[OnlineRow], dp_size: int, dp_rank: int, group_size: int
) -> tuple[int, int]:
    """Contiguous ``(lo, hi)`` DP slice of the rollout, whole groups only.

    Advantages are computed over whole groups and then each rank trains its
    shard; a shard that cut a group would train half of a comparison, whose
    normalization was measured over the other half the shard never sees. The
    rows therefore must split into ``dp_size`` equal shards, each a whole
    number of ``group_size``-row groups.
    """
    n_rows = len(rows)
    if dp_size < 1:
        raise ValueError(
            f"shard_rows: dp_size {dp_size} is below 1 of at least 1 required rank(s); "
            f"{n_rows} rows on 0 ranks train nothing"
        )
    if group_size < 1:
        raise ValueError(
            f"shard_rows: group_size {group_size} is below 1 of at least 1 required "
            f"completion(s); a 0-sized group has no group-normalized advantage"
        )
    if not 0 <= dp_rank < dp_size:
        raise ValueError(
            f"shard_rows: dp_rank {dp_rank} is outside dp_size {dp_size} "
            f"((0..{dp_size - 1}) are the admissible ranks)"
        )
    if n_rows % dp_size:
        raise ValueError(
            f"shard_rows: {n_rows} rows do not split into {dp_size} equal shards "
            f"({n_rows} % {dp_size} = {n_rows % dp_size}); an unequal shard would let "
            f"one rank train on rows the DP average has no counterpart for"
        )
    per_rank = n_rows // dp_size
    if per_rank % group_size:
        raise ValueError(
            f"shard_rows: {n_rows} rows give {per_rank} rows per shard over dp_size "
            f"{dp_size}, and {per_rank} is not a whole multiple of group_size "
            f"{group_size} ({per_rank} % {group_size} = {per_rank % group_size}); "
            f"a shard would cut a group in half"
        )
    return dp_rank * per_rank, (dp_rank + 1) * per_rank


# Bridge exports MoE experts under the HF *checkpoint* names (one tensor per expert);
# transformers >= 5 fuses them in memory (its checkpoint conversion mapping:
# [gate_proj, up_proj] -> experts.gate_up_proj [E, 2I, H], down_proj -> experts.down_proj
# [E, H, I]; Mixtral spells them w1/w3/w2 under block_sparse_moe).
_EXPERT_NAME = re.compile(
    r"^(?P<prefix>.+?)\.(?:mlp|block_sparse_moe)\.experts\.(?P<e>\d+)\."
    r"(?P<part>gate_proj|up_proj|down_proj|w1|w3|w2)\.weight$"
)
_GATE, _UP, _DOWN = ("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")


def _fused_expert_slice(state: Mapping[str, Any], name: str) -> tuple[str, Any] | None:
    """Return ``(fused_name, view)`` for a per-expert export name, else ``None``."""
    match = _EXPERT_NAME.match(name)
    if match is None:
        return None
    prefix, expert, part = match["prefix"], int(match["e"]), match["part"]
    target = f"{prefix}.mlp.experts.{'down_proj' if part in _DOWN else 'gate_up_proj'}"
    fused = state.get(target)
    if fused is None or fused.dim() != 3 or expert >= fused.shape[0]:
        return None
    if part in _DOWN:
        return target, fused[expert]
    half = fused.shape[1] // 2
    return target, fused[expert, :half] if part in _GATE else fused[expert, half:]


def refit_hf_policy(
    bridge: Any,
    model_list: Any,
    hf_model: Any,
    *,
    is_writer: bool,
) -> dict[str, int]:
    """Gather Megatron weights into the generation-side HF copy (COLLECTIVE).

    ``bridge.export_hf_weights(model_list, cpu=False, show_progress=False)`` is
    a collective that yields ``(hf_name, tensor)`` with full TP/PP-gathered
    tensors, so EVERY rank drains the generator to exhaustion: a rank that
    returned early would leave its peers waiting inside the gather forever.
    Only the writer copies -- ``dest.copy_(tensor.to(dtype, device))`` under
    ``torch.no_grad()`` keeps the HF parameter's dtype/device and writes through
    the state dict's shared storage; everyone else consumes and drops.

    Returns ``{"written", "unwritten"}`` on the writer and zeros elsewhere.
    "written" counts DISTINCT names written (a name yielded twice is still one
    parameter). Two failures are kept apart DELIBERATELY:

    - a yielded name the HF state dict does not hold is a LEAK and RAISES -- the
      generated text would come from weights one version stale. The raise
      happens AFTER the generator is drained so no peer is left inside the
      gather while the writer unwinds;
    - an HF entry the export never yields is REPORTED as ``unwritten``, never
      silently ignored, so a stale embedding cannot go undiscussed. The one
      legitimate case is excluded: with ``config.tie_word_embeddings`` the HF
      checkpoint HAS no ``lm_head.weight`` -- it is the tied embedding -- and the
      export correctly has nothing to write for it.
    """
    import torch

    # Only the writer holds an HF copy; peers pass None and just drain the export.
    state = hf_model.state_dict() if is_writer else {}
    written: set[str] = set()
    missing: list[str] = []
    fused_hits: dict[str, set[str]] = {}
    with torch.no_grad():
        for hf_name, tensor in bridge.export_hf_weights(model_list, cpu=False, show_progress=False):
            name = str(hf_name)
            if not is_writer:
                continue  # drain-and-drop: the export is collective on every rank
            dest = state.get(name)
            if dest is None:
                fused = _fused_expert_slice(state, name)
                if fused is None:
                    missing.append(name)  # refused AFTER the drain; see docstring
                    continue
                target, dest = fused
                if tuple(dest.shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"refit_hf_policy: {name} has shape {tuple(tensor.shape)} but its "
                        f"slice of {target} has shape {tuple(dest.shape)}"
                    )
                dest.copy_(tensor.to(dtype=dest.dtype, device=dest.device))
                fused_hits.setdefault(target, set()).add(name)
                continue
            dest.copy_(tensor.to(dtype=dest.dtype, device=dest.device))
            written.add(name)
    # A fused expert tensor is written only when EVERY expert slice arrived: E slices
    # for down_proj, 2E for gate_up_proj (gate and up halves of each expert).
    for target, names in fused_hits.items():
        experts = int(state[target].shape[0])
        if len(names) == experts * (2 if target.endswith("gate_up_proj") else 1):
            written.add(target)
    if not is_writer:
        return {"written": 0, "unwritten": 0}
    if missing:
        raise ValueError(
            f"refit_hf_policy: {len(missing)} of {len(written) + len(missing)} yielded "
            f"weight name(s) -- {missing} -- are absent from the HF state dict of "
            f"{len(state)} entries; copying fewer names than were gathered would leave "
            f"stale weights in the generation copy"
        )
    tied = bool(getattr(getattr(hf_model, "config", None), "tie_word_embeddings", False))
    unwritten = [
        name for name in state if name not in written and not (tied and name == "lm_head.weight")
    ]
    if unwritten:
        # report, not raise: an unexported entry is exhaustively named below, and
        # silence is exactly the failure this count exists to prevent.
        print(
            f"[online] refit_hf_policy: {len(unwritten)} of {len(state)} HF state "
            f"dict entries were never yielded by export_hf_weights: {unwritten}",
            flush=True,
        )
    return {"written": len(written), "unwritten": len(unwritten)}


def render_prompt_ids(tokenizer: Any, prompt_turns: Sequence[tuple[str, str]]) -> list[int]:
    """Chat-render one prompt to token ids, once, via the model's own template.

    The template that ships with the checkpoint is the only authority on how its
    roles are framed; a hand-rolled role prefix would render text the policy was
    never trained on. Tokenizers answer in two shapes (a bare id list, or a
    BatchEncoding/dict that carries ``input_ids``), and both are accepted -- so
    this refuses neither the slow tokenizer nor the fast one.
    """
    messages = [{"role": role, "content": text} for role, text in prompt_turns]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if rendered and isinstance(rendered[0], (list, tuple)):
        rendered = rendered[0]
    return [int(token) for token in rendered]


def _cut_after_eos(ids: Sequence[int], eos_id: int, pad_id: int) -> tuple[list[int], bool]:
    """Cut one completion just after its first EOS (keeping it), else strip fill.

    The EOS is KEPT: it is the completion's own stop token and the loss sees it
    as a real prediction. A completion with no EOS burned its whole
    ``max_new_tokens`` budget -- ``finished=False``, reported as truncation --
    and only its trailing PAD is stripped, and only when pad != EOS. When pad IS
    EOS the generation-fill at the tail of a short row is indistinguishable
    from a real stop token, so it is kept rather than guessed away: a stripped
    EOS would delete the very token that says the row ended.
    """
    tokens = [int(token) for token in ids]
    for position, token in enumerate(tokens):
        if token == eos_id:
            return tokens[: position + 1], True
    if pad_id != eos_id:
        while tokens and tokens[-1] == pad_id:
            tokens.pop()
    return tokens, False


def generate_group_rows(
    hf_model: Any,
    tokenizer: Any,
    samples: Sequence[Sample],
    prompt_indices: Sequence[int],
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    reward: Any,
    device: Any,
) -> list[OnlineRow]:
    """Sample ``group_size`` completions per chosen prompt and score each one.

    One ``hf_model.generate`` per prompt (not one call for the batch: a group
    must come from the same decoding draw per prompt, and per-prompt calls keep
    the ids un-padded on the right so the completion is exactly ``output[:, P:]``).
    Sampling is ordinary HF multinomial (``do_sample=True``, ``top_k=0`` keeps
    the whole distribution; nothing here adds repetition tricks the old-logprob
    capture would then disagree with).

    The group index is the prompt's POSITION in ``prompt_indices`` -- rows of one
    group share a value and are normalized against each other downstream. Each
    row is cut after its first EOS (below), decoded WITHOUT special tokens, and
    handed to the verifiable scorer, which may abstain: ``reward=None`` travels
    with the row and is masked later, never repaired here.
    """
    import torch

    if group_size < 1:
        raise ValueError(
            f"generate_group_rows: group_size {group_size} is below 1 of at least 1 "
            f"required completion(s); a 0-sized group cannot be normalized"
        )
    if max_new_tokens < 1:
        raise ValueError(
            f"generate_group_rows: max_new_tokens {max_new_tokens} is below 1 of at "
            f"least 1 required token(s); 0 budget samples an empty completion"
        )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise ValueError(
            "generate_group_rows: tokenizer exposes no eos_token_id; without a stop "
            "token a completion cannot be told finished from budget-truncated"
        )
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = int(eos_id) if pad_id is None else int(pad_id)
    rows: list[OnlineRow] = []
    with torch.no_grad():
        hf_model.eval()  # generation must not see dropout: the logprobs it implies are targets
        for group, position in enumerate(prompt_indices):
            if not 0 <= int(position) < len(samples):
                raise ValueError(
                    f"generate_group_rows: prompt index {position} is outside the "
                    f"{len(samples)} sample(s) offered"
                )
            sample = samples[int(position)]
            prompt_ids = render_prompt_ids(tokenizer, sample.prompt_turns)
            input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            output = hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=0,
                num_return_sequences=group_size,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                eos_token_id=int(eos_id),
                use_cache=True,
            )
            for encoded in output[:, len(prompt_ids) :].tolist():
                completion_ids, finished = _cut_after_eos(encoded, int(eos_id), pad_id)
                response = tokenizer.decode(completion_ids, skip_special_tokens=True)
                rows.append(
                    OnlineRow(
                        prompt_ids=tuple(prompt_ids),
                        completion_ids=tuple(completion_ids),
                        reward=reward.score(response=response, gold=sample.gold),
                        group=group,
                        finished=finished,
                    )
                )
    return rows


def greedy_heldout(
    hf_model: Any,
    tokenizer: Any,
    samples: Sequence[Sample],
    *,
    max_new_tokens: int,
    reward: Any,
    device: Any,
    batch_size: int = 16,
) -> dict[str, float | int]:
    """Greedy decode every held-out sample; report verdict and truncation counts.

    Evaluation uses the DECODE behaviour being measured -- no sampling, no
    advantage arithmetic -- and left-padded batches, whose attention mask keeps
    the pad on the left out of every softmax that matters (``padding_side`` is
    set for the call and RESTORED afterwards: an evaluation that leaves the
    trainer's tokenizer on "left" changes training batches it will never own).

    ``unparsed`` counts rows the verifier could NOT read (its abstention);
    ``truncated`` counts rows with no EOS inside the budget. ``accuracy`` is over
    ALL rows (``correct / n``), the same denominator the HF lane's held-out hook
    reports, so the two lanes' numbers compare directly; ``parsed_accuracy``
    (``correct / (n - unparsed)``) is reported beside it. ``correct`` counts
    verdicts equal to the scorer's declared ``correct`` value (``MCQLetterReward``
    scores 1.0/0.0), not merely positive ones.
    """
    import torch

    if max_new_tokens < 1:
        raise ValueError(
            f"greedy_heldout: max_new_tokens {max_new_tokens} is below 1 of at least 1 "
            f"required token(s); 0 budget would score 0 completions"
        )
    if batch_size < 1:
        raise ValueError(
            f"greedy_heldout: batch_size {batch_size} is below 1 of at least 1 required "
            f"row(s); a 0-sized batch decodes nothing"
        )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise ValueError(
            "greedy_heldout: tokenizer exposes no eos_token_id; truncation cannot be "
            "measured without a stop token"
        )
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = int(eos_id) if pad_id is None else int(pad_id)
    prompts = [render_prompt_ids(tokenizer, sample.prompt_turns) for sample in samples]
    previous_side = getattr(tokenizer, "padding_side", "right")
    n_rows = 0
    correct = 0
    correct_value = float(getattr(reward, "correct", 1.0))
    unparsed = 0
    truncated = 0
    completion_tokens = 0
    try:
        tokenizer.padding_side = "left"
        with torch.no_grad():
            hf_model.eval()
            for start in range(0, len(prompts), batch_size):
                chunk = prompts[start : start + batch_size]
                width = max(len(prompt) for prompt in chunk)
                input_ids = torch.tensor(
                    [[pad_id] * (width - len(prompt)) + list(prompt) for prompt in chunk],
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.tensor(
                    [[0] * (width - len(prompt)) + [1] * len(prompt) for prompt in chunk],
                    dtype=torch.long,
                    device=device,
                )
                output = hf_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_id,
                    eos_token_id=int(eos_id),
                    use_cache=True,
                )
                for offset, encoded in enumerate(output[:, width:].tolist()):
                    completion_ids, finished = _cut_after_eos(encoded, int(eos_id), pad_id)
                    response = tokenizer.decode(completion_ids, skip_special_tokens=True)
                    verdict = reward.score(response=response, gold=samples[start + offset].gold)
                    n_rows += 1
                    completion_tokens += len(completion_ids)
                    if not finished:
                        truncated += 1
                    if verdict is None:
                        unparsed += 1
                    elif verdict == correct_value:
                        correct += 1
    finally:
        tokenizer.padding_side = previous_side
    parsed = n_rows - unparsed
    return {
        "n": n_rows,
        "correct": correct,
        "unparsed": unparsed,
        "truncated": truncated,
        "accuracy": (correct / n_rows) if n_rows else 0.0,
        "parsed_accuracy": (correct / parsed) if parsed else 0.0,
        "mean_completion_tokens": (completion_tokens / n_rows) if n_rows else 0.0,
    }


def broadcast_rows(rows: list[OnlineRow] | None, *, src: int = 0) -> list[OnlineRow]:
    """Ship rank-0's rollout rows to every rank (or pass them through locally).

    Rank 0 samples, generates and scores; every other rank must train on the SAME
    rows or the DP average compares policies that never saw one batch. The
    payload is plain Python (ids plus a float verdict) and is cheap next to one
    forward pass. Without an initialized ``torch.distributed`` there is exactly
    one process and the rows are returned unchanged -- and then ``rows`` cannot
    be ``None``, since nobody is left to supply them.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        if rows is None:
            raise ValueError(
                "broadcast_rows: rows is None but torch.distributed is not initialized; "
                "with 1 process there is no source rank to receive the rollout from"
            )
        return rows
    payload: list[Any] = [rows]
    dist.broadcast_object_list(payload, src=src)
    landed = payload[0]
    if landed is None:
        raise ValueError(
            f"broadcast_rows: rank {src} broadcast 0 of at least 1 required row(s) as "
            f"None; a step whose source generated nothing would average garbage"
        )
    return list(landed)


def step_rollout_metrics(rows: Sequence[OnlineRow]) -> dict[str, float]:
    """Per-step rollout report, named for what each number counts.

    ``reward_mean`` is over SCORED rows only and is nan-free: 0.0 with
    ``scored == 0`` rather than a mean of the empty set (``used < offered`` stays
    visible in ``unparsed``). ``finished_frac`` is the share that stopped on its
    own EOS; ``mean_completion_tokens`` is the kept completion length across ALL
    rows. ``flat_groups`` counts groups whose scored rewards are all EQUAL --
    including groups with fewer than 2 scored members, where a single verdict is
    a group that taught no comparison. Flatness is not sanity; it names where
    the advantage is zero.
    """
    n_rows = len(rows)
    scored_values = [row.reward for row in rows if row.reward is not None]
    scored = len(scored_values)
    values = [float(v) for v in scored_values]
    groups: dict[int, list[float]] = {}
    for row in rows:
        bucket = groups.setdefault(int(row.group), [])
        if row.reward is not None:
            bucket.append(float(row.reward))
    flat_groups = sum(1 for bucket in groups.values() if all(v == bucket[0] for v in bucket))
    return {
        "reward_mean": (sum(values) / scored) if scored else 0.0,
        "scored": float(scored),
        "unparsed": float(n_rows - scored),
        "finished_frac": (sum(1 for row in rows if row.finished) / n_rows) if n_rows else 0.0,
        "mean_completion_tokens": (
            sum(len(row.completion_ids) for row in rows) / n_rows if n_rows else 0.0
        ),
        "flat_groups": float(flat_groups),
    }
