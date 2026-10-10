"""The PPO step: a learned value head, GAE, and the clipped surrogate.

``_one_step`` prices group-centred rewards through ``advantage_fn``. The
registry's ``ppo`` binding (``PPOAlgorithm``) is the one policy-gradient
entry that loop cannot serve: its advantage is TEMPORAL -- generalised
advantage estimation against a learned per-token value -- so it needs a value
head, a second optimiser, and a value loss the group loop has no slot for.
This module owns that whole step, the same way ``online_pref_step`` owns the
pairwise one, including the FSDP collective choreography that keeps every
rank issuing the identical forward/backward sequence when the number of
scorable rows differs per rank.

WHAT IS CLAIMED: ``ppo_objective`` reads clip bounds, KL stance, value clip
and the declared component/metric names off the ``PPOAlgorithm`` instance, so
the step prices what the binding declared; ``build_value_head`` returns a
zero-initialised fp32 linear head (V == 0 before the first update);
``gae`` implements the textbook reverse recursion over masked tokens;
``ppo_step`` issues the SAME sequence of collectives on every rank
(agree_all, one row-count sum, agree_max, three whitening sums, the per-slice forwards and
backwards, one replicated-grad average, then a fixed list of metric sums), so a
rank with no scorable row participates with zero-weight dummy rows instead of
deadlocking the peers' FSDP collectives; a step where NO rank has a scorable
row is UNMEASURED and skipped after exactly one collective; the old
log-probs are read in a no_grad pass immediately before the graded pass over
the same weights, so at every step ratio == 1 up to kernel nondeterminism and
clip_fraction == 0 (the step-1 invariant, printed at step 0).

WHAT IS NOT CLAIMED: that one inner epoch is PPO's sample efficiency (it is
the first epoch of PPO, which is exactly the regime the ratio invariant
checks); that the value head shares the policy's gradient -- it reads
DETACHED final hidden states, so it is a linear critic over the policy's
features and the value loss never moves the policy; or that the constants
below are tuned.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from foundationscale.gates.objective_gates import LossComponent, MetricObservation
from foundationscale.rl import distributed as dist_mod
from foundationscale.rl.advantage import RewardStats
from foundationscale.rl.algorithm import StepReport
from foundationscale.rl.corpus import Sample
from foundationscale.rl.distributed import DistContext
from foundationscale.rl.interfaces import LossOutput

# Not declared by PPOAlgorithm, so they live here, named, rather than inline.
GAMMA = 1.0
GAE_LAMBDA = 0.95
VALUE_COEF = 0.1
KL_WEIGHT = 0.04
WHITEN_EPS = 1e-8

_TEXT_KEYS = {"input_ids", "attention_mask"}


@dataclass(frozen=True)
class PPOStepObjective:
    """What ``ppo_step`` prices, read off one ``PPOAlgorithm`` instance."""

    clip_bounds: tuple[float, float]
    kl_weight: float
    value_clip_epsilon: float | None
    components: tuple[str, ...]
    metrics: tuple[str, ...]
    value_coef: float = VALUE_COEF
    gamma: float = GAMMA
    gae_lambda: float = GAE_LAMBDA
    ratio_scope: str = "token"

    def declaration(self) -> Any:
        return SimpleNamespace(components=self.components, metrics=self.metrics)


def ppo_objective(algorithm: Any) -> PPOStepObjective | None:
    """The step objective for a ``PPOAlgorithm``; ``None`` for anything else."""
    from foundationscale.rl.ppo import PPOAlgorithm

    if not isinstance(algorithm, PPOAlgorithm):
        return None
    semantics = algorithm.semantics()
    requirements = algorithm.requirements()
    metrics = tuple(requirements.declared_metrics)
    value_clip = getattr(algorithm, "value_clip_epsilon", None)
    if ("value_clip_fraction" in metrics) != (value_clip is not None):
        # The binding declares the metric iff a value clip was configured, but
        # does not keep the width; guessing one would price an undeclared clip.
        raise ValueError(
            "ppo: the binding declares value_clip_fraction but exposes no "
            "value_clip_epsilon (or the reverse); refusing to guess the width"
        )
    if semantics.clip_bounds is None:
        raise ValueError("ppo: the binding declares no clip_bounds; the surrogate has no band")
    low, high = semantics.clip_bounds
    return PPOStepObjective(
        clip_bounds=(float(low), float(high)),
        kl_weight=KL_WEIGHT if semantics.kl_estimator == "k3" else 0.0,
        value_clip_epsilon=None if value_clip is None else float(value_clip),
        components=tuple(requirements.declared_components),
        metrics=metrics,
    )


def is_ppo(objective: Any) -> bool:
    """True iff ``objective`` is the step objective ``ppo_objective`` built."""
    return isinstance(objective, PPOStepObjective)


def _hidden_size(model: Any) -> int:
    config = getattr(getattr(model, "module", model), "config", None)
    config = getattr(config, "text_config", config)
    size = getattr(config, "hidden_size", None)
    if not isinstance(size, int) or size < 1:
        raise ValueError(
            f"value head: model config exposes hidden_size={size!r}; a linear "
            "critic needs the width of the final hidden state"
        )
    return size


def build_value_head(model: Any, device: Any) -> Any:
    """A zero-initialised fp32 ``Linear(hidden_size, 1)``: V == 0 at step 0."""
    import torch

    head = torch.nn.Linear(_hidden_size(model), 1, dtype=torch.float32, device=device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    return head


def gae(rewards: Any, values: Any, mask: Any, *, gamma: float, lam: float) -> Any:
    """Generalised advantage estimates over masked token positions.

    ``delta_t = r_t + gamma * V_{t+1} * m_{t+1} - V_t`` and
    ``A_t = delta_t + gamma * lam * m_{t+1} * A_{t+1}``, zeroed off-mask. The
    value past the last masked token is 0 (a terminal state).
    """
    import torch

    width = rewards.shape[-1]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[..., 0])
    for t in range(width - 1, -1, -1):
        if t + 1 < width:
            next_mask = mask[..., t + 1]
            next_value = values[..., t + 1] * next_mask
        else:
            next_mask = torch.zeros_like(running)
            next_value = torch.zeros_like(running)
        delta = rewards[..., t] + gamma * next_value - values[..., t]
        running = (delta + gamma * lam * next_mask * running) * mask[..., t]
        advantages[..., t] = running
    return advantages


def ppo_step(
    trainer: Any,
    *,
    step: int,
    chunk: list[Sample],
    model: Any,
    tokenizer: Any,
    surface: Any,
    reward: Any,
    objective: PPOStepObjective,
    optimizer: Any,
    ref_model: Any,
    device: str,
    ctx: DistContext,
    value_head: Any,
    value_optimizer: Any,
) -> StepReport | None:
    """One PPO step, or ``None`` when no rank has a scorable completion."""
    import torch

    from foundationscale.rl import trainer as trainer_mod

    cfg = trainer.config
    group_size = cfg.group_size
    golds: list[str | None] = [sample.gold for sample in chunk]

    prompt_ids = trainer_mod.encode_prompts(surface, chunk, device)
    with torch.no_grad(), trainer_mod._generation_mode(model) as gen_model:
        generated = gen_model.generate(
            **prompt_ids,
            max_new_tokens=cfg.max_new_tokens,
            num_return_sequences=group_size,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            pad_token_id=tokenizer.pad_token_id,
            **dist_mod.generate_kwargs_for(ctx, cfg.sharding),
        )
    prompt_width = prompt_ids["input_ids"].shape[1]
    completions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)

    # An abstention (None) drops the row; coercing it to 0.0 would teach the
    # critic a return the verifier never stated.
    scorable: list[tuple[int, float]] = []
    for index, completion in enumerate(completions):
        scored = reward.score(response=completion, gold=golds[index // group_size])
        if scored is not None:
            scorable.append((index, scored))
    local_rows = len(scorable)

    if dist_mod.agree_all(local_rows == 0, ctx):
        print(
            f"UNMEASURED step {step}: 0 of {len(completions)} completion(s) were "
            "scorable on any rank; no return exists to estimate an advantage "
            "from, so no step is claimed.",
            file=sys.stderr,
        )
        return None
    if local_rows == 0:
        print(
            f"[trainer] step {step} rank {ctx.rank}: no scorable completion on "
            "this rank; others have some -- participating with zero loss "
            "(UNMEASURED here)",
            file=sys.stderr,
        )

    modality_kwargs = {
        key: value.repeat_interleave(group_size, dim=0)
        for key, value in prompt_ids.items()
        if key not in _TEXT_KEYS and hasattr(value, "index_select")
    }

    global_rows = dist_mod.all_reduce_sum(float(local_rows), ctx)
    micro_batch = int(cfg.logprob_micro_batch)
    local_slice_count = max(1, -(-local_rows // micro_batch)) if micro_batch > 0 else 1
    slice_count = dist_mod.agree_max(local_slice_count, ctx)

    def slice_rows(slice_at: int) -> list[tuple[int, float]]:
        if micro_batch > 0:
            start = slice_at * micro_batch
            end = min(start + micro_batch, local_rows)
        else:
            start, end = 0, local_rows
        return scorable[start:end] if start < end else []

    def forward_rows(scorer: Any, rows_idx: Any, *, with_values: bool) -> tuple[Any, Any, Any]:
        """(token log-probs, completion mask, values or None) for the rows."""
        sequences = generated.index_select(0, rows_idx)
        attention = (sequences != tokenizer.pad_token_id).long()
        target_ids = sequences[:, 1:]
        mask = torch.zeros_like(target_ids, dtype=torch.float32)
        mask[:, prompt_width - 1 :] = attention[:, 1:][:, prompt_width - 1 :]
        sliced = {key: value.index_select(0, rows_idx) for key, value in modality_kwargs.items()}
        out = scorer(
            input_ids=sequences,
            attention_mask=attention,
            output_hidden_states=with_values,
            **sliced,
        )
        logprobs = trainer_mod._token_logprobs(out.logits, target_ids)
        values = None
        if with_values:
            # Position t's state predicts token t+1, the same shift as the logits.
            hidden = out.hidden_states[-1][:, :-1].detach().to(torch.float32)
            values = value_head(hidden).squeeze(-1)
        return logprobs, mask, values

    # (a) No-grad pass: old log-probs, old values, reference log-probs, and
    # the advantages. Whitening needs every slice first, so this pass runs to
    # the end before any graded forward.
    plans: list[dict[str, Any]] = []
    adv_count = adv_sum = adv_sq = 0.0
    for slice_at in range(slice_count):
        rows = slice_rows(slice_at)
        real = bool(rows)
        row_ids = [row for row, _ in rows] if real else [0]
        rows_idx = torch.tensor(row_ids, device=device)
        with torch.no_grad():
            old_logprobs, mask, old_values = forward_rows(model, rows_idx, with_values=True)
            ref_logprobs = None
            if ref_model is not None:
                ref_logprobs, _, _ = forward_rows(ref_model, rows_idx, with_values=False)
        token_rewards = torch.zeros_like(mask)
        if real:
            # The outcome reward lands on each row's last completion token.
            positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
            last = (positions * (mask > 0)).amax(dim=1)
            outcome = torch.tensor(
                [value for _, value in rows], dtype=mask.dtype, device=mask.device
            )
            token_rewards[torch.arange(len(rows), device=mask.device), last] = outcome
        else:
            mask = mask * 0.0
        advantages = gae(
            token_rewards, old_values, mask, gamma=objective.gamma, lam=objective.gae_lambda
        )
        returns = advantages + old_values
        adv_count += float(mask.sum())
        adv_sum += float((advantages * mask).sum())
        adv_sq += float((advantages * advantages * mask).sum())
        plans.append(
            {
                "rows_idx": rows_idx,
                "real": real,
                "mask": mask,
                "old_logprobs": old_logprobs,
                "old_values": old_values,
                "ref_logprobs": ref_logprobs,
                "advantages": advantages,
                "returns": returns,
            }
        )

    # (b) Global whitening: three sums, unconditional, identical on every rank.
    global_tokens = dist_mod.all_reduce_sum(adv_count, ctx)
    global_sum = dist_mod.all_reduce_sum(adv_sum, ctx)
    global_sq = dist_mod.all_reduce_sum(adv_sq, ctx)
    denominator = max(global_tokens, 1.0)
    adv_mean = global_sum / denominator
    adv_std = max(global_sq / denominator - adv_mean * adv_mean, 0.0) ** 0.5

    # (c) Graded pass: one policy forward + one backward per agreed slice.
    low, high = objective.clip_bounds
    sums = {"policy": 0.0, "value": 0.0, "kl": 0.0, "clipped": 0.0, "ratio": 0.0, "vclip": 0.0}
    for plan in plans:
        mask = plan["mask"]
        logprobs, _, values = forward_rows(model, plan["rows_idx"], with_values=True)
        advantages = (plan["advantages"] - adv_mean) / (adv_std + WHITEN_EPS)
        ratio = torch.exp(logprobs - plan["old_logprobs"])
        surrogate = torch.minimum(ratio * advantages, ratio.clamp(low, high) * advantages)
        policy_tokens = -surrogate * mask
        value_error = (values - plan["returns"]) ** 2
        vclipped = torch.zeros_like(mask)
        if objective.value_clip_epsilon is not None:
            eps = objective.value_clip_epsilon
            old_values = plan["old_values"]
            clipped_values = old_values + (values - old_values).clamp(-eps, eps)
            clipped_error = (clipped_values - plan["returns"]) ** 2
            vclipped = (clipped_error > value_error).to(mask.dtype)
            value_error = torch.maximum(value_error, clipped_error)
        value_tokens = 0.5 * value_error * mask
        kl_tokens = torch.zeros_like(mask)
        if plan["ref_logprobs"] is not None and objective.kl_weight != 0.0:
            log_ref_ratio = (plan["ref_logprobs"] - logprobs) * mask
            kl_tokens = (torch.expm1(log_ref_ratio) - log_ref_ratio).clamp(min=0.0) * mask
        slice_loss = (
            policy_tokens.sum()
            + objective.value_coef * value_tokens.sum()
            + objective.kl_weight * kl_tokens.sum()
        ) / denominator
        if not plan["real"]:
            slice_loss = slice_loss * 0.0
        else:
            outside = ((ratio < low) | (ratio > high)).to(mask.dtype)
            sums["policy"] += float(policy_tokens.detach().sum())
            sums["value"] += float(value_tokens.detach().sum())
            sums["kl"] += float(kl_tokens.detach().sum())
            sums["clipped"] += float((outside * mask).sum())
            sums["ratio"] += float((ratio.detach() * mask).sum())
            sums["vclip"] += float((vclipped * mask).sum())
        slice_loss.backward()

    # (d) The value head is replicated, not sharded: average its grads the
    # way FSDP/DDP average the policy's, then step both.
    dist_mod.all_reduce_grads_mean(value_head, ctx)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    value_optimizer.step()
    value_optimizer.zero_grad(set_to_none=True)

    reduced = {
        key: dist_mod.all_reduce_sum(value, ctx) / denominator for key, value in sums.items()
    }
    policy_loss = reduced["policy"]
    value_contribution = objective.value_coef * reduced["value"]
    kl_contribution = objective.kl_weight * reduced["kl"]
    total = policy_loss + value_contribution + kl_contribution
    if step == 0:
        print(
            f"[trainer] step 0 PPO invariant: ratio_mean={reduced['ratio']:.6f} "
            f"(expect 1) clip_fraction={reduced['clipped']:.6f} (expect 0) "
            f"value_loss={reduced['value']:.6f}",
            file=sys.stderr,
        )

    contributions = {
        "ppo_policy_loss": (1.0, policy_loss),
        "value_loss": (objective.value_coef, value_contribution),
        "kl_penalty": (objective.kl_weight, kl_contribution),
    }
    components: list[Any] = []
    for name in objective.components:
        if name in contributions:
            weight, contribution = contributions[name]
            components.append(
                LossComponent(name=name, weight=weight, observed=True, contribution=contribution)
            )
        else:
            components.append(
                LossComponent(name=name, weight=0.0, observed=False, contribution=None)
            )
    observed = {
        "clip_fraction": reduced["clipped"],
        "kl_estimate": reduced["kl"],
        "value_clip_fraction": reduced["vclip"],
    }
    metrics = [MetricObservation(name="ratio_mean", value=reduced["ratio"])]
    metrics.extend(
        MetricObservation(name=name, value=observed[name])
        for name in objective.metrics
        if name in observed
    )
    scored_values = tuple(value for _, value in scorable)
    return StepReport(
        step=step,
        loss=LossOutput(loss=total, components=tuple(components), metrics=tuple(metrics)),
        rows=local_rows if local_rows else int(global_rows),
        reward_stats=RewardStats.over(scored_values) if scored_values else None,
        sync=None,
    )
