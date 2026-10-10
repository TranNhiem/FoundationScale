"""The online-preference step: online DPO and iterative DPO over mined pairs.

``_one_step`` prices group-relative objectives (a reward centred against its
own group via ``advantage_fn``). The online-preference family is the one
registry branch that loop cannot serve: ``OnlineDPOLoss`` and
``IterativeDPOLoss`` declare no ``advantage_fn`` because their unit of account
is not a centred reward but a chosen/rejected PAIR mined from the group's own
scored completions. This module owns that whole step -- rollout, pairing,
pairwise pricing against a frozen reference, and the FSDP collective
choreography that keeps every rank issuing the identical forward/backward
sequence when the pair count differs per rank. It also owns the reference
policy's refresh cadence, which only iterative DPO has.

WHAT IS CLAIMED: is_online_pref recognises exactly the two registry
objectives this step can price; refresh_cadence maps them to "never" (online,
frozen reference) and "every N steps" (iterative); maybe_refresh_reference
copies policy shards into reference shards elementwise (valid under FSDP2
because both models are wrapped identically, so local shards align and no
collective is needed) and refuses loudly on any structural mismatch rather
than training against a half-copied reference; online_pref_step issues the
SAME sequence of collectives on every rank (agree_all, agree_max,
all_reduce_sum, then one policy forward + one reference forward + one
backward per agreed slice), so a rank that mined zero pairs participates
with zero-weight dummy pairs instead of deadlocking the peers' FSDP
collectives; a step where NO rank mines a pair is UNMEASURED and skipped
after exactly one collective, never reported as a zero-row step; at step 0,
when policy == reference, the priced loss is exactly -log sigmoid(0) = ln 2.

WHAT IS NOT CLAIMED: that any particular step mines a pair (ties and
abstentions produce none, and that is the normal state, not an error); that
the reference refresh is bit-exact across FSDP resharding events (both
models are assumed to stay wrapped identically for the whole run); or that
beta is tuned -- it is read from the objective, never hard-coded here.
"""

from __future__ import annotations

import sys
from typing import Any

from foundationscale.gates.objective_gates import LossComponent, MetricObservation
from foundationscale.rl import distributed as dist_mod
from foundationscale.rl.advantage import RewardStats
from foundationscale.rl.algorithm import StepReport
from foundationscale.rl.corpus import Sample
from foundationscale.rl.distributed import DistContext
from foundationscale.rl.interfaces import LossOutput
from foundationscale.rl.online_objectives import IterativeDPOLoss, OnlineDPOLoss

# Mirrors _one_step's modality split: everything the surface emitted beyond
# the text pair is conditioning (pixel_values et al.) and must be forwarded
# to BOTH scorers per row, or the policy/reference log-probs would price an
# image-free conditional the rollout never sampled.
_TEXT_KEYS = {"input_ids", "attention_mask"}


def is_online_pref(objective: Any) -> bool:
    """True iff ``objective`` is one of the two online-preference losses.

    Membership is by TYPE, not by capability: the registry binds exactly
    these two classes, and duck-typing (e.g. any objective with ``beta``)
    would silently adopt offline preference losses into a loop that mines
    pairs from rewards they never asked for.
    """
    return isinstance(objective, (OnlineDPOLoss, IterativeDPOLoss))


def refresh_cadence(objective: Any, configured: int) -> int:
    """Steps between policy->reference refreshes; 0 means never refresh.

    Online DPO prices every pair against the FROZEN initial policy, so its
    cadence is 0 whatever the operator configured; only iterative DPO moves
    its reference, on the operator's cadence. A negative cadence is never
    meaningful -- for either objective -- and is refused rather than clamped,
    because a clamped cadence would silently refresh more or less often than
    the run card says.
    """
    if configured < 0:
        raise ValueError(
            f"configured={configured}: a negative reference-refresh cadence is not "
            "meaningful; use 0 to keep the reference frozen"
        )
    if isinstance(objective, IterativeDPOLoss):
        return configured
    return 0


def maybe_refresh_reference(*, step: int, cadence: int, model: Any, ref_model: Any) -> bool:
    """Copy policy weights into the reference when ``(step + 1)`` hits the cadence.

    The copy is elementwise over the two parameter lists under no_grad. No
    collective is issued: under FSDP2 both models are wrapped identically, so
    each rank's local shard of parameter i lines up with the peer's, and the
    copy is local .data movement. Off-cadence steps touch nothing and return
    False, so callers can invoke this unconditionally after every step.
    """
    if cadence <= 0 or (step + 1) % cadence != 0:
        return False
    import torch

    ref_params = list(ref_model.parameters())
    pol_params = list(model.parameters())
    # Validate the WHOLE correspondence before copying anything: a reference
    # refreshed over only the matching prefix is a subtly wrong reference for
    # every later step, and that is worse than a loud failure here.
    if len(ref_params) != len(pol_params):
        raise RuntimeError(
            f"reference refresh at step {step}: parameter-count mismatch -- the "
            f"reference exposes {len(ref_params)} tensor(s), the policy "
            f"{len(pol_params)}; the copy cannot be made correspond, so no "
            "parameter was touched"
        )
    for index, (ref, pol) in enumerate(zip(ref_params, pol_params, strict=True)):
        if tuple(ref.shape) != tuple(pol.shape):
            raise RuntimeError(
                f"reference refresh at step {step}: shape mismatch at parameter "
                f"index {index}: reference {tuple(ref.shape)} vs policy "
                f"{tuple(pol.shape)}; the copy cannot be made correspond, so no "
                "parameter was touched"
            )
    with torch.no_grad():
        for ref, pol in zip(ref_params, pol_params, strict=True):
            ref.data.copy_(pol.data)
    return True


def online_pref_step(
    trainer: Any,
    *,
    step: int,
    chunk: list[Sample],
    model: Any,
    tokenizer: Any,
    surface: Any,
    reward: Any,
    objective: Any,
    optimizer: Any,
    ref_model: Any,
    device: str,
    ctx: DistContext,
) -> StepReport | None:
    """One online-preference step, or ``None`` when no rank mined a pair.

    WHAT IS CLAIMED: identical collective sequence on every rank; per-pair
    loss sums are normalised by the GLOBAL pair count so the report and the
    gradient describe the same priced pairs; a pair-less rank takes the step
    with exactly-zero loss rather than skipping.

    WHAT IS NOT CLAIMED: that equal-reward prompts teach anything -- they
    yield no pair by construction.
    """
    # Lazy: trainer.py imports THIS module at its top, so a module-level
    # import back would cycle. Tests monkeypatch trainer_mod.encode_prompts,
    # so the lookup happens here, at call time -- never at import time.
    import torch
    import torch.nn.functional as F

    from foundationscale.rl import trainer as trainer_mod

    cfg = trainer.config
    group_size = cfg.group_size
    # beta and every other priced hyper-parameter come from the objective;
    # the step owns choreography, not constants.
    beta = float(objective.beta)
    golds: list[str | None] = [sample.gold for sample in chunk]

    # Rollout, verbatim from _one_step: encode through the surface (so a
    # multimodal chunk is scored conditioned on the same pixels it sampled
    # from), then sample group_size completions per prompt.
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

    # Score row by row; an abstention (None) is excluded from ranking, not
    # coerced to 0.0 -- a zero coerced reward could fabricate a preference.
    scorable: list[tuple[int, float]] = []
    for index, completion in enumerate(completions):
        # generate(num_return_sequences=G) returns rows prompt-major, so the
        # gold of row i is golds[i // G].
        scored = reward.score(response=completion, gold=golds[index // group_size])
        if scored is not None:
            scorable.append((index, scored))
    scored_values = [score for _, score in scorable]

    # Pairing per prompt: chosen = argmax reward, rejected = argmin, FIRST
    # index winning ties (strict comparisons only). A prompt whose scorable
    # rewards are all equal -- or that has fewer than two scorable rows --
    # states no preference, so it contributes no pair.
    by_prompt: dict[int, list[tuple[int, float]]] = {}
    for index, score in scorable:
        by_prompt.setdefault(index // group_size, []).append((index, score))
    pairs: list[tuple[int, int]] = []
    for members in by_prompt.values():
        if len(members) < 2:
            continue
        chosen_at = rejected_at = 0
        for at in range(1, len(members)):
            if members[at][1] > members[chosen_at][1]:
                chosen_at = at
            if members[at][1] < members[rejected_at][1]:
                rejected_at = at
        if members[chosen_at][1] == members[rejected_at][1]:
            continue
        pairs.append((members[chosen_at][0], members[rejected_at][0]))

    local_pairs = len(pairs)
    # (a) The null-step vote is rank-local first, agreed second: a rank must
    # never skip alone. Exactly ONE collective fires on the skip path.
    if dist_mod.agree_all(local_pairs == 0, ctx):
        print(
            f"UNMEASURED step {step}: 0 of {len(chunk)} prompt(s) yielded a "
            "chosen/rejected pair on any rank; every group's scorable rewards "
            "tied or abstained, so no preference exists to price. No gradient "
            "exists to take, so no step is claimed.",
            file=sys.stderr,
        )
        return None
    if local_pairs == 0:
        # A peer has pairs: this rank must still walk the identical slice
        # loop with zero-weight dummies, or the busiest rank deadlocks in
        # FSDP all-gathers. Same structure as _one_step's null rank.
        print(
            f"[trainer] step {step} rank {ctx.rank}: no prompt on this rank "
            "yielded a chosen/rejected pair; others did -- participating with "
            "zero loss (UNMEASURED here)",
            file=sys.stderr,
        )

    # Modality conditioning, exactly as _one_step: per-prompt processor
    # output repeated to per-completion rows. Row selection happens per
    # slice against these expanded tensors, in the same index order as the
    # generated ids.
    modality_kwargs = {
        key: value
        for key, value in prompt_ids.items()
        if key not in _TEXT_KEYS and hasattr(value, "index_select")
    }
    if modality_kwargs:
        modality_kwargs = {
            key: value.repeat_interleave(group_size, dim=0)
            for key, value in modality_kwargs.items()
        }
        print(
            "[trainer] forwarding modality keys to the scorer: "
            + ", ".join(sorted(modality_kwargs)),
            file=sys.stderr,
        )

    # (b) The slice COUNT must agree across ranks: forwards and backwards
    # fire per slice, so ranks with fewer real pairs pad with dummies. A
    # pair-less rank advertises one slice so agree_max has a well-formed vote.
    micro_batch = int(cfg.logprob_micro_batch)
    local_slice_count = max(1, -(-local_pairs // micro_batch)) if micro_batch > 0 else 1
    slice_count = dist_mod.agree_max(local_slice_count, ctx)
    # (before the slices) Global pair count, reduced ONCE: every slice's
    # loss sums are denominated by it, so the per-slice backwards add up to
    # the gradient of the global mean pairwise loss.
    global_pairs = dist_mod.all_reduce_sum(float(local_pairs), ctx)

    def summed_logprobs(scorer: Any, rows_idx: Any) -> Any:
        """Completion-token SUM log-probs for the selected generated rows.

        Chosen and rejected rows share ONE forward so both sides of a pair
        see identical conditioning. The supervision mask is _one_step's:
        full-width attention for the model, a shifted completion-only mask
        for the targets, one column narrower.
        """
        sequences = generated.index_select(0, rows_idx)
        attention = (sequences != tokenizer.pad_token_id).long()
        target_ids = sequences[:, 1:]
        response_mask = torch.zeros_like(target_ids, dtype=torch.float32)
        response_mask[:, prompt_width - 1 :] = attention[:, 1:][:, prompt_width - 1 :]
        sliced_modalities = {
            key: value.index_select(0, rows_idx) for key, value in modality_kwargs.items()
        }
        logits = scorer(
            input_ids=sequences,
            attention_mask=attention,
            **sliced_modalities,
        ).logits
        token_logprobs = trainer_mod._token_logprobs(logits, target_ids)
        return (token_logprobs * response_mask).sum(dim=-1)

    def real_pairs_in(slice_at: int) -> list[tuple[int, int]]:
        if micro_batch > 0:
            start = slice_at * micro_batch
            end = min(start + micro_batch, local_pairs)
        else:
            start, end = 0, local_pairs
        return pairs[start:end] if start < end else []

    # (c) The slice loop. Detached per-pair sums are accumulated alongside
    # the graph, so the metrics below are measured off the exact tensors the
    # backward priced.
    local_loss_sum = 0.0
    local_margin_sum = 0.0
    local_correct_sum = 0.0
    for slice_at in range(slice_count):
        slice_pairs = real_pairs_in(slice_at)
        if slice_pairs:  # noqa: SIM108 -- the dummy arm carries its rationale
            row_ids = [row for pair in slice_pairs for row in pair]
        else:
            # Dummy: any valid rows do -- the first completion as both the
            # chosen and the rejected side. The 0.0 factor below keeps the
            # graph (and its collectives) alive while zeroing the gradient.
            row_ids = [0, 0]
        rows_idx = torch.tensor(row_ids, device=device)
        n_rows = len(row_ids) // 2
        policy_sums = summed_logprobs(model, rows_idx)
        with torch.no_grad():
            reference_sums = summed_logprobs(ref_model, rows_idx)
        # pi_c/ref_c are the first half of the batch, pi_r/ref_r the second.
        logratio_gap = (policy_sums[:n_rows] - reference_sums[:n_rows]) - (
            policy_sums[n_rows:] - reference_sums[n_rows:]
        )
        # -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))): the same
        # pairwise DPO kernel the offline plane prices. Inline here because
        # the ONLINE reduction denominates by the global pair count, which no
        # batch-local helper can know.
        pair_losses = -F.logsigmoid(beta * logratio_gap)
        slice_loss = pair_losses.sum() / global_pairs
        if not slice_pairs:
            slice_loss = slice_loss * 0.0
        else:
            detached = logratio_gap.detach()
            local_loss_sum += float(pair_losses.detach().sum())
            local_margin_sum += float(detached.sum())
            local_correct_sum += float((detached > 0).to(detached.dtype).sum())
        slice_loss.backward()

    # (d) One optimiser step for the whole slice stack, as _one_step does;
    # the trailing zero_grad(set_to_none=True) leaves no stale graph state
    # for the next step.
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    declaration = objective.declaration()
    measured_loss = dist_mod.all_reduce_sum(local_loss_sum, ctx) / global_pairs
    # The objective's declaration names the one component and its accuracy /
    # margin metrics; report under exactly those names so the objective gates
    # compare like with like.
    margin = dist_mod.all_reduce_sum(local_margin_sum, ctx) / global_pairs
    accuracy = dist_mod.all_reduce_sum(local_correct_sum, ctx) / global_pairs
    loss_output = LossOutput(
        loss=measured_loss,
        components=tuple(
            LossComponent(name=name, weight=1.0, observed=True, contribution=measured_loss)
            for name in declaration.components
        ),
        metrics=(
            MetricObservation(name=objective.accuracy_metric_name, value=accuracy),
            MetricObservation(name=objective.margin_metric_name, value=margin),
        ),
    )
    return StepReport(
        step=step,
        loss=loss_output,
        # rows stays LOCAL so it matches the local reward_stats count; a null
        # rank has no local rows and zero is refused, so it names the global pairs.
        rows=len(scored_values) if scored_values else int(global_pairs),
        reward_stats=RewardStats.over(tuple(scored_values)) if scored_values else None,
        sync=None,
    )
