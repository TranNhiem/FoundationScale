"""The concrete RL training loop over the tensor plane.

This module closes the loop the contract stack opened: it samples prompts,
generates groups of completions, scores them with a verifiable reward,
prices the group-relative objective through the differentiable tensor kernel,
and takes one optimiser step per ``StepReport``. The shipped pure-python
objective arithmetic remains the oracle and is untouched; the loss actually
backpropagated is ``TensorPolicyLoss``'s evaluation of the SAME declared
axes read off the SAME objective instance.

torch and transformers are imported lazily INSIDE ``run()`` -- the same
idiom ``train/loop.py`` uses -- so importing this module on a torch-free
host always succeeds. A host without either dependency is refused
(exit-contract 96) with the missing dependency NAMED, never an unraised
ImportError and never a silent fall-back.

Model loading is family-aware only for media-declared corpora: a text-only
corpus loads exactly as before (``AutoModelForCausalLM``, falling back to
``AutoModelForImageTextToText`` only on an exception -- byte-identical to
every run before this paragraph was true). A corpus whose samples carry
images picks the auto class with ``train/loop.py``'s own
``_media_capable_auto_class``/``_model_can_consume_pixels`` (imported, never
duplicated) instead of the try/except fallback: MEASURED on transformers
5.18.0, ``AutoModelForCausalLM.from_pretrained`` on a qwen3_5/qwen3_5_moe
checkpoint SUCCEEDS with a text-only class
(``Qwen3_5ForCausalLM``/``Qwen3_5MoeForCausalLM``) that skips
``model.visual`` on load and absorbs ``pixel_values``/``image_grid_thw``
through a bare ``**kwargs`` without reading them, so the try/except never
fires and ``generate()`` either trains on silently-dropped images or raises
late on an unused-kwargs ValueError. A media-declared load whose selected
class still cannot consume pixels REFUSES (exit 96) naming the model type,
rather than training blind. Gemma-4 is unaffected either way: both auto
mappings already resolve it to the same ``ForConditionalGeneration`` class.
prompt templating goes exclusively through ``tokenizer.apply_chat_template``
-- a tokenizer without a chat template is REFUSED, because silently
concatenating strings would silently change the prompt distribution the
objective is priced against.

WHAT IS CLAIMED: one ``run()`` performs at most ``max_steps`` optimiser
steps; every step recomputes ``old_logprobs`` under ``torch.no_grad()`` on
the same rows it backpropagates (generation scores are never reused -- their
shapes differ and the bug is silent); abstaining rows are dropped and the
report is built only from rows the gradient actually touched; and the
emitted ``StepReport`` carries a real ``LossOutput`` with
``loss=float(tensor)`` and with the #546 observability pair
``ratio_mean``/``clip_fraction`` in its metrics channel, measured off the
same kept tensors the loss was priced from; and an objective declaring a
non-zero ``kl_weight`` (grpo's k3) is priced against a frozen reference copy
of the initial policy, loaded once from ``reference_model or model`` only
when the objective needs it -- a reference-free objective never pays that
memory.

WHAT IS NOT CLAIMED: convergence, benchmark results, or equivalence with
any published implementation -- the equivalence test proves the tensor path
agrees with the audited arithmetic and nothing more; that generation is
correctly tuned; or that any row survives scoring on any given step.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    from collections.abc import Callable, Iterable, Iterator, Sequence

    import torch

from foundationscale.rl.advantage import AdvantageRefusal, RewardStats
from foundationscale.rl.algorithm import StepReport, StepReportRefusal
from foundationscale.rl.corpus import Sample, load_sharegpt
from foundationscale.rl.group_policy_objectives import prompt_mean_row_weights
from foundationscale.rl.interfaces import BatchRefusal, ExperienceBatch, LossOutput
from foundationscale.rl.online_objectives import BestOfNLoss, RAFTLoss
from foundationscale.rl.online_pref_step import (
    is_online_pref,
    maybe_refresh_reference,
    online_pref_step,
    refresh_cadence,
)
from foundationscale.rl.ppo_step import build_value_head, is_ppo, ppo_objective, ppo_step
from foundationscale.rl.prompt_surface import encode_prompts, resolve_prompt_surface
from foundationscale.rl.registry import lookup_algorithm
from foundationscale.rl.rewards import MCQLetterReward
from foundationscale.rl.torch_backend import (
    TensorMaskedSFTLoss,
    TensorPolicyLoss,
    TensorREINFORCELoss,
)

__all__ = (
    "RLTrainConfig",
    "RLTrainer",
    "TrainerRefusal",
)


class TrainerRefusal(RuntimeError):
    # Raised when the trainer cannot proceed for a configuration, environment
    # or data reason: a missing dependency, an unknown template surface, an
    # unparseable corpus. Distinct from BatchRefusal (a data-shape fact the
    # downstream contracts own) and from StepReportRefusal (one step's record
    # being incoherent): this is the loop's own refusal, before any step runs.
    pass


def _refuse_exit_96(message: str) -> NoReturn:
    # Exit contract: 96 REFUSE. Print the named reason and exit; never raise
    # ImportError for a missing optional dependency and never return 1.
    # The NoReturn annotation is load-bearing, not decoration: it is what lets
    # the type checker see that a refused load never falls through to a use of
    # the unbound name, so the call sites need no silencing comments.
    print(f"REFUSE: {message}", file=sys.stderr)
    raise SystemExit(96)


# How many groups the saturated-step line names individually before it stops and
# counts the rest. The message exists to be READ, and a run with many prompts per
# step would otherwise emit a line long enough that nobody reads any of it. The
# remainder is stated rather than dropped: a truncated list that does not say it
# was truncated is a different lie from the one this message was fixed to stop.
_MAX_GROUPS_REPORTED: int = 8


def _token_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Return per-token target log-probabilities without a dense log-softmax.

    ``log_softmax(logits).gather(...)`` first materialises a B x T x V
    tensor. This expression asks for the same scalar field as two B x T
    planes instead: the target score selected by ``gather`` and the
    full-vocabulary normaliser selected by ``logsumexp``. Upcasting the
    logits before the reduction keeps the reduction numerics explicit while
    leaving the full-vocabulary activation budget with the caller's row
    slice, which is the quantity ``logprob_micro_batch`` controls.
    """
    import torch  # function-local: see module docstring

    # The shift is EXPLICIT here. The dense form got it implicitly: gather
    # accepts an index shorter than its input on the non-gathered dims, so
    # it read logits[:, :T-1] -- the positions that predict target_ids --
    # and ignored the last. logsumexp has no such index and would normalise
    # over all T positions, so the logits are narrowed to the targets first.
    logits = logits[:, : target_ids.shape[1]]
    target = target_ids.unsqueeze(-1)
    return logits.gather(-1, target).squeeze(-1).float() - torch.logsumexp(logits.float(), dim=-1)


def _micro_batched_backward(
    *,
    loss_tensor: torch.Tensor,
    current_logprobs: torch.Tensor,
    row_slices: Sequence[tuple[int, int]],
    forward_slice: Callable[[int, int], torch.Tensor],
) -> None:
    """Deliver a surrogate leaf gradient through fresh row-sliced graphs.

    ``current_logprobs`` must be the detached batch-sized leaf already used
    by ``loss_tensor``. The first backward therefore computes only the
    derivative of the declared surrogate with respect to that leaf.
    Re-forwarding one slice and calling ``backward`` with the matching
    leaf-gradient slice states the chain rule directly, so parameter
    gradients accumulate to the same value as the whole-batch graph without
    retaining every row graph at once.

    ``optimizer.zero_grad()`` and ``optimizer.step()`` deliberately remain
    with the caller: this helper owns backward ordering, not the optimiser
    protocol around it.
    """
    loss_tensor.backward()
    output_grad = current_logprobs.grad
    if output_grad is None:
        raise RuntimeError(
            "the surrogate produced no gradient for the detached log-probability leaf; "
            "micro-batched replay cannot be chained"
        )
    for start, end in row_slices:
        slice_logp = forward_slice(start, end)
        slice_logp.backward(output_grad[start:end])


def _padding_backwards(
    *, forward_slice: Callable[[int, int], torch.Tensor], extra_slices: int, n_rows: int
) -> None:
    """Issue the busiest rank's surplus forward/backward pairs at zero gradient.

    FSDP all-gathers once per forward and reduce-scatters once per backward,
    so a rank holding fewer micro-batch slices than its busiest peer must
    still issue one pass per missing slice or the peers deadlock. The
    gradient of ``sum * 0.0`` is exactly zero, so the step is unchanged.
    """
    for _ in range(extra_slices):
        lane = forward_slice(0, min(1, n_rows))
        (lane.sum() * 0.0).backward()


def _dp_weighted_loss(local_loss: float, weight: float, ctx: Any) -> tuple[float, float]:
    """Row-weighted mean loss over the ranks that measured, plus that weight.

    A null rank passes ``weight=0.0`` so its zeroed dummy loss is excluded
    rather than averaged in as a spurious 0.0. Both reductions are issued on
    every rank in the same order, whatever the local data.
    """
    from foundationscale.rl.distributed import all_reduce_sum

    measured_weight = all_reduce_sum(weight, ctx)
    if not ctx.is_distributed:
        return local_loss, measured_weight
    return all_reduce_sum(local_loss * weight, ctx) / measured_weight, measured_weight


def _per_group_reward_summary(
    kept_rows: Sequence[int],
    prompt_ids: Sequence[str],
    rewards: Sequence[float],
) -> str:
    """Distinct rewards per GROUP, for the saturated-step line.

    Kept separate from the step so the shape of this evidence is testable
    without a model, a tokenizer and a GPU -- which is why the pooled version it
    replaces shipped unexamined.

    The pooled version computed one set over all used rows, so a step whose
    first group scored all 0.0 and whose second scored all 1.0 printed "no
    within-group variance (distinct rewards: [0.0, 1.0])": evidence that appears
    to refute the sentence attached to it, on a step where the sentence is true.
    Variance is a per-group property and the report has to be too.
    """
    per_group: dict[str, set[float]] = {}
    for row in kept_rows:
        per_group.setdefault(prompt_ids[row], set()).add(float(rewards[row]))
    shown = sorted(per_group.items())[:_MAX_GROUPS_REPORTED]
    summary = "; ".join(f"{name}: {sorted(values)}" for name, values in shown)
    if len(per_group) > len(shown):
        summary += f"; (+{len(per_group) - len(shown)} more group(s))"
    return summary


def _refuse_vacuous_run(*, attempted: int, measured: int) -> None:
    # Every UNMEASURED step names itself on stderr as it is skipped, but a
    # caller holding the returned list cannot tell a run that measured nothing
    # from a run that was never asked to measure: both are []. max_steps < 1 is
    # already refused at construction because "an empty run measures nothing";
    # a run that attempts its steps and lands zero of them is empty by the same
    # argument, discovered later. Refusing here is what keeps the emptiness from
    # being read as a quiet success -- this framework's founding failure.
    if attempted > 0 and measured == 0:
        raise TrainerRefusal(
            f"{attempted} step(s) attempted and 0 produced a measurable update; "
            f"every step was UNMEASURED and said so above. A run that trained "
            f"nothing measures nothing and is refused as vacuous"
        )


class MasterWeightOptimizer:
    """Host-fp32-master wrapper around torch.optim.AdamW.

    WHY THIS EXISTS (measured, do not re-derive): model params load as
    bf16, and at the default lr=1e-6 each AdamW update is BELOW the bf16
    ulp of the param it targets. bf16 arithmetic DISCARDS such an update
    outright -- it does not attenuate it, and nothing accumulates. Over 50
    direct-bf16 steps only 0.98% of param entries ever move, IDENTICAL to
    after 1 step: the optimiser runs, the model does not learn. Holding
    fp32 MASTER copies on the HOST (device peak 109.51 -> 68.08 GiB, the
    Adam state leaves the GPU) and casting results back down moves 20.58%
    of entries by step 50, 21x. Full fp32 params ON DEVICE do not fit
    (183.41 of 184.31 GiB, OOM), so the masters live on the host.

    SCHEME: at construction, snapshot every requires-grad device param as
    an fp32 CPU tensor; that CPU list is what AdamW holds and steps. At
    each step(): (1) copy each device grad UP into the matching master as
    fp32, (2) run the wrapped AdamW step over the masters (true fp32
    arithmetic, so updates at lr=1e-6 are representable and accumulate),
    (3) copy each master back DOWN into its device param, cast to that
    param's dtype. zero_grad() clears BOTH planes: the device grads are
    the ones the training loop writes, and the master grads are the ones
    AdamW reads, so leaving either stale would double-count.

    Drop-in for the existing ``optimizer`` variable: exposes step() and
    zero_grad() only. zip uses strict=True on every pairing of the two
    parallel lists -- a silent length mismatch here would train the wrong
    tensors, which is exactly the class of invisible failure this wrapper
    exists to eliminate.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], **adamw_kwargs: Any) -> None:
        import torch  # function-local: see module docstring

        self.device_params = [p for p in params if p.requires_grad]
        self.masters = [
            p.detach().to(device="cpu", dtype=torch.float32, copy=True) for p in self.device_params
        ]
        for master in self.masters:
            master.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(self.masters, **adamw_kwargs)

    def zero_grad(self, set_to_none: bool = True) -> None:
        import torch  # function-local: see module docstring

        for p in self.device_params:
            p.grad = None if set_to_none else torch.zeros_like(p)
        for m in self.masters:
            m.grad = None if set_to_none else torch.zeros_like(m)

    def step(self, closure: Any = None) -> Any:
        import torch  # function-local: see module docstring

        # (1) device grads up to fp32 masters.
        for p, m in zip(self.device_params, self.masters, strict=True):
            if p.grad is None:
                m.grad = None
            else:
                m.grad = p.grad.detach().to(device="cpu", dtype=torch.float32, copy=True)
        # (2) true fp32 step: lr=1e-6 updates are representable here and
        # therefore ACCUMULATE instead of being discarded at the bf16 ulp.
        result = (
            self.optimizer.step(closure=closure) if closure is not None else self.optimizer.step()
        )
        # (3) masters back down to the device params' dtype; the cast cost
        # is one bf16 rounding per step, not one per micro-update.
        for p, m in zip(self.device_params, self.masters, strict=True):
            p.data.copy_(m.to(dtype=p.dtype, copy=False))
        return result


def _expand_video_samples(samples: Sequence[Sample], config: Any) -> tuple[Sample, ...]:
    """Each video-carrying sample becomes its declared frames, appended as images.

    Refuses (96) when video is present but no frame budget was declared, and when
    a clip cannot be decoded -- naming the sample, never substituting a blank
    frame or silently training on the text alone.
    """
    import dataclasses
    from pathlib import Path

    carriers = [sample for sample in samples if sample.video is not None]
    if not carriers:
        return tuple(samples)
    if int(getattr(config, "video_frames", 0) or 0) < 1:
        shown = ", ".join(f"{s.sample_id} (video)" for s in carriers[:5])
        more = f" and {len(carriers) - 5} more" if len(carriers) > 5 else ""
        _refuse_exit_96(
            f"{len(carriers)} of {len(samples)} sample(s) carry video: {shown}{more}. "
            "Frame count, sampling strategy and per-frame resolution are dataset "
            "decisions with no safe default; declare them with video_frames (>= 1), "
            "video_sampling and video_max_side, and each clip is routed as that many "
            "frames through the image path."
        )
    from foundationscale.video import FrameBudget, VideoDecodeError, video_frame_paths

    try:
        budget = FrameBudget(
            frames=int(config.video_frames),
            sampling=str(config.video_sampling),
            max_side=config.video_max_side,
        )
    except ValueError as exc:
        _refuse_exit_96(f"invalid video frame budget: {exc}")
    cache = config.video_cache_dir or str(
        Path(config.dataset).resolve().parent / ".fs_video_frames"
    )
    expanded: list[Sample] = []
    for sample in samples:
        if sample.video is None:
            expanded.append(sample)
            continue
        clip = Path(sample.video)
        if not clip.is_absolute():
            clip = Path(config.dataset).resolve().parent / clip
        try:
            frames = video_frame_paths(clip, budget, cache)
        except (FileNotFoundError, VideoDecodeError) as exc:
            _refuse_exit_96(f"sample {sample.sample_id!r}: {exc}")
        expanded.append(
            dataclasses.replace(sample, images=tuple(sample.images) + tuple(frames), video=None)
        )
    return tuple(expanded)


# The only adapter kind either trainer in this module wires. A second kind
# that someone types into a config dict and silently gets ignored is exactly
# the class of defect __init__'s validation exists to turn into a refusal.
_ADAPTERS: tuple[str, ...] = ("lora",)


class _AdapterDisabledReference:
    """A frozen reference forward, read off the SAME peft-wrapped policy.

    WHY THIS EXISTS: under ``adapter='lora'`` the base weights ARE the
    frozen reference -- peft's LoRA delta is the only trained quantity --
    so loading a second full model copy would duplicate every frozen byte
    next to the one already resident, which is exactly the memory
    ``wrap_fsdp2``'s own module docstring says FSDP2 exists to avoid for the
    full-parameter case. ``model.disable_adapter()`` (a peft context
    manager) already turns every LoRA module's forward back into the bare
    base-model computation for its duration; this class makes that context
    manager answer the SAME calling convention every ``ref_model(...)``
    call site in this package already uses (``ref_model(input_ids=...,
    attention_mask=..., **kwargs).logits``), so none of those call sites --
    ``_one_step``, ``_priced_tail``, ``ppo_step.forward_rows``,
    ``online_pref_step.summed_logprobs`` -- change at all.

    Puts ``policy_model`` into ``eval()`` for the duration of the call and
    restores whatever mode it found before returning: peft's own
    ``disable_adapter()`` toggles only the adapter, not dropout/train-vs-
    eval behaviour, and the step-0 "reference equals policy" invariant
    (grpo's k3 term, DPO's margin) needs the SAME deterministic forward the
    real second-copy path got from a freshly loaded, ``.eval()``'d model.

    WHAT IS CLAIMED: one call is one forward of the base model (no LoRA
    delta applied), under ``torch.no_grad()``, and the policy model's
    training mode and adapter are both restored before the call returns --
    this proxy never leaves the policy model in a different state than it
    found it.

    WHAT IS NOT CLAIMED: that ``.eval()``/``.parameters()`` return anything
    meaningful. The only caller that wants the reference's OWN parameter
    list is ``maybe_refresh_reference``'s periodic resync (iterative DPO),
    and the call site that would hand it this proxy refuses first instead
    (see its comment in ``RLTrainer.run``) -- a resync has no base weights
    to copy TO here, since disabling the adapter always reads the one,
    unchanging base checkpoint.

    Unwraps one ``.module`` on construction when present: under
    ``sharding='ddp'`` the caller's ``model`` is a
    ``DistributedDataParallel`` instance, which does NOT proxy arbitrary
    attributes (``disable_adapter``, peft's own methods) through to the
    wrapped module the way this package's own ``getattr(model, "module",
    model)`` idiom (``save_checkpoint``, ``_one_step``'s generate call)
    already has to account for -- calling ``.disable_adapter()`` on the DDP
    wrapper itself would raise ``AttributeError``. Reading the raw module
    directly is also the CORRECT choice, not merely the one that does not
    crash: the reference forward runs under ``torch.no_grad()``, so there is
    no gradient for DDP's wrapper to synchronise and no reason to pay its
    hook overhead. FSDP2's ``fully_shard`` needs no such unwrap -- it
    augments the SAME module object in place rather than wrapping it in a
    container, which is exactly why GPU proof (a) (sharding='fsdp') did not
    surface this; ddp would have hit it on its first reference call.
    """

    def __init__(self, policy_model: Any) -> None:
        self._policy_model = getattr(policy_model, "module", policy_model)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch  # function-local: see module docstring

        policy = self._policy_model
        was_training = bool(policy.training)
        policy.eval()
        try:
            with torch.no_grad(), policy.disable_adapter():
                return policy(*args, **kwargs)
        finally:
            policy.train(was_training)

    def eval(self) -> _AdapterDisabledReference:
        # No persistent mode to flip: __call__ already brackets every
        # forward in eval()/train() around the policy model itself.
        return self

    def parameters(self) -> Iterable[Any]:
        # Deliberately empty, not the policy's own parameters: handing
        # those out would let a caller expecting an INDEPENDENT reference
        # silently read (or, worse, write) the policy's live tensors.
        return iter(())


def _apply_lora_adapter(
    model: Any,
    *,
    adapter: str | None,
    adapter_rank: int | None,
    adapter_alpha: float | None,
    adapter_targets: tuple[str, ...] | None,
    adapter_dropout: float | None,
    log_prefix: str,
) -> tuple[Any, dict[str, str]]:
    """Wrap ``model`` with peft LoRA, or return it unchanged when ``adapter`` is None.

    Reuses the SAME family-registry target selection
    ``train/loop.py``'s SFT plane uses (:func:`foundationscale.families.plan_adapter_targets`),
    so a VLM's vision tower is excluded and target coverage is checked the
    identical way on both planes -- this policy lives in ``families``
    precisely so no second copy of it could drift from the first.

    Call this AFTER the model is loaded and BEFORE any FSDP/DDP wrap: peft
    replaces specific ``nn.Linear`` leaves in place, which ``wrap_fsdp2``
    (composable ``fully_shard``) and ``find_decoder_blocks`` (class-name
    matching on the surrounding decoder block, untouched by the leaf swap)
    both tolerate; wrapping the other way round would hand peft an
    already-sharded DTensor tree to replace leaves inside, which it does
    not support.

    Refuses (exit 96, via ``_refuse_exit_96``) when: peft is not installed;
    the family/target plan itself refuses (no family and no declared
    targets, or a declared target that resolves nothing); or the adapter
    attaches to 0 modules. A ``get_peft_model`` construction exception is
    DELIBERATELY left to propagate uncaught -- same classification
    ``train/loop.py`` gives it and for the same reason stated there: it
    rewrites an already-constructed model in memory and opens no file and
    no socket, so there is no environment errno for a refusal classifier to
    read, and a branch here would be unfireable by any test.
    """
    if adapter is None:
        return model, {}
    if adapter not in _ADAPTERS:  # pragma: no cover -- callers validate first
        _refuse_exit_96(f"adapter={adapter!r} is not one of {_ADAPTERS}")
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        _refuse_exit_96(
            f"adapter={adapter!r} is declared but the optional dependency "
            "'peft' is not installed. Refusing rather than silently running "
            "a full fine-tune"
        )
    from foundationscale.families import plan_adapter_targets, torch_linear_predicate

    lora_config: dict[str, Any] = {"r": adapter_rank}
    if adapter_alpha is not None:
        lora_config["lora_alpha"] = adapter_alpha
    family_config: Any = {}
    config_to_dict = getattr(getattr(model, "config", None), "to_dict", None)
    if callable(config_to_dict):
        as_dict = config_to_dict()
        if isinstance(as_dict, dict):
            family_config = as_dict
    plan = plan_adapter_targets(
        family_config,
        adapter_targets,
        model.named_modules(),
        torch_linear_predicate(),
    )
    for line in plan.announcements:
        print(f"{log_prefix} adapter: {line}", file=sys.stderr)
    if plan.refusal is not None:
        _refuse_exit_96(f"adapter={adapter!r}: {plan.refusal}")
    lora_config["target_modules"] = list(plan.targets)
    if adapter_dropout is not None:
        lora_config["lora_dropout"] = adapter_dropout
    model = get_peft_model(model, LoraConfig(**lora_config))
    lora_param_names = [name for name, _ in model.named_parameters() if ".lora_" in name]
    attached_modules = sorted({name.split(".lora_")[0] for name in lora_param_names})
    if not attached_modules:
        _refuse_exit_96(
            f"adapter={adapter!r} attached to 0 modules (targets="
            f"{list(adapter_targets) if adapter_targets is not None else None!r}); "
            "refusing as vacuous -- an adapter that targets nothing trains "
            "nothing while looking like it trained"
        )
    trainable = sum(int(p.numel()) for _, p in model.named_parameters() if p.requires_grad)
    total_params = sum(int(p.numel()) for _, p in model.named_parameters())
    notes = {
        "adapter.mode": adapter,
        "adapter.rank": str(adapter_rank),
        "adapter.alpha": str(adapter_alpha),
        "adapter.targets_declared": (
            ",".join(adapter_targets) if adapter_targets is not None else "(peft defaults)"
        ),
        "adapter.attached_modules": str(len(attached_modules)),
        "adapter.resolved_targets": ",".join(attached_modules),
        "adapter.trainable_params": str(trainable),
        "adapter.total_params": str(total_params),
    }
    print(
        f"{log_prefix} adapter: lora attached to {len(attached_modules)} module(s); "
        f"{trainable}/{total_params} parameters trainable; undeclared knobs left "
        "to peft defaults",
        file=sys.stderr,
    )
    return model, notes


def _trainable_parameters(model: Any) -> Iterable[Any]:
    """Parameters an optimizer should step: ``requires_grad`` only.

    A no-op filter for a full fine-tune -- every parameter already requires
    grad there, so the filtered generator yields the exact same parameters
    in the exact same order as ``model.parameters()`` -- and the difference,
    under ``adapter='lora'``, between training only the LoRA delta and
    quietly handing AdamW (or ``MasterWeightOptimizer``, which already
    filters internally -- see its own docstring -- making this a harmless
    second pass there) optimiser state for every frozen base tensor too.
    """
    return (p for p in model.parameters() if p.requires_grad)


def _reference_plan(
    *,
    adapter: str | None,
    reference_model: str | None,
    model: str,
    refresh_every: int,
) -> str:
    """Which reference strategy a trainer's reference-loading block should use.

    Returns ``"disable_adapter"`` when the frozen reference should be read
    off the SAME policy with its LoRA adapter disabled (see
    :class:`_AdapterDisabledReference`) -- true exactly when an adapter is
    declared AND the reference is this run's OWN model (``reference_model``
    is ``None`` or equal to ``model``; a distinct ``reference_model`` is a
    genuinely different checkpoint that disabling an adapter cannot reach).
    Returns ``"second_copy"`` for every other case, INCLUDING ``adapter is
    None`` -- the historical, unconditional second-model-load path, so a
    caller branching on this return value reproduces that path byte for
    byte when no adapter is declared.

    Raises :class:`TrainerRefusal` when the resolved strategy is
    ``"disable_adapter"`` but ``refresh_every > 0``: iterative-DPO-style
    periodic reference refresh copies the CURRENT policy into the
    reference, and under LoRA disabling the adapter always reads the
    ORIGINAL, unchanging base checkpoint -- there is no policy drift for a
    refresh to capture, so running one would silently do nothing while
    reporting a refresh.
    """
    reference_is_this_model = reference_model is None or reference_model == model
    if adapter is not None and reference_is_this_model:
        if refresh_every > 0:
            raise TrainerRefusal(
                f"a reference refresh every {refresh_every} step(s) "
                f"(ref_refresh_steps={refresh_every}) was requested, but "
                "adapter='lora' with no distinct reference_model makes the "
                "reference model.disable_adapter() -- always the ORIGINAL "
                "frozen base, which cannot be refreshed to track the "
                "policy's LoRA updates. Set ref_refresh_steps=0, declare "
                "adapter=None, or name a distinct reference_model"
            )
        return "disable_adapter"
    return "second_copy"


def _load_causal_lm(model_id: str, *, needs_images: bool) -> Any:
    """Load ``model_id`` under the auto class the corpus actually needs.

    Module level (not a closure inside ``run()``) so the policy load and the
    frozen reference load -- two call sites that must never drift -- share
    ONE decision, and so this function is directly unit-testable the same
    way ``_expand_video_samples``/``_reference_plan`` are, with no need to
    drive a whole ``run()``.

    Text-only corpora (``needs_images=False``) are untouched: the historical
    try-``AutoModelForCausalLM``-except-try-``AutoModelForImageTextToText``
    fallback, byte-identical to every run before this function existed.
    That fallback is deliberately NOT reused for media-declared corpora --
    doing so is the defect this closes. MEASURED on transformers 5.18.0:
    ``AutoModelForCausalLM.from_pretrained`` on a qwen3_5/qwen3_5_moe
    checkpoint (Qwen3.6-27B/35B-A3B) SUCCEEDS, handing back
    ``Qwen3_5ForCausalLM``/``Qwen3_5MoeForCausalLM`` -- text-only classes
    that skip ``model.visual`` on load and absorb
    ``pixel_values``/``image_grid_thw``/``mm_token_type_ids`` through a bare
    ``**kwargs`` without reading them -- so the try/except never raises and
    the mismatch surfaces later, at the first ``generate()``, as "model_kwargs
    are not used by the model", or worse: silent, pixel-free training under a
    multimodal label.

    A media-declared corpus instead asks ``train/loop.py``'s own
    ``_media_capable_auto_class`` (imported, never duplicated -- the SFT
    plane already fixed this exact defect, see
    tests/train/test_model_auto_class_selection.py) which auto class to use,
    BEFORE calling ``from_pretrained``, then proves the loaded instance can
    actually consume pixels with ``_model_can_consume_pixels`` and REFUSES
    (exit 96, naming the model type) rather than training blind if it
    cannot. ``gemma4_unified`` is unaffected: both auto-mappings already
    resolve it to the one ``Gemma4UnifiedForConditionalGeneration`` class.

    Annotated Any: transformers 5.x wraps ``from_pretrained`` in a decorator
    whose return type does not survive inference, so a caller's subsequent
    ``.to(device)`` would resolve against the wrapper rather than the model
    and report the device string as a bad `self`. The alternative -- a cast
    to PreTrainedModel -- would assert a class neither branch promises.
    """
    if not needs_images:
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
        )

        try:
            loaded: Any = AutoModelForCausalLM.from_pretrained(model_id)
        except Exception:
            try:
                loaded = AutoModelForImageTextToText.from_pretrained(model_id)
            except Exception as exc:  # noqa: BLE001
                _refuse_exit_96(
                    f"model load failed for {model_id!r} under both auto classes: {exc}"
                )
        return loaded

    # Media-declared: decide the class BEFORE from_pretrained, deterministically
    # -- never via try/except-after-the-fact, which is exactly how the
    # Qwen3_5 defect this closes went undetected (AutoModelForCausalLM's own
    # from_pretrained call never raises on that family).
    from transformers import AutoConfig  # noqa: PLC0415

    from foundationscale.train.loop import (  # noqa: PLC0415
        _family_config_mapping,
        _media_capable_auto_class,
        _model_can_consume_pixels,
    )

    try:
        model_config = AutoConfig.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001
        _refuse_exit_96(f"config load failed for {model_id!r}: {exc}")
    auto_class, auto_class_name, verify_pixel_capability = _media_capable_auto_class(
        model_config, media_declared=True
    )
    try:
        loaded = auto_class.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001
        _refuse_exit_96(f"model load failed for {model_id!r} under {auto_class_name}: {exc}")
    if verify_pixel_capability and not _model_can_consume_pixels(loaded):
        # Read through the SAME mapping _model_can_consume_pixels' own family
        # resolution uses (handles both a real transformers config object and
        # a plain-dict double identically) -- naming the model_type here must
        # not invent a second, narrower reading of "config" than the check
        # it is reporting on.
        model_type = _family_config_mapping(loaded).get("model_type")
        _refuse_exit_96(
            f"model_id={model_id!r} carries images in this corpus "
            f"(needs_images=True) and model_type={model_type!r} is registered "
            f"under AutoModelForImageTextToText, but the loaded "
            f"{auto_class_name} instance cannot consume pixels: its forward() "
            "names no pixel_values parameter, or its family registry's "
            "declared image tower does not resolve on the loaded module "
            "tree. Training anyway would silently drop every image under a "
            "multimodal label -- refusing rather than training blind"
        )
    return loaded


# (flattened-patch value key, its per-image patch-count key). MEASURED
# transformers 5.18.0 naming on Qwen2-VL/Qwen2.5-VL/Qwen3-VL-family
# processors; a family without a matching pair (gemma-4's pixel_values is
# already per-image, no *_grid_thw key at all) is untouched by this table.
_FLATTENED_PATCH_PAIRS: tuple[tuple[str, str], ...] = (
    ("pixel_values", "image_grid_thw"),
    ("pixel_values_videos", "video_grid_thw"),
)


def _expand_modality_kwargs_for_group(
    modality_kwargs: dict[str, torch.Tensor], *, group: int, kept_indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Expand one row of modality tensors per PROMPT into one per KEPT, grouped row.

    MEASURED (GRPO+images, Qwen3.6-27B/35B-A3B): not every per-image key has
    one row per prompt. gemma-4's ``pixel_values`` does -- shape
    ``(n_images, ...)``, one image per row -- so repeating rows ``group``
    times and narrowing to ``kept_indices`` (both indexed in the SAME
    group-expanded row space ``group_policy``/``generate()`` use) is exactly
    right, and is what this function still does for every key with no
    flattened-patch pairing below.

    Qwen2-VL/Qwen2.5-VL/Qwen3-VL's own ``pixel_values``, though, is
    FLATTENED PATCHES across the whole encoded chunk: shape
    ``(sum_i patches_i, patch_dim)``, with ``image_grid_thw`` (shape
    ``(n_images, 3)``, each row ``(t, h, w)``) giving each image's own
    ``patches_i = t*h*w`` contiguous block length. Row-wise
    ``repeat_interleave``/``index_select`` on that tensor slices by RAW
    PATCH position, not by image -- MEASURED, confirmed by printing shapes
    at this exact call site before this fix existed: two images with patch
    counts ``[320, 288]`` (608 total rows) and ``kept_indices=[0, 1]`` (both
    group-expanded rows mapping to prompt 0) produced a 2-row
    ``pixel_values`` (two individual duplicated PATCHES, not prompt 0's 320)
    against an ``image_grid_thw`` still correctly claiming 2 images of 320
    patches each (640) -- exactly the
    ``RuntimeError: size of tensor a (2) must match b (640)`` the vision
    tower's position-embedding add raised.

    The fix: for a (value, count) pair whose value width equals the count
    tensor's ``prod(-1).sum()`` -- i.e. is genuinely flattened patches, not
    coincidentally already per-row -- the ORIGINAL per-image block
    boundaries are read off ``image_grid_thw`` (cumulative ``prod(-1)``), and
    each KEPT, group-expanded row's own image block (``kept_indices //
    group`` recovers the original image index: ONE image per prompt is this
    plane's only declared shape, the same arithmetic ``group_ids`` already
    uses) is concatenated in order -- never repeated/selected by raw patch
    position. ``image_grid_thw`` itself is expanded the same row-wise way
    every other per-image key is (its own width IS one row per image), so
    the two stay in lockstep by construction, not by coincidence.

    WHAT IS NOT CLAIMED: more than one image per prompt. A future corpus
    declaring that would need ``kept_indices // group`` replaced with a real
    per-prompt image-count mapping; this function has no such input and
    would mis-divide silently, which is why the shape-equality test above is
    the ONLY detector -- a family whose flattened-patch total does not match
    is left on the per-row path rather than guessed into this one.
    """
    import torch  # function-local: see module docstring

    flattened_keys: set[str] = set()
    expanded: dict[str, torch.Tensor] = {}
    source_images = torch.div(kept_indices, group, rounding_mode="floor")

    for value_key, count_key in _FLATTENED_PATCH_PAIRS:
        value = modality_kwargs.get(value_key)
        counts = modality_kwargs.get(count_key)
        if value is None or counts is None:
            continue
        patch_counts = counts.prod(dim=-1).to(torch.long)
        if int(value.shape[0]) != int(patch_counts.sum().item()):
            # Not actually flattened patches on this family/build (e.g.
            # already per-row) -- left for the generic path below rather
            # than block-expanded on a guess.
            continue
        offsets = torch.cumsum(torch.cat([patch_counts.new_zeros(1), patch_counts]), dim=0)
        blocks = [value[offsets[i] : offsets[i + 1]] for i in source_images.tolist()]
        expanded[value_key] = (
            torch.cat(blocks, dim=0) if blocks else value.new_zeros((0, *value.shape[1:]))
        )
        expanded[count_key] = counts.index_select(0, source_images)
        flattened_keys.add(value_key)
        flattened_keys.add(count_key)

    for key, value in modality_kwargs.items():
        if key in flattened_keys:
            continue
        expanded[key] = value.repeat_interleave(group, dim=0).index_select(0, kept_indices)

    return expanded


def _narrow_modality_kwargs_by_row(
    modality_kwargs: dict[str, torch.Tensor], *, start: int, end: int
) -> dict[str, torch.Tensor]:
    """Row-range-narrow modality tensors, treating flattened-patch keys as BLOCKS.

    MEASURED (GRPO+images, Qwen3.6-27B/35B-A3B, ``logprob_micro_batch``
    slicing): after :func:`_expand_modality_kwargs_for_group`,
    ``image_grid_thw``/``video_grid_thw`` has exactly ONE row per kept
    training row -- ``forward_logprob_slice``'s plain
    ``value.narrow(0, start, end - start)`` is already correct for it, the
    same as ``input_ids``/``attention``. Its paired flattened-patch value
    (``pixel_values``/``pixel_values_videos``) is still the concatenation of
    each row's own patch BLOCK in row order; row-wise narrowing it the same
    way truncates to the first ``end - start`` raw PATCHES, not the patches
    belonging to rows ``[start, end)``. MEASURED on GPU: a 1-row logprob
    slice over a 320-patch image produced a 1-patch ``hidden_states`` against
    a 320-patch ``pos_embeds`` inside the vision tower's position-embedding
    add (``RuntimeError: size of tensor a (1) must match b (320)``; a 2-row
    slice gave ``(2)`` vs ``(640)``) -- the SAME flattened-vs-per-row
    confusion :func:`_expand_modality_kwargs_for_group` closes for the
    group-expansion step, recurring here at the micro-batch-slicing step.

    The fix: for a (value, count) pair whose value width equals the count
    tensor's ``prod(-1).sum()``, the per-row block boundaries are read off
    the count tensor (cumulative ``prod(-1)``, already in row order by
    construction) and the SINGLE contiguous span covering rows
    ``[start, end)`` is sliced out -- no gather needed, unlike the
    group-expansion step, because the blocks are already laid out in that
    exact order.
    """
    import torch  # function-local: see module docstring

    narrowed: dict[str, torch.Tensor] = {}
    patch_keys: set[str] = set()

    for value_key, count_key in _FLATTENED_PATCH_PAIRS:
        value = modality_kwargs.get(value_key)
        counts = modality_kwargs.get(count_key)
        if value is None or counts is None:
            continue
        patch_counts = counts.prod(dim=-1).to(torch.long)
        if int(value.shape[0]) != int(patch_counts.sum().item()):
            continue
        offsets = torch.cumsum(torch.cat([patch_counts.new_zeros(1), patch_counts]), dim=0)
        narrowed[value_key] = value[int(offsets[start].item()) : int(offsets[end].item())]
        patch_keys.add(value_key)

    for key, value in modality_kwargs.items():
        if key in patch_keys:
            continue
        narrowed[key] = value.narrow(0, start, end - start)

    return narrowed


def _align_modality_keys_to_scored_width(
    modality_kwargs: dict[str, torch.Tensor], *, prompt_width: int, sequence_width: int
) -> dict[str, torch.Tensor]:
    """Extend per-token modality tensors from prompt width to the scored width.

    MEASURED (GRPO+images, Qwen3.6-27B/35B-A3B): every key ``_one_step`` pulls
    out of ``prompt_ids`` comes from the SAME processor call as
    ``prompt_ids["input_ids"]`` and so is produced at PROMPT width -- but not
    all of those keys are per-TOKEN. ``pixel_values``/``image_grid_thw`` are
    per-IMAGE: their non-batch dims (patch features, ``(t, h, w)``) have
    nothing to do with sequence length and must pass through unchanged.
    ``mm_token_type_ids`` (Qwen3VL) and ``image_position_ids`` (gemma-4) ARE
    per-token: shape ``(rows, prompt_width)``, one entry per prompt token.
    The scorer forward, though, runs over ``kept_sequences`` -- prompt_width
    + max_new_tokens columns, since #371 scores the GENERATED completion
    too -- while this dict was only ever repeat_interleaved/index_selected
    along the ROW axis, never extended along the sequence axis. Qwen3.5's own
    ``get_rope_index`` then indexes a full-width attention_mask against a
    prompt-width ``mm_token_type_ids`` and raises: MEASURED, mask shape
    ``[927]`` vs tensor shape ``[527]`` on one rank and ``[787]`` vs ``[387]``
    on another, same step -- both exactly that rank's prompt_width plus the
    400-token completion, confirmed by printing the shapes at the call site
    before this fix existed.

    A per-token key is told apart from a per-image key by shape alone --
    there is no family-registry lookup available this deep in the generic
    scoring path -- and this holds for both measured per-token keys above
    because they share ``prompt_ids["input_ids"]``'s exact width by
    construction. The completion columns are always actually-generated TEXT
    (never a second image), so they are padded with 0 -- plain text's own
    value in Qwen3VL's token-type convention. gemma-4 was never observed to
    crash on this (its rope path does not index by ``image_position_ids`` the
    way qwen3_5's does), so this is behaviourally a no-op for it either way;
    the pad only removes a silently-truncated tensor that some OTHER reader
    could trip on later.
    """
    import torch  # function-local: see module docstring

    if sequence_width <= prompt_width:
        return modality_kwargs
    pad_width = sequence_width - prompt_width
    aligned: dict[str, Any] = {}
    for key, value in modality_kwargs.items():
        if value.dim() >= 2 and value.shape[1] == prompt_width:
            pad = value.new_zeros((value.shape[0], pad_width, *value.shape[2:]))
            aligned[key] = torch.cat([value, pad], dim=1)
        else:
            aligned[key] = value
    return aligned


@contextmanager
def _generation_mode(model: Any) -> Iterator[Any]:
    """Put ``model`` into the state ``generate()`` needs, and restore it after.

    MEASURED ROOT CAUSE (gemma-4-12B-it, GRPO rollout, 2026-10-09): every
    online algorithm's rollout calls ``.generate()`` while the model is
    still in ``.train()`` mode -- set once, before the step loop starts, and
    never toggled for the rollout -- because ``gradient_checkpointing_enable()``
    (called once at setup) leaves ``model.config.use_cache = False`` and
    ``.train()`` is what the surrounding code calls after wrapping.
    ``Gemma4UnifiedTextDecoderLayer.forward`` checks ``self.training and
    self.gradient_checkpointing`` and, when both are true, drops its KV
    cache and sets ``past_key_values=None`` -- CORRECT for a training
    forward (checkpointing recomputes activations and a cache would be
    stale by the recompute), but ``generate()``'s incremental decode loop
    assumes a working cache: each new token is produced from ONLY the
    single newest input id plus whatever the cache remembers, so a cache
    silently dropped mid-generation conditions every token after the first
    on almost no context. MEASURED: the first generated token is correct
    (full-prompt forward, no cache needed yet) and every token after it is
    near-random -- ``'A Sqh有意х 나오"--ᇲο¬一切 ...'`` instead of a bare
    letter. This is SILENT: ``generate()`` raises nothing and returns a
    full-shaped tensor, so a caller that does not read the decoded text
    never learns the rollout was corrupted -- exactly the failure class
    this repository's doctrine refuses to let ship unmeasured.

    THE FIX: bracket the ``generate()`` call in ``.eval()`` (so
    ``self.training`` is False throughout the generated module tree,
    including every decoder layer, for the duration of the call -- peft's
    LoRA layers and the adapter's enabled/disabled state are UNCHANGED by
    eval/train, so rollouts still sample the POLICY, adapter included) and
    in ``config.use_cache = True`` when the family exposes that attribute
    (checked via ``hasattr``, never assumed -- a family with no ``config``
    or no ``use_cache`` field is left alone rather than crashing or
    fabricating the attribute). Both are restored to their EXACT prior
    values before this returns, so a caller resuming training afterward
    sees byte-identical state to what it would have without this fix:
    ``.train(was_training)`` (not unconditionally ``.train()`` -- a caller
    already in eval for some other reason must not be flipped to train),
    and ``config.use_cache = prior_use_cache`` (not unconditionally
    restored to False -- a caller that never set it at all must not gain a
    new attribute).

    Unwraps ``.module`` first, the same idiom ``save_checkpoint`` and every
    ``.generate(`` call site already use: under ``sharding='ddp'`` the live
    ``model`` is a ``DistributedDataParallel`` instance, and toggling
    ``.training``/``.config`` on the wrapper reads/writes the SAME
    underlying attributes as the wrapped module (DDP does forward plain
    attribute access for ``training``, unlike the custom methods
    ``_AdapterDisabledReference`` has to unwrap for), but unwrapping once
    here keeps this function's contract identical regardless of sharding,
    rather than relying on that forwarding behaviour implicitly.

    Also explicitly unshards every FSDP2 unit for the duration (see
    :func:`foundationscale.rl.distributed.unshard_for_generation`'s
    docstring for the measured reason a forward-hook-only fix is not
    enough: a submodule ``generate()``'s multimodal preprocessing reaches
    directly, before the root's own first forward, can leave the root's
    OWN unit un-materialised on its very next regular call). A no-op on
    DDP/single-process models -- :func:`unshard_for_generation` finds zero
    FSDP2 units on either.
    """
    from foundationscale.rl.distributed import unshard_for_generation

    target = getattr(model, "module", model)
    was_training = bool(target.training)
    config: Any = getattr(target, "config", None)
    has_use_cache = hasattr(config, "use_cache")
    prior_use_cache = config.use_cache if has_use_cache else None
    target.eval()
    if has_use_cache:
        config.use_cache = True
    reshard = unshard_for_generation(target)
    try:
        yield target
    finally:
        reshard()
        if has_use_cache:
            config.use_cache = prior_use_cache
        target.train(was_training)


@dataclass
class RLTrainConfig:
    """Configuration for one RL training run.

    ``model`` is a local path or hub id and is NEVER defaulted: hardcoding a
    default model would smuggle an untested surface into every run that
    forgot the flag. ``algorithm`` names a registry entry
    (``"grpo"``/``"gspo"``/``"dr_grpo"``/``"dapo"``/``"agentic_grpo"``). ``device=None`` means
    auto-select -- metal when available, else CPU.

    WHAT IS CLAIMED: these are the only knobs the loop reads.

    WHAT IS NOT CLAIMED: that any particular value trains well; nothing here
    is tuned.
    """

    model: str
    # The ShareGPT corpus the built-in generate-and-score leg draws prompts
    # from. None only together with rollout_source, which replaces that leg
    # entirely; each of the two other combinations is refused in run().
    dataset: str | None = None
    # dr_grpo, not grpo. dr_grpo is reference-free, so the default run never
    # pays the memory for the frozen reference copy that grpo's k3 term
    # requires. The default should be the cheapest entry that trains end to
    # end; selecting "grpo" remains legal and now auto-loads that reference
    # (see reference_policy below).
    algorithm: str = "dr_grpo"
    # None = auto: a frozen reference copy of the initial policy is loaded
    # iff the resolved objective declares kl_weight != 0.0. False forbids the
    # load -- and _resolve_objective then refuses any objective that needs
    # the term, with the missing input named. True forces the load even for
    # a reference-free objective (declared, and therefore not silent waste).
    reference_policy: bool | None = None
    # Which checkpoint the reference copy is loaded from; None means the
    # policy's own initial weights, which is what makes the step-1 k3
    # contribution exactly zero.
    reference_model: str | None = None
    # The answer surface the prompt asked the policy for, as a regex with one
    # capture group, or None to treat every A--Z in the completion as a candidate.
    # Paired with gold_key: gold_key says where the TRUTH is, answer_pattern says
    # where the model's CLAIM is, and scoring needs both to be locatable.
    answer_pattern: str | None = None
    # Which record key holds the verifiable answer, or None to infer it from the
    # assistant turn. This is a property OF THE CORPUS, so it belongs in the run
    # declaration and not in the loader: #512, measured 2026-09-19, zero of 5,098
    # corpus files carry the English prompt marker the inference path requires, so
    # a plane that cannot be told where gold lives cannot score real data at all.
    gold_key: str | None = None
    group_size: int = 4
    learning_rate: float = 1e-6
    # None means "decide by measurement": use host-fp32 master weights whenever
    # the parameters are not fp32. At this lr a bf16 parameter's AdamW update is
    # below its ulp and is DISCARDED rather than attenuated, so 50 steps move the
    # same 0.98% of entries as 1 -- the loop runs and the model does not learn,
    # with every emitted signal (loss, grad-norm, throughput, changed checkpoint
    # bytes) looking healthy. True/False force the choice; forcing it is an
    # operator decision and is recorded either way.
    master_weights: bool | None = None
    max_steps: int = 10
    # Measured 2026-10-09 (gemma-4-12B-it, ScienceQA, terminal-pattern reward
    # "Answer: X"): this default truncates the model's own reasoning before it
    # reaches the answer line on effectively every rollout -- 0/20 sampled
    # completions reached a parseable answer at max_new_tokens=64, 5/20 at 200,
    # 17/20 at 400. A terminal-pattern reward needs enough budget for the
    # model to FINISH its chain of thought, not just name an answer; callers
    # using MCQLetterReward-shaped rewards should raise this explicitly rather
    # than rely on the default. Left at 64 rather than changed here because
    # not every reward shape needs long completions and the right budget is
    # reward- and model-specific -- silently raising the default would just
    # move the silent-default problem rather than remove it.
    max_new_tokens: int = 64
    # #370: sampling is DECLARED here, never inherited. Before this the trainer
    # called generate(do_sample=True) with no temperature/top_p/top_k, so the
    # sampling distribution came from the checkpoint's generation_config.json --
    # a file the training config never mentions. Group-relative objectives
    # (GRPO/GSPO/Dr.GRPO/DAPO) are DEFINED by within-group reward variance, and
    # temperature is the primary lever on it, so the trainer could not influence
    # the one quantity its objective family depends on. Measured consequence: an
    # end-to-end run refused every step with "advantage is identically zero over
    # 4 of 4 rows, distinct rewards: [1.0]".
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    prompts_per_step: int = 2
    # Caps how many rows each log-probability forward keeps resident. Zero
    # preserves the historical whole-batch scorer; a positive value first
    # prices the surrogate from one batch-sized detached leaf, then replays
    # row slices only to deliver that leaf's gradient to the parameters.
    # The loss, kept-row selection and metrics still describe one logical
    # batch, while full-vocabulary scorer activations are bounded per slice.
    logprob_micro_batch: int = 0
    seed: int = 0
    device: str | None = None
    # Data-parallel sharding: "none" (default, single process) / "ddp" /
    # "fsdp" (FSDP2 fully_shard). Defaults preserve the existing single-GPU
    # behaviour byte-for-byte. "none" refuses to run under torchrun rather
    # than silently training N replicas.
    sharding: str = "none"
    # None keeps the historical behaviour of never saving weights. When set,
    # checkpoints land at save_dir/step_N every save_every steps (save_every
    # of 0 means only the final save) and at save_dir/final at the end.
    save_dir: str | None = None
    save_every: int = 0
    gradient_checkpointing: bool = False
    # iterative DPO only: copy the policy into the frozen reference every N
    # steps (0 never refreshes). Online DPO keeps its initial reference.
    ref_refresh_steps: int = 0
    # reinforce_baseline only: what a FLAT group (every completion of one
    # prompt scored the same) contributes. "zero" (default) gives a flat group
    # advantage 0, as a per-prompt baseline would (NeMo-RL GRPO's leave-one-out
    # baseline and DAPO's trivial-group filter both make it 0). "keep" prices it
    # against the scalar EMA like any row, so a flat group at reward r gets
    # advantage r - baseline: a correct-everywhere prompt is pushed up and a
    # wrong-everywhere prompt pushed down with no within-prompt contrast. On a
    # 1000-step held-out MCQ comparison (Qwen2.5-7B, n=800) "keep" collapsed on
    # 4 of 5 seeds (0-48%) while "zero" held 63-65% on 5 of 5. The EMA still
    # folds in every raw return either way.
    reinforce_flat_groups: str = "zero"
    # None (default) keeps the built-in encode -> generate -> decode -> MCQ-score
    # leg untouched. Set to an object exposing ``rollout(step: int) ->
    # ExperienceBatch`` (and optionally ``publish(model, tokenizer, ctx, step) ->
    # None``, a weight-export hook) to have ``run()`` price externally generated
    # multi-turn rows instead: each step calls ``rollout_source.rollout(step)``,
    # shards the returned groups whole across ranks, and prices them through
    # ``_one_step_rows`` / ``_priced_tail`` -- the SAME kept-plane-onward pricing
    # the built-in leg uses. Refused (named) together with ppo, the
    # online-preference family and the estimator-free family in this slice: none
    # of those three price a group-relative advantage over externally supplied
    # rows yet.
    rollout_source: Any | None = None
    # Video as a DECLARED frame budget (foundationscale.video): 0 keeps video
    # refused; N >= 1 turns each clip into N centred-uniform frames, routed as
    # images. Frames are cached under video_cache_dir (default: a
    # ".fs_video_frames" directory beside the dataset).
    video_frames: int = 0
    video_sampling: str = "uniform"
    video_max_side: int | None = None
    video_cache_dir: str | None = None
    # Declared adapter mode. None means FULL FINE-TUNE, stated as data rather
    # than implied by the absence of peft wiring -- the SAME convention
    # train/loop.py's SFT plane uses. Every adapter_* knob is None by default,
    # and a partial specification (one set while adapter is None, or adapter
    # set without adapter_rank) is refused in RLTrainer.__init__.
    adapter: str | None = None
    adapter_rank: int | None = None
    adapter_alpha: float | None = None
    adapter_targets: tuple[str, ...] | None = None
    adapter_dropout: float | None = None


class RLTrainer:
    """One loop: sample, generate, score, price, step.

    WHAT IS CLAIMED: per step, ``prompts_per_step`` samples are taken,
    ``group_size`` completions are generated per prompt under sampling, every
    completion is scored with abstentions DROPPED (an abstaining row is
    unmeasured, never scored 0.0), old log-probabilities are recomputed under
    ``torch.no_grad()`` over exactly the surviving rows, current
    log-probabilities are recomputed WITH grad over those same rows, and one
    ``TensorPolicyLoss`` backward/optimiser step is taken.

    WHAT IS NOT CLAIMED: that generation is scheduled, that checkpoints are
    saved, or that a step is attempted when fewer usable rows survive than
    are needed -- such a step is skipped as UNMEASURED.
    """

    def __init__(self, config: RLTrainConfig) -> None:
        if isinstance(config.group_size, bool) or not isinstance(config.group_size, int):
            raise TrainerRefusal(f"group_size={config.group_size!r}: an int >= 2 is required")
        if config.group_size < 2:
            raise TrainerRefusal(
                f"group_size={config.group_size}: a group-relative objective needs at "
                f"least 2 samples per group; 1 of at least 2 supplied"
            )
        if config.reinforce_flat_groups not in ("keep", "zero"):
            raise TrainerRefusal(
                f"reinforce_flat_groups={config.reinforce_flat_groups!r}: one of "
                f"'keep' or 'zero' is required"
            )
        if config.max_steps < 1:
            raise TrainerRefusal(
                f"max_steps={config.max_steps}: 0 steps of at least 1 requested; an "
                f"empty run measures nothing and is refused as vacuous"
            )
        if config.prompts_per_step < 1:
            raise TrainerRefusal(
                f"prompts_per_step={config.prompts_per_step}: 0 prompts of at least 1 "
                f"required per step"
            )
        if config.logprob_micro_batch < 0:
            raise ValueError(
                f"logprob_micro_batch={config.logprob_micro_batch}: a negative row "
                "budget is not meaningful; use 0 for the whole batch"
            )
        if config.sharding not in ("none", "ddp", "fsdp"):
            raise TrainerRefusal(
                f"sharding={config.sharding!r}: one of 'none', 'ddp', 'fsdp' is required"
            )
        if config.save_every < 0:
            raise TrainerRefusal(
                f"save_every={config.save_every}: a negative interval is not "
                f"meaningful; use 0 for final-only saving"
            )
        if config.adapter is not None and config.adapter not in _ADAPTERS:
            raise TrainerRefusal(f"adapter={config.adapter!r} is not one of {_ADAPTERS}")
        if config.adapter is None:
            # A partial specification is a refusal, not a hint: every
            # adapter_* field with adapter unset is a statement about nothing.
            for field_name in (
                "adapter_rank",
                "adapter_alpha",
                "adapter_targets",
                "adapter_dropout",
            ):
                value = getattr(config, field_name)
                if value is not None:
                    raise TrainerRefusal(
                        f"{field_name}={value!r} is set while adapter is None: a "
                        f"partial adapter specification is refused. Set adapter to "
                        f"one of {_ADAPTERS}, or clear {field_name}"
                    )
        elif (
            config.adapter_rank is None
            or isinstance(config.adapter_rank, bool)
            or (not isinstance(config.adapter_rank, int) or int(config.adapter_rank) < 1)
        ):
            raise TrainerRefusal(
                f"adapter={config.adapter!r} requires adapter_rank to be a "
                f"positive int; got adapter_rank={config.adapter_rank!r}. A "
                "missing or non-positive rank silently defines the adapter's "
                "capacity, which is exactly the unrecorded-config failure"
            )
        if config.adapter_targets is not None:
            config.adapter_targets = tuple(config.adapter_targets)
            if not config.adapter_targets:
                # all([]) is True, and an adapter that targets nothing trains
                # nothing while looking like it trained.
                raise TrainerRefusal(
                    "adapter_targets=() is refused as vacuous: name at least "
                    "one target, or pass None to use peft's per-model defaults"
                )
        self.config = config
        # reinforce_baseline's carried EMA state: None means UNSEEDED (no
        # batch priced yet), never a baseline of 0.0. The tail seeds it from
        # the first measured mean return and updates it only AFTER the step
        # that priced against it. Other families never read this.
        self._reinforce_baseline: float | None = None

    def _resolve_objective(self) -> Any:
        """Build the algorithm's objective instance from the registry entry.

        WHAT IS CLAIMED: the returned object exposes the declared axes
        ``TensorPolicyLoss`` reads and an ``advantage_fn``.

        WHAT IS NOT CLAIMED: that the objective is tuned.
        """
        algorithm = lookup_algorithm(self.config.algorithm)
        ppo = ppo_objective(algorithm)
        if ppo is not None:
            if self.config.rollout_source is not None:
                raise TrainerRefusal(
                    f"rollout_source is set together with algorithm="
                    f"{self.config.algorithm!r}, which resolves to a PPO objective: "
                    f"ppo is refused together with rollout_source in this slice -- "
                    f"its advantage is temporal (GAE over a learned value), not the "
                    f"group-relative advantage_fn family rollout_source rows price"
                )
            # PPOAlgorithm carries no ``_objective``: its advantage is temporal
            # (GAE over a learned value), which ppo_step owns, not _one_step.
            return ppo
        # Registry factories (gspo_algorithm, dr_grpo_algorithm, dapo_algorithm,
        # the grpo entry) construct a SequencePolicyAlgorithm; its objective is
        # the single source of truth for ratio scope, clip bounds, reduction,
        # advantage estimator and KL weight.
        objective = getattr(algorithm, "_objective", None)
        if objective is None:
            # The grpo entry binds GRPOAlgorithm -- the torch-free objective
            # binding -- which carries no ``_objective`` by construction. That
            # absence is EXPECTED, so the refusal names the algorithm's own
            # declared reason (the k3 reference plane this one-model loop does
            # not hold) instead of reading as a broken factory. Measured hole:
            # before this arm, 'grpo' got the generic no-axes message.
            reference_declared = False
            requirements_fn = getattr(algorithm, "requirements", None)
            if callable(requirements_fn):
                try:
                    declared_requires = requirements_fn().requires
                except Exception:  # noqa: BLE001 -- metadata lookup must never mask the refusal
                    declared_requires = {}
                reference_declared = bool(declared_requires.get("reference_policy"))
            if reference_declared:
                raise TrainerRefusal(
                    f"algorithm {self.config.algorithm!r} declares "
                    f"requires.reference_policy=True: its k3 objective term prices "
                    f"log-ratios against a frozen copy of the initial policy, but "
                    f"the binding exposes no `_objective` instance, so the "
                    f"reference plane has nothing to feed. Bind the objective "
                    f"(as grpo does) to run it under the reference-policy path."
                )
            raise TrainerRefusal(
                f"algorithm {self.config.algorithm!r}: 0 of 1 required objective "
                f"instances expose the declared axes; the tensor kernel reads the "
                f"objective's declarations rather than a per-algorithm branch"
            )
        # Exposing an objective is NOT the same as exposing the axes this loop
        # reads, and the gap is the majority case. The preference family
        # (DPO/CPO/IPO/KTO/ORPO/SimPO) prices a PAIR and the online family
        # (RAFT/best-of-n/online-DPO/iterative-DPO) prices a RANKED set; neither
        # centres a reward against its group, so neither declares an
        # ``advantage_fn`` -- absent by design, not missing by oversight.
        # #513, measured 2026-09-19: 10 of the 14 registry entries that expose
        # objective have none, while ``_one_step`` reads
        # ``objective.advantage_fn.compute(...)`` unconditionally. Naming one of
        # them therefore raised AttributeError several frames deeper, after the
        # model was loaded and a full group had been generated -- a crash where
        # the contract owes a refusal that names the missing input, and one that
        # arrives only after the expensive part of the step has been paid for.
        if is_online_pref(objective):
            if self.config.rollout_source is not None:
                raise TrainerRefusal(
                    f"rollout_source is set together with algorithm="
                    f"{self.config.algorithm!r}, which resolves to an online-"
                    f"preference objective: the online-preference family is "
                    f"refused together with rollout_source in this slice -- it "
                    f"prices a PAIR mined from the group's own rewards, not the "
                    f"group-relative advantage_fn family rollout_source rows price"
                )
            # Online/iterative DPO price a PAIR mined from the group's own
            # rewards; online_pref_step owns that loop, not _one_step.
            return objective
        # reinforce_baseline and reinforce_pp are the two estimator-free
        # tails (design section 5): one subtracts a carried EMA baseline,
        # the other folds a k1 penalty into the return and normalises
        # globally. Neither declares advantage_fn, by design and not by
        # oversight, so the estimator refusal below is not for them. The SFT
        # pair (raft, best_of_n) is estimator-free too: its loss is masked NLL
        # over the argmax-reward winners of each prompt group (_sft_tail).
        estimator_free = self.config.algorithm in (
            "reinforce_baseline",
            "reinforce_pp",
            "raft",
            "best_of_n",
        )
        if estimator_free and self.config.rollout_source is not None:
            raise TrainerRefusal(
                f"rollout_source is set together with algorithm="
                f"{self.config.algorithm!r}: the estimator-free family "
                f"(reinforce_baseline, reinforce_pp, raft, best_of_n) is refused "
                f"together with rollout_source in this slice -- none of them price "
                f"a group-relative advantage_fn over externally supplied rows"
            )
        if not estimator_free and not hasattr(objective, "advantage_fn"):
            raise TrainerRefusal(
                f"algorithm {self.config.algorithm!r}: 0 of 1 required advantage "
                f"estimators are declared by its objective "
                f"({type(objective).__name__}); this loop prices group-centred "
                f"rewards, so a pairwise or rank-based objective has no estimator "
                f"for it to read. The group-relative family runs today; the "
                f"preference and online families are not wired to this loop."
            )
        # An active k3 term needs a frozen reference copy, and so does
        # reinforce_pp's k1 fold (its declared kl_weight stays 0.0 because
        # the penalty folds into the return, so the axis does not see it).
        # run() now loads that copy (auto unless reference_policy says
        # otherwise); what survives here as a refusal is the DECLARED
        # refusal to build one: reference_policy=False against an objective
        # that needs the plane.
        kl_weight = float(getattr(objective, "kl_weight", 0.0))
        needs_reference = kl_weight != 0.0 or self.config.algorithm == "reinforce_pp"
        if needs_reference and self.config.reference_policy is False:
            raise TrainerRefusal(
                f"algorithm {self.config.algorithm!r} declares kl_weight={kl_weight}; "
                f"reference_policy=False forbids the 1 of 1 reference policies "
                f"the k3 term or k1 fold requires, so the term cannot be "
                f"measured. Set reference_policy to None (auto) or True to "
                f"load one."
            )
        return objective

    def run(self) -> list[StepReport]:
        """Run the training loop and return one report per completed step."""

        # The EMA baseline is per-run state: a second run() on the same trainer
        # must seed from its own first batch, not inherit the previous run's value.
        self._reinforce_baseline = None

        # #370: refuse BEFORE any allocation is burned. Greedy decoding makes
        # every completion in a group byte-identical, so their rewards are equal,
        # so the group-relative advantage is identically zero and no gradient
        # exists -- for the whole GRPO/GSPO/Dr.GRPO/DAPO family this is not a
        # bad hyperparameter, it is a configuration in which training cannot
        # happen. Refusing is louder than emitting N identical rows and letting
        # the advantage estimator abstain once per step forever.
        if self.config.group_size > 1 and self.config.temperature <= 0.0:
            _refuse_exit_96(
                f"temperature={self.config.temperature} with "
                f"group_size={self.config.group_size}: greedy decoding yields "
                "identical completions, so within-group reward variance is zero "
                "by construction and a group-relative objective has no gradient "
                "to take. Raise the temperature or set group_size=1."
            )

        if self.config.rollout_source is not None and self.config.dataset is not None:
            _refuse_exit_96(
                f"dataset={self.config.dataset!r} together with rollout_source: the "
                f"rollout source replaces the corpus draw entirely, so the dataset "
                f"would be declared and never read. Drop one of the two."
            )
        if self.config.rollout_source is None and self.config.dataset is None:
            _refuse_exit_96(
                "dataset=None without a rollout_source: the built-in leg draws its "
                "prompts from the corpus, and there is none to draw from"
            )
        samples: tuple[Sample, ...] | list[Sample] = (
            ()
            if self.config.dataset is None
            else load_sharegpt(self.config.dataset, gold_key=self.config.gold_key)
        )
        # #371: corpus.py PARSES `image` and `video` into Sample.images/.video,
        # and its docstring advertises "text-only, image-text, multi-image, and
        # video" records. This trainer references neither field: it builds every
        # prompt from prompt_turns alone through a TOKENIZER, never a processor,
        # and `pixel_values` appears nowhere in this file. So a vision record
        # used to load cleanly, pass every validation, and train on the TEXT
        # ALONE -- the pixels dropped between loader and model with no error and
        # no UNMEASURED line, while the run looked completely healthy.
        #
        # Until the processor path lands, that silent drop becomes a loud
        # refusal. Naming the sample and the modality matters: "some records
        # have images" is not actionable, and a count alone would let a single
        # stray record look like a corpus-wide problem.
        # NARROWED once the processor path landed. IMAGES are now routed: an
        # image-carrying batch resolves an AutoProcessor (prompt_surface), whose
        # extra keys -- pixel_values and friends -- are group-expanded and
        # forwarded to the scorer alongside input_ids. Measured on gemma-4-E4B:
        # 258.0 prompt tokens per image, exactly linear at n = 1, 2, 4, 8.
        #
        # VIDEO used to be refused outright: nothing here sampled frames, and the
        # missing pieces (frame count, spacing, resolution) are DATASET decisions
        # with no safe default. They are now a declared FrameBudget
        # (foundationscale.video): with video_frames >= 1 each clip becomes that
        # many centred-uniform frames, cached on disk and routed as IMAGES through
        # the processor path above. With video_frames == 0 video still refuses,
        # naming the samples and the knob. Against the measured 131,072-token
        # context, 258 tok/frame puts the ceiling near 508 frames -- ample, which
        # is precisely why the budget is chosen by the caller, not inherited.
        samples = _expand_video_samples(samples, self.config)
        try:
            import torch
        except ImportError:
            _refuse_exit_96(
                "1 of 2 required dependencies absent: torch; the tensor plane "
                "needs it and no pure-python fall-back exists for weight updates"
            )
        try:
            import transformers  # noqa: F401 -- probe only; _load_causal_lm imports its own names
        except ImportError:
            _refuse_exit_96(
                "1 of 2 required dependencies absent: transformers; models are "
                "loaded through Auto classes and templates through "
                "apply_chat_template, never by this package"
            )

        from foundationscale.rl.distributed import (
            destroy,
            init_distributed,
            is_main,
            save_checkpoint,
            shard_indices,
            wrap_ddp,
            wrap_fsdp2,
        )

        torch.manual_seed(self.config.seed)
        # Process group comes up BEFORE any device movement; "none" under
        # torchrun is refused inside init_distributed, never replicated.
        ctx = init_distributed(self.config.sharding)
        if self.config.prompts_per_step < ctx.world_size:
            # shard_indices would hand every rank ZERO prompts and each step
            # would abstain forever; refuse identically on every rank instead.
            destroy(ctx)
            raise TrainerRefusal(
                f"prompts_per_step={self.config.prompts_per_step} < world_size="
                f"{ctx.world_size}: sharding is over prompts, so some rank would "
                "hold none; raise prompts_per_step to at least the world size"
            )
        device = self.config.device
        if ctx.is_distributed:
            device = str(ctx.device)
        if device is None:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )

        try:
            # #371: resolve the prompt SURFACE, not just a tokenizer. A batch
            # carrying images needs an AutoProcessor -- it is the only thing
            # that produces pixel_values -- and prompt_surface REFUSES (96)
            # rather than downgrading to a tokenizer, because that downgrade
            # is exactly the silent drop this work removed.
            #
            # `tokenizer` stays bound for the 16 existing call sites
            # (pad_token_id, batch_decode, apply_chat_template): a processor
            # carries its own .tokenizer, so both paths expose the same
            # surface and no call site had to change to gain the capability.
            needs_images = any(sample.images for sample in samples)
            prompt_surface = resolve_prompt_surface(self.config.model, needs_images)
            tokenizer = (
                prompt_surface.surface
                if prompt_surface.kind == "tokenizer"
                else prompt_surface.surface.tokenizer
            )
        except Exception as exc:  # noqa: BLE001 -- load surface failure is a refusal
            _refuse_exit_96(f"tokenizer load failed for {self.config.model!r}: {exc}")

        model = _load_causal_lm(self.config.model, needs_images=needs_images)
        # adapter_notes is a dict a manifest-keeping caller could consume;
        # this loop has no manifest of its own (unlike train/loop.py's SFT
        # plane) -- _apply_lora_adapter already PRINTS the same facts to
        # stderr, which is this loop's existing "record the config" surface
        # (see the optimizer= print a few lines down).
        model, _adapter_notes = _apply_lora_adapter(
            model,
            adapter=self.config.adapter,
            adapter_rank=self.config.adapter_rank,
            adapter_alpha=self.config.adapter_alpha,
            adapter_targets=self.config.adapter_targets,
            adapter_dropout=self.config.adapter_dropout,
            log_prefix="[trainer]",
        )
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if self.config.adapter is not None:
                # peft freezes every base-model parameter, so the first
                # (embedding) activation in the checkpointed chain carries
                # requires_grad=False and torch.utils.checkpoint has nothing
                # to build a backward graph through. This hooks the input
                # embedding's output to require grad regardless -- the
                # standard peft + gradient-checkpointing pairing.
                model.enable_input_require_grads()
            # Training forwards must not cache; generate() is called with
            # use_cache=True independently below.
            model.config.use_cache = False
        if self.config.sharding == "fsdp":
            # Load on CPU and shard from there: fully_shard materialises the
            # per-rank DTensor shards from these tensors, so pre-moving one
            # full copy to the GPU would only double the peak. wrap_fsdp2
            # casts to fp32 first -- the sharded params ARE the fp32 masters.
            model = wrap_fsdp2(model, ctx)
            model.train()
        elif self.config.sharding == "ddp":
            model.to(device)
            model = wrap_ddp(model, ctx)
            model.train()
        else:
            model.to(device)
            model.train()

        if not getattr(tokenizer, "chat_template", None):
            _refuse_exit_96(
                f"tokenizer for {self.config.model!r} has no chat template: "
                f"prompt construction REFUSES to concatenate strings silently; "
                f"set tokenizer.chat_template explicitly"
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # #546: pad on the LEFT before any prompt is encoded for
        # generation. Right padding is the tokenizer default, and with it
        # one batched generate() call over prompts of different lengths
        # puts pad tokens BETWEEN a shorter prompt and the continuation it
        # appends -- transformers itself warns "right-padding was
        # detected" on this call -- so the model attends across pad slots
        # mid-sequence and every real token after them is conditioned on
        # pads. Left padding makes the pads a prefix instead. The things
        # the code below actually relies on are unchanged -- verified by
        # reading it, not assumed:
        #   * completions are generated[:, prompt_width:]; generate()
        #     returns rows of width prompt_width + max_new_tokens with the
        #     prompt (pads included, on whichever side) occupying the FIRST
        #     prompt_width columns either way, so the slice boundary moves
        #     only with the width, never with the side;
        #   * the attention mask in _one_step is built from pad ids on the
        #     kept sequences, so left pads are excluded from attention
        #     exactly as right pads were;
        #   * response_mask only writes into columns >= prompt_width - 1, a
        #     region no left pad can reach;
        #   * forward_logprobs derives no positions itself -- it hands the
        #     model input_ids plus the full attention mask, and the no-grad
        #     old pass and the graph-carrying current pass see the SAME
        #     padded batch, so old and current readings stay conditioned
        #     identically and the importance ratio compares like with
        #     like. What left padding fixes is the conditioning INSIDE
        #     generate(), which is the pass that was wrong.
        # On the processor surface the trainer's `tokenizer` IS the
        # processor's tokenizer, but the attribute is set on both spellings
        # so the invariant survives a future surface that separates them;
        # the second set is free when they are one object.
        tokenizer.padding_side = "left"
        inner_tokenizer = getattr(prompt_surface.surface, "tokenizer", None)
        if inner_tokenizer is not None:
            inner_tokenizer.padding_side = "left"

        objective = self._resolve_objective()
        kl_weight = float(getattr(objective, "kl_weight", 0.0))
        # reinforce_pp joins the auto rule despite declaring kl_weight 0.0:
        # its k1 fold reads the reference plane in the trainer tail, where
        # the kernel's reference axis cannot see it.
        needs_reference = kl_weight != 0.0 or self.config.algorithm == "reinforce_pp"
        ref_model: Any = None
        online_pref = is_online_pref(objective)
        refresh_every = refresh_cadence(objective, self.config.ref_refresh_steps)
        if needs_reference or self.config.reference_policy or online_pref:
            try:
                plan = _reference_plan(
                    adapter=self.config.adapter,
                    reference_model=self.config.reference_model,
                    model=self.config.model,
                    refresh_every=refresh_every,
                )
            except TrainerRefusal as exc:
                _refuse_exit_96(str(exc))
            if plan == "disable_adapter":
                ref_model = _AdapterDisabledReference(model)
                print(
                    "[trainer] reference: adapter='lora', no distinct "
                    "reference_model -- reusing the policy with the adapter "
                    "disabled instead of loading a second model copy",
                    file=sys.stderr,
                )
            else:
                # The frozen reference plane: loaded before any optimizer step
                # so it IS the initial policy -- which is what makes the
                # step-1 k3 contribution exactly zero. Only an objective
                # declaring a non-zero kl_weight (or an operator forcing
                # reference_policy=True) pays this memory; _resolve_objective
                # already refused the needs-one-but-forbidden combination.
                ref_model = _load_causal_lm(
                    self.config.reference_model or self.config.model, needs_images=needs_images
                )
                if self.config.sharding == "fsdp":
                    # Sharded too: a frozen full-size replica on each rank
                    # would rescale memory exactly the way fsdp exists to
                    # prevent. DDP keeps it plain -- no gradient averaging is
                    # wanted over a frozen model, so no DDP wrapper.
                    ref_model = wrap_fsdp2(ref_model, ctx)
                else:
                    ref_model.to(device)
                ref_model.eval()
                for parameter in ref_model.parameters():
                    parameter.requires_grad_(False)
        reward = MCQLetterReward(answer_pattern=self.config.answer_pattern)
        use_ppo = is_ppo(objective)
        loss_fn = None if online_pref or use_ppo else TensorPolicyLoss(objective=objective)
        # #369: bf16 params stepped directly by AdamW at lr=1e-6 discard every
        # sub-ulp update, so the loop trains ~nothing while loss, grad-norm,
        # throughput and changed checkpoint bytes all look healthy. Selection is
        # EXPLICIT and printed, because a silent choice here IS the defect.
        param_dtype = next(model.parameters()).dtype
        if self.config.sharding == "fsdp":
            # wrap_fsdp2 already cast the sharded DTensors to fp32 and
            # MixedPrecisionPolicy handles the bf16 compute cast: the
            # masters ARE the parameters. MasterWeightOptimizer's host-side
            # copies would break the DTensor plane, so plain AdamW over the
            # sharded fp32 params is used -- the optimiser still steps fp32
            # state against fp32 params, which is the whole point of
            # mastering.
            use_masters = False
            master_reason = "fsdp: sharded params are fp32 masters via MixedPrecisionPolicy"
        elif self.config.master_weights is not None:
            use_masters = self.config.master_weights
            master_reason = f"forced by config master_weights={self.config.master_weights}"
        else:
            use_masters = param_dtype != torch.float32
            master_reason = (
                f"auto: params are {param_dtype}, whose ulp exceeds the update at this lr"
                if use_masters
                else "auto: params already fp32, no mastering needed"
            )
        optimizer: Any
        if use_masters:
            optimizer = MasterWeightOptimizer(
                _trainable_parameters(model), lr=self.config.learning_rate
            )
        else:
            optimizer = torch.optim.AdamW(  # noqa: B014
                _trainable_parameters(model), lr=self.config.learning_rate
            )
        print(
            "[trainer] optimizer="
            + ("MasterWeightOptimizer(host-fp32)" if use_masters else "AdamW(direct)")
            + f" param_dtype={param_dtype} lr={self.config.learning_rate}"
            + f" reason={master_reason}",
            file=sys.stderr,
        )

        value_head: Any = None
        value_optimizer: Any = None
        if use_ppo:
            # Replicated on every rank (not FSDP-wrapped); ppo_step averages
            # its grads explicitly so the replicas never drift.
            value_head = build_value_head(model, device)
            value_optimizer = torch.optim.AdamW(
                value_head.parameters(), lr=self.config.learning_rate
            )

        usable = tuple(sample for sample in samples if sample.gold is not None)
        if not usable and self.config.rollout_source is None:
            _refuse_exit_96(
                f"0 of {len(samples)} loaded samples carry a parseable gold letter; "
                f"a run with no verifiable row is vacuous"
            )

        reports: list[StepReport] = []
        cursor = 0
        for step in range(self.config.max_steps):
            if self.config.rollout_source is not None:
                # The additive rows lane: an externally generated
                # ExperienceBatch replaces the corpus draw AND the per-step
                # encode -> generate -> decode -> score leg entirely.
                # _resolve_objective already refused ppo / online-preference /
                # estimator-free algorithms together with rollout_source, so
                # loss_fn here is always the group-relative TensorPolicyLoss.
                assert loss_fn is not None
                batch = self.config.rollout_source.rollout(step)
                group_key_column = batch.column("prompt_ids")
                unique_groups = sorted(set(group_key_column))
                if ctx.is_distributed:
                    # Shard whole GROUPS, never rows within one: the same
                    # "a group's baseline stays on one rank" invariant the
                    # built-in leg keeps by sharding prompts instead of
                    # completions.
                    my_group_positions = shard_indices(len(unique_groups), ctx)
                    my_groups = {unique_groups[i] for i in my_group_positions}
                    if not my_groups and unique_groups:
                        # More ranks than groups this step: replicate one
                        # group so this rank still participates and every
                        # collective fires.
                        my_groups = {unique_groups[0]}
                else:
                    my_groups = set(unique_groups)
                row_indices = [i for i, key in enumerate(group_key_column) if key in my_groups]
                sharded_batch = ExperienceBatch(
                    columns={
                        name: tuple(column[i] for i in row_indices)
                        for name, column in batch.columns.items()
                    },
                    required=batch.required,
                )
                report = self._one_step_rows(
                    step,
                    sharded_batch,
                    model=model,
                    optimizer=optimizer,
                    ref_model=ref_model,
                    objective=objective,
                    loss_fn=loss_fn,
                    device=device,
                    pad_token_id=tokenizer.pad_token_id,
                    ctx=ctx,
                )
                if report is not None:
                    publish = getattr(self.config.rollout_source, "publish", None)
                    if publish is not None:
                        # S0: synchronous, one rollout per optimizer step --
                        # the next iteration's rollout(step + 1) is free to
                        # read whatever this publish just exported.
                        publish(model, tokenizer, ctx, step)
            else:
                global_chunk = [
                    usable[(cursor + offset) % len(usable)]
                    for offset in range(self.config.prompts_per_step)
                ]
                cursor += self.config.prompts_per_step
                if ctx.is_distributed:
                    # Shard PROMPTS, never completions: all G completions of one
                    # prompt stay on one rank so the group baseline stays local.
                    my_indices = shard_indices(len(global_chunk), ctx)
                    chunk = [global_chunk[i] for i in my_indices]
                    if not chunk:
                        # More ranks than prompts this step: replicate one prompt
                        # so generation still returns rows and collectives fire.
                        chunk = [global_chunk[0]]
                else:
                    chunk = global_chunk
                if use_ppo:
                    report = ppo_step(
                        self,
                        step=step,
                        chunk=chunk,
                        model=model,
                        tokenizer=tokenizer,
                        surface=prompt_surface,
                        reward=reward,
                        objective=objective,
                        optimizer=optimizer,
                        ref_model=ref_model,
                        device=device,
                        ctx=ctx,
                        value_head=value_head,
                        value_optimizer=value_optimizer,
                    )
                elif online_pref:
                    report = online_pref_step(
                        self,
                        step=step,
                        chunk=chunk,
                        model=model,
                        tokenizer=tokenizer,
                        surface=prompt_surface,
                        reward=reward,
                        objective=objective,
                        optimizer=optimizer,
                        ref_model=ref_model,
                        device=device,
                        ctx=ctx,
                    )
                    if maybe_refresh_reference(
                        step=step, cadence=refresh_every, model=model, ref_model=ref_model
                    ):
                        print(
                            f"[trainer] reference refreshed from policy after step {step}",
                            file=sys.stderr,
                        )
                else:
                    assert loss_fn is not None
                    report = self._one_step(
                        step=step,
                        chunk=chunk,
                        model=model,
                        tokenizer=tokenizer,
                        surface=prompt_surface,
                        reward=reward,
                        objective=objective,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        ref_model=ref_model,
                        device=device,
                        ctx=ctx,
                    )
            if report is not None:
                reports.append(report)
            if (
                self.config.save_dir is not None
                and self.config.save_every > 0
                and (step + 1) % self.config.save_every == 0
            ):
                save_checkpoint(
                    model,
                    tokenizer,
                    f"{self.config.save_dir}/step_{step + 1}",
                    ctx,
                    sharding=self.config.sharding,
                    step=step + 1,
                )
        _refuse_vacuous_run(attempted=self.config.max_steps, measured=len(reports))
        if self.config.save_dir is not None:
            save_checkpoint(
                model,
                tokenizer,
                f"{self.config.save_dir}/final",
                ctx,
                sharding=self.config.sharding,
                step=self.config.max_steps,
            )
            if value_head is not None and is_main(ctx):
                # The critic is replicated, so rank 0's copy is the whole of it.
                torch.save(value_head.state_dict(), f"{self.config.save_dir}/final/value_head.pt")
        destroy(ctx)
        return reports

    def _one_step(
        self,
        *,
        step: int,
        chunk: list[Sample],
        model: Any,
        tokenizer: Any,
        surface: Any,
        reward: MCQLetterReward,
        objective: Any,
        loss_fn: TensorPolicyLoss,
        optimizer: Any,
        ref_model: Any | None,
        device: str,
        ctx: Any = None,
    ) -> StepReport | None:
        """One step over surviving rows, or ``None`` when none survive.

        WHAT IS CLAIMED: used < offered is the normal state; a step whose
        every row abstains is UNMEASURED and skipped, never reported as a
        zero-row step.

        WHAT IS NOT CLAIMED: that any particular step produces a report.
        """
        from foundationscale.rl.distributed import DistContext, agree_all, generate_kwargs_for

        if ctx is None:
            ctx = DistContext(
                rank=0, world_size=1, local_rank=0, device=device, is_distributed=False
            )
        import torch

        golds: list[str | None] = [sample.gold for sample in chunk]

        # #371: encode through the SURFACE, not the bare tokenizer. For a
        # text-only chunk this is the same tokenizer call as before; for a
        # chunk carrying images it is the processor, which is the only thing
        # that emits pixel_values. Everything downstream is unchanged --
        # prompt_width still comes from the encoded tensor, and the extra
        # modality keys are group-expanded and forwarded to the scorer below.
        prompt_ids = encode_prompts(surface, chunk, device)
        with torch.no_grad(), _generation_mode(model) as gen_model:
            # Under DDP the generative path bypasses the wrapper (no_grad --
            # no grad sharing is wanted during rollout); under fsdp the
            # wrapper IS the module generate must run on. _generation_mode
            # unwraps .module either way and restores eval/use_cache state
            # on return -- see its docstring for the measured reason this
            # is not optional under gradient_checkpointing.
            generated = gen_model.generate(
                **prompt_ids,
                max_new_tokens=self.config.max_new_tokens,
                num_return_sequences=self.config.group_size,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                # top_k=0 disables the cutoff; generate() wants the sentinel, and
                # passing None would silently restore the checkpoint's value --
                # the very inheritance #370 removed.
                top_k=self.config.top_k,
                pad_token_id=tokenizer.pad_token_id,
                # synced_gpus under fsdp: generate must keep stepping while
                # ANY rank is still generating, or an early-EOS rank would
                # deadlock the next all-gather.
                **generate_kwargs_for(ctx, self.config.sharding),
            )
        prompt_width = prompt_ids["input_ids"].shape[1]
        completions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)

        # score, then drop abstentions row by row.
        rows: list[tuple[int, float]] = []
        for index, completion in enumerate(completions):
            # generate(num_return_sequences=G) returns rows prompt-major:
            # [p0g0 .. p0gG-1, p1g0 ..], so the gold of row i is golds[i // G].
            scored = reward.score(response=completion, gold=golds[index // self.config.group_size])
            if scored is not None:
                rows.append((index, scored))
        # A NULL rank continues down the identical code path with one dummy
        # row and a loss multiplied by 0.0, so it issues exactly the same
        # sequence of collectives as a rank with real rows. Returning early
        # after one private forward/backward/step deadlocked FSDP: the other
        # ranks went on to agree_max, the logprob forwards, three more votes,
        # the sliced backward and the report reductions. MEASURED on 4x GB200.
        # A null rank votes True at every later "skip?" vote.
        null_rank = False
        every_rank_empty = agree_all(not rows, ctx)
        if not rows and not every_rank_empty:
            null_rank = True
            rows = [(0, 0.0)]
            print(
                f"[trainer] step {step} rank {ctx.rank}: "
                "no row survived reward scoring on this rank; others did"
                " -- participating with zero loss (UNMEASURED here)",
                file=sys.stderr,
            )
        if not rows:
            # offered > used == 0: an entirely abstaining step is UNMEASURED.
            print(
                f"UNMEASURED step {step}: 0 of {len(completions)} completion(s) "
                f"earned a score; every rollout abstained, so the step carries no "
                f"reward at all. No gradient exists to take, so no step is claimed.",
                file=sys.stderr,
            )
            return None

        kept_indices = torch.tensor([index for index, _ in rows], device=device)
        rewards = torch.tensor([score for _, score in rows], dtype=torch.float32, device=device)
        kept_sequences = generated.index_select(0, kept_indices)

        target_ids = kept_sequences[:, 1:]
        # Two DIFFERENT widths, and conflating them is silent. The attention
        # mask the model receives must span the full input width; the
        # supervision mask is defined over the SHIFTED targets, one column
        # narrower. Passing the shifted mask as attention_mask hands the
        # model a mask that disagrees with input_ids -- either a hard shape
        # error or, worse, a one-position attention shift.
        attention = (kept_sequences != tokenizer.pad_token_id).long()
        shifted_attention = attention[:, 1:]
        response_mask = torch.zeros_like(target_ids, dtype=torch.float32)
        response_mask[:, prompt_width - 1 :] = shifted_attention[:, prompt_width - 1 :]

        # #371: on a vision model the sequence is GENERATED conditioned on the
        # image, so it must be SCORED conditioned on the same image. Passing
        # only input_ids here would score an image-free conditional against
        # image-conditioned tokens -- the importance ratio would then compare
        # two different distributions, and it would stay finite and plausible
        # the whole way. Measured on gemma-4-E4B: one image expands the prompt
        # 16 -> 273 tokens and the processor emits pixel_values,
        # mm_token_type_ids and image_position_ids alongside input_ids.
        #
        # Every non-text key the surface produced is forwarded verbatim. It is
        # built once, outside the closure, so BOTH the no_grad old-logprob pass
        # and the graph-carrying current pass see exactly the same conditioning
        # -- recomputing it per call would let them drift.
        _TEXT_KEYS = {"input_ids", "attention_mask"}
        modality_kwargs = {
            key: value
            for key, value in prompt_ids.items()
            if key not in _TEXT_KEYS and hasattr(value, "index_select")
        }
        if modality_kwargs:
            # generate() expanded each prompt into group_size rows; most
            # modality tensors are still one row per PROMPT, so they are
            # repeated to match and then narrowed to the kept rows, in that
            # order -- doing it the other way round selects against the wrong
            # axis silently. A flattened-patch key (Qwen-VL family
            # pixel_values) is NOT one row per prompt, though -- see
            # _expand_modality_kwargs_for_group's docstring for the measured
            # defect this avoids.
            group = self.config.group_size
            modality_kwargs = _expand_modality_kwargs_for_group(
                modality_kwargs, group=group, kept_indices=kept_indices
            )
            # MEASURED: per-TOKEN keys (mm_token_type_ids,
            # image_position_ids) are still PROMPT width here; the scorer
            # below is called over kept_sequences, prompt_width +
            # max_new_tokens wide. Extending them keeps every per-token
            # modality tensor the same width the model actually scores --
            # per-IMAGE keys (pixel_values, image_grid_thw) are untouched by
            # this call, see its docstring for the shape test that tells them
            # apart.
            modality_kwargs = _align_modality_keys_to_scored_width(
                modality_kwargs,
                prompt_width=prompt_width,
                sequence_width=kept_sequences.shape[1],
            )
            print(
                "[trainer] forwarding modality keys to the scorer: "
                + ", ".join(sorted(modality_kwargs)),
                file=sys.stderr,
            )

        group_ids = [f"row-{index // self.config.group_size}" for index, _ in rows]
        return self._priced_tail(
            step=step,
            model=model,
            optimizer=optimizer,
            ref_model=ref_model,
            device=device,
            ctx=ctx,
            objective=objective,
            loss_fn=loss_fn,
            kept_sequences=kept_sequences,
            response_mask=response_mask,
            attention=attention,
            modality_kwargs=modality_kwargs,
            rewards=rewards,
            group_ids=group_ids,
            null_rank=null_rank,
            rows=rows,
        )

    def _priced_tail(
        self,
        *,
        step: int,
        model: Any,
        optimizer: Any,
        ref_model: Any | None,
        device: str,
        ctx: Any,
        objective: Any,
        loss_fn: TensorPolicyLoss,
        kept_sequences: torch.Tensor,
        response_mask: torch.Tensor,
        attention: torch.Tensor,
        modality_kwargs: dict[str, torch.Tensor],
        rewards: torch.Tensor,
        group_ids: list[str],
        null_rank: bool,
        rows: list[tuple[int, float]] | None = None,
    ) -> StepReport | None:
        """Price the kept planes: logprobs, advantage, loss, backward, report.

        WHAT IS CLAIMED: this is EVERY optimizer step this module takes from
        "the kept rows are known" onward, shared VERBATIM by the built-in
        encode -> generate -> decode -> score leg (``_one_step``) and the
        externally generated multi-turn rows leg (``_one_step_rows``):
        ``old_logprobs`` are recomputed under ``torch.no_grad()`` on exactly
        ``kept_sequences`` regardless of where those rows came from, and
        ``group_ids`` is the ONLY group-identity input the advantage estimator
        reads -- a caller that synthesizes ``f"row-{index // group_size}"``
        (the built-in leg) and a caller that reads a batch's own
        ``prompt_ids`` column (the rows leg) are indistinguishable from here
        on. ``rows`` -- the pre-compaction ``(index, score)`` pairs -- is read
        only by the SFT and REINFORCE-family tails, which ``_resolve_objective``
        already refuses together with ``rollout_source``, so the rows leg
        never needs to supply it; every other branch counts offered rows off
        ``kept_sequences`` itself.

        WHAT IS NOT CLAIMED: that any particular step produces a report, or
        that a caller may reach the SFT / REINFORCE branches without ``rows``.
        """
        import torch

        from foundationscale.rl.distributed import agree_all, agree_max, all_reduce_sum

        target_ids = kept_sequences[:, 1:]
        n_rows = int(kept_sequences.shape[0])
        micro_batch = self.config.logprob_micro_batch
        use_logprob_micro_batching = 0 < micro_batch < n_rows
        row_slices: tuple[tuple[int, int], ...] = (
            tuple(
                (start, min(start + micro_batch, n_rows)) for start in range(0, n_rows, micro_batch)
            )
            if use_logprob_micro_batching
            else ((0, n_rows),)
        )
        # The micro-slice COUNT must agree across ranks: FSDP's collectives
        # fire once per sliced forward/backward, so a rank with fewer real
        # slices pads with zero-weight dummies below.
        extra_slices = 0
        if ctx.is_distributed:
            extra_slices = max(0, agree_max(len(row_slices), ctx) - len(row_slices))
            # The PATH must agree too, not just the count: the sliced path
            # forwards each slice three times (old, current, backward replay)
            # and the whole-batch path twice (its backward reuses the live
            # graph). MEASURED on 2x GB200: a rank with 2 kept rows beside a
            # rank with 8 hung in the first FSDP all-gather after the split.
            # So if any rank slices, every rank slices; its whole batch is
            # then one slice.
            use_logprob_micro_batching = not agree_all(not use_logprob_micro_batching, ctx)
        # Padding slices sit INSIDE each pass, so every rank issues the same
        # model sequence (policy, policy, reference) slice by slice.
        pad_slice = (0, min(1, n_rows))
        passes = row_slices + (pad_slice,) * extra_slices

        def forward_logprob_slice(
            start: int, end: int, *, scorer_model: Any = None
        ) -> torch.Tensor:
            # Every tensor with a per-row leading dimension follows the same
            # half-open row range: the generated ids, their full-width
            # attention mask, the shifted targets and every per-row modality
            # tensor. A flattened-patch modality tensor (Qwen-VL family
            # pixel_values) is NOT per-row, though -- see
            # _narrow_modality_kwargs_by_row's docstring for the measured
            # defect this avoids (a micro-batch slice truncating raw patches
            # instead of selecting the sliced rows' own patch blocks).
            width = end - start
            sliced_modalities = _narrow_modality_kwargs_by_row(
                modality_kwargs, start=start, end=end
            )
            # The scorer defaults to the policy; the frozen reference is the
            # only other caller, and every per-row tensor still follows the
            # same half-open row range.
            logits = (model if scorer_model is None else scorer_model)(
                input_ids=kept_sequences.narrow(0, start, width),
                attention_mask=attention.narrow(0, start, width),
                **sliced_modalities,
            ).logits
            return _token_logprobs(logits, target_ids.narrow(0, start, width))

        if isinstance(objective, (RAFTLoss, BestOfNLoss)):
            # The SFT pair has no advantage estimator, no old plane and no
            # reference plane: route to masked-NLL-on-winners BEFORE any of
            # those three computations are paid for. Generation, scoring,
            # abstention dropping and the mask construction above are shared
            # with the PPO-clip tail.
            if rows is None:
                raise TrainerRefusal(
                    f"algorithm {self.config.algorithm!r} resolves to the SFT "
                    f"tail, which groups by the pre-compaction (index, score) "
                    f"pairs this caller did not supply; rollout_source callers "
                    f"refuse this algorithm family in _resolve_objective before "
                    f"reaching here"
                )
            return self._sft_tail(
                step=step,
                objective=objective,
                rows=rows,
                response_mask=response_mask,
                forward_slice=forward_logprob_slice,
                row_slices=row_slices,
                use_logprob_micro_batching=use_logprob_micro_batching,
                optimizer=optimizer,
                ctx=ctx,
                null_rank=null_rank,
                extra_slices=extra_slices,
            )

        # Old logprobs are RECOMPUTED under no_grad over the same rows -- the
        # generation scores are never reused: their shapes differ and the bug
        # is silent. When micro-batching, current logprobs are also read
        # under no_grad and become ONE detached leaf. The loss and metrics
        # below therefore see the same batch-shaped tensors as the
        # whole-batch path; only the way parameter gradients are delivered
        # changes.
        real = len(row_slices)
        if use_logprob_micro_batching:
            with torch.no_grad():
                old_logprobs = torch.cat(
                    [forward_logprob_slice(start, end) for start, end in passes][:real],
                    dim=0,
                )
                current_logprobs = (
                    torch.cat(
                        [forward_logprob_slice(start, end) for start, end in passes][:real],
                        dim=0,
                    )
                    .detach()
                    .requires_grad_(True)
                )
        else:
            # A zero budget, and a budget spanning every row, retain the
            # historical single current forward as one live graph.
            with torch.no_grad():
                old_logprobs = forward_logprob_slice(0, n_rows)
            current_logprobs = forward_logprob_slice(0, n_rows).requires_grad_(True)

        # The reference plane: same kept rows, same modality conditioning,
        # same row slices as the policy planes, under no_grad. Computed only
        # when a reference is loaded, which run() guarantees iff the
        # objective's k3 term needs it (or the operator forced the load).
        reference_logprobs: torch.Tensor | None = None
        if ref_model is not None:
            with torch.no_grad():
                reference_logprobs = torch.cat(
                    [
                        forward_logprob_slice(start, end, scorer_model=ref_model)
                        for start, end in passes
                    ][:real],
                    dim=0,
                )

        # Family dispatch (design section 5): the two estimator-free
        # bindings price straight off these planes and never call
        # advantage_fn -- one subtracts a carried EMA baseline, the other
        # folds a k1 penalty into the return and normalises globally. Every
        # kept scored row is used, so the row gather is the identity on the
        # kept planes, never a compaction.
        if self.config.algorithm in ("reinforce_baseline", "reinforce_pp"):
            if rows is None:
                raise TrainerRefusal(
                    f"algorithm {self.config.algorithm!r} resolves to the "
                    f"estimator-free REINFORCE tail, which needs the pre-"
                    f"compaction (index, score) pairs this caller did not "
                    f"supply; rollout_source callers refuse this algorithm "
                    f"family in _resolve_objective before reaching here"
                )
            keep_all = torch.arange(n_rows, device=device)
            tail_current = current_logprobs.index_select(0, keep_all)
            tail_old = old_logprobs.index_select(0, keep_all).detach()
            tail_mask = response_mask.index_select(0, keep_all).detach()
            tail_ref: torch.Tensor | None = None
            if reference_logprobs is not None:
                tail_ref = reference_logprobs.index_select(0, keep_all)
            tail_scores = [float(score) for _, score in rows]
            if self.config.algorithm == "reinforce_baseline":
                return self._reinforce_baseline_tail(
                    step=step,
                    scores=tail_scores,
                    group_keys=[index // self.config.group_size for index, _ in rows],
                    kept_current=tail_current,
                    kept_mask=tail_mask,
                    current_logprobs=current_logprobs,
                    row_slices=row_slices,
                    forward_slice=forward_logprob_slice,
                    use_logprob_micro_batching=use_logprob_micro_batching,
                    objective=objective,
                    optimizer=optimizer,
                    ctx=ctx,
                    null_rank=null_rank,
                    extra_slices=extra_slices,
                )
            return self._reinforce_pp_tail(
                step=step,
                scores=tail_scores,
                kept_current=tail_current,
                kept_old=tail_old,
                kept_mask=tail_mask,
                kept_ref=tail_ref,
                current_logprobs=current_logprobs,
                row_slices=row_slices,
                forward_slice=forward_logprob_slice,
                use_logprob_micro_batching=use_logprob_micro_batching,
                objective=objective,
                loss_fn=loss_fn,
                optimizer=optimizer,
                ctx=ctx,
                null_rank=null_rank,
                extra_slices=extra_slices,
            )

        # The estimator reads the PER-TOKEN supervision mask, not a per-row
        # flag: it denominates each response by its own supervised length.
        # Handing it a 1-D tensor of ones made every row non-iterable and
        # refused the batch, and the row count it would have implied is not
        # the quantity the estimator needs. ``group_ids`` is the caller's
        # group-identity input -- synthesized by the built-in leg, read off
        # the batch's own column by the rows leg -- never recomputed here.
        advantage: Any = None
        adv_refusal: str | None = None
        try:
            if not null_rank:
                advantage = objective.advantage_fn.compute(
                    prompt_ids=tuple(group_ids),
                    rewards=tuple(float(value) for value in rewards.tolist()),
                    mask=tuple(tuple(int(e) for e in row) for row in response_mask.tolist()),
                )
        except AdvantageRefusal as exc:
            adv_refusal = str(exc)
        # A rank must never skip alone: agree, then act identically. If SOME
        # ranks have a usable advantage, this rank participates with a
        # zero-weight dummy so FSDP's collectives stay aligned.
        every_rank_refused = agree_all(null_rank or adv_refusal is not None, ctx)
        if adv_refusal is not None and not every_rank_refused and not null_rank:
            null_rank = True
            print(
                f"[trainer] step {step} rank {ctx.rank}: "
                "advantage refused on this rank; others have a usable advantage"
                " -- participating with zero loss (UNMEASURED here)",
                file=sys.stderr,
            )
        if every_rank_refused:
            if null_rank:
                print(
                    f"[trainer] step {step}: every rank is null; step UNMEASURED",
                    file=sys.stderr,
                )
                return None
            # The estimator refuses when NO row survives: every group was too
            # small for a baseline once abstentions were dropped. That is a
            # genuine UNMEASURED step, not a crash. Letting the refusal
            # propagate ended an entire multi-step run on one unlucky draw --
            # and a run that dies at step 4 of 8 reports nothing about steps
            # 5 to 8, which is a worse outcome than saying "this step taught
            # nothing" and continuing.
            print(
                f"UNMEASURED step {step}: the advantage estimator used 0 of "
                f"{n_rows} offered row(s) -- {adv_refusal}",
                file=sys.stderr,
            )
            return None
        # AdvantageResult is a RECORD, not a per-row sequence: `rows` names the
        # batch indices it used and `weights` carries one per-token weight row
        # for each. Enumerating the record itself treated its fields as
        # advantages. `used < offered` is the estimator's own visible account
        # of what it dropped -- a group too small for a baseline leaves here.
        kept_rows = [0] if null_rank else list(advantage.rows)
        every_rank_kept_none = agree_all(null_rank or not kept_rows, ctx)
        if not kept_rows and not every_rank_kept_none:
            null_rank = True
            kept_rows = [0]
            print(
                f"[trainer] step {step} rank {ctx.rank}: "
                "the advantage kept no row on this rank; others kept rows"
                " -- participating with zero loss (UNMEASURED here)",
                file=sys.stderr,
            )
        if every_rank_kept_none:
            if null_rank:
                print(
                    f"[trainer] step {step}: every rank is null; step UNMEASURED",
                    file=sys.stderr,
                )
                return None
            print(
                f"UNMEASURED step {step}: the advantage estimator kept 0 of "
                f"{n_rows} scored row(s); no group was large enough to admit a "
                f"baseline. No gradient exists to take, so no step is claimed.",
                file=sys.stderr,
            )
            return None
        keep = torch.tensor(kept_rows, device=device)
        if null_rank:
            advantage_tensor = torch.zeros(
                (1, int(response_mask.shape[1])), dtype=torch.float32, device=device
            )
        else:
            advantage_tensor = torch.tensor(
                [list(weights) for weights in advantage.weights],
                dtype=torch.float32,
                device=device,
            ).detach()

        # A SATURATED step is unmeasured, not a step. When every completion in
        # a group earns the same reward -- all correct or all wrong -- the
        # group-relative advantage is identically zero, the surrogate is zero,
        # and the gradient is zero. The optimiser then "steps" without moving,
        # and the report would carry loss=-0.0 over a healthy-looking row
        # count: a run that taught nothing, reported as a run that trained.
        # That is this framework's founding failure wearing new clothes, so it
        # is named and skipped rather than counted.
        zero_advantage = not bool(advantage_tensor.abs().any())
        every_rank_zero = agree_all(null_rank or zero_advantage, ctx)
        if zero_advantage and not every_rank_zero and not null_rank:
            null_rank = True
            print(
                f"[trainer] step {step} rank {ctx.rank}: "
                "every advantage is zero on this rank; others have signal"
                " -- participating with zero loss (UNMEASURED here)",
                file=sys.stderr,
            )
        if every_rank_zero:
            if null_rank:
                print(
                    f"[trainer] step {step}: every rank is null; step UNMEASURED",
                    file=sys.stderr,
                )
                return None
            # Reported PER GROUP, because the group is the axis the baseline is
            # formed over and the claim is about variance WITHIN it. Pooling the
            # rewards across groups printed evidence that contradicted the
            # sentence carrying it: a step whose first group scored all 0.0 and
            # whose second scored all 1.0 has no within-group variance anywhere,
            # yet announced "no within-group variance (distinct rewards:
            # [0.0, 1.0])". A reader is entitled to conclude from that line that
            # the diagnosis is wrong, and a diagnostic nobody can trust is worse
            # than no diagnostic. MEASURED on GB200, step 17 of a 20-step run.
            summary = _per_group_reward_summary(
                kept_rows, group_ids, [float(value) for value in rewards.tolist()]
            )
            print(
                f"UNMEASURED step {step}: advantage is identically zero over "
                f"{len(kept_rows)} of {n_rows} used row(s); no group's reward "
                f"varies within that group (rewards per group -- {summary}). "
                f"No gradient exists to take, so no step is claimed.",
                file=sys.stderr,
            )
            return None

        # #546: bind the kept tensors ONCE. The step-observability metrics
        # emitted below must be measured off the exact tensors the kernel
        # prices -- gathering a second old/current pair for reporting would
        # let the report describe a gradient it never took.
        kept_current = current_logprobs.index_select(0, keep)
        kept_old = old_logprobs.index_select(0, keep).detach()
        kept_mask = response_mask.index_select(0, keep).detach()
        kept_ref: torch.Tensor | None = None
        if reference_logprobs is not None:
            kept_ref = reference_logprobs.index_select(0, keep)
        # prompt_mean's denominator is PER GROUP and is built from group ids,
        # which the tensor kernel never receives. So the row weights are
        # computed ONCE per optimizer step, here -- after row compaction and
        # before the loss call -- from the KEPT rows' group ids and the kept
        # mask's supervised counts, and ride unchanged through
        # _micro_batched_backward and through the padding slices (which
        # carry no loss). The denominator is supplied by
        # prompt_mean_row_weights, the same torch-free function the oracle
        # uses, so this plane cannot disagree with it on shape.
        #
        # P is STEP-GLOBAL. Prompts -- never completions -- are sharded whole,
        # so groups are rank-local and the active-group count must be
        # all-reduce-summed before the division. Under DDP gradient averaging
        # (and FSDP2, which averages identically) the weights are then scaled
        # by world_size so the GLOBAL objective is the prompt mean over ALL
        # ranks' groups: the wrapper divides the averaged gradient by
        # world_size, and w_row = world_size / (P_global * group_tokens)
        # renormalises it back onto the global denominator. A null rank
        # contributes 0 groups and 0 weight.
        reduction_row_weights: Any = None
        if getattr(objective, "reduction", None) == "prompt_mean":
            counts = [int(value) for value in kept_mask.sum(dim=-1).tolist()]
            kept_group_ids = [group_ids[row] for row in kept_rows]
            local_groups = (
                0
                if null_rank
                else len(
                    {key for key, count in zip(kept_group_ids, counts, strict=True) if count > 0}
                )
            )
            p_global = all_reduce_sum(float(local_groups), ctx)
            if p_global == 0.0:
                # Saturated/unmeasured step is not a step, in the same voice
                # as the zero-advantage path below: no active group means no
                # measured denominator, and 0.0 would report a perfect loss
                # over no gradient.
                print(
                    f"UNMEASURED step {step}: 0 prompt group(s) carry a "
                    f"supervised token over {len(kept_rows)} kept row(s) "
                    f"across every rank; the prompt_mean denominator is "
                    f"UNMEASURED and never 0.0. No gradient exists to take, "
                    f"so no step is claimed.",
                    file=sys.stderr,
                )
                return None
            if local_groups == 0:
                weights: tuple[float, ...] = (0.0,) * len(kept_rows)
            else:
                base = prompt_mean_row_weights(kept_group_ids, counts)
                scale = float(ctx.world_size) * float(local_groups) / p_global
                weights = tuple(value * scale for value in base)
            # The kernel's seam is a (rows,) tensor on the loss's device; the
            # torch-free owner produced plain floats. float64 so a 1/6 is not
            # rounded here -- the kernel casts once, to its own dtype.
            import torch

            reduction_row_weights = torch.tensor(
                weights, dtype=torch.float64, device=kept_current.device
            )
        loss_tensor = loss_fn(
            current_logprobs=kept_current,
            old_logprobs=kept_old,
            advantages=advantage_tensor,
            mask=kept_mask,
            reference_logprobs=kept_ref,
            reduction_row_weights=reduction_row_weights,
        )
        if null_rank:
            # Mandatory, not cosmetic: with a zero advantage the surrogate is
            # zero but the KL/reference terms are not.
            loss_tensor = loss_tensor * 0.0
        optimizer.zero_grad()
        if use_logprob_micro_batching:
            # zero_grad precedes both backward phases. The first fills the
            # detached leaf; each slice then re-creates only its own model
            # graph, consumes its matching leaf gradient and frees the graph.
            # step() is still issued once, after every slice contribution.
            _micro_batched_backward(
                loss_tensor=loss_tensor,
                current_logprobs=current_logprobs,
                row_slices=row_slices,
                forward_slice=forward_logprob_slice,
            )
        else:
            loss_tensor.backward()
        for _ in range(extra_slices):
            # Padding backward: one collective-matching backward whose
            # gradient is exactly zero, so the busiest rank's extra slices
            # do not deadlock the reduce-scatter.
            lane = forward_logprob_slice(0, min(1, n_rows))
            (lane.sum() * 0.0).backward()
        optimizer.step()

        # Averaged scalars are reduced MEAN, counts SUM, so every rank holds
        # the same report; only rank 0's caller-visible stream prints.
        # A null rank's loss is a zeroed dummy, so it is excluded from the
        # mean by weight rather than averaged in as a spurious 0.0: the
        # reported loss is the row-weighted mean over the ranks that measured.
        local_loss = float(loss_tensor.detach())
        weight = 0.0 if null_rank else float(len(kept_rows))
        measured_weight = all_reduce_sum(weight, ctx)
        measured = (
            all_reduce_sum(local_loss * weight, ctx) / measured_weight
            if ctx.is_distributed
            else local_loss
        )
        loss_output = LossOutput(
            loss=measured,
            components=_loss_components(
                objective=objective,
                total=measured,
                current_logprobs=kept_current,
                reference_logprobs=kept_ref,
                mask=kept_mask,
            ),
            # #546: the step-1 invariant (ratio == 1.0, clip fraction ==
            # 0.0, because old and current are read off the same weights) is
            # carried as OBSERVED metrics on the LossOutput -- never as new
            # StepReport fields, whose contract forbids derived quantities
            # and second copies; an unmeasurable entry stays absent, never
            # 0.0.
            metrics=_ratio_and_clip_metrics(
                objective=objective,
                current_logprobs=kept_current,
                old_logprobs=kept_old,
                mask=kept_mask,
            ),
        )
        # Reward telemetry over the rows the gradient ACTUALLY touched, not
        # over everything offered. Leaving this None made the one number that
        # explains a step invisible: a saturated batch and a healthy one
        # produce the same-looking report, and on the first step -- where the
        # ratio is exactly 1 because old and current come from the same
        # weights -- the surrogate is the mean of centred advantages and reads
        # ~0 even when the gradient is large. Without the reward spread there
        # is no way to tell those two apart from the report alone.
        return StepReport(
            step=step,
            loss=loss_output,
            # rows stays LOCAL so it matches the local reward_stats count
            # (StepReport refuses two denominators); a null rank has no local
            # rows and zero is refused, so it names the peers' measured rows.
            rows=int(measured_weight) if null_rank else len(kept_rows),
            reward_stats=(
                None
                if null_rank
                else RewardStats.over(tuple(float(rewards[row]) for row in kept_rows))
            ),
            sync=None,
        )

    def _planes_from_batch(
        self, batch: ExperienceBatch, *, device: str, pad_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int]]:
        """Build the kept planes from an externally generated ``ExperienceBatch``.

        WHAT IS CLAIMED: reads exactly the columns ``prompt_token_ids``,
        ``response_ids``, ``loss_mask``, ``prompt_ids`` and ``reward``; a row
        with ``reward is None`` is DROPPED as an abstention (never priced as
        0.0), the same rule the built-in MCQ leg applies, and the drop count
        is printed. Each kept row's sequence is
        ``prompt_token_ids + response_ids``, RIGHT-padded by LENGTH to the
        batch's longest kept sequence with ``pad_token_id``; ``attention`` is
        built from those same per-row LENGTHS, never by scanning for a pad
        id (a tool token or an EOS-as-pad token must not be mistaken for
        padding). ``response_mask[i, t]`` for target position ``t``
        (predicting ``kept_sequences[i, t + 1]``) is the response token's own
        ``loss_mask`` entry at response position ``t + 1 - len(prompt_i)``
        when that position is inside the response, else 0 -- so a tool-result
        token recorded with ``loss_mask`` 0 stays unsupervised wherever it
        lands, unlike the built-in leg's single completion-only boundary.
        Refuses (``TrainerRefusal``, naming both sides) when a kept row's
        ``loss_mask`` length disagrees with its ``response_ids`` length, or
        when every remaining row supervises 0 response tokens in total.

        WHAT IS NOT CLAIMED: that any row survived -- an all-abstention batch
        returns zero-row planes rather than refusing, so the caller can apply
        the same null-rank voice the built-in leg uses when every local row
        abstains while a peer rank still has rows.
        """
        import torch

        prompt_token_ids_col = batch.column("prompt_token_ids")
        response_ids_col = batch.column("response_ids")
        loss_mask_col = batch.column("loss_mask")
        prompt_ids_col = batch.column("prompt_ids")
        reward_col = batch.column("reward")

        sequences: list[tuple[int, ...]] = []
        prompt_lengths: list[int] = []
        response_lengths: list[int] = []
        row_loss_masks: list[tuple[int, ...]] = []
        group_ids: list[str] = []
        rewards_list: list[float] = []
        kept_row_indices: list[int] = []
        dropped = 0

        for row in range(len(batch)):
            reward_value = reward_col[row]
            if reward_value is None:
                # Abstention: dropped, never priced as 0.0 -- the same rule
                # the built-in MCQ leg applies to a scorer that declined.
                dropped += 1
                continue
            response = tuple(response_ids_col[row])
            mask = tuple(loss_mask_col[row])
            if len(mask) != len(response):
                raise TrainerRefusal(
                    f"_planes_from_batch: row {row} (prompt_ids="
                    f"{prompt_ids_col[row]!r}): loss_mask has {len(mask)} "
                    f"entries for a response of {len(response)} token(s) -- a "
                    f"per-token mask must carry exactly one entry per "
                    f"response token"
                )
            prompt = tuple(prompt_token_ids_col[row])
            sequences.append(prompt + response)
            prompt_lengths.append(len(prompt))
            response_lengths.append(len(response))
            row_loss_masks.append(mask)
            group_ids.append(str(prompt_ids_col[row]))
            rewards_list.append(float(reward_value))
            kept_row_indices.append(row)

        if dropped:
            print(
                f"[trainer] _planes_from_batch: dropped {dropped} of "
                f"{len(batch)} row(s) to abstention (reward=None)",
                file=sys.stderr,
            )

        if not sequences:
            # Every row abstained: an empty result, not a refusal -- the
            # caller (_one_step_rows) owes the same null-rank/UNMEASURED
            # voice the built-in leg uses when every local row abstains
            # while a peer rank still has rows.
            empty_sequences = torch.empty((0, 1), dtype=torch.long, device=device)
            empty_mask = torch.empty((0, 0), dtype=torch.float32, device=device)
            empty_rewards = torch.empty((0,), dtype=torch.float32, device=device)
            return empty_sequences, empty_mask, empty_sequences.clone(), empty_rewards, [], []

        max_len = max(len(seq) for seq in sequences)
        n_rows = len(sequences)
        kept_sequences = torch.full(
            (n_rows, max_len), pad_token_id, dtype=torch.long, device=device
        )
        attention = torch.zeros((n_rows, max_len), dtype=torch.long, device=device)
        response_mask = torch.zeros((n_rows, max_len - 1), dtype=torch.float32, device=device)
        total_supervised = 0
        for i, seq in enumerate(sequences):
            length = len(seq)
            kept_sequences[i, :length] = torch.tensor(seq, dtype=torch.long, device=device)
            attention[i, :length] = 1
            prompt_len = prompt_lengths[i]
            response_len = response_lengths[i]
            mask_i = row_loss_masks[i]
            # Target position t predicts kept_sequences[i, t + 1]; that
            # position is a response token iff prompt_len <= t + 1 <
            # prompt_len + response_len, i.e. t in [prompt_len - 1,
            # prompt_len + response_len - 1). Clamped into [0, max_len - 1)
            # and re-based onto the response's own indices so a 0-length
            # prompt (no column ever predicts the response's own first
            # token, same as the built-in shift) is handled without a
            # negative slice.
            lo = max(prompt_len - 1, 0)
            hi = min(prompt_len + response_len - 1, max_len - 1)
            if hi > lo:
                resp_lo = lo + 1 - prompt_len
                row_values = torch.tensor(
                    mask_i[resp_lo : resp_lo + (hi - lo)], dtype=torch.float32, device=device
                )
                response_mask[i, lo:hi] = row_values
                total_supervised += int(row_values.sum())

        if total_supervised == 0:
            raise TrainerRefusal(
                f"_planes_from_batch: {n_rows} of {len(batch)} row(s) survived "
                f"abstention-dropping but supervise 0 response token(s) in "
                f"total; a step with no supervised token has no gradient to "
                f"take and is refused rather than priced as a zero-loss step"
            )

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        return kept_sequences, response_mask, attention, rewards, group_ids, kept_row_indices

    def _one_step_rows(
        self,
        step: int,
        batch: ExperienceBatch,
        *,
        model: Any,
        optimizer: Any,
        ref_model: Any | None,
        objective: Any,
        loss_fn: TensorPolicyLoss,
        device: str,
        pad_token_id: int,
        ctx: Any = None,
    ) -> StepReport | None:
        """One step over an externally generated multi-turn ``ExperienceBatch``.

        WHAT IS CLAIMED: ``_planes_from_batch`` builds the kept planes and
        ``_priced_tail`` prices them -- the SAME pricing the built-in
        encode -> generate -> decode -> score leg uses from the kept rows
        onward. Old logprobs are recomputed under ``torch.no_grad()`` inside
        ``_priced_tail``; the batch's own ``rollout_logprobs`` column (if any)
        is never read here or passed to it. A rank left with 0 kept rows
        while a peer rank still has rows participates with a zero-weight
        dummy row and votes UNMEASURED, in the same spirit as the built-in
        leg's null-rank dummy -- this lane has no ``generated`` tensor to
        borrow a row from, so the dummy is built fresh. A step where EVERY
        rank's batch is empty after abstention-dropping is UNMEASURED and
        returns ``None``, naming the same voice the built-in leg uses.

        WHAT IS NOT CLAIMED: that any particular step produces a report, or
        that a malformed row (mask/response length mismatch, or an
        all-zero-supervision batch) is tolerated -- ``_planes_from_batch``
        refuses those loudly rather than returning here.
        """
        import torch

        from foundationscale.rl.distributed import DistContext, agree_all

        if ctx is None:
            ctx = DistContext(
                rank=0, world_size=1, local_rank=0, device=device, is_distributed=False
            )

        # The 6th element (kept_row_indices) maps surviving rows back onto the
        # original batch; nothing in this tail needs that map, so it is read
        # here only to keep the tuple contract honest, under a `_`-prefixed
        # name ruff's unused-local check treats as deliberate.
        kept_sequences, response_mask, attention, rewards, group_ids, _kept_row_indices = (
            self._planes_from_batch(batch, device=device, pad_token_id=pad_token_id)
        )
        n_rows = int(kept_sequences.shape[0])
        null_rank = False
        every_rank_empty = agree_all(n_rows == 0, ctx)
        if n_rows == 0 and not every_rank_empty:
            null_rank = True
            # No `generated` tensor exists on this lane to borrow row 0 from
            # the way the built-in leg's null-rank dummy does: build the
            # smallest participating row instead. Width 2 keeps target_ids
            # non-empty, attention 1 everywhere keeps the forward well
            # defined, and an all-zero response_mask supervises nothing even
            # before null_rank zeroes the loss below.
            kept_sequences = torch.full((1, 2), pad_token_id, dtype=torch.long, device=device)
            attention = torch.ones((1, 2), dtype=torch.long, device=device)
            response_mask = torch.zeros((1, 1), dtype=torch.float32, device=device)
            rewards = torch.zeros((1,), dtype=torch.float32, device=device)
            group_ids = ["null-rank-dummy"]
            print(
                f"[trainer] step {step} rank {ctx.rank}: "
                "0 row(s) survived _planes_from_batch on this rank; others "
                "kept rows -- participating with zero loss (UNMEASURED here)",
                file=sys.stderr,
            )
        if n_rows == 0 and every_rank_empty:
            print(
                f"UNMEASURED step {step}: 0 of {len(batch)} row(s) survived "
                f"_planes_from_batch on every rank; every rollout abstained, "
                f"so the step carries no reward at all. No gradient exists to "
                f"take, so no step is claimed.",
                file=sys.stderr,
            )
            return None

        return self._priced_tail(
            step=step,
            model=model,
            optimizer=optimizer,
            ref_model=ref_model,
            device=device,
            ctx=ctx,
            objective=objective,
            loss_fn=loss_fn,
            kept_sequences=kept_sequences,
            response_mask=response_mask,
            attention=attention,
            modality_kwargs={},
            rewards=rewards,
            group_ids=group_ids,
            null_rank=null_rank,
            rows=None,
        )

    def _sft_tail(
        self,
        *,
        step: int,
        objective: Any,
        rows: list[tuple[int, float]],
        response_mask: torch.Tensor,
        forward_slice: Callable[[int, int], torch.Tensor],
        row_slices: tuple[tuple[int, int], ...],
        use_logprob_micro_batching: bool,
        optimizer: Any,
        ctx: Any,
        null_rank: bool,
        extra_slices: int,
    ) -> StepReport | None:
        """Masked-NLL tail for the SFT pair (RAFT, best-of-N).

        WHAT IS CLAIMED: the scored rows are grouped by prompt id; each group
        contributes its argmax-reward members -- and the loss is
        :class:`TensorMaskedSFTLoss` over those winners. A TIED maximum is
        neither broken nor dropped: every row at the maximum is a winner,
        because a first-row tiebreak would smuggle row order into a reward
        measurement, and dropping the group would discard nearly every group
        under a binary reward (any group with two correct rows ties). A group
        whose rows ALL share one reward carries no ranking and is dropped,
        printed with the count -- the SFT analogue of a zero-advantage group.
        The gradient travels the same ``forward_slice`` /
        ``row_slices`` machinery as the PPO-clip tail, so
        ``logprob_micro_batch`` slices this tail identically. The report's
        ``rows`` counts winner rows -- at least one per priced group -- and the loss
        carries the objective's own declared metrics where recoverable (the
        NLL mean, and for best-of-N the winner reward mean).

        WHAT IS NOT CLAIMED: that any group produced a winner. When every
        group is flat, the step is UNMEASURED with the reason named, never priced
        as a zero-loss step. No old-logprob plane and no reference plane are
        computed: masked NLL has no ratio to anchor, and neither objective
        declares a KL term, so ``reference_policy=None`` loads no frozen copy
        here.

        Under data parallelism "no winner" is voted, never acted on alone: a
        rank without winners (or with no rows at all) still issues the same
        forwards, backwards and reductions as its peers, at zero loss. Only
        when EVERY rank is without winners is the step UNMEASURED.
        """
        import torch

        from foundationscale.rl.distributed import agree_all

        # Grouped by POSITION, not batch index: `rows` pairs a batch index
        # with a score, and the log-probability / mask planes are already
        # narrowed to the scored rows, so the grouping keys must address those
        # planes positionally.
        groups: dict[str, list[int]] = {}
        for position, (index, _score) in enumerate(rows):
            prompt_id = f"row-{index // self.config.group_size}"
            groups.setdefault(prompt_id, []).append(position)
        winner_positions: list[int] = []
        flat: list[tuple[str, float]] = []
        tied_groups = 0
        for prompt_id, positions in groups.items():
            scores = [rows[position][1] for position in positions]
            best = max(scores)
            if min(scores) == best:
                flat.append((prompt_id, best))
                continue
            leaders = [position for position in positions if rows[position][1] == best]
            tied_groups += len(leaders) > 1
            winner_positions.extend(leaders)
        if flat and not null_rank:
            shown = "; ".join(
                f"{prompt_id}: all rows at reward {best}"
                for prompt_id, best in flat[:_MAX_GROUPS_REPORTED]
            )
            if len(flat) > _MAX_GROUPS_REPORTED:
                shown += f"; (+{len(flat) - _MAX_GROUPS_REPORTED} more group(s))"
            print(
                f"[trainer] step {step}: {len(flat)} of {len(groups)} group(s) "
                f"dropped from the SFT tail; every row shares one reward, so "
                f"there is no ranking to fine-tune on ({shown})",
                file=sys.stderr,
            )
        if tied_groups and not null_rank:
            print(
                f"[trainer] step {step}: {tied_groups} of {len(groups)} group(s) "
                f"have a tied maximum; every tied leader is a winner "
                f"({len(winner_positions)} winner row(s) in total)",
                file=sys.stderr,
            )
        no_winners = null_rank or not winner_positions
        if agree_all(no_winners, ctx):
            print(
                f"UNMEASURED step {step}: every group is flat ({len(flat)} of "
                f"{len(groups)} group(s) dropped); the SFT tail selected 0 "
                f"winners, no gradient exists to take and no step is claimed.",
                file=sys.stderr,
            )
            return None
        if no_winners:
            print(
                f"[trainer] step {step}: this rank selected 0 winners but a peer "
                f"did; participating with zero loss (UNMEASURED here)",
                file=sys.stderr,
            )
            # A placeholder winner keeps the kernel's shapes; its loss is
            # zeroed below, so the gradient is exactly zero.
            winner_positions = [0]

        winners = torch.tensor(winner_positions, device=response_mask.device)
        winner_mask = response_mask.index_select(0, winners).detach()
        n_rows = int(response_mask.shape[0])
        if use_logprob_micro_batching:
            # The same two-pass scheme as the PPO-clip tail: price one
            # detached batch-shaped leaf, then deliver its gradient through
            # fresh row-sliced graphs. The only swap is the kernel.
            with torch.no_grad():
                # Padding slices ride inside the pass so every rank issues
                # the busiest rank's forward count; their rows are dropped.
                passes = row_slices + ((0, min(1, n_rows)),) * extra_slices
                sliced = [forward_slice(start, end) for start, end in passes]
                current_full = (
                    torch.cat(sliced[: len(row_slices)], dim=0).detach().requires_grad_(True)
                )
        else:
            current_full = forward_slice(0, n_rows).requires_grad_(True)
        winner_current = current_full.index_select(0, winners)
        loss_tensor = TensorMaskedSFTLoss(objective=objective)(
            current_logprobs=winner_current,
            mask=winner_mask,
        )
        if no_winners:
            loss_tensor = loss_tensor * 0.0
        optimizer.zero_grad()
        if use_logprob_micro_batching:
            _micro_batched_backward(
                loss_tensor=loss_tensor,
                current_logprobs=current_full,
                row_slices=row_slices,
                forward_slice=forward_slice,
            )
        else:
            loss_tensor.backward()
        _padding_backwards(forward_slice=forward_slice, extra_slices=extra_slices, n_rows=n_rows)
        optimizer.step()

        from foundationscale.gates.objective_gates import MetricObservation

        measured, measured_weight = _dp_weighted_loss(
            float(loss_tensor.detach()),
            0.0 if no_winners else float(len(winner_positions)),
            ctx,
        )
        weight = float(getattr(objective, "weight", 1.0))
        metrics: list[Any] = []
        nll_metric_name = getattr(objective, "nll_metric_name", None)
        if isinstance(nll_metric_name, str) and nll_metric_name:
            # loss == weight * mean per-row NLL by construction of the
            # kernel, so the objective's declared NLL metric is recoverable
            # from the measured scalar -- a derivation, not a second pass.
            metrics.append(MetricObservation(name=nll_metric_name, value=measured / weight))
        reward_metric_name = getattr(objective, "reward_metric_name", None)
        if isinstance(reward_metric_name, str) and reward_metric_name and not no_winners:
            winner_rewards = [rows[position][1] for position in winner_positions]
            metrics.append(
                MetricObservation(
                    name=reward_metric_name,
                    value=sum(winner_rewards) / len(winner_rewards),
                )
            )
        loss_output = LossOutput(
            loss=measured,
            components=_loss_components(
                objective=objective,
                total=measured,
                current_logprobs=winner_current,
                reference_logprobs=None,
                mask=winner_mask,
            ),
            metrics=tuple(metrics),
        )
        return StepReport(
            step=step,
            loss=loss_output,
            # A rank without winners names the peers' measured rows, as the
            # main path does for a null rank (zero rows is refused).
            rows=int(measured_weight) if no_winners else len(winner_positions),
            # The SFT bindings declare advantage_fn False, and verify_step
            # grades the reward_stats <-> advantage_fn pairing both ways, so
            # this stays None rather than a pooled summary over survivors.
            reward_stats=None,
            sync=None,
        )

    def _reinforce_baseline_tail(
        self,
        *,
        step: int,
        scores: list[float],
        group_keys: list[int],
        kept_current: torch.Tensor,
        kept_mask: torch.Tensor,
        current_logprobs: torch.Tensor,
        row_slices: tuple[tuple[int, int], ...],
        forward_slice: Callable[[int, int], torch.Tensor],
        use_logprob_micro_batching: bool,
        objective: Any,
        optimizer: Any,
        ctx: Any,
        null_rank: bool,
        extra_slices: int,
    ) -> StepReport | None:
        """REINFORCE tail: subtract the carried EMA baseline from each return.

        With ``reinforce_flat_groups="zero"`` a group whose kept returns are
        all equal gets advantage 0 instead (counted from the kept rows, so a
        group reduced to one row by abstention is flat too), and the count of
        zeroed groups is printed every step.

        WHAT IS CLAIMED: the baseline subtracted is the carried state, or
        the batch's own mean return on the unseeded first step -- reported
        through the metric channel, never hidden; the EMA update happens
        only AFTER the optimiser step it priced; ``rows`` counts every kept
        row, because nothing compacts; and ``reward_stats`` abstains,
        because this binding declares ``advantage_fn: False`` and the
        pairing is graded both ways.

        Under data parallelism the batch mean is the GLOBAL mean over every
        rank's rows (a null rank contributes none), so the carried baseline
        stays identical on every rank; and "zero surrogate" is voted, never
        acted on alone -- a rank whose advantages are all zero still issues
        its peers' forwards, backwards and reductions, at zero loss.

        WHAT IS NOT CLAIMED: that the EMA tracks any optimum.
        """
        import torch

        from foundationscale.gates.objective_gates import MetricObservation
        from foundationscale.rl.distributed import agree_all, all_reduce_sum

        if not hasattr(objective, "baseline_momentum"):
            # No silent 0.99: the EMA rate is a declared field of the objective.
            raise TrainerRefusal(
                f"reinforce_baseline: objective {type(objective).__name__} declares no "
                "baseline_momentum; refusing to substitute a default EMA rate"
            )
        momentum = float(objective.baseline_momentum)
        # A null rank's scores are a placeholder, so it contributes nothing.
        local_n = 0.0 if null_rank else float(len(scores))
        total_n = all_reduce_sum(local_n, ctx)
        batch_mean = all_reduce_sum(0.0 if null_rank else sum(scores), ctx) / total_n
        used_baseline = self._reinforce_baseline
        if used_baseline is None:
            # Seeding from the first measured mean return, not from a
            # defaulted 0.0: abstention resolves into a measurement.
            used_baseline = batch_mean
        advantages = [score - used_baseline for score in scores]
        if self.config.reinforce_flat_groups == "zero" and not null_rank:
            group_returns: dict[int, set[float]] = {}
            for key, score in zip(group_keys, scores, strict=True):
                group_returns.setdefault(key, set()).add(score)
            flat = {key for key, values in group_returns.items() if len(values) == 1}
            advantages = [
                0.0 if key in flat else value
                for key, value in zip(group_keys, advantages, strict=True)
            ]
            print(
                f"[trainer] step {step}: reinforce_baseline zeroed "
                f"{len(flat)}/{len(group_returns)} flat group(s)",
                file=sys.stderr,
            )
        local_zero = null_rank or not any(value != 0.0 for value in advantages)
        if agree_all(local_zero, ctx):
            # Every kept return equals the baseline: no stimulus in the
            # whole batch, no gradient, no claimed step.
            print(
                f"UNMEASURED step {step}: every kept return equals the "
                f"baseline {used_baseline!r} over {int(total_n)} used "
                f"row(s); the REINFORCE surrogate is identically zero, "
                f"so no gradient exists and no step is claimed.",
                file=sys.stderr,
            )
            return None
        if local_zero:
            print(
                f"[trainer] step {step}: every local return equals the baseline "
                f"but a peer's does not; participating with zero loss "
                f"(UNMEASURED here)",
                file=sys.stderr,
            )
            advantages = [0.0] * len(scores)
        advantage_tensor = torch.tensor(advantages, dtype=torch.float32, device=kept_current.device)
        loss_fn = TensorREINFORCELoss(objective=objective)
        loss_tensor = loss_fn(
            current_logprobs=kept_current,
            advantages=advantage_tensor,
            mask=kept_mask,
        )
        if local_zero:
            loss_tensor = loss_tensor * 0.0
        optimizer.zero_grad()
        if use_logprob_micro_batching:
            _micro_batched_backward(
                loss_tensor=loss_tensor,
                current_logprobs=current_logprobs,
                row_slices=row_slices,
                forward_slice=forward_slice,
            )
        else:
            loss_tensor.backward()
        _padding_backwards(
            forward_slice=forward_slice,
            extra_slices=extra_slices,
            n_rows=int(current_logprobs.shape[0]),
        )
        optimizer.step()
        # The state update happens only after the price: step N reports the
        # baseline it USED, and the EMA folds in this batch's mean so step
        # N + 1 subtracts the updated value.
        state = self._reinforce_baseline
        self._reinforce_baseline = (
            batch_mean if state is None else momentum * state + (1.0 - momentum) * batch_mean
        )
        measured, measured_weight = _dp_weighted_loss(
            float(loss_tensor.detach()), 0.0 if local_zero else local_n, ctx
        )
        # The declared bounded diagnostic: strictly above, so a batch whose
        # returns all EQUAL the baseline reads 0.0 -- degenerate, truthfully.
        # Counted over every rank's real rows, like the baseline itself.
        local_above = 0.0 if null_rank else float(sum(1 for s in scores if s > used_baseline))
        frac_above = all_reduce_sum(local_above, ctx) / total_n
        loss_output = LossOutput(
            loss=measured,
            components=_loss_components(
                objective=objective,
                total=measured,
                current_logprobs=kept_current,
                reference_logprobs=None,
                mask=kept_mask,
            ),
            metrics=(
                MetricObservation(
                    name=str(objective.baseline_metric_name),
                    value=frac_above,
                ),
            ),
        )
        print(
            f"[trainer] step {step}: reinforce_baseline "
            f"baseline_used={used_baseline:.6g} "
            f"baseline_next={self._reinforce_baseline:.6g}",
            file=sys.stderr,
        )
        return StepReport(
            step=step,
            loss=loss_output,
            # A zero-loss rank names the peers' measured rows (zero is refused).
            rows=int(measured_weight) if local_zero else len(scores),
            reward_stats=None,
            sync=None,
        )

    def _reinforce_pp_tail(
        self,
        *,
        step: int,
        scores: list[float],
        kept_current: torch.Tensor,
        kept_old: torch.Tensor,
        kept_mask: torch.Tensor,
        kept_ref: torch.Tensor | None,
        current_logprobs: torch.Tensor,
        row_slices: tuple[tuple[int, int], ...],
        forward_slice: Callable[[int, int], torch.Tensor],
        use_logprob_micro_batching: bool,
        objective: Any,
        loss_fn: TensorPolicyLoss,
        optimizer: Any,
        ctx: Any,
        null_rank: bool,
        extra_slices: int,
    ) -> StepReport | None:
        """Reinforce++ tail: k1 fold into the return, GLOBAL z-score, PPO clip.

        WHAT IS CLAIMED: each kept row's penalised return is its scalar
        reward minus ``kl_beta * sum_supervised(current - reference)`` over
        DETACHED readings; the normalisation is global over kept rows with
        population statistics via RewardStats.over; a zero global spread is
        UNMEASURED, never a manufactured z-score; and the PPO-clipped
        token-ratio tail is the shared tensor kernel, whose declared axes
        (token scope, symmetric clip) it reads off the objective itself.

        Under data parallelism "global" means over every rank's kept rows:
        the mean and population spread are reduced across ranks (a null rank
        contributes none), so the zero-spread verdict is collective and a
        null rank participates at zero loss.

        WHAT IS NOT CLAIMED: any equivalence with a reference Reinforce++
        implementation, and any comparability of the penalised scale with
        the raw reward scale.
        """
        import math

        import torch

        from foundationscale.rl.distributed import all_reduce_sum

        if kept_ref is None:
            raise TrainerRefusal(
                "reinforce_pp folds reference log-probabilities into every "
                "row's return (the k1 penalty) but 0 of 1 reference planes "
                "are loaded; run() loads one automatically unless "
                "reference_policy=False refused it upstream"
            )
        if not hasattr(objective, "kl_beta"):
            # No silent 0.04: the k1 fold strength is a declared field of the objective.
            raise TrainerRefusal(
                f"reinforce_pp: objective {type(objective).__name__} declares no kl_beta; "
                "refusing to substitute a default penalty strength"
            )
        kl_beta = float(objective.kl_beta)
        cur = kept_current.detach()
        ref_plane = kept_ref.detach()
        kl_per_row = ((cur - ref_plane) * kept_mask).sum(dim=-1).to(dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=cur.device)
        # The penalty folds into the RETURN, never a separable loss term:
        # one folded component is what the objective's declaration states.
        penalised_list = [float(value) for value in (scores_tensor - kl_beta * kl_per_row).tolist()]
        if ctx.is_distributed:
            # The same two-pass population formula as RewardStats.over, with
            # each sum reduced across ranks; the count is the global count.
            local_n = 0.0 if null_rank else float(len(penalised_list))
            total_n = all_reduce_sum(local_n, ctx)
            mean = all_reduce_sum(0.0 if null_rank else sum(penalised_list), ctx) / total_n
            local_sq = 0.0 if null_rank else sum((v - mean) ** 2 for v in penalised_list)
            std = math.sqrt(all_reduce_sum(local_sq, ctx) / total_n)
        else:
            local_n = float(len(penalised_list))
            stats = RewardStats.over(tuple(penalised_list))
            mean, std = stats.mean, stats.std
        if std == 0.0:
            print(
                f"UNMEASURED step {step}: global advantage normalisation "
                f"over {len(scores)} kept row(s) found zero spread in the "
                f"penalised returns: every z-score would be a manufactured "
                f"0.0, and a manufactured zero gradient is not a "
                f"measurement -- no step is claimed.",
                file=sys.stderr,
            )
            return None
        z_scores = torch.tensor(
            [(value - mean) / std for value in penalised_list],
            dtype=torch.float32,
            device=cur.device,
        )
        loss_tensor = loss_fn(
            current_logprobs=kept_current,
            old_logprobs=kept_old,
            advantages=z_scores,
            mask=kept_mask,
        )
        if null_rank:
            loss_tensor = loss_tensor * 0.0
        optimizer.zero_grad()
        if use_logprob_micro_batching:
            _micro_batched_backward(
                loss_tensor=loss_tensor,
                current_logprobs=current_logprobs,
                row_slices=row_slices,
                forward_slice=forward_slice,
            )
        else:
            loss_tensor.backward()
        _padding_backwards(
            forward_slice=forward_slice,
            extra_slices=extra_slices,
            n_rows=int(current_logprobs.shape[0]),
        )
        optimizer.step()
        measured, measured_weight = _dp_weighted_loss(
            float(loss_tensor.detach()), 0.0 if null_rank else local_n, ctx
        )
        loss_output = LossOutput(
            loss=measured,
            components=_loss_components(
                objective=objective,
                total=measured,
                current_logprobs=kept_current,
                reference_logprobs=None,
                mask=kept_mask,
            ),
            metrics=_ratio_and_clip_metrics(
                objective=objective,
                current_logprobs=kept_current,
                old_logprobs=kept_old,
                mask=kept_mask,
            ),
        )
        print(
            f"[trainer] step {step}: reinforce_pp "
            f"penalised_mean={mean:.6g} penalised_std={std:.6g}",
            file=sys.stderr,
        )
        return StepReport(
            step=step,
            loss=loss_output,
            rows=int(measured_weight) if null_rank else len(scores),
            reward_stats=None,
            sync=None,
        )


def _ratio_and_clip_metrics(
    *,
    objective: Any,
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[Any, ...]:
    """Per-step observability metrics for the step's ``LossOutput.metrics`` (#546).

    WHAT IS CLAIMED: the tuple carries a ``MetricObservation`` named
    ``ratio_mean`` -- the mean importance ratio over exactly the supervised
    positions -- and, when the objective declares ``clip_bounds``, one named
    ``clip_fraction`` -- the fraction of those ratios OUTSIDE the declared
    ``(low, high)`` band. Both are measured from the same kept
    ``current_logprobs`` / ``old_logprobs`` / ``mask`` tensors the kernel
    priced, so on the first step -- old and current read off the same
    weights -- the pair is exactly (1.0, 0.0), and the invariant becomes
    checkable from the public report instead of from a debugger. The pair
    rides in the ``metrics`` channel precisely because the ``StepReport``
    contract (algorithm.py) forbids derived quantities and second copies of
    a count the ``LossOutput`` already holds: an observation belongs in the
    observation's own record.

    Ratios are token-level, ``exp(current - old)`` per supervised position,
    unless the objective declares ``ratio_scope == "sequence"``: then each
    row contributes its sequence-level ratio, ``exp`` of the masked mean
    log-ratio -- the very expression ``TensorPolicyLoss`` forms for that
    scope -- and both the mean and the clipped fraction are taken over rows.
    When the objective declares no ``clip_bounds`` the clip fraction is
    UNMEASURABLE, so that entry is omitted rather than reported as 0.0 --
    absent is not zero. An all-zero mask omits both, for the same reason.

    WHAT IS NOT CLAIMED: that these names are DECLARED on the objective.
    Declared metric expectations live on each objective's ``declaration()``
    -- the way ``DPOLoss`` declares ``accuracy`` with bounds and a degenerate
    tuple -- and nothing this loop runs reconciles observed metrics against
    that declaration: ``StepReport.__post_init__`` validates the components,
    the reward-stats count and the sync record only, and ``verify_step`` --
    the check that does reconcile observed metrics against declared ones --
    is not on the trainer's path. Emitting the pair therefore keeps every
    accounting check in algorithm.py green, which is the coherence the
    channel requires here.
    """
    import torch

    from foundationscale.gates.objective_gates import MetricObservation

    # Detached float64 readings: the ratio and the clip fraction are
    # reported, never differentiated, and the upcast keeps the reported mean
    # honest when the log-probability plane itself runs bf16.
    cur = current_logprobs.detach().to(dtype=torch.float64)
    old = old_logprobs.detach().to(dtype=torch.float64)
    mask_f = mask.detach().to(dtype=torch.float64)
    if not bool(mask_f.sum() > 0):
        return ()
    log_ratio = (cur - old) * mask_f
    if getattr(objective, "ratio_scope", "token") == "sequence":
        # One ratio per row -- the masked-mean-of-log-ratios the kernel
        # exponentiates for sequence scope -- so the clipped fraction here is
        # over rows, not tokens. Rows are the kernel's kept rows, which the
        # kernel refused unless each carried supervision, so no denominator
        # can be zero on this path.
        ratios = torch.exp(log_ratio.sum(dim=-1) / mask_f.sum(dim=-1))
    else:
        ratios = torch.exp(log_ratio)[mask_f > 0.5]
    metrics: list[Any] = [MetricObservation(name="ratio_mean", value=float(ratios.mean()))]
    clip_bounds = getattr(objective, "clip_bounds", None)
    if clip_bounds is not None:
        low, high = float(clip_bounds[0]), float(clip_bounds[1])
        outside = ((ratios < low) | (ratios > high)).to(dtype=torch.float64)
        metrics.append(MetricObservation(name="clip_fraction", value=float(outside.mean())))
    return tuple(metrics)


def _declared_component_names(objective: Any) -> tuple[str, ...]:
    declaration = objective.declaration()
    return tuple(declaration.components)


def _loss_components(
    *,
    objective: Any,
    total: float,
    current_logprobs: torch.Tensor | None = None,
    reference_logprobs: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[Any, ...]:
    """Decompose the measured scalar across the objective's declared components.

    The kernel returns ONE scalar, so without a reference term exactly one
    component can carry a measured contribution -- the first declared
    component IS the whole loss. With an active k3 term the KL contribution
    is re-measured off the kept tensors (detached, fp64: reported, never
    differentiated) with the kernel's own expression and clamp, attributed
    to the second declared component, and the policy component gets the
    remainder ``total - kl``. Attributing the whole scalar to the policy
    while the kl component read ``None`` would misstate both.

    Any further declared component is emitted with ``contribution=None`` --
    UNMEASURED, never 0.0: reporting an unmeasured term as zero would
    assert it is inert when nothing measured whether it is.
    """
    import torch

    from foundationscale.gates.objective_gates import LossComponent

    names = _declared_component_names(objective)
    kl_weight = float(getattr(objective, "kl_weight", 0.0))
    kl_contribution: float | None = None
    if (
        kl_weight != 0.0
        and current_logprobs is not None
        and reference_logprobs is not None
        and mask is not None
        and len(names) >= 2
    ):
        cur = current_logprobs.detach().to(dtype=torch.float64)
        ref = reference_logprobs.detach().to(dtype=torch.float64)
        mask_f = mask.detach().to(dtype=torch.float64)
        log_reference_ratio = (ref - cur) * mask_f
        # The kernel's own k3 expression, clamp included, upcast so the
        # REPORTED split does not inherit the plane's bf16 read.
        k3 = ((torch.expm1(log_reference_ratio) - log_reference_ratio) * mask_f).clamp(min=0.0)
        kl_contribution = kl_weight * float(k3.sum() / mask_f.sum())
    components: list[Any] = []
    for index, name in enumerate(names):
        if index == 0:
            rest = total - kl_contribution if kl_contribution is not None else total
            components.append(
                LossComponent(name=name, weight=1.0, observed=True, contribution=rest)
            )
        elif index == 1 and kl_contribution is not None:
            components.append(
                LossComponent(
                    name=name,
                    weight=kl_weight,
                    observed=True,
                    contribution=kl_contribution,
                )
            )
        else:
            components.append(
                LossComponent(name=name, weight=0.0, observed=False, contribution=None)
            )
    return tuple(components)


# Sanity: BatchRefusal is imported so a caller can catch the tensor plane's
# refusals through this module without importing torch.
_ = (BatchRefusal, StepReportRefusal)
