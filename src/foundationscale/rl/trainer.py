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

No model-family branching exists anywhere here. Model loading tries
``AutoModelForCausalLM`` and falls back to ``AutoModelForImageTextToText``
only (transformers 5.x removed ``AutoModelForVision2Seq``, and it is the
successor class Gemma-4 registers under);
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    from collections.abc import Callable, Iterable, Sequence

    import torch

from foundationscale.rl.advantage import AdvantageRefusal, RewardStats
from foundationscale.rl.algorithm import StepReport, StepReportRefusal
from foundationscale.rl.corpus import Sample, load_sharegpt
from foundationscale.rl.interfaces import BatchRefusal, LossOutput
from foundationscale.rl.online_objectives import BestOfNLoss, RAFTLoss
from foundationscale.rl.online_pref_step import (
    is_online_pref,
    maybe_refresh_reference,
    online_pref_step,
    refresh_cadence,
)
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


@dataclass
class RLTrainConfig:
    """Configuration for one RL training run.

    ``model`` is a local path or hub id and is NEVER defaulted: hardcoding a
    default model would smuggle an untested surface into every run that
    forgot the flag. ``algorithm`` names a registry entry
    (``"grpo"``/``"gspo"``/``"dr_grpo"``/``"dapo"``). ``device=None`` means
    auto-select -- metal when available, else CPU.

    WHAT IS CLAIMED: these are the only knobs the loop reads.

    WHAT IS NOT CLAIMED: that any particular value trains well; nothing here
    is tuned.
    """

    model: str
    dataset: str
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

        samples = load_sharegpt(self.config.dataset, gold_key=self.config.gold_key)
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
        # VIDEO is still refused, and deliberately so. Nothing in this repo
        # samples frames from a clip, and the missing pieces are DATA decisions
        # rather than plumbing: frame count, sampling strategy, per-frame
        # resolution. Any default chosen here would silently redefine the
        # dataset. Against the measured 131,072-token context, 258 tok/frame
        # puts the ceiling near 508 frames -- ample, which is precisely why the
        # frame budget should be chosen rather than inherited from whoever
        # wrote this line.
        carriers = [(sample.sample_id, "video") for sample in samples if sample.video is not None]
        if carriers:
            shown = ", ".join(f"{sid} ({kind})" for sid, kind in carriers[:5])
            more = f" and {len(carriers) - 5} more" if len(carriers) > 5 else ""
            _refuse_exit_96(
                f"{len(carriers)} of {len(samples)} sample(s) carry video, which "
                f"no component in this repository decodes: {shown}{more}. "
                "Frame count, sampling strategy and per-frame resolution are "
                "dataset decisions with no safe default -- picking one here "
                "would silently redefine the corpus. Images ARE supported and "
                "are routed through the processor; video refuses until the "
                "frame budget is declared."
            )
        try:
            import torch
        except ImportError:
            _refuse_exit_96(
                "1 of 2 required dependencies absent: torch; the tensor plane "
                "needs it and no pure-python fall-back exists for weight updates"
            )
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
            )
        except ImportError:
            _refuse_exit_96(
                "1 of 2 required dependencies absent: transformers; models are "
                "loaded through Auto classes and templates through "
                "apply_chat_template, never by this package"
            )

        from foundationscale.rl.distributed import (
            destroy,
            init_distributed,
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

        def _load_causal_lm(model_id: str) -> Any:
            # Annotated Any: transformers 5.x wraps ``from_pretrained`` in a
            # decorator whose return type does not survive inference, so the
            # subsequent ``.to(device)`` resolves against the wrapper rather
            # than the model and reports the device string as a bad `self`.
            # The alternative -- a cast to PreTrainedModel -- would assert a
            # class the auto-loader does not promise across both branches.
            # Shared by the policy and the frozen reference copy: two inline
            # copies of this try/except would be two chances for them to drift.
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

        model = _load_causal_lm(self.config.model)
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
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
            # The frozen reference plane: loaded before any optimizer step so
            # it IS the initial policy -- which is what makes the step-1 k3
            # contribution exactly zero. Only an objective declaring a
            # non-zero kl_weight (or an operator forcing reference_policy=True)
            # pays this memory; _resolve_objective already refused the
            # needs-one-but-forbidden combination.
            ref_model = _load_causal_lm(self.config.reference_model or self.config.model)
            if self.config.sharding == "fsdp":
                # Sharded too: a frozen full-size replica on each rank would
                # rescale memory exactly the way fsdp exists to prevent. DDP
                # keeps it plain -- no gradient averaging is wanted over a
                # frozen model, so no DDP wrapper.
                ref_model = wrap_fsdp2(ref_model, ctx)
            else:
                ref_model.to(device)
            ref_model.eval()
            for parameter in ref_model.parameters():
                parameter.requires_grad_(False)
        reward = MCQLetterReward(answer_pattern=self.config.answer_pattern)
        loss_fn = None if online_pref else TensorPolicyLoss(objective=objective)
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
            optimizer = MasterWeightOptimizer(model.parameters(), lr=self.config.learning_rate)
        else:
            optimizer = torch.optim.AdamW(  # noqa: B014
                model.parameters(), lr=self.config.learning_rate
            )
        print(
            "[trainer] optimizer="
            + ("MasterWeightOptimizer(host-fp32)" if use_masters else "AdamW(direct)")
            + f" param_dtype={param_dtype} lr={self.config.learning_rate}"
            + f" reason={master_reason}",
            file=sys.stderr,
        )

        usable = tuple(sample for sample in samples if sample.gold is not None)
        if not usable:
            _refuse_exit_96(
                f"0 of {len(samples)} loaded samples carry a parseable gold letter; "
                f"a run with no verifiable row is vacuous"
            )

        reports: list[StepReport] = []
        cursor = 0
        for step in range(self.config.max_steps):
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
            if online_pref:
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
        from foundationscale.rl.distributed import (
            DistContext,
            agree_all,
            agree_max,
            all_reduce_sum,
            generate_kwargs_for,
        )

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
        with torch.no_grad():
            # Under DDP the generative path bypasses the wrapper (no_grad --
            # no grad sharing is wanted during rollout); under fsdp the
            # wrapper IS the module generate must run on.
            generated = getattr(model, "module", model).generate(
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
            # generate() expanded each prompt into group_size rows; the modality
            # tensors are still one row per PROMPT, so they are repeated to
            # match and then narrowed to the kept rows, in that order. Doing it
            # the other way round selects against the wrong axis silently.
            group = self.config.group_size
            modality_kwargs = {
                key: value.repeat_interleave(group, dim=0).index_select(0, kept_indices)
                for key, value in modality_kwargs.items()
            }
            print(
                "[trainer] forwarding modality keys to the scorer: "
                + ", ".join(sorted(modality_kwargs)),
                file=sys.stderr,
            )

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
            # attention mask, the shifted targets and every modality tensor.
            # ``narrow`` names dimension 0 explicitly; silently slicing a
            # modality's feature axis would score a different condition.
            width = end - start
            sliced_modalities = {
                key: value.narrow(0, start, width) for key, value in modality_kwargs.items()
            }
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

        prompt_id_values = [f"row-{index // self.config.group_size}" for index, _ in rows]
        # The estimator reads the PER-TOKEN supervision mask, not a per-row
        # flag: it denominates each response by its own supervised length.
        # Handing it a 1-D tensor of ones made every row non-iterable and
        # refused the batch, and the row count it would have implied is not
        # the quantity the estimator needs.
        advantage: Any = None
        adv_refusal: str | None = None
        try:
            if not null_rank:
                advantage = objective.advantage_fn.compute(
                    prompt_ids=tuple(prompt_id_values),
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
                f"{len(rows)} offered row(s) -- {adv_refusal}",
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
                f"{len(rows)} scored row(s); no group was large enough to admit a "
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
                kept_rows, prompt_id_values, [float(value) for value in rewards.tolist()]
            )
            print(
                f"UNMEASURED step {step}: advantage is identically zero over "
                f"{len(kept_rows)} of {len(rows)} used row(s); no group's reward "
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
        loss_tensor = loss_fn(
            current_logprobs=kept_current,
            old_logprobs=kept_old,
            advantages=advantage_tensor,
            mask=kept_mask,
            reference_logprobs=kept_ref,
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
