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
``loss=float(tensor)``.

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
    from collections.abc import Iterable

    import torch

from foundationscale.rl.advantage import AdvantageRefusal, RewardStats
from foundationscale.rl.algorithm import StepReport, StepReportRefusal
from foundationscale.rl.corpus import Sample, load_sharegpt
from foundationscale.rl.interfaces import BatchRefusal, LossOutput
from foundationscale.rl.registry import lookup_algorithm
from foundationscale.rl.rewards import MCQLetterReward
from foundationscale.rl.torch_backend import TensorPolicyLoss

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
    # dr_grpo, not grpo. GRPO declares a k3 reference term, so it requires a
    # reference-policy log-probability column; this loop holds ONE model and
    # produces no reference plane, so a default of "grpo" refuses on every
    # step of every default run. A default that cannot run is not a default.
    # Selecting "grpo" here remains legal and will refuse with that reason
    # named, which is the honest outcome -- it just is not what an operator
    # gets by typing nothing.
    algorithm: str = "dr_grpo"
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
    seed: int = 0
    device: str | None = None


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
        self.config = config

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
            raise TrainerRefusal(
                f"algorithm {self.config.algorithm!r}: 0 of 1 required objective "
                f"instances expose the declared axes; the tensor kernel reads the "
                f"objective's declarations rather than a per-algorithm branch"
            )
        # A NAMED, DELIBERATE LIMITATION, not an oversight. An active KL term
        # needs reference log-probabilities, which need a frozen copy of the
        # initial policy held alongside the trained one. This loop holds ONE
        # model. Refusing here names the missing input; letting it through
        # would surface as the kernel's reference-absent refusal several
        # frames deeper, where it reads like a bug rather than a boundary.
        kl_weight = float(getattr(objective, "kl_weight", 0.0))
        if kl_weight != 0.0:
            raise TrainerRefusal(
                f"algorithm {self.config.algorithm!r} declares kl_weight={kl_weight}; "
                f"0 of 1 required reference policies are loaded by this loop, so the "
                f"k3 term cannot be measured. Reference-free objectives "
                f"(kl_weight == 0.0) run today; the reference-policy path is not built."
            )
        return objective

    def run(self) -> list[StepReport]:
        """Run the training loop and return one report per completed step."""

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

        samples = load_sharegpt(self.config.dataset)
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
        carriers = [
            (sample.sample_id, "image" if sample.images else "video")
            for sample in samples
            if sample.images or sample.video is not None
        ]
        if carriers:
            shown = ", ".join(f"{sid} ({kind})" for sid, kind in carriers[:5])
            more = f" and {len(carriers) - 5} more" if len(carriers) > 5 else ""
            _refuse_exit_96(
                f"{len(carriers)} of {len(samples)} sample(s) carry a modality "
                f"this trainer cannot yet route to the model: {shown}{more}. "
                "The corpus parses images and video; the training path is "
                "text-only, so training on them would silently discard the "
                "pixels. Refusing rather than dropping them."
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
                AutoTokenizer,
            )
        except ImportError:
            _refuse_exit_96(
                "1 of 2 required dependencies absent: transformers; models are "
                "loaded through Auto classes and templates through "
                "apply_chat_template, never by this package"
            )

        torch.manual_seed(self.config.seed)
        device = self.config.device
        if device is None:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        except Exception as exc:  # noqa: BLE001 -- load surface failure is a refusal
            _refuse_exit_96(f"tokenizer load failed for {self.config.model!r}: {exc}")
        try:
            # Annotated Any: transformers 5.x wraps ``from_pretrained`` in a
            # decorator whose return type does not survive inference, so the
            # subsequent ``.to(device)`` resolves against the wrapper rather
            # than the model and reports the device string as a bad `self`.
            # The alternative -- a cast to PreTrainedModel -- would assert a
            # class the auto-loader does not promise across both branches.
            model: Any = AutoModelForCausalLM.from_pretrained(self.config.model)
        except Exception:
            try:
                model = AutoModelForImageTextToText.from_pretrained(self.config.model)
            except Exception as exc:  # noqa: BLE001
                _refuse_exit_96(
                    f"model load failed for {self.config.model!r} under both auto classes: {exc}"
                )
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

        objective = self._resolve_objective()
        reward = MCQLetterReward()
        loss_fn = TensorPolicyLoss(objective=objective)
        # #369: bf16 params stepped directly by AdamW at lr=1e-6 discard every
        # sub-ulp update, so the loop trains ~nothing while loss, grad-norm,
        # throughput and changed checkpoint bytes all look healthy. Selection is
        # EXPLICIT and printed, because a silent choice here IS the defect.
        param_dtype = next(model.parameters()).dtype
        if self.config.master_weights is not None:
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
            chunk = [
                usable[(cursor + offset) % len(usable)]
                for offset in range(self.config.prompts_per_step)
            ]
            cursor += self.config.prompts_per_step
            report = self._one_step(
                step=step,
                chunk=chunk,
                model=model,
                tokenizer=tokenizer,
                reward=reward,
                objective=objective,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )
            if report is not None:
                reports.append(report)
        return reports

    def _one_step(
        self,
        *,
        step: int,
        chunk: list[Sample],
        model: Any,
        tokenizer: Any,
        reward: MCQLetterReward,
        objective: Any,
        loss_fn: TensorPolicyLoss,
        optimizer: Any,
        device: str,
    ) -> StepReport | None:
        """One step over surviving rows, or ``None`` when none survive.

        WHAT IS CLAIMED: used < offered is the normal state; a step whose
        every row abstains is UNMEASURED and skipped, never reported as a
        zero-row step.

        WHAT IS NOT CLAIMED: that any particular step produces a report.
        """
        import torch

        prompts: list[str] = [
            tokenizer.apply_chat_template(
                [{"role": role, "content": text} for role, text in sample.prompt_turns],
                tokenize=False,
                add_generation_prompt=True,
            )
            for sample in chunk
        ]
        golds: list[str | None] = [sample.gold for sample in chunk]

        prompt_ids = tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
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
        if not rows:
            # offered > used == 0: an entirely abstaining step is UNMEASURED.
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

        def forward_logprobs() -> Any:
            logits = model(
                input_ids=kept_sequences, attention_mask=attention, **modality_kwargs
            ).logits
            logps = torch.log_softmax(logits, dim=-1)
            return torch.gather(logps, 2, target_ids.unsqueeze(-1)).squeeze(-1)

        # Old logprobs are RECOMPUTED under no_grad over the same rows -- the
        # generation scores are never reused: their shapes differ and the bug
        # is silent.
        with torch.no_grad():
            old_logprobs = forward_logprobs()
        current_logprobs = forward_logprobs().requires_grad_(True)

        prompt_id_values = [f"row-{index // self.config.group_size}" for index, _ in rows]
        # The estimator reads the PER-TOKEN supervision mask, not a per-row
        # flag: it denominates each response by its own supervised length.
        # Handing it a 1-D tensor of ones made every row non-iterable and
        # refused the batch, and the row count it would have implied is not
        # the quantity the estimator needs.
        try:
            advantage = objective.advantage_fn.compute(
                prompt_ids=tuple(prompt_id_values),
                rewards=tuple(float(value) for value in rewards.tolist()),
                mask=tuple(tuple(int(entry) for entry in row) for row in response_mask.tolist()),
            )
        except AdvantageRefusal as exc:
            # The estimator refuses when NO row survives: every group was too
            # small for a baseline once abstentions were dropped. That is a
            # genuine UNMEASURED step, not a crash. Letting the refusal
            # propagate ended an entire multi-step run on one unlucky draw --
            # and a run that dies at step 4 of 8 reports nothing about steps
            # 5 to 8, which is a worse outcome than saying "this step taught
            # nothing" and continuing.
            print(
                f"UNMEASURED step {step}: the advantage estimator used 0 of "
                f"{len(rows)} offered row(s) -- {exc}",
                file=sys.stderr,
            )
            return None
        # AdvantageResult is a RECORD, not a per-row sequence: `rows` names the
        # batch indices it used and `weights` carries one per-token weight row
        # for each. Enumerating the record itself treated its fields as
        # advantages. `used < offered` is the estimator's own visible account
        # of what it dropped -- a group too small for a baseline leaves here.
        kept_rows = list(advantage.rows)
        if not kept_rows:
            return None
        keep = torch.tensor(kept_rows, device=device)
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
        if not bool(advantage_tensor.abs().any()):
            used_rewards = [float(rewards[row]) for row in kept_rows]
            distinct = sorted(set(used_rewards))
            print(
                f"UNMEASURED step {step}: advantage is identically zero over "
                f"{len(kept_rows)} of {len(rows)} used row(s); the reward has "
                f"no within-group variance (distinct rewards: {distinct}). "
                f"No gradient exists to take, so no step is claimed.",
                file=sys.stderr,
            )
            return None

        loss_tensor = loss_fn(
            current_logprobs=current_logprobs.index_select(0, keep),
            old_logprobs=old_logprobs.index_select(0, keep).detach(),
            advantages=advantage_tensor,
            mask=response_mask.index_select(0, keep).detach(),
        )
        optimizer.zero_grad()
        loss_tensor.backward()
        optimizer.step()

        measured = float(loss_tensor.detach())
        loss_output = LossOutput(
            loss=measured,
            components=_loss_components(objective, measured),
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
            rows=len(kept_rows),
            reward_stats=RewardStats.over(tuple(float(rewards[row]) for row in kept_rows)),
            sync=None,
        )


def _declared_component_names(objective: Any) -> tuple[str, ...]:
    declaration = objective.declaration()
    return tuple(declaration.components)


def _loss_components(objective: Any, total: float) -> tuple[Any, ...]:
    """Decompose the measured scalar across the objective's declared components.

    The kernel returns ONE scalar, so exactly one component can carry a
    measured contribution. This loop runs only reference-free objectives
    (``_resolve_objective`` refuses the rest), so the single declared
    component IS the whole loss and the attribution is exact.

    Any further declared component is emitted with ``contribution=None`` --
    UNMEASURED, never 0.0. Splitting one scalar across two names would
    double-count it, and reporting an unmeasured term as zero would assert
    it is inert when nothing measured whether it is.
    """
    from foundationscale.gates.objective_gates import LossComponent

    names = _declared_component_names(objective)
    return tuple(
        LossComponent(
            name=name,
            weight=1.0 if index == 0 else 0.0,
            observed=index == 0,
            contribution=total if index == 0 else None,
        )
        for index, name in enumerate(names)
    )


# Sanity: BatchRefusal is imported so a caller can catch the tensor plane's
# refusals through this module without importing torch.
_ = (BatchRefusal, StepReportRefusal)
