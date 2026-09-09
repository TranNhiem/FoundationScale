"""The tensor plane: ONE differentiable kernel for every declared objective.

This module is the parallel tensor plane to the torch-free oracle modules
(``losses.py``, ``group_policy_objectives.py``). It is deliberately NOT in
the torch-free set: torch is imported at module scope here, and nowhere in
any oracle module.

There is exactly ONE kernel, :class:`TensorPolicyLoss`. It does not know
GRPO from GSPO from DAPO. It reads four declarations off the objective --
``ratio_scope``, ``clip_bounds``, ``reduction``, ``advantage_fn``'s output
supplied by the caller, and ``kl_weight`` -- and branches on those declared
axes only. A new group-relative algorithm that declares its axes receives a
differentiable loss for free; writing a per-algorithm kernel would
reintroduce exactly the duplication the axes matrix measured away.

WHAT IS CLAIMED: given the same per-row log-probability readings, mask,
and advantage weights the oracle priced, the returned scalar agrees with
the oracle's ``LossOutput.loss`` to floating-point tolerance, for every
combination of the declared axes; every guard the oracle carries over its
inputs (shape agreement, finiteness, a non-empty supervision denominator)
has a counterpart here, refusing with both sides of every count named; and
the returned tensor is a 0-dim scalar whose gradient lands on
``current_logprobs``.

WHAT IS NOT CLAIMED: that the guards here replace the oracle's batch-level
validation (group composition and column schema belong to the oracle and
the trainer, which call this kernel only on already-priced rows); any
convergence, benchmark, or paper-equivalence property; or that this
module is torch-free -- it is not, by design.

COVERAGE, MEASURED: a 14-row mutation battery over this file kills 13.
The survivor is the ``.clamp(min=0.0)`` on the k3 term. It survives
because it is UNFALSIFIABLE here, not because the tests are weak: a sweep
of ``expm1(x) - x`` over 50 magnitudes in float32 and float64 produced no
negative value, so no input reaching this kernel can distinguish the
clamped form from the unclamped one. It is retained as parity with the
oracle's own roundoff clamp and is declared here rather than left to read
as a verified guard. Every other axis -- loss sign, both ratio scopes,
all three reductions, the clip bounds INCLUDING the asymmetric upper
bound only DAPO declares, the k3 direction, the supervision mask, the
empty-row refusal and the finiteness guard -- has a mutant that dies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    import torch

from foundationscale.rl.interfaces import BatchRefusal

__all__ = ("TensorPolicyLoss",)


def _require_2d(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() != 2:
        raise BatchRefusal(
            f"{name} has shape {tuple(tensor.shape)} ({tensor.dim()} dims); "
            f"a (rows, tokens) matrix of exactly 2 dims is required"
        )


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    import torch

    if not bool(torch.isfinite(tensor).all()):
        count = int((~torch.isfinite(tensor)).sum())
        raise BatchRefusal(
            f"{count} of {tensor.numel()} entries in {name} are non-finite; "
            f"one non-finite reading would poison the importance ratio for "
            f"its whole response"
        )


def _require_same_shape(
    name_a: str,
    tensor_a: torch.Tensor,
    name_b: str,
    tensor_b: torch.Tensor,
) -> None:
    if tuple(tensor_a.shape) != tuple(tensor_b.shape):
        raise BatchRefusal(
            f"{name_a} has shape {tuple(tensor_a.shape)} but {name_b} has "
            f"shape {tuple(tensor_b.shape)}; 1 pair of required inputs "
            f"disagrees and every token reading must be attributable to "
            f"exactly one (row, position)"
        )


@dataclass(frozen=True, slots=True)
class TensorPolicyLoss:
    """Differentiable evaluation of ANY declared group-relative objective.

    The kernel reads its behaviour off the objective's declarations and
    contains no per-algorithm branch:

    * ``objective.ratio_scope == "token"`` -- per-token ratio
      ``exp(current - old)``; ``== "sequence"`` -- one ratio per row,
      ``exp(sum of masked log-ratios / supervised count)``, broadcast back
      over the row's tokens.
    * ``objective.clip_bounds`` -- the ``(low, high)`` PPO clip interval.
    * ``objective.reduction`` -- ``"token_mean"`` divides the summed
      surrogate by the supervised-token count; ``"sequence_mean"`` divides
      each row by its own supervised count then means over rows;
      ``"constant"`` divides by ``rows * objective.constant_length``.
    * ``getattr(objective, "kl_weight", 0.0)`` -- when non-zero, the k3
      estimator ``exp(ref - cur) - (ref - cur) - 1`` over supervised tokens,
      denominated by the supervised-token mean exactly as the oracle
      denominates it, whatever ``reduction`` the policy side declares.
    * ``getattr(objective, "weight", 1.0)`` -- the policy component weight.

    THE LOSS IS NEGATED: the returned value is
    ``-(surrogate) + kl_weight * kl``, so gradient DESCENT on it ASCENDS
    the clipped surrogate. This matches the oracle, whose
    ``LossOutput.loss`` carries the same negation.

    All inputs are ``torch.Tensor`` of shape ``(rows, tokens)`` except
    ``advantages``, which is ``(rows,)`` (one scalar weight per row,
    broadcast over tokens) or ``(rows, tokens)`` (the advantage estimator's
    per-token weight rows, which already carry a literal 0.0 at masked
    positions). ``current_logprobs`` MUST require grad; every other input
    is detached here. The batch passed in must already be restricted to
    the estimator's USED rows -- row dropping is the estimator's affair,
    not this kernel's.

    WHAT IS CLAIMED: see the module docstring. WHAT IS NOT CLAIMED: that
    this kernel replicates the oracle's PER-ELEMENT k3 refusal. The oracle
    refuses any single k3 term below ``-1e-15``; doing that here would read
    a boolean off the device on every step, so the tensor plane clamps the
    roundoff to zero instead and leaves per-element adjudication to the
    oracle. The scalar loss IS checked for finiteness, because that costs
    one sync on a value the caller reads anyway.
    """

    objective: Any  # a SequenceObjective / GRPO-family loss

    def __call__(
        self,
        *,
        current_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        reference_logprobs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # torch is imported HERE, not at module scope, because the packaging
        # census forbids an import-time torch import anywhere under src/:
        # the base install ships without torch, and a module-scope import
        # would make this file unimportable there rather than merely
        # uncallable. The module still DECLARES itself torch-dependent --
        # calling it without torch raises here, which is the honest failure.
        import torch

        objective = self.objective
        origin = type(objective).__name__

        if not isinstance(current_logprobs, torch.Tensor):
            raise BatchRefusal(
                f"current_logprobs is {type(current_logprobs).__name__}, "
                f"not torch.Tensor; 1 of 1 gradient-carrying inputs must "
                f"be a tensor"
            )
        _require_2d("current_logprobs", current_logprobs)
        _require_finite("current_logprobs", current_logprobs)
        if not current_logprobs.requires_grad:
            raise BatchRefusal(
                f"current_logprobs.requires_grad is False for {origin}; 1 "
                f"of 1 live inputs carries no gradient, and a loss computed "
                f"from it can never produce a weight update"
            )

        old = old_logprobs.detach()
        mask_f = mask.detach().to(dtype=current_logprobs.dtype)
        adv = advantages.detach().to(dtype=current_logprobs.dtype)
        for name, tensor in (
            ("old_logprobs", old),
            ("mask", mask_f),
            ("advantages", adv),
        ):
            _require_finite(name, tensor)
        _require_2d("old_logprobs", old)
        _require_2d("mask", mask_f)
        _require_same_shape("current_logprobs", current_logprobs, "old_logprobs", old)
        _require_same_shape("current_logprobs", current_logprobs, "mask", mask_f)

        rows, tokens = current_logprobs.shape
        if adv.dim() == 1:
            if adv.shape[0] != rows:
                raise BatchRefusal(
                    f"advantages has {adv.shape[0]} rows but "
                    f"current_logprobs has {rows} rows; 1 of 1 advantage "
                    f"vectors must carry one weight per batch row"
                )
            adv = adv.unsqueeze(-1)
        elif adv.dim() == 2:
            _require_same_shape("current_logprobs", current_logprobs, "advantages", adv)
        else:
            raise BatchRefusal(
                f"advantages has shape {tuple(advantages.shape)} "
                f"({advantages.dim()} dims); a (rows,) vector or a "
                f"(rows, tokens) matrix is required"
            )

        bad_mask = int(((mask_f != 0.0) & (mask_f != 1.0)).sum())
        if bad_mask:
            raise BatchRefusal(
                f"{bad_mask} of {mask_f.numel()} mask entries are neither "
                f"0 nor 1; supervision mask entries must be 0 or 1, and a "
                f"fractional entry cannot state which positions carry "
                f"gradient"
            )
        supervised_total = mask_f.sum()
        if not bool(supervised_total > 0):
            raise BatchRefusal(
                f"0 of {mask_f.numel()} mask entries are supervised for "
                f"{origin}; the loss is unmeasured on a fully masked "
                f"batch, never 0.0"
            )
        # A row with no supervised token is refused, not absorbed. Dividing
        # it by a clamped denominator would let it enter a sequence_mean as
        # a zero term, shrinking the loss by diluting the row average -- an
        # empty row would then read as a row that scored zero. The oracle
        # refuses such a row; a plane that quietly averaged it in would
        # disagree with the oracle in the one direction nothing detects.
        row_supervised = mask_f.sum(dim=-1)
        empty_rows = int((row_supervised == 0).sum())
        if empty_rows:
            raise BatchRefusal(
                f"{empty_rows} of {mask_f.shape[0]} rows carry 0 supervised "
                f"tokens for {origin}; an unsupervised row has no ratio and "
                f"no advantage, and averaging it in as a zero term would "
                f"understate the loss rather than report the gap"
            )

        ratio_scope = objective.ratio_scope
        clip_low, clip_high = objective.clip_bounds
        weight = float(getattr(objective, "weight", 1.0))
        kl_weight = float(getattr(objective, "kl_weight", 0.0))

        # Masked positions of log_ratio read 0.0: they are multiplied by
        # the mask again before any reduction, so they contribute nothing.
        log_ratio = (current_logprobs - old) * mask_f
        if ratio_scope == "token":
            ratio = torch.exp(log_ratio)
        elif ratio_scope == "sequence":
            sequence_log_ratio = log_ratio.sum(dim=-1) / row_supervised
            ratio = torch.exp(sequence_log_ratio).unsqueeze(-1)
            # Under a SEQUENCE-scope ratio the advantage must be collapsed to
            # one weight per row BEFORE the clip, because the oracle clips a
            # single (ratio, advantage) pair per response. Clipping per token
            # and averaging afterwards is a different objective whenever the
            # per-token weights vary within a row: mean_t min(r*a_t, c*a_t)
            # != min(r*abar, c*abar) once some a_t change sign, since the min
            # selects the clipped branch per token rather than per response.
            # Today's estimators emit one constant weight per row, so this
            # collapse is an identity on them -- it is here so that pairing a
            # token-level advantage with a sequence-level ratio stays the
            # oracle's objective instead of silently becoming another one.
            adv = (adv * mask_f).sum(dim=-1, keepdim=True) / row_supervised.unsqueeze(-1)
        else:
            raise BatchRefusal(
                f"{origin} declares ratio_scope={ratio_scope!r}; the "
                f"kernel knows 2 scopes ('token', 'sequence') and this is "
                f"neither -- an undeclared axis is not a free declaration"
            )

        clipped = torch.clamp(ratio, min=clip_low, max=clip_high)
        # The PPO min: min(ratio * adv, clip(ratio) * adv) per position,
        # then masked. Advantage weights already read 0.0 at masked
        # positions when supplied per-token; the extra mask multiplication
        # covers the broadcast-per-row form.
        surrogate_terms = torch.minimum(ratio * adv, clipped * adv) * mask_f

        reduction = objective.reduction
        if reduction == "token_mean":
            surrogate = surrogate_terms.sum() / supervised_total
        elif reduction == "sequence_mean":
            per_row = surrogate_terms.sum(dim=-1) / row_supervised
            surrogate = per_row.mean()
        elif reduction == "constant":
            constant_length = int(getattr(objective, "constant_length", tokens))
            if constant_length <= 0:
                raise BatchRefusal(
                    f"{origin} declares constant_length={constant_length}; "
                    f"the constant-reduction denominator requires a "
                    f"positive integer length"
                )
            surrogate = surrogate_terms.sum() / float(rows * constant_length)
        else:
            raise BatchRefusal(
                f"{origin} declares reduction={reduction!r}; the kernel "
                f"knows 3 reductions ('token_mean', 'sequence_mean', "
                f"'constant') and this is none of them"
            )

        loss = -weight * surrogate
        if kl_weight != 0.0:
            if reference_logprobs is None:
                raise BatchRefusal(
                    f"kl_weight={kl_weight!r} is active for {origin} but "
                    f"0 of 1 required reference_logprobs inputs were "
                    f"supplied; an active k3 term requires a reference"
                )
            reference = reference_logprobs.detach().to(dtype=current_logprobs.dtype)
            _require_2d("reference_logprobs", reference)
            _require_finite("reference_logprobs", reference)
            _require_same_shape(
                "current_logprobs", current_logprobs, "reference_logprobs", reference
            )
            # k3 estimator, token-relative, denominated by the supervised-
            # token mean exactly as the oracle denominates it.
            log_reference_ratio = (reference - current_logprobs) * mask_f
            # expm1(x) - x is nonnegative for every real x, so a negative
            # entry here is float roundoff near x == 0. The oracle clamps
            # that same roundoff to zero; clamping matches it without
            # reading a per-element verdict back off the device.
            k3 = ((torch.expm1(log_reference_ratio) - log_reference_ratio) * mask_f).clamp(min=0.0)
            loss = loss + kl_weight * (k3.sum() / supervised_total)

        if not bool(torch.isfinite(loss)):
            # Inputs are checked for finiteness, but exp() of a large
            # supervised log-ratio overflows to inf and propagates a nan
            # into every parameter on the first backward. The oracle
            # refuses a non-finite ratio per element; the tensor plane
            # catches the same class once, on the scalar, where the check
            # costs one sync the caller pays for anyway.
            raise BatchRefusal(
                f"{origin} produced a non-finite loss ({loss.detach()!r}); "
                f"inputs were finite, so the overflow arose inside the "
                f"kernel -- most often exp() of an unclipped log-ratio"
            )
        if loss.dim() != 0 or not loss.requires_grad:
            # Unreachable through the guards above: the guards ensure
            # current_logprobs requires grad and every reduction returns a
            # 0-dim tensor. Fail closed rather than hand the optimizer a
            # scalar it cannot see.
            raise BatchRefusal(  # pragma: no cover
                f"{origin} produced a loss with dim={loss.dim()} and "
                f"requires_grad={loss.requires_grad}; the optimizer's "
                f"scalar must be 0-dim and differentiable"
            )
        return loss
